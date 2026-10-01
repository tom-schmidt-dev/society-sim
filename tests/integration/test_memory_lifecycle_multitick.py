from __future__ import annotations

import re
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from src.application.services.lifecycle.day_night_service import DayNightService
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.cognition import (
    BlockedResolution,
    DialogueResolution,
    GoalDecision,
    GoalEvaluation,
)
from src.domain.models.planning.planning import (
    ActionType,
    AgentCognitiveContext,
    PlanDecomposition,
    SubGoalIntent,
)
from src.domain.models.world.position import Position
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.presenter import IPresenter
from src.domain.ports.vector_memory_store import IVectorMemoryStore
from src.infrastructure.container import ApplicationContainer


# ==============================================================================
# In-Memory Test Doubles (Keine Netzwerk- oder LiteLLM-Abhängigkeiten)
# ==============================================================================


class EphemeralVectorStore(IVectorMemoryStore):
    """Speichert und filtert Vektoreinträge im Speicher für Integrationstests."""

    def __init__(self) -> None:
        self._entries: dict[str, list[tuple[str, dict[str, Any]]]] = {}

    def add_memories(
        self,
        agent_id: str,
        memories: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self._entries.setdefault(agent_id, [])
        metas = metadatas if metadatas is not None else [{} for _ in memories]
        for mem, meta in zip(memories, metas):
            self._entries[agent_id].append((mem, meta))

    def retrieve_relevant(
        self,
        agent_id: str,
        query: str,
        limit: int = 3,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        items = self._entries.get(agent_id, [])
        results: list[str] = []
        for text, meta in items:
            if metadata_filter and not all(meta.get(k) == v for k, v in metadata_filter.items()):
                continue
            results.append(text)
            if len(results) >= limit:
                break
        return results


class DeterministicCognitionStub(ICognitionProvider):
    """Vollständig deterministischer Stub ohne Hintergrund-Tasks oder Netzwerkaufrufe."""

    async def decide_next_goal(self, context: dict[str, Any]) -> Any:
        class SimpleDecision:
            thought = "Testgedanke"
            chosen_goal = "Erkunden"
        return SimpleDecision()

    async def resolve_blockage(self, context: dict[str, Any]) -> Any:
        class SimpleResolution:
            action = "wait"
            thought = "Warten im Test"
            new_sub_goal = None
        return SimpleResolution()

    async def evaluate_goal_status(self, context: dict[str, Any]) -> Any:
        class SimpleEvaluation:
            is_completed = True
            thought = "Testziel beendet"
            reason = "Erreicht"
        return SimpleEvaluation()

    async def respond_to_dialogue(self, context: dict[str, Any]) -> Any:
        class SimpleDialogue:
            action = "end_dialogue"
            thought = "Dialog beendet"
        return SimpleDialogue()

    async def decompose_plan(self, context: AgentCognitiveContext) -> PlanDecomposition:
        # 1. Erinnerungsbasierte Zielplanung (Tag 2: Retrieval aus dem Vektorspeicher)
        if context.episodic_memories:
            for mem in context.episodic_memories:
                match = re.search(r"bei \((\d+),\s*(\d+)\)", mem)
                if match:
                    target_pos = (int(match.group(1)), int(match.group(2)))
                    return PlanDecomposition(
                        thought=f"Erinnere Quelle bei {target_pos}.",
                        primary_goal="Nahrungssuche",
                        sub_goals=[
                            SubGoalIntent(
                                action_type=ActionType.MOVE_TO,
                                target_position=target_pos,
                                description=f"Gehe zu {target_pos}",
                            ),
                            SubGoalIntent(
                                action_type=ActionType.CONSUME,
                                description="Konsumiere gefundene Nahrung",
                            ),
                        ],
                    )

        # 2. Wahrnehmungsbasierte Zielplanung (Tag 1: Entität im Sichtfeld vorhanden)
        consumable = next((e for e in context.known_entities if e.is_consumable), None)
        if consumable and consumable.last_known_position:
            target_pos = (consumable.last_known_position.x, consumable.last_known_position.y)
            return PlanDecomposition(
                thought=f"Gesehene Ressource {consumable.name} ansteuern.",
                primary_goal="Nahrungssuche",
                sub_goals=[
                    SubGoalIntent(
                        action_type=ActionType.MOVE_TO,
                        target_position=target_pos,
                        description=f"Gehe zu {consumable.name}",
                    ),
                    SubGoalIntent(
                        action_type=ActionType.CONSUME,
                        target_entity_id=consumable.entity_id,
                        description=f"Konsumiere {consumable.name}",
                    ),
                ],
            )

        # 3. Fallback: Erkundung
        return PlanDecomposition(
            thought="Keine Ressource bekannt. Erkunde.",
            primary_goal="Nahrungssuche",
            sub_goals=[
                SubGoalIntent(
                    action_type=ActionType.EXPLORE,
                    description="Erkunde Terrain",
                )
            ],
        )

        # 2. Wahrnehmungsbasierte Zielplanung (Tag 1: Entität im Sichtfeld vorhanden)
        consumable = next((e for e in context.known_entities if e.is_consumable), None)
        if consumable and consumable.last_known_position:
            target_pos = (consumable.last_known_position.x, consumable.last_known_position.y)
            return PlanDecomposition(
                thought=f"Gesehene Ressource {consumable.name} ansteuern.",
                primary_goal="Nahrungssuche",
                sub_goals=[
                    SubGoalIntent(
                        action_type=ActionType.MOVE_TO,
                        target_position=target_pos,
                        description=f"Gehe zu {consumable.name}",
                    ),
                    SubGoalIntent(
                        action_type=ActionType.CONSUME,
                        target_entity_id=consumable.entity_id,
                        description=f"Konsumiere {consumable.name}",
                    ),
                ],
            )

        # 3. Fallback: Erkundung
        return PlanDecomposition(
            thought="Keine Ressource bekannt. Erkunde.",
            primary_goal="Nahrungssuche",
            sub_goals=[
                SubGoalIntent(
                    action_type=ActionType.EXPLORE,
                    description="Erkunde Terrain",
                )
            ],
        )


# ==============================================================================
# End-to-End Integrationstest
# ==============================================================================


@pytest.mark.asyncio
async def test_full_memory_lifecycle_e2e() -> None:
    # 1. Setup: 15 Takte Tag, 5 Takte Nacht (Gesamtzyklus: 20 Takte)
    vector_store = EphemeralVectorStore()
    day_night_service = DayNightService(day_ticks=15, night_ticks=5)
    cognition = DeterministicCognitionStub()

    container = ApplicationContainer.build(
        width=20,
        height=10,
        tick_interval=0.0,
        day_night_service=day_night_service,
        vector_store=vector_store,
        cognition_provider=cognition,
    )
    container.engine._presenter = MagicMock(spec=IPresenter)

    # 2. Betretbare Nahrungsquelle bei (6, 5) platzieren
    apple_tree = WorldEntity(
        id="tree_1",
        name="Apfelbaum",
        position=Position(6, 5),
        entity_type="food",
        is_conversational=False,
        is_passable=True,
    )
    setattr(apple_tree, "is_consumable", True)
    setattr(apple_tree, "nutrition_value", 0.8)
    setattr(apple_tree, "is_depletable", False)
    container.engine.register_entity(apple_tree)

    # 3. Agent Alice bei (4, 5) initialisieren (Distanz 2 liegt im Sichtradius von 3)
    alice = Agent(id="alice", name="Alice", position=Position(4, 5))
    alice.needs["hunger"] = 0.75
    alice.needs["thirst"] = 0.0
    alice.needs["energy"] = 0.0
    container.engine.register_agent(alice)

    # ------------------------------------------------------------------
    # Phase 1: Tag 1 (Takte 1 bis 14) -> Wahrnehmung, Annäherung, Konsum
    # ------------------------------------------------------------------
    for _ in range(14):
        await container.engine.process_tick()

    # Hunger muss durch Verzehr signifikant unter den Startwert von 0.75 gesunken sein
    assert alice.needs["hunger"] < 0.75

    # ------------------------------------------------------------------
    # Phase 2: Nacht 1 (Takte 15 bis 19) -> Schlaf & Konsolidierung
    # ------------------------------------------------------------------
    for _ in range(5):
        await container.engine.process_tick()

    assert alice.is_sleeping is True

    # Verifikation: Erinnerung mit Fundortkoordinaten wurde in den Vektorspeicher geschrieben
    memories = vector_store.retrieve_relevant(
        agent_id="alice", query="Nahrung", metadata_filter={"category": "resource"}
    )
    assert len(memories) >= 1
    assert "Apfelbaum bei (6, 5)" in memories[0]

    # ------------------------------------------------------------------
    # Phase 3: Tag 2 (Takt 20) -> Erwachen & Autonome Zielansteuerung via Gedächtnis
    # ------------------------------------------------------------------
    # Alice räumlich versetzen, ihr lokales Sichtgedächtnis leeren und Hunger akut setzen
    alice.position = Position(1, 1)
    alice.memory.known_entities.clear()
    alice.needs["hunger"] = 0.85

    # Takt 20 ausführen (Tag 2 startet)
    await container.engine.process_tick()

    assert alice.is_sleeping is False

    # Verifikation: Plandekomposition nutzte die Erinnerung zur direkten Navigation
    active_goal = alice.active_goal
    assert active_goal is not None
    assert active_goal.name == "SubGoal: move_to"
    assert active_goal.target_position == Position(6, 5)
    assert alice.has_path is True
    assert alice.path[-1] == Position(6, 5)