from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from src.application.services.cognition_orchestrator import CognitionOrchestrator
from src.application.services.daily_event_buffer import DailyEventBuffer
from src.application.services.day_night_service import DayNightService
from src.application.services.memory_consolidation_service import MemoryConsolidationService
from src.application.services.movement_orchestrator import MovementOrchestrator
from src.application.services.need_service import NeedService
from src.application.services.plan_decomposition_service import PlanDecompositionService
from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent import Agent
from src.domain.models.events import SimulationEvent
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter
from src.domain.ports.vector_memory_store import IVectorMemoryStore
from src.domain.services.perception_service import PerceptionService
from src.infrastructure.cognition.instructor_adapter import InstructorCognitionAdapter


# ==============================================================================
# In-Memory Test Doubles
# ==============================================================================


class InMemoryVectorMemoryStore(IVectorMemoryStore):
    """Speichert und filtert Vektoreinträge im Speicher für Integrationstests."""

    def __init__(self) -> None:
        self._entries: dict[str, list[tuple[str, dict[str, Any]]]] = {}

    def add_memories(
        self,
        agent_id: str,
        memories: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        if agent_id not in self._entries:
            self._entries[agent_id] = []
        metas = metadatas if metadatas is not None else [{} for _ in memories]
        for memory, meta in zip(memories, metas):
            self._entries[agent_id].append((memory, meta))

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
            if metadata_filter:
                match = all(meta.get(k) == v for k, v in metadata_filter.items())
                if not match:
                    continue
            results.append(text)
            if len(results) >= limit:
                break
        return results


class InMemoryDailyEventBuffer(DailyEventBuffer):
    """Deterministischer Puffer für Tagesereignisse."""

    def __init__(self) -> None:
        try:
            super().__init__()
        except Exception:
            pass
        self._events: dict[str, list[SimulationEvent]] = {}

    def record_event(self, event: SimulationEvent) -> None:
        self._events.setdefault(event.agent_id, []).append(event)

    def get_events_for_agent(self, agent_id: str) -> list[SimulationEvent]:
        return list(self._events.get(agent_id, []))

    def clear_agent(self, agent_id: str) -> None:
        self._events.pop(agent_id, None)


class InMemoryEventLogger(IEventLogger):
    """Erfasst emittierte Simulationsereignisse."""

    def __init__(self) -> None:
        self.events: list[SimulationEvent] = []

    def log(self, event: SimulationEvent) -> None:
        self.events.append(event)


# ==============================================================================
# Integrationstest: Kognitiver Gedächtniskreislauf
# ==============================================================================


@pytest.mark.asyncio
async def test_end_to_end_cognitive_retrieval_and_revisit() -> None:
    """
    Validiert das Zusammenspiel:
    1. Tag 1: Nutzung einer Wasserquelle bei (4, 4).
    2. Nacht 1: Konsolidierung in den Vektorspeicher.
    3. Tag 2: Durst triggert Plandekomposition; Erinnerung führt zur Zielansteuerung von (4, 4).
    """
    # 1. Infrastruktur- und Domain-Komponenten
    grid = WorldGrid(width=10, height=10)
    logger = InMemoryEventLogger()
    presenter = MagicMock(spec=IPresenter)

    # Deterministischer Pathfinder
    pathfinder = MagicMock(spec=IPathfinder)
    pathfinder.find_path.side_effect = lambda start, target, _: [
        start,
        Position((start.x + target.x) // 2, (start.y + target.y) // 2),
        target,
    ]

    vector_store = InMemoryVectorMemoryStore()
    event_buffer = InMemoryDailyEventBuffer()
    cognition_adapter = InstructorCognitionAdapter()

    # 1 Takt Tag, 1 Takt Nacht -> Zykluslänge = 2
    # Takt 1 = Tag 1, Takt 2 = Nacht 1, Takt 3 = Tag 2
    day_night_service = DayNightService(day_ticks=2, night_ticks=1, logger=logger)

    consolidation_service = MemoryConsolidationService(
        vector_store=vector_store,
        event_buffer=event_buffer,
        logger=logger,
    )

    need_service = NeedService()
    plan_decomp_service = PlanDecompositionService(
        cognition_provider=cognition_adapter,
        vector_memory_store=vector_store,
    )

    perception_service = MagicMock(spec=PerceptionService)
    perception_service.get_visible_positions.return_value = []
    perception_service.get_visible_entities.return_value = []

    movement_orchestrator = MagicMock(spec=MovementOrchestrator)

    engine = SimulationEngine(
        grid=grid,
        pathfinder=pathfinder,
        presenter=presenter,
        logger=logger,
        cognition_provider=cognition_adapter,
        vector_memory_store=vector_store,
        day_night_service=day_night_service,
        memory_consolidation_service=consolidation_service,
        need_service=need_service,
        plan_decomposition_service=plan_decomp_service,
        perception_service=perception_service,
        movement_orchestrator=movement_orchestrator,
    )

    agent = Agent(id="agent_explorer", name="Explorer", position=Position(1, 1))
    agent.needs["hunger"] = 0.0
    agent.needs["thirst"] = 0.0
    agent.needs["energy"] = 0.0
    engine.register_agent(agent)

    # ------------------------------------------------------------------
    # Phase 1: Tag 1 - Ereignis der Wassernutzung bei (4, 4) im Puffer
    # ------------------------------------------------------------------
    event_buffer.record_event(
        SimulationEvent(
            tick=1,
            agent_id=agent.id,
            event_type="resource_consumed",
            summary="Wasserquelle genutzt",
            payload={"resource_type": "Wasserquelle", "position": (4, 4)},
        )
    )

    await engine.process_tick()  # Takt 1 (Tag 1)
    assert agent.is_sleeping is False
    assert agent.id not in vector_store._entries

    # ------------------------------------------------------------------
    # Phase 2: Nacht 1 - Schlaf & Konsolidierung
    # ------------------------------------------------------------------
    await engine.process_tick()  # Takt 2 (Nacht 1, phase_tick == 1 == day_ticks)
    assert agent.is_sleeping is True

    # Verifikation: Erinnerung wurde in den Vektorspeicher geschrieben
    memories = vector_store.retrieve_relevant(
        agent_id=agent.id,
        query="Wasser",
        metadata_filter={"category": "resource"},
    )
    assert len(memories) == 1
    assert "Wasserquelle bei (4, 4) erfolgreich genutzt." in memories[0]
    assert event_buffer.get_events_for_agent(agent.id) == []

    # ------------------------------------------------------------------
    # Phase 3: Tag 2 - Erwachen & Autonome Zielansteuerung durch Erinnerung
    # ------------------------------------------------------------------
    # Durst wird vor dem Takt akut gesetzt; keine Entitäten im aktuellen Sichtfeld
    agent.needs["thirst"] = 0.95
    assert len(agent.memory.known_entities) == 0

    await engine.process_tick()  # Takt 3 (Tag 2 startet, phase_tick == 0)

    # Agent ist wach
    assert agent.is_sleeping is False

    # Verifikation: Plandekomposition hat Sub-Goals erzeugt
    assert len(agent.goals) >= 2
    active_goal = agent.active_goal
    assert active_goal is not None
    assert active_goal.name == "SubGoal: move_to"
    assert active_goal.target_position == Position(4, 4)

    # Verifikation: Pfadfinder hat den Weg zur erinnerten Position zugewiesen
    assert agent.has_path is True
    assert agent.path[-1] == Position(4, 4)