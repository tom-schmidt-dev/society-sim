from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import pytest

from main_dialogue_showcase import build_corridor_world
from src.domain.models.agent.agent_state import AgentLifecycleState
from src.domain.models.coordination.evasion_phase import EvasionPhase
from src.domain.models.planning.cognition import (
    BlockedResolution,
    DialogueResolution,
    GoalDecision,
    GoalEvaluation,
    GoalIntent,
    SocialReflection,
    TalkAction, MoveToAction, WaitAction,
)
from src.domain.models.planning.planning import AgentCognitiveContext, PlanDecomposition
from src.domain.models.world.position import Position
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.vector_memory_store import IVectorMemoryStore
from src.infrastructure.container import ApplicationContainer


class MockShowcaseCognition(ICognitionProvider):
    """Deterministischer Kognitionsadapter für das Korridor-Showcase."""

    async def decide_next_goal(self, context: dict[str, Any]) -> GoalDecision:
        return GoalDecision(
            thought="Kein weiteres Ziel erforderlich",
            action=MoveToAction(
                destination_name="Kein Ziel",
                target_position=Position(0, 0),
                reason="Showcase abgeschlossen",
            ),
        )

    async def resolve_blockage(self, context: dict[str, Any]) -> BlockedResolution:
        agent_id = str(context.get("agent_id", ""))
        blocker_id = str(context.get("blocker_id", ""))

        if agent_id == "1":  # Alice (fordernd)
            return BlockedResolution(
                thought="Der Korridor ist versperrt. Ich fordere Bob zum Ausweichen in die Nische auf.",
                action=TalkAction(
                    target_agent_id=blocker_id,
                    message="Der Weg ist zu schmal. Bitte weichen Sie in die Nische aus!",
                    intent="request_yield",
                ),
            )
        else:  # Bob (nachgiebig: wartet auf Alice' Anfrage statt unabgesprochen wegzulaufen)
            return BlockedResolution(
                thought="Ich sehe Alice und warte auf ihr Anliegen zur Klärung.",
                action=WaitAction(
                    ticks=2,
                    reason="Warte auf Klärung",
                ),
            )


    async def respond_to_dialogue(self, context: dict[str, Any]) -> DialogueResolution:
        agent_id = str(context.get("agent_id", ""))
        partner_id = str(context.get("partner_id", ""))
        incoming_intent = context.get("incoming_intent")

        if agent_id == "2":  # Bob reagiert auf Alice's Bitte
            return DialogueResolution(
                thought="Ich akzeptiere die Bitte und mache Platz in der Nische (15, 3).",
                action=TalkAction(
                    target_agent_id=partner_id,
                    message="Ich weiche nach (15, 3) aus.",
                    intent="offer_yield",
                ),
                negotiation_intent="offer_yield",
                new_goal=GoalIntent(name="In Nische ausweichen", intent_type="evade"),
            )
        else:  # Alice nimmt Bobs Ausweichangebot an
            return DialogueResolution(
                thought="Bob macht Platz, ich nehme das Angebot dankend an und passiere.",
                action=TalkAction(
                    target_agent_id=partner_id,
                    message="Vielen Dank, ich passiere den Korridor.",
                    intent="accept",
                ),
                negotiation_intent="accept",
            )

    async def reflect_on_dialogue(self, context: dict[str, Any]) -> SocialReflection:
        return SocialReflection(
            assessment="Sehr kooperative Einigung an der Engstelle erzielt.",
            progression_summary="Absprache erfolgreich: Bob weicht in Nische aus, Alice passiert.",
        )

    async def evaluate_goal_status(self, context: dict[str, Any]) -> GoalEvaluation:
        return GoalEvaluation(is_completed=True, reason="Zielkriterien erfüllt")

    async def decompose_plan(
        self, context: AgentCognitiveContext
    ) -> PlanDecomposition:
        return PlanDecomposition(primary_goal="Showcase", sub_goals=[])


@pytest.mark.asyncio
async def test_corridor_negotiation_full_cycle_no_exceptions(tmp_path: Path) -> None:
    log_file = str(tmp_path / "test_showcase_events.jsonl")
    cognition = MockShowcaseCognition()

    vector_store = MagicMock(spec=IVectorMemoryStore)
    vector_store.retrieve_relevant.return_value = []

    # 1. Container gemäß Showcase initialisieren
    container = ApplicationContainer.build(
        width=30,
        height=9,
        tick_interval=0.001,  # Schneller Testdurchlauf
        auditory_radius=4,
        log_file=log_file,
        cognition_provider=cognition,
        vector_store=vector_store,
        enable_deterministic_corridor=False,
        enable_day_night=False,
    )

    # 2. Welt und Agenten konfigurieren
    alice, bob = build_corridor_world(container)

    # 3. Simulation starten und bis zum Erreichen der Ziele durchlaufen lassen
    await container.engine.run(max_ticks=250, stop_when_idle=True)

    # 4. Status- und Positionsvalidierung
    # Alice muss ihr Ost-Portal bei (27, 4) erreicht haben
    assert alice.position == Position(27, 4), f"Alice hat Ost-Portal nicht erreicht: {alice.position}"
    assert alice.lifecycle_state == AgentLifecycleState.IDLE

    # Bob muss sein West-Portal bei (2, 4) erreicht haben
    assert bob.position == Position(2, 4), f"Bob hat West-Portal nicht erreicht: {bob.position}"
    assert bob.lifecycle_state == AgentLifecycleState.IDLE

    # Zielstacks müssen nach Abschluss vollständig abgearbeitet sein
    assert len(alice.goals) == 0 or all(g.status == "completed" for g in alice.goals)
    assert len(bob.goals) == 0 or all(g.status == "completed" for g in bob.goals)

    # 5. Log- und Resilienz-Validierung
    logged_events = []
    if Path(log_file).exists():
        import json
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logged_events.append(json.loads(line))

    event_types = [e.get("event_type") for e in logged_events]

    # Keine Kognitions- oder Dialogabbrüche
    assert "dialogue_failed" not in event_types
    assert "interaction_request_expired" not in event_types

    # Phasenwechsel und Ausweichkaskade verifizieren
    assert "dialogue_agreement_confirmed" in event_types or any("offer_yield" in str(e) for e in logged_events)
    assert any(e.get("payload", {}).get("phase") == EvasionPhase.YIELDING_WAIT.value for e in logged_events)
    assert any(e.get("payload", {}).get("phase") == EvasionPhase.CLEARANCE_CONFIRMED.value for e in logged_events)
    assert any(e.get("payload", {}).get("phase") == EvasionPhase.EGRESS.value for e in logged_events)