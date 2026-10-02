from unittest.mock import AsyncMock
import pytest

from src.domain.models.agent.agent import Agent
from src.domain.models.planning.cognition import (
    BlockedResolution,
    DialogueResolution,
    GoalEvaluation,
    GoalIntent,
    SocialReflection,
    TalkAction,
)
from src.domain.models.world.position import Position
from src.domain.ports.cognition_provider import ICognitionProvider
from src.infrastructure.container import ApplicationContainer


@pytest.mark.asyncio
async def test_corridor_negotiation_full_cycle_no_exceptions() -> None:
    cognition_mock = AsyncMock(spec=ICognitionProvider)

    # 1. Mock für die initiale Blockadeentscheidung
    async def mock_resolve_blockage(context: dict) -> BlockedResolution:
        blocker_id = context.get("blocker_id")
        return BlockedResolution(
            thought="Weg blockiert, Verhandlung einleiten.",
            action=TalkAction(
                target_agent_id=str(blocker_id),
                message="Bitte weichen Sie aus!",
                intent="request_yield",
            ),
        )

    cognition_mock.resolve_blockage.side_effect = mock_resolve_blockage

    # 2. Mock für den Verhandlungsdialog
    def mock_respond(context: dict) -> DialogueResolution:
        agent_id = context.get("agent_id")
        incoming_intent = context.get("incoming_intent")

        if agent_id == "1":  # Alice (dominant)
            if incoming_intent == "offer_yield" or context.get("partner_is_yielding"):
                return DialogueResolution(
                    thought="Danke für das Ausweichen, ich passiere.",
                    action=TalkAction(
                        target_agent_id="2",
                        message="Danke, ich passiere.",
                        intent="accept",
                    ),
                    negotiation_intent="accept",
                )
            return DialogueResolution(
                thought="Ich passiere und fordere Vorrang.",
                action=TalkAction(
                    target_agent_id="2",
                    message="Bitte weichen Sie aus!",
                    intent="request_yield",
                ),
                negotiation_intent="request_yield",
            )
        else:  # Bob (kooperativ)
            if incoming_intent in ("request_yield", "reject"):
                return DialogueResolution(
                    thought="Ich gebe nach und weiche in die Nische aus.",
                    action=TalkAction(
                        target_agent_id="1",
                        message="In Ordnung, ich trete in die Nische.",
                        intent="offer_yield",
                    ),
                    negotiation_intent="offer_yield",
                    new_goal=GoalIntent(name="In Nische ausweichen", intent_type="evade"),
                )
            return DialogueResolution(
                thought="Warte ab.",
                action=TalkAction(
                    target_agent_id="1",
                    message="Gehe vor.",
                    intent="accept",
                ),
                negotiation_intent="accept",
            )

    cognition_mock.respond_to_dialogue.side_effect = mock_respond
    cognition_mock.reflect_on_dialogue.return_value = SocialReflection(
        assessment="kooperativ",
        progression_summary="Einigung erzielt, Bob weicht aus.",
    )
    cognition_mock.evaluate_goal_status.return_value = GoalEvaluation(
        thought="Ziel aktiv",
        is_completed=False,
        reason="Unterwegs",
    )

    # 3. Container initialisieren
    container = ApplicationContainer.build(
        width=30,
        height=9,
        tick_interval=0.01,
        auditory_radius=4,
        enable_deterministic_corridor=False,
        enable_day_night=False,
        cognition_provider=cognition_mock,
    )

    grid = container.grid
    width = 30
    height = 9

    # Korridorwände bei y=3 und y=5 (Nische bei x=15, y=3)
    for x in range(5, 25):
        if x != 15:
            grid.set_obstacle(Position(x, 3))
        grid.set_obstacle(Position(x, 5))

    # Nischeneinfassung bei (15, 3)
    grid.set_obstacle(Position(14, 2))
    grid.set_obstacle(Position(15, 1))
    grid.set_obstacle(Position(16, 2))

    # Agenten instanziieren
    alice = Agent(
        id="1",
        name="Alice",
        position=Position(14, 4),
        assertiveness=0.85,
        charisma=0.7,
        is_conversational=True,
    )
    bob = Agent(
        id="2",
        name="Bob",
        position=Position(15, 4),
        assertiveness=0.2,
        charisma=0.6,
        is_conversational=True,
    )

    # 4. Mentale Karten vorab mit Korridorwissen synchronisieren
    alice.mental_map.set_bounds(width, height)
    bob.mental_map.set_bounds(width, height)
    for x in range(width):
        for y in range(height):
            pos = Position(x, y)
            is_w = grid.is_walkable(pos)
            alice.mental_map.update_tile(pos, is_walkable=is_w, tick=0)
            bob.mental_map.update_tile(pos, is_walkable=is_w, tick=0)

    # Agenten und Zielpositionen registrieren
    container.engine.register_agent(alice)
    container.engine.set_agent_target("1", Position(27, 4), "Ost-Portal")

    container.engine.register_agent(bob)
    container.engine.set_agent_target("2", Position(2, 4), "West-Portal")

    # 5. Simulationsschleife
    for _ in range(40):
        await container.engine.process_tick()
        if alice.position.x > 15:
            break

    # 6. Validierung
    assert alice.position.x > 15
    assert not alice.is_thinking
    assert not bob.is_thinking