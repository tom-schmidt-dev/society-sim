from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.infrastructure.container import ApplicationContainer
from src.domain.models.agent.agent import Agent
from src.domain.models.world.position import Position
from src.domain.models.planning.cognition import BlockedResolution, DialogueResolution, TalkAction, InspectAction
from src.domain.ports.cognition_provider import ICognitionProvider
from src.main2 import build_corridor


@pytest.mark.asyncio
async def test_corridor_niche_evasion_and_clearance_reawakening() -> None:
    mock_cognition = MagicMock(spec=ICognitionProvider)
    mock_cognition.resolve_blockage = AsyncMock()
    mock_cognition.resolve_blockage.side_effect = [
        BlockedResolution(
            thought="Ich bitte Bob auszuweichen.",
            action=TalkAction(target_agent_id="2", message="Bitte weichen Sie aus!", reason="Chokepoint blockiert", intent="request_yield"),
        ),
        BlockedResolution(
            thought="Ich inspiziere Alice.",
            action=InspectAction(target_agent_id="1", reason="Typ feststellen"),
        ),
    ]

    mock_cognition.respond_to_dialogue = AsyncMock()
    mock_cognition.respond_to_dialogue.side_effect = [
        DialogueResolution(
            thought="Ich weiche in die Nische aus.",
            action=TalkAction(target_agent_id="1", message="Ich weiche aus.", reason="Rolle yield", intent="offer_yield"),
            negotiation_intent="offer_yield",
        ),
        DialogueResolution(
            thought="Danke, ich passiere.",
            action=TalkAction(target_agent_id="2", message="Danke, ich passiere.", reason="Rolle pass", intent="accept"),
            negotiation_intent="accept",
        ),
    ]

    container = ApplicationContainer.build(
        width=90,
        height=45,
        tick_interval=0.0,
        blockage_strategy="action_masking",
        cognition_provider=mock_cognition,
    )
    build_corridor(container)

    alice = Agent(id="1", name="Alice", position=Position(2, 22), entity_type="agent", is_conversational=True)
    container.engine.register_agent(alice)
    container.engine.set_agent_target("1", Position(87, 22), "Ost-Tor")

    bob = Agent(id="2", name="Bob", position=Position(87, 22), entity_type="agent", is_conversational=True)
    container.engine.register_agent(bob)
    container.engine.set_agent_target("2", Position(2, 22), "West-Tor")

    for _ in range(1, 55):
        await container.engine.process_tick()

    # 1. Alice hat den Chokepoint erfolgreich passiert
    assert alice.position.x > 47

    # 2. Bob hat das Clearance-Signal empfangen und die Nische verlassen
    assert bob.active_goal is not None
    assert bob.active_goal.name == "West-Tor"
    assert bob.is_evasion_locked is False
    assert bob.has_path is True

    # 3. Bob ist wieder auf der Korridorlinie (y=22) in Richtung Westen unterwegs
    assert bob.position.y == 22
    assert bob.position.x < 45

    # 4. Im Dialogverlauf ist das Freigabe- und Dankessignal festgehalten
    formatted_history = container.dialogue_history.get_recent_formatted(10)
    assert any("Danke fürs Platz machen!" in entry for entry in formatted_history)
    assert any("Gern geschehen!" in entry for entry in formatted_history)
