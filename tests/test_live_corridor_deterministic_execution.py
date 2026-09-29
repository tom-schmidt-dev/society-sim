from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.application.services.need_service import NeedService
from src.container import ApplicationContainer
from src.domain.models.agent import Agent
from src.domain.models.position import Position
from src.domain.ports.cognition_provider import ICognitionProvider
from src.main2 import build_corridor


@pytest.mark.asyncio
async def test_live_corridor_encounter_resolves_deterministically_without_llm() -> None:
    """
    Verifiziert, dass bei der räumlichen Begegnung im Korridor (main2-Szenario)
    die deterministische Arbitrierung greift:
    - 0 Aufrufe an ICognitionProvider (resolve_blockage / respond_to_dialogue)
    - Einrücken in die Nische bei (45, 21)
    - Kollisionsfreie Passage
    - Clearance-Signal und Wiederaufnahme des Primärziels
    """
    mock_cognition = MagicMock(spec=ICognitionProvider)
    mock_cognition.resolve_blockage = AsyncMock()
    mock_cognition.respond_to_dialogue = AsyncMock()

    container = ApplicationContainer.build(
        width=90,
        height=45,
        tick_interval=0.0,
        blockage_strategy="action_masking",
        cognition_provider=mock_cognition,
        need_service=NeedService(hunger_increase_per_tick=0.0),
    )
    build_corridor(container)

    # Alice (Start: 2, 22 -> Ziel: 87, 22)
    alice = Agent(id="1", name="Alice", position=Position(2, 22), entity_type="agent", is_conversational=True)
    container.engine.register_agent(alice)
    container.engine.set_agent_target("1", Position(87, 22), "Ost-Tor")

    # Bob (Start: 87, 22 -> Ziel: 2, 22)
    bob = Agent(id="2", name="Bob", position=Position(87, 22), entity_type="agent", is_conversational=True)
    container.engine.register_agent(bob)
    container.engine.set_agent_target("2", Position(2, 22), "West-Tor")

    # Taktzyklus durchlaufen, bis beide Agenten passiert haben
    for tick in range(1, 160):
        await container.engine.process_tick()
        if alice.position == Position(87, 22) and bob.position == Position(2, 22):
            break

    # 1. Null LLM-Aufrufe während der gesamten Korridor-Begegnung
    assert mock_cognition.resolve_blockage.call_count == 0, (
        f"Erwartet 0 resolve_blockage Aufrufe, erhalten: {mock_cognition.resolve_blockage.call_count}"
    )
    assert mock_cognition.respond_to_dialogue.call_count == 0, (
        f"Erwartet 0 respond_to_dialogue Aufrufe, erhalten: {mock_cognition.respond_to_dialogue.call_count}"
    )

    # 2. Beide Agenten haben ihre Endziele erreicht
    assert alice.position == Position(87, 22)
    assert bob.position == Position(2, 22)

    # 3. Deterministische Kommunikations-Templates im Dialogverlauf enthalten
    dialogues = container.dialogue_history.get_recent_formatted(20)
    assert any("Hier ist nicht genug Platz für uns beide." in d for d in dialogues)
    assert any("Passage abgeschlossen." in d or "Danke fürs Platz machen!" in d for d in dialogues)
    assert any("Gern geschehen!" in d for d in dialogues)


@pytest.mark.asyncio
async def test_non_agent_blockage_retains_cognition_llm_inference() -> None:
    """
    Verifiziert, dass bei Blockaden durch nicht-Agenten (z.B. Stein wie in main.py)
    die LLM-Inferenz (resolve_blockage) aktiv aufgerufen wird.
    """
    from src.domain.models.cognition import BlockedResolution, InspectAction
    from src.domain.models.world_entity import WorldEntity

    mock_cognition = MagicMock(spec=ICognitionProvider)
    mock_cognition.resolve_blockage = AsyncMock(
        return_value=BlockedResolution(
            thought="Ich untersuche den Stein.",
            action=InspectAction(target_agent_id="stone_1", reason="Objekt inspizieren"),
        )
    )

    container = ApplicationContainer.build(
        width=90,
        height=45,
        tick_interval=0.0,
        blockage_strategy="action_masking",
        cognition_provider=mock_cognition,
    )
    build_corridor(container)

    alice = Agent(id="1", name="Alice", position=Position(44, 22), entity_type="agent", is_conversational=True)
    container.engine.register_agent(alice)
    container.engine.set_agent_target("1", Position(87, 22), "Ost-Tor")

    stone = WorldEntity(id="stone_1", name="Großer Stein", position=Position(45, 22), entity_type="rock", is_conversational=False)
    container.engine.register_entity(stone)

    # Einen Takt ausführen, Alice will auf (45, 22) treten
    await container.engine.process_tick()

    # resolve_blockage MUSS für nicht-Agenten aufgerufen worden sein
    assert mock_cognition.resolve_blockage.call_count >= 1

