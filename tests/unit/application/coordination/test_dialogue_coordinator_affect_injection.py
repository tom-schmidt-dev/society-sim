from unittest.mock import AsyncMock, MagicMock
import pytest

from src.application.services.coordination.dialogue_coordinator import DialogueCoordinator
from src.application.services.coordination.dialogue_history import DialogueHistory
from src.application.services.coordination.dialogue_session_manager import DialogueSessionManager
from src.domain.models.agent.agent import Agent
from src.domain.models.communication.message import CommunicationChannel, IncomingMessage
from src.domain.models.planning.cognition import DialogueResolution, TalkAction
from src.domain.models.world.position import Position
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.vector_memory_store import IVectorMemoryStore
from src.domain.services.affect_service import AffectService


@pytest.fixture
def coordinator_setup():
    logger = MagicMock(spec=IEventLogger)
    cognition_provider = AsyncMock(spec=ICognitionProvider)
    pathfinder = MagicMock(spec=IPathfinder)
    goal_service = MagicMock()
    evasion_finder = MagicMock()
    evasion_finder.compare_evasion_distances.return_value = (None, None)
    action_executor = MagicMock()
    session_manager = DialogueSessionManager()
    dialogue_history = DialogueHistory()
    vector_store = MagicMock(spec=IVectorMemoryStore)
    affect_service = AffectService()

    coordinator = DialogueCoordinator(
        logger=logger,
        cognition_provider=cognition_provider,
        pathfinder=pathfinder,
        goal_service=goal_service,
        evasion_finder=evasion_finder,
        action_executor=action_executor,
        session_manager=session_manager,
        dialogue_history=dialogue_history,
        vector_memory_store=vector_store,
        affect_service=affect_service,
    )

    alice = Agent(id="1", name="Alice", position=Position(14, 4), assertiveness=0.85)
    bob = Agent(id="2", name="Bob", position=Position(15, 4), assertiveness=0.2)

    return coordinator, cognition_provider, affect_service, dialogue_history, vector_store, alice, bob


@pytest.mark.asyncio
async def test_situational_notes_passed_to_context_and_not_stored_in_history(coordinator_setup) -> None:
    coordinator, cognition_provider, affect_service, dialogue_history, vector_store, alice, bob = coordinator_setup

    # Simuliere Vor-Frustration durch vorherige Ablehnungen
    for _ in range(3):
        affect_service.record_turn(agent_id=alice.id, partner_id=bob.id, incoming_intent="reject")

    # Eingehende Nachricht an Alice
    alice.receive_message(
        IncomingMessage(
            from_agent_id=bob.id,
            from_agent_name=bob.name,
            message="Ich weiche nicht aus.",
            channel=CommunicationChannel.LOCAL_TALK,
            intent="reject",
        )
    )

    cognition_provider.respond_to_dialogue.return_value = DialogueResolution(
        thought="Ich beharre auf meinem Vorrang.",
        action=TalkAction(
            target_agent_id=bob.id,
            message="Bitte machen Sie den Weg frei!",
            intent="request_yield",
        ),
        negotiation_intent="request_yield",
    )

    await coordinator.handle_incoming_dialogue(alice, [alice, bob])

    # 1. Verifikation: situational_notes wurden im context an das LLM übergeben
    call_args = cognition_provider.respond_to_dialogue.call_args
    assert call_args is not None
    context = call_args[0][0]
    assert "situational_notes" in context
    notes = context["situational_notes"]
    assert len(notes) > 0
    assert any("Geduld" in note or "Zeitdruck" in note or "Vorrang" in note for note in notes)

    # 2. Verifikation: Ephemerität (Notizen erscheinen nicht im persistenten Gesprächsverlauf)
    recent_dialogues = dialogue_history.get_recent_formatted()
    for note in notes:
        for logged_text in recent_dialogues:
            assert note not in logged_text

    # 3. Verifikation: Keine Kontamination des Vektorspeichers mit flüchtigen Notizen
    for call in vector_store.add_memories.call_args_list:
        memories = call.kwargs.get("memories", [])
        for mem in memories:
            for note in notes:
                assert note not in mem