from unittest.mock import MagicMock
import pytest

from src.application.services.coordination.agent_protocol_service import AgentProtocolService
from src.application.services.coordination.dialogue_history import DialogueHistory
from src.domain.models.agent.agent import Agent
from src.domain.models.communication.interaction_request import InteractionRequest
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder


@pytest.fixture
def service_setup():
    grid = MagicMock(spec=WorldGrid)
    pathfinder = MagicMock(spec=IPathfinder)
    logger = MagicMock(spec=IEventLogger)
    goal_service = MagicMock()
    evasion_finder = MagicMock()
    dialogue_history = DialogueHistory()

    current_tick = 0

    def tick_provider() -> int:
        nonlocal current_tick
        return current_tick

    def set_tick(t: int) -> None:
        nonlocal current_tick
        current_tick = t

    service = AgentProtocolService(
        grid=grid,
        pathfinder=pathfinder,
        logger=logger,
        goal_service=goal_service,
        dialogue_history=dialogue_history,
        evasion_finder=evasion_finder,
        tick_provider=tick_provider,
    )

    alice = Agent(id="1", name="Alice", position=Position(14, 4))
    bob = Agent(id="2", name="Bob", position=Position(15, 4))

    return service, logger, alice, bob, set_tick


def test_interaction_request_does_not_expire_while_target_is_thinking(service_setup) -> None:
    service, logger, alice, bob, set_tick = service_setup

    # Alice reiht Anfrage bei Bob ein
    req = InteractionRequest(
        requester_id=alice.id,
        target_id=bob.id,
        blocked_pos=Position(15, 4),
        tick=0,
        ttl_ticks=3,
    )
    bob.interaction_queue.append(req)

    # Bob denkt (z. B. asynchrone LLM-Inferenz)
    bob.set_thinking(True, reason="Wägt Blockadelösung ab")
    set_tick(50)

    service.process_interaction_queue(bob, [alice, bob], background_tasks=set())

    # Anfrage muss unangetastet in Bobs Queue verbleiben
    assert len(bob.interaction_queue) == 1
    # Es darf kein Expired-Event geloggt worden sein
    for call in logger.log.call_args_list:
        event = call[0][0]
        assert event.event_type != "interaction_request_expired"


def test_interaction_request_does_not_expire_while_requester_is_thinking(service_setup) -> None:
    service, logger, alice, bob, set_tick = service_setup

    req = InteractionRequest(
        requester_id=alice.id,
        target_id=bob.id,
        blocked_pos=Position(15, 4),
        tick=0,
        ttl_ticks=3,
    )
    bob.interaction_queue.append(req)

    # Bob ist frei, aber Alice denkt
    bob.set_thinking(False)
    alice.set_thinking(True, reason="Führt anderen Gedankengang")
    set_tick(50)

    service.process_interaction_queue(bob, [alice, bob], background_tasks=set())

    # Anfrage muss in der Queue bleiben, bis Alice wieder aufnahmebereit ist
    assert len(bob.interaction_queue) == 1
    for call in logger.log.call_args_list:
        event = call[0][0]
        assert event.event_type != "interaction_request_expired"


def test_interaction_request_expires_normally_when_nobody_is_thinking(service_setup) -> None:
    service, logger, alice, bob, set_tick = service_setup

    req = InteractionRequest(
        requester_id=alice.id,
        target_id=bob.id,
        blocked_pos=Position(15, 4),
        tick=0,
        ttl_ticks=3,
    )
    bob.interaction_queue.append(req)

    bob.set_thinking(False)
    alice.set_thinking(False)
    set_tick(4)  # 4 > 0 + 3 -> abgelaufen

    service.process_interaction_queue(bob, [alice, bob], background_tasks=set())

    # Anfrage wurde ordnungsgemäß wegen Inaktivität verworfen
    assert len(bob.interaction_queue) == 0
    logged_event_types = [call[0][0].event_type for call in logger.log.call_args_list]
    assert "interaction_request_expired" in logged_event_types