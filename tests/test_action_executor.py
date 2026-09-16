from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from src.application.services.action_executor import ActionExecutor
from src.application.services.dialogue_history import DialogueHistory
from src.application.services.evasion_finder import EvasionFinder
from src.application.services.goal_service import GoalService
from src.domain.models.agent import Agent
from src.domain.models.cognition import TalkAction
from src.domain.models.goal import Goal
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
from src.infrastructure.pathfinding.astar import AStarPathfinder


@pytest.fixture
def test_setup():
    grid = WorldGrid(width=20, height=20)
    logger = MagicMock()
    dialogue_history = DialogueHistory()
    pathfinder = AStarPathfinder()
    cognition = MagicMock()
    current_tick = 5

    goal_service = GoalService(
        logger=logger,
        cognition_provider=cognition,
        pathfinder=pathfinder,
        tick_provider=lambda: current_tick,
    )
    evasion_finder = EvasionFinder(pathfinder=pathfinder)
    executor = ActionExecutor(
        grid=grid,
        logger=logger,
        dialogue_history=dialogue_history,
        goal_service=goal_service,
        pathfinder=pathfinder,
        evasion_finder=evasion_finder,
        tick_provider=lambda: current_tick,
    )
    return executor, grid, goal_service, dialogue_history


def test_execute_blockage_action_talk_with_blocked_pos_parameter(test_setup) -> None:
    # TC-ACT-01: Regressionstest für Signatur-Kompatibilität (blocked_pos Parameter)
    executor, grid, goal_service, history = test_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    all_entities = [alice, bob]

    action = TalkAction(
        target_agent_id="2",
        message="Bitte ausweichen!",
        reason="Blockade lösen",
        intent="request_yield",
    )

    # Aufruf mit explizitem blocked_pos (darf keinen TypeError werfen)
    executor.execute_blockage_action(
        agent=alice,
        blocker=bob,
        action=action,
        incident_id="inc-test-01",
        all_entities=all_entities,
        blocked_pos=Position(6, 5),
        current_tick=5,
        thought="Weg ist blockiert",
    )

    assert len(bob.inbox) == 1
    incoming = bob.inbox[0]
    assert incoming.message == "Bitte ausweichen!"
    assert incoming.intent == "request_yield"
    assert incoming.from_agent_id == "1"

    # Alice wartet auf Antwort, Bob pausiert im Gespräch
    assert alice.active_goal is not None
    assert alice.active_goal.name == "Warten auf Antwort"
    assert bob.active_goal is not None
    assert bob.active_goal.name == "Konversation mit Alice"


def test_execute_evasion_pushes_goal_and_notifies_partner(test_setup) -> None:
    # TC-ACT-02: Deterministische Ausweichkaskade via EvasionFinder
    executor, grid, goal_service, history = test_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    all_entities = [alice, bob]

    for x in range(10):
        alice.mental_map.update_tile(Position(x, 5), is_walkable=True, tick=1)
    # Nische bei (5, 4)
    alice.mental_map.update_tile(Position(5, 4), is_walkable=True, tick=1)

    executor.execute_evasion(
        agent=alice,
        partner=bob,
        blocked_pos=Position(6, 5),
        all_entities=all_entities,
        incident_id="inc-evade-01",
        thought="Ich weiche nach Norden aus",
    )

    assert alice.active_goal is not None
    assert alice.active_goal.name == "In Nische ausweichen"
    assert alice.active_goal.target_position == Position(5, 4)
    assert alice.active_goal.yield_for_agent_id == "2"

    # Bob erhält Ausweichankündigung
    assert len(bob.inbox) == 1
    notice = bob.inbox[0]
    assert notice.is_evasion_notice is True
    assert "Ich mache Platz" in notice.message


def test_handle_non_conversational_talk(test_setup) -> None:
    # TC-ACT-03: Ansprache eines passiven Objekts erzeugt sofort leere Rückmeldung
    executor, grid, goal_service, history = test_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    rock = WorldEntity(
        id="stone_1",
        name="Großer Stein",
        position=Position(6, 5),
        entity_type="rock",
        is_conversational=False,
    )
    all_entities = [alice, rock]

    action = TalkAction(
        target_agent_id="stone_1",
        message="Hallo Stein, kannst du weggehen?",
        reason="Versuch der Kontaktaufnahme",
    )

    executor.execute_blockage_action(
        agent=alice,
        blocker=rock,
        action=action,
        incident_id="inc-rock-01",
        all_entities=all_entities,
        blocked_pos=Position(6, 5),
        current_tick=5,
    )

    # Alice assimiliert sofort die leere Antwort
    assert alice.memory.get_assumed_conversational("stone_1") is False
    assert alice.memory.can_talk("stone_1") is False