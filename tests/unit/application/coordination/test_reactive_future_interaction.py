from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from src.application.services.coordination.agent_protocol_service import AgentProtocolService
from src.application.services.coordination.conflict_coordinator import ConflictCoordinator
from src.application.services.coordination.dialogue_history import DialogueHistory
from src.application.services.coordination.dialogue_session_manager import DialogueSessionManager
from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.interaction.interaction_dispatcher import InteractionDispatcher
from src.application.services.movement.evasion_finder import EvasionFinder
from src.application.services.cognition.goal_service import GoalService
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_state import AgentLifecycleState
from src.domain.models.interaction.commands import TalkCommand
from src.domain.models.interaction.interaction_result import ImmediateResult, PendingFuture
from src.domain.models.planning.cognition import BlockedResolution, WaitAction
from src.domain.models.planning.goal import Goal
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.infrastructure.pathfinding.astar import AStarPathfinder


@pytest.fixture
def reactive_setup():
    grid = WorldGrid(width=20, height=20)
    for x in range(20):
        for y in range(20):
            grid.remove_obstacle(Position(x, y))

    logger = MagicMock()
    cognition = AsyncMock()
    pathfinder = AStarPathfinder()
    tick_box = [10]

    def current_tick() -> int:
        return tick_box[0]

    goal_service = GoalService(
        logger=logger,
        cognition_provider=cognition,
        pathfinder=pathfinder,
        tick_provider=current_tick,
    )
    evasion_finder = EvasionFinder(pathfinder=pathfinder)
    dialogue_history = DialogueHistory()
    session_manager = DialogueSessionManager(max_dialogue_turns=2)

    executor = ActionExecutor(
        grid=grid,
        logger=logger,
        dialogue_history=dialogue_history,
        goal_service=goal_service,
        pathfinder=pathfinder,
        evasion_finder=evasion_finder,
        tick_provider=current_tick,
    )

    dispatcher = InteractionDispatcher(logger=logger, tick_provider=current_tick)

    coordinator = ConflictCoordinator(
        logger=logger,
        cognition_provider=cognition,
        pathfinder=pathfinder,
        goal_service=goal_service,
        evasion_finder=evasion_finder,
        action_executor=executor,
        interaction_dispatcher=dispatcher,
        session_manager=session_manager,
        dialogue_history=dialogue_history,
        tick_provider=current_tick,
        enable_deterministic_corridor=False,
    )

    protocol_service = AgentProtocolService(
        grid=grid,
        pathfinder=pathfinder,
        logger=logger,
        goal_service=goal_service,
        dialogue_history=dialogue_history,
        evasion_finder=evasion_finder,
        conflict_coordinator=coordinator,
        tick_provider=current_tick,
    )

    return coordinator, dispatcher, cognition, goal_service, protocol_service, tick_box


@pytest.mark.asyncio
async def test_free_partner_initiates_pending_future_and_waiting_state(reactive_setup) -> None:
    _, dispatcher, _, _, _, _ = reactive_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    bob = Agent(id="2", name="Bob", position=Position(6, 5), is_conversational=True)

    cmd = TalkCommand(
        source_entity_id=alice.id,
        target_entity_id=bob.id,
        message="Bitte Platz machen",
        intent="request_yield",
    )
    res = dispatcher.dispatch(alice, bob, cmd)

    assert isinstance(res, PendingFuture)
    assert alice.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER
    assert len(bob.interaction_mailbox) == 1


@pytest.mark.asyncio
async def test_busy_partner_returns_immediate_busy_result(reactive_setup) -> None:
    _, dispatcher, _, _, _, _ = reactive_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    bob = Agent(id="2", name="Bob", position=Position(6, 5), is_conversational=True)
    bob.interaction_partner_id = "other_agent"

    cmd = TalkCommand(
        source_entity_id=alice.id,
        target_entity_id=bob.id,
        message="Hallo?",
        intent="request_yield",
    )
    res = dispatcher.dispatch(alice, bob, cmd)

    assert isinstance(res, ImmediateResult)
    assert res.success is False
    assert res.reason == "BUSY"
    assert alice.lifecycle_state == AgentLifecycleState.IDLE


@pytest.mark.asyncio
async def test_conflict_coordinator_passes_time_unit_metrics_when_busy(reactive_setup) -> None:
    coordinator, _, cognition, _, _, tick_box = reactive_setup
    tick_box[0] = 50

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    alice.assign_path([Position(6, 5), Position(7, 5), Position(8, 5)])
    alice.push_goal(Goal(name="Zielort", target_position=Position(8, 5)))

    bob = Agent(id="2", name="Bob", position=Position(6, 5), is_conversational=True)
    bob.interaction_partner_id = "charlie"

    all_entities = [alice, bob]

    cognition.resolve_blockage.return_value = BlockedResolution(
        thought="Partner ist beschäftigt, ich warte.",
        action=WaitAction(ticks=2, reason="Partner beschäftigt"),
    )

    await coordinator.resolve_blockage(
        agent=alice,
        blocker=bob,
        blocked_pos=Position(6, 5),
        all_entities=all_entities,
    )

    cognition.resolve_blockage.assert_awaited_once()
    context = cognition.resolve_blockage.call_args[0][0]

    assert context["blocker_busy"] is True
    assert context["restweg_aktuell_zeiteinheiten"] == 3
    assert context["alternativweg_gesamt_zeiteinheiten"] is not None
    assert context["umweg_mehr_zeiteinheiten"] is not None
    assert context["bisher_gewartete_zeiteinheiten"] == 0

    assert alice.active_goal is not None
    assert alice.active_goal.name == "Warten auf Partner"
    assert alice.active_goal.remaining_ticks == 200
    assert alice.active_goal.initial_wait_tick == 50


@pytest.mark.asyncio
async def test_consecutive_wait_calculates_exact_elapsed_wait_time(reactive_setup) -> None:
    coordinator, _, cognition, goal_service, _, tick_box = reactive_setup
    tick_box[0] = 100

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    alice.assign_path([Position(6, 5)])
    alice.push_goal(Goal(name="Hauptziel", target_position=Position(6, 5)))

    wait_goal = Goal(
        name="Warten auf Partner",
        target_position=None,
        remaining_ticks=200,
        yield_for_agent_id="2",
        initial_wait_tick=80,
    )
    goal_service.push_goal(alice, wait_goal)

    bob = Agent(id="2", name="Bob", position=Position(6, 5), is_conversational=True)
    bob.interaction_partner_id = "charlie"

    all_entities = [alice, bob]

    cognition.resolve_blockage.return_value = BlockedResolution(
        thought="Warte erneut.",
        action=WaitAction(ticks=2, reason="Noch beschäftigt"),
    )

    await coordinator.resolve_blockage(
        agent=alice,
        blocker=bob,
        blocked_pos=Position(6, 5),
        all_entities=all_entities,
    )

    context = cognition.resolve_blockage.call_args[0][0]
    assert context["bisher_gewartete_zeiteinheiten"] == 20
    assert alice.active_goal.initial_wait_tick == 80


def test_fast_path_interrupts_wait_goal_when_blocker_frees(reactive_setup) -> None:
    _, _, _, goal_service, protocol_service, tick_box = reactive_setup
    tick_box[0] = 30

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    alice.assign_path([Position(6, 5)])
    alice.push_goal(Goal(name="Hauptziel", target_position=Position(6, 5)))

    wait_goal = Goal(
        name="Warten auf Partner",
        remaining_ticks=190,
        yield_for_agent_id="2",
        initial_wait_tick=20,
    )
    goal_service.push_goal(alice, wait_goal)

    bob = Agent(id="2", name="Bob", position=Position(6, 5), is_conversational=True)
    bob.interaction_partner_id = None
    bob.is_thinking = False

    protocol_service.check_waiting_partner_fast_path(alice, [alice, bob])

    assert alice.active_goal.name == "Hauptziel"


def test_fast_path_interrupts_wait_goal_when_blocker_clears_tile(reactive_setup) -> None:
    _, _, _, goal_service, protocol_service, tick_box = reactive_setup
    tick_box[0] = 30

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    alice.assign_path([Position(6, 5)])
    alice.push_goal(Goal(name="Hauptziel", target_position=Position(6, 5)))

    wait_goal = Goal(
        name="Warten auf Partner",
        remaining_ticks=190,
        yield_for_agent_id="2",
        initial_wait_tick=20,
    )
    goal_service.push_goal(alice, wait_goal)

    bob = Agent(id="2", name="Bob", position=Position(7, 5), is_conversational=True)
    bob.interaction_partner_id = "charlie"

    protocol_service.check_waiting_partner_fast_path(alice, [alice, bob])

    assert alice.active_goal.name == "Hauptziel"