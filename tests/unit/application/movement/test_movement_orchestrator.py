from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from src.application.services.coordination.agent_protocol_service import AgentProtocolService
from src.application.services.movement.movement_sync_service import MovementSyncService
from src.application.services.movement.movement_orchestrator import MovementOrchestrator
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.goal import ExecutionPriority, Goal
from src.domain.models.world.position import Position
from src.domain.models.coordination.reservation_table import ReservationTable
from src.domain.models.world.world import WorldGrid
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.conflict_coordinator import IConflictCoordinator
from src.domain.ports.pathfinder import IPathfinder


@pytest.fixture
def movement_orchestrator_setup():
    grid = WorldGrid(width=10, height=10)
    for x in range(10):
        for y in range(10):
            grid.remove_obstacle(Position(x, y))

    pathfinder = MagicMock(spec=IPathfinder)
    pathfinder.find_path.return_value = [Position(2, 1)]
    movement_sync = MagicMock(spec=MovementSyncService)
    reservation_table = ReservationTable()
    protocol_service = MagicMock(spec=AgentProtocolService)
    conflict_coordinator = MagicMock(spec=IConflictCoordinator)
    on_agent_moved = MagicMock()

    orchestrator = MovementOrchestrator(
        grid=grid,
        pathfinder=pathfinder,
        movement_sync_service=movement_sync,
        reservation_table=reservation_table,
        protocol_service=protocol_service,
        conflict_coordinator=conflict_coordinator,
        on_agent_moved=on_agent_moved,
        tick_provider=lambda: 1,
    )

    return (
        orchestrator,
        grid,
        pathfinder,
        movement_sync,
        reservation_table,
        protocol_service,
        conflict_coordinator,
        on_agent_moved,
    )


class TestMovementOrchestrator:
    def test_execute_movement_phase_requests_intents_and_handles_commit(
        self, movement_orchestrator_setup
    ) -> None:
        (
            orchestrator,
            grid,
            _,
            movement_sync,
            reservation_table,
            protocol_service,
            _,
            on_agent_moved,
        ) = movement_orchestrator_setup

        agent = Agent(id="a1", name="Alice", position=Position(1, 1))
        agent.assign_path([Position(2, 1)])
        agents = [agent]
        entities: list[WorldEntity] = [agent]
        background_tasks: set[asyncio.Task] = set()

        sync_result = MagicMock()
        sync_result.committed_agents = {"a1"}
        movement_sync.execute_two_phase_commit.return_value = sync_result

        orchestrator.execute_physical_movement(
            agents=agents,
            entities=entities,
            delayed_agent_ids=set(),
            background_tasks=background_tasks,
        )

        assert movement_sync.execute_two_phase_commit.call_count == 1
        on_agent_moved.assert_called_once_with(agent)
        protocol_service.check_and_signal_clearance.assert_called_once_with(agent, entities)

    def test_execute_movement_phase_skips_delayed_agents(
        self, movement_orchestrator_setup
    ) -> None:
        orchestrator, _, _, movement_sync, reservation_table, _, _, _ = movement_orchestrator_setup

        agent = Agent(id="a1", name="Alice", position=Position(1, 1))
        agent.assign_path([Position(2, 1)])
        agents = [agent]
        entities: list[WorldEntity] = [agent]

        sync_result = MagicMock()
        sync_result.committed_agents = set()
        movement_sync.execute_two_phase_commit.return_value = sync_result

        orchestrator.execute_physical_movement(
            agents=agents,
            entities=entities,
            delayed_agent_ids={"a1"},
            background_tasks=set(),
        )

        intents_arg = movement_sync.execute_two_phase_commit.call_args[1]["intents"]
        assert len(intents_arg) == 0

    def test_handle_committed_agent_signals_niche_events(
        self, movement_orchestrator_setup
    ) -> None:
        orchestrator, _, _, _, _, protocol_service, _, on_agent_moved = movement_orchestrator_setup

        agent = Agent(id="a1", name="Alice", position=Position(2, 2))
        goal = Goal(
            name="Ausweichen",
            target_position=Position(2, 2),
            junction_position=Position(1, 2),
            yield_for_agent_id="a2",
        )
        agent.push_goal(goal)
        entities: list[WorldEntity] = [agent]

        orchestrator.handle_committed_agent_post_move(agent, entities)

        on_agent_moved.assert_called_once_with(agent)
        protocol_service.signal_niche_junction_entry.assert_called_once_with(agent, goal, entities)
        protocol_service.handle_niche_arrival.assert_called_once_with(agent, goal, entities)
        protocol_service.check_and_signal_clearance.assert_called_once_with(agent, entities)

    @pytest.mark.asyncio
    async def test_handle_blocked_agent_delegates_to_conflict_coordinator(
            self, movement_orchestrator_setup
    ) -> None:
        (
            orchestrator,
            _,
            _,
            _,
            _,
            _,
            conflict_coordinator,
            _,
        ) = movement_orchestrator_setup

        agent = Agent(id="a1", name="Alice", position=Position(1, 1))
        agent.assign_path([Position(2, 1)])
        blocker = Agent(id="a2", name="Bob", position=Position(2, 1))
        entities: list[WorldEntity] = [agent, blocker]
        background_tasks: set[asyncio.Task] = set()

        conflict_coordinator.resolve_blockage = AsyncMock()

        orchestrator.handle_blocked_agent(agent, entities, background_tasks)

        assert agent.is_thinking is True
        assert len(background_tasks) == 1


    def test_handle_blocked_agent_replans_on_static_obstacle(
        self, movement_orchestrator_setup
    ) -> None:
        orchestrator, grid, pathfinder, _, _, _, _, _ = movement_orchestrator_setup

        agent = Agent(id="a1", name="Alice", position=Position(1, 1))
        agent.assign_path([Position(2, 1)])
        agent.push_goal(Goal(name="Ziel", target_position=Position(5, 1)))

        grid.set_obstacle(Position(2, 1))
        pathfinder.find_path.return_value = [Position(1, 2), Position(5, 1)]

        orchestrator.handle_blocked_agent(agent, [agent], set())

        assert agent.mental_map.is_walkable(Position(2, 1)) is False
        pathfinder.find_path.assert_called_once_with(
            Position(1, 1), Position(5, 1), agent.mental_map
        )
        assert agent.path == [Position(1, 2), Position(5, 1)]