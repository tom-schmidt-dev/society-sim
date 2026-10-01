from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.cognition.cognition_orchestrator import CognitionOrchestrator
from src.application.services.cognition.frontier_explorer import FrontierExplorer
from src.application.services.cognition.goal_service import GoalService
from src.application.services.lifecycle.need_service import NeedService
from src.application.services.cognition.plan_decomposition_service import PlanDecompositionService
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_memory import EntityFact
from src.domain.models.planning.goal import Goal
from src.domain.models.world.position import Position
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.services.precondition_evaluator import PreconditionEvaluator


@pytest.fixture
def cognition_orchestrator_setup():
    pathfinder = MagicMock(spec=IPathfinder)
    logger = MagicMock(spec=IEventLogger)
    goal_service = MagicMock(spec=GoalService)
    need_service = MagicMock(spec=NeedService)
    plan_decomposition_service = MagicMock(spec=PlanDecompositionService)
    action_executor = MagicMock(spec=ActionExecutor)
    frontier_explorer = MagicMock(spec=FrontierExplorer)
    precondition_evaluator = MagicMock(spec=PreconditionEvaluator)

    orchestrator = CognitionOrchestrator(
        pathfinder=pathfinder,
        logger=logger,
        goal_service=goal_service,
        need_service=need_service,
        plan_decomposition_service=plan_decomposition_service,
        action_executor=action_executor,
        frontier_explorer=frontier_explorer,
        precondition_evaluator=precondition_evaluator,
        tick_provider=lambda: 5,
    )

    return (
        orchestrator,
        pathfinder,
        logger,
        goal_service,
        need_service,
        plan_decomposition_service,
        action_executor,
        frontier_explorer,
        precondition_evaluator,
    )


class TestCognitionOrchestrator:
    @pytest.mark.asyncio
    async def test_process_agent_needs_triggers_decomposition_on_urgent_need(
        self, cognition_orchestrator_setup
    ) -> None:
        (
            orchestrator,
            _,
            _,
            goal_service,
            need_service,
            plan_decomp_service,
            _,
            _,
            _,
        ) = cognition_orchestrator_setup

        agent = Agent(id="a1", name="Alice", position=Position(1, 1))
        need_service.is_need_urgent.return_value = True

        mock_plan = MagicMock()
        mock_plan.primary_goal = "Nahrungsbeschaffung"
        intent_mock = MagicMock()
        intent_mock.target_position = (2, 2)
        intent_mock.target_entity_id = "apple_1"
        intent_mock.action_type.value = "consume"
        intent_mock.description = "Apfel essen"
        mock_plan.sub_goals = [intent_mock]

        plan_decomp_service.create_plan_for_need = AsyncMock(return_value=mock_plan)

        await orchestrator.process_agent_needs_and_cognition(agents=[agent], entities=[agent])

        plan_decomp_service.create_plan_for_need.assert_awaited_once_with(agent, "hunger")
        assert goal_service.push_goal.call_count == 2
        need_service.update_needs.assert_called_once_with(agent)

    def test_handle_subgoal_move_to_assigns_path(
        self, cognition_orchestrator_setup
    ) -> None:
        orchestrator, pathfinder, _, _, _, _, _, _, _ = cognition_orchestrator_setup

        agent = Agent(id="a1", name="Alice", position=Position(1, 1))
        goal = Goal(name="SubGoal: move_to", target_position=Position(3, 1))
        pathfinder.find_path.return_value = [Position(2, 1), Position(3, 1)]

        orchestrator.handle_subgoal_move_to(agent, goal)

        pathfinder.find_path.assert_called_once_with(
            Position(1, 1), Position(3, 1), agent.mental_map
        )
        assert agent.path == [Position(2, 1), Position(3, 1)]

    def test_handle_subgoal_explore_interrupts_when_resource_discovered(
        self, cognition_orchestrator_setup
    ) -> None:
        (
            orchestrator,
            _,
            logger,
            _,
            _,
            _,
            _,
            _,
            precondition_evaluator,
        ) = cognition_orchestrator_setup

        agent = Agent(id="a1", name="Alice", position=Position(1, 1))
        goal = Goal(name="SubGoal: explore", target_position=Position(5, 5))
        agent.push_goal(goal)
        agent.assign_path([Position(2, 1)])

        fact = EntityFact(
            entity_id="food_1",
            name="Beere",
            last_known_position=Position(2, 2),
            entity_type="food",
        )
        precondition_evaluator.find_discovered_entity.return_value = fact

        orchestrator.handle_subgoal_explore(agent, goal)

        assert not agent.goals
        assert not agent.has_path
        event = next(
            e
            for e in [call_args[0][0] for call_args in logger.log.call_args_list]
            if e.event_type == "goal_interrupted_for_replan"
        )
        assert event.payload["discovered_entity_id"] == "food_1"

    def test_handle_subgoal_consume_executes_action_and_cleans_goals(
        self, cognition_orchestrator_setup
    ) -> None:
        (
            orchestrator,
            _,
            _,
            goal_service,
            need_service,
            _,
            action_executor,
            _,
            _,
        ) = cognition_orchestrator_setup

        agent = Agent(id="a1", name="Alice", position=Position(1, 1))
        target_item = WorldEntity(id="apple_1", name="Apfel", position=Position(1, 1))
        goal = Goal(name="SubGoal: consume", target_entity_id="apple_1")
        agent.push_goal(goal)

        action_executor.execute_consume.return_value = True
        need_service.is_need_urgent.return_value = False

        result = orchestrator.handle_subgoal_consume(
            agent=agent, current_goal=goal, entities=[agent, target_item]
        )

        assert result is True
        action_executor.execute_consume.assert_called_once()
        assert goal_service.pop_goal.call_count >= 1

    @pytest.mark.asyncio
    async def test_vital_values_not_updated_when_consumed(
        self, cognition_orchestrator_setup
    ) -> None:
        (
            orchestrator,
            _,
            _,
            _,
            need_service,
            _,
            action_executor,
            _,
            _,
        ) = cognition_orchestrator_setup

        agent = Agent(id="a1", name="Alice", position=Position(1, 1))
        target_item = WorldEntity(id="apple_1", name="Apfel", position=Position(1, 1))
        goal = Goal(name="SubGoal: consume", target_entity_id="apple_1")
        agent.push_goal(goal)

        need_service.is_need_urgent.return_value = False
        action_executor.execute_consume.return_value = True

        await orchestrator.process_agent_needs_and_cognition(
            agents=[agent], entities=[agent, target_item]
        )

        need_service.update_needs.assert_not_called()