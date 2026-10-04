from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.application.services.cognition.cognition_orchestrator import CognitionOrchestrator
from src.application.services.cognition.goal_service import GoalService
from src.application.services.cognition.plan_decomposition_service import PlanDecompositionService
from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.lifecycle.need_service import NeedService
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.goal import Goal
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.services.precondition_evaluator import PreconditionEvaluator
from tests.integration.test_level2_plan_decomposition_with_store import MockCognitionProvider


class SilentLogger(IEventLogger):
    def log(self, event: object) -> None:
        pass


@pytest.mark.asyncio
async def test_level4_adjacent_consume_execution() -> None:
    grid = WorldGrid(width=10, height=10)
    logger = SilentLogger()
    pathfinder = MagicMock(spec=IPathfinder)
    precondition_evaluator = PreconditionEvaluator()
    need_service = NeedService()
    cognition = MockCognitionProvider()
    goal_service = GoalService(logger=logger, cognition_provider=cognition, pathfinder=pathfinder)

    action_executor = ActionExecutor(
        grid=grid,
        logger=logger,
        dialogue_history=MagicMock(),
        goal_service=goal_service,
        pathfinder=pathfinder,
        perception_service=MagicMock(),
        evasion_finder=MagicMock(),
        critical_section_coordinator=MagicMock(),
        need_service=need_service,
        tick_provider=lambda: 1,
    )

    orchestrator = CognitionOrchestrator(
        pathfinder=pathfinder,
        logger=logger,
        goal_service=goal_service,
        need_service=need_service,
        plan_decomposition_service=MagicMock(spec=PlanDecompositionService),
        action_executor=action_executor,
        precondition_evaluator=precondition_evaluator,
        tick_provider=lambda: 1,
    )

    # Entität bei (6, 5)
    apple_tree = WorldEntity(
        id="tree_1",
        name="Apfelbaum",
        position=Position(6, 5),
        entity_type="food",
        is_conversational=False,
    )
    setattr(apple_tree, "is_consumable", True)
    setattr(apple_tree, "nutrition_value", 0.7)
    setattr(apple_tree, "is_depletable", False)

    # Agent steht benachbart auf (5, 5) mit Hunger 0.80
    agent = Agent(id="a1", name="Alice", position=Position(5, 5))
    agent.needs["hunger"] = 0.80
    agent.memory.update_entity_perception("tree_1", "Apfelbaum", Position(6, 5), 1, "food")

    consume_goal = Goal(name="SubGoal: consume", target_entity_id="tree_1", target_position=Position(6, 5))
    agent.goals.append(consume_goal)

    # Ausführung
    success = orchestrator.handle_subgoal_consume(agent, consume_goal, entities=[apple_tree])

    assert success is True
    # Hunger muss um exakt 0.70 gesunken sein
    assert pytest.approx(agent.needs["hunger"], rel=1e-3) == 0.10
    # Ziel muss erledigt und gepoppt sein
    assert len(agent.goals) == 0