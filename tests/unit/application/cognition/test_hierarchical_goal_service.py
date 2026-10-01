from __future__ import annotations

from unittest.mock import MagicMock
from src.application.services.cognition.goal_service import GoalService
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.goal import Goal
from src.domain.models.planning.planning import ActionType, PlanDecomposition, SubGoalIntent
from src.domain.models.world.position import Position
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder


def test_apply_plan_to_agent_goals():
    mock_logger = MagicMock(spec=IEventLogger)
    mock_cognition = MagicMock(spec=ICognitionProvider)
    mock_pathfinder = MagicMock(spec=IPathfinder)

    goal_service = GoalService(
        logger=mock_logger,
        cognition_provider=mock_cognition,
        pathfinder=mock_pathfinder,
        tick_provider=lambda: 0,
    )

    agent = Agent(id="1", name="Alice", position=Position(2, 2))
    plan = PlanDecomposition(
        thought="Hunger beheben durch Konsum.",
        primary_goal="Hunger stillen",
        sub_goals=[
            SubGoalIntent(
                action_type=ActionType.MOVE_TO,
                target_entity_id="apple_1",
                target_position=(2, 3),
            ),
            SubGoalIntent(
                action_type=ActionType.CONSUME,
                target_entity_id="apple_1",
            ),
        ],
    )

    # 1. Übergeordnetes Primärziel anlegen
    primary_goal = Goal(name=plan.primary_goal)
    goal_service.push_goal(agent, primary_goal)

    # 2. Teilziele sequentiell ablegen (LIFO-Stack: letzter Schritt zuunterst)
    for intent in reversed(plan.sub_goals):
        target_pos = (
            Position(intent.target_position[0], intent.target_position[1])
            if intent.target_position
            else None
        )
        sub_goal = Goal(
            name=f"SubGoal: {intent.action_type.value}",
            target_position=target_pos,
            target_entity_id=intent.target_entity_id,
            description=intent.description,
        )
        goal_service.push_goal(agent, sub_goal)

    # Stack-Reihenfolge: MOVE_TO liegt ganz oben (aktiv), gefolgt von CONSUME und Primärziel
    assert len(agent.goals) == 3
    assert agent.active_goal is not None
    assert agent.active_goal.name == "SubGoal: move_to"
    assert agent.active_goal.target_position == Position(2, 3)