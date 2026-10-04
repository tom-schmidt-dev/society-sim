from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.domain.models.planning.planning import ActionType, PlanDecomposition, SubGoalIntent


def test_plan_decomposition_valid_schema():
    plan = PlanDecomposition(
        thought="Nahrung liegt in Sichtweite.",
        primary_goal="Nahrung aufnehmen",
        sub_goals=[
            SubGoalIntent(
                action_type=ActionType.MOVE_TO,
                target_entity_id="apple_1",
                target_position=(2, 3),
                description="Zum Apfel bewegen",
            ),
            SubGoalIntent(
                action_type=ActionType.CONSUME,
                target_entity_id="apple_1",
                description="Apfel verzehren",
            ),
        ],
    )
    assert plan.primary_goal == "Nahrung aufnehmen"
    assert len(plan.sub_goals) == 2
    assert plan.sub_goals[0].action_type == ActionType.MOVE_TO


def test_plan_decomposition_rejects_empty_subgoals():
    with pytest.raises(ValidationError):
        PlanDecomposition(
            thought="Kein Plan vorhanden.",
            primary_goal="Leerlauf",
            sub_goals=[],
        )


def test_plan_decomposition_enforces_maximum_chain_length():
    with pytest.raises(ValidationError):
        PlanDecomposition(
            thought="Zu viele Schritte.",
            primary_goal="Überdimensionierter Plan",
            sub_goals=[
                SubGoalIntent(action_type=ActionType.WAIT)
                for _ in range(6)
            ],
        )

def test_action_type_rejects_invalid_strings():
    with pytest.raises(ValidationError):
        SubGoalIntent(action_type="invalid_action_name")  # type: ignore[arg-type]


def test_goal_initial_wait_tick_retention_and_serialization() -> None:
    from src.domain.models.planning.goal import Goal

    goal = Goal(name="Warten auf Partner", remaining_ticks=200, initial_wait_tick=42)
    assert goal.initial_wait_tick == 42
    serialized = goal.to_dict()
    assert serialized["initial_wait_tick"] == 42


def test_agent_has_no_interaction_queue() -> None:
    from src.domain.models.agent.agent import Agent
    from src.domain.models.world.position import Position

    agent = Agent(id="1", name="Alice", position=Position(0, 0))
    assert not hasattr(agent, "interaction_queue")