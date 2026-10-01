from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.coordination.dialogue_history import DialogueHistory
from src.application.services.cognition.goal_service import GoalService
from src.application.services.lifecycle.need_service import NeedService
from src.domain.models.agent.agent import Agent
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder


@pytest.fixture
def executor_setup():
    grid = WorldGrid(10, 10)
    logger = MagicMock(spec=IEventLogger)
    dialogue_history = DialogueHistory()
    cognition = MagicMock(spec=ICognitionProvider)
    pathfinder = MagicMock(spec=IPathfinder)
    goal_service = GoalService(logger=logger, cognition_provider=cognition, pathfinder=pathfinder)
    need_service = NeedService()

    executor = ActionExecutor(
        grid=grid,
        logger=logger,
        dialogue_history=dialogue_history,
        goal_service=goal_service,
        pathfinder=pathfinder,
        need_service=need_service,
        tick_provider=lambda: 5,
    )
    return executor, logger, need_service


def test_execute_consume_success(executor_setup):
    executor, logger, _ = executor_setup
    agent = Agent(id="agent_1", name="Alice", position=Position(2, 2))
    agent.needs["hunger"] = 0.8

    apple = WorldEntity(
        id="apple_1",
        name="Apfel",
        position=Position(2, 3),
        is_consumable=True,
        nutrition_value=0.5,
    )
    entities: list[WorldEntity] = [agent, apple]

    success = executor.execute_consume(
        agent=agent,
        target_entity=apple,
        all_entities=entities,
        incident_id="test-consume-1",
    )

    assert success is True
    assert apple not in entities
    assert pytest.approx(agent.needs["hunger"], 0.001) == 0.3

    logged_events = [call.args[0] for call in logger.log.call_args_list]
    assert any(ev.event_type == "entity_consumed" for ev in logged_events)


def test_execute_consume_fails_when_not_adjacent(executor_setup):
    executor, logger, _ = executor_setup
    agent = Agent(id="agent_1", name="Alice", position=Position(2, 2))
    apple = WorldEntity(
        id="apple_1",
        name="Apfel",
        position=Position(5, 5),
        is_consumable=True,
        nutrition_value=0.5,
    )
    entities: list[WorldEntity] = [agent, apple]

    success = executor.execute_consume(
        agent=agent,
        target_entity=apple,
        all_entities=entities,
        incident_id="test-consume-2",
    )

    assert success is False
    assert apple in entities
    assert not any(call.args[0].event_type == "entity_consumed" for call in logger.log.call_args_list)


def test_execute_consume_fails_when_not_consumable(executor_setup):
    executor, logger, _ = executor_setup
    agent = Agent(id="agent_1", name="Alice", position=Position(2, 2))
    stone = WorldEntity(
        id="stone_1",
        name="Stein",
        position=Position(2, 3),
        is_consumable=False,
    )
    entities: list[WorldEntity] = [agent, stone]

    success = executor.execute_consume(
        agent=agent,
        target_entity=stone,
        all_entities=entities,
        incident_id="test-consume-3",
    )

    assert success is False
    assert stone in entities