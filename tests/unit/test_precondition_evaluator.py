from __future__ import annotations

import pytest
from src.domain.models.agent import Agent
from src.domain.models.position import Position
from src.domain.models.world_entity import WorldEntity
from src.domain.services.precondition_evaluator import PreconditionEvaluator


def test_can_consume_adjacent_consumable_entity():
    agent = Agent(id="1", name="Alice", position=Position(2, 2))
    apple = WorldEntity(
        id="apple_1",
        name="Apfel",
        position=Position(2, 3),
        is_consumable=True,
        nutrition_value=0.5,
    )

    evaluator = PreconditionEvaluator()
    assert evaluator.can_consume(agent, apple) is True


def test_cannot_consume_distant_entity():
    agent = Agent(id="1", name="Alice", position=Position(2, 2))
    apple = WorldEntity(
        id="apple_1",
        name="Apfel",
        position=Position(5, 5),
        is_consumable=True,
    )

    evaluator = PreconditionEvaluator()
    assert evaluator.can_consume(agent, apple) is False


def test_cannot_consume_non_consumable_entity():
    agent = Agent(id="1", name="Alice", position=Position(2, 2))
    stone = WorldEntity(
        id="stone_1",
        name="Stein",
        position=Position(2, 3),
        is_consumable=False,
    )

    evaluator = PreconditionEvaluator()
    assert evaluator.can_consume(agent, stone) is False

from src.domain.models.agent_memory import EntityFact, PerceptionSource


def test_find_discovered_entity_by_category_success():
    agent = Agent(id="1", name="Alice", position=Position(2, 2))
    apple_fact = EntityFact(
        entity_id="apple_1",
        name="Apfel",
        last_known_position=Position(3, 3),
        source=PerceptionSource.VISUAL,
        observed_tick=1,
        entity_type="apple",
    )
    agent.memory.known_entities["apple_1"] = apple_fact

    evaluator = PreconditionEvaluator()
    consumable_types = evaluator.RESOURCE_CATEGORIES["consumable"]
    found = evaluator.find_discovered_entity(agent, categories=consumable_types)

    assert found is not None
    assert found.entity_id == "apple_1"
    assert found.entity_type == "apple"


def test_find_discovered_entity_by_category_no_match():
    agent = Agent(id="1", name="Alice", position=Position(2, 2))
    stone_fact = EntityFact(
        entity_id="stone_1",
        name="Stein",
        last_known_position=Position(3, 3),
        source=PerceptionSource.VISUAL,
        observed_tick=1,
        entity_type="stone",
    )
    agent.memory.known_entities["stone_1"] = stone_fact

    evaluator = PreconditionEvaluator()
    consumable_types = evaluator.RESOURCE_CATEGORIES["consumable"]
    found = evaluator.find_discovered_entity(agent, categories=consumable_types)

    assert found is None


def test_find_discovered_entity_by_predicate():
    agent = Agent(id="1", name="Alice", position=Position(2, 2))
    tool_fact = EntityFact(
        entity_id="axe_1",
        name="Axt",
        last_known_position=Position(4, 4),
        source=PerceptionSource.VISUAL,
        observed_tick=2,
        entity_type="axe",
        confidence=0.9,
    )
    agent.memory.known_entities["axe_1"] = tool_fact

    evaluator = PreconditionEvaluator()
    found = evaluator.find_discovered_entity(
        agent, predicate=lambda fact: fact.confidence > 0.8 and fact.entity_type == "axe"
    )

    assert found is not None
    assert found.entity_id == "axe_1"


def test_find_discovered_entity_empty_memory_returns_none():
    agent = Agent(id="1", name="Alice", position=Position(2, 2))
    evaluator = PreconditionEvaluator()

    found = evaluator.find_discovered_entity(
        agent, categories=evaluator.RESOURCE_CATEGORIES["consumable"]
    )

    assert found is None