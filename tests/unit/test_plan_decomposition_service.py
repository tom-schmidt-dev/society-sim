from __future__ import annotations

import pytest
from unittest.mock import AsyncMock
from src.domain.models.agent import Agent
from src.domain.models.agent_memory import EntityFact, PerceptionSource
from src.domain.models.planning import ActionType, PlanDecomposition, SubGoalIntent, AgentCognitiveContext
from src.domain.models.position import Position
from src.domain.ports.cognition_provider import ICognitionProvider
from src.application.services.plan_decomposition_service import PlanDecompositionService

@pytest.mark.asyncio
async def test_plan_decomposition_creates_valid_cognitive_context():
    cognition = AsyncMock(spec=ICognitionProvider)
    cognition.decompose_plan.return_value = PlanDecomposition(
        thought="Nahrung bekannt",
        primary_goal="Essen",
        sub_goals=[SubGoalIntent(action_type=ActionType.CONSUME, target_entity_id="apple_1")],
    )

    service = PlanDecompositionService(cognition_provider=cognition)

    agent = Agent(id="agent_1", name="Alice", position=Position(1, 1))
    agent.needs["hunger"] = 0.85

    # Visuelle Beobachtung
    agent.memory.known_entities["apple_1"] = EntityFact(
        entity_id="apple_1",
        name="Apfel",
        last_known_position=Position(1, 2),
        source=PerceptionSource.VISUAL,
        observed_tick=1,
        entity_type="food",
        confidence=1.0,
    )

    await service.create_plan_for_need(agent, "hunger")

    cognition.decompose_plan.assert_awaited_once()
    context: AgentCognitiveContext = cognition.decompose_plan.call_args[0][0]

    assert isinstance(context, AgentCognitiveContext)
    assert context.agent_id == "agent_1"
    assert context.current_position == Position(1, 1)
    assert context.urgent_need == "hunger"
    assert len(context.known_entities) == 1

    fact = context.known_entities[0]
    assert fact.entity_id == "apple_1"
    assert fact.last_known_position == Position(1, 2)
    assert fact.source == "visual"
    assert fact.is_consumable is True

@pytest.mark.asyncio
async def test_plan_decomposition_when_food_is_known():
    agent = Agent(id="1", name="Alice", position=Position(2, 2))
    agent.memory.known_entities["apple_1"] = EntityFact(
        entity_id="apple_1",
        name="Apfel",
        entity_type="food",
        last_known_position=Position(2, 3),
        source=PerceptionSource.VISUAL,
    )

    mock_cognition = AsyncMock(spec=ICognitionProvider)
    mock_cognition.decompose_plan.return_value = PlanDecomposition(
        thought="Ein Apfel ist direkt bekannt, ich gehe hin und esse ihn.",
        primary_goal="Hunger stillen",
        sub_goals=[
            SubGoalIntent(action_type=ActionType.MOVE_TO, target_entity_id="apple_1", target_position=(2, 3)),
            SubGoalIntent(action_type=ActionType.CONSUME, target_entity_id="apple_1"),
        ],
    )

    service = PlanDecompositionService(cognition_provider=mock_cognition)
    plan = await service.create_plan_for_need(agent, "hunger")

    assert len(plan.sub_goals) == 2
    assert plan.sub_goals[0].action_type == ActionType.MOVE_TO
    assert plan.sub_goals[1].action_type == ActionType.CONSUME
    mock_cognition.decompose_plan.assert_awaited_once()


@pytest.mark.asyncio
async def test_plan_decomposition_when_food_is_unknown():
    agent = Agent(id="1", name="Alice", position=Position(2, 2))
    # Kein Eintrag in agent.memory

    mock_cognition = AsyncMock(spec=ICognitionProvider)
    mock_cognition.decompose_plan.return_value = PlanDecomposition(
        thought="Keine Nahrung sichtbar oder im Gedächtnis. Ich muss die Umgebung absuchen.",
        primary_goal="Nahrung finden und verzehren",
        sub_goals=[
            SubGoalIntent(action_type=ActionType.EXPLORE, description="Suche nach essbaren Objekten"),
            SubGoalIntent(action_type=ActionType.CONSUME, description="Gefundene Nahrung verzehren"),
        ],
    )

    service = PlanDecompositionService(cognition_provider=mock_cognition)
    plan = await service.create_plan_for_need(agent, "hunger")

    assert len(plan.sub_goals) == 2
    assert plan.sub_goals[0].action_type == ActionType.EXPLORE
    assert plan.sub_goals[1].action_type == ActionType.CONSUME