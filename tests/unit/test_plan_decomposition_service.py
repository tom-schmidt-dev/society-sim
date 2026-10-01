from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from src.application.services.plan_decomposition_service import PlanDecompositionService
from src.domain.models.agent import Agent
from src.domain.models.agent_memory import EntityFact, PerceptionSource
from src.domain.models.planning import (
    ActionType,
    AgentCognitiveContext,
    PlanDecomposition,
    SubGoalIntent,
)
from src.domain.models.position import Position
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.vector_memory_store import IVectorMemoryStore


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_cognition() -> AsyncMock:
    """Bereitstellung eines isolierten Kognitionsproviders."""
    return AsyncMock(spec=ICognitionProvider)


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Bereitstellung eines gemockten Vektorspeichers."""
    return MagicMock(spec=IVectorMemoryStore)


@pytest.fixture
def sample_agent() -> Agent:
    """Bereitstellung eines standardisierten Agenten."""
    return Agent(id="agent_1", name="Alice", position=Position(1, 1))


@pytest.fixture
def service(mock_cognition: AsyncMock) -> PlanDecompositionService:
    """Standard-Serviceinstanz ohne Vektorspeicher."""
    return PlanDecompositionService(cognition_provider=mock_cognition)


# ==============================================================================
# Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_plan_decomposition_creates_valid_cognitive_context(
    mock_cognition: AsyncMock,
    service: PlanDecompositionService,
    sample_agent: Agent,
) -> None:
    mock_cognition.decompose_plan.return_value = PlanDecomposition(
        thought="Nahrung bekannt",
        primary_goal="Essen",
        sub_goals=[SubGoalIntent(action_type=ActionType.CONSUME, target_entity_id="apple_1")],
    )

    sample_agent.needs["hunger"] = 0.85
    sample_agent.memory.known_entities["apple_1"] = EntityFact(
        entity_id="apple_1",
        name="Apfel",
        last_known_position=Position(1, 2),
        source=PerceptionSource.VISUAL,
        observed_tick=1,
        entity_type="food",
        confidence=1.0,
    )

    await service.create_plan_for_need(sample_agent, "hunger")

    mock_cognition.decompose_plan.assert_awaited_once()
    context: AgentCognitiveContext = mock_cognition.decompose_plan.call_args[0][0]

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
    assert context.episodic_memories == []


@pytest.mark.asyncio
async def test_plan_decomposition_when_food_is_known(
    mock_cognition: AsyncMock,
    service: PlanDecompositionService,
    sample_agent: Agent,
) -> None:
    sample_agent.position = Position(2, 2)
    sample_agent.memory.known_entities["apple_1"] = EntityFact(
        entity_id="apple_1",
        name="Apfel",
        entity_type="food",
        last_known_position=Position(2, 3),
        source=PerceptionSource.VISUAL,
    )

    mock_cognition.decompose_plan.return_value = PlanDecomposition(
        thought="Ein Apfel ist direkt bekannt, ich gehe hin und esse ihn.",
        primary_goal="Hunger stillen",
        sub_goals=[
            SubGoalIntent(action_type=ActionType.MOVE_TO, target_entity_id="apple_1", target_position=(2, 3)),
            SubGoalIntent(action_type=ActionType.CONSUME, target_entity_id="apple_1"),
        ],
    )

    plan = await service.create_plan_for_need(sample_agent, "hunger")

    assert len(plan.sub_goals) == 2
    assert plan.sub_goals[0].action_type == ActionType.MOVE_TO
    assert plan.sub_goals[1].action_type == ActionType.CONSUME
    mock_cognition.decompose_plan.assert_awaited_once()


@pytest.mark.asyncio
async def test_plan_decomposition_when_food_is_unknown(
    mock_cognition: AsyncMock,
    service: PlanDecompositionService,
    sample_agent: Agent,
) -> None:
    sample_agent.position = Position(2, 2)

    mock_cognition.decompose_plan.return_value = PlanDecomposition(
        thought="Keine Nahrung sichtbar oder im Gedächtnis. Ich muss die Umgebung absuchen.",
        primary_goal="Nahrung finden und verzehren",
        sub_goals=[
            SubGoalIntent(action_type=ActionType.EXPLORE, description="Suche nach essbaren Objekten"),
            SubGoalIntent(action_type=ActionType.CONSUME, description="Gefundene Nahrung verzehren"),
        ],
    )

    plan = await service.create_plan_for_need(sample_agent, "hunger")

    assert len(plan.sub_goals) == 2
    assert plan.sub_goals[0].action_type == ActionType.EXPLORE
    assert plan.sub_goals[1].action_type == ActionType.CONSUME
    mock_cognition.decompose_plan.assert_awaited_once()


@pytest.mark.asyncio
async def test_plan_decomposition_retrieves_episodic_memories_with_filter(
    mock_cognition: AsyncMock,
    mock_vector_store: MagicMock,
    sample_agent: Agent,
) -> None:
    mock_cognition.decompose_plan.return_value = PlanDecomposition(
        thought="Erinnerung genutzt",
        primary_goal="Durst stillen",
        sub_goals=[SubGoalIntent(action_type=ActionType.EXPLORE)],
    )
    mock_vector_store.retrieve_relevant.return_value = [
        "Tag 1: Wasserquelle bei (2, 5) erfolgreich genutzt."
    ]

    service_with_store = PlanDecompositionService(
        cognition_provider=mock_cognition,
        vector_memory_store=mock_vector_store,
    )

    await service_with_store.create_plan_for_need(sample_agent, "thirst")

    mock_vector_store.retrieve_relevant.assert_called_once_with(
        agent_id="agent_1",
        query="Wasser Trinken Quelle",
        limit=3,
        metadata_filter={"category": "resource"},
    )

    context: AgentCognitiveContext = mock_cognition.decompose_plan.call_args[0][0]
    assert context.episodic_memories == [
        "Tag 1: Wasserquelle bei (2, 5) erfolgreich genutzt."
    ]


@pytest.mark.asyncio
async def test_plan_decomposition_fallback_for_unknown_need(
    mock_cognition: AsyncMock,
    mock_vector_store: MagicMock,
    sample_agent: Agent,
) -> None:
    mock_cognition.decompose_plan.return_value = PlanDecomposition(
        thought="Unbekanntes Bedürfnis",
        primary_goal="Ruhe",
        sub_goals=[SubGoalIntent(action_type=ActionType.WAIT)],
    )
    mock_vector_store.retrieve_relevant.return_value = []

    service_with_store = PlanDecompositionService(
        cognition_provider=mock_cognition,
        vector_memory_store=mock_vector_store,
    )

    await service_with_store.create_plan_for_need(sample_agent, "custom_need")

    mock_vector_store.retrieve_relevant.assert_called_once_with(
        agent_id="agent_1",
        query="custom_need Quelle",
        limit=3,
        metadata_filter={"category": "resource"},
    )

    context: AgentCognitiveContext = mock_cognition.decompose_plan.call_args[0][0]
    assert context.episodic_memories == []