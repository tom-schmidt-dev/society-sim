from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.domain.models.planning.planning import (
    ActionType,
    AgentCognitiveContext,
    EntityPerceptionFact,
    PlanDecomposition,
    SubGoalIntent,
)
from src.domain.models.world.position import Position
from src.infrastructure.cognition.adapters.instructor_adapter import InstructorCognitionAdapter


@pytest.fixture
def sample_context_with_memories() -> AgentCognitiveContext:
    return AgentCognitiveContext(
        agent_id="agent_1",
        current_position=Position(1, 1),
        vital_status={"hunger": 0.9},
        urgent_need="hunger",
        known_entities=[],
        episodic_memories=[
            "Tag 1: Apfelbaum bei (5, 6) erfolgreich genutzt.",
        ],
    )


@pytest.mark.asyncio
async def test_decompose_plan_injects_episodic_memories_into_prompt(
    sample_context_with_memories: AgentCognitiveContext,
) -> None:
    adapter = InstructorCognitionAdapter()
    expected_plan = PlanDecomposition(
        thought="Nutze Erinnerung an Apfelbaum.",
        primary_goal="Hunger stillen",
        sub_goals=[
            SubGoalIntent(
                action_type=ActionType.MOVE_TO,
                target_position=(5, 6),
                description="Gehe zu (5, 6)",
            ),
            SubGoalIntent(
                action_type=ActionType.CONSUME,
                description="Konsumiere Nahrung",
            ),
        ],
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=expected_plan)
    adapter._client = mock_client

    result = await adapter.decompose_plan(sample_context_with_memories)

    assert result == expected_plan
    mock_client.chat.completions.create.assert_awaited_once()

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]

    assert "episodic_memories" in system_prompt
    assert "Tag 1: Apfelbaum bei (5, 6) erfolgreich genutzt." in user_prompt


@pytest.mark.asyncio
async def test_heuristic_fallback_uses_episodic_memories_when_llm_fails(
    sample_context_with_memories: AgentCognitiveContext,
) -> None:
    adapter = InstructorCognitionAdapter()

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("LLM offline"))
    adapter._client = mock_client

    plan = await adapter.decompose_plan(sample_context_with_memories)

    assert plan.primary_goal == "hunger stillen"
    assert len(plan.sub_goals) == 2
    assert plan.sub_goals[0].action_type == ActionType.MOVE_TO
    assert plan.sub_goals[0].target_position == (5, 6)
    assert plan.sub_goals[1].action_type == ActionType.CONSUME


@pytest.mark.asyncio
async def test_heuristic_fallback_defaults_to_explore_without_memories_or_entities() -> None:
    adapter = InstructorCognitionAdapter()
    empty_context = AgentCognitiveContext(
        agent_id="agent_2",
        current_position=Position(0, 0),
        vital_status={"thirst": 0.8},
        urgent_need="thirst",
        known_entities=[],
        episodic_memories=[],
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("LLM offline"))
    adapter._client = mock_client

    plan = await adapter.decompose_plan(empty_context)

    assert plan.sub_goals[0].action_type == ActionType.EXPLORE
    assert plan.sub_goals[1].action_type == ActionType.CONSUME