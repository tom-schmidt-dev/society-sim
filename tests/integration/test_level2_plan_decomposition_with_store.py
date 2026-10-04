from __future__ import annotations

from typing import Any, Optional

import pytest

from src.application.services.cognition.plan_decomposition_service import PlanDecompositionService
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.cognition import SocialReflection
from src.domain.models.planning.planning import (
    ActionType,
    AgentCognitiveContext,
    PlanDecomposition,
    SubGoalIntent,
)
from src.domain.models.world.position import Position
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.vector_memory_store import IVectorMemoryStore


class Level2VectorStoreStub(IVectorMemoryStore):
    def __init__(self) -> None:
        self.memories: list[str] = []

    def add_memories(
        self, agent_id: str, memories: list[str], metadatas: Optional[list[dict[str, Any]]] = None
    ) -> None:
        self.memories.extend(memories)

    def retrieve_relevant(
        self, agent_id: str, query: str, limit: int = 3, metadata_filter: Optional[dict[str, Any]] = None
    ) -> list[str]:
        return list(self.memories[:limit])


class MockCognitionProvider(ICognitionProvider):
    async def decide_next_goal(self, context: dict[str, Any]) -> Any:
        raise NotImplementedError

    async def resolve_blockage(self, context: dict[str, Any]) -> Any:
        raise NotImplementedError

    async def evaluate_goal_status(self, context: dict[str, Any]) -> Any:
        raise NotImplementedError

    async def respond_to_dialogue(self, context: dict[str, Any]) -> Any:
        raise NotImplementedError

    async def reflect_on_dialogue(self, context: dict[str, Any]) -> SocialReflection:
        return SocialReflection(
            assessment="Mock-Einschätzung",
            progression_summary="Mock-Zusammenfassung",
        )

    async def decompose_plan(self, context: AgentCognitiveContext) -> PlanDecomposition:
        if context.episodic_memories:
            return PlanDecomposition(
                thought="Erinnerung genutzt",
                primary_goal="Nahrungssuche",
                sub_goals=[
                    SubGoalIntent(
                        action_type=ActionType.MOVE_TO,
                        target_position=(6, 5),
                        description="Gehe zu (6, 5)",
                    )
                ],
            )
        return PlanDecomposition(
            thought="Exploration",
            primary_goal="Erkundung",
            sub_goals=[SubGoalIntent(action_type=ActionType.EXPLORE, description="Exploriere")],
        )


@pytest.mark.asyncio
async def test_plan_decomposition_retrieves_from_store() -> None:
    vector_store = Level2VectorStoreStub()
    vector_store.add_memories("a1", ["Tag 1: Apfelbaum bei (6, 5) erfolgreich genutzt."])

    service = PlanDecompositionService(
        cognition_provider=MockCognitionProvider(),
        vector_memory_store=vector_store,
    )

    agent = Agent(id="a1", name="Alice", position=Position(1, 1))
    agent.needs["hunger"] = 0.8

    plan = await service.create_plan_for_need(agent, "hunger")

    assert plan.primary_goal == "Nahrungssuche"
    assert len(plan.sub_goals) == 1
    assert plan.sub_goals[0].action_type == ActionType.MOVE_TO
    assert plan.sub_goals[0].target_position == (6, 5)