from __future__ import annotations

from typing import Optional
from src.domain.models.agent import Agent
from src.domain.models.planning import (
    AgentCognitiveContext,
    EntityPerceptionFact,
    PlanDecomposition,
)
from src.domain.ports.cognition_provider import ICognitionProvider


class PlanDecompositionService:
    """Orchestriert die kognitive Zerlegung von übergeordneten Bedürfnissen in Teilzielketten."""

    def __init__(self, cognition_provider: ICognitionProvider) -> None:
        self._cognition_provider = cognition_provider

    def _build_context(self, agent: Agent, need_name: str) -> AgentCognitiveContext:
        known_facts: list[EntityPerceptionFact] = []

        for fact in agent.memory.known_entities.values():
            if fact.last_known_position is None:
                continue

            is_consumable = fact.entity_type in {"food", "apple", "resource", "drink"}

            known_facts.append(
                EntityPerceptionFact(
                    entity_id=fact.entity_id,
                    name=fact.name,
                    entity_type=fact.entity_type,
                    last_known_position=fact.last_known_position,
                    is_consumable=is_consumable,
                    source=fact.source.value if hasattr(fact.source, "value") else str(fact.source),
                    confidence=fact.confidence,
                )
            )

        return AgentCognitiveContext(
            agent_id=agent.id,
            current_position=agent.position,
            vital_status=dict(agent.needs),
            urgent_need=need_name,
            known_entities=known_facts,
            unexplored_frontiers_available=True,
        )

    async def create_plan_for_need(self, agent: Agent, need_name: str) -> PlanDecomposition:
        context = self._build_context(agent, need_name)
        return await self._cognition_provider.decompose_plan(context)