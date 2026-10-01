from __future__ import annotations

from typing import Optional
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.planning import (
    AgentCognitiveContext,
    EntityPerceptionFact,
    PlanDecomposition,
)
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.vector_memory_store import IVectorMemoryStore
from src.domain.services.precondition_evaluator import PreconditionEvaluator


class PlanDecompositionService:
    """Orchestriert die kognitive Zerlegung von übergeordneten Bedürfnissen in Teilzielketten."""

    _NEED_QUERY_MAP: dict[str, str] = {
        "hunger": "Nahrung Essen Quelle",
        "thirst": "Wasser Trinken Quelle",
        "energy": "Erholung Schlaf Rastplatz",
    }

    def __init__(
        self,
        cognition_provider: ICognitionProvider,
        precondition_evaluator: Optional[PreconditionEvaluator] = None,
        vector_memory_store: Optional[IVectorMemoryStore] = None,
    ) -> None:
        self._cognition_provider = cognition_provider
        self._precondition_evaluator = precondition_evaluator or PreconditionEvaluator()
        self._vector_memory_store = vector_memory_store

    def _is_relevant_target(self, entity_type: Optional[str], need_name: str) -> bool:
        """Prüft, ob eine Entität das gegebene oder ein allgemeines Vitalbedürfnis bedienen kann."""
        if not entity_type:
            return False
        if need_name == "thirst":
            return entity_type in self._precondition_evaluator.RESOURCE_CATEGORIES.get("drinkable", set())
        if need_name == "energy":
            return entity_type in self._precondition_evaluator.RESOURCE_CATEGORIES.get("rest_area", set())

        # Standard: Hunger / Consumable
        return (
            entity_type in self._precondition_evaluator.RESOURCE_CATEGORIES.get("consumable", set())
            or entity_type in self._precondition_evaluator.RESOURCE_CATEGORIES.get("food", set())
        )

    def _retrieve_episodic_memories(self, agent_id: str, need_name: str) -> list[str]:
        """Ruft relevante Vergangenheitserfahrungen mit Metadatenfilterung resilient aus dem Vektorspeicher ab."""
        if not self._vector_memory_store:
            return []

        query = self._NEED_QUERY_MAP.get(need_name, f"{need_name} Quelle")
        try:
            return self._vector_memory_store.retrieve_relevant(
                agent_id=agent_id,
                query=query,
                limit=3,
                metadata_filter={"category": "resource"},
            )
        except Exception:
            return []

    def _build_context(self, agent: Agent, need_name: str) -> AgentCognitiveContext:
        known_facts: list[EntityPerceptionFact] = []

        for fact in agent.memory.known_entities.values():
            if fact.last_known_position is None:
                continue

            is_consumable = self._is_relevant_target(fact.entity_type, need_name)

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

        episodic_memories = self._retrieve_episodic_memories(agent.id, need_name)

        return AgentCognitiveContext(
            agent_id=agent.id,
            current_position=agent.position,
            vital_status=dict(agent.needs),
            urgent_need=need_name,
            known_entities=known_facts,
            unexplored_frontiers_available=True,
            traits={
                "charisma": agent.charisma,
                "assertiveness": agent.assertiveness,
            },
            episodic_memories=episodic_memories,
        )

    async def create_plan_for_need(self, agent: Agent, need_name: str) -> PlanDecomposition:
        context = self._build_context(agent, need_name)
        return await self._cognition_provider.decompose_plan(context)