from typing import Callable, Iterable, Optional
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_memory import EntityFact
from src.domain.models.world.world_entity import WorldEntity


class PreconditionEvaluator:
    """Deterministische Überprüfung von Vor- und Nachbedingungen für Aktionen."""

    # Standard-Kategoriengruppen für Systembedürfnisse
    RESOURCE_CATEGORIES: dict[str, set[str]] = {
        "consumable": {"food", "apple", "resource", "drink"},
        "food": {"food", "apple", "resource", "bread", "berry"},
        "drinkable": {"water", "well", "drink", "stream", "fountain"},
        "rest_area": {"bed", "camp", "shelter", "bench", "home"},
        "building_material": {"wood", "stone", "iron", "plank"},
        "tool": {"axe", "pickaxe", "hammer"},
    }

    @staticmethod
    def is_adjacent(agent: Agent, entity: WorldEntity) -> bool:
        return agent.position.manhattan_distance(entity.position) <= 1

    def can_consume(self, agent: Agent, entity: WorldEntity) -> bool:
        if not getattr(entity, "is_consumable", False):
            return False
        return self.is_adjacent(agent, entity)

    def can_drink(self, agent: Agent, entity: WorldEntity) -> bool:
        if not (
            getattr(entity, "is_drinkable", False)
            or getattr(entity, "is_consumable", False)
            or entity.entity_type in self.RESOURCE_CATEGORIES["drinkable"]
        ):
            return False
        return self.is_adjacent(agent, entity)

    def can_rest(self, agent: Agent, entity: WorldEntity) -> bool:
        if not (
            getattr(entity, "is_rest_area", False)
            or entity.entity_type in self.RESOURCE_CATEGORIES["rest_area"]
        ):
            return False
        return agent.position == entity.position or self.is_adjacent(agent, entity)

    def find_discovered_entity(
        self,
        agent: Agent,
        categories: Optional[Iterable[str]] = None,
        predicate: Optional[Callable[[EntityFact], bool]] = None,
    ) -> Optional[EntityFact]:
        """Sucht im Gedächtnis des Agenten nach einer Entität, die den Kriterien entspricht."""
        category_set = set(categories) if categories else None

        for fact in agent.memory.known_entities.values():
            if category_set and fact.entity_type in category_set:
                return fact
            if predicate and predicate(fact):
                return fact
        return None