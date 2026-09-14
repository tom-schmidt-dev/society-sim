from __future__ import annotations

from src.domain.models.agent import Agent
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity


class PerceptionService:
    def __init__(self, default_radius: int = 3) -> None:
        self._default_radius: int = default_radius

    def get_visible_positions(
        self,
        origin: Position,
        grid: WorldGrid,
        radius: int | None = None,
    ) -> list[Position]:
        """Ermittelt alle Kacheln innerhalb der Manhattan-Distanz L1 <= radius innerhalb der Grid-Grenzen."""
        effective_radius = radius if radius is not None else self._default_radius
        visible_positions: list[Position] = []

        for dx in range(-effective_radius, effective_radius + 1):
            remaining_y = effective_radius - abs(dx)
            for dy in range(-remaining_y, remaining_y + 1):
                pos = Position(origin.x + dx, origin.y + dy)
                if grid.is_within_bounds(pos):
                    visible_positions.append(pos)

        return visible_positions

    def get_visible_entities(
        self,
        agent: Agent,
        entities: list[WorldEntity],
        radius: int | None = None,
    ) -> list[WorldEntity]:
        """Filtert alle Entitäten heraus, die sich innerhalb des Sichtfelds des Agenten befinden (ohne sich selbst)."""
        effective_radius = radius if radius is not None else self._default_radius
        return [
            entity
            for entity in entities
            if entity.id != agent.id
            and agent.position.manhattan_distance(entity.position) <= effective_radius
        ]