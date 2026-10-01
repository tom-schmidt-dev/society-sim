from __future__ import annotations

from typing import Optional
from src.domain.models.agent.agent import Agent
from src.domain.models.communication.message import CommunicationChannel
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.models.world.world_entity import WorldEntity


class PerceptionService:
    # Feste Reichweitengrenzen nach L1-Manhattan-Distanz
    CHANNEL_RANGES: dict[CommunicationChannel, Optional[int]] = {
        CommunicationChannel.LOCAL_TALK: 3,
        CommunicationChannel.PUBLIC_ADDRESS: 15,
        CommunicationChannel.DIGITAL_NETWORK: None,  # Unbegrenzte Reichweite
    }

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
        """Filtert alle Entitäten heraus, die sich innerhalb des Sichtfelds des Agenten befinden."""
        effective_radius = radius if radius is not None else self._default_radius
        return [
            entity
            for entity in entities
            if entity.id != agent.id
            and agent.position.manhattan_distance(entity.position) <= effective_radius
        ]

    def is_in_channel_range(
        self,
        sender_pos: Position,
        recipient_pos: Position,
        channel: CommunicationChannel,
    ) -> bool:
        """Prüft, ob eine Nachricht die Distanzbeschränkung des Übertragungskanals einhält."""
        max_dist = self.CHANNEL_RANGES.get(channel)
        if max_dist is None:
            return True
        return sender_pos.manhattan_distance(recipient_pos) <= max_dist

    class PreconditionEvaluator:
        """Deterministische Überprüfung von Vor- und Nachbedingungen für Aktionen."""

        @staticmethod
        def is_adjacent(agent: Agent, entity: WorldEntity) -> bool:
            """Prüft Manhattan-Distanz <= 1 (direkte Nachbarschaft)."""
            return agent.position.manhattan_distance(entity.position) <= 1

        def can_consume(self, agent: Agent, entity: WorldEntity) -> bool:
            """Prüft, ob ein Agent eine Entität unmittelbar verzehren kann."""
            if not getattr(entity, "is_consumable", False):
                return False
            return self.is_adjacent(agent, entity)