from __future__ import annotations

from abc import ABC, abstractmethod
from src.domain.models.agent.agent import Agent
from src.domain.models.world.position import Position
from src.domain.models.world.world_entity import WorldEntity


class IConflictCoordinator(ABC):
    @abstractmethod
    async def resolve_blockage(
        self,
        agent: Agent,
        blocker: WorldEntity,
        blocked_pos: Position,
        all_entities: list[WorldEntity],
    ) -> None:
        """Initiiert und koordiniert die Auflösung einer räumlichen Blockadesituation."""
        pass