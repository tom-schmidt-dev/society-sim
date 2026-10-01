from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.domain.models.agent.agent_cognition import AgentCognitiveSnapshot
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.models.world.world_entity import WorldEntity


class IPresenter(ABC):
    @abstractmethod
    def render(
        self,
        grid: WorldGrid,
        entities: list[WorldEntity],
        tick: int,
        dialogues: Optional[list[str]] = None,
        known_positions: Optional[set[Position]] = None,
        snapshots: Optional[list[AgentCognitiveSnapshot]] = None,
    ) -> None:
        pass