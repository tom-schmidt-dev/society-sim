from abc import ABC, abstractmethod
from typing import Optional

from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity


from src.domain.models.position import Position


class IPresenter(ABC):
    @abstractmethod
    def render(
        self,
        grid: WorldGrid,
        entities: list[WorldEntity],
        tick: int,
        dialogues: Optional[list[str]] = None,
        known_positions: Optional[set[Position]] = None,
    ) -> None:
        pass