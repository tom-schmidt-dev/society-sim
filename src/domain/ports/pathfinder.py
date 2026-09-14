from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from src.domain.models.position import Position

class IPathfinder(ABC):
    @abstractmethod
    def find_path(
            self, start: Position, goal: Position, grid: Any
    ) -> list[Position]:
        pass