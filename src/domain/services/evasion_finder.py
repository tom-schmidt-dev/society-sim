from __future__ import annotations

from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.ports.pathfinder import IPathfinder
from typing import Any, Optional


class EvasionFinder:
    def __init__(self, pathfinder: IPathfinder) -> None:
        self._pathfinder: IPathfinder = pathfinder

    def find_nearest_evasion_tile(
            self,
            start: Position,
            blocked_pos: Position,
            grid: Any,
            occupied_positions: set[Position],
    ) -> Optional[Position]:
        """Ermittelt per Breitensuche das nächstgelegene freie, erreichbare Feld abseits der Blockade."""
        queue: list[Position] = [start]
        visited: set[Position] = {start, blocked_pos}

        while queue:
            current = queue.pop(0)
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                neighbor = Position(current.x + dx, current.y + dy)
                if (
                        0 <= neighbor.x < grid.width
                        and 0 <= neighbor.y < grid.height
                        and neighbor not in visited
                ):
                    visited.add(neighbor)
                    if grid.is_walkable(neighbor) and neighbor not in occupied_positions:
                        path = self._pathfinder.find_path(start, neighbor, grid)
                        if path:
                            return neighbor
                    if grid.is_walkable(neighbor):
                        queue.append(neighbor)
        return None