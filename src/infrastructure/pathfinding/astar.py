from __future__ import annotations
import heapq
from typing import Any
from src.domain.models.position import Position
from src.domain.ports.pathfinder import IPathfinder

class AStarPathfinder(IPathfinder):
    def find_path(
            self, start: Position, goal: Position, grid: Any
    ) -> list[Position]:
        if not grid.is_walkable(start) or not grid.is_walkable(goal):
            return []
        if start == goal:
            return []

        counter: int = 0
        open_set: list[tuple[int, int, Position]] = []
        heapq.heappush(open_set, (0, counter, start))

        came_from: dict[Position, Position] = {}
        g_score: dict[Position, int] = {start: 0}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            for neighbor in grid.get_valid_neighbors(current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = neighbor.manhattan_distance(goal)
                    f_score = tentative_g + h_score
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return []

    @staticmethod
    def _reconstruct_path(
        came_from: dict[Position, Position], current: Position
    ) -> list[Position]:
        path: list[Position] = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path
