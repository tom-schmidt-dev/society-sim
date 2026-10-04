from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from src.domain.models.agent.mental_map import AgentMentalMap, TileKnowledge
from src.domain.models.world.position import Position
from src.domain.ports.pathfinder import IPathfinder


@dataclass(frozen=True, slots=True)
class EvasionResult:
    target_tile: Position
    junction_tile: Position
    is_frontier: bool = False
    path: list[Position] = field(default_factory=list)


class EvasionFinder:
    def __init__(self, pathfinder: IPathfinder) -> None:
        self._pathfinder: IPathfinder = pathfinder

    @staticmethod
    def _is_known_walkable(grid: Any, pos: Position) -> bool:
        """Prüft, ob eine Kachel verifiziert passierbar ist (schließt UNKNOWN aus)."""
        if isinstance(grid, AgentMentalMap):
            tile = grid.tiles.get(pos)
            return tile is not None and tile.knowledge == TileKnowledge.WALKABLE
        return bool(grid.is_walkable(pos))

    def find_nearest_evasion_tile(
            self,
            start: Position,
            blocked_pos: Position,
            grid: Any,
            occupied_positions: set[Position],
            partner_trajectory: Optional[Sequence[Position] | set[Position]] = None,
            search_direction: Optional[tuple[float, float]] = None,
            allow_frontier: bool = True,
    ) -> Optional[EvasionResult]:
        """Ermittelt per 3-Stufen-Kaskade das Ausweichfeld und den Schnittpunkt zur Räumung."""
        trajectory_set = set(partner_trajectory) if partner_trajectory else set()

        # Nur real belegte Kacheln (occupied_positions) blockieren die Traversierung physisch
        physical_obstacles: set[Position] = (set(occupied_positions) - {start})
        forbidden_targets: set[Position] = set(occupied_positions) | {blocked_pos} | trajectory_set

        queue: list[Position] = [start]
        visited: set[Position] = {start}
        came_from: dict[Position, Position] = {}

        stage1_match: Optional[Position] = None
        stage2_match: Optional[Position] = None

        while queue:
            current = queue.pop(0)

            if current != start and current not in forbidden_targets and self._is_known_walkable(grid, current):
                dist = start.manhattan_distance(current)
                if dist <= 3:
                    stage1_match = current
                    break
                elif stage2_match is None:
                    stage2_match = current

            for neighbor in current.get_neighbors():
                if neighbor not in visited and neighbor not in physical_obstacles and grid.is_within_bounds(neighbor):
                    visited.add(neighbor)
                    if self._is_known_walkable(grid, neighbor):
                        came_from[neighbor] = current
                        queue.append(neighbor)

        chosen_tile = stage1_match or stage2_match
        is_frontier_match = False

        if allow_frontier and not chosen_tile and isinstance(grid, AgentMentalMap):
            frontier_forbidden = set(occupied_positions) | {blocked_pos}
            chosen_tile, frontier_came_from = self._find_nearest_frontier(
                start=start,
                grid=grid,
                frontier_forbidden=frontier_forbidden,
                direction_vector=search_direction,
            )
            if chosen_tile:
                came_from = frontier_came_from
                is_frontier_match = True

        if chosen_tile:
            evasion_path = self._reconstruct_path(start, chosen_tile, came_from)
            junction = self._determine_junction(evasion_path, trajectory_set, start)
            return EvasionResult(
                target_tile=chosen_tile,
                junction_tile=junction,
                is_frontier=is_frontier_match,
                path=evasion_path,
            )

        return None

    @staticmethod
    def _sort_frontier_neighbors(
            neighbor_candidates: list[Position],
            current: Position,
            direction_vector: Optional[tuple[float, float]],
    ) -> list[Position]:
        if direction_vector is None:
            return neighbor_candidates
        vx = float(direction_vector[0])
        vy = float(direction_vector[1])
        if vx == 0.0 and vy == 0.0:
            return neighbor_candidates
        return sorted(
            neighbor_candidates,
            key=lambda nb: (nb.x - current.x) * vx + (nb.y - current.y) * vy,
            reverse=True,
        )

    def _find_nearest_frontier(
            self,
            start: Position,
            grid: AgentMentalMap,
            frontier_forbidden: set[Position],
            direction_vector: Optional[tuple[float, float]] = None,
            max_depth: int = 15,
    ) -> tuple[Optional[Position], dict[Position, Position]]:
        """Findet die nächstgelegene Grenzkachel mit Tiefenbegrenzung und Sektor-Gewichtung."""
        frontier_queue: list[tuple[Position, int]] = [(start, 0)]
        frontier_visited: set[Position] = {start} | frontier_forbidden
        came_from: dict[Position, Position] = {}

        while frontier_queue:
            curr, depth = frontier_queue.pop(0)

            if curr != start:
                for neighbor in curr.get_neighbors():
                    if grid.is_within_bounds(neighbor):
                        tile = grid.tiles.get(neighbor)
                        if tile is None or tile.knowledge == TileKnowledge.UNKNOWN:
                            return curr, came_from

            # Abbruch bei Erreichen der maximalen Suchtiefe
            if depth >= max_depth:
                continue

            sorted_neighbors = self._sort_frontier_neighbors(
                curr.get_neighbors(), curr, direction_vector
            )
            for neighbor in sorted_neighbors:
                if neighbor not in frontier_visited and grid.is_within_bounds(neighbor):
                    frontier_visited.add(neighbor)
                    if self._is_known_walkable(grid, neighbor):
                        came_from[neighbor] = curr
                        frontier_queue.append((neighbor, depth + 1))

        return None, came_from

    @staticmethod
    def _reconstruct_path(
        start: Position,
        target: Position,
        came_from: dict[Position, Position],
    ) -> list[Position]:
        curr = target
        path: list[Position] = []
        while curr != start and curr in came_from:
            path.append(curr)
            curr = came_from[curr]
        path.reverse()
        return path

    @staticmethod
    def _determine_junction(
        evasion_path: list[Position],
        partner_trajectory: set[Position],
        default_pos: Position,
    ) -> Position:
        """Bestimmt die letzte Kachel des Ausweichwegs, die auf dem Partnerpfad lag."""
        last_intersection = default_pos
        for pos in evasion_path:
            if pos in partner_trajectory:
                last_intersection = pos
            else:
                break
        return last_intersection


    def compare_evasion_distances(
            self,
            agent_a: Any,
            agent_b: Any,
            all_entities: list[Any],
    ) -> tuple[Optional[EvasionResult], Optional[EvasionResult]]:
        """Ermittelt und vergleicht die Nischenwege für zwei Agenten unter Beachtung realer Pfaderreichbarkeit."""
        other_occupied = {e.position for e in all_entities if e.id not in (agent_a.id, agent_b.id)}

        res_a = self.find_nearest_evasion_tile(
            start=agent_a.position,
            blocked_pos=agent_b.position,
            grid=agent_a.mental_map,
            occupied_positions=other_occupied | {agent_b.position},
            partner_trajectory=agent_b.path if agent_b.has_path else None,
            allow_frontier=False,
        )

        res_b = self.find_nearest_evasion_tile(
            start=agent_b.position,
            blocked_pos=agent_a.position,
            grid=agent_b.mental_map,
            occupied_positions=other_occupied | {agent_a.position},
            partner_trajectory=agent_a.path if agent_a.has_path else None,
            allow_frontier=False,
        )

        return res_a, res_b
