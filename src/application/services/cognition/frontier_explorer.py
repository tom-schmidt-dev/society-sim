from __future__ import annotations

from typing import Optional
from src.domain.models.agent.mental_map import AgentMentalMap
from src.domain.models.world.position import Position


class FrontierExplorer:
    """Ermittelt deterministisch die nächstgelegene Kachel an der Sichtgrenze."""

    @staticmethod
    def find_nearest_frontier(
        current_pos: Position,
        mental_map: AgentMentalMap,
    ) -> Optional[Position]:
        frontier_candidates: list[Position] = []

        for pos in mental_map.tiles.keys():
            if not mental_map.is_walkable(pos):
                continue

            # Prüfe, ob mindestens ein Nachbar innerhalb der Kartengrenzen unentdeckt ist
            has_unknown_neighbor = False
            for neighbor in pos.get_neighbors():
                if 0 <= neighbor.x < mental_map.width and 0 <= neighbor.y < mental_map.height:
                    if neighbor not in mental_map.tiles:
                        has_unknown_neighbor = True
                        break

            if has_unknown_neighbor:
                frontier_candidates.append(pos)

        if not frontier_candidates:
            return None

        # Wähle den Kandidaten mit der geringsten Manhattan-Distanz
        frontier_candidates.sort(key=lambda p: current_pos.manhattan_distance(p))
        return frontier_candidates[0]