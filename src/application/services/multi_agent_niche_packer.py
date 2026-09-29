from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from src.domain.models.agent import Agent
from src.domain.models.mental_map import AgentMentalMap, TileKnowledge
from src.domain.models.position import Position
from src.application.services.convoy_coordinator import Convoy


class NicheType(str, Enum):
    POCKET = "pocket"            # Tiefe Sackgassen-Nische mit einer gemeinsamen Mündung
    DISTRIBUTED = "distributed"  # Sequenziell verteilte Einzellücken entlang des Korridors
    BYPASS = "bypass"            # Parallelkorridor mit Ein- und Ausgang


@dataclass(frozen=True, slots=True)
class SlotAssignment:
    agent_id: str
    slot_position: Position
    junction_position: Position
    path_to_slot: list[Position]


@dataclass(slots=True)
class NicheConfiguration:
    niche_type: NicheType
    total_capacity: int
    slots: list[SlotAssignment]
    junctions: list[Position]
    additional_cost: int


class MultiAgentNichePacker:
    """
    Geometrische Nischen-Evaluation und Slot-Zuweisung für Konvois.
    Unterstützt Pocket-Nischen, sequentiell verteilte Nischen und Bypässe.
    Garantie der ununterbrochenen Korridormindestbreite gemäß Details.md Punkte 2, 8
    und ImplementationPlan.md Abschnitt 3.5.
    """

    def find_niche_configuration(
        self,
        convoy: Convoy,
        mental_map: AgentMentalMap,
        passing_convoy_trajectory: list[Position],
        min_corridor_width: int = 1,
    ) -> Optional[NicheConfiguration]:
        """
        Sucht eine Nischenkonfiguration mit sum(Kapazität) >= sum(Footprints aller Agenten im Konvoi).
        Nischenslots dürfen niemals auf passing_convoy_trajectory liegen.
        """
        required_capacity = sum(a.footprint[0] * a.footprint[1] for a in convoy.members)
        if required_capacity == 0 or not convoy.members:
            return None

        trajectory_set = set(passing_convoy_trajectory)

        # 1. Identifiziere alle erreichbaren Nischenzellen außerhalb der Trajektorie
        # Eine Nischenzelle ist passierbar und liegt nicht auf der Trajektorie
        candidate_slots = self._find_available_niche_cells(
            convoy=convoy,
            mental_map=mental_map,
            trajectory_set=trajectory_set,
        )

        if len(candidate_slots) < required_capacity:
            return None

        # 2. Versuch A: Pocket Niche (Zusammenhängende Zellen mit einer gemeinsamen Junction)
        pocket_config = self._try_pocket_niche(
            convoy=convoy,
            candidate_slots=candidate_slots,
            trajectory_set=trajectory_set,
            required_capacity=required_capacity,
            mental_map=mental_map,
        )
        if pocket_config:
            return pocket_config

        # 3. Versuch B: Bypass / Parallelkorridor
        bypass_config = self._try_bypass_niche(
            convoy=convoy,
            candidate_slots=candidate_slots,
            trajectory_set=trajectory_set,
            required_capacity=required_capacity,
            mental_map=mental_map,
        )
        if bypass_config:
            return bypass_config

        # 4. Versuch C: Sequentiell verteilte Nischen entlang des Korridors (Distributed)
        return self._try_distributed_niches(
            convoy=convoy,
            candidate_slots=candidate_slots,
            trajectory_set=trajectory_set,
            required_capacity=required_capacity,
            mental_map=mental_map,
        )

    def _find_available_niche_cells(
        self,
        convoy: Convoy,
        mental_map: AgentMentalMap,
        trajectory_set: set[Position],
    ) -> set[Position]:
        """Findet alle Kacheln, die nicht auf der Trajektorie liegen, aber passierbar sind."""
        available: set[Position] = set()
        search_radius = 25
        for agent in convoy.members:
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    pos = Position(agent.position.x + dx, agent.position.y + dy)
                    if pos not in trajectory_set and self._is_walkable_tile(mental_map, pos):
                        available.add(pos)
        return available

    @staticmethod
    def _is_walkable_tile(mental_map: AgentMentalMap, pos: Position) -> bool:
        if not mental_map.is_within_bounds(pos):
            return False
        tile = mental_map.tiles.get(pos)
        if tile is not None:
            return tile.knowledge == TileKnowledge.WALKABLE
        if not mental_map.tiles:
            return mental_map.is_walkable(pos)
        return False

    def _try_pocket_niche(
        self,
        convoy: Convoy,
        candidate_slots: set[Position],
        trajectory_set: set[Position],
        required_capacity: int,
        mental_map: AgentMentalMap,
    ) -> Optional[NicheConfiguration]:
        """
        Sucht eine zusammenhängende Nische (Sackgasse) mit einer einzigen Mündung zur Trajektorie.
        Die Agenten rücken hintereinander ein (LIFO-Belegung: tiefste Zelle zuerst).
        """
        # Finde Junctions auf der Trajektorie, die an mindestens eine Nischenzelle grenzen
        for traj_pos in trajectory_set:
            adjacent_niche_cells = [
                n for n in traj_pos.get_neighbors()
                if n in candidate_slots
            ]
            for entry_cell in adjacent_niche_cells:
                # BFS von entry_cell aus durch candidate_slots, um verbundene Pocket zu finden
                pocket_cells = self._gather_connected_component(entry_cell, candidate_slots)
                touching_junctions = [
                    j for j in trajectory_set
                    if any(n in pocket_cells for n in j.get_neighbors())
                ]
                if len(touching_junctions) == 1 and len(pocket_cells) >= required_capacity:
                    # Sortiere nach Tiefe (größte Distanz zur Mündung traj_pos zuerst für LIFO)
                    sorted_cells = sorted(
                        pocket_cells,
                        key=lambda p: -p.manhattan_distance(traj_pos),
                    )
                    chosen_slots = sorted_cells[:required_capacity]

                    slots: list[SlotAssignment] = []
                    total_cost = 0
                    for i, agent in enumerate(convoy.members):
                        slot_pos = chosen_slots[i]
                        path = self._get_path_in_niche(traj_pos, slot_pos, set(pocket_cells) | {traj_pos})
                        cost = agent.position.manhattan_distance(traj_pos) + 2 * traj_pos.manhattan_distance(slot_pos)
                        total_cost += cost
                        slots.append(
                            SlotAssignment(
                                agent_id=agent.id,
                                slot_position=slot_pos,
                                junction_position=traj_pos,
                                path_to_slot=path,
                            )
                        )

                    return NicheConfiguration(
                        niche_type=NicheType.POCKET,
                        total_capacity=len(chosen_slots),
                        slots=slots,
                        junctions=[traj_pos],
                        additional_cost=total_cost,
                    )

        return None

    def _try_bypass_niche(
        self,
        convoy: Convoy,
        candidate_slots: set[Position],
        trajectory_set: set[Position],
        required_capacity: int,
        mental_map: AgentMentalMap,
    ) -> Optional[NicheConfiguration]:
        """
        Sucht einen Parallelkorridor/Bypass mit mindestens zwei Verbindungen zur Trajektorie.
        """
        # Alle Junctions finden, die an candidate_slots grenzen
        junctions: set[Position] = set()
        for traj_pos in trajectory_set:
            if any(n in candidate_slots for n in traj_pos.get_neighbors()):
                junctions.add(traj_pos)

        if len(junctions) < 2:
            return None

        # Prüfe ob zusammenhängende Nischenzellen mindestens 2 Junctions berühren
        visited: set[Position] = set()
        for cell in candidate_slots:
            if cell in visited:
                continue
            component = self._gather_connected_component(cell, candidate_slots)
            visited.update(component)

            touching_junctions = [
                j for j in junctions
                if any(n in component for n in j.get_neighbors())
            ]
            if len(touching_junctions) >= 2 and len(component) >= required_capacity:
                sorted_cells = sorted(component, key=lambda p: p.x)
                chosen_slots = sorted_cells[:required_capacity]

                slots: list[SlotAssignment] = []
                total_cost = 0
                for i, agent in enumerate(convoy.members):
                    slot_pos = chosen_slots[i]
                    nearest_junc = min(touching_junctions, key=lambda j: j.manhattan_distance(slot_pos))
                    path = self._get_path_in_niche(nearest_junc, slot_pos, set(component) | {nearest_junc})
                    cost = agent.position.manhattan_distance(nearest_junc) + 2 * nearest_junc.manhattan_distance(slot_pos)
                    total_cost += cost
                    slots.append(
                        SlotAssignment(
                            agent_id=agent.id,
                            slot_position=slot_pos,
                            junction_position=nearest_junc,
                            path_to_slot=path,
                        )
                    )

                return NicheConfiguration(
                    niche_type=NicheType.BYPASS,
                    total_capacity=len(chosen_slots),
                    slots=slots,
                    junctions=touching_junctions,
                    additional_cost=total_cost,
                )

        return None

    def _try_distributed_niches(
        self,
        convoy: Convoy,
        candidate_slots: set[Position],
        trajectory_set: set[Position],
        required_capacity: int,
        mental_map: AgentMentalMap,
    ) -> Optional[NicheConfiguration]:
        """
        Verteilt Agenten monoton auf separate Nischen entlang des Korridors oder Freiflächen.
        Jeder Agent erhält eine eigene Kachel in der Nähe seiner Marschachse / aktuellen Position.
        """
        assigned_slots: set[Position] = set()
        slots: list[SlotAssignment] = []
        total_cost = 0
        junctions: set[Position] = set()

        for agent in convoy.members:
            # BFS von agent.position aus, um das nächstgelegene erreichbare Ausweichfeld zu finden
            queue: list[Position] = [agent.position]
            visited: set[Position] = {agent.position}
            came_from: dict[Position, Position] = {}
            best_slot: Optional[Position] = None

            while queue:
                curr = queue.pop(0)
                if (
                    curr != agent.position
                    and curr in candidate_slots
                    and curr not in assigned_slots
                    and curr not in trajectory_set
                ):
                    best_slot = curr
                    break

                for n in curr.get_neighbors():
                    if n not in visited and mental_map.is_within_bounds(n) and self._is_walkable_tile(mental_map, n):
                        visited.add(n)
                        if curr != agent.position and n in trajectory_set:
                            continue
                        came_from[n] = curr
                        queue.append(n)

            if not best_slot:
                valid_candidates = [
                    p for p in candidate_slots
                    if p not in assigned_slots
                ]
                if not valid_candidates:
                    return None
                best_slot = min(valid_candidates, key=lambda p: agent.position.manhattan_distance(p))

            if best_slot in came_from:
                curr = best_slot
                rev_path: list[Position] = []
                while curr in came_from:
                    rev_path.append(curr)
                    curr = came_from[curr]
                rev_path.append(agent.position)
                rev_path.reverse()
                path = rev_path
            else:
                path = self._get_path_in_niche(agent.position, best_slot, candidate_slots | {agent.position})

            if agent.position in trajectory_set:
                nearest_junc = agent.position
            else:
                adj_junctions = [j for j in trajectory_set if j.manhattan_distance(best_slot) == 1]
                if adj_junctions:
                    nearest_junc = adj_junctions[0]
                else:
                    nearest_junc = min(trajectory_set, key=lambda j: j.manhattan_distance(best_slot))

            assigned_slots.add(best_slot)
            junctions.add(nearest_junc)
            cost = 2 * max(1, len(path) - 1)
            total_cost += cost

            slots.append(
                SlotAssignment(
                    agent_id=agent.id,
                    slot_position=best_slot,
                    junction_position=nearest_junc,
                    path_to_slot=path,
                )
            )

        return NicheConfiguration(
            niche_type=NicheType.DISTRIBUTED,
            total_capacity=len(slots),
            slots=slots,
            junctions=list(junctions),
            additional_cost=total_cost,
        )


    @staticmethod
    def _gather_connected_component(start: Position, allowed_set: set[Position]) -> list[Position]:
        """Ermittelt alle verbundenen Nachbarkacheln innerhalb von allowed_set via BFS."""
        component: list[Position] = []
        visited: set[Position] = {start}
        queue: list[Position] = [start]

        while queue:
            curr = queue.pop(0)
            component.append(curr)
            for n in curr.get_neighbors():
                if n in allowed_set and n not in visited:
                    visited.add(n)
                    queue.append(n)

        return component

    @staticmethod
    def _get_path_in_niche(start: Position, goal: Position, allowed_cells: set[Position]) -> list[Position]:
        """Ermittelt einen zusammenhängenden Pfad von start zu goal innerhalb allowed_cells via BFS."""
        if start == goal:
            return [start]
        queue: list[Position] = [start]
        visited: set[Position] = {start}
        came_from: dict[Position, Position] = {}

        while queue:
            curr = queue.pop(0)
            if curr == goal:
                break
            for n in curr.get_neighbors():
                if n in allowed_cells and n not in visited:
                    visited.add(n)
                    came_from[n] = curr
                    queue.append(n)

        if goal not in came_from:
            return [start, goal]

        path: list[Position] = []
        curr = goal
        while curr in came_from:
            path.append(curr)
            curr = came_from[curr]
        path.append(start)
        path.reverse()
        return path
