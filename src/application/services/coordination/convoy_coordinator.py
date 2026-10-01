from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.mental_map import AgentMentalMap, TileKnowledge
from src.domain.models.world.position import Position
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder


class DrainMode(str, Enum):
    ALTERNATING = "alternating"  # Reißverschlussverfahren (Modus A)
    BLOCKWISE = "blockwise"      # Sequenzielles Einfließen (Modus B)


class PermutationStep(str, Enum):
    LATERAL_STEP_OUT = "lateral_step_out"
    GAP_CLOSURE = "gap_closure"
    RE_INSERTION = "re_insertion"
    COMPLETED = "completed"


@dataclass
class Convoy:
    id: str
    members: list[Agent] = field(default_factory=list)
    direction_vector: tuple[int, int] = (0, 0)

    @property
    def leader(self) -> Agent:
        if not self.members:
            raise ValueError("Convoy has no members")
        return self.members[0]

    @property
    def tail(self) -> Agent:
        if not self.members:
            raise ValueError("Convoy has no members")
        return self.members[-1]

    def swap_members(self, i: int, j: int) -> None:
        self.members[i], self.members[j] = self.members[j], self.members[i]


@dataclass
class PermutationPlan:
    leader: Agent
    follower: Agent
    lateral_tile: Position
    original_leader_pos: Position
    step: PermutationStep = PermutationStep.LATERAL_STEP_OUT


class ConvoyCoordinator:
    """
    Deterministische Konvoikoordination, Permutation, Verzweigung (Mid-Convoy Branching)
    und Backtracking-Arbitrierung gemäß Details.md Punkte 7, 9, 10, 11 und ImplementationPlan.md 3.1, 3.5, 3.6.
    """

    def __init__(
        self,
        pathfinder: Optional[IPathfinder] = None,
        logger: Optional[IEventLogger] = None,
    ) -> None:
        self._pathfinder: Optional[IPathfinder] = pathfinder
        self._logger: Optional[IEventLogger] = logger

    @staticmethod
    def derive_direction(agent: Agent) -> tuple[int, int]:
        """Ermittelt den aktuellen Bewegungsvektor des Agenten aus seinem Pfad oder aktiven Ziel."""
        if agent.path:
            dx = agent.path[0].x - agent.position.x
            dy = agent.path[0].y - agent.position.y
            # Vorzeichen normieren auf -1, 0, 1
            sx = 0 if dx == 0 else (1 if dx > 0 else -1)
            sy = 0 if dy == 0 else (1 if dy > 0 else -1)
            return (sx, sy)

        goal = agent.active_goal
        if goal and goal.target_position:
            dx = goal.target_position.x - agent.position.x
            dy = goal.target_position.y - agent.position.y
            sx = 0 if dx == 0 else (1 if dx > 0 else -1)
            sy = 0 if dy == 0 else (1 if dy > 0 else -1)
            return (sx, sy)

        return (0, 0)

    def identify_convoys(self, agents: list[Agent]) -> list[Convoy]:
        """
        Gruppiert Agenten zu Konvois gemäß Konvoikriterium:
        Parallele/identische Trajektorien, gleicher Richtungsvektor und 1 <= L1(pos_i, pos_j) <= 2.
        Sortiert Agenten im Konvoi von Front (Leader) zu Tail.
        """
        if not agents:
            return []

        # Nach Richtungsvektoren gruppieren
        by_direction: dict[tuple[int, int], list[Agent]] = {}
        for agent in agents:
            dir_vec = self.derive_direction(agent)
            by_direction.setdefault(dir_vec, []).append(agent)

        convoys: list[Convoy] = []
        convoy_idx = 1

        for dir_vec, dir_agents in by_direction.items():
            if dir_vec == (0, 0):
                # Stationäre Agenten bilden Einzelkonvois
                for single in dir_agents:
                    convoys.append(Convoy(id=f"convoy_{convoy_idx}", members=[single], direction_vector=dir_vec))
                    convoy_idx += 1
                continue

            # Komponentenbildung über 1 <= L1 <= 2 Distanz
            unvisited_map: dict[str, Agent] = {a.id: a for a in dir_agents}
            while unvisited_map:
                start_id, start = unvisited_map.popitem()
                group = [start]
                queue = [start]

                while queue:
                    curr = queue.pop(0)
                    neighbors = [
                        other for other in list(unvisited_map.values())
                        if 1 <= curr.position.manhattan_distance(other.position) <= 2
                    ]
                    for n in neighbors:
                        del unvisited_map[n.id]
                        group.append(n)
                        queue.append(n)

                # Sortieren von Front zu Tail (in Richtung dir_vec am weitesten vorne)
                # Skalarprodukt mit dir_vec maximieren
                dx, dy = dir_vec
                group.sort(key=lambda a: -(a.position.x * dx + a.position.y * dy))

                convoys.append(Convoy(id=f"convoy_{convoy_idx}", members=group, direction_vector=dir_vec))
                convoy_idx += 1

        return convoys

    @staticmethod
    def select_leader(convoy: Convoy, conflict_point: Position) -> Agent:
        """Deterministische Leader-Wahl: Agent mit minimaler Manhattan-Distanz zum Konfliktbereich."""
        if not convoy.members:
            raise ValueError("Convoy has no members")
        return min(convoy.members, key=lambda a: a.position.manhattan_distance(conflict_point))

    @staticmethod
    def check_permutation_needed(convoy: Convoy) -> bool:
        """
        Prüft, ob das Ziel des Leaders topologisch hinter dem Ziel des Nachfolgers liegt.
        (Details.md Punkt 9; ImplementationPlan.md 3.5.1).
        """
        if len(convoy.members) < 2:
            return False

        leader = convoy.members[0]
        follower = convoy.members[1]

        l_goal = leader.active_goal
        f_goal = follower.active_goal

        if not l_goal or not f_goal or not l_goal.target_position or not f_goal.target_position:
            return False

        # In Bewegungsrichtung: wenn leader.target_position vor follower.target_position erreicht wird
        dx, dy = convoy.direction_vector
        leader_dist_to_goal = leader.position.manhattan_distance(l_goal.target_position)
        follower_dist_to_goal = follower.position.manhattan_distance(f_goal.target_position)

        # Wenn der Leader ein entfernteres Ziel hat als der Nachfolger, muss der Nachfolger zuerst vorbei
        return leader_dist_to_goal > follower_dist_to_goal

    def plan_permutation(
        self,
        convoy: Convoy,
        mental_map: AgentMentalMap,
        occupied_positions: Optional[set[Position]] = None,
    ) -> Optional[PermutationPlan]:
        """
        Plant einen Lateral Sidestep für den Leader auf eine freie Nachbarzelle orthogonal zur Marschachse.
        """
        if len(convoy.members) < 2:
            return None

        leader = convoy.members[0]
        follower = convoy.members[1]
        dx, dy = convoy.direction_vector
        occupied = occupied_positions or set()

        # Orthogonale Vektoren: (dx, dy) -> (-dy, dx) und (dy, -dx)
        orthogonal_offsets = [(-dy, dx), (dy, -dx)] if (dx, dy) != (0, 0) else [(0, 1), (0, -1), (1, 0), (-1, 0)]

        candidates: list[Position] = []
        for ox, oy in orthogonal_offsets:
            lateral_pos = Position(leader.position.x + ox, leader.position.y + oy)
            if (
                mental_map.is_walkable(lateral_pos)
                and lateral_pos not in occupied
                and lateral_pos != follower.position
            ):
                candidates.append(lateral_pos)

        if not candidates:
            return None

        # Bevorzuge Kacheln, die explizit in tiles als WALKABLE registriert sind
        candidates.sort(
            key=lambda p: 0 if (p in mental_map.tiles and mental_map.tiles[p].knowledge == TileKnowledge.WALKABLE) else 1
        )
        chosen_pos = candidates[0]

        return PermutationPlan(
            leader=leader,
            follower=follower,
            lateral_tile=chosen_pos,
            original_leader_pos=leader.position,
            step=PermutationStep.LATERAL_STEP_OUT,
        )

    def execute_permutation_step(
        self,
        convoy: Convoy,
        plan: PermutationPlan,
    ) -> PermutationStep:
        """
        Führt den nächsten atomaren Schritt der Permutation aus:
        LATERAL_STEP_OUT -> GAP_CLOSURE -> RE_INSERTION -> COMPLETED.
        """
        if plan.step == PermutationStep.LATERAL_STEP_OUT:
            plan.leader.position = plan.lateral_tile
            plan.step = PermutationStep.GAP_CLOSURE
            return plan.step

        elif plan.step == PermutationStep.GAP_CLOSURE:
            plan.follower.position = plan.original_leader_pos
            plan.step = PermutationStep.RE_INSERTION
            return plan.step

        elif plan.step == PermutationStep.RE_INSERTION:
            # Leader reiht sich hinter Follower ein (dessen ursprüngliche Position)
            plan.leader.position = plan.original_leader_pos
            convoy.swap_members(0, 1)
            plan.step = PermutationStep.COMPLETED
            return plan.step

        return PermutationStep.COMPLETED

    @staticmethod
    def plan_mid_convoy_branching(
        convoy: Convoy,
        junction_pos: Position,
        mode: DrainMode = DrainMode.ALTERNATING,
    ) -> list[Agent]:
        """
        Geordnetes Abfließen bei Mid-Convoy Branching (Details.md Punkt 10; ImplementationPlan.md 3.5.2).
        Agent an der Abzweigung A_m wird neuer Leader.
        Modus A: Reißverschlussverfahren (alternierend aus Tail und Head).
        Modus B: Sequenziell (kürzere Räumdistanz zuerst komplett).
        """
        members = list(convoy.members)
        if not members:
            return []

        # Index des Agenten an der Junction finden
        idx = next((i for i, a in enumerate(members) if a.position == junction_pos), None)
        if idx is None:
            # Nächsten Agenten zur Junction wählen
            idx = min(range(len(members)), key=lambda i: members[i].position.manhattan_distance(junction_pos))

        m = members[idx]
        h_head = list(reversed(members[:idx]))  # A_{m-1} bis A_0 (rückwärts zur Junction)
        h_tail = members[idx + 1:]             # A_{m+1} bis A_k (vorwärts zur Junction)

        reconstituted: list[Agent] = [m]

        if mode == DrainMode.ALTERNATING:
            # Reißverschlussverfahren: alternierend aus Tail und Head
            while h_tail or h_head:
                if h_tail:
                    reconstituted.append(h_tail.pop(0))
                if h_head:
                    reconstituted.append(h_head.pop(0))
        else:
            # Modus B: Sequenziell (kürzere Resthälfte fließt zuerst ab)
            if len(h_tail) <= len(h_head):
                reconstituted.extend(h_tail)
                reconstituted.extend(h_head)
            else:
                reconstituted.extend(h_head)
                reconstituted.extend(h_tail)

        return reconstituted

    @staticmethod
    def find_nearest_backward_junction(
        tail_pos: Position,
        direction_vector: tuple[int, int],
        mental_map: AgentMentalMap,
        max_search_depth: int = 30,
    ) -> Optional[Position]:
        """
        Sucht vom Tail ausgehend rückwärts (entgegen direction_vector) die nächste Kachel mit einer Verzweigung
        (mehr als 2 passierbare Nachbarn).
        """
        dx, dy = direction_vector
        backward_dx, backward_dy = -dx, -dy
        curr = tail_pos

        ortho_offsets = [(-dy, dx), (dy, -dx)] if (dx, dy) != (0, 0) else [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for _ in range(max_search_depth):
            curr = Position(curr.x + backward_dx, curr.y + backward_dy)
            if not mental_map.is_walkable(curr):
                break
            # Prüfen ob an curr eine transversale Abzweigung existiert, die explizit passierbar ist
            for ox, oy in ortho_offsets:
                lateral = Position(curr.x + ox, curr.y + oy)
                tile = mental_map.tiles.get(lateral)
                if tile is not None and tile.knowledge == TileKnowledge.WALKABLE:
                    return curr

        return None

    def evaluate_backtracking(
        self,
        convoy_1: Convoy,
        convoy_2: Convoy,
        map_1: AgentMentalMap,
        map_2: AgentMentalMap,
    ) -> tuple[Convoy, Position]:
        """
        Backtracking-Protokoll bei Null-Nischen (Details.md Punkt 11; ImplementationPlan.md 3.6):
        Berechnet für beide Gruppen die Distanz des jeweiligen Tails zur nächsten freien Verzweigung.
        Die Gruppe mit kürzerer Distanz weicht zurück bis zu 'backtracking_junction_target'.
        """
        junc_1 = self.find_nearest_backward_junction(convoy_1.tail.position, convoy_1.direction_vector, map_1)
        junc_2 = self.find_nearest_backward_junction(convoy_2.tail.position, convoy_2.direction_vector, map_2)

        dist_1 = convoy_1.tail.position.manhattan_distance(junc_1) if junc_1 else 9999
        dist_2 = convoy_2.tail.position.manhattan_distance(junc_2) if junc_2 else 9999

        if dist_1 <= dist_2 and junc_1 is not None:
            return (convoy_1, junc_1)
        elif junc_2 is not None:
            return (convoy_2, junc_2)
        else:
            # Fallback: Falls keine Verzweigung gefunden, Position des Tails als Ankerpunkt
            fallback_pos = convoy_1.tail.position if dist_1 <= dist_2 else convoy_2.tail.position
            return (convoy_1 if dist_1 <= dist_2 else convoy_2, fallback_pos)
