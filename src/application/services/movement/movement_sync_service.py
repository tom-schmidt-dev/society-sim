from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.events import SimulationEvent
from src.domain.models.world.position import Position
from src.domain.models.coordination.reservation_table import TileReservationIntent, ReservationTable
from src.domain.ports.event_logger import IEventLogger


class IGridWalkable(Protocol):
    def is_walkable(self, pos: Position) -> bool: ...
    def is_within_bounds(self, pos: Position) -> bool: ...


@dataclass(slots=True)
class ComponentSyncResult:
    component_id: int
    agent_ids: set[str]
    is_committed: bool
    reason: Optional[str] = None
    movements: dict[str, Position] = field(default_factory=dict)


@dataclass(slots=True)
class MovementSyncResult:
    committed_agents: set[str] = field(default_factory=set)
    failed_agents: set[str] = field(default_factory=set)
    component_results: list[ComponentSyncResult] = field(default_factory=list)


class MovementSyncService:
    """
    Lokal beschränkter Zweiphasen-Commit für Bewegungssynchronisation.
    Gemäß Details.md Punkt 12 und ImplementationPlan.md Abschnitt 3.3.

    Phase 1: Intents sammeln, Konflikt-Zusammenhangskomponenten isolieren,
             und gerichtet validieren (Front-to-Tail bei Vorwärtsfahrt,
             Tail-to-Front bei Backtracking).
    Phase 2: Atomare Grid-Übertragung ausschließlich für fehlerfreie Komponenten;
             isolierter Abbruch und Verharren für gestörte Komponenten.
    """

    def __init__(
        self,
        logger: Optional[IEventLogger] = None,
        reservation_table: Optional[ReservationTable] = None,
    ) -> None:
        self._logger: Optional[IEventLogger] = logger
        self._reservation_table: ReservationTable = reservation_table or ReservationTable()

    def execute_two_phase_commit(
        self,
        agents: list[Agent],
        intents: list[TileReservationIntent],
        grid: IGridWalkable,
        occupied_positions: Optional[set[Position]] = None,
        tick: int = 0,
    ) -> MovementSyncResult:
        """
        Führt den zweiphasigen Bewegungstransaktions-Commit aus:
        1. Phase 1: Validierung aller Intents pro zusammenhängender Komponente.
        2. Phase 2: Atomarer Commit fehlerfreier Komponenten, isoliertes Verharren fehlerhafter Komponenten.
        """
        agent_map: dict[str, Agent] = {a.id: a for a in agents}
        intent_map: dict[str, TileReservationIntent] = {i.agent_id: i for i in intents}

        result = MovementSyncResult()

        if not intent_map:
            return result

        # Stationäre Positionen (Agenten ohne Intent oder externe statische Hindernisse)
        stationary_positions: set[Position] = set(occupied_positions or set())
        for a in agents:
            if a.id not in intent_map:
                stationary_positions.add(a.position)

        # Schritt 1: Konflikt-Graph aufbauen und Zusammenhangskomponenten ermitteln
        components = self._build_connected_components(intents)

        # Schritt 2: Validierung & Commit komponentenweise
        for comp_idx, comp_agents in enumerate(components):
            comp_intents = [intent_map[aid] for aid in comp_agents if aid in intent_map]
            if not comp_intents:
                continue

            is_valid, reason, validation_order = self._validate_component(
                comp_intents=comp_intents,
                grid=grid,
                stationary_positions=stationary_positions,
            )

            if is_valid:
                # Phase 2: Atomarer Commit für die Komponente
                movements: dict[str, Position] = {}
                for intent in comp_intents:
                    ag = agent_map[intent.agent_id]
                    ag.position = intent.desired_position
                    if ag.path and ag.path[0] == intent.desired_position:
                        ag.path.pop(0)
                    movements[intent.agent_id] = intent.desired_position
                    result.committed_agents.add(intent.agent_id)

                comp_res = ComponentSyncResult(
                    component_id=comp_idx,
                    agent_ids=comp_agents,
                    is_committed=True,
                    movements=movements,
                )
                result.component_results.append(comp_res)

                if self._logger:
                    self._logger.log(
                        SimulationEvent(
                            tick=tick,
                            agent_id="SYSTEM",
                            event_type="movement_component_committed",
                            summary=f"Bewegungskomponente {comp_idx} ({len(comp_agents)} Agenten) atomar committed.",
                            payload={
                                "component_id": comp_idx,
                                "agent_ids": list(comp_agents),
                                "movements": {k: (v.x, v.y) for k, v in movements.items()},
                            },
                        )
                    )
            else:
                # Isolierter Abbruch für diese Komponente
                for aid in comp_agents:
                    result.failed_agents.add(aid)

                comp_res = ComponentSyncResult(
                    component_id=comp_idx,
                    agent_ids=comp_agents,
                    is_committed=False,
                    reason=reason,
                )
                result.component_results.append(comp_res)

                if self._logger:
                    self._logger.log(
                        SimulationEvent(
                            tick=tick,
                            agent_id="SYSTEM",
                            event_type="movement_component_failed",
                            summary=f"Bewegungskomponente {comp_idx} blockiert: {reason}. Agenten verharren.",
                            payload={
                                "component_id": comp_idx,
                                "agent_ids": list(comp_agents),
                                "reason": reason,
                            },
                        )
                    )

        return result

    def _build_connected_components(
        self,
        intents: list[TileReservationIntent],
    ) -> list[set[str]]:
        """
        Gruppiert Agenten basierend auf räumlichen und relationalen Abhängigkeiten
        in disjunkte Zusammenhangskomponenten.
        Kanten entstehen bei:
        - Gemeinsamem Ziel (Contention)
        - Abhängigem Nachrücken (Following: desired == current eines anderen Agenten)
        - Swap-Konflikten
        """
        agent_ids = [i.agent_id for i in intents]
        adj: dict[str, set[str]] = {aid: set() for aid in agent_ids}

        for i in range(len(intents)):
            for j in range(i + 1, len(intents)):
                a1 = intents[i]
                a2 = intents[j]

                has_conflict = False

                # 1. Zielkollision (gleiches desired_position)
                if a1.desired_position == a2.desired_position:
                    has_conflict = True

                # 2. Aufrücken / Hinterherfahren (desired_position == current_position)
                if a1.desired_position == a2.current_position or a2.desired_position == a1.current_position:
                    has_conflict = True

                # 3. Swap (A will zu B, B will zu A)
                if a1.desired_position == a2.current_position and a2.desired_position == a1.current_position:
                    has_conflict = True

                if has_conflict:
                    adj[a1.agent_id].add(a2.agent_id)
                    adj[a2.agent_id].add(a1.agent_id)

        # BFS zur Ermittlung der Zusammenhangskomponenten
        components: list[set[str]] = []
        visited: set[str] = set()

        for aid in agent_ids:
            if aid not in visited:
                comp: set[str] = set()
                queue = [aid]
                visited.add(aid)
                while queue:
                    curr = queue.pop(0)
                    comp.add(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(comp)

        return components

    def _validate_component(
        self,
        comp_intents: list[TileReservationIntent],
        grid: IGridWalkable,
        stationary_positions: set[Position],
    ) -> tuple[bool, Optional[str], list[TileReservationIntent]]:
        """
        Validiert eine Zusammenhangskomponente:
        1. Statische Passierbarkeit und Bounds
        2. Keine Kollision mit externen stationären Entitäten
        3. Keine mehrfache Belegung desselben Zieles
        4. Keine direkten Swap-Kollisionen
        5. Gerichtete Validierungsreihenfolge:
           - Vorwärts: Front-to-Tail
           - Backtracking: Tail-to-Front
        """
        # 1. Bounds und statische Hindernisse
        for intent in comp_intents:
            if not grid.is_within_bounds(intent.desired_position):
                return False, f"Agent {intent.agent_id}: Ziel {intent.desired_position} liegt außerhalb der Map-Grenzen", []
            if not grid.is_walkable(intent.desired_position):
                return False, f"Agent {intent.agent_id}: Ziel {intent.desired_position} ist statisches Hindernis", []

        # 2. Stationäre Entitäten
        for intent in comp_intents:
            if intent.desired_position in stationary_positions:
                return False, f"Agent {intent.agent_id}: Ziel {intent.desired_position} durch stationäre Entität belegt", []

        # 3. Mehrfache Belegung desselben Zielfeldes
        targets: set[Position] = set()
        for intent in comp_intents:
            if intent.desired_position in targets:
                return False, f"Mehrfache Reservierung für Zielkachel {intent.desired_position}", []
            targets.add(intent.desired_position)

        # 4. Swap-Kollisionen prüfen
        for i in range(len(comp_intents)):
            for j in range(i + 1, len(comp_intents)):
                a1 = comp_intents[i]
                a2 = comp_intents[j]
                if a1.desired_position == a2.current_position and a2.desired_position == a1.current_position:
                    return False, f"Swap-Kollision zwischen Agent {a1.agent_id} und {a2.agent_id}", []

        # 5. Gerichtete Validierungsreihenfolge (Front-to-Tail vs. Tail-to-Front)
        is_backtracking = any(i.is_backtracking for i in comp_intents)

        # Gerichteter Abhängigkeitsgraph: A hängt von B ab, wenn A.desired == B.current
        # (B muss erst sein Feld räumen, bevor A hineintreten darf)
        dependencies: dict[str, set[str]] = {i.agent_id: set() for i in comp_intents}
        current_pos_to_agent: dict[Position, str] = {i.current_position: i.agent_id for i in comp_intents}

        for intent in comp_intents:
            occupant_id = current_pos_to_agent.get(intent.desired_position)
            if occupant_id and occupant_id != intent.agent_id:
                dependencies[intent.agent_id].add(occupant_id)

        # Topologische Simulation: Wer kann als Erstes ziehen?
        # Ein Agent kann ziehen, wenn sein Ziel nicht von einem anderen Agenten der Komponente besetzt ist,
        # der noch nicht gezogen ist.
        intent_by_id = {i.agent_id: i for i in comp_intents}
        vacated_positions: set[Position] = set()
        remaining = set(intent_by_id.keys())
        ordered_intents: list[TileReservationIntent] = []

        while remaining:
            # Finde alle Agenten, deren gewünschtes Feld frei ist:
            # d.h. gewünschtes Feld ist nicht von einem verbleibenden Agenten belegt
            ready_agents = [
                aid for aid in remaining
                if intent_by_id[aid].desired_position not in {intent_by_id[r].current_position for r in remaining if r != aid}
            ]

            if not ready_agents:
                # Zyklische Blockade / Deadlock
                return False, "Zyklische Blockade in der Bewegungskomponente erkannt", []

            # Determinismus: Bei mehreren abzugsbereiten Agenten sortieren
            # Bei Backtracking: Tail zuerst (höchste Distanz oder is_backtracking Flag)
            # Bei Vorwärts: Front zuerst (niedrigste Restdistanz)
            if is_backtracking:
                ready_agents.sort(
                    key=lambda aid: (-intent_by_id[aid].distance_to_goal, aid)
                )
            else:
                ready_agents.sort(
                    key=lambda aid: (intent_by_id[aid].distance_to_goal, aid)
                )

            chosen = ready_agents[0]
            ordered_intents.append(intent_by_id[chosen])
            vacated_positions.add(intent_by_id[chosen].current_position)
            remaining.remove(chosen)

        return True, None, ordered_intents
