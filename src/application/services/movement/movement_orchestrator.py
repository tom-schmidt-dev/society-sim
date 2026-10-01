from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from src.application.services.coordination.agent_protocol_service import AgentProtocolService
from src.application.services.movement.movement_sync_service import MovementSyncService
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.goal import ExecutionPriority, Goal
from src.domain.models.world.position import Position
from src.domain.models.coordination.reservation_table import ReservationTable, TileReservationIntent
from src.domain.models.world.world import WorldGrid
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.conflict_coordinator import IConflictCoordinator
from src.domain.ports.pathfinder import IPathfinder


class MovementOrchestrator:
    """Orchestriert Phase 2 des Simulationszyklus: Absichtserfassung, Arbitrierung, Bewegung und Kollisionsauflösung."""

    def __init__(
        self,
        grid: WorldGrid,
        pathfinder: IPathfinder,
        movement_sync_service: MovementSyncService,
        reservation_table: ReservationTable,
        protocol_service: AgentProtocolService,
        conflict_coordinator: Optional[IConflictCoordinator] = None,
        on_agent_moved: Optional[Callable[[Agent], Any]] = None,
        tick_provider: Optional[Callable[[], int]] = None,
    ) -> None:
        self._grid = grid
        self._pathfinder = pathfinder
        self._movement_sync_service = movement_sync_service
        self._reservation_table = reservation_table
        self._protocol_service = protocol_service
        self._conflict_coordinator = conflict_coordinator
        self._on_agent_moved = on_agent_moved
        self._tick_provider = tick_provider or (lambda: 0)

    def execute_physical_movement(
        self,
        agents: list[Agent],
        entities: list[WorldEntity],
        delayed_agent_ids: set[str],
        background_tasks: set[asyncio.Task[Any]],
    ) -> None:
        """Führt Arbitrierung, Zweiphasen-Commit und Konfliktbehandlung für alle Agenten aus."""
        self._reservation_table.clear()

        # 1. Schrittwünsche sammeln und reservieren
        intents: list[TileReservationIntent] = []
        for agent in agents:
            if agent.id in delayed_agent_ids:
                continue
            if agent.has_path and not agent.is_busy:
                target_pos = agent.path[0]
                active_goal = agent.active_goal
                prio = active_goal.priority if active_goal else ExecutionPriority.ROUTINE
                dist = len(agent.path)
                is_backtracking = bool(
                    active_goal and active_goal.backtracking_junction_target is not None
                )
                intent = TileReservationIntent(
                    agent_id=agent.id,
                    current_position=agent.position,
                    desired_position=target_pos,
                    priority=prio,
                    distance_to_goal=dist,
                    is_backtracking=is_backtracking,
                )
                intents.append(intent)
                self._reservation_table.request_reservation(intent)

        # 2. Physische Ausführung via Zweiphasen-Commit
        occupied_stationary = {e.position for e in entities if not isinstance(e, Agent)}
        current_tick = self._tick_provider()
        sync_result = self._movement_sync_service.execute_two_phase_commit(
            agents=agents,
            intents=intents,
            grid=self._grid,
            occupied_positions=occupied_stationary,
            tick=current_tick,
        )

        for agent in agents:
            if agent.id in sync_result.committed_agents:
                self.handle_committed_agent_post_move(agent, entities)
            elif not agent.has_path or agent.is_busy:
                self._protocol_service.check_and_signal_clearance(agent, entities)
            else:
                self.handle_blocked_agent(agent, entities, background_tasks)

    def handle_committed_agent_post_move(
        self, agent: Agent, entities: list[WorldEntity]
    ) -> None:
        """Aktualisiert Sensorik und Signalisierung für erfolgreich bewegte Agenten."""
        if self._on_agent_moved:
            self._on_agent_moved(agent)

        current_goal = agent.active_goal
        if current_goal is not None:
            self._protocol_service.signal_niche_junction_entry(agent, current_goal, entities)

            if (
                current_goal.target_position is not None
                and agent.position == current_goal.target_position
                and not current_goal.is_evasion_hold
                and current_goal.junction_position is not None
            ):
                self._protocol_service.handle_niche_arrival(agent, current_goal, entities)

        self._protocol_service.check_and_signal_clearance(agent, entities)

    def handle_blocked_agent(
        self,
        agent: Agent,
        entities: list[WorldEntity],
        background_tasks: set[asyncio.Task[Any]],
    ) -> None:
        """Initiiert Konfliktauflösung für blockierte Bewegungsschritte."""
        next_pos = agent.path[0]
        blocker = next((e for e in entities if e.position == next_pos), None)
        current_tick = self._tick_provider()

        if blocker:
            if isinstance(blocker, Agent):
                blocker_goal = blocker.active_goal
                if (
                    blocker.is_evasion_locked
                    or (blocker_goal and blocker_goal.yield_for_agent_id == agent.id)
                ):
                    return

                if self._conflict_coordinator:
                    agent.is_thinking = True
                    task = asyncio.create_task(
                        self._conflict_coordinator.resolve_blockage(
                            agent, blocker, next_pos, entities
                        )
                    )
                    background_tasks.add(task)
                    task.add_done_callback(background_tasks.discard)
            else:
                if (
                    not agent.memory.is_epistemically_exhausted(blocker.id)
                    and self._conflict_coordinator
                ):
                    agent.is_thinking = True
                    task = asyncio.create_task(
                        self._conflict_coordinator.resolve_blockage(
                            agent, blocker, next_pos, entities
                        )
                    )
                    background_tasks.add(task)
                    task.add_done_callback(background_tasks.discard)
                else:
                    agent.mental_map.mark_obstacle(next_pos, current_tick)
                    self.replan_around_obstacle(agent)
        elif not self._grid.is_walkable(next_pos):
            agent.mental_map.mark_obstacle(next_pos, current_tick)
            self.replan_around_obstacle(agent)

    def replan_around_obstacle(self, agent: Agent) -> None:
        """Berechnet Pfad neu, wenn ein statisches Hindernis festgestellt wurde."""
        active_goal = agent.active_goal
        if active_goal and active_goal.target_position and not active_goal.is_evasion_hold:
            new_path = self._pathfinder.find_path(
                agent.position, active_goal.target_position, agent.mental_map
            )
            if new_path:
                agent.assign_path(new_path)
            else:
                agent.clear_path()