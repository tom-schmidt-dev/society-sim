from __future__ import annotations

import asyncio
from typing import Any, Optional
from src.application.services.action_executor import ActionExecutor
from src.application.services.conflict_coordinator import ConflictCoordinator
from src.application.services.convoy_arbitrator import ConvoyArbitrator
from src.application.services.convoy_coordinator import ConvoyCoordinator
from src.application.services.critical_section_coordinator import CriticalSectionCoordinator
from src.application.services.dialogue_coordinator import DialogueCoordinator
from src.application.services.dialogue_history import DialogueHistory
from src.application.services.dialogue_session_manager import DialogueSessionManager
from src.application.services.evasion_finder import EvasionFinder
from src.application.services.goal_service import GoalService
from src.application.services.movement_sync_service import MovementSyncService
from src.application.services.multi_agent_niche_packer import MultiAgentNichePacker
from src.application.services.target_search_service import TargetSearchService
from src.application.services.need_service import NeedService
from src.application.services.plan_decomposition_service import PlanDecompositionService
from src.domain.models.agent import Agent
from src.domain.models.communication_templates import DialogueTemplates
from src.domain.models.events import SimulationEvent
from src.domain.models.evasion_phase import EvasionPhase
from src.domain.models.goal import ExecutionPriority, Goal
from src.domain.models.message import CommunicationChannel, IncomingMessage
from src.domain.models.position import Position
from src.domain.models.reservation_table import ReservationTable, TileReservationIntent
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.conflict_coordinator import IConflictCoordinator
from src.domain.ports.dialogue_coordinator import IDialogueCoordinator
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter
from src.domain.services.perception_service import PerceptionService

from src.domain.models.agent_memory import EntityFact
from src.application.services.frontier_explorer import FrontierExplorer
from src.domain.services.precondition_evaluator import PreconditionEvaluator
from src.application.services.agent_protocol_service import AgentProtocolService

class SimulationEngine:
    def __init__(
        self,
        grid: WorldGrid,
        pathfinder: IPathfinder,
        presenter: IPresenter,
        logger: IEventLogger,
        cognition_provider: ICognitionProvider,
        tick_interval: float = 0.3,
        goal_service: Optional[GoalService] = None,
        conflict_coordinator: Optional[IConflictCoordinator] = None,
        dialogue_coordinator: Optional[IDialogueCoordinator] = None,
        dialogue_history: Optional[DialogueHistory] = None,
        perception_service: Optional[PerceptionService] = None,
        target_search_service: Optional[TargetSearchService] = None,
        critical_section_coordinator: Optional[CriticalSectionCoordinator] = None,
        auditory_radius: int = 3,
        movement_sync_service: Optional[MovementSyncService] = None,
        convoy_coordinator: Optional[ConvoyCoordinator] = None,
        niche_packer: Optional[MultiAgentNichePacker] = None,
        convoy_arbitrator: Optional[ConvoyArbitrator] = None,
        need_service: Optional[NeedService] = None,  # <-- NEU
        plan_decomposition_service: Optional[PlanDecompositionService] = None,  # <-- NEU
        enable_deterministic_corridor: bool = True,
    ) -> None:
        self._entities: list[WorldEntity] = []
        self._grid: WorldGrid = grid
        self._pathfinder: IPathfinder = pathfinder
        self._presenter: IPresenter = presenter
        self._logger: IEventLogger = logger
        self._tick_interval: float = tick_interval
        self._auditory_radius: int = auditory_radius

        self._agents: list[Agent] = []
        self._current_tick: int = 0
        self._is_running: bool = False
        self._background_tasks: set[asyncio.Task[Any]] = set()

        self._reservation_table = ReservationTable()
        self._dialogue_history = dialogue_history or DialogueHistory()
        self._perception_service = perception_service or PerceptionService(default_radius=3)
        self._goal_service = goal_service or GoalService(
            logger, cognition_provider, pathfinder, tick_provider=lambda: self._current_tick
        )
        self._evasion_finder = EvasionFinder(pathfinder)
        self._session_manager = DialogueSessionManager(max_dialogue_turns=2, logger=self._logger)
        self._critical_section_coordinator = (
            critical_section_coordinator
            or CriticalSectionCoordinator(logger=self._logger, tick_provider=lambda: self._current_tick)
        )
        self._frontier_explorer = FrontierExplorer()
        self._precondition_evaluator = PreconditionEvaluator()

        self._need_service = need_service or NeedService()

        self._delayed_agent_ids: set[str] = set()

        self._plan_decomposition_service = (
            plan_decomposition_service
            or PlanDecompositionService(cognition_provider=cognition_provider)
        )

        self._target_search_service = target_search_service or TargetSearchService(
            pathfinder=self._pathfinder,
            perception_service=self._perception_service,
            evasion_finder=self._evasion_finder,
            logger=self._logger,
        )

        # ANPASSUNG: need_service an ActionExecutor übergeben
        self._action_executor = ActionExecutor(
            grid=self._grid,
            logger=self._logger,
            dialogue_history=self._dialogue_history,
            goal_service=self._goal_service,
            pathfinder=self._pathfinder,
            perception_service=self._perception_service,
            evasion_finder=self._evasion_finder,
            critical_section_coordinator=self._critical_section_coordinator,
            need_service=self._need_service,
            tick_provider=lambda: self._current_tick,
        )

        self._movement_sync_service = movement_sync_service or MovementSyncService(
            logger=self._logger,
            reservation_table=self._reservation_table,
        )
        self._convoy_coordinator = convoy_coordinator or ConvoyCoordinator(
            pathfinder=self._pathfinder,
            logger=self._logger,
        )
        self._niche_packer = niche_packer or MultiAgentNichePacker()
        self._convoy_arbitrator = convoy_arbitrator or ConvoyArbitrator()
        self._conflict_coordinator = conflict_coordinator
        self._dialogue_coordinator = dialogue_coordinator
        self._conflict_coordinator = conflict_coordinator or ConflictCoordinator(
            logger=self._logger,
            cognition_provider=cognition_provider,
            pathfinder=self._pathfinder,
            goal_service=self._goal_service,
            evasion_finder=self._evasion_finder,
            action_executor=self._action_executor,
            session_manager=self._session_manager,
            dialogue_history=self._dialogue_history,
            tick_provider=lambda: self._current_tick,
            convoy_coordinator=self._convoy_coordinator,
            niche_packer=self._niche_packer,
            convoy_arbitrator=self._convoy_arbitrator,
            enable_deterministic_corridor=enable_deterministic_corridor,

        )
        self._dialogue_coordinator = dialogue_coordinator or DialogueCoordinator(
            logger=self._logger,
            cognition_provider=cognition_provider,
            pathfinder=self._pathfinder,
            goal_service=self._goal_service,
            evasion_finder=self._evasion_finder,
            action_executor=self._action_executor,
            session_manager=self._session_manager,
            dialogue_history=self._dialogue_history,
            tick_provider=lambda: self._current_tick,
        )

        self._protocol_service = AgentProtocolService(
            grid=self._grid,
            pathfinder=self._pathfinder,
            logger=self._logger,
            goal_service=self._goal_service,
            dialogue_history=self._dialogue_history,
            evasion_finder=self._evasion_finder,
            dialogue_coordinator=self._dialogue_coordinator,
            conflict_coordinator=self._conflict_coordinator,
            auditory_radius=self._auditory_radius,
            tick_provider=lambda: self._current_tick,


        )

    @property
    def current_tick(self) -> int:
        return self._current_tick

    @property
    def dialogue_history(self) -> DialogueHistory:
        return self._dialogue_history

    @property
    def critical_section_coordinator(self) -> CriticalSectionCoordinator:
        return self._critical_section_coordinator

    @property
    def movement_sync_service(self) -> MovementSyncService:
        return self._movement_sync_service

    @property
    def convoy_coordinator(self) -> ConvoyCoordinator:
        return self._convoy_coordinator

    @property
    def niche_packer(self) -> MultiAgentNichePacker:
        return self._niche_packer

    @property
    def convoy_arbitrator(self) -> ConvoyArbitrator:
        return self._convoy_arbitrator

    def register_entity(self, entity: WorldEntity) -> None:
        if not self._grid.is_walkable(entity.position):
            raise ValueError(f"Position {entity.position} für Objekt {entity.name} ist blockiert.")
        self._entities.append(entity)
        self._logger.log(
            SimulationEvent(
                tick=self._current_tick,
                agent_id=entity.id,
                event_type="entity_registered",
                summary=f"Objekt '{entity.name}' platziert.",
                payload={"x": entity.position.x, "y": entity.position.y},
            )
        )

    def register_agent(self, agent: Agent) -> None:
        if not self._grid.is_walkable(agent.position):
            raise ValueError(f"Startposition {agent.position} für Agent {agent.name} blockiert.")
        if agent not in self._entities:
            self._entities.append(agent)
        self._agents.append(agent)
        agent.mental_map.set_bounds(self._grid.width, self._grid.height)
        self._logger.log(
            SimulationEvent(
                tick=self._current_tick,
                agent_id=agent.id,
                event_type="agent_registered",
                summary=f"Agent {agent.name} registriert an ({agent.position.x}, {agent.position.y}).",
            )
        )

    def set_agent_target(self, agent_id: str, target: Position, destination_name: str = "Ziel") -> None:
        agent = next((a for a in self._agents if a.id == agent_id), None)
        if not agent:
            raise ValueError(f"Agent '{agent_id}' nicht gefunden.")

        agent.goals.clear()
        self._goal_service.push_goal(agent, Goal(name=destination_name, target_position=target))
        self._update_agent_perception(agent)
        path = self._pathfinder.find_path(agent.position, target, agent.mental_map)
        if path:
            agent.assign_path(path)

    def _update_agent_perception(self, agent: Agent) -> list[WorldEntity]:
        visible_positions = self._perception_service.get_visible_positions(agent.position, self._grid)
        for pos in visible_positions:
            tile_type = self._grid.get_tile_type(pos)
            if not self._grid.is_walkable(pos) or agent.memory.get_type_assumed_walkable(tile_type) is False:
                agent.mental_map.mark_obstacle(pos, self._current_tick)
            else:
                agent.mental_map.update_tile(pos, is_walkable=True, tick=self._current_tick)

        visible_entities = self._perception_service.get_visible_entities(agent, self._entities)
        for visible_entity in visible_entities:
            agent.memory.update_entity_perception(
                entity_id=visible_entity.id,
                name=visible_entity.name,
                pos=visible_entity.position,
                tick=self._current_tick,
                entity_type=visible_entity.entity_type,
            )
            if (
                agent.memory.is_epistemically_exhausted(visible_entity.id)
                and agent.memory.get_assumed_walkable(visible_entity.id) is False
            ):
                agent.mental_map.mark_obstacle(visible_entity.position, self._current_tick)

        return visible_entities

    def _collect_known_positions(self) -> set[Position]:
        discovered: set[Position] = set()
        for agent in self._agents:
            discovered.update(agent.mental_map.tiles.keys())
        return discovered



    def _notify_next_critical_section_holder(self, resource_key: str, next_agent_id: str) -> None:
        """Aktiviert den nächsten Agenten in der Warteschlange und stößt die Notwendigkeitsprüfung an."""
        next_agent = next((a for a in self._agents if a.id == next_agent_id), None)
        if not next_agent or not next_agent.active_goal:
            return

        is_completed = self._critical_section_coordinator.is_completed(resource_key)
        is_still_needed = self._goal_service.validate_goal_necessity(
            agent=next_agent,
            goal=next_agent.active_goal,
            is_already_completed=is_completed,
            incident_id=f"cs-notify-t{self._current_tick}-{resource_key}",
        )
        if is_still_needed:
            self._goal_service.resume_goal(next_agent)
            active_goal = next_agent.active_goal
            if active_goal and active_goal.target_position:
                new_path = self._pathfinder.find_path(
                    next_agent.position, active_goal.target_position, next_agent.mental_map
                )
                if new_path:
                    next_agent.assign_path(new_path)


    def _interrupt_goal_for_replan(
        self,
        agent: Agent,
        reason: str,
        discovered_fact: Optional[EntityFact] = None,
    ) -> None:
        """Bricht aktive Pfade und Ziele kontrolliert ab, um eine Neu-Dekompensation anzustoßen."""
        agent.clear_path()
        agent.goals.clear()

        payload: dict[str, Any] = {"reason": reason}
        if discovered_fact:
            payload["discovered_entity_id"] = discovered_fact.entity_id
            payload["entity_type"] = discovered_fact.entity_type
            if discovered_fact.last_known_position:
                payload["position"] = [
                    discovered_fact.last_known_position.x,
                    discovered_fact.last_known_position.y,
                ]

        self._logger.log(
            SimulationEvent(
                tick=self._current_tick,
                agent_id=agent.id,
                event_type="goal_interrupted_for_replan",
                summary=f"Agent {agent.name}: Ziel abgebrochen wegen '{reason}'. Re-Planung initiiert.",
                payload=payload,
            )
        )

    async def process_tick(self) -> None:
        """Führt einen deterministischen zweiphasigen Simulationszyklus aus."""
        self._current_tick += 1

        # ==========================================================
        # PHASE 1: Intention, Doppel-Puffer-Commit & Kognition
        # ==========================================================
        self._commit_staging_messages()
        self._update_perceptions_and_timed_goals()
        await self._process_agent_needs_and_cognition()
        self._process_agent_communications_and_protocols()

        # ==========================================================
        # PHASE 2: Arbitrierung & Physische Bewegung
        # ==========================================================
        self._execute_physical_movement()
        self._commit_staging_messages()
        await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # Phase 1: Teilmethoden
    # ------------------------------------------------------------------

    def _commit_staging_messages(self) -> None:
        """Überführt Staging-Nachrichten aller Entitäten deterministisch."""
        for entity in self._entities:
            entity.commit_staging_messages()

    def _update_perceptions_and_timed_goals(self) -> None:
        """Aktualisiert Zeitziele und Sensorik der Agenten inklusive Pfadvalidierung."""
        for agent in self._agents:
            if agent.active_goal:
                self._goal_service.process_timed_goal(agent)

            self._update_agent_perception(agent)
            if agent.has_path and not agent.is_busy:
                if any(not agent.mental_map.is_walkable(p) for p in agent.path):
                    active_goal = agent.active_goal
                    if active_goal and active_goal.target_position and not active_goal.is_evasion_hold:
                        new_path = self._pathfinder.find_path(
                            agent.position, active_goal.target_position, agent.mental_map
                        )
                        if new_path:
                            agent.assign_path(new_path)
                        else:
                            agent.clear_path()

    async def _process_agent_needs_and_cognition(self) -> None:
        """Überwacht Bedürfnisse, triggert Plandekomposition und führt Sub-Goals aus."""
        for agent in self._agents:
            consumed_in_tick = False

            # 1. Plandekomposition bei akutem Hunger & leerem Zielstack
            if (
                not agent.is_busy
                and not agent.goals
                and self._need_service.is_need_urgent(agent, "hunger")
            ):
                await self._trigger_plan_decomposition(agent, "hunger")

            # 2. Ausführung & Evaluation aktiver Sub-Goals
            current_goal = agent.active_goal
            if current_goal and not agent.is_busy:
                if current_goal.name == "SubGoal: move_to":
                    self._handle_subgoal_move_to(agent, current_goal)
                elif current_goal.name.startswith("SubGoal: explore"):
                    self._handle_subgoal_explore(agent, current_goal)
                elif current_goal.name == "SubGoal: consume":
                    consumed_in_tick = self._handle_subgoal_consume(agent, current_goal)

            # 3. Zyklischer Vitalwertzuwachs pro Takt
            if not consumed_in_tick:
                self._need_service.update_needs(agent)

    async def _trigger_plan_decomposition(self, agent: Agent, need_name: str) -> None:
        """Erzeugt einen neuen hierarchischen Plan über den Kognitionsservice."""
        plan = await self._plan_decomposition_service.create_plan_for_need(agent, need_name)
        primary_goal = Goal(name=plan.primary_goal)
        self._goal_service.push_goal(agent, primary_goal)

        for intent in reversed(plan.sub_goals):
            target_pos = (
                Position(intent.target_position[0], intent.target_position[1])
                if intent.target_position
                else None
            )
            sub_goal = Goal(
                name=f"SubGoal: {intent.action_type.value}",
                target_position=target_pos,
                target_entity_id=intent.target_entity_id,
                description=intent.description,
            )
            self._goal_service.push_goal(agent, sub_goal)

    def _handle_subgoal_move_to(self, agent: Agent, current_goal: Goal) -> None:
        """Verwaltet Pfadzuweisung für direkte Navigations-Teilziele."""
        if current_goal.target_position and not agent.has_path:
            if agent.position != current_goal.target_position:
                path = self._pathfinder.find_path(
                    agent.position,
                    current_goal.target_position,
                    agent.mental_map,
                )
                if path:
                    agent.assign_path(path)

    def _handle_subgoal_explore(self, agent: Agent, current_goal: Goal) -> None:
        """Evaluiert Ressourcenfunde und steuert Grenzkachel-Exploration."""
        target_categories = self._precondition_evaluator.RESOURCE_CATEGORIES.get("consumable", set())
        discovered = self._precondition_evaluator.find_discovered_entity(
            agent, categories=target_categories
        )

        if discovered:
            self._interrupt_goal_for_replan(
                agent=agent,
                reason=f"Ressource vom Typ '{discovered.entity_type}' entdeckt",
                discovered_fact=discovered,
            )
        else:
            if not agent.has_path and not agent.is_busy:
                target_frontier = self._frontier_explorer.find_nearest_frontier(
                    agent.position, agent.mental_map
                )
                if target_frontier:
                    path = self._pathfinder.find_path(
                        agent.position, target_frontier, agent.mental_map
                    )
                    if path:
                        agent.assign_path(path)
                        current_goal.target_position = target_frontier

            if current_goal.target_position and agent.position == current_goal.target_position:
                self._goal_service.pop_goal(agent)

    def _handle_subgoal_consume(self, agent: Agent, current_goal: Goal) -> bool:
        """Führt Konsumaktion deterministisch aus."""
        target_entity = None
        if current_goal.target_entity_id:
            target_entity = next(
                (e for e in self._entities if e.id == current_goal.target_entity_id), None
            )
        if not target_entity and current_goal.target_position:
            target_entity = next(
                (e for e in self._entities if e.position == current_goal.target_position), None
            )

        if not target_entity:
            return False

        consumed = self._action_executor.execute_consume(
            agent=agent,
            target_entity=target_entity,
            all_entities=self._entities,
            incident_id=f"consume-t{self._current_tick}-{agent.id}",
        )
        if consumed:
            self._goal_service.pop_goal(agent)
            if not self._need_service.is_need_urgent(agent, "hunger"):
                active_g = agent.active_goal
                if active_g and not any(g.name.startswith("SubGoal:") for g in agent.goals):
                    self._goal_service.pop_goal(agent)
            return True
        return False

    def _process_agent_communications_and_protocols(self) -> None:
        """Bearbeitet Protokolle, Inboxes und Interaktionsqueues via ProtocolService."""
        self._delayed_agent_ids = {
            agent.id for agent in self._agents if agent.inbox and not agent.is_thinking
        }
        self._protocol_service.process_protocols(
            agents=self._agents,
            entities=self._entities,
            background_tasks=self._background_tasks,
        )
        for agent in self._agents:
            self._process_epistemic_target_search(agent)
            self._process_goal_arrival(agent)


    def _process_epistemic_target_search(self, agent: Agent) -> None:
        """Initiiert Zielsuche bei Zielen mit bekannter Entitäts-ID ohne Pfad."""
        active_goal = agent.active_goal
        if not agent.is_busy and active_goal and active_goal.target_entity_id and not agent.has_path:
            search_res = self._target_search_service.search_target(
                agent=agent,
                target_entity_id=active_goal.target_entity_id,
                grid=self._grid,
                entities=self._entities,
                current_tick=self._current_tick,
            )
            if search_res.path:
                agent.assign_path(search_res.path)

    def _process_goal_arrival(self, agent: Agent) -> None:
        """Überprüft Nischenankunft und Freigabe kritischer Abschnitte an Zielkoordinaten."""
        current_goal = agent.active_goal
        if not (current_goal and current_goal.target_position and agent.position == current_goal.target_position):
            return
        if current_goal.is_evasion_hold:
            return

        if current_goal.junction_position is not None:
            self._protocol_service.handle_niche_arrival(agent, current_goal, self._entities)
        else:
            resource_key = f"pos:{current_goal.target_position.x},{current_goal.target_position.y}"
            next_agent_id = self._critical_section_coordinator.release(
                agent_id=agent.id, resource_key=resource_key, mark_completed=True
            )
            self._goal_service.pop_goal(agent)
            if next_agent_id:
                self._notify_next_critical_section_holder(resource_key, next_agent_id)

    # ------------------------------------------------------------------
    # Phase 2: Teilmethoden
    # ------------------------------------------------------------------

    def _execute_physical_movement(self) -> None:
        """Führt Arbitrierung, Zweiphasen-Commit und Konfliktbehandlung aus."""
        self._reservation_table.clear()

        # 1. Schrittwünsche sammeln und reservieren
        intents: list[TileReservationIntent] = []
        for agent in self._agents:
            if agent.id in self._delayed_agent_ids:
                continue
            if agent.has_path and not agent.is_busy:
                target_pos = agent.path[0]
                active_goal = agent.active_goal
                prio = active_goal.priority if active_goal else ExecutionPriority.ROUTINE
                dist = len(agent.path)
                is_backtracking = bool(active_goal and active_goal.backtracking_junction_target is not None)
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
        occupied_stationary = {e.position for e in self._entities if not isinstance(e, Agent)}
        sync_result = self._movement_sync_service.execute_two_phase_commit(
            agents=self._agents,
            intents=intents,
            grid=self._grid,
            occupied_positions=occupied_stationary,
            tick=self._current_tick,
        )

        for agent in self._agents:
            if agent.id in sync_result.committed_agents:
                self._handle_committed_agent_post_move(agent)
            elif not agent.has_path or agent.is_busy:
                self._protocol_service.check_and_signal_clearance(agent, self._entities)
            else:
                self._handle_blocked_agent(agent)

    def _handle_committed_agent_post_move(self, agent: Agent) -> None:
        """Aktualisiert Sensorik und Signalisierung für erfolgreich bewegte Agenten."""
        self._update_agent_perception(agent)
        current_goal = agent.active_goal

        if (
            current_goal is not None
            and current_goal.junction_position is not None
            and not current_goal.is_evasion_hold
            and not current_goal.halt_signaled
            and agent.position == current_goal.junction_position
        ):
            current_goal.halt_signaled = True
            partner_id = current_goal.yield_for_agent_id
            partner = next((e for e in self._entities if e.id == partner_id), None)
            if partner:
                partner.receive_message(
                    IncomingMessage(
                        from_agent_id=agent.id,
                        from_agent_name=agent.name,
                        message="HALT WARTE!",
                        channel=CommunicationChannel.LOCAL_TALK,
                        is_halt_request=True,
                        correlation_key=f"niche-entry-{agent.id}",
                    )
                )

        if (
            current_goal is not None
            and current_goal.target_position is not None
            and agent.position == current_goal.target_position
            and not current_goal.is_evasion_hold
            and current_goal.junction_position is not None
        ):
            self._protocol_service.handle_niche_arrival(agent, current_goal, self._entities)

        self._protocol_service.check_and_signal_clearance(agent, self._entities)

    def _handle_blocked_agent(self, agent: Agent) -> None:
        """Initiiert Konfliktauflösung für blockierte Bewegungsschritte."""
        next_pos = agent.path[0]
        blocker = next((e for e in self._entities if e.position == next_pos), None)
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
                            agent, blocker, next_pos, self._entities
                        )
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
            else:
                if not agent.memory.is_epistemically_exhausted(blocker.id) and self._conflict_coordinator:
                    agent.is_thinking = True
                    task = asyncio.create_task(
                        self._conflict_coordinator.resolve_blockage(
                            agent, blocker, next_pos, self._entities
                        )
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                else:
                    agent.mental_map.mark_obstacle(next_pos, self._current_tick)
                    self._replan_around_obstacle(agent)
        elif not self._grid.is_walkable(next_pos):
            agent.mental_map.mark_obstacle(next_pos, self._current_tick)
            self._replan_around_obstacle(agent)

    def _replan_around_obstacle(self, agent: Agent) -> None:
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

    async def run(self, max_ticks: int = 20) -> None:
        self._is_running = True
        for agent in self._agents:
            self._update_agent_perception(agent)

        known_tiles = self._collect_known_positions()
        self._presenter.render(
            self._grid,
            self._entities,
            self._current_tick,
            self._dialogue_history.get_recent_formatted(limit=8),
            known_positions=known_tiles,
        )

        while self._is_running and self._current_tick < max_ticks:
            await asyncio.sleep(self._tick_interval)
            await self.process_tick()
            known_tiles = self._collect_known_positions()
            self._presenter.render(
                self._grid,
                self._entities,
                self._current_tick,
                self._dialogue_history.get_recent_formatted(limit=8),
                known_positions=known_tiles,
            )
            if all(not a.has_path and not a.is_busy and not a.is_thinking for a in self._agents):
                break

        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)

        self._is_running = False

    def _check_and_signal_clearance(self, agent: Agent) -> None:
        self._protocol_service.check_and_signal_clearance(agent, self._entities)

    def _handle_niche_arrival(self, agent: Agent, current_goal: Goal) -> None:
        self._protocol_service.handle_niche_arrival(agent, current_goal, self._entities)

    def _replan_agent_path_avoiding(self, agent: Agent, forbidden_tiles: set[Position]) -> None:
        self._protocol_service.replan_agent_path_avoiding(agent, forbidden_tiles)

    def _is_in_corridor_zone(self, pos: Position) -> bool:
        return self._protocol_service.is_in_corridor_zone(pos)

