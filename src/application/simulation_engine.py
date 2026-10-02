from __future__ import annotations

import asyncio
from typing import Any, Optional

from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.coordination.agent_protocol_service import AgentProtocolService
from src.application.services.cognition.cognition_orchestrator import CognitionOrchestrator
from src.application.services.coordination.conflict_coordinator import ConflictCoordinator
from src.application.services.coordination.convoy_arbitrator import ConvoyArbitrator
from src.application.services.coordination.convoy_coordinator import ConvoyCoordinator
from src.application.services.coordination.critical_section_coordinator import CriticalSectionCoordinator
from src.application.services.lifecycle.day_night_service import DayNightService
from src.application.services.coordination.dialogue_coordinator import DialogueCoordinator
from src.application.services.coordination.dialogue_history import DialogueHistory
from src.application.services.coordination.dialogue_session_manager import DialogueSessionManager
from src.application.services.movement.evasion_finder import EvasionFinder
from src.application.services.cognition.frontier_explorer import FrontierExplorer
from src.application.services.cognition.goal_service import GoalService
from src.application.services.lifecycle.memory_consolidation_service import MemoryConsolidationService
from src.application.services.movement.movement_orchestrator import MovementOrchestrator
from src.application.services.movement.movement_sync_service import MovementSyncService
from src.application.services.movement.multi_agent_niche_packer import MultiAgentNichePacker
from src.application.services.lifecycle.need_service import NeedService
from src.application.services.cognition.plan_decomposition_service import PlanDecompositionService
from src.application.services.movement.target_search_service import TargetSearchService
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_cognition import AgentCognitiveSnapshot
from src.domain.models.agent.agent_memory import EntityFact
from src.domain.models.planning.goal import Goal
from src.domain.models.world.position import Position
from src.domain.models.coordination.reservation_table import ReservationTable
from src.domain.models.world.world import WorldGrid
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.conflict_coordinator import IConflictCoordinator
from src.domain.ports.dialogue_coordinator import IDialogueCoordinator
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter
from src.domain.ports.vector_memory_store import IVectorMemoryStore
from src.domain.services.perception_service import PerceptionService
from src.domain.services.precondition_evaluator import PreconditionEvaluator


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
        need_service: Optional[NeedService] = None,
        day_night_service: Optional[DayNightService] = None,
        plan_decomposition_service: Optional[PlanDecompositionService] = None,
        frontier_explorer: Optional[FrontierExplorer] = None,
        precondition_evaluator: Optional[PreconditionEvaluator] = None,
        movement_orchestrator: Optional[MovementOrchestrator] = None,
        cognition_orchestrator: Optional[CognitionOrchestrator] = None,
        protocol_service: Optional[AgentProtocolService] = None,
        enable_deterministic_corridor: bool = True,
        enable_day_night: bool = True,  # Neu: Tag-Nacht-Steuerung umschaltbar
        memory_consolidation_service: Optional[MemoryConsolidationService] = None,
        vector_memory_store: Optional[IVectorMemoryStore] = None,
    ) -> None:
        self._enable_day_night = enable_day_night
        """Initialisiert Simulationszustand, Basisdienste sowie Kognitions-, Bewegungs- und Protokoll-Orchestratoren."""
        self._grid: WorldGrid = grid
        self._pathfinder: IPathfinder = pathfinder
        self._presenter: IPresenter = presenter
        self._logger: IEventLogger = logger
        self._tick_interval: float = tick_interval
        self._auditory_radius: int = auditory_radius

        self._entities: list[WorldEntity] = []
        self._agents: list[Agent] = []
        self._current_tick: int = 0
        self._is_running: bool = False
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._delayed_agent_ids: set[str] = set()

        # Basisdienste
        self._dialogue_history = dialogue_history or DialogueHistory()
        self._perception_service = perception_service or PerceptionService(default_radius=3)
        self._goal_service = goal_service or GoalService(
            logger, cognition_provider, pathfinder, tick_provider=lambda: self._current_tick
        )
        self._evasion_finder = EvasionFinder(pathfinder)
        self._session_manager = DialogueSessionManager(max_dialogue_turns=None, logger=self._logger)
        self._critical_section_coordinator = (
            critical_section_coordinator
            or CriticalSectionCoordinator(
                logger=self._logger, tick_provider=lambda: self._current_tick
            )
        )
        self._convoy_coordinator = convoy_coordinator or ConvoyCoordinator(
            pathfinder=self._pathfinder,
            logger=self._logger,
        )
        self._niche_packer = niche_packer or MultiAgentNichePacker()
        self._convoy_arbitrator = convoy_arbitrator or ConvoyArbitrator()

        effective_need_service = need_service or NeedService()
        self._day_night_service = day_night_service or DayNightService(logger=self._logger)

        effective_plan_decomp_service = (
            plan_decomposition_service
            or PlanDecompositionService(cognition_provider=cognition_provider)
        )

        effective_plan_decomp_service = (
                plan_decomposition_service
                or PlanDecompositionService(
            cognition_provider=cognition_provider,
            vector_memory_store=vector_memory_store,
        )
        )

        self._target_search_service = target_search_service or TargetSearchService(
            pathfinder=self._pathfinder,
            perception_service=self._perception_service,
            evasion_finder=self._evasion_finder,
            logger=self._logger,
        )

        self._action_executor = ActionExecutor(
            grid=self._grid,
            logger=self._logger,
            dialogue_history=self._dialogue_history,
            goal_service=self._goal_service,
            pathfinder=self._pathfinder,
            perception_service=self._perception_service,
            evasion_finder=self._evasion_finder,
            critical_section_coordinator=self._critical_section_coordinator,
            need_service=effective_need_service,
            tick_provider=lambda: self._current_tick,
        )

        reservation_table = ReservationTable()
        self._movement_sync_service = movement_sync_service or MovementSyncService(
            logger=self._logger,
            reservation_table=reservation_table,
        )

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
            vector_memory_store=vector_memory_store,
        )

        # Extrahierte Orchestratoren
        self._protocol_service = protocol_service or AgentProtocolService(
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

        self._movement_orchestrator = movement_orchestrator or MovementOrchestrator(
            grid=self._grid,
            pathfinder=self._pathfinder,
            movement_sync_service=self._movement_sync_service,
            reservation_table=reservation_table,
            protocol_service=self._protocol_service,
            conflict_coordinator=self._conflict_coordinator,
            on_agent_moved=self._update_agent_perception,
            tick_provider=lambda: self._current_tick,
        )

        self._cognition_orchestrator = cognition_orchestrator or CognitionOrchestrator(
            pathfinder=self._pathfinder,
            logger=self._logger,
            goal_service=self._goal_service,
            need_service=effective_need_service,
            plan_decomposition_service=effective_plan_decomp_service,
            action_executor=self._action_executor,
            frontier_explorer=frontier_explorer,
            precondition_evaluator=precondition_evaluator,
            tick_provider=lambda: self._current_tick,
        )
        self._memory_consolidation_service = memory_consolidation_service
        self._vector_memory_store = vector_memory_store

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def latest_cognitive_snapshots(self) -> list[AgentCognitiveSnapshot]:
        """Liefert die aktuellen Kognitions-Snapshots aller Agenten."""
        return self._cognition_orchestrator.latest_snapshots

    def _render_current_state(self, known_tiles: set[Position]) -> None:
        dialogues = self._dialogue_history.get_recent_formatted(limit=8)
        snapshots = self.latest_cognitive_snapshots
        try:
            self._presenter.render(
                self._grid,
                self._entities,
                self._current_tick,
                dialogues=dialogues,
                known_positions=known_tiles,
                snapshots=snapshots,
            )
        except TypeError:
            self._presenter.render(
                self._grid,
                self._entities,
                self._current_tick,
                dialogues=dialogues,
                known_positions=known_tiles,
            )

    @property
    def current_tick(self) -> int:
        """Gibt den aktuellen Simulationszyklus zurück."""
        return self._current_tick

    @property
    def dialogue_history(self) -> DialogueHistory:
        """Liefert das Protokoll aller getätigten Dialoge."""
        return self._dialogue_history

    @property
    def critical_section_coordinator(self) -> CriticalSectionCoordinator:
        """Verwaltet exklusive Belegungsrechte für kritische räumliche Abschnitte."""
        return self._critical_section_coordinator

    @property
    def movement_sync_service(self) -> MovementSyncService:
        """Führt den atomaren Zwei-Phasen-Commit für Positionswechsel aus."""
        return self._movement_sync_service

    @property
    def convoy_coordinator(self) -> ConvoyCoordinator:
        """Verwaltet koordinierte Gruppenbewegungen in Engpässen."""
        return self._convoy_coordinator

    @property
    def niche_packer(self) -> MultiAgentNichePacker:
        """Berechnet Ausweichkonfigurationen für mehrere Agenten in Nischen."""
        return self._niche_packer

    @property
    def convoy_arbitrator(self) -> ConvoyArbitrator:
        """Trifft Vorfahrtsentscheidungen bei aufeinandertreffenden Konvois."""
        return self._convoy_arbitrator

    @property
    def movement_orchestrator(self) -> MovementOrchestrator:
        """Orchestriert Phase 2: Arbitrierung, Ausweichmanöver und physische Bewegung."""
        return self._movement_orchestrator

    @property
    def cognition_orchestrator(self) -> CognitionOrchestrator:
        """Orchestriert Phase 1: Vitalwerte, hierarchische Dekomposition und Sub-Goals."""
        return self._cognition_orchestrator

    @property
    def protocol_service(self) -> AgentProtocolService:
        """Verwaltet Nachrichten-Routing, Dialogabstände und Interaktionsanfragen."""
        return self._protocol_service

    # ------------------------------------------------------------------
    # Registrierung & Entitätsmanagement
    # ------------------------------------------------------------------

    def register_entity(self, entity: WorldEntity) -> None:
        """
        Registriert ein statisches oder interaktives Objekt im Gitter.
        - Validiert die Begehbarkeit der Zielposition.
        - Fügt die Entität der globalen Objektliste hinzu.
        """
        if not self._grid.is_walkable(entity.position):
            raise ValueError(f"Position {entity.position} für Objekt {entity.name} ist blockiert.")
        self._entities.append(entity)

    def register_agent(self, agent: Agent) -> None:
        """
        Fügt einen neuen Agenten zur Simulation hinzu.
        - Prüft die Startposition auf Kollisionsfreiheit.
        - Initialisiert die Abmessungen der agentenspezifischen mentalen Karte.
        - Trägt den Agenten in die Entitäts- und Agentenverwaltung ein.
        """
        if not self._grid.is_walkable(agent.position):
            raise ValueError(f"Startposition {agent.position} für Agent {agent.name} blockiert.")
        if agent not in self._entities:
            self._entities.append(agent)
        self._agents.append(agent)
        agent.mental_map.set_bounds(self._grid.width, self._grid.height)

    def set_agent_target(
        self, agent_id: str, target: Position, destination_name: str = "Ziel"
    ) -> None:
        """
        Weist einem Agenten ein explizites Navigationsziel von außen zu.
        - Setzt bestehende Zielstacks zurück.
        - Registriert das neue Ziel im GoalService.
        - Führt eine Sensorik-Aktualisierung durch und berechnet den initialen Pfad.
        """
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
        """
        Aktualisiert das Umweltwissen eines einzelnen Agenten.
        - Erfasst sichtbare Kacheln und trägt statische Hindernisse in die mentale Karte ein.
        - Aktualisiert bekannte Entitätsfakten (Position, Typ) im Gedächtnis.
        - Markiert unpassierbare, epistemisch erschöpfte Entitäten als Hindernis.
        """
        visible_positions = self._perception_service.get_visible_positions(
            agent.position, self._grid
        )
        for pos in visible_positions:
            tile_type = self._grid.get_tile_type(pos)
            if (
                not self._grid.is_walkable(pos)
                or agent.memory.get_type_assumed_walkable(tile_type) is False
            ):
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
        """Sammelt alle Positionen, die von mindestens einem Agenten bereits erkundet wurden (für den Presenter)."""
        discovered: set[Position] = set()
        for agent in self._agents:
            discovered.update(agent.mental_map.tiles.keys())
        return discovered

    # ------------------------------------------------------------------
    # Taktzyklus (Zweiphasen-Commit)
    # ------------------------------------------------------------------

    async def process_tick(self) -> None:
        self._current_tick += 1

        # 0. Tag-Nacht-Synchronisation & nächtliche Konsolidierung
        if self._enable_day_night:
            phase_event = self._day_night_service.process_tick(self._current_tick, self._agents)
            is_night = self._day_night_service.is_night(self._current_tick)

            if phase_event == "night_started" and self._memory_consolidation_service:
                day_number = self._day_night_service.get_day_number(self._current_tick)
                self._memory_consolidation_service.consolidate_all(
                    agents=self._agents,
                    day_number=day_number,
                    tick=self._current_tick,
                )
        else:
            is_night = False

        # ==========================================================
        # PHASE 1: Intention, Doppel-Puffer-Commit & Kognition
        # ==========================================================
        self._commit_staging_messages()
        self._update_perceptions_and_timed_goals()
        await self._cognition_orchestrator.process_agent_needs_and_cognition(
            self._agents, self._entities
        )
        self._process_agent_communications_and_protocols()

        # ==========================================================
        # PHASE 2: Arbitrierung & Physische Bewegung (pausiert bei Nacht)
        # ==========================================================
        if not is_night:
            self._movement_orchestrator.execute_physical_movement(
                agents=self._agents,
                entities=self._entities,
                delayed_agent_ids=self._delayed_agent_ids,
                background_tasks=self._background_tasks,
            )
            self._commit_staging_messages()

        await asyncio.sleep(0)

    def _commit_staging_messages(self) -> None:
        """Überträgt zwischengespeicherte Staging-Nachrichten in die aktiven Inboxes aller Entitäten."""
        for entity in self._entities:
            entity.commit_staging_messages()

    def _update_perceptions_and_timed_goals(self) -> None:
        """
        Bereitet Agenten auf die Kognition vor:
        - Zählt Time-to-Live für zeitbefristete Ziele herunter.
        - Aktualisiert das Sichtfeld und revalidiert geplante Pfade gegen neu entdeckte Hindernisse.
        """
        for agent in self._agents:
            if agent.active_goal:
                self._goal_service.process_timed_goal(agent)

            self._update_agent_perception(agent)
            if agent.has_path and not agent.is_busy:
                if any(not agent.mental_map.is_walkable(p) for p in agent.path):
                    active_goal = agent.active_goal
                    if (
                        active_goal
                        and active_goal.target_position
                        and not active_goal.is_evasion_hold
                    ):
                        new_path = self._pathfinder.find_path(
                            agent.position, active_goal.target_position, agent.mental_map
                        )
                        if new_path:
                            agent.assign_path(new_path)
                        else:
                            agent.clear_path()

    def _process_agent_communications_and_protocols(self) -> None:
        """
        Koordiniert die nachgelagerten Protokolle nach der Kognition:
        - Identifiziert verzögerte Agenten mit Inbox-Nachrichten.
        - Verarbeitet Inboxes, Interaktionsqueues und Distanzen via AgentProtocolService.
        - Führt epistemische Zielsuchen durch und prüft Zielankünfte.
        """
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
        """Löst Pfadsuche aus, wenn ein Agent ein Entitätsziel verfolgt, dessen genaue Position erst ermittelt werden muss."""
        active_goal = agent.active_goal
        if (
            not agent.is_busy
            and active_goal
            and active_goal.target_entity_id
            and not agent.has_path
            and not active_goal.name.startswith("SubGoal:")
        ):
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
        """
        Prüft, ob der Agent sein Ziel erreicht hat:
        - Löst bei Nischenankunft das entsprechende Ausweichhalteprotokoll aus.
        - Gibt bei Ankunft an einem Zielkachel-Ressourcenschloss die kritische Sektion frei und weckt wartende Agenten.
        """
        current_goal = agent.active_goal
        if not (
            current_goal
            and current_goal.target_position
            and agent.position == current_goal.target_position
        ):
            return
        if current_goal.is_evasion_hold:
            return

        if current_goal.junction_position is not None:
            self._protocol_service.handle_niche_arrival(agent, current_goal, self._entities)
        else:
            resource_key = (
                f"pos:{current_goal.target_position.x},{current_goal.target_position.y}"
            )
            next_agent_id = self._critical_section_coordinator.release(
                agent_id=agent.id, resource_key=resource_key, mark_completed=True
            )
            self._goal_service.pop_goal(agent)
            if next_agent_id:
                self._notify_next_critical_section_holder(resource_key, next_agent_id)

    def _notify_next_critical_section_holder(self, resource_key: str, next_agent_id: str) -> None:
        """
        Aktiviert den nächsten wartenden Agenten nach Freigabe einer Ressource:
        - Prüft per Notwendigkeitscheck, ob das Ziel weiterhin erforderlich ist.
        - Weckt den Agenten auf und weist einen neuen Pfad zum Ziel zu.
        """
        next_agent = next((a for a in self._agents if a.id == next_agent_id), None)
        if not next_agent:
            return

        active_goal = next_agent.active_goal
        if not active_goal:
            return

        is_completed = self._critical_section_coordinator.is_completed(resource_key)
        is_still_needed = self._goal_service.validate_goal_necessity(
            agent=next_agent,
            goal=active_goal,
            is_already_completed=is_completed,
            incident_id=f"cs-notify-t{self._current_tick}-{resource_key}",
        )
        if is_still_needed:
            self._goal_service.resume_goal(next_agent)
            resumed_goal = next_agent.active_goal
            if resumed_goal and resumed_goal.target_position:
                new_path = self._pathfinder.find_path(
                    next_agent.position, resumed_goal.target_position, next_agent.mental_map
                )
                if new_path:
                    next_agent.assign_path(new_path)

    # ------------------------------------------------------------------
    # Präsentations- und Lebenszyklussteuerung
    # ------------------------------------------------------------------

    async def run(self, max_ticks: int = 20, stop_when_idle: bool = True) -> None:
        """
        Führt die Simulations-Hauptschleife aus:
        - Rendert Initialzustand mit Kognitions-Snapshots.
        - Führt Takte in festgelegten Intervallen aus, bis max_ticks erreicht sind oder alle Agenten ruhen.
        - Wartet auf den Abschluss aller asynchronen Hintergrund-Tasks bei Simulationsende.
        """
        self._is_running = True
        for agent in self._agents:
            self._update_agent_perception(agent)
            self._cognition_orchestrator._capture_and_log_snapshot(agent, self._entities)

        known_tiles = self._collect_known_positions()
        self._render_current_state(known_tiles)

        while self._is_running and self._current_tick < max_ticks:
            await asyncio.sleep(self._tick_interval)
            await self.process_tick()
            known_tiles = self._collect_known_positions()
            self._render_current_state(known_tiles)

            if stop_when_idle and all(
                not a.has_path and not a.is_busy and not a.is_thinking for a in self._agents
            ):
                break

        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)

        self._is_running = False

    # ------------------------------------------------------------------
    # Fassaden-Delegationen (Rückwärtskompatibilität & Test-Verträge)
    # ------------------------------------------------------------------

    def _check_and_signal_clearance(self, agent: Agent) -> None:
        """Delegiert Chokepoint-Clearance-Signalisierung an den ProtocolService."""
        self._protocol_service.check_and_signal_clearance(agent, self._entities)

    def _handle_niche_arrival(self, agent: Agent, current_goal: Goal) -> None:
        """Delegiert Nischenankunft und Halteabsicherung an den ProtocolService."""
        self._protocol_service.handle_niche_arrival(agent, current_goal, self._entities)

    def _replan_agent_path_avoiding(self, agent: Agent, forbidden_tiles: set[Position]) -> None:
        """Delegiert Pfadneuberechnung unter Ausschluss gesperrter Kacheln an den ProtocolService."""
        self._protocol_service.replan_agent_path_avoiding(agent, forbidden_tiles)

    def _is_in_corridor_zone(self, pos: Position) -> bool:
        """Delegiert Prüfung auf Korridorgeometrie an den ProtocolService."""
        return self._protocol_service.is_in_corridor_zone(pos)

    def _interrupt_goal_for_replan(
        self,
        agent: Agent,
        reason: str,
        discovered_fact: Optional[EntityFact] = None,
    ) -> None:
        """Delegiert opportunistischen Zielabbruch und Neuplanung an den CognitionOrchestrator."""
        self._cognition_orchestrator.interrupt_goal_for_replan(agent, reason, discovered_fact)

    def _execute_physical_movement(self) -> None:
        """Delegiert Phase 2 (Arbitrierung, Zwei-Phasen-Commit, Kollisionsbehandlung) an den MovementOrchestrator."""
        self._movement_orchestrator.execute_physical_movement(
            agents=self._agents,
            entities=self._entities,
            delayed_agent_ids=self._delayed_agent_ids,
            background_tasks=self._background_tasks,
        )

    def _handle_committed_agent_post_move(self, agent: Agent) -> None:
        """Delegiert Nachbereitung erfolgreicher Schritte (Sensorik, Nischensignale) an den MovementOrchestrator."""
        self._movement_orchestrator.handle_committed_agent_post_move(agent, self._entities)

    def _handle_blocked_agent(self, agent: Agent) -> None:
        """Delegiert Kollisionsbehandlung und Konflikt-Tasks an den MovementOrchestrator."""
        self._movement_orchestrator.handle_blocked_agent(
            agent, self._entities, self._background_tasks
        )

    def _replan_around_obstacle(self, agent: Agent) -> None:
        """Delegiert statische Hindernis-Neuberechnung an den MovementOrchestrator."""
        self._movement_orchestrator.replan_around_obstacle(agent)

    async def _process_agent_needs_and_cognition(self) -> None:
        """Delegiert Phase 1 (Vitalwerte, Plandekomposition, Sub-Goal-Ausführung) an den CognitionOrchestrator."""
        await self._cognition_orchestrator.process_agent_needs_and_cognition(
            self._agents, self._entities
        )