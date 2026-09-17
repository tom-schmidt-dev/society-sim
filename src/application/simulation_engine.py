from __future__ import annotations

import asyncio
from typing import Any, Optional
from src.application.services.action_executor import ActionExecutor
from src.application.services.conflict_coordinator import ConflictCoordinator
from src.application.services.critical_section_coordinator import CriticalSectionCoordinator
from src.application.services.dialogue_coordinator import DialogueCoordinator
from src.application.services.dialogue_history import DialogueHistory
from src.application.services.dialogue_session_manager import DialogueSessionManager
from src.application.services.evasion_finder import EvasionFinder
from src.application.services.goal_service import GoalService
from src.application.services.target_search_service import TargetSearchService
from src.domain.models.agent import Agent
from src.domain.models.events import SimulationEvent
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
            tick_provider=lambda: self._current_tick,
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

    @property
    def current_tick(self) -> int:
        return self._current_tick

    @property
    def dialogue_history(self) -> DialogueHistory:
        return self._dialogue_history

    @property
    def critical_section_coordinator(self) -> CriticalSectionCoordinator:
        return self._critical_section_coordinator

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

    def _handle_niche_arrival(self, agent: Agent, current_goal: Goal) -> None:
        junction = current_goal.junction_position
        partner_id = current_goal.yield_for_agent_id

        self._goal_service.pop_goal(agent)
        agent.is_evasion_locked = True
        agent.clear_path()

        self._goal_service.push_goal(
            agent,
            Goal(
                name="Nischen-Halt",
                holds_position=True,
                is_evasion_hold=True,
                junction_position=junction,
                yield_for_agent_id=partner_id,
                priority=ExecutionPriority.URGENT,
                description="Wartet in der Nische, bis der Partner den Chokepoint passiert hat.",
            ),
        )

        partner = next((e for e in self._entities if e.id == partner_id), None)
        if partner:
            partner.receive_message(
                IncomingMessage(
                    from_agent_id=agent.id,
                    from_agent_name=agent.name,
                    message="Ok, geh weiter.",
                    channel=CommunicationChannel.LOCAL_TALK,
                    is_resume_signal=True,
                    correlation_key=f"niche-entry-{agent.id}",
                )
            )

    def _check_and_signal_clearance(self, agent: Agent) -> None:
        for other in self._entities:
            if not isinstance(other, Agent) or other.id == agent.id:
                continue
            other_goal = other.active_goal
            if (
                (other.is_evasion_locked or (other_goal and other_goal.is_evasion_hold))
                and (other_goal and other_goal.yield_for_agent_id == agent.id)
            ):
                junction = other_goal.junction_position
                if (
                    junction is not None
                    and agent.position != junction
                    and junction not in agent.path
                    and agent.position.manhattan_distance(junction) >= 2
                ):
                    other_goal.yield_for_agent_id = None
                    other.receive_message(
                        IncomingMessage(
                            from_agent_id=agent.id,
                            from_agent_name=agent.name,
                            message="Danke fürs Platz machen!",
                            channel=CommunicationChannel.LOCAL_TALK,
                            is_courtesy=True,
                            is_resume_signal=True,
                        )
                    )
                    self._dialogue_history.record_dialogue(
                        tick=self._current_tick,
                        sender_id=agent.id,
                        sender_name=agent.name,
                        recipient_id=other.id,
                        recipient_name=other.name,
                        message="Danke fürs Platz machen!",
                    )

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

    def _replan_agent_path_avoiding(self, agent: Agent, forbidden_tiles: set[Position]) -> None:
        """Berechnet den Pfad zum aktiven Ziel neu unter Meidung der Sperrkacheln."""
        active_goal = agent.active_goal
        if not active_goal or not active_goal.target_position:
            return

        target = active_goal.target_position
        temp_obstacles: set[Position] = set()

        for pos in forbidden_tiles:
            if pos != target and pos != agent.position:
                if agent.mental_map.is_walkable(pos):
                    temp_obstacles.add(pos)
                    agent.mental_map.update_tile(pos, is_walkable=False, tick=self._current_tick)

        new_path = self._pathfinder.find_path(agent.position, target, agent.mental_map)

        for pos in temp_obstacles:
            agent.mental_map.update_tile(pos, is_walkable=True, tick=self._current_tick)

        if new_path:
            agent.assign_path(new_path)

    async def process_tick(self) -> None:
        """Führt einen zweiphasigen Simulationszyklus aus."""
        self._current_tick += 1

        # ==========================================================
        # PHASE 1: Intention, Doppel-Puffer-Commit & Kognition
        # ==========================================================
        for entity in self._entities:
            entity.commit_staging_messages()

        for agent in self._agents:
            if agent.active_goal:
                self._goal_service.process_timed_goal(agent)

        for entity in list(self._entities):
            if not isinstance(entity, Agent):
                continue

            agent = entity

            # 1. Reichweiten-Prüfung für bestehende Dialoge
            if agent.interaction_partner_id:
                partner_entity = next((e for e in self._entities if e.id == agent.interaction_partner_id), None)
                if partner_entity:
                    dist = agent.position.manhattan_distance(partner_entity.position)
                    if dist > self._auditory_radius:
                        agent.is_waiting_for_reply = False
                        agent.is_listening_to_peer = False
                        agent.has_bid_farewell = False
                        agent.peer_bid_farewell = False
                        agent.interaction_partner_id = None
                        self._logger.log(
                            SimulationEvent(
                                tick=self._current_tick,
                                agent_id=agent.id,
                                event_type="message_undeliverable",
                                summary=f"Agent {agent.name}: Interaktionspartner {partner_entity.name} außer Hörweite. Wartezustand gelöst.",
                                payload={"partner_id": partner_entity.id, "distance": dist},
                            )
                        )

            # 2. Gegenseitiger Verabschiedungsabschluss
            if agent.has_bid_farewell and agent.interaction_partner_id:
                partner_agent = next(
                    (a for a in self._agents if a.id == agent.interaction_partner_id),
                    None,
                )
                if isinstance(partner_agent, Agent) and partner_agent.has_bid_farewell:
                    self._logger.log(
                        SimulationEvent(
                            tick=self._current_tick,
                            agent_id=agent.id,
                            event_type="farewell_handshake_completed",
                            summary=f"Verabschiedung zwischen {agent.name} und {partner_agent.name} abgeschlossen.",
                            payload={"partner_id": partner_agent.id},
                        )
                    )
                    agent.has_bid_farewell = False
                    agent.peer_bid_farewell = False
                    agent.is_listening_to_peer = False
                    agent.is_waiting_for_reply = False
                    agent.interaction_partner_id = None

                    partner_agent.has_bid_farewell = False
                    partner_agent.peer_bid_farewell = False
                    partner_agent.is_listening_to_peer = False
                    partner_agent.is_waiting_for_reply = False
                    partner_agent.interaction_partner_id = None

            # 3. Eingehende Nachrichten auswerten
            if agent.inbox and not agent.is_thinking:
                courtesy_msgs = [m for m in agent.inbox if m.is_courtesy]
                for msg in courtesy_msgs:
                    agent.inbox.remove(msg)
                    agent.assimilate_message(msg, tick=self._current_tick)
                    sender = next((e for e in self._entities if e.id == msg.from_agent_id), None)
                    sender_id = sender.id if sender else msg.from_agent_id
                    sender_name = sender.name if sender else "Partner"

                    self._dialogue_history.record_dialogue(
                        tick=self._current_tick,
                        sender_id=agent.id,
                        sender_name=agent.name,
                        recipient_id=sender_id,
                        recipient_name=sender_name,
                        message="Gern geschehen!",
                    )

                    agent.is_evasion_locked = False
                    goal = agent.active_goal
                    if goal is not None and (goal.is_evasion_hold or "Nischen-Halt" in goal.name or "In Nische ausweichen" in goal.name):
                        junction_pos = goal.junction_position
                        self._goal_service.pop_goal(agent)
                        while True:
                            curr_g = agent.active_goal
                            if curr_g and ("Warten" in curr_g.name or "Konversation" in curr_g.name):
                                self._goal_service.pop_goal(agent)
                            else:
                                break

                        self._goal_service.resume_goal(agent)

                        active = agent.active_goal
                        if active is not None and active.target_position:
                            agent.mental_map.update_tile(agent.position, is_walkable=True, tick=self._current_tick)
                            if junction_pos:
                                agent.mental_map.update_tile(junction_pos, is_walkable=True, tick=self._current_tick)
                            new_path = self._pathfinder.find_path(
                                agent.position,
                                active.target_position,
                                agent.mental_map,
                            )
                            if new_path:
                                agent.assign_path(new_path)

                halt_msgs = [m for m in agent.inbox if m.is_halt_request]
                for msg in halt_msgs:
                    agent.inbox.remove(msg)
                    agent.assimilate_message(msg, tick=self._current_tick)
                    agent.is_holding_for_junction = True

                resume_msgs = [m for m in agent.inbox if m.is_resume_signal]
                for msg in resume_msgs:
                    agent.inbox.remove(msg)
                    agent.assimilate_message(msg, tick=self._current_tick)
                    agent.is_holding_for_junction = False

                    if msg.correlation_key:
                        self._goal_service.pop_goal_by_key(agent, msg.correlation_key)
                    else:
                        for i in range(len(agent.goals) - 1, -1, -1):
                            if agent.goals[i].yield_for_agent_id == msg.from_agent_id:
                                agent.goals.pop(i)
                                if agent.goals:
                                    agent.goals[-1].status = "active"
                                break

                    fact = agent.memory.known_entities.get(msg.from_agent_id)
                    if msg.correlation_key and msg.correlation_key.startswith("niche-entry"):
                        if fact:
                            fact.partner_planned_path = None
                    elif fact and fact.partner_planned_path and agent.has_path:
                        partner_tiles = set(fact.partner_planned_path)
                        if set(agent.path) & partner_tiles:
                            self._replan_agent_path_avoiding(agent, partner_tiles)

                path_update_msgs = [m for m in agent.inbox if m.is_path_update]
                if path_update_msgs:
                    for msg in path_update_msgs:
                        agent.inbox.remove(msg)
                        agent.assimilate_message(msg, tick=self._current_tick)
                        corr_key = msg.correlation_key or f"path-upd-{self._current_tick}"
                        partner_path = msg.planned_path or []
                        partner_tiles = set(partner_path)

                        current_target_goal = agent.active_goal
                        if current_target_goal and current_target_goal.target_position:
                            if current_target_goal.target_position in partner_tiles:
                                res = self._evasion_finder.find_nearest_evasion_tile(
                                    start=agent.position,
                                    blocked_pos=agent.position,
                                    grid=agent.mental_map,
                                    occupied_positions={e.position for e in self._entities if e.id != agent.id},
                                    partner_trajectory=partner_path,
                                )
                                if res:
                                    current_target_goal.target_position = res.target_tile
                                    current_target_goal.junction_position = res.junction_tile
                                    agent.assign_path(res.path)
                            else:
                                self._replan_agent_path_avoiding(agent, partner_tiles)

                        sender = next((e for e in self._entities if e.id == msg.from_agent_id), None)
                        if sender:
                            sender.receive_message(
                                IncomingMessage(
                                    from_agent_id=agent.id,
                                    from_agent_name=agent.name,
                                    message="Ok, weiter.",
                                    channel=CommunicationChannel.LOCAL_TALK,
                                    is_resume_signal=True,
                                    correlation_key=corr_key,
                                )
                            )
                    continue

                evasion_notices = [m for m in agent.inbox if m.is_evasion_notice]
                for msg in evasion_notices:
                    agent.inbox.remove(msg)
                    agent.assimilate_message(msg, tick=self._current_tick)
                    agent.is_waiting_for_reply = False
                    agent.has_bid_farewell = False
                    agent.peer_bid_farewell = False
                    if agent.interaction_partner_id == msg.from_agent_id:
                        agent.interaction_partner_id = None

                    farewells_from_evader = [
                        m for m in agent.inbox
                        if m.from_agent_id == msg.from_agent_id and m.is_farewell
                    ]
                    for fm in farewells_from_evader:
                        agent.inbox.remove(fm)
                        agent.assimilate_message(fm, tick=self._current_tick)

                    current_goal = agent.active_goal
                    if current_goal and (
                        "In Nische ausweichen" in current_goal.name or "Warten" in current_goal.name
                    ):
                        self._goal_service.pop_goal(agent, target_goal=current_goal)

                if not agent.interaction_partner_id:
                    farewell_echoes = [m for m in agent.inbox if m.is_farewell]
                    for msg in farewell_echoes:
                        agent.inbox.remove(msg)
                        agent.assimilate_message(msg, tick=self._current_tick)
                        sender = next(
                            (a for a in self._agents if a.id == msg.from_agent_id),
                            None,
                        )
                        if isinstance(sender, Agent):
                            sender.has_bid_farewell = False
                            sender.peer_bid_farewell = False
                            sender.is_waiting_for_reply = False
                            sender.interaction_partner_id = None

                if agent.inbox:
                    agent.is_thinking = True
                    task = asyncio.create_task(
                        self._dialogue_coordinator.handle_incoming_dialogue(
                            agent, self._entities
                        )
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                    continue

            # 4. Abarbeitung der FIFO-Warteschlange mit TTL-Verfall & Anti-Starvation
            while not agent.is_busy and agent.interaction_queue:
                req = agent.interaction_queue.pop(0)
                requester_entity = next((e for e in self._entities if e.id == req.requester_id), None)

                # Type-Guard für statische Typprüfung
                if not isinstance(requester_entity, Agent):
                    continue

                requester: Agent = requester_entity

                if req.is_expired(self._current_tick):
                    requester.is_waiting_for_reply = False
                    requester.interaction_partner_id = None
                    self._logger.log(
                        SimulationEvent(
                            tick=self._current_tick,
                            agent_id=agent.id,
                            event_type="interaction_request_expired",
                            summary=f"Anfrage von {req.requester_id} an {agent.name} nach {req.ttl_ticks} Ticks verworfen.",
                            payload={"requester_id": req.requester_id, "tick_created": req.tick},
                        )
                    )
                    continue

                requester.is_waiting_for_reply = False
                requester.interaction_partner_id = None

                dist = agent.position.manhattan_distance(requester.position)
                is_within_range = dist <= self._auditory_radius
                is_still_heading_to_pos = requester.has_path and requester.path[0] == req.blocked_pos
                is_agent_still_at_pos = agent.position == req.blocked_pos

                if is_within_range and is_still_heading_to_pos and is_agent_still_at_pos:
                    self._logger.log(
                        SimulationEvent(
                            tick=self._current_tick,
                            agent_id=agent.id,
                            event_type="interaction_dequeued",
                            summary=f"Agent {agent.name} bearbeitet Anfrage von {requester.name}.",
                            payload={"requester_id": requester.id, "distance": dist, "blocked_pos": [req.blocked_pos.x, req.blocked_pos.y]},
                        )
                    )
                    requester.is_thinking = True
                    task = asyncio.create_task(
                        self._conflict_coordinator.resolve_blockage(
                            requester, agent, req.blocked_pos, self._entities
                        )
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                    break
                else:
                    self._logger.log(
                        SimulationEvent(
                            tick=self._current_tick,
                            agent_id=agent.id,
                            event_type="interaction_request_dropped",
                            summary=f"Anfrage von {requester.name} an {agent.name} verworfen (Bedingung nicht mehr erfüllt).",
                            payload={
                                "requester_id": requester.id,
                                "is_within_range": is_within_range,
                                "is_still_heading_to_pos": is_still_heading_to_pos,
                                "is_agent_still_at_pos": is_agent_still_at_pos,
                            },
                        )
                    )

            # 5. Epistemische Zielsuche
            active_goal = agent.active_goal
            if (
                not agent.is_busy
                and active_goal
                and active_goal.target_entity_id
                and not agent.has_path
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

            # 6. Nischenankunft prüfen
            current_goal = agent.active_goal
            if (
                current_goal is not None
                and current_goal.target_position is not None
                and agent.position == current_goal.target_position
                and not current_goal.is_evasion_hold
                and current_goal.junction_position is not None
            ):
                self._handle_niche_arrival(agent, current_goal)

            # 7. Zielabschluss am statischen Zielort
            if (
                current_goal is not None
                and current_goal.target_position is not None
                and agent.position == current_goal.target_position
                and not current_goal.is_evasion_hold
                and current_goal.junction_position is None
            ):
                resource_key = f"pos:{current_goal.target_position.x},{current_goal.target_position.y}"
                next_agent_id = self._critical_section_coordinator.release(
                    agent_id=agent.id, resource_key=resource_key, mark_completed=True
                )
                self._goal_service.pop_goal(agent)
                if next_agent_id:
                    self._notify_next_critical_section_holder(resource_key, next_agent_id)

        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)

        # ==========================================================
        # PHASE 2: Arbitrierung (Reservation Table) & Physische Schritte
        # ==========================================================
        self._reservation_table.clear()

        # Schritt 1: Schrittwünsche sammeln und reservieren
        for agent in self._agents:
            if agent.has_path and not agent.is_busy:
                target_pos = agent.path[0]
                active_goal = agent.active_goal
                prio = active_goal.priority if active_goal else ExecutionPriority.ROUTINE
                dist = len(agent.path)
                intent = TileReservationIntent(
                    agent_id=agent.id,
                    current_position=agent.position,
                    desired_position=target_pos,
                    priority=prio,
                    distance_to_goal=dist,
                )
                self._reservation_table.request_reservation(intent)

        # Schritt 2: Physische Ausführung & Kollisionsbehandlung
        for agent in self._agents:
            if not agent.has_path or agent.is_busy:
                self._check_and_signal_clearance(agent)
                continue

            next_pos = agent.path[0]
            is_static_walkable = self._grid.is_walkable(next_pos)
            has_reservation = self._reservation_table.is_granted(agent.id, next_pos)
            is_occupied = any(e.position == next_pos for e in self._entities if e.id != agent.id)

            if is_static_walkable and has_reservation and not is_occupied:
                agent.step()
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
                    self._handle_niche_arrival(agent, current_goal)

                self._check_and_signal_clearance(agent)
            else:
                blocker = next((e for e in self._entities if e.position == next_pos), None)
                if blocker:
                    if isinstance(blocker, Agent):
                        blocker_goal = blocker.active_goal
                        if (
                            blocker.is_evasion_locked
                            or (blocker_goal and blocker_goal.yield_for_agent_id == agent.id)
                        ):
                            continue

                    agent.is_thinking = True
                    task = asyncio.create_task(
                        self._conflict_coordinator.resolve_blockage(
                            agent, blocker, next_pos, self._entities
                        )
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)

        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)

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

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._is_running = False