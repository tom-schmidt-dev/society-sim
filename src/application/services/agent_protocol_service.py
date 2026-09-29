from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from src.application.services.dialogue_history import DialogueHistory
from src.application.services.evasion_finder import EvasionFinder
from src.application.services.goal_service import GoalService
from src.domain.models.agent import Agent
from src.domain.models.communication_templates import DialogueTemplates
from src.domain.models.events import SimulationEvent
from src.domain.models.evasion_phase import EvasionPhase
from src.domain.models.goal import ExecutionPriority, Goal
from src.domain.models.message import CommunicationChannel, IncomingMessage
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
from src.domain.ports.conflict_coordinator import IConflictCoordinator
from src.domain.ports.dialogue_coordinator import IDialogueCoordinator
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder


class AgentProtocolService:
    """Kapselt Nachrichten-Routing, Dialog-Distanzen, Interaktionsqueues und Ausweichprotokolle."""

    def __init__(
        self,
        grid: WorldGrid,
        pathfinder: IPathfinder,
        logger: IEventLogger,
        goal_service: GoalService,
        dialogue_history: DialogueHistory,
        evasion_finder: EvasionFinder,
        dialogue_coordinator: Optional[IDialogueCoordinator] = None,
        conflict_coordinator: Optional[IConflictCoordinator] = None,
        auditory_radius: int = 3,
        tick_provider: Optional[Callable[[], int]] = None,
    ) -> None:
        self._grid = grid
        self._pathfinder = pathfinder
        self._logger = logger
        self._goal_service = goal_service
        self._dialogue_history = dialogue_history
        self._evasion_finder = evasion_finder
        self._dialogue_coordinator = dialogue_coordinator
        self._conflict_coordinator = conflict_coordinator
        self._auditory_radius = auditory_radius
        self._tick_provider = tick_provider or (lambda: 0)

    def process_protocols(
        self,
        agents: list[Agent],
        entities: list[WorldEntity],
        background_tasks: set[asyncio.Task[Any]],
    ) -> None:
        """Führt alle protokollspezifischen Prüfungen für alle Agenten in einem Takt aus."""
        for agent in agents:
            self.verify_dialogue_distance(agent, entities)
            self.handle_farewell_handshake(agent, agents)
            self.process_agent_inbox(agent, entities, agents, background_tasks)
            self.process_interaction_queue(agent, entities, background_tasks)

    def verify_dialogue_distance(self, agent: Agent, entities: list[WorldEntity]) -> None:
        if not agent.interaction_partner_id:
            return
        partner = next((e for e in entities if e.id == agent.interaction_partner_id), None)
        if partner:
            dist = agent.position.manhattan_distance(partner.position)
            if dist > self._auditory_radius:
                agent.is_waiting_for_reply = False
                agent.is_listening_to_peer = False
                agent.has_bid_farewell = False
                agent.peer_bid_farewell = False
                agent.interaction_partner_id = None
                self._logger.log(
                    SimulationEvent(
                        tick=self._tick_provider(),
                        agent_id=agent.id,
                        event_type="message_undeliverable",
                        summary=f"Agent {agent.name}: Interaktionspartner {partner.name} außer Hörweite.",
                        payload={"partner_id": partner.id, "distance": dist},
                    )
                )

    def handle_farewell_handshake(self, agent: Agent, agents: list[Agent]) -> None:
        if not (agent.has_bid_farewell and agent.interaction_partner_id):
            return
        partner = next((a for a in agents if a.id == agent.interaction_partner_id), None)
        if isinstance(partner, Agent) and partner.has_bid_farewell:
            self._logger.log(
                SimulationEvent(
                    tick=self._tick_provider(),
                    agent_id=agent.id,
                    event_type="farewell_handshake_completed",
                    summary=f"Verabschiedung zwischen {agent.name} und {partner.name} abgeschlossen.",
                    payload={"partner_id": partner.id},
                )
            )
            for a in (agent, partner):
                a.has_bid_farewell = False
                a.peer_bid_farewell = False
                a.is_listening_to_peer = False
                a.is_waiting_for_reply = False
                a.interaction_partner_id = None

    def process_agent_inbox(
        self,
        agent: Agent,
        entities: list[WorldEntity],
        agents: list[Agent],
        background_tasks: set[asyncio.Task[Any]],
    ) -> None:
        if not agent.inbox or agent.is_thinking:
            return

        current_tick = self._tick_provider()

        # 1. Höflichkeitsnachrichten
        for msg in [m for m in agent.inbox if m.is_courtesy]:
            agent.inbox.remove(msg)
            agent.assimilate_message(msg, tick=current_tick)
            self._handle_courtesy_message(agent, msg, entities)

        # 2. Halt-Anfragen
        for msg in [m for m in agent.inbox if m.is_halt_request]:
            agent.inbox.remove(msg)
            agent.assimilate_message(msg, tick=current_tick)
            agent.is_holding_for_junction = True

        # 3. Resume-Signale
        for msg in [m for m in agent.inbox if m.is_resume_signal]:
            agent.inbox.remove(msg)
            agent.assimilate_message(msg, tick=current_tick)
            self._handle_resume_signal(agent, msg)

        # 4. Pfad-Updates
        for msg in [m for m in agent.inbox if m.is_path_update]:
            agent.inbox.remove(msg)
            agent.assimilate_message(msg, tick=current_tick)
            self._handle_path_update_message(agent, msg, entities)

        # 5. Ausweich-Hinweise
        for msg in [m for m in agent.inbox if m.is_evasion_notice]:
            agent.inbox.remove(msg)
            agent.assimilate_message(msg, tick=current_tick)
            self._handle_evasion_notice(agent, msg)

        # 6. Verabschiedungs-Echos
        if not agent.interaction_partner_id:
            for msg in [m for m in agent.inbox if m.is_farewell]:
                agent.inbox.remove(msg)
                agent.assimilate_message(msg, tick=current_tick)
                sender = next((a for a in agents if a.id == msg.from_agent_id), None)
                if isinstance(sender, Agent):
                    sender.has_bid_farewell = False
                    sender.peer_bid_farewell = False
                    sender.is_waiting_for_reply = False
                    sender.interaction_partner_id = None

        # 7. Asynchroner Dialogstart
        if agent.inbox and self._dialogue_coordinator:
            agent.is_thinking = True
            task = asyncio.create_task(
                self._dialogue_coordinator.handle_incoming_dialogue(agent, entities)
            )
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

    def _handle_courtesy_message(
        self, agent: Agent, msg: Any, entities: list[WorldEntity]
    ) -> None:
        current_tick = self._tick_provider()
        sender = next((e for e in entities if e.id == msg.from_agent_id), None)
        sender_id = sender.id if sender else msg.from_agent_id
        sender_name = sender.name if sender else "Partner"

        self._dialogue_history.record_dialogue(
            tick=current_tick,
            sender_id=agent.id,
            sender_name=agent.name,
            recipient_id=sender_id,
            recipient_name=sender_name,
            message="Gern geschehen!",
        )
        agent.is_evasion_locked = False
        agent.evasion_phase = EvasionPhase.EGRESS
        self._logger.log(
            SimulationEvent(
                tick=current_tick,
                agent_id=agent.id,
                event_type="evasion_phase_changed",
                summary=f"Agent {agent.name}: Status gewechselt zu {EvasionPhase.EGRESS.value}.",
                payload={"phase": EvasionPhase.EGRESS.value},
            )
        )
        goal = agent.active_goal
        if goal is not None and (
            goal.is_evasion_hold or "Nischen-Halt" in goal.name or "In Nische ausweichen" in goal.name
        ):
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
                agent.mental_map.update_tile(agent.position, is_walkable=True, tick=current_tick)
                if junction_pos:
                    agent.mental_map.update_tile(junction_pos, is_walkable=True, tick=current_tick)
                new_path = self._pathfinder.find_path(
                    agent.position, active.target_position, agent.mental_map
                )
                if new_path:
                    agent.assign_path(new_path)

    def _handle_resume_signal(self, agent: Agent, msg: Any) -> None:
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
                self.replan_agent_path_avoiding(agent, partner_tiles)

    def _handle_path_update_message(
        self, agent: Agent, msg: Any, entities: list[WorldEntity]
    ) -> None:
        current_tick = self._tick_provider()
        corr_key = msg.correlation_key or f"path-upd-{current_tick}"
        partner_path = msg.planned_path or []
        partner_tiles = set(partner_path)

        current_target_goal = agent.active_goal
        if current_target_goal and current_target_goal.target_position:
            if current_target_goal.target_position in partner_tiles:
                res = self._evasion_finder.find_nearest_evasion_tile(
                    start=agent.position,
                    blocked_pos=agent.position,
                    grid=agent.mental_map,
                    occupied_positions={e.position for e in entities if e.id != agent.id},
                    partner_trajectory=partner_path,
                )
                if res:
                    current_target_goal.target_position = res.target_tile
                    current_target_goal.junction_position = res.junction_tile
                    agent.assign_path(res.path)
            else:
                self.replan_agent_path_avoiding(agent, partner_tiles)

        sender = next((e for e in entities if e.id == msg.from_agent_id), None)
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

    def _handle_evasion_notice(self, agent: Agent, msg: Any) -> None:
        current_tick = self._tick_provider()
        agent.is_waiting_for_reply = False
        agent.has_bid_farewell = False
        agent.peer_bid_farewell = False
        if agent.interaction_partner_id == msg.from_agent_id:
            agent.interaction_partner_id = None

        for fm in [m for m in agent.inbox if m.from_agent_id == msg.from_agent_id and m.is_farewell]:
            agent.inbox.remove(fm)
            agent.assimilate_message(fm, tick=current_tick)

        current_goal = agent.active_goal
        if current_goal and (
            "In Nische ausweichen" in current_goal.name or "Warten" in current_goal.name
        ):
            self._goal_service.pop_goal(agent, target_goal=current_goal)

    def process_interaction_queue(
        self,
        agent: Agent,
        entities: list[WorldEntity],
        background_tasks: set[asyncio.Task[Any]],
    ) -> None:
        current_tick = self._tick_provider()
        while not agent.is_busy and agent.interaction_queue:
            req = agent.interaction_queue.pop(0)
            requester_entity = next((e for e in entities if e.id == req.requester_id), None)
            if not isinstance(requester_entity, Agent):
                continue

            requester: Agent = requester_entity
            if req.is_expired(current_tick):
                requester.is_waiting_for_reply = False
                requester.interaction_partner_id = None
                self._logger.log(
                    SimulationEvent(
                        tick=current_tick,
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
                        tick=current_tick,
                        agent_id=agent.id,
                        event_type="interaction_dequeued",
                        summary=f"Agent {agent.name} bearbeitet Anfrage von {requester.name}.",
                        payload={
                            "requester_id": requester.id,
                            "distance": dist,
                            "blocked_pos": [req.blocked_pos.x, req.blocked_pos.y],
                        },
                    )
                )
                if self._conflict_coordinator:
                    requester.is_thinking = True
                    task = asyncio.create_task(
                        self._conflict_coordinator.resolve_blockage(
                            requester, agent, req.blocked_pos, entities
                        )
                    )
                    background_tasks.add(task)
                    task.add_done_callback(background_tasks.discard)
                break
            else:
                self._logger.log(
                    SimulationEvent(
                        tick=current_tick,
                        agent_id=agent.id,
                        event_type="interaction_request_dropped",
                        summary=f"Anfrage von {requester.name} an {agent.name} verworfen.",
                        payload={
                            "requester_id": requester.id,
                            "is_within_range": is_within_range,
                            "is_still_heading_to_pos": is_still_heading_to_pos,
                            "is_agent_still_at_pos": is_agent_still_at_pos,
                        },
                    )
                )

    def handle_niche_arrival(
            self, agent: Agent, current_goal: Goal, entities: list[WorldEntity]
    ) -> None:
        current_tick = self._tick_provider()
        junction = current_goal.junction_position
        partner_id = current_goal.yield_for_agent_id

        self._goal_service.pop_goal(agent)
        agent.clear_path()

        partner = next((e for e in entities if e.id == partner_id), None)
        if partner is None or not isinstance(partner, Agent):
            agent.is_evasion_locked = False
            agent.evasion_phase = EvasionPhase.EGRESS
            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=agent.id,
                    event_type="evasion_phase_changed",
                    summary=f"Agent {agent.name}: Status gewechselt zu {EvasionPhase.EGRESS.value} (Partner ist nicht-Agent/stationär).",
                    payload={"phase": EvasionPhase.EGRESS.value},
                )
            )
            self._goal_service.resume_goal(agent)
            active = agent.active_goal
            if active and active.target_position:
                new_path = self._pathfinder.find_path(agent.position, active.target_position, agent.mental_map)
                if new_path:
                    agent.assign_path(new_path)
            return

        agent.is_evasion_locked = True
        agent.evasion_phase = EvasionPhase.YIELDING_WAIT
        self._logger.log(
            SimulationEvent(
                tick=current_tick,
                agent_id=agent.id,
                event_type="evasion_phase_changed",
                summary=f"Agent {agent.name}: Status gewechselt zu {EvasionPhase.YIELDING_WAIT.value}.",
                payload={"phase": EvasionPhase.YIELDING_WAIT.value,
                         "junction": [junction.x, junction.y] if junction else None},
            )
        )

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

    def check_and_signal_clearance(self, agent: Agent, entities: list[WorldEntity]) -> None:
        current_tick = self._tick_provider()
        for other in entities:
            if not isinstance(other, Agent) or other.id == agent.id:
                continue
            other_goal = other.active_goal
            if (
                    (other.is_evasion_locked or (other_goal and other_goal.is_evasion_hold))
                    and (other_goal and other_goal.yield_for_agent_id == agent.id)
            ):
                junction = other_goal.junction_position
                if junction is None:
                    continue

                margin = max(agent.footprint[0], agent.footprint[1]) + 1
                dist = agent.position.manhattan_distance(junction)
                has_margin = dist >= margin

                active_goal = agent.active_goal
                is_at_destination = (
                        active_goal is not None
                        and active_goal.target_position is not None
                        and agent.position == active_goal.target_position
                        and not agent.has_path
                )
                is_outside_corridor = not self.is_in_corridor_zone(agent.position)
                destination_cleared = is_at_destination and is_outside_corridor
                junction_passed = agent.position != junction and junction not in agent.path

                if (has_margin or destination_cleared) and junction_passed:
                    other_goal.yield_for_agent_id = None
                    agent.evasion_phase = EvasionPhase.CLEARANCE_CONFIRMED
                    self._logger.log(
                        SimulationEvent(
                            tick=current_tick,
                            agent_id=agent.id,
                            event_type="evasion_phase_changed",
                            summary=f"Agent {agent.name}: Status gewechselt zu {EvasionPhase.CLEARANCE_CONFIRMED.value}.",
                            payload={"phase": EvasionPhase.CLEARANCE_CONFIRMED.value, "recipient_id": other.id},
                        )
                    )
                    is_group = bool(other_goal.is_group_goal)
                    clearance_msg = DialogueTemplates.clearance(is_group=is_group)
                    other.receive_message(
                        IncomingMessage(
                            from_agent_id=agent.id,
                            from_agent_name=agent.name,
                            message=clearance_msg,
                            channel=CommunicationChannel.LOCAL_TALK,
                            is_courtesy=True,
                            is_resume_signal=True,
                        )
                    )
                    self._dialogue_history.record_dialogue(
                        tick=current_tick,
                        sender_id=agent.id,
                        sender_name=agent.name,
                        recipient_id=other.id,
                        recipient_name=other.name,
                        message=clearance_msg,
                    )

    def is_in_corridor_zone(self, pos: Position) -> bool:
        up = Position(pos.x, pos.y - 1)
        down = Position(pos.x, pos.y + 1)
        if not self._grid.is_walkable(up) and not self._grid.is_walkable(down):
            return True
        left = Position(pos.x - 1, pos.y)
        right = Position(pos.x + 1, pos.y)
        if not self._grid.is_walkable(left) and not self._grid.is_walkable(right):
            return True
        return False

    def replan_agent_path_avoiding(self, agent: Agent, forbidden_tiles: set[Position]) -> None:
        active_goal = agent.active_goal
        if not active_goal or not active_goal.target_position:
            return

        target = active_goal.target_position
        temp_obstacles: set[Position] = set()
        current_tick = self._tick_provider()

        for pos in forbidden_tiles:
            if pos != target and pos != agent.position:
                if agent.mental_map.is_walkable(pos):
                    temp_obstacles.add(pos)
                    agent.mental_map.update_tile(pos, is_walkable=False, tick=current_tick)

        new_path = self._pathfinder.find_path(agent.position, target, agent.mental_map)

        for pos in temp_obstacles:
            agent.mental_map.update_tile(pos, is_walkable=True, tick=current_tick)

        if new_path:
            agent.assign_path(new_path)