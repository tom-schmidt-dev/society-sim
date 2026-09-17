from __future__ import annotations

from typing import Any, Callable, Optional
from src.application.services.critical_section_coordinator import CriticalSectionCoordinator
from src.application.services.dialogue_history import DialogueHistory
from src.application.services.evasion_finder import EvasionFinder
from src.application.services.goal_service import GoalService
from src.domain.models.agent import Agent
from src.domain.models.cognition import (
    AbortAction,
    EndDialogueAction,
    InspectAction,
    ProbeAction,
    RerouteAction,
    TalkAction,
    WaitAction,
)
from src.domain.models.events import SimulationEvent
from src.domain.models.goal import ExecutionPriority, Goal
from src.domain.models.message import CommunicationChannel, IncomingMessage
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.services.perception_service import PerceptionService


class ActionExecutor:
    def __init__(
        self,
        grid: WorldGrid,
        logger: IEventLogger,
        dialogue_history: DialogueHistory,
        goal_service: GoalService,
        pathfinder: IPathfinder,
        perception_service: Optional[PerceptionService] = None,
        evasion_finder: Optional[EvasionFinder] = None,
        critical_section_coordinator: Optional[CriticalSectionCoordinator] = None,
        tick_provider: Optional[Callable[[], int]] = None,
    ) -> None:
        self._grid = grid
        self._logger = logger
        self._dialogue_history = dialogue_history
        self._goal_service = goal_service
        self._pathfinder = pathfinder
        self._perception_service = perception_service or PerceptionService(default_radius=3)
        self._evasion_finder = evasion_finder or EvasionFinder(pathfinder)
        self._tick_provider = tick_provider or (lambda: 0)
        self._critical_section_coordinator = (
            critical_section_coordinator
            or CriticalSectionCoordinator(logger=self._logger, tick_provider=self._tick_provider)
        )

    @property
    def critical_section_coordinator(self) -> CriticalSectionCoordinator:
        return self._critical_section_coordinator

    @staticmethod
    def find_entity(
        target_id: str, all_entities: list[WorldEntity]
    ) -> Optional[WorldEntity]:
        target_lower = target_id.lower()
        for entity in all_entities:
            if (
                entity.id == target_id
                or entity.name.lower() == target_lower
                or f"agent {entity.name.lower()}" == target_lower
            ):
                return entity
        return None

    @staticmethod
    def create_talk_message(
        agent: Agent,
        message_text: str,
        channel: CommunicationChannel = CommunicationChannel.LOCAL_TALK,
        share_path: bool = True,
        is_path_update: bool = False,
        correlation_key: Optional[str] = None,
        is_evasion_notice: bool = False,
        intent: Optional[str] = None,
        is_farewell: bool = False,
    ) -> IncomingMessage:
        path_payload = None
        active_goal = agent.active_goal
        if share_path and active_goal and agent.has_path:
            path_payload = [agent.position] + list(agent.path)

        return IncomingMessage(
            from_agent_id=agent.id,
            from_agent_name=agent.name,
            message=message_text,
            channel=channel,
            planned_path=path_payload,
            is_path_update=is_path_update,
            correlation_key=correlation_key,
            is_evasion_notice=is_evasion_notice,
            intent=intent,
            is_farewell=is_farewell,
        )

    @staticmethod
    def validate_pre_commit(
        agent: Agent,
        blocker: WorldEntity,
        blocked_pos: Position,
    ) -> bool:
        """TOCTOU-Schutz: Prüft unmittelbar vor Ausführung, ob die Blockade noch besteht."""
        if not agent.has_path or agent.path[0] != blocked_pos:
            return False
        if blocker.position != blocked_pos:
            return False
        return True

    def _handle_non_conversational_talk(
        self,
        agent: Agent,
        target: WorldEntity,
        message: str,
        incident_id: str,
        current_tick: int,
    ) -> None:
        target.receive_message(
            IncomingMessage(
                from_agent_id=agent.id,
                from_agent_name=agent.name,
                message=message,
                channel=CommunicationChannel.LOCAL_TALK,
            )
        )
        empty_resp = IncomingMessage(
            from_agent_id=target.id,
            from_agent_name=target.name,
            message="",
            channel=CommunicationChannel.LOCAL_TALK,
            is_empty_response=True,
        )
        agent.receive_message(empty_resp)
        agent.assimilate_message(empty_resp, tick=current_tick)

        if agent.memory.get_entity_walkability(target.id) is False:
            agent.mental_map.mark_obstacle(target.position, current_tick)

        self._dialogue_history.record_dialogue(
            tick=current_tick,
            sender_id=target.id,
            sender_name=target.name,
            recipient_id=agent.id,
            recipient_name=agent.name,
            message="",
            is_empty_response=True,
        )

    def dispatch_message(
            self,
            sender: Agent,
            recipient: WorldEntity,
            message: IncomingMessage,
            incident_id: str,
    ) -> bool:
        """Prüft die Kanalreichweite und stellt die Nachricht bei Erfolg in den Staging-Puffer."""
        current_tick = self._tick_provider()
        in_range = self._perception_service.is_in_channel_range(
            sender.position, recipient.position, message.channel
        )
        if not in_range:
            dist = sender.position.manhattan_distance(recipient.position)
            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=sender.id,
                    event_type="message_out_of_range",
                    summary=f"Nachricht von {sender.name} an {recipient.name} verworfen: Außerhalb Kanalreichweite ({message.channel.value}, Distanz: {dist}).",
                    payload={
                        "incident_id": incident_id,
                        "recipient_id": recipient.id,
                        "channel": message.channel.value,
                        "distance": dist,
                    },
                )
            )
            return False

        recipient.receive_message(message)
        return True

    def execute_evasion(
        self,
        agent: Agent,
        partner: Optional[WorldEntity],
        blocked_pos: Position,
        all_entities: list[WorldEntity],
        incident_id: str,
        thought: str,
        sub_goal_name: Optional[str] = None,
    ) -> bool:
        """Führt die Ausweichkaskade mit atomarer Pre-Commit-Pfadvalidierung aus."""
        # 1. Pre-Commit Check: Ist die Blockadesituation überhaupt noch existent?
        if partner and partner.position != blocked_pos:
            self._logger.log(
                SimulationEvent(
                    tick=self._tick_provider(),
                    agent_id=agent.id,
                    event_type="evasion_precommit_invalidated",
                    summary=f"Agent {agent.name}: Ausweichen verworfen, da Partner {partner.name} Feld ({blocked_pos.x}, {blocked_pos.y}) geräumt hat.",
                    payload={"incident_id": incident_id, "partner_id": partner.id},
                )
            )
            return False

        occupied = {other.position for other in all_entities if other.id != agent.id}
        partner_id = partner.id if partner else None
        partner_fact = agent.memory.known_entities.get(partner_id) if partner_id else None
        partner_trajectory = partner_fact.partner_planned_path if partner_fact else None

        search_dir: Optional[tuple[float, float]] = None
        if partner_fact:
            if partner_fact.last_observed_velocity != (0.0, 0.0):
                search_dir = partner_fact.last_observed_velocity
            elif partner_fact.smoothed_velocity != (0.0, 0.0):
                search_dir = partner_fact.smoothed_velocity

        evasion_res = self._evasion_finder.find_nearest_evasion_tile(
            start=agent.position,
            blocked_pos=blocked_pos,
            grid=agent.mental_map,
            occupied_positions=occupied,
            partner_trajectory=partner_trajectory,
            search_direction=search_dir,
        )

        # 2. Pre-Commit Check: Ist der berechnete Ausweichpfad weiterhin kollisionsfrei?
        if evasion_res and evasion_res.path:
            is_path_blocked = any(step in occupied for step in evasion_res.path)
            if is_path_blocked:
                self._logger.log(
                    SimulationEvent(
                        tick=self._tick_provider(),
                        agent_id=agent.id,
                        event_type="evasion_precommit_invalidated",
                        summary=f"Agent {agent.name}: Ausweichpfad nach ({evasion_res.target_tile.x}, {evasion_res.target_tile.y}) ist belegt.",
                        payload={"incident_id": incident_id},
                    )
                )
                return False

        agent.is_waiting_for_reply = False
        agent.is_listening_to_peer = False

        active_goal = agent.active_goal
        if active_goal and active_goal.priority > ExecutionPriority.URGENT:
            self._goal_service.pause_goal(agent, active_goal, incident_id=incident_id)

        if evasion_res and not evasion_res.is_frontier:
            goal_name = sub_goal_name or "In Nische ausweichen"
            self._goal_service.push_goal(
                agent,
                Goal(
                    name=goal_name,
                    target_position=evasion_res.target_tile,
                    junction_position=evasion_res.junction_tile,
                    yield_for_agent_id=partner_id,
                    is_evasion_hold=False,
                    priority=ExecutionPriority.URGENT,
                    description=thought,
                ),
                incident_id=incident_id,
            )
            if evasion_res.path:
                agent.assign_path(evasion_res.path)

            if partner:
                msg = self.create_talk_message(
                    agent=agent,
                    message_text=f"Ich mache Platz und weiche nach ({evasion_res.target_tile.x}, {evasion_res.target_tile.y}) aus.",
                    channel=CommunicationChannel.LOCAL_TALK,
                    share_path=True,
                    is_evasion_notice=True,
                )
                partner.receive_message(msg)
            return True

        elif evasion_res and evasion_res.is_frontier:
            self._goal_service.push_goal(
                agent,
                Goal(
                    name="Erkunde Terrain",
                    target_position=evasion_res.target_tile,
                    yield_for_agent_id=partner_id,
                    is_evasion_hold=False,
                    priority=ExecutionPriority.URGENT,
                    description="Erkundet unbekanntes Gebiet auf der Suche nach einer Nische.",
                ),
                incident_id=incident_id,
            )
            if evasion_res.path:
                agent.assign_path(evasion_res.path)

            if partner:
                msg = self.create_talk_message(
                    agent=agent,
                    message_text="Ich suche nach einer Nische und erkunde den Bereich.",
                    channel=CommunicationChannel.LOCAL_TALK,
                    share_path=True,
                )
                partner.receive_message(msg)
            return True

        else:
            if partner:
                msg = IncomingMessage(
                    from_agent_id=agent.id,
                    from_agent_name=agent.name,
                    message="Ich kann nicht ausweichen, kein freies Feld gefunden.",
                    channel=CommunicationChannel.LOCAL_TALK,
                )
                partner.receive_message(msg)
            return False

    def execute_inspection(
        self,
        agent: Agent,
        blocker: WorldEntity,
        incident_id: str,
        current_tick: int,
    ) -> None:
        resource_key = f"entity:{blocker.id}"
        active_goal = agent.active_goal
        prio = active_goal.priority if active_goal else ExecutionPriority.ROUTINE
        has_lock = self._critical_section_coordinator.acquire_or_queue(agent, resource_key, prio)

        if not has_lock:
            self._goal_service.pause_goal(agent, incident_id=incident_id)
            return

        agent.memory.record_inspection(blocker.id, blocker.entity_type)
        if isinstance(blocker, Agent) or blocker.is_conversational:
            agent.memory.record_walkability_result(blocker.id, is_walkable=False)

        self._dialogue_history.record_dialogue(
            tick=current_tick,
            sender_id=agent.id,
            sender_name=agent.name,
            recipient_id=blocker.id,
            recipient_name=blocker.name,
            message=blocker.entity_type,
            is_inspection=True,
        )
        self._logger.log(
            SimulationEvent(
                tick=current_tick,
                agent_id=agent.id,
                event_type="entity_inspected",
                summary=f"Agent {agent.name} inspiziert {blocker.name}: Typ={blocker.entity_type}.",
                payload={
                    "incident_id": incident_id,
                    "target_id": blocker.id,
                    "entity_type": blocker.entity_type,
                },
            )
        )
        self._critical_section_coordinator.release(agent.id, resource_key, mark_completed=True)

    def execute_probe(
        self,
        agent: Agent,
        blocker: WorldEntity,
        incident_id: str,
        current_tick: int,
    ) -> None:
        resource_key = f"entity:{blocker.id}"
        active_goal = agent.active_goal
        prio = active_goal.priority if active_goal else ExecutionPriority.ROUTINE
        has_lock = self._critical_section_coordinator.acquire_or_queue(agent, resource_key, prio)

        if not has_lock:
            self._goal_service.pause_goal(agent, incident_id=incident_id)
            return

        is_walkable = getattr(blocker, "is_passable", False)
        agent.memory.record_walkability_result(blocker.id, is_walkable=is_walkable)
        agent.memory.record_type_walkability(
            blocker.entity_type, is_walkable=is_walkable, pos=blocker.position
        )

        if not is_walkable and (
            not blocker.is_conversational
            or not isinstance(blocker, Agent)
            or agent.memory.get_assumed_conversational(blocker.id) is False
        ):
            agent.mental_map.mark_obstacle(blocker.position, current_tick)

        self._dialogue_history.record_dialogue(
            tick=current_tick,
            sender_id=agent.id,
            sender_name=agent.name,
            recipient_id=blocker.id,
            recipient_name=blocker.name,
            message=f"Erprobt: Passierbar={'Ja' if is_walkable else 'Nein'}",
            is_inspection=True,
        )
        self._logger.log(
            SimulationEvent(
                tick=current_tick,
                agent_id=agent.id,
                event_type="entity_probed",
                summary=f"Agent {agent.name} erprobt {blocker.name}: Passierbar={is_walkable}.",
                payload={
                    "incident_id": incident_id,
                    "target_id": blocker.id,
                    "entity_type": blocker.entity_type,
                    "is_walkable": is_walkable,
                },
            )
        )
        self._critical_section_coordinator.release(agent.id, resource_key, mark_completed=True)

    def execute_blockage_action(
        self,
        agent: Agent,
        blocker: WorldEntity,
        action: Any,
        incident_id: str,
        all_entities: list[WorldEntity],
        blocked_pos: Position,
        current_tick: Optional[int] = None,
        thought: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        tick = current_tick if current_tick is not None else self._tick_provider()

        # TOCTOU-Validierung vor Commit
        if not self.validate_pre_commit(agent, blocker, blocked_pos):
            self._logger.log(
                SimulationEvent(
                    tick=tick,
                    agent_id=agent.id,
                    event_type="action_precommit_invalidated",
                    summary=f"Agent {agent.name}: Aktion verworfen, da Blockadesituation an ({blocked_pos.x}, {blocked_pos.y}) nicht mehr besteht.",
                    payload={"incident_id": incident_id, "action": action.action_type},
                )
            )
            return

        if isinstance(action, TalkAction):
            target = self.find_entity(action.target_agent_id, all_entities) or blocker
            target_name = target.name if target else action.target_agent_id
            target_id = target.id if target else None

            self._dialogue_history.record_dialogue(
                tick=tick,
                sender_id=agent.id,
                sender_name=agent.name,
                recipient_id=target_id,
                recipient_name=target_name,
                message=action.message,
                intent=action.intent,
            )

            if not target.is_conversational:
                self._handle_non_conversational_talk(
                    agent, target, action.message, incident_id, tick
                )
            elif isinstance(target, Agent):
                msg = self.create_talk_message(agent, action.message, channel=CommunicationChannel.LOCAL_TALK,
                                               intent=action.intent)
                delivered = self.dispatch_message(agent, target, msg, incident_id)
                if delivered:
                    agent.is_waiting_for_reply = True
                    agent.interaction_partner_id = target.id
                    target.interaction_partner_id = agent.id

        elif isinstance(action, InspectAction):
            self.execute_inspection(agent, blocker, incident_id, tick)

        elif isinstance(action, ProbeAction):
            target = self.find_entity(action.target_agent_id, all_entities) or blocker
            self.execute_probe(agent, target, incident_id, tick)

        elif isinstance(action, WaitAction):
            self._goal_service.push_goal(
                agent,
                Goal(
                    name=f"Warten ({action.ticks} Ticks)",
                    holds_position=True,
                    remaining_ticks=action.ticks,
                    description=action.reason,
                ),
                incident_id=incident_id,
            )

        elif isinstance(action, RerouteAction):
            active_goal = agent.active_goal
            target_pos = None
            if active_goal and active_goal.target_position:
                target_pos = active_goal.target_position
            elif agent.has_path:
                target_pos = agent.path[-1]

            if target_pos:
                new_path = self._pathfinder.find_path(agent.position, target_pos, agent.mental_map)
                if new_path:
                    agent.assign_path(new_path)

        elif isinstance(action, AbortAction):
            agent.clear_path()
            agent.abandon_active_goal()

        self._logger.log(
            SimulationEvent(
                tick=tick,
                agent_id=agent.id,
                event_type="blockage_resolved",
                summary=f"{agent.name} führt {action.action_type} aus. Grund: {action.reason}.",
                payload={
                    "incident_id": incident_id,
                    "duration_ms": duration_ms,
                    "action": action.action_type,
                    "reason": action.reason,
                    "internal_thought": thought,
                },
            )
        )

    def execute_dialogue_action(
        self,
        agent: Agent,
        partner: Optional[WorldEntity],
        action: TalkAction | EndDialogueAction,
        incident_id: str,
        all_entities: list[WorldEntity],
    ) -> None:
        current_tick = self._tick_provider()

        if isinstance(action, EndDialogueAction):
            final_msg = action.final_message or f"Ich beende das Gespräch: {action.reason}"
            partner_name = partner.name if partner else "Raum"
            partner_id = partner.id if partner else None

            self._dialogue_history.record_dialogue(
                tick=current_tick,
                sender_id=agent.id,
                sender_name=agent.name,
                recipient_id=partner_id,
                recipient_name=partner_name,
                message=final_msg,
            )

            active_goal = agent.active_goal
            is_evading = active_goal is not None and active_goal.priority == ExecutionPriority.URGENT
            agent.has_bid_farewell = True
            agent.is_waiting_for_reply = not is_evading

            if partner and partner.is_conversational:
                farewell_msg = IncomingMessage(
                    from_agent_id=agent.id,
                    from_agent_name=agent.name,
                    message=final_msg,
                    channel=CommunicationChannel.LOCAL_TALK,
                    is_farewell=True,
                )
                self.dispatch_message(agent, partner, farewell_msg, incident_id)
                if isinstance(partner, Agent):
                    partner.peer_bid_farewell = True
                    if partner.has_bid_farewell or is_evading or not partner.interaction_partner_id:
                        agent.has_bid_farewell = False
                        agent.peer_bid_farewell = False
                        agent.is_listening_to_peer = False
                        agent.is_waiting_for_reply = False
                        agent.interaction_partner_id = None

                        partner.has_bid_farewell = False
                        partner.peer_bid_farewell = False
                        partner.is_listening_to_peer = False
                        partner.is_waiting_for_reply = False
                        partner.interaction_partner_id = None

        elif isinstance(action, TalkAction):
            target = self.find_entity(action.target_agent_id, all_entities) or partner
            target_name = target.name if target else action.target_agent_id
            target_id = target.id if target else None

            self._dialogue_history.record_dialogue(
                tick=current_tick,
                sender_id=agent.id,
                sender_name=agent.name,
                recipient_id=target_id,
                recipient_name=target_name,
                message=action.message,
                intent=action.intent,
            )

            if target:
                if not target.is_conversational:
                    self._handle_non_conversational_talk(
                        agent, target, action.message, incident_id, current_tick
                    )
                else:
                    msg = self.create_talk_message(agent, action.message, channel=CommunicationChannel.LOCAL_TALK, intent=action.intent)
                    delivered = self.dispatch_message(agent, target, msg, incident_id)
                    if delivered and action.intent not in ("accept", "offer_yield"):
                        agent.is_waiting_for_reply = True
                        agent.interaction_partner_id = target.id
                        if isinstance(target, Agent):
                            target.interaction_partner_id = agent.id