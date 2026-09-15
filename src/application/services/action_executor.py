from __future__ import annotations

from typing import Callable, Optional
from src.application.services.dialogue_history import DialogueHistory
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
from src.domain.models.goal import Goal
from src.domain.models.message import IncomingMessage
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder


class ActionExecutor:
    def __init__(
        self,
        grid: WorldGrid,
        logger: IEventLogger,
        dialogue_history: DialogueHistory,
        goal_service: GoalService,
        pathfinder: IPathfinder,
        tick_provider: Optional[Callable[[], int]] = None,
    ) -> None:
        self._grid = grid
        self._logger = logger
        self._dialogue_history = dialogue_history
        self._goal_service = goal_service
        self._pathfinder = pathfinder
        self._tick_provider = tick_provider or (lambda: 0)

    def find_entity(
        self, target_id: str, all_entities: list[WorldEntity]
    ) -> Optional[WorldEntity]:
        """Ermittelt eine Entität über ID oder Namensabgleich."""
        target_lower = target_id.lower()
        for entity in all_entities:
            if (
                entity.id == target_id
                or entity.name.lower() == target_lower
                or f"agent {entity.name.lower()}" == target_lower
            ):
                return entity
        return None

    def _handle_non_conversational_talk(
        self,
        agent: Agent,
        target: WorldEntity,
        message: str,
        incident_id: str,
        current_tick: int,
    ) -> None:
        """Behandelt die Interaktion mit passiven Entitäten einheitlich an zentraler Stelle."""
        target.receive_message(
            IncomingMessage(
                from_agent_id=agent.id,
                from_agent_name=agent.name,
                message=message,
            )
        )
        empty_resp = IncomingMessage(
            from_agent_id=target.id,
            from_agent_name=target.name,
            message="",
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

    def execute_blockage_action(
        self,
        agent: Agent,
        blocker: WorldEntity,
        blocked_pos: Position,
        action: InspectAction | ProbeAction | WaitAction | TalkAction | RerouteAction | AbortAction,
        incident_id: str,
        duration_ms: float,
        thought: str,
        all_entities: list[WorldEntity],
    ) -> None:
        """Führt eine Handlungsentscheidung aus einer Blockadesituation deterministisch aus."""
        current_tick = self._tick_provider()

        if isinstance(action, InspectAction):
            target = self.find_entity(action.target_agent_id, all_entities) or blocker
            inspection_msg = IncomingMessage(
                from_agent_id=target.id,
                from_agent_name=target.name,
                message=f"Typ: {target.entity_type}",
                is_inspection=True,
                inspected_entity_type=target.entity_type,
            )
            agent.assimilate_message(inspection_msg, tick=current_tick)

            self._dialogue_history.record_dialogue(
                tick=current_tick,
                sender_id=agent.id,
                sender_name=agent.name,
                recipient_id=target.id,
                recipient_name=target.name,
                message=target.entity_type,
                is_inspection=True,
            )

            if agent.memory.get_assumed_walkable(target.id) is False:
                agent.mental_map.mark_obstacle(target.position, current_tick)

            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=agent.id,
                    event_type="entity_inspected",
                    summary=f"Agent {agent.name} inspiziert {target.name}: Typ='{target.entity_type}'.",
                    payload={
                        "incident_id": incident_id,
                        "target_id": target.id,
                        "entity_type": target.entity_type,
                        "reason": action.reason,
                    },
                )
            )

        elif isinstance(action, ProbeAction):
            target = self.find_entity(action.target_agent_id, all_entities) or blocker
            is_physically_walkable = self._grid.is_walkable(target.position) and target.is_passable

            agent.memory.record_walkability_result(target.id, is_walkable=is_physically_walkable)
            agent.memory.record_type_walkability(target.entity_type, is_walkable=is_physically_walkable, pos=target.position)

            if not is_physically_walkable:
                if agent.memory.get_assumed_conversational(target.id) is False:
                    agent.mental_map.mark_obstacle(target.position, current_tick)

            self._dialogue_history.record_dialogue(
                tick=current_tick,
                sender_id=agent.id,
                sender_name=agent.name,
                recipient_id=target.id,
                recipient_name=target.name,
                message=f"Erprobt: Passierbar={'Ja' if is_physically_walkable else 'Nein'}",
                is_inspection=True,
            )

            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=agent.id,
                    event_type="entity_probed",
                    summary=f"Agent {agent.name} erprobt {target.name}: Passierbar={is_physically_walkable}.",
                    payload={
                        "incident_id": incident_id,
                        "target_id": target.id,
                        "entity_type": target.entity_type,
                        "is_walkable": is_physically_walkable,
                        "reason": action.reason,
                    },
                )
            )

        elif isinstance(action, WaitAction):
            self._goal_service.push_goal(
                agent,
                Goal(
                    name=f"Auswertung von {target.name}",
                    holds_position=True,
                    remaining_ticks=1,
                    description=f"Erprobung: {target.name} ist passierbar={is_physically_walkable}.",
                ),
                incident_id=incident_id,
            )

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

        elif isinstance(action, TalkAction):
            target = self.find_entity(action.target_agent_id, all_entities) or blocker
            self._dialogue_history.record_dialogue(
                tick=current_tick,
                sender_id=agent.id,
                sender_name=agent.name,
                recipient_id=target.id,
                recipient_name=target.name,
                message=action.message,
            )
            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=agent.id,
                    event_type="message_delivered",
                    summary=f"Agent {agent.name} sagt zu {target.name}: '{action.message}'",
                    payload={
                        "incident_id": incident_id,
                        "sender_id": agent.id,
                        "recipient_id": target.id,
                        "message": action.message,
                    },
                )
            )

            if not target.is_conversational:
                self._handle_non_conversational_talk(
                    agent, target, action.message, incident_id, current_tick
                )
            else:
                target.receive_message(
                    IncomingMessage(
                        from_agent_id=agent.id,
                        from_agent_name=agent.name,
                        message=action.message,
                    )
                )
                self._goal_service.push_goal(
                    target,
                    Goal(
                        name=f"Konversation mit {agent.name}",
                        holds_position=True,
                        description="Pausiert zur Beantwortung der Nachricht.",
                    ),
                    incident_id=incident_id,
                )
                self._goal_service.push_goal(
                    agent,
                    Goal(
                        name="Warten auf Antwort",
                        holds_position=True,
                        remaining_ticks=4,
                        description="Wartet nach Nachricht auf Reaktion.",
                    ),
                    incident_id=incident_id,
                )


        elif isinstance(action, RerouteAction):
            active_goal = agent.active_goal
            target_pos = (
                active_goal.target_position
                if active_goal and active_goal.target_position
                else (agent.path[-1] if agent.has_path else None)
            )
            if target_pos:
                # Kachelmarkierung erfolgt ausschließlich über verifizierte Fakten in ProbeAction/TalkAction
                new_path = self._pathfinder.find_path(agent.position, target_pos, agent.mental_map)
                if new_path:
                    agent.assign_path(new_path)

        elif isinstance(action, AbortAction):
            agent.clear_path()
            agent.abandon_active_goal()

        self._logger.log(
            SimulationEvent(
                tick=current_tick,
                agent_id=agent.id,
                event_type="blockage_resolved",
                summary=f"{agent.name} führt {action.action_type} aus. Grund: {action.reason}.",
                payload={
                    "incident_id": incident_id,
                    "duration_ms": duration_ms,
                    "action": action.action_type,
                    "reason": action.reason,
                    "internal_thought": thought,
                    "action_details": action.model_dump(),
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
        """Führt eine Konversationsentscheidung deterministisch aus."""
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
            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=agent.id,
                    event_type="message_delivered",
                    summary=f"Agent {agent.name} verabschiedet sich von {partner_name}: '{final_msg}'",
                    payload={
                        "incident_id": incident_id,
                        "sender_id": agent.id,
                        "recipient_id": partner_id,
                        "message": final_msg,
                        "reason": action.reason,
                    },
                )
            )

            if partner and partner.is_conversational:
                partner.receive_message(
                    IncomingMessage(
                        from_agent_id=agent.id,
                        from_agent_name=agent.name,
                        message=final_msg,
                        is_farewell=True,
                    )
                )
                if isinstance(partner, Agent) and partner.active_goal and (
                    "Konversation" in partner.active_goal.name or "Warten" in partner.active_goal.name
                ):
                    self._goal_service.pop_goal(
                        partner, target_goal=partner.active_goal, incident_id=incident_id
                    )

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
            )

            if target:
                if not target.is_conversational:
                    self._handle_non_conversational_talk(
                        agent, target, action.message, incident_id, current_tick
                    )
                else:
                    target.receive_message(
                        IncomingMessage(
                            from_agent_id=agent.id,
                            from_agent_name=agent.name,
                            message=action.message,
                        )
                    )
                    self._goal_service.push_goal(
                        agent,
                        Goal(
                            name="Warten auf Antwort",
                            holds_position=True,
                            remaining_ticks=4,
                            description="Wartet nach Gegenantwort.",
                        ),
                        incident_id=incident_id,
                    )