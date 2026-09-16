from __future__ import annotations

import uuid
from typing import Callable, Optional
from src.application.services.action_executor import ActionExecutor
from src.application.services.dialogue_history import DialogueHistory
from src.application.services.dialogue_session_manager import DialogueSessionManager
from src.application.services.evasion_finder import EvasionFinder
from src.application.services.goal_service import GoalService
from src.domain.models.agent import Agent
from src.domain.models.cognition import (
    DialogueResolution,
    EndDialogueAction,
    GoalIntent,
    TalkAction,
)
from src.domain.models.events import SimulationEvent
from src.domain.models.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.dialogue_coordinator import IDialogueCoordinator
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder


class DialogueCoordinator(IDialogueCoordinator):
    def __init__(
        self,
        logger: IEventLogger,
        cognition_provider: ICognitionProvider,
        pathfinder: IPathfinder,
        goal_service: GoalService,
        evasion_finder: EvasionFinder,
        action_executor: ActionExecutor,
        session_manager: DialogueSessionManager,
        dialogue_history: DialogueHistory,
        tick_provider: Optional[Callable[[], int]] = None,
    ) -> None:
        self._logger = logger
        self._cognition_provider = cognition_provider
        self._pathfinder = pathfinder
        self._goal_service = goal_service
        self._evasion_finder = evasion_finder
        self._action_executor = action_executor
        self._session_manager = session_manager
        self._dialogue_history = dialogue_history
        self._tick_provider = tick_provider or (lambda: 0)

    async def handle_incoming_dialogue(
        self,
        agent: Agent,
        all_entities: list[WorldEntity],
    ) -> None:
        current_tick = self._tick_provider()
        incident_id = f"dlg-t{current_tick}-{agent.id}-{uuid.uuid4().hex[:6]}"

        incoming_messages = agent.drain_inbox(assimilate=True, tick=current_tick)
        received_dicts = [msg.to_dict() for msg in incoming_messages]

        partner_id = incoming_messages[-1].from_agent_id if incoming_messages else None
        partner = next((other for other in all_entities if other.id == partner_id), None)

        if partner:
            agent.memory.update_entity_perception(
                entity_id=partner.id,
                name=partner.name,
                pos=partner.position,
                tick=current_tick,
            )

        turn_count = self._session_manager.increment_turn(agent.id, partner_id)

        res_self, res_partner = (None, None)
        if partner and isinstance(partner, Agent):
            res_self, res_partner = self._evasion_finder.compare_evasion_distances(
                agent_a=agent,
                agent_b=partner,
                all_entities=all_entities,
            )

        len_self = len(res_self.path) if res_self and res_self.path else None
        len_partner = len(res_partner.path) if res_partner and res_partner.path else None

        recommended_role = "yield"
        if len_self is not None and len_partner is not None:
            recommended_role = "yield" if len_self <= len_partner else "pass"
        elif len_self is None and len_partner is not None:
            recommended_role = "pass"

        latest_intent = incoming_messages[-1].intent if incoming_messages else None

        context = {
            "agent_id": agent.id,
            "name": agent.name,
            "partner_id": partner.id if partner else None,
            "partner_name": partner.name if partner else "Unbekannt",
            "partner_is_conversational": partner.is_conversational if partner else False,
            "current_x": agent.position.x,
            "current_y": agent.position.y,
            "received_messages": received_dicts,
            "recent_dialogues": self._dialogue_history.get_recent_formatted(limit=8),
            "recent_dialogues_structured": self._dialogue_history.get_recent_structured(limit=8),
            "active_goal": agent.active_goal.to_dict() if agent.active_goal else None,
            "goal_stack": [g.to_dict() for g in agent.goals],
            "dialogue_turn_count": turn_count,
            "evasion_distance_self": len_self,
            "evasion_distance_partner": len_partner,
            "recommended_role": recommended_role,
            "incoming_intent": latest_intent,
            "peer_bid_farewell": agent.peer_bid_farewell,
        }

        try:
            resolution: DialogueResolution = await self._cognition_provider.respond_to_dialogue(context)

            chosen_intent = resolution.negotiation_intent
            if isinstance(resolution.action, TalkAction) and chosen_intent:
                resolution.action.intent = chosen_intent

            # Wenn der Agent Verabschiedung ablehnt (reject), Reset beider Seiten
            if chosen_intent == "reject":
                agent.has_bid_farewell = False
                agent.peer_bid_farewell = False
                if partner and isinstance(partner, Agent):
                    partner.has_bid_farewell = False
                    partner.peer_bid_farewell = False

            if isinstance(resolution.action, EndDialogueAction):
                self._session_manager.reset_session(agent.id, partner_id)

            self._action_executor.execute_dialogue_action(
                agent=agent,
                partner=partner,
                action=resolution.action,
                incident_id=incident_id,
                all_entities=all_entities,
            )

            partner_is_yielding = (
                isinstance(partner, Agent)
                and (
                    partner.is_evasion_locked
                    or (partner.active_goal is not None and partner.active_goal.yield_for_agent_id == agent.id)
                )
            )

            if latest_intent == "offer_yield" and chosen_intent == "accept":
                self._session_manager.reset_session(agent.id, partner_id)
                return

            wants_to_evade = (
                (latest_intent == "request_yield" and chosen_intent == "accept")
                or chosen_intent == "offer_yield"
                or (resolution.new_goal is not None and resolution.new_goal.intent_type == "evade")
            )

            if wants_to_evade:
                if partner_is_yielding:
                    self._logger.log(
                        SimulationEvent(
                            tick=current_tick,
                            agent_id=agent.id,
                            event_type="evasion_suppressed",
                            summary=f"Agent {agent.name} weicht nicht aus, da {partner.name} bereits Platz macht.",
                            payload={"incident_id": incident_id, "partner_id": partner.id},
                        )
                    )
                else:
                    partner_pos = partner.position if partner else agent.position
                    sub_goal_name = (
                        resolution.new_goal.name
                        if resolution.new_goal
                        else "In Nische ausweichen"
                    )
                    self._action_executor.execute_evasion(
                        agent=agent,
                        partner=partner,
                        blocked_pos=partner_pos,
                        all_entities=all_entities,
                        incident_id=incident_id,
                        thought=resolution.thought,
                        sub_goal_name=sub_goal_name,
                    )
                return

        except Exception as err:
            self._logger.log(
                SimulationEvent(
                    tick=self._tick_provider(),
                    agent_id=agent.id,
                    event_type="dialogue_failed",
                    summary=f"Fehler im Dialog von Agent {agent.name}: {err}",
                    payload={"incident_id": incident_id, "error": str(err)},
                )
            )
        finally:
            agent.is_thinking = False