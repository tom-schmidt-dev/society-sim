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

        try:
            agent.commit_staging_messages()
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

            partner_goal = partner.active_goal if isinstance(partner, Agent) else None
            partner_is_yielding = (
                isinstance(partner, Agent)
                and (
                    partner.is_evasion_locked
                    or (partner_goal is not None and partner_goal.yield_for_agent_id == agent.id)
                )
            )
            agent_goal = agent.active_goal
            agent_is_yielding = (
                agent.is_evasion_locked
                or (agent_goal is not None and agent_goal.yield_for_agent_id == partner_id)
            )

            latest_intent = incoming_messages[-1].intent if incoming_messages else None

            # Terminal-Guard (Agreement Semaphore): Wenn der Agent bereits für den Partner ausweicht
            # und der Partner dies mit 'accept' bestätigt ("Danke, ich passiere."), ist die Einigung besiegelt.
            # Es bedarf keiner weiteren Kognitionsanfrage und keiner Antwortnachricht.
            if agent_is_yielding and latest_intent == "accept":
                self._session_manager.reset_session(agent.id, partner_id)
                agent.is_waiting_for_reply = False
                if agent.interaction_partner_id == partner_id:
                    agent.interaction_partner_id = None
                partner_name = partner.name if partner else (partner_id or "Partner")
                self._logger.log(
                    SimulationEvent(
                        tick=current_tick,
                        agent_id=agent.id,
                        event_type="dialogue_agreement_confirmed",
                        summary=f"Agent {agent.name} schließt Dialog ab: Ausweichen für {partner_name} wurde bestätigt.",
                        payload={"incident_id": incident_id, "partner_id": partner_id},
                    )
                )
                return

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
            if partner_is_yielding:
                recommended_role = "pass"
            elif agent_is_yielding:
                recommended_role = "yield"
            elif len_self is not None and len_partner is not None:
                recommended_role = "yield" if len_self <= len_partner else "pass"
            elif len_self is None and len_partner is not None:
                recommended_role = "pass"

            active_goal = agent.active_goal
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
                "active_goal": active_goal.to_dict() if active_goal else None,
                "goal_stack": [g.to_dict() for g in agent.goals],
                "dialogue_turn_count": turn_count,
                "evasion_distance_self": len_self,
                "evasion_distance_partner": len_partner,
                "recommended_role": recommended_role,
                "incoming_intent": latest_intent,
                "peer_bid_farewell": agent.peer_bid_farewell,
                "partner_is_yielding": partner_is_yielding,
                "agent_is_yielding": agent_is_yielding,
            }

            if partner_id and self._session_manager.is_turn_limit_exceeded(agent.id, partner_id):
                if agent_is_yielding:
                    resolution: DialogueResolution = DialogueResolution(
                        thought="Maximale Gesprächsrunden erreicht. Ich weiche bereits wie vereinbart aus.",
                        action=EndDialogueAction(
                            reason="Maximale Rundenanzahl überschritten.",
                            final_message="Die Absprache steht, ich weiche aus.",
                        ),
                        negotiation_intent="accept",
                    )
                elif recommended_role == "yield" and not partner_is_yielding:
                    resolution = DialogueResolution(
                        thought="Maximale Gesprächsrunden erreicht. Schlichter entscheidet: Ich weiche aus.",
                        action=EndDialogueAction(
                            reason="Maximale Rundenanzahl überschritten.",
                            final_message="Wir kommen hier zu keiner Einigung. Ich beende das Gespräch und weiche aus.",
                        ),
                        negotiation_intent="accept",
                        new_goal=GoalIntent(name="In Nische ausweichen", intent_type="evade"),
                    )
                else:
                    resolution = DialogueResolution(
                        thought="Maximale Gesprächsrunden erreicht. Schlichter entscheidet: Ich passiere.",
                        action=EndDialogueAction(
                            reason="Maximale Rundenanzahl überschritten.",
                            final_message="Wir kommen hier zu keiner Einigung. Ich passiere.",
                        ),
                        negotiation_intent="accept",
                    )
            else:
                resolution = await self._cognition_provider.respond_to_dialogue(context)

            chosen_intent = resolution.negotiation_intent
            if isinstance(resolution.action, TalkAction) and chosen_intent:
                resolution.action.intent = chosen_intent

            # Ausweich-Semaphor / Pre-Dispatch Guard:
            # Wenn der Partner bereits ausweicht, darf kein konkurrierendes 'offer_yield'
            # gesendet oder ein eigenes Ausweichziel gesetzt werden.
            if partner_is_yielding:
                is_offering_evasion = (
                    chosen_intent == "offer_yield"
                    or (isinstance(resolution.action, TalkAction) and resolution.action.intent == "offer_yield")
                    or (resolution.new_goal is not None and resolution.new_goal.intent_type == "evade")
                )
                if is_offering_evasion:
                    if partner is not None:
                        self._logger.log(
                            SimulationEvent(
                                tick=current_tick,
                                agent_id=agent.id,
                                event_type="evasion_suppressed",
                                summary=f"Agent {agent.name} weicht nicht aus, da {partner.name} bereits Platz macht.",
                                payload={"incident_id": incident_id, "partner_id": partner.id},
                            )
                        )
                    chosen_intent = "accept"
                    if isinstance(resolution.action, TalkAction):
                        resolution.action.intent = "accept"
                        resolution.action.message = "Danke, ich passiere."
                    resolution.new_goal = None

            # Wenn der Agent Verabschiedung ablehnt (reject), Reset beider Seiten
            if chosen_intent == "reject":
                agent.has_bid_farewell = False
                agent.peer_bid_farewell = False
                if partner and isinstance(partner, Agent):
                    partner.has_bid_farewell = False
                    partner.peer_bid_farewell = False

            # Terminal-Handshake Guard gegen wechselseitige accept-Schleifen:
            # Hat der Partner bereits 'accept' gesendet und der Agent wählt ebenfalls 'accept',
            # ist die Einigung beidseitig final. Es darf keine weitere TalkAction gesendet werden,
            # die den Partner erneut zu einer Antwort verleiten würde.
            if latest_intent == "accept" and chosen_intent == "accept":
                if isinstance(resolution.action, TalkAction):
                    self._dialogue_history.record_dialogue(
                        tick=current_tick,
                        sender_id=agent.id,
                        sender_name=agent.name,
                        recipient_id=partner.id if partner else None,
                        recipient_name=partner.name if partner else "Partner",
                        message=resolution.action.message,
                        intent="accept",
                    )
                elif isinstance(resolution.action, EndDialogueAction):
                    self._action_executor.execute_dialogue_action(
                        agent=agent,
                        partner=partner,
                        action=resolution.action,
                        incident_id=incident_id,
                        all_entities=all_entities,
                    )
                self._session_manager.reset_session(agent.id, partner_id)
                agent.is_waiting_for_reply = False
                if agent.interaction_partner_id == partner_id:
                    agent.interaction_partner_id = None
                return

            if isinstance(resolution.action, EndDialogueAction):
                self._session_manager.reset_session(agent.id, partner_id)

            self._action_executor.execute_dialogue_action(
                agent=agent,
                partner=partner,
                action=resolution.action,
                incident_id=incident_id,
                all_entities=all_entities,
            )

            if (latest_intent == "offer_yield" and chosen_intent == "accept") or (partner_is_yielding and chosen_intent == "accept"):
                self._session_manager.reset_session(agent.id, partner_id)
                agent.is_waiting_for_reply = False
                if agent.interaction_partner_id == partner_id:
                    agent.interaction_partner_id = None
                return

            wants_to_evade = (
                (latest_intent == "request_yield" and chosen_intent == "accept")
                or chosen_intent == "offer_yield"
                or (resolution.new_goal is not None and resolution.new_goal.intent_type == "evade")
            )

            if wants_to_evade:
                agent.is_waiting_for_reply = False
                if agent.interaction_partner_id == partner_id:
                    agent.interaction_partner_id = None
                if partner_is_yielding and partner is not None:
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