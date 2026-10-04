from __future__ import annotations

import time
from typing import Any
import uuid
from typing import Callable, Optional
from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.coordination.dialogue_history import DialogueHistory
from src.application.services.coordination.dialogue_session_manager import DialogueSessionManager
from src.application.services.movement.evasion_finder import EvasionFinder
from src.application.services.cognition.goal_service import GoalService
from src.domain.models.agent.agent import Agent
from typing import cast
from src.domain.models.agent.agent_state import AgentLifecycleState
from src.domain.models.interaction.interaction_result import ImmediateResult
from src.domain.models.planning.cognition import (
    DialogueResolution,
    EndDialogueAction,
    GoalIntent,
    TalkAction,
)
from src.domain.models.planning.events import SimulationEvent
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.dialogue_coordinator import IDialogueCoordinator
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.vector_memory_store import IVectorMemoryStore
from src.domain.services.affect_service import AffectService


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
        vector_memory_store: Optional[IVectorMemoryStore] = None,
        max_context_dialogues: int = 20,
        affect_service: Optional[AffectService] = None,
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
        self._vector_memory_store = vector_memory_store
        self._max_context_dialogues = max_context_dialogues
        self._affect_service = affect_service or AffectService()

    def _retrieve_social_memories(
        self, agent_id: str, partner_id: Optional[str]
    ) -> list[str]:
        """Ruft Vergangenheitserfahrungen über den Interaktionspartner resilient ab."""
        if not self._vector_memory_store or not partner_id:
            return []

        query = f"Begegnung mit {partner_id}"
        try:
            return self._vector_memory_store.retrieve_relevant(
                agent_id=agent_id,
                query=query,
                limit=3,
                metadata_filter={"category": "social"},
            )
        except Exception:
            return []

    async def _synthesize_and_store_social_reflection(
        self,
        agent: Agent,
        partner: Optional[WorldEntity],
        incident_id: str,
        current_tick: int,
        recent_dialogues: list[str],
    ) -> None:
        """
        HINWEIS ZUR ARCHITEKTUR / PERFORMANCE:
        Die Vektorisierung und Meinungsbildung erfolgt aktuell unmittelbar am Tag nach Gesprächsende.
        Falls die zusätzliche LLM-Inferenz und das Schreiben in ChromaDB den Echtzeitfluss der
        Darstellungsschicht (ConsolePresenter / FrameBufferService) beeinträchtigen, kann diese
        Vektorisierung modular in die nächtliche Konsolidierungsphase (MemoryConsolidationService)
        verlagert werden, indem am Tag lediglich ein flüchtiges Event gepuffert wird.
        """
        partner_name = partner.name if partner else "Unbekannt"
        partner_id = partner.id if partner else "unknown"

        reflection_context = {
            "agent_id": agent.id,
            "agent_name": agent.name,
            "partner_id": partner_id,
            "partner_name": partner_name,
            "recent_dialogues": recent_dialogues,
        }

        try:
            start_time = time.perf_counter()
            reflection = await self._cognition_provider.reflect_on_dialogue(reflection_context)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        except Exception as err:
            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=agent.id,
                    event_type="social_reflection_failed",
                    summary=f"Soziale Reflexion für {agent.name} über {partner_name} fehlgeschlagen: {err}",
                    payload={"incident_id": incident_id, "error": str(err)},
                )
            )
            return

        memory_text = (
            f"[Soziale Interaktion mit {partner_name}]: {reflection.assessment} "
            f"Verlauf: {reflection.progression_summary}"
        )

        # 1. Direkte Vektorisierung
        if self._vector_memory_store:
            try:
                self._vector_memory_store.add_memories(
                    agent_id=agent.id,
                    memories=[memory_text],
                    metadatas=[{
                        "category": "social",
                        "partner_id": partner_id,
                        "tick": current_tick,
                    }],
                )
            except Exception as err:
                self._logger.log(
                    SimulationEvent(
                        tick=current_tick,
                        agent_id=agent.id,
                        event_type="vector_memory_failed",
                        summary=f"Vektorspeicherung für {agent.name} fehlgeschlagen: {err}",
                        payload={"incident_id": incident_id, "error": str(err)},
                    )
                )

        # 2. Strukturiertes Fine-Tuning-Log für SLM
        self._logger.log(
            SimulationEvent(
                tick=current_tick,
                agent_id=agent.id,
                event_type="social_reflection_completed",
                summary=f"Agent {agent.name}: Soziale Reflexion über {partner_name} abgeschlossen.",
                payload={
                    "incident_id": incident_id,
                    "agent_id": agent.id,
                    "partner_id": partner_id,
                    "assessment": reflection.assessment,
                    "progression_summary": reflection.progression_summary,
                    "persisted_memory": memory_text,
                    "dialogue_history": recent_dialogues,
                    "duration_ms": duration_ms,
                },
            )
        )

    async def handle_incoming_dialogue(
            self,
            agent: Agent,
            all_entities: list[WorldEntity],
    ) -> None:
        current_tick = self._tick_provider()
        incident_id = f"dlg-t{current_tick}-{agent.id}-{uuid.uuid4().hex[:6]}"
        chosen_intent: Optional[str] = None

        try:
            agent.commit_staging_messages()
            incoming_messages = agent.drain_inbox(assimilate=True, tick=current_tick)
            received_dicts = [msg.to_dict() for msg in incoming_messages]

            dialogue_messages = [
                m for m in incoming_messages
                if not (
                    m.is_courtesy
                    or m.is_halt_request
                    or m.is_resume_signal
                    or m.is_evasion_notice
                    or m.is_path_update
                )
            ]
            latest_dialogue_msg = dialogue_messages[-1] if dialogue_messages else (
                incoming_messages[-1] if incoming_messages else None
            )

            partner_id = (
                latest_dialogue_msg.from_agent_id
                if latest_dialogue_msg
                else agent.interaction_partner_id
            )
            partner = next((other for other in all_entities if other.id == partner_id), None)

            if partner:
                agent.memory.update_entity_perception(
                    entity_id=partner.id,
                    name=partner.name,
                    pos=partner.position,
                    tick=current_tick,
                )

            if partner_id:
                recent_formatted = self._dialogue_history.get_recent_formatted_for_pair(
                    agent.id, partner_id, limit=self._max_context_dialogues
                )
                recent_structured = [
                    r.to_dict()
                    for r in self._dialogue_history.get_recent_records_for_pair(
                        agent.id, partner_id, limit=self._max_context_dialogues
                    )
                ]
                conversation_summary = self._session_manager.get_summary(agent.id, partner_id)
            else:
                recent_formatted = self._dialogue_history.get_recent_formatted(limit=self._max_context_dialogues)
                recent_structured = self._dialogue_history.get_recent_structured(limit=self._max_context_dialogues)
                conversation_summary = ""

            turn_count = (
                self._session_manager.increment_turn(agent.id, partner_id)
                if partner_id
                else 0
            )

            partner_goal = partner.active_goal if isinstance(partner, Agent) else None
            partner_is_yielding = (
                isinstance(partner, Agent)
                and (
                    partner.lifecycle_state == AgentLifecycleState.YIELDING
                    or partner.is_evasion_locked
                    or (partner_goal is not None and partner_goal.yield_for_agent_id == agent.id)
                )
            )
            agent_goal = agent.active_goal
            agent_is_yielding = (
                agent.lifecycle_state == AgentLifecycleState.YIELDING
                or agent.is_evasion_locked
                or (agent_goal is not None and agent_goal.yield_for_agent_id == partner_id)
            )

            agent.set_thinking(True, reason=f"Führt Dialog mit {partner.name if partner else 'Partner'}")
            latest_intent = latest_dialogue_msg.intent if latest_dialogue_msg else None

            if partner_id:
                self._affect_service.record_turn(agent.id, partner_id, incoming_intent=latest_intent)
                situational_notes = self._affect_service.generate_situational_notes(
                    agent_id=agent.id,
                    partner_id=partner_id,
                    assertiveness=agent.assertiveness,
                )
            else:
                situational_notes = []

            if agent_is_yielding and latest_intent == "accept":
                await self._synthesize_and_store_social_reflection(
                    agent, partner, incident_id, current_tick, recent_formatted
                )
                if partner_id:
                    for cmd, fut in list(agent.interaction_mailbox):
                        if cmd.source_entity_id == partner_id and not fut.done():
                            fut.set_result(ImmediateResult(success=True, reason="DIALOGUE_ENDED"))
                    agent.interaction_mailbox = [
                        (c, f) for c, f in agent.interaction_mailbox if not f.done()
                    ]
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
            social_memories = self._retrieve_social_memories(agent.id, partner_id)
            context = {
                "agent_id": agent.id,
                "name": agent.name,
                "assertiveness": agent.assertiveness,
                "charisma": agent.charisma,
                "partner_id": partner.id if partner else None,
                "partner_name": partner.name if partner else "Unbekannt",
                "partner_is_conversational": partner.is_conversational if partner else False,
                "current_x": agent.position.x,
                "current_y": agent.position.y,
                "received_messages": received_dicts,
                "conversation_summary": conversation_summary,
                "recent_dialogues": recent_formatted,
                "recent_dialogues_structured": recent_structured,
                "active_goal": active_goal.to_dict() if active_goal else None,
                "goal_stack": [g.to_dict() for g in agent.goals],
                "dialogue_turn_count": turn_count,
                "evasion_distance_self": len_self,
                "evasion_distance_partner": len_partner,
                "recommended_role": recommended_role,
                "incoming_intent": latest_intent,
                "peer_bid_farewell": False,
                "partner_is_yielding": partner_is_yielding,
                "agent_is_yielding": agent_is_yielding,
                "social_memories": social_memories,
                "episodic_memories": social_memories,
                "situational_notes": situational_notes,
            }

            # Autonome Kognitionsentscheidung (kein Zwangsschlichter)
            resolution = await self._cognition_provider.respond_to_dialogue(context)

            chosen_intent = resolution.negotiation_intent or (
                resolution.action.intent if isinstance(resolution.action, TalkAction) else None
            )
            if isinstance(resolution.action, TalkAction) and chosen_intent:
                resolution.action.intent = cast(Any, chosen_intent)

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

            wants_to_evade = (
                (latest_intent == "request_yield" and chosen_intent == "accept")
                or chosen_intent == "offer_yield"
                or (resolution.new_goal is not None and resolution.new_goal.intent_type == "evade")
            )

            # Rejection-Verarbeitung
            if (chosen_intent == "reject" or latest_intent == "reject") and not wants_to_evade:
                self._action_executor.execute_dialogue_action(
                    agent=agent,
                    partner=partner,
                    action=resolution.action,
                    incident_id=incident_id,
                    all_entities=all_entities,
                )
                if partner and isinstance(partner, Agent):
                    partner.is_waiting_for_reply = False
                    partner.interaction_partner_id = None
                    if partner.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER:
                        partner.transition_to(AgentLifecycleState.IDLE)
                    for p_cmd, p_fut in list(partner.interaction_mailbox):
                        if p_cmd.source_entity_id == agent.id and not p_fut.done():
                            p_fut.set_result(
                                ImmediateResult(
                                    success=False,
                                    reason="REJECTED",
                                    payload={"intent": chosen_intent},
                                )
                            )
                    partner.interaction_mailbox = [
                        (c, f) for c, f in partner.interaction_mailbox if not f.done()
                    ]
                agent.is_waiting_for_reply = False
                agent.interaction_partner_id = None
                if agent.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER:
                    agent.transition_to(AgentLifecycleState.IDLE)
                for a_cmd, a_fut in list(agent.interaction_mailbox):
                    if a_cmd.source_entity_id == (partner.id if partner else None) and not a_fut.done():
                        a_fut.set_result(
                            ImmediateResult(
                                success=False,
                                reason="REJECTED",
                                payload={"intent": chosen_intent},
                            )
                        )
                agent.interaction_mailbox = [
                    (c, f) for c, f in agent.interaction_mailbox if not f.done()
                ]
                return

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
                await self._synthesize_and_store_social_reflection(
                    agent, partner, incident_id, current_tick, recent_formatted
                )
                if partner_id:
                    self._session_manager.reset_session(agent.id, partner_id)
                agent.is_waiting_for_reply = False
                if agent.interaction_partner_id == partner_id:
                    agent.interaction_partner_id = None
                if agent.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER:
                    agent.transition_to(AgentLifecycleState.IDLE)
                return

            if isinstance(resolution.action, EndDialogueAction) and partner_id:
                self._session_manager.reset_session(agent.id, partner_id)

            self._action_executor.execute_dialogue_action(
                agent=agent,
                partner=partner,
                action=resolution.action,
                incident_id=incident_id,
                all_entities=all_entities,
            )

            if partner_id:
                for cmd, fut in list(agent.interaction_mailbox):
                    if cmd.source_entity_id == partner_id and not fut.done():
                        fut.set_result(
                            ImmediateResult(
                                success=True,
                                reason="TALK_REPLY" if isinstance(resolution.action, TalkAction) else "DIALOGUE_ENDED",
                                payload={
                                    "message": getattr(
                                        resolution.action,
                                        "message",
                                        getattr(resolution.action, "final_message", ""),
                                    ),
                                    "intent": chosen_intent,
                                },
                            )
                        )
                agent.interaction_mailbox = [
                    (c, f) for c, f in agent.interaction_mailbox if not f.done()
                ]

                is_dialogue_ending = isinstance(resolution.action, EndDialogueAction)
                is_agreement_reached = (
                    (latest_intent == "offer_yield" and chosen_intent == "accept")
                    or (partner_is_yielding and chosen_intent == "accept")
                )

                if is_dialogue_ending or is_agreement_reached:
                    await self._synthesize_and_store_social_reflection(
                        agent, partner, incident_id, current_tick, recent_formatted
                    )

                if is_agreement_reached:
                    if partner_is_yielding or latest_intent == "offer_yield":
                        agent.transition_to(AgentLifecycleState.PASSING, reason="Passiere Korridor")
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
                            summary=f"Agent {agent.name} schließt Dialog ab: Einigung mit {partner_name} erzielt.",
                            payload={"incident_id": incident_id, "partner_id": partner_id},
                        )
                    )
                    return

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
                    agent.transition_to(
                        AgentLifecycleState.YIELDING, reason="In Nische ausweichen"
                    )
                return

            if isinstance(resolution.action, EndDialogueAction):
                agent.is_waiting_for_reply = False
                if agent.interaction_partner_id == partner_id:
                    agent.interaction_partner_id = None
                if agent.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER:
                    agent.transition_to(AgentLifecycleState.IDLE)
                if partner and isinstance(partner, Agent):
                    partner.is_waiting_for_reply = False
                    if partner.interaction_partner_id == agent.id:
                        partner.interaction_partner_id = None
                    if partner.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER:
                        partner.transition_to(AgentLifecycleState.IDLE)
                    for p_cmd, p_fut in list(partner.interaction_mailbox):
                        if p_cmd.source_entity_id == agent.id and not p_fut.done():
                            p_fut.set_result(
                                ImmediateResult(
                                    success=True,
                                    reason="DIALOGUE_ENDED",
                                    payload={"intent": chosen_intent},
                                )
                            )
                    partner.interaction_mailbox = [
                        (c, f) for c, f in partner.interaction_mailbox if not f.done()
                    ]
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
            agent.set_thinking(False)