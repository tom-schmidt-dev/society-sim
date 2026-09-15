from __future__ import annotations

import uuid
from typing import Callable, Optional
from src.application.services.action_executor import ActionExecutor
from src.application.services.dialogue_history import DialogueHistory
from src.application.services.dialogue_session_manager import DialogueSessionManager
from src.application.services.goal_service import GoalService
from src.domain.models.agent import Agent
from src.domain.models.cognition import (
    DialogueResolution,
    EndDialogueAction,
    GoalIntent,
)
from src.domain.models.events import SimulationEvent
from src.domain.models.goal import Goal
from src.domain.models.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.dialogue_coordinator import IDialogueCoordinator
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.application.services.evasion_finder import EvasionFinder


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
            "active_goal": agent.active_goal.to_dict() if agent.active_goal else None,
            "goal_stack": [g.to_dict() for g in agent.goals],
            "dialogue_turn_count": turn_count,
            "max_dialogue_turns": self._session_manager.max_dialogue_turns,
        }

        try:
            if self._session_manager.is_turn_limit_exceeded(agent.id, partner_id):
                resolution = DialogueResolution(
                    thought="Maximale Gesprächsrunden erreicht. Ich beende das Gespräch und weiche aus.",
                    action=EndDialogueAction(
                        reason="Maximale Rundenanzahl überschritten.",
                        final_message="Wir kommen hier zu keiner Einigung. Ich beende das Gespräch und suche einen Ausweg.",
                    ),
                    new_goal=GoalIntent(name="In Nische ausweichen", intent_type="evade"),
                )
            else:
                resolution = await self._cognition_provider.respond_to_dialogue(context)

            if agent.active_goal and ("Konversation" in agent.active_goal.name or "Warten" in agent.active_goal.name):
                self._goal_service.pop_goal(agent, target_goal=agent.active_goal, incident_id=incident_id)

            if isinstance(resolution.action, EndDialogueAction):
                self._session_manager.reset_session(agent.id, partner_id)

            self._action_executor.execute_dialogue_action(
                agent=agent,
                partner=partner,
                action=resolution.action,
                incident_id=incident_id,
                all_entities=all_entities,
            )

            if resolution.new_goal:
                intent = resolution.new_goal
                if intent.intent_type == "evade":
                    occupied = {other.position for other in all_entities if other.id != agent.id}
                    evasion_target = self._evasion_finder.find_nearest_evasion_tile(
                        agent.position,
                        partner.position if partner else agent.position,
                        agent.mental_map,
                        occupied,
                    )
                    if evasion_target:
                        self._goal_service.push_goal(
                            agent,
                            Goal(name=intent.name, target_position=evasion_target, description=resolution.thought),
                            incident_id=incident_id,
                        )
                        path = self._pathfinder.find_path(agent.position, evasion_target, agent.mental_map)
                        if path:
                            agent.assign_path(path)
                elif intent.intent_type == "hold":
                    self._goal_service.push_goal(
                        agent,
                        Goal(name=intent.name, holds_position=True, remaining_ticks=3, description=resolution.thought),
                        incident_id=incident_id,
                    )

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