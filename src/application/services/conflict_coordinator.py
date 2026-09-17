from __future__ import annotations

import time
import uuid
from typing import Callable, Optional
from src.application.services.action_executor import ActionExecutor
from src.application.services.dialogue_history import DialogueHistory
from src.application.services.dialogue_session_manager import DialogueSessionManager
from src.application.services.evasion_finder import EvasionFinder
from src.application.services.goal_service import GoalService
from src.domain.models.agent import Agent
from src.domain.models.cognition import (
    BlockedResolution,
    InspectAction,
    ProbeAction,
    TalkAction,
)
from src.domain.models.events import SimulationEvent
from src.domain.models.goal import Goal
from src.domain.models.position import Position
from src.domain.models.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.conflict_coordinator import IConflictCoordinator
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder


class ConflictCoordinator(IConflictCoordinator):
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

        # In src/application/services/conflict_coordinator.py
        # Ersetzung der Methode: resolve_blockage

    async def resolve_blockage(
            self,
            agent: Agent,
            blocker: WorldEntity,
            blocked_pos: Position,
            all_entities: list[WorldEntity],
    ) -> None:
        lock = self._session_manager.get_lock(agent.id, blocker.id)

        current_tick = self._tick_provider()
        incident_id = f"inc-t{current_tick}-{agent.id}x{blocker.id}-{uuid.uuid4().hex[:6]}"

        try:
            async with self._session_manager.acquire_session(agent.id, blocker.id, tick=current_tick) as lock_granted:
                if not lock_granted:
                    agent.is_thinking = False
                    return

                # 1. Schnelle Vorabprüfung
                if not agent.has_path or agent.path[0] != blocked_pos or blocker.position != blocked_pos:
                    return

                if agent.inbox and any(m.from_agent_id == blocker.id for m in agent.inbox):
                    return

                is_busy_with_third_party = (
                        isinstance(blocker, Agent)
                        and blocker.interaction_partner_id is not None
                        and blocker.interaction_partner_id != agent.id
                )

                if is_busy_with_third_party and isinstance(blocker, Agent):
                    if not any(req.requester_id == agent.id for req in blocker.interaction_queue):
                        from src.domain.models.interaction_request import InteractionRequest
                        blocker.interaction_queue.append(
                            InteractionRequest(
                                requester_id=agent.id,
                                target_id=blocker.id,
                                blocked_pos=blocked_pos,
                                tick=current_tick,
                            )
                        )
                        agent.is_waiting_for_reply = True
                        agent.interaction_partner_id = blocker.id
                        self._logger.log(
                            SimulationEvent(
                                tick=current_tick,
                                agent_id=agent.id,
                                event_type="interaction_queued",
                                summary=f"Agent {agent.name} wartet auf {blocker.name} (in Warteschlange eingereiht).",
                                payload={"target_id": blocker.id, "queue_length": len(blocker.interaction_queue)},
                            )
                        )
                    return

                if isinstance(blocker, Agent):
                    blocker_goal = blocker.active_goal
                    if (
                        blocker.is_evasion_locked
                        or (blocker_goal is not None and blocker_goal.yield_for_agent_id == agent.id)
                    ):
                        return

                if isinstance(blocker, Agent):
                    blocker.is_listening_to_peer = True
                    blocker.interaction_partner_id = agent.id

                try:
                    agent.memory.update_entity_perception(
                        entity_id=blocker.id,
                        name=blocker.name,
                        pos=blocker.position,
                        tick=current_tick,
                    )

                    allow_talk = agent.memory.can_talk(blocker.id)
                    allow_probe = agent.memory.can_probe(blocker.id)
                    inspected = agent.memory.is_inspected(blocker.id)

                    fact = agent.memory.known_entities.get(blocker.id)
                    known_type = fact.entity_type if fact else None
                    belief = agent.memory.type_beliefs.get(known_type) if known_type else None
                    empirical_summary = (
                        f"Kategorie '{known_type}': {belief.empirical_walkability}; {belief.empirical_conversational}"
                        if belief
                        else "Keine empirischen Typ-Erfahrungen vorhanden."
                    )

                    current_goal = agent.active_goal
                    context = {
                        "agent_id": agent.id,
                        "name": agent.name,
                        "blocker_id": blocker.id,
                        "blocker_name": blocker.name,
                        "blocker_inspected": inspected,
                        "allow_talk": allow_talk,
                        "allow_probe": allow_probe,
                        "empirical_record": empirical_summary,
                        "current_pos": [agent.position.x, agent.position.y],
                        "blocked_pos": [blocked_pos.x, blocked_pos.y],
                        "active_goal": current_goal.name if current_goal else "Keines",
                    }

                    start_time = time.perf_counter()
                    resolution: BlockedResolution = await self._cognition_provider.resolve_blockage(context)
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

                    if resolution.complete_sub_goal and len(agent.goals) > 1:
                        self._goal_service.pop_goal(agent, incident_id=incident_id)

                    if resolution.new_sub_goal:
                        self._action_executor.execute_evasion(
                            agent=agent,
                            partner=blocker,
                            blocked_pos=blocked_pos,
                            all_entities=all_entities,
                            incident_id=incident_id,
                            thought=resolution.thought,
                            sub_goal_name=resolution.new_sub_goal,
                        )

                    # TOCTOU-gesicherte Ausführung
                    self._action_executor.execute_blockage_action(
                        agent=agent,
                        blocker=blocker,
                        action=resolution.action,
                        incident_id=incident_id,
                        all_entities=all_entities,
                        blocked_pos=blocked_pos,
                        current_tick=current_tick,
                        thought=resolution.thought,
                        duration_ms=duration_ms,
                    )

                finally:
                    if isinstance(blocker, Agent):
                        blocker.is_listening_to_peer = False
        finally:
            agent.is_thinking = False