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
            async with lock:
                if not agent.has_path or agent.path[0] != blocked_pos or blocker.position != blocked_pos:
                    return

                listening_goal: Optional[Goal] = None
                if isinstance(blocker, Agent):
                    listening_goal = Goal(
                        name=f"Lauscht Agent {agent.name}",
                        target_position=blocker.position,
                        holds_position=True,
                        description="Fixiert die Position während der Inferenz des Partners.",
                    )
                    self._goal_service.push_goal(blocker, listening_goal, incident_id=incident_id)

                try:
                    self._logger.log(
                        SimulationEvent(
                            tick=current_tick,
                            agent_id=agent.id,
                            event_type="movement_blocked",
                            summary=f"Agent {agent.name} wurde bei Bewegung nach ({blocked_pos.x}, {blocked_pos.y}) von {blocker.name} blockiert.",
                            payload={
                                "incident_id": incident_id,
                                "attempted_pos": {"x": blocked_pos.x, "y": blocked_pos.y},
                                "blocked_by_agent_id": blocker.id,
                            },
                        )
                    )

                    agent.memory.update_entity_perception(
                        entity_id=blocker.id,
                        name=blocker.name,
                        pos=blocker.position,
                        tick=current_tick,
                    )

                    received = [m.to_dict() for m in agent.drain_inbox(assimilate=True, tick=current_tick)]

                    active_goal = agent.active_goal
                    target_pos = (
                        active_goal.target_position
                        if active_goal and active_goal.target_position
                        else (agent.path[-1] if agent.has_path else None)
                    )

                    test_path: list[Position] = []
                    if target_pos:
                        test_path = self._pathfinder.find_path(agent.position, target_pos, agent.mental_map)

                    stack_repr = [
                        f"Ebene {idx}: {g.name} (Status: {g.status}, Koordinate: {g.target_position})"
                        for idx, g in enumerate(agent.goals)
                    ]

                    allow_talk = agent.memory.can_talk(blocker.id)
                    allow_probe = agent.memory.can_probe(blocker.id)
                    inspected = agent.memory.is_inspected(blocker.id)

                    fact = agent.memory.known_entities.get(blocker.id)
                    known_type = fact.entity_type if fact else None
                    belief = agent.memory.type_beliefs.get(known_type) if known_type else None

                    individual_walkable = agent.memory.get_entity_walkability(blocker.id)
                    if individual_walkable is False:
                        walkability_text = (
                            f"Dieses konkrete Objekt wurde bereits erprobt und ist UNPASSIERBAR. "
                            f"(Kategorie '{known_type}': {belief.empirical_walkability if belief else 'Keine Daten'})"
                        )
                    elif individual_walkable is True:
                        walkability_text = (
                            f"Dieses konkrete Objekt wurde bereits erprobt und ist PASSIERBAR. "
                            f"(Kategorie '{known_type}': {belief.empirical_walkability if belief else 'Keine Daten'})"
                        )
                    else:
                        walkability_text = (
                            belief.empirical_walkability
                            if belief
                            else "Bisher keine Erfahrungswerte zu diesem Typ"
                        )

                    context = {
                        "agent_id": agent.id,
                        "name": agent.name,
                        "blocker_id": blocker.id,
                        "blocker_name": blocker.name,
                        "blocker_inspected": inspected,
                        "blocker_entity_type": known_type or "unbekannt",
                        "blocker_memory_status": agent.memory.get_fact_text(blocker.id),
                        "empirical_walkability": walkability_text,
                        "empirical_conversational": (
                            belief.empirical_conversational
                            if belief
                            else "Bisher keine Erfahrungswerte zu diesem Typ"
                        ),
                        "allow_talk": allow_talk,
                        "allow_probe": allow_probe,
                        "current_x": agent.position.x,
                        "current_y": agent.position.y,
                        "blocked_x": blocked_pos.x,
                        "blocked_y": blocked_pos.y,
                        "energy": agent.energy,
                        "goal_stack": stack_repr,
                        "active_goal": agent.active_goal.to_dict() if agent.active_goal else None,
                        "can_reroute": bool(test_path),
                        "received_messages": received,
                        "recent_dialogues": self._dialogue_history.get_recent_formatted(limit=6),
                    }

                    self._logger.log(
                        SimulationEvent(
                            tick=current_tick,
                            agent_id=agent.id,
                            event_type="cognition_started",
                            summary=f"Agent {agent.name} startet Inferenz zur Blockadelösung mit {blocker.name}.",
                            payload={
                                "incident_id": incident_id,
                                "context_snapshot": {
                                    "blocked_pos": {"x": blocked_pos.x, "y": blocked_pos.y},
                                    "can_reroute": bool(test_path),
                                    "allow_talk": allow_talk,
                                    "allow_probe": allow_probe,
                                    "inspected": inspected,
                                },
                            },
                        )
                    )

                    start_time = time.perf_counter()
                    resolution: BlockedResolution = await self._cognition_provider.resolve_blockage(context)

                    if isinstance(resolution.action, (TalkAction, InspectAction, ProbeAction)):
                        if resolution.action.target_agent_id == agent.id or resolution.action.target_agent_id not in (
                            blocker.id,
                            blocker.name,
                        ):
                            resolution.action.target_agent_id = blocker.id

                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

                    if resolution.complete_sub_goal and len(agent.goals) > 1:
                        self._goal_service.pop_goal(agent, incident_id=incident_id)

                    if resolution.new_sub_goal:
                        occupied = {other.position for other in all_entities if other.id != agent.id}
                        evasion_target = self._evasion_finder.find_nearest_evasion_tile(
                            agent.position, blocked_pos, agent.mental_map, occupied
                        )
                        self._goal_service.push_goal(
                            agent,
                            Goal(
                                name=resolution.new_sub_goal,
                                target_position=evasion_target,
                                description=resolution.thought,
                            ),
                            incident_id=incident_id,
                        )
                        if evasion_target:
                            evasion_path = self._pathfinder.find_path(
                                agent.position, evasion_target, agent.mental_map
                            )
                            if evasion_path:
                                agent.assign_path(evasion_path)

                    self._action_executor.execute_blockage_action(
                        agent=agent,
                        blocker=blocker,
                        blocked_pos=blocked_pos,
                        action=resolution.action,
                        incident_id=incident_id,
                        duration_ms=duration_ms,
                        thought=resolution.thought,
                        all_entities=all_entities,
                    )

                except Exception as err:
                    self._goal_service.push_goal(
                        agent,
                        Goal(
                            name="Warten (Fallback)",
                            holds_position=True,
                            remaining_ticks=2,
                            description="Kognitions-Fallback nach Fehler.",
                        ),
                        incident_id=incident_id,
                    )
                    self._logger.log(
                        SimulationEvent(
                            tick=self._tick_provider(),
                            agent_id=agent.id,
                            event_type="cognition_failed",
                            summary=f"Kognitionsfehler bei Agent {agent.name}: {err}. Fallback: Warten.",
                            payload={"incident_id": incident_id, "error": str(err)},
                        )
                    )
                finally:
                    if isinstance(blocker, Agent) and listening_goal:
                        self._goal_service.pop_goal(blocker, target_goal=listening_goal, incident_id=incident_id)
        finally:
            agent.is_thinking = False