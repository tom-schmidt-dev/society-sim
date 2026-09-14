from __future__ import annotations

import asyncio
import time
import uuid
from typing import Callable, Optional
from src.domain.models.agent import Agent
from src.domain.models.cognition import (
    AbortAction,
    InspectAction,
    RerouteAction,
    WaitAction,
    BlockedResolution,
    DialogueResolution,
    EndDialogueAction,
    TalkAction,
    GoalIntent,
)
from src.domain.models.events import SimulationEvent
from src.domain.models.goal import Goal
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.services.evasion_finder import EvasionFinder
from src.application.services.goal_service import GoalService


class ConflictCoordinator:
    def __init__(
            self,
            grid: WorldGrid,
            logger: IEventLogger,
            cognition_provider: ICognitionProvider,
            pathfinder: IPathfinder,
            goal_service: GoalService,
            evasion_finder: EvasionFinder,
            tick_provider: Optional[Callable[[], int]] = None,
            max_dialogue_turns: int = 2,
    ) -> None:
        self._grid: WorldGrid = grid
        self._logger: IEventLogger = logger
        self._cognition_provider: ICognitionProvider = cognition_provider
        self._pathfinder: IPathfinder = pathfinder
        self._goal_service: GoalService = goal_service
        self._evasion_finder: EvasionFinder = evasion_finder
        self._tick_provider: Callable[[], int] = tick_provider or (lambda: 0)
        self._max_dialogue_turns: int = max_dialogue_turns

        self._dialogues: list[str] = []
        self._conversation_locks: dict[frozenset[str], asyncio.Lock] = {}
        self._dialogue_turns: dict[frozenset[str], int] = {}

    @property
    def dialogues(self) -> list[str]:
        return self._dialogues

    async def resolve_blockage(
            self,
            agent: Agent,
            blocker: WorldEntity,
            blocked_pos: Position,
            grid: WorldGrid,
            all_agents: list[WorldEntity],
    ) -> None:
        pair_key = frozenset({agent.id, blocker.id})
        if pair_key not in self._conversation_locks:
            self._conversation_locks[pair_key] = asyncio.Lock()
        lock = self._conversation_locks[pair_key]

        incident_id = f"inc-t{self._tick_provider()}-{agent.id}x{blocker.id}-{uuid.uuid4().hex[:6]}"

        try:
            async with lock:
                if not agent.has_path or agent.path[0] != blocked_pos or blocker.position != blocked_pos:
                    return

                listening_goal = None
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
                            tick=self._tick_provider(),
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

                    agent.memory.update_entity_perception(blocker.id, blocker.name, blocker.position)

                    received = list(agent.inbox)
                    agent.inbox.clear()

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

                    # Ermittlung von allow_talk rein auf Basis des subjektiven Gedächtnisses
                    assumed_conv = agent.memory.get_assumed_conversational(blocker.id)
                    allow_talk = assumed_conv is not False

                    fact = agent.memory.known_entities.get(blocker.id)
                    inspected = fact.inspected if fact else False
                    known_type = fact.entity_type if fact else None

                    context = {
                        "agent_id": agent.id,
                        "name": agent.name,
                        "blocker_id": blocker.id,
                        "blocker_name": blocker.name,
                        "blocker_inspected": inspected,
                        "blocker_entity_type": known_type or "unbekannt",
                        "blocker_memory_status": agent.memory.get_fact_text(blocker.id),
                        "allow_talk": allow_talk,
                        "current_x": agent.position.x,
                        "current_y": agent.position.y,
                        "blocked_x": blocked_pos.x,
                        "blocked_y": blocked_pos.y,
                        "energy": agent.energy,
                        "goal_stack": stack_repr,
                        "active_goal": agent.active_goal.to_dict() if agent.active_goal else None,
                        "can_reroute": bool(test_path),
                        "received_messages": received,
                        "recent_dialogues": self._dialogues[-6:],
                    }

                    self._logger.log(
                        SimulationEvent(
                            tick=self._tick_provider(),
                            agent_id=agent.id,
                            event_type="cognition_started",
                            summary=f"Agent {agent.name} startet Inferenz zur Blockadelösung mit {blocker.name}.",
                            payload={
                                "incident_id": incident_id,
                                "context_snapshot": {
                                    "blocked_pos": {"x": blocked_pos.x, "y": blocked_pos.y},
                                    "can_reroute": bool(test_path),
                                    "allow_talk": allow_talk,
                                    "inspected": inspected,
                                },
                            },
                        )
                    )

                    start_time = time.perf_counter()
                    resolution: BlockedResolution = await self._cognition_provider.resolve_blockage(context)

                    if isinstance(resolution.action, (TalkAction, InspectAction)):
                        if resolution.action.target_agent_id == agent.id or resolution.action.target_agent_id not in (
                                blocker.id, blocker.name):
                            resolution.action.target_agent_id = blocker.id

                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

                    if resolution.complete_sub_goal and len(agent.goals) > 1:
                        self._goal_service.pop_goal(agent, incident_id=incident_id)

                    if resolution.new_sub_goal:
                        occupied = {other.position for other in all_agents if other.id != agent.id}
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
                            evasion_path = self._pathfinder.find_path(agent.position, evasion_target, agent.mental_map)
                            if evasion_path:
                                agent.path = evasion_path

                    self._apply_resolution(
                        agent, blocker, blocked_pos, grid, all_agents, resolution, incident_id, duration_ms
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

    def _apply_resolution(
            self,
            agent: Agent,
            blocker: WorldEntity,
            blocked_pos: Position,
            grid: WorldGrid,
            all_agents: list[WorldEntity],
            resolution: BlockedResolution,
            incident_id: str,
            duration_ms: float,
    ) -> None:
        action = resolution.action
        current_tick = self._tick_provider()

        if isinstance(action, InspectAction):
            target_entity = next(
                (
                    other for other in all_agents
                    if other.id == action.target_agent_id
                       or other.name.lower() == action.target_agent_id.lower()
                       or f"agent {other.name.lower()}" == action.target_agent_id.lower()
                ),
                None,
            )
            if target_entity:
                agent.memory.record_inspection(target_entity.id, target_entity.entity_type)
                info_msg = f"Inspektion: {target_entity.name} ist vom Typ '{target_entity.entity_type}'."
                self._dialogues.append(
                    f"{agent.name} -> {target_entity.name}: [Inspiziert: Typ={target_entity.entity_type}]")

                # Falls der Typ als unpassierbar bekannt ist, Kachel in MentalMap sperren
                if agent.memory.get_assumed_walkable(target_entity.id) is False:
                    agent.mental_map.mark_obstacle(target_entity.position, current_tick)

                self._logger.log(
                    SimulationEvent(
                        tick=current_tick,
                        agent_id=agent.id,
                        event_type="entity_inspected",
                        summary=f"Agent {agent.name} inspiziert {target_entity.name}: Typ='{target_entity.entity_type}'.",
                        payload={
                            "incident_id": incident_id,
                            "target_id": target_entity.id,
                            "entity_type": target_entity.entity_type,
                            "reason": action.reason,
                        },
                    )
                )
                self._goal_service.push_goal(
                    agent,
                    Goal(
                        name=f"Auswertung von {target_entity.name}",
                        holds_position=True,
                        remaining_ticks=1,
                        description=info_msg,
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
            target_entity = next(
                (
                    other for other in all_agents
                    if other.id == action.target_agent_id
                       or other.name.lower() == action.target_agent_id.lower()
                       or f"agent {other.name.lower()}" == action.target_agent_id.lower()
                ),
                None,
            )

            if target_entity:
                self._dialogues.append(f"{agent.name} -> {target_entity.name}: '{action.message}'")
                self._logger.log(
                    SimulationEvent(
                        tick=current_tick,
                        agent_id=agent.id,
                        event_type="message_delivered",
                        summary=f"Agent {agent.name} sagt zu {target_entity.name}: '{action.message}'",
                        payload={
                            "incident_id": incident_id,
                            "sender_id": agent.id,
                            "recipient_id": target_entity.id,
                            "message": action.message,
                        },
                    )
                )

                if not target_entity.is_conversational:
                    agent.memory.record_interaction_result(target_entity.id, responded=False)
                    agent.memory.record_walkability_result(target_entity.id, is_walkable=False)
                    agent.mental_map.mark_obstacle(target_entity.position, current_tick)

                    target_entity.inbox.append({
                        "from_agent_id": agent.id,
                        "from_agent_name": agent.name,
                        "message": action.message,
                    })
                    agent.inbox.append({
                        "from_agent_id": target_entity.id,
                        "from_agent_name": target_entity.name,
                        "message": "",
                        "is_empty_response": True,
                    })
                    self._dialogues.append(f"{target_entity.name} -> {agent.name}: ''")
                    self._goal_service.push_goal(
                        agent,
                        Goal(
                            name=f"Warten auf Klärung ({target_entity.name})",
                            holds_position=True,
                            remaining_ticks=2,
                            description="Wartet auf Verarbeitung der Objektreaktion.",
                        ),
                        incident_id=incident_id,
                    )
                else:
                    target_entity.inbox.append({
                        "from_agent_id": agent.id,
                        "from_agent_name": agent.name,
                        "message": action.message,
                    })
                    self._goal_service.push_goal(
                        target_entity,
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
            target = (
                active_goal.target_position
                if active_goal and active_goal.target_position
                else (agent.path[-1] if agent.has_path else None)
            )
            if target:
                agent.mental_map.mark_obstacle(blocked_pos, current_tick)
                new_path = self._pathfinder.find_path(agent.position, target, agent.mental_map)
                if new_path:
                    agent.path = new_path
        elif isinstance(action, AbortAction):
            agent.path.clear()
            if agent.goals:
                agent.goals[-1].status = "abandoned"

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
                    "internal_thought": resolution.thought,
                    "action_details": action.model_dump(),
                },
            )
        )

    async def handle_incoming_dialogue(
            self,
            agent: Agent,
            grid: WorldGrid,
            all_entities: list[WorldEntity],
    ) -> None:
        incident_id = f"dlg-t{self._tick_provider()}-{agent.id}-{uuid.uuid4().hex[:6]}"
        received = list(agent.inbox)
        agent.inbox.clear()

        partner_id = received[-1].get("from_agent_id") if received else None
        partner = next((other for other in all_entities if other.id == partner_id), None)

        if partner:
            agent.memory.update_entity_perception(partner.id, partner.name, partner.position)
            if partner.is_conversational:
                agent.memory.record_interaction_result(partner.id, responded=True)
            else:
                agent.memory.record_interaction_result(partner.id, responded=False)
                agent.memory.record_walkability_result(partner.id, is_walkable=False)
                agent.mental_map.mark_obstacle(partner.position, self._tick_provider())

        pair_key = frozenset({agent.id, partner.id}) if partner else frozenset({agent.id})
        turn_count = self._dialogue_turns.get(pair_key, 0) + 1
        self._dialogue_turns[pair_key] = turn_count

        context = {
            "agent_id": agent.id,
            "name": agent.name,
            "partner_id": partner.id if partner else None,
            "partner_name": partner.name if partner else "Unbekannt",
            "partner_is_conversational": partner.is_conversational if partner else False,
            "current_x": agent.position.x,
            "current_y": agent.position.y,
            "received_messages": received,
            "recent_dialogues": self._dialogues[-8:],
            "active_goal": agent.active_goal.to_dict() if agent.active_goal else None,
            "goal_stack": [g.to_dict() for g in agent.goals],
            "dialogue_turn_count": turn_count,
            "max_dialogue_turns": self._max_dialogue_turns,
        }

        try:
            if turn_count > self._max_dialogue_turns:
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
                self._dialogue_turns.pop(pair_key, None)
                final_msg = resolution.action.final_message or f"Ich beende das Gespräch: {resolution.action.reason}"

                self._dialogues.append(f"{agent.name} -> {partner.name if partner else 'Raum'}: '{final_msg}'")
                self._logger.log(
                    SimulationEvent(
                        tick=self._tick_provider(),
                        agent_id=agent.id,
                        event_type="message_delivered",
                        summary=f"Agent {agent.name} verabschiedet sich von {partner.name if partner else 'Unbekannt'}: '{final_msg}'",
                        payload={
                            "incident_id": incident_id,
                            "sender_id": agent.id,
                            "recipient_id": partner.id if partner else None,
                            "message": final_msg,
                            "reason": resolution.action.reason,
                        },
                    )
                )

                if partner and partner.is_conversational:
                    partner.inbox.append({
                        "from_agent_id": agent.id,
                        "from_agent_name": agent.name,
                        "message": final_msg,
                        "is_farewell": True,
                    })
                    if isinstance(partner, Agent) and partner.active_goal and (
                            "Konversation" in partner.active_goal.name or "Warten" in partner.active_goal.name
                    ):
                        self._goal_service.pop_goal(partner, target_goal=partner.active_goal, incident_id=incident_id)

            elif isinstance(resolution.action, TalkAction):
                target_entity = next(
                    (
                        other for other in all_entities
                        if other.id == resolution.action.target_agent_id
                           or other.name.lower() == resolution.action.target_agent_id.lower()
                    ),
                    partner,
                )
                target_name = target_entity.name if target_entity else resolution.action.target_agent_id
                self._dialogues.append(f"{agent.name} -> {target_name}: '{resolution.action.message}'")

                if target_entity:
                    if not target_entity.is_conversational:
                        agent.memory.record_interaction_result(target_entity.id, responded=False)
                        agent.memory.record_walkability_result(target_entity.id, is_walkable=False)
                        agent.mental_map.mark_obstacle(target_entity.position, self._tick_provider())
                        target_entity.inbox.append({
                            "from_agent_id": agent.id,
                            "from_agent_name": agent.name,
                            "message": resolution.action.message,
                        })
                        agent.inbox.append({
                            "from_agent_id": target_entity.id,
                            "from_agent_name": target_entity.name,
                            "message": "",
                            "is_empty_response": True,
                        })
                        self._dialogues.append(f"{target_entity.name} -> {agent.name}: ''")
                        self._goal_service.push_goal(
                            agent,
                            Goal(
                                name=f"Warten auf Klärung ({target_entity.name})",
                                holds_position=True,
                                remaining_ticks=2,
                                description="Wartet auf Verarbeitung der Objektreaktion.",
                            ),
                            incident_id=incident_id,
                        )
                    else:
                        target_entity.inbox.append({
                            "from_agent_id": agent.id,
                            "from_agent_name": agent.name,
                            "message": resolution.action.message,
                        })
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

            if resolution.new_goal:
                intent = resolution.new_goal
                if intent.intent_type == "evade":
                    occupied = {other.position for other in all_entities if other.id != agent.id}
                    evasion_target = self._evasion_finder.find_nearest_evasion_tile(
                        agent.position, partner.position if partner else agent.position, agent.mental_map, occupied
                    )
                    if evasion_target:
                        self._goal_service.push_goal(
                            agent,
                            Goal(name=intent.name, target_position=evasion_target, description=resolution.thought),
                            incident_id=incident_id,
                        )
                        path = self._pathfinder.find_path(agent.position, evasion_target, agent.mental_map)
                        if path:
                            agent.path = path
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