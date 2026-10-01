from __future__ import annotations

import time
import uuid
from typing import Callable, Optional
from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.coordination.convoy_arbitrator import ArbitrationOutcome, ConvoyArbitrator
from src.application.services.coordination.convoy_coordinator import Convoy, ConvoyCoordinator
from src.application.services.coordination.dialogue_history import DialogueHistory
from src.application.services.coordination.dialogue_session_manager import DialogueSessionManager
from src.application.services.movement.evasion_finder import EvasionFinder
from src.application.services.cognition.goal_service import GoalService
from src.application.services.movement.multi_agent_niche_packer import MultiAgentNichePacker, NicheConfiguration
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.cognition import (
    BlockedResolution,
    InspectAction,
    ProbeAction,
    TalkAction,
)
from src.domain.models.communication.communication_templates import DialogueTemplates
from src.domain.models.planning.events import SimulationEvent
from src.domain.models.coordination.evasion_phase import EvasionPhase
from src.domain.models.planning.goal import ExecutionPriority, Goal
from src.domain.models.agent.mental_map import AgentMentalMap, FusedMentalMap
from src.domain.models.world.position import Position
from src.domain.models.world.world_entity import WorldEntity
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
        convoy_coordinator: Optional[ConvoyCoordinator] = None,
        niche_packer: Optional[MultiAgentNichePacker] = None,
        convoy_arbitrator: Optional[ConvoyArbitrator] = None,
        enable_deterministic_corridor: bool = True,
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
        self._convoy_coordinator = convoy_coordinator or ConvoyCoordinator(pathfinder=self._pathfinder, logger=self._logger)
        self._niche_packer = niche_packer or MultiAgentNichePacker()
        self._convoy_arbitrator = convoy_arbitrator or ConvoyArbitrator()
        self._enable_deterministic_corridor = enable_deterministic_corridor

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
                        and blocker.interaction_partner_id != agent.id
                        and (blocker.is_busy or blocker.interaction_partner_id is not None)
                )

                if is_busy_with_third_party and isinstance(blocker, Agent):
                    if not any(req.requester_id == agent.id for req in blocker.interaction_queue):
                        from src.domain.models.communication.interaction_request import InteractionRequest
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
                    if self._session_manager.was_negotiated_in_tick(agent.id, blocker.id, current_tick):
                        return

                    blocker_goal = blocker.active_goal
                    if (
                        blocker.is_evasion_locked
                        or (blocker_goal is not None and blocker_goal.yield_for_agent_id == agent.id)
                        or blocker.evasion_phase in (EvasionPhase.YIELDING_INGRESS, EvasionPhase.YIELDING_WAIT)
                    ):
                        return

                    agent_goal = agent.active_goal
                    if (
                        agent.is_evasion_locked
                        or (agent_goal is not None and agent_goal.yield_for_agent_id == blocker.id)
                        or agent.evasion_phase in (EvasionPhase.YIELDING_INGRESS, EvasionPhase.YIELDING_WAIT)
                    ):
                        return

                if isinstance(blocker, Agent) and agent.is_conversational and blocker.is_conversational:
                    if self._enable_deterministic_corridor and self._is_corridor_encounter(agent, blocker, blocked_pos):
                        await self._resolve_corridor_blockage_deterministically(
                            agent=agent,
                            blocker=blocker,
                            blocked_pos=blocked_pos,
                            all_entities=all_entities,
                            incident_id=incident_id,
                            current_tick=current_tick,
                        )
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

                    try:
                        start_time = time.perf_counter()
                        resolution: BlockedResolution = await self._cognition_provider.resolve_blockage(context)
                        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    except Exception as err:
                        self._logger.log(
                            SimulationEvent(
                                tick=current_tick,
                                agent_id=agent.id,
                                event_type="cognition_failed",
                                summary=f"Kognition fehlgeschlagen für {agent.name}: {err}",
                                payload={"incident_id": incident_id, "error": str(err)},
                            )
                        )
                        from src.domain.models.planning.goal import ExecutionPriority
                        fallback_goal = Goal(
                            name="Warten (Fallback)",
                            holds_position=True,
                            remaining_ticks=2,
                            priority=ExecutionPriority.URGENT,
                            description="Kognitionsfehler Fallback",
                        )
                        agent.push_goal(fallback_goal)
                        return

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

                    if isinstance(blocker, Agent):
                        self._session_manager.mark_negotiated(agent.id, blocker.id, current_tick)

                finally:
                    if isinstance(blocker, Agent):
                        blocker.is_listening_to_peer = False
        finally:
            agent.is_thinking = False

    def _is_corridor_encounter(
        self,
        agent: Agent,
        blocker: Agent,
        blocked_pos: Position,
    ) -> bool:
        """
        Prüft gemäß Details.md Punkt 3 & 5 und Integration.md, ob eine räumliche Begegnung
        im Korridor / Engpass vorliegt, deren freie Durchgangsbreite kleiner als die Summe
        der transversalen Ausdehnungen ist (Footprint-Summe = 2), oder ob Gegenverkehr
        (opposing traffic / collision course) zweier aktiver Agenten vorliegt.
        """
        if not (agent.is_conversational and blocker.is_conversational):
            return False

        dir_agent = self._convoy_coordinator.derive_direction(agent)
        dir_blocker = self._convoy_coordinator.derive_direction(blocker)

        # 1. Gegenverkehr (opposing traffic) oder direkter Kollisionskurs
        if dir_agent != (0, 0) and dir_blocker != (0, 0):
            if (dir_agent[0] * dir_blocker[0] < 0) or (dir_agent[1] * dir_blocker[1] < 0):
                return True

        if blocker.has_path and blocker.path and agent.position == blocker.path[0]:
            return True

        if blocker.has_path and agent.position in blocker.path and blocker.position in agent.path:
            return True

        # 2. Räumliche Korridorverengung (Wände/Hindernisse transversal)
        def is_obs(p: Position) -> bool:
            return (not agent.mental_map.is_walkable(p)) or (not blocker.mental_map.is_walkable(p))

        for pos in (agent.position, blocked_pos):
            if dir_agent[0] != 0:
                up = Position(pos.x, pos.y - 1)
                down = Position(pos.x, pos.y + 1)
                if is_obs(up) or is_obs(down):
                    return True
            elif dir_agent[1] != 0:
                left = Position(pos.x - 1, pos.y)
                right = Position(pos.x + 1, pos.y)
                if is_obs(left) or is_obs(right):
                    return True
            else:
                up = Position(pos.x, pos.y - 1)
                down = Position(pos.x, pos.y + 1)
                left = Position(pos.x - 1, pos.y)
                right = Position(pos.x + 1, pos.y)
                if (is_obs(up) or is_obs(down)) or (is_obs(left) or is_obs(right)):
                    return True

        return False

    async def _resolve_corridor_blockage_deterministically(
        self,
        agent: Agent,
        blocker: Agent,
        blocked_pos: Position,
        all_entities: list[WorldEntity],
        incident_id: str,
        current_tick: int,
    ) -> None:
        all_agents = [e for e in all_entities if isinstance(e, Agent)]
        convoys = self._convoy_coordinator.identify_convoys(all_agents)

        convoy_agent = next((c for c in convoys if agent in c.members), Convoy(id=f"convoy_{agent.id}", members=[agent]))
        convoy_blocker = next((c for c in convoys if blocker in c.members), Convoy(id=f"convoy_{blocker.id}", members=[blocker]))

        leader_agent = self._convoy_coordinator.select_leader(convoy_agent, blocked_pos)
        leader_blocker = self._convoy_coordinator.select_leader(convoy_blocker, agent.position)

        is_group = len(convoy_agent.members) > 1 or len(convoy_blocker.members) > 1

        # 1. Phase: CONFLICT_DETECTED
        for a in convoy_agent.members + convoy_blocker.members:
            a.evasion_phase = EvasionPhase.CONFLICT_DETECTED
            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=a.id,
                    event_type="evasion_phase_changed",
                    summary=f"Agent {a.name}: Status gewechselt zu {EvasionPhase.CONFLICT_DETECTED.value}.",
                    payload={"phase": EvasionPhase.CONFLICT_DETECTED.value, "incident_id": incident_id},
                )
            )

        conflict_prompt = DialogueTemplates.conflict_notice(is_group=is_group)
        self._dialogue_history.record_dialogue(
            tick=current_tick,
            sender_id=leader_agent.id,
            sender_name=leader_agent.name,
            recipient_id=leader_blocker.id,
            recipient_name=leader_blocker.name,
            message=conflict_prompt,
            intent="request_yield",
        )
        self._logger.log(
            SimulationEvent(
                tick=current_tick,
                agent_id=agent.id,
                event_type="corridor_conflict_detected",
                summary=f"Korridorkonflikt zwischen {agent.name} und {blocker.name} erkannt. Deterministische FSM gestartet.",
                payload={
                    "incident_id": incident_id,
                    "prompt": conflict_prompt,
                    "convoy_agent": [a.id for a in convoy_agent.members],
                    "convoy_blocker": [a.id for a in convoy_blocker.members],
                },
            )
        )

        # 2. Phase: NEGOTIATING
        for a in convoy_agent.members + convoy_blocker.members:
            a.evasion_phase = EvasionPhase.NEGOTIATING
            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=a.id,
                    event_type="evasion_phase_changed",
                    summary=f"Agent {a.name}: Status gewechselt zu {EvasionPhase.NEGOTIATING.value}.",
                    payload={"phase": EvasionPhase.NEGOTIATING.value, "incident_id": incident_id},
                )
            )

        involved_agents = convoy_agent.members + convoy_blocker.members
        involved_maps = [a.mental_map for a in involved_agents]
        dyn_positions = {a.position for a in all_agents}
        fused_map = FusedMentalMap.fuse(involved_maps, dynamic_positions=dyn_positions)
        self._logger.log(
            SimulationEvent(
                tick=current_tick,
                agent_id=agent.id,
                event_type="fused_mental_map_aggregated",
                summary=f"Temporäre FusedMentalMap für Konvois {convoy_agent.id} und {convoy_blocker.id} aggregiert.",
                payload={
                    "incident_id": incident_id,
                    "convoy_agent": convoy_agent.id,
                    "convoy_blocker": convoy_blocker.id,
                    "total_tiles": len(fused_map.tiles),
                },
            )
        )

        # 3. Trajektorien ermitteln
        traj_agent = [leader_agent.position] + list(leader_agent.path) if leader_agent.has_path else [leader_agent.position]
        if not leader_agent.has_path and leader_agent.active_goal and leader_agent.active_goal.target_position:
            p_agent = self._pathfinder.find_path(leader_agent.position, leader_agent.active_goal.target_position, fused_map)
            if p_agent:
                traj_agent = [leader_agent.position] + p_agent

        traj_blocker = [leader_blocker.position] + list(leader_blocker.path) if leader_blocker.has_path else [leader_blocker.position]
        if not leader_blocker.has_path and leader_blocker.active_goal and leader_blocker.active_goal.target_position:
            p_blocker = self._pathfinder.find_path(leader_blocker.position, leader_blocker.active_goal.target_position, fused_map)
            if p_blocker:
                traj_blocker = [leader_blocker.position] + p_blocker

        # 4. Nischen-Konfigurationen evaluieren
        niche_config_agent = self._niche_packer.find_niche_configuration(
            convoy=convoy_agent,
            mental_map=fused_map,
            passing_convoy_trajectory=traj_blocker,
        )
        niche_config_blocker = self._niche_packer.find_niche_configuration(
            convoy=convoy_blocker,
            mental_map=fused_map,
            passing_convoy_trajectory=traj_agent,
        )

        delta_c_agent = niche_config_agent.additional_cost if niche_config_agent else None
        delta_c_blocker = niche_config_blocker.additional_cost if niche_config_blocker else None

        # 5. Arbitrierung via ConvoyArbitrator (70/30-Nutzenfunktion & SHA-256-Münzwurf)
        arbitration = self._convoy_arbitrator.arbitrate(
            group_1_agents=convoy_agent.members,
            group_2_agents=convoy_blocker.members,
            delta_c_1=delta_c_agent,
            delta_c_2=delta_c_blocker,
            tick=current_tick,
            leader_1_id=leader_agent.id,
            leader_2_id=leader_blocker.id,
        )

        self._logger.log(
            SimulationEvent(
                tick=current_tick,
                agent_id=agent.id,
                event_type="corridor_arbitration_completed",
                summary=f"Arbitrierung abgeschlossen: {arbitration.outcome.value} (Score: {arbitration.score:.2f}, {arbitration.reason})",
                payload={
                    "incident_id": incident_id,
                    "outcome": arbitration.outcome.value,
                    "score": arbitration.score,
                    "reason": arbitration.reason,
                    "coin_flip_winner": arbitration.coin_flip_winner_id,
                },
            )
        )

        if arbitration.coin_flip_winner_id:
            coin_msg = DialogueTemplates.coin_flip(arbitration.coin_flip_winner_id)
            self._dialogue_history.record_dialogue(
                tick=current_tick,
                sender_id=leader_agent.id,
                sender_name=leader_agent.name,
                recipient_id=leader_blocker.id,
                recipient_name=leader_blocker.name,
                message=coin_msg,
                intent="coin_flip",
            )

        if arbitration.outcome in (ArbitrationOutcome.GROUP_1_YIELDS, ArbitrationOutcome.GROUP_2_YIELDS):
            yielding_convoy = convoy_agent if arbitration.outcome == ArbitrationOutcome.GROUP_1_YIELDS else convoy_blocker
            passing_convoy = convoy_blocker if arbitration.outcome == ArbitrationOutcome.GROUP_1_YIELDS else convoy_agent
            niche_config = niche_config_agent if arbitration.outcome == ArbitrationOutcome.GROUP_1_YIELDS else niche_config_blocker
            cost = delta_c_agent if arbitration.outcome == ArbitrationOutcome.GROUP_1_YIELDS else delta_c_blocker
            assert niche_config is not None

            offer_text = DialogueTemplates.offer_yield_short(cost or 1)
            self._dialogue_history.record_dialogue(
                tick=current_tick,
                sender_id=yielding_convoy.leader.id,
                sender_name=yielding_convoy.leader.name,
                recipient_id=passing_convoy.leader.id,
                recipient_name=passing_convoy.leader.name,
                message=offer_text,
                intent="offer_yield",
            )
            accept_text = DialogueTemplates.accept_passage(other_gid=yielding_convoy.id, is_group=is_group)
            self._dialogue_history.record_dialogue(
                tick=current_tick,
                sender_id=passing_convoy.leader.id,
                sender_name=passing_convoy.leader.name,
                recipient_id=yielding_convoy.leader.id,
                recipient_name=yielding_convoy.leader.name,
                message=accept_text,
                intent="accept",
            )

            if is_group:
                self._goal_service.suspend_convoy_goals(
                    agents=yielding_convoy.members,
                    group_id=yielding_convoy.id,
                    participant_ids=[a.id for a in yielding_convoy.members],
                    incident_id=incident_id,
                )

            for slot_assign in niche_config.slots:
                member = next((m for m in yielding_convoy.members if m.id == slot_assign.agent_id), None)
                if member:
                    member.evasion_phase = EvasionPhase.YIELDING_INGRESS
                    self._logger.log(
                        SimulationEvent(
                            tick=current_tick,
                            agent_id=member.id,
                            event_type="evasion_phase_changed",
                            summary=f"Agent {member.name}: Status gewechselt zu {EvasionPhase.YIELDING_INGRESS.value}.",
                            payload={"phase": EvasionPhase.YIELDING_INGRESS.value, "incident_id": incident_id},
                        )
                    )
                    forbidden_for_path = {a.position for a in passing_convoy.members}
                    clean_slot_path = [p for p in slot_assign.path_to_slot if p != member.position]
                    if clean_slot_path and not any(p in forbidden_for_path for p in clean_slot_path):
                        path_to_slot = clean_slot_path
                    else:
                        temp_map = AgentMentalMap(fused_map.width, fused_map.height)
                        temp_map.tiles = dict(fused_map.tiles)
                        for fpos in forbidden_for_path:
                            temp_map.mark_obstacle(fpos, current_tick)
                        path_to_slot = self._pathfinder.find_path(member.position, slot_assign.slot_position, temp_map)
                        if not path_to_slot:
                            path_to_slot = []
                    member.assign_path(path_to_slot)
                    evasion_goal = Goal(
                        name="In Nische ausweichen",
                        target_position=slot_assign.slot_position,
                        junction_position=slot_assign.junction_position,
                        yield_for_agent_id=passing_convoy.leader.id,
                        priority=ExecutionPriority.URGENT,
                        is_group_goal=is_group,
                        group_id=yielding_convoy.id if is_group else None,
                        participant_ids=[a.id for a in yielding_convoy.members] if is_group else [],
                        holds_position=False,
                    )
                    self._goal_service.push_goal(member, evasion_goal, incident_id=incident_id)
                    member.is_waiting_for_reply = False
                    member.interaction_partner_id = None

            for p_member in passing_convoy.members:
                p_member.evasion_phase = EvasionPhase.PASSING
                self._logger.log(
                    SimulationEvent(
                        tick=current_tick,
                        agent_id=p_member.id,
                        event_type="evasion_phase_changed",
                        summary=f"Agent {p_member.name}: Status gewechselt zu {EvasionPhase.PASSING.value}.",
                        payload={"phase": EvasionPhase.PASSING.value, "incident_id": incident_id},
                    )
                )
                p_member.is_waiting_for_reply = False
                p_member.interaction_partner_id = None

        else:
            yielding_convoy, target_pos = self._convoy_coordinator.evaluate_backtracking(
                convoy_1=convoy_agent,
                convoy_2=convoy_blocker,
                map_1=fused_map,
                map_2=fused_map,
            )
            passing_convoy = convoy_blocker if yielding_convoy.id == convoy_agent.id else convoy_agent

            for m in yielding_convoy.members:
                m.evasion_phase = EvasionPhase.YIELDING_INGRESS
                self._logger.log(
                    SimulationEvent(
                        tick=current_tick,
                        agent_id=m.id,
                        event_type="evasion_phase_changed",
                        summary=f"Agent {m.name}: Status gewechselt zu {EvasionPhase.YIELDING_INGRESS.value}.",
                        payload={"phase": EvasionPhase.YIELDING_INGRESS.value, "incident_id": incident_id},
                    )
                )
                m.push_goal(
                    Goal(
                        name="Backtracking",
                        target_position=target_pos,
                        backtracking_junction_target=target_pos,
                        yield_for_agent_id=passing_convoy.leader.id,
                        priority=ExecutionPriority.URGENT,
                    )
                )
                m.is_evasion_locked = True
            for m in passing_convoy.members:
                m.evasion_phase = EvasionPhase.PASSING

        fused_map.persist_to_agent_maps(involved_maps)
        self._session_manager.mark_negotiated(agent.id, blocker.id, current_tick)
        self._session_manager.mark_negotiated(leader_agent.id, leader_blocker.id, current_tick)
        self._session_manager.reset_session(agent.id, blocker.id)
        agent.is_thinking = False
        blocker.is_thinking = False