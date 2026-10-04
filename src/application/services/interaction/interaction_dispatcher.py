from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional, Union

from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_state import AgentLifecycleState
from src.domain.models.interaction.commands import (
    EndDialogueCommand,
    InspectCommand,
    InteractionCommand,
    ProbeCommand,
    TalkCommand,
)
from src.domain.models.interaction.interaction_result import ImmediateResult, PendingFuture
from src.domain.models.planning.events import SimulationEvent
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.interaction_dispatcher import IInteractionDispatcher


class InteractionDispatcher(IInteractionDispatcher):
    """Vermittelt Interaktionsbefehle und aktualisiert das epistemische Gedächtnis reaktiv."""

    def __init__(
        self,
        logger: Optional[IEventLogger] = None,
        tick_provider: Optional[Callable[[], int]] = None,
    ) -> None:
        self._logger = logger
        self._tick_provider = tick_provider or (lambda: 0)

    def dispatch(
        self,
        requester: Agent,
        target: WorldEntity,
        command: InteractionCommand,
    ) -> Union[ImmediateResult, PendingFuture]:
        current_tick = self._tick_provider()

        # 1. Umweltwissen im Gedächtnis initial verankern
        if target.id not in requester.memory.known_entities:
            requester.memory.update_entity_perception(
                entity_id=target.id,
                name=target.name,
                pos=target.position,
                tick=current_tick,
                entity_type=target.entity_type,
            )

        # 2. Fail-Fast bei fremdbeschäftigtem oder deliberierendem Agenten
        if isinstance(command, TalkCommand) and isinstance(target, Agent):
            is_partner = target.interaction_partner_id == requester.id
            is_mutual = bool(target.path and target.path[0] == requester.position)

            if not (is_partner or is_mutual):
                if (
                    target.interaction_partner_id not in (None, requester.id)
                    or target.lifecycle_state == AgentLifecycleState.DELIBERATING
                ):
                    if self._logger:
                        self._logger.log(
                            SimulationEvent(
                                tick=current_tick,
                                agent_id=requester.id,
                                event_type="interaction_busy_rejected",
                                summary=f"{requester.name} -> {target.name} abgewiesen: Partner ist beschäftigt.",
                                payload={"target_id": target.id, "reason": "BUSY"},
                            )
                        )
                    return ImmediateResult(success=False, reason="BUSY")

        # 3. Double-Dispatch an das Zielobjekt
        result = target.receive_interaction(command)

        # 4. Synchrones Ergebnis verarbeiten
        if isinstance(result, ImmediateResult):
            self._apply_immediate_feedback(requester, target, command, result)
            if self._logger:
                self._logger.log(
                    SimulationEvent(
                        tick=current_tick,
                        agent_id=requester.id,
                        event_type="interaction_immediate_result",
                        summary=f"{requester.name} -> {target.name} [{type(command).__name__}]: {result.reason}",
                        payload={"success": result.success, "reason": result.reason},
                    )
                )
            return result

        # 5. Asynchrones PendingFuture registrieren
        requester.transition_to(AgentLifecycleState.WAITING_FOR_PEER)
        requester.is_waiting_for_reply = True
        requester.interaction_partner_id = target.id

        def _on_future_completed(fut: asyncio.Future[Any]) -> None:
            if fut.cancelled():
                if requester.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER:
                    requester.transition_to(AgentLifecycleState.IDLE)
                return

            try:
                outcome = fut.result()
            except Exception:
                requester.memory.record_interaction_result(target.id, responded=False)
                if requester.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER:
                    requester.transition_to(AgentLifecycleState.IDLE)
                return

            if isinstance(outcome, ImmediateResult) and (
                not outcome.success
                or outcome.reason in ("DIALOGUE_ENDED", "REJECTED", "FAREWELL_COMPLETED")
            ):
                requester.memory.record_interaction_result(target.id, responded=False)
                if requester.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER:
                    requester.transition_to(AgentLifecycleState.IDLE)
            else:
                requester.memory.record_interaction_result(target.id, responded=True)
                if requester.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER:
                    requester.transition_to(
                        AgentLifecycleState.DELIBERATING,
                        reason=f"Antwort von {target.name} empfangen",
                    )

        result.future.add_done_callback(_on_future_completed)

        if self._logger:
            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=requester.id,
                    event_type="interaction_pending_started",
                    summary=f"{requester.name} wartet auf Antwort von {target.name}.",
                    payload={"target_id": target.id},
                )
            )

        return result


    def _apply_immediate_feedback(
        self,
        requester: Agent,
        target: WorldEntity,
        command: InteractionCommand,
        result: ImmediateResult,
    ) -> None:
        """Aktualisiert Fakten im Gedächtnis des anfragenden Agenten deterministisch."""
        if isinstance(command, TalkCommand):
            requester.memory.record_interaction_result(target.id, responded=result.success)

        elif isinstance(command, InspectCommand) and result.success:
            entity_type = result.payload.get("entity_type") or target.entity_type
            requester.memory.record_inspection(target.id, entity_type)
            if "is_passable" in result.payload:
                requester.memory.record_walkability_result(
                    target.id, is_walkable=result.payload["is_passable"]
                )

        elif isinstance(command, ProbeCommand) and result.success:
            if "is_passable" in result.payload:
                requester.memory.record_walkability_result(
                    target.id, is_walkable=result.payload["is_passable"]
                )

        elif isinstance(command, EndDialogueCommand):
            requester.is_waiting_for_reply = False
            requester.interaction_partner_id = None