from __future__ import annotations

from typing import Protocol, Union
from src.domain.models.agent.agent import Agent
from src.domain.models.interaction.commands import InteractionCommand
from src.domain.models.interaction.interaction_result import ImmediateResult, PendingFuture
from src.domain.models.world.world_entity import WorldEntity


class IInteractionDispatcher(Protocol):
    """Port für das typisierte Command-Dispatching zwischen Entitäten."""

    def dispatch(
        self,
        requester: Agent,
        target: WorldEntity,
        command: InteractionCommand,
    ) -> Union[ImmediateResult, PendingFuture]:
        ...