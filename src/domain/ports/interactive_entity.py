from __future__ import annotations

from typing import Protocol, Union
from src.domain.models.interaction.commands import InteractionCommand
from src.domain.models.interaction.interaction_result import ImmediateResult, PendingFuture


class IInteractiveEntity(Protocol):
    """Protokoll für alle Entitäten, die typisierte Interaktionsbefehle empfangen können."""

    def receive_interaction(
        self, command: InteractionCommand
    ) -> Union[ImmediateResult, PendingFuture]:
        ...