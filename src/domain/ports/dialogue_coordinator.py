from __future__ import annotations

from abc import ABC, abstractmethod
from src.domain.models.agent.agent import Agent
from src.domain.models.world.world_entity import WorldEntity


class IDialogueCoordinator(ABC):
    @abstractmethod
    async def handle_incoming_dialogue(
        self,
        agent: Agent,
        all_entities: list[WorldEntity],
    ) -> None:
        """Verarbeitet eingegangene Nachrichten eines Agenten und führt Dialoge fort oder beendet sie."""
        pass