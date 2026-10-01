from __future__ import annotations
from abc import ABC, abstractmethod
from src.domain.models.planning.events import SimulationEvent


class IEventLogger(ABC):
    @abstractmethod
    def log(self, event: SimulationEvent) -> None:
        """Persistiert ein einzelnes Simulationsereignis."""
        pass