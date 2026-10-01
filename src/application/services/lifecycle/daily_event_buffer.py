
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from src.domain.models.planning.events import SimulationEvent


class DailyEventBuffer:
    """Hält relevante Tagesereignisse pro Agent im Arbeitsspeicher bis zur nächtlichen Konsolidierung."""

    def __init__(self) -> None:
        self._events_by_agent: dict[str, list[SimulationEvent]] = defaultdict(list)

    def record_event(self, agent_id: str, event: SimulationEvent) -> None:
        """Fügt ein Ereignis zur Tagesliste des Agenten hinzu."""
        self._events_by_agent[agent_id].append(event)

    def get_events_for_agent(self, agent_id: str) -> list[SimulationEvent]:
        """Liefert die gesammelten Ereignisse eines Agenten."""
        return list(self._events_by_agent.get(agent_id, []))

    def clear_agent(self, agent_id: str) -> None:
        """Bereinigt den Ereignispuffer eines Agenten nach erfolgter Konsolidierung."""
        self._events_by_agent.pop(agent_id, None)

    def clear_all(self) -> None:
        """Setzt den gesamten Puffer zurück."""
        self._events_by_agent.clear()