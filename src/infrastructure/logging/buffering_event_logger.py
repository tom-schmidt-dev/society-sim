from __future__ import annotations

from src.application.services.daily_event_buffer import DailyEventBuffer
from src.domain.models.events import SimulationEvent
from src.domain.ports.event_logger import IEventLogger


class BufferingEventLogger(IEventLogger):
    """Protokolliert Ereignisse persistent und leitet agentenbezogene Ereignisse in den Tagespuffer weiter."""

    def __init__(self, inner_logger: IEventLogger, buffer: DailyEventBuffer) -> None:
        self._inner_logger = inner_logger
        self._buffer = buffer

    def log(self, event: SimulationEvent) -> None:
        self._inner_logger.log(event)
        if event.agent_id and event.agent_id != "system":
            self._buffer.record_event(event.agent_id, event)