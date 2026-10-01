from __future__ import annotations

from typing import Optional

from src.domain.models.agent.agent import Agent
from src.domain.models.planning.events import SimulationEvent
from src.domain.ports.event_logger import IEventLogger


class DayNightService:
    """Verwaltet den globalen Tag-Nacht-Zyklus und steuert Schlaf- sowie Wachphasen."""

    def __init__(
        self,
        day_ticks: int = 100,
        night_ticks: int = 20,
        logger: Optional[IEventLogger] = None,
    ) -> None:
        if day_ticks <= 0 or night_ticks <= 0:
            raise ValueError("day_ticks and night_ticks must be positive integers.")
        self.day_ticks = day_ticks
        self.night_ticks = night_ticks
        self._logger = logger

    @property
    def cycle_length(self) -> int:
        """Gesamtlänge eines Zyklus aus Tag- und Nachttakten."""
        return self.day_ticks + self.night_ticks

    def is_night(self, tick: int) -> bool:
        """Gibt True zurück, wenn der Takt in die Nachtphase fällt."""
        return (tick % self.cycle_length) >= self.day_ticks

    def is_day(self, tick: int) -> bool:
        """Gibt True zurück, wenn der Takt in die Tagphase fällt."""
        return not self.is_night(tick)

    def get_day_number(self, tick: int) -> int:
        """Gibt die aktuelle Tagesnummer zurück (beginnend bei 1)."""
        return (tick // self.cycle_length) + 1

    def get_phase_progress(self, tick: int) -> float:
        """Berechnet den Fortschritt der aktuellen Phase im Intervall [0.0, 1.0)."""
        phase_tick = tick % self.cycle_length
        if self.is_night(tick):
            return (phase_tick - self.day_ticks) / self.night_ticks
        return phase_tick / self.day_ticks

    def process_tick(self, tick: int, agents: list[Agent]) -> Optional[str]:
        """
        Synchronisiert den Status aller Agenten mit dem Tag-Nacht-Zyklus.
        Gibt 'night_started' oder 'day_started' bei Phasenwechseln zurück, andernfalls None.
        """
        phase_tick = tick % self.cycle_length
        day_num = self.get_day_number(tick)

        if phase_tick == self.day_ticks:
            for agent in agents:
                agent.is_sleeping = True

            if self._logger:
                self._logger.log(
                    SimulationEvent(
                        tick=tick,
                        agent_id="system",
                        event_type="night_started",
                        summary=f"Nacht von Tag {day_num} bricht an. Agenten schlafen.",
                        payload={"day": day_num, "tick": tick},
                    )
                )
            return "night_started"

        if phase_tick == 0 and tick > 0:
            for agent in agents:
                agent.is_sleeping = False

            if self._logger:
                self._logger.log(
                    SimulationEvent(
                        tick=tick,
                        agent_id="system",
                        event_type="day_started",
                        summary=f"Tag {day_num} bricht an. Agenten erwachen.",
                        payload={"day": day_num, "tick": tick},
                    )
                )
            return "day_started"

        return None