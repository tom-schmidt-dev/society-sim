from __future__ import annotations

from typing import Optional
from src.domain.models.agent import Agent


class NeedService:
    """Überwacht, aktualisiert und priorisiert physiologische Vitalwerte und Mängel."""

    DEFAULT_INCREASE_RATES: dict[str, float] = {
        "hunger": 0.01,
        "thirst": 0.015,
        "energy": 0.008,
    }

    DEFAULT_THRESHOLDS: dict[str, float] = {
        "hunger": 0.70,
        "thirst": 0.70,
        "energy": 0.70,
    }

    DEFAULT_WEIGHTS: dict[str, float] = {
        "thirst": 1.2,
        "hunger": 1.0,
        "energy": 0.8,
    }

    def __init__(
        self,
        hunger_increase_per_tick: Optional[float] = None,
        thirst_increase_per_tick: Optional[float] = None,
        energy_increase_per_tick: Optional[float] = None,
        increase_rates: Optional[dict[str, float]] = None,
        hunger_threshold: Optional[float] = None,
        thresholds: Optional[dict[str, float]] = None,
        weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.increase_rates = dict(self.DEFAULT_INCREASE_RATES)
        if increase_rates:
            self.increase_rates.update(increase_rates)

        if hunger_increase_per_tick is not None:
            self.increase_rates["hunger"] = hunger_increase_per_tick
            if (
                hunger_increase_per_tick == 0.0
                and thirst_increase_per_tick is None
                and energy_increase_per_tick is None
                and not increase_rates
            ):
                self.increase_rates["thirst"] = 0.0
                self.increase_rates["energy"] = 0.0

        if thirst_increase_per_tick is not None:
            self.increase_rates["thirst"] = thirst_increase_per_tick
        if energy_increase_per_tick is not None:
            self.increase_rates["energy"] = energy_increase_per_tick

        self.thresholds = dict(self.DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        if hunger_threshold is not None:
            self.thresholds["hunger"] = hunger_threshold

        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    @property
    def hunger_increase_per_tick(self) -> float:
        return self.increase_rates.get("hunger", 0.01)

    @property
    def hunger_threshold(self) -> float:
        return self.thresholds.get("hunger", 0.70)

    def update_needs(self, agent: Agent) -> None:
        """Erhöht physiologische Mängel pro Zeittakt deterministisch bis maximal 1.0."""
        if getattr(agent, "is_sleeping", False):
            # Regeneration während der Schlafphase
            self.satisfy_need(agent, "energy", reduction=0.03)
            # Reduzierter Grundumsatz bei Schlaf
            for need_name in ("hunger", "thirst"):
                current_val = agent.needs.get(need_name, 0.0)
                rate = self.increase_rates.get(need_name, 0.0) * 0.5
                agent.needs[need_name] = min(1.0, current_val + rate)
            return

        for need_name, rate in self.increase_rates.items():
            current_val = agent.needs.get(need_name, 0.0)
            agent.needs[need_name] = min(1.0, current_val + rate)

    def is_need_urgent(self, agent: Agent, need_name: str = "hunger") -> bool:
        """Prüft, ob ein spezifisches Bedürfnis den Schwellenwert für Handlungsbedarf erreicht."""
        threshold = self.thresholds.get(need_name, 0.70)
        return agent.needs.get(need_name, 0.0) >= threshold

    def is_any_need_urgent(self, agent: Agent) -> bool:
        """Ermittelt, ob mindestens ein physiologisches Bedürfnis Handlungsbedarf signalisiert."""
        return any(self.is_need_urgent(agent, need_name) for need_name in self.thresholds)

    def calculate_priority(self, agent: Agent, need_name: str) -> float:
        """Berechnet den gewichteten Dringlichkeitswert eines Bedürfnisses."""
        level = agent.needs.get(need_name, 0.0)
        weight = self.weights.get(need_name, 1.0)
        return level * weight

    def get_dominant_need(self, agent: Agent, only_urgent: bool = True) -> Optional[str]:
        """
        Gibt den Namen des Bedürfnisses mit der höchsten gewichteten Priorität zurück.
        Wird only_urgent=True gesetzt, werden nur Bedürfnisse über dem Schwellenwert bewertet.
        """
        candidate_needs = (
            [name for name in self.thresholds if self.is_need_urgent(agent, name)]
            if only_urgent
            else list(self.thresholds.keys())
        )

        if not candidate_needs:
            return None

        return max(candidate_needs, key=lambda name: self.calculate_priority(agent, name))

    def satisfy_need(self, agent: Agent, need_name: str, reduction: float) -> None:
        """Reduziert den Mangelwert nach erfolgreicher Befriedigung bis minimal 0.0."""
        if need_name in agent.needs:
            agent.needs[need_name] = max(0.0, agent.needs[need_name] - reduction)