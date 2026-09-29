from __future__ import annotations

from src.domain.models.agent import Agent


class NeedService:
    """Überwacht und aktualisiert Vitalwerte und physiologische Bedürfnisse."""

    def __init__(
        self,
        hunger_increase_per_tick: float = 0.01,
        hunger_threshold: float = 0.7,
    ) -> None:
        self.hunger_increase_per_tick = hunger_increase_per_tick
        self.hunger_threshold = hunger_threshold

    def update_needs(self, agent: Agent) -> None:
        """Erhöht physiologische Mängel pro Zeittakt."""
        current_hunger = agent.needs.get("hunger", 0.0)
        agent.needs["hunger"] = current_hunger + self.hunger_increase_per_tick

    def is_need_urgent(self, agent: Agent, need_name: str = "hunger") -> bool:
        """Prüft, ob ein Bedürfnis den Schwellenwert für Handlungsbedarf überschreitet."""
        return agent.needs.get(need_name, 0.0) >= self.hunger_threshold

    def satisfy_need(self, agent: Agent, need_name: str, reduction: float) -> None:
        """Reduziert den Mangelwert nach erfolgreicher Befriedigung."""
        if need_name in agent.needs:
            agent.needs[need_name] = max(0.0, agent.needs[need_name] - reduction)