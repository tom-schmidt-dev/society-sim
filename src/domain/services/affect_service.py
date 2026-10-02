from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class AgentAffectState:
    frustration: float = 0.0  # Bereich [0.0, 1.0]
    patience: float = 1.0     # Bereich [0.0, 1.0]


class AffectService:
    """
    Verwaltet ephemere emotionale Zustände (Geduld, Frustration) während einer Verhandlung.

    ARCHITEKTUR-HINWEIS / ROADMAP:
    Dieses deterministische Affekt-Modul dient als Interimslösung.
    In der geplanten Dual-Model-Architektur übernimmt das lokale SLM (Small Language Model)
    diese hochfrequente psychologische Bewertung und dynamische Satzgenerierung mit minimaler Latenz.
    """

    def __init__(self) -> None:
        self._states: dict[tuple[str, str], AgentAffectState] = {}

    @staticmethod
    def _make_key(agent_a_id: str, agent_b_id: str) -> tuple[str, str]:
        return (agent_a_id, agent_b_id)

    def get_affect(self, agent_id: str, partner_id: str) -> AgentAffectState:
        key = self._make_key(agent_id, partner_id)
        if key not in self._states:
            self._states[key] = AgentAffectState()
        return self._states[key]

    def record_turn(
        self,
        agent_id: str,
        partner_id: str,
        incoming_intent: Optional[str],
    ) -> None:
        """Aktualisiert die Affektwerte nach einer Interaktionsrunde als neues Value Object."""
        current = self.get_affect(agent_id, partner_id)
        new_frustration = current.frustration
        new_patience = current.patience

        if incoming_intent == "reject":
            new_frustration = min(1.0, round(current.frustration + 0.25, 2))
            new_patience = max(0.0, round(current.patience - 0.25, 2))
        elif incoming_intent in ("accept", "offer_yield"):
            new_frustration = 0.0
            new_patience = 1.0

        key = self._make_key(agent_id, partner_id)
        self._states[key] = AgentAffectState(
            frustration=new_frustration,
            patience=new_patience,
        )

    def reset_affect(self, agent_id: str, partner_id: str) -> None:
        """Setzt die Affektwerte bei Beendigung des Dialogs zurück."""
        key = self._make_key(agent_id, partner_id)
        self._states.pop(key, None)

    def generate_situational_notes(
        self,
        agent_id: str,
        partner_id: str,
        assertiveness: float = 0.5,
    ) -> list[str]:
        """
        Erzeugt flüchtige, situative Verhaltenshinweise für den System-Prompt des LLMs.
        Diese Sätze existieren ausschließlich für den jeweiligen API-Call und werden nicht
        in das persistente Gedächtnis übernommen.
        """
        state = self.get_affect(agent_id, partner_id)
        notes: list[str] = []

        if state.frustration >= 0.7:
            notes.append("Du verlierst allmählich die Geduld, da dein Gegenüber uneinsichtig bleibt.")

        if state.patience <= 0.3:
            notes.append("Du stehst unter Zeitdruck und kannst keine weiteren Verzögerungen hinnehmen.")

        if state.frustration >= 0.5 and assertiveness >= 0.7:
            notes.append("Du empfindest das Beharren deines Gegenübers als unbegründet und forderst entschieden Vorrang.")
        elif state.frustration >= 0.5 and assertiveness <= 0.3:
            notes.append("Die angespannte Situation belastet dich, du suchst dringend nach einem Ausweg.")

        return notes