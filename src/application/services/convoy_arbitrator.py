from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from src.domain.models.agent import Agent


class ArbitrationOutcome(str, Enum):
    GROUP_1_YIELDS = "group_1_yields"
    GROUP_2_YIELDS = "group_2_yields"
    BACKTRACK_REQUIRED = "backtrack_required"


@dataclass(frozen=True, slots=True)
class ArbitrationResult:
    outcome: ArbitrationOutcome
    score: float
    advantage_cost_norm: float
    advantage_trait_norm: float
    coin_flip_winner_id: Optional[str] = None
    reason: str = ""


class ConvoyArbitrator:
    """
    Deterministische Arbitrierung und 70/30-Nutzenfunktion für Korridor- und Nischenkonflikte.
    Gemäß Details.md Punkte 3, 5, 6 und ImplementationPlan.md Abschnitte 3.6, 3.7.
    """

    MAX_TRAIT_RANGE: float = 2.0  # Max(charisma + assertiveness) = 1.0 + 1.0 = 2.0

    @staticmethod
    def calculate_trait_score(agents: list[Agent]) -> float:
        """Berechnet den durchschnittlichen Trait-Score einer Gruppe: 1/|G| * sum(charisma + assertiveness)."""
        if not agents:
            return 0.0
        total_traits = sum(agent.charisma + agent.assertiveness for agent in agents)
        return total_traits / len(agents)

    @staticmethod
    def deterministic_coin_flip(leader_a_id: str, leader_b_id: str, tick: int) -> str:
        """
        Deterministischer, prozess- und replay-stabiler SHA-256-Münzwurf bei Gleichstand (Score == 0.0).
        Gibt die ID des Gewinners zurück (derjenige, der zuerst passieren darf).
        """
        sorted_ids = sorted([leader_a_id, leader_b_id])
        payload = f"{sorted_ids[0]}:{sorted_ids[1]}:{tick}".encode("utf-8")
        digest = int(hashlib.sha256(payload).hexdigest(), 16)
        return sorted_ids[0] if (digest % 2 == 0) else sorted_ids[1]

    def arbitrate(
        self,
        group_1_agents: list[Agent],
        group_2_agents: list[Agent],
        delta_c_1: Optional[int],  # Zusatzkosten wenn Gruppe 1 ausweicht (None wenn keine Nische)
        delta_c_2: Optional[int],  # Zusatzkosten wenn Gruppe 2 ausweicht (None wenn keine Nische)
        tick: int,
        leader_1_id: Optional[str] = None,
        leader_2_id: Optional[str] = None,
    ) -> ArbitrationResult:
        """
        Entscheidet deterministisch, welche Gruppe ausweicht bzw. ob Backtracking initiiert wird.
        Score(G1) > 0 -> G1 weicht aus (G2 passiert).
        Score(G1) < 0 -> G2 weicht aus (G1 passiert).
        Score(G1) == 0.0 -> Münzwurf bestimmt den Passierenden (der andere weicht aus).
        """
        # Grenzfallprüfung vor Quotientenbildung (Details.md Punkt 5)
        if delta_c_1 is None and delta_c_2 is None:
            return ArbitrationResult(
                outcome=ArbitrationOutcome.BACKTRACK_REQUIRED,
                score=0.0,
                advantage_cost_norm=0.0,
                advantage_trait_norm=0.0,
                reason="Keine Gruppe hat eine Nische gefunden. Backtracking-Protokoll initiiert.",
            )

        if delta_c_1 is not None and delta_c_2 is None:
            return ArbitrationResult(
                outcome=ArbitrationOutcome.GROUP_1_YIELDS,
                score=1.0,
                advantage_cost_norm=1.0,
                advantage_trait_norm=0.0,
                reason="Nur Gruppe 1 hat eine Nische gefunden. Gruppe 1 weicht deterministisch aus.",
            )

        if delta_c_1 is None and delta_c_2 is not None:
            return ArbitrationResult(
                outcome=ArbitrationOutcome.GROUP_2_YIELDS,
                score=-1.0,
                advantage_cost_norm=-1.0,
                advantage_trait_norm=0.0,
                reason="Nur Gruppe 2 hat eine Nische gefunden. Gruppe 2 weicht deterministisch aus.",
            )

        # Beide Gruppen haben valide Nischen (delta_c_1 is not None and delta_c_2 is not None)
        assert delta_c_1 is not None and delta_c_2 is not None

        cost_denom = max(delta_c_2 + delta_c_1, 1)
        adv_cost_norm = (delta_c_2 - delta_c_1) / cost_denom
        # Begrenzung auf [-1.0, 1.0]
        adv_cost_norm = max(-1.0, min(1.0, adv_cost_norm))

        trait_1 = self.calculate_trait_score(group_1_agents)
        trait_2 = self.calculate_trait_score(group_2_agents)
        adv_trait_norm = (trait_1 - trait_2) / self.MAX_TRAIT_RANGE
        # Begrenzung auf [-1.0, 1.0]
        adv_trait_norm = max(-1.0, min(1.0, adv_trait_norm))

        # 70 % Wegunterschied, 30 % Charaktereigenschaften
        score = 0.7 * adv_cost_norm + 0.3 * adv_trait_norm
        # Runden auf 6 Nachkommastellen gegen Fließkomma-Ungenauigkeiten bei Gleichstand
        score = round(score, 6)

        l1_id = leader_1_id or (group_1_agents[0].id if group_1_agents else "1")
        l2_id = leader_2_id or (group_2_agents[0].id if group_2_agents else "2")

        if score > 0.0:
            return ArbitrationResult(
                outcome=ArbitrationOutcome.GROUP_1_YIELDS,
                score=score,
                advantage_cost_norm=adv_cost_norm,
                advantage_trait_norm=adv_trait_norm,
                reason="Gruppe 1 hat geringere Ausweich-Zusatzkosten (Score > 0). Gruppe 1 weicht aus.",
            )
        elif score < 0.0:
            return ArbitrationResult(
                outcome=ArbitrationOutcome.GROUP_2_YIELDS,
                score=score,
                advantage_cost_norm=adv_cost_norm,
                advantage_trait_norm=adv_trait_norm,
                reason="Gruppe 2 hat geringere Ausweich-Zusatzkosten (Score < 0). Gruppe 2 weicht aus.",
            )
        else:
            # Absoluter Gleichstand (Score == 0.0) -> Deterministischer SHA-256-Münzwurf
            winner_id = self.deterministic_coin_flip(l1_id, l2_id, tick)
            # Der Gewinner passiert zuerst; die andere Gruppe weicht aus
            outcome = (
                ArbitrationOutcome.GROUP_2_YIELDS
                if winner_id == l1_id
                else ArbitrationOutcome.GROUP_1_YIELDS
            )
            return ArbitrationResult(
                outcome=outcome,
                score=0.0,
                advantage_cost_norm=adv_cost_norm,
                advantage_trait_norm=adv_trait_norm,
                coin_flip_winner_id=winner_id,
                reason=f"Gleichstand der Nutzenfunktion (Score=0.0). Münzwurf bestimmt: Agent/Gruppe {winner_id} passiert zuerst.",
            )
