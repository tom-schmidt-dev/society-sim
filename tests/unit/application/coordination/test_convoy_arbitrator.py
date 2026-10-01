from __future__ import annotations

import pytest
from src.domain.models.agent.agent import Agent
from src.domain.models.communication.communication_templates import DialogueTemplates
from src.domain.models.world.position import Position
from src.application.services.coordination.convoy_arbitrator import (
    ConvoyArbitrator,
    ArbitrationOutcome,
)


class TestConvoyArbitrator:
    @pytest.fixture
    def arbitrator(self) -> ConvoyArbitrator:
        return ConvoyArbitrator()

    def test_deterministic_coin_flip_symmetry_and_stability(self, arbitrator: ConvoyArbitrator) -> None:
        # Replay-Stabilität: Gleiche Eingaben -> identisches Ergebnis
        w1 = arbitrator.deterministic_coin_flip("agent_a", "agent_b", 42)
        w2 = arbitrator.deterministic_coin_flip("agent_a", "agent_b", 42)
        assert w1 == w2

        # Symmetrie: Vertauschte Reihenfolge der Argumente -> identischer Gewinner
        w3 = arbitrator.deterministic_coin_flip("agent_b", "agent_a", 42)
        assert w1 == w3

        # Gewinner ist zwingend einer der beiden IDs
        assert w1 in ("agent_a", "agent_b")

    def test_calculate_trait_score(self, arbitrator: ConvoyArbitrator) -> None:
        assert arbitrator.calculate_trait_score([]) == 0.0

        a1 = Agent(id="1", name="A1", position=Position(0, 0), charisma=0.2, assertiveness=0.4)
        a2 = Agent(id="2", name="A2", position=Position(0, 1), charisma=0.6, assertiveness=0.8)
        # Average = ((0.2 + 0.4) + (0.6 + 0.8)) / 2 = 2.0 / 2 = 1.0
        assert arbitrator.calculate_trait_score([a1, a2]) == pytest.approx(1.0)

    def test_arbitrate_border_cases(self, arbitrator: ConvoyArbitrator) -> None:
        a1 = Agent(id="1", name="Alice", position=Position(1, 1))
        b1 = Agent(id="2", name="Bob", position=Position(2, 1))

        # Fall A: Keine Nische für beide -> Backtracking
        res_none = arbitrator.arbitrate([a1], [b1], delta_c_1=None, delta_c_2=None, tick=1)
        assert res_none.outcome == ArbitrationOutcome.BACKTRACK_REQUIRED

        # Fall B: Nur Gruppe 1 hat Nische -> Gruppe 1 weicht aus
        res_only_1 = arbitrator.arbitrate([a1], [b1], delta_c_1=3, delta_c_2=None, tick=1)
        assert res_only_1.outcome == ArbitrationOutcome.GROUP_1_YIELDS

        # Fall C: Nur Gruppe 2 hat Nische -> Gruppe 2 weicht aus
        res_only_2 = arbitrator.arbitrate([a1], [b1], delta_c_1=None, delta_c_2=4, tick=1)
        assert res_only_2.outcome == ArbitrationOutcome.GROUP_2_YIELDS

    def test_arbitrate_cost_advantage_with_default_traits(self, arbitrator: ConvoyArbitrator) -> None:
        # Default Traits (0.0): Kostenvorteil entscheidet allein
        a1 = Agent(id="1", name="Alice", position=Position(1, 1))
        b1 = Agent(id="2", name="Bob", position=Position(2, 1))

        # Gruppe 1 Ausweichkosten 2, Gruppe 2 Ausweichkosten 8
        # delta_c_1 < delta_c_2 -> Gruppe 1 hat geringere Kosten -> Gruppe 1 weicht aus
        res_1_yields = arbitrator.arbitrate([a1], [b1], delta_c_1=2, delta_c_2=8, tick=1)
        assert res_1_yields.outcome == ArbitrationOutcome.GROUP_1_YIELDS
        assert res_1_yields.score > 0.0

        # Umgekehrt: Gruppe 1 Ausweichkosten 10, Gruppe 2 Ausweichkosten 3 -> Gruppe 2 weicht aus
        res_2_yields = arbitrator.arbitrate([a1], [b1], delta_c_1=10, delta_c_2=3, tick=1)
        assert res_2_yields.outcome == ArbitrationOutcome.GROUP_2_YIELDS
        assert res_2_yields.score < 0.0

    def test_arbitrate_equal_scores_triggers_coin_flip(self, arbitrator: ConvoyArbitrator) -> None:
        a1 = Agent(id="1", name="Alice", position=Position(1, 1))
        b1 = Agent(id="2", name="Bob", position=Position(2, 1))

        # Identische Kosten (5 vs 5) und identische Traits (0.0 vs 0.0) -> Score == 0.0
        res_flip = arbitrator.arbitrate([a1], [b1], delta_c_1=5, delta_c_2=5, tick=10)
        assert res_flip.score == 0.0
        assert res_flip.coin_flip_winner_id in ("1", "2")
        if res_flip.coin_flip_winner_id == "1":
            assert res_flip.outcome == ArbitrationOutcome.GROUP_2_YIELDS
        else:
            assert res_flip.outcome == ArbitrationOutcome.GROUP_1_YIELDS

    def test_dialogue_templates_exact_wording(self) -> None:
        # Prüfung gegen Details.md Punkt 3
        assert DialogueTemplates.conflict_notice(is_group=False) == "Hier ist nicht genug Platz für uns beide."
        assert DialogueTemplates.conflict_notice(is_group=True) == "Hier ist nicht genug Platz für unsere Gruppen."
        assert DialogueTemplates.niche_seen(3) == "Ja das stimmt. Ich habe eine Nische gesehen. Sie ist 3 Felder von mir entfernt. Ich könnte Platz machen."
        assert DialogueTemplates.offer_yield_short(2) == "Meine Lücke ist 2 Felder entfernt. Ich mache dir kurz Platz."
        assert DialogueTemplates.request_yield_far(7) == "Für mich wäre der Weg weiter, sie ist 7 Felder entfernt. Bitte sei so lieb und mache mir kurz Platz."
        assert "verlängert sich mein Weg um 4 Felder" in DialogueTemplates.counter_offer("(12, 5)", 4, 2)
        assert DialogueTemplates.accept_passage_group("Alpha") == "Zusatzkosten für Gruppe Alpha sind geringer. Wir übernehmen die Passage. Los geht's."
        assert DialogueTemplates.reject_request_group("Alpha") == "Zusatzkosten für Gruppe Alpha sind höher. Bitte weicht aus."
        assert "Münzwurf hat entschieden" in DialogueTemplates.coin_flip("Alice")
        assert DialogueTemplates.clearance(is_group=False) == "Passage abgeschlossen. Danke fürs Platz machen!"
        assert DialogueTemplates.courtesy_reply(is_group=False) == "Gern geschehen! Setze Weg fort."

    def test_arbitrate_assertive_agent_forces_cooperative_agent_to_yield(
            self, arbitrator: ConvoyArbitrator
    ) -> None:
        # Gruppe 1 (Bob): kooperativ (Trait 0.8). Gruppe 2 (Alice): fordernd (Trait 1.55).
        # Bei gleichen Kosten (5 vs 5) muss Gruppe 1 weichen (Score > 0 -> GROUP_1_YIELDS).
        bob = Agent(id="2", name="Bob", position=Position(15, 4), charisma=0.6, assertiveness=0.2)
        alice = Agent(id="1", name="Alice", position=Position(14, 4), charisma=0.7, assertiveness=0.85)

        res = arbitrator.arbitrate(
            group_1_agents=[bob],
            group_2_agents=[alice],
            delta_c_1=5,
            delta_c_2=5,
            tick=13,
        )
        assert res.outcome == ArbitrationOutcome.GROUP_1_YIELDS
        assert res.score > 0.0