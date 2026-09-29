from __future__ import annotations

import pytest
from src.domain.models.agent import Agent
from src.domain.models.goal import Goal
from src.domain.models.mental_map import AgentMentalMap
from src.domain.models.position import Position
from src.application.services.convoy_coordinator import (
    ConvoyCoordinator,
    Convoy,
    DrainMode,
    PermutationStep,
)


class TestConvoyCoordinator:
    @pytest.fixture
    def coordinator(self) -> ConvoyCoordinator:
        return ConvoyCoordinator()

    def test_identify_convoys_distance_and_direction(self, coordinator: ConvoyCoordinator) -> None:
        # Alice und Bob bewegen sich nach Osten (dx=1, dy=0) mit L1=1 -> Bilden gemeinsamen Konvoi
        alice = Agent(id="1", name="Alice", position=Position(10, 5))
        alice.path = [Position(11, 5), Position(12, 5)]

        bob = Agent(id="2", name="Bob", position=Position(9, 5))
        bob.path = [Position(10, 5), Position(11, 5)]

        # Charlie bewegt sich nach Westen -> Separater Konvoi
        charlie = Agent(id="3", name="Charlie", position=Position(20, 5))
        charlie.path = [Position(19, 5), Position(18, 5)]

        # Dave ist zu weit weg (L1 > 2) -> Separater Konvoi
        dave = Agent(id="4", name="Dave", position=Position(5, 5))
        dave.path = [Position(6, 5), Position(7, 5)]

        convoys = coordinator.identify_convoys([alice, bob, charlie, dave])

        # Es gibt 3 Konvois: [Alice, Bob], [Charlie], [Dave]
        assert len(convoys) == 3

        east_convoy = next(c for c in convoys if c.direction_vector == (1, 0) and len(c.members) == 2)
        # Alice ist vorne (x=10 > x=9), also Leader
        assert east_convoy.leader.id == "1"
        assert east_convoy.tail.id == "2"

    def test_select_leader_closest_to_conflict(self, coordinator: ConvoyCoordinator) -> None:
        a1 = Agent(id="1", name="A1", position=Position(10, 5))
        a2 = Agent(id="2", name="A2", position=Position(9, 5))
        convoy = Convoy(id="c1", members=[a1, a2], direction_vector=(1, 0))

        conflict = Position(12, 5)
        leader = coordinator.select_leader(convoy, conflict)
        assert leader.id == "1"

    def test_permutation_lifecycle(self, coordinator: ConvoyCoordinator) -> None:
        # Alice (Leader bei x=10) will zu x=30. Bob (Follower bei x=9) will zu x=15.
        # Da Alice ein weiter entferntes Ziel hat als Bob, muss Bob vorbei!
        alice = Agent(id="1", name="Alice", position=Position(10, 5))
        alice.push_goal(Goal(name="G_Far", target_position=Position(30, 5)))

        bob = Agent(id="2", name="Bob", position=Position(9, 5))
        bob.push_goal(Goal(name="G_Near", target_position=Position(15, 5)))

        convoy = Convoy(id="c1", members=[alice, bob], direction_vector=(1, 0))
        assert coordinator.check_permutation_needed(convoy) is True

        # Karte mit freier Kachel orthogonal (Norden: y=4)
        mmap = AgentMentalMap(width=50, height=20)
        mmap.update_tile(Position(10, 4), is_walkable=True, tick=1)

        plan = coordinator.plan_permutation(convoy, mmap)
        assert plan is not None
        assert plan.lateral_tile == Position(10, 4)

        # Schritt 1: LATERAL_STEP_OUT
        step1 = coordinator.execute_permutation_step(convoy, plan)
        assert step1 == PermutationStep.GAP_CLOSURE
        assert alice.position == Position(10, 4)

        # Schritt 2: GAP_CLOSURE
        step2 = coordinator.execute_permutation_step(convoy, plan)
        assert step2 == PermutationStep.RE_INSERTION
        assert bob.position == Position(10, 5)

        # Schritt 3: RE_INSERTION
        step3 = coordinator.execute_permutation_step(convoy, plan)
        assert step3 == PermutationStep.COMPLETED
        # Konvoi-Reihenfolge ist nun getauscht: Bob ist Leader, Alice ist Follower
        assert convoy.members[0].id == "2"
        assert convoy.members[1].id == "1"

    def test_mid_convoy_branching_alternating_and_blockwise(self, coordinator: ConvoyCoordinator) -> None:
        # Konvoi aus 5 Agenten auf y=5 von x=1 bis x=5
        agents = [Agent(id=str(i), name=f"A{i}", position=Position(i, 5)) for i in range(1, 6)]
        convoy = Convoy(id="c5", members=agents, direction_vector=(1, 0))

        # Abzweigung liegt bei A3 (Position 3, 5)
        junc = Position(3, 5)

        # Modus A: Reißverschlussverfahren (Alternating)
        # Leader: A3
        # Tail: [A4, A5]
        # Head: [A2, A1]
        # Erwartet: [A3, A4, A2, A5, A1]
        reconstituted_alt = coordinator.plan_mid_convoy_branching(convoy, junc, mode=DrainMode.ALTERNATING)
        assert [a.id for a in reconstituted_alt] == ["3", "4", "2", "5", "1"]

        # Modus B: Blockwise
        # Erwartet (gleiche Länge 2 == 2 -> Tail zuerst): [A3, A4, A5, A2, A1]
        reconstituted_block = coordinator.plan_mid_convoy_branching(convoy, junc, mode=DrainMode.BLOCKWISE)
        assert [a.id for a in reconstituted_block] == ["3", "4", "5", "2", "1"]

    def test_evaluate_backtracking_selects_closest_tail_to_junction(self, coordinator: ConvoyCoordinator) -> None:
        # Konvoi 1 bewegt sich nach Osten (dx=1), Tail bei x=10
        # Verzweigung rückwärts bei x=7 (Distanz 3)
        c1 = Convoy(
            id="c1",
            members=[Agent(id="1", name="A1", position=Position(12, 5)), Agent(id="2", name="A2", position=Position(10, 5))],
            direction_vector=(1, 0),
        )
        map1 = AgentMentalMap(width=30, height=10)
        for x in range(5, 15):
            map1.update_tile(Position(x, 5), is_walkable=True, tick=1)
        # Kreuzung bei (7, 5): hat Nachbarn nach Norden (7, 4)
        map1.update_tile(Position(7, 4), is_walkable=True, tick=1)

        # Konvoi 2 bewegt sich nach Westen (dx=-1), Tail bei x=20
        # Verzweigung rückwärts bei x=25 (Distanz 5)
        c2 = Convoy(
            id="c2",
            members=[Agent(id="3", name="B1", position=Position(18, 5)), Agent(id="4", name="B2", position=Position(20, 5))],
            direction_vector=(-1, 0),
        )
        map2 = AgentMentalMap(width=30, height=10)
        for x in range(15, 27):
            map2.update_tile(Position(x, 5), is_walkable=True, tick=1)
        map2.update_tile(Position(25, 6), is_walkable=True, tick=1)

        yielding_convoy, target_junc = coordinator.evaluate_backtracking(c1, c2, map1, map2)

        # Konvoi 1 hat Distanz 3 <= 5 -> weicht zurück zu (7, 5)
        assert yielding_convoy.id == "c1"
        assert target_junc == Position(7, 5)
