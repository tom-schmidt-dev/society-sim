from __future__ import annotations

import pytest
from src.domain.models.agent import Agent
from src.domain.models.evasion_phase import EvasionPhase
from src.domain.models.goal import Goal, ExecutionPriority
from src.domain.models.mental_map import AgentMentalMap, FusedMentalMap, TileKnowledge
from src.domain.models.position import Position


class TestDomainConvoyModels:
    def test_evasion_phase_enum_values(self) -> None:
        expected_phases = {
            "idle",
            "moving",
            "conflict_detected",
            "negotiating",
            "yielding_ingress",
            "yielding_wait",
            "passing",
            "clearance_confirmed",
            "egress",
            "reconstitution",
            "suspended",
        }
        actual_phases = {phase.value for phase in EvasionPhase}
        assert actual_phases == expected_phases

    def test_agent_defaults_and_trait_bounds(self) -> None:
        agent = Agent(id="a1", name="Alice", position=Position(5, 5))
        assert agent.footprint == (1, 1)
        assert agent.charisma == 0.0
        assert agent.assertiveness == 0.0
        assert agent.evasion_phase == EvasionPhase.IDLE

        # Valid custom traits
        agent_custom = Agent(
            id="a2",
            name="Bob",
            position=Position(5, 6),
            footprint=(2, 1),
            charisma=0.75,
            assertiveness=0.3,
            evasion_phase=EvasionPhase.NEGOTIATING,
        )
        assert agent_custom.footprint == (2, 1)
        assert agent_custom.charisma == 0.75
        assert agent_custom.assertiveness == 0.3
        assert agent_custom.evasion_phase == EvasionPhase.NEGOTIATING

        # Invalid charisma
        with pytest.raises(ValueError, match="charisma"):
            Agent(id="err1", name="Err", position=Position(0, 0), charisma=1.5)
        with pytest.raises(ValueError, match="charisma"):
            Agent(id="err2", name="Err", position=Position(0, 0), charisma=-0.1)

        # Invalid assertiveness
        with pytest.raises(ValueError, match="assertiveness"):
            Agent(id="err3", name="Err", position=Position(0, 0), assertiveness=2.0)
        with pytest.raises(ValueError, match="assertiveness"):
            Agent(id="err4", name="Err", position=Position(0, 0), assertiveness=-0.5)

        # Invalid footprint
        with pytest.raises(ValueError, match="footprint"):
            Agent(id="err5", name="Err", position=Position(0, 0), footprint=(0, 1))

    def test_goal_statuses_and_group_fields(self) -> None:
        goal_suspended = Goal(
            name="ConvoyWait",
            target_position=Position(10, 20),
            status="suspended",
            priority=ExecutionPriority.URGENT,
            is_group_goal=True,
            group_id="convoy_alpha",
            participant_ids=["a1", "a2"],
            state_snapshot={"active_step": 3, "interrupted_at_tick": 42},
            backtracking_junction_target=Position(8, 20),
        )
        assert goal_suspended.status == "suspended"
        assert goal_suspended.is_group_goal is True
        assert goal_suspended.group_id == "convoy_alpha"
        assert goal_suspended.participant_ids == ["a1", "a2"]
        assert goal_suspended.state_snapshot == {"active_step": 3, "interrupted_at_tick": 42}
        assert goal_suspended.backtracking_junction_target == Position(8, 20)

        goal_dict = goal_suspended.to_dict()
        assert goal_dict["status"] == "suspended"
        assert goal_dict["is_group_goal"] is True
        assert goal_dict["group_id"] == "convoy_alpha"
        assert goal_dict["participant_ids"] == ["a1", "a2"]
        assert goal_dict["state_snapshot"] == {"active_step": 3, "interrupted_at_tick": 42}
        assert goal_dict["backtracking_junction_target"] == (8, 20)

        # Re-evaluating status
        goal_reeval = Goal(name="RouteReeval", status="re_evaluating")
        assert goal_reeval.status == "re_evaluating"

    def test_fused_mental_map_timestamp_and_precautionary_principle(self) -> None:
        map_a = AgentMentalMap(width=20, height=20)
        map_b = AgentMentalMap(width=20, height=20)

        # Fall 1: Unterschiedliche Zeitstempel -> Neuerer Zeitstempel gewinnt
        pos1 = Position(3, 3)
        map_a.update_tile(pos1, is_walkable=True, tick=10)
        map_b.update_tile(pos1, is_walkable=False, tick=15)  # Neuer

        # Fall 2: Identische Zeitstempel, Konflikt -> Vorsichtsprinzip: OBSTACLE > WALKABLE
        pos2 = Position(4, 4)
        map_a.update_tile(pos2, is_walkable=True, tick=20)
        map_b.update_tile(pos2, is_walkable=False, tick=20)  # Gleich alt, Hindernis hat Vorrang

        # Fall 3: Einseitiges Wissen
        pos3 = Position(5, 5)
        map_a.update_tile(pos3, is_walkable=True, tick=5)

        fused = FusedMentalMap.fuse([map_a, map_b])

        # pos1: map_b (tick 15) gewinnt
        assert fused.tiles[pos1].knowledge == TileKnowledge.OBSTACLE
        assert fused.tiles[pos1].last_observed_tick == 15

        # pos2: Vorsichtsprinzip greift bei tick 20
        assert fused.tiles[pos2].knowledge == TileKnowledge.OBSTACLE
        assert fused.tiles[pos2].last_observed_tick == 20

        # pos3: Von map_a übernommen
        assert fused.tiles[pos3].knowledge == TileKnowledge.WALKABLE
        assert fused.tiles[pos3].last_observed_tick == 5

    def test_fused_mental_map_epistemic_persistence_excludes_dynamic_entities(self) -> None:
        map_a = AgentMentalMap(width=20, height=20)
        map_b = AgentMentalMap(width=20, height=20)

        # Statische Wand
        wall_pos = Position(10, 10)
        map_a.update_tile(wall_pos, is_walkable=False, tick=1, is_static=True)

        # Dynamischer Agent B steht auf (10, 11)
        agent_b_pos = Position(10, 11)
        map_a.update_tile(agent_b_pos, is_walkable=False, tick=2, is_static=False)

        # Fusion unter Deklaration von agent_b_pos als dynamisch
        fused = FusedMentalMap.fuse([map_a, map_b], dynamic_positions={agent_b_pos})

        assert fused.tiles[wall_pos].is_static is True
        assert fused.tiles[agent_b_pos].is_static is False

        # Persistenz auf eine neue / uninformierte Karte
        map_c = AgentMentalMap(width=20, height=20)
        fused.persist_to_agent_maps([map_c])

        # Die statische Wand MUSS in map_c existieren
        assert wall_pos in map_c.tiles
        assert map_c.tiles[wall_pos].knowledge == TileKnowledge.OBSTACLE

        # Die dynamische Agentenbelegung DARF NICHT in map_c übernommen werden
        assert agent_b_pos not in map_c.tiles
