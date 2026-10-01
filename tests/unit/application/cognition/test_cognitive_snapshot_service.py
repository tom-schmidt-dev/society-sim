from __future__ import annotations

import pytest

from src.application.services.cognition.cognitive_snapshot_service import CognitiveSnapshotService
from src.application.services.lifecycle.need_service import NeedService
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.goal import Goal
from src.domain.models.world.position import Position
from src.domain.models.world.world_entity import WorldEntity


class TestCognitiveSnapshotService:
    @pytest.fixture
    def service(self) -> CognitiveSnapshotService:
        need_service = NeedService()
        return CognitiveSnapshotService(need_service=need_service)

    def test_idle_agent_thought(self, service: CognitiveSnapshotService) -> None:
        agent = Agent(id="a1", name="Alice", position=Position(1, 1))
        snapshot = service.create_snapshot(agent, tick=5)

        assert snapshot.agent_id == "a1"
        assert snapshot.tick == 5
        assert snapshot.primary_goal is None
        assert snapshot.intended_strategy == "Bereitschaft"
        assert "keine Aufgaben vor" in snapshot.formatted_thought

    def test_navigation_with_primary_goal(self, service: CognitiveSnapshotService) -> None:
        agent = Agent(id="a1", name="Alice", position=Position(1, 1))
        agent.goals.append(Goal(name="Ost-Tor", target_position=Position(10, 1)))
        agent.assign_path([Position(2, 1), Position(3, 1)])

        snapshot = service.create_snapshot(agent, tick=2)

        assert snapshot.primary_goal == "Ost-Tor"
        assert snapshot.intended_strategy == "Pfadnavigation"
        assert "Mein Ziel ist 'Ost-Tor'" in snapshot.formatted_thought

    def test_blocked_path_thought(self, service: CognitiveSnapshotService) -> None:
        agent = Agent(id="a1", name="Alice", position=Position(1, 1))
        agent.goals.append(Goal(name="Ost-Tor", target_position=Position(10, 1)))
        agent.assign_path([Position(2, 1), Position(3, 1)])

        blocker = WorldEntity(id="b1", name="Bob", position=Position(2, 1))
        snapshot = service.create_snapshot(agent, tick=3, all_entities=[blocker])

        assert snapshot.perceived_obstacle == "Bob auf Kachel (2, 1)"
        assert "Lösungsansatz: Konfliktkoordination" in snapshot.formatted_thought

    def test_evasion_hold_thought(self, service: CognitiveSnapshotService) -> None:
        agent = Agent(id="a1", name="Alice", position=Position(4, 5))
        agent.goals.append(
            Goal(
                name="In Nische halten",
                target_position=Position(4, 5),
                is_evasion_hold=True,
                yield_for_agent_id="b1",
            )
        )

        snapshot = service.create_snapshot(agent, tick=12)

        assert snapshot.intended_strategy == "Defensives Halten in Nische"
        assert "b1 die Durchfahrt zu gewähren" in snapshot.formatted_thought

    def test_subgoal_consume_thought(self, service: CognitiveSnapshotService) -> None:
        agent = Agent(id="a1", name="Alice", position=Position(2, 2))
        agent.needs["hunger"] = 0.85
        agent.goals.append(Goal(name="Nahrungsbeschaffung"))
        agent.goals.append(Goal(name="SubGoal: consume"))

        snapshot = service.create_snapshot(agent, tick=20)

        assert snapshot.intended_strategy == "Nahrungsaufnahme"
        assert "verzehre die Nahrung" in snapshot.formatted_thought
        assert snapshot.dominant_need == "hunger"