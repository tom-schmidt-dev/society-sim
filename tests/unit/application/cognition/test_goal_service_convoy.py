from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from src.domain.models.agent.agent import Agent
from src.domain.models.coordination.evasion_phase import EvasionPhase
from src.domain.models.planning.goal import Goal, ExecutionPriority
from src.domain.models.agent.mental_map import FusedMentalMap, AgentMentalMap
from src.domain.models.world.position import Position
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.application.services.cognition.goal_service import GoalService


class TestGoalServiceConvoy:
    def test_suspend_convoy_goals_snapshots_and_inverts_priority(self) -> None:
        mock_logger = MagicMock(spec=IEventLogger)
        mock_cognition = MagicMock(spec=ICognitionProvider)
        mock_pathfinder = MagicMock(spec=IPathfinder)
        current_tick = 50
        service = GoalService(
            logger=mock_logger,
            cognition_provider=mock_cognition,
            pathfinder=mock_pathfinder,
            tick_provider=lambda: current_tick,
        )

        alice = Agent(id="1", name="Alice", position=Position(10, 5))
        alice.assign_path([Position(11, 5), Position(12, 5)])
        alice.push_goal(Goal(name="Ost-Tor", target_position=Position(30, 5)))

        bob = Agent(id="2", name="Bob", position=Position(9, 5))
        bob.assign_path([Position(10, 5), Position(11, 5)])
        bob.push_goal(Goal(name="Ost-Tor-Bob", target_position=Position(30, 5)))

        service.suspend_convoy_goals(
            agents=[alice, bob],
            group_id="group_north",
            participant_ids=["1", "2"],
        )

        for agent in (alice, bob):
            assert agent.evasion_phase == EvasionPhase.SUSPENDED
            assert agent.active_goal is not None
            assert agent.active_goal.name == "ConvoyResolution"
            assert agent.active_goal.priority == ExecutionPriority.URGENT

            # Suspendiertes Ursprungsziel
            orig_goal = agent.goals[0]
            assert orig_goal.status == "suspended"
            assert orig_goal.is_group_goal is True
            assert orig_goal.group_id == "group_north"
            assert orig_goal.participant_ids == ["1", "2"]
            assert orig_goal.state_snapshot is not None
            assert orig_goal.state_snapshot["position"] == (agent.position.x, agent.position.y)
            assert orig_goal.state_snapshot["suspended_at_tick"] == 50

    def test_reevaluate_and_resume_convoy_goals_success(self) -> None:
        mock_logger = MagicMock(spec=IEventLogger)
        mock_cognition = MagicMock(spec=ICognitionProvider)
        mock_pathfinder = MagicMock(spec=IPathfinder)
        service = GoalService(
            logger=mock_logger,
            cognition_provider=mock_cognition,
            pathfinder=mock_pathfinder,
            tick_provider=lambda: 60,
        )

        alice = Agent(id="1", name="Alice", position=Position(10, 5))
        alice.push_goal(Goal(name="Ost-Tor", target_position=Position(30, 5)))
        bob = Agent(id="2", name="Bob", position=Position(9, 5))
        bob.push_goal(Goal(name="Ost-Tor-Bob", target_position=Position(30, 5)))

        service.suspend_convoy_goals([alice, bob], group_id="g1", participant_ids=["1", "2"])

        fused_map = FusedMentalMap(width=50, height=20)
        mock_pathfinder.find_path.side_effect = [
            [Position(11, 5), Position(12, 5)],
            [Position(10, 5), Position(11, 5)],
        ]

        success = service.reevaluate_and_resume_convoy_goals([alice, bob], fused_map)
        assert success is True

        for agent in (alice, bob):
            assert agent.evasion_phase == EvasionPhase.MOVING
            assert agent.active_goal is not None
            assert agent.active_goal.status == "active"
            assert "ConvoyResolution" not in [g.name for g in agent.goals]
            assert agent.has_path is True

    def test_reevaluate_and_resume_convoy_goals_fails_when_unreachable(self) -> None:
        mock_logger = MagicMock(spec=IEventLogger)
        mock_cognition = MagicMock(spec=ICognitionProvider)
        mock_pathfinder = MagicMock(spec=IPathfinder)
        service = GoalService(
            logger=mock_logger,
            cognition_provider=mock_cognition,
            pathfinder=mock_pathfinder,
            tick_provider=lambda: 60,
        )

        alice = Agent(id="1", name="Alice", position=Position(10, 5))
        alice.push_goal(Goal(name="Ost-Tor", target_position=Position(30, 5)))
        service.suspend_convoy_goals([alice], group_id="g1", participant_ids=["1"])

        fused_map = FusedMentalMap(width=50, height=20)
        mock_pathfinder.find_path.return_value = None  # Kein Pfad

        success = service.reevaluate_and_resume_convoy_goals([alice], fused_map)
        assert success is False

    def test_deadlock_protection_timeout_at_1000_ticks(self) -> None:
        mock_logger = MagicMock(spec=IEventLogger)
        mock_cognition = MagicMock(spec=ICognitionProvider)
        mock_pathfinder = MagicMock(spec=IPathfinder)
        tick = 100
        service = GoalService(
            logger=mock_logger,
            cognition_provider=mock_cognition,
            pathfinder=mock_pathfinder,
            tick_provider=lambda: tick,
        )

        alice = Agent(id="1", name="Alice", position=Position(10, 5))
        alice.push_goal(Goal(name="Ost-Tor", target_position=Position(30, 5)))
        service.suspend_convoy_goals([alice], group_id="g1", participant_ids=["1"])

        # Innerhalb der 1000 Ticks (100 + 1000 = 1100)
        tick = 1050
        assert service.check_suspension_timeout(alice) is False
        suspended_g = next(g for g in alice.goals if g.is_group_goal and g.name == "Ost-Tor")
        assert suspended_g.status == "suspended"

        # Überschreitung: 100 + 1001 = 1101
        tick = 1101
        assert service.check_suspension_timeout(alice) is True
        assert suspended_g.status == "abandoned"
        assert alice.active_goal is None or alice.active_goal.name != "ConvoyResolution"
