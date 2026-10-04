from __future__ import annotations

from unittest.mock import AsyncMock
import pytest

from src.domain.ports.cognition_provider import ICognitionProvider
from src.infrastructure.container import ApplicationContainer
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.goal import Goal, ExecutionPriority
from src.domain.models.agent.mental_map import AgentMentalMap
from src.domain.models.world.position import Position


class TestConvoyCorridorScenarios:
    @pytest.mark.asyncio
    async def test_topological_clearance_corridor_zone_exclusion(self) -> None:
        mock_cognition = AsyncMock(spec=ICognitionProvider)
        container = ApplicationContainer.build(
            width=30,
            height=10,
            tick_interval=0.0,
            cognition_provider=mock_cognition,
        )

        for x in range(2, 21):
            container.grid.set_obstacle(Position(x, 4), "wall")
            if x != 10:
                container.grid.set_obstacle(Position(x, 6), "wall")

        alice = Agent(id="alice", name="Alice", position=Position(12, 5))
        alice.path = []
        container.engine.register_agent(alice)

        bob = Agent(id="bob", name="Bob", position=Position(10, 7))
        bob.is_evasion_locked = True
        bob_goal = Goal(
            name="EvasionHold",
            priority=ExecutionPriority.URGENT,
            junction_position=Position(10, 5),
            yield_for_agent_id="alice",
            is_evasion_hold=True,
        )
        bob.push_goal(bob_goal)
        container.engine.register_agent(bob)

        alice.position = Position(11, 5)
        container.engine._check_and_signal_clearance(alice)

        assert bob.active_goal is not None
        assert bob.active_goal.yield_for_agent_id == "alice"
        assert bob.is_evasion_locked is True

        alice.position = Position(12, 5)
        container.engine._check_and_signal_clearance(alice)

        assert bob.active_goal is not None
        assert bob.active_goal.yield_for_agent_id is None
        recent = container.dialogue_history.get_recent_formatted(5)
        assert any("Danke fürs Platz machen!" in entry for entry in recent)

    @pytest.mark.asyncio
    async def test_topological_clearance_outside_corridor_at_destination(self) -> None:
        mock_cognition = AsyncMock(spec=ICognitionProvider)
        container = ApplicationContainer.build(
            width=30,
            height=10,
            tick_interval=0.0,
            cognition_provider=mock_cognition,
        )
        alice = Agent(id="alice", name="Alice", position=Position(11, 5))
        alice.push_goal(Goal(name="Ziel", target_position=Position(11, 5)))
        alice.path = []

        bob = Agent(id="bob", name="Bob", position=Position(10, 6))
        bob.is_evasion_locked = True
        bob_goal = Goal(
            name="EvasionHold",
            priority=ExecutionPriority.URGENT,
            junction_position=Position(10, 5),
            yield_for_agent_id="alice",
            is_evasion_hold=True,
        )
        bob.push_goal(bob_goal)

        container.engine.register_agent(alice)
        container.engine.register_agent(bob)

        assert not container.engine._is_in_corridor_zone(alice.position)

        container.engine._check_and_signal_clearance(alice)

        assert bob.active_goal is not None
        assert bob.active_goal.yield_for_agent_id is None
        recent = container.dialogue_history.get_recent_formatted(5)
        assert any("Danke fürs Platz machen!" in entry for entry in recent)

    @pytest.mark.asyncio
    async def test_convoy_synchronized_movement_in_engine(self) -> None:
        mock_cognition = AsyncMock(spec=ICognitionProvider)
        container = ApplicationContainer.build(
            width=30,
            height=10,
            tick_interval=0.0,
            cognition_provider=mock_cognition,
        )

        a1 = Agent(id="a1", name="A1", position=Position(5, 5))
        a2 = Agent(id="a2", name="A2", position=Position(4, 5))
        a3 = Agent(id="a3", name="A3", position=Position(3, 5))

        a1.assign_path([Position(6, 5), Position(7, 5), Position(8, 5)])
        a2.assign_path([Position(5, 5), Position(6, 5), Position(7, 5)])
        a3.assign_path([Position(4, 5), Position(5, 5), Position(6, 5)])

        container.engine.register_agent(a1)
        container.engine.register_agent(a2)
        container.engine.register_agent(a3)

        for _ in range(3):
            await container.engine.process_tick()

        assert a1.position == Position(8, 5)
        assert a2.position == Position(7, 5)
        assert a3.position == Position(6, 5)
        assert a1.path == []
        assert a2.path == []
        assert a3.path == []

    @pytest.mark.asyncio
    async def test_convoy_meets_agent_arbitration_and_niche_packing(self) -> None:
        mock_cognition = AsyncMock(spec=ICognitionProvider)
        container = ApplicationContainer.build(
            width=30,
            height=10,
            tick_interval=0.0,
            cognition_provider=mock_cognition,
        )

        for x in range(1, 21):
            container.grid.set_obstacle(Position(x, 4), "wall")
            if x != 10:
                container.grid.set_obstacle(Position(x, 6), "wall")

        c1 = Agent(id="c1", name="Leader", position=Position(6, 5))
        c1.assign_path([Position(x, 5) for x in range(7, 20)])
        c2 = Agent(id="c2", name="Follower", position=Position(5, 5))
        c2.assign_path([Position(x, 5) for x in range(6, 20)])

        bob = Agent(id="bob", name="Bob", position=Position(14, 5))
        bob.assign_path([Position(x, 5) for x in range(13, 0, -1)])

        convoys = container.engine.convoy_coordinator.identify_convoys([c1, c2, bob])
        assert len(convoys) == 2
        c_convoy = next(c for c in convoys if len(c.members) == 2)
        assert c_convoy.leader.id == "c1"

        mmap = AgentMentalMap(width=30, height=10)
        for x in range(1, 21):
            mmap.update_tile(Position(x, 5), is_walkable=True, tick=1)
        mmap.update_tile(Position(10, 6), is_walkable=True, tick=1)

        c_trajectory = [Position(x, 5) for x in range(1, 21)]
        bob_trajectory = [Position(x, 5) for x in range(1, 21)]

        c_niche = container.engine.niche_packer.find_niche_configuration(
            convoy=c_convoy,
            mental_map=mmap,
            passing_convoy_trajectory=bob_trajectory,
        )
        assert c_niche is None

        bob_convoy = next(c for c in convoys if c.leader.id == "bob")
        bob_niche = container.engine.niche_packer.find_niche_configuration(
            convoy=bob_convoy,
            mental_map=mmap,
            passing_convoy_trajectory=c_trajectory,
        )
        assert bob_niche is not None
        assert bob_niche.total_capacity == 1
        assert bob_niche.slots[0].slot_position == Position(10, 6)

        from src.application.services.coordination.convoy_arbitrator import ConvoyArbitrator, ArbitrationOutcome
        arbitrator = ConvoyArbitrator()
        outcome = arbitrator.arbitrate(
            group_1_agents=c_convoy.members,
            group_2_agents=bob_convoy.members,
            delta_c_1=None,
            delta_c_2=bob_niche.additional_cost,
            tick=1,
        )
        assert outcome.outcome == ArbitrationOutcome.GROUP_2_YIELDS