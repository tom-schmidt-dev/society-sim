from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_memory import EntityFact
from src.domain.models.planning.goal import Goal
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter


@pytest.fixture
def base_engine_setup():
    grid = WorldGrid(width=15, height=15)
    for x in range(15):
        for y in range(15):
            grid.remove_obstacle(Position(x, y))

    pathfinder = MagicMock(spec=IPathfinder)
    pathfinder.find_path.return_value = [Position(2, 2)]
    presenter = MagicMock(spec=IPresenter)
    logger = MagicMock(spec=IEventLogger)
    cognition = AsyncMock(spec=ICognitionProvider)

    engine = SimulationEngine(
        grid=grid,
        pathfinder=pathfinder,
        presenter=presenter,
        logger=logger,
        cognition_provider=cognition,
        tick_interval=0.0,
    )
    return engine, grid, presenter, logger


class TestSimulationEngineValidationAndProperties:
    def test_register_entity_raises_on_blocked_tile(self, base_engine_setup) -> None:
        engine, grid, _, _ = base_engine_setup
        grid.set_obstacle(Position(5, 5))
        entity = WorldEntity(id="box_1", name="Kiste", position=Position(5, 5))

        with pytest.raises(ValueError, match="Position Position\\(x=5, y=5\\) für Objekt Kiste ist blockiert."):
            engine.register_entity(entity)

    def test_register_agent_raises_on_blocked_tile(self, base_engine_setup) -> None:
        engine, grid, _, _ = base_engine_setup
        grid.set_obstacle(Position(3, 3))
        agent = Agent(id="a1", name="Alice", position=Position(3, 3))

        with pytest.raises(ValueError, match="Startposition Position\\(x=3, y=3\\) für Agent Alice blockiert."):
            engine.register_agent(agent)

    def test_set_agent_target_raises_for_unknown_agent(self, base_engine_setup) -> None:
        engine, _, _, _ = base_engine_setup
        with pytest.raises(ValueError, match="Agent 'unknown' nicht gefunden."):
            engine.set_agent_target("unknown", target=Position(1, 1))

    def test_properties_expose_subservices(self, base_engine_setup) -> None:
        engine, _, _, _ = base_engine_setup
        assert engine.niche_packer is not None
        assert engine.convoy_arbitrator is not None
        assert engine.convoy_coordinator is not None
        assert engine.movement_sync_service is not None
        assert engine.critical_section_coordinator is not None
        assert engine.dialogue_history is not None


class TestSimulationEngineLifecycleAndReplan:
    def test_interrupt_goal_for_replan_logs_discovered_fact_payload(self, base_engine_setup) -> None:
        engine, _, _, logger = base_engine_setup
        agent = Agent(id="1", name="Alice", position=Position(1, 1))
        engine.register_agent(agent)
        agent.push_goal(Goal(name="Altes Ziel", target_position=Position(5, 5)))
        agent.assign_path([Position(2, 1), Position(3, 1)])

        fact = EntityFact(
            entity_id="food_1",
            name="Apfel",
            last_known_position=Position(4, 4),
            entity_type="food",
        )

        engine._interrupt_goal_for_replan(agent, reason="Ressource entdeckt", discovered_fact=fact)

        assert not agent.goals
        assert not agent.has_path
        event = next(e for e in [call_args[0][0] for call_args in logger.log.call_args_list] if e.event_type == "goal_interrupted_for_replan")
        assert event.payload["discovered_entity_id"] == "food_1"
        assert event.payload["entity_type"] == "food"
        assert event.payload["position"] == [4, 4]

    @pytest.mark.asyncio
    async def test_run_terminates_when_all_agents_idle(self, base_engine_setup) -> None:
        engine, _, presenter, _ = base_engine_setup
        agent = Agent(id="1", name="Alice", position=Position(1, 1))
        engine.register_agent(agent)

        await engine.run(max_ticks=10)

        assert engine.current_tick == 1
        assert presenter.render.call_count >= 2
        assert engine._is_running is False

    @pytest.mark.asyncio
    async def test_run_cleans_up_background_tasks(self, base_engine_setup) -> None:
        engine, _, _, _ = base_engine_setup
        agent = Agent(id="1", name="Alice", position=Position(1, 1))
        engine.register_agent(agent)

        task_executed = False

        async def dummy_task() -> None:
            nonlocal task_executed
            await asyncio.sleep(0.001)
            task_executed = True

        bg_task = asyncio.create_task(dummy_task())
        engine._background_tasks.add(bg_task)

        await engine.run(max_ticks=2)

        assert task_executed is True
        assert engine._is_running is False


class TestSimulationEngineDelegationWrappers:
    def test_delegation_wrappers(self, base_engine_setup) -> None:
        engine, _, _, _ = base_engine_setup
        agent = Agent(id="1", name="Alice", position=Position(1, 1))
        engine.register_agent(agent)

        protocol_mock = MagicMock()
        engine._protocol_service = protocol_mock

        engine._check_and_signal_clearance(agent)
        protocol_mock.check_and_signal_clearance.assert_called_once_with(agent, engine._entities)

        goal = Goal(name="Nische", junction_position=Position(1, 2))
        engine._handle_niche_arrival(agent, goal)
        protocol_mock.handle_niche_arrival.assert_called_once_with(agent, goal, engine._entities)

        forbidden = {Position(2, 2)}
        engine._replan_agent_path_avoiding(agent, forbidden)
        protocol_mock.replan_agent_path_avoiding.assert_called_once_with(agent, forbidden)

        protocol_mock.is_in_corridor_zone.return_value = True
        assert engine._is_in_corridor_zone(Position(1, 1)) is True
        protocol_mock.is_in_corridor_zone.assert_called_once_with(Position(1, 1))