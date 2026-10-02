import pytest
from unittest.mock import AsyncMock, MagicMock

from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent.agent import Agent
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter


@pytest.fixture
def base_engine_setup():
    grid = WorldGrid(width=10, height=10)
    pathfinder = MagicMock(spec=IPathfinder)
    presenter = MagicMock(spec=IPresenter)
    logger = MagicMock(spec=IEventLogger)
    cognition_provider = MagicMock(spec=ICognitionProvider)

    return grid, pathfinder, presenter, logger, cognition_provider


@pytest.mark.asyncio
async def test_day_night_disabled_prevents_night_phase_and_sleep(base_engine_setup) -> None:
    grid, pathfinder, presenter, logger, cognition_provider = base_engine_setup

    engine = SimulationEngine(
        grid=grid,
        pathfinder=pathfinder,
        presenter=presenter,
        logger=logger,
        cognition_provider=cognition_provider,
        enable_day_night=False,
    )

    agent = Agent(id="1", name="Alice", position=Position(1, 1))
    engine.register_agent(agent)

    # Setze Tick direkt in die reguläre Nachtphase (standardmäßig Tick 100 bis 119)
    engine._current_tick = 99

    await engine.process_tick()

    assert engine.current_tick == 100
    # Bei deaktiviertem Tag-Nacht-Zyklus darf der Agent nicht schlafen
    assert agent.is_sleeping is False

    # Es darf kein night_started-Event geloggt worden sein
    logged_event_types = [call[0][0].event_type for call in logger.log.call_args_list]
    assert "night_started" not in logged_event_types


@pytest.mark.asyncio
async def test_day_night_enabled_triggers_night_phase(base_engine_setup) -> None:
    grid, pathfinder, presenter, logger, cognition_provider = base_engine_setup

    engine = SimulationEngine(
        grid=grid,
        pathfinder=pathfinder,
        presenter=presenter,
        logger=logger,
        cognition_provider=cognition_provider,
        enable_day_night=True,
    )

    agent = Agent(id="1", name="Alice", position=Position(1, 1))
    engine.register_agent(agent)

    engine._current_tick = 99

    await engine.process_tick()

    assert engine.current_tick == 100
    # Bei aktiviertem Tag-Nacht-Zyklus tritt die Nacht regulär ein
    assert agent.is_sleeping is True

    logged_event_types = [call[0][0].event_type for call in logger.log.call_args_list]
    assert "night_started" in logged_event_types