from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from src.application.services.lifecycle.day_night_service import DayNightService
from src.application.services.lifecycle.need_service import NeedService
from src.domain.models.agent.agent import Agent
from src.domain.models.world.position import Position
from src.domain.ports.event_logger import IEventLogger


class TestDayNightService:
    def test_init_raises_on_invalid_cycle_parameters(self) -> None:
        with pytest.raises(ValueError):
            DayNightService(day_ticks=0, night_ticks=20)
        with pytest.raises(ValueError):
            DayNightService(day_ticks=100, night_ticks=-5)

    def test_phase_detection(self) -> None:
        service = DayNightService(day_ticks=100, night_ticks=20)

        assert service.is_day(0) is True
        assert service.is_night(0) is False

        assert service.is_day(99) is True
        assert service.is_night(99) is False

        assert service.is_day(100) is False
        assert service.is_night(100) is True

        assert service.is_day(119) is False
        assert service.is_night(119) is True

        # Neuer Zyklus ab Takt 120
        assert service.is_day(120) is True
        assert service.is_night(120) is False

    def test_day_number_and_progress(self) -> None:
        service = DayNightService(day_ticks=100, night_ticks=20)

        assert service.get_day_number(0) == 1
        assert service.get_day_number(119) == 1
        assert service.get_day_number(120) == 2

        assert pytest.approx(service.get_phase_progress(50)) == 0.5
        assert pytest.approx(service.get_phase_progress(100)) == 0.0
        assert pytest.approx(service.get_phase_progress(110)) == 0.5

    def test_process_tick_transitions_and_logging(self) -> None:
        mock_logger = MagicMock(spec=IEventLogger)
        service = DayNightService(day_ticks=10, night_ticks=5, logger=mock_logger)
        agent = Agent(id="a1", name="Alice", position=Position(0, 0))

        # Regulärer Tagestakt
        event = service.process_tick(5, [agent])
        assert event is None
        assert agent.is_sleeping is False

        # Übergang in die Nacht bei Takt 10
        event = service.process_tick(10, [agent])
        assert event == "night_started"
        assert agent.is_sleeping is True
        assert agent.is_busy is True
        mock_logger.log.assert_called_once()
        assert mock_logger.log.call_args[0][0].event_type == "night_started"

        # Übergang zurück zum Tag bei Takt 15
        mock_logger.reset_mock()
        event = service.process_tick(15, [agent])
        assert event == "day_started"
        assert agent.is_sleeping is False
        assert agent.is_busy is False
        mock_logger.log.assert_called_once()
        assert mock_logger.log.call_args[0][0].event_type == "day_started"

    def test_need_service_regenerates_energy_during_sleep(self) -> None:
        need_service = NeedService()
        agent = Agent(id="a1", name="Alice", position=Position(0, 0))
        agent.needs["energy"] = 0.5
        agent.needs["hunger"] = 0.2
        agent.is_sleeping = True

        need_service.update_needs(agent)

        # Energie regeneriert
        assert pytest.approx(agent.needs["energy"], rel=1e-3) == 0.47
        # Hunger steigt mit halber Rate (0.01 * 0.5 = 0.005)
        assert pytest.approx(agent.needs["hunger"], rel=1e-3) == 0.205