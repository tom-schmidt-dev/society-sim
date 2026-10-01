from __future__ import annotations

import pytest

from src.application.services.need_service import NeedService
from src.domain.models.agent import Agent
from src.domain.models.position import Position


class TestNeedService:
    def test_default_agent_has_all_vital_needs_initialized(self) -> None:
        agent = Agent(id="a1", name="Alice", position=Position(0, 0))
        assert "hunger" in agent.needs
        assert "thirst" in agent.needs
        assert "energy" in agent.needs
        assert agent.needs["hunger"] == 0.0
        assert agent.needs["thirst"] == 0.0
        assert agent.needs["energy"] == 0.0

    def test_backwards_compatible_init_properties(self) -> None:
        service = NeedService(hunger_increase_per_tick=0.03, hunger_threshold=0.85)
        assert service.hunger_increase_per_tick == 0.03
        assert service.hunger_threshold == 0.85

    def test_update_needs_increases_all_vital_deficits(self) -> None:
        service = NeedService(
            hunger_increase_per_tick=0.01,
            thirst_increase_per_tick=0.02,
            energy_increase_per_tick=0.005,
        )
        agent = Agent(id="a1", name="Alice", position=Position(0, 0))

        service.update_needs(agent)

        assert pytest.approx(agent.needs["hunger"], rel=1e-3) == 0.01
        assert pytest.approx(agent.needs["thirst"], rel=1e-3) == 0.02
        assert pytest.approx(agent.needs["energy"], rel=1e-3) == 0.005

    def test_update_needs_clamps_at_maximum_one(self) -> None:
        service = NeedService(hunger_increase_per_tick=0.5)
        agent = Agent(id="a1", name="Alice", position=Position(0, 0))
        agent.needs["hunger"] = 0.8

        service.update_needs(agent)
        assert agent.needs["hunger"] == 1.0

        service.update_needs(agent)
        assert agent.needs["hunger"] == 1.0

    def test_is_need_urgent_respects_individual_thresholds(self) -> None:
        service = NeedService(
            thresholds={"hunger": 0.7, "thirst": 0.6, "energy": 0.8}
        )
        agent = Agent(id="a1", name="Alice", position=Position(0, 0))
        agent.needs["thirst"] = 0.65
        agent.needs["hunger"] = 0.65

        assert service.is_need_urgent(agent, "thirst") is True
        assert service.is_need_urgent(agent, "hunger") is False
        assert service.is_any_need_urgent(agent) is True

    def test_get_dominant_need_prioritizes_higher_weighted_need(self) -> None:
        service = NeedService(
            thresholds={"hunger": 0.7, "thirst": 0.7, "energy": 0.7},
            weights={"hunger": 1.0, "thirst": 1.2, "energy": 0.8},
        )
        agent = Agent(id="a1", name="Alice", position=Position(0, 0))
        agent.needs["hunger"] = 0.75
        agent.needs["thirst"] = 0.75

        dominant = service.get_dominant_need(agent)
        assert dominant == "thirst"

    def test_get_dominant_need_returns_none_if_no_need_is_urgent(self) -> None:
        service = NeedService(hunger_threshold=0.7)
        agent = Agent(id="a1", name="Alice", position=Position(0, 0))
        agent.needs["hunger"] = 0.5
        agent.needs["thirst"] = 0.4
        agent.needs["energy"] = 0.3

        assert service.is_any_need_urgent(agent) is False
        assert service.get_dominant_need(agent) is None

    def test_get_dominant_need_can_ignore_urgency_if_requested(self) -> None:
        service = NeedService(
            weights={"hunger": 1.0, "thirst": 1.2, "energy": 0.8}
        )
        agent = Agent(id="a1", name="Alice", position=Position(0, 0))
        agent.needs["hunger"] = 0.5
        agent.needs["thirst"] = 0.4

        assert service.get_dominant_need(agent, only_urgent=False) == "hunger"

    def test_satisfy_need_reduces_deficit_and_clamps_at_zero(self) -> None:
        service = NeedService()
        agent = Agent(id="a1", name="Alice", position=Position(0, 0))
        agent.needs["thirst"] = 0.5

        service.satisfy_need(agent, "thirst", 0.3)
        assert pytest.approx(agent.needs["thirst"], rel=1e-3) == 0.2

        service.satisfy_need(agent, "thirst", 0.5)
        assert agent.needs["thirst"] == 0.0