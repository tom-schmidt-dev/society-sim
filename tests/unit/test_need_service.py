from __future__ import annotations

import pytest
from src.domain.models.agent import Agent
from src.domain.models.position import Position
from src.application.services.need_service import NeedService


def test_need_service_increases_hunger_per_tick():
    agent = Agent(id="agent_1", name="Alice", position=Position(0, 0))
    agent.needs["hunger"] = 0.1

    service = NeedService(hunger_increase_per_tick=0.05, hunger_threshold=0.7)
    service.update_needs(agent)

    assert pytest.approx(agent.needs["hunger"], 0.001) == 0.15


def test_need_service_detects_urgent_need():
    agent = Agent(id="agent_1", name="Alice", position=Position(0, 0))
    agent.needs["hunger"] = 0.75

    service = NeedService(hunger_threshold=0.7)
    assert service.is_need_urgent(agent, "hunger") is True


def test_need_service_resets_need_on_consumption():
    agent = Agent(id="agent_1", name="Alice", position=Position(0, 0))
    agent.needs["hunger"] = 0.85

    service = NeedService()
    service.satisfy_need(agent, "hunger", reduction=0.5)

    assert pytest.approx(agent.needs["hunger"], 0.001) == 0.35