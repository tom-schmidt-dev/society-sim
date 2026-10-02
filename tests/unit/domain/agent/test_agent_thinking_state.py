import pytest
from src.domain.models.agent.agent import Agent
from src.domain.models.world.position import Position


def test_agent_thinking_state_default() -> None:
    agent = Agent(
        id="1",
        name="Alice",
        position=Position(0, 0),
    )
    assert agent.is_thinking is False
    assert agent.thinking_reason is None


def test_agent_set_thinking_with_reason() -> None:
    agent = Agent(
        id="1",
        name="Alice",
        position=Position(0, 0),
    )

    agent.set_thinking(True, reason="Wägt Ausweichen für Bob ab")
    assert agent.is_thinking is True
    assert agent.thinking_reason == "Wägt Ausweichen für Bob ab"

    agent.set_thinking(False)
    assert agent.is_thinking is False
    assert agent.thinking_reason is None


def test_agent_set_thinking_without_explicit_reason() -> None:
    agent = Agent(
        id="1",
        name="Alice",
        position=Position(0, 0),
    )

    agent.set_thinking(True)
    assert agent.is_thinking is True
    assert agent.thinking_reason is None