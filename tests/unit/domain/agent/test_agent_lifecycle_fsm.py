from __future__ import annotations
from dataclasses import fields
import pytest

from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_state import AgentLifecycleState, InvalidStateTransitionError
from src.domain.models.world.position import Position


def test_agent_initial_lifecycle_state() -> None:
    agent = Agent(id="1", name="Alice", position=Position(2, 4))
    assert agent.lifecycle_state == AgentLifecycleState.IDLE
    assert not agent.is_thinking
    assert agent.thinking_reason is None


def test_agent_state_transitions_valid() -> None:
    agent = Agent(id="1", name="Alice", position=Position(2, 4))

    # IDLE -> DELIBERATING
    agent.transition_to(AgentLifecycleState.DELIBERATING, reason="Verhandle Chokepoint")
    assert agent.lifecycle_state == AgentLifecycleState.DELIBERATING
    assert agent.is_thinking is True
    assert agent.thinking_reason == "Verhandle Chokepoint"
    assert agent.is_busy is True

    # DELIBERATING -> WAITING_FOR_PEER
    agent.transition_to(AgentLifecycleState.WAITING_FOR_PEER)
    assert agent.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER
    assert agent.is_thinking is False
    assert agent.thinking_reason is None
    assert agent.is_busy is True

    # WAITING_FOR_PEER -> DELIBERATING (Antwort eingetroffen)
    agent.transition_to(AgentLifecycleState.DELIBERATING, reason="Verarbeite Antwort")
    assert agent.lifecycle_state == AgentLifecycleState.DELIBERATING
    assert agent.is_thinking is True

    # DELIBERATING -> YIELDING
    agent.transition_to(AgentLifecycleState.YIELDING)
    assert agent.lifecycle_state == AgentLifecycleState.YIELDING
    assert agent.is_thinking is False

    # YIELDING -> IDLE
    agent.transition_to(AgentLifecycleState.IDLE)
    assert agent.lifecycle_state == AgentLifecycleState.IDLE
    assert agent.is_busy is False


def test_agent_state_transitions_invalid_raises_error() -> None:
    agent = Agent(id="1", name="Alice", position=Position(2, 4))
    agent.transition_to(AgentLifecycleState.WAITING_FOR_PEER)

    # Direkter Sprung von WAITING_FOR_PEER zu YIELDING ohne Kognitionsauflösung ist verboten
    with pytest.raises(InvalidStateTransitionError):
        agent.transition_to(AgentLifecycleState.YIELDING)

    # Direkter Sprung von WAITING_FOR_PEER zu PASSING ist verboten
    with pytest.raises(InvalidStateTransitionError):
        agent.transition_to(AgentLifecycleState.PASSING)


def test_set_thinking_syncs_with_lifecycle_state() -> None:
    agent = Agent(id="1", name="Alice", position=Position(2, 4))

    agent.set_thinking(True, reason="Suche Weg")
    assert agent.lifecycle_state == AgentLifecycleState.DELIBERATING
    assert agent.is_thinking is True
    assert agent.thinking_reason == "Suche Weg"

    agent.set_thinking(False)
    assert agent.lifecycle_state == AgentLifecycleState.IDLE
    assert agent.is_thinking is False
    assert agent.thinking_reason is None


def test_set_thinking_method_protection() -> None:
    agent = Agent(id="1", name="Alice", position=Position(2, 4))

    # Defensive Absicherung: Überschreiben der Methode mit einem Boolean wirft AttributeError
    with pytest.raises(AttributeError):
        agent.set_thinking = False  # type: ignore[assignment]


def test_is_thinking_assignment_syncs_lifecycle_state() -> None:
    agent = Agent(id="1", name="Alice", position=Position(2, 4))

    # Rückwärtskompatibilität für direkte Flag-Zuweisung
    agent.is_thinking = True
    assert agent.lifecycle_state == AgentLifecycleState.DELIBERATING
    assert agent.is_busy is True

    agent.is_thinking = False
    assert agent.lifecycle_state == AgentLifecycleState.IDLE


def test_waiting_properties_reflect_lifecycle_state() -> None:
    agent = Agent(id="1", name="Alice", position=Position(2, 4))
    assert not agent.is_waiting_for_reply
    assert not agent.is_listening_to_peer

    agent.transition_to(AgentLifecycleState.WAITING_FOR_PEER)
    assert agent.is_waiting_for_reply is True
    assert agent.is_listening_to_peer is True
    assert agent.is_busy is True

    agent.transition_to(AgentLifecycleState.IDLE)
    assert agent.is_waiting_for_reply is False
    assert agent.is_listening_to_peer is False
    assert agent.is_busy is False


def test_farewell_flags_no_longer_exist_on_agent() -> None:
    agent = Agent(id="1", name="Alice", position=Position(2, 4))
    field_names = {f.name for f in fields(agent)}
    assert "has_bid_farewell" not in field_names
    assert "peer_bid_farewell" not in field_names
    assert not hasattr(agent, "has_bid_farewell")
    assert not hasattr(agent, "peer_bid_farewell")

def test_legacy_flags_removed_from_agent_fields() -> None:
    agent = Agent(id="1", name="Alice", position=Position(2, 4))
    field_names = {f.name for f in fields(agent)}
    for flag in (
        "is_waiting_for_reply",
        "is_listening_to_peer",
        "has_bid_farewell",
        "peer_bid_farewell",
        "interaction_queue",
    ):
        assert flag not in field_names

    for flag in ("has_bid_farewell", "peer_bid_farewell", "interaction_queue"):
        assert not hasattr(agent, flag)

    for prop in ("is_waiting_for_reply", "is_listening_to_peer"):
        assert isinstance(getattr(type(agent), prop), property)