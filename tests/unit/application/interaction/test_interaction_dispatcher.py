from __future__ import annotations

import asyncio
import pytest

from src.application.services.interaction.interaction_dispatcher import InteractionDispatcher
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_state import AgentLifecycleState
from src.domain.models.interaction.commands import (
    InspectCommand,
    ProbeCommand,
    TalkCommand,
)
from src.domain.models.interaction.interaction_result import ImmediateResult, PendingFuture
from src.domain.models.planning.events import SimulationEvent
from src.domain.models.world.position import Position
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.event_logger import IEventLogger

class MockLogger(IEventLogger):
    def __init__(self) -> None:
        self.logged_events: list[SimulationEvent] = []

    def log(self, event: SimulationEvent) -> None:
        self.logged_events.append(event)

def test_dispatch_immediate_failure_updates_requester_memory() -> None:
    dispatcher = InteractionDispatcher()
    alice = Agent(id="1", name="Alice", position=Position(14, 4))
    stone = WorldEntity(
        id="stone_1",
        name="Rollender Stein",
        position=Position(15, 4),
        entity_type="obstacle",
        is_conversational=False,
    )

    cmd = TalkCommand(source_entity_id="1", target_entity_id="stone_1", message="Platz da!")
    res = dispatcher.dispatch(alice, stone, cmd)

    assert isinstance(res, ImmediateResult)
    assert res.success is False
    assert res.reason == "NOT_COMMUNICATIVE"

    # Epistemisches Gedächtnis wurde aktualisiert
    assert alice.memory.can_talk("stone_1") is False
    assert alice.memory.get_assumed_conversational("stone_1") is False
    assert alice.lifecycle_state == AgentLifecycleState.IDLE


def test_dispatch_inspect_command_updates_requester_memory() -> None:
    dispatcher = InteractionDispatcher()
    alice = Agent(id="1", name="Alice", position=Position(14, 4))
    stone = WorldEntity(
        id="stone_1",
        name="Rollender Stein",
        position=Position(15, 4),
        entity_type="obstacle",
        is_passable=False,
    )

    cmd = InspectCommand(source_entity_id="1", target_entity_id="stone_1")
    res = dispatcher.dispatch(alice, stone, cmd)

    assert isinstance(res, ImmediateResult)
    assert res.success is True
    assert alice.memory.is_inspected("stone_1") is True
    assert alice.memory.get_assumed_walkable("stone_1") is False


def test_dispatch_probe_command_updates_requester_memory() -> None:
    dispatcher = InteractionDispatcher()
    alice = Agent(id="1", name="Alice", position=Position(14, 4))
    obstacle = WorldEntity(
        id="obs_1",
        name="Holzkiste",
        position=Position(15, 4),
        entity_type="crate",
        is_passable=False,
    )

    cmd = ProbeCommand(source_entity_id="1", target_entity_id="obs_1")
    res = dispatcher.dispatch(alice, obstacle, cmd)

    assert isinstance(res, ImmediateResult)
    assert res.success is True
    assert alice.memory.can_probe("obs_1") is False
    assert alice.memory.get_entity_walkability("obs_1") is False


@pytest.mark.asyncio
async def test_dispatch_future_resolves_when_partner_replies() -> None:
    dispatcher = InteractionDispatcher()
    alice = Agent(id="1", name="Alice", position=Position(14, 4))
    bob = Agent(id="2", name="Bob", position=Position(15, 4), is_conversational=True)

    cmd = TalkCommand(
        source_entity_id="1",
        target_entity_id="2",
        message="Bitte weichen Sie aus!",
        intent="request_yield",
    )
    res = dispatcher.dispatch(alice, bob, cmd)

    assert isinstance(res, PendingFuture)
    assert not res.future.done()
    assert alice.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER
    assert alice.is_waiting_for_reply is True

    # Bob löst das Future auf (Antwort eingetroffen)
    reply = TalkCommand(source_entity_id="2", target_entity_id="1", message="Ich mache Platz.", intent="offer_yield")
    res.future.set_result(reply)
    await asyncio.sleep(0)

    assert alice.memory.get_assumed_conversational("2") is True
    assert alice.lifecycle_state == AgentLifecycleState.DELIBERATING
    assert alice.is_waiting_for_reply is False


@pytest.mark.asyncio
async def test_dispatch_future_cancelled_resets_state() -> None:
    dispatcher = InteractionDispatcher()
    alice = Agent(id="1", name="Alice", position=Position(14, 4))
    bob = Agent(id="2", name="Bob", position=Position(15, 4), is_conversational=True)

    cmd = TalkCommand(source_entity_id="1", target_entity_id="2", message="Hallo?")
    res = dispatcher.dispatch(alice, bob, cmd)

    assert isinstance(res, PendingFuture)
    res.future.cancel()
    await asyncio.sleep(0)

    assert alice.lifecycle_state == AgentLifecycleState.IDLE
    assert alice.is_waiting_for_reply is False


def test_dispatch_talk_to_busy_partner_returns_busy_immediate_result() -> None:
    dispatcher = InteractionDispatcher()
    alice = Agent(id="1", name="Alice", position=Position(14, 4))
    bob = Agent(id="2", name="Bob", position=Position(15, 4), is_conversational=True)
    bob.interaction_partner_id = "3"  # Bob spricht bereits mit jemand anderem

    cmd = TalkCommand(source_entity_id="1", target_entity_id="2", message="Kann ich vorbei?")
    res = dispatcher.dispatch(alice, bob, cmd)

    assert isinstance(res, ImmediateResult)
    assert res.success is False
    assert res.reason == "BUSY"
    # Autonomie bleibt erhalten: Alice wird nicht in WAITING_FOR_PEER versetzt
    assert alice.lifecycle_state == AgentLifecycleState.IDLE
    assert not alice.is_busy
    assert len(bob.interaction_mailbox) == 0


def test_dispatch_talk_to_deliberating_partner_returns_busy_immediate_result() -> None:
    dispatcher = InteractionDispatcher()
    alice = Agent(id="1", name="Alice", position=Position(14, 4))
    bob = Agent(id="2", name="Bob", position=Position(15, 4), is_conversational=True)
    bob.transition_to(AgentLifecycleState.DELIBERATING, reason="Überlegt Weg")

    cmd = TalkCommand(source_entity_id="1", target_entity_id="2", message="Hallo?")
    res = dispatcher.dispatch(alice, bob, cmd)

    assert isinstance(res, ImmediateResult)
    assert res.success is False
    assert res.reason == "BUSY"
    assert alice.lifecycle_state == AgentLifecycleState.IDLE
    assert len(bob.interaction_mailbox) == 0


def test_dispatch_talk_busy_emits_interaction_busy_rejected_event() -> None:
    mock_logger = MockLogger()
    dispatcher = InteractionDispatcher(logger=mock_logger, tick_provider=lambda: 12)
    alice = Agent(id="1", name="Alice", position=Position(14, 4))
    bob = Agent(id="2", name="Bob", position=Position(15, 4), is_conversational=True)
    bob.interaction_partner_id = "3"

    cmd = TalkCommand(source_entity_id="1", target_entity_id="2", message="Hallo?")
    dispatcher.dispatch(alice, bob, cmd)

    assert len(mock_logger.logged_events) == 1
    recorded_event = mock_logger.logged_events[0]
    assert recorded_event.event_type == "interaction_busy_rejected"
    assert recorded_event.tick == 12
    assert recorded_event.payload["reason"] == "BUSY"
    assert recorded_event.payload["target_id"] == "2"


def test_dispatch_talk_to_mutual_partner_is_not_busy() -> None:
    dispatcher = InteractionDispatcher()
    alice = Agent(id="1", name="Alice", position=Position(14, 4))
    bob = Agent(id="2", name="Bob", position=Position(15, 4), is_conversational=True)
    bob.interaction_partner_id = "1"  # Bob wartet bereits genau auf Alice

    cmd = TalkCommand(source_entity_id="1", target_entity_id="2", message="Hier bin ich.")
    res = dispatcher.dispatch(alice, bob, cmd)

    assert isinstance(res, PendingFuture)
    assert alice.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER