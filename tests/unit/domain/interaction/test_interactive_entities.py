from __future__ import annotations

import pytest

from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_state import AgentLifecycleState
from src.domain.models.interaction.commands import (
    EndDialogueCommand,
    InspectCommand,
    ProbeCommand,
    TalkCommand,
)
from src.domain.models.interaction.entity_capabilities import EntityCapability
from src.domain.models.interaction.interaction_result import ImmediateResult, PendingFuture
from src.domain.models.world.position import Position
from src.domain.models.world.world_entity import WorldEntity


def test_static_obstacle_rejects_talk_immediately() -> None:
    stone = WorldEntity(
        id="stone_1",
        name="Rollender Stein",
        position=Position(15, 4),
        entity_type="obstacle",
        is_conversational=False,
    )

    cmd = TalkCommand(
        source_entity_id="1",
        target_entity_id="stone_1",
        message="Bitte aus dem Weg rollen!",
    )
    res = stone.receive_interaction(cmd)

    assert isinstance(res, ImmediateResult)
    assert res.success is False
    assert res.reason == "NOT_COMMUNICATIVE"


def test_static_obstacle_accepts_inspect_and_probe() -> None:
    stone = WorldEntity(
        id="stone_1",
        name="Rollender Stein",
        position=Position(15, 4),
        entity_type="obstacle",
        is_passable=False,
    )

    inspect_res = stone.receive_interaction(InspectCommand(source_entity_id="1", target_entity_id="stone_1"))
    assert isinstance(inspect_res, ImmediateResult)
    assert inspect_res.success is True
    assert inspect_res.reason == "INSPECTED"
    assert inspect_res.payload["entity_type"] == "obstacle"
    assert inspect_res.payload["is_passable"] is False

    probe_res = stone.receive_interaction(ProbeCommand(source_entity_id="1", target_entity_id="stone_1"))
    assert isinstance(probe_res, ImmediateResult)
    assert probe_res.success is True
    assert probe_res.reason == "PROBED"
    assert probe_res.payload["is_passable"] is False

@pytest.mark.asyncio
def test_agent_accepts_talk_and_returns_pending_future() -> None:
    bob = Agent(
        id="2",
        name="Bob",
        position=Position(15, 4),
        is_conversational=True,
    )

    cmd = TalkCommand(
        source_entity_id="1",
        target_entity_id="2",
        message="Bitte weichen Sie aus!",
        intent="request_yield",
    )
    res = bob.receive_interaction(cmd)

    assert isinstance(res, PendingFuture)
    assert not res.future.done()
    assert res.context["source_entity_id"] == "1"
    assert len(bob.interaction_mailbox) == 1

@pytest.mark.asyncio
def test_agent_busy_queues_future_without_drop() -> None:
    bob = Agent(
        id="2",
        name="Bob",
        position=Position(15, 4),
        is_conversational=True,
    )
    bob.transition_to(AgentLifecycleState.DELIBERATING, reason="Eigene Wegfindung")

    cmd = TalkCommand(
        source_entity_id="1",
        target_entity_id="2",
        message="Können wir sprechen?",
    )
    res = bob.receive_interaction(cmd)

    # Future wird trotz aktiver Kognition nicht verworfen
    assert isinstance(res, PendingFuture)
    assert not res.future.done()
    assert len(bob.interaction_mailbox) == 1


def test_non_conversational_agent_rejects_talk_immediately() -> None:
    mute_agent = Agent(
        id="3",
        name="Stummer Agent",
        position=Position(15, 4),
        is_conversational=False,
    )

    cmd = TalkCommand(
        source_entity_id="1",
        target_entity_id="3",
        message="Hallo?",
    )
    res = mute_agent.receive_interaction(cmd)

    assert isinstance(res, ImmediateResult)
    assert res.success is False
    assert res.reason == "NOT_COMMUNICATIVE"
    assert len(mute_agent.interaction_mailbox) == 0

@pytest.mark.asyncio
def test_agent_resolves_future_on_end_dialogue() -> None:
    bob = Agent(
        id="2",
        name="Bob",
        position=Position(15, 4),
        is_conversational=True,
    )

    talk_cmd = TalkCommand(source_entity_id="1", target_entity_id="2", message="Hi")
    pending = bob.receive_interaction(talk_cmd)
    assert isinstance(pending, PendingFuture)

    end_cmd = EndDialogueCommand(source_entity_id="1", target_entity_id="2", reason="Tschüss")
    end_res = bob.receive_interaction(end_cmd)

    assert isinstance(end_res, ImmediateResult)
    assert end_res.success is True
    assert pending.future.done()
    assert pending.future.result().reason == "DIALOGUE_ENDED"