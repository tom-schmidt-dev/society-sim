from __future__ import annotations

import asyncio
import pytest

from src.domain.models.interaction.commands import (
    EndDialogueCommand,
    InspectCommand,
    InteractionCommand,
    ProbeCommand,
    TalkCommand,
)
from src.domain.models.interaction.entity_capabilities import EntityCapability
from src.domain.models.interaction.interaction_result import ImmediateResult, PendingFuture


def test_entity_capability_flags() -> None:
    # Einzelne Capabilities und Bitflag-Kombinationen
    empty_caps = EntityCapability.NONE
    assert empty_caps.value == 0

    caps = EntityCapability.COMMUNICATIVE | EntityCapability.INSPECTABLE
    assert EntityCapability.COMMUNICATIVE in caps
    assert EntityCapability.INSPECTABLE in caps
    assert EntityCapability.PUSHABLE not in caps
    assert EntityCapability.TRAVERSABLE not in caps


def test_command_instantiation_and_immutability() -> None:
    talk = TalkCommand(
        source_entity_id="1",
        target_entity_id="2",
        message="Bitte weichen Sie aus!",
        intent="request_yield",
    )
    assert isinstance(talk, InteractionCommand)
    assert talk.source_entity_id == "1"
    assert talk.target_entity_id == "2"
    assert talk.message == "Bitte weichen Sie aus!"
    assert talk.intent == "request_yield"

    # Frozen Dataclass Immutability
    with pytest.raises((AttributeError, TypeError)):
        talk.message = "Neue Nachricht"  # type: ignore[misc]

    inspect = InspectCommand(source_entity_id="1", target_entity_id="2")
    assert isinstance(inspect, InteractionCommand)

    probe = ProbeCommand(source_entity_id="1", target_entity_id="2")
    assert isinstance(probe, InteractionCommand)

    end_dlg = EndDialogueCommand(
        source_entity_id="2",
        target_entity_id="1",
        reason="Einigung erzielt.",
        final_message="Ich weiche aus.",
        intent="accept",
    )
    assert isinstance(end_dlg, InteractionCommand)
    assert end_dlg.intent == "accept"


def test_immediate_result_properties() -> None:
    rejected = ImmediateResult(success=False, reason="NOT_COMMUNICATIVE")
    assert not rejected.success
    assert rejected.reason == "NOT_COMMUNICATIVE"
    assert rejected.payload == {}

    with pytest.raises((AttributeError, TypeError)):
        rejected.success = True  # type: ignore[misc]


@pytest.mark.asyncio
async def test_pending_future_properties() -> None:
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[ImmediateResult] = loop.create_future()

    pending = PendingFuture(future=fut, context={"partner_id": "2"})
    assert not pending.future.done()
    assert pending.context["partner_id"] == "2"

    fut.set_result(ImmediateResult(success=True, reason="ACCEPTED"))
    assert pending.future.done()
    res = await pending.future
    assert res.success is True