import pytest
from src.domain.models.planning.cognition import (
    BlockedResolution,
    DialogueResolution,
    EndDialogueAction,
    TalkAction,
)


def test_dialogue_resolution_unwraps_flat_end_dialogue() -> None:
    # Reproduktion des Fehlers aus Tick 237 im Log
    raw_payload = {
        "action_type": "end_dialogue",
        "final_message": None,
        "reason": "Peer has bid farewell",
    }
    resolution = DialogueResolution.model_validate(raw_payload)
    assert isinstance(resolution.action, EndDialogueAction)
    assert resolution.action.reason == "Peer has bid farewell"
    assert resolution.thought == "Peer has bid farewell"


def test_blocked_resolution_unwraps_nested_talk_arguments() -> None:
    # Reproduktion des Fehlers aus Tick 171 im Log
    raw_payload = {
        "thought": "Ich bin blockiert.",
        "action": {
            "action_type": "talk",
            "arguments": {
                "target_agent_id": "2",
                "message": "Bitte weichen Sie aus!",
                "reason": "Ich muss zum Ziel.",
                "intent": "request_yield",
            },
        },
    }
    resolution = BlockedResolution.model_validate(raw_payload)
    assert isinstance(resolution.action, TalkAction)
    assert resolution.action.target_agent_id == "2"
    assert resolution.action.message == "Bitte weichen Sie aus!"
    assert resolution.action.intent == "request_yield"