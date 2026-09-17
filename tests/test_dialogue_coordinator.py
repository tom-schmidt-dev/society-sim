from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest
from src.application.services.action_executor import ActionExecutor
from src.application.services.dialogue_coordinator import DialogueCoordinator
from src.application.services.dialogue_history import DialogueHistory
from src.application.services.dialogue_session_manager import DialogueSessionManager
from src.application.services.evasion_finder import EvasionFinder
from src.application.services.goal_service import GoalService
from src.domain.models.agent import Agent
from src.domain.models.cognition import (
    DialogueResolution,
    EndDialogueAction,
    GoalIntent,
    TalkAction,
)
from src.domain.models.goal import Goal
from src.domain.models.message import IncomingMessage
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.infrastructure.pathfinding.astar import AStarPathfinder


@pytest.fixture
def dialogue_setup():
    grid = WorldGrid(width=20, height=20)
    logger = MagicMock()
    cognition = AsyncMock()
    pathfinder = AStarPathfinder()
    current_tick = 5

    goal_service = GoalService(
        logger=logger,
        cognition_provider=cognition,
        pathfinder=pathfinder,
        tick_provider=lambda: current_tick,
    )
    evasion_finder = EvasionFinder(pathfinder=pathfinder)
    dialogue_history = DialogueHistory()
    session_manager = DialogueSessionManager(max_dialogue_turns=2)

    executor = ActionExecutor(
        grid=grid,
        logger=logger,
        dialogue_history=dialogue_history,
        goal_service=goal_service,
        pathfinder=pathfinder,
        evasion_finder=evasion_finder,
        tick_provider=lambda: current_tick,
    )

    coordinator = DialogueCoordinator(
        logger=logger,
        cognition_provider=cognition,
        pathfinder=pathfinder,
        goal_service=goal_service,
        evasion_finder=evasion_finder,
        action_executor=executor,
        session_manager=session_manager,
        dialogue_history=dialogue_history,
        tick_provider=lambda: current_tick,
    )

    return coordinator, cognition, session_manager, logger


@pytest.mark.asyncio
async def test_dialogue_accept_offer_yield_does_not_evade(dialogue_setup) -> None:
    # TC-NEG-01: Partner bietet Platz an, Agent akzeptiert -> Agent weicht nicht aus
    coordinator, cognition, _, _ = dialogue_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(10, 5)))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))

    alice.receive_message(
        IncomingMessage(
            from_agent_id="2",
            from_agent_name="Bob",
            message="Ich mache Platz.",
            intent="offer_yield",
        )
    )

    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Bob macht Platz, ich bedanke mich und passiere.",
        action=TalkAction(target_agent_id="2", message="Danke!", reason="Annahme"),
        negotiation_intent="accept",
    )

    await coordinator.handle_incoming_dialogue(alice, [alice, bob])

    # Alice übernimmt die Passierrolle und generiert kein Ausweichziel
    assert alice.active_goal.name == "Ost-Tor"
    assert alice.is_thinking is False


@pytest.mark.asyncio
async def test_dialogue_accept_request_yield_triggers_evasion(dialogue_setup) -> None:
    # TC-NEG-02: Partner fordert Platz, Agent stimmt zu -> Agent leitet Ausweichen ein
    coordinator, cognition, _, _ = dialogue_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(10, 5)))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))

    for x in range(10):
        alice.mental_map.update_tile(Position(x, 5), is_walkable=True, tick=1)
    alice.mental_map.update_tile(Position(5, 4), is_walkable=True, tick=1)

    alice.receive_message(
        IncomingMessage(
            from_agent_id="2",
            from_agent_name="Bob",
            message="Kannst du bitte ausweichen?",
            intent="request_yield",
        )
    )

    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Ich habe eine Nische nahebei und stimme zu.",
        action=EndDialogueAction(reason="Zustimmung", final_message="Ich weiche aus."),
        negotiation_intent="accept",
    )

    await coordinator.handle_incoming_dialogue(alice, [alice, bob])

    assert alice.active_goal.name == "In Nische ausweichen"
    assert alice.active_goal.target_position == Position(5, 4)
    assert alice.is_thinking is False


@pytest.mark.asyncio
async def test_dialogue_symmetric_offer_yield_suppresses_redundant_evasion(dialogue_setup) -> None:
    # TC-NEG-03: Beide wollen ausweichen; Partner weicht bereits aus -> eigene Nische unterdrückt
    coordinator, cognition, _, logger = dialogue_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(10, 5)))

    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    # Bob weicht bereits aktiv für Alice aus
    bob.push_goal(
        Goal(
            name="In Nische ausweichen",
            target_position=Position(6, 4),
            yield_for_agent_id="1",
        )
    )

    alice.receive_message(
        IncomingMessage(
            from_agent_id="2",
            from_agent_name="Bob",
            message="Ich suche eine Nische.",
            intent="offer_yield",
        )
    )

    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Ich weiche ebenfalls aus.",
        action=TalkAction(target_agent_id="2", message="Ich weiche auch aus.", reason="Höflichkeit"),
        negotiation_intent="offer_yield",
        new_goal=GoalIntent(name="In Nische ausweichen", intent_type="evade"),
    )

    await coordinator.handle_incoming_dialogue(alice, [alice, bob])

    # Evasion wird unterdrückt, da Bob bereits Platz macht
    logged_event_types = [call.args[0].event_type for call in logger.log.call_args_list]
    assert "evasion_suppressed" in logged_event_types
    assert alice.active_goal.name == "Ost-Tor"

    bob.commit_staging_messages()
    assert len(bob.inbox) == 1
    assert bob.inbox[0].intent == "accept"
    assert bob.inbox[0].message == "Danke, ich passiere."


@pytest.mark.asyncio
async def test_dialogue_circuit_breaker_arbitration_on_turn_limit(dialogue_setup) -> None:
    # TC-NEG-04: Nach Überschreiten des Rundenlimits greift der Schlichter ohne LLM-Inferenz
    coordinator, cognition, session_manager, _ = dialogue_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(10, 5)))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))

    for x in range(10):
        alice.mental_map.update_tile(Position(x, 5), is_walkable=True, tick=1)
        bob.mental_map.update_tile(Position(x, 5), is_walkable=True, tick=1)

    # Nische näher an Alice bei (5, 4)
    alice.mental_map.update_tile(Position(5, 4), is_walkable=True, tick=1)

    # Rundenlimit künstlich überschreiten (> 2)
    session_manager.increment_turn("1", "2")
    session_manager.increment_turn("1", "2")
    session_manager.increment_turn("1", "2")

    alice.receive_message(
        IncomingMessage(
            from_agent_id="2",
            from_agent_name="Bob",
            message="Ich gehe nicht zurück.",
            intent="reject",
        )
    )

    await coordinator.handle_incoming_dialogue(alice, [alice, bob])

    # Kognition wurde nicht befragt; Schlichter hat entschieden
    cognition.respond_to_dialogue.assert_not_awaited()
    # Alice hatte den kürzeren Weg zur Nische und muss ausweichen
    assert alice.active_goal.name == "In Nische ausweichen"


@pytest.mark.asyncio
async def test_dialogue_arbitration_equal_distance_resolves_deterministically(dialogue_setup) -> None:
    # TC-NEG-05: Gleiche Nischendistanz beider Agenten löst deterministisch auf Rolle 'yield' auf
    coordinator, cognition, session_manager, _ = dialogue_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(10, 5)))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    bob.push_goal(Goal(name="West-Tor", target_position=Position(1, 5)))

    for x in range(10):
        alice.mental_map.update_tile(Position(x, 5), is_walkable=True, tick=1)
        bob.mental_map.update_tile(Position(x, 5), is_walkable=True, tick=1)

    # Beide haben eine Nische in genau 1 Schritt Distanz
    alice.mental_map.update_tile(Position(5, 4), is_walkable=True, tick=1)
    bob.mental_map.update_tile(Position(6, 4), is_walkable=True, tick=1)

    session_manager.increment_turn("1", "2")
    session_manager.increment_turn("1", "2")
    session_manager.increment_turn("1", "2")

    alice.receive_message(
        IncomingMessage(
            from_agent_id="2",
            from_agent_name="Bob",
            message="Patt.",
            intent="reject",
        )
    )

    await coordinator.handle_incoming_dialogue(alice, [alice, bob])

    cognition.respond_to_dialogue.assert_not_awaited()
    # Bei Distanzgleichstand len_self <= len_partner greift deterministisch 'yield'
    assert alice.active_goal.name == "In Nische ausweichen"


@pytest.mark.asyncio
async def test_dialogue_accept_received_by_yielding_agent_terminates_without_reply(dialogue_setup) -> None:
    # TC-NEG-06: Wenn der Agent bereits ausweicht und der Partner 'accept' sendet,
    # wird die Einigung terminal bestätigt: keine LLM-Anfrage, keine Antwortnachricht.
    coordinator, cognition, session_manager, logger = dialogue_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    bob.push_goal(
        Goal(
            name="In Nische ausweichen",
            target_position=Position(6, 4),
            yield_for_agent_id="1",
        )
    )
    bob.is_waiting_for_reply = True
    bob.interaction_partner_id = "1"
    session_manager.increment_turn("2", "1")

    bob.receive_message(
        IncomingMessage(
            from_agent_id="1",
            from_agent_name="Alice",
            message="Danke, ich passiere.",
            intent="accept",
        )
    )

    await coordinator.handle_incoming_dialogue(bob, [alice, bob])

    # Kognition wurde nicht befragt
    cognition.respond_to_dialogue.assert_not_awaited()

    # Alice hat keine Nachricht zurückerhalten (weder in inbox noch staging_inbox)
    alice.commit_staging_messages()
    assert len(alice.inbox) == 0

    # Event wurde geloggt
    logged_event_types = [call.args[0].event_type for call in logger.log.call_args_list]
    assert "dialogue_agreement_confirmed" in logged_event_types

    # Wartezustände und Session wurden zurückgesetzt
    assert bob.is_waiting_for_reply is False
    assert bob.interaction_partner_id is None
    assert session_manager.get_turn_count("2", "1") == 0


@pytest.mark.asyncio
async def test_dialogue_mutual_accept_terminates_without_dispatching_talk_reply(dialogue_setup) -> None:
    # TC-NEG-07: Wenn ein Agent auf ein eingehendes 'accept' ebenfalls mit 'accept' reagiert,
    # wird keine weitere TalkAction an den Partner dispatcht (Verhinderung der accept-Echo-Schleife).
    coordinator, cognition, session_manager, _ = dialogue_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))

    alice.receive_message(
        IncomingMessage(
            from_agent_id="2",
            from_agent_name="Bob",
            message="Ich stimme dem Vorschlag zu.",
            intent="accept",
        )
    )

    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Wir sind uns einig.",
        action=TalkAction(
            target_agent_id="2",
            message="Ich stimme dem Vorschlag zu.",
            reason="Einigung",
        ),
        negotiation_intent="accept",
    )

    await coordinator.handle_incoming_dialogue(alice, [alice, bob])

    # Bob erhält keine neue Nachricht, die ihn erneut triggern würde
    bob.commit_staging_messages()
    assert len(bob.inbox) == 0

    # Session wurde zurückgesetzt
    assert session_manager.get_turn_count("1", "2") == 0
    assert alice.is_waiting_for_reply is False


@pytest.mark.asyncio
async def test_dialogue_accept_offer_yield_clears_waiting_flags_and_allows_movement(dialogue_setup) -> None:
    # TC-NEG-08: Wenn Alice Bobs Ausweichangebot annimmt, werden ihre Wartezustände
    # vollständig freigegeben, sodass sie nicht blockiert ist und den Chokepoint passieren kann.
    coordinator, cognition, session_manager, _ = dialogue_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(10, 5)))
    alice.is_waiting_for_reply = True
    alice.interaction_partner_id = "2"

    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    bob.push_goal(
        Goal(
            name="In Nische ausweichen",
            target_position=Position(6, 4),
            yield_for_agent_id="1",
        )
    )

    alice.receive_message(
        IncomingMessage(
            from_agent_id="2",
            from_agent_name="Bob",
            message="Ich weiche aus.",
            intent="offer_yield",
        )
    )

    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Bob weicht aus, ich passiere.",
        action=TalkAction(target_agent_id="2", message="Danke, ich passiere.", reason="Passieren"),
        negotiation_intent="accept",
    )

    await coordinator.handle_incoming_dialogue(alice, [alice, bob])

    # Alices Wartezustände sind vollständig gelöst
    assert alice.is_waiting_for_reply is False
    assert alice.interaction_partner_id is None
    assert alice.is_busy is False
    assert session_manager.get_turn_count("1", "2") == 0