from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest
from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.coordination.dialogue_coordinator import DialogueCoordinator
from src.application.services.coordination.dialogue_history import DialogueHistory
from src.application.services.coordination.dialogue_session_manager import DialogueSessionManager
from src.application.services.movement.evasion_finder import EvasionFinder
from src.application.services.cognition.goal_service import GoalService
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.cognition import (
    DialogueResolution,
    EndDialogueAction,
    GoalIntent,
    TalkAction,
)
from src.domain.models.planning.goal import Goal
from src.domain.models.communication.message import IncomingMessage
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
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

@pytest.mark.asyncio
async def test_dialogue_retrieves_social_memories_for_partner() -> None:
    # TC-SOC-01: DialogCoordinator ruft soziale Erinnerungen über den Partner ab und bettet sie in den Kontext ein
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

    mock_vector_store = MagicMock()
    mock_vector_store.retrieve_relevant.return_value = [
        "Tag 1: Begegnung mit Bob verlief cooperative."
    ]

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
        vector_memory_store=mock_vector_store,
    )

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    alice.receive_message(
        IncomingMessage(
            from_agent_id="2",
            from_agent_name="Bob",
            message="Hallo",
            intent="talk",
        )
    )

    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Bob kenne ich als kooperativ.",
        action=TalkAction(target_agent_id="2", message="Hallo Bob!", reason="Gruß"),
        negotiation_intent="offer_yield",
    )

    await coordinator.handle_incoming_dialogue(alice, [alice, bob])

    mock_vector_store.retrieve_relevant.assert_called_once_with(
        agent_id="1",
        query="Begegnung mit 2",
        limit=3,
        metadata_filter={"category": "social"},
    )

    context_arg = cognition.respond_to_dialogue.call_args[0][0]
    assert context_arg["social_memories"] == [
        "Tag 1: Begegnung mit Bob verlief cooperative."
    ]


@pytest.mark.asyncio
async def test_dialogue_retrieval_resilient_on_vector_store_failure() -> None:
    # TC-SOC-02: Bei einem Fehler im Vektorspeicher stürzt der Dialog nicht ab
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

    mock_vector_store = MagicMock()
    mock_vector_store.retrieve_relevant.side_effect = RuntimeError("Vektor-DB nicht erreichbar")

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
        vector_memory_store=mock_vector_store,
    )

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    alice.receive_message(
        IncomingMessage(
            from_agent_id="2",
            from_agent_name="Bob",
            message="Hallo",
            intent="talk",
        )
    )

    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Antwort ohne Gedächtnis",
        action=TalkAction(target_agent_id="2", message="Hallo!", reason="Gruß"),
        negotiation_intent="accept",
    )

    await coordinator.handle_incoming_dialogue(alice, [alice, bob])

    assert alice.is_thinking is False
    context_arg = cognition.respond_to_dialogue.call_args[0][0]
    assert context_arg["social_memories"] == []


@pytest.mark.asyncio
async def test_dialogue_unlimited_turns_allows_persistent_dispute(dialogue_setup) -> None:
    # TC-DISP-01: Bei max_dialogue_turns=None greift kein Schlichter; Streit geht über Turn 2 hinaus
    coordinator, cognition, _, _ = dialogue_setup
    coordinator._session_manager._max_dialogue_turns = None

    alice = Agent(id="1", name="Alice", position=Position(5, 5), assertiveness=0.9)
    bob = Agent(id="2", name="Bob", position=Position(6, 5), assertiveness=0.8)

    # 5 Runden künstlich hochzählen
    for _ in range(5):
        coordinator._session_manager.increment_turn("1", "2")

    alice.receive_message(
        IncomingMessage(
            from_agent_id="2",
            from_agent_name="Bob",
            message="Ich weiche nicht aus!",
            intent="reject",
        )
    )

    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Ich beharre auf meinem Recht.",
        action=TalkAction(target_agent_id="2", message="Ich aber auch nicht!", reason="Beharren"),
        negotiation_intent="reject",
    )

    await coordinator.handle_incoming_dialogue(alice, [alice, bob])

    # Kognition wurde regulär befragt, kein Zwangsausweichen
    cognition.respond_to_dialogue.assert_awaited_once()
    assert alice.active_goal is None or "In Nische ausweichen" not in alice.active_goal.name
    assert alice.is_thinking is False


@pytest.mark.asyncio
async def test_dialogue_sliding_window_and_summary_in_context(dialogue_setup) -> None:
    # TC-DISP-02: Maximal 20 Nachrichten im Kontext, conversation_summary initial vorhanden
    coordinator, cognition, session_manager, _ = dialogue_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))

    # 25 Nachrichten in die Historie einfügen
    for i in range(25):
        coordinator._dialogue_history.record_dialogue(
            tick=i,
            sender_id="1" if i % 2 == 0 else "2",
            sender_name="Alice" if i % 2 == 0 else "Bob",
            recipient_id="2" if i % 2 == 0 else "1",
            recipient_name="Bob" if i % 2 == 0 else "Alice",
            message=f"Argument {i}",
            intent="reject",
        )

    session_manager.set_summary("Bisheriger Streit: Beide beharren auf Durchgang.", "1", "2")

    alice.receive_message(
        IncomingMessage(
            from_agent_id="2",
            from_agent_name="Bob",
            message="Argument 24",
            intent="reject",
        )
    )

    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Weiter streiten.",
        action=TalkAction(target_agent_id="2", message="Nein!", reason="Streit"),
        negotiation_intent="reject",
    )

    await coordinator.handle_incoming_dialogue(alice, [alice, bob])

    context_arg = cognition.respond_to_dialogue.call_args[0][0]
    # Maximal 20 Nachrichten im gleitenden Fenster
    assert len(context_arg["recent_dialogues"]) == 20
    # Älteste Nachrichten (0 bis 4) wurden verdrängt, Argument 24 ist enthalten
    assert "Argument 24" in context_arg["recent_dialogues"][-1]
    assert "Argument 0" not in context_arg["recent_dialogues"][0]
    # Zusammenfassung ist im Kontext präsent
    assert context_arg["conversation_summary"] == "Bisheriger Streit: Beide beharren auf Durchgang."


@pytest.mark.asyncio
async def test_dialogue_triggers_social_reflection_and_vector_persistence(dialogue_setup) -> None:
    # TC-SOC-03: Dialogende triggert soziale Reflexion, Vektorspeicherung und Fine-Tuning-Log
    coordinator, cognition, _, logger = dialogue_setup

    mock_vector_store = MagicMock()
    coordinator._vector_memory_store = mock_vector_store

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))

    alice.receive_message(
        IncomingMessage(
            from_agent_id="2",
            from_agent_name="Bob",
            message="Ich mache dir Platz und weiche aus.",
            intent="offer_yield",
        )
    )

    from src.domain.models.planning.cognition import SocialReflection
    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Bob weicht aus, ich nehme an.",
        action=TalkAction(target_agent_id="2", message="Danke, ich gehe durch.", reason="Annahme"),
        negotiation_intent="accept",
    )
    cognition.reflect_on_dialogue.return_value = SocialReflection(
        assessment="Kooperativ und verständnisvoll.",
        progression_summary="Nach kurzer Ansprache sofort in die Nische ausgewichen.",
    )

    await coordinator.handle_incoming_dialogue(alice, [alice, bob])

    # 1. Reflexion wurde aufgerufen
    cognition.reflect_on_dialogue.assert_awaited_once()
    refl_context = cognition.reflect_on_dialogue.call_args[0][0]
    assert refl_context["agent_id"] == "1"
    assert refl_context["partner_id"] == "2"

    # 2. Vektorspeicher wurde mit zweigeteiltem Eintrag und Metadaten beliefert
    mock_vector_store.add_memories.assert_called_once()
    mem_call = mock_vector_store.add_memories.call_args
    assert mem_call.kwargs["agent_id"] == "1"
    memories = mem_call.kwargs["memories"]
    assert len(memories) == 1
    assert "Kooperativ und verständnisvoll." in memories[0]
    assert "in die Nische ausgewichen." in memories[0]
    metas = mem_call.kwargs["metadatas"]
    assert metas[0]["category"] == "social"
    assert metas[0]["partner_id"] == "2"

    # 3. Log-Ereignis für Fine-Tuning wurde erfasst
    logged_event_types = [call.args[0].event_type for call in logger.log.call_args_list]
    assert "social_reflection_completed" in logged_event_types