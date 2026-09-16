from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest
from src.application.services.action_executor import ActionExecutor
from src.application.services.conflict_coordinator import ConflictCoordinator
from src.application.services.dialogue_history import DialogueHistory
from src.application.services.dialogue_session_manager import DialogueSessionManager
from src.application.services.evasion_finder import EvasionFinder
from src.application.services.goal_service import GoalService
from src.domain.models.agent import Agent
from src.domain.models.cognition import BlockedResolution, InspectAction, TalkAction
from src.domain.models.goal import Goal
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
from src.infrastructure.pathfinding.astar import AStarPathfinder


@pytest.fixture
def conflict_setup():
    grid = WorldGrid(width=20, height=20)
    for x in range(20):
        for y in range(20):
            grid.remove_obstacle(Position(x, y))

    logger = MagicMock()
    cognition = AsyncMock()
    pathfinder = AStarPathfinder()
    current_tick = 10

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

    coordinator = ConflictCoordinator(
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

    return coordinator, cognition, logger, goal_service


@pytest.mark.asyncio
async def test_conflict_first_contact_inspects_unknown_entity(conflict_setup) -> None:
    # TC-CON-01: Uninspizierte Entität triggert Inferenz und Inspektion
    coordinator, cognition, logger, _ = conflict_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    alice.assign_path([Position(6, 5)])
    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(10, 5)))

    stone = WorldEntity(
        id="stone_1",
        name="Großer Stein",
        position=Position(6, 5),
        entity_type="rock",
        is_conversational=False,
    )
    all_entities = [alice, stone]

    cognition.resolve_blockage.return_value = BlockedResolution(
        thought="Objekt ist unbekannt, muss inspiziert werden.",
        action=InspectAction(target_agent_id="stone_1", reason="Erstkontakt"),
    )

    await coordinator.resolve_blockage(
        agent=alice,
        blocker=stone,
        blocked_pos=Position(6, 5),
        all_entities=all_entities,
    )

    cognition.resolve_blockage.assert_awaited_once()
    assert alice.memory.is_inspected("stone_1") is True
    assert alice.is_thinking is False


@pytest.mark.asyncio
async def test_conflict_locks_and_serializes_concurrent_blockage(conflict_setup) -> None:
    # TC-CON-02: Symmetrische Blockade im selben Takt serialisiert via Lock
    coordinator, cognition, _, _ = conflict_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    alice.assign_path([Position(6, 5)])
    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(10, 5)))

    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    bob.assign_path([Position(5, 5)])
    bob.push_goal(Goal(name="West-Tor", target_position=Position(1, 5)))

    all_entities = [alice, bob]

    async def delayed_resolve(context):
        await asyncio.sleep(0.02)
        return BlockedResolution(
            thought="Ich spreche mit Bob.",
            action=TalkAction(target_agent_id="2", message="Hallo Bob", reason="Weg frei machen"),
        )

    cognition.resolve_blockage.side_effect = delayed_resolve

    # Beide Agenten starten zeitgleich resolve_blockage
    await asyncio.gather(
        coordinator.resolve_blockage(alice, bob, Position(6, 5), all_entities),
        coordinator.resolve_blockage(bob, alice, Position(5, 5), all_entities),
    )

    # Der erste Agent hat gesprochen und Bob ein Warteziel gesetzt
    assert alice.active_goal is not None
    assert bob.active_goal is not None
    assert not alice.is_thinking
    assert not bob.is_thinking


@pytest.mark.asyncio
async def test_conflict_listening_goal_popped_in_finally_on_error(conflict_setup) -> None:
    # TC-CON-03: Listening-Goal auf Partner wird im Fehlerfall sicher abgebaut
    coordinator, cognition, _, _ = conflict_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    alice.assign_path([Position(6, 5)])
    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(10, 5)))

    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    bob.push_goal(Goal(name="West-Tor", target_position=Position(1, 5)))
    all_entities = [alice, bob]

    cognition.resolve_blockage.side_effect = RuntimeError("LLM-Verbindungsabbruch")

    await coordinator.resolve_blockage(alice, bob, Position(6, 5), all_entities)

    # Bob darf kein verwaistes 'Lauscht Agent Alice' mehr besitzen
    assert bob.active_goal.name == "West-Tor"
    assert alice.is_thinking is False


@pytest.mark.asyncio
async def test_conflict_fallback_on_cognition_failure(conflict_setup) -> None:
    # TC-CON-04: Kognitionsfehler fängt Exception und setzt befristetes Warten
    coordinator, cognition, logger, _ = conflict_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    alice.assign_path([Position(6, 5)])
    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(10, 5)))

    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    all_entities = [alice, bob]

    cognition.resolve_blockage.side_effect = TimeoutError("Inferenz-Timeout")

    await coordinator.resolve_blockage(alice, bob, Position(6, 5), all_entities)

    assert alice.active_goal is not None
    assert alice.active_goal.name == "Warten (Fallback)"
    assert alice.active_goal.remaining_ticks == 2
    assert alice.is_thinking is False

    # Event 'cognition_failed' wurde geloggt
    logged_event_types = [call.args[0].event_type for call in logger.log.call_args_list]
    assert "cognition_failed" in logged_event_types