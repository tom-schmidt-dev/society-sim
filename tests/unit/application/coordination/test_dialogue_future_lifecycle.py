from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock
import pytest

from src.application.services.coordination.dialogue_coordinator import DialogueCoordinator
from src.application.services.coordination.dialogue_history import DialogueHistory
from src.application.services.coordination.dialogue_session_manager import DialogueSessionManager
from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.cognition.goal_service import GoalService
from src.application.services.interaction.interaction_dispatcher import InteractionDispatcher
from src.application.services.movement.evasion_finder import EvasionFinder
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_state import AgentLifecycleState
from src.domain.models.interaction.commands import TalkCommand
from src.domain.models.interaction.interaction_result import PendingFuture
from src.domain.models.planning.cognition import (
    BlockedResolution,
    DialogueResolution,
    GoalDecision,
    GoalEvaluation,
    GoalIntent,
    SocialReflection,
    TalkAction,
)
from src.domain.models.planning.goal import Goal
from src.domain.models.planning.planning import AgentCognitiveContext, PlanDecomposition
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.infrastructure.pathfinding.astar import AStarPathfinder


class MockCognitionProvider(ICognitionProvider):
    def __init__(self, responses: list[DialogueResolution]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def decide_next_goal(self, context: dict[str, Any]) -> GoalDecision:
        raise NotImplementedError

    async def resolve_blockage(self, context: dict[str, Any]) -> BlockedResolution:
        raise NotImplementedError

    async def evaluate_goal_status(self, context: dict[str, Any]) -> GoalEvaluation:
        raise NotImplementedError

    async def respond_to_dialogue(self, context: dict[str, Any]) -> DialogueResolution:
        self.call_count += 1
        if self._responses:
            return self._responses.pop(0)
        return DialogueResolution(
            thought="Standard-Antwort",
            action=TalkAction(
                target_agent_id=str(context.get("partner_id", "")),
                message="Ok.",
                intent="accept",
            ),
            negotiation_intent="accept",
        )

    async def reflect_on_dialogue(self, context: dict[str, Any]) -> SocialReflection:
        return SocialReflection(
            assessment="kooperativ",
            progression_summary="Einigung erzielt",
        )

    async def decompose_plan(
        self, context: AgentCognitiveContext
    ) -> PlanDecomposition:
        raise NotImplementedError


def _setup_corridor_environment() -> tuple[WorldGrid, Agent, Agent, ActionExecutor, DialogueHistory, MagicMock]:
    grid = WorldGrid(width=30, height=9)
    for x in range(5, 25):
        if x != 15:
            grid.set_obstacle(Position(x, 3))
        grid.set_obstacle(Position(x, 5))

    alice = Agent(id="1", name="Alice", position=Position(14, 4), is_conversational=True)
    alice.push_goal(Goal(name="Ost-Portal", target_position=Position(27, 4)))

    bob = Agent(id="2", name="Bob", position=Position(15, 4), is_conversational=True)
    bob.push_goal(Goal(name="West-Portal", target_position=Position(2, 4)))

    # Initialisierung der mentalen Karten analog zu SimulationEngine.register_agent
    for a in (alice, bob):
        a.mental_map.set_bounds(grid.width, grid.height)
        for y in range(grid.height):
            for x in range(grid.width):
                p = Position(x, y)
                if grid.is_walkable(p):
                    a.mental_map.update_tile(p, is_walkable=True, tick=0)
                else:
                    a.mental_map.mark_obstacle(p, tick=0)

    logger = MagicMock(spec=IEventLogger)
    pathfinder = AStarPathfinder()
    history = DialogueHistory()
    dummy_cognition = MagicMock(spec=ICognitionProvider)
    goal_service = GoalService(logger=logger, cognition_provider=dummy_cognition, pathfinder=pathfinder)
    executor = ActionExecutor(
        grid=grid,
        logger=logger,
        dialogue_history=history,
        goal_service=goal_service,
        pathfinder=pathfinder,
    )
    return grid, alice, bob, executor, history, logger


@pytest.mark.asyncio
async def test_dialogue_agreement_pushes_evasion_goal_to_stack() -> None:
    grid, alice, bob, executor, history, logger = _setup_corridor_environment()
    pathfinder = AStarPathfinder()
    goal_service = GoalService(logger=logger, cognition_provider=MagicMock(spec=ICognitionProvider), pathfinder=pathfinder)
    evasion_finder = EvasionFinder(pathfinder)
    session_manager = DialogueSessionManager(logger=logger)

    bob_response = DialogueResolution(
        thought="Ich mache Platz in der Nische.",
        action=TalkAction(
            target_agent_id="1",
            message="Ich weiche nach (15, 3) aus.",
            intent="offer_yield",
        ),
        negotiation_intent="offer_yield",
        new_goal=GoalIntent(name="In Nische ausweichen", intent_type="evade"),
    )
    cognition = MockCognitionProvider(responses=[bob_response])

    coordinator = DialogueCoordinator(
        logger=logger,
        cognition_provider=cognition,
        pathfinder=pathfinder,
        goal_service=goal_service,
        evasion_finder=evasion_finder,
        action_executor=executor,
        session_manager=session_manager,
        dialogue_history=history,
    )

    dispatcher = InteractionDispatcher(logger=logger)
    cmd = TalkCommand(
        source_entity_id="1",
        target_entity_id="2",
        message="Bitte ausweichen!",
        intent="request_yield",
    )
    pending = dispatcher.dispatch(alice, bob, cmd)
    assert isinstance(pending, PendingFuture)
    assert alice.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER

    bob.commit_staging_messages()
    await coordinator.handle_incoming_dialogue(bob, [alice, bob])

    # Prüfung auf abgefangene Coordinator-Fehler
    for call in logger.log.call_args_list:
        event = call.args[0]
        if getattr(event, "event_type", None) == "dialogue_failed":
            raise RuntimeError(f"Dialogue failed: {event.payload.get('error')}")

    # 1. Zielstack-Integrität bei Bob
    assert len(bob.goals) == 2
    assert bob.goals[0].name == "West-Portal"
    assert bob.goals[0].status == "paused"
    assert bob.goals[1].name == "In Nische ausweichen"
    assert bob.goals[1].status == "active"
    assert bob.goals[1].target_position == Position(15, 3)

    # 2. Status-Prüfung
    assert bob.lifecycle_state == AgentLifecycleState.YIELDING
    assert len(bob.path) > 0
    assert bob.path[-1] == Position(15, 3)

    # 3. Future-Auflösung von Alice
    assert pending.future.done()
    result = pending.future.result()
    assert result.success is True
    assert result.reason == "TALK_REPLY"
    assert result.payload["intent"] == "offer_yield"


@pytest.mark.asyncio
async def test_dialogue_ping_pong_turn_taking() -> None:
    grid, alice, bob, executor, history, logger = _setup_corridor_environment()
    pathfinder = AStarPathfinder()
    goal_service = GoalService(logger=logger, cognition_provider=MagicMock(spec=ICognitionProvider), pathfinder=pathfinder)
    evasion_finder = EvasionFinder(pathfinder)
    session_manager = DialogueSessionManager(logger=logger)
    dispatcher = InteractionDispatcher(logger=logger)

    bob_resp = DialogueResolution(
        thought="Ich biete an zu weichen.",
        action=TalkAction(target_agent_id="1", message="Ich kann ausweichen.", intent="offer_yield"),
        negotiation_intent="offer_yield",
        new_goal=GoalIntent(name="In Nische ausweichen", intent_type="evade"),
    )
    alice_resp = DialogueResolution(
        thought="Ich nehme das Angebot an.",
        action=TalkAction(target_agent_id="2", message="Danke, ich passiere.", intent="accept"),
        negotiation_intent="accept",
    )

    bob_cognition = MockCognitionProvider(responses=[bob_resp])
    alice_cognition = MockCognitionProvider(responses=[alice_resp])

    coord_bob = DialogueCoordinator(
        logger=logger,
        cognition_provider=bob_cognition,
        pathfinder=pathfinder,
        goal_service=goal_service,
        evasion_finder=evasion_finder,
        action_executor=executor,
        session_manager=session_manager,
        dialogue_history=history,
    )

    coord_alice = DialogueCoordinator(
        logger=logger,
        cognition_provider=alice_cognition,
        pathfinder=pathfinder,
        goal_service=goal_service,
        evasion_finder=evasion_finder,
        action_executor=executor,
        session_manager=session_manager,
        dialogue_history=history,
    )

    # Runde 1: Alice -> Bob
    cmd1 = TalkCommand(source_entity_id="1", target_entity_id="2", message="Weg versperrt.", intent="request_yield")
    fut1 = dispatcher.dispatch(alice, bob, cmd1)
    assert isinstance(fut1, PendingFuture)

    # Runde 2: Bob antwortet
    await asyncio.sleep(0.01)
    bob.commit_staging_messages()
    await coord_bob.handle_incoming_dialogue(bob, [alice, bob])

    for call in logger.log.call_args_list:
        event = call.args[0]
        if getattr(event, "event_type", None) == "dialogue_failed":
            raise RuntimeError(f"Dialogue failed: {event.payload.get('error')}")

    assert fut1.future.done()
    await asyncio.sleep(0)

    # Runde 3: Alice verarbeitet Bobs Antwort und akzeptiert
    alice.commit_staging_messages()
    await coord_alice.handle_incoming_dialogue(alice, [alice, bob])

    for call in logger.log.call_args_list:
        event = call.args[0]
        if getattr(event, "event_type", None) == "dialogue_failed":
            raise RuntimeError(f"Dialogue failed: {event.payload.get('error')}")

    # Endzustand verifizieren: Bob weicht aus, Alice passiert
    assert bob.lifecycle_state == AgentLifecycleState.YIELDING
    assert alice.lifecycle_state in (AgentLifecycleState.PASSING, AgentLifecycleState.IDLE)
    assert any("offer_yield" in r.intent for r in history.records if r.intent)