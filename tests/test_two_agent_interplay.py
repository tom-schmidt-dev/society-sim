from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from src.application.services.action_executor import ActionExecutor
from src.application.services.conflict_coordinator import ConflictCoordinator
from src.application.services.dialogue_coordinator import DialogueCoordinator
from src.application.services.dialogue_history import DialogueHistory
from src.application.services.dialogue_session_manager import DialogueSessionManager
from src.application.services.evasion_finder import EvasionFinder, EvasionResult
from src.application.services.goal_service import GoalService
from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent import Agent
from src.domain.models.cognition import (
    BlockedResolution,
    DialogueResolution,
    EndDialogueAction,
    GoalIntent,
    TalkAction,
)
from src.domain.models.goal import ExecutionPriority, Goal
from src.domain.models.interaction_request import InteractionRequest
from src.domain.models.message import IncomingMessage
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter
from src.domain.services.perception_service import PerceptionService


@pytest.fixture
def mock_pathfinder() -> MagicMock:
    pf = MagicMock(spec=IPathfinder)
    pf.find_path.side_effect = None
    pf.find_path.return_value = []
    return pf


@pytest.fixture
def mock_logger() -> MagicMock:
    return MagicMock(spec=IEventLogger)


@pytest.fixture
def mock_cognition() -> AsyncMock:
    return AsyncMock(spec=ICognitionProvider)


@pytest.fixture
def mock_presenter() -> MagicMock:
    return MagicMock(spec=IPresenter)


@pytest.fixture
def test_env(
    mock_pathfinder: MagicMock,
    mock_logger: MagicMock,
    mock_cognition: AsyncMock,
    mock_presenter: MagicMock,
) -> dict:
    grid = WorldGrid(width=60, height=25)
    for x in range(60):
        grid.set_obstacle(Position(x, 11))
        grid.set_obstacle(Position(x, 13))
    grid.remove_obstacle(Position(30, 11))

    dialogue_history = DialogueHistory()
    session_manager = DialogueSessionManager(max_dialogue_turns=4)
    goal_service = GoalService(mock_logger, mock_cognition, mock_pathfinder)
    evasion_finder = EvasionFinder(mock_pathfinder)
    perception_service = PerceptionService(default_radius=3)

    action_executor = ActionExecutor(
        grid=grid,
        logger=mock_logger,
        dialogue_history=dialogue_history,
        goal_service=goal_service,
        pathfinder=mock_pathfinder,
        evasion_finder=evasion_finder,
    )

    conflict_coordinator = ConflictCoordinator(
        logger=mock_logger,
        cognition_provider=mock_cognition,
        pathfinder=mock_pathfinder,
        goal_service=goal_service,
        evasion_finder=evasion_finder,
        action_executor=action_executor,
        session_manager=session_manager,
        dialogue_history=dialogue_history,
    )

    dialogue_coordinator = DialogueCoordinator(
        logger=mock_logger,
        cognition_provider=mock_cognition,
        pathfinder=mock_pathfinder,
        goal_service=goal_service,
        evasion_finder=evasion_finder,
        action_executor=action_executor,
        session_manager=session_manager,
        dialogue_history=dialogue_history,
    )

    engine = SimulationEngine(
        grid=grid,
        pathfinder=mock_pathfinder,
        presenter=mock_presenter,
        logger=mock_logger,
        cognition_provider=mock_cognition,
        goal_service=goal_service,
        conflict_coordinator=conflict_coordinator,
        dialogue_coordinator=dialogue_coordinator,
        dialogue_history=dialogue_history,
        perception_service=perception_service,
        auditory_radius=3,
    )

    return {
        "engine": engine,
        "grid": grid,
        "action_executor": action_executor,
        "conflict_coordinator": conflict_coordinator,
        "dialogue_coordinator": dialogue_coordinator,
        "goal_service": goal_service,
        "evasion_finder": evasion_finder,
        "cognition": mock_cognition,
        "pathfinder": mock_pathfinder,
        "logger": mock_logger,
    }


class TestTwoAgentFullEvasionLifecycle:
    @pytest.mark.asyncio
    async def test_full_evasion_and_clearance_cycle(self, test_env: dict) -> None:
        engine: SimulationEngine = test_env["engine"]
        cognition: AsyncMock = test_env["cognition"]
        pathfinder: MagicMock = test_env["pathfinder"]
        evasion_finder: EvasionFinder = test_env["evasion_finder"]

        alice = Agent(id="1", name="Alice", position=Position(29, 12))
        alice.push_goal(Goal(name="Ost-Tor", target_position=Position(50, 12)))

        bob = Agent(id="2", name="Bob", position=Position(30, 12))
        bob.push_goal(Goal(name="West-Tor", target_position=Position(10, 12)))

        engine.register_agent(alice)
        engine.register_agent(bob)

        # 1. Blockade auf Kachel (30, 12)
        alice.path = [Position(30, 12)]

        cognition.resolve_blockage.return_value = BlockedResolution(
            thought="Ich bitte Bob, mir Platz zu machen.",
            action=TalkAction(
                target_agent_id="2",
                message="Kannst du bitte ausweichen?",
                reason="Weg versperrt",
                intent="request_yield",
            ),
        )

        await engine.process_tick()

        assert alice.is_waiting_for_reply
        assert alice.interaction_partner_id == "2"
        assert len(bob.inbox) == 1
        assert bob.inbox[0].intent == "request_yield"

        # 2. Bob verhandelt und übernimmt das Ausweichen
        niche_tile = Position(30, 10)
        junction_tile = Position(30, 12)
        niche_path = [Position(30, 11), Position(30, 10)]

        evasion_finder.find_nearest_evasion_tile = MagicMock(
            return_value=EvasionResult(
                target_tile=niche_tile,
                junction_tile=junction_tile,
                path=niche_path,
            )
        )

        cognition.respond_to_dialogue.return_value = DialogueResolution(
            thought="Ich habe eine Nische nahebei, ich weiche aus.",
            action=EndDialogueAction(
                reason="Ich übernehme das Ausweichen",
                final_message="Ich weiche in die Nische aus.",
            ),
            negotiation_intent="offer_yield",
            new_goal=GoalIntent(name="In Nische ausweichen", intent_type="evade"),
        )

        await engine.process_tick()

        assert bob.active_goal is not None
        assert bob.active_goal.name == "In Nische ausweichen"
        assert bob.active_goal.priority == ExecutionPriority.URGENT
        assert bob.path == niche_path
        assert any(msg.is_evasion_notice for msg in alice.inbox)

        # 3. Bob betritt die Junction (30, 12) und sendet Halt-Signal
        bob.position = Position(31, 12)
        bob.active_goal.halt_signaled = False
        bob.path = [junction_tile, Position(30, 11), Position(30, 10)]

        await engine.process_tick()

        assert any(msg.is_halt_request for msg in alice.inbox)

        # 4. Bob erreicht die Nische (30, 10)
        bob.position = niche_tile
        bob.path.clear()

        await engine.process_tick()

        assert bob.is_evasion_locked
        assert bob.active_goal.name == "Nischen-Halt"
        assert any(msg.is_resume_signal for msg in alice.inbox)

        # 5. Alice passiert die Nische (Clearance-Prüfung) & Bob reaktiviert Hauptziel
        alice.position = Position(33, 12)
        alice.path = [Position(34, 12)]

        path_back = [Position(30, 11), Position(30, 12), Position(29, 12)]
        pathfinder.find_path.return_value = path_back

        await engine.process_tick()

        # Bob wurde in Tick 5 durch Alice' Clearance sofort entsperrt und neu bepfadet
        assert not bob.is_evasion_locked
        current_goal_name = getattr(bob.active_goal, "name", None)
        assert current_goal_name == "West-Tor"
        assert bob.path == path_back

        # 6. Beide Agenten setzen ihre reguläre Bewegung fort
        await engine.process_tick()

        assert bob.position == Position(30, 11)
        assert alice.position == Position(34, 12)


class TestDialogueNegotiationInteractions:
    @pytest.mark.asyncio
    async def test_rejection_resets_farewell_flags_on_both_agents(self, test_env: dict) -> None:
        engine: SimulationEngine = test_env["engine"]
        cognition: AsyncMock = test_env["cognition"]

        alice = Agent(id="1", name="Alice", position=Position(10, 12))
        bob = Agent(id="2", name="Bob", position=Position(11, 12))

        engine.register_agent(alice)
        engine.register_agent(bob)

        alice.has_bid_farewell = True
        alice.is_waiting_for_reply = True
        alice.interaction_partner_id = "2"
        bob.peer_bid_farewell = True
        bob.interaction_partner_id = "1"

        bob.receive_message(
            IncomingMessage(
                from_agent_id="1",
                from_agent_name="Alice",
                message="Tschüss!",
                is_farewell=True,
            )
        )

        cognition.respond_to_dialogue.return_value = DialogueResolution(
            thought="Ich kann nicht gehen, ich stecke fest.",
            action=TalkAction(
                target_agent_id="1",
                message="Warte, wir haben die Blockade noch nicht gelöst!",
                reason="Blockade ungelöst",
                intent="reject",
            ),
            negotiation_intent="reject",
        )

        await engine.process_tick()

        assert not alice.has_bid_farewell
        assert not alice.peer_bid_farewell
        assert not bob.has_bid_farewell
        assert not bob.peer_bid_farewell

    @pytest.mark.asyncio
    async def test_mutual_farewell_handshake_clears_all_dialogue_locks(self, test_env: dict) -> None:
        engine: SimulationEngine = test_env["engine"]
        cognition: AsyncMock = test_env["cognition"]

        alice = Agent(id="1", name="Alice", position=Position(10, 12))
        bob = Agent(id="2", name="Bob", position=Position(11, 12))

        engine.register_agent(alice)
        engine.register_agent(bob)

        alice.has_bid_farewell = True
        alice.is_waiting_for_reply = True
        alice.interaction_partner_id = "2"

        bob.peer_bid_farewell = True
        bob.interaction_partner_id = "1"
        bob.receive_message(
            IncomingMessage(
                from_agent_id="1",
                from_agent_name="Alice",
                message="Ich wünsche dir einen schönen Tag, tschüss!",
                is_farewell=True,
            )
        )

        cognition.respond_to_dialogue.return_value = DialogueResolution(
            thought="Alles geklärt, ich verabschiede mich ebenfalls.",
            action=EndDialogueAction(
                reason="Verabschiedung erwidert",
                final_message="Auf Wiedersehen Alice!",
            ),
        )

        await engine.process_tick()

        assert not alice.is_busy
        assert not alice.has_bid_farewell
        assert alice.interaction_partner_id is None

        assert not bob.is_busy
        assert not bob.has_bid_farewell
        assert bob.interaction_partner_id is None


class TestInteractionQueueSequentialResolution:
    @pytest.mark.asyncio
    async def test_queued_agent_is_released_when_blocker_becomes_available(self, test_env: dict) -> None:
        engine: SimulationEngine = test_env["engine"]
        conflict_coordinator: ConflictCoordinator = test_env["conflict_coordinator"]

        alice = Agent(id="1", name="Alice", position=Position(10, 12))
        alice.is_thinking = True

        bob = Agent(id="2", name="Bob", position=Position(9, 12))
        bob.path = [Position(10, 12)]

        engine.register_agent(alice)
        engine.register_agent(bob)

        await conflict_coordinator.resolve_blockage(bob, alice, Position(10, 12), [alice, bob])

        assert len(alice.interaction_queue) == 1
        assert bob.is_waiting_for_reply
        assert bob.interaction_partner_id == "1"

        alice.is_thinking = False
        await engine.process_tick()

        assert len(alice.interaction_queue) == 0
        assert not bob.is_waiting_for_reply
        assert bob.interaction_partner_id is None

    @pytest.mark.asyncio
    async def test_stale_request_in_queue_is_dropped_if_agent_already_rerouted(self, test_env: dict) -> None:
        engine: SimulationEngine = test_env["engine"]

        alice = Agent(id="1", name="Alice", position=Position(10, 12))
        bob = Agent(id="2", name="Bob", position=Position(9, 12))
        bob.path = [Position(9, 11)]
        bob.is_waiting_for_reply = True
        bob.interaction_partner_id = "1"

        alice.interaction_queue.append(
            InteractionRequest(requester_id="2", target_id="1", blocked_pos=Position(10, 12), tick=1)
        )

        engine.register_agent(alice)
        engine.register_agent(bob)

        await engine.process_tick()

        assert len(alice.interaction_queue) == 0
        assert not bob.is_waiting_for_reply


class TestDistanceConstraintsAndPreemption:
    @pytest.mark.asyncio
    async def test_agent_breaks_wait_loop_when_partner_moves_out_of_earshot(self, test_env: dict) -> None:
        engine: SimulationEngine = test_env["engine"]

        alice = Agent(id="1", name="Alice", position=Position(10, 12))
        bob = Agent(id="2", name="Bob", position=Position(11, 12))

        engine.register_agent(alice)
        engine.register_agent(bob)

        alice.is_waiting_for_reply = True
        alice.interaction_partner_id = "2"

        bob.position = Position(20, 12)

        await engine.process_tick()

        assert not alice.is_waiting_for_reply
        assert alice.interaction_partner_id is None

    def test_evasion_preempts_cooperative_goal_and_resumes_cleanly(self, test_env: dict) -> None:
        action_executor: ActionExecutor = test_env["action_executor"]
        goal_service: GoalService = test_env["goal_service"]
        evasion_finder: EvasionFinder = test_env["evasion_finder"]

        alice = Agent(id="1", name="Alice", position=Position(10, 12))
        routine_goal = Goal(name="Hauptziel", priority=ExecutionPriority.ROUTINE)
        coop_goal = Goal(
            name="Partner begleiten",
            priority=ExecutionPriority.COOPERATIVE,
            target_entity_id="2",
        )

        goal_service.push_goal(alice, routine_goal)
        goal_service.push_goal(alice, coop_goal)
        assert alice.active_goal == coop_goal

        evasion_finder.find_nearest_evasion_tile = MagicMock(
            return_value=EvasionResult(
                target_tile=Position(10, 10),
                junction_tile=Position(10, 12),
                path=[Position(10, 11), Position(10, 10)],
            )
        )

        action_executor.execute_evasion(
            agent=alice,
            partner=None,
            blocked_pos=Position(11, 12),
            all_entities=[alice],
            incident_id="inc-test-1",
            thought="Muss Platz machen.",
        )

        assert coop_goal.status == "paused"
        assert alice.active_goal.name == "In Nische ausweichen"
        assert alice.active_goal.priority == ExecutionPriority.URGENT

        goal_service.pop_goal(alice)
        goal_service.resume_goal(alice)

        assert alice.active_goal == coop_goal
        assert coop_goal.status == "active"