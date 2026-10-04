from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.coordination.conflict_coordinator import ConflictCoordinator
from src.application.services.coordination.dialogue_coordinator import DialogueCoordinator
from src.application.services.coordination.dialogue_history import DialogueHistory
from src.application.services.coordination.dialogue_session_manager import DialogueSessionManager
from src.application.services.movement.evasion_finder import EvasionFinder, EvasionResult
from src.application.services.cognition.goal_service import GoalService
from src.application.services.interaction.interaction_dispatcher import InteractionDispatcher
from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_state import AgentLifecycleState
from src.domain.models.planning.cognition import (
    BlockedResolution,
    DialogueResolution,
    EndDialogueAction,
    GoalIntent,
    TalkAction,
    WaitAction,
)
from src.domain.models.planning.goal import ExecutionPriority, Goal
from src.domain.models.communication.message import IncomingMessage
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
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

    interaction_dispatcher = InteractionDispatcher(logger=mock_logger)

    conflict_coordinator = ConflictCoordinator(
        logger=mock_logger,
        cognition_provider=mock_cognition,
        pathfinder=mock_pathfinder,
        goal_service=goal_service,
        evasion_finder=evasion_finder,
        action_executor=action_executor,
        interaction_dispatcher=interaction_dispatcher,
        session_manager=session_manager,
        dialogue_history=dialogue_history,
        enable_deterministic_corridor=False,
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
        enable_deterministic_corridor=False,
        interaction_dispatcher=interaction_dispatcher,
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

        bob.commit_staging_messages()
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

        alice.commit_staging_messages()
        evasion_goal = bob.active_goal
        assert evasion_goal is not None
        assert evasion_goal.name == "In Nische ausweichen"
        assert evasion_goal.priority == ExecutionPriority.URGENT
        evasion_goal.junction_position = junction_tile
        evasion_goal.yield_for_agent_id = alice.id
        evasion_goal.target_position = niche_tile
        assert bob.path == niche_path
        assert any(msg.is_evasion_notice for msg in alice.inbox)

        # 3. Bob betritt die Junction (30, 12) und sendet Halt-Signal
        bob.position = Position(31, 12)
        evasion_goal.halt_signaled = False
        bob.path = [junction_tile, Position(30, 11), Position(30, 10)]

        await engine.process_tick()

        alice.commit_staging_messages()
        assert any(msg.is_halt_request for msg in alice.inbox)

        # 4. Bob erreicht die Nische (30, 10)
        bob.position = niche_tile
        bob.path.clear()

        await engine.process_tick()

        alice.commit_staging_messages()
        assert bob.is_evasion_locked
        hold_goal = bob.active_goal
        assert hold_goal is not None
        assert hold_goal.name == "Nischen-Halt"
        assert any(msg.is_resume_signal for msg in alice.inbox)

        # 5. Alice passiert die Nische & Bob reaktiviert Hauptziel
        alice.position = Position(33, 12)
        alice.path = [Position(34, 12)]

        path_back = [Position(30, 11), Position(30, 12), Position(29, 12)]
        pathfinder.find_path.return_value = path_back

        # Takt 5: Alice absorbiert Resume-Signal (Inbox-Delay)
        await engine.process_tick()
        # Takt 6: Alice rückt auf (34, 12) vor und erteilt Clearance an Bob
        await engine.process_tick()
        # Takt 7: Bob verarbeitet Clearance und reaktiviert West-Tor
        await engine.process_tick()

        assert not bob.is_evasion_locked
        resumed_goal = bob.active_goal
        assert resumed_goal is not None
        assert resumed_goal.name == "West-Tor"
        assert bob.path == path_back

        # 6. Beide Agenten setzen reguläre Bewegung fort
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

        alice.transition_to(AgentLifecycleState.WAITING_FOR_PEER)
        alice.interaction_partner_id = "2"
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

        assert alice.lifecycle_state == AgentLifecycleState.IDLE
        assert bob.lifecycle_state == AgentLifecycleState.IDLE
        assert alice.interaction_partner_id is None
        assert bob.interaction_partner_id is None
        assert len(alice.interaction_mailbox) == 0
        assert len(bob.interaction_mailbox) == 0

    @pytest.mark.asyncio
    async def test_mutual_farewell_handshake_clears_all_dialogue_locks(self, test_env: dict) -> None:
        engine: SimulationEngine = test_env["engine"]
        cognition: AsyncMock = test_env["cognition"]

        alice = Agent(id="1", name="Alice", position=Position(10, 12))
        bob = Agent(id="2", name="Bob", position=Position(11, 12))

        engine.register_agent(alice)
        engine.register_agent(bob)

        alice.transition_to(AgentLifecycleState.WAITING_FOR_PEER)
        alice.interaction_partner_id = "2"
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
        assert alice.lifecycle_state == AgentLifecycleState.IDLE
        assert alice.interaction_partner_id is None

        assert not bob.is_busy
        assert bob.lifecycle_state == AgentLifecycleState.IDLE
        assert bob.interaction_partner_id is None

class TestInteractionQueueSequentialResolution:
    @pytest.mark.asyncio
    async def test_busy_partner_triggers_wait_goal_and_fast_path_releases_when_free(self, test_env: dict) -> None:
        """TC-INT-01: Bei fremdbeschäftigtem Blocker entscheidet die Kognition auf Warten; Fast-Path weckt den Agenten."""
        engine: SimulationEngine = test_env["engine"]
        conflict_coordinator: ConflictCoordinator = test_env["conflict_coordinator"]
        cognition: AsyncMock = test_env["cognition"]

        alice = Agent(id="1", name="Alice", position=Position(10, 12))
        alice.transition_to(AgentLifecycleState.DELIBERATING, reason="Denkt nach")

        bob = Agent(id="2", name="Bob", position=Position(9, 12))
        bob.assign_path([Position(10, 12)])
        bob.push_goal(Goal(name="Hauptziel", target_position=Position(15, 12)))

        engine.register_agent(alice)
        engine.register_agent(bob)

        cognition.resolve_blockage.return_value = BlockedResolution(
            thought="Alice ist beschäftigt, ich warte kurz.",
            action=WaitAction(ticks=2, reason="Partner beschäftigt"),
        )

        await conflict_coordinator.resolve_blockage(bob, alice, Position(10, 12), [alice, bob])

        # Bob hat ein Warteziel erhalten, ohne in einer Queue zu hängen
        wait_goal = bob.active_goal
        assert wait_goal is not None
        assert wait_goal.name == "Warten auf Partner"
        assert wait_goal.yield_for_agent_id == "1"

        # Alice wird frei und räumt das Feld -> Fast-Path beendet das Warten im nächsten Taktzyklus
        alice.transition_to(AgentLifecycleState.IDLE)
        alice.position = Position(10, 13)
        await engine.process_tick()

        resumed_goal = bob.active_goal
        assert resumed_goal is not None
        assert resumed_goal.name == "Hauptziel"

    @pytest.mark.asyncio
    async def test_free_partner_binds_to_mailbox_future_and_waiting_state(self, test_env: dict) -> None:
        """TC-INT-02: Bei freiem Blocker wird ein TalkCommand in die Mailbox übergeben und Bob wartet reaktiv."""
        engine: SimulationEngine = test_env["engine"]
        conflict_coordinator: ConflictCoordinator = test_env["conflict_coordinator"]
        cognition: AsyncMock = test_env["cognition"]

        alice = Agent(id="1", name="Alice", position=Position(10, 12), is_conversational=True)
        bob = Agent(id="2", name="Bob", position=Position(9, 12), is_conversational=True)
        bob.assign_path([Position(10, 12)])
        bob.push_goal(Goal(name="Hauptziel", target_position=Position(15, 12)))

        engine.register_agent(alice)
        engine.register_agent(bob)

        cognition.resolve_blockage.return_value = BlockedResolution(
            thought="Ich bitte um Durchgang.",
            action=TalkAction(target_agent_id="1", message="Platz bitte", intent="request_yield"),
        )

        await conflict_coordinator.resolve_blockage(bob, alice, Position(10, 12), [alice, bob])

        # Bob wartet reaktiv auf Antwort; Alice hat den Befehl in ihrer Mailbox
        assert bob.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER
        assert bob.interaction_partner_id == "1"
        assert len(alice.interaction_mailbox) == 1

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
        evasion_goal = alice.active_goal
        assert evasion_goal is not None
        assert evasion_goal.name == "In Nische ausweichen"
        assert evasion_goal.priority == ExecutionPriority.URGENT

        goal_service.pop_goal(alice)
        goal_service.resume_goal(alice)

        assert alice.active_goal == coop_goal
        assert coop_goal.status == "active"