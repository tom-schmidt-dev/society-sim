from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.coordination.conflict_coordinator import ConflictCoordinator
from src.application.services.coordination.dialogue_history import DialogueHistory
from src.application.services.coordination.dialogue_session_manager import DialogueSessionManager
from src.application.services.movement.evasion_finder import EvasionFinder
from src.application.services.cognition.goal_service import GoalService
from src.application.services.movement.target_search_service import TargetSearchService
from src.application.services.interaction.interaction_dispatcher import InteractionDispatcher
from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_state import AgentLifecycleState
from src.domain.models.interaction.commands import TalkCommand
from src.domain.models.planning.cognition import (
    BlockedResolution,
    DialogueResolution,
    EndDialogueAction,
    GoalEvaluation,
    WaitAction,
)
from src.domain.models.planning.goal import ExecutionPriority, Goal
from src.domain.models.agent.mental_map import AgentMentalMap
from src.domain.models.communication.message import IncomingMessage
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter
from src.domain.services.perception_service import PerceptionService
from src.infrastructure.pathfinding.astar import AStarPathfinder


@pytest.fixture
def mock_pathfinder() -> MagicMock:
    pf = MagicMock(spec=IPathfinder)
    pf.find_path.side_effect = None
    pf.find_path.return_value = [Position(45, 22), Position(46, 22)]
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


# ---------------------------------------------------------------------------
# 1. Test-Klasse: Agenten-Zustände & Goal-Priorisierung
# ---------------------------------------------------------------------------
class TestAgentStateAndPriorities:
    def test_agent_is_busy_derived_purely_from_boolean_flags(self) -> None:
        agent = Agent(id="1", name="Alice", position=Position(5, 5))
        assert not agent.is_busy

        agent.is_listening_to_peer = True
        assert agent.is_busy
        agent.is_listening_to_peer = False

        agent.is_waiting_for_reply = True
        assert agent.is_busy
        agent.is_waiting_for_reply = False

        agent.is_holding_for_junction = True
        assert agent.is_busy
        agent.is_holding_for_junction = False

        agent.is_evasion_locked = True
        assert agent.is_busy
        agent.is_evasion_locked = False

        agent.is_thinking = True
        assert agent.is_busy

    def test_goal_service_pause_and_resume_preserves_stack_hierarchy(
        self, mock_logger: MagicMock, mock_cognition: AsyncMock, mock_pathfinder: MagicMock
    ) -> None:
        service = GoalService(mock_logger, mock_cognition, mock_pathfinder)
        agent = Agent(id="1", name="Alice", position=Position(5, 5))

        routine_goal = Goal(name="Ost-Tor", priority=ExecutionPriority.ROUTINE)
        coop_goal = Goal(name="Partner aufsuchen", priority=ExecutionPriority.COOPERATIVE)

        service.push_goal(agent, routine_goal)
        service.push_goal(agent, coop_goal)
        assert agent.active_goal == coop_goal

        # Pausieren des obersten Ziels
        paused = service.pause_goal(agent)
        assert paused == coop_goal
        assert coop_goal.status == "paused"
        # Nach Pausierung ist das darunterliegende Ziel aktiv
        assert agent.active_goal == routine_goal

        # Reaktivierung
        resumed = service.resume_goal(agent)
        assert resumed == coop_goal
        assert coop_goal.status == "active"
        assert agent.active_goal == coop_goal


# ---------------------------------------------------------------------------
# 2. Test-Klasse: EvasionFinder & Geometrie-Reparatur
# ---------------------------------------------------------------------------
class TestEvasionFinderGeometry:
    def test_evasion_finder_traverses_blocked_pos_neighbor_to_reach_niche(self) -> None:
        pathfinder = AStarPathfinder()
        finder = EvasionFinder(pathfinder)
        mmap = AgentMentalMap(width=50, height=30)

        for x in range(50):
            mmap.update_tile(Position(x, 15), is_walkable=True, tick=1)
        # Nische direkt an Position (25, 14)
        mmap.update_tile(Position(25, 14), is_walkable=True, tick=1)

        start = Position(24, 15)
        blocked_pos = Position(25, 15)
        partner_trajectory = [Position(x, 15) for x in range(40, 20, -1)]

        result = finder.find_nearest_evasion_tile(
            start=start,
            blocked_pos=blocked_pos,
            grid=mmap,
            occupied_positions=set(),
            partner_trajectory=partner_trajectory,
        )

        assert result is not None
        assert result.target_tile == Position(25, 14)
        # Junction muss strikt orthogonal an die Nische angrenzen
        assert result.junction_tile.manhattan_distance(result.target_tile) == 1

    def test_directed_frontier_search_prioritizes_movement_vector(
        self, mock_pathfinder: MagicMock
    ) -> None:
        finder = EvasionFinder(mock_pathfinder)
        mmap = AgentMentalMap(width=50, height=50)

        # Offener Raum um (20, 20)
        for x in range(15, 26):
            for y in range(15, 26):
                mmap.update_tile(Position(x, y), is_walkable=True, tick=1)

        # Vektor nach Osten (1.0, 0.0)
        frontier_east, _ = finder._find_nearest_frontier(
            start=Position(20, 20),
            grid=mmap,
            frontier_forbidden=set(),
            direction_vector=(1.0, 0.0),
        )

        # Vektor nach Westen (-1.0, 0.0)
        frontier_west, _ = finder._find_nearest_frontier(
            start=Position(20, 20),
            grid=mmap,
            frontier_forbidden=set(),
            direction_vector=(-1.0, 0.0),
        )

        assert frontier_east is not None
        assert frontier_west is not None
        assert frontier_east.x > 20
        assert frontier_west.x < 20


# ---------------------------------------------------------------------------
# 3. Test-Klasse: Schutz vor fehlerhaftem Goal-Popping
# ---------------------------------------------------------------------------
class TestGoalServiceProtection:
    @pytest.mark.asyncio
    async def test_evaluate_sub_goal_completion_strictly_protects_evasion_lock(
        self, mock_logger: MagicMock, mock_cognition: AsyncMock, mock_pathfinder: MagicMock
    ) -> None:
        service = GoalService(mock_logger, mock_cognition, mock_pathfinder)
        grid = WorldGrid(width=20, height=20)
        agent = Agent(id="1", name="Alice", position=Position(5, 5))

        hold_goal = Goal(
            name="Nischen-Halt",
            is_evasion_hold=True,
            holds_position=True,
            priority=ExecutionPriority.URGENT,
        )
        service.push_goal(agent, hold_goal)
        agent.is_evasion_locked = True

        # LLM behauptet fälschlicherweise Abschluss
        mock_cognition.evaluate_goal_status.return_value = GoalEvaluation(
            thought="Ich bin fertig.", is_completed=True, reason="Fertig"
        )

        await service.evaluate_sub_goal_completion(agent, grid, recent_dialogues=[])

        # Ziel darf nicht gepoppt worden sein, keine LLM-Inferenz gestartet
        assert agent.active_goal == hold_goal
        assert agent.is_evasion_locked
        mock_cognition.evaluate_goal_status.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Test-Klasse: Auditive Reichweite & Verabschiedungs-Handshake
# ---------------------------------------------------------------------------
class TestAuditoryRadiusAndFarewellHandshake:
    @pytest.mark.asyncio
    async def test_auditory_range_exceeded_immediately_releases_waiting_agent(
        self, mock_pathfinder: MagicMock, mock_presenter: MagicMock, mock_logger: MagicMock, mock_cognition: AsyncMock
    ) -> None:
        grid = WorldGrid(width=30, height=30)
        engine = SimulationEngine(
            grid=grid,
            pathfinder=mock_pathfinder,
            presenter=mock_presenter,
            logger=mock_logger,
            cognition_provider=mock_cognition,
            auditory_radius=3,
        )

        alice = Agent(id="1", name="Alice", position=Position(5, 5))
        bob = Agent(id="2", name="Bob", position=Position(15, 5))  # Distanz = 10 > 3

        engine.register_agent(alice)
        engine.register_agent(bob)

        alice.is_waiting_for_reply = True
        alice.interaction_partner_id = "2"

        await engine.process_tick()

        # Alice muss wegen Überschreitung von auditory_radius entsperrt sein
        assert not alice.is_waiting_for_reply
        assert alice.interaction_partner_id is None

    @pytest.mark.asyncio
    async def test_mutual_farewell_handshake_unlocks_both_agents(
            self, mock_pathfinder: MagicMock, mock_presenter: MagicMock, mock_logger: MagicMock,
            mock_cognition: AsyncMock
    ) -> None:
        grid = WorldGrid(width=30, height=30)
        engine = SimulationEngine(
            grid=grid,
            pathfinder=mock_pathfinder,
            presenter=mock_presenter,
            logger=mock_logger,
            cognition_provider=mock_cognition,
            auditory_radius=3,
        )

        alice = Agent(id="1", name="Alice", position=Position(5, 5))
        bob = Agent(id="2", name="Bob", position=Position(6, 5))  # Distanz = 1 <= 3

        engine.register_agent(alice)
        engine.register_agent(bob)

        alice.interaction_partner_id = "2"
        alice.transition_to(AgentLifecycleState.WAITING_FOR_PEER)

        bob.interaction_partner_id = "1"
        bob.receive_message(
            IncomingMessage(
                from_agent_id="1",
                from_agent_name="Alice",
                message="Tschüss!",
                is_farewell=True,
            )
        )

        mock_cognition.respond_to_dialogue.return_value = DialogueResolution(
            thought="Verabschiedung erwidert",
            action=EndDialogueAction(
                reason="Verabschiedung erwidert",
                final_message="Auf Wiedersehen!",
            ),
        )

        await engine.process_tick()

        assert not alice.is_waiting_for_reply
        assert alice.interaction_partner_id is None
        assert alice.lifecycle_state == AgentLifecycleState.IDLE

        assert not bob.is_waiting_for_reply
        assert bob.interaction_partner_id is None
        assert bob.lifecycle_state == AgentLifecycleState.IDLE


# ---------------------------------------------------------------------------
# 5. Test-Klasse: FIFO-Warteschlange & Lazy Validation
# ---------------------------------------------------------------------------
class TestInteractionQueueMechanics:
    @pytest.mark.asyncio
    async def test_conflict_coordinator_queues_request_if_blocker_is_busy(
        self, mock_pathfinder: MagicMock, mock_logger: MagicMock, mock_cognition: AsyncMock
    ) -> None:
        """Prüft, dass bei beschäftigtem Blocker autonomes Warten über Zeiteinheiten statt Queues greift."""
        grid = WorldGrid(width=30, height=30)
        dialogue_history = DialogueHistory()
        session_manager = DialogueSessionManager()
        goal_service = GoalService(mock_logger, mock_cognition, mock_pathfinder)
        evasion_finder = EvasionFinder(mock_pathfinder)
        dispatcher = InteractionDispatcher(logger=mock_logger)
        action_executor = ActionExecutor(
            grid, mock_logger, dialogue_history, goal_service, mock_pathfinder, evasion_finder=evasion_finder
        )

        coordinator = ConflictCoordinator(
            logger=mock_logger,
            cognition_provider=mock_cognition,
            pathfinder=mock_pathfinder,
            goal_service=goal_service,
            evasion_finder=evasion_finder,
            action_executor=action_executor,
            interaction_dispatcher=dispatcher,
            session_manager=session_manager,
            dialogue_history=dialogue_history,
            enable_deterministic_corridor=False,
        )

        alice = Agent(id="1", name="Alice", position=Position(10, 10))
        alice.assign_path([Position(11, 10)])
        alice.push_goal(Goal(name="Hauptziel", target_position=Position(15, 10)))

        bob = Agent(id="2", name="Bob", position=Position(11, 10))
        bob.interaction_partner_id = "3"  # Bob ist fremdbeschäftigt

        mock_cognition.resolve_blockage.return_value = BlockedResolution(
            thought="Bob ist beschäftigt, ich warte.",
            action=WaitAction(ticks=2, reason="Partner beschäftigt"),
        )

        await coordinator.resolve_blockage(alice, bob, Position(11, 10), [alice, bob])

        assert alice.active_goal is not None
        assert alice.active_goal.name == "Warten auf Partner"
        assert alice.active_goal.yield_for_agent_id == "2"
        assert alice.lifecycle_state == AgentLifecycleState.IDLE

    @pytest.mark.asyncio
    async def test_lazy_validation_drops_stale_queue_entry_when_blocker_cleared(
        self, mock_pathfinder: MagicMock, mock_presenter: MagicMock, mock_logger: MagicMock, mock_cognition: AsyncMock
    ) -> None:
        """Prüft TOCTOU-Schutz: Räumt der Blocker während Deliberation das Feld, wird die Aktion verworfen."""
        grid = WorldGrid(width=30, height=30)
        engine = SimulationEngine(
            grid=grid,
            pathfinder=mock_pathfinder,
            presenter=mock_presenter,
            logger=mock_logger,
            cognition_provider=mock_cognition,
            auditory_radius=3,
            enable_deterministic_corridor=False,
        )

        alice = Agent(id="1", name="Alice", position=Position(10, 10))
        alice.assign_path([Position(11, 10)])
        alice.push_goal(Goal(name="Hauptziel", target_position=Position(15, 10)))

        bob = Agent(id="2", name="Bob", position=Position(11, 10))
        bob.interaction_partner_id = "3"

        engine.register_agent(alice)
        engine.register_agent(bob)

        async def move_blocker_during_thought(context):
            bob.position = Position(11, 9)
            return BlockedResolution(
                thought="Wollte eigentlich warten.",
                action=WaitAction(ticks=2, reason="Warten"),
            )

        mock_cognition.resolve_blockage.side_effect = move_blocker_during_thought

        await engine._conflict_coordinator.resolve_blockage(
            agent=alice,
            blocker=bob,
            blocked_pos=Position(11, 10),
            all_entities=engine._entities,
        )

        logged_events = [call_args[0][0].event_type for call_args in mock_logger.log.call_args_list]
        assert "action_dropped_stale" in logged_events
        assert alice.active_goal.name == "Hauptziel"

# ---------------------------------------------------------------------------
# 6. Test-Klasse: Epistemische Zielsuche (TargetSearchService)
# ---------------------------------------------------------------------------
class TestTargetSearchServiceCascade:
    def test_stage_1_visual_search_finds_target_directly_in_range(
        self, mock_pathfinder: MagicMock, mock_logger: MagicMock
    ) -> None:
        grid = WorldGrid(width=30, height=30)
        perception = PerceptionService(default_radius=3)
        evasion_finder = EvasionFinder(mock_pathfinder)
        service = TargetSearchService(mock_pathfinder, perception, evasion_finder, mock_logger)

        alice = Agent(id="1", name="Alice", position=Position(10, 10))
        bob = Agent(id="2", name="Bob", position=Position(12, 10))  # Distanz = 2 <= 3

        res = service.search_target(alice, "2", grid, [alice, bob], current_tick=1)

        assert res.phase == "visual"
        assert res.target_position == Position(12, 10)
        mock_pathfinder.find_path.assert_called_with(alice.position, Position(12, 10), alice.mental_map)

    def test_stage_2_projected_memory_search_when_outside_visual_range(
        self, mock_pathfinder: MagicMock, mock_logger: MagicMock
    ) -> None:
        grid = WorldGrid(width=30, height=30)
        perception = PerceptionService(default_radius=3)
        evasion_finder = EvasionFinder(mock_pathfinder)
        service = TargetSearchService(mock_pathfinder, perception, evasion_finder, mock_logger)

        alice = Agent(id="1", name="Alice", position=Position(5, 5))
        bob = Agent(id="2", name="Bob", position=Position(20, 20))

        # Gedächtniseintrag mit Geschwindigkeitsvektor simulieren
        alice.memory.update_entity_perception("2", "Bob", pos=Position(10, 5), tick=1)
        alice.memory.known_entities["2"].last_observed_velocity = (1.0, 0.0)

        res = service.search_target(alice, "2", grid, [alice, bob], current_tick=2)

        assert res.phase == "projected"
        assert res.target_position == Position(11, 5)