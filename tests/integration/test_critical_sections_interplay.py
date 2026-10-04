from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.coordination.conflict_coordinator import ConflictCoordinator
from src.application.services.coordination.critical_section_coordinator import CriticalSectionCoordinator
from src.application.services.coordination.dialogue_coordinator import DialogueCoordinator
from src.application.services.coordination.dialogue_history import DialogueHistory
from src.application.services.coordination.dialogue_session_manager import DialogueSessionManager
from src.application.services.movement.evasion_finder import EvasionFinder
from src.application.services.cognition.goal_service import GoalService
from src.application.services.interaction.interaction_dispatcher import InteractionDispatcher
from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_state import AgentLifecycleState
from src.domain.models.interaction.commands import TalkCommand
from src.domain.models.planning.cognition import (
    BlockedResolution,
    DialogueResolution,
    EndDialogueAction,
    TalkAction,
    WaitAction,
)
from src.domain.models.planning.goal import ExecutionPriority, Goal
from src.domain.models.communication.message import IncomingMessage
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter


@pytest.fixture
def mock_logger() -> MagicMock:
    logger = MagicMock(spec=IEventLogger)
    logger.log = MagicMock()
    return logger


@pytest.fixture
def mock_pathfinder() -> MagicMock:
    pathfinder = MagicMock(spec=IPathfinder)
    pathfinder.find_path = MagicMock(return_value=[Position(1, 1), Position(1, 2)])
    return pathfinder


@pytest.fixture
def mock_cognition() -> MagicMock:
    cognition = MagicMock(spec=ICognitionProvider)
    cognition.resolve_blockage = AsyncMock(
        return_value=BlockedResolution(
            thought="Standard-Blockadeloesung",
            action=WaitAction(action_type="wait", ticks=1, reason="Warten"),
        )
    )
    cognition.respond_to_dialogue = AsyncMock(
        return_value=DialogueResolution(
            thought="Standard-Dialog",
            action=EndDialogueAction(action_type="end_dialogue", reason="Gespraechsende"),
        )
    )
    cognition.evaluate_goal_status = AsyncMock()
    return cognition


@pytest.fixture
def mock_presenter() -> MagicMock:
    presenter = MagicMock(spec=IPresenter)
    presenter.render = MagicMock()
    return presenter


def create_agent(
    agent_id: str,
    name: str,
    pos: Position,
    energy: int = 100,
    grid_size: tuple[int, int] = (20, 20),
) -> Agent:
    agent = Agent(id=agent_id, name=name, position=pos, energy=energy)
    agent.mental_map.set_bounds(grid_size[0], grid_size[1])
    return agent


def create_entity(
    entity_id: str,
    name: str,
    pos: Position,
    is_conversational: bool = False,
    is_passable: bool = False,
    entity_type: str = "chest",
) -> WorldEntity:
    entity = WorldEntity(id=entity_id, name=name, position=pos, is_conversational=is_conversational)
    entity.is_passable = is_passable
    entity.entity_type = entity_type
    return entity


class TestCriticalSectionCoordinator:
    def test_acquire_initial_success(self, mock_logger: MagicMock) -> None:
        coordinator = CriticalSectionCoordinator(logger=mock_logger, tick_provider=lambda: 1)
        agent = create_agent("agent_1", "Alice", Position(0, 0))

        acquired = coordinator.acquire_or_queue(
            agent=agent,
            resource_key="target:(5,5)",
            priority=ExecutionPriority.ROUTINE,
        )

        assert acquired is True
        assert coordinator.is_holder("target:(5,5)", agent.id) is True
        assert coordinator.get_holder("target:(5,5)") == agent.id

    def test_acquire_idempotent_for_current_holder(self, mock_logger: MagicMock) -> None:
        coordinator = CriticalSectionCoordinator(logger=mock_logger, tick_provider=lambda: 1)
        agent = create_agent("agent_1", "Alice", Position(0, 0))

        assert coordinator.acquire_or_queue(agent, "res:1", ExecutionPriority.ROUTINE) is True
        assert coordinator.acquire_or_queue(agent, "res:1", ExecutionPriority.ROUTINE) is True

    def test_priority_queueing_three_agents(self, mock_logger: MagicMock) -> None:
        coordinator = CriticalSectionCoordinator(logger=mock_logger, tick_provider=lambda: 1)
        agent_holder = create_agent("agent_holder", "Holder", Position(0, 0))
        agent_routine = create_agent("agent_routine", "Routine", Position(1, 0))
        agent_urgent = create_agent("agent_urgent", "Urgent", Position(2, 0))
        agent_coop = create_agent("agent_coop", "Coop", Position(3, 0))

        coordinator.acquire_or_queue(agent_holder, "chokepoint", ExecutionPriority.ROUTINE)

        assert coordinator.acquire_or_queue(agent_routine, "chokepoint", ExecutionPriority.ROUTINE) is False
        assert coordinator.acquire_or_queue(agent_urgent, "chokepoint", ExecutionPriority.URGENT) is False
        assert coordinator.acquire_or_queue(agent_coop, "chokepoint", ExecutionPriority.COOPERATIVE) is False

        section = coordinator.get_or_create_section("chokepoint")
        queued_ids = [req.agent_id for req in section.wait_queue]

        assert queued_ids == ["agent_urgent", "agent_coop", "agent_routine"]

    def test_proposed_order_tie_breaker(self, mock_logger: MagicMock) -> None:
        coordinator = CriticalSectionCoordinator(logger=mock_logger, tick_provider=lambda: 1)
        agent_holder = create_agent("holder", "Holder", Position(0, 0))
        agent_a = create_agent("agent_a", "Alice", Position(1, 0))
        agent_b = create_agent("agent_b", "Bob", Position(2, 0))

        coordinator.acquire_or_queue(agent_holder, "res:target", ExecutionPriority.ROUTINE)

        coordinator.acquire_or_queue(agent_a, "res:target", ExecutionPriority.COOPERATIVE, proposed_order=2)
        coordinator.acquire_or_queue(agent_b, "res:target", ExecutionPriority.COOPERATIVE, proposed_order=1)

        section = coordinator.get_or_create_section("res:target")
        assert section.wait_queue[0].agent_id == "agent_b"
        assert section.wait_queue[1].agent_id == "agent_a"

    def test_fifo_fallback_on_identical_priority_and_order(self, mock_logger: MagicMock) -> None:
        current_tick = 1
        coordinator = CriticalSectionCoordinator(logger=mock_logger, tick_provider=lambda: current_tick)
        agent_holder = create_agent("holder", "Holder", Position(0, 0))
        agent_first = create_agent("agent_1", "First", Position(1, 0))
        agent_second = create_agent("agent_2", "Second", Position(2, 0))

        coordinator.acquire_or_queue(agent_holder, "door", ExecutionPriority.ROUTINE)

        current_tick = 2
        coordinator.acquire_or_queue(agent_first, "door", ExecutionPriority.ROUTINE)
        current_tick = 5
        coordinator.acquire_or_queue(agent_second, "door", ExecutionPriority.ROUTINE)

        section = coordinator.get_or_create_section("door")
        assert section.wait_queue[0].agent_id == "agent_1"
        assert section.wait_queue[1].agent_id == "agent_2"

    def test_release_transfers_to_next_in_queue(self, mock_logger: MagicMock) -> None:
        coordinator = CriticalSectionCoordinator(logger=mock_logger, tick_provider=lambda: 1)
        agent_1 = create_agent("agent_1", "A1", Position(0, 0))
        agent_2 = create_agent("agent_2", "A2", Position(1, 0))

        coordinator.acquire_or_queue(agent_1, "terminal", ExecutionPriority.ROUTINE)
        coordinator.acquire_or_queue(agent_2, "terminal", ExecutionPriority.ROUTINE)

        next_agent = coordinator.release("agent_1", "terminal", mark_completed=False)

        assert next_agent == "agent_2"
        assert coordinator.get_holder("terminal") == "agent_2"
        assert coordinator.is_holder("terminal", "agent_2") is True

    def test_release_with_mark_completed(self, mock_logger: MagicMock) -> None:
        coordinator = CriticalSectionCoordinator(logger=mock_logger, tick_provider=lambda: 42)
        agent = create_agent("agent_1", "A1", Position(0, 0))

        coordinator.acquire_or_queue(agent, "chest:1", ExecutionPriority.ROUTINE)
        next_agent = coordinator.release("agent_1", "chest:1", mark_completed=True)

        assert next_agent is None
        assert coordinator.is_completed("chest:1") is True
        section = coordinator.get_or_create_section("chest:1")
        assert section.completed_by == "agent_1"
        assert section.completed_tick == 42

    def test_unauthorized_release_is_rejected(self, mock_logger: MagicMock) -> None:
        coordinator = CriticalSectionCoordinator(logger=mock_logger, tick_provider=lambda: 1)
        holder = create_agent("holder", "Holder", Position(0, 0))
        intruder = create_agent("intruder", "Intruder", Position(1, 0))

        coordinator.acquire_or_queue(holder, "key", ExecutionPriority.ROUTINE)
        result = coordinator.release(intruder.id, "key", mark_completed=True)

        assert result is None
        assert coordinator.get_holder("key") == holder.id
        assert coordinator.is_completed("key") is False


class TestGoalServiceCriticalSections:
    def test_validate_goal_necessity_when_not_completed(self, mock_logger: MagicMock, mock_pathfinder: MagicMock, mock_cognition: MagicMock) -> None:
        service = GoalService(mock_logger, mock_cognition, mock_pathfinder)
        agent = create_agent("agent_1", "A1", Position(0, 0))
        goal = Goal(name="Hebel umlegen", target_position=Position(5, 5))
        service.push_goal(agent, goal)

        is_needed = service.validate_goal_necessity(agent, goal, is_already_completed=False)

        assert is_needed is True
        assert agent.active_goal == goal

    def test_validate_goal_necessity_when_already_completed(self, mock_logger: MagicMock, mock_pathfinder: MagicMock, mock_cognition: MagicMock) -> None:
        service = GoalService(mock_logger, mock_cognition, mock_pathfinder)
        agent = create_agent("agent_1", "A1", Position(0, 0))
        goal = Goal(name="Hebel umlegen", target_position=Position(5, 5))
        agent.path = [Position(1, 0), Position(2, 0)]
        service.push_goal(agent, goal)

        is_needed = service.validate_goal_necessity(agent, goal, is_already_completed=True)

        assert is_needed is False
        assert agent.active_goal is None
        assert agent.has_path is False
        logged_types = [call_args[0][0].event_type for call_args in mock_logger.log.call_args_list]
        assert "goal_deemed_unnecessary" in logged_types

    def test_pause_and_resume_goal(self, mock_logger: MagicMock, mock_pathfinder: MagicMock, mock_cognition: MagicMock) -> None:
        service = GoalService(mock_logger, mock_cognition, mock_pathfinder)
        agent = create_agent("agent_1", "A1", Position(0, 0))
        routine_goal = Goal(name="Erkunde", priority=ExecutionPriority.ROUTINE)
        service.push_goal(agent, routine_goal)

        paused = service.pause_goal(agent, routine_goal)
        assert paused is not None
        assert paused.status == "paused"

        resumed = service.resume_goal(agent)
        assert resumed is not None
        assert resumed.status == "active"
        assert agent.active_goal is not None
        assert agent.active_goal.status == "active"


class TestActionExecutorCriticalSections:
    def test_execute_inspection_competition_between_two_agents(
        self,
        mock_logger: MagicMock,
        mock_pathfinder: MagicMock,
    ) -> None:
        grid = WorldGrid(width=10, height=10)
        history = DialogueHistory()
        goal_service = GoalService(mock_logger, MagicMock(), mock_pathfinder)
        coordinator = CriticalSectionCoordinator(logger=mock_logger)
        executor = ActionExecutor(
            grid=grid,
            logger=mock_logger,
            dialogue_history=history,
            goal_service=goal_service,
            pathfinder=mock_pathfinder,
            critical_section_coordinator=coordinator,
        )

        agent_1 = create_agent("agent_1", "Alice", Position(1, 1))
        agent_2 = create_agent("agent_2", "Bob", Position(1, 2))
        target_entity = create_entity("box_1", "Kiste", Position(2, 2))

        goal_1 = Goal(name="Inspiziere Kiste", priority=ExecutionPriority.ROUTINE)
        goal_2 = Goal(name="Inspiziere Kiste", priority=ExecutionPriority.ROUTINE)
        goal_service.push_goal(agent_1, goal_1)
        goal_service.push_goal(agent_2, goal_2)

        executor.execute_inspection(agent_1, target_entity, "inc-1", current_tick=1)
        assert coordinator.is_completed("entity:box_1") is True

        coordinator_busy = CriticalSectionCoordinator(logger=mock_logger)
        coordinator_busy.acquire_or_queue(agent_1, "entity:box_1", ExecutionPriority.ROUTINE)

        executor_busy = ActionExecutor(
            grid=grid,
            logger=mock_logger,
            dialogue_history=history,
            goal_service=goal_service,
            pathfinder=mock_pathfinder,
            critical_section_coordinator=coordinator_busy,
        )

        executor_busy.execute_inspection(agent_2, target_entity, "inc-2", current_tick=2)
        assert agent_2.active_goal is not None
        assert agent_2.active_goal.status == "paused"

    def test_execute_probe_critical_section(
        self,
        mock_logger: MagicMock,
        mock_pathfinder: MagicMock,
    ) -> None:
        grid = WorldGrid(width=10, height=10)
        history = DialogueHistory()
        goal_service = GoalService(mock_logger, MagicMock(), mock_pathfinder)
        coordinator = CriticalSectionCoordinator(logger=mock_logger)
        executor = ActionExecutor(
            grid=grid,
            logger=mock_logger,
            dialogue_history=history,
            goal_service=goal_service,
            pathfinder=mock_pathfinder,
            critical_section_coordinator=coordinator,
        )

        agent = create_agent("agent_1", "Alice", Position(0, 0))
        target = create_entity("wall_1", "Wand", Position(0, 1), is_passable=False, entity_type="wall")
        agent.memory.update_entity_perception(target.id, target.name, target.position, tick=1)
        agent.memory.record_inspection(target.id, target.entity_type)
        goal = Goal(name="Erprobe Wand", priority=ExecutionPriority.ROUTINE)
        goal_service.push_goal(agent, goal)

        executor.execute_probe(agent, target, "inc-probe", current_tick=1)

        assert coordinator.is_completed("entity:wall_1") is True
        assert agent.memory.get_entity_walkability("wall_1") is False


class TestLLMErrorHandlingAndEdgeCases:
    @pytest.mark.asyncio
    async def test_resolve_blockage_cognition_timeout_exception(
            self,
            mock_logger: MagicMock,
            mock_pathfinder: MagicMock,
    ) -> None:
        cognition = MagicMock(spec=ICognitionProvider)
        cognition.resolve_blockage = AsyncMock(side_effect=TimeoutError("LLM Inferenz überschritten"))

        session_mgr = DialogueSessionManager(max_dialogue_turns=2)
        goal_service = GoalService(mock_logger, cognition, mock_pathfinder)
        history = DialogueHistory()
        dispatcher = InteractionDispatcher(logger=mock_logger)
        executor = ActionExecutor(
            grid=WorldGrid(10, 10),
            logger=mock_logger,
            dialogue_history=history,
            goal_service=goal_service,
            pathfinder=mock_pathfinder,
        )
        coordinator = ConflictCoordinator(
            logger=mock_logger,
            cognition_provider=cognition,
            pathfinder=mock_pathfinder,
            goal_service=goal_service,
            evasion_finder=EvasionFinder(mock_pathfinder),
            action_executor=executor,
            interaction_dispatcher=dispatcher,
            session_manager=session_mgr,
            dialogue_history=history,
            enable_deterministic_corridor=False,
        )

        agent = create_agent("agent_1", "Alice", Position(1, 1))
        partner = create_agent("agent_2", "Bob", Position(1, 2))
        agent.path = [Position(1, 2)]

        agent.is_thinking = True
        await coordinator.resolve_blockage(agent, partner, Position(1, 2), [agent, partner])

        assert agent.is_thinking is False
        assert agent.is_waiting_for_reply is False
        logged_types = [call_args[0][0].event_type for call_args in mock_logger.log.call_args_list]
        assert "cognition_failed" in logged_types

    @pytest.mark.asyncio
    async def test_dialogue_coordinator_llm_runtime_error(
        self,
        mock_logger: MagicMock,
        mock_pathfinder: MagicMock,
    ) -> None:
        cognition = MagicMock(spec=ICognitionProvider)
        cognition.respond_to_dialogue = AsyncMock(side_effect=RuntimeError("Syntaxfehler im JSON"))

        session_mgr = DialogueSessionManager(max_dialogue_turns=2)
        goal_service = GoalService(mock_logger, cognition, mock_pathfinder)
        history = DialogueHistory()
        executor = ActionExecutor(
            grid=WorldGrid(10, 10),
            logger=mock_logger,
            dialogue_history=history,
            goal_service=goal_service,
            pathfinder=mock_pathfinder,
        )
        dialogue_coord = DialogueCoordinator(
            logger=mock_logger,
            cognition_provider=cognition,
            pathfinder=mock_pathfinder,
            goal_service=goal_service,
            evasion_finder=EvasionFinder(mock_pathfinder),
            action_executor=executor,
            session_manager=session_mgr,
            dialogue_history=history,
        )

        agent = create_agent("agent_1", "Alice", Position(1, 1))
        partner = create_agent("agent_2", "Bob", Position(1, 2))
        agent.inbox.append(IncomingMessage(from_agent_id="agent_2", from_agent_name="Bob", message="Hallo"))

        agent.is_thinking = True
        await dialogue_coord.handle_incoming_dialogue(agent, [agent, partner])

        assert agent.is_thinking is False
        logged_types = [call_args[0][0].event_type for call_args in mock_logger.log.call_args_list]
        assert "dialogue_failed" in logged_types

    @pytest.mark.asyncio
    async def test_empty_or_faulty_action_target_sanitization(
        self,
        mock_logger: MagicMock,
        mock_pathfinder: MagicMock,
    ) -> None:
        cognition = MagicMock(spec=ICognitionProvider)
        cognition.resolve_blockage = AsyncMock(
            return_value=BlockedResolution(
                thought="Ich rede mit mir selbst.",
                action=TalkAction(
                    action_type="talk",
                    target_agent_id="agent_1",
                    message="Selbstgespräch",
                    reason="Klärungsversuch",
                ),
            )
        )

        session_mgr = DialogueSessionManager(max_dialogue_turns=2)
        goal_service = GoalService(mock_logger, cognition, mock_pathfinder)
        history = DialogueHistory()
        dispatcher = InteractionDispatcher(logger=mock_logger)
        executor = ActionExecutor(
            grid=WorldGrid(10, 10),
            logger=mock_logger,
            dialogue_history=history,
            goal_service=goal_service,
            pathfinder=mock_pathfinder,
        )
        coordinator = ConflictCoordinator(
            logger=mock_logger,
            cognition_provider=cognition,
            pathfinder=mock_pathfinder,
            goal_service=goal_service,
            evasion_finder=EvasionFinder(mock_pathfinder),
            action_executor=executor,
            interaction_dispatcher=dispatcher,
            session_manager=session_mgr,
            dialogue_history=history,
            enable_deterministic_corridor=False,
        )

        agent = create_agent("agent_1", "Alice", Position(1, 1))
        blocker = create_agent("agent_2", "Bob", Position(1, 2))
        agent.path = [Position(1, 2)]

        await coordinator.resolve_blockage(agent, blocker, Position(1, 2), [agent, blocker])

        blocker.commit_staging_messages()
        assert len(blocker.inbox) == 1
        assert blocker.inbox[0].from_agent_id == "agent_1"
        assert blocker.inbox[0].message == "Selbstgespräch"


class TestMultiAgentCriticalSectionScenarios:
    @pytest.mark.asyncio
    async def test_three_agents_competing_for_same_goal_necessity_abort(
        self,
        mock_logger: MagicMock,
        mock_presenter: MagicMock,
        mock_cognition: MagicMock,
    ) -> None:
        grid = WorldGrid(width=10, height=10)
        pathfinder = MagicMock(spec=IPathfinder)
        target_pos = Position(5, 5)

        pathfinder.find_path = MagicMock(return_value=[Position(5, 5)])

        engine = SimulationEngine(
            grid=grid,
            pathfinder=pathfinder,
            presenter=mock_presenter,
            logger=mock_logger,
            cognition_provider=mock_cognition,
        )

        agent_1 = create_agent("a1", "Alice", Position(5, 4))
        agent_2 = create_agent("a2", "Bob", Position(4, 5))
        agent_3 = create_agent("a3", "Charlie", Position(6, 5))

        engine.register_agent(agent_1)
        engine.register_agent(agent_2)
        engine.register_agent(agent_3)

        resource_key = f"pos:{target_pos.x},{target_pos.y}"

        assert engine.critical_section_coordinator.acquire_or_queue(
            agent_1, resource_key, ExecutionPriority.ROUTINE
        ) is True

        assert engine.critical_section_coordinator.acquire_or_queue(
            agent_2, resource_key, ExecutionPriority.ROUTINE, proposed_order=1
        ) is False
        assert engine.critical_section_coordinator.acquire_or_queue(
            agent_3, resource_key, ExecutionPriority.ROUTINE, proposed_order=2
        ) is False

        engine.set_agent_target(agent_1.id, target_pos, "Schatz heben")
        engine.set_agent_target(agent_2.id, target_pos, "Schatz heben")
        engine.set_agent_target(agent_3.id, target_pos, "Schatz heben")

        engine._goal_service.pause_goal(agent_2)
        engine._goal_service.pause_goal(agent_3)
        agent_2.clear_path()
        agent_3.clear_path()

        assert agent_2.active_goal is not None
        assert agent_2.active_goal.status == "paused"
        assert agent_3.active_goal is not None
        assert agent_3.active_goal.status == "paused"

        agent_1.position = target_pos

        await engine.process_tick()

        assert agent_1.active_goal is None
        assert engine.critical_section_coordinator.is_completed(resource_key) is True
        assert agent_2.active_goal is None

    @pytest.mark.asyncio
    async def test_urgent_preemption_reorders_queue_over_routine_agents(
        self,
        mock_logger: MagicMock,
        mock_presenter: MagicMock,
        mock_cognition: MagicMock,
        mock_pathfinder: MagicMock,
    ) -> None:
        grid = WorldGrid(width=10, height=10)
        engine = SimulationEngine(
            grid=grid,
            pathfinder=mock_pathfinder,
            presenter=mock_presenter,
            logger=mock_logger,
            cognition_provider=mock_cognition,
        )

        agent_holder = create_agent("holder", "Holder", Position(0, 0))
        agent_routine_1 = create_agent("r1", "Routine 1", Position(1, 0))
        agent_routine_2 = create_agent("r2", "Routine 2", Position(2, 0))
        agent_urgent = create_agent("urg", "Urgent Evasion", Position(3, 0))

        engine.register_agent(agent_holder)
        engine.register_agent(agent_routine_1)
        engine.register_agent(agent_routine_2)
        engine.register_agent(agent_urgent)

        chokepoint = "chokepoint_pos:(3,3)"

        engine.critical_section_coordinator.acquire_or_queue(agent_holder, chokepoint, ExecutionPriority.ROUTINE)
        engine.critical_section_coordinator.acquire_or_queue(agent_routine_1, chokepoint, ExecutionPriority.ROUTINE)
        engine.critical_section_coordinator.acquire_or_queue(agent_routine_2, chokepoint, ExecutionPriority.ROUTINE)
        engine.critical_section_coordinator.acquire_or_queue(agent_urgent, chokepoint, ExecutionPriority.URGENT)

        section = engine.critical_section_coordinator.get_or_create_section(chokepoint)
        assert section.wait_queue[0].agent_id == "urg"
        assert section.wait_queue[1].agent_id == "r1"
        assert section.wait_queue[2].agent_id == "r2"

        promoted = engine.critical_section_coordinator.release(agent_holder.id, chokepoint)
        assert promoted == "urg"
        assert engine.critical_section_coordinator.is_holder(chokepoint, "urg") is True

    @pytest.mark.asyncio
    async def test_interaction_queue_lazy_validation_and_dequeue_trigger(
            self,
            mock_logger: MagicMock,
            mock_presenter: MagicMock,
            mock_cognition: MagicMock,
            mock_pathfinder: MagicMock,
    ) -> None:
        """Prüft reaktive Mailbox-Zustellung und Zustandsbindung statt Legacy-Queue."""
        grid = WorldGrid(width=10, height=10)
        engine = SimulationEngine(
            grid=grid,
            pathfinder=mock_pathfinder,
            presenter=mock_presenter,
            logger=mock_logger,
            cognition_provider=mock_cognition,
        )

        agent_target = create_agent("target", "TargetAgent", Position(2, 2))
        agent_target.capabilities |= agent_target.capabilities.COMMUNICATIVE
        agent_requester = create_agent("req", "RequesterAgent", Position(2, 1))

        engine.register_agent(agent_target)
        engine.register_agent(agent_requester)

        cmd = TalkCommand(
            source_entity_id=agent_requester.id,
            target_entity_id=agent_target.id,
            message="Hallo",
            intent="request_yield",
        )
        engine.interaction_dispatcher.dispatch(agent_requester, agent_target, cmd)

        assert agent_requester.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER
        assert len(agent_target.interaction_mailbox) == 1

        await engine.process_tick()

        logged_events = [call_args[0][0].event_type for call_args in mock_logger.log.call_args_list]
        assert "interaction_pending_started" in logged_events

    @pytest.mark.asyncio
    async def test_interaction_queue_drops_invalid_request(
            self,
            mock_logger: MagicMock,
            mock_presenter: MagicMock,
            mock_cognition: MagicMock,
            mock_pathfinder: MagicMock,
    ) -> None:
        """Prüft, dass fremdbeschäftigte Partner sofort mit BUSY abgewiesen werden (Fail-Fast)."""
        grid = WorldGrid(width=10, height=10)
        engine = SimulationEngine(
            grid=grid,
            pathfinder=mock_pathfinder,
            presenter=mock_presenter,
            logger=mock_logger,
            cognition_provider=mock_cognition,
        )

        agent_target = create_agent("target", "TargetAgent", Position(2, 2))
        agent_target.interaction_partner_id = "other_agent"
        agent_requester = create_agent("req", "RequesterAgent", Position(8, 8))

        engine.register_agent(agent_target)
        engine.register_agent(agent_requester)

        cmd = TalkCommand(
            source_entity_id=agent_requester.id,
            target_entity_id=agent_target.id,
            message="Kann ich durch?",
            intent="request_yield",
        )
        res = engine.interaction_dispatcher.dispatch(agent_requester, agent_target, cmd)

        assert res.success is False
        assert res.reason == "BUSY"
        assert agent_requester.lifecycle_state == AgentLifecycleState.IDLE

        logged_events = [call_args[0][0].event_type for call_args in mock_logger.log.call_args_list]
        assert "interaction_busy_rejected" in logged_events