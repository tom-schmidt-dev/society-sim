from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from src.application.services.action_executor import ActionExecutor
from src.application.services.conflict_coordinator import ConflictCoordinator
from src.application.services.critical_section_coordinator import CriticalSectionCoordinator
from src.application.services.dialogue_coordinator import DialogueCoordinator
from src.application.services.dialogue_history import DialogueHistory
from src.application.services.dialogue_session_manager import DialogueSessionManager
from src.application.services.evasion_finder import EvasionFinder
from src.application.services.goal_service import GoalService
from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent import Agent
from src.domain.models.cognition import (
    AbortAction,
    BlockedResolution,
    DialogueResolution,
    EndDialogueAction,
    InspectAction,
    ProbeAction,
    RerouteAction,
    TalkAction,
    WaitAction,
)
from src.domain.models.critical_section import CriticalSection, CriticalSectionRequest
from src.domain.models.goal import ExecutionPriority, Goal
from src.domain.models.interaction_request import InteractionRequest
from src.domain.models.message import IncomingMessage
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter


# ============================================================================
# Fixtures und Test-Fabriken
# ============================================================================

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
    """Erzeugt eine standardisierte Agent-Instanz mit initialisierter MentalMap."""
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
    """Erzeugt eine standardisierte WorldEntity-Instanz für Interaktionsprüfungen."""
    entity = WorldEntity(id=entity_id, name=name, position=pos, is_conversational=is_conversational)
    entity.is_passable = is_passable  # type: ignore[attr-defined]
    entity.entity_type = entity_type
    return entity


# ============================================================================
# 1. Tests: CriticalSectionCoordinator (Unit-Tests)
# ============================================================================

class TestCriticalSectionCoordinator:
    """Testet die Kernfunktionalität des CriticalSectionCoordinators isoliert."""

    def test_acquire_initial_success(self, mock_logger: MagicMock) -> None:
        """Der erste anfordernde Agent erhält sofortigen Zugriff auf die Ressource."""
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
        """Wiederholtes Anfordern durch den aktuellen Inhaber blockiert nicht."""
        coordinator = CriticalSectionCoordinator(logger=mock_logger, tick_provider=lambda: 1)
        agent = create_agent("agent_1", "Alice", Position(0, 0))

        assert coordinator.acquire_or_queue(agent, "res:1", ExecutionPriority.ROUTINE) is True
        assert coordinator.acquire_or_queue(agent, "res:1", ExecutionPriority.ROUTINE) is True

    def test_priority_queueing_three_agents(self, mock_logger: MagicMock) -> None:
        """Agenten werden strikt nach Priorität eingereiht: URGENT vor COOPERATIVE vor ROUTINE."""
        coordinator = CriticalSectionCoordinator(logger=mock_logger, tick_provider=lambda: 1)
        agent_holder = create_agent("agent_holder", "Holder", Position(0, 0))
        agent_routine = create_agent("agent_routine", "Routine", Position(1, 0))
        agent_urgent = create_agent("agent_urgent", "Urgent", Position(2, 0))
        agent_coop = create_agent("agent_coop", "Coop", Position(3, 0))

        coordinator.acquire_or_queue(agent_holder, "chokepoint", ExecutionPriority.ROUTINE)

        # Einreihen in suboptimaler Reihenfolge
        assert coordinator.acquire_or_queue(agent_routine, "chokepoint", ExecutionPriority.ROUTINE) is False
        assert coordinator.acquire_or_queue(agent_urgent, "chokepoint", ExecutionPriority.URGENT) is False
        assert coordinator.acquire_or_queue(agent_coop, "chokepoint", ExecutionPriority.COOPERATIVE) is False

        section = coordinator.get_or_create_section("chokepoint")
        queued_ids = [req.agent_id for req in section.wait_queue]

        # Erwartete Sortierung: URGENT (agent_urgent) -> COOPERATIVE (agent_coop) -> ROUTINE (agent_routine)
        assert queued_ids == ["agent_urgent", "agent_coop", "agent_routine"]

    def test_proposed_order_tie_breaker(self, mock_logger: MagicMock) -> None:
        """Bei gleicher Priorität entscheidet die vorgeschlagene LLM-Reihenfolge (proposed_order)."""
        coordinator = CriticalSectionCoordinator(logger=mock_logger, tick_provider=lambda: 1)
        agent_holder = create_agent("holder", "Holder", Position(0, 0))
        agent_a = create_agent("agent_a", "Alice", Position(1, 0))
        agent_b = create_agent("agent_b", "Bob", Position(2, 0))

        coordinator.acquire_or_queue(agent_holder, "res:target", ExecutionPriority.ROUTINE)

        # Beide COOPERATIVE, aber Bob hat niedrigere Rangnummer (Rang 1 vor Rang 2)
        coordinator.acquire_or_queue(agent_a, "res:target", ExecutionPriority.COOPERATIVE, proposed_order=2)
        coordinator.acquire_or_queue(agent_b, "res:target", ExecutionPriority.COOPERATIVE, proposed_order=1)

        section = coordinator.get_or_create_section("res:target")
        assert section.wait_queue[0].agent_id == "agent_b"
        assert section.wait_queue[1].agent_id == "agent_a"

    def test_fifo_fallback_on_identical_priority_and_order(self, mock_logger: MagicMock) -> None:
        """Bei identischer Priorität und fehlendem proposed_order entscheidet der Ankunfts-Tick (FIFO)."""
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
        """Freigabe übergibt die Ressource an den nächsten wartenden Agenten."""
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
        """Freigabe mit mark_completed persistiert den Status und den Abschluss-Tick."""
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
        """Ein unberechtigter Agent kann eine fremde Critical Section nicht freigeben."""
        coordinator = CriticalSectionCoordinator(logger=mock_logger, tick_provider=lambda: 1)
        holder = create_agent("holder", "Holder", Position(0, 0))
        intruder = create_agent("intruder", "Intruder", Position(1, 0))

        coordinator.acquire_or_queue(holder, "key", ExecutionPriority.ROUTINE)
        result = coordinator.release(intruder.id, "key", mark_completed=True)

        assert result is None
        assert coordinator.get_holder("key") == holder.id
        assert coordinator.is_completed("key") is False


# ============================================================================
# 2. Tests: GoalService & Notwendigkeitsprüfung
# ============================================================================

class TestGoalServiceCriticalSections:
    """Testet Zielpausierung, Reaktivierung und Notwendigkeitsprüfung im GoalService."""

    def test_validate_goal_necessity_when_not_completed(self, mock_logger: MagicMock, mock_pathfinder: MagicMock, mock_cognition: MagicMock) -> None:
        """Ist das Ziel noch nicht abgeschlossen, bleibt es aktiv erhalten."""
        service = GoalService(mock_logger, mock_cognition, mock_pathfinder)
        agent = create_agent("agent_1", "A1", Position(0, 0))
        goal = Goal(name="Hebel umlegen", target_position=Position(5, 5))
        service.push_goal(agent, goal)

        is_needed = service.validate_goal_necessity(agent, goal, is_already_completed=False)

        assert is_needed is True
        assert agent.active_goal == goal

    def test_validate_goal_necessity_when_already_completed(self, mock_logger: MagicMock, mock_pathfinder: MagicMock, mock_cognition: MagicMock) -> None:
        """Ist das Ziel bereits erledigt, wird es verworfen und der Pfad gelöscht."""
        service = GoalService(mock_logger, mock_cognition, mock_pathfinder)
        agent = create_agent("agent_1", "A1", Position(0, 0))
        goal = Goal(name="Hebel umlegen", target_position=Position(5, 5))
        agent.path = [Position(1, 0), Position(2, 0)]
        service.push_goal(agent, goal)

        is_needed = service.validate_goal_necessity(agent, goal, is_already_completed=True)

        assert is_needed is False
        assert agent.active_goal is None
        assert agent.has_path is False
        # Prüfen, ob das Event geloggt wurde
        logged_types = [call_args[0][0].event_type for call_args in mock_logger.log.call_args_list]
        assert "goal_deemed_unnecessary" in logged_types

    def test_pause_and_resume_goal(self, mock_logger: MagicMock, mock_pathfinder: MagicMock, mock_cognition: MagicMock) -> None:
        """Ziele können pausiert und exakt in den Zustand 'active' reaktiviert werden."""
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
        assert agent.active_goal.status == "active"


# ============================================================================
# 3. Tests: ActionExecutor & Kritische Bereiche
# ============================================================================

class TestActionExecutorCriticalSections:
    """Testet den wechselseitigen Ausschluss bei atomaren Aktionen wie Inspektion und Probe."""

    def test_execute_inspection_competition_between_two_agents(
        self,
        mock_logger: MagicMock,
        mock_pathfinder: MagicMock,
    ) -> None:
        """Greifen zwei Agenten auf dieselbe Entität zu, wird der zweite pausiert."""
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

        # Agent 1 führt Inspektion durch -> erlangt Lock, schließt ab, markiert completed
        executor.execute_inspection(agent_1, target_entity, "inc-1", current_tick=1)
        assert coordinator.is_completed("entity:box_1") is True

        # Wenn Agent 2 danach versucht, dieselbe Kiste zu sperren
        # Simuliere Lock-Konkurrenz: Lock wird von Agent 1 während der Ausführung noch gehalten
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
        # Agent 2 konnte den Lock nicht erhalten -> sein Ziel wurde pausiert
        assert agent_2.active_goal.status == "paused"

    def test_execute_probe_critical_section(
        self,
        mock_logger: MagicMock,
        mock_pathfinder: MagicMock,
    ) -> None:
        """Physische Erprobung (Probe) erfordert exklusiven Zugriff und gibt nach Erfolg frei."""
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


# ============================================================================
# 4. Tests: LLM-Fehlerszenarien & Resilienz
# ============================================================================

class TestLLMErrorHandlingAndEdgeCases:
    """Testet Ausnahmezustände, leere Rückgaben und Fehlformatierungen der Kognition."""

    @pytest.mark.asyncio
    async def test_resolve_blockage_cognition_timeout_exception(
        self,
        mock_logger: MagicMock,
        mock_pathfinder: MagicMock,
    ) -> None:
        """Bei Timeout/Exception des LLMs verfällt der Agent nicht in Dauer-Thinking."""
        cognition = MagicMock(spec=ICognitionProvider)
        cognition.resolve_blockage = AsyncMock(side_effect=TimeoutError("LLM Inferenz überschritten"))

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
        coordinator = ConflictCoordinator(
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
        blocker = create_agent("agent_2", "Bob", Position(1, 2))
        agent.path = [Position(1, 2)]

        agent.is_thinking = True
        await coordinator.resolve_blockage(agent, blocker, Position(1, 2), [agent, blocker])

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
        """Laufzeitfehler im Dialog-LLM loggen 'dialogue_failed' und setzen den Zustand zurück."""
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
        """Gibt das LLM fälschlicherweise die eigene ID als Talk-Target zurück, wird auf Blocker korrigiert."""
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
            session_manager=session_mgr,
            dialogue_history=history,
        )

        agent = create_agent("agent_1", "Alice", Position(1, 1))
        blocker = create_agent("agent_2", "Bob", Position(1, 2))
        agent.path = [Position(1, 2)]

        await coordinator.resolve_blockage(agent, blocker, Position(1, 2), [agent, blocker])

        assert len(blocker.inbox) == 1
        assert blocker.inbox[0].from_agent_id == "agent_1"
        assert blocker.inbox[0].message == "Selbstgespräch"


# ============================================================================
# 5. Tests: Multi-Agenten-Szenarien (>= 3 Agenten) & End-to-End
# ============================================================================

class TestMultiAgentCriticalSectionScenarios:
    """Testet komplexe Konstellationen mit drei oder mehr Agenten im Simulationsablauf."""

    @pytest.mark.asyncio
    async def test_three_agents_competing_for_same_goal_necessity_abort(
        self,
        mock_logger: MagicMock,
        mock_presenter: MagicMock,
        mock_cognition: MagicMock,
    ) -> None:
        """Drei Agenten wollen dasselbe Ziel bearbeiten.

        Agent 1 schließt es ab. Agent 2 und 3 brechen nach Notwendigkeitsprüfung deterministisch ab.
        """
        grid = WorldGrid(width=10, height=10)
        pathfinder = MagicMock(spec=IPathfinder)
        target_pos = Position(5, 5)

        # Deterministische Pfade für alle Agenten
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

        # Agent 1 belegt die Critical Section
        assert engine.critical_section_coordinator.acquire_or_queue(
            agent_1, resource_key, ExecutionPriority.ROUTINE
        ) is True

        # Agent 2 und 3 reihen sich ein und werden pausiert
        assert engine.critical_section_coordinator.acquire_or_queue(
            agent_2, resource_key, ExecutionPriority.ROUTINE, proposed_order=1
        ) is False
        assert engine.critical_section_coordinator.acquire_or_queue(
            agent_3, resource_key, ExecutionPriority.ROUTINE, proposed_order=2
        ) is False

        # Ziele vergeben
        engine.set_agent_target(agent_1.id, target_pos, "Schatz heben")
        engine.set_agent_target(agent_2.id, target_pos, "Schatz heben")
        engine.set_agent_target(agent_3.id, target_pos, "Schatz heben")

        # Agent 2 und 3 wurden pausiert, da sie in der Warteschlange stehen
        engine._goal_service.pause_goal(agent_2)
        engine._goal_service.pause_goal(agent_3)

        assert agent_2.active_goal.status == "paused"
        assert agent_3.active_goal.status == "paused"

        # Agent 1 betritt das Zielfeld
        agent_1.position = target_pos

        # Tick ausführen: Agent 1 schließt Ziel ab und löst Freigabe + Notifikation aus
        await engine.process_tick()

        # Überprüfen: Agent 1 hat das Ziel abgeschlossen
        assert agent_1.active_goal is None
        assert engine.critical_section_coordinator.is_completed(resource_key) is True

        # Durch die Freigabe wurde Agent 2 als nächster Inhaber notifiziert.
        # Dessen Notwendigkeitsprüfung ergab: is_completed == True -> Ziel wurde verworfen!
        assert agent_2.active_goal is None

    @pytest.mark.asyncio
    async def test_urgent_preemption_reorders_queue_over_routine_agents(
        self,
        mock_logger: MagicMock,
        mock_presenter: MagicMock,
        mock_cognition: MagicMock,
        mock_pathfinder: MagicMock,
    ) -> None:
        """Ein dringendes Ausweichziel (URGENT) überholt reguläre Routine-Ziele in der Warteschlange."""
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

        # Holder besetzt die Ressource
        engine.critical_section_coordinator.acquire_or_queue(agent_holder, chokepoint, ExecutionPriority.ROUTINE)

        # Routine 1 und 2 fordern an
        engine.critical_section_coordinator.acquire_or_queue(agent_routine_1, chokepoint, ExecutionPriority.ROUTINE)
        engine.critical_section_coordinator.acquire_or_queue(agent_routine_2, chokepoint, ExecutionPriority.ROUTINE)

        # Jetzt fordert ein Agent mit Notfall-/Ausweichpriorität an
        engine.critical_section_coordinator.acquire_or_queue(agent_urgent, chokepoint, ExecutionPriority.URGENT)

        # Warteschlange prüfen: Der Urgent-Agent muss an erster Stelle stehen
        section = engine.critical_section_coordinator.get_or_create_section(chokepoint)
        assert section.wait_queue[0].agent_id == "urg"
        assert section.wait_queue[1].agent_id == "r1"
        assert section.wait_queue[2].agent_id == "r2"

        # Freigabe durch Holder: Der Urgent-Agent muss die Ressource erhalten
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
        """In der interaction_queue wartende Agenten werden korrekt dequeued und lösen Kognition aus."""
        grid = WorldGrid(width=10, height=10)
        engine = SimulationEngine(
            grid=grid,
            pathfinder=mock_pathfinder,
            presenter=mock_presenter,
            logger=mock_logger,
            cognition_provider=mock_cognition,
        )

        agent_target = create_agent("target", "TargetAgent", Position(2, 2))
        agent_requester = create_agent("req", "RequesterAgent", Position(2, 1))

        engine.register_agent(agent_target)
        engine.register_agent(agent_requester)

        # Requester plant Bewegung auf das Feld von Target
        agent_requester.path = [Position(2, 2)]

        # Requester in Warteschlange einhängen
        agent_target.interaction_queue.append(
            InteractionRequest(
                requester_id=agent_requester.id,
                target_id=agent_target.id,
                blocked_pos=Position(2, 2),
                tick=1,
            )
        )

        # Tick abarbeiten: Dequeue-Bedingungen sind erfüllt (in Hörweite, Position stimmt noch)
        await engine.process_tick()

        # Überprüfen, ob die Dequeue-Abarbeitung stattfand
        logged_events = [call_args[0][0].event_type for call_args in mock_logger.log.call_args_list]
        assert "interaction_dequeued" in logged_events
        assert len(agent_target.interaction_queue) == 0

    @pytest.mark.asyncio
    async def test_interaction_queue_drops_invalid_request(
        self,
        mock_logger: MagicMock,
        mock_presenter: MagicMock,
        mock_cognition: MagicMock,
        mock_pathfinder: MagicMock,
    ) -> None:
        """Hat sich die Situation verändert (z. B. Requester ging weg), wird der Request verworfen."""
        grid = WorldGrid(width=10, height=10)
        engine = SimulationEngine(
            grid=grid,
            pathfinder=mock_pathfinder,
            presenter=mock_presenter,
            logger=mock_logger,
            cognition_provider=mock_cognition,
        )

        agent_target = create_agent("target", "TargetAgent", Position(2, 2))
        agent_requester = create_agent("req", "RequesterAgent", Position(8, 8))  # Außer Hörweite

        engine.register_agent(agent_target)
        engine.register_agent(agent_requester)

        agent_target.interaction_queue.append(
            InteractionRequest(
                requester_id=agent_requester.id,
                target_id=agent_target.id,
                blocked_pos=Position(2, 2),
                tick=1,
            )
        )

        await engine.process_tick()

        logged_events = [call_args[0][0].event_type for call_args in mock_logger.log.call_args_list]
        assert "interaction_request_dropped" in logged_events
        assert len(agent_target.interaction_queue) == 0