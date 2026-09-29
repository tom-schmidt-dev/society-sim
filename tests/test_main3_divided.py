from __future__ import annotations

import heapq
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.action_executor import ActionExecutor
from src.application.services.critical_section_coordinator import CriticalSectionCoordinator
from src.application.services.dialogue_history import DialogueHistory
from src.application.services.dialogue_session_manager import DialogueSessionManager
from src.application.services.evasion_finder import EvasionFinder
from src.application.services.goal_service import GoalService

try:
    from src.application.simulation_engine import SimulationEngine
except ModuleNotFoundError:
    try:
        from src.simulation_engine import SimulationEngine
    except ModuleNotFoundError:
        from src.application.services.simulation_engine import SimulationEngine

from src.domain.models.agent import Agent
from src.domain.models.cognition import (
    BlockedResolution,
    DialogueResolution,
    EndDialogueAction,
    GoalIntent,
    TalkAction,
)
from src.domain.models.goal import ExecutionPriority, Goal
from src.domain.models.mental_map import AgentMentalMap, TileKnowledge
from src.domain.models.message import IncomingMessage
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter


class DeterministicAStarPathfinder(IPathfinder):
    """Deterministischer A*-Pathfinder für Testszenarien."""

    def find_path(self, start: Position, target: Position, grid: Any) -> list[Position]:
        if start == target:
            return []

        open_set: list[tuple[int, int, Position]] = []
        heapq.heappush(open_set, (0, 0, start))
        came_from: dict[Position, Position] = {}
        g_score: dict[Position, int] = {start: 0}
        counter = 0

        def is_walkable(pos: Position) -> bool:
            if isinstance(grid, AgentMentalMap):
                tile = grid.tiles.get(pos)
                if tile is None:
                    return True
                return tile.knowledge != TileKnowledge.OBSTACLE
            if hasattr(grid, "is_walkable"):
                return grid.is_walkable(pos)
            return True

        def within_bounds(pos: Position) -> bool:
            if hasattr(grid, "is_within_bounds"):
                return grid.is_within_bounds(pos)
            return 0 <= pos.x < 90 and 0 <= pos.y < 45

        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current == target:
                path: list[Position] = []
                curr = current
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.reverse()
                return path

            for neighbor in current.get_neighbors():
                if not within_bounds(neighbor):
                    continue
                if not is_walkable(neighbor) and neighbor != target:
                    continue

                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + neighbor.manhattan_distance(target)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return []


def setup_west_corridor_grid() -> WorldGrid:
    """Erzeugt ein 30x30 Grid mit dem exakten West-Korridor aus main3.py."""
    grid = WorldGrid(width=30, height=30)
    for x in range(5, 30):
        if x != 15:
            grid.set_obstacle(Position(x, 21))
        grid.set_obstacle(Position(x, 23))

    grid.set_obstacle(Position(14, 20))
    grid.set_obstacle(Position(15, 19))
    grid.set_obstacle(Position(16, 20))
    return grid


# ============================================================================
# Stufe 1: Isolierte Geometrie- und Pfadberechnung (EvasionFinder)
# ============================================================================

class TestStep1EvasionFinderIsolated:
    def test_reconstruct_path_excludes_start_position(self) -> None:
        pathfinder = DeterministicAStarPathfinder()
        finder = EvasionFinder(pathfinder)

        start = Position(15, 22)
        target = Position(15, 21)
        came_from = {target: start}

        reconstructed = finder._reconstruct_path(start, target, came_from)
        assert reconstructed == [Position(15, 21)]
        assert start not in reconstructed

    def test_find_nearest_evasion_tile_identifies_west_niche(self) -> None:
        grid = setup_west_corridor_grid()
        pathfinder = DeterministicAStarPathfinder()
        finder = EvasionFinder(pathfinder)

        start = Position(15, 22)
        blocked_pos = Position(14, 22)
        occupied = {start, blocked_pos}
        partner_trajectory = [Position(14, 22), Position(15, 22), Position(16, 22)]

        result = finder.find_nearest_evasion_tile(
            start=start,
            blocked_pos=blocked_pos,
            grid=grid,
            occupied_positions=occupied,
            partner_trajectory=partner_trajectory,
        )

        assert result is not None
        assert result.target_tile == Position(15, 21)
        assert result.junction_tile == Position(15, 22)
        assert result.path == [Position(15, 21)]
        assert result.is_frontier is False


# ============================================================================
# Stufe 2: Ziel-Stack und Unterbrechungslogik (GoalService & ActionExecutor)
# ============================================================================

class TestStep2GoalServiceAndEvasionExecution:
    def test_execute_evasion_pauses_routine_goal_and_pushes_urgent(self) -> None:
        grid = setup_west_corridor_grid()
        logger = MagicMock(spec=IEventLogger)
        pathfinder = DeterministicAStarPathfinder()
        goal_service = GoalService(logger, MagicMock(), pathfinder)
        history = DialogueHistory()
        executor = ActionExecutor(
            grid=grid,
            logger=logger,
            dialogue_history=history,
            goal_service=goal_service,
            pathfinder=pathfinder,
        )

        alice = Agent(id="1", name="Alice", position=Position(14, 22))
        bob = Agent(id="2", name="Bob", position=Position(15, 22))
        alice.mental_map.set_bounds(30, 30)
        bob.mental_map.set_bounds(30, 30)

        bob.mental_map.update_tile(Position(15, 21), is_walkable=True, tick=1)

        routine_goal = Goal(name="West-Tor", target_position=Position(2, 22), priority=ExecutionPriority.ROUTINE)
        goal_service.push_goal(bob, routine_goal)

        executor.execute_evasion(
            agent=bob,
            partner=alice,
            blocked_pos=alice.position,
            all_entities=[alice, bob],
            incident_id="inc-1",
            thought="Ich weiche aus.",
        )

        assert routine_goal.status == "paused"
        assert bob.active_goal is not None
        assert bob.active_goal.priority == ExecutionPriority.URGENT
        assert bob.active_goal.target_position == Position(15, 21)
        assert bob.active_goal.junction_position == Position(15, 22)
        assert bob.active_goal.yield_for_agent_id == alice.id
        assert bob.path == [Position(15, 21)]

        alice.commit_staging_messages()
        assert len(alice.inbox) == 1


# ============================================================================
# Stufe 3: Nischenankunft und Haltezustand (SimulationEngine._handle_niche_arrival)
# ============================================================================

class TestStep3NicheArrivalTransition:
    def test_handle_niche_arrival_locks_agent_and_signals_partner(self) -> None:
        grid = setup_west_corridor_grid()
        logger = MagicMock(spec=IEventLogger)
        pathfinder = DeterministicAStarPathfinder()
        engine = SimulationEngine(
            grid=grid,
            pathfinder=pathfinder,
            presenter=MagicMock(),
            logger=logger,
            cognition_provider=MagicMock(),
        )

        alice = Agent(id="1", name="Alice", position=Position(14, 22))
        bob = Agent(id="2", name="Bob", position=Position(15, 21))
        engine.register_agent(alice)
        engine.register_agent(bob)

        evasion_goal = Goal(
            name="In Nische ausweichen",
            target_position=Position(15, 21),
            junction_position=Position(15, 22),
            yield_for_agent_id=alice.id,
            priority=ExecutionPriority.URGENT,
        )
        engine._goal_service.push_goal(bob, evasion_goal)

        engine._handle_niche_arrival(bob, evasion_goal)

        assert bob.is_evasion_locked is True
        assert bob.has_path is False
        assert bob.active_goal is not None
        assert bob.active_goal.name == "Nischen-Halt"
        assert bob.active_goal.is_evasion_hold is True
        assert bob.active_goal.junction_position == Position(15, 22)
        assert bob.active_goal.yield_for_agent_id == alice.id

        alice.commit_staging_messages()
        assert len(alice.inbox) == 1
        assert alice.inbox[0].is_resume_signal is True
        assert alice.inbox[0].message == "Ok, geh weiter."


# ============================================================================
# Stufe 4: Posteingang-Taktung und Schritt-Verzögerung (SimulationEngine.process_tick)
# ============================================================================

class TestStep4InboxProcessingTickDelay:
    @pytest.mark.asyncio
    async def test_inbox_consumption_delays_physical_step_by_one_tick(self) -> None:
        grid = setup_west_corridor_grid()
        logger = MagicMock(spec=IEventLogger)
        pathfinder = DeterministicAStarPathfinder()
        engine = SimulationEngine(
            grid=grid,
            pathfinder=pathfinder,
            presenter=MagicMock(),
            logger=logger,
            cognition_provider=MagicMock(),
        )

        alice = Agent(id="1", name="Alice", position=Position(14, 22))
        engine.register_agent(alice)
        alice.path = [Position(15, 22), Position(16, 22)]

        alice.inbox.append(
            IncomingMessage(
                from_agent_id="2",
                from_agent_name="Bob",
                message="Ok, weiter.",
                is_resume_signal=True,
            )
        )

        await engine.process_tick()
        assert len(alice.inbox) == 0
        assert alice.position == Position(14, 22)
        assert alice.path == [Position(15, 22), Position(16, 22)]

        await engine.process_tick()
        assert alice.position == Position(15, 22)
        assert alice.path == [Position(16, 22)]


# ============================================================================
# Stufe 5: Abstandsvalidierung und Räumungssignal (_check_and_signal_clearance)
# ============================================================================

class TestStep5ClearanceDistanceValidation:
    def test_clearance_suppressed_when_distance_below_two(self) -> None:
        grid = setup_west_corridor_grid()
        logger = MagicMock(spec=IEventLogger)
        engine = SimulationEngine(
            grid=grid,
            pathfinder=DeterministicAStarPathfinder(),
            presenter=MagicMock(),
            logger=logger,
            cognition_provider=MagicMock(),
        )

        alice = Agent(id="1", name="Alice", position=Position(15, 22))
        bob = Agent(id="2", name="Bob", position=Position(15, 21))
        engine.register_agent(alice)
        engine.register_agent(bob)

        bob.is_evasion_locked = True
        engine._goal_service.push_goal(
            bob,
            Goal(
                name="Nischen-Halt",
                is_evasion_hold=True,
                junction_position=Position(15, 22),
                yield_for_agent_id=alice.id,
            ),
        )

        engine._check_and_signal_clearance(alice)
        assert len(bob.inbox) == 0

        alice.position = Position(16, 22)
        engine._check_and_signal_clearance(alice)
        assert len(bob.inbox) == 0

    def test_clearance_emitted_when_distance_is_two_or_more(self) -> None:
        grid = setup_west_corridor_grid()
        logger = MagicMock(spec=IEventLogger)
        engine = SimulationEngine(
            grid=grid,
            pathfinder=DeterministicAStarPathfinder(),
            presenter=MagicMock(),
            logger=logger,
            cognition_provider=MagicMock(),
        )

        alice = Agent(id="1", name="Alice", position=Position(17, 22))
        bob = Agent(id="2", name="Bob", position=Position(15, 21))
        engine.register_agent(alice)
        engine.register_agent(bob)

        bob.is_evasion_locked = True
        niche_goal = Goal(
            name="Nischen-Halt",
            is_evasion_hold=True,
            junction_position=Position(15, 22),
            yield_for_agent_id=alice.id,
        )
        engine._goal_service.push_goal(bob, niche_goal)

        engine._check_and_signal_clearance(alice)
        bob.commit_staging_messages()

        assert len(bob.inbox) == 1
        assert bob.inbox[0].is_courtesy is True
        assert bob.inbox[0].is_resume_signal is True
        assert niche_goal.yield_for_agent_id is None


# ============================================================================
# Stufe 6: Nischenentlassung und Zielreaktivierung (Courtesy-Processing)
# ============================================================================

class TestStep6CourtesyReleaseFromNiche:
    @pytest.mark.asyncio
    async def test_courtesy_message_releases_niche_lock_and_resumes_goal(self) -> None:
        grid = setup_west_corridor_grid()
        logger = MagicMock(spec=IEventLogger)
        pathfinder = DeterministicAStarPathfinder()
        engine = SimulationEngine(
            grid=grid,
            pathfinder=pathfinder,
            presenter=MagicMock(),
            logger=logger,
            cognition_provider=MagicMock(),
        )

        alice = Agent(id="1", name="Alice", position=Position(17, 22))
        bob = Agent(id="2", name="Bob", position=Position(15, 21))
        engine.register_agent(alice)
        engine.register_agent(bob)

        routine_goal = Goal(name="West-Tor", target_position=Position(2, 22), status="paused")
        engine._goal_service.push_goal(bob, routine_goal)

        niche_goal = Goal(
            name="Nischen-Halt",
            is_evasion_hold=True,
            junction_position=Position(15, 22),
            yield_for_agent_id=None,
        )
        engine._goal_service.push_goal(bob, niche_goal)
        bob.is_evasion_locked = True

        bob.inbox.append(
            IncomingMessage(
                from_agent_id=alice.id,
                from_agent_name=alice.name,
                message="Danke fürs Platz machen!",
                is_courtesy=True,
                is_resume_signal=True,
            )
        )

        await engine.process_tick()

        assert bob.is_evasion_locked is False
        assert bob.active_goal is not None
        assert bob.active_goal.name == "West-Tor"
        assert bob.active_goal.status == "active"
        assert bob.has_path is True
        assert bob.path[0] == Position(15, 22)


# ============================================================================
# Stufe 7: Integrierter Gesamtablauf (Tick-by-Tick End-to-End)
# ============================================================================

class TestStep7FullIntegrationCorridorEvasion:
    @pytest.mark.asyncio
    async def test_full_frontal_evasion_cycle_with_exact_ticks(self) -> None:
        grid = setup_west_corridor_grid()
        logger = MagicMock(spec=IEventLogger)
        cognition = MagicMock(spec=ICognitionProvider)
        pathfinder = DeterministicAStarPathfinder()

        engine = SimulationEngine(
            grid=grid,
            pathfinder=pathfinder,
            presenter=MagicMock(),
            logger=logger,
            cognition_provider=cognition,
            enable_deterministic_corridor=False,
        )

        alice = Agent(id="1", name="Alice", position=Position(14, 22), is_conversational=True)
        bob = Agent(id="2", name="Bob", position=Position(15, 22), is_conversational=True)
        engine.register_agent(alice)
        engine.register_agent(bob)

        alice.path = [Position(15, 22), Position(16, 22), Position(17, 22)]
        bob.path = [Position(14, 22), Position(13, 22)]

        engine._goal_service.push_goal(alice, Goal(name="Ost-Hafen", target_position=Position(25, 22)))
        engine._goal_service.push_goal(bob, Goal(name="West-Tor", target_position=Position(2, 22)))

        engine._update_agent_perception(alice)
        engine._update_agent_perception(bob)

        # Alice als Gesprächsinitiatorin gegenüber Bob festlegen
        bob.interaction_partner_id = alice.id

        # Tick 1: Alice erkennt Blockade und initiiert Dialog
        cognition.resolve_blockage.return_value = BlockedResolution(
            thought="Weg blockiert. Ich spreche Bob an.",
            action=TalkAction(
                action_type="talk",
                target_agent_id=bob.id,
                message="Ich muss nach Osten, kannst du ausweichen?",
                reason="Frontale Blockade",
            ),
        )

        await engine.process_tick()
        bob.commit_staging_messages()
        assert len(bob.inbox) == 1
        assert bob.inbox[0].from_agent_id == alice.id

        # Tick 2: Bob antwortet mit Ausweichabsicht
        cognition.respond_to_dialogue.return_value = DialogueResolution(
            thought="Nische frei. Ich weiche aus.",
            action=EndDialogueAction(
                action_type="end_dialogue",
                reason="Nische erreichbar",
                final_message="Ich weiche nach (15, 21) aus.",
            ),
            negotiation_intent="offer_yield",
            new_goal=GoalIntent(intent_type="evade", name="In Nische ausweichen"),
        )

        await engine.process_tick()
        assert bob.active_goal.priority == ExecutionPriority.URGENT
        assert bob.active_goal.target_position == Position(15, 21)
        assert bob.path == [Position(15, 21)]

        # Tick 3: Bob betritt Nische und sendet resume_signal
        await engine.process_tick()
        assert bob.position == Position(15, 21)
        assert bob.is_evasion_locked is True
        assert bob.active_goal.name == "Nischen-Halt"
        assert any(m.is_resume_signal for m in alice.inbox)

        # Tick 4: Alice verarbeitet Posteingang (Takt wird absorbiert, kein Schritt)
        await engine.process_tick()
        assert alice.position == Position(14, 22)
        assert len(alice.inbox) == 0

        # Tick 5: Alice zieht auf (15, 22)
        await engine.process_tick()
        assert alice.position == Position(15, 22)

        # Tick 6: Alice zieht auf (16, 22) (Distanz 1 zur Junction)
        await engine.process_tick()
        assert alice.position == Position(16, 22)
        assert len(bob.inbox) == 0

        # Tick 7: Alice zieht auf (17, 22) (Distanz 2 zur Junction -> Clearance)
        await engine.process_tick()
        assert alice.position == Position(17, 22)
        assert any(m.is_courtesy for m in bob.inbox)

        # Tick 8: Bob verarbeitet Courtesy-Signal und nimmt Hauptpfad wieder auf
        await engine.process_tick()
        assert bob.is_evasion_locked is False
        assert bob.active_goal.name == "West-Tor"
        assert bob.active_goal.status == "active"
        assert bob.path[0] == Position(15, 22)