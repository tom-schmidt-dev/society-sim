from __future__ import annotations

import heapq
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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
    BlockedResolution,
    DialogueResolution,
    EndDialogueAction,
    GoalIntent,
    TalkAction,
    WaitAction,
)
from src.domain.models.goal import ExecutionPriority, Goal
from src.domain.models.mental_map import AgentMentalMap, TileKnowledge
from src.domain.models.message import IncomingMessage
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter
from tests.test_main3_divided import setup_west_corridor_grid


class DeterministicAStarPathfinder(IPathfinder):
    """Vollständig deterministischer A*-Pathfinder für Testszenarien."""

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


def setup_complex_grid(grid: WorldGrid) -> None:
    width = 90
    height = 45

    for x in range(width):
        grid.set_obstacle(Position(x, 0))
        grid.set_obstacle(Position(x, height - 1))
    for y in range(height):
        grid.set_obstacle(Position(0, y))
        grid.set_obstacle(Position(width - 1, y))

    for y in range(1, height - 1):
        if not (18 <= y <= 26):
            grid.set_obstacle(Position(30, y))
            grid.set_obstacle(Position(60, y))

    for x in range(1, width - 1):
        if not (40 <= x <= 50):
            grid.set_obstacle(Position(x, 15))
            grid.set_obstacle(Position(x, 30))

    for x in range(5, 30):
        if x != 15:
            grid.set_obstacle(Position(x, 21))
        grid.set_obstacle(Position(x, 23))

    grid.set_obstacle(Position(14, 20))
    grid.set_obstacle(Position(15, 19))
    grid.set_obstacle(Position(16, 20))

    for x in range(61, 85):
        if x != 72:
            grid.set_obstacle(Position(x, 21))
        grid.set_obstacle(Position(x, 23))

    grid.set_obstacle(Position(71, 20))
    grid.set_obstacle(Position(72, 19))
    grid.set_obstacle(Position(73, 20))

    for y in range(31, 44):
        if y != 37:
            grid.set_obstacle(Position(44, y))
        grid.set_obstacle(Position(46, y))

    grid.remove_obstacle(Position(46, 37))
    grid.set_obstacle(Position(47, 36))
    grid.set_obstacle(Position(48, 37))
    grid.set_obstacle(Position(47, 38))

    rocks = [
        Position(10, 8),
        Position(20, 38),
        Position(75, 8),
        Position(80, 38),
        Position(45, 22),
    ]
    for r in rocks:
        grid.set_obstacle(r)


@pytest.fixture
def complex_environment() -> dict[str, Any]:
    grid = WorldGrid(width=90, height=45)
    setup_complex_grid(grid)

    logger = MagicMock(spec=IEventLogger)
    logger.log = MagicMock()

    presenter = MagicMock(spec=IPresenter)
    presenter.render = MagicMock()

    cognition = MagicMock(spec=ICognitionProvider)
    cognition.resolve_blockage = AsyncMock()
    cognition.respond_to_dialogue = AsyncMock()
    cognition.evaluate_goal_status = AsyncMock()

    pathfinder = DeterministicAStarPathfinder()

    engine = SimulationEngine(
        grid=grid,
        pathfinder=pathfinder,
        presenter=presenter,
        logger=logger,
        cognition_provider=cognition,
        tick_interval=0.0,
        auditory_radius=3,
    )

    statue = WorldEntity(
        id="statue_center",
        name="Antike Statue",
        position=Position(46, 22),
        entity_type="monument",
        is_conversational=False,
    )
    statue.is_passable = False  # type: ignore[attr-defined]
    engine.register_entity(statue)

    alice = Agent(id="1", name="Alice", position=Position(2, 22), is_conversational=True)
    bob = Agent(id="2", name="Bob", position=Position(87, 22), is_conversational=True)
    charlie = Agent(id="3", name="Charlie", position=Position(45, 42), is_conversational=True)
    dana = Agent(id="4", name="Dana", position=Position(45, 3), is_conversational=True)
    elena = Agent(id="5", name="Elena", position=Position(5, 5), is_conversational=True)

    for ag in (alice, bob, charlie, dana, elena):
        engine.register_agent(ag)

    return {
        "grid": grid,
        "engine": engine,
        "logger": logger,
        "cognition": cognition,
        "pathfinder": pathfinder,
        "statue": statue,
        "alice": alice,
        "bob": bob,
        "charlie": charlie,
        "dana": dana,
        "elena": elena,
    }


class TestComplexWorldMultiAgentScenario:
    def test_world_topology_and_niche_geometry(self, complex_environment: dict[str, Any]) -> None:
        grid: WorldGrid = complex_environment["grid"]

        assert grid.is_walkable(Position(10, 22)) is True
        assert grid.is_walkable(Position(10, 21)) is False
        assert grid.is_walkable(Position(10, 23)) is False

        assert grid.is_walkable(Position(15, 21)) is True
        assert grid.is_walkable(Position(15, 19)) is False
        assert grid.is_walkable(Position(14, 20)) is False
        assert grid.is_walkable(Position(16, 20)) is False

        assert grid.is_walkable(Position(72, 21)) is True
        assert grid.is_walkable(Position(72, 19)) is False

        assert grid.is_walkable(Position(45, 35)) is True
        assert grid.is_walkable(Position(44, 35)) is False
        assert grid.is_walkable(Position(46, 35)) is False
        assert grid.is_walkable(Position(46, 37)) is True
        assert grid.is_walkable(Position(48, 37)) is False

        assert grid.is_walkable(Position(45, 22)) is False

    def test_initial_pathfinding_across_chokepoints(self, complex_environment: dict[str, Any]) -> None:
        engine: SimulationEngine = complex_environment["engine"]
        alice: Agent = complex_environment["alice"]
        bob: Agent = complex_environment["bob"]
        charlie: Agent = complex_environment["charlie"]
        dana: Agent = complex_environment["dana"]
        elena: Agent = complex_environment["elena"]

        engine.set_agent_target(alice.id, Position(87, 22), "Ost-Hafen")
        engine.set_agent_target(bob.id, Position(2, 22), "West-Tor")
        engine.set_agent_target(charlie.id, Position(45, 3), "Nord-Palast")
        engine.set_agent_target(dana.id, Position(45, 42), "Süd-Kaserne")
        engine.set_agent_target(elena.id, Position(85, 40), "Süd-Ost-Markt")

        assert alice.has_path is True
        assert bob.has_path is True
        assert charlie.has_path is True
        assert dana.has_path is True
        assert elena.has_path is True

        assert Position(15, 22) in alice.path
        assert Position(15, 22) in bob.path

        assert Position(45, 37) in charlie.path
        assert Position(45, 37) in dana.path

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
        # Da Alice in _entities vor Bob verarbeitet wird, wird das Signal im selben Takt zugestellt und von Bob direkt konsumiert
        await engine.process_tick()
        assert alice.position == Position(17, 22)
        assert bob.is_evasion_locked is False
        assert bob.active_goal.name == "West-Tor"
        assert bob.active_goal.status == "active"
        assert bob.path[0] == Position(15, 22)

        # Tick 8: Bob zieht physisch aus der Nische zurück auf die Junction (15, 22)
        await engine.process_tick()
        assert bob.position == Position(15, 22)

    @pytest.mark.asyncio
    async def test_plaza_multi_agent_critical_section_and_goal_necessity(
        self, complex_environment: dict[str, Any]
    ) -> None:
        engine: SimulationEngine = complex_environment["engine"]
        alice: Agent = complex_environment["alice"]
        elena: Agent = complex_environment["elena"]
        charlie: Agent = complex_environment["charlie"]
        statue: WorldEntity = complex_environment["statue"]

        resource_key = f"entity:{statue.id}"
        cs_coord: CriticalSectionCoordinator = engine.critical_section_coordinator

        acquired_alice = cs_coord.acquire_or_queue(alice, resource_key, ExecutionPriority.ROUTINE)
        assert acquired_alice is True
        assert cs_coord.get_holder(resource_key) == alice.id

        acquired_elena = cs_coord.acquire_or_queue(
            elena, resource_key, ExecutionPriority.ROUTINE, proposed_order=1
        )
        acquired_charlie = cs_coord.acquire_or_queue(
            charlie, resource_key, ExecutionPriority.ROUTINE, proposed_order=2
        )
        assert acquired_elena is False
        assert acquired_charlie is False

        goal_alice = Goal(name="Statue untersuchen", priority=ExecutionPriority.ROUTINE)
        goal_elena = Goal(name="Statue untersuchen", priority=ExecutionPriority.ROUTINE)
        goal_charlie = Goal(name="Statue untersuchen", priority=ExecutionPriority.ROUTINE)

        engine._goal_service.push_goal(alice, goal_alice)
        engine._goal_service.push_goal(elena, goal_elena)
        engine._goal_service.push_goal(charlie, goal_charlie)

        engine._goal_service.pause_goal(elena)
        engine._goal_service.pause_goal(charlie)

        alice.memory.update_entity_perception(statue.id, statue.name, statue.position, tick=engine.current_tick)
        alice.memory.record_inspection(statue.id, statue.entity_type)

        next_agent_id = cs_coord.release(alice.id, resource_key, mark_completed=True)
        engine._goal_service.pop_goal(alice)

        assert next_agent_id == elena.id
        assert cs_coord.is_completed(resource_key) is True

        engine._notify_next_critical_section_holder(resource_key, next_agent_id)
        assert elena.active_goal is None

        next_agent_after_elena = cs_coord.release(elena.id, resource_key, mark_completed=True)
        assert next_agent_after_elena == charlie.id
        engine._notify_next_critical_section_holder(resource_key, next_agent_after_elena)
        assert charlie.active_goal is None

    @pytest.mark.asyncio
    async def test_urgent_evasion_preempts_routine_passage_in_south_chokepoint(
        self, complex_environment: dict[str, Any]
    ) -> None:
        engine: SimulationEngine = complex_environment["engine"]
        charlie: Agent = complex_environment["charlie"]
        dana: Agent = complex_environment["dana"]
        elena: Agent = complex_environment["elena"]

        chokepoint_key = "chokepoint_pos:(45,37)"
        cs_coord = engine.critical_section_coordinator

        cs_coord.acquire_or_queue(dana, chokepoint_key, ExecutionPriority.ROUTINE)
        cs_coord.acquire_or_queue(charlie, chokepoint_key, ExecutionPriority.ROUTINE)
        cs_coord.acquire_or_queue(elena, chokepoint_key, ExecutionPriority.URGENT)

        section = cs_coord.get_or_create_section(chokepoint_key)
        assert section.wait_queue[0].agent_id == elena.id
        assert section.wait_queue[1].agent_id == charlie.id

        promoted = cs_coord.release(dana.id, chokepoint_key)
        assert promoted == elena.id
        assert cs_coord.is_holder(chokepoint_key, elena.id) is True

    @pytest.mark.asyncio
    async def test_llm_faulty_and_empty_responses_resilience(
        self, complex_environment: dict[str, Any]
    ) -> None:
        engine: SimulationEngine = complex_environment["engine"]
        cognition: MagicMock = complex_environment["cognition"]
        alice: Agent = complex_environment["alice"]
        bob: Agent = complex_environment["bob"]
        statue: WorldEntity = complex_environment["statue"]

        alice.position = Position(14, 22)
        bob.position = Position(15, 22)
        alice.path = [Position(15, 22)]

        cognition.resolve_blockage.side_effect = TimeoutError("Lokaler LLM-Worker antwortet nicht")

        await engine._conflict_coordinator.resolve_blockage(
            alice, bob, Position(15, 22), engine._entities
        )

        assert alice.is_thinking is False
        assert alice.is_waiting_for_reply is False

        bob.interaction_partner_id = None
        alice.interaction_partner_id = None
        bob.inbox.clear()
        alice.inbox.clear()

        cognition.resolve_blockage.side_effect = None
        cognition.resolve_blockage.return_value = BlockedResolution(
            thought="Fehlerhafte Adressierung.",
            action=TalkAction(
                action_type="talk",
                target_agent_id=alice.id,
                message="Hallo?",
                reason="Verwirrung",
            ),
        )

        await engine._conflict_coordinator.resolve_blockage(
            alice, bob, Position(15, 22), engine._entities
        )

        assert len(bob.inbox) == 1
        assert bob.inbox[0].from_agent_id == alice.id

        statue_talk = TalkAction(
            action_type="talk",
            target_agent_id=statue.id,
            message="Sprich mit mir!",
            reason="Erkundung",
        )
        engine._action_executor.execute_blockage_action(
            agent=alice,
            blocker=statue,
            action=statue_talk,
            incident_id="inc-statue-talk",
            all_entities=engine._entities,
            current_tick=engine.current_tick,
        )

        assert any(m.is_empty_response for m in alice.inbox)

    @pytest.mark.asyncio
    async def test_dialogue_rejection_resets_farewell_handshake(
        self, complex_environment: dict[str, Any]
    ) -> None:
        engine: SimulationEngine = complex_environment["engine"]
        cognition: MagicMock = complex_environment["cognition"]
        alice: Agent = complex_environment["alice"]
        bob: Agent = complex_environment["bob"]

        alice.has_bid_farewell = True
        bob.peer_bid_farewell = True
        bob.inbox.append(
            IncomingMessage(
                from_agent_id=alice.id,
                from_agent_name=alice.name,
                message="Tschüss!",
                intent="farewell",
            )
        )

        cognition.respond_to_dialogue.return_value = DialogueResolution(
            thought="Nein, ich bin noch nicht fertig!",
            action=TalkAction(
                action_type="talk",
                target_agent_id=alice.id,
                message="Warte, ich muss noch etwas klären!",
                reason="Klärungsbedarf",
            ),
            negotiation_intent="reject",
        )

        await engine._dialogue_coordinator.handle_incoming_dialogue(bob, engine._entities)

        assert alice.has_bid_farewell is False
        assert bob.peer_bid_farewell is False