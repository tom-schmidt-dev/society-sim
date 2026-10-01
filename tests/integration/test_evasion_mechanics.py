from __future__ import annotations

import pytest
from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.coordination.dialogue_history import DialogueHistory
from src.application.services.movement.evasion_finder import EvasionFinder, EvasionResult
from src.application.services.cognition.goal_service import GoalService
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.events import SimulationEvent
from src.domain.models.agent.mental_map import AgentMentalMap, TileKnowledge
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.event_logger import IEventLogger
from src.infrastructure.pathfinding.astar import AStarPathfinder


class MockEventLogger(IEventLogger):
    def __init__(self) -> None:
        self.events: list[SimulationEvent] = []

    def log(self, event: SimulationEvent) -> None:
        self.events.append(event)


class DummyCognitionProvider:
    pass


@pytest.fixture
def pathfinder() -> AStarPathfinder:
    return AStarPathfinder()


@pytest.fixture
def evasion_finder(pathfinder: AStarPathfinder) -> EvasionFinder:
    return EvasionFinder(pathfinder)


def setup_known_corridor(
    mental_map: AgentMentalMap,
    corridor_y: int = 22,
    x_range: tuple[int, int] = (40, 50),
    niche_pos: Position | None = None,
) -> None:
    """Initialisiert einen verifizierten Korridorabschnitt in der mentalen Karte."""
    mental_map.set_bounds(90, 45)
    for x in range(x_range[0], x_range[1] + 1):
        mental_map.update_tile(Position(x, corridor_y), is_walkable=True, tick=1)
        mental_map.mark_obstacle(Position(x, corridor_y - 1), tick=1)
        mental_map.mark_obstacle(Position(x, corridor_y + 1), tick=1)

    if niche_pos:
        mental_map.update_tile(niche_pos, is_walkable=True, tick=1)


class TestEvasionFinderCascade:
    def test_stage_1_finds_immediate_niche(self, evasion_finder: EvasionFinder) -> None:
        """Stufe 1 findet eine bekannte Nische innerhalb L1 <= 3."""
        mental_map = AgentMentalMap(90, 45)
        niche = Position(44, 21)
        setup_known_corridor(mental_map, corridor_y=22, niche_pos=niche)

        start = Position(44, 22)
        blocked_pos = Position(45, 22)
        partner_trajectory = [Position(x, 22) for x in range(40, 47)]

        result: EvasionResult | None = evasion_finder.find_nearest_evasion_tile(
            start=start,
            blocked_pos=blocked_pos,
            grid=mental_map,
            occupied_positions={blocked_pos},
            partner_trajectory=partner_trajectory,
        )

        assert result is not None
        assert result.target_tile == niche
        assert result.is_frontier is False
        assert len(result.path) > 0
        assert result.path[-1] == niche

    def test_stage_1_and_2_ignore_unseen_wall(self, evasion_finder: EvasionFinder) -> None:
        """Kacheln mit Status UNKNOWN hinter Wänden dürfen nicht als Nische gewählt werden."""
        mental_map = AgentMentalMap(90, 45)
        setup_known_corridor(mental_map, corridor_y=22, niche_pos=None)

        start = Position(44, 22)
        blocked_pos = Position(45, 22)
        partner_trajectory = [Position(x, 22) for x in range(40, 47)]

        result = evasion_finder.find_nearest_evasion_tile(
            start=start,
            blocked_pos=blocked_pos,
            grid=mental_map,
            occupied_positions={blocked_pos},
            partner_trajectory=partner_trajectory,
        )

        if result and not result.is_frontier:
            tile = mental_map.tiles.get(result.target_tile)
            assert tile is not None and tile.knowledge == TileKnowledge.WALKABLE

    def test_junction_tile_determination(self, evasion_finder: EvasionFinder) -> None:
        """Die Junction-Kachel ist das letzte gemeinsame Wegstück auf der Partnertrajektorie."""
        mental_map = AgentMentalMap(90, 45)
        niche = Position(45, 21)
        setup_known_corridor(mental_map, corridor_y=22, niche_pos=niche)

        start = Position(43, 22)
        blocked_pos = Position(46, 22)
        partner_trajectory = [Position(x, 22) for x in range(40, 47)]

        result = evasion_finder.find_nearest_evasion_tile(
            start=start,
            blocked_pos=blocked_pos,
            grid=mental_map,
            occupied_positions={blocked_pos},
            partner_trajectory=partner_trajectory,
        )

        assert result is not None
        assert result.target_tile == niche
        assert result.junction_tile == Position(45, 22)

    def test_stage_3_frontier_fallback_when_no_niche_known(self, evasion_finder: EvasionFinder) -> None:
        """Wenn keine bekannte Nische existiert, greift Stufe 3 (Frontier-Exploration)."""
        mental_map = AgentMentalMap(90, 45)
        mental_map.set_bounds(90, 45)
        mental_map.update_tile(Position(12, 22), is_walkable=True, tick=1)
        mental_map.update_tile(Position(11, 22), is_walkable=True, tick=1)
        mental_map.update_tile(Position(10, 22), is_walkable=True, tick=1)
        for x in (10, 11, 12):
            mental_map.mark_obstacle(Position(x, 21), tick=1)
            mental_map.mark_obstacle(Position(x, 23), tick=1)

        start = Position(11, 22)
        blocked_pos = Position(12, 22)
        partner_trajectory = [Position(12, 22), Position(11, 22), Position(10, 22)]

        result = evasion_finder.find_nearest_evasion_tile(
            start=start,
            blocked_pos=blocked_pos,
            grid=mental_map,
            occupied_positions={blocked_pos},
            partner_trajectory=partner_trajectory,
        )

        assert result is not None
        assert result.is_frontier is True
        assert result.target_tile == Position(10, 22)


class TestActionExecutorEvasionIntegration:
    @pytest.fixture
    def setup_services(self, pathfinder: AStarPathfinder, evasion_finder: EvasionFinder):
        grid = WorldGrid(width=90, height=45)
        logger = MockEventLogger()
        history = DialogueHistory()
        goal_service = GoalService(
            logger=logger,
            cognition_provider=DummyCognitionProvider(),  # type: ignore
            pathfinder=pathfinder,
            tick_provider=lambda: 10,
        )
        executor = ActionExecutor(
            grid=grid,
            logger=logger,
            dialogue_history=history,
            goal_service=goal_service,
            pathfinder=pathfinder,
            evasion_finder=evasion_finder,
            tick_provider=lambda: 10,
        )
        return grid, logger, goal_service, executor

    def test_execute_evasion_sets_goal_and_signals_partner(
        self, setup_services
    ) -> None:
        """Verifiziert, dass execute_evasion den Goal-Stack befüllt und eine Evasions-Nachricht sendet."""
        grid, logger, goal_service, executor = setup_services

        alice = Agent(id="1", name="Alice", position=Position(44, 22))
        bob = Agent(id="2", name="Bob", position=Position(45, 22))
        all_entities: list[WorldEntity] = [alice, bob]

        niche = Position(44, 21)
        setup_known_corridor(alice.mental_map, corridor_y=22, niche_pos=niche)
        bob.path = [Position(x, 22) for x in range(44, 39, -1)]
        alice.memory.record_partner_path(bob.id, [bob.position] + bob.path)

        executor.execute_evasion(
            agent=alice,
            partner=bob,
            blocked_pos=bob.position,
            all_entities=all_entities,
            incident_id="inc-test-1",
            thought="Ich weiche in die Nische aus.",
            sub_goal_name="In Nische ausweichen",
        )

        # 1. Goal-Stack von Alice prüfen
        assert alice.active_goal is not None
        assert alice.active_goal.name == "In Nische ausweichen"
        assert alice.active_goal.target_position == niche
        assert alice.active_goal.junction_position == Position(44, 22)
        assert alice.active_goal.yield_for_agent_id == bob.id
        assert alice.active_goal.is_evasion_hold is False
        assert alice.has_path is True

        # 2. Nachricht an Bob prüfen
        bob.commit_staging_messages()
        assert len(bob.inbox) == 1
        msg = bob.inbox[0]
        assert msg.is_evasion_notice is True
        assert msg.planned_path is not None
        assert "Ich mache Platz" in msg.message

    def test_execute_evasion_handles_complete_exhaustion(
        self, setup_services
    ) -> None:
        """Kaskadenerschöpfung sendet Absage ohne Zielerstellung."""
        grid, logger, goal_service, executor = setup_services

        alice = Agent(id="1", name="Alice", position=Position(5, 5))
        bob = Agent(id="2", name="Bob", position=Position(6, 5))
        all_entities: list[WorldEntity] = [alice, bob]

        alice.mental_map.set_bounds(90, 45)
        alice.mental_map.update_tile(Position(5, 5), is_walkable=True, tick=1)
        for p in Position(5, 5).get_neighbors():
            alice.mental_map.mark_obstacle(p, tick=1)

        executor.execute_evasion(
            agent=alice,
            partner=bob,
            blocked_pos=bob.position,
            all_entities=all_entities,
            incident_id="inc-test-fail",
            thought="Kein Ausweg möglich.",
        )

        bob.commit_staging_messages()
        assert alice.active_goal is None
        assert len(bob.inbox) == 1
        assert "Ich kann nicht ausweichen" in bob.inbox[0].message