from __future__ import annotations

import pytest
from src.domain.models.agent import Agent
from src.domain.models.goal import ExecutionPriority
from src.domain.models.position import Position
from src.domain.models.reservation_table import TileReservationIntent
from src.application.services.movement_sync_service import (
    MovementSyncService,
    IGridWalkable,
    MovementSyncResult,
)


class MockGrid:
    def __init__(self, width: int = 50, height: int = 50, obstacles: set[Position] | None = None) -> None:
        self.width = width
        self.height = height
        self.obstacles = obstacles or set()

    def is_within_bounds(self, pos: Position) -> bool:
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

    def is_walkable(self, pos: Position) -> bool:
        return self.is_within_bounds(pos) and pos not in self.obstacles


class TestMovementSyncService:
    @pytest.fixture
    def service(self) -> MovementSyncService:
        return MovementSyncService()

    @pytest.fixture
    def grid(self) -> MockGrid:
        return MockGrid(width=30, height=30)

    def test_isolated_component_failure_does_not_block_independent_agents(
        self, service: MovementSyncService, grid: MockGrid
    ) -> None:
        """
        Zwei unabhängige Komponenten:
        - Komponente 1: Agent A will auf (5, 5), das als Hindernis markiert ist.
        - Komponente 2: Agent C will auf (15, 15) ziehen (völlig freies Terrain).
        Ergebnis: Komponente 1 scheitert isoliert. Komponente 2 wird erfolgreich committed.
        """
        grid.obstacles.add(Position(5, 5))

        agent_a = Agent(id="a", name="Alice", position=Position(4, 5))
        agent_c = Agent(id="c", name="Charlie", position=Position(14, 15), path=[Position(15, 15)])

        intent_a = TileReservationIntent(
            agent_id="a",
            current_position=Position(4, 5),
            desired_position=Position(5, 5),
        )
        intent_c = TileReservationIntent(
            agent_id="c",
            current_position=Position(14, 15),
            desired_position=Position(15, 15),
        )

        result = service.execute_two_phase_commit(
            agents=[agent_a, agent_c],
            intents=[intent_a, intent_c],
            grid=grid,
        )

        # Komponente 1 fehlgeschlagen
        assert "a" in result.failed_agents
        assert agent_a.position == Position(4, 5)

        # Komponente 2 erfolgreich
        assert "c" in result.committed_agents
        assert agent_c.position == Position(15, 15)
        assert agent_c.path == []

    def test_forward_convoy_front_to_tail_validation(
        self, service: MovementSyncService, grid: MockGrid
    ) -> None:
        """
        Vorwärts-Konvoi: [A, B, C]
        A steht auf (5, 5), B auf (4, 5), C auf (3, 5).
        A zieht nach (6, 5), B nach (5, 5), C nach (4, 5).
        Front-to-Tail: A zieht zuerst frei -> B kann nachziehen -> C kann nachziehen.
        """
        a = Agent(id="a", name="A", position=Position(5, 5), path=[Position(6, 5)])
        b = Agent(id="b", name="B", position=Position(4, 5), path=[Position(5, 5)])
        c = Agent(id="c", name="C", position=Position(3, 5), path=[Position(4, 5)])

        intents = [
            TileReservationIntent("a", Position(5, 5), Position(6, 5), distance_to_goal=1),
            TileReservationIntent("b", Position(4, 5), Position(5, 5), distance_to_goal=2),
            TileReservationIntent("c", Position(3, 5), Position(4, 5), distance_to_goal=3),
        ]

        result = service.execute_two_phase_commit(
            agents=[a, b, c],
            intents=intents,
            grid=grid,
        )

        assert result.failed_agents == set()
        assert result.committed_agents == {"a", "b", "c"}
        assert a.position == Position(6, 5)
        assert b.position == Position(5, 5)
        assert c.position == Position(4, 5)

    def test_backtracking_convoy_tail_to_front_validation(
        self, service: MovementSyncService, grid: MockGrid
    ) -> None:
        """
        Rückwärts-Konvoi (Backtracking): [A, B, C]
        A steht auf (5, 5), B auf (4, 5), C auf (3, 5).
        Rückzug in negative X-Richtung:
        C (Tail) zieht nach (2, 5), B zieht nach (3, 5), A (Leader) zieht nach (4, 5).
        Tail-to-Front: C räumt zuerst (3, 5) -> B rückt nach -> A rückt nach.
        """
        a = Agent(id="a", name="A", position=Position(5, 5), path=[Position(4, 5)])
        b = Agent(id="b", name="B", position=Position(4, 5), path=[Position(3, 5)])
        c = Agent(id="c", name="C", position=Position(3, 5), path=[Position(2, 5)])

        intents = [
            TileReservationIntent("a", Position(5, 5), Position(4, 5), is_backtracking=True, distance_to_goal=1),
            TileReservationIntent("b", Position(4, 5), Position(3, 5), is_backtracking=True, distance_to_goal=2),
            TileReservationIntent("c", Position(3, 5), Position(2, 5), is_backtracking=True, distance_to_goal=3),
        ]

        result = service.execute_two_phase_commit(
            agents=[a, b, c],
            intents=intents,
            grid=grid,
        )

        assert result.failed_agents == set()
        assert result.committed_agents == {"a", "b", "c"}
        assert c.position == Position(2, 5)
        assert b.position == Position(3, 5)
        assert a.position == Position(4, 5)

    def test_swap_collision_fails_and_holds_both_agents(
        self, service: MovementSyncService, grid: MockGrid
    ) -> None:
        """
        Head-on Swap-Konflikt:
        Agent A steht auf (5, 5) und will nach (6, 5).
        Agent B steht auf (6, 5) und will zeitgleich nach (5, 5).
        Ergebnis: Validierung schlägt fehl, beide verharren an ihren Ursprungspositionen.
        """
        a = Agent(id="a", name="A", position=Position(5, 5))
        b = Agent(id="b", name="B", position=Position(6, 5))

        intents = [
            TileReservationIntent("a", Position(5, 5), Position(6, 5)),
            TileReservationIntent("b", Position(6, 5), Position(5, 5)),
        ]

        result = service.execute_two_phase_commit(
            agents=[a, b],
            intents=intents,
            grid=grid,
        )

        assert result.failed_agents == {"a", "b"}
        assert result.committed_agents == set()
        assert a.position == Position(5, 5)
        assert b.position == Position(6, 5)

    def test_mutual_target_collision_fails_component(
        self, service: MovementSyncService, grid: MockGrid
    ) -> None:
        """
        Zwei Agenten wollen auf dasselbe freie Zielfeld (5, 5) ziehen.
        """
        a = Agent(id="a", name="A", position=Position(4, 5))
        b = Agent(id="b", name="B", position=Position(6, 5))

        intents = [
            TileReservationIntent("a", Position(4, 5), Position(5, 5)),
            TileReservationIntent("b", Position(6, 5), Position(5, 5)),
        ]

        result = service.execute_two_phase_commit(
            agents=[a, b],
            intents=intents,
            grid=grid,
        )

        assert result.failed_agents == {"a", "b"}
        assert result.committed_agents == set()
        assert a.position == Position(4, 5)
        assert b.position == Position(6, 5)

    def test_stationary_entity_blocks_movement(
        self, service: MovementSyncService, grid: MockGrid
    ) -> None:
        """
        Agent A will auf (5, 5) ziehen, wo eine stationäre Entität ohne Intent verharrt.
        """
        a = Agent(id="a", name="A", position=Position(4, 5))
        stat_agent = Agent(id="stat", name="Stationary", position=Position(5, 5))

        intents = [
            TileReservationIntent("a", Position(4, 5), Position(5, 5)),
        ]

        result = service.execute_two_phase_commit(
            agents=[a, stat_agent],
            intents=intents,
            grid=grid,
        )

        assert "a" in result.failed_agents
        assert a.position == Position(4, 5)
