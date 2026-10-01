from __future__ import annotations

import pytest
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.infrastructure.pathfinding.astar import AStarPathfinder


@pytest.fixture
def pathfinder() -> AStarPathfinder:
    return AStarPathfinder()


@pytest.fixture
def empty_grid() -> WorldGrid:
    return WorldGrid(width=10, height=10)


def test_astar_finds_direct_path(pathfinder: AStarPathfinder, empty_grid: WorldGrid) -> None:
    start = Position(1, 1)
    goal = Position(4, 1)

    path = pathfinder.find_path(start, goal, empty_grid)

    assert len(path) == 3
    assert path == [Position(2, 1), Position(3, 1), Position(4, 1)]
    assert start not in path


def test_astar_navigates_around_obstacle(pathfinder: AStarPathfinder, empty_grid: WorldGrid) -> None:
    empty_grid.set_obstacle(Position(2, 1))
    start = Position(1, 1)
    goal = Position(3, 1)

    path = pathfinder.find_path(start, goal, empty_grid)

    assert len(path) == 4
    assert Position(2, 1) not in path
    assert path[-1] == goal


def test_astar_returns_empty_when_start_equals_goal(
    pathfinder: AStarPathfinder, empty_grid: WorldGrid
) -> None:
    pos = Position(2, 2)
    path = pathfinder.find_path(pos, pos, empty_grid)
    assert path == []


def test_astar_returns_empty_when_goal_is_obstacle(
    pathfinder: AStarPathfinder, empty_grid: WorldGrid
) -> None:
    goal = Position(5, 5)
    empty_grid.set_obstacle(goal)
    path = pathfinder.find_path(Position(1, 1), goal, empty_grid)
    assert path == []


def test_astar_returns_empty_when_completely_blocked(
    pathfinder: AStarPathfinder, empty_grid: WorldGrid
) -> None:
    # Vollständige vertikale Wand bei x=5
    for y in range(10):
        empty_grid.set_obstacle(Position(5, y))

    path = pathfinder.find_path(Position(1, 1), Position(8, 8), empty_grid)
    assert path == []