from unittest.mock import MagicMock
import pytest

from src.application.services.movement.evasion_finder import EvasionFinder
from src.domain.models.agent.agent import Agent
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.ports.pathfinder import IPathfinder


def _build_test_corridor(rear_niche: bool = False) -> tuple[WorldGrid, Agent, Agent]:
    width = 30
    height = 9
    grid = WorldGrid(width=width, height=height)

    # Korridorwände bei y=3 und y=5 von x=5 bis x=24
    for x in range(5, 25):
        if x != 15 and not (rear_niche and x == 10):
            grid.set_obstacle(Position(x, 3))
        grid.set_obstacle(Position(x, 5))

    # Nischeneinfassung vordere Nische (15, 3)
    grid.set_obstacle(Position(14, 2))
    grid.set_obstacle(Position(15, 1))
    grid.set_obstacle(Position(16, 2))

    # Sackgasse am Korridorende West
    grid.set_obstacle(Position(5, 4))

    if rear_niche:
        # Nischeneinfassung rückwärtige Nische (10, 3)
        grid.set_obstacle(Position(9, 2))
        grid.set_obstacle(Position(10, 2))  # Nischen-Rückwand
        grid.set_obstacle(Position(11, 2))
    else:
        # Sackgasse direkt hinter Alice bei (10, 4)
        grid.set_obstacle(Position(10, 4))

    alice = Agent(id="1", name="Alice", position=Position(14, 4))
    alice.mental_map.set_bounds(width, height)
    alice.assign_path([Position(15, 4), Position(16, 4), Position(17, 4), Position(27, 4)])

    bob = Agent(id="2", name="Bob", position=Position(15, 4))
    bob.mental_map.set_bounds(width, height)
    bob_target_x = 5
    bob.assign_path([Position(x, 4) for x in range(14, bob_target_x, -1)])

    # Mentale Karten initialisieren
    for x in range(width):
        for y in range(height):
            pos = Position(x, y)
            is_w = grid.is_walkable(pos)
            alice.mental_map.update_tile(pos, is_walkable=is_w, tick=0)
            bob.mental_map.update_tile(pos, is_walkable=is_w, tick=0)

    return grid, alice, bob


def test_blocked_partner_tile_invalidates_forward_niche() -> None:
    pathfinder = MagicMock(spec=IPathfinder)
    finder = EvasionFinder(pathfinder)
    _, alice, bob = _build_test_corridor(rear_niche=False)

    res_alice, res_bob = finder.compare_evasion_distances(
        agent_a=alice,
        agent_b=bob,
        all_entities=[alice, bob],
    )

    # Alice steht vor Bob (15, 4) - Nische (15, 3) ist durch Bob blockiert, Sackgasse nach hinten
    assert res_alice is None

    # Bob steht unmittelbar vor Nische (15, 3) und kann ausweichen
    assert res_bob is not None
    assert res_bob.target_tile == Position(15, 3)
    assert res_bob.junction_tile == Position(15, 4)
    assert res_bob.path == [Position(15, 3)]


def test_backtracking_to_rear_niche_when_forward_is_blocked() -> None:
    pathfinder = MagicMock(spec=IPathfinder)
    finder = EvasionFinder(pathfinder)
    _, alice, bob = _build_test_corridor(rear_niche=True)

    res_alice, res_bob = finder.compare_evasion_distances(
        agent_a=alice,
        agent_b=bob,
        all_entities=[alice, bob],
    )

    # Bob kann in Nische (15, 3)
    assert res_bob is not None
    assert res_bob.target_tile == Position(15, 3)

    # Alice weicht rückwärts in Nische (10, 3) aus
    assert res_alice is not None
    assert res_alice.target_tile == Position(10, 3)
    assert res_alice.junction_tile == Position(10, 4)
    assert res_alice.path == [
        Position(13, 4),
        Position(12, 4),
        Position(11, 4),
        Position(10, 4),
        Position(10, 3),
    ]