from __future__ import annotations

import pytest
from src.domain.models.agent import Agent
from src.domain.models.position import Position
from src.application.services.frontier_explorer import FrontierExplorer


def test_frontier_explorer_finds_nearest_unexplored_boundary():
    agent = Agent(id="agent_1", name="Alice", position=Position(2, 2))
    agent.mental_map.set_bounds(5, 5)

    # Agent kennt bisher nur ein 3x3 Feld um sich herum
    for x in range(1, 4):
        for y in range(1, 4):
            agent.mental_map.update_tile(Position(x, y), is_walkable=True, tick=1)

    explorer = FrontierExplorer()
    target = explorer.find_nearest_frontier(agent.position, agent.mental_map)

    assert target is not None
    # Ziel muss eine bekannte begehbare Kachel sein, die an Unbekanntes grenzt
    assert agent.mental_map.is_walkable(target)
    assert any(
        n not in agent.mental_map.tiles
        for n in target.get_neighbors()
        if 0 <= n.x < 5 and 0 <= n.y < 5
    )


def test_frontier_explorer_returns_none_when_map_fully_explored():
    agent = Agent(id="agent_1", name="Alice", position=Position(1, 1))
    agent.mental_map.set_bounds(3, 3)

    # Gesamte 3x3 Karte ist exploriert
    for x in range(3):
        for y in range(3):
            agent.mental_map.update_tile(Position(x, y), is_walkable=True, tick=1)

    explorer = FrontierExplorer()
    target = explorer.find_nearest_frontier(agent.position, agent.mental_map)

    assert target is None