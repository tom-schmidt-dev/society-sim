from __future__ import annotations

import pytest
from src.application.services.movement.evasion_finder import EvasionFinder
from src.domain.models.agent.mental_map import AgentMentalMap, TileKnowledge
from src.domain.models.world.position import Position
from src.infrastructure.pathfinding.astar import AStarPathfinder


@pytest.fixture
def evasion_finder() -> EvasionFinder:
    return EvasionFinder(pathfinder=AStarPathfinder())


@pytest.fixture
def configured_mental_map() -> AgentMentalMap:
    m = AgentMentalMap(width=20, height=20)
    for x in range(11):
        m.update_tile(Position(x, 5), is_walkable=True, tick=1)
    return m


def test_evasion_stage1_start_already_adjacent_to_niche(
    evasion_finder: EvasionFinder, configured_mental_map: AgentMentalMap
) -> None:
    # TC-GEO-01: Startfeld grenzt unmittelbar an freie Nische außerhalb der Trajektorie
    configured_mental_map.update_tile(Position(4, 4), is_walkable=True, tick=1)

    start = Position(4, 5)
    blocked_pos = Position(5, 5)
    partner_trajectory = [Position(x, 5) for x in range(11)]

    result = evasion_finder.find_nearest_evasion_tile(
        start=start,
        blocked_pos=blocked_pos,
        grid=configured_mental_map,
        occupied_positions=set(),
        partner_trajectory=partner_trajectory,
    )

    assert result is not None
    assert result.target_tile == Position(4, 4)
    assert result.junction_tile == Position(4, 5)
    assert result.is_frontier is False


def test_evasion_stage1_near_niche(
    evasion_finder: EvasionFinder, configured_mental_map: AgentMentalMap
) -> None:
    # TC-GEO-02: Nische innerhalb L1 <= 3 bei (3, 4), erreichbar von Start (4, 5)
    configured_mental_map.update_tile(Position(3, 4), is_walkable=True, tick=1)

    start = Position(4, 5)
    blocked_pos = Position(5, 5)
    # Gesamter Flur y=5 ist Partnerpfad, Ausweichen ist nur nach y!=5 möglich
    partner_trajectory = [Position(x, 5) for x in range(11)]

    result = evasion_finder.find_nearest_evasion_tile(
        start=start,
        blocked_pos=blocked_pos,
        grid=configured_mental_map,
        occupied_positions=set(),
        partner_trajectory=partner_trajectory,
    )

    assert result is not None
    assert result.is_frontier is False
    assert result.target_tile == Position(3, 4)
    assert start.manhattan_distance(result.target_tile) <= 3
    assert result.junction_tile == Position(3, 5)


def test_evasion_stage2_distant_niche(
    evasion_finder: EvasionFinder, configured_mental_map: AgentMentalMap
) -> None:
    # TC-GEO-03: Nische außerhalb L1 > 3 auf bekannter Karte bei (0, 4)
    configured_mental_map.update_tile(Position(0, 4), is_walkable=True, tick=1)

    start = Position(5, 5)
    blocked_pos = Position(6, 5)
    partner_trajectory = [Position(x, 5) for x in range(11)]

    result = evasion_finder.find_nearest_evasion_tile(
        start=start,
        blocked_pos=blocked_pos,
        grid=configured_mental_map,
        occupied_positions=set(),
        partner_trajectory=partner_trajectory,
    )

    assert result is not None
    assert result.is_frontier is False
    assert result.target_tile == Position(0, 4)
    assert start.manhattan_distance(result.target_tile) > 3
    assert result.junction_tile == Position(0, 5)


def test_evasion_stage3_frontier_exploration(
    evasion_finder: EvasionFinder, configured_mental_map: AgentMentalMap
) -> None:
    # TC-GEO-04: Flur y=5 bekannt; gesamter Flur ist Partnerpfad.
    # Kacheln y=4 sind UNKNOWN; Grenzkachel zu UNKNOWN wird als Frontier ermittelt.
    start = Position(5, 5)
    blocked_pos = Position(6, 5)
    partner_trajectory = [Position(x, 5) for x in range(11)]

    result = evasion_finder.find_nearest_evasion_tile(
        start=start,
        blocked_pos=blocked_pos,
        grid=configured_mental_map,
        occupied_positions=set(),
        partner_trajectory=partner_trajectory,
    )

    assert result is not None
    assert result.is_frontier is True
    assert configured_mental_map.tiles.get(result.target_tile).knowledge == TileKnowledge.WALKABLE


def test_evasion_stage4_complete_exhaustion(evasion_finder: EvasionFinder) -> None:
    # TC-GEO-05: Vollständig eingemauerter Bereich ohne Nische und ohne Frontier
    m = AgentMentalMap(width=3, height=3)
    start = Position(1, 1)
    blocked_pos = Position(1, 2)
    m.update_tile(start, is_walkable=True, tick=1)
    m.update_tile(blocked_pos, is_walkable=True, tick=1)

    for x in range(3):
        for y in range(3):
            p = Position(x, y)
            if p not in (start, blocked_pos):
                m.mark_obstacle(p, tick=1)

    result = evasion_finder.find_nearest_evasion_tile(
        start=start,
        blocked_pos=blocked_pos,
        grid=m,
        occupied_positions=set(),
        partner_trajectory=[blocked_pos],
    )

    assert result is None


def test_evasion_determines_correct_junction_tile(
    evasion_finder: EvasionFinder, configured_mental_map: AgentMentalMap
) -> None:
    # TC-GEO-06: Letzte gemeinsame Wegmarke auf Partnertrajektorie
    configured_mental_map.update_tile(Position(3, 6), is_walkable=True, tick=1)

    start = Position(5, 5)
    blocked_pos = Position(6, 5)
    partner_trajectory = [Position(x, 5) for x in range(11)]

    result = evasion_finder.find_nearest_evasion_tile(
        start=start,
        blocked_pos=blocked_pos,
        grid=configured_mental_map,
        occupied_positions=set(),
        partner_trajectory=partner_trajectory,
    )

    assert result is not None
    assert result.junction_tile == Position(3, 5)
    assert result.target_tile == Position(3, 6)