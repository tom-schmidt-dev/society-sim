from __future__ import annotations

import pytest
from src.application.services.evasion_finder import EvasionFinder
from src.domain.models.mental_map import AgentMentalMap, TileKnowledge
from src.domain.models.position import Position
from src.infrastructure.pathfinding.astar import AStarPathfinder


@pytest.fixture
def evasion_finder() -> EvasionFinder:
    return EvasionFinder(pathfinder=AStarPathfinder())


@pytest.fixture
def configured_mental_map() -> AgentMentalMap:
    m = AgentMentalMap(width=20, height=20)
    # Flur bei y=5 von x=0 bis x=10 als WALKABLE bekannt machen
    for x in range(11):
        m.update_tile(Position(x, 5), is_walkable=True, tick=1)
    return m


def test_evasion_stage1_near_niche(
    evasion_finder: EvasionFinder, configured_mental_map: AgentMentalMap
) -> None:
    # Nische bei (5, 4) innerhalb von L1 <= 3
    configured_mental_map.update_tile(Position(5, 4), is_walkable=True, tick=1)

    start = Position(4, 5)
    blocked_pos = Position(5, 5)
    partner_trajectory = [Position(x, 5) for x in range(5, 11)]

    result = evasion_finder.find_nearest_evasion_tile(
        start=start,
        blocked_pos=blocked_pos,
        grid=configured_mental_map,
        occupied_positions=set(),
        partner_trajectory=partner_trajectory,
    )

    assert result is not None
    assert result.is_frontier is False
    assert result.target_tile == Position(5, 4)
    assert start.manhattan_distance(result.target_tile) <= 3
    assert result.junction_tile == Position(5, 5) or result.junction_tile == Position(4, 5)


def test_evasion_stage2_distant_niche(
    evasion_finder: EvasionFinder, configured_mental_map: AgentMentalMap
) -> None:
    # Keine Nische im Nahbereich L1 <= 3, aber bei (0, 4) mit Distanz 5
    configured_mental_map.update_tile(Position(0, 4), is_walkable=True, tick=1)

    start = Position(5, 5)
    blocked_pos = Position(6, 5)
    partner_trajectory = [Position(x, 5) for x in range(6, 11)]

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


def test_evasion_stage3_frontier_exploration(
    evasion_finder: EvasionFinder, configured_mental_map: AgentMentalMap
) -> None:
    # Keine bekannten Nischen; Kachel (0, 5) grenzt an UNKNOWN-Terrain bei (0, 4)
    start = Position(5, 5)
    blocked_pos = Position(6, 5)
    partner_trajectory = [Position(x, 5) for x in range(6, 11)]

    result = evasion_finder.find_nearest_evasion_tile(
        start=start,
        blocked_pos=blocked_pos,
        grid=configured_mental_map,
        occupied_positions=set(),
        partner_trajectory=partner_trajectory,
    )

    assert result is not None
    assert result.is_frontier is True
    # Nächste Grenzkachel zu unbekanntem Terrain
    assert configured_mental_map.tiles.get(result.target_tile).knowledge == TileKnowledge.WALKABLE


def test_evasion_stage4_complete_exhaustion(evasion_finder: EvasionFinder) -> None:
    # Vollständig eingemauerter 1x1-Bereich ohne Frontier
    m = AgentMentalMap(width=3, height=3)
    start = Position(1, 1)
    blocked_pos = Position(1, 2)
    m.update_tile(start, is_walkable=True, tick=1)
    m.update_tile(blocked_pos, is_walkable=True, tick=1)

    # Alle umgebenden Kacheln als Wände verifiziert bekannt machen (kein UNKNOWN)
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
    # Flur verläuft auf y=5. Nische öffnet sich bei (3, 6) nach Süden.
    configured_mental_map.update_tile(Position(3, 6), is_walkable=True, tick=1)

    start = Position(5, 5)
    blocked_pos = Position(6, 5)
    # Trajektorie des Partners belegt die gesamte Flurachse y=5
    partner_trajectory = [Position(x, 5) for x in range(11)]

    result = evasion_finder.find_nearest_evasion_tile(
        start=start,
        blocked_pos=blocked_pos,
        grid=configured_mental_map,
        occupied_positions=set(),
        partner_trajectory=partner_trajectory,
    )

    assert result is not None
    # Letzte Kachel auf der gemeinsamen Achse y=5 vor dem Einbiegen nach y=6
    assert result.junction_tile == Position(3, 5)
    assert result.target_tile == Position(3, 6)