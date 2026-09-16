from __future__ import annotations

import pytest
from src.domain.models.agent import Agent
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
from src.domain.services.perception_service import PerceptionService


@pytest.fixture
def perception_service() -> PerceptionService:
    return PerceptionService(default_radius=3)


@pytest.fixture
def grid() -> WorldGrid:
    return WorldGrid(width=20, height=20)


def test_perception_positions_bounded_by_manhattan_radius(
    perception_service: PerceptionService, grid: WorldGrid
) -> None:
    origin = Position(10, 10)
    visible = perception_service.get_visible_positions(origin, grid, radius=3)

    assert origin in visible
    for pos in visible:
        assert origin.manhattan_distance(pos) <= 3
        assert grid.is_within_bounds(pos)

    # Bei L1 <= 3 im unbegrenzten Raum: 1 + 4 + 8 + 12 = 25 Kacheln
    assert len(visible) == 25


def test_perception_positions_clamped_to_grid_borders(
    perception_service: PerceptionService, grid: WorldGrid
) -> None:
    origin = Position(0, 0)
    visible = perception_service.get_visible_positions(origin, grid, radius=3)

    assert len(visible) < 25
    for pos in visible:
        assert pos.x >= 0 and pos.y >= 0
        assert origin.manhattan_distance(pos) <= 3


def test_perception_filters_entities_by_distance_and_identity(
    perception_service: PerceptionService,
) -> None:
    agent = Agent(id="agent_1", name="Alice", position=Position(5, 5))

    near_entity = WorldEntity(id="e_near", name="Stein", position=Position(6, 6))      # Distanz: 2
    border_entity = WorldEntity(id="e_border", name="Kiste", position=Position(5, 8))   # Distanz: 3
    far_entity = WorldEntity(id="e_far", name="Baum", position=Position(5, 9))        # Distanz: 4

    entities: list[WorldEntity] = [agent, near_entity, border_entity, far_entity]
    visible = perception_service.get_visible_entities(agent, entities, radius=3)

    visible_ids = {e.id for e in visible}
    assert "agent_1" not in visible_ids
    assert "e_near" in visible_ids
    assert "e_border" in visible_ids
    assert "e_far" not in visible_ids