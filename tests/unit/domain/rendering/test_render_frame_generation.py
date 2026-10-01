from __future__ import annotations

import pytest

from src.application.services.rendering.frame_buffer_service import FrameBufferService
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_cognition import AgentCognitiveSnapshot
from src.domain.models.rendering.render_frame import RenderFrame
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.models.world.world_entity import WorldEntity
from src.infrastructure.presentation.console.console_presenter import ConsolePresenter


def test_render_frame_immutability() -> None:
    frame = RenderFrame(
        tick=1,
        timestamp=1000.0,
        grid_matrix=[[" . "]],
        entities=[],
    )
    with pytest.raises(Exception):
        frame.tick = 2  # type: ignore[misc]


def test_render_frame_to_dict() -> None:
    snapshot = AgentCognitiveSnapshot(
        agent_id="1",
        agent_name="Alice",
        tick=1,
        dominant_need="hunger",
        dominant_need_level=0.5,
        primary_goal="Nahrung",
        active_subgoal="move_to",
        current_position=Position(1, 1),
        target_position=Position(2, 2),
        perceived_obstacle=None,
        intended_strategy="direkt",
        formatted_thought="Ich suche Nahrung.",
    )
    frame = RenderFrame(
        tick=1,
        timestamp=1000.0,
        grid_matrix=[[" . ", " # "]],
        entities=[{"id": "1", "name": "Alice"}],
        dialogues=["Alice: Hallo"],
        snapshots=[snapshot],
    )
    data = frame.to_dict()
    assert data["tick"] == 1
    assert data["timestamp"] == 1000.0
    assert len(data["grid_matrix"]) == 1
    assert len(data["dialogues"]) == 1
    assert data["snapshots"][0]["agent_name"] == "Alice"


def test_console_presenter_generate_frame() -> None:
    grid = WorldGrid(width=3, height=3)
    grid.set_obstacle(Position(0, 0))

    agent = Agent(id="a1", name="Alice", position=Position(1, 1))
    rock = WorldEntity(id="r1", name="Fels", position=Position(2, 2), entity_type="rock")
    entities = [agent, rock]

    frame = ConsolePresenter.generate_frame(
        grid=grid,
        entities=entities,
        tick=5,
        dialogues=["Alice: Test"],
        timestamp=123.456,
    )

    assert frame.tick == 5
    assert frame.timestamp == 123.456
    assert len(frame.grid_matrix) == 3
    assert len(frame.grid_matrix[0]) == 3

    # Kacheln prüfen
    assert frame.grid_matrix[0][0] == " # "
    assert frame.grid_matrix[1][1] == " A "
    assert frame.grid_matrix[2][2] == " f "
    assert frame.grid_matrix[0][1] == " . "

    # Entitäten-Metadaten prüfen
    assert len(frame.entities) == 2
    agent_entry = next(e for e in frame.entities if e["id"] == "a1")
    assert agent_entry["symbol"] == "A"
    assert agent_entry["is_agent"] is True


def test_console_presenter_fog_of_war() -> None:
    grid = WorldGrid(width=3, height=3)
    agent = Agent(id="a1", name="Alice", position=Position(1, 1))
    known = {Position(1, 1)}

    frame = ConsolePresenter.generate_frame(
        grid=grid,
        entities=[agent],
        tick=1,
        known_positions=known,
    )

    assert frame.grid_matrix[1][1] == " A "
    assert frame.grid_matrix[0][0] == "   "
    assert frame.grid_matrix[2][2] == "   "


def test_console_presenter_render_pushes_to_buffer() -> None:
    buffer = FrameBufferService()
    presenter = ConsolePresenter(frame_buffer=buffer)
    grid = WorldGrid(width=2, height=2)

    presenter.render(
        grid=grid,
        entities=[],
        tick=42,
    )

    assert buffer.size == 1
    popped = buffer.pop_frame()
    assert popped is not None
    assert popped.tick == 42