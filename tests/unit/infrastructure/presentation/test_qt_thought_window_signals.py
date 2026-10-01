from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

# Headless-Modus für CI und reine Terminal-Sitzungen erzwingen
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication

from src.application.services.rendering.frame_buffer_service import FrameBufferService
from src.domain.models.agent.agent_cognition import AgentCognitiveSnapshot
from src.domain.models.rendering.render_frame import RenderFrame
from src.domain.models.world.position import Position
from src.infrastructure.presentation.qt.simulation_worker import SimulationWorker
from src.infrastructure.presentation.qt.thought_stream_window import ThoughtStreamWindow


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    return app


def test_thought_stream_window_render_frame(qapp: QApplication) -> None:
    buffer = FrameBufferService()
    window = ThoughtStreamWindow(frame_buffer=buffer, timer_interval_ms=10)

    snapshot = AgentCognitiveSnapshot(
        agent_id="agent_1",
        agent_name="Alice",
        tick=1,
        dominant_need="hunger",
        dominant_need_level=0.75,
        primary_goal="Nahrungssuche",
        active_subgoal="move_to",
        current_position=Position(5, 5),
        target_position=Position(5, 8),
        perceived_obstacle=None,
        intended_strategy="Direkte Navigation",
        formatted_thought="Ich habe Hunger und bewege mich zur Nahrungsquelle.",
    )

    frame = RenderFrame(
        tick=1,
        timestamp=100.0,
        grid_matrix=[[" . "]],
        entities=[],
        dialogues=["Alice: Hallo Welt"],
        snapshots=[snapshot],
    )

    # Frame direkt an das Fenster übergeben
    window.process_frame(frame)

    assert "agent_1" in window.agent_tabs
    tab = window.agent_tabs["agent_1"]

    assert "hunger" in tab.need_label.text().lower()
    assert "0.75" in tab.need_label.text()
    assert "Nahrungssuche" in tab.goal_label.text()
    assert "Ich habe Hunger" in tab.log_view.toPlainText()


def test_thought_stream_window_consumes_buffer(qapp: QApplication) -> None:
    buffer = FrameBufferService()
    window = ThoughtStreamWindow(frame_buffer=buffer, timer_interval_ms=10)

    frame = RenderFrame(
        tick=42,
        timestamp=200.0,
        grid_matrix=[[" . "]],
        entities=[],
        dialogues=[],
        snapshots=[],
    )
    buffer.push_frame(frame)

    # Taktzyklus des Timers synchron auslösen
    window.consume_next_frame()

    assert window.current_tick == 42
    assert buffer.is_empty is True


def test_simulation_worker_lifecycle(qapp: QApplication) -> None:
    mock_engine = MagicMock()
    mock_engine.run = AsyncMock()

    worker = SimulationWorker(engine=mock_engine, max_ticks=10)

    finished_called = False

    def on_finished() -> None:
        nonlocal finished_called
        finished_called = True

    worker.simulation_finished.connect(on_finished)

    # Synchrone Ausführung der Run-Logik zur Testvalidierung
    worker.run()

    assert finished_called is True
    mock_engine.run.assert_called_once_with(max_ticks=10)