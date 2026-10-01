#!/usr/bin/env python3
"""
main_dialogue_showcase.py
Demonstriert die autonome Begegnung und Verhandlung zweier Agenten in einer Engstelle:
Alice (hohe assertiveness) und Bob (kooperativ) begegnen sich gegenläufig in einem Korridor.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

if sys.prefix == sys.base_prefix:
    venv_python = project_root / ".venv" / "bin" / "python3"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

from PyQt6.QtWidgets import QApplication

from src.domain.models.agent.agent import Agent
from src.domain.models.world.position import Position
from src.infrastructure.container import ApplicationContainer
from src.infrastructure.presentation.qt.simulation_worker import SimulationWorker
from src.infrastructure.presentation.qt.thought_stream_window import ThoughtStreamWindow


def build_corridor_world(container: ApplicationContainer) -> tuple[Agent, Agent]:
    grid = container.grid
    width = 30
    height = 9

    # Umrandung
    for x in range(width):
        grid.set_obstacle(Position(x, 0))
        grid.set_obstacle(Position(x, height - 1))
    for y in range(height):
        grid.set_obstacle(Position(0, y))
        grid.set_obstacle(Position(width - 1, y))

    # Engpass-Wände oberhalb und unterhalb von y=4
    for x in range(5, 25):
        if x != 15:  # Nische bei (15, 3)
            grid.set_obstacle(Position(x, 3))
        grid.set_obstacle(Position(x, 5))

    # Nischeneinfassung bei (15, 3)
    grid.set_obstacle(Position(14, 2))
    grid.set_obstacle(Position(15, 1))
    grid.set_obstacle(Position(16, 2))

    # Agent 1: Alice (West nach Ost, fordernd)
    alice = Agent(
        id="1",
        name="Alice",
        position=Position(2, 4),
        assertiveness=0.85,
        charisma=0.7,
        is_conversational=True,
    )
    container.engine.register_agent(alice)
    container.engine.set_agent_target("1", Position(27, 4), "Ost-Portal")

    # Agent 2: Bob (Ost nach West, nachgiebig)
    bob = Agent(
        id="2",
        name="Bob",
        position=Position(27, 4),
        assertiveness=0.2,
        charisma=0.6,
        is_conversational=True,
    )
    container.engine.register_agent(bob)
    container.engine.set_agent_target("2", Position(2, 4), "West-Portal")

    return alice, bob


def main() -> None:
    container = ApplicationContainer.build(
        width=30,
        height=9,
        tick_interval=0.15,
        auditory_radius=4,
    )

    build_corridor_world(container)

    app = QApplication(sys.argv)
    window = ThoughtStreamWindow(frame_buffer=container.frame_buffer, timer_interval_ms=50)
    window.show()

    worker = SimulationWorker(
        engine=container.engine,
        max_ticks=450,  # Analog zu main3.py für vollständige Passage
        stop_when_idle=True,
    )
    worker.simulation_finished.connect(lambda: print("\n[INFO] Korridor-Verhandlung abgeschlossen."))
    worker.error_occurred.connect(lambda err: print(f"\n[FEHLER] Simulationsabbruch: {err}"))
    worker.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()