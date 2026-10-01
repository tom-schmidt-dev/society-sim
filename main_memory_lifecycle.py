#!/usr/bin/env python3
"""
main_memory_lifecycle.py
Demonstriert den vollständigen kognitiven Lebenszyklus eines Agenten:
Bedürfnisentstehung -> Exploration -> Konsum -> Nacht-Konsolidierung (Vektorspeicher) ->
Erwachen -> erneuter Hunger -> zielgerichtete Rückkehr über semantisches Retrieval.
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
from src.domain.models.world.world_entity import WorldEntity
from src.infrastructure.container import ApplicationContainer
from src.infrastructure.presentation.qt.simulation_worker import SimulationWorker
from src.infrastructure.presentation.qt.thought_stream_window import ThoughtStreamWindow


def setup_world(container: ApplicationContainer) -> Agent:
    grid = container.grid
    width = grid.width
    height = grid.height

    # Grenzwände
    for x in range(width):
        grid.set_obstacle(Position(x, 0))
        grid.set_obstacle(Position(x, height - 1))
    for y in range(height):
        grid.set_obstacle(Position(0, y))
        grid.set_obstacle(Position(width - 1, y))

    # Nahrungsquelle (Apfelbaum) platzieren
    apple_tree = WorldEntity(
        id="apple_tree_1",
        name="Apfelbaum",
        position=Position(18, 6),
        entity_type="food",
        is_conversational=False,
    )
    # Markierung für Verzehrbarkeit
    setattr(apple_tree, "is_consumable", True)
    setattr(apple_tree, "nutrition_value", 0.6)
    setattr(apple_tree, "is_depletable", False)  # Bleibt als unerschöpfliche Quelle bestehen
    container.engine.register_entity(apple_tree)

    # Agent ohne vorgegebenes Ziel initialisieren
    alice = Agent(
        id="agent_alice",
        name="Alice",
        position=Position(3, 6),
        charisma=0.7,
        assertiveness=0.8,
    )
    # Initial moderater Hunger
    alice.needs["hunger"] = 0.70
    alice.needs["thirst"] = 0.0
    alice.needs["energy"] = 0.0

    container.engine.register_agent(alice)
    return alice


# Oben bei den Imports ergänzen:
from src.application.services.lifecycle.day_night_service import DayNightService


def main() -> None:
    # 1. DayNightService mit Zyklus (25 Takte Tag, 10 Takte Nacht) vorbereiten
    day_night_service = DayNightService(day_ticks=25, night_ticks=10)

    # 2. Container mit vorbereitetem DayNightService instanziieren
    container = ApplicationContainer.build(
        width=30,
        height=14,
        tick_interval=0.10,
        day_night_service=day_night_service,
    )

    setup_world(container)

    # 3. PyQt6-Applikation & Fenster initialisieren
    app = QApplication(sys.argv)
    window = ThoughtStreamWindow(frame_buffer=container.frame_buffer, timer_interval_ms=40)
    window.show()

    # 4. Simulation ausführen (150 Takte, kein Abbruch im Leerlauf)
    worker = SimulationWorker(
        engine=container.engine,
        max_ticks=150,
        stop_when_idle=False,
    )
    worker.simulation_finished.connect(lambda: print("\n[INFO] Simulation erfolgreich beendet."))
    worker.error_occurred.connect(lambda err: print(f"\n[FEHLER] Simulationsabbruch: {err}"))
    worker.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()