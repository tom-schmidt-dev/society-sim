from __future__ import annotations

import asyncio
from typing import Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.application.simulation_engine import SimulationEngine


class SimulationWorker(QThread):
    """Führt die asynchrone Simulations-Engine auf einem dedizierten Worker-Thread aus."""

    simulation_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        engine: SimulationEngine,
        max_ticks: int = 200,
        stop_when_idle: Optional[bool] = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._max_ticks = max_ticks
        self._stop_when_idle = stop_when_idle

    def run(self) -> None:
        """Startet den asyncio-Event-Loop für die SimulationEngine."""
        try:
            kwargs: dict[str, Any] = {"max_ticks": self._max_ticks}
            if self._stop_when_idle is not None:
                kwargs["stop_when_idle"] = self._stop_when_idle
            asyncio.run(self._engine.run(**kwargs))
            self.simulation_finished.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))