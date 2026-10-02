from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Optional
from src.domain.models.planning.events import SimulationEvent
from src.domain.ports.event_logger import IEventLogger


class JsonlEventLogger(IEventLogger):
    def __init__(
        self,
        base_dir: str = "logs",
        prefix: str = "simulation_events",
        file_path: Optional[str | Path] = None,
    ) -> None:
        if file_path is not None:
            self._file_path = Path(file_path)
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._file_path = Path(base_dir) / f"{prefix}_{timestamp}.jsonl"
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: SimulationEvent) -> None:
        data = {
            "tick": event.tick,
            "timestamp": event.timestamp,
            "agent_id": event.agent_id,
            "event_type": event.event_type,
            "summary": event.summary,
            "payload": event.payload,
        }
        with open(self._file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")