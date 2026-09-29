from __future__ import annotations

import json
from pathlib import Path
from src.domain.models.world_definition import WorldDefinition


class JsonWorldRepository:
    """Lädt und speichert WorldDefinition-Instanzen im JSON-Format."""

    def __init__(self, worlds_dir: Path) -> None:
        self._worlds_dir = worlds_dir
        self._worlds_dir.mkdir(parents=True, exist_ok=True)

    def save(self, world: WorldDefinition, filename: str) -> Path:
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        path = self._worlds_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(world.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    def load(self, filename: str) -> WorldDefinition:
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        path = self._worlds_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Welt-Datei '{path}' existiert nicht.")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return WorldDefinition.from_dict(data)

    def list_worlds(self) -> list[str]:
        return [p.name for p in self._worlds_dir.glob("*.json")]