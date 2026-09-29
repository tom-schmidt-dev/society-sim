from __future__ import annotations

import json
from pathlib import Path
from src.domain.models.entity_blueprint import EntityBlueprint


class JsonBlueprintRepository:
    """Verwaltet Objekttyp-Blueprints auf dem Dateisystem."""

    DEFAULT_BLUEPRINTS = [
        # Kategorie: Statische Architektur / Wände
        EntityBlueprint(
            id="wall",
            name="Wand (Mauerwerk)",
            category="obstacle",
            entity_type="wall",
            is_conversational=False,
            is_obstacle=True,
            color="#2D3748",
            char="#",
        ),
        EntityBlueprint(
            id="wall_reinforced",
            name="Verstärkte Wand",
            category="obstacle",
            entity_type="wall",
            is_conversational=False,
            is_obstacle=True,
            color="#1A202C",
            char="X",
        ),

        # Kategorie: Natur- und Geländehindernisse
        EntityBlueprint(
            id="rock",
            name="Großer Stein",
            category="entity",
            entity_type="rock",
            is_conversational=False,
            is_obstacle=True,
            color="#718096",
            char="O",
        ),
        EntityBlueprint(
            id="boulder",
            name="Felsbrocken",
            category="obstacle",
            entity_type="boulder",
            is_conversational=False,
            is_obstacle=True,
            color="#4A5568",
            char="B",
        ),
        EntityBlueprint(
            id="tree",
            name="Baum / Busch",
            category="obstacle",
            entity_type="tree",
            is_conversational=False,
            is_obstacle=True,
            color="#2F855A",
            char="T",
        ),
        EntityBlueprint(
            id="water",
            name="Wasserfläche",
            category="obstacle",
            entity_type="water",
            is_conversational=False,
            is_obstacle=True,
            color="#3182CE",
            char="~",
        ),

        # Kategorie: Interaktive / Dynamische Objekte
        EntityBlueprint(
            id="crate",
            name="Holzkiste",
            category="entity",
            entity_type="box",
            is_conversational=False,
            is_obstacle=True,
            color="#B7791F",
            char="K",
        ),
        EntityBlueprint(
            id="door_locked",
            name="Verschlossene Tür",
            category="entity",
            entity_type="door",
            is_conversational=False,
            is_obstacle=True,
            color="#9B2C2C",
            char="D",
        ),

        # Kategorie: Agenten & Akteure
        EntityBlueprint(
            id="agent_alice",
            name="Agent (Alice)",
            category="agent",
            entity_type="agent",
            is_conversational=True,
            is_obstacle=True,
            color="#3182CE",
            char="A",
            default_properties={"agent_id": "1", "name": "Alice"},
        ),
        EntityBlueprint(
            id="agent_bob",
            name="Agent (Bob)",
            category="agent",
            entity_type="agent",
            is_conversational=True,
            is_obstacle=True,
            color="#805AD5",
            char="B",
            default_properties={"agent_id": "2", "name": "Bob"},
        ),
        EntityBlueprint(
            id="agent_npc",
            name="NPC (Wanderer)",
            category="agent",
            entity_type="agent",
            is_conversational=True,
            is_obstacle=True,
            color="#DD6B20",
            char="N",
            default_properties={"agent_id": "3", "name": "Wanderer"},
        ),

        # Kategorie: Marker & Zonen
        EntityBlueprint(
            id="target_east",
            name="Ziel (Ost-Tor)",
            category="obstacle",
            entity_type="target",
            is_conversational=False,
            is_obstacle=False,
            color="#38A169",
            char="E",
            default_properties={"name": "Ost-Tor"},
        ),
        EntityBlueprint(
            id="target_base",
            name="Ziel (Lager/Basis)",
            category="obstacle",
            entity_type="target",
            is_conversational=False,
            is_obstacle=False,
            color="#319795",
            char="Z",
            default_properties={"name": "Lager"},
        ),
    ]

    def __init__(self, filepath: Path) -> None:
        self._filepath = filepath
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self._filepath.exists():
            self._filepath.parent.mkdir(parents=True, exist_ok=True)
            self.save_all(self.DEFAULT_BLUEPRINTS)

    def get_all(self) -> list[EntityBlueprint]:
        if not self._filepath.exists():
            return list(self.DEFAULT_BLUEPRINTS)
        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [EntityBlueprint.from_dict(item) for item in data]
        except Exception:
            return list(self.DEFAULT_BLUEPRINTS)

    def get_by_id(self, blueprint_id: str) -> EntityBlueprint | None:
        return next((b for b in self.get_all() if b.id == blueprint_id), None)

    def add_or_update(self, blueprint: EntityBlueprint) -> None:
        """Fügt einen Blueprint hinzu oder aktualisiert einen bestehenden."""
        blueprints = self.get_all()
        # Vorhandenen Eintrag ersetzen oder neuen anhängen
        updated = False
        for idx, item in enumerate(blueprints):
            if item.id == blueprint.id:
                blueprints[idx] = blueprint
                updated = True
                break
        if not updated:
            blueprints.append(blueprint)
        self.save_all(blueprints)

    def save_all(self, blueprints: list[EntityBlueprint]) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump([b.to_dict() for b in blueprints], f, indent=2, ensure_ascii=False)