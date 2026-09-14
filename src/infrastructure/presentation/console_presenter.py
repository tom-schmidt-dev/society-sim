from __future__ import annotations
import textwrap
from typing import Optional
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
from src.domain.models.agent import Agent
from src.domain.models.position import Position
from src.domain.ports.presenter import IPresenter


class ConsolePresenter(IPresenter):
    def render(
        self,
        grid: WorldGrid,
        entities: list[WorldEntity],
        tick: int,
        dialogues: Optional[list[str]] = None,
        known_positions: Optional[set[Position]] = None,
    ) -> None:
        entity_positions: dict[Position, WorldEntity] = {
            entity.position: entity for entity in entities
        }

        # 1. Linke Spalte: World-Grid mit Fog-of-War-Filter
        map_lines: list[str] = []
        border = "+" + "---" * grid.width + "+"
        map_lines.append(border)

        for y in range(grid.height):
            row = ["|"]
            for x in range(grid.width):
                pos = Position(x, y)
                if known_positions is not None and pos not in known_positions:
                    row.append("   ")
                elif pos in entity_positions:
                    entity = entity_positions[pos]
                    if isinstance(entity, Agent):
                        row.append(f" {entity.name[0].upper()} ")
                    else:
                        row.append(f" {entity.name[0].lower()} ")
                elif not grid.is_walkable(pos):
                    row.append(" # ")
                else:
                    row.append(" . ")
            row.append("|")
            map_lines.append("".join(row))

        map_lines.append(border)

        # 2. Rechte Spalte: Konversationen (Umbruch bei Länge von "'Tschüss, ich verstehe, dass du" = 31 Zeichen)
        dialogue_entries = dialogues or []
        right_lines: list[str] = ["Letzte Konversationen:"]
        wrap_width = len("'Tschüss, ich verstehe, dass du")

        if dialogue_entries:
            for entry in dialogue_entries[-8:]:
                wrapped = textwrap.wrap(entry, width=wrap_width)
                right_lines.extend(wrapped)
        else:
            right_lines.append("(Keine Nachrichten)")

        # 3. Zeilenweise Zusammenführung
        map_width = len(border)
        total_rows = max(len(map_lines), len(right_lines))
        combined: list[str] = [f"\033[H\033[J=== SIMULATION TICK: {tick} ==="]

        for i in range(total_rows):
            left = map_lines[i] if i < len(map_lines) else " " * map_width
            right = right_lines[i] if i < len(right_lines) else ""
            combined.append(f"{left}  │  {right}")

        # 4. Statuszeilen unterhalb der Spalten
        combined.append("Entitäten-Status:")
        for entity in entities:
            # Entitäten außerhalb des aufgedeckten Bereichs werden im Status ausgeblendet
            if known_positions is not None and entity.position not in known_positions:
                continue
            if isinstance(entity, Agent):
                status = f"Bewegung zu {entity.path[-1]}" if entity.has_path else "Wartend"
                combined.append(
                    f" - [{entity.name[0].upper()}] {entity.name}: Pos={entity.position}, "
                    f"Energie={entity.energy}, Status={status}"
                )
            else:
                combined.append(
                    f" - [{entity.name[0].lower()}] {entity.name}: Pos={entity.position}, "
                    f"Objekt (spricht={'ja' if entity.is_conversational else 'nein'})"
                )

        print("\n".join(combined))