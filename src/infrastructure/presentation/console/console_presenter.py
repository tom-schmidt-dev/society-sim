from __future__ import annotations

import textwrap
import time
from typing import Any, Optional

from src.application.services.rendering.frame_buffer_service import FrameBufferService
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_cognition import AgentCognitiveSnapshot
from src.domain.models.rendering.render_frame import RenderFrame
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.presenter import IPresenter


class ConsolePresenter(IPresenter):
    """Präsentiert Simulationszustände auf der Konsole und übergibt Frames optional an den Puffer."""

    def __init__(self, frame_buffer: Optional[FrameBufferService] = None) -> None:
        self._frame_buffer = frame_buffer

    @property
    def frame_buffer(self) -> Optional[FrameBufferService]:
        return self._frame_buffer

    @frame_buffer.setter
    def frame_buffer(self, buffer: Optional[FrameBufferService]) -> None:
        self._frame_buffer = buffer

    @staticmethod
    def generate_frame(
        grid: WorldGrid,
        entities: list[WorldEntity],
        tick: int,
        dialogues: Optional[list[str]] = None,
        known_positions: Optional[set[Position]] = None,
        snapshots: Optional[list[AgentCognitiveSnapshot]] = None,
        timestamp: Optional[float] = None,
    ) -> RenderFrame:
        """Erzeugt ein unveränderliches RenderFrame-Objekt aus dem aktuellen Weltzustand."""
        entity_positions: dict[Position, WorldEntity] = {
            entity.position: entity for entity in entities
        }

        grid_matrix: list[list[str]] = []
        for y in range(grid.height):
            row: list[str] = []
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
            grid_matrix.append(row)

        entities_data: list[dict[str, Any]] = []
        for entity in entities:
            if known_positions is not None and entity.position not in known_positions:
                continue
            is_agent = isinstance(entity, Agent)
            target_pos = None
            if is_agent and entity.has_path and entity.path:
                target_pos = (entity.path[-1].x, entity.path[-1].y)

            entities_data.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": getattr(
                        entity, "entity_type", "agent" if is_agent else "entity"
                    ),
                    "position": (entity.position.x, entity.position.y),
                    "symbol": entity.name[0].upper() if is_agent else entity.name[0].lower(),
                    "is_agent": is_agent,
                    "energy": getattr(entity, "energy", None),
                    "has_path": getattr(entity, "has_path", False),
                    "target": target_pos,
                    "is_conversational": getattr(entity, "is_conversational", False),
                }
            )

        return RenderFrame(
            tick=tick,
            timestamp=timestamp if timestamp is not None else time.time(),
            grid_matrix=grid_matrix,
            entities=entities_data,
            dialogues=list(dialogues) if dialogues else [],
            snapshots=list(snapshots) if snapshots else [],
        )

    @staticmethod
    def format_frame(frame: RenderFrame, width: int, height: int) -> str:
        """Formatiert ein RenderFrame in das zweispaltige Terminal-Layout."""
        # 1. Linke Spalte: World-Grid
        map_lines: list[str] = []
        border = "+" + "---" * width + "+"
        map_lines.append(border)

        for y in range(height):
            row = ["|"]
            for x in range(width):
                row.append(frame.grid_matrix[y][x])
            row.append("|")
            map_lines.append("".join(row))

        map_lines.append(border)

        # 2. Rechte Spalte: Konversationen
        right_lines: list[str] = ["Letzte Konversationen:"]
        wrap_width = len("'Tschüss, ich verstehe, dass du")

        if frame.dialogues:
            for entry in frame.dialogues[-8:]:
                wrapped = textwrap.wrap(entry, width=wrap_width)
                right_lines.extend(wrapped)
        else:
            right_lines.append("(Keine Nachrichten)")

        # 3. Zeilenweise Zusammenführung
        map_width = len(border)
        total_rows = max(len(map_lines), len(right_lines))
        combined: list[str] = [f"\033[H\033[J=== SIMULATION TICK: {frame.tick} ==="]

        for i in range(total_rows):
            left = map_lines[i] if i < len(map_lines) else " " * map_width
            right = right_lines[i] if i < len(right_lines) else ""
            combined.append(f"{left}  │  {right}")

        # 4. Statuszeilen unterhalb der Spalten
        combined.append("Entitäten-Status:")
        for ent in frame.entities:
            pos_str = f"Position(x={ent['position'][0]}, y={ent['position'][1]})"
            if ent["is_agent"]:
                target = ent["target"]
                status = (
                    f"Bewegung zu Position(x={target[0]}, y={target[1]})"
                    if ent["has_path"] and target
                    else "Wartend"
                )
                combined.append(
                    f" - [{ent['symbol']}] {ent['name']}: Pos={pos_str}, "
                    f"Energie={ent['energy']}, Status={status}"
                )
            else:
                conv = "ja" if ent["is_conversational"] else "nein"
                combined.append(
                    f" - [{ent['symbol']}] {ent['name']}: Pos={pos_str}, "
                    f"Objekt (spricht={conv})"
                )

        return "\n".join(combined)

    def render(
        self,
        grid: Any = None,
        entities: Any = None,
        tick: Any = None,
        dialogues: Optional[list[str]] = None,
        known_positions: Optional[set[Position]] = None,
        snapshots: Optional[list[AgentCognitiveSnapshot]] = None,
    ) -> None:
        """Rendert den Zustand auf der Konsole und leitet das Frame an den Puffer weiter."""
        if not isinstance(self, ConsolePresenter):
            # Abwärtskompatibilität für statischen Aufruf
            actual_grid: WorldGrid = self
            actual_entities: list[WorldEntity] = grid
            actual_tick: int = tick
            actual_dialogues = dialogues
            actual_known = known_positions
            actual_snapshots = snapshots
            target_buffer = None
        else:
            actual_grid = grid
            actual_entities = entities
            actual_tick = tick
            actual_dialogues = dialogues
            actual_known = known_positions
            actual_snapshots = snapshots
            target_buffer = self._frame_buffer

        frame = self.generate_frame(
            grid=actual_grid,
            entities=actual_entities,
            tick=actual_tick,
            dialogues=actual_dialogues,
            known_positions=actual_known,
            snapshots=actual_snapshots,
        )

        if target_buffer is not None:
            target_buffer.push_frame(frame)

        formatted = self.format_frame(frame, actual_grid.width, actual_grid.height)
        print(formatted)