from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Literal

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

if sys.prefix == sys.base_prefix:
    venv_python = project_root / ".venv" / "bin" / "python3"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

from src.application.services.world_loader import WorldLoader
from src.infrastructure.container import ApplicationContainer
from src.domain.models.position import Position
from src.domain.models.world_definition import (
    PlacedAgentData,
    PlacedEntityData,
    WorldDefinition,
)
from src.infrastructure.repositories.json_world_repository import JsonWorldRepository


def create_default_labyrinth_definition() -> WorldDefinition:
    """Erzeugt die Standard-Labyrinth-Definition als Fallback."""
    obstacles: list[Position] = []

    # 1. Grenzwände
    for x in range(90):
        obstacles.append(Position(x, 0))
        obstacles.append(Position(x, 44))
    for y in range(45):
        obstacles.append(Position(0, y))
        obstacles.append(Position(89, y))

    # 2. Barrieren
    for y in range(1, 36):
        obstacles.append(Position(15, y))
    for x in range(1, 11):
        obstacles.append(Position(x, 28))

    for y in range(9, 44):
        obstacles.append(Position(30, y))
    for x in range(16, 26):
        obstacles.append(Position(x, 20))

    for y in range(1, 21):
        obstacles.append(Position(45, y))
    obstacles.append(Position(45, 21))
    obstacles.append(Position(45, 23))
    for y in range(24, 36):
        obstacles.append(Position(45, y))

    for x in range(38, 45):
        if x != 44:
            obstacles.append(Position(x, 20))
        obstacles.append(Position(x, 24))

    for y in range(9, 44):
        obstacles.append(Position(60, y))
    for x in range(48, 57):
        obstacles.append(Position(x, 22))

    for y in range(1, 36):
        obstacles.append(Position(75, y))
    for x in range(76, 86):
        obstacles.append(Position(x, 15))

    for rock in [Position(7, 10), Position(22, 35), Position(36, 12), Position(52, 30), Position(68, 18), Position(82, 32)]:
        obstacles.append(rock)

    entities = [
        PlacedEntityData(
            id="stone_1",
            name="Großer Stein",
            blueprint_id="rock",
            position=Position(45, 22),
            entity_type="rock",
            is_conversational=False,
        )
    ]

    agents = [
        PlacedAgentData(
            id="1",
            name="Alice",
            position=Position(2, 22),
            target_position=Position(87, 22),
            destination_name="Ost-Tor",
        )
    ]

    return WorldDefinition(
        name="Labyrinth",
        width=90,
        height=45,
        obstacles=obstacles,
        entities=entities,
        agents=agents,
    )


async def main() -> None:
    strategy: Literal["action_masking", "reflection"] = "action_masking"
    world_name = "labyrinth3"

    world_repo = JsonWorldRepository(project_root / "data" / "worlds")

    # Welt laden oder Fallback generieren und persistieren
    try:
        world = world_repo.load(world_name)
    except FileNotFoundError:
        world = create_default_labyrinth_definition()
        world_repo.save(world, world_name)

    container = ApplicationContainer.build(
        width=world.width,
        height=world.height,
        tick_interval=0.15,
        blockage_strategy=strategy,
    )

    # Deterministische Befüllung ohne prozedurale Schleifen in main
    WorldLoader.apply_to_container(world, container)

    await container.engine.run(max_ticks=600)


if __name__ == "__main__":
    asyncio.run(main())