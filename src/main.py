from __future__ import annotations

import asyncio
from typing import Literal
from src.container import ApplicationContainer
from src.domain.models.agent import Agent
from src.domain.models.position import Position
from src.domain.models.world_entity import WorldEntity


def build_labyrinth(container: ApplicationContainer) -> None:
    grid = container.grid

    # 1. Äußere Grenzwände (90x45)
    for x in range(90):
        grid.set_obstacle(Position(x, 0))
        grid.set_obstacle(Position(x, 44))
    for y in range(45):
        grid.set_obstacle(Position(0, y))
        grid.set_obstacle(Position(89, y))

    # 2. Vertikale Barriere 1 (x=15) - Durchgang unten (y: 36..43)
    for y in range(1, 36):
        grid.set_obstacle(Position(15, y))
    for x in range(1, 11):
        grid.set_obstacle(Position(x, 28))

    # 3. Vertikale Barriere 2 (x=30) - Durchgang oben (y: 1..8)
    for y in range(9, 44):
        grid.set_obstacle(Position(30, y))
    for x in range(16, 26):
        grid.set_obstacle(Position(x, 20))

    # 4. Vertikale Barriere 3 (x=45) - Mittlerer Pfad mit Chokepoint und Umweg unten
    for y in range(1, 21):
        grid.set_obstacle(Position(45, y))
    grid.set_obstacle(Position(45, 21))
    grid.set_obstacle(Position(45, 23))
    for y in range(24, 36):
        grid.set_obstacle(Position(45, y))

    # Führungskorridor vor dem Hindernis bei y=22 mit Ausweichnische bei (44, 20)
    for x in range(38, 45):
        if x != 44:
            grid.set_obstacle(Position(x, 20))
        grid.set_obstacle(Position(x, 24))

    # 5. Vertikale Barriere 4 (x=60) - Durchgang oben (y: 1..8)
    for y in range(9, 44):
        grid.set_obstacle(Position(60, y))
    for x in range(48, 57):
        grid.set_obstacle(Position(x, 22))

    # 6. Vertikale Barriere 5 (x=75) - Durchgang unten (y: 36..43)
    for y in range(1, 36):
        grid.set_obstacle(Position(75, y))
    for x in range(76, 86):
        grid.set_obstacle(Position(x, 15))

    # 7. Statische Felsblöcke
    scattered_rocks = [
        Position(7, 10),
        Position(22, 35),
        Position(36, 12),
        Position(52, 30),
        Position(68, 18),
        Position(82, 32),
    ]
    for rock in scattered_rocks:
        grid.set_obstacle(rock)


async def main() -> None:
    # Wähle die Konfliktlösungs-Strategie:
    # "action_masking" = A1 (Pydantic-Schema schließt 'talk' aus)
    # "reflection"     = A3 (Kognitive Korrekturschleife bei Fehlentscheidungen)
    strategy: Literal["action_masking", "reflection"] = "action_masking"

    container = ApplicationContainer.build(
        width=90,
        height=45,
        tick_interval=0.15,
        blockage_strategy=strategy,
    )
    build_labyrinth(container)

    alice = Agent(id="1", name="Alice", position=Position(2, 22))
    container.engine.register_agent(alice)

    container.engine.set_agent_target(
        agent_id="1",
        target=Position(87, 22),
        destination_name="Ost-Tor",
    )

    # src/main.py
    stone = WorldEntity(
        id="stone_1",
        name="Großer Stein",
        position=Position(45, 22),
        entity_type="rock",  # Expliziter Typ statt "generic"
        is_conversational=False,
    )
    container.engine.register_entity(stone)

    await container.engine.run(max_ticks=400)


if __name__ == "__main__":
    asyncio.run(main())