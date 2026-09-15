from __future__ import annotations

import asyncio
from typing import Literal
from src.container import ApplicationContainer
from src.domain.models.agent import Agent
from src.domain.models.position import Position


def build_corridor(container: ApplicationContainer) -> None:
    grid = container.grid

    # 1. Äußere Grenzwände (90x45)
    for x in range(90):
        grid.set_obstacle(Position(x, 0))
        grid.set_obstacle(Position(x, 44))
    for y in range(45):
        grid.set_obstacle(Position(0, y))
        grid.set_obstacle(Position(89, y))

    # 2. Korridorwände oben (y=21) und unten (y=23) von x=10 bis x=80
    for x in range(10, 80):
        grid.set_obstacle(Position(x, 21))
        grid.set_obstacle(Position(x, 23))

    # 3. Ausweichnische bei x=45 nach Norden öffnen und einfassen
    grid.remove_obstacle(Position(45, 21))
    grid.set_obstacle(Position(44, 20))
    grid.set_obstacle(Position(45, 20))
    grid.set_obstacle(Position(46, 20))


async def main() -> None:
    strategy: Literal["action_masking", "reflection"] = "action_masking"

    container = ApplicationContainer.build(
        width=90,
        height=45,
        tick_interval=0.15,
        blockage_strategy=strategy,
    )
    build_corridor(container)

    # Agent Alice (West -> Ost)
    alice = Agent(
        id="1",
        name="Alice",
        position=Position(2, 22),
        entity_type="agent",
        is_conversational=True,
    )
    container.engine.register_agent(alice)
    container.engine.set_agent_target(
        agent_id="1",
        target=Position(87, 22),
        destination_name="Ost-Tor",
    )

    # Agent Bob (Ost -> West)
    bob = Agent(
        id="2",
        name="Bob",
        position=Position(87, 22),
        entity_type="agent",
        is_conversational=True,
    )
    container.engine.register_agent(bob)
    container.engine.set_agent_target(
        agent_id="2",
        target=Position(2, 22),
        destination_name="West-Tor",
    )

    await container.engine.run(max_ticks=400)


if __name__ == "__main__":
    asyncio.run(main())