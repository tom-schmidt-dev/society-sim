from __future__ import annotations

import asyncio
from src.infrastructure.container import ApplicationContainer
from src.domain.models.agent import Agent
from src.domain.models.goal import ExecutionPriority, Goal
from src.domain.models.position import Position
from src.domain.models.world_entity import WorldEntity


def build_obstacle_course(container: ApplicationContainer) -> None:
    grid = container.grid

    # Grenzwände (50x30)
    for x in range(50):
        grid.set_obstacle(Position(x, 0))
        grid.set_obstacle(Position(x, 29))
    for y in range(30):
        grid.set_obstacle(Position(0, y))
        grid.set_obstacle(Position(49, y))

    # Trennwand bei x=25 mit Durchgang bei y=15
    for y in range(1, 29):
        if y != 15:
            grid.set_obstacle(Position(25, y))


async def main() -> None:
    container = ApplicationContainer.build(
        width=50,
        height=30,
        tick_interval=0.15,
        auditory_radius=3,
        blockage_strategy="action_masking",
    )
    build_obstacle_course(container)

    # Passives Hindernis im einzigen Durchgang (Stein)
    rock = WorldEntity(
        id="rock_doorway",
        name="Massiver Fels",
        position=Position(25, 15),
        entity_type="rock",
        is_conversational=False,
        is_passable=False,
    )
    container.engine.register_entity(rock)

    # Ziel-Agent hinter der Wand (außerhalb des Sicht- und Hörfelds)
    scout = Agent(id="scout_bob", name="Bob", position=Position(40, 20), is_conversational=True)
    container.engine.register_agent(scout)
    container.engine.set_agent_target("scout_bob", Position(40, 5), "Nord-Posten")

    # Such-Agent Alice mit prioritätsgesteuerter Partner-Suche
    alice = Agent(id="searcher_alice", name="Alice", position=Position(10, 15), is_conversational=True)
    container.engine.register_agent(alice)

    # Ziel: Partner über dreistufige Kaskade aufspüren
    alice.push_goal(
        Goal(
            name="Finde Bob",
            target_entity_id="scout_bob",
            priority=ExecutionPriority.COOPERATIVE,
            description="Initiiert die epistemische Suchkaskade über Wahrnehmung, Schätzung und Sektorsuche.",
        )
    )

    await container.engine.run(max_ticks=150)


if __name__ == "__main__":
    asyncio.run(main())