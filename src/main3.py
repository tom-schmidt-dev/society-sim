import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# If running outside the project virtualenv, re-exec with .venv python
if sys.prefix == sys.base_prefix:
    venv_python = project_root / ".venv" / "bin" / "python3"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

from src.infrastructure.container import ApplicationContainer
from src.domain.models.agent.agent import Agent
from src.domain.models.world.position import Position
from src.domain.models.world.world_entity import WorldEntity


def build_complex_world(container: ApplicationContainer) -> None:
    grid = container.grid
    width = 90
    height = 45

    # 1. Äußere Umfassung
    for x in range(width):
        grid.set_obstacle(Position(x, 0))
        grid.set_obstacle(Position(x, height - 1))
    for y in range(height):
        grid.set_obstacle(Position(0, y))
        grid.set_obstacle(Position(width - 1, y))

    # 2. Zentraler Platz (Plaza): von x=35 bis x=55, y=17 bis y=27 offen halten
    # Vertikale Trennwände West und Ost
    for y in range(1, height - 1):
        if not (18 <= y <= 26):
            grid.set_obstacle(Position(30, y))
            grid.set_obstacle(Position(60, y))

    # Horizontale Trennwände Nord und Süd
    for x in range(1, width - 1):
        if not (40 <= x <= 50):
            grid.set_obstacle(Position(x, 15))
            grid.set_obstacle(Position(x, 30))

    # 3. Engpässe und Chokepoints (West-Korridor auf y=22)
    for x in range(5, 30):
        if x != 15:  # Nische bei (15, 21)
            grid.set_obstacle(Position(x, 21))
        grid.set_obstacle(Position(x, 23))

    # Nischeneinfassung für West-Korridor
    grid.set_obstacle(Position(14, 20))
    grid.set_obstacle(Position(15, 19))
    grid.set_obstacle(Position(16, 20))

    # 4. Ost-Korridor auf y=22
    for x in range(61, 85):
        if x != 72:  # Nische bei (72, 21)
            grid.set_obstacle(Position(x, 21))
        grid.set_obstacle(Position(x, 23))

    # Nischeneinfassung für Ost-Korridor
    grid.set_obstacle(Position(71, 20))
    grid.set_obstacle(Position(72, 19))
    grid.set_obstacle(Position(73, 20))

    # 5. Vertikaler Zentral-Chokepoint auf x=45 (Süd-Verbindung)
    for y in range(31, 44):
        if y != 37:  # Nische bei (46, 37)
            grid.set_obstacle(Position(44, y))
        grid.set_obstacle(Position(46, y))

    # Nische bei (47, 37) freigeben
    grid.remove_obstacle(Position(46, 37))
    grid.set_obstacle(Position(47, 36))
    grid.set_obstacle(Position(48, 37))
    grid.set_obstacle(Position(47, 38))

    # 6. Statische Dekorationsobjekte / Felsen
    rocks = [
        Position(10, 8),
        Position(20, 38),
        Position(75, 8),
        Position(80, 38),
        Position(45, 22),  # Zentraler Brunnen/Monolith auf der Plaza
    ]
    for r in rocks:
        grid.set_obstacle(r)


async def main() -> None:
    container = ApplicationContainer.build(
        width=90,
        height=45,
        tick_interval=0.10,
        auditory_radius=3,
        blockage_strategy="action_masking",
    )
    build_complex_world(container)

    # Statische nicht-ansprechbare Entität platzieren
    statue = WorldEntity(
        id="statue_center",
        name="Antike Statue",
        position=Position(46, 22),
        entity_type="monument",
        is_conversational=False,
        is_passable=False,
    )
    container.engine.register_entity(statue)

    # Agent 1: Alice (West -> Ost durch engen Korridor)
    alice = Agent(id="1", name="Alice", position=Position(2, 22), is_conversational=True)
    container.engine.register_agent(alice)
    container.engine.set_agent_target("1", Position(87, 22), "Ost-Hafen")

    # Agent 2: Bob (Ost -> West direkt gegen Alice)
    bob = Agent(id="2", name="Bob", position=Position(87, 22), is_conversational=True)
    container.engine.register_agent(bob)
    container.engine.set_agent_target("2", Position(2, 22), "West-Tor")

    # Agent 3: Charlie (Süd -> Nord durch Plaza)
    charlie = Agent(id="3", name="Charlie", position=Position(45, 42), is_conversational=True)
    container.engine.register_agent(charlie)
    container.engine.set_agent_target("3", Position(45, 3), "Nord-Palast")

    # Agent 4: Dana (Nord -> Süd kreuzt Charlie)
    dana = Agent(id="4", name="Dana", position=Position(45, 3), is_conversational=True)
    container.engine.register_agent(dana)
    container.engine.set_agent_target("4", Position(45, 42), "Süd-Kaserne")

    # Agent 5: Elena (Querläuferin vom Nord-Westen in den Süd-Osten)
    elena = Agent(id="5", name="Elena", position=Position(5, 5), is_conversational=True)
    container.engine.register_agent(elena)
    container.engine.set_agent_target("5", Position(85, 40), "Süd-Ost-Markt")

    await container.engine.run(max_ticks=250)


if __name__ == "__main__":
    asyncio.run(main())