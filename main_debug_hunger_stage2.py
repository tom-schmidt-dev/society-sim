from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.planning import (
    ActionType,
    PlanDecomposition,
    SubGoalIntent,
)
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter


class ConsoleLogger(IEventLogger):
    """Gibt alle getriggerten Simulationsevents direkt auf der Standardausgabe aus."""

    def log(self, event: object) -> None:
        event_name = getattr(event, "name", type(event).__name__)
        payload = getattr(event, "payload", str(event))
        print(f"  [EVENT] {event_name}: {payload}")


async def run_stage_2_debug() -> None:
    grid = WorldGrid(width=5, height=5)
    presenter = MagicMock(spec=IPresenter)
    logger = ConsoleLogger()
    cognition = AsyncMock(spec=ICognitionProvider)

    # 1. Kognition liefert Kette: Erst heranbewegen nach (1, 2), dann konsumieren
    cognition.decompose_plan.return_value = PlanDecomposition(
        thought="Apfel sichtbar auf (1, 3). Ich bewege mich auf (1, 2) und esse ihn.",
        primary_goal="Nahrung beschaffen",
        sub_goals=[
            SubGoalIntent(
                action_type=ActionType.MOVE_TO,
                target_position=(1, 2),
                description="Zum Apfel laufen",
            ),
            SubGoalIntent(
                action_type=ActionType.CONSUME,
                target_entity_id="apple_1",
                description="Apfel essen",
            ),
        ],
    )

    # 2. Pathfinder über Port mocken: Liefert bei Ziel (1, 2) genau den Zwischenschritt [goal]
    pathfinder = MagicMock(spec=IPathfinder)
    pathfinder.find_path.side_effect = lambda start, goal, mmap: [goal] if start != goal else []

    engine = SimulationEngine(
        grid=grid,
        pathfinder=pathfinder,
        presenter=presenter,
        logger=logger,
        cognition_provider=cognition,
        tick_interval=0.0,
    )

    # Agent Alice bei (1, 1) mit Hunger 0.85
    agent = Agent(id="agent_1", name="Alice", position=Position(1, 1))
    agent.needs["hunger"] = 0.85
    engine.register_agent(agent)

    # Apfel bei (1, 3)
    apple = WorldEntity(
        id="apple_1",
        name="Apfel",
        position=Position(1, 3),
        is_consumable=True,
        nutrition_value=0.5,
    )
    engine.register_entity(apple)

    print("======================================================================")
    print("START DEBUG: test_hunger_simulation_stage_2_approach_and_consume")
    print(f"Init: Agent Alice bei {agent.position}, Hunger={agent.needs['hunger']}")
    print(f"Init: Apfel bei {apple.position}, Consumable={apple.is_consumable}")
    print("======================================================================\n")

    for tick in range(1, 4):
        print(f"--- [TAKT {tick}] VOR engine.process_tick() ---")
        active_goal_name = agent.active_goal.name if agent.active_goal else "None"
        goal_stack = [g.name for g in agent.goals]
        print(f"  Position:    {agent.position}")
        print(f"  Path:        {agent.path}")
        print(f"  Has Path:    {agent.has_path}")
        print(f"  Active Goal: {active_goal_name}")
        print(f"  Goal Stack:  {goal_stack}")
        print(f"  Hunger:      {agent.needs.get('hunger')}")

        await engine.process_tick()

        print(f"--- [TAKT {tick}] NACH engine.process_tick() ---")
        active_goal_name = agent.active_goal.name if agent.active_goal else "None"
        goal_stack = [g.name for g in agent.goals]
        entities_ids = [e.id for e in engine._entities]
        print(f"  Position:    {agent.position}")
        print(f"  Path:        {agent.path}")
        print(f"  Has Path:    {agent.has_path}")
        print(f"  Active Goal: {active_goal_name}")
        print(f"  Goal Stack:  {goal_stack}")
        print(f"  Hunger:      {agent.needs.get('hunger')}")
        print(f"  Entities:    {entities_ids}")
        print("--------------------------------------------------\n")

        if apple not in engine._entities:
            print(f">>> Erfolg: Apfel wurde in Takt {tick} konsumiert und aus _entities entfernt. <<<\n")
            break

    print("======================================================================")
    print(f"Ergebnis: apple not in engine._entities -> {apple not in engine._entities}")
    print(f"Finale Position Alice: {agent.position}")
    print(f"Finaler Hunger Alice:   {agent.needs.get('hunger')}")
    print("======================================================================")


if __name__ == "__main__":
    asyncio.run(run_stage_2_debug())