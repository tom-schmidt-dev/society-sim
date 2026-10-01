from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.application.simulation_engine import SimulationEngine
from src.application.services.lifecycle.need_service import NeedService
from src.application.services.cognition.plan_decomposition_service import PlanDecompositionService
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.planning import ActionType, PlanDecomposition, SubGoalIntent
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter


@pytest.mark.asyncio
async def test_hunger_simulation_stage_1_direct_consumption():
    grid = WorldGrid(width=5, height=5)
    pathfinder = MagicMock(spec=IPathfinder)
    presenter = MagicMock(spec=IPresenter)
    logger = MagicMock(spec=IEventLogger)
    cognition = AsyncMock(spec=ICognitionProvider)

    # 1. Kognition liefert direkten Konsum-Plan
    cognition.decompose_plan.return_value = PlanDecomposition(
        thought="Nahrung liegt direkt neben mir. Verzehre Apfel.",
        primary_goal="Hunger stillen",
        sub_goals=[
            SubGoalIntent(
                action_type=ActionType.CONSUME,
                target_entity_id="apple_1",
                description="Apfel essen",
            )
        ],
    )

    engine = SimulationEngine(
        grid=grid,
        pathfinder=pathfinder,
        presenter=presenter,
        logger=logger,
        cognition_provider=cognition,
        tick_interval=0.0,
    )

    agent = Agent(id="agent_1", name="Alice", position=Position(2, 2))
    agent.needs["hunger"] = 0.85
    engine.register_agent(agent)

    apple = WorldEntity(
        id="apple_1",
        name="Apfel",
        position=Position(2, 3),
        is_consumable=True,
        nutrition_value=0.5,
    )
    engine.register_entity(apple)

    # 2. Takt ausführen
    await engine.process_tick()

    # 3. Assertions
    assert apple not in engine._entities
    assert pytest.approx(agent.needs["hunger"], 0.001) == 0.35
    assert agent.active_goal is None or agent.active_goal.status == "completed"

    logged_event_types = [call.args[0].event_type for call in logger.log.call_args_list]
    assert "entity_consumed" in logged_event_types

@pytest.mark.asyncio
async def test_hunger_simulation_stage_2_approach_and_consume():
    grid = WorldGrid(width=5, height=5)
    presenter = MagicMock(spec=IPresenter)
    logger = MagicMock(spec=IEventLogger)
    cognition = AsyncMock(spec=ICognitionProvider)

    # 1. Kognition liefert Kette: Erst heranbewegen, dann konsumieren
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

    # 2. Pathfinder über Port mocken: Liefert bei Ziel (1, 2) genau den Zwischenschritt
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

    agent = Agent(id="agent_1", name="Alice", position=Position(1, 1))
    agent.needs["hunger"] = 0.85
    engine.register_agent(agent)

    apple = WorldEntity(
        id="apple_1",
        name="Apfel",
        position=Position(1, 3),
        is_consumable=True,
        nutrition_value=0.5,
    )
    engine.register_entity(apple)

    # 3. Simulation über bis zu 3 Takte ausführen (Planung/Schritt -> Zielabschluss -> Konsum)
    for _ in range(3):
        await engine.process_tick()
        if apple not in engine._entities:
            break

    # 4. Assertions
    assert apple not in engine._entities
    assert pytest.approx(agent.needs["hunger"], 0.001) == 0.36
    assert agent.position == Position(1, 2)
    assert not any(g.name.startswith("SubGoal:") for g in agent.goals)

@pytest.mark.asyncio
async def test_hunger_simulation_stage_3_explore_and_consume():
    grid = WorldGrid(width=10, height=10)
    presenter = MagicMock(spec=IPresenter)
    logger = MagicMock(spec=IEventLogger)
    cognition = AsyncMock(spec=ICognitionProvider)

    # 1. Kognitionspläne vorbereiten:
    # Phase A: Keine Nahrung im Speicher -> Erkundungsabsicht
    explore_plan = PlanDecomposition(
        thought="Keine Nahrung bekannt. Ich erkunde das Gebiet.",
        primary_goal="Nahrung suchen",
        sub_goals=[
            SubGoalIntent(action_type=ActionType.EXPLORE, description="Gebiet erkunden"),
            SubGoalIntent(action_type=ActionType.CONSUME, description="Nahrung verzehren"),
        ],
    )

    # Phase B: Nach Entdeckung des Apfels -> Gezielter Anlauf und Verzehr
    consume_plan = PlanDecomposition(
        thought="Apfel entdeckt. Ich laufe hin und esse ihn.",
        primary_goal="Hunger stillen",
        sub_goals=[
            SubGoalIntent(
                action_type=ActionType.MOVE_TO,
                target_position=(1, 3),
                description="Zum Apfel laufen",
            ),
            SubGoalIntent(
                action_type=ActionType.CONSUME,
                target_entity_id="apple_1",
                description="Apfel essen",
            ),
        ],
    )

    cognition.decompose_plan.side_effect = [explore_plan, consume_plan]

    # Deterministischer Manhattan-Pathfinder für schrittweise Wegfindung
    def mock_find_path(start: Position, goal: Position, mmap) -> list[Position]:
        if start == goal:
            return []
        path = []
        curr = start
        while curr != goal:
            dx = 1 if goal.x > curr.x else (-1 if goal.x < curr.x else 0)
            dy = 1 if goal.y > curr.y else (-1 if goal.y < curr.y else 0)
            if dx != 0:
                curr = Position(curr.x + dx, curr.y)
            else:
                curr = Position(curr.x, curr.y + dy)
            path.append(curr)
        return path

    pathfinder = MagicMock(spec=IPathfinder)
    pathfinder.find_path.side_effect = mock_find_path

    engine = SimulationEngine(
        grid=grid,
        pathfinder=pathfinder,
        presenter=presenter,
        logger=logger,
        cognition_provider=cognition,
        tick_interval=0.0,
    )

    # Agent auf (1, 1) mit akutem Hunger und beschränkter mentaler Karte
    agent = Agent(id="agent_1", name="Alice", position=Position(1, 1))
    agent.needs["hunger"] = 0.85
    agent.mental_map.set_bounds(10, 10)
    # Nur Ausgangskachel ist initial bekannt, Rest gilt als Frontier
    agent.mental_map.update_tile(Position(1, 1), is_walkable=True, tick=1)
    engine.register_agent(agent)

    # Apfel außerhalb des initialen Sichtfelds auf (1, 4)
    apple = WorldEntity(
        id="apple_1",
        name="Apfel",
        position=Position(1, 4),
        entity_type="food",
        is_consumable=True,
        nutrition_value=0.5,
    )
    engine.register_entity(apple)

    # Simulation für bis zu 8 Takte ausführen (Exploration -> Fund -> Re-Planung -> Annäherung -> Verzehr)
    for _ in range(8):
        await engine.process_tick()
        if apple not in engine._entities:
            break

    # Zusicherungen (Postkonditionen)
    assert apple not in engine._entities
    assert agent.needs["hunger"] < 0.70
    assert not any(g.name.startswith("SubGoal:") for g in agent.goals)

    # Ereignisprotokolle verifizieren
    logged_event_types = [call.args[0].event_type for call in logger.log.call_args_list]
    assert "goal_interrupted_for_replan" in logged_event_types
    assert "entity_consumed" in logged_event_types