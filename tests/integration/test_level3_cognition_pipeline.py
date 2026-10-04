from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.application.services.cognition.cognition_orchestrator import CognitionOrchestrator
from src.application.services.cognition.goal_service import GoalService
from src.application.services.cognition.plan_decomposition_service import PlanDecompositionService
from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.lifecycle.need_service import NeedService
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.goal import Goal
from src.domain.models.world.position import Position
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from tests.integration.test_level2_plan_decomposition_with_store import (
    MockCognitionProvider,
    Level2VectorStoreStub,
)


@pytest.mark.asyncio
async def test_cognition_orchestrator_with_plan_decomp_and_vector_store() -> None:
    # 1. Test Doubles vorbereiten
    vector_store = Level2VectorStoreStub()
    vector_store.add_memories("a1", ["Tag 1: Apfelbaum bei (6, 5) erfolgreich genutzt."])

    cognition_provider = MockCognitionProvider()
    plan_decomp = PlanDecompositionService(
        cognition_provider=cognition_provider,
        vector_memory_store=vector_store,
    )

    pathfinder = MagicMock(spec=IPathfinder)
    pathfinder.find_path.side_effect = lambda start, target, _: [start, target]

    goal_service = GoalService(
        logger=MagicMock(spec=IEventLogger),
        cognition_provider=cognition_provider,
        pathfinder=pathfinder,
    )
    need_service = NeedService()

    orchestrator = CognitionOrchestrator(
        pathfinder=pathfinder,
        logger=MagicMock(spec=IEventLogger),
        goal_service=goal_service,
        need_service=need_service,
        plan_decomposition_service=plan_decomp,
        action_executor=MagicMock(spec=ActionExecutor),
    )

    # 2. Agent mit akutem Hunger und aktivem Fallback-Warten initialisieren
    agent = Agent(id="a1", name="Alice", position=Position(1, 1))
    agent.needs["hunger"] = 0.85
    agent.goals.append(Goal(name="Warten (Fallback)"))

    # 3. Kognitionsschritt ausführen
    await orchestrator.process_agent_needs_and_cognition(agents=[agent], entities=[])

    # 4. Verifikation: Fallback-Ziel wurde durch die dekodierte Erinnerung ersetzt
    active_goal = agent.active_goal
    assert active_goal is not None
    assert active_goal.name == "SubGoal: move_to"
    assert active_goal.target_position == Position(6, 5)