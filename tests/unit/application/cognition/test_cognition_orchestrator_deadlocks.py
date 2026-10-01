from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.application.services.cognition.cognition_orchestrator import CognitionOrchestrator
from src.application.services.cognition.frontier_explorer import FrontierExplorer
from src.application.services.cognition.goal_service import GoalService
from src.application.services.cognition.plan_decomposition_service import PlanDecompositionService
from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.lifecycle.need_service import NeedService
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.goal import Goal
from src.domain.models.planning.planning import ActionType, PlanDecomposition, SubGoalIntent
from src.domain.models.world.position import Position
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.services.precondition_evaluator import PreconditionEvaluator


@pytest.fixture
def orchestrator_setup() -> tuple[CognitionOrchestrator, dict[str, MagicMock]]:
    pathfinder = MagicMock(spec=IPathfinder)
    logger = MagicMock(spec=IEventLogger)
    goal_service = MagicMock(spec=GoalService)
    need_service = MagicMock(spec=NeedService)
    plan_decomp = MagicMock(spec=PlanDecompositionService)
    action_executor = MagicMock(spec=ActionExecutor)
    frontier_explorer = MagicMock(spec=FrontierExplorer)
    precondition_evaluator = MagicMock(spec=PreconditionEvaluator)

    orchestrator = CognitionOrchestrator(
        pathfinder=pathfinder,
        logger=logger,
        goal_service=goal_service,
        need_service=need_service,
        plan_decomposition_service=plan_decomp,
        action_executor=action_executor,
        frontier_explorer=frontier_explorer,
        precondition_evaluator=precondition_evaluator,
    )

    mocks = {
        "pathfinder": pathfinder,
        "logger": logger,
        "goal_service": goal_service,
        "need_service": need_service,
        "plan_decomp": plan_decomp,
        "action_executor": action_executor,
        "frontier_explorer": frontier_explorer,
        "precondition_evaluator": precondition_evaluator,
    }
    return orchestrator, mocks


def test_handle_subgoal_move_to_without_target_triggers_replan(
    orchestrator_setup: tuple[CognitionOrchestrator, dict[str, MagicMock]]
) -> None:
    """Prüft, dass move_to ohne target_position die Re-Planung erzwingt statt zu blockieren."""
    orchestrator, _ = orchestrator_setup
    agent = Agent(id="a1", name="Alice", position=Position(1, 1))
    invalid_goal = Goal(name="SubGoal: move_to", target_position=None)
    agent.goals.append(invalid_goal)

    orchestrator.handle_subgoal_move_to(agent, invalid_goal)

    # Muss Ziele leeren und Pfad zurücksetzen
    assert len(agent.goals) == 0
    assert agent.has_path is False


def test_handle_subgoal_explore_continues_when_frontiers_exist(
    orchestrator_setup: tuple[CognitionOrchestrator, dict[str, MagicMock]]
) -> None:
    """Prüft, dass explore bei Ankunft an Zwischenzielen neue Grenzkacheln wählt, solange unentdecktes Terrain existiert."""
    orchestrator, mocks = orchestrator_setup
    agent = Agent(id="a1", name="Alice", position=Position(4, 9))
    current_goal = Goal(name="SubGoal: explore", target_position=Position(4, 9))
    agent.goals.append(current_goal)

    # Keine Ressource bisher im Sichtfeld
    mocks["precondition_evaluator"].find_discovered_entity.return_value = None

    # Nächste Grenzkachel verfügbar
    next_frontier = Position(5, 9)
    mocks["frontier_explorer"].find_nearest_frontier.return_value = next_frontier
    mocks["pathfinder"].find_path.return_value = [Position(4, 9), next_frontier]

    orchestrator.handle_subgoal_explore(agent, current_goal)

    # Ziel darf nicht gepoppt werden, sondern wird auf die neue Koordinate gesetzt
    mocks["goal_service"].pop_goal.assert_not_called()
    assert current_goal.target_position == next_frontier
    assert agent.path == [Position(4, 9), next_frontier]


def test_handle_subgoal_consume_triggers_replan_if_target_missing(
    orchestrator_setup: tuple[CognitionOrchestrator, dict[str, MagicMock]]
) -> None:
    """Prüft, dass consume sofort Re-Planung anfordert, wenn an der Zielposition kein Objekt auffindbar ist."""
    orchestrator, _ = orchestrator_setup
    agent = Agent(id="a1", name="Alice", position=Position(3, 6))
    consume_goal = Goal(name="SubGoal: consume", target_position=Position(3, 6))
    agent.goals.append(consume_goal)

    # Leere Entitätenliste
    success = orchestrator.handle_subgoal_consume(agent, consume_goal, entities=[])

    assert success is False
    assert len(agent.goals) == 0


@pytest.mark.asyncio
async def test_trigger_plan_decomposition_filters_empty_move_to(
    orchestrator_setup: tuple[CognitionOrchestrator, dict[str, MagicMock]]
) -> None:
    """Prüft, dass Dekompositionen mit ungültigen move_to-Aktionen (ohne Koordinaten) gefiltert werden."""
    orchestrator, mocks = orchestrator_setup
    agent = Agent(id="a1", name="Alice", position=Position(3, 6))

    # LLM-Ausgabe mit ungültigem move_to-Intent
    plan = PlanDecomposition(
        thought="Ungültiger Plan",
        primary_goal="Essen beschaffen",
        sub_goals=[
            SubGoalIntent(
                action_type=ActionType.MOVE_TO,
                target_position=None,
                description="Move to current position",
            ),
            SubGoalIntent(
                action_type=ActionType.EXPLORE,
                description="Exploriere",
            ),
        ],
    )
    mocks["plan_decomp"].create_plan_for_need.return_value = plan

    await orchestrator.trigger_plan_decomposition(agent, "hunger")

    pushed_goals = [call.args[1].name for call in mocks["goal_service"].push_goal.call_args_list]

    # 'SubGoal: move_to' ohne Koordinaten darf nicht auf den Stack geschoben werden
    assert "SubGoal: move_to" not in pushed_goals
    assert "SubGoal: explore" in pushed_goals

@pytest.mark.asyncio
async def test_urgent_need_overrides_fallback_wait_goal(
    orchestrator_setup: tuple[CognitionOrchestrator, dict[str, MagicMock]]
) -> None:
    """Prüft, dass ein akutes Bedürfnis das Ziel 'Warten (Fallback)' verdrängt."""
    orchestrator, mocks = orchestrator_setup
    agent = Agent(id="a1", name="Alice", position=Position(1, 1))
    agent.goals.append(Goal(name="Warten (Fallback)"))

    mocks["need_service"].get_dominant_need.return_value = "hunger"
    mocks["need_service"].is_need_urgent.return_value = True

    plan = PlanDecomposition(
        thought="Plan zur Nahrungssuche",
        primary_goal="Nahrungssuche",
        sub_goals=[
            SubGoalIntent(
                action_type=ActionType.EXPLORE,
                description="Erkunde Terrain",
            )
        ],
    )
    mocks["plan_decomp"].create_plan_for_need.return_value = plan

    await orchestrator.process_agent_needs_and_cognition(agents=[agent], entities=[])

    # Warten (Fallback) muss gelöscht und durch den neuen Plan ersetzt worden sein
    pushed_goals = [call.args[1].name for call in mocks["goal_service"].push_goal.call_args_list]
    assert "Warten (Fallback)" not in [g.name for g in agent.goals]
    assert "Nahrungssuche" in pushed_goals
    assert "SubGoal: explore" in pushed_goals

def test_handle_subgoal_move_to_completes_when_adjacent_to_resource(
        orchestrator_setup: tuple[CognitionOrchestrator, dict[str, MagicMock]]
) -> None:
    """Prüft, dass move_to bereits bei Adjazenz zu einer Zielressource abschließt und den Pfad leert."""
    orchestrator, mocks = orchestrator_setup
    agent = Agent(id="a1", name="Alice", position=Position(5, 5))
    consume_goal = Goal(name="SubGoal: consume", target_position=Position(6, 5))
    move_goal = Goal(name="SubGoal: move_to", target_position=Position(6, 5))
    agent.goals.extend([consume_goal, move_goal])
    agent.assign_path([Position(6, 5)])

    orchestrator.handle_subgoal_move_to(agent, move_goal)

    # Ziel muss gepoppt und Pfad sofort gelöscht werden, damit kein Betreten des Baumes erfolgt
    mocks["goal_service"].pop_goal.assert_called_once_with(agent)
    assert agent.has_path is False