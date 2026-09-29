from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.application.services.action_executor import ActionExecutor
from src.application.services.dialogue_history import DialogueHistory
from src.application.services.dialogue_session_manager import DialogueSessionManager
from src.application.services.evasion_finder import EvasionFinder
from src.application.services.goal_service import GoalService
from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent import Agent
from src.domain.models.cognition import GoalEvaluation
from src.domain.models.goal import Goal
from src.domain.models.mental_map import AgentMentalMap
from src.domain.models.message import IncomingMessage
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter
from src.domain.services.perception_service import PerceptionService
from src.infrastructure.pathfinding.astar import AStarPathfinder


@pytest.fixture
def mocked_env():
    grid = MagicMock(spec=WorldGrid)
    grid.is_walkable.return_value = True
    grid.is_within_bounds.return_value = True
    grid.width = 50
    grid.height = 50

    logger = MagicMock(spec=IEventLogger)
    presenter = MagicMock(spec=IPresenter)
    cognition = AsyncMock(spec=ICognitionProvider)
    pathfinder = MagicMock(spec=IPathfinder)
    dialogue_history = MagicMock(spec=DialogueHistory)
    perception = MagicMock(spec=PerceptionService)

    pathfinder.find_path.side_effect = lambda start, goal, mmap: [goal]

    current_tick = 100
    goal_service = GoalService(
        logger=logger,
        cognition_provider=cognition,
        pathfinder=pathfinder,
        tick_provider=lambda: current_tick,
    )
    evasion_finder = EvasionFinder(pathfinder=pathfinder)

    executor = ActionExecutor(
        grid=grid,
        logger=logger,
        dialogue_history=dialogue_history,
        goal_service=goal_service,
        pathfinder=pathfinder,
        evasion_finder=evasion_finder,
        tick_provider=lambda: current_tick,
    )

    engine = SimulationEngine(
        grid=grid,
        pathfinder=pathfinder,
        presenter=presenter,
        logger=logger,
        cognition_provider=cognition,
        tick_interval=0.001,
        goal_service=goal_service,
        perception_service=perception,
        dialogue_history=dialogue_history,
    )

    return {
        "engine": engine,
        "grid": grid,
        "cognition": cognition,
        "pathfinder": pathfinder,
        "goal_service": goal_service,
        "executor": executor,
        "logger": logger,
    }


# ==============================================================================
# TEST 1: Kognitive Fehlauswertung darf Nischen-Halt NICHT vorzeitig poppen
# ==============================================================================
@pytest.mark.asyncio
async def test_sub_goal_evaluation_must_never_pop_evasion_hold(mocked_env) -> None:
    """
    Testet Tick 215 des Live-Logs: Wenn das LLM fälschlicherweise behauptet,
    das Zwischenziel sei erfüllt, darf ein is_evasion_hold niemals durch
    evaluate_sub_goal_completion entfernt werden, sondern NUR durch ein Signal.
    """
    env = mocked_env
    cognition: AsyncMock = env["cognition"]
    goal_service: GoalService = env["goal_service"]

    alice = Agent(id="1", name="Alice", position=Position(43, 20))
    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(87, 22)))
    hold_goal = Goal(
        name="Nischen-Halt",
        holds_position=True,
        is_evasion_hold=True,
        junction_position=Position(44, 22),
        yield_for_agent_id="2",
    )
    alice.push_goal(hold_goal)

    cognition.evaluate_goal_status.return_value = GoalEvaluation(
        thought="Partner ist noch da, aber ich bin in der Nische, also fertig.",
        is_completed=True,
        reason="Nische erreicht",
    )

    await goal_service.evaluate_sub_goal_completion(
        agent=alice,
        grid=env["grid"],
        recent_dialogues=[],
    )

    assert alice.active_goal is not None
    assert alice.active_goal.name == "Nischen-Halt"
    assert alice.active_goal.is_evasion_hold is True


# ==============================================================================
# TEST 2: Falsche Junction-Berechnung (Reproduktion von junction=[9, 22])
# ==============================================================================
def test_junction_must_be_strictly_adjacent_to_evasion_tile() -> None:
    """
    Reproduziert den Fehler im Live-Run, bei dem junction_position=(9,22) berechnet
    wurde, obwohl Start bei (44,22) und Nische bei (45,21) lag.
    """
    pathfinder = AStarPathfinder()
    evasion_finder = EvasionFinder(pathfinder=pathfinder)

    mmap = AgentMentalMap(width=90, height=30)
    for x in range(90):
        mmap.update_tile(Position(x, 22), is_walkable=True, tick=1)
    mmap.update_tile(Position(45, 21), is_walkable=True, tick=1)

    start = Position(44, 22)
    blocked_pos = Position(45, 22)

    partner_trajectory = [Position(x, 22) for x in range(87, 1, -1)]

    res = evasion_finder.find_nearest_evasion_tile(
        start=start,
        blocked_pos=blocked_pos,
        grid=mmap,
        occupied_positions=set(),
        partner_trajectory=partner_trajectory,
    )

    assert res is not None
    assert res.junction_tile.y == 22
    assert abs(res.junction_tile.x - res.target_tile.x) <= 1
    assert res.target_tile == Position(45, 21)


# ==============================================================================
# TEST 3: Strikter Clearance-Trigger - Verfrühte Clearance verhindern
# ==============================================================================
@pytest.mark.asyncio
async def test_clearance_strictly_requires_both_l1_distance_and_junction_past(mocked_env) -> None:
    """
    Prüft die 2-Kachel-Bedingung: Wenn Bob sich von (46,22) über die Junction (45,22)
    nach Westen bewegt, darf erst bei x <= 43 das Courtesy-Signal ausgelöst werden.
    """
    env = mocked_env
    engine: SimulationEngine = env["engine"]

    alice = Agent(id="1", name="Alice", position=Position(45, 21))
    bob = Agent(id="2", name="Bob", position=Position(47, 22))
    engine.register_agent(alice)
    engine.register_agent(bob)

    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(87, 22)))
    alice.push_goal(
        Goal(
            name="Nischen-Halt",
            holds_position=True,
            is_evasion_hold=True,
            junction_position=Position(45, 22),
            yield_for_agent_id="2",
        )
    )

    bob.push_goal(Goal(name="West-Tor", target_position=Position(2, 22)))

    # 1. Schritt: Bob zieht auf (46, 22) -> Noch vor Junction
    bob.path = [Position(46, 22), Position(45, 22), Position(44, 22), Position(43, 22)]
    await engine.process_tick()
    assert bob.position == Position(46, 22)
    assert len(alice.inbox) == 0

    # 2. Schritt: Bob betritt Junction (45, 22) -> Darf KEINE Freigabe geben
    await engine.process_tick()
    assert bob.position == Position(45, 22)
    assert len(alice.inbox) == 0

    # 3. Schritt: Bob zieht auf (44, 22) -> Distanz ist 1 -> KEINE Freigabe
    await engine.process_tick()
    assert bob.position == Position(44, 22)
    assert len(alice.inbox) == 0

    # 4. Schritt: Bob zieht auf (43, 22) -> Distanz ist 2 und Junction liegt hinter ihm -> FREIGABE!
    await engine.process_tick()
    assert bob.position == Position(43, 22)
    assert len(alice.inbox) == 1
    msg = alice.inbox[0]
    assert msg.is_courtesy is True
    assert msg.is_resume_signal is True


# ==============================================================================
# TEST 4: Verhindern von Evasion-Hold Stack-Duplikaten
# ==============================================================================
@pytest.mark.asyncio
async def test_no_stacked_niche_holds_on_subsequent_ticks(mocked_env) -> None:
    """
    Testet Tick 322/397 des Live-Logs: Agenten durften nicht mehrfach 'Nischen-Halt'
    aufeinander stapeln, wenn sie bereits in der Nische stehen.
    """
    env = mocked_env
    engine: SimulationEngine = env["engine"]

    alice = Agent(id="1", name="Alice", position=Position(45, 21))
    engine.register_agent(alice)

    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(87, 22)))
    alice.push_goal(
        Goal(
            name="Nischen-Halt",
            holds_position=True,
            is_evasion_hold=True,
            junction_position=Position(45, 22),
            yield_for_agent_id="2",
            target_position=Position(45, 21),
        )
    )

    for _ in range(5):
        await engine.process_tick()

    hold_goals = [g for g in alice.goals if g.name == "Nischen-Halt"]
    assert len(hold_goals) == 1
    assert len(alice.goals) == 2


# ==============================================================================
# TEST 5: Nach Freigabe muss Re-Planning sofort neuen Pfad setzen
# ==============================================================================
@pytest.mark.asyncio
async def test_reawakening_immediately_assigns_path_out_of_niche(mocked_env) -> None:
    """
    Prüft, dass der Agent im exakt selben Takt, in dem er die Courtesy-Nachricht
    verarbeitet, einen neuen Pfad erhält und nicht im Status 'busy' verbleibt.
    """
    env = mocked_env
    engine: SimulationEngine = env["engine"]
    pathfinder = env["pathfinder"]

    alice = Agent(id="1", name="Alice", position=Position(45, 21))
    engine.register_agent(alice)

    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(87, 22)))
    alice.push_goal(
        Goal(
            name="Nischen-Halt",
            holds_position=True,
            is_evasion_hold=True,
            junction_position=Position(45, 22),
            yield_for_agent_id="2",
        )
    )

    path_out = [Position(45, 22), Position(46, 22), Position(47, 22)]
    pathfinder.find_path.side_effect = None
    pathfinder.find_path.return_value = path_out

    alice.receive_message(
        IncomingMessage(
            from_agent_id="2",
            from_agent_name="Bob",
            message="Danke fürs Platz machen!",
            is_courtesy=True,
            is_resume_signal=True,
        )
    )

    await engine.process_tick()

    assert alice.active_goal.name == "Ost-Tor"
    assert alice.path == path_out
    assert alice.is_busy is False
    assert alice.has_path is True