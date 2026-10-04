from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any, Callable, Coroutine, Optional, TypeVar

from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.coordination.conflict_coordinator import ConflictCoordinator
from src.application.services.coordination.dialogue_history import DialogueHistory
from src.application.services.coordination.dialogue_session_manager import DialogueSessionManager
from src.application.services.interaction.interaction_dispatcher import InteractionDispatcher
from src.application.services.movement.evasion_finder import EvasionFinder
from src.application.services.cognition.goal_service import GoalService
from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent.agent import Agent
from src.domain.models.communication.message import IncomingMessage
from src.domain.models.planning.cognition import SocialReflection
from src.domain.models.planning.events import SimulationEvent
from src.domain.models.planning.goal import Goal
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.presenter import IPresenter
from src.infrastructure.pathfinding.astar import AStarPathfinder

T = TypeVar("T")


def async_test(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return asyncio.run(func(*args, **kwargs))
    return wrapper


class MockEventLogger(IEventLogger):
    def __init__(self) -> None:
        self.events: list[SimulationEvent] = []

    def log(self, event: SimulationEvent) -> None:
        self.events.append(event)


class MockPresenter(IPresenter):
    def render(
        self,
        grid: WorldGrid,
        entities: list[Any],
        tick: int,
        dialogues: Optional[list[str]] = None,
        known_positions: Optional[set[Position]] = None,
        snapshots: Optional[list[Any]] = None,
    ) -> None:
        pass


class MockCognitionProvider(ICognitionProvider):
    async def decide_next_goal(self, context: dict[str, Any]) -> Any:
        raise NotImplementedError

    async def resolve_blockage(self, context: dict[str, Any]) -> Any:
        raise NotImplementedError

    async def evaluate_goal_status(self, context: dict[str, Any]) -> Any:
        raise NotImplementedError

    async def respond_to_dialogue(self, context: dict[str, Any]) -> Any:
        raise NotImplementedError

    async def reflect_on_dialogue(self, context: dict[str, Any]) -> SocialReflection:
        return SocialReflection(
            assessment="Mock-Einschätzung",
            progression_summary="Mock-Zusammenfassung",
        )


def setup_engine() -> tuple[SimulationEngine, WorldGrid, MockEventLogger]:
    grid = WorldGrid(width=90, height=45)
    for x in range(10, 80):
        grid.set_obstacle(Position(x, 21))
        grid.set_obstacle(Position(x, 23))

    grid.remove_obstacle(Position(45, 21))
    grid.set_obstacle(Position(44, 20))
    grid.set_obstacle(Position(45, 20))
    grid.set_obstacle(Position(46, 20))

    pathfinder = AStarPathfinder()
    presenter = MockPresenter()
    logger = MockEventLogger()
    cognition = MockCognitionProvider()

    engine = SimulationEngine(
        grid=grid,
        pathfinder=pathfinder,
        presenter=presenter,
        logger=logger,
        cognition_provider=cognition,
        tick_interval=0.0,
    )
    return engine, grid, logger


def populate_agent_mental_corridor(agent: Agent, width: int = 90, height: int = 45) -> None:
    agent.mental_map.set_bounds(width, height)
    for x in range(10, 80):
        agent.mental_map.update_tile(Position(x, 22), is_walkable=True, tick=1)
        agent.mental_map.mark_obstacle(Position(x, 21), tick=1)
        agent.mental_map.mark_obstacle(Position(x, 23), tick=1)

    agent.mental_map.update_tile(Position(45, 21), is_walkable=True, tick=1)
    agent.mental_map.mark_obstacle(Position(44, 20), tick=1)
    agent.mental_map.mark_obstacle(Position(45, 20), tick=1)
    agent.mental_map.mark_obstacle(Position(46, 20), tick=1)


@async_test
async def test_junction_halt_and_niche_entry_handshake() -> None:
    engine, grid, logger = setup_engine()

    alice = Agent(id="1", name="Alice", position=Position(44, 22))
    bob = Agent(id="2", name="Bob", position=Position(47, 22))
    engine.register_agent(alice)
    engine.register_agent(bob)

    populate_agent_mental_corridor(alice)
    populate_agent_mental_corridor(bob)

    bob.push_goal(Goal(name="West-Tor", target_position=Position(10, 22)))
    bob.assign_path([Position(46, 22), Position(45, 22), Position(44, 22)])

    alice.assign_path([Position(45, 22), Position(45, 21)])
    alice.push_goal(
        Goal(
            name="Anfahrt Nische",
            target_position=Position(45, 21),
            junction_position=Position(45, 22),
            yield_for_agent_id=bob.id,
            is_evasion_hold=False,
        )
    )

    await engine.process_tick()
    assert alice.position == Position(45, 22)
    alice_goal = alice.active_goal
    assert alice_goal is not None
    assert alice_goal.halt_signaled is True
    assert any(m.is_halt_request for m in bob.inbox)

    await engine.process_tick()
    assert alice.position == Position(45, 21)
    alice_goal = alice.active_goal
    assert alice_goal is not None
    assert alice_goal.name == "Nischen-Halt"
    assert alice_goal.is_evasion_hold is True
    assert any(m.is_resume_signal for m in bob.inbox)

    await engine.process_tick()
    assert alice.position == Position(45, 21)
    bob_goal = bob.active_goal
    assert bob_goal is not None
    assert bob_goal.name == "West-Tor"
    assert bob.is_busy is False


@async_test
async def test_clearance_and_courtesy_handshake() -> None:
    engine, grid, logger = setup_engine()

    alice = Agent(id="1", name="Alice", position=Position(45, 21))
    bob = Agent(id="2", name="Bob", position=Position(46, 22))
    engine.register_agent(alice)
    engine.register_agent(bob)

    populate_agent_mental_corridor(alice)
    populate_agent_mental_corridor(bob)

    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(87, 22)))
    alice.push_goal(
        Goal(
            name="Nischen-Halt",
            holds_position=True,
            is_evasion_hold=True,
            junction_position=Position(45, 22),
            yield_for_agent_id=bob.id,
        )
    )

    bob.push_goal(Goal(name="West-Tor", target_position=Position(10, 22)))
    bob.assign_path([Position(45, 22), Position(44, 22), Position(43, 22)])

    await engine.process_tick()
    assert bob.position == Position(45, 22)
    alice_goal = alice.active_goal
    assert alice_goal is not None
    assert alice_goal.name == "Nischen-Halt"

    await engine.process_tick()
    assert bob.position == Position(44, 22)
    alice_goal = alice.active_goal
    assert alice_goal is not None
    assert alice_goal.name == "Nischen-Halt"

    await engine.process_tick()
    assert bob.position == Position(43, 22)
    courtesy_msgs = [m for m in alice.inbox if m.is_courtesy]
    assert len(courtesy_msgs) == 1
    assert "Danke fürs Platz machen" in courtesy_msgs[0].message

    await engine.process_tick()
    alice_goal = alice.active_goal
    assert alice_goal is not None
    assert alice_goal.name == "Ost-Tor"
    assert alice.has_path is True
    assert alice.path[0] == Position(45, 22)


@async_test
async def test_underway_path_update_synchronization() -> None:
    engine, grid, logger = setup_engine()

    alice = Agent(id="1", name="Alice", position=Position(40, 22))
    bob = Agent(id="2", name="Bob", position=Position(48, 22))
    engine.register_agent(alice)
    engine.register_agent(bob)

    populate_agent_mental_corridor(alice)
    populate_agent_mental_corridor(bob)

    alice.push_goal(Goal(name="Anfahrt Nische", target_position=Position(45, 21), holds_position=False))
    alice.assign_path([Position(41, 22), Position(42, 22)])

    corr_key = "upd-test-123"
    bob.push_goal(
        Goal(
            name="Wartet auf Freigabe",
            holds_position=True,
            yield_for_agent_id=alice.id,
            correlation_key=corr_key,
        )
    )

    new_planned_path = [Position(48, 22), Position(47, 22), Position(46, 22)]
    alice.receive_message(
        IncomingMessage(
            from_agent_id=bob.id,
            from_agent_name=bob.name,
            message="HALT WARTE!",
            is_path_update=True,
            planned_path=new_planned_path,
            correlation_key=corr_key,
        )
    )

    await engine.process_tick()
    assert not any(g.name == "Unterwegs-Halt" for g in alice.goals)
    assert any(m.is_resume_signal for m in bob.inbox)

    await engine.process_tick()
    assert not any(g.correlation_key == corr_key for g in bob.goals)
    assert bob.is_busy is False


@async_test
async def test_farewell_message_does_not_spawn_dialogue_or_evasion() -> None:
    engine, grid, logger = setup_engine()

    alice = Agent(id="1", name="Alice", position=Position(20, 22))
    bob = Agent(id="2", name="Bob", position=Position(25, 22))
    engine.register_agent(alice)
    engine.register_agent(bob)

    alice.push_goal(Goal(name="Hauptziel", target_position=Position(80, 22)))

    alice.receive_message(
        IncomingMessage(
            from_agent_id=bob.id,
            from_agent_name=bob.name,
            message="Tschüss, ich muss weiter.",
            is_farewell=True,
        )
    )

    await engine.process_tick()

    assert len(alice.inbox) == 0
    alice_goal = alice.active_goal
    assert alice_goal is not None
    assert alice_goal.name == "Hauptziel"
    assert alice.is_thinking is False
    assert not any("In Nische ausweichen" in g.name for g in alice.goals)


@async_test
async def test_evasion_suppressed_when_partner_already_yielding() -> None:
    engine, grid, logger = setup_engine()

    alice = Agent(id="1", name="Alice", position=Position(44, 22))
    bob = Agent(id="2", name="Bob", position=Position(45, 22))
    engine.register_agent(alice)
    engine.register_agent(bob)

    populate_agent_mental_corridor(alice)
    populate_agent_mental_corridor(bob)

    alice.push_goal(Goal(name="Hauptziel", target_position=Position(80, 22)))

    bob.push_goal(
        Goal(
            name="In Nische ausweichen",
            target_position=Position(45, 21),
            yield_for_agent_id=alice.id,
            is_evasion_hold=False,
        )
    )

    alice.assign_path([Position(45, 22)])

    pathfinder = AStarPathfinder()
    history = DialogueHistory()
    goal_service = GoalService(logger, MockCognitionProvider(), pathfinder)
    evasion_finder = EvasionFinder(pathfinder)
    session_manager = DialogueSessionManager()
    executor = ActionExecutor(grid, logger, history, goal_service, pathfinder, evasion_finder=evasion_finder)
    dispatcher = InteractionDispatcher(logger=logger)

    coordinator = ConflictCoordinator(
        logger=logger,
        cognition_provider=MockCognitionProvider(),
        pathfinder=pathfinder,
        goal_service=goal_service,
        evasion_finder=evasion_finder,
        action_executor=executor,
        interaction_dispatcher=dispatcher,
        session_manager=session_manager,
        dialogue_history=history,
    )

    await coordinator.resolve_blockage(
        agent=alice,
        blocker=bob,
        blocked_pos=Position(45, 22),
        all_entities=[alice, bob],
    )

    alice_goal = alice.active_goal
    assert alice_goal is not None
    assert alice_goal.name == "Hauptziel"
    assert not any("In Nische ausweichen" in g.name for g in alice.goals)