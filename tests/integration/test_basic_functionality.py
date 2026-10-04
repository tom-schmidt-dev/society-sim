from __future__ import annotations

from typing import Any, Optional
import pytest

from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_state import AgentLifecycleState
from src.domain.models.interaction.commands import TalkCommand
from src.domain.models.interaction.interaction_result import ImmediateResult, PendingFuture
from src.domain.models.planning.cognition import BlockedResolution, SocialReflection, WaitAction
from src.domain.models.planning.events import SimulationEvent
from src.domain.models.planning.goal import Goal
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.presenter import IPresenter
from src.infrastructure.pathfinding.astar import AStarPathfinder


class MockEventLogger(IEventLogger):
    def __init__(self) -> None:
        self.events: list[SimulationEvent] = []

    def log(self, event: SimulationEvent) -> None:
        self.events.append(event)

    def has_event(self, event_type: str) -> bool:
        return any(e.event_type == event_type for e in self.events)

    def get_events(self, event_type: str) -> list[SimulationEvent]:
        return [e for e in self.events if e.event_type == event_type]


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


@pytest.fixture
def test_setup():
    grid = WorldGrid(width=20, height=20)
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
        auditory_radius=3,
        enable_deterministic_corridor=False,
    )
    return engine, logger, grid


@pytest.mark.asyncio
async def test_interaction_queued_when_blocker_is_busy_with_third_party(test_setup):
    """Prüft Fail-Fast: Fremdbeschäftigte Blocker weisen Anfragen sofort mit BUSY ab; Autonomie bleibt erhalten."""
    engine, logger, _ = test_setup

    agent_a = Agent(id="agent_a", name="Alice", position=Position(5, 5))
    agent_b = Agent(id="agent_b", name="Bob", position=Position(6, 5), is_conversational=True)
    agent_c = Agent(id="agent_c", name="Charlie", position=Position(7, 5), is_conversational=True)

    engine.register_agent(agent_a)
    engine.register_agent(agent_b)
    engine.register_agent(agent_c)

    agent_b.interaction_partner_id = agent_a.id
    agent_a.interaction_partner_id = agent_b.id

    cmd = TalkCommand(
        source_entity_id=agent_c.id,
        target_entity_id=agent_b.id,
        message="Bitte Durchgang freigeben",
        intent="request_yield",
    )
    res = engine.interaction_dispatcher.dispatch(agent_c, agent_b, cmd)

    assert isinstance(res, ImmediateResult)
    assert res.success is False
    assert res.reason == "BUSY"
    assert agent_c.lifecycle_state == AgentLifecycleState.IDLE
    assert logger.has_event("interaction_busy_rejected")


@pytest.mark.asyncio
async def test_duplicate_interaction_request_prevented(test_setup):
    """Prüft, dass bei freiem Blocker eine Bindung an ein PendingFuture erfolgt und der Status konsistent bleibt."""
    engine, logger, _ = test_setup

    agent_b = Agent(id="agent_b", name="Bob", position=Position(6, 5), is_conversational=True)
    agent_c = Agent(id="agent_c", name="Charlie", position=Position(7, 5), is_conversational=True)

    engine.register_agent(agent_b)
    engine.register_agent(agent_c)

    cmd = TalkCommand(
        source_entity_id=agent_c.id,
        target_entity_id=agent_b.id,
        message="Hallo",
        intent="request_yield",
    )
    res = engine.interaction_dispatcher.dispatch(agent_c, agent_b, cmd)

    assert isinstance(res, PendingFuture)
    assert agent_c.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER
    assert agent_c.interaction_partner_id == agent_b.id
    assert len(agent_b.interaction_mailbox) == 1


@pytest.mark.asyncio
async def test_queue_request_expires_with_ttl(test_setup):
    """Prüft, dass der Fast-Path ein Warteziel beendet, sobald der Blocker frei wird."""
    engine, logger, _ = test_setup

    agent_b = Agent(id="agent_b", name="Bob", position=Position(5, 5))
    agent_c = Agent(id="agent_c", name="Charlie", position=Position(6, 5))
    engine.register_agent(agent_b)
    engine.register_agent(agent_c)

    agent_c.push_goal(Goal(name="Hauptziel", target_position=Position(1, 5)))
    wait_goal = Goal(
        name="Warten auf Partner",
        remaining_ticks=200,
        yield_for_agent_id=agent_b.id,
        initial_wait_tick=1,
    )
    agent_c.push_goal(wait_goal)

    agent_b.interaction_partner_id = None
    agent_b.is_thinking = False

    engine.protocol_service.check_waiting_partner_fast_path(agent_c, engine._entities)

    assert agent_c.active_goal.name == "Hauptziel"
    assert logger.has_event("wait_interrupted_fast_path")


@pytest.mark.asyncio
async def test_queue_lazy_validation_drops_when_target_moved(test_setup):
    """Prüft, dass der Fast-Path greift, wenn der Blocker die Kachel vor dem Agenten verlässt."""
    engine, logger, _ = test_setup

    agent_b = Agent(id="agent_b", name="Bob", position=Position(5, 5))
    agent_c = Agent(id="agent_c", name="Charlie", position=Position(4, 5))
    engine.register_agent(agent_b)
    engine.register_agent(agent_c)

    agent_c.assign_path([Position(5, 5)])
    agent_c.push_goal(Goal(name="Hauptziel", target_position=Position(10, 5)))
    wait_goal = Goal(
        name="Warten auf Partner",
        remaining_ticks=200,
        yield_for_agent_id=agent_b.id,
        initial_wait_tick=1,
    )
    agent_c.push_goal(wait_goal)

    agent_b.position = Position(5, 6)

    engine.protocol_service.check_waiting_partner_fast_path(agent_c, engine._entities)

    assert agent_c.active_goal.name == "Hauptziel"
    assert logger.has_event("wait_interrupted_fast_path")


@pytest.mark.asyncio
async def test_queue_fifo_order_processing(test_setup):
    """Prüft, dass asynchrone Mailbox-Befehle deterministisch registriert und aufgelöst werden."""
    engine, logger, _ = test_setup

    agent_b = Agent(id="agent_b", name="Bob", position=Position(5, 5), is_conversational=True)
    agent_c = Agent(id="agent_c", name="Charlie", position=Position(4, 5), is_conversational=True)

    engine.register_agent(agent_b)
    engine.register_agent(agent_c)

    cmd_c = TalkCommand(source_entity_id=agent_c.id, target_entity_id=agent_b.id, message="Hi Bob")
    res_c = engine.interaction_dispatcher.dispatch(agent_c, agent_b, cmd_c)

    assert isinstance(res_c, PendingFuture)
    assert len(agent_b.interaction_mailbox) == 1
    assert agent_c.lifecycle_state == AgentLifecycleState.WAITING_FOR_PEER

    reply = ImmediateResult(success=True, reason="TALK_REPLY")
    agent_b.interaction_mailbox[0][1].set_result(reply)

    import asyncio
    await asyncio.sleep(0)

    assert agent_c.lifecycle_state == AgentLifecycleState.DELIBERATING