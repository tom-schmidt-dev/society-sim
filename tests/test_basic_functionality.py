from __future__ import annotations

import pytest
from typing import Any, Optional
from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent import Agent
from src.domain.models.cognition import (
    BlockedResolution,
    DialogueResolution,
    EndDialogueAction,
    GoalDecision,
    GoalEvaluation,
    MoveToAction,
    WaitAction,
)
from src.domain.models.events import SimulationEvent
from src.domain.models.goal import ExecutionPriority, Goal
from src.domain.models.interaction_request import InteractionRequest
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
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
        entities: list[WorldEntity],
        tick: int,
        dialogues: Optional[list[str]] = None,
        known_positions: Optional[set[Position]] = None,
    ) -> None:
        pass


class MockCognitionProvider(ICognitionProvider):
    async def decide_next_goal(self, context: dict[str, Any]) -> GoalDecision:
        return GoalDecision(
            thought="Dummy thought",
            action=MoveToAction(
                destination_name="Ziel",
                target_position=Position(0, 0),
                reason="Test",
            ),
        )

    async def resolve_blockage(self, context: dict[str, Any]) -> BlockedResolution:
        return BlockedResolution(
            thought="Warte kurz",
            action=WaitAction(ticks=1, reason="Test-Warten"),
        )

    async def evaluate_goal_status(self, context: dict[str, Any]) -> GoalEvaluation:
        return GoalEvaluation(
            thought="Eval",
            is_completed=False,
            reason="In Arbeit",
        )

    async def respond_to_dialogue(self, context: dict[str, Any]) -> DialogueResolution:
        return DialogueResolution(
            thought="Beenden",
            action=EndDialogueAction(reason="Test-Ende"),
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
    )
    return engine, logger, grid


@pytest.mark.asyncio
async def test_interaction_queued_when_blocker_is_busy_with_third_party(test_setup):
    """Prüft, ob ein Agent in die Warteschlange eingereiht wird, wenn der Blocker anderweitig beschäftigt ist."""
    engine, logger, _ = test_setup

    agent_a = Agent(id="agent_a", name="Alice", position=Position(5, 5))
    agent_b = Agent(id="agent_b", name="Bob", position=Position(6, 5))
    agent_c = Agent(id="agent_c", name="Charlie", position=Position(7, 5))

    engine.register_agent(agent_a)
    engine.register_agent(agent_b)
    engine.register_agent(agent_c)

    agent_b.interaction_partner_id = agent_a.id
    agent_a.interaction_partner_id = agent_b.id

    agent_c.path = [Position(6, 5)]

    await engine._conflict_coordinator.resolve_blockage(
        agent=agent_c,
        blocker=agent_b,
        blocked_pos=Position(6, 5),
        all_entities=engine._entities,
    )

    assert len(agent_b.interaction_queue) == 1
    assert agent_b.interaction_queue[0].requester_id == agent_c.id
    assert agent_c.is_waiting_for_reply is True
    assert agent_c.interaction_partner_id == agent_b.id
    assert logger.has_event("interaction_queued")


@pytest.mark.asyncio
async def test_duplicate_interaction_request_prevented(test_setup):
    """Prüft, dass wiederholte Blockaden desselben Agenten nicht zu Duplikaten in der Queue führen."""
    engine, logger, _ = test_setup

    agent_a = Agent(id="agent_a", name="Alice", position=Position(5, 5))
    agent_b = Agent(id="agent_b", name="Bob", position=Position(6, 5))
    agent_c = Agent(id="agent_c", name="Charlie", position=Position(7, 5))

    engine.register_agent(agent_a)
    engine.register_agent(agent_b)
    engine.register_agent(agent_c)

    agent_b.interaction_partner_id = agent_a.id
    agent_a.interaction_partner_id = agent_b.id
    agent_c.path = [Position(6, 5)]

    # Erster Blockade-Trigger: reiht Charlie ein
    await engine._conflict_coordinator.resolve_blockage(
        agent=agent_c,
        blocker=agent_b,
        blocked_pos=Position(6, 5),
        all_entities=engine._entities,
    )
    assert len(agent_b.interaction_queue) == 1

    # Zweiter Blockade-Trigger im selben Zustand: darf keine zweite Anfrage anhängen
    await engine._conflict_coordinator.resolve_blockage(
        agent=agent_c,
        blocker=agent_b,
        blocked_pos=Position(6, 5),
        all_entities=engine._entities,
    )
    assert len(agent_b.interaction_queue) == 1


@pytest.mark.asyncio
async def test_queue_request_expires_with_ttl(test_setup):
    """Prüft, ob veraltete Anfragen nach Ablauf der TTL deterministisch verworfen und Bereinigungen ausgelöst werden."""
    engine, logger, _ = test_setup

    agent_b = Agent(id="agent_b", name="Bob", position=Position(5, 5))
    agent_c = Agent(id="agent_c", name="Charlie", position=Position(6, 5))
    engine.register_agent(agent_b)
    engine.register_agent(agent_c)

    engine._current_tick = 5
    expired_req = InteractionRequest(
        requester_id=agent_c.id,
        target_id=agent_b.id,
        blocked_pos=Position(5, 5),
        tick=1,
        ttl_ticks=3,
    )
    agent_b.interaction_queue.append(expired_req)
    agent_c.is_waiting_for_reply = True
    agent_c.interaction_partner_id = agent_b.id

    await engine.process_tick()

    assert len(agent_b.interaction_queue) == 0
    assert agent_c.is_waiting_for_reply is False
    assert agent_c.interaction_partner_id is None
    assert logger.has_event("interaction_request_expired")


@pytest.mark.asyncio
async def test_queue_lazy_validation_drops_when_target_moved(test_setup):
    """Prüft, ob die Abarbeitung verworfen wird, wenn der Blocker die Kachel vor Entnahme bereits geräumt hat."""
    engine, logger, _ = test_setup

    agent_b = Agent(id="agent_b", name="Bob", position=Position(5, 5))
    agent_c = Agent(id="agent_c", name="Charlie", position=Position(4, 5))
    engine.register_agent(agent_b)
    engine.register_agent(agent_c)

    agent_c.path = [Position(5, 5)]
    valid_req = InteractionRequest(
        requester_id=agent_c.id,
        target_id=agent_b.id,
        blocked_pos=Position(5, 5),
        tick=engine.current_tick,
        ttl_ticks=3,
    )
    agent_b.interaction_queue.append(valid_req)
    agent_c.is_waiting_for_reply = True
    agent_c.interaction_partner_id = agent_b.id

    agent_b.position = Position(5, 6)

    await engine.process_tick()

    assert len(agent_b.interaction_queue) == 0
    assert logger.has_event("interaction_request_dropped")


@pytest.mark.asyncio
async def test_queue_fifo_order_processing(test_setup):
    """Prüft, dass mehrere Anfragen strikt nach First-In-First-Out (FIFO) abgearbeitet werden."""
    engine, logger, _ = test_setup

    agent_b = Agent(id="agent_b", name="Bob", position=Position(5, 5))
    agent_c = Agent(id="agent_c", name="Charlie", position=Position(4, 5))
    agent_d = Agent(id="agent_d", name="Dana", position=Position(6, 5))

    engine.register_agent(agent_b)
    engine.register_agent(agent_c)
    engine.register_agent(agent_d)

    agent_c.path = [Position(5, 5)]
    agent_d.path = [Position(5, 5)]

    req_c = InteractionRequest(
        requester_id=agent_c.id,
        target_id=agent_b.id,
        blocked_pos=Position(5, 5),
        tick=engine.current_tick,
    )
    req_d = InteractionRequest(
        requester_id=agent_d.id,
        target_id=agent_b.id,
        blocked_pos=Position(5, 5),
        tick=engine.current_tick,
    )

    agent_b.interaction_queue.extend([req_c, req_d])

    await engine.process_tick()

    dequeued_events = logger.get_events("interaction_dequeued")
    assert len(dequeued_events) == 1
    assert dequeued_events[0].payload["requester_id"] == agent_c.id
    assert len(agent_b.interaction_queue) == 1
    assert agent_b.interaction_queue[0].requester_id == agent_d.id