from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from src.application.services.cognition.plan_decomposition_service import PlanDecompositionService
from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.events import SimulationEvent
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter
from src.domain.ports.vector_memory_store import IVectorMemoryStore
from src.infrastructure.cognition.adapters.instructor_adapter import InstructorCognitionAdapter


class LocalInMemoryVectorStore(IVectorMemoryStore):
    def __init__(self) -> None:
        self._store: dict[str, list[tuple[str, dict[str, Any]]]] = {}

    def add_memories(
        self,
        agent_id: str,
        memories: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self._store.setdefault(agent_id, [])
        metas = metadatas if metadatas is not None else [{} for _ in memories]
        for mem, meta in zip(memories, metas):
            self._store[agent_id].append((mem, meta))

    def retrieve_relevant(
        self,
        agent_id: str,
        query: str,
        limit: int = 3,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        entries = self._store.get(agent_id, [])
        results: list[str] = []
        for text, meta in entries:
            if metadata_filter:
                if not all(meta.get(k) == v for k, v in metadata_filter.items()):
                    continue
            results.append(text)
            if len(results) >= limit:
                break
        return results


class QuietEventLogger(IEventLogger):
    def log(self, event: SimulationEvent) -> None:
        pass


@pytest.mark.asyncio
async def test_autonomous_goal_generation_from_memory() -> None:
    """Prüft, ob ein hungriger Agent ohne Startziel autonom ein Ziel aus dem Vektorspeicher ableitet."""
    grid = WorldGrid(width=15, height=15)
    logger = QuietEventLogger()
    presenter = MagicMock(spec=IPresenter)

    pathfinder = MagicMock(spec=IPathfinder)
    pathfinder.find_path.side_effect = lambda start, target, _: [start, target]

    vector_store = LocalInMemoryVectorStore()
    cognition = InstructorCognitionAdapter()

    plan_decomp = PlanDecompositionService(
        cognition_provider=cognition,
        vector_memory_store=vector_store,
    )

    engine = SimulationEngine(
        grid=grid,
        pathfinder=pathfinder,
        presenter=presenter,
        logger=logger,
        cognition_provider=cognition,
        plan_decomposition_service=plan_decomp,
        vector_memory_store=vector_store,
    )

    agent = Agent(id="agent_alice", name="Alice", position=Position(2, 2))
    agent.needs["hunger"] = 0.85  # Akuter Hunger
    engine.register_agent(agent)

    # Erinnerung im Vektorspeicher hinterlegen
    vector_store.add_memories(
        agent_id=agent.id,
        memories=["Tag 1: Apfelbaum bei (8, 8) erfolgreich genutzt."],
        metadatas=[{"category": "resource"}],
    )

    # Vor dem Takt: Kein Ziel aktiv
    assert len(agent.goals) == 0

    await engine.process_tick()

    # Nach dem Takt: Autonome Plandekomposition hat Sub-Goals erzeugt
    assert len(agent.goals) >= 2
    active_goal = agent.active_goal
    assert active_goal is not None
    assert active_goal.name == "SubGoal: move_to"
    assert active_goal.target_position == Position(8, 8)
    assert agent.has_path is True
    assert agent.path[-1] == Position(8, 8)


@pytest.mark.asyncio
async def test_autonomous_exploration_when_no_memory() -> None:
    """Prüft, ob ein Agent ohne Gedächtnis autonom in den Erkundungsmodus wechselt."""
    grid = WorldGrid(width=10, height=10)
    logger = QuietEventLogger()
    presenter = MagicMock(spec=IPresenter)

    pathfinder = MagicMock(spec=IPathfinder)
    pathfinder.find_path.side_effect = lambda start, target, _: [start, target]

    vector_store = LocalInMemoryVectorStore()
    cognition = InstructorCognitionAdapter()

    engine = SimulationEngine(
        grid=grid,
        pathfinder=pathfinder,
        presenter=presenter,
        logger=logger,
        cognition_provider=cognition,
        vector_memory_store=vector_store,
    )

    agent = Agent(id="agent_bob", name="Bob", position=Position(1, 1))
    agent.needs["hunger"] = 0.90
    engine.register_agent(agent)

    await engine.process_tick()

    assert len(agent.goals) >= 1
    active_goal = agent.active_goal
    assert active_goal is not None
    assert "explore" in active_goal.name.lower()