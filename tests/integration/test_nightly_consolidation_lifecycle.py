from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.lifecycle.daily_event_buffer import DailyEventBuffer
from src.application.services.lifecycle.day_night_service import DayNightService
from src.application.services.lifecycle.memory_consolidation_service import MemoryConsolidationService
from src.application.services.movement.movement_orchestrator import MovementOrchestrator
from src.application.services.cognition.cognition_orchestrator import CognitionOrchestrator
from src.application.services.coordination.agent_protocol_service import AgentProtocolService
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.events import SimulationEvent
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter
from src.domain.ports.vector_memory_store import IVectorMemoryStore
from src.domain.services.perception_service import PerceptionService
from src.application.simulation_engine import SimulationEngine


# ==============================================================================
# Test Doubles & In-Memory Fakes
# ==============================================================================


class InMemoryVectorMemoryStore(IVectorMemoryStore):
    """Konkreter In-Memory-Vektorspeicher für isolierte Integrationstests."""

    def __init__(self) -> None:
        self.stored_memories: dict[str, list[str]] = {}
        self.stored_metadatas: dict[str, list[dict[str, Any]]] = {}

    def add_memories(
        self,
        agent_id: str,
        memories: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.stored_memories.setdefault(agent_id, []).extend(memories)
        if metadatas:
            self.stored_metadatas.setdefault(agent_id, []).extend(metadatas)

    def retrieve_relevant(
        self,
        agent_id: str,
        query: str,
        limit: int = 3,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        return self.stored_memories.get(agent_id, [])[:limit]


class InMemoryDailyEventBuffer(DailyEventBuffer):
    """In-Memory Puffer für deterministische Tagesereignisse in Tests."""

    def __init__(self) -> None:
        try:
            super().__init__()
        except Exception:
            pass
        self._events: dict[str, list[SimulationEvent]] = {}

    def record_event(self, event: SimulationEvent) -> None:
        """Erfasst ein Ereignis für den im Event hinterlegten Agenten."""
        self._events.setdefault(event.agent_id, []).append(event)

    def get_events_for_agent(self, agent_id: str) -> list[SimulationEvent]:
        """Gibt alle gepufferten Ereignisse für den Agenten zurück."""
        return list(self._events.get(agent_id, []))

    def clear_agent(self, agent_id: str) -> None:
        """Leert den Ereignispuffer des Agenten nach erfolgreicher Konsolidierung."""
        self._events.pop(agent_id, None)


class RecordingEventLogger(IEventLogger):
    """Protokolliert emittierte Simulationsereignisse zur Testüberprüfung."""

    def __init__(self) -> None:
        self.events: list[SimulationEvent] = []

    def log(self, event: SimulationEvent) -> None:
        self.events.append(event)


# ==============================================================================
# Hilfsfunktionen zur Instanziierung
# ==============================================================================


def create_test_agent(
    agent_id: str = "agent_alpha",
    name: str = "Alpha",
    position: Optional[Position] = None,
) -> Agent:
    """Erzeugt eine Test-Agenteninstanz mit robuster Fallback-Behandlung."""
    pos = position or Position(1, 1)
    try:
        agent = Agent(id=agent_id, name=name, position=pos)
    except TypeError:
        agent = MagicMock(spec=Agent)
        agent.id = agent_id
        agent.name = name
        agent.position = pos
        agent.is_sleeping = False
        agent.is_busy = False
        agent.is_thinking = False
        agent.has_path = False
        agent.inbox = []
        agent.active_goal = None
        agent.mental_map = MagicMock()
        agent.memory = MagicMock()
        agent.commit_staging_messages = MagicMock()

    return agent


def create_test_grid(width: int = 10, height: int = 10) -> WorldGrid:
    """Erzeugt ein passierbares Standardgitter für Tests."""
    try:
        return WorldGrid(width=width, height=height)
    except TypeError:
        grid = MagicMock(spec=WorldGrid)
        grid.width = width
        grid.height = height
        grid.is_walkable.return_value = True
        grid.get_tile_type.return_value = "floor"
        return grid


def setup_simulation_environment(
    day_ticks: int = 3,
    night_ticks: int = 2,
    with_consolidation: bool = True,
) -> tuple[
    SimulationEngine,
    InMemoryDailyEventBuffer,
    InMemoryVectorMemoryStore,
    RecordingEventLogger,
    MagicMock,
]:
    """Konfiguriert die Simulationsengine mit isolierten Ports und Spies."""
    grid = create_test_grid()
    pathfinder = MagicMock(spec=IPathfinder)
    presenter = MagicMock(spec=IPresenter)
    logger = RecordingEventLogger()
    cognition_provider = MagicMock(spec=ICognitionProvider)

    day_night_service = DayNightService(
        day_ticks=day_ticks,
        night_ticks=night_ticks,
        logger=logger,
    )

    event_buffer = InMemoryDailyEventBuffer()
    vector_store = InMemoryVectorMemoryStore()

    consolidation_service: Optional[MemoryConsolidationService] = None
    if with_consolidation:
        consolidation_service = MemoryConsolidationService(
            vector_store=vector_store,
            event_buffer=event_buffer,
            logger=logger,
        )

    movement_orchestrator = MagicMock(spec=MovementOrchestrator)
    cognition_orchestrator = MagicMock(spec=CognitionOrchestrator)
    cognition_orchestrator.process_agent_needs_and_cognition = AsyncMock()
    cognition_orchestrator.latest_snapshots = []

    protocol_service = MagicMock(spec=AgentProtocolService)

    perception_service = MagicMock(spec=PerceptionService)
    perception_service.get_visible_positions.return_value = []
    perception_service.get_visible_entities.return_value = []

    engine = SimulationEngine(
        grid=grid,
        pathfinder=pathfinder,
        presenter=presenter,
        logger=logger,
        cognition_provider=cognition_provider,
        day_night_service=day_night_service,
        memory_consolidation_service=consolidation_service,
        movement_orchestrator=movement_orchestrator,
        cognition_orchestrator=cognition_orchestrator,
        protocol_service=protocol_service,
        perception_service=perception_service,
    )

    return engine, event_buffer, vector_store, logger, movement_orchestrator


# ==============================================================================
# Integrationstests: Nächtlicher Konsolidierungs-Lebenszyklus
# ==============================================================================


@pytest.mark.asyncio
async def test_night_transition_triggers_sleep_and_consolidation() -> None:
    """Validiert, dass bei Erreichen von day_ticks der Schlafzustand gesetzt und konsolidiert wird."""
    engine, event_buffer, vector_store, logger, movement_orchestrator = (
        setup_simulation_environment(day_ticks=3, night_ticks=2)
    )
    agent = create_test_agent("agent_1", "Alpha")
    engine.register_agent(agent)

    event_buffer.record_event(
        SimulationEvent(
            tick=1,
            agent_id="agent_1",
            event_type="resource_consumed",
            summary="Apfel gegessen",
            payload={"resource_type": "Apfelbaum", "position": (3, 4)},
        )
    )

    # Takt 1 (Tag)
    await engine.process_tick()
    assert agent.is_sleeping is False
    assert movement_orchestrator.execute_physical_movement.call_count == 1
    assert "agent_1" not in vector_store.stored_memories

    # Takt 2 (Tag)
    await engine.process_tick()
    assert agent.is_sleeping is False
    assert movement_orchestrator.execute_physical_movement.call_count == 2
    assert "agent_1" not in vector_store.stored_memories

    # Takt 3 (Nacht bricht an: phase_tick == day_ticks)
    await engine.process_tick()
    assert agent.is_sleeping is True

    # Phase 2 (Bewegung) muss in der Nacht pausieren
    assert movement_orchestrator.execute_physical_movement.call_count == 2

    # Vektorspeicher muss konsolidierte Erinnerung enthalten
    assert "agent_1" in vector_store.stored_memories
    memories = vector_store.stored_memories["agent_1"]
    assert len(memories) == 1
    assert "Tag 1: Apfelbaum bei (3, 4) erfolgreich genutzt." in memories[0]

    # Puffer muss geleert sein
    assert event_buffer.get_events_for_agent("agent_1") == []

    # Logger muss 'night_started' und 'memory_consolidated' verzeichnet haben
    event_types = [e.event_type for e in logger.events]
    assert "night_started" in event_types
    assert "memory_consolidated" in event_types


@pytest.mark.asyncio
async def test_consolidation_synthesizes_diverse_event_categories() -> None:
    """Prüft die korrekte semantische Synthese und Metadaten-Kategorisierung mehrerer Ereignisse."""
    engine, event_buffer, vector_store, _, _ = setup_simulation_environment(
        day_ticks=2, night_ticks=2
    )
    agent = create_test_agent("agent_1", "Alpha")
    engine.register_agent(agent)

    # Verschiedene Ereignistypen im Puffer registrieren
    event_buffer.record_event(
        SimulationEvent(
            tick=1,
            agent_id="agent_1",
            event_type="resource_consumed",
            summary="Wasser konsumiert",
            payload={"resource_type": "Wasserquelle", "position": (2, 2)},
        )
    )
    event_buffer.record_event(
        SimulationEvent(
            tick=1,
            agent_id="agent_1",
            event_type="dialogue_resolved",
            summary="Dialog abgeschlossen",
            payload={"partner_id": "agent_2", "outcome": "cooperative"},
        )
    )
    event_buffer.record_event(
        SimulationEvent(
            tick=2,
            agent_id="agent_1",
            event_type="evasion_hold_started",
            summary="Ausweichhalt begonnen",
            payload={"yield_for_agent_id": "agent_2"},
        )
    )
    event_buffer.record_event(
        SimulationEvent(
            tick=2,
            agent_id="agent_1",
            event_type="cognitive_snapshot",
            summary="Kognitiver Zustand erfasst",
            payload={"perceived_obstacle": "Engpass_Nord", "intended_strategy": "Ausweichen"},
        )
    )

    # Takt 1 (Tag)
    await engine.process_tick()
    # Takt 2 (Nachteinbruch)
    await engine.process_tick()

    memories = vector_store.stored_memories.get("agent_1", [])
    metadatas = vector_store.stored_metadatas.get("agent_1", [])
    assert len(memories) == 4
    assert len(metadatas) == 4

    categories = [meta["category"] for meta in metadatas]
    assert "resource" in categories
    assert "social" in categories
    assert "conflict" in categories

    # Überprüfung der Textrepräsentationen
    joined_memories = " ".join(memories)
    assert "Wasserquelle bei (2, 2) erfolgreich genutzt." in joined_memories
    assert "Begegnung mit agent_2 verlief cooperative." in joined_memories
    assert "Vorfahrt für agent_2 gewährt." in joined_memories
    assert "Fortschritt behindert durch Engpass_Nord." in joined_memories


@pytest.mark.asyncio
async def test_multi_agent_memory_isolation() -> None:
    """Stellt sicher, dass Erinnerungen strikt agentenspezifisch konsolidiert werden."""
    engine, event_buffer, vector_store, _, _ = setup_simulation_environment(
        day_ticks=2, night_ticks=2
    )
    agent_1 = create_test_agent("agent_1", "Alpha")
    agent_2 = create_test_agent("agent_2", "Beta")
    engine.register_agent(agent_1)
    engine.register_agent(agent_2)

    # Nur Agent 1 hat Ereignisse
    event_buffer.record_event(
        SimulationEvent(
            tick=1,
            agent_id="agent_1",
            event_type="resource_consumed",
            summary="Beeren konsumiert",
            payload={"resource_type": "Beerenstrauch"},
        )
    )

    # Tag 1 ablaufen lassen bis Nachtstart
    await engine.process_tick()  # Takt 1
    await engine.process_tick()  # Takt 2 (Nachtstart)

    # Beide Agenten müssen schlafen
    assert agent_1.is_sleeping is True
    assert agent_2.is_sleeping is True

    # Nur Agent 1 hat Einträge im Vektorspeicher
    assert "agent_1" in vector_store.stored_memories
    assert "agent_2" not in vector_store.stored_memories
    assert len(vector_store.stored_memories["agent_1"]) == 1


@pytest.mark.asyncio
async def test_movement_suppression_and_no_duplicate_consolidation_during_night() -> None:
    """Prüft, dass während der gesamten Nacht Bewegung pausiert und keine Re-Konsolidierung erfolgt."""
    engine, event_buffer, vector_store, _, movement_orchestrator = (
        setup_simulation_environment(day_ticks=2, night_ticks=3)
    )
    agent = create_test_agent("agent_1", "Alpha")
    engine.register_agent(agent)

    event_buffer.record_event(
        SimulationEvent(
            tick=1,
            agent_id="agent_1",
            event_type="path_assigned",
            summary="Pfad zu Kachel (5, 5) zugewiesen.",
        )
    )

    await engine.process_tick()  # Takt 1 (Tag): Bewegung läuft
    assert movement_orchestrator.execute_physical_movement.call_count == 1

    await engine.process_tick()  # Takt 2 (Nachtstart): Konsolidierung, Bewegung stoppt
    assert movement_orchestrator.execute_physical_movement.call_count == 1
    assert len(vector_store.stored_memories["agent_1"]) == 1

    await engine.process_tick()  # Takt 3 (Nacht): Bewegung weiterhin blockiert
    assert movement_orchestrator.execute_physical_movement.call_count == 1
    # Keine erneute Konsolidierung
    assert len(vector_store.stored_memories["agent_1"]) == 1

    await engine.process_tick()  # Takt 4 (Nacht): Bewegung blockiert
    assert movement_orchestrator.execute_physical_movement.call_count == 1


@pytest.mark.asyncio
async def test_night_to_day_transition_wakes_agents_and_resumes_movement() -> None:
    """Validiert den Aufwachprozess bei Takt 0 (Modulolänge) und das Einsetzen von Phase 2."""
    # Zyklus: 2 Takte Tag, 2 Takte Nacht -> Zykluslänge = 4
    engine, _, _, logger, movement_orchestrator = setup_simulation_environment(
        day_ticks=2, night_ticks=2
    )
    agent = create_test_agent("agent_1", "Alpha")
    engine.register_agent(agent)

    # Takt 1: Tag (Wach)
    await engine.process_tick()
    assert agent.is_sleeping is False

    # Takt 2: Nachtstart (Schlafend)
    await engine.process_tick()
    assert agent.is_sleeping is True

    # Takt 3: Nacht (Schlafend)
    await engine.process_tick()
    assert agent.is_sleeping is True

    # Takt 4: phase_tick == 0 -> Tagesbeginn Tag 2
    await engine.process_tick()
    assert agent.is_sleeping is False
    assert any(e.event_type == "day_started" for e in logger.events)

    # Takt 5: Regulärer Tagtakt -> Physische Bewegung wird ausgeführt
    movement_calls_before = movement_orchestrator.execute_physical_movement.call_count
    await engine.process_tick()
    assert movement_orchestrator.execute_physical_movement.call_count == movement_calls_before + 1


@pytest.mark.asyncio
async def test_multi_day_cycles_increment_day_number() -> None:
    """Überprüft die korrekte Inkrementierung von day_number über mehrere Tag-Nacht-Zyklen."""
    # Zykluslänge = 2 + 1 = 3 (Takt 2 = Nacht Tag 1, Takt 5 = Nacht Tag 2)
    engine, event_buffer, vector_store, _, _ = setup_simulation_environment(
        day_ticks=2, night_ticks=1
    )
    agent = create_test_agent("agent_1", "Alpha")
    engine.register_agent(agent)

    # Tag 1 Event
    event_buffer.record_event(
        SimulationEvent(
            tick=1,
            agent_id="agent_1",
            event_type="resource_consumed",
            summary="Quelle Tag 1 genutzt",
            payload={"resource_type": "Quelle_Tag1"},
        )
    )

    await engine.process_tick()  # Takt 1 (Tag 1)
    await engine.process_tick()  # Takt 2 (Nacht Tag 1 -> Konsolidierung)
    await engine.process_tick()  # Takt 3 (Tag 2 startet, phase_tick == 0)

    # Tag 2 Event nach Aufwachen
    event_buffer.record_event(
        SimulationEvent(
            tick=4,
            agent_id="agent_1",
            event_type="resource_consumed",
            summary="Quelle Tag 2 genutzt",
            payload={"resource_type": "Quelle_Tag2"},
        )
    )

    await engine.process_tick()  # Takt 4 (Tag 2)
    await engine.process_tick()  # Takt 5 (Nacht Tag 2 -> Konsolidierung)

    memories = vector_store.stored_memories.get("agent_1", [])
    assert len(memories) == 2
    assert "Tag 1: Quelle_Tag1 erfolgreich genutzt." in memories[0]
    assert "Tag 2: Quelle_Tag2 erfolgreich genutzt." in memories[1]


@pytest.mark.asyncio
async def test_lifecycle_resilience_without_consolidation_service() -> None:
    """Stellt sicher, dass die Engine auch ohne MemoryConsolidationService stabil schaltet."""
    engine, _, _, logger, movement_orchestrator = setup_simulation_environment(
        day_ticks=2,
        night_ticks=2,
        with_consolidation=False,
    )
    agent = create_test_agent("agent_1", "Alpha")
    engine.register_agent(agent)

    # Zyklus durchlaufen (Tag -> Nacht -> Tag)
    await engine.process_tick()  # Takt 1 (Tag)
    assert agent.is_sleeping is False

    await engine.process_tick()  # Takt 2 (Nachtbeginn)
    assert agent.is_sleeping is True
    assert movement_orchestrator.execute_physical_movement.call_count == 1

    await engine.process_tick()  # Takt 3 (Nacht)
    assert agent.is_sleeping is True
    assert movement_orchestrator.execute_physical_movement.call_count == 1

    await engine.process_tick()  # Takt 4 (Tagesanbruch)
    assert agent.is_sleeping is False