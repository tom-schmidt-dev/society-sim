from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest
from src.application.services.dialogue_history import DialogueHistory
from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent import Agent
from src.domain.models.cognition import (
    BlockedResolution,
    DialogueResolution,
    EndDialogueAction,
    InspectAction,
    ProbeAction,
    RerouteAction,
    TalkAction,
)
from src.domain.models.goal import Goal
from src.domain.models.message import IncomingMessage
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.presenter import IPresenter
from src.domain.services.perception_service import PerceptionService
from src.infrastructure.pathfinding.astar import AStarPathfinder


def build_test_corridor_engine() -> tuple[SimulationEngine, WorldGrid, AsyncMock]:
    grid = WorldGrid(width=25, height=7)
    for x in range(25):
        for y in range(7):
            grid.remove_obstacle(Position(x, y))

    for x in range(1, 24):
        grid.set_obstacle(Position(x, 2))
        grid.set_obstacle(Position(x, 4))

    # Nische bei (12, 2)
    grid.remove_obstacle(Position(12, 2))
    grid.set_obstacle(Position(11, 1))
    grid.set_obstacle(Position(12, 1))
    grid.set_obstacle(Position(13, 1))

    cognition = AsyncMock(spec=ICognitionProvider)
    engine = SimulationEngine(
        grid=grid,
        pathfinder=AStarPathfinder(),
        presenter=MagicMock(spec=IPresenter),
        logger=MagicMock(spec=IEventLogger),
        cognition_provider=cognition,
        tick_interval=0.001,
        perception_service=PerceptionService(default_radius=3),
        dialogue_history=DialogueHistory(),
    )
    return engine, grid, cognition


@pytest.mark.asyncio
async def test_corridor_substep1_approach_to_blockage() -> None:
    # Schritt 1: Annäherung bis zur Blockade an x=12 und x=13
    engine, _, _ = build_test_corridor_engine()

    alice = Agent(id="1", name="Alice", position=Position(2, 3))
    bob = Agent(id="2", name="Bob", position=Position(22, 3))
    engine.register_agent(alice)
    engine.register_agent(bob)

    engine.set_agent_target("1", target=Position(22, 3), destination_name="Ost-Tor")
    engine.set_agent_target("2", target=Position(2, 3), destination_name="West-Tor")

    for _ in range(10):
        await engine.process_tick()

    assert alice.position == Position(12, 3)
    assert bob.position == Position(13, 3)


@pytest.mark.asyncio
async def test_corridor_substep2_alice_evades_into_niche() -> None:
    # Schritt 2: Alice weicht nach Ansprache in Nische (12, 2) aus
    engine, _, cognition = build_test_corridor_engine()

    alice = Agent(id="1", name="Alice", position=Position(12, 3))
    bob = Agent(id="2", name="Bob", position=Position(13, 3))
    engine.register_agent(alice)
    engine.register_agent(bob)

    engine.set_agent_target("1", target=Position(22, 3), destination_name="Ost-Tor")
    engine.set_agent_target("2", target=Position(2, 3), destination_name="West-Tor")

    cognition.resolve_blockage.return_value = BlockedResolution(
        thought="Blockade. Bitte ausweichen.",
        action=TalkAction(
            target_agent_id="1",
            message="Bitte Platz machen.",
            reason="Weg frei machen",
            intent="request_yield",
        ),
    )
    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Ich weiche nach Norden aus.",
        action=EndDialogueAction(reason="Zustimmung", final_message="Ich weiche aus."),
        negotiation_intent="accept",
    )

    for _ in range(5):
        await engine.process_tick()
        await asyncio.sleep(0.001)

    assert alice.position == Position(12, 2)
    assert alice.active_goal is not None
    assert alice.active_goal.name == "Nischen-Halt"


@pytest.mark.asyncio
async def test_corridor_substep3_clearance_and_resumption() -> None:
    # Schritt 3: Bob passiert Nische; Alice empfängt Danke-Signal und verlässt Nische
    engine, _, _ = build_test_corridor_engine()

    alice = Agent(id="1", name="Alice", position=Position(12, 2))
    bob = Agent(id="2", name="Bob", position=Position(12, 3))
    engine.register_agent(alice)
    engine.register_agent(bob)

    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(22, 3)))
    alice.push_goal(
        Goal(
            name="Nischen-Halt",
            holds_position=True,
            is_evasion_hold=True,
            junction_position=Position(12, 3),
            yield_for_agent_id="2",
        )
    )

    bob.push_goal(Goal(name="West-Tor", target_position=Position(2, 3)))
    bob.assign_path([Position(11, 3), Position(10, 3), Position(9, 3)])

    # Bob passiert Chokepoint bis L1 >= 2 (Position 10, 3)
    await engine.process_tick()  # Bob -> (11, 3)
    await engine.process_tick()  # Bob -> (10, 3), sendet is_courtesy an Alice

    # Takt für Alices Reaktivierung
    await engine.process_tick()

    assert alice.active_goal.name == "Ost-Tor"
    assert alice.has_path is True
    assert alice.path[0] == Position(12, 3)


@pytest.mark.asyncio
async def test_e2e_corridor_scenario_handshake_and_completion() -> None:
    # TC-E2E-01: Vollständiger Lebenszyklus von Start bis Ziel
    engine, _, cognition = build_test_corridor_engine()

    alice = Agent(id="1", name="Alice", position=Position(2, 3))
    bob = Agent(id="2", name="Bob", position=Position(22, 3))
    engine.register_agent(alice)
    engine.register_agent(bob)

    target_alice = Position(22, 3)
    target_bob = Position(2, 3)
    engine.set_agent_target("1", target=target_alice, destination_name="Ost-Tor")
    engine.set_agent_target("2", target=target_bob, destination_name="West-Tor")

    cognition.resolve_blockage.return_value = BlockedResolution(
        thought="Blockade. Ich bitte um Ausweichen.",
        action=TalkAction(
            target_agent_id="1",
            message="Bitte ausweichen.",
            reason="Weg frei machen",
            intent="request_yield",
        ),
    )
    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Ich weiche in Nische aus.",
        action=EndDialogueAction(reason="Zustimmung", final_message="Ich weiche aus."),
        negotiation_intent="accept",
    )

    max_ticks = 140
    for tick_no in range(max_ticks):
        if alice.position == target_alice and bob.position == target_bob:
            break
        await engine.process_tick()
        await asyncio.sleep(0.001)

    diag = (
        f"Tick: {engine.current_tick} | "
        f"Alice Pos: {alice.position}, Goals: {[g.name for g in alice.goals]}, Path: {alice.path}, Busy: {alice.is_busy} | "
        f"Bob Pos: {bob.position}, Goals: {[g.name for g in bob.goals]}, Path: {bob.path}, Busy: {bob.is_busy}"
    )
    assert alice.position == target_alice, diag


@pytest.mark.asyncio
async def test_e2e_labyrinth_scenario_inspection_probing_and_reroute() -> None:
    # TC-E2E-02: Labyrinth-Szenario: Inspektion, Erprobung und Umgehung eines Felsblocks
    grid = WorldGrid(width=16, height=7)
    for x in range(16):
        for y in range(7):
            grid.remove_obstacle(Position(x, y))

    for x in range(1, 15):
        if x not in (5, 9):
            grid.set_obstacle(Position(x, 2))
        grid.set_obstacle(Position(x, 4))

    for x in range(4, 11):
        grid.set_obstacle(Position(x, 0))

    cognition = AsyncMock(spec=ICognitionProvider)
    engine = SimulationEngine(
        grid=grid,
        pathfinder=AStarPathfinder(),
        presenter=MagicMock(spec=IPresenter),
        logger=MagicMock(spec=IEventLogger),
        cognition_provider=cognition,
        tick_interval=0.001,
        perception_service=PerceptionService(default_radius=3),
        dialogue_history=DialogueHistory(),
    )

    alice = Agent(id="1", name="Alice", position=Position(2, 3))
    engine.register_agent(alice)

    rock = WorldEntity(
        id="stone_1",
        name="Großer Stein",
        position=Position(7, 3),
        entity_type="rock",
        is_conversational=False,
        is_passable=False,
    )
    engine.register_entity(rock)

    target_alice = Position(13, 3)
    engine.set_agent_target("1", target=target_alice, destination_name="Ost-Tor")

    step_counter = 0

    async def mock_resolve_blockage(context):
        nonlocal step_counter
        step_counter += 1
        if step_counter == 1:
            return BlockedResolution(
                thought="Unbekanntes Objekt, ich inspiziere es.",
                action=InspectAction(target_agent_id="stone_1", reason="Erstkontakt"),
            )
        elif step_counter == 2:
            return BlockedResolution(
                thought="Objekt inspiziert, ich erprobe Passierbarkeit.",
                action=ProbeAction(target_agent_id="stone_1", reason="Tasttest"),
            )
        else:
            return BlockedResolution(
                thought="Objekt unpassierbar. Ich plane den Weg neu.",
                action=RerouteAction(reason="Umgehen"),
            )

    cognition.resolve_blockage.side_effect = mock_resolve_blockage

    max_ticks = 60
    for _ in range(max_ticks):
        if alice.position == target_alice:
            break
        await engine.process_tick()
        await asyncio.sleep(0.001)

    assert alice.position == target_alice
    assert alice.memory.is_inspected("stone_1") is True
    assert alice.memory.get_entity_walkability("stone_1") is False
    assert alice.active_goal.name == "Ost-Tor"