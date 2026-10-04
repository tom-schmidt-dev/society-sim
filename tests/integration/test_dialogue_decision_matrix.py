from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock
import pytest

from src.application.services.coordination.dialogue_history import DialogueHistory
from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_state import AgentLifecycleState
from src.domain.models.planning.cognition import (
    AbortAction,
    BlockedResolution,
    DialogueResolution,
    EndDialogueAction,
    GoalIntent,
    RerouteAction,
    TalkAction,
    WaitAction,
)
from src.domain.models.world.position import Position
from src.domain.models.world.world import WorldGrid
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.presenter import IPresenter
from src.domain.services.perception_service import PerceptionService
from src.infrastructure.pathfinding.astar import AStarPathfinder


def build_matrix_corridor_engine(allow_loop: bool = False) -> tuple[SimulationEngine, WorldGrid, AsyncMock]:
    """Erzeugt eine Testumgebung mit Korridor, Nische und optionalem Umgehungspfad."""
    grid = WorldGrid(width=25, height=9)
    for x in range(25):
        for y in range(9):
            grid.remove_obstacle(Position(x, y))

    # Hauptkorridor auf y=4 begrenzen
    for x in range(1, 24):
        grid.set_obstacle(Position(x, 3))
        grid.set_obstacle(Position(x, 5))

    # Nische bei (12, 3)
    grid.remove_obstacle(Position(12, 3))
    grid.set_obstacle(Position(11, 2))
    grid.set_obstacle(Position(12, 2))
    grid.set_obstacle(Position(13, 2))

    # Optionaler oberer Umgehungspfad für Reroute-Szenarien
    if allow_loop:
        for x in range(4, 21):
            grid.remove_obstacle(Position(x, 1))
        grid.remove_obstacle(Position(4, 2))
        grid.remove_obstacle(Position(4, 3))
        grid.remove_obstacle(Position(20, 2))
        grid.remove_obstacle(Position(20, 3))

    cognition = AsyncMock(spec=ICognitionProvider)
    engine = SimulationEngine(
        grid=grid,
        pathfinder=AStarPathfinder(),
        presenter=MagicMock(spec=IPresenter),
        logger=MagicMock(spec=IEventLogger),
        cognition_provider=cognition,
        tick_interval=0.001,
        perception_service=PerceptionService(default_radius=4),
        dialogue_history=DialogueHistory(),
        enable_deterministic_corridor=False,
        enable_day_night=False,
    )
    return engine, grid, cognition


@pytest.mark.asyncio
async def test_matrix_case1_cooperation_immediate_yield() -> None:
    """Fall 1: Bob bittet Alice um Ausweichen; Alice kooperiert sofort und weicht in Nische aus."""
    engine, _, cognition = build_matrix_corridor_engine()

    bob = Agent(id="1", name="Bob", position=Position(13, 4))
    alice = Agent(id="2", name="Alice", position=Position(12, 4))
    engine.register_agent(bob)
    engine.register_agent(alice)

    engine.set_agent_target("1", target=Position(2, 4), destination_name="West-Tor")
    engine.set_agent_target("2", target=Position(22, 4), destination_name="Ost-Tor")

    cognition.resolve_blockage.return_value = BlockedResolution(
        thought="Weg blockiert. Ich fordere Alice zum Ausweichen auf.",
        action=TalkAction(
            target_agent_id="2",
            message="Bitte in die Nische ausweichen.",
            intent="request_yield",
        ),
    )
    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Ich mache Platz in der Nische.",
        action=TalkAction(
            target_agent_id="1",
            message="Ich weiche nach Norden aus.",
            intent="offer_yield",
        ),
        negotiation_intent="offer_yield",
        new_goal=GoalIntent(name="In Nische ausweichen", intent_type="evade"),
    )

    for _ in range(6):
        await engine.process_tick()
        await asyncio.sleep(0.001)

    assert alice.position == Position(12, 3)
    active_goal = alice.active_goal
    assert active_goal is not None
    assert active_goal.name == "Nischen-Halt"


@pytest.mark.asyncio
async def test_matrix_case2_rejection_then_reroute() -> None:
    """Fall 2: Bob bittet um Ausweichen; Alice lehnt ab (reject); Bob wählt Reroute-Umweg."""
    engine, _, cognition = build_matrix_corridor_engine(allow_loop=True)

    bob = Agent(id="1", name="Bob", position=Position(13, 4))
    alice = Agent(id="2", name="Alice", position=Position(12, 4))
    engine.register_agent(bob)
    engine.register_agent(alice)

    engine.set_agent_target("1", target=Position(2, 4), destination_name="West-Tor")
    engine.set_agent_target("2", target=Position(22, 4), destination_name="Ost-Tor")

    step = 0

    async def mock_resolve_blockage(context: dict[str, Any]) -> BlockedResolution:
        nonlocal step
        step += 1
        if step == 1:
            return BlockedResolution(
                thought="Ich bitte um Durchgang.",
                action=TalkAction(target_agent_id="2", message="Ausweichen bitte.", intent="request_yield"),
            )
        return BlockedResolution(
            thought="Partner verweigert Ausweichen. Ich wähle den Umweg.",
            action=RerouteAction(reason="Umgehen via Nordpfad"),
        )

    cognition.resolve_blockage.side_effect = mock_resolve_blockage
    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Ich habe es eilig und weiche nicht aus.",
        action=TalkAction(target_agent_id="1", message="Nein, ich bleibe hier.", intent="reject"),
        negotiation_intent="reject",
    )

    await engine.process_tick()
    await engine.process_tick()
    await engine.process_tick()

    assert bob.has_path is True
    assert Position(12, 4) not in bob.path


@pytest.mark.asyncio
async def test_matrix_case3_rejection_then_wait() -> None:
    """Fall 3: Bob bittet um Ausweichen; Alice lehnt ab (reject); Bob entscheidet sich zu warten."""
    engine, _, cognition = build_matrix_corridor_engine()

    bob = Agent(id="1", name="Bob", position=Position(13, 4))
    alice = Agent(id="2", name="Alice", position=Position(12, 4))
    engine.register_agent(bob)
    engine.register_agent(alice)

    engine.set_agent_target("1", target=Position(2, 4), destination_name="West-Tor")
    engine.set_agent_target("2", target=Position(22, 4), destination_name="Ost-Tor")

    step = 0

    async def mock_resolve_blockage(context: dict[str, Any]) -> BlockedResolution:
        nonlocal step
        step += 1
        if step == 1:
            return BlockedResolution(
                thought="Bitte um Durchgang.",
                action=TalkAction(target_agent_id="2", message="Platz da.", intent="request_yield"),
            )
        return BlockedResolution(
            thought="Ablehnung erhalten. Ich warte geduldig ab.",
            action=WaitAction(ticks=5, reason="Warte auf spätere Klärung"),
        )

    cognition.resolve_blockage.side_effect = mock_resolve_blockage
    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Ich weiche nicht.",
        action=TalkAction(target_agent_id="1", message="Nein.", intent="reject"),
        negotiation_intent="reject",
    )

    await engine.process_tick()
    await engine.process_tick()
    await engine.process_tick()

    active_goal = bob.active_goal
    assert active_goal is not None
    assert "Warten" in active_goal.name
    assert bob.position == Position(13, 4)


@pytest.mark.asyncio
async def test_matrix_case4_rejection_then_self_evade() -> None:
    """Fall 4: Bob bittet um Ausweichen; Alice lehnt ab; Bob weicht daraufhin selbst in die Nische aus."""
    engine, _, cognition = build_matrix_corridor_engine()

    # Bob steht direkt an der Nische (12, 4); Alice blockiert von Westen (11, 4)
    bob = Agent(id="1", name="Bob", position=Position(12, 4))
    alice = Agent(id="2", name="Alice", position=Position(11, 4))
    engine.register_agent(bob)
    engine.register_agent(alice)

    engine.set_agent_target("1", target=Position(2, 4), destination_name="West-Tor")
    engine.set_agent_target("2", target=Position(22, 4), destination_name="Ost-Tor")

    step = 0

    async def mock_resolve_blockage(context: dict[str, Any]) -> BlockedResolution:
        nonlocal step
        step += 1
        if step == 1:
            return BlockedResolution(
                thought="Ausweichbitte stellen.",
                action=TalkAction(target_agent_id="2", message="Bitte weichen.", intent="request_yield"),
            )
        return BlockedResolution(
            thought="Partner weigert sich. Ich mache selbst Platz.",
            action=TalkAction(target_agent_id="2", message="Gut, ich weiche aus.", intent="offer_yield"),
            new_sub_goal="In Nische ausweichen",
        )

    cognition.resolve_blockage.side_effect = mock_resolve_blockage
    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Ich weiche nicht aus.",
        action=TalkAction(target_agent_id="1", message="Ich bleibe.", intent="reject"),
        negotiation_intent="reject",
    )

    reached_niche = False
    for _ in range(8):
        await engine.process_tick()
        await asyncio.sleep(0.001)
        if bob.position == Position(12, 3):
            reached_niche = True
            break

    assert reached_niche is True
    active_goal = bob.active_goal
    assert active_goal is not None
    assert active_goal.name in ("Nischen-Halt", "In Nische ausweichen")


@pytest.mark.asyncio
async def test_matrix_case5_rejection_then_end_dialogue() -> None:
    """Fall 5: Bob bittet um Ausweichen; Alice lehnt ab; Bob bricht die Zielverfolgung ab."""
    engine, _, cognition = build_matrix_corridor_engine()

    bob = Agent(id="1", name="Bob", position=Position(13, 4))
    alice = Agent(id="2", name="Alice", position=Position(12, 4))
    engine.register_agent(bob)
    engine.register_agent(alice)

    engine.set_agent_target("1", target=Position(2, 4), destination_name="West-Tor")
    engine.set_agent_target("2", target=Position(22, 4), destination_name="Ost-Tor")

    step = 0

    async def mock_resolve_blockage(context: dict[str, Any]) -> BlockedResolution:
        nonlocal step
        step += 1
        if step == 1:
            return BlockedResolution(
                thought="Ich fordere Ausweichen.",
                action=TalkAction(target_agent_id="2", message="Bitte Platz machen.", intent="request_yield"),
            )
        return BlockedResolution(
            thought="Gespräch ist zwecklos. Ich breche ab.",
            action=AbortAction(reason="Uneinigkeit"),
        )

    cognition.resolve_blockage.side_effect = mock_resolve_blockage
    cognition.respond_to_dialogue.return_value = DialogueResolution(
        thought="Kein Interesse an Kooperation.",
        action=TalkAction(target_agent_id="1", message="Nein.", intent="reject"),
        negotiation_intent="reject",
    )

    await engine.process_tick()
    await engine.process_tick()
    await engine.process_tick()

    assert bob.lifecycle_state == AgentLifecycleState.IDLE
    assert alice.lifecycle_state == AgentLifecycleState.IDLE
    assert bob.interaction_partner_id is None
    assert alice.interaction_partner_id is None

@pytest.mark.asyncio
async def test_matrix_case6_multi_turn_negotiation_success() -> None:
    """Fall 6: Alice hinterfragt initial (negotiate), lenkt in Runde 2 nach erneutem Appell ein."""
    engine, _, cognition = build_matrix_corridor_engine()

    bob = Agent(id="1", name="Bob", position=Position(13, 4))
    alice = Agent(id="2", name="Alice", position=Position(12, 4))
    engine.register_agent(bob)
    engine.register_agent(alice)

    engine.set_agent_target("1", target=Position(2, 4), destination_name="West-Tor")
    engine.set_agent_target("2", target=Position(22, 4), destination_name="Ost-Tor")

    alice_turn = 0

    async def mock_respond_to_dialogue(context: dict[str, Any]) -> DialogueResolution:
        nonlocal alice_turn
        agent_id = str(context.get("agent_id"))
        if agent_id == "2":  # Alice
            alice_turn += 1
            if alice_turn == 1:
                return DialogueResolution(
                    thought="Ich hinterfrage das Anliegen.",
                    action=TalkAction(target_agent_id="1", message="Warum sollte ich?", intent="negotiate"),
                    negotiation_intent=None,
                )
            return DialogueResolution(
                thought="Die Begründung überzeugt mich. Ich weiche aus.",
                action=TalkAction(target_agent_id="1", message="In Ordnung, ich weiche aus.", intent="offer_yield"),
                negotiation_intent="offer_yield",
                new_goal=GoalIntent(name="In Nische ausweichen", intent_type="evade"),
            )
        else:  # Bob
            return DialogueResolution(
                thought="Ich bekräftige die Dringlichkeit.",
                action=TalkAction(target_agent_id="2", message="Mein Weg ist wesentlich weiter.", intent="request_yield"),
                negotiation_intent="request_yield",
            )

    cognition.resolve_blockage.return_value = BlockedResolution(
        thought="Blockade. Bitte ausweichen.",
        action=TalkAction(target_agent_id="2", message="Bitte Platz machen.", intent="request_yield"),
    )
    cognition.respond_to_dialogue.side_effect = mock_respond_to_dialogue

    reached_niche = False
    for _ in range(12):
        await engine.process_tick()
        await asyncio.sleep(0.001)
        if alice.position == Position(12, 3):
            reached_niche = True
            break

    assert reached_niche is True
    active_goal = alice.active_goal
    assert active_goal is not None
    assert active_goal.name in ("Nischen-Halt", "In Nische ausweichen")