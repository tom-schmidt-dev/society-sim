from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest
from src.application.services.dialogue_history import DialogueHistory
from src.application.simulation_engine import SimulationEngine
from src.domain.models.agent import Agent
from src.domain.models.goal import Goal
from src.domain.models.message import IncomingMessage
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.presenter import IPresenter
from src.domain.services.perception_service import PerceptionService
from src.infrastructure.pathfinding.astar import AStarPathfinder


@pytest.fixture
def engine_setup():
    grid = WorldGrid(width=30, height=30)
    for x in range(30):
        for y in range(30):
            grid.remove_obstacle(Position(x, y))

    pathfinder = AStarPathfinder()
    presenter = MagicMock(spec=IPresenter)
    logger = MagicMock(spec=IEventLogger)
    cognition = AsyncMock(spec=ICognitionProvider)
    dialogue_history = DialogueHistory()
    perception = PerceptionService(default_radius=3)

    engine = SimulationEngine(
        grid=grid,
        pathfinder=pathfinder,
        presenter=presenter,
        logger=logger,
        cognition_provider=cognition,
        tick_interval=0.01,
        perception_service=perception,
        dialogue_history=dialogue_history,
    )
    return engine, cognition, logger, dialogue_history


@pytest.mark.asyncio
async def test_signal_halt_request_pushes_wait_goal(engine_setup) -> None:
    # TC-SIG-01: Empfang von is_halt_request erzeugt Halteziel mit correlation_key
    engine, _, _, _ = engine_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    engine.register_agent(alice)
    engine.register_agent(bob)

    bob.push_goal(Goal(name="West-Tor", target_position=Position(1, 5)))

    bob.receive_message(
        IncomingMessage(
            from_agent_id="1",
            from_agent_name="Alice",
            message="HALT WARTE!",
            is_halt_request=True,
            correlation_key="niche-entry-1",
        )
    )

    await engine.process_tick()

    assert bob.active_goal is not None
    assert bob.active_goal.name == "Wartet auf Freigabe (Alice)"
    assert bob.active_goal.holds_position is True
    assert bob.active_goal.correlation_key == "niche-entry-1"
    assert bob.active_goal.yield_for_agent_id == "1"


@pytest.mark.asyncio
async def test_signal_halt_suppresses_movement(engine_setup) -> None:
    # TC-SIG-02: Bei aktivem Halteziel führt der Agent keinen Bewegungsschritt aus
    engine, _, _, _ = engine_setup

    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    engine.register_agent(bob)

    bob.push_goal(Goal(name="West-Tor", target_position=Position(1, 5)))
    bob.push_goal(
        Goal(
            name="Wartet auf Freigabe (Alice)",
            holds_position=True,
            correlation_key="niche-entry-1",
        )
    )
    bob.assign_path([Position(5, 5), Position(4, 5)])

    await engine.process_tick()

    assert bob.position == Position(6, 5)
    assert len(bob.path) == 2


@pytest.mark.asyncio
async def test_signal_resume_with_matching_key_pops_goal(engine_setup) -> None:
    # TC-SIG-03: Empfang von is_resume_signal mit passendem Key entfernt das Halteziel
    engine, _, _, _ = engine_setup

    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    engine.register_agent(bob)

    bob.push_goal(Goal(name="West-Tor", target_position=Position(1, 5)))
    bob.push_goal(
        Goal(
            name="Wartet auf Freigabe (Alice)",
            holds_position=True,
            correlation_key="niche-entry-1",
        )
    )

    bob.receive_message(
        IncomingMessage(
            from_agent_id="1",
            from_agent_name="Alice",
            message="Ok, weiter.",
            is_resume_signal=True,
            correlation_key="niche-entry-1",
        )
    )

    await engine.process_tick()

    assert bob.active_goal is not None
    assert bob.active_goal.name == "West-Tor"
    assert bob.active_goal.holds_position is False


@pytest.mark.asyncio
async def test_signal_resume_with_key_mismatch_retains_goal(engine_setup) -> None:
    # TC-SIG-04: Freigabesignal mit fremdem Key lässt das Halteziel unberührt
    engine, _, _, _ = engine_setup

    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    engine.register_agent(bob)

    bob.push_goal(Goal(name="West-Tor", target_position=Position(1, 5)))
    bob.push_goal(
        Goal(
            name="Wartet auf Freigabe (Alice)",
            holds_position=True,
            correlation_key="niche-entry-1",
        )
    )

    bob.receive_message(
        IncomingMessage(
            from_agent_id="1",
            from_agent_name="Alice",
            message="Ok, weiter.",
            is_resume_signal=True,
            correlation_key="wrong-key-999",
        )
    )

    await engine.process_tick()

    assert bob.active_goal.name == "Wartet auf Freigabe (Alice)"
    assert bob.active_goal.correlation_key == "niche-entry-1"


@pytest.mark.asyncio
async def test_signal_resume_without_key_removes_by_partner_id(engine_setup) -> None:
    # TC-SIG-05: Fallback-Freigabe ohne Key entfernt Ziel anhand von yield_for_agent_id
    engine, _, _, _ = engine_setup

    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    engine.register_agent(bob)

    bob.push_goal(Goal(name="West-Tor", target_position=Position(1, 5)))
    bob.push_goal(
        Goal(
            name="Wartet auf Freigabe",
            holds_position=True,
            yield_for_agent_id="1",
            correlation_key=None,
        )
    )

    bob.receive_message(
        IncomingMessage(
            from_agent_id="1",
            from_agent_name="Alice",
            message="Ok, weiter.",
            is_resume_signal=True,
            correlation_key=None,
        )
    )

    await engine.process_tick()

    assert bob.active_goal.name == "West-Tor"


@pytest.mark.asyncio
async def test_signal_path_update_halts_and_acknowledges(engine_setup) -> None:
    # TC-SIG-06: Agent A pausiert bei Pfadänderung des Partners und sendet Freigabebestätigung
    engine, _, _, _ = engine_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    bob = Agent(id="2", name="Bob", position=Position(10, 5))
    bob.is_thinking = True  # Verhindert vorzeitigen Nachrichtenverbrauch im selben Taktzyklus
    engine.register_agent(alice)
    engine.register_agent(bob)

    alice.push_goal(Goal(name="In Nische ausweichen", target_position=Position(5, 4)))
    alice.assign_path([Position(5, 4)])

    alice.receive_message(
        IncomingMessage(
            from_agent_id="2",
            from_agent_name="Bob",
            message="HALT WARTE!",
            is_path_update=True,
            planned_path=[Position(10, 5), Position(9, 5), Position(8, 5)],
            correlation_key="path-upd-42",
        )
    )

    await engine.process_tick()

    # Bob empfängt Freigabebestätigung mit identischem correlation_key
    assert len(bob.inbox) == 1
    confirm = bob.inbox[0]
    assert confirm.is_resume_signal is True
    assert confirm.correlation_key == "path-upd-42"
    assert confirm.message == "Ok, weiter."

    # Alice's temporäres Halteziel wurde direkt wieder abgebaut
    assert alice.active_goal.name == "In Nische ausweichen"


@pytest.mark.asyncio
async def test_signal_path_update_recalculates_niche_on_collision(engine_setup) -> None:
    # TC-SIG-07: Neuer Pfad des Partners schneidet Nische -> alternative Nische wird berechnet
    engine, _, _, _ = engine_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    bob = Agent(id="2", name="Bob", position=Position(8, 5))
    engine.register_agent(alice)
    engine.register_agent(bob)

    # Alice kennt Nische 1 bei (5, 4) und Nische 2 bei (5, 6)
    for x in range(10):
        alice.mental_map.update_tile(Position(x, 5), is_walkable=True, tick=1)
    alice.mental_map.update_tile(Position(5, 4), is_walkable=True, tick=1)
    alice.mental_map.update_tile(Position(5, 6), is_walkable=True, tick=1)

    # Alice visiert ursprünglich (5, 4) an
    alice.push_goal(
        Goal(
            name="In Nische ausweichen",
            target_position=Position(5, 4),
            junction_position=Position(5, 5),
        )
    )

    # Bob's Pfad führt nun direkt über (5, 4)
    alice.receive_message(
        IncomingMessage(
            from_agent_id="2",
            from_agent_name="Bob",
            message="Pfad geändert",
            is_path_update=True,
            planned_path=[Position(8, 5), Position(7, 5), Position(6, 5), Position(5, 4)],
            correlation_key="collision-reroute",
        )
    )

    await engine.process_tick()

    # Alice weicht auf alternative Nische (5, 6) aus
    assert alice.active_goal.target_position == Position(5, 6)


@pytest.mark.asyncio
async def test_signal_clearance_triggered_at_distance_greater_equal_two(engine_setup) -> None:
    # TC-SIG-08: Agent B erreicht Distanz >= 2 zur Junction und sendet Courtesy-Signal
    engine, _, _, _ = engine_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 4))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    engine.register_agent(alice)
    engine.register_agent(bob)

    # Alice wartet in der Nische
    alice.push_goal(
        Goal(
            name="Nischen-Halt",
            holds_position=True,
            is_evasion_hold=True,
            junction_position=Position(5, 5),
            yield_for_agent_id="2",
        )
    )

    # Bob zieht von (6, 5) auf (7, 5); Distanz zu Junction (5, 5) ist danach 2
    bob.assign_path([Position(7, 5)])

    await engine.process_tick()

    assert bob.position == Position(7, 5)
    assert len(alice.inbox) == 1
    courtesy = alice.inbox[0]
    assert courtesy.is_courtesy is True
    assert courtesy.is_resume_signal is True
    assert "Danke fürs Platz machen!" in courtesy.message


@pytest.mark.asyncio
async def test_signal_clearance_suppressed_when_distance_below_two(engine_setup) -> None:
    # TC-SIG-09: Agent B hat Distanz < 2 zur Junction -> kein Courtesy-Signal
    engine, _, _, _ = engine_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 4))
    bob = Agent(id="2", name="Bob", position=Position(5, 5))
    engine.register_agent(alice)
    engine.register_agent(bob)

    alice.push_goal(
        Goal(
            name="Nischen-Halt",
            holds_position=True,
            is_evasion_hold=True,
            junction_position=Position(5, 5),
            yield_for_agent_id="2",
        )
    )

    # Bob zieht nur 1 Schritt nach (6, 5); Distanz = 1
    bob.assign_path([Position(6, 5)])

    await engine.process_tick()

    assert bob.position == Position(6, 5)
    assert len(alice.inbox) == 0


@pytest.mark.asyncio
async def test_signal_courtesy_reawakens_evading_agent(engine_setup) -> None:
    # TC-SIG-10: Empfang von Courtesy-Signal baut Nischen-Halt ab und reaktiviert Hauptroute
    engine, _, _, _ = engine_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 4))
    bob = Agent(id="2", name="Bob", position=Position(7, 5))
    engine.register_agent(alice)
    engine.register_agent(bob)

    for x in range(15):
        for y in range(10):
            alice.mental_map.update_tile(Position(x, y), is_walkable=True, tick=1)

    alice.push_goal(Goal(name="Ost-Tor", target_position=Position(12, 5)))
    alice.push_goal(
        Goal(
            name="Nischen-Halt",
            holds_position=True,
            is_evasion_hold=True,
            junction_position=Position(5, 5),
            yield_for_agent_id="2",
        )
    )

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

    # Nischen-Halt abgebaut, Primärziel aktiv und neuer Pfad berechnet
    assert alice.active_goal.name == "Ost-Tor"
    assert alice.has_path is True
    assert alice.is_busy is False

    # Bob erhält deterministische Antwort
    assert len(bob.inbox) == 1
    assert bob.inbox[0].message == "Gern geschehen!"


@pytest.mark.asyncio
async def test_signal_evasion_notice_clears_partner_evasion_goals(engine_setup) -> None:
    # TC-SIG-11: Evasion-Notice baut überflüssige Warte- und Ausweichziele des Partners ab
    engine, _, _, _ = engine_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    engine.register_agent(alice)
    engine.register_agent(bob)

    bob.push_goal(Goal(name="West-Tor", target_position=Position(1, 5)))
    bob.push_goal(Goal(name="In Nische ausweichen", holds_position=True))

    bob.receive_message(
        IncomingMessage(
            from_agent_id="1",
            from_agent_name="Alice",
            message="Ich mache Platz.",
            is_evasion_notice=True,
        )
    )

    await engine.process_tick()

    assert bob.active_goal.name == "West-Tor"
    assert bob.is_busy is False


@pytest.mark.asyncio
async def test_signal_farewell_resets_session_without_echo(engine_setup) -> None:
    # TC-SIG-12: Verabschiedung setzt Session zurück; keine weitere Antwort-Inferenz
    engine, _, _, _ = engine_setup

    alice = Agent(id="1", name="Alice", position=Position(5, 5))
    bob = Agent(id="2", name="Bob", position=Position(6, 5))
    engine.register_agent(alice)
    engine.register_agent(bob)

    bob.push_goal(Goal(name="West-Tor", target_position=Position(1, 5)))
    bob.push_goal(Goal(name="Konversation mit Alice", holds_position=True))
    engine._session_manager.increment_turn("2", "1")

    bob.receive_message(
        IncomingMessage(
            from_agent_id="1",
            from_agent_name="Alice",
            message="Tschüss!",
            is_farewell=True,
        )
    )

    await engine.process_tick()

    assert engine._session_manager.get_turn_count("2", "1") == 0
    assert bob.active_goal.name == "West-Tor"
    assert bob.is_thinking is False