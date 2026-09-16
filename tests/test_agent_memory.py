from __future__ import annotations

import pytest
from src.domain.models.agent_memory import AgentMemory, PerceptionSource
from src.domain.models.mental_map import AgentMentalMap
from src.domain.models.position import Position


@pytest.fixture
def memory() -> AgentMemory:
    return AgentMemory(standstill_threshold=3)


def test_trajectory_smoothing_and_standstill_detection(memory: AgentMemory) -> None:
    # TC-EPI-01: Trajektorienglättung über k=3 und Stillstand
    entity_id = "bob"
    name = "Bob"

    # Schritt 1: Tick 1 an (1, 1)
    memory.update_entity_perception(entity_id, name, Position(1, 1), PerceptionSource.VISUAL, tick=1)
    fact = memory.known_entities[entity_id]
    assert fact.last_known_position == Position(1, 1)
    assert fact.standstill_ticks == 0

    # Schritt 2: Tick 2 an (2, 1) -> Bewegung nach Osten
    memory.update_entity_perception(entity_id, name, Position(2, 1), PerceptionSource.VISUAL, tick=2)
    assert fact.last_observed_velocity == (1.0, 0.0)
    assert fact.smoothed_velocity == (1.0, 0.0)
    assert fact.standstill_ticks == 0

    # Schritt 3: Tick 3 an (3, 1) -> Historienfenster voll (Länge 3)
    memory.update_entity_perception(entity_id, name, Position(3, 1), PerceptionSource.VISUAL, tick=3)
    assert len(fact.position_history) == 3
    assert fact.smoothed_velocity == (1.0, 0.0)

    # Stillstand: 3 Takte ohne Positionsveränderung
    memory.update_entity_perception(entity_id, name, Position(3, 1), PerceptionSource.VISUAL, tick=4)
    memory.update_entity_perception(entity_id, name, Position(3, 1), PerceptionSource.VISUAL, tick=5)
    memory.update_entity_perception(entity_id, name, Position(3, 1), PerceptionSource.VISUAL, tick=6)

    assert fact.standstill_ticks == 3
    assert fact.last_observed_velocity == (0.0, 0.0)


def test_lazy_decay_confidence_and_extrapolation(memory: AgentMemory) -> None:
    # TC-EPI-02: Linearer Konfidenzabfall (tau = 5) und Extrapolation
    entity_id = "bob"
    memory.update_entity_perception(entity_id, "Bob", Position(10, 5), PerceptionSource.VISUAL, tick=10)
    memory.update_entity_perception(entity_id, "Bob", Position(11, 5), PerceptionSource.VISUAL, tick=11)

    # delta_t = 0: Volle Konfidenz
    p_t11 = memory.get_projected_fact(entity_id, current_tick=11)
    assert p_t11 is not None
    assert p_t11.confidence == 1.0
    assert p_t11.projected_position == Position(11, 5)

    # delta_t = 2: Konfidenz = 1.0 - (2 / 5) = 0.6; Extrapolation um 2 Kacheln nach Osten
    p_t13 = memory.get_projected_fact(entity_id, current_tick=13)
    assert p_t13 is not None
    assert p_t13.confidence == pytest.approx(0.6)
    assert p_t13.projected_position == Position(13, 5)

    # delta_t = 5: Konfidenz = 0.0; keine Position mehr
    p_t16 = memory.get_projected_fact(entity_id, current_tick=16)
    assert p_t16 is not None
    assert p_t16.confidence == 0.0
    assert p_t16.projected_position is None


def test_falsification_and_wall_protection(memory: AgentMemory) -> None:
    # TC-EPI-03: Kollision mit Wand oder Sichtfeld bricht Konfidenz sofort ab
    entity_id = "bob"
    memory.update_entity_perception(entity_id, "Bob", Position(5, 5), PerceptionSource.VISUAL, tick=1)
    memory.update_entity_perception(entity_id, "Bob", Position(6, 5), PerceptionSource.VISUAL, tick=2)

    mental_map = AgentMentalMap(width=20, height=20)
    for x in range(20):
        for y in range(20):
            mental_map.update_tile(Position(x, y), is_walkable=True, tick=1)

    # Fall A: Extrapolierte Position (7, 5) stößt auf bekannte Wand
    mental_map.mark_obstacle(Position(7, 5), tick=1)
    projected = memory.get_projected_fact(entity_id, current_tick=3, mental_map=mental_map)
    assert projected is not None
    assert projected.confidence == 0.0
    assert projected.projected_position is None

    # Fall B: Kachel (7, 5) ist begehbar, liegt aber im aktuellen Sichtfeld und ist leer
    mental_map.update_tile(Position(7, 5), is_walkable=True, tick=2)
    visible_tiles = {Position(6, 5), Position(7, 5), Position(8, 5)}
    falsified = memory.get_projected_fact(
        entity_id, current_tick=3, mental_map=mental_map, visible_positions=visible_tiles
    )
    assert falsified is not None
    assert falsified.confidence == 0.0
    assert falsified.projected_position is None


def test_entity_probing_and_walkability_recording(memory: AgentMemory) -> None:
    # TC-EPI-04: Objekt-Erprobung und Persistierung
    entity_id = "stone_1"
    memory.update_entity_perception(entity_id, "Fels", Position(4, 4), PerceptionSource.VISUAL, tick=1)
    memory.record_inspection(entity_id, entity_type="rock")

    assert memory.is_inspected(entity_id) is True
    assert memory.can_probe(entity_id) is True

    # Tastung ergibt: Unpassierbar
    memory.record_walkability_result(entity_id, is_walkable=False)
    assert memory.get_entity_walkability(entity_id) is False
    assert memory.can_probe(entity_id) is False


def test_non_responsive_entity_blocks_further_talk(memory: AgentMemory) -> None:
    # TC-EPI-05: Stumme Entitäten schließen Talk-Aktionen aus
    entity_id = "rock_blocker"
    memory.update_entity_perception(entity_id, "Stein", Position(2, 2), PerceptionSource.VISUAL, tick=1)
    memory.record_inspection(entity_id, entity_type="rock")

    assert memory.can_talk(entity_id) is True

    # Leere Rückmeldung (keine Antwort)
    memory.record_interaction_result(entity_id, responded=False)

    assert memory.get_assumed_conversational(entity_id) is False
    assert memory.can_talk(entity_id) is False
    assert memory.is_epistemically_exhausted(entity_id) is False

    # Nach zusätzlicher Tastung: Epistemisch vollständig erschöpft
    memory.record_walkability_result(entity_id, is_walkable=False)
    assert memory.is_epistemically_exhausted(entity_id) is True