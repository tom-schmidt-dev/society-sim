from src.domain.services.affect_service import AffectService


def test_reject_increases_frustration_and_drops_patience() -> None:
    service = AffectService()
    agent_id = "1"
    partner_id = "2"

    initial_affect = service.get_affect(agent_id, partner_id)
    assert initial_affect.frustration == 0.0
    assert initial_affect.patience == 1.0

    # Erste Ablehnung
    service.record_turn(agent_id=agent_id, partner_id=partner_id, incoming_intent="reject")
    affect_1 = service.get_affect(agent_id, partner_id)
    assert affect_1.frustration > 0.0
    assert affect_1.patience < 1.0

    # Wiederholte Ablehnung steigert Frustration weiter
    service.record_turn(agent_id=agent_id, partner_id=partner_id, incoming_intent="reject")
    affect_2 = service.get_affect(agent_id, partner_id)
    assert affect_2.frustration > affect_1.frustration
    assert affect_2.patience < affect_1.patience


def test_agreement_resets_affect() -> None:
    service = AffectService()
    agent_id = "1"
    partner_id = "2"

    service.record_turn(agent_id=agent_id, partner_id=partner_id, incoming_intent="reject")
    service.record_turn(agent_id=agent_id, partner_id=partner_id, incoming_intent="reject")

    # Einigung setzt die Affektwerte zurück
    service.record_turn(agent_id=agent_id, partner_id=partner_id, incoming_intent="accept")
    affect = service.get_affect(agent_id, partner_id)
    assert affect.frustration == 0.0
    assert affect.patience == 1.0


def test_generate_situational_notes_based_on_thresholds() -> None:
    service = AffectService()
    agent_id = "1"
    partner_id = "2"

    # Initialzustand: Keine situativen Warnungen
    notes_initial = service.generate_situational_notes(
        agent_id=agent_id, partner_id=partner_id, assertiveness=0.8
    )
    assert notes_initial == []

    # Mehrfache Ablehnung provoziert hohe Frustration
    for _ in range(4):
        service.record_turn(agent_id=agent_id, partner_id=partner_id, incoming_intent="reject")

    notes_frustrated = service.generate_situational_notes(
        agent_id=agent_id, partner_id=partner_id, assertiveness=0.85
    )
    assert len(notes_frustrated) > 0
    assert any("Geduld" in note or "Zeitdruck" in note for note in notes_frustrated)