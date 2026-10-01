# AUDIT- & INTEGRATIONS-AUFTRAG: VERIFIKATION DER LAUFZEIT-VERDRAHTUNG (EXECUTION PATH)

Beim Ausführen der Simulation über `main.py` bzw. `main2.py` verhält sich das System identisch zum Altbestand: Blockaden und Korridorkonflikte laufen weiterhin über die LLM-Inferenz (`ICognitionProvider`), während die neu konzipierten deterministischen Komponenten zur Laufzeit nicht aktiv sind.

Führe umgehend ein systematisches Audit durch und binde die deterministische Architektur vollständig in den aktiven Ausführungspfad ein.

---

### 1. DIAGNOSE & ROOT-CAUSE-ANALYSE (IST-ZUSTAND PRÜFEN)

Untersuche deterministisch die Aufrufkette ausgehend vom Einstiegspunkt bis zur Kognitionsabfrage:
1. **Composition Root (`src/infrastructure/container.py`):**
   - Prüfe, welche Dienste in `ApplicationContainer.build()` instanziiert und an die `SimulationEngine` übergeben werden.
   - Werden die neuen Module (`MovementSyncService`, `MultiAgentNichePacker`, `ConvoyCoordinator`) dort instanziiert oder fehlen sie vollständig in der Verdrahtung?
2. **Simulations-Schleife (`src/application/simulation_engine.py`):**
   - Prüfe `process_tick()`: Werden bei Blockaden in Phase 1 und Phase 2 weiterhin `self._conflict_coordinator.resolve_blockage` und `self._dialogue_coordinator.handle_incoming_dialogue` aufgerufen, die intern `ICognitionProvider` triggern?
   - Wird in Phase 2 weiterhin die alte, sequenzielle `ReservationTable` statt des lokal beschränkten Zweiphasen-Commits (`MovementSyncService`) genutzt?
3. **Konfliktkoordination (`src/application/services/conflict_coordinator.py` & `dialogue_coordinator.py`):**
   - Prüfe, ob die Korridor-Konfliktlösung noch immer hardcodiert auf `await self._cognition_provider.resolve_blockage(context)` verzweigt, anstatt die deterministische Arbitrierung (FusedMentalMap, 70/30-Nutzenfunktion, deterministische Templates) zu durchlaufen.

---

### 2. PFLICHT-REFACTORING & AKTIVIERUNG IM AUSFÜHRUNGSPFAD

Bringe die Architektur mit den Spezifikationen aus `Implementierungsplan.md` in Übereinstimmung:

1. **Composition Root aktualisieren (`src/infrastructure/container.py`):**
   - Instanziiere die neuen Dienste (`MovementSyncService`, `MultiAgentNichePacker`, `ConvoyCoordinator`) zentral im Container.
   - Injiziere sie ordnungsgemäß in `SimulationEngine`, `GoalService` und die Koordinatoren.
2. **Entkopplung von LLM-Inferenz bei Korridorkonflikten:**
   - Modifiziere die Verzweigung in `SimulationEngine` und `ConflictCoordinator`: Handelt es sich um eine räumliche Begegnung im Korridor (Gegenverkehr zweier Agenten/Konvois), darf KEIN Aufruf an `ICognitionProvider` erfolgen.
   - Starte stattdessen deterministisch die FSM:
     `CONFLICT_DETECTED` -> temporäre Fusion zur `FusedMentalMap` -> Nischenberechnung via `MultiAgentNichePacker` -> Kosten-/Trait-Arbitrierung (70/30, SHA-256-Münzwurf) -> Konvoi-Invertierung/Egress.
   - Der Informationsaustausch MUSS zwingend über die festen String-Templates ohne Sprachmodell generiert werden.
3. **Zweiphasen-Commit scharfschalten (`src/application/simulation_engine.py`):**
   - Ersetze die unkoordinierte Schrittausführung in Phase 2 durch den atomaren Aufruf des `MovementSyncService`.
   - Stelle sicher, dass Intent-Validierungen für Front-to-Tail (Vorwärts) und Tail-to-Front (Backtracking) greifen.
4. **Goal-Lifecycle & Gruppen-Suspension aktivieren:**
   - Stelle sicher, dass bei Nischenfahrten und Zwischenhalten die synchronisierte Gruppensuspension (`group_id`, `participant_ids`) und das `ConvoyResolutionGoal` im `GoalService` gesetzt werden.

---

### 3. VERIFIKATION & ERFOLGSKRITERIEN

1. **Führe die Test-Suite aus:**
   - Alle bestehenden Unit-Tests und die neuen Tests für den deterministischen Ablauf müssen fehlerfrei durchlaufen.
2. **Prüfe die Protokollierung (`JsonlEventLogger` & Konsole):**
   - Starte `main2.py` / `main.py`.
   - Verifiziere anhand der erzeugten Events:
     - Keine Events vom Typ `dialogue_failed`, kein Warten auf LLM-Timeouts bei Korridorbegegnung.
     - Sichtbare Emission der deterministischen Templates (*"Hier ist nicht genug Platz für uns beide."*, Nischenangebote, Clearance-Signale).
     - Deterministischer Eintritt in die Nische bei x=45 und kollisionsfreie Passage beider Agenten.
3. **Dokumentation:**
   - Aktualisiere nach erfolgreicher Verifikation den Status in `Implementierungsplan.md`.