# implPlan10_03.md: Resiliente Multi-Agenten-Verhandlung, Geometrie-Validierung, Kognitions-Transparenz & Ephemeres Affektmodell

## 1. Problemstellung & Ausführliche Begründungen der Lösungsansätze

### 1.1 Kognitions-Transparenz (`thinking_reason`) & Entkopplung der TTL von Inferenzlatenzen
* **Problem:** 
  In den Testläufen verfielen `InteractionRequest`-Objekte regelmäßig nach Ablauf ihrer Lebensdauer (`ttl_ticks=3`), weil lokale Sprachmodelle für Inferenzschritte zwischen 1,3 und 6,4 Sekunden benötigen[cite: 4, 9]. Während das Modell inferiert, schreitet die Taktzeit der Simulationsschleife (`tick_interval=0.15s`) um 10 bis 40 Ticks voran[cite: 2, 4]. Der wartende Agent verwirft die Anfrage daraufhin (`interaction_request_expired`), obwohl der Partner aktiv an der Lösung arbeitet[cite: 4, 9].
* **Lösungsansatz:**
  1. Erweiterung des Agentenstatus im Domänenmodell um `thinking_reason: Optional[str] = None`.
  2. Aussetzen des TTL-Ablaufs in `AgentProtocolService.process_interaction_queue`, solange entweder der anfragende Agent oder der Adressat im Zustand `is_thinking == True` ist[cite: 9].
* **Begründung:**
  * Ein rein boolesches Flag `is_thinking` reicht für verteilte Kognition und Beobachtbarkeit nicht aus. Indem der konkrete Grund (z. B. `"Wägt Ausweichen für Bob ab"`) explizit im Modell geführt wird, können Logging-Systeme, Benutzeroberflächen (`ThoughtStreamWindow`) und Sensorikmodule anderer Agenten semantisch differenzieren, womit ein Akteur gebunden ist[cite: 2].
  * Inferenzlatenzen sind Artefakte der Ausführungsumgebung (Hardware/Modellgröße) und dürfen nicht die kausale Logik der simulierten Welt korrumpieren. Ein starrer Takt-Ablauf bestraft Agenten für Rechenzeit. Das Einfrieren der TTL ist hier nicht vorgesehen, da andere Agenten (ggf. erst bei künftigen Implementierungen) die TTL parallel benötigen.

---

### 1.2 Geometrische Vorab-Validierung von Ausweichnischen (`EvasionFinder`)
* **Problem:**
  Im Showcase main_dialogue_showcase.py blockierten sich Alice auf (14, 4) und Bob auf (15, 4)[cite: 2, 4]. Die Nische lag bei (15, 3)[cite: 2]. Es fand keine Konversation statt. Da der Eingang zur Nische durch Bobs Kachel (15, 4) verläuft, könnte die Reservierung (`evasion_precommit_invalidated`) gescheitert sein, sodass Alice bewegungsunfähig vor Bob stehenbleibt.
* **Lösungsansatz:**
  *zu evaluieren*
---

### 1.3 Entkopplung der Tag-Nacht-Steuerung (`enable_day_night`)
* **Problem:**
  Mitten in der Engpass-Verhandlung trat bei Tick 100 die Nachtphase ein[cite: 4]. Da Agenten nachts schlafen und physische Bewegungen blockiert sind, wurden asynchrone Kommunikations-Tasks und Zustandsübergänge unterbrochen oder führten zu Hängern nach Tagesanbruch[cite: 1, 4].
* **Lösungsansatz:**
  Einführung des Parameters `enable_day_night: bool = True` in `ApplicationContainer.build()` und `SimulationEngine`[cite: 1, 2]. In `main_dialogue_showcase.py` wird dieser standardmäßig auf `False` gesetzt[cite: 2].
* **Begründung:**
  * Orthogonalität von Testdomänen: Ein Verhandlungsszenario in einer Engstelle dient der Validierung von Kommunikations- und Koordinationsprotokollen. Der makroskopische Tag-Nacht- und Schlafrhythmus ist eine orthogonale Dynamik[cite: 1]. Um Nebeneffekte und Artefakt-Überlagerungen während der Entwicklung zu eliminieren, muss der Lebenszyklus deterministisch isolierbar sein.

---

### 1.4 Affektives Modell (`AffectService`) & Ephemere Kontext-Injektion
* **Problem:**
  Agenten wiederholten bei Uneinigkeit identische Phrasen (`"Bitte weichen Sie aus!"` vs. `"Ich kann nicht einfach ausweichen"`), da ihr interner Zustand rein statisch blieb[cite: 4]. Es gab keine psychologische Entwicklung oder Verhandlungsermüdung.
* **Lösungsansatz:**
  1. Implementierung eines zustandsbasierten `AffectService`, der Parameter wie `frustration` und `patience` pro Agent und Interaktionspartner nachverfolgt.
  2. Dynamische Generierung situativer Sätze basierend auf Schwellenwerten (z. B. bei `frustration > 0.7`: `"Du verlierst allmählich die Geduld, da dein Gegenüber uneinsichtig bleibt."`).
  3. Injektion dieser Sätze als `situational_notes: list[str]` in das ephemere `context`-Dictionary für den System-Prompt des Sprachmodells[cite: 8].
  4. Sobald der Zustand abklingt oder die Situation gelöst ist, entfallen diese Sätze im nächsten Takt automatisch[cite: 5].
  5. **Architektur-Hinweis für künftige Evolutionsstufe:** Der `AffectService` ist interimistisch als deterministisches Domänenmodul ausgelegt. Langfristig übernimmt das lokale Small Language Model (SLM) im Dual-Model-Verbund diese kontinuierliche, feingranulare Affektbewertung mit minimaler Latenz[cite: 3].
* **Begründung:**
  * **Flüchtigkeit gegen Kontext-Verschmutzung:** Affekte sind temporäre psychologische Zustände. Würden sie in die persistente `DialogueHistory` oder den Vektorspeicher geschrieben, würde das Langzeitgedächtnis mit flüchtigen situativen Verärgerungen kontaminiert[cite: 5]. Durch die Übergabe im Prompt existiert die Information genau für die Dauer der Entscheidungsfindung und hinterlässt keine unerwünschten Seiteneffekte[cite: 5, 8].
  * **Verhandlungsdynamik ohne Deadlocks:** Ein Agent mit steigender Frustration formuliert schärfer und bricht bei Erreichen der Geduldsgrenze die Verhandlung ab oder schaltet auf Notfallprotokolle um. Dadurch werden Endlosschleifen durchbrochen.

---

### 1.5 Defensive Schema-Normalisierung (Pydantic-Härtung)
* **Problem:**
  Lokale Modelle variieren gelegentlich im Ausgabeformat: Mal fehlen umschließende Container-Keys (`{"action_type": "end_dialogue", ...}` statt `{"action": {"action_type": "end_dialogue"}}`), mal werden Parameter fälschlich unter `arguments` verschachtelt[cite: 4]. Dies führte zu Pydantic-Validierungsfehlern und unkontrollierten Fallback-Abbrüchen[cite: 4].
* **Lösungsansatz:**
  Erweiterung der `@model_validator(mode="before")`-Methoden in `cognition.py`, um typische Fehlstrukturen vor der eigentlichen Typ-Prüfung deterministisch in die kanonische Form zu überführen[cite: 7].
* **Begründung:**
  * Kleinere Strukturabweichungen des Sprachmodells dürfen nicht zum Absturz des Agenten führen, wenn der semantische Inhalt eindeutig rekonstruierbar ist.

---

## 2. Architektur & Komponenten-Design

src/
├── domain/
│   ├── models/
│   │   ├── agent/
│   │   │   └── agent.py                     # Erweitert: thinking_reason: Optional[str]
│   │   └── planning/
│   │       └── cognition.py                 # Gehärtet: Robuste Root- & Arguments-Entpacker
│   └── services/
│       └── affect_service.py                # Neu: Nachverfolgung von Frustration/Geduld (Interim für SLM)
├── application/
│   ├── services/
│   │   ├── coordination/
│   │   │   ├── agent_protocol_service.py    # Erweitert: TTL-Freeze bei is_thinking
│   │   │   └── dialogue_coordinator.py      # Erweitert: Ephemere situational_notes-Injektion
│   │   └── movement/
│   │       └── evasion_finder.py            # Erweitert: Strikte Pfaderreichbarkeits-Filterung
│   └── simulation_engine.py                 # Erweitert: Flag enable_day_night
└── infrastructure/
    ├── cognition/adapters/
    │   └── instructor_adapter.py            # Erweitert: Berücksichtigung von situational_notes
    └── container.py                         # Erweitert: Flag enable_day_night Durchreichung

---

## 3. Phasenbasierter TDD-Implementierungsplan

Jeder Teilschritt folgt dem strikten Zyklus:
1. Vorbedingungsprüfung
2. Test schreiben (Red)
3. Implementieren (Green)
4. Refaktorisieren & Validieren

---

### Phase 1: Domänenmodell & Schema-Robustheit

#### Schritt 1.1: Erweiterung des Modells `Agent`
* **Ziel:** `thinking_reason` einführen und an `is_thinking` koppeln[cite: 5, 9].
* **Tests:** `tests/unit/domain/agent/test_agent_thinking_state.py`
  * Test: `test_agent_set_thinking_with_reason()`: Setzen von `set_thinking(True, reason="Warte auf Bob")` setzt beide Felder; `set_thinking(False)` setzt `is_thinking=False` und `thinking_reason=None`.
* **Komponenten:** `src/domain/models/agent/agent.py`.
* **Voraussetzungsprüfung für Schritt 1.2:**
  * `pytest tests/unit/domain/agent/test_agent_thinking_state.py` ist grün.

#### Schritt 1.2: Pydantic-Normalisierung für `DialogueResolution` & `BlockedResolution`
* **Ziel:** Robuste Validierung gegen flache Dictionaries und verschachtelte `arguments`-Keys[cite: 4, 7].
* **Tests:** `tests/unit/domain/planning/test_cognition_schemas_resilience.py`
  * Test: `test_dialogue_resolution_unwraps_flat_end_dialogue()`: Validiere Payload ohne `action`-Key: `{"action_type": "end_dialogue", "reason": "Tschüss"}`[cite: 4, 7].
  * Test: `test_blocked_resolution_unwraps_nested_talk_arguments()`: Validiere Payload mit verschachtelten Argumenten: `{"action": {"action_type": "talk", "arguments": {"target_agent_id": "2", "message": "Hi"}}}`[cite: 4, 7].
* **Komponenten:** `src/domain/models/planning/cognition.py`[cite: 7].
* **Voraussetzungsprüfung für Phase 2:**
  * Alle Schema-Tests laufen ohne Pydantic-Validation-Errors durch.

---

### Phase 2: Kognitions-Transparenz & TTL-Ablaufschutz

#### Schritt 2.1: TTL-Schutz bei aktiver Kognition in `AgentProtocolService`
* **Ziel:** Eine `InteractionRequest` darf in `process_interaction_queue` nicht ablaufen, wenn `agent.is_thinking` oder `requester.is_thinking` wahr ist[cite: 9].
* **Tests:** `tests/unit/application/coordination/test_protocol_service_ttl_freeze.py`
  * Test: `test_interaction_request_does_not_expire_while_target_is_thinking()`: Ticks werden um 50 erhöht, aber `agent.is_thinking = True` -> Request bleibt in Queue[cite: 9].
  * Test: `test_interaction_request_does_not_expire_while_requester_is_thinking()`: Ticks werden um 50 erhöht, aber `requester.is_thinking = True` -> Request bleibt in Queue[cite: 9].
  * Test: `test_interaction_request_expires_normally_when_nobody_is_thinking()`: Regulärer TTL-Ablauf nach 3 Ticks ohne Kognition[cite: 9].
* **Komponenten:** `src/application/services/coordination/agent_protocol_service.py`[cite: 9].
* **Voraussetzungsprüfung für Phase 3:**
  * Keine `interaction_request_expired`-Events mehr bei laufenden Denkprozessen im Testlauf[cite: 4, 9].

---

### Phase 3: Geometrische Erreichbarkeit & Nischen-Filterung

#### Schritt 3.1: Strikte Erreichbarkeitsprüfung im `EvasionFinder`
* **Ziel:** Eine Nische gilt nur als Ausweichziel, wenn ein kollisionsfreier Pfad existiert, der die Kachel des Blockers nicht schneidet. Kann der Agent nicht vorwärts ausweichen, wird der rückwärtige Raum geprüft.
* **Tests:** `tests/unit/application/movement/test_evasion_finder_reachability.py`
  * Test: `test_blocked_partner_tile_invalidates_forward_niche()`: Agent A bei (14, 4), Agent B bei (15, 4), Nische bei (15, 3)[cite: 2, 4]. Ergebnis für A muss `None` sein, für B ein gültiger Pfad[cite: 5].
  * Test: `test_backtracking_to_rear_niche_when_forward_is_blocked()`: Agent A bei (14, 4), B bei (15, 4). Nische bei (10, 3) hinter A. Suche liefert Pfad zurück nach (10, 3).
* **Komponenten:** `src/application/services/movement/evasion_finder.py`.
* **Voraussetzungsprüfung für Phase 4:**
  * `compare_evasion_distances` liefert für eingesperrte Agenten deterministisch `None`[cite: 5]. Keine ungültigen `evasion_precommit_invalidated`-Ereignisse mehr[cite: 4].

---

### Phase 4: Konfigurierbarer Tag-Nacht-Zyklus

#### Schritt 4.1: Flag `enable_day_night` in Engine und Container
* **Ziel:** Möglichkeit zur vollständigen Deaktivierung der Nachtphase für fokussierte Verhandlungstests[cite: 1, 2].
* **Tests:** `tests/unit/application/lifecycle/test_day_night_toggle.py`
  * Test: `test_day_night_disabled_prevents_night_phase()`: Bei `enable_day_night=False` bleibt `is_night` auch bei Ticks > 100 immer `False`[cite: 1, 4].
* **Komponenten:**
  * `src/infrastructure/container.py`[cite: 2]
  * `src/application/simulation_engine.py`[cite: 1]
* **Voraussetzungsprüfung für Phase 5:**
  * `main_dialogue_showcase.py` kann ohne nächtliche Pausen kontinuierlich durchlaufen[cite: 2].

---

### Phase 5: Affektmodell & Ephemere Kontext-Injektion

#### Schritt 5.1: Implementierung des `AffectService` (Interim für künftiges SLM)
* **Ziel:** Verwaltung flüchtiger Werte für `frustration` und `patience`.
* **Tests:** `tests/unit/domain/services/test_affect_service.py`
  * Test: `test_reject_increases_frustration_and_drops_patience()`: Wiederholte `reject`-Intents steigern Frustration.
  * Test: `test_agreement_resets_affect()`: Bei `accept` erfolgt ein Reset.
  * Test: `test_generate_situational_notes()`: Liefert passende Verhaltenshinweise basierend auf Schwellenwerten.
* **Komponenten:**
  * `src/domain/services/affect_service.py` (inklusive Dokumentation zur späteren Migration auf das SLM)[cite: 3].

#### Schritt 5.2: Injektion in `DialogueCoordinator` & `InstructorCognitionAdapter`
* **Ziel:** Ephemere Einbindung der `situational_notes` in den System-Prompt; kein persistentes Speichern im Langzeitgedächtnis[cite: 5, 8].
* **Tests:** `tests/unit/application/coordination/test_dialogue_coordinator_affect_injection.py`
  * Test: `test_situational_notes_passed_to_context_and_not_stored_in_history()`: Prüft, dass die Notizen im Aufruf-Kontext enthalten sind, aber weder in `DialogueHistory` noch im `VectorMemoryStore` auftauchen[cite: 5].
* **Komponenten:**
  * `src/application/services/coordination/dialogue_coordinator.py`[cite: 5]
  * `src/infrastructure/cognition/adapters/instructor_adapter.py`[cite: 8]
* **Voraussetzungsprüfung für Phase 6:**
  * Vollständige Unit-Test-Abdeckung für Affekt-Steuerung und ephemere Notizen.

---

### Phase 6: Integration & Showcase-Validierung

#### Schritt 6.1: End-to-End-Test der Showcase-Konfiguration
* **Ziel:** Alice und Bob verhandeln autonom, Bob erkennt seine exklusive Nischenerreichbarkeit, weicht aus und lässt Alice passieren[cite: 2, 4].
* **Tests:** `tests/integration/test_dialogue_showcase_resilience.py`
  * Test: `test_corridor_negotiation_full_cycle_no_exceptions()`: Simuliert das Showcase-Szenario mit Mock-LLM unter extremen Latenzen (simuliertes Tick-Drifting) und verifiziert die kollisionsfreie Passage[cite: 2, 4].
* **Showcase-Anpassung:** `main_dialogue_showcase.py` nutzt `enable_day_night=False` und `enable_deterministic_corridor=False`[cite: 2].

---

## 4. Definition of Done (DoD)

1. **Testabdeckung:** Alle neuen Tests unter `tests/unit/` und `tests/integration/` laufen unter `python3 -m pytest` fehlerfrei durch.
2. **Kognitions-Transparenz:** Jeder Denkprozess setzt `is_thinking=True` und einen aussagekräftigen `thinking_reason`.
3. **Keine Latenz-Timeouts:** Bei langsamer Modellinferenz verfallen keine `InteractionRequest`-Objekte mehr fälschlich durch Takt-Drifting[cite: 4, 9].
4. **Geometrische Korrektheit:** Ein Agent versucht niemals, eine Nische anzusteuern, deren Weg durch den Partner oder andere Objekte versperrt ist[cite: 4, 5].
5. **Ephemere Dynamik:** Affekt-Hinweise beeinflussen die Dialoggenerierung nachweislich dynamisch, ohne das Langzeitgedächtnis zu verschmutzen[cite: 5].
6. **Code-Qualität:** Einhaltung der SOLID-Prinzipien, vollständige Typisierung mit `mypy`.