# Zukünftige Erweiterungen: Kognitive Tiefe & Kommunikationsinfrastruktur

## 1. Persona- & Biografie-RAG für Intent-Entscheidungen
* **Konzept:** Trennung zwischen physikalischer Validität und kognitiver Präferenz. Die Simulations-Engine ermittelt deterministisch alle im aktuellen Zustand begehbaren bzw. zulässigen Handlungsoptionen (z. B. Ausweichen, Vorrang beanspruchen, Alternativroute).
* **RAG-Mechanismus:** 
  * Der Konfliktkontext (Gegenüber, Raumtyp, aktuelles Hauptziel) dient als Query an eine lokale Vektordatenbank.
  * Relevante Auszüge aus der Biografie, Persönlichkeitsdimensionen (z. B. Durchsetzungsvermögen, Kooperationsbereitschaft) sowie historische Erinnerungen an das Gegenüber werden semantisch abgerufen.
  * Das Ergebnis wird in 2–3 prägnanten Sätzen als `character_bias` in das Prompt injiziert, um dem LLM die Begründung (`thought`) und Auswahl (`action_id`) konsistent zu seiner Rolle zu ermöglichen, ohne den gesamten Lebenslauf zu übergeben.

## 2. Abstraktion von Kommunikationskanälen & Reichweiten
* **Konzept:** Ablösung harter Kopplung zwischen Sprecher und Empfänger durch typisierte Kommunikationsmedien.
* **Kanal-Typen und Reichweiten:**
  * `LOCAL_TALK`: Direkte Sprache von Angesicht zu Angesicht ($L_1 \le 3$). Physische Blockadeverhandlungen finden ausschließlich auf diesem Kanal statt.
  * `PUBLIC_ADDRESS`: Lokale Lautsprecher oder Rufe ($L_1 \le 15$). Ermöglicht Warnungen vor Staus oder Ankündigungen in Räumen.
  * `DIGITAL_NETWORK`: Funk, SMS oder E-Mail ($L_1 = \infty$). Keine Distanzbeschränkung; erfordert jedoch ein definiertes Kommunikationsprotokoll und Adressierung.
* **Architektonische Umsetzung:**
  * Erweiterung von `IncomingMessage` um `channel_type: CommunicationChannel`.
  * `PerceptionService` prüft vor der Zustellung die Reichweitenbedingung bezogen auf den Kanal. Außerhalb der Distanz liegende Nachrichten werden nicht in die `inbox` abgelegt.

# Implementierungsziel: Epistemische Zielsuche und prioritätsgesteuerte Zielunterbrechung

## 1. Dreistufige epistemische Suchkaskade (Partner-/Zielannäherung)
Muss ein Agent für eine Interaktion räumliche Nähe zu einem Ziel oder Partner herstellen, konsultiert das System die Wissensschichten in fester Reihenfolge:

1. **Direkte Wahrnehmung:**
   * Abfrage über `PerceptionService` im lokalen Radius ($L_1 \le r_{\text{auditiv}}$).
   * Liegt das Ziel im Radius, wird die direkte Interaktion initiiert.
2. **MentalMap & Extrapolation:**
   * Liegt das Ziel außerhalb des Radius: Abfrage von `AgentMemory.get_projected_fact(entity_id)`.
   * Ist eine extrapolierte Position mit Konfidenz > 0 verfügbar, berechnet der Pathfinder einen Pfad dorthin über die bekannte `AgentMentalMap`.
3. **Frontier-Exploration:**
   * Ist die Position unbekannt oder die extrapolierte Zielkachel erreicht, ohne dass die Ziel-Entität wahrgenommen wird (Falsifikation):
   * Der Agent ermittelt die nächste unentdeckte Grenzkachel (`FrontierTile`) auf Basis seiner `AgentMentalMap` und setzt den Pfad dorthin, um das Terrain aufzuklären.

Jeder Phasenübergang (Wahrnehmung -> Pfad zu Schätzposition -> Exploration) wird im Agentengedächtnis (`AgentMemory`) und als Simulationsereignis (`search_phase_transition`) dokumentiert.

## 2. Zielstruktur und Prioritätskopplung
Die Klasse `Goal` wird um Prioritätsstufen und Referenzfelder erweitert:

* `priority: ExecutionPriority`:
  * `URGENT` (Prio 1): Physische Räumung, Ausweichen in Nische, Nischenhalt. Blockiert andere Aktionen vollständig (`is_busy = True`).
  * `COOPERATIVE` (Prio 2): Partner aufsuchen, Begleiten, Warten auf Gesprächsannahme. Ist durch höherwertige Anfragen unterbrechbar (`is_busy_interruptible = True`).
  * `ROUTINE` (Prio 3): Reguläre Navigation zum Hauptziel (z. B. "Ost-Tor").
* `target_entity_id: Optional[str]`: ID der gesuchten oder verfolgten Entität.
* `status: Literal["active", "paused", "completed", "abandoned"]`: Expliziter Status für unterbrochene Ziele.

## 3. Ziel-Pausierung und Wiederaufnahme bei Verfolger-Blockaden
Wird ein Agent, der sich im Verfolgungs- oder Annäherungsmodus (Prio 2) befindet, von einer anderen Entität physisch blockiert oder zum Platzmachen aufgefordert (Prio 1):

1. **Pausierung:**
   * Das aktive Ziel `Goal(name="Partner aufsuchen", priority=COOPERATIVE)` wird auf `status = "paused"` gesetzt.
   * Der bisherige Pfad wird gesichert.
2. **Präemption (Platz machen):**
   * Das höherrangige Ausweichziel `Goal(name="In Nische ausweichen", priority=URGENT)` wird auf den Stack gelegt und ausgeführt.
   * Selbst wenn der Anfragende der verfolgte Partner selbst ist, weicht der Verfolger deterministisch aus, da Prio 1 vor Prio 2 rangiert.
3. **Reaktivierung:**
   * Nach Abschluss des Nischen-Lebenszyklus und erfolgter Freigabe wird das Prio-1-Ziel entfernt.
   * Das oberste verbleibende Ziel wechselt von `"paused"` zu `"active"`.
   * Der Agent führt Stufe 1 oder 2 der Suchkaskade aus, berechnet von seiner aktuellen Position aus einen neuen Pfad zum Partner und setzt die Annäherung fort.

# Härtungsmaßnahmen gegen Race Conditions und Nebenläufigkeitsprobleme

### 1. Phased Execution & Double-Buffered Messaging
* **Double-Buffered Inboxes (Staging-Puffer)**:
  * Trennung von Schreib- (`staging_inbox`) und Lesepuffer (`inbox`).
  * Alle während Takt $t$ generierten Nachrichten (`TalkAction`, `is_resume_signal`, `is_courtesy`) landen im Staging-Puffer.
  * Atomarer Transfer: Zu Beginn von Takt $t+1$ werden alle Staging-Nachrichten in die aktive `inbox` überführt.
  * **Ziel**: Vollständige Unabhängigkeit von der Iterationsreihenfolge der Agenten in `_entities`.
* **Zweiphasiger Simulationszyklus**:
  * **Phase 1 (Intention & Kommunikation)**:
    * Verarbeitung der aktiven Inboxes.
    * Ausführung von LLM-Inferenz / Heuristiken.
    * Deklaration beabsichtigter Aktionen (Bewegungsvektor, Ausweichabsicht, Zielanspruch) ohne Ausführung auf dem physischen Grid.
  * **Phase 2 (Arbitrierung & Ausführung)**:
    * Zentrale Validierung und Konfliktauflösung aller deklarierten Absichten.
    * Physische Ausführung gültiger Bewegungen (`step()`).
    * Erfassung von Kollisionen und Bereitstellung von Signalen im Staging-Puffer für den Folgetakt.

---

### 2. Räumliche Reservierung & Nischen-Arbitrierung (Spatial Contention)
* **Kachel-Reservierungstabelle (Reservation Table)**:
  * Einführung einer flüchtigen Reservierungsebene auf dem Grid für den Folgetakt $t+1$.
  * Belegt ein Agent eine Kachel als intendierten nächsten Schritt oder als Nischen-Ziel, wird diese temporär reserviert.
  * **Konkurrierender Nischenzugriff**: Wollen zwei Agenten im selben Takt in dieselbe freie Nische ausweichen, entscheidet die Arbitrierung vor Pfadzuweisung deterministisch nach:
    1. Ziel-Priorität (`ExecutionPriority`).
    2. Manhattan-Distanz zur Nische.
    3. Tie-Breaker via kanonischer Agenten-ID (`agent_id_a < agent_id_b`).
* **Swap- und Crossing-Kollisionsschutz**:
  * Erkennung symmetrischer Tausch-Konflikte ($A \to \text{Pos}_B$ und zeitgleich $B \to \text{Pos}_A$).
  * Erkennung von Diagonalkreuzungen, falls diagonale Bewegungen zulässig werden.
  * Verhindert das gleichzeitige Betreten derselben freien Zielkachel durch mehrere Agenten.

---

### 3. TOCTOU-Schutz bei asynchroner Inferenz (Time-of-Check to Time-of-Use)
* **Pre-Commit Action Validation**:
  * Da LLM-Aufrufe asynchron laufen (`asyncio.create_task`), vergeht reale Zeit, in der sich die Umwelt verändert haben kann.
  * Unmittelbar vor der Ausführung der generierten Aktion (`execute_blockage_action`, `execute_evasion`) erfolgt ein Plausibilitätscheck:
    * Befindet sich der Blocker noch auf der Zielkachel?
    * Ist der geplante Ausweichpfad noch kollisionsfrei?
    * Ist das Ziel noch unerledigt?
  * **Fallback**: Schlägt die Validierung fehl, wird die Aktion verworfen und ein Re-Evaluation-Event geloggt, statt eine ungültige Aktion blind auszuführen.
* **Snapshot-Isolation für Inferenz-Kontexte**:
  * Dem LLM wird ein unveränderlicher Kontext-Snapshot zum Zeitpunkt der Blockade übergeben.
  * Rückgabewerte werden stets gegen den *aktuellen* Weltzustand gemappt, niemals gegen den Snapshot-Zustand.

---

### 4. Deadlock-Prävention bei asynchronen Locks
* **Kanonische Lock-Hierarchie im `DialogueSessionManager`**:
  * Sortierung der Lock-Schlüssel zur Vermeidung zirkulärer Warteabhängigkeiten bei $\ge 3$ Agenten:
    * `key = tuple(sorted([agent_a_id, agent_b_id]))`
  * Verhindert Deadlocks, wenn Agent A mit B verhandeln will und zeitgleich Agent B mit A.
* **Lock-Timeouts & Circuit-Breaker**:
  * Asynchrone Locks (`async with lock:`) erhalten einen festen Timeout.
  * Bei Überschreitung des Zeitfensters wird die Sperre zwangsweise freigegeben und ein `dialogue_timeout`-Ereignis protokolliert.

---

### 5. Warteschlangen-Lebenszyklus & Anti-Starvation
* **TTL (Time-To-Live) für `InteractionRequest`**:
  * Anfragen in der `interaction_queue` erhalten ein Ablaufdatum (z. B. `current_tick + 3`).
  * Ist die Anfrage beim Dequeueing älter als die TTL, wird sie ohne Kognitionsaufruf verworfen.
* **Fairness-Arbitrierung im `CriticalSectionCoordinator`**:
  * Vermeidung von Starvation niedriger priorisierter Agenten (`ROUTINE`):
    * Ein dynamischer Dringlichkeits-Boost (Aging), wenn ein Agent länger als $N$ Takte in der Warteschlange verweilt.
  * Erkennung verwaister Locks: Gibt ein Inhaber-Agent infolge eines Aborts (`AbortAction`) oder Fehlers die Critical Section nicht frei, erzwingt ein Timeout die Übergabe an den nächsten Wartenden.

---

### 6. Synchronisation von MentalMap und Ground Truth
* **Invalidierung veralteter Trajektorien**:
  * Erhält ein Agent ein Ausweich- oder Räumungssignal (`is_resume_signal`, `is_evasion_notice`), müssen zuvor gecachte Pfade (`partner_planned_path`) im Speicher explizit invalidiert oder aktualisiert werden.
  * Temporär als Hindernis markierte Kacheln müssen deterministisch auf `WALKABLE` zurückgesetzt werden, sobald das Clearance-Signal verarbeitet wurde.

### Weiterführend

1. Soll für die Verifikation der Zustandsübergänge im Logging ein dediziertes Event `farewell_handshake_completed` erfasst werden, sobald beide Agenten `has_bid_farewell == True` erreicht haben?
2. Soll in `DialogueHistory` vermerkt werden, wenn ein Agent eine Verabschiedung via `reject` abweist, um dieses Ereignis in nachfolgenden Prompts gezielt hervorzuheben?
3. Soll `ActionExecutor.execute_evasion` den Parameter `search_direction` automatisch aus `agent.memory.known_entities[partner_id].last_observed_velocity` ableiten, falls der direkte Pfad in eine Nische blockiert ist?
4. Soll in `AgentMemory` zusätzlich ein Schwellenwert für signifikante Richtungswechsel definiert werden, ab dem die historische Glättung (`smoothed_velocity`) verworfen wird?
5. Soll `TargetSearchService` den Pfad zu einem sich bewegenden Ziel in Sichtweite automatisch abbrechen und neu berechnen, sobald das Ziel mehr als 2 Kacheln von seinem letzten Pfad-Endpunkt abweicht?
6. Soll die maximale Suchtiefe bei der Frontier-Exploration über ein Konfigurationsattribut begrenzt werden, um Performance-Einbrüche auf großen Maps (> 100x100) zu vermeiden?
7. Soll als nächster Schritt die dynamische Options-Generierung für Blockaden (`available_choices` im Prompt statt statischer Pydantic-Masken) angegangen werden, um auch bei `resolve_blockage` den Kontext für Llama-3-8B drastisch zu reduzieren?
8. Soll ein dedizierter Regressionstest für die FIFO-Warteschlange (`interaction_queue`) erstellt werden, der das Einreihen, die Lazy Validation und die zeitversetzte Abarbeitung automatisiert abprüft?
