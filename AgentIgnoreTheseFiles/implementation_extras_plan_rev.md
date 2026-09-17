# Architekturplan: Kognitive Tiefe, Zielsteuerung und Systemhärtung

## 1. Kognitive Modellierung, Persona-RAG & Entscheidungsfindung
* **Persona- & Biografie-RAG**:
  * Trennung zwischen physikalischer Validität (deterministische Engine-Regeln) und kognitiver Präferenz (LLM-Auswahl)[cite: 8].
  * Der Konfliktkontext (Gegenüber, Raumtyp, aktives Ziel) dient als Query an eine lokale Vektordatenbank[cite: 8].
  * Semantischer Abruf von Biografie-Auszügen, Persönlichkeitsdimensionen (z. B. Durchsetzungsvermögen, Kooperationsbereitschaft) und Interaktionshistorien[cite: 8].
  * Injektion von 2–3 prägnanten Sätzen als `character_bias` in den Prompt zur konsistenten Rollenbegründung (`thought`) und Aktionsauswahl (`action_id`) ohne Übermittlung des vollen Lebenslaufs[cite: 8].
* **Snapshot-Isolation für Inferenz-Kontexte**:
  * Übergabe eines unveränderlichen Kontext-Snapshots zum Zeitpunkt des Blockade-Ereignisses an das LLM[cite: 8].
  * Rückgabewerte und Aktionen werden stets gegen den aktuellen Weltzustand validiert und niemals blind gegen den veralteten Snapshot-Zustand ausgeführt[cite: 8].
* **Pre-Commit Action Validation (TOCTOU-Schutz)**:
  * Asynchrone Inferenz via `asyncio.create_task` entkoppelt Anfrage und Ausführung zeitlich[cite: 8].
  * Unmittelbar vor Ausführung generierter Aktionen (`execute_blockage_action`, `execute_evasion`) erfolgt ein atomarer Plausibilitätscheck[cite: 8]:
    * Befindet sich der Blocker weiterhin auf der Zielkachel[cite: 8]?
    * Ist der berechnete Ausweichpfad weiterhin kollisionsfrei[cite: 8]?
    * Ist das adressierte Ziel noch unerledigt[cite: 8]?
  * Schlägt die Validierung fehl, wird die Aktion verworfen und ein Re-Evaluation-Event protokolliert[cite: 8].

---

## 2. Kommunikationsinfrastruktur & Nebenläufigkeitsschutz
* **Typisierte Kommunikationskanäle & Reichweiten**:
  * Entkopplung harter Sprecher-Empfänger-Bindungen über `channel_type: CommunicationChannel` in `IncomingMessage`[cite: 8].
  * `LOCAL_TALK`: Direkte Konversation von Angesicht zu Angesicht ($L_1 \le 3$); ausschließlicher Kanal für physische Blockadeverhandlungen[cite: 8].
  * `PUBLIC_ADDRESS`: Lokale Lautsprecher/Rufe ($L_1 \le 15$) für Raum- und Stauwarnungen[cite: 8].
  * `DIGITAL_NETWORK`: Protokollbasierter Funk/Netzwerkverkehr ($L_1 = \infty$) ohne räumliche Beschränkung[cite: 8].
  * `PerceptionService` filtert Nachrichten vor Ablage in die `inbox` anhand der distanzbasierten Kanalreichweite[cite: 8].
* **Double-Buffered Messaging (Staging-Puffer)**:
  * Vollständige Trennung von Schreibpuffer (`staging_inbox`) und Lesepuffer (`inbox`)[cite: 8].
  * Alle während Takt $t$ erzeugten Nachrichten (`TalkAction`, `is_resume_signal`, `is_courtesy`) werden im Staging-Puffer gesammelt[cite: 8].
  * Zu Beginn von Takt $t+1$ erfolgt ein atomarer Transfer aller Staging-Nachrichten in die aktive `inbox` zur Beseitigung von Iterationsreihenfolge-Artefakten (`_entities`)[cite: 8].
* **Asynchrone Lock-Hierarchie & Circuit-Breaker**:
  * Kanonische Sortierung von Lock-Schlüsseln im `DialogueSessionManager` zur Vermeidung zirkulärer Deadlocks: `key = tuple(sorted([agent_a_id, agent_b_id]))`[cite: 8].
  * Feste Timeouts für asynchrone Locks (`async with lock:`) erzwingen bei Blockaden die Freigabe und protokollieren ein `dialogue_timeout`-Ereignis[cite: 8].
* **Warteschlangen-Lebenszyklus & Anti-Starvation**:
  * `InteractionRequest` in der `interaction_queue` erhält eine feste Time-To-Live (z. B. `current_tick + 3`) und verfällt bei veralteter Relevanz ohne Kognitionsaufruf[cite: 8].
  * Dynamischer Dringlichkeits-Boost (Aging) im `CriticalSectionCoordinator` verhindert das Verhungern von `ROUTINE`-Zielen bei anhaltender Belegung[cite: 8].
  * Automatische Timeout-Freigabe verwaister Critical-Section-Locks bei Agenten-Abbrüchen (`AbortAction`) oder Laufzeitfehlern[cite: 8].

---

## 3. Zielsystem, Prioritäten & Epistemische Trajektorien
* **Zielstruktur & Prioritätskopplung (`Goal`)**:
  * `priority: ExecutionPriority` steuert Unterbrechbarkeit und Blockadestatus[cite: 8]:
    * `URGENT` (Prio 1): Physische Räumung, Nischenausweichen, Nischenhalt; belegt den Agenten vollständig (`is_busy = True`)[cite: 8].
    * `COOPERATIVE` (Prio 2): Partnerannäherung, Begleitung, Dialogannahme; durch Prio 1 unterbrechbar (`is_busy_interruptible = True`)[cite: 8].
    * `ROUTINE` (Prio 3): Hauptzielnavigation (z. B. "Ost-Tor")[cite: 8].
  * `target_entity_id: Optional[str]`: Referenz auf die gesuchte oder verfolgte Entität[cite: 8].
  * `status: Literal["active", "paused", "completed", "abandoned"]`: Expliziter Lebenszyklusstatus[cite: 8].
* **Ziel-Pausierung, Präemption & Reaktivierung**:
  * Wird ein Agent auf Prio 2 (`COOPERATIVE`) blockiert oder zum Platzmachen aufgefordert, wechselt das Ziel auf `status = "paused"` unter Sicherung des bisherigen Pfades[cite: 8].
  * Das höherrangige Prio-1-Ziel (`URGENT`) wird auf den Stack gelegt und präemptiv ausgeführt – auch wenn der Anfragende das verfolgte Ziel selbst ist[cite: 8].
  * Nach Räumung und Abschluss des Prio-1-Ziels wechselt das pausierte Ziel zurück auf `active` und stößt eine Neuberechnung des Annäherungspfads an[cite: 8].
* **Dreistufige epistemische Suchkaskade (Ziel- & Partnerannäherung)**:
  1. *Direkte Wahrnehmung*: Lokaler Radius ($L_1 \le r_{\text{auditiv}}$) über `PerceptionService`[cite: 8].
  2. *MentalMap & Extrapolation*: Abfrage von `AgentMemory.get_projected_fact(entity_id)` bei Distanzüberschreitung; Pfadfindung dorthin via `AgentMentalMap` bei Konfidenz > 0[cite: 8].
  3. *Frontier-Exploration*: Ermittlung und Ansteuerung der nächsten unentdeckten Grenzkachel (`FrontierTile`) bei unbekannter Position oder falsifizierter Schätzung[cite: 8].
  * Phasenwechsel werden in `AgentMemory` persistiert und als `search_phase_transition` geloggt[cite: 8].
* **Synchronisation von MentalMap & Ground Truth**:
  * Beim Empfang von Ausweich- und Räumungssignalen (`is_resume_signal`, `is_evasion_notice`) werden gespeicherte Partnertrajektorien (`partner_planned_path`) im Speicher invalidiert[cite: 8].
  * Temporär gesperrte Ausweichkacheln werden nach Signalverarbeitung deterministisch auf `WALKABLE` zurückgesetzt[cite: 8].

---

## 4. Physische Arbitrierung & Räumliche Reservierung
* **Zweiphasiger Simulationszyklus**:
  * *Phase 1 (Intention & Kommunikation)*: Inboxes konsumieren, LLM-/Heuristik-Inferenz ausführen, Aktionen und Zielansprüche ohne direkte Grid-Mutation deklarieren[cite: 8].
  * *Phase 2 (Arbitrierung & Ausführung)*: Deklarierte Absichten zentral auf Konflikte prüfen, Schritte (`step()`) physisch ausführen, Kollisionen erfassen und Signale in Staging-Puffer schreiben[cite: 8].
* **Kachel-Reservierungstabelle (Reservation Table)**:
  * Flüchtige Reservierungsebene für den Zielschritt in Takt $t+1$[cite: 8].
  * Deterministische Nischen-Arbitrierung bei gleichzeitigem Zugriff zweier Agenten auf dieselbe Kachel nach[cite: 8]:
    1. Ziel-Priorität (`ExecutionPriority`)[cite: 8].
    2. Manhattan-Distanz zur Nische[cite: 8].
    3. Kanonischer ID-Vergleich (`agent_id_a < agent_id_b`) als Tie-Breaker[cite: 8].
* **Swap- und Crossing-Kollisionsschutz**:
  * Erkennung symmetrischer Tauschkonflikte ($A \to \text{Pos}_B$ zeitgleich zu $B \to \text{Pos}_A$)[cite: 8].
  * Schutz gegen diagonale Pfadkreuzungen und gleichzeitiges Betreten desselben freien Zielfeldes[cite: 8].

---

## 5. Offene Evaluations- & Implementierungsfragen
1. Erfassung eines dedizierten Logging-Events `farewell_handshake_completed`, sobald beide Agenten `has_bid_farewell == True` erreicht haben[cite: 8]?
2. Protokollierung von abgewiesenen Verabschiedungen (`reject`) in `DialogueHistory`, um diese in Folge-Prompts hervorzuheben[cite: 8]?
3. Automatische Ableitung von `search_direction` in `ActionExecutor.execute_evasion` aus `agent.memory.known_entities[partner_id].last_observed_velocity`, falls der direkte Nischenweg versperrt ist[cite: 8]?
4. Definition eines Richtungswechsel-Schwellenwerts in `AgentMemory`, ab welchem die historische Glättung (`smoothed_velocity`) verworfen wird[cite: 8]?
5. Automatischer Abbruch und Neuplanung in `TargetSearchService`, sobald ein sichtbares Ziel mehr als 2 Kacheln von seinem letzten Pfad-Endpunkt abweicht[cite: 8]?
6. Konfigurierbare Begrenzung der maximalen Suchtiefe bei der Frontier-Exploration zur Performance-Sicherung auf großen Karten (> 100x100)[cite: 8]?
7. Dynamische Options-Generierung (`available_choices` im Prompt statt statischer Pydantic-Masken) zur drastischen Kontextreduktion bei `resolve_blockage`[cite: 8]?
8. Dedizierter Regressionstest für die FIFO-Warteschlange (`interaction_queue`) zur Validierung von Einreihung, Lazy Validation und zeitversetzter Abarbeitung[cite: 8]?
