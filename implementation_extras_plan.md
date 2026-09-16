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

### Weiterführend

1. Soll für die Verifikation der Zustandsübergänge im Logging ein dediziertes Event `farewell_handshake_completed` erfasst werden, sobald beide Agenten `has_bid_farewell == True` erreicht haben?
2. Soll in `DialogueHistory` vermerkt werden, wenn ein Agent eine Verabschiedung via `reject` abweist, um dieses Ereignis in nachfolgenden Prompts gezielt hervorzuheben?
3. Soll `ActionExecutor.execute_evasion` den Parameter `search_direction` automatisch aus `agent.memory.known_entities[partner_id].last_observed_velocity` ableiten, falls der direkte Pfad in eine Nische blockiert ist?
4. Soll in `AgentMemory` zusätzlich ein Schwellenwert für signifikante Richtungswechsel definiert werden, ab dem die historische Glättung (`smoothed_velocity`) verworfen wird?
5. Soll `TargetSearchService` den Pfad zu einem sich bewegenden Ziel in Sichtweite automatisch abbrechen und neu berechnen, sobald das Ziel mehr als 2 Kacheln von seinem letzten Pfad-Endpunkt abweicht?
6. Soll die maximale Suchtiefe bei der Frontier-Exploration über ein Konfigurationsattribut begrenzt werden, um Performance-Einbrüche auf großen Maps (> 100x100) zu vermeiden?
7. Soll als nächster Schritt die dynamische Options-Generierung für Blockaden (`available_choices` im Prompt statt statischer Pydantic-Masken) angegangen werden, um auch bei `resolve_blockage` den Kontext für Llama-3-8B drastisch zu reduzieren?
8. Soll ein dedizierter Regressionstest für die FIFO-Warteschlange (`interaction_queue`) erstellt werden, der das Einreihen, die Lazy Validation und die zeitversetzte Abarbeitung automatisiert abprüft?
