# society_sim: Status & Entwicklungsplan

## 1. Erreichter Meilenstein: Integrationsstufe 3
* **Status:** Erfolgreich implementiert und per Integrationstest abgesichert (`test_hunger_simulation_stage_3_explore_and_consume`).
* **Funktionsumfang:**
  * Deterministische Zustandserfassung via `AgentCognitiveContext` und `EntityPerceptionFact`.
  * Autonome Zielzerlegung: Agent initiiert bei akutem Mangel und fehlendem Wissen `explore`.
  * Raumerschließung über `FrontierExplorer` entlang der Sichtgrenzen.
  * Epistemische Re-Evaluation: Opportunistischer Zielabbruch (`_interrupt_goal_for_replan`) bei sensorischer Erfassung passender Ressourcen.
  * Re-Planung im Folgezyklus mit neuem Wissen (`move_to` -> `consume`) bis zur Bedürfnisbefriedigung.

---

### Ergänzung zum Implementierungsplan (Stand: Lauffähige Baseline)

#### 1. Abgeschlossene Reparaturen & Baseline-Wiederherstellung
* **LSP-Konformität in `InstructorCognitionAdapter`:**
  * Implementierung von `decompose_plan` mit strukturierter Pydantic-Validierung.
  * Deterministischer Heuristik-Fallback (`_heuristic_fallback_plan`) für netzwerkunabhängigen und robusten Testbetrieb.
* **Koordinator-Initialisierung:**
  * Standard-Instanziierung von `ConflictCoordinator` und `DialogueCoordinator` in `SimulationEngine.__init__`.
  * Null-Checks in `_handle_blocked_agent`, `_process_interaction_queue` und `_process_agent_inbox` integriert.
* **Systemstatus:**
  * `python3 main.py` führt das Labyrinth-Szenario deterministisch und fehlerfrei aus.
  * Alle Unit- und Integrationstests (inklusive Stufe 3) sind grün. (noch nicht komplett durch veraltete Tests. Traceback folgt)

---

#### 2. Refactoring-Fahrplan: Dekonstruktion der `SimulationEngine`
* **Phase 1: Extraktion `AgentProtocolService` (`src/application/services/agent_protocol_service.py`)**
  * Kapselung aller Agent-zu-Agent-Nachrichten- und Ausweichprotokolle:
    * Inbox-Routing (`courtesy`, `halt`, `resume`, `path_update`, `evasion_notice`, `farewell`).
    * TTL-Warteschlangen für Blockaden (`interaction_queue`).
    * Topologische Nischenankunft und Chokepoint-Clearance (`_check_and_signal_clearance`, `_handle_niche_arrival`).
    * Distanzvalidierung aktiver Dialoge (`_verify_dialogue_distance`).
* **Phase 2: Extraktion `AgentMovementOrchestrator` (`src/application/services/agent_movement_orchestrator.py`)**
  * Kapselung der Bewegungsphase (Phase 2):
    * Sammeln und Einreichen von `TileReservationIntent`.
    * Ausführung des Zwei-Phasen-Commits über `MovementSyncService`.
    * Statische Hinderniserkennung und lokale Pfadneuberechnung.
* **Phase 3: Extraktion `AgentCognitionCoordinator` (`src/application/services/agent_cognition_coordinator.py`)**
  * Kapselung der Absichten und Vitalwerte (Phase 1):
    * Zyklische Bedürfnisüberwachung (`NeedService`).
    * Plandekompositions-Trigger via `PlanDecompositionService`.
    * Abarbeitung von Sub-Goals (`move_to`, `explore`, `consume`).
* **Phase 4: Bereinigung `SimulationEngine`**
  * Reduktion auf einen reinen Takt-Orchestrator (Facade) mit maximal 6 primären Abhängigkeiten.

## 3. Nächster Meilenstein: Kognitive Autonomie & SLM-Pipeline
Ziel: Agenten entwickeln Handlungspläne emergierend aus innerem Zustand, Umweltwissen, Charaktereigenschaften und Historie, statt über fest verdrahtete Schwellenwerte.

### Phase A: Erweiterung des kognitiven Entscheidungsraums
* **Multi-Bedürfnis-Dynamik:**
  * Konkurrierende Vitalparameter (Hunger, Durst, Energie/Schlaf) simultan überwachen.
  * Gewichtung und Priorisierung durch den Kognitionsservice statt harter `hunger >= 0.70`-Trigger.
* **Erweiterung von `AgentCognitiveContext`:**
  * Einbindung von Charaktereigenschaften (`traits`, z. B. Risikobereitschaft, Geduld).
  * Anbindung des episodischen Speichers (`episodic_memories`, z. B. frühere Ressourcenstandorte, Interaktionserfahrungen).
  * Semantische Informationsaufnahme: Gerüchte und Dritt-Informationen (`source`, `confidence`) aus Dialogen in Handlungsoptionen überführen.

### Phase B: SLM-Finetuning & Lokale Inferenz
* **Trainingsdaten-Pipeline:**
  * Aufzeichnung valider Zustands-Aktions-Sequenzen (`AgentCognitiveContext` -> `PlanDecomposition`) als JSONL-Datensätze.
  * Vorbereitung des Datensatzes für Unsloth (QLoRA-Finetuning kompakter SLMs).
* **Adapter-Integration:**
  * Fertigstellung des `InstructorAdapter` (`src/infrastructure/cognition/instructor_adapter.py`) für strukturierte Ausgaben.
  * Anbindung an lokale Inferenz-Backends (vLLM / Ollama).
  * Absicherung gegen Halluzinationen und ungültige Aktionsschemata über Pydantic-Validierungs-Fallbacks.

---

## 3. Aktuell eingeschobener Zwischenschritt: Refactoring `SimulationEngine`
Ziel: Dekonstruktion des God-Objects zur Sicherung der Wartbarkeit vor Ausbau der Kognition.

* **Schnitt 1:** Extraktion von `AgentProtocolService` (Inbox-Routing, Handshakes, Nischen-Ausweichen, Hörweitenprüfung).
* **Schnitt 2:** Extraktion von `AgentMovementOrchestrator` (Phase 2: Arbitrierung, `MovementSyncService`, Kollisions-Replanning).
* **Schnitt 3:** Extraktion von `AgentCognitionCoordinator` (Phase 1: Vitalstoffwechsel, Plandekomposition, Sub-Goal-Ausführung).
* **Schnitt 4:** Reduktion der `SimulationEngine` auf eine schlanke Facade für den Taktzyklus.