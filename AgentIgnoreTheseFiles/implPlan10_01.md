# implPlan10_01.md: Frame-Pufferung, Live-Kognitions-Stream & Autonome Zielableitung

## 1. Systemübersicht & Architektur-Ziele

* Entkopplung der Taktraten (Producer-Consumer-Muster):
  * Die Simulation läuft mit nativer Geschwindigkeit ohne künstliche Verzögerung in Phase 2 (Physik) und pausiert während LLM-Aufrufen in Phase 1[cite: 1, 2].
  * Der ConsolePresenter gibt den aktuellen Live-Zustand ungebremst im Terminal aus[cite: 5].
  * Ein FrameBufferService puffert unveränderliche RenderFrame-Instanzen thread-sicher in einer FIFO-Warteschlange.
  * Zukünftige grafische Darstellungsschichten konsumieren Frames mit einer definierten Maximalgeschwindigkeit, um Schwankungen durch KI-Latenzen auszugleichen.
* Fail-Fast Inferenz-Prüfung:
  * Überprüfung der Verfügbarkeit des Ollama-Dienstes und des konfigurierten Modells vor Takt 1.
  * Auslösen eines RuntimeError ohne Start der Simulation, falls der Dienst offline ist oder das Modell fehlt.
* Separates Kognitions-Fenster (PyQt6):
  * Live-Visualisierung der inneren Gedankenprozesse, Bedürfnisse und Zielhierarchien synchron zur Simulation.
  * Thread-Trennung: PyQt6-Event-Loop im Haupt-Thread, SimulationEngine (asyncio) in einem separaten Worker-Thread[cite: 1].
* Autonome Zielableitung:
  * Agenten generieren Ziele eigenständig auf Basis von Vitalbedürfnissen (NeedService), Gedächtnisinhalten (IVectorMemoryStore), Umgebungsdaten (PerceptionService) und Charaktereigenschaften[cite: 1, 2].

---

## 2. Detaillierte Teilschritte

### Teilschritt 1: Pre-Flight-Validierung für Inferenz (Ollama)
* Ziel: Unmittelbarer Programmabbruch bei fehlender Inferenz-Bereitschaft ohne Mocking-Fallback.
* Komponenten:
  * src/infrastructure/cognition/health/ollama_health_checker.py:
    * Klasse: OllamaHealthChecker mit synchroner Methode check_status(api_base: str, model_name: str) -> None.
    * Sendet eine HTTP-GET-Anfrage an {api_base}/api/tags.
    * Prüft die Erreichbarkeit des Dienstes und das Vorhandensein des Modells in der Rückgabe.
    * Wirft RuntimeError("Ollama-Dienst nicht erreichbar unter {api_base} oder Modell '{model_name}' nicht geladen.") bei Fehlern oder fehlendem Modell.
* Integration:
  * Aufruf in ApplicationContainer.build() vor der Instanziierung der Kognitionsadapter.
* Tests:
  * tests/unit/infrastructure/cognition/test_ollama_health.py (Verbindungsfehler, Modell fehlt, erfolgreiche Validierung via unittest.mock).

### Teilschritt 2: Domänenmodell & Pufferung für Render-Frames
* Ziel: Konsistente, zeitstempelbasierte Kapselung visueller Taktzustände.
* Komponenten:
  * src/domain/models/rendering/render_frame.py:
    * Unveränderliches Value Object RenderFrame:
      * tick: int
      * timestamp: float
      * grid_matrix: list[list[str]] (deterministische Zeichenmatrix der Kacheln und Hindernisse)
      * entities: list[dict[str, Any]] (ID, Name, Typ, Position, Symbol)
      * dialogues: list[str]
      * snapshots: list[AgentCognitiveSnapshot]
  * src/application/services/rendering/frame_buffer_service.py:
    * Thread-sichere FIFO-Warteschlange (queue.Queue[RenderFrame]) mit konfigurierbarer Maximalkapazität.
    * Methoden: push_frame(frame: RenderFrame) -> None, pop_frame(timeout: Optional[float] = None) -> Optional[RenderFrame], clear() -> None.
  * src/infrastructure/presentation/console/console_presenter.py:
    * Trennung in generate_frame(...) -> RenderFrame (reine Zustandstransformation) und render(...) (Terminal-I/O und optionale Weiterleitung an den FrameBufferService)[cite: 5].
* Tests:
  * tests/unit/domain/rendering/test_render_frame_generation.py
  * tests/unit/application/rendering/test_frame_buffer_service.py

### Teilschritt 3: Live-Kognitions-Stream-Fenster (PyQt6)
* Ziel: Parallele Visualisierung der inneren Gedankenprozesse in einem separaten GUI-Fenster.
* Komponenten:
  * src/infrastructure/presentation/qt/simulation_worker.py:
    * SimulationWorker(QThread): Führt den asyncio-Event-Loop der SimulationEngine aus[cite: 1].
    * Sendet das Qt-Signal frame_emitted = pyqtSignal(object) nach jedem verarbeiteten Takt an das Interface.
  * src/infrastructure/presentation/qt/thought_stream_window.py:
    * ThoughtStreamWindow(QMainWindow):
      * Tabs oder Split-Panes für jeden registrierten Agenten.
      * Live-Statusanzeige: Aktuelles dominantes Bedürfnis, Zielstack, Hunger/Durst/Energie-Werte[cite: 2].
      * Fortlaufender Gedanken-Log mit automatischem Zeilensprung (Auto-Scroll) bei neuen Einträgen.
* Tests:
  * tests/unit/infrastructure/presentation/test_qt_thought_window_signals.py

### Teilschritt 4: Autonome Zielableitung auf Basis von Gedächtnis und Bedürfnissen
* Ziel: Vollständige Selbstständigkeit der Agenten bei der Handlungsplanung ohne extern vorgegebene Navigationsziele.
* Komponenten:
  * src/application/services/cognition/plan_decomposition_service.py:
    * Integration der semantischen Suche über IVectorMemoryStore bei Mangelzuständen (z. B. Abfrage nach bekannten Nahrungs- oder Wasserquellen)[cite: 1, 2].
    * Berücksichtigung von Persönlichkeitsmerkmalen (assertiveness, charisma) bei der Zielpriorisierung.
  * src/application/services/cognition/cognition_orchestrator.py:
    * Autonome Generierung von Sub-Goals, sobald der Zielstack leer ist und Vitalwerte definierte Grenzwerte überschreiten[cite: 1, 2].
    * Rückfall auf Frontier-Exploration (FrontierExplorer), falls keine passende Position im Gedächtnis auffindbar ist[cite: 1].
* Tests:
  * tests/integration/test_autonomous_goal_generation.py

### Teilschritt 5: Demonstrations- und Ausführungsskripte
* Skript 1 (src/main_memory_lifecycle.py):
  * Einzelszenario: Ein einzelner Agent in einer Umgebung mit verteilten Ressourcen.
  * Ablauf:
    1. Agent beginnt ohne vorgegebenes Ziel; Vitalbedürfnisse steigen schrittweise an[cite: 2].
    2. Agent erkundet das Areal, entdeckt eine Nahrungsquelle und verzehrt diese[cite: 2].
    3. Beginn der Nachtphase (DayNightService): Agent wechselt in den Schlafzustand, MemoryConsolidationService synthetisiert und vektorisiert die Tageserfahrungen in ChromaDB[cite: 1, 2].
    4. Folge-Tag: Steigender Hunger veranlasst den Agenten zum Vektor-Retrieval; er navigiert gezielt zur gestern entdeckten Position zurück[cite: 2].
  * Darstellung: Live-Terminal-Gitter und paralleles PyQt6-Kognitionsfenster.
* Skript 2 (src/main_dialogue_showcase.py):
  * Mehragenten-Szenario: Zwei Agenten begegnen sich gegenläufig in einem Korridor.
  * Ablauf:
    1. Agent 1 (hohe assertiveness) und Agent 2 (kooperativ) bewegen sich aufeinander zu.
    2. Blockadeerkennung und Initiierung des Dialogs über das LLM.
    3. Verhandlung eines Nischenausweichmanövers.
    4. Vollzug des Ausweichens und Passieren der Engstelle.
  * Darstellung: Live-Terminal-Gitter, Konsolenausgabe des Dialogverlaufs und Anzeige der internen Kognition im PyQt6-Fenster.

---

## 3. Dateistruktur

Projekt-Root
- main.py
- main_dialogue_showcase.py
- main_memory_lifecycle.py
- src/
  - application/
    - services/
      - cognition/
        - cognition_orchestrator.py
        - cognitive_snapshot_service.py
        - frontier_explorer.py
        - goal_service.py
        - plan_decomposition_service.py
      - coordination/
        - agent_protocol_service.py
        - conflict_coordinator.py
        - convoy_arbitrator.py
        - convoy_coordinator.py
        - critical_section_coordinator.py
        - dialogue_coordinator.py
        - dialogue_history.py
        - dialogue_session_manager.py
      - execution/
        - action_executor.py
      - lifecycle/
        - daily_event_buffer.py
        - day_night_service.py
        - memory_consolidation_service.py
        - need_service.py
      - movement/
        - evasion_finder.py
        - movement_orchestrator.py
        - movement_sync_service.py
        - multi_agent_niche_packer.py
        - target_search_service.py
      - rendering/
        - frame_buffer_service.py
      - world/
        - world_loader.py
    - simulation_engine.py
  - domain/
    - models/
      - agent/
        - agent.py
        - agent_cognition.py
        - agent_memory.py
        - mental_map.py
      - communication/
        - communication_templates.py
        - dialogue.py
        - interaction_request.py
        - message.py
      - coordination/
        - critical_section.py
        - evasion_phase.py
        - reservation_table.py
        - travel_tracker.py
      - planning/
        - cognition.py
        - events.py
        - goal.py
        - planning.py
      - rendering/
        - render_frame.py
      - world/
        - entity_blueprint.py
        - position.py
        - world.py
        - world_definition.py
        - world_entity.py
    - ports/
      - cognition_provider.py
      - conflict_coordinator.py
      - dialogue_coordinator.py
      - event_logger.py
      - pathfinder.py
      - presenter.py
      - vector_memory_store.py
    - services/
      - perception_service.py
      - precondition_evaluator.py
  - infrastructure/
    - adapters/
      - chroma_memory_adapter.py
    - cognition/
      - adapters/
        - instructor_adapter.py
      - health/
        - ollama_health_checker.py
    - container.py
    - logging/
      - buffering_event_logger.py
      - jsonl_logger.py
    - pathfinding/
      - astar.py
    - presentation/
      - console/
        - console_presenter.py
      - qt/
        - simulation_worker.py
        - thought_stream_window.py
    - repositories/
      - json_blueprint_repository.py
      - json_world_repository.py
  - tools/
    - world_editor.py
- tests/
  - integration/
  - unit/
    - application/
      - cognition/
      - coordination/
      - execution/
      - lifecycle/
      - movement/
      - rendering/
    - domain/
      - agent/
      - coordination/
      - planning/
      - rendering/
      - services/
    - infrastructure/
      - adapters/
      - cognition/
      - pathfinding/
      - presentation/

---

## 4. Validierungskriterien & Definition of Done

1. Alle bestehenden und neuen Tests laufen unter python3 -m pytest fehlerfrei durch.
2. Beim Aufruf ohne aktiven Ollama-Server oder bei fehlendem Modell bricht das System vor Beginn des ersten Takts mit einem RuntimeError ab.
3. Der ConsolePresenter liefert über generate_frame deterministische RenderFrame-Objekte mit Zeitstempel und Taktnummer.
4. Der FrameBufferService puffert diese Frames thread-sicher für nachgelagerte Schichten.
5. Das PyQt6-Fenster spiegelt den inneren Kognitionszustand der Agenten live und thread-entkoppelt wider.
6. Im Lebenszyklus-Szenario navigiert der Agent am Folgetag nachweislich über Vektorspeicher-Erinnerungen autonom zu einer Nahrungsquelle.