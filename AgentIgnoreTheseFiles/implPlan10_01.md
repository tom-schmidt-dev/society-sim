# implPlan10_01.md: Frame-Pufferung, Live-Kognitions-Stream & Autonome Zielableitung

## 1. Systemübersicht & Architektur-Ziele

* **Entkopplung der Taktraten (Producer-Consumer-Muster):**
  * Simulation läuft nativ ungebremst (schnelle Takte bei Physik, Pausen während LLM-Inferenz).
  * `ConsolePresenter` gibt den Live-Zustand direkt im Terminal aus.
  * `FrameBufferService` puffert unveränderliche `RenderFrame`-Instanzen thread-sicher in einer FIFO-Warteschlange.
  * Zukünftige/externe Darstellungsschichten konsumieren Frames mit konfigurierbarer Maximalgeschwindigkeit und glätten Wartezeiten.
* **Fail-Fast Inferenz-Prüfung:**
  * Verbindungs- und Modellprüfung gegen Ollama vor Takt 1.
  * Abbruch per `RuntimeError` ohne Start der Simulation, falls Ollama offline ist oder das Modell fehlt.
* **Separates Kognitions-Fenster (PyQt6):**
  * Live-Visualisierung der inneren Gedankenprozesse, Bedürfnisse und Zielhierarchien synchron zur Simulation.
  * Thread-Entkopplung: PyQt6 im Haupt-Thread, `SimulationEngine` (`asyncio`) im Worker-Thread.
* **Autonome Zielableitung:**
  * Agenten generieren Ziele selbstständig auf Basis von Vitalbedürfnissen (`NeedService`), Vektorerinnerungen (`IVectorMemoryStore`), Umgebungsdaten (`PerceptionService`) und Charaktereigenschaften.
* **Strukturierte Modularisierung:**
  * Vermeidung flacher Ordner durch thematische Unterordner (`rendering/`, `qt/`, `console/`, `health/`).
  * Alphabetische Sortierung innerhalb der Ordner.

---

## 2. Detaillierte Teilschritte

### Teilschritt 1: Pre-Flight-Validierung für Inferenz (Ollama)
* **Ziel:** Sofortiger Abbruch bei fehlender Inferenz-Bereitschaft ohne Mocking.
* **Komponenten:**
  * Modul: `src/infrastructure/cognition/health/ollama_health_checker.py`
  * Methode: `check_ollama_status(api_base: str, model_name: str) -> None`
  * Führt synchronen HTTP-Request gegen `api_base/api/tags` aus.
  * Prüft Erreichbarkeit und Vorhandensein des konfigurierten Modells.
  * Wirft bei Fehlschlag `RuntimeError("Ollama-Dienst nicht erreichbar unter {api_base} oder Modell '{model_name}' nicht geladen.")`.
* **Integration:** Aufruf in `ApplicationContainer.build()` vor Instanziierung der Kognitionsdienste.
* **Tests:** `tests/unit/cognition/test_ollama_health.py`.

---

### Teilschritt 2: Domänenmodell & Pufferung für Render-Frames
* **Ziel:** Konsistente, zeitstempelbasierte Kapselung visueller Taktzustände.
* **Komponenten:**
  * Domänenmodell: `src/domain/models/rendering/render_frame.py`
    * Value Object `RenderFrame`:
      * `tick: int`
      * `timestamp: float`
      * `grid_matrix: list[list[str]]` (deterministische Zeichenmatrix)
      * `entities: list[dict[str, Any]]` (ID, Typ, Position, Symbol)
      * `dialogues: list[str]`
      * `snapshots: list[AgentCognitiveSnapshot]`
  * Dienst: `src/application/services/rendering/frame_buffer_service.py`
    * Thread-sichere FIFO-Warteschlange (`queue.Queue[RenderFrame]`) mit konfigurierbarer Maximalkapazität.
    * Methoden: `push_frame(frame: RenderFrame) -> None`, `pop_frame(timeout: Optional[float]) -> Optional[RenderFrame]`, `clear() -> None`.
* **Erweiterung `ConsolePresenter`:**
  * Verschiebung nach `src/infrastructure/presentation/console/console_presenter.py`.
  * Trennung in:
    * `generate_frame(...) -> RenderFrame`: Reine Transformation von Gitter und Entitäten in Zeichen.
    * `render(...)`: Ausgabe des Frames im Terminal und optionale Übergabe an `FrameBufferService`.
* **Tests:**
  * `tests/unit/rendering/test_render_frame_generation.py`
  * `tests/unit/rendering/test_frame_buffer_service.py`

---

### Teilschritt 3: Live-Kognitions-Stream-Fenster (PyQt6)
* **Ziel:** Visualisierung der inneren Gedankenprozesse in einem separaten Fenster.
* **Komponenten:**
  * Modul: `src/infrastructure/presentation/qt/thought_stream_window.py`
    * `ThoughtStreamWindow(QMainWindow)`:
      * Bereiche pro Agent mit automatischer Nachführung (Auto-Scroll).
      * Anzeige von Vitalwerten (Hunger, Durst, Energie) und Zielen.
  * Modul: `src/infrastructure/presentation/qt/simulation_worker.py`
    * `SimulationWorker(QThread)`:
      * Führt den `asyncio`-Event-Loop der `SimulationEngine` aus.
      * Sendet `frame_emitted = pyqtSignal(object)` bei Taktende an das Hauptfenster.
* **Tests:** `tests/unit/presentation/test_qt_thought_window_signals.py`.

---

### Teilschritt 4: Autonome Zielableitung auf Basis von Gedächtnis und Bedürfnissen
* **Ziel:** Vollständige Selbstständigkeit der Agenten bei Priorisierung und Planung ohne externe Zielvorgaben.
* **Komponenten:**
  * `PlanDecompositionService`:
    * Vektorabfrage über `IVectorMemoryStore` basierend auf dem dominanten Bedürfnis (z. B. "Wo gab es zuletzt Wasser/Nahrung?").
    * Berücksichtigung von Persönlichkeitsmerkmalen (`charisma`, `assertiveness`, `curiosity`).
  * `CognitionOrchestrator`:
    * Wenn der Zielstack leer ist und Vitalwerte Schwellenwerte überschreiten: Autonome Initiierung von `decompose_need_goal(...)`.
    * Falls keine Position im Gedächtnis vorhanden ist: Übergang in Explorationsmodus (`frontier_explorer`).
* **Tests:** `tests/integration/test_autonomous_goal_generation.py`.

---

### Teilschritt 5: Demonstrations- und Ausführungsskripte

* **Skript 1 (`src/main_memory_lifecycle.py`):**
  * **Szenario:** Einzelner Agent in einer Umgebung mit verstreuten Ressourcen.
  * **Ablauf:**
    1. Agent startet ohne Ziel, Vitalbedürfnisse steigen über Takte an.
    2. Agent erkundet Terrain, lokalisiert Nahrungsquelle und verzehrt Nahrung.
    3. Nachtphase (`DayNightService`): Agent schläft, `MemoryConsolidationService` indexiert Tagesereignisse in ChromaDB.
    4. Folge-Tag: Agent wird erneut hungrig, ruft den Fundort via Vektor-Retrieval ab und navigiert zielgerichtet dorthin.
  * **Darstellung:** Live-Terminal-Ausgabe + paralleles PyQt6-Kognitionsfenster.

* **Skript 2 (`src/main_dialogue_showcase.py`):**
  * **Szenario:** Zwei Agenten mit unterschiedlichen Eigenschaften treffen im Engpass aufeinander.
  * **Ablauf:**
    1. Agent A (hohe `assertiveness`) und Agent B (kooperativ) navigieren gegeneinander.
    2. Verhandlungsprozess via LLM bei Sichtkontakt / Blockade.
    3. Autonome Einigung auf Nischen-Ausweichen.
    4. Fortsetzung der Pfade nach Freigabe des Korridors.
  * **Darstellung:** Live-Terminal-Ausgabe + Dialogprotokoll + detaillierter Gedankenstrom im PyQt6-Fenster.

---

## 3. Dateistruktur-Übersicht