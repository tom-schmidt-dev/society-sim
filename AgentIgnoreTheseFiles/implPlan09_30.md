# society_sim: Meilenstein 3 – Kognitive Autonomie, Vektor-Gedächtnis & SLM-Pipeline (Stand: 30.09.2026)

## 1. Systemübersicht & Modell-Allokation

Die Architektur trennt strikt zwischen deterministischer Physik (Gitter, A*, Zwei-Phasen-Commit) und kognitiver Entscheidungsfindung:

| Komponente / Modell | Schicht & Rolle | Ausführungszeitpunkt | Latenz / Anforderung |
| :--- | :--- | :--- | :--- |
| **Deterministische Physik** | Application (`MovementOrchestrator`) | Phase 2 des Taktzyklus | < 1 ms pro Agent (kollisionsfrei) |
| **SLM (1B–3B, QLoRA)** | Infrastructure (`InstructorCognitionAdapter`) | Phase 1 des Taktzyklus | 50–150 ms (Pydantic-validierte Sub-Goals) |
| **LLM (8B, z. B. Llama 3)** | Application / Infrastructure | Asynchron bei Begegnungen (`is_thinking`) | 1–3 s (semantische Verhandlung & Dialoge) |
| **LLM (8B, Batch)** | Application (`MemoryConsolidationService`) | Nachtzyklus (Schlafphase) | Batch-Verarbeitung der Tagesereignisse |
| **Bi-Encoder Embedding** | Infrastructure (`SentenceTransformerAdapter`) | Tag (Bedarfsabfrage) & Nacht (Indexierung) | < 10 ms (ChromaDB Top-k Retrieval) |
| **JsonlEventLogger** | Infrastructure (`src/infrastructure/logging/`) | Laufend pro Takt / Event | Append-Only für Audit, SFT-Generierung & Nacht-Synthese[cite: 2, 3] |

---

## 2. Phase A: Multi-Bedürfnis-Dynamik, Gedankenverlauf & Inspektion

### Teilschritt A.1: Paralleler Vitalstoffwechsel (`NeedService`)
* **Ziel:** Ablösung des isolierten Einzelschwellenwerts durch konkurrierende Bedürfnisse.
* **Modellierung (`Agent` & `NeedService`):**
  * Vitalwerte als normalisierte Fließkommazahlen im Intervall [0.0, 1.0]: `hunger`, `thirst`, `energy`.
  * Konfigurierbare Zuwachsraten pro Takt (Metabolismus).
  * Dringlichkeitsbewertung über eine gewichtete Priorisierungsfunktion zur Auswahl des dominanten Handlungsantriebs:
    $$\text{priority} = w_{\text{need}} \cdot f(\text{level})$$
* **Aktionserweiterung:**
  * Erweiterung von `ActionExecutor` um Sub-Goals für `drink` und `rest`.
  * Einbindung von Vorbedingungen in `PreconditionEvaluator`.
* **Tests:** `tests/unit/test_need_service.py` (Metabolismus-Zyklen, Schwellenwert-Triggers, Prioritätsentscheidungen).

### Teilschritt A.2: Transparenter Gedankenverlauf & Kognitions-Snapshots
* **Ziel:** Vollständige Nachvollziehbarkeit des inneren Zustands jedes Agenten pro Takt ohne LLM-Overhead.
* **Datenstruktur (`AgentCognitiveSnapshot`):**
  * `dominant_need`: Name und Wert des dringendsten Bedürfnisses.
  * `primary_goal`: Aktives Hauptziel aus dem Zielstack.
  * `active_subgoal`: Aktives atomares Teilziel (`move_to`, `explore`, `consume`, `drink`, `rest`).
  * `perceived_obstacle`: Eventuelle Blockadeursache (z. B. Agenten-ID, Koordinate).
  * `intended_strategy`: Gewählte Lösungsstrategie (z. B. Verhandlung, Nischen-Ausweichen, Warten).
  * `formatted_thought`: Deterministisch erzeugter Volltextsatz (Template Engine).
* **Inspektion & Logging:**
  * Methode `agent.get_cognitive_snapshot() -> AgentCognitiveSnapshot`.
  * Integration in `JsonlEventLogger`: Protokollierung wesentlicher Denk- und Strategiewechsel als `SimulationEvent`[cite: 3].
  * Integration in `ConsolePresenter`: Ausgabe des formatierten Gedankenverlaufs im Konsolen-Interface.

---

## 3. Phase B: Tag-Nacht-Zyklus, Vektorspeicher & Speicherpartitionierung

### Teilschritt B.1: Takt-Orchestrierung des Tag-Nacht-Wechsels
* **Ziel:** Trennung von aktiver Simulationszeit und rechenintensiver Konsolidierungszeit.
* **Architektur (`DayNightService`):**
  * Parameter: Zyklenlänge (z. B. 100 Takte Tag, 20 Takte Nacht).
  * Umschaltung:
    * Tag: Reguläre Ausführung von Phase 1 (Kognition) und Phase 2 (Bewegung).
    * Nacht: Alle Agenten erhalten Status `is_sleeping = True`. Phase 2 pausiert vollständig.
* **Integration in `SimulationEngine`:** Einbindung des Phasenübergangs in die Takt-Schleife.

### Teilschritt B.2: Vektorspeicher-Architektur & Metadaten-Partitionierung
* **Architektur-Entscheidung:** Eine einzige, geteilte ChromaDB-Collection für alle Agenten zur Minimierung von HNSW-Index-Overhead und Datei-Handles, abgesichert durch zwingende Metadaten-Filterung auf Adapter-Ebene.
* **Domain Port (`IVectorMemoryStore`):**
  ```python
  def add_memories(self, agent_id: str, memories: list[str], metadatas: list[dict[str, Any]]) -> None: ...
  def retrieve_relevant(self, agent_id: str, query: str, limit: int = 3, metadata_filter: Optional[dict[str, Any]] = None) -> list[str]: ...
  

Alles abgeschlossen. Empfohlene nächste Schritte

Schritt 1 (Absicherung): Erstellung von tests/integration/test_nightly_consolidation_lifecycle.py zur Validierung des End-to-End-Zyklus über die SimulationEngine.

Schritt 2 (Phase C: Retrieval & Kognitive Nutzung):

Teilschritt C.1: Integration des semantischen Retrievals in PlanDecompositionService / CognitionOrchestrator (Abruf relevanter Vergangenheitserfahrungen bei Zielkonflikten oder Ressourcenmangel).

Teilschritt C.2: Prompt-Injektion der abgerufenen Erinnerungen in die Inferenz-Pipeline.