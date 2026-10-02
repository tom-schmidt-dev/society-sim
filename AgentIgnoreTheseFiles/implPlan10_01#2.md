# implPlan10_02.md: Autonome LLM-Korridorverhandlung, Soziale Reflexion & Dual-Model-Architektur

## 1. Systemübersicht & Dual-Model-Vision

* **Dual-Model-Konzept (SLM + LLM):**
  * **Großes Modell (LLM):** Fokussiert auf hochwertige sprachliche Formulierungen, offene Verhandlungen, Argumentation und differenzierte Meinungsbildung[cite: 22].
  * **Kleines Modell (SLM):** Soll künftig parallel operieren und hochfrequente, taktische Entscheidungen (Blockadeauflösung, Wegeprüfung, operative Sub-Goals) mit minimaler Latenz abarbeiten[cite: 21, 22].
  * **Data Flywheel:** Die strukturierten JSONL-Ereignislogs der Simulation dienen unmittelbar als instruktionsbasiertes Trainingsset (Supervised Fine-Tuning / SFT), um das SLM auf Basis der Entscheidungen des großen Modells zu trainieren[cite: 2].
* **Entkoppelte Darstellungsschicht & Vektorisierungs-Performance:**
  * Geplant ist eine nachgelagerte grafische Darstellungsschicht auf Basis des `ConsolePresenter` / `FrameBufferService`, die mit gedrosselter, gleichmäßiger Geschwindigkeit läuft[cite: 1, 2].
  * Die soziale Meinungsbildung wird zunächst direkt am Ende einer Konversation am Tag vektorisiert und in ChromaDB geschrieben[cite: 2, 27].
  * Sollte sich im Zusammenspiel mit der Darstellungsschicht zeigen, dass die Tages-Inferenz den Simulationsfluss zu stark ausbremst, kann das Vektorisieren architektonisch modular in die nächtliche Schlafphase (`MemoryConsolidationService`) verlagert werden[cite: 1, 2, 3].

---

## 2. Architektonische Grundprinzipien

* **Clean Architecture & Hexagonale Entkopplung:**
  * Strikte Trennung von Domäne (`domain`), Anwendungslogik (`application`) und Infrastruktur (`infrastructure`)[cite: 1, 2, 3].
  * Domänenmodelle und Applikationsservices bleiben vollständig frei von Abhängigkeiten zu Inferenz-Frameworks (Instructor, LiteLLM) oder Speicher-Backends (ChromaDB)[cite: 2, 21, 27].
* **SOLID-Prinzipien:**
  * *Single Responsibility Principle (SRP):* Trennung von Sitzungsverfolgung (`DialogueSessionManager`), Dialoghistorie (`DialogueHistory`), Verhandlungskoordination (`DialogueCoordinator`) und Aktionsausführung (`ActionExecutor`)[cite: 20, 23, 25].
  * *Open/Closed Principle (OCP):* Erweiterung des Handlungs- und Reflexionsraums über typisierte Pydantic-Schemata ohne Eingriffe in die Simulationsschleife[cite: 3, 21].
  * *Dependency Inversion Principle (DIP):* Koordination erfolgt ausschließlich gegen Schnittstellen (`ICognitionProvider`, `IVectorMemoryStore`, `IEventLogger`)[cite: 2, 3, 27].
* **GoF-Entwurfsmuster:**
  * *Strategy Pattern:* Austauschbare Konfliktlösungsstrategien (deterministische Korridor-FSM vs. autonome LLM-Verhandlung)[cite: 7].
  * *Observer Pattern:* Strukturierte, verlustfreie Protokollierung aller Kognitions- und Reflexionsschritte über `IEventLogger`[cite: 2, 3].
  * *Facade Pattern:* `SimulationEngine` als einheitlicher Einstiegspunkt für den zweiphasigen Simulationszyklus[cite: 3].
* **Test-Driven Development (TDD):**
  * Vorab-Definition fehlschlagender Unit- und Integrationstests vor jeder Code-Anpassung.
  * Regressionstest-Absicherung gegen bestehende Szenarien[cite: 1].

---

## 3. Detaillierte Teilschritte

### Teilschritt 1: Konfiguration & Deaktivierung der deterministischen Korridor-FSM
* **Ziel:** Ermöglichung der rein sprachmodellbasierten Verhandlung in Engpässen ohne automatischen FSM-Bypass[cite: 7].
* **Komponenten:**
  * `src/infrastructure/container.py`:
    * Parameter `enable_deterministic_corridor: bool = True` in `ApplicationContainer.build()` exponieren[cite: 2].
    * Standardwert für Showcase- und Verhandlungsszenarien auf `False` konfigurierbar machen[cite: 2].
  * `src/application/simulation_engine.py`:
    * Durchreichung des Parameters an den internen `ConflictCoordinator`[cite: 3, 7].
  * `src/application/services/coordination/conflict_coordinator.py`:
    * Bei `enable_deterministic_corridor=False` werden frontale Korridorblockaden nicht durch die deterministische FSM abgefangen, sondern initiieren eine reguläre `TalkAction` (`request_yield` oder `talk`)[cite: 7, 21].
* **Voraussetzungsprüfung für Teilschritt 1:**
  * Vergewissere dich vor Teilschritt 2, dass bei deaktivierter FSM blockierte Agenten zuverlässig Nachrichten in die `staging_inbox` des Partners einspeisen und keine unkontrollierten Ausweichpfade generiert werden[cite: 7, 23].

---

### Teilschritt 2: Offene Verhandlung & Gleitendes Dialogfenster (Sliding Window)
* **Ziel:** Zulassen beliebig langer Diskussionen und Streits ohne Rundenbegrenzung sowie Begrenzung des LLM-Kontexts auf die letzten 20 Beiträge[cite: 20].
* **Komponenten:**
  * `src/application/services/coordination/dialogue_session_manager.py`:
    * `max_dialogue_turns: Optional[int] = None` zulassen[cite: 20].
    * `is_turn_limit_exceeded()` liefert bei `None` immer `False`[cite: 20].
    * Speicherung und Bereinigung von `conversation_summaries: dict[tuple[str, ...], str]` integrieren[cite: 20].
  * `src/application/services/coordination/dialogue_history.py`:
    * Methode `get_recent_records_for_pair(agent_a_id: str, agent_b_id: str, limit: int = 20) -> list[DialogueRecord]` implementieren[cite: 25, 26].
    * Methode `get_recent_formatted_for_pair(agent_a_id: str, agent_b_id: str, limit: int = 20) -> list[str]` implementieren[cite: 25, 26].
  * `src/application/services/coordination/dialogue_coordinator.py`:
    * Übergabe von `conversation_summary` (initial leerer String) an das `context`-Dictionary[cite: 11, 20].
    * Begrenzung von `recent_dialogues` auf maximal 20 Einträge via Sliding Window[cite: 11].
    * Übergabe der Agentenattribute `assertiveness` und `charisma` in den Kontext[cite: 11, 24].
  * `src/infrastructure/cognition/adapters/instructor_adapter.py`:
    * System-Prompt anpassen: Verhaltensregeln basierend auf `assertiveness` und `charisma` verankern[cite: 22].
    * Entfernung von Aufforderungen zur Gesprächskürze; Zulassung von Beharren (`reject`) und Argumentationen[cite: 21, 22].
    * Einbettung von `conversation_summary`, falls vorhanden[cite: 20, 22].
* **Voraussetzungsprüfung für Teilschritt 2:**
  * Vergewissere dich vor Teilschritt 3, dass Teilschritt 1 aktiv ist und Unit-Tests belegen, dass wiederholte `reject`-Intents über mehr als 5 Takte hinweg stabil im Dialog bleiben, ohne Exceptions oder vorzeitiges Ausweichen auszulösen[cite: 11, 21].

---

### Teilschritt 3: Soziale Reflexion & Direkte Vektorisierung (Daytime mit Nightly-Option)
* **Ziel:** Nach Abschluss einer Konversation bildet der Agent eine zweiteilige Meinung und speichert diese unmittelbar in ChromaDB; Vorbereitung auf spätere nächtliche Verlagerung[cite: 1, 2, 11, 27].
* **Komponenten:**
  * `src/domain/models/planning/cognition.py`:
    * Pydantic-Schema `SocialReflection`:
      * `assessment: str` (Einschätzung der Persönlichkeit des Gegenübers)
      * `progression_summary: str` (Kurzer Verlauf und Resultat des Gesprächs)
  * `src/domain/ports/cognition_provider.py`:
    * Definition der Methode `reflect_on_dialogue(self, context: dict[str, Any]) -> SocialReflection`.
  * `src/infrastructure/cognition/adapters/instructor_adapter.py`:
    * Implementierung von `reflect_on_dialogue()` mit strukturierter Inferenz[cite: 22].
  * `src/application/services/coordination/dialogue_coordinator.py`:
    * Trigger bei Einigung (`accept`), Verabschiedung (`is_farewell`) oder Dialogende (`EndDialogueAction`)[cite: 11, 23].
    * Generierung des formatierten Eintrags:
      `"[Soziale Interaktion mit {partner_name}]: {assessment}. Verlauf: {progression_summary}"`[cite: 11, 27].
    * Direktes Schreiben via `self._vector_memory_store.add_memories(...)` mit Metadaten `{"category": "social", "partner_id": partner_id, "tick": current_tick}`[cite: 11, 27].
    * **Architektur-Hinweis im Code:** Dokumentieren, dass dieser Vektorisierungsschritt tagsüber ausgeführt wird, jedoch bei Performance-Engpässen im Kontext der Darstellungsschicht modular in die Nachtphase (`MemoryConsolidationService`) ausgelagert werden kann[cite: 1, 2].
* **Voraussetzungsprüfung für Teilschritt 3:**
  * Vergewissere dich vor Teilschritt 4, dass Teilschritt 2 fehlerfrei arbeitet und nachweisbar ein valides `SocialReflection`-Objekt aus den letzten Dialogdaten abgeleitet werden kann[cite: 22].

---

### Teilschritt 4: Strukturiertes SLM-Fine-Tuning-Logging
* **Ziel:** Vollständige Serialisierung der Kognitions- und Verhandlungsentscheidungen im JSONL-Ereignisstrom als strukturierter Trainingsdatensatz für das kleinere Entscheidungsmodell[cite: 2].
* **Komponenten:**
  * `src/application/services/coordination/dialogue_coordinator.py`:
    * Ausgabe des Events `social_reflection_completed` an `self._logger.log()`[cite: 11].
    * Payload-Struktur (vollständiges SFT-Prompt-Completion-Paar):
      * `agent_id: str`[cite: 26]
      * `partner_id: str`[cite: 26]
      * `dialogue_turns: int`[cite: 20]
      * `input_context: dict[str, Any]` (bereinigter Eingabekontext aus Distanzen, Persönlichkeitswerten und Dialoghistorie)[cite: 11, 22]
      * `model_used: str`[cite: 2]
      * `assessment: str`
      * `progression_summary: str`
      * `persisted_memory: str`[cite: 27]
      * `duration_ms: float`
* **Voraussetzungsprüfung für Teilschritt 4:**
  * Vergewissere dich vor Teilschritt 5, dass das Event-Schema vollständig typisiert ist und das JSONL-Protokoll valide Datensätze für nachgelagerte Fine-Tuning-Pipelines liefert[cite: 2].

---

### Teilschritt 5: Demonstrationsskript & Validierung
* **Ziel:** Verifikation des vollständigen Verhandlungs- und Meinungsbildungszyklus unter realer LLM-Inferenz[cite: 1, 5].
* **Komponenten:**
  * `main_dialogue_showcase.py`:
    * Initialisierung des Containers mit `enable_deterministic_corridor=False`[cite: 2, 5].
    * Testlauf mit Alice (`assertiveness=0.85`) und Bob (`assertiveness=0.2`)[cite: 5].
    * Testlauf mit symmetrisch dominanten Agenten (beide `assertiveness=0.9`) zur Verifikation von anhaltenden Streitgesprächen.
* **Voraussetzungsprüfung für Teilschritt 5:**
  * Vergewissere dich, dass alle Teilschritte 1 bis 4 implementiert sind und die Test-Suite ohne Regressionen durchläuft.

---

## 4. Definition of Done (DoD)

* Alle Tests unter `tests/unit/application/coordination/` und `tests/integration/` laufen unter `python3 -m pytest` fehlerfrei durch.
* Bei deaktivierter Korridor-FSM kommunizieren Agenten autonom über das LLM; es kommt zu keinen unaufgelösten Swap-Deadlocks vor der Engstelle[cite: 4, 7].
* Der Dialog bleibt bei wechselseitiger Ablehnung beliebig lange aktiv; die Prompt-Historie wird strikt auf maximal 20 Einträge beschränkt[cite: 20].
* Nach Abschluss der Verhandlung ist ein strukturierter sozialer Gedächtniseintrag im Vektorspeicher vorhanden und für nachfolgende Takte abrufbar[cite: 11, 27].
* Das JSONL-Protokoll enthält das vollständige Ereignis `social_reflection_completed` zur direkten Weiternutzung für das Modell-Finetuning[cite: 2].
* Der Code enthält explizite Architekturhinweise für die optionale Umstellung auf nächtliche Vektorisierung[cite: 1, 2].