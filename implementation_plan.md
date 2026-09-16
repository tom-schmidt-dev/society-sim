# Implementierungsplan

## Plan 1: Bugfixes und Architektur-Refactoring

Dieser Plan beseitigt alle identifizierten Syntaxfehler, Laufzeit-Bugs und Architekturverletzungen (DRY, Kapselung). Nach jedem Schritt werden die Voraussetzungen für den Folgeschritt geprüft und sichergestellt.

* **Schritt 1: Bereinigung des Domänen-Datenmodells (`message.py`)**[cite: 11]
  * **Maßnahme:** Entfernen der redundanten Deklaration `correlation_key: Optional[str] = None` in `IncomingMessage`[cite: 11].
  * **Voraussetzungsprüfung für Schritt 2:** Das Datenmodell `IncomingMessage` ist syntaktisch valide und unverändert abwärtskompatibel zu allen Aufrufern[cite: 11].

* **Schritt 2: Erweiterung von `EvasionResult` (`evasion_finder.py`)**[cite: 4]
  * **Maßnahme:** Hinzufügen des Feldes `path: list[Position]` zu `EvasionResult`[cite: 4].
  * **Begründung:** Der Ausweichpfad wird in `find_nearest_evasion_tile` via `_reconstruct_path` bereits vollständig ermittelt[cite: 4]. Die direkte Rückgabe im Ergebnisobjekt verhindert redundante A*-Neuberechnungen und behebt den Laufzeitfehler in `simulation_engine.py` (Zeile 260)[cite: 4, 8].
  * **Voraussetzungsprüfung für Schritt 3:** `EvasionResult` liefert Zielkachel, Junction-Kachel und den konkreten Pfad[cite: 4]. Damit stehen dem aufrufenden Service alle Daten zur Pfadzuweisung zur Verfügung.

* **Schritt 3: Zentralisierung der Ausweichlogik & Kapselung (`action_executor.py`)**[cite: 5]
  * **Maßnahme 1:** Sichtbarkeit von `_create_talk_message` auf öffentlich (`create_talk_message`) anpassen[cite: 5].
  * **Maßnahme 2:** Neue Methode `execute_evasion(agent, partner, blocked_pos, all_entities, incident_id, thought)` implementieren:
    * Kaskadierte Nischensuche via `EvasionFinder`[cite: 2, 3, 4].
    * Fallunterscheidung: Bekannte Nische vs. Frontier-Exploration vs. Kaskaden-Erschöpfung[cite: 2, 3].
    * Goal-Stack-Manipulation (`push_goal`) und Pfadzuweisung[cite: 1, 2, 3].
    * Benachrichtigung des Partners mit Pfadübermittlung[cite: 2, 3].
  * **Voraussetzungsprüfung für Schritt 4:** Der `ActionExecutor` stellt eine geschlossene Schnittstelle für Ausweichvorgänge bereit[cite: 5]. Weder `ConflictCoordinator` noch `DialogueCoordinator` müssen direkt auf interne Hilfsmethoden zugreifen oder Ziel-Stacks für Ausweichmanöver manipulieren[cite: 2, 3, 5].

* **Schritt 4: Entschlackung der Koordinatoren (`conflict_coordinator.py` & `dialogue_coordinator.py`)**[cite: 2, 3]
  * **Maßnahme:** Entfernen der redundanten Nischensuch- und Benachrichtigungsblöcke in beiden Koordinatoren[cite: 2, 3]. Ersatz durch den Aufruf von `self._action_executor.execute_evasion(...)`[cite: 2, 3, 5].
  * **Voraussetzungsprüfung für Schritt 5:** Keine DRY-Verletzungen mehr zwischen Dialog- und Blockadekoordination[cite: 2, 3]. Beide Klassen beschränken sich auf die Orchestrierung der Inferenz und delegieren die physische Ausführung an den `ActionExecutor`[cite: 2, 3, 5].

* **Schritt 5: Korrektur der Signal- und Pfad-Updates (`simulation_engine.py`)**[cite: 8]
  * **Maßnahme 1:** In Zeile 260 `agent.assign_path(res.path_to_niche)` auf das neue Feld `res.path` korrigieren[cite: 8].
  * **Maßnahme 2:** Validierung der Clearance-Bedingung in Fall 4:
    * Überprüfung, dass $L_1(\text{pos}_B, \text{junction}) \ge 2$ und $\text{junction} \notin \text{path}_B$ erfüllt sind[cite: 8].
    * Sicherstellen, dass das Absenden von `is_courtesy=True` durch Agent B atomar das Verlassen der Nische bei Agent A triggert[cite: 8].
  * **Voraussetzungsprüfung für Schritt 6:** Die SimulationEngine kompiliert fehlerfrei, enthält keine ungültigen Attributzugriffe mehr und setzt das Handshake-Protokoll deterministisch um[cite: 8].

* **Schritt 6: Validierungslauf des Basissystems (`main2.py`)**[cite: 32]
  * **Maßnahme:** Ausführen des Korridorszenarios über 600 Ticks und Überprüfung der JSONL-Logs auf vollständige Deadlock-Auflösung ohne Ausnahmen[cite: 31, 32].

---

## Plan 2: Geplante Features und funktionale Erweiterungen

Dieser Plan baut auf dem bereinigten Fundament auf und implementiert die erweiterten kognitiven und sensorischen Fähigkeiten.

* **Schritt 1: Dynamische Anpassung des Kognitions-Prompts (`instructor_adapter.py`)**[cite: 28]
  * **Maßnahme:** System-Prompt in `resolve_blockage` erweitern, sodass bekannte unbewegliche oder stumme Objekte (`is_conversational=False`) bei erneuter Begegnung deterministisch umgangen und nicht erneut angesprochen werden[cite: 28].
  * **Voraussetzungsprüfung für Schritt 2:** Kognitive Fehlversuche bei bekannten statischen Entitäten werden modellseitig minimiert[cite: 28].

* **Schritt 2: Konsistente Pfadplanung über mentale Karten (`AStarPathfinder`)**[cite: 18, 29]
  * **Maßnahme:** Sicherstellen, dass bei jeder Routenneuberechnung ausschließlich die `mental_map` des jeweiligen Agenten als Grid übergeben wird, niemals das globale `WorldGrid`[cite: 1, 2, 3, 8, 16, 18, 29].
  * **Voraussetzungsprüfung für Schritt 3:** Fog-of-War-Integrität ist vollständig gewahrt; kein Agent besitzt globales Kartenwissen[cite: 18, 30].

* **Schritt 3: Taktweise Gedächtnis-Konsolidierung (`PerceptionService` & `AgentMemory`)**[cite: 9, 27]
  * **Maßnahme:** Automatisches Aktualisieren von `EntityFact` für alle Entitäten im Sichtfeld innerhalb von `SimulationEngine._update_agent_perception` (Quelle: `PerceptionSource.VISUAL`)[cite: 8, 9, 27].
  * **Voraussetzungsprüfung für Schritt 4:** Gedächtniseinträge altern und aktualisieren sich kontinuierlich ohne manuelle Kognitionsaufrufe[cite: 9].

* **Schritt 4: Aufbau der isolierten Test-Suite**
  * **Maßnahme 1:** Unit-Tests mit Mocks für `ICognitionProvider` zur Verifikation der Kaskadierungsentscheidungen (Stufe 1 vs. 2 vs. 3)[cite: 4, 25].
  * **Maßnahme 2:** Integrationstests für den Handshake: Stopp an Junction $\rightarrow$ Nischenhalt $\rightarrow$ Passage von Agent B $\rightarrow$ Courtesy-Signal $\rightarrow$ Weiterreise von Agent A[cite: 8].
  * **Voraussetzungsprüfung für den Abschluss:** Vollständige Testabdeckung aller deterministischen Leitplanken ohne Abhängigkeit von einer laufenden LLM-Instanz.
