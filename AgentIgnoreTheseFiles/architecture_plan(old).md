# Architektur und Konzeption

## 1. Modularität & Schichtenarchitektur (Hexagonal / Clean Architecture)

### Domain-Layer (Kern)
* Kapselt Zustand und Fachlogik (`Agent`, `Position`, `WorldGrid`, `Goal`, `AgentMemory`, `AgentMentalMap`, `WorldEntity`) vollständig isoliert von Frameworks, I/O und Sprachmodellen[cite: 1, 9, 12, 13, 15, 16, 18].

### Application-Layer (Orchestrierung)
* Koordiniert Abläufe und Zustandsübergänge (`SimulationEngine`, `GoalService`, `ConflictCoordinator`, `DialogueCoordinator`, `DialogueSessionManager`, `DialogueHistory`, `EvasionFinder`, `ActionExecutor`)[cite: 1, 2, 3, 4, 5, 6, 7, 8].

### Infrastructure-Layer (Adapter)
* Kapselt externe Systeme:
  * Kognition / LLM: `InstructorCognitionAdapter` via LiteLLM[cite: 28].
  * Persistenz: `JsonlEventLogger` für strukturiertes JSONL-Event-Logging[cite: 31].
  * Pfadfindung: `AStarPathfinder` für A*-Suchen auf Graphen[cite: 29].
  * Präsentation: `ConsolePresenter` für Terminalausgabe mit Nebel des Krieges (Fog of War)[cite: 30].

### Entkopplung über Abstraktionen (Ports)
* `ICognitionProvider`, `IEventLogger`, `IPathfinder`, `IPresenter`, `IConflictCoordinator`, `IDialogueCoordinator` ermöglichen isoliertes Unit-Testing mittels Mocks ohne LLM-Inferenz[cite: 21, 22, 23, 24, 25, 26].

---

## 2. Deterministisch-Kreativer Handlungsspielraum

### Kreativer Spielraum (LLM als Kognitionskomponente)
* Bewertung unvollständiger Situationen und Formulierung interner Überlegungen (`thought`)[cite: 14].
* Formulierung sprachlicher Äußerungen (`message`, `final_message`) in Dialogen[cite: 14].
* Auswahl strategischer Handlungsoptionen (`wait`, `talk`, `reroute`, `abort`, `probe`, `inspect`) anhand injizierter Fakten[cite: 14, 28].

### Deterministische Leitplanken (Simulations-Engine)
* **Physik & Raum:** Kollisionen, Koordinatenvalidierung und Pfadfindung (A*) unterliegen unveränderlichen mathematischen Grid-Regeln[cite: 8, 16, 29].
* **Intent-Auflösung:** Kognition liefert abstrakte Intentionen (`intent_type="evade"`); das physische Zielfeld bestimmt die deterministische Kaskade (`EvasionFinder`)[cite: 4, 14].
* **Circuit-Breaker:** Feste Obergrenzen (`max_dialogue_turns`) im `DialogueSessionManager` erzwingen Handlungsabbrüche ohne weitere Inferenz[cite: 3, 6].
* **Schema-Härtung:** Pydantic-Modelle mit `@model_validator` normalisieren Modell-Ausgaben deterministisch vor Erreichen der Domäne[cite: 14].
* **Fakten-Injektion & Masking:** Bekannte Eigenschaften (`is_conversational=False`, unpassierbar) steuern die Prompt-Auswahl und schließen unzulässige Aktionen über Pydantic-Teilschemata strukturell aus[cite: 14, 28].

---

## 3. SOLID-Prinzipien

* **Single Responsibility Principle (SRP):**
  * `Agent`: Physische Attribute, Pfadführung, Ziel-Stack[cite: 13].
  * `AgentMemory`: Epistemisches Wissen, Glaubenssätze (`TypeBelief`), Trajektorienhistorie[cite: 9].
  * `GoalService`: Verwaltung und Manipulation des Ziel-Stacks inklusive Event-Logging[cite: 1].
  * `ConflictCoordinator`: Orchestrierung von Blockade-Inferenzen[cite: 2].
  * `DialogueCoordinator`: Orchestrierung von Konversations-Inferenzen[cite: 3].
  * `ActionExecutor`: Physische Ausführung von Aktionen, Nachrichtenversand, Nischen-Handling[cite: 5].
* **Open/Closed Principle (OCP):**
  * Erweiterung von Aktionen über diskriminierte Union-Typen (`Union[WaitAction, TalkAction, ...]`) ohne Modifikation der Kernschleife[cite: 14].
* **Liskov Substitution Principle (LSP):**
  * `Agent` erweitert `WorldEntity` typsicher; Engine operiert polymorph auf `list[WorldEntity]`[cite: 8, 13, 15].
* **Interface Segregation Principle (ISP):**
  * Aufgabenbezogene, schlanke Schnittstellen (`ICognitionProvider`, `IEventLogger`, `IPathfinder`)[cite: 23, 25, 26].
* **Dependency Inversion Principle (DIP):**
  * High-Level-Dienste hängen ausschließlich von Schnittstellen ab, nicht von konkreten Infrastruktur-Treibern[cite: 1, 2, 3, 8].

---

## 4. Epistemische Trajektorien- und Nischenkoordination

### Sensorik und Gedächtnis (`AgentMemory` & `EntityFact`)
* **Trajektorienerfassung:** `EntityFact` speichert `position_history` ($k=3$), `smoothed_velocity` ($\vec{v}_{smooth}$) und `standstill_ticks`[cite: 9].
* **Lazy Evaluation:** `get_projected_fact` berechnet Konfidenzabfall ($\tau = 5$ Takte) und lineare Extrapolation bei Sichtverlust nur bei Abruf[cite: 9].
* **Validierung gegen Mental Map:** Führt eine Extrapolation in bekannte Wände, wird die Konfidenz sofort auf $0.0$ gesetzt[cite: 9].

### Lokale Weltkarte (Fog of War)
* Jeder Agent pflegt eine eigene `AgentMentalMap`[cite: 13, 18].
* Unbekannte Bereiche (`TileKnowledge.UNKNOWN`) gelten optimistisch als passierbar[cite: 18].
* Pfadplanung und Nischensuche operieren primär auf der individuellen mentalen Karte[cite: 1, 2, 3, 4, 8].

### Kaskadierte Nischensuche (`EvasionFinder`)
* **Stufe 1:** BFS im Nahbereich ($L_1 \le 3$)[cite: 4].
* **Stufe 2:** BFS auf bekannten begehbaren Kacheln der Mental Map ($L_1 > 3$, keine künstliche Reichweitenbegrenzung)[cite: 4].
* **Stufe 3:** Frontier-Exploration an der Grenze zu unbekanntem Terrain (`TileKnowledge.UNKNOWN`), falls Stufe 1 und 2 scheitern[cite: 4].
* **Junction-Tile:** Das letzte Feld auf dem berechneten Ausweichweg, das noch Teil der ursprünglichen Partnertrajektorie war[cite: 4].

---

## 5. Protokoll der Blockadelösung

### 1. Unterwegs-Synchronisation
* Agent B folgt Agent A im Korridor[cite: 8].
* Erkennt Agent B eine zwingende Pfadänderung, sendet er `is_path_update=True` mit neuem Pfad und `correlation_key`[cite: 8, 11].
* Agent A stoppt temporär, berechnet die Ausweichroute neu und sendet `is_resume_signal=True` ("Ok, weiter.") an Agent B zurück[cite: 8, 11].

### 2. Nischenankunft und Handshake
* **Stopp vor dem Abzweig:** Erreicht Agent A die `junction_position`, sendet er `is_halt_request=True` ("HALT WARTE!") an Agent B[cite: 8, 11].
* **Einbiegen:** Agent B stoppt[cite: 8]. Agent A rückt auf das Ausweichfeld ein[cite: 8].
* **Haltezustand:** In der Nische setzt Agent A `Goal(is_evasion_hold=True, junction_position=..., yield_for_agent_id=B.id)` und sendet `is_resume_signal=True` ("Ok, weiter.") an Agent B[cite: 8, 11, 12].
* **Durchfahrt:** Agent B nimmt die Fahrt wieder auf und passiert den Engpass[cite: 8].

### 3. Clearance und Höflichkeits-Trigger
* **Clearance-Bedingung:** Agent B hat die Nische passiert, sobald $L_1(\text{pos}_B, \text{junction}) \ge 2$ und $\text{junction} \notin \text{path}_B$ erfüllt sind[cite: 8].
* **Trigger-Signal:** Agent B sendet im Vorbeigehen `is_courtesy=True` ("Danke fürs Platz machen!")[cite: 8, 11].
* **Reaktivierung:** Der Empfang der Höflichkeitsnachricht triggert bei Agent A deterministisch das Antwortsignal ("Gern geschehen!"), den Abbau des Halteziels und die A*-Neuberechnung zum Primärziel[cite: 8, 11].

### 4. Ablaufgraph

```text
[Phase 1: Verhandlung & Nischenfund]
       |
       v
  Agent A sucht Nische via EvasionFinder (Stufe 1 & 2)
       |
       +---> [Nische gefunden] ---------> Ziel "Anfahrt Nische", Pfad via Nachricht an B
       |
       +---> [Keine Nische bekannt] ----> Ziel "Erkunde Terrain" (Frontier), B folgt A
       |
       +---> [Alles erschöpft] ---------> Nachricht: "Ich kann nicht ausweichen...", B am Zug

[Phase 2: Verfolgung & dynamisches Pfad-Update]
       |
       v
  Agent B folgt Agent A im Korridor
       |
       +---> B ändert Pfad: B sendet is_path_update=True + "HALT WARTE!" + correlation_key
       |     A stoppt -> A rechnet Route/Nische neu -> A sendet "Ok, weiter." -> Beide fahren an

[Phase 3: Nischenankunft & Handshake]
       |
       v
  Agent A erreicht junction_position (unmittelbar vor Nische)
       |
       +---> A sendet "HALT WARTE!" (is_halt_request=True, correlation_key)
       +---> B stoppt
       +---> A zieht auf target_position in die Nische
       +---> A setzt is_evasion_hold=True
       +---> A sendet "Ok, weiter." (is_resume_signal=True)
       +---> B fährt an

[Phase 4: Passage, Clearance & Quittung]
       |
       v
  Agent B passiert junction_position und erreicht Distanz >= 2
       |
       +---> B sendet "Danke fürs Platz machen!" (is_courtesy=True, is_resume_signal=True)
       +---> A antwortet deterministisch "Gern geschehen!"
       +---> A baut Halteziel ab, berechnet A*-Pfad zum Hauptziel neu, verlässt Nische
