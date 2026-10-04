# Implementierungsplan 10_04: Reaktive Future-Architektur, Generische Gruppen-Verhandlung ($1 \dots N$) & Modulare Kognitions-Dekomposition

## 1. Grundsätzliche Ziele & Leitprinzipien der Architektur

1. **Echte Kognitive Autonomie statt starrer Heuristiken:**
   Agenten lösen räumliche Engpässe nicht über starre Heuristiken, Schwellenwerte oder erzwungene Schlichterregeln. Sie wägen rational zwischen sozialem Aufwand (Verhandeln, Nachgeben, Warten) und physischem Aufwand (konkrete Umwegkosten) ab. Entscheidungen trifft das Sprachmodell (LLM/SLM) intrinsisch anhand quantitativer Umweltmetriken, individueller Persönlichkeitsmerkmale (`assertiveness`, `charisma`) und episodischer Gedächtniseinträge[cite: 6].

2. **Harmonisierte Entscheidungswährung („Zeiteinheiten“ statt Ticks vs. Raum):**
   Da im Simulationsmodell 1 Takt genau 1 Zeiteinheit und 1 Bewegungsschritt genau 1 Zeiteinheit entspricht, werden räumliche und zeitliche Kosten auf die einheitliche Währung **Zeiteinheiten** normiert:
   * `restweg_aktuell_zeiteinheiten`: Verbleibende Zeiteinheiten auf der blockierten Originalroute ($\text{len}(\text{agent.path})$).
   * `alternativweg_gesamt_zeiteinheiten`: Gesamte Zeiteinheiten bei Umgehung der Blockade ($\text{len}(\text{alt\_path})$) oder `None`, falls topologisch kein Weg existiert.
   * `umweg_mehr_zeiteinheiten`: Konkrete zusätzliche Zeiteinheiten ($\text{Alternativweg} - \text{Restweg}$) oder `None` bei Sackgassen.
   * `bisher_gewartete_zeiteinheiten`: Reale Takt-Differenz seit Beginn des Warteentscheids ($\text{current\_tick} - \text{initial\_wait\_tick}$). Wiederholtes Warten misst die verstrichene Zeit deterministisch über diesen initialen Ankerpunkt.

3. **Getaktete Welt mit lokaler Verhandlungs-Sperre (Option B):**
   Die globale Welt steht während einer Konversation nicht still. Unbeteiligte Agenten bewegen sich weiter, und die globale Taktzeit schreitet voran[cite: 4]. Die beiden interagierenden Agenten verharren während ihres Dialogs physisch an Ort und Stelle (`is_busy == True`, Zustand `IN_INTERACTION`)[cite: 4, 6]. Pro Welt-Takt wird genau ein Interaktionsschritt ausgetauscht, wodurch Raum-Zeit-Drifts vermieden werden[cite: 6].

4. **Universelles Interaktions-Future-Paradigma (Request-Response):**
   Sämtliche Interaktionen – ob zwischen Agenten oder mit statischen Objekten – werden über stark typisierte Futures abgewickelt[cite: 5]. Jede Anfrage bindet an ein vertragliches Future:
   * **Agent $\leftrightarrow$ Agent:** Der Partner *muss* das Future bedienen (durch `ReplyAction`, `YieldAction`, `LeaveInteractionAction` oder künftig `HostileAction`).
   * **Agent $\leftrightarrow$ Objekt:** Passive Objekte bedienen das Future synchron im selben Takt via `ImmediateResult`[cite: 5].
   * **Fremdbeschäftigter Partner (Fail-Fast):** Ist der Zielagent anderweitig gebunden, liefert der `InteractionDispatcher` synchron `ImmediateResult(success=False, reason="BUSY")`, ohne den Anfragenden zu blockieren[cite: 5].
   * **Idempotenz-Schutz:** Vor einer Blockadeauflösung prüfen Agenten ihre Mailbox und Inbox, um zirkuläre Gegenanfragen bei frontalem Aufeinandertreffen deterministisch zu unterdrücken.

5. **Generisches Gruppenparadigma ($1 \dots N$):**
   Ein Einzelagent wird architektonisch als Konvoi der Größe 1 behandelt. Dieselbe Nischenberechnung, dieselbe Kognitionsabstraktion (`NegotiatingParty`) und dieselbe Clearance-Logik steuern Einzelbegegnungen wie auch Gruppenkonflikte ohne redundante Codezweige. Der Gruppenanführer verhandelt autokratisch für seine Mitglieder anhand der kumulierten Gruppenzeiteinheiten.

6. **Vertragliche Engstellen-Räumung via `ClearanceContract`:**
   Ablösung des periodischen Distanz-Pollings in jedem Takt. Die ausweichende Partei bindet das Halteziel in der Nische an ein `ClearanceContract`-Future. Die passierende Partei löst dieses Future beim Erreichen des Sicherheitsabstands reaktiv auf, woraufhin die Nische unmittelbar im LIFO-Verfahren geräumt wird. Bei Routenänderungen oder Blockaden der passierenden Partei schützt ein Fast-Path-Timeout vor Deadlocks.

7. **Kausales, Change-Driven SFT-Logging (Data Flywheel für SLMs):**
   Das Logging in `simulation_events.jsonl` dient primär der Erzeugung hochwertiger Trainingsdatensätze für das Fine-Tuning kleiner Sprachmodelle (SLM, 1B–3B Parameter via Unsloth). Da repetitive No-Op-Zustände (z. B. 20 Takte reines Geradeauslaufen) ein Sprachmodell zur Passivität verziehen, filtert der `CognitionOrchestrator` unveränderte Takte vollständig heraus. Ein Log-Eintrag entsteht ausschließlich bei kausalen Zustandsänderungen, echten Konflikten oder sozialen Reflexionen.

8. **TOCTOU-Resilienz (Time-of-Check to Time-of-Use):**
   Verifikation aller räumlichen und situativen Vorbedingungen unmittelbar vor dem Commit einer Aktion (`action_dropped_stale`, `action_precommit_invalidated`), um Zustandsdrifts durch variable Inferenzzeiten deterministisch abzufangen.

---

## 2. Verbindliche Handlungsanweisungen & Richtlinien für den autonomen Entwickler-Agenten

### 2.1 Phasenweises TDD (Contract-First), Matrix-Testing & Migration von Alt-Tests
* **Kein Micro-TDD, sondern Contract- & Matrix-First pro Phase:**
  * Es wird auf zeilenweises Micro-TDD verzichtet, um die Entstehung fragiler Hilfskonstrukte bei tiefgreifenden architektonischen Umbauten zu vermeiden[cite: 4].
  * Vor Beginn einer Umsetzungsphase (Phase 1 bis Phase 4) werden die Ziel-Kontrakte und Matrix-Tests **ausschließlich für die Komponenten der aktuellen Phase** formuliert oder aktualisiert („Red“)[cite: 4].
  * Es darf kein Vorab-Testcode für spätere Phasen geschrieben werden, damit der Fokus erhalten bleibt und nicht gegen eine unüberschaubare Menge fehlschlagender Tests gearbeitet werden muss.
  * Die Fachlogik wird zielgerichtet implementiert, bis sämtliche Phasentests fehlerfrei durchlaufen („Green“)[cite: 4].
  * Erst nach Erreichen des grünen Status erfolgt das strukturelle Refactoring (Dekomposition, Beseitigung von Redundanzen, SOLID-Bereinigung)[cite: 4].
* **Systematisches Matrix-Testing:**
  * Die Datei `tests/integration/test_dialogue_decision_matrix.py` dient als primäre Leitmatrix und wird sukzessive um die relevanten Pfade der Kognitions- und Interaktionsmatrix erweitert[cite: 4].
  * Für neu strukturierte Komponenten (`UnifiedNicheService`, `ClearanceContract`, `NegotiatingParty`) wird pro Phase ein isolierter Testkatalog angelegt, der Vorbedingungen, Übergangszustände und Endergebnisse deterministisch validiert[cite: 4].
  * Tests dürfen nicht starr auf transitorische Zwischenzustände nach festen Takt-Schleifen prüfen (z. B. Halten in der Nische vor der Ausfahrt); temporäre Zustände müssen ereignis- oder zustandsgesteuert abgefangen werden[cite: 4].
* **Kanonische Klassifizierung & Migration bestehender Alt-Tests:**
  Bestehende Tests dürfen weder unkritisch mitgeschleift noch pauschal verworfen werden. Sie sind strikt in drei Kategorien zu unterteilen:
  1. **Kategorie A: Verhaltens- und End-to-End-Szenarien schützen (Beibehalten):**
     * Integrationsprüfungen (`tests/integration/test_scenarios.py`, `tests/integration/test_dialogue_showcase_resilience.py`) sichern das funktionale Gesamtsystem ab und verbleiben dauerhaft als Regressionsnetz.
     * Bei Schnittstellenänderungen werden lediglich die Fixture-Aufrufe und Konstruktorsignaturen angepasst; die fachlichen Verhaltens-Assertions bleiben unangetastet.
  2. **Kategorie B: Struktur-Tests konsolidierter Module überführen (Migrieren):**
     * Unit-Tests für Komponenten, die im Rahmen der Neugestaltung verschmolzen werden (z. B. `EvasionFinder` und `MultiAgentNichePacker`), werden in die Testsuite der neuen Komponente überführt (z. B. `tests/unit/application/movement/test_unified_niche_service.py`)[cite: 4].
     * Die alten Testdateien werden erst dann gelöscht, wenn der neue Service alle bisherigen Randfälle und Konfigurationen nachweislich abdeckt.
  3. **Kategorie C: Whitebox-Tests gelöschter Implementierungsdetails entfernen (Ablösen):**
     * Tests, die spezifisch auf bewusst abgeschaffte Legacy-Mechanismen abzielen (z. B. `Agent.interaction_queue`, manuelle Kartenmanipulationen, veraltete Flag-Kombinationen oder zyklisches Distanz-Polling), werden gelöscht, sobald der jeweilige Nachfolge-Kontrakt (`InteractionDispatcher`, `IPathfinder.find_path(..., excluded_positions=...)`, `ClearanceContract`) implementiert und getestet ist[cite: 4].

### 2.2 Tooling, Verifikationsbefehle & Dependency-Injection-Integrität
* **Python-Umgebung:** Es wird ausschließlich `python3` verwendet (kein `python`)[cite: 6].
* **Testausführung:** Tests werden zielgerichtet pro Phase mit `python3 -m pytest <pfad>` ausgeführt[cite: 6].
* **Statische Analyse:** Nach jeder Dateiänderung muss die Datei mit `pyright <dateipfad>` auf Typfehler geprüft werden[cite: 6]. Fehler wie `Member None of Goal | None does not have attribute name` müssen über lokales Variablen-Narrowing aufgelöst werden[cite: 6].
* **Dependency-Injection-Integrität:** Sobald Schnittstellen (`ports/`) oder Services neu strukturiert werden, ist zwingend `src/infrastructure/container.py` synchron anzupassen, um die Singleton-Verdrahtung konsistent zu halten[cite: 6].

### 2.3 Anti-Halluzinations-Schranken & Eskalation
* **Eskalationsgrenze (3-Fehlversuche-Regel):** Gelingt es nach drei aufeinanderfolgenden Korrekturversuchen nicht, einen fehlschlagenden Test grün zu schalten, muss die Bearbeitung sofort angehalten und der Nutzer konsultiert werden.
* **Verbot von Testaufweichung:** Es ist strikt untersagt, Assertions in Integrations- oder Szenariotests abzuschwächen oder gelöschte Methoden durch Dummy-Mocks zu ersetzen, um Tests künstlich zu bestehen.
* **Typstrenge Mocks:** Mocks für `ICognitionProvider` müssen vollständige, valide Pydantic-Instanzen (`BlockedResolution`, `DialogueResolution`) zurückliefern; Dummy-Dicts sind unzulässig[cite: 4].

### 2.4 Git- und Branch-Disziplin
* **Strikte Branch-Bindung:** Es wird ausnahmslos auf dem **aktuell ausgecheckten Branch** gearbeitet[cite: 4].
* **Branch-Schutz:** Der Agent darf unter keinen Umständen neue Branches erstellen, auf andere Branches wechseln oder `git checkout` / `git switch` ausführen[cite: 4].
* **Strikte Scope-Isolation:** Vor Abschluss und Commit einer Phase dürfen keine Dateien oder Komponenten nachgelagerter Phasen angefasst oder vorab refaktorisiert werden[cite: 6].
* **Commit-Hygiene & Conventional Commits:** 
  * Erst wenn das Quality Gate einer Phase zu 100 % grün ist, erfolgt der Commit[cite: 6].
  * Vor `git add` muss `git status` geprüft werden: Es dürfen **keine** Laufzeit-Artefakte, Caches oder Log-Dateien (`logs/simulation_events.jsonl`, `__pycache__`) gestaged werden[cite: 4, 6].
  * Commit-Nachrichten folgen strikt Conventional Commits (z. B. `refactor(core): ...`, `feat(movement): ...`)[cite: 6].

### 2.5 Konsultation des Nutzers bei Unklarheiten
Bestehen bei der Implementierung mehrere gleichwertige Entwürfe oder hängen Abläufe von Nutzerpräferenzen ab (insbesondere bei Vorfahrts- und Konvoiregeln), muss der Nutzer nach folgendem verbindlichen Schema konsultiert werden[cite: 4]:

1. **Beteiligte:** Akteure und Rollen (z. B. „Agent A als Konvoi-Führer mit Follower C trifft auf Einzelagent B“).
2. **Koordinaten & Topologie:** Ausgangspositionen (z. B. „A auf (12, 4), C auf (11, 4), B auf (13, 4); Nische bei (12, 3)“).
3. **Taktverlauf:** Was geschieht in Takt $t$, was droht in Takt $t+1$?[cite: 4]
4. **Optionen im Vergleich:**
   * *Option 1:* Verhalten, Vor- und Nachteile.
   * *Option 2:* Verhalten, Vor- und Nachteile.
5. **Konkrete Frage:** Präzise Frage nach der bevorzugten Nutzerentscheidung.

---

## 3. Matrix aller Handlungsstränge & gewünschtes Verhalten

Die nachfolgende Matrix definiert alle möglichen Konstellationen beim Aufeinandertreffen zweier Parteien (Partei A = Initiator, Partei B = Partner; jeweils 1 bis $N$ Agenten).

| Strang / Szenario | Aktion Partei A (Initiator) | Reaktion Partei B (Partner) | Folgeentscheidung Partei A | Physische & Systemische Auswirkung |
| :--- | :--- | :--- | :--- | :--- |
| **1. Unmittelbare Kooperation** | Sendet `TalkAction(intent="request_yield")`. | Erwidert `TalkAction(intent="offer_yield")` und setzt Ziel `In Nische ausweichen`. | Registriert Zusage, quittiert (`accept`), passiert den Chokepoint. | B zieht in Nische (bei $N > 1$ nach Slot-Tiefe geordnet) und bindet an `ClearanceContract`. A passiert. Nach Erreichen des Sicherheitsabstands bedient A das `ClearanceContract`-Future. B tritt im LIFO-Verfahren wieder aus. |
| **2. Rejection $\to$ Reroute (Umweg)** | Sendet `TalkAction(intent="request_yield")`. | Erwidert `TalkAction(intent="reject")`. | Wählt `RerouteAction` basierend auf Kognitionskontext (`detour_additional_steps`). | Beide Parteien gehen in `IDLE`. A berechnet neuen Pfad via `IPathfinder` unter explizitem Ausschluss der Kacheln von B (`excluded_positions`) und umgeht die Blockade autonom. |
| **3. Rejection $\to$ Warten** | Sendet `TalkAction(intent="request_yield")`. | Erwidert `TalkAction(intent="reject")`. | Wählt `WaitAction(ticks=N)`. | A pusht Warteziel mit `initial_wait_tick`, verharrt an Ort und Stelle und misst die Wartezeit. Fast-Path weckt A vorzeitig, falls B das Feld räumt. |
| **4. Rejection $\to$ Selbst-Ausweichen** | Sendet `TalkAction(intent="request_yield")`. | Erwidert `TalkAction(intent="reject")`. | Wählt `new_sub_goal="In Nische ausweichen"` (lenkt ein). | A weicht selbst in die erreichbare Nische aus und bindet an `ClearanceContract` gegenüber B. B passiert. A setzt nach Clearance die Hauptroute fort. |
| **5. Rejection $\to$ Zielabbruch** | Sendet `TalkAction(intent="request_yield")`. | Erwidert `TalkAction(intent="reject")`. | Wählt `AbortAction(reason=...)`. | A leert seinen Pfad, markiert das Ziel als abgebrochen (`abandoned`) und wechselt in `IDLE`[cite: 6]. B kann ungehindert passieren. |
| **6. Multi-Turn Verhandlung** | T1: `TalkAction(intent="request_yield")`. | T1: Erwidert `TalkAction(intent="negotiate", message="Warum?")`. | T2: Bekräftigt Dringlichkeit via `TalkAction(intent="request_yield")`. | Nachrichten fließen rundenbasiert pro Welt-Takt über typisierte Interaktions-Futures[cite: 6]. B lenkt in Runde 2 ein (`offer_yield`), woraufhin die Nischenausweichkaskade startet. |
| **7. Beidseitige Verweigerung (Deadlock)** | Sendet `TalkAction(intent="request_yield")`. | Erwidert `TalkAction(intent="reject")`. | Antwortet erneut mit `reject` oder verharrt. | Beide Parteien bleiben physisch blockiert und initiieren in Folgetakten Neubewertungen anhand der akkumulierten Wartezeit. |
| **8. Nicht-kommunikatives Hindernis** | Versucht `TalkAction` oder Bewegung[cite: 4]. | Keine Antwort (`NOT_COMMUNICATIVE` oder stationäres Objekt)[cite: 5]. | Kognition wählt `InspectAction` $\to$ `ProbeAction` $\to$ `RerouteAction`. | Agent aktualisiert epistemisches Gedächtnis (`can_talk=False`, `is_passable=False`), markiert Hindernis und plant Umweg[cite: 5]. |
| **9. Kognitions-Timeout / Inferenzfehler** | Sendet Anfrage oder berechnet Weg[cite: 5]. | Exception / Timeout im Kognitionsadapter. | Automatischer Fallback greift. | `SimulationEvent(event_type="cognition_failed")` wird geloggt. Agent erhält temporäres Schutz-Warteziel (`remaining_ticks=2`), um die Simulation zu stabilisieren. |

---

## 4. Beteiligte Dateien und Aufgaben

### Anwendungs- & Koordinationsschicht (`src/application/`)
* **`simulation_engine.py`**: Zentraler Taktgeber und System-Fassade. Koordiniert Phase 1 (Kognition, Dialoge) und Phase 2 (Zwei-Phasen-Commit der Bewegung, Zielerreichung). Verwaltet `IInteractionDispatcher` als Singleton.
* **`services/movement/movement_orchestrator.py`**: Verwaltet Schrittabsichten (`TileReservationIntent`), führt über `MovementSyncService` physische Bewegungen aus und übergibt blockierte Paare entprellt an die Konfliktkoordination[cite: 4].
* **`services/coordination/conflict_coordinator.py`**: Koordiniert Blockadesituationen. Berechnet metrische Zeiteinheiten, aggregiert Parteien (`NegotiatingParty`) und delegiert Entscheidungen via Strategy Pattern an spezialisierte Handler.
* **`services/coordination/dialogue_coordinator.py`**: Steuert die rundenbasierte Ausführung von Dialogen, bereitet Kognitionskontexte auf und stößt soziale Reflexionen (`social_reflection_completed`) für den Vektorspeicher an.
* **`services/coordination/interaction_lifecycle_manager.py` (Neu)**: Zentraler Verwalter aller Interaktions-Futures. Kapselt Erstellung, Timeouts, Abbrechen und sauberes Schließen von Futures ohne Code-Duplikate[cite: 5].
* **`services/movement/unified_niche_service.py` (Neu)**: Verschmilzt `EvasionFinder` und `MultiAgentNichePacker`. Berechnet Nischenslots, Grenzkacheln und Ingress-/Egress-Reihenfolgen (LIFO) generisch für $1 \dots N$ Agenten.
* **`services/execution/action_executor.py`**: Führt physische Aktionen kognitiver Beschlüsse aus (`execute_evasion`, `execute_inspection`, `execute_probe`, `execute_consume`, `execute_rest`).
* **`services/interaction/interaction_dispatcher.py`**: Zentraler Dispatcher für Interaktionsbefehle[cite: 5]. Behandelt synchrones Fail-Fast (`BUSY`) und bindet asynchrone Interaktions-Futures[cite: 5].
* **`services/coordination/dialogue_session_manager.py`**: Asynchrone Session-Locks pro Agentenpaar zur Vermeidung zirkulärer Deadlocks. Frei von künstlichen Zwangslimits.
* **`services/cognition/cognition_orchestrator.py`**: Führt Change-Driven Snapshot-Logging durch und unterdrückt No-Op-Events über Caching-Instanz `_last_logged_snapshots`.

### Domänenschicht (`src/domain/`)
* **`models/agent/agent.py`**: Entitätsmodell mit FSM-Zustand (`AgentLifecycleState`), Zielstack, mentaler Karte und reaktiver Mailbox[cite: 6].
* **`models/coordination/negotiating_party.py` (Neu)**: Abstraktion für $1 \dots N$ Agenten mit Leader, Mitgliedern und aggregierten Zeiteinheiten.
* **`models/coordination/clearance_contract.py` (Neu)**: Vertragliches Future zur Absicherung des Nischenhalts mit Fast-Path-Timeout gegen Blockaden.
* **`models/planning/goal.py`**: Zieldefinition inklusive `initial_wait_tick` zur akkumulierten Wartezeitmessung[cite: 6].
* **`models/planning/cognition.py`**: Pydantic-Domänenschemata für Kognitions- und Dialogaktionen (`BlockedResolution`, `DialogueResolution`, `TalkAction`, `RerouteAction`, `WaitAction`, `AbortAction`, `EndDialogueAction`).
* **`ports/pathfinder.py`**: Erweitert um `excluded_positions: Optional[set[Position]] = None`.

### Testdateien (`tests/`)
* **`tests/integration/test_dialogue_decision_matrix.py`**: Vollständige Matrixprüfung aller 6 Kognitionspfade (Kooperation, Reroute, Warten, Selbstevaluation, Abort, Multi-Turn).
* **`tests/integration/test_scenarios.py`**: End-to-End-Szenarien für Korridorbegegnungen und Labyrinth-Erprobungen.
* **`tests/integration/test_dialogue_showcase_resilience.py`**: Resilienzprüfung unter extremen Latenzen mit vollständigem Lebenszyklus.
* **`tests/unit/application/movement/test_unified_niche_service.py` (Neu)**: Testet Slot-Zuweisung und LIFO-Reihenfolge für $N=1$ bis $N=4$.
* **`tests/unit/application/coordination/test_reactive_future_interaction.py`**: Validierung von Zeiteinheiten, Fail-Fast und Fast-Path-Interrupts[cite: 2].

---

## 5. Fundierte architektonische Analyse & Refactoring-Konzepte

### 5.1 Topologie: Standardisierter Pfadausschluss & `UnifiedNicheService`
* **Pfadausschluss ohne Seiteneffekte:** `IPathfinder.find_path` erhält den Parameter `excluded_positions: Optional[set[Position]] = None`. Der A*-Algorithmus ignoriert ausgeschlossene Kacheln direkt im Heap-Durchlauf, wodurch alle fehleranfälligen temporären Kopien der mentalen Karten entfallen.
* **Composite Pattern für Nischen:** `UnifiedNicheService` löst die Trennung zwischen Einzelagenten und Konvois auf. Für $1 \dots N$ Agenten ermittelt er die optimalen Nischenkoordinaten und berechnet deterministisch die Ingress- und Egress-Reihenfolge: Bei Sackgassen-Nischen muss der Agent mit dem tiefsten Zielslot die Nische als erster betreten (Ingress) und als letzter verlassen (Egress, LIFO-Prinzip), um Blockaden am Eingang auszuschließen.

### 5.2 Verhandlung: `NegotiatingParty` & Strategy Pattern
* **Kanonische Partei-Abstraktion:** Begegnungen werden stets zwischen zwei `NegotiatingParty`-Instanzen verhandelt. Bei einem Einzelagenten besteht die Partei aus genau einem Mitglied. Das LLM erhält stets dieselbe quantifizierte Datenstruktur aus Gesamtrestweg und Gruppen-Umwegkosten.
* **Single Responsibility Principle (SRP):** Der monolithische Block `resolve_blockage` wird in eigenständige Strategie-Klassen aufgeteilt:
  * `TalkActionHandler`: Erzeugt Interaktions-Future und versetzt Agent in `IN_INTERACTION`[cite: 5].
  * `RerouteActionHandler`: Weist Alternativpfad unter Ausschluss der Gegenpartei zu.
  * `WaitActionHandler`: Initialisiert Warteziel mit konserviertem `initial_wait_tick`.
  * `EvasionActionHandler`: Initialisiert `UnifiedNicheService`, weist Nischenpfade zu und bindet an den `ClearanceContract`.

### 5.3 Reaktivität: `InteractionLifecycleManager` & `ClearanceContract`
* **Vermeidung von Codeduplikaten:** Der `InteractionLifecycleManager` kapselt das Registrieren, Überwachen, Auflösen und Bereinigen von Futures. Die vierfach duplizierte Schleife über `agent.interaction_mailbox` entfällt restlos[cite: 5].
* **Reaktive Nischenräumung:** `ClearanceContract` löst das ineffiziente Distanz-Polling ab. Das Passieren des letzten Gegenagenten löst das Future direkt auf; bei unvorhergesehenen Blockaden schützt ein Fast-Path-Mechanismus vor partiellem Verharren.

### 5.4 FSM-Konsolidierung
* Zusammenführung redundanter Flags (`AgentLifecycleState`, `EvasionPhase`, `Goal`-Flags) in eine konsistente, kanonische FSM auf `Agent`[cite: 6]:
  * `IDLE`: Bereit für neue Aufgaben[cite: 6].
  * `NAVIGATING`: Folgt aktivem Pfad[cite: 6].
  * `DELIBERATING`: Führt aktive Inferenz / Pfadberechnung durch (`is_thinking == True`)[cite: 6].
  * `IN_INTERACTION`: Verhandelt aktiv oder wartet auf Erwiderung[cite: 6].
  * `EVADING`: Bewegt sich in zugewiesenen Nischenslot[cite: 6].
  * `HOLDING_IN_NICHE`: Hält Position unter Bindung an `ClearanceContract`[cite: 6].

---

## 6. Phasenweiser Durchführungsplan mit Quality Gates

Jede Phase ist eine in sich geschlossene funktionale Einheit mit eigenem Quality Gate[cite: 4]. Phase $k+1$ darf erst begonnen werden, wenn das Quality Gate für Phase $k$ vollständig und nachweislich bestanden ist[cite: 4].

```
+-------------------------------------------------------------------------+
| Phase 1: Schnittstellen-Bereinigung & Entkopplung (Clean Code)          |
| -> IPathfinder excluded_positions umsetzen                              |
| -> InteractionLifecycleManager extrahieren                              |
| Quality Gate 1: python3 -m pytest tests/unit/infrastructure/pathfinding/ |
|                 tests/unit/application/interaction/                     |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| Phase 2: Generischer UnifiedNicheService & ClearanceContract            |
| -> UnifiedNicheService (1...N mit LIFO-Reihenfolge)                     |
| -> ClearanceContract (Reaktives Wecken statt Distanz-Polling)           |
| Quality Gate 2: python3 -m pytest tests/unit/application/movement/      |
|                 tests/integration/test_scenarios.py                     |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| Phase 3: Dekomposition von ConflictCoordinator & DialogueCoord          |
| -> NegotiatingParty Abstraktion einführen                               |
| -> Strategy-Handler (Talk, Reroute, Wait, Evasion) aufteilen            |
| Quality Gate 3: python3 -m pytest tests/unit/application/coordination/  |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| Phase 4: Matrix-Verifikation & SFT-Export                               |
| -> test_dialogue_decision_matrix.py (100% grün)                         |
| -> export_sft_dataset.py verifizieren                                  |
| Quality Gate 4: python3 -m pytest (100% Gesamtsuite grün)               |
+-------------------------------------------------------------------------+
```

### Phase 1: Schnittstellen-Bereinigung & Entkopplung (Clean Code)
* **Ziel:** Beseitigung von Seiteneffekten bei der Pfadsuche und Konsolidierung der Mailbox-Bereinigung[cite: 4].
* **Schritt 1.1 (Expliziter Startpunkt):**
  * `src/domain/ports/pathfinder.py` und `src/infrastructure/pathfinding/astar.py` erweitern um:
    ```python
    def find_path(
        self,
        start: Position,
        goal: Position,
        grid: IGrid,
        excluded_positions: Optional[set[Position]] = None,
    ) -> Optional[list[Position]]:
    ```
  * Unit-Tests in `tests/unit/infrastructure/pathfinding/test_astar.py` anlegen: Pfadsuche ignoriert `excluded_positions` deterministisch.
  * Manuelle Modifikationen der mentalen Karte in `ActionExecutor` und `ConflictCoordinator` durch direkte Übergabe von `excluded_positions={blocked_pos}` ersetzen[cite: 4].
* **Schritt 1.2:**
  * `src/application/services/coordination/interaction_lifecycle_manager.py` implementieren[cite: 4]:
    * `cancel_mailbox_futures(agent: Agent, reason: str = "CANCELLED") -> None`
    * `complete_matching_futures(agent: Agent, partner_id: str, success: bool, reason: str, payload: Optional[dict] = None) -> None`
  * Vierfach duplizierte Schleifen in `InteractionDispatcher`, `DialogueCoordinator` und `AgentProtocolService` durch den Manager ersetzen[cite: 4].
* **Quality Gate Phase 1:**
  ```bash
  python3 -m pytest tests/unit/infrastructure/pathfinding/ tests/unit/application/interaction/
  pyright src/infrastructure/pathfinding/ src/application/services/coordination/interaction_lifecycle_manager.py
  git commit -m "refactor(core): implement pathfinder position exclusion and interaction lifecycle manager"
  ```

### Phase 2: Generischer `UnifiedNicheService` & `ClearanceContract`
* **Ziel:** Zusammenführung der Nischenlogik ($1 \dots N$) und Ablösung des zyklischen Distanz-Pollings[cite: 4].
* **Schritt 2.1:**
  * `src/application/services/movement/unified_niche_service.py` implementieren[cite: 4]:
    * Nimmt `agents: list[Agent]` entgegen[cite: 4].
    * Ermittelt zusammenhängende oder verteilte Nischenplätze[cite: 4].
    * Weist Ingress- und Egress-Reihenfolge nach dem LIFO-Prinzip zu (tiefster Slot betritt zuerst, verlässt zuletzt)[cite: 4].
  * Neue Testsuite anlegen: `tests/unit/application/movement/test_unified_niche_service.py` ($N=1$ bis $N=4$ Nischenbelegung)[cite: 4].
  * Bisherige Tests aus `tests/unit/application/movement/test_evasion_finder.py` migrieren; Alt-Dateien bereinigen[cite: 4].
* **Schritt 2.2:**
  * `src/domain/models/coordination/clearance_contract.py` implementieren: Future mit Sicherheitsabstand, Initiator-ID und Fast-Path-Timeout[cite: 4].
  * `AgentProtocolService.check_and_signal_clearance` refaktorisieren: Löst das `ClearanceContract`-Future auf, sobald der letzte Gegenagent die Engstelle passiert hat[cite: 4].
* **Quality Gate Phase 2:**
  ```bash
  python3 -m pytest tests/unit/application/movement/test_unified_niche_service.py tests/integration/test_scenarios.py
  pyright src/application/services/movement/unified_niche_service.py src/domain/models/coordination/clearance_contract.py
  git commit -m "feat(movement): implement unified niche service and reactive clearance contract"
  ```

### Phase 3: Dekomposition von `ConflictCoordinator` & `DialogueCoordinator`
* **Ziel:** Einhaltung des Single Responsibility Principle (SRP) und Einführung der $1 \dots N$ Partei-Abstraktion[cite: 4].
* **Schritt 3.1:**
  * `src/domain/models/coordination/negotiating_party.py` anlegen (`leader`, `members`, aggregierte Zeiteinheiten)[cite: 4].
* **Schritt 3.2:**
  * Extraktion der Strategy-Handler in `src/application/services/coordination/strategies/`:
    * `TalkActionHandler`[cite: 4]
    * `RerouteActionHandler`[cite: 4]
    * `WaitActionHandler`[cite: 4]
    * `EvasionActionHandler`[cite: 4]
  * `ConflictCoordinator.resolve_blockage` auf reine Delegation an die Strategy-Handler reduzieren[cite: 4].
* **Schritt 3.3:**
  * `DialogueCoordinator` verschlanken: Vektorgedächtnis-Reflexion in separaten Service auslagern, feste Rundenlimits restlos entfernen[cite: 4].
* **Quality Gate Phase 3:**
  ```bash
  python3 -m pytest tests/unit/application/coordination/
  pyright src/application/services/coordination/
  git commit -m "refactor(coordination): decompose conflict coordinator into strategy handlers with negotiating party"
  ```

### Phase 4: Matrix-Verifikation & SFT-Export
* **Ziel:** Vollständige Absicherung aller Verhandlungspfade und Bereinigung der Trainingsdaten[cite: 4].
* **Schritt 4.1:**
  * `tests/integration/test_dialogue_decision_matrix.py` finalisieren: Alle 6 Fälle (inklusive Selbstevaluation und Multi-Turn) müssen fehlerfrei durchlaufen[cite: 4].
* **Schritt 4.2:**
  * `src/tools/export_sft_dataset.py` ausführen und validieren: Sicherstellen, dass keine No-Op-Zustände im Output-Stream landen[cite: 4].
* **Quality Gate Phase 4 (Gesamtabnahme):**
  ```bash
  python3 -m pytest
  pyright src/ tests/
  git commit -m "test(matrix): verify all 6 decision matrix cases and finalize sft export"
  ```

---

## 7. Definition of Done (DoD)

1. Sämtliche vier Quality Gates wurden chronologisch durchlaufen und committet.
2. `Agent.interaction_queue`, veraltete Status-Flags (`has_bid_farewell`, `peer_bid_farewell`, etc.) und `max_dialogue_turns` existieren im gesamten Projekt nicht mehr[cite: 4].
3. Alle Interaktionen (Agent $\leftrightarrow$ Agent, Agent $\leftrightarrow$ Objekt, Clearance) folgen dem reaktiven Future-Muster[cite: 4].
4. Pfadsuche mit Kachelausschluss erfolgt über `IPathfinder.find_path(..., excluded_positions=...)` ohne Seiteneffekte auf mentalen Karten[cite: 4].
5. Nischenzuweisung für $1 \dots N$ Agenten erfolgt zentral über `UnifiedNicheService` nach dem LIFO-Prinzip[cite: 4].
6. Die gesamte Testsuite (`python3 -m pytest`) und die statische Typprüfung (`pyright src/ tests/`) sind zu 100 % fehler- und warnungsfrei[cite: 4].