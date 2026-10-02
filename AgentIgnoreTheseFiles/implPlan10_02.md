# implPlan10_04.md: Capability-basiertes Interaktionsmodell, Command-Dispatch & Ereignisgesteuerte Future-Verhandlung

## 1. Problemstellung & Architekturbegründung

### 1.1 Entkopplung von Inferenzlatenzen (Ausschluss von Takt- und Timeout-Drift)
* Problem:
  Simulations-Ticks (tick_interval=0.15s) und Realzeit-Timeouts (asyncio.wait_for) messen die Dauer der Ausführungsumgebung (Hardware, LLM-Inferenz) und nicht die kausale Logik der Spielwelt. Taktbasierte Time-to-Live (TTL) führt dazu, dass Anfragen während der Inferenzzeit des Partners verfallen (interaction_request_expired), obwohl dieser aktiv rechnet.
* Lösungsansatz:
  Vollständiger Verzicht auf ttl_ticks, tickbasiertes Queue-Polling und asyncio.wait_for.
  Interaktionen werden über asyncio.Future abgewickelt:
  * Unterstützt das Ziel die Aktion nicht sofort physisch/kognitiv, liefert es ein synchrones ImmediateResult.
  * Kann und will das Ziel reagieren, übergibt es ein PendingFuture. Der Anfragende wartet reaktiv auf die Erfüllung dieses Futures.

### 1.2 Universelles Interaktionsmodell (Command Pattern & Double Dispatch)
* Problem:
  Bisherige Koordinatoren (ConflictCoordinator, AgentProtocolService) besitzen fest verdrahtete Annahmen über Interaktionspartner (z. B. dass Blocker stets Agenten sind oder nur geredet werden kann). Ein stummer Agent oder ein unbeweglicher Stein führt zu Fehlannahmen oder Deadlocks.
* Lösungsansatz:
  Einführung des Command- und Double-Dispatch-Musters:
  1. InteractionCommand: Kapselt jede Aktion (talk, inspect, probe, push, evade).
  2. IInteractiveEntity: Jedes Objekt der Welt deklariert seine Fähigkeiten (EntityCapabilities) und entscheidet autonom in receive_interaction(), wie es auf ein Command reagiert.
  3. Scheitert eine Aktion an der Beschaffenheit des Ziels (z. B. talk an einem Stein), meldet das Ziel unmittelbar ImmediateResult(success=False, reason="NOT_COMMUNICATIVE").
  4. Dieses Feedback wird im Gedächtnis des Agenten (AgentMemory) dauerhaft verbucht, wodurch das LLM in Folgeentscheidungen alternative Strategien wählt.

### 1.3 Konsolidierung der Agenten-Zustandsverwaltung (FSM)
* Problem:
  Verteilte boolesche Flags (is_thinking, is_busy, is_waiting_for_reply, is_listening_to_peer) erzeugen inkonsistente Zwischenzustände und Race Conditions (wie versehentliches Überschreiben via agent.set_thinking = False).
* Lösungsansatz:
  Einführung einer expliziten Zustands-FSM (AgentLifecycleState) im Domänenmodell:
  * IDLE: Bereit für Aktionen oder Taktbewegung.
  * DELIBERATING: LLM-Inferenz läuft (physisch stationär, nimmt Nachrichten an).
  * WAITING_FOR_PEER: Wartet auf die Erfüllung eines offenen PendingFuture.
  * YIELDING: Führt Ausweichmanöver durch.
  * PASSING: Passiert die Engstelle.

### 1.4 Zielstack-Integration von Verhandlungsentscheidungen
* Problem:
  Entscheidungen dürfen den übergeordneten Kontext (z. B. Ost-Portal) nicht überschreiben.
* Lösungsansatz:
  * Primärziel auf Ebene 0 wird pausiert.
  * Vereinbarte Aktionen (Ausweichen, Nischen-Halt, kooperative Aufgaben) werden als temporäre Sub-Goals oben auf den Stack gepusht.
  * Nach Freigabe (Clearance) poppen die Sub-Goals ab und das Primärziel wird reaktiviert.

---

## 2. Komponenten-Architektur & Klassendesign

### 2.1 Domänenschicht (src/domain/)

#### A. Interaktionsmodelle (src/domain/models/interaction/)
* entity_capabilities.py:
  Definiert die Bitflags der Fähigkeiten:
  - NONE = 0
  - INSPECTABLE = 1
  - COMMUNICATIVE = 2
  - PUSHABLE = 4
  - TRAVERSABLE = 8

* commands.py:
  Basisklasse InteractionCommand und konkrete Commands:
  - TalkCommand(message: str, intent: Optional[str])
  - InspectCommand()
  - ProbeCommand()
  - EndDialogueCommand(reason: str, final_message: Optional[str])

* interaction_result.py:
  - ImmediateResult(success: bool, reason: str, payload: dict)
  - PendingFuture(future: asyncio.Future)

#### B. Agentenstatus (src/domain/models/agent/)
* agent_state.py:
  AgentLifecycleState (Enum) ersetzt redundante Status-Flags auf Agent.

#### C. Schnittstelle für Entitäten (src/domain/ports/interactive_entity.py)
* IInteractiveEntity:
  Definiert die Protokollmethode:
  def receive_interaction(self, command: InteractionCommand) -> Union[ImmediateResult, PendingFuture]

---

### 2.2 Anwendungsschicht (src/application/)

#### A. Vermittlungsdienst (src/application/services/interaction/interaction_dispatcher.py)
* Ersetzt die starre Queue-Verarbeitung in AgentProtocolService.
* Nimmt ein InteractionCommand von Agent A entgegen, leitet es an Entität B weiter und steuert die Rückmeldung:
  - Bei ImmediateResult: Direkte Rückgabe an Agent A und Aktualisierung von AgentMemory.
  - Bei PendingFuture: Versetzt Agent A in den Zustand WAITING_FOR_PEER und bindet die Reaktivierung an die Erfüllung des Futures.

#### B. Dialog-Orchestrator (src/application/services/coordination/dialogue_coordinator.py)
* Führt die Konversation als Ping-Pong-Sequenz.
* Bei Empfang einer Antwort wird das offene Future des Partners aufgelöst und bei Fortführung ein neues Future für den Folgezug geöffnet.
* Bei Einigung (accept) oder Abbruch (end_dialogue) wird das Future mit terminalem Status aufgelöst und die Session bereinigt.

---

## 3. Ablaufdiagramme

### 3.1 Unpassierbares / Stummes Objekt (z. B. Stein)
Agent A                      InteractionDispatcher                     Stein (WorldEntity)
   │                                   │                                        │
   │── dispatch(TalkCommand) ─────────>│                                        │
   │                                   │── receive_interaction(TalkCommand) ───>│
   │                                   │                                        │── Prüft Capabilities:
   │                                   │                                        │   COMMUNICATIVE nicht vorhanden
   │                                   │<── ImmediateResult(success=False, ─────┘
   │                                   │                    reason="NOT_COMMUNICATIVE")
   │<── ImmediateResult ───────────────┘
   │
   │── update_memory(Stein, is_conversational=False)
   │── Kognition wählt nächste Strategie (Inspect / Probe / Evasion)

### 3.2 Wechselseitiger Dialog zwischen zwei Agenten
Agent A (West)               InteractionDispatcher               Agent B (Ost)
   │                                   │                               │
   │── dispatch(TalkCommand) ─────────>│                               │
   │                                   │── receive_interaction(Talk) ──>│
   │                                   │                               │── Prüft Capabilities: COMMUNICATIVE aktiv
   │                                   │                               │── Erzeugt response_future_1
   │                                   │<── PendingFuture(fut_1) ──────┘
   │<── PendingFuture(fut_1) ──────────┘
   │
   │ [Zustand: WAITING_FOR_PEER]                                       │ [Zustand: DELIBERATING]
   │ (physikalisch stationär)                                          │ Inferenz läuft via LLM...
   │                                                                   │
   │                                                                   │── Antwort fertig: TalkAction("Bitte weichen Sie aus")
   │                                                                   │── Erzeugt eigenes response_future_2
   │                                                                   │── fut_1.set_result(Antwort, response_future_2)
   │<──────────────────────────────────────────────────────────────────│
   │
   │ [Zustand: DELIBERATING]                                           │ [Zustand: WAITING_FOR_PEER]
   │ Inferenz läuft via LLM...                                         │ (physikalisch stationär)
   │── Entscheidung: EndDialogueAction("Ich weiche aus", intent=accept)│
   │── fut_2.set_result(Terminierung) ────────────────────────────────>│
   │
   │ [Zustand: YIELDING]                                               │ [Zustand: PASSING]
   │ Stack: [Ost-Portal (paused), In Nische ausweichen (active)]        │ Stack: [West-Portal (active)]

---

## 4. Phasenbasierter TDD-Implementierungsplan

### Phase 1: Domänenmodelle, Capabilities & Command-Strukturen

#### Schritt 1.1: Capabilities & Commands
* Tests: tests/unit/domain/interaction/test_interaction_commands.py
  - test_entity_capability_flags: Prüft Kombinationen von Capabilities (COMMUNICATIVE, INSPECTABLE, PUSHABLE).
  - test_command_instantiation_and_validation: Prüft Payloads von TalkCommand, InspectCommand und EndDialogueCommand.
* Komponenten:
  - src/domain/models/interaction/entity_capabilities.py
  - src/domain/models/interaction/commands.py
  - src/domain/models/interaction/interaction_result.py

#### Schritt 1.2: AgentLifecycleState FSM
* Tests: tests/unit/domain/agent/test_agent_lifecycle_fsm.py
  - test_agent_state_transitions_valid: Erlaubte Wechsel (IDLE -> DELIBERATING -> WAITING_FOR_PEER -> IDLE).
  - test_agent_state_transitions_invalid_raises_error: Direkter Wechsel von WAITING_FOR_PEER zu YIELDING ohne Kognitionsauflösung wird defensiv verhindert.
* Komponenten:
  - src/domain/models/agent/agent_state.py
  - src/domain/models/agent/agent.py (Integration FSM, Beseitigung der Zuweisungsschwachstelle set_thinking = False)

---

### Phase 2: Double Dispatch & Reaktivität auf Entitätsebene

#### Schritt 2.1: receive_interaction auf WorldEntity und Agent
* Tests: tests/unit/domain/interaction/test_interactive_entities.py
  - test_static_obstacle_rejects_talk_immediately: Eine Wand oder ein Stein liefert synchron ImmediateResult(success=False, reason="NOT_COMMUNICATIVE").
  - test_agent_accepts_talk_and_returns_pending_future: Ein Agent mit COMMUNICATIVE liefert PendingFuture mit ungelöstem asyncio.Future.
  - test_agent_busy_queues_future_without_drop: Erhält ein Agent ein Command während DELIBERATING, wird das Future in der Mailbox gepuffert und nicht verworfen.
* Komponenten:
  - src/domain/models/world/world_entity.py
  - src/domain/models/agent/agent.py

---

### Phase 3: InteractionDispatcher & Asynchroner Handshake

#### Schritt 3.1: Dispatching & Epistemisches Feedback
* Tests: tests/unit/application/interaction/test_interaction_dispatcher.py
  - test_dispatch_immediate_failure_updates_requester_memory: Nach Ablehnung durch einen Stein ist agent.memory.is_conversational(target_id) == False.
  - test_dispatch_future_resolves_when_partner_replies: Simuliert asynchrones Lösen des Futures durch Partner; anfragender Agent erhält die Nachricht ohne Takt-Timeout.
* Komponenten:
  - src/application/services/interaction/interaction_dispatcher.py
  - src/domain/models/agent/agent_memory.py

---

### Phase 4: Integration in DialogueCoordinator & ConflictCoordinator

#### Schritt 4.1: Entkopplung der Verhandlung von Ticks & Stack-Steuerung
* Tests: tests/unit/application/coordination/test_dialogue_future_lifecycle.py
  - test_dialogue_ping_pong_turn_taking: Simuliert 3 Gesprächsrunden über Futures mit simulierter variabler Inferenzlatenz (ohne Taktverlust).
  - test_dialogue_agreement_pushes_evasion_goal_to_stack: Einigung versetzt den Weichenden in YIELDING und pusht "In Nische ausweichen" auf den Zielstack; Primärziel bleibt paused.
* Komponenten:
  - src/application/services/coordination/dialogue_coordinator.py
  - src/application/services/coordination/conflict_coordinator.py
  - src/application/services/coordination/agent_protocol_service.py (Bereinigung von veralteten TTL-Routinen)

---

### Phase 5: Integration & Showcase-Validierung

#### Schritt 5.1: Verhandlungsszenario im Engpass
* Tests: tests/integration/test_dialogue_showcase_resilience.py
  - test_corridor_negotiation_full_cycle_no_exceptions:
    Vollständiger Lauf aus main_dialogue_showcase.py:
    1. Alice und Bob treffen aufeinander (Swap-Kollision).
    2. Dialog startet über Futures ohne Lock-Timeout und ohne TTL-Drop.
    3. Bob akzeptiert, pusht Nischen-Ziel auf Stack und weicht nach (15, 3) aus.
    4. Alice passiert, Clearance signalisiert, Bobs Nischen-Halt poppt ab und Primärziel reaktiviert.
* Showcase-Verifikation:
  - Ausführung von python3 main_dialogue_showcase.py ohne Hänger und ohne Ausnahmen im Log.

---

## 5. Definition of Done (DoD)

1. Keine Takt-/Timeout-Kopplung: Weder ttl_ticks, noch Takt-Zähler noch asyncio.wait_for steuern das Warten auf Kognitionsergebnisse.
2. Capability-Konformität: Jede Welt-Entität implementiert receive_interaction(); inkompatible Anfragen liefern synchrones Feedback ohne Ressourcenblockade.
3. Deterministische Zustände: AgentLifecycleState schließt undefinierte Mischzustände aus; keine Überschreibung von Methoden durch Status-Booleans.
4. Zielstack-Integrität: Verhandlungsergebnisse werden sauber als Sub-Goals auf den Stack gepusht; das Primärziel bleibt pausiert erhalten und wird nach Abschluss reaktiviert.
5. Epistemische Gedächtnisaktualisierung: Ein gescheiterter Interaktionsversuch aktualisiert das Entitätswissen im Gedächtnis des Agenten sofort deterministisch.
6. Vollständige Testabdeckung: Alle Unit- und Integrationstests laufen unter python3 -m pytest fehlerfrei durch.