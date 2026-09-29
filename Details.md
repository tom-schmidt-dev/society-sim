# DOKUMENTATION & FACHLICHES BACKUP: DETERMINISTISCHE MULTI-AGENTEN-KORRIDORSTEUERUNG

================================================================================
ARBEITSANWEISUNG ZUR NUTZUNG DIESES REFERENZ-DOKUMENTS
================================================================================
Dieses Dokument dient als deterministisches Referenz- und Validierungs-Backup 
für das gesamte Systemverhalten der Multi-Agenten-Simulation.

PFLICHTANWEISUNG FÜR DEN AGENTEN VOR UND NACH JEDEM IMPLEMENTIERUNGSSCHRITT:
1. VOR Beginn der Bearbeitung eines Teilmoduls oder Schritts aus Implementierungsplan.md:
   - Lies den entsprechenden Abschnitt in den folgenden Ausformulierungen detailliert gegen.
   - Verifiziere das korrekte, vollständige mathematische und architektonische Verständnis
     der jeweiligen Invarianten, Abhängigkeiten und Schnittstellen, bevor Code generiert wird.
2. NACH Abschluss der Implementierung und vor dem Commit/Abschluss des Schrittes:
   - Gleiche den neu implementierten Code sowie die zugehörigen Unit-Tests erneut mit 
     den Ausformulierungen ab.
   - Prüfe deterministisch, ob alle definierten Randfälle, Formeln, Zustandsübergänge 
     und Fehlergrenzen exakt wie spezifiziert eingehalten wurden.
================================================================================

---

### FACHLICHE AUSFORMULIERUNGEN & INVARIANTEN

Die Wegfindung wird deterministisch ausgelegt, um Fehleranfälligkeit zu minimieren und Skalierbarkeit zu sichern. KI-Inferenz bleibt für übergeordnete Entscheidungen reserviert, die auf Kontext (Umgebungsbewertung, zukünftige Agenten-Eigenschaften und Erfahrungswerte aus dem aktuellen Durchlauf) basieren.

Im Standardfall (z. B. Agent A und Agent B oder ganze Konvois begegnen sich in einem schmalen Korridor ohne ausreichende Durchgangsbreite) entfällt die wiederholte KI-Anfrage vollständig. Die Konfliktlösung läuft deterministisch ab:

1. Wissensbasis & Kartenaggregation:
   - Agenten sammeln Wissen über die Map über Sensorik in ihrer individuellen 'AgentMentalMap'. Ein Zugriff auf eine globale Karte ('WorldGrid') findet für Entscheidungen zu keinem Zeitpunkt statt.
   - Für die Konfliktlösung und Pfadberechnung werden die mentalen Karten aller beteiligten Agenten temporär zu einer gemeinsamen Arbeitskarte ('FusedMentalMap') aggregiert.
   - Statisches Geländewissen (Wände, feste Hindernisse, erkundete Korridore) wird nach dem Manöver dauerhaft in die individuellen mentalen Karten übernommen. Dynamische Belegungen durch fremde Agenten verbleiben flüchtig und werden nach Beendigung des Manövers verworfen.
   - Bei Kachelkonflikten mit identischem Zeitstempel gilt das Vorsichtsprinzip: 'TileKnowledge.OBSTACLE' überschreibt 'TileKnowledge.WALKABLE', dieses überschreibt 'TileKnowledge.UNKNOWN'.

2. Nischensuche & Geometrie:
   - Auf Basis der aggregierten Karte wird deterministisch für jeden Agenten bzw. Konvoi der kürzeste Weg zu einer passenden Nische berechnet.
   - Die Nische muss ausreichend Platz für die Körpergröße des ausweichenden Agenten bieten (Standard: 1x1 für alle Agenten, jedoch als Attribut 'footprint' auf der Entität hinterlegt und konfigurierbar).
   - Während der ausweichende Agent die Nische besetzt, muss der verbleibende Korridor eine freie Mindestbreite aufweisen, die dem transversalen Footprint des passierenden Agenten entspricht.

3. Deterministische Kommunikation:
   - Der gesamte Informationsaustausch erfolgt ausschließlich über standardisierte Text-Templates:
     * "Hier ist nicht genug Platz für uns beide." (bzw. "für unsere Gruppen.")
     * "Ja das stimmt. Ich habe eine Nische gesehen. Sie ist {x} Felder von mir entfernt. Ich könnte Platz machen."
     * a) "Meine Lücke ist {x} Felder entfernt. Ich mache dir kurz Platz."
     * b) "Für mich wäre der Weg weiter, sie ist {x} Felder entfernt. Bitte sei so lieb und mache mir kurz Platz."
     * "Aber wenn du in diese Lücke bei {coords} gehst, verlängert sich mein Weg um {x} Felder. Das wäre ein riesen Umweg für mich. Ich könnte dir auch anbieten in meine Lücke in {y} Feldern zu gehen. Passt das?"
     * a) "Ja, mein Weg ist dann insgesamt kürzer. Los geht's."
     * b) "Ja aber mein Weg wäre dann noch weiter. Bitte mach Platz."
     * "Gleichstand der Nutzenfunktion. Der Münzwurf hat entschieden: Agent/Gruppe {winner_id} passiert zuerst."
     * "Passage abgeschlossen. Danke fürs Platz machen!"
     * "Gern geschehen! Setze Weg fort."

4. Phasentrennung & Topologische Clearance:
   - Das Einrücken in die Nische, das Warten in der Nische und das Fortsetzen des Weges sind streng getrennte Phasen ('EvasionPhase').
   - Der ausweichende Agent verharrt in der Nische ('is_evasion_hold=True'), bis der Partner den Engpass passiert hat.
   - Das Kriterium für die Wiederaufnahme des Weges ist NIEMALS das bloße Verlassen des Sichtfeldes (Gefahr von Starvation bei langen Korridoren oder Kollisionen an Kurven).
   - Es gilt ausschließlich die topologische Clearance: Der passierende Agent muss eine Sicherheitsdistanz zur Kreuzung/Mündung ('junction') erreicht haben:
     L1(pos, junction) >= Margin (mit Margin = max(Width, Length) + 1, bei 1x1: Margin = 2)
     ODER sein Ziel außerhalb der eingleisigen Konfliktzone erreicht haben (pos == destination AND pos NOT IN CorridorZone),
     UND die Junction darf nicht mehr im weiteren Pfad des passierenden Agenten liegen.
   - Erst nach Erfüllung dieser Bedingung erfolgt das Dankessignal und die Quittung.

5. Kostenminimierung & 70/30-Nutzenfunktion:
   - Es wird stets das Szenario gewählt, das die Summe der Zusatzschritte aller beteiligten Einheiten minimiert (Weg zur Nische plus Re-Integration in den Primärpfad plus eventuelle Warteverzögerungen der Gegenpartei).
   - Bei der Arbitrierung gehen der normalisierte Wegunterschied zu 70 % und die normalisierten Charaktereigenschaften (Charisma, Assertiveness) zu 30 % ein.
   - Wertebereich der Traits: [0.0, 1.0] pro Merkmal, MaxTraitRange = 2.0.
   - Formel: Score = 0.7 * Advantage_cost_norm + 0.3 * Advantage_trait_norm.
   - Beide Terme werden vorab auf [-1.0, 1.0] normalisiert.
   - Aktuell betragen die Traits aller Agenten standardmäßig 0.0. Dies führt dank des additiven Modells nicht zu Divisions- oder Multiplikationsfehlern: Bei identischen Merkmalen (0.0 == 0.0) entscheidet allein der normalisierte Kostenvorteil.
   - Grenzfallprüfung vor Quotientenbildung: Findet nur eine Partei eine Nische, weicht diese ohne Score-Berechnung direkt aus. Findet keine Seite eine Nische, wird direkt Backtracking initiiert.

6. Deterministischer Münzwurf:
   - Ein Münzwurf wird ausschließlich dann ausgelöst, wenn die Nutzenfunktion einen absoluten Gleichstand aufweist (Score == 0.0).
   - Der Münzwurf muss prozess- und replay-stabil sein. Pythons internes 'hash()' ist verboten.
   - Es wird SHA-256 über die lexikografisch sortierten IDs der Verhandlungsführer und den aktuellen Simulationstakt verwendet:
     digest = int(sha256(f"{min_id}:{max_id}:{tick}").hexdigest(), 16)
     winner = min_id if (digest % 2 == 0) else max_id

7. Konvoiführung vs. Richtungssuche:
   - Bewegen sich Agenten gemeinsam in dieselbe Richtung (z. B. ein Verband oder Partner, die gemeinsam eine Nische ansteuern), schalten sie in den synchronisierten Konvoi-Modus.
   - Abstandsinvariante: In jedem Takt t gilt auf gerader Strecke 1 <= L1(pos_A, pos_B) <= 2. Ein Auseinanderdriften oder Verlieren ist mathematisch ausgeschlossen.
   - Die heuristische Suche nach Entitäten mit unvollständigen Koordinaten ('DirectionalFrontierSearch' / 'TargetSearchService') bleibt als separates Werkzeug vollständig isoliert von der Konvoikoordination erhalten.

8. Konvois in Nischen (Multi-Agent-Nischenpackung):
   - Müssen mehrere Agenten eines Verbands ausweichen, evaluiert 'MultiAgentNichePacker' Nischenkonfigurationen, deren Gesamtzellenzahl mindestens der Summe aller Agentenausdehnungen entspricht.
   - Unterstützte Geometrien:
     * Tiefe Sackgassen-Nischen (Pocket Niches) mit einer gemeinsamen Mündung.
     * Sequenziell verteilte Nischen entlang des Korridors (kollisionsfreie, monotone Zuweisung).
     * Bypässe und Parallelkorridore.
   - Während alle Agenten des nachgebenden Konvois ihre Nischenplätze einnehmen, muss die Mindestdurchgangsbreite für den passierenden Konvoi ununterbrochen gewährleistet sein.

9. Konvoi-Permutation (Lateral Sidestep & Gap Closure):
   - Verlässt ein Konvoi eine Sackgassen-Nische (LIFO) oder liegt das Ziel des vorderen Agenten hinter dem Ziel des nachfolgenden Agenten, wird die Reihenfolge dynamisch permutiert:
     * 'LATERAL_STEP_OUT': Der vordere Agent tritt transversal auf eine freie Nachbarzelle orthogonal zur Marschachse aus.
     * 'GAP_CLOSURE': Der nachfolgende Agent rückt auf der Hauptachse vor und schließt die Lücke.
     * 'RE_INSERTION': Der ausgetretene Agent reiht sich hinter dem Vorbeiziehenden wieder ein.
   - Die interne Konvoi-Liste wird über swap(i, i+1) deterministisch aktualisiert.

10. Geordnetes Abfließen (Mid-Convoy Branching):
    - Wird ein neuer Pfad an einer Position frei, an der sich die Mitte eines Konvois befindet:
      * Der an der Abzweigung stehende Agent schert als erster ein und wird zum Leader des neuen Konvois bestimmt.
      * Die verbleibenden Hälften (Head und Tail) fließen geordnet über die Abzweigung ein:
        - Modus A: Reißverschlussverfahren (alternierend ein Schritt aus Head, ein Schritt aus Tail).
        - Modus B: Sequenzielles Einfließen (die Hälfte mit kürzerer Räumdistanz fließt zuerst komplett ab).

11. Backtracking bei Null-Nischen:
    - Findet keine Partei eine Nische, wird für beide Gruppen die Distanz des jeweiligen Konvoi-Endes (Tail) zur nächsten freien Verzweigung berechnet.
    - Die Gruppe mit der kürzeren Distanz weicht im Rückwärtsgang zurück bis zu einer explizit definierten Koordinate ('backtracking_junction_target').
    - Die Gegenpartei wartet oder rückt mit Sicherheitsabstand (L1 >= 2) nach, bis die Verzweigung vollständig geräumt und gesichert ist.

12. Zweiphasen-Commit der Bewegung ('MovementSyncService'):
    - Schrittwünsche werden im Taktzyklus erst dann atomar auf das Grid übertragen, wenn die Bewegungswünsche aller beteiligten Agenten der lokalen Konflikt-Zusammenhangskomponente kollisionsfrei validiert wurden.
    - Konfliktfreie Agenten außerhalb dieser Zusammenhangskomponente werden nicht blockiert.
    - Validierungsreihenfolge:
      * Vorwärtsbewegung: Front-to-Tail (vom Leader zum Tail).
      * Rückwärtsbewegung (Backtracking): Tail-to-Front (vom Tail zum Leader).
    - Erst nach erfolgreicher Validierung aller Schritte innerhalb der Komponente erfolgt der Commit synchron im selben Takt.

13. Gruppen-Suspension & Unterbrechbarkeit:
    - Alle Phasen sind deterministisch deklariert und unterbrechbar.
    - Tritt während des Ablaufs eine Blockade auf (z. B. Nische durch dritten Agenten besetzt oder unvorhergesehenes Hindernis aufgedeckt), wird die Phase unterbrochen ('SUSPENDED').
    - Die Ziele aller Gruppenmitglieder werden synchron auf 'status="suspended"' gesetzt, erhalten das Flag 'is_group_goal=True', eine einheitliche 'group_id' sowie die Liste aller 'participant_ids' inklusive State-Snapshot.
    - Ein übergeordnetes 'ResolutionGoal' übernimmt synchron auf allen Stacks die Steuerung.
    - Nach Auflösung des Problems prüfen alle Mitglieder der Gruppe synchron die Gültigkeit ihrer Ursprungspfade auf der aktualisierten 'FusedMentalMap', bevor sie synchron auf 'active' zurückkehren.
    - Deadlock-Schutz: 'MAX_SUSPENSION_TICKS = 1000'.

14. Prüfung und Refactoring des Zielmechanismus:
    - Der bestehende Mechanismus in 'GoalService', 'Goal' und 'InteractionRequest' ist daraufhin zu prüfen, ob er Gruppen-Flags, State-Snapshots, Konvoiziele und die Phasen-FSM vollständig unterstützt, und bei Bedarf konsequent zu erweitern bzw. zu refaktorisieren.