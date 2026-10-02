from __future__ import annotations

import json
from typing import Any, Literal, Optional, cast
import re
import instructor
import litellm

from src.domain.models.planning.cognition import (
    BlockedResolution,
    DialogueResolution,
    GoalDecision,
    GoalEvaluation,
    SocialReflection,
)
from src.domain.ports.cognition_provider import ICognitionProvider
from pydantic import BaseModel
from src.domain.models.planning.planning import (
    ActionType,
    AgentCognitiveContext,
    PlanDecomposition,
    SubGoalIntent,
)


class InstructorCognitionAdapter(ICognitionProvider):
    def __init__(
        self,
        model_name: str = "ollama/llama3",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        blockage_strategy: Literal["action_masking", "reflection"] = "action_masking",
    ) -> None:
        self._model_name = model_name
        self._api_base = api_base
        self._api_key = api_key
        self._temperature = temperature
        self._blockage_strategy: Literal["action_masking", "reflection"] = blockage_strategy

        self._client = cast(
            instructor.AsyncInstructor,
            instructor.from_litellm(litellm.acompletion),
        )

    def _build_request_params(
        self, response_model: type[Any], messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._model_name,
            "response_model": response_model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._api_base:
            params["api_base"] = self._api_base
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    async def decide_next_goal(self, context: dict[str, Any]) -> GoalDecision:
        system_prompt = (
            "Du bist das Kognitionsmodul eines autonomen Agenten in einer Gesellschaftssimulation.\n"
            "Analysiere den Status und die Umgebung des Agenten und wähle ein plausibles nächstes Ziel.\n"
            "Formuliere deine internen Gedanken im Feld 'thought' und gib ein valides Ziel mit Begründung an."
        )

        user_content = (
            f"Agenten-ID: {context.get('agent_id')}\n"
            f"Name: {context.get('name')}\n"
            f"Aktuelle Position: ({context.get('current_x')}, {context.get('current_y')})\n"
            f"Energie: {context.get('energy')}/100\n"
            f"Verfügbare Orte auf der Karte: {context.get('available_locations')}\n"
            f"Kürzliche Ereignisse:\n{context.get('recent_events_summary', 'Keine.')}"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        params = self._build_request_params(GoalDecision, messages)
        decision: GoalDecision = await self._client.chat.completions.create(**params)
        return decision

    async def resolve_blockage(self, context: dict[str, Any]) -> BlockedResolution:
        allow_talk = context.get("allow_talk", True)
        allow_probe = context.get("allow_probe", True)
        inspected = context.get("blocker_inspected", False)
        can_reroute = context.get("can_reroute", False)
        blocker_id = context.get("blocker_id")
        blocker_name = context.get("blocker_name", "Unbekannt")

        # 1. Dynamische Ermittlung der situativ zulässigen Handlungsoptionen
        available_choices: list[str] = []
        instructions: list[str] = []

        if not inspected:
            available_choices.extend(["inspect", "probe", "wait", "abort"])
            instructions.append(
                f"- 'inspect': Untersuche {blocker_name} (target_agent_id='{blocker_id}'), "
                "um den Typ der Entität zu ermitteln."
            )
            instructions.append(
                f"- 'probe': Führe einen physischen Tast-/Drucktest auf {blocker_name} durch "
                f"(target_agent_id='{blocker_id}'), um direkte Passierbarkeit zu testen."
            )
        else:
            if allow_talk:
                available_choices.append("talk")
                instructions.append(
                    f"- 'talk': Sprich mit {blocker_name} (target_agent_id='{blocker_id}'). "
                    "Setze 'intent' zwingend auf 'request_yield' (um Platz bitten) oder 'offer_yield' (selbst ausweichen)."
                )
            if allow_probe:
                available_choices.append("probe")
                instructions.append(
                    f"- 'probe': Prüfe physisch die Passierbarkeit von {blocker_name} (target_agent_id='{blocker_id}')."
                )
            if can_reroute:
                available_choices.append("reroute")
                instructions.append("- 'reroute': Berechne einen Alternativweg um das Hindernis herum.")

            available_choices.extend(["wait", "abort"])

        instructions.extend([
            "- 'wait': Halte die Position für einige Ticks, falls das Hindernis temporär ist.",
            "- 'abort': Breche das aktuelle Vorhaben ab und gib den Ziel-Stack frei.",
        ])

        # 2. Prägnanter, fokussierter System-Prompt mit situativen Leitplanken
        system_prompt = (
            "Du steuerst einen autonomen Agenten bei einer Blockadesituation in einer 2D-Grid-Simulation.\n\n"
            f"Zulässige Aktionen in dieser Situation: {', '.join(available_choices)}.\n\n"
            "Handlungsanweisungen:\n" + "\n".join(instructions) + "\n\n"
            "Regeln:\n"
            "1. Wähle ausschließlich eine der explizit als zulässig genannten Aktionen.\n"
            "2. Begründe deine Entscheidung nachvollziehbar im Feld 'thought'.\n"
            "3. Bei 'talk', 'inspect' und 'probe' MUSS target_agent_id exakt der Blocker-ID entsprechen.\n"
            "4. Bei 'talk' MUSS ein passender 'intent' ('request_yield' oder 'offer_yield') gesetzt werden."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]

        params = self._build_request_params(BlockedResolution, messages)
        resolution: BlockedResolution = await self._client.chat.completions.create(**params)
        return resolution

    async def evaluate_goal_status(self, context: dict[str, Any]) -> GoalEvaluation:
        system_prompt = (
            "Du analysierst den Fortschritt eines Agenten in einer Simulation.\n"
            "Prüfe anhand des Status, der Position und der jüngsten Ereignisse, ob das aktuelle Teilziel erfüllt ist.\n"
            "Setze 'is_completed' auf True, wenn das Teilziel beendet werden kann und der Agent zum übergeordneten Hauptziel zurückkehren soll.\n"
            "Begründe deine Entscheidung in 'thought' und 'reason'."
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]

        params = self._build_request_params(GoalEvaluation, messages)
        evaluation: GoalEvaluation = await self._client.chat.completions.create(**params)
        return evaluation

    async def respond_to_dialogue(self, context: dict[str, Any]) -> DialogueResolution:
        turn_count = context.get("dialogue_turn_count", 1)
        recommended_role = context.get("recommended_role", "yield")
        dist_self = context.get("evasion_distance_self")
        dist_partner = context.get("evasion_distance_partner")
        incoming_intent = context.get("incoming_intent")
        peer_bid_farewell = context.get("peer_bid_farewell", False)

        assertiveness = context.get("assertiveness", 0.5)
        charisma = context.get("charisma", 0.5)
        conversation_summary = context.get("conversation_summary", "")


        prompt_lines = [
            "Du steuerst die Verhandlung eines Agenten bei einer Blockade in einer 2D-Simulation.\n",
            f"Aktuelle Verhandlungsrunde: {turn_count}.",
            f"Charaktereigenschaften: Durchsetzungsstärke (Assertiveness)={assertiveness:.2f}, Charisma={charisma:.2f}.",
            f"Eigener Weg zur nächsten Nische: {dist_self} Schritte.",
            f"Weg des Partners zur nächsten Nische: {dist_partner} Schritte.",
            f"Empfohlene Rolle laut Geometrie: '{recommended_role}'.",
            f"Letzter eingehender Intent des Partners: '{incoming_intent}'.",
        ]

        if conversation_summary:
            prompt_lines.append(f"\nBisherige Zusammenfassung des bisherigen Gesprächsverlaufs:\n{conversation_summary}")

        social_memories = context.get("social_memories") or context.get("episodic_memories") or []
        if social_memories:
            prompt_lines.append("\nErinnerungen an frühere Interaktionen mit dem Partner:")
            for mem in social_memories:
                prompt_lines.append(f"- {mem}")

        if peer_bid_farewell:
            prompt_lines.extend([
                "\nSTATUS: Dein Partner hat sich verabschiedet (peer_bid_farewell=True).",
                "- Reguläre Aktion: Bestätige die Verabschiedung mit 'end_dialogue' (final_message formulieren).",
                "- Ausnahme: Nur wenn du zwingend noch Hilfe benötigst oder die Situation für dich ungelöst ist, "
                "wähle 'talk' mit negotiation_intent='reject'. Du musst im Feld 'reason' und in 'message' "
                "zwingend begründen, was genau noch ungeklärt ist.",
            ])
        else:
            prompt_lines.extend([
                "\nVerhaltens- und Entscheidungsregeln basierend auf deiner Persönlichkeit:",
                "- Hohe Durchsetzungsstärke (assertiveness >= 0.7): Beharre auf deinem Vorrang, argumentiere nachdrücklich und nutze 'request_yield' oder 'reject'. Gib nur bei zwingenden Gründen nach.",
                "- Niedrige Durchsetzungsstärke (assertiveness <= 0.3): Sei kompromissbereit, weiche Konflikten aus und biete eher 'offer_yield' an, wenn eine Lücke nahe ist.",
                "- Hohes Charisma (charisma >= 0.7): Formuliere überzeugende, diplomatische Argumente statt reiner Befehle.",
                "\nEntscheidungsregeln für 'negotiation_intent':",
                "- 'offer_yield': Du bietest an, selbst in eine Nische auszuweichen (nur wenn du nachgeben willst).",
                "- 'request_yield': Du forderst den Partner auf, Platz zu machen.",
                "- 'accept': Du nimmst das Ausweichangebot des Partners an und passierst.",
                "- 'reject': Du lehnst die Forderung des Partners ab, widersprichst oder beharrst auf deiner Position.",
                "\nKommunikationsregeln:",
                "- Setze 'negotiation_intent' konsistent zu deiner Äußerung.",
                "- Es gibt keine Rundenbeschränkung: Argumentiere, diskutiere oder streite, solange keine Einigung besteht.",
                "- Wenn eine Einigung erzielt wurde: Wähle 'end_dialogue' mit finaler Bestätigung.",
            ])

        situational_notes = context.get("situational_notes", [])
        if situational_notes:
            prompt_lines.append("\nAktuelle situative Verfassung / Innere Haltung:")
            for note in situational_notes:
                prompt_lines.append(f"- {note}")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": "\n".join(prompt_lines)},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]

        params = self._build_request_params(DialogueResolution, messages)
        resolution: DialogueResolution = await self._client.chat.completions.create(**params)
        return resolution

    async def reflect_on_dialogue(self, context: dict[str, Any]) -> SocialReflection:
        partner_name = context.get("partner_name", "Unbekannt")
        recent_dialogues = context.get("recent_dialogues", [])

        system_prompt = (
            "Du bist das Reflexionsmodul eines autonomen Agenten in einer Gesellschaftssimulation.\n"
            f"Ein Gespräch mit {partner_name} wurde soeben beendet.\n"
            "Analysiere den vorliegenden Gesprächsverlauf und erstelle eine prägnante soziale Einschätzung:\n"
            "1. 'assessment': Beurteile den Charakter und das Verhalten der Person (z. B. dominant, kompromissbereit, stur, kooperativ).\n"
            "2. 'progression_summary': Fasse in 1-2 kurzen Sätzen zusammen, wie das Gespräch verlaufen ist und wer nachgegeben hat.\n"
            "Fasse dich kurz und präzise."
        )

        user_content = (
                f"Gesprächsverlauf mit {partner_name}:\n"
                + ("\n".join(recent_dialogues) if recent_dialogues else "Keine Aufzeichnungen vorhanden.")
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        params = self._build_request_params(SocialReflection, messages)
        try:
            reflection: SocialReflection = await self._client.chat.completions.create(**params)
            return reflection
        except Exception:
            return SocialReflection(
                assessment="Verhalten unauffällig.",
                progression_summary=f"Gespräch mit {partner_name} abgeschlossen.",
            )

    async def decompose_plan(
            self, context: AgentCognitiveContext
    ) -> PlanDecomposition:
        system_prompt = (
            "Du bist das Kognitions- und Planungsmodul eines autonomen Agenten in einer 2D-Gitter-Simulation.\n"
            "Deine Aufgabe ist es, für ein dringendes Bedürfnis des Agenten (z. B. 'hunger') einen strukturierten Handlungsplan zu erstellen.\n"
            "Analysiere die Vitalwerte, die aktuelle Position, bekannte Entitäten sowie abgerufene Erinnerungen ('episodic_memories').\n"
            "Formuliere deine Gedanken im Feld 'thought', benenne ein Primärziel ('primary_goal') "
            "und zerlege es in eine geordnete Liste von atomaren Teilzielen ('sub_goals').\n\n"
            "Zulässige Aktionen für Sub-Goals sind: 'move_to', 'explore', 'consume', 'wait', 'inspect'.\n"
            "Regeln:\n"
            "1. Wenn eine passende Ressource in 'known_entities' bekannt ist, plane 'move_to' gefolgt von 'consume'.\n"
            "2. Wenn keine Ressource in 'known_entities' bekannt ist, aber 'episodic_memories' relevante Quellen oder Standorte enthalten, plane 'move_to' zu den erinnerten Koordinaten.\n"
            "3. Wenn weder bekannte Entitäten noch zielführende Erinnerungen vorliegen, wähle zuerst 'explore'.\n"
            "4. Gib bei 'move_to' die Zielkoordinaten [x, y] und bei 'consume' die target_entity_id an."
        )

        user_content = (
            context.model_dump_json(indent=2)
            if isinstance(context, BaseModel)
            else json.dumps(context, ensure_ascii=False)
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        params = self._build_request_params(PlanDecomposition, messages)
        try:
            decomposition: PlanDecomposition = await self._client.chat.completions.create(**params)
            return decomposition
        except Exception:
            # Robuster Fallback bei Offline-Betrieb oder Verbindungsabbrüchen
            return self._heuristic_fallback_plan(context)

    def _heuristic_fallback_plan(
            self, context: AgentCognitiveContext
    ) -> PlanDecomposition:
        """Deterministischer Fallback-Planer bei nicht erreichbarem Sprachmodell."""
        if not any(e.is_consumable and e.last_known_position for e in
                   context.known_entities) and context.episodic_memories:
            for memory_text in context.episodic_memories:
                match = re.search(r"bei \((\d+),\s*(\d+)\)", memory_text)
                if match:
                    target_pos = (int(match.group(1)), int(match.group(2)))
                    need = context.urgent_need or "Bedürfnis"
                    return PlanDecomposition(
                        thought=f"Fallback: Erinnere mich an Quelle bei {target_pos}.",
                        primary_goal=f"{need} stillen",
                        sub_goals=[
                            SubGoalIntent(
                                action_type=ActionType.MOVE_TO,
                                target_position=target_pos,
                                description=f"Gehe zu erinnerter Position {target_pos}",
                            ),
                            SubGoalIntent(
                                action_type=ActionType.CONSUME,
                                description="Konsumiere gefundene Ressource",
                            ),
                        ],
                    )
        consumable = next(
            (e for e in context.known_entities if e.is_consumable and e.last_known_position),
            None,
        )
        if consumable and consumable.last_known_position:
            target_pos = (consumable.last_known_position.x, consumable.last_known_position.y)
            return PlanDecomposition(
                thought="Fallback: Bekannte Ressource direkt ansteuern und konsumieren.",
                primary_goal="Hunger stillen",
                sub_goals=[
                    SubGoalIntent(
                        action_type=ActionType.MOVE_TO,
                        target_position=target_pos,
                        description=f"Gehe zu {consumable.name}",
                    ),
                    SubGoalIntent(
                        action_type=ActionType.CONSUME,
                        target_entity_id=consumable.entity_id,
                        description=f"Konsumiere {consumable.name}",
                    ),
                ],
            )

        return PlanDecomposition(
            thought="Fallback: Keine Ressource bekannt. Starte Exploration.",
            primary_goal="Nahrung suchen",
            sub_goals=[
                SubGoalIntent(
                    action_type=ActionType.EXPLORE,
                    description="Erkunde unbekanntes Terrain nach Nahrung",
                ),
                SubGoalIntent(
                    action_type=ActionType.CONSUME,
                    description="Konsumiere gefundene Nahrung",
                ),
            ],
        )