from __future__ import annotations

import json
from typing import Any, Literal, Optional, cast
import instructor
import litellm

from src.domain.models.cognition import (
    BlockedResolution,
    DialogueResolution,
    GoalDecision,
    GoalEvaluation,
)
from src.domain.ports.cognition_provider import ICognitionProvider


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

        prompt_lines = [
            "Du steuerst die Verhandlung eines Agenten bei einer Blockade in einer 2D-Simulation.\n",
            f"Aktuelle Verhandlungsrunde: {turn_count}.",
            f"Eigener Weg zur nächsten Nische: {dist_self} Schritte.",
            f"Weg des Partners zur nächsten Nische: {dist_partner} Schritte.",
            f"Empfohlene Rolle laut Geometrie: '{recommended_role}'.",
            f"Letzter eingehender Intent des Partners: '{incoming_intent}'.",
        ]

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
                "\nEntscheidungsregeln für 'negotiation_intent':",
                "- 'offer_yield': Du bietest an, in die Nische auszuweichen.",
                "- 'request_yield': Du forderst den Partner auf, auszuweichen.",
                "- 'accept': Du stimmst dem Vorschlag des Partners zu.",
                "- 'reject': Du lehnst den Vorschlag ab und machst einen Gegenvorschlag.",
                "\nKommunikationsregeln:",
                "- Setze 'negotiation_intent' passend zu deiner Absicht.",
                "- Wenn eine Einigung erzielt wurde: Wähle 'end_dialogue' mit finaler Bestätigung.",
                "- Halte die Unterhaltung kurz und zielführend.",
            ])

        messages: list[dict[str, str]] = [
            {"role": "system", "content": "\n".join(prompt_lines)},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]

        params = self._build_request_params(DialogueResolution, messages)
        resolution: DialogueResolution = await self._client.chat.completions.create(**params)
        return resolution