from __future__ import annotations

from typing import Any, Literal, Optional, cast
import instructor
import litellm
import json

from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.models.cognition import (
    BlockedResolution,
    NonVerbalBlockedResolution,
    DialogueResolution,
    GoalDecision,
    GoalEvaluation,
    TalkAction,
)


class InstructorCognitionAdapter(ICognitionProvider):
    def __init__(
        self,
        model_name: str = "ollama/llama3",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
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
        inspected = context.get("blocker_inspected", False)

        # STRATEGIE A1: Dynamisches Action-Masking
        if not allow_talk and self._blockage_strategy == "action_masking":
            system_prompt = (
                "Du steuerst einen Agenten in einer 2D-Grid-Simulation bei einer Blockade.\n\n"
                "Die blockierende Entität spricht nicht oder reagiert nicht. Verbale Kommunikation ('talk') steht nicht zur Auswahl.\n\n"
                "Handlungsanweisungen:\n"
                f"- Ist 'blocker_inspected' False: Wähle bevorzugt 'inspect' (target_agent_id='{context.get('blocker_id')}'), "
                "um den Typ der Entität zu analysieren und Schlussfolgerungen für ähnliche Objekte zu ziehen.\n"
                "- Ist das Objekt bereits inspiziert oder bekannt unpassierbar: Wähle 'reroute' (falls can_reroute True) oder 'wait'.\n"
                "- Formuliere deine Begründung im Feld 'thought'."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ]
            params = self._build_request_params(NonVerbalBlockedResolution, messages)
            res: NonVerbalBlockedResolution = await self._client.chat.completions.create(**params)
            return BlockedResolution(
                thought=res.thought,
                action=res.action,
                new_sub_goal=res.new_sub_goal,
                complete_sub_goal=res.complete_sub_goal,
            )

        # STRATEGIE A3: Kognitive Selbstkorrektur / Reflexions-Schleife
        elif not allow_talk and self._blockage_strategy == "reflection":
            system_prompt = (
                "Du steuerst einen Agenten in einer 2D-Grid-Simulation bei einer Blockade.\n\n"
                "Verhaltensregeln:\n"
                "- Prüfe den Eintrag 'blocker_memory_status' im Kontext:\n"
                "  -> Falls 'blocker_inspected' False ist: Wähle 'inspect', um den Objekttyp zu ermitteln.\n"
                "  -> Falls die Entität stumm oder unpassierbar ist: Sprich sie NICHT an. Wähle 'reroute' oder 'wait'.\n"
                "- Formuliere deine Begründung im Feld 'thought'."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ]
            params = self._build_request_params(BlockedResolution, messages)
            resolution: BlockedResolution = await self._client.chat.completions.create(**params)

            if isinstance(resolution.action, TalkAction):
                messages.append({"role": "assistant", "content": resolution.model_dump_json()})
                messages.append({
                    "role": "user",
                    "content": (
                        f"KRITISCHER FEHLER: Du hast 'talk' gewählt, obwohl '{context.get('blocker_name')}' "
                        f"nicht spricht oder als stumm eingestuft ist.\n"
                        "Wähle 'inspect' (falls noch uninspiziert), 'reroute' oder 'wait'."
                    ),
                })
                retry_params = self._build_request_params(BlockedResolution, messages)
                resolution = await self._client.chat.completions.create(**retry_params)

            return resolution

        # Standardpfad: Verbale Interaktion potentiell möglich
        else:
            system_prompt = (
                "Du steuerst einen Agenten in einer 2D-Grid-Simulation bei einer Blockade.\n\n"
                "Verhaltensregeln:\n"
                f"- Ist 'blocker_inspected' False: Es wird dringend empfohlen, die Entität zuerst per 'inspect' "
                f"(target_agent_id='{context.get('blocker_id')}') zu analysieren, um ihren Typ und ihr Wesen zu bestimmen.\n"
                "- Ist die Entität inspiziert und potentiell ansprechbar: Wähle 'talk'.\n"
                "- 'target_agent_id' MUSS exakt der 'blocker_id' entsprechen.\n"
                "- Wähle 'reroute' nur, wenn 'can_reroute' True ist.\n"
                "- Formuliere deine Begründung im Feld 'thought'."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ]
            params = self._build_request_params(BlockedResolution, messages)
            return await self._client.chat.completions.create(**params)

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
        max_turns = context.get("max_dialogue_turns", 2)

        system_prompt = (
            "Du steuerst einen Agenten im Dialog in einer 2D-Grid-Simulation.\n\n"
            f"Aktuelle Runde: {turn_count} von maximal {max_turns}.\n\n"
            "Verhaltensregeln:\n"
            "- Prüfe 'partner_is_conversational', 'is_empty_response' und die bisherige Rundenanzahl.\n"
            "- Wenn das Gegenüber nicht sprechen kann, keine Antwort gibt oder die maximale Rundenanzahl erreicht ist:\n"
            "  -> Wähle zwingend 'action': {'action_type': 'end_dialogue', 'reason': '...'}.\n"
            "  -> Formuliere in 'final_message' eine explizite Verabschiedung mit Begründung.\n"
            "  -> Setze 'new_goal': {'name': 'In Nische ausweichen', 'intent_type': 'evade'}.\n"
            "- Wähle 'talk' nur, wenn der Partner spricht und Klärungsbedarf besteht.\n"
            "- 'target_agent_id' muss der 'partner_id' entsprechen."
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]

        params = self._build_request_params(DialogueResolution, messages)
        resolution: DialogueResolution = await self._client.chat.completions.create(**params)
        return resolution