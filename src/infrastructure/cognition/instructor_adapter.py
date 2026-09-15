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
    ImpassableNonVerbalBlockedResolution,
    NonVerbalBlockedResolution,
    PostInspectionBlockedResolution,
    ProbeOnlyBlockedResolution,
    TalkOnlyBlockedResolution,
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
        blocker_id = context.get("blocker_id")

        # 1. Epistemisch vollständig erschöpft: Weder Sprache noch Tastung offen
        if not allow_talk and not allow_probe:
            system_prompt = (
                "Du steuerst einen Agenten bei einer unüberwindbaren Blockade.\n\n"
                "Das Hindernis ist bereits vollständig erprobt: Es reagiert nicht auf Sprache und ist unpassierbar.\n"
                "Aktionen wie 'talk', 'inspect' und 'probe' sind unzulässig.\n\n"
                "Handlungsanweisungen:\n"
                "- Wähle 'reroute' (falls can_reroute True), um einen Alternativweg zu berechnen.\n"
                "- Wähle 'wait', falls du warten möchtest.\n"
                "- Formuliere deine Begründung im Feld 'thought'."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ]
            params = self._build_request_params(ImpassableNonVerbalBlockedResolution, messages)
            res_imp: ImpassableNonVerbalBlockedResolution = await self._client.chat.completions.create(**params)
            return BlockedResolution(
                thought=res_imp.thought,
                action=res_imp.action,
                new_sub_goal=res_imp.new_sub_goal,
                complete_sub_goal=res_imp.complete_sub_goal,
            )

        # 2. Inspiziert, aber noch mindestens ein Kanal offen
        if inspected:
            # Fall A: Beide Kanäle offen (talk und probe)
            if allow_talk and allow_probe:
                system_prompt = (
                    "Du steuerst einen Agenten bei einer Blockade.\n\n"
                    f"Das Hindernis wurde inspiziert. Ein erneutes 'inspect' ist NICHT zulässig.\n"
                    "Ebenso sind 'reroute' und 'wait' unzulässig, solange das Hindernis nicht vollständig erprobt ist.\n\n"
                    "Handlungsanweisungen:\n"
                    f"- Wähle 'probe' (target_agent_id='{blocker_id}'), um die Passierbarkeit durch Tasten zu prüfen.\n"
                    f"- Wähle 'talk' (target_agent_id='{blocker_id}'), um das Gegenüber anzusprechen.\n"
                    "- Formuliere deine Begründung im Feld 'thought'."
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                ]
                params = self._build_request_params(PostInspectionBlockedResolution, messages)
                res_post: PostInspectionBlockedResolution = await self._client.chat.completions.create(**params)
                return BlockedResolution(
                    thought=res_post.thought,
                    action=res_post.action,
                    new_sub_goal=res_post.new_sub_goal,
                    complete_sub_goal=res_post.complete_sub_goal,
                )

            # Fall B: Nur noch Erprobung offen (bereits als stumm verifiziert)
            if allow_probe and not allow_talk:
                system_prompt = (
                    "Du steuerst einen Agenten bei einer Blockade.\n\n"
                    "Das Gegenüber reagiert nicht auf Sprache. Das Hindernis ist jedoch physisch noch nicht erprobt.\n"
                    "Aktionen wie 'talk', 'wait' und 'reroute' sind unzulässig.\n\n"
                    f"- Wähle zwingend 'probe' (target_agent_id='{blocker_id}'), um die Passierbarkeit zu testen.\n"
                    "- Formuliere deine Begründung im Feld 'thought'."
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                ]
                params = self._build_request_params(ProbeOnlyBlockedResolution, messages)
                res_probe: ProbeOnlyBlockedResolution = await self._client.chat.completions.create(**params)
                return BlockedResolution(
                    thought=res_probe.thought,
                    action=res_probe.action,
                    new_sub_goal=res_probe.new_sub_goal,
                    complete_sub_goal=res_probe.complete_sub_goal,
                )

            # Fall C: Nur noch Sprache offen (bereits physisch erprobt)
            if allow_talk and not allow_probe:
                system_prompt = (
                    "Du steuerst einen Agenten bei einer Blockade.\n\n"
                    "Die physische Passierbarkeit ist bereits erprobt. Der verbale Kanal ist noch offen.\n"
                    "Aktionen wie 'probe', 'wait' und 'reroute' sind unzulässig.\n\n"
                    f"- Wähle 'talk' (target_agent_id='{blocker_id}'), um mit dem Gegenüber zu interagieren.\n"
                    "- Formuliere deine Begründung im Feld 'thought'."
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                ]
                params = self._build_request_params(TalkOnlyBlockedResolution, messages)
                res_talk: TalkOnlyBlockedResolution = await self._client.chat.completions.create(**params)
                return BlockedResolution(
                    thought=res_talk.thought,
                    action=res_talk.action,
                    new_sub_goal=res_talk.new_sub_goal,
                    complete_sub_goal=res_talk.complete_sub_goal,
                )

        # 3. Nicht inspiziert und keine Sprachoption
        if not allow_talk and self._blockage_strategy == "action_masking":
            system_prompt = (
                "Du steuerst einen Agenten bei einer Blockade. Verbale Kommunikation ('talk') steht nicht zur Auswahl.\n\n"
                "Handlungsanweisungen:\n"
                f"- Da das Objekt noch nicht inspiziert ist: Wähle bevorzugt 'inspect' (target_agent_id='{blocker_id}'), "
                "um Typ und Identität festzustellen.\n"
                f"- Wähle 'probe' (target_agent_id='{blocker_id}'), falls du direkt die Passierbarkeit prüfen willst.\n"
                "- Formuliere deine Begründung im Feld 'thought'."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ]
            params = self._build_request_params(NonVerbalBlockedResolution, messages)
            res_nonv: NonVerbalBlockedResolution = await self._client.chat.completions.create(**params)
            return BlockedResolution(
                thought=res_nonv.thought,
                action=res_nonv.action,
                new_sub_goal=res_nonv.new_sub_goal,
                complete_sub_goal=res_nonv.complete_sub_goal,
            )

        # 4. Standard-Erstkontakt vor Inspektion
        system_prompt = (
            "Du steuerst einen Agenten in einer 2D-Grid-Simulation bei einer Blockade.\n\n"
            "Handlungsanweisungen:\n"
            f"- Da 'blocker_inspected' False ist: Wähle zwingend 'inspect' (target_agent_id='{blocker_id}'), "
            "um den Typ der Entität oder Kachel zu ermitteln.\n"
            f"- 'target_agent_id' MUSS exakt '{blocker_id}' entsprechen.\n"
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