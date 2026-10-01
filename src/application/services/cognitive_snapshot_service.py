from __future__ import annotations

from typing import Optional

from src.application.services.need_service import NeedService
from src.domain.models.agent import Agent
from src.domain.models.agent_cognition import AgentCognitiveSnapshot
from src.domain.models.world_entity import WorldEntity


class CognitiveSnapshotService:
    """Erzeugt deterministische Kognitions-Snapshots und formulierte Gedankengänge."""

    def __init__(self, need_service: NeedService) -> None:
        self._need_service = need_service

    def create_snapshot(
        self,
        agent: Agent,
        tick: int,
        all_entities: Optional[list[WorldEntity]] = None,
    ) -> AgentCognitiveSnapshot:
        dominant_need = self._need_service.get_dominant_need(agent, only_urgent=False)
        dominant_level = agent.needs.get(dominant_need, 0.0) if dominant_need else 0.0

        primary_goal = agent.goals[0].name if agent.goals else None
        active_goal = agent.active_goal
        active_subgoal = (
            active_goal.name if active_goal and active_goal != agent.goals[0] else None
        )
        target_pos = active_goal.target_position if active_goal else None

        obstacle_desc, strategy, thought = self._derive_thought_and_strategy(
            agent=agent,
            dominant_need=dominant_need,
            dominant_level=dominant_level,
            primary_goal=primary_goal,
            active_subgoal=active_subgoal,
            target_pos=target_pos,
            all_entities=all_entities or [],
        )

        return AgentCognitiveSnapshot(
            agent_id=agent.id,
            agent_name=agent.name,
            tick=tick,
            dominant_need=dominant_need,
            dominant_need_level=dominant_level,
            primary_goal=primary_goal,
            active_subgoal=active_subgoal,
            current_position=agent.position,
            target_position=target_pos,
            perceived_obstacle=obstacle_desc,
            intended_strategy=strategy,
            formatted_thought=thought,
        )

    def _derive_thought_and_strategy(
        self,
        agent: Agent,
        dominant_need: Optional[str],
        dominant_level: float,
        primary_goal: Optional[str],
        active_subgoal: Optional[str],
        target_pos: Optional[object],
        all_entities: list[WorldEntity],
    ) -> tuple[Optional[str], str, str]:
        # 1. Dialog- & Denkzustand
        if agent.is_thinking:
            strategy = "Kognitive Reflexion"
            thought = "Ich überdenke meine nächsten Schritte und verarbeite Eindrücke."
            return None, strategy, thought

        if agent.is_waiting_for_reply and agent.interaction_partner_id:
            partner = next((e for e in all_entities if e.id == agent.interaction_partner_id), None)
            partner_name = partner.name if partner else agent.interaction_partner_id
            obstacle = f"Warte auf Antwort von {partner_name}"
            strategy = "Kommunikationsabstimmung"
            thought = f"Ich warte auf die Antwort von {partner_name}, um die Wegerechte zu klären."
            return obstacle, strategy, thought

        # 2. Ausweich- und Haltesituationen
        active_goal = agent.active_goal
        if active_goal and active_goal.is_evasion_hold:
            obstacle = (
                f"Vorfahrt für Agent {active_goal.yield_for_agent_id}"
                if active_goal.yield_for_agent_id
                else "Engpass blockiert"
            )
            strategy = "Defensives Halten in Nische"
            thought = f"Ich halte meine Position in der Nische, um {active_goal.yield_for_agent_id or 'dem Gegenverkehr'} die Durchfahrt zu gewähren."
            return obstacle, strategy, thought

        if agent.is_holding_for_junction:
            strategy = "Warten an Nischeneinmündung"
            thought = "Ich sichere die Nischeneinmündung und warte auf Freigabe des Korridors."
            return "Nischeneinmündung belegt", strategy, thought

        # 3. Physische Blockade auf Pfad
        if agent.has_path and not agent.is_busy:
            next_step = agent.path[0]
            blocker = next((e for e in all_entities if e.position == next_step and e.id != agent.id), None)
            if blocker:
                obstacle = f"{blocker.name} auf Kachel ({next_step.x}, {next_step.y})"
                strategy = "Konfliktkoordination & Ausweicharbitrierung"
                thought = f"Entität '{blocker.name}' versperrt mir bei ({next_step.x}, {next_step.y}) den Weg. Lösungsansatz: {strategy}."
                return obstacle, strategy, thought

        # 4. Zielbezogene Aktionen (Sub-Goals)
        if active_subgoal:
            if active_subgoal == "SubGoal: consume":
                strategy = "Nahrungsaufnahme"
                thought = f"Ziel erreicht. Ich verzehre die Nahrung, um mein Bedürfnis {dominant_need} ({dominant_level:.2f}) zu stillen."
                return None, strategy, thought

            if active_subgoal == "SubGoal: drink":
                strategy = "Flüssigkeitsaufnahme"
                thought = f"Wasserstelle erreicht. Ich trinke, um meinen Durst ({dominant_level:.2f}) zu stillen."
                return None, strategy, thought

            if active_subgoal == "SubGoal: rest":
                strategy = "Regeneration"
                thought = f"Ruhepunkt erreicht. Ich ruhe mich aus, um meine Erschöpfung ({dominant_level:.2f}) abzubauen."
                return None, strategy, thought

            if active_subgoal.startswith("SubGoal: explore"):
                strategy = "Gezielte Grenzerkundung"
                target_str = f" nach ({target_pos.x}, {target_pos.y})" if hasattr(target_pos, "x") else ""
                thought = f"Ich benötige {dominant_need or 'Ressourcen'}. Da keine Quelle bekannt ist, erkunde ich unbekanntes Terrain{target_str}."
                return None, strategy, thought

            if active_subgoal == "SubGoal: move_to":
                strategy = "Zielgerichtete Navigation"
                target_str = f" nach ({target_pos.x}, {target_pos.y})" if hasattr(target_pos, "x") else ""
                thought = f"Ich bewege mich im Rahmen von '{primary_goal}'{target_str}."
                return None, strategy, thought

        # 5. Primärziel aktiv ohne Sub-Goals
        if primary_goal:
            strategy = "Pfadnavigation"
            target_str = f" nach ({target_pos.x}, {target_pos.y})" if hasattr(target_pos, "x") else ""
            thought = f"Mein Ziel ist '{primary_goal}'{target_str}. Der Pfad ist frei."
            return None, strategy, thought

        # 6. Leerlauf
        strategy = "Bereitschaft"
        if dominant_need and self._need_service.is_need_urgent(agent, dominant_need):
            thought = f"Ich spüre dringenden Bedarf an {dominant_need} ({dominant_level:.2f}) und benötige einen Handlungsplan."
        else:
            thought = "Aktuell liegen keine Aufgaben vor. Ich beobachte die Umgebung."
        return None, strategy, thought