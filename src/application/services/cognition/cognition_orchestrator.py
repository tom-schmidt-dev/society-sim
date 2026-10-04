from __future__ import annotations

from typing import Any, Callable, Optional

from src.application.services.execution.action_executor import ActionExecutor
from src.application.services.cognition.cognitive_snapshot_service import CognitiveSnapshotService
from src.application.services.cognition.frontier_explorer import FrontierExplorer
from src.application.services.cognition.goal_service import GoalService
from src.application.services.lifecycle.need_service import NeedService
from src.application.services.cognition.plan_decomposition_service import PlanDecompositionService
from src.domain.models.agent.agent import Agent
from src.domain.models.agent.agent_cognition import AgentCognitiveSnapshot
from src.domain.models.agent.agent_memory import EntityFact
from src.domain.models.planning.events import SimulationEvent
from src.domain.models.planning.goal import Goal
from src.domain.models.world.position import Position
from src.domain.models.world.world_entity import WorldEntity
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.services.precondition_evaluator import PreconditionEvaluator


class CognitionOrchestrator:
    """Orchestriert Phase 1: Vitalwerte, hierarchische Plandekomposition und Sub-Goal-Ausführung."""

    def __init__(
        self,
        pathfinder: IPathfinder,
        logger: IEventLogger,
        goal_service: GoalService,
        need_service: NeedService,
        plan_decomposition_service: PlanDecompositionService,
        action_executor: ActionExecutor,
        frontier_explorer: Optional[FrontierExplorer] = None,
        precondition_evaluator: Optional[PreconditionEvaluator] = None,
        snapshot_service: Optional[CognitiveSnapshotService] = None,
        tick_provider: Optional[Callable[[], int]] = None,
    ) -> None:
        self._pathfinder = pathfinder
        self._logger = logger
        self._goal_service = goal_service
        self._need_service = need_service
        self._plan_decomposition_service = plan_decomposition_service
        self._action_executor = action_executor
        self._frontier_explorer = frontier_explorer or FrontierExplorer()
        self._precondition_evaluator = precondition_evaluator or PreconditionEvaluator()
        self._snapshot_service = snapshot_service or CognitiveSnapshotService(self._need_service)
        self._tick_provider = tick_provider or (lambda: 0)
        self._latest_snapshots: dict[str, AgentCognitiveSnapshot] = {}
        self._last_logged_snapshots: dict[str, AgentCognitiveSnapshot] = {}

    @property
    def latest_snapshots(self) -> list[AgentCognitiveSnapshot]:
        """Gibt die zuletzt erfassten Kognitions-Snapshots aller Agenten zurück."""
        return list(self._latest_snapshots.values())

    async def process_agent_needs_and_cognition(
        self, agents: list[Agent], entities: list[WorldEntity]
    ) -> None:
        """
        Phase 1 des Taktzyklus:
        - Stößt bei dringenden Bedürfnissen und leerem Zielstack die hierarchische Plandekomposition an.
        - Führt atomare Sub-Goals aus (Navigation, Erkundung, Konsum).
        - Aktualisiert Vitalwerte über den NeedService nur dann, wenn in diesem Takt keine Nahrung verzehrt wurde.
        """
        for agent in agents:
            if agent.is_sleeping:
                self._need_service.update_needs(agent)
                continue

            # 1. Plandekomposition nur bei leerem Zielstack (oder Fallback-Warten)
            has_only_fallback_goal = (
                len(agent.goals) == 1 and agent.goals[0].name.startswith("Warten")
            )

            if not agent.is_busy and (not agent.goals or has_only_fallback_goal):
                dominant_need = self._need_service.get_dominant_need(agent)
                if not isinstance(dominant_need, str):
                    if self._need_service.is_need_urgent(agent, "hunger"):
                        dominant_need = "hunger"
                    elif self._need_service.is_need_urgent(agent, "thirst"):
                        dominant_need = "thirst"
                    elif self._need_service.is_need_urgent(agent, "energy"):
                        dominant_need = "energy"
                    else:
                        dominant_need = None

                # Dringlichkeitsvalidierung verhindert Re-Planung nach Sättigung
                if dominant_need and self._need_service.is_need_urgent(agent, dominant_need):
                    if has_only_fallback_goal:
                        agent.goals.clear()
                    await self.trigger_plan_decomposition(agent, dominant_need)

            # 2. Sub-Goal-Ausführung (auf Schleifenebene: läuft in jedem aktiven Takt)
            consumed = False
            current_goal = agent.active_goal
            if current_goal and not agent.is_busy:
                if current_goal.name.startswith("SubGoal: move_to"):
                    self.handle_subgoal_move_to(agent, current_goal, entities)
                    current_goal = agent.active_goal

                if current_goal and not agent.is_busy:
                    if current_goal.name.startswith("SubGoal: consume"):
                        consumed = self.handle_subgoal_consume(agent, current_goal, entities)
                    elif current_goal.name.startswith("SubGoal: explore"):
                        self.handle_subgoal_explore(agent, current_goal)

            # 3. Vitalwerte nur aktualisieren, wenn in diesem Takt keine Nahrung verzehrt wurde
            if not consumed:
                self._need_service.update_needs(agent)

    def handle_subgoal_move_to(
        self,
        agent: Agent,
        current_goal: Goal,
        entities: Optional[list[WorldEntity]] = None,
    ) -> None:
        """Verwaltet Pfadzuweisung für direkte Navigations-Teilziele."""
        if not current_goal.target_position:
            self.interrupt_goal_for_replan(agent, reason="move_to ohne Zielkoordinaten")
            return

        # 1. Exakte Zielkachel erreicht
        if agent.position == current_goal.target_position:
            agent.clear_path()
            self._goal_service.pop_goal(agent)
            return

        # 2. Vorzeitiger Abschluss: Zielposition war die Ressource selbst und Agent grenzt bereits daran an
        if len(agent.goals) >= 2:
            next_goal = agent.goals[-2]
            if any(
                next_goal.name.startswith(prefix)
                for prefix in ("SubGoal: consume", "SubGoal: drink", "SubGoal: rest")
            ):
                resource_pos: Optional[Position] = None
                target_ent = self._find_target_entity_for_goal(next_goal, entities or [])
                if target_ent:
                    resource_pos = target_ent.position
                elif (
                    next_goal.target_entity_id
                    and next_goal.target_entity_id in agent.memory.known_entities
                ):
                    resource_pos = agent.memory.known_entities[
                        next_goal.target_entity_id
                    ].last_known_position
                elif next_goal.target_position:
                    resource_pos = next_goal.target_position

                if (
                    resource_pos is not None
                    and current_goal.target_position == resource_pos
                    and agent.position.manhattan_distance(resource_pos) <= 1
                ):
                    agent.clear_path()
                    self._goal_service.pop_goal(agent)
                    return

        # 3. Pfad berechnen und zuweisen, falls noch keiner existiert
        if not agent.has_path:
            path = self._pathfinder.find_path(
                agent.position,
                current_goal.target_position,
                agent.mental_map,
            )
            if path:
                agent.assign_path(path)
            else:
                self.interrupt_goal_for_replan(
                    agent, reason=f"Kein Pfad nach {current_goal.target_position}"
                )

    def _capture_and_log_snapshot(self, agent: Agent, entities: list[WorldEntity]) -> None:
        current_tick = self._tick_provider()
        snapshot = self._snapshot_service.create_snapshot(agent, current_tick, entities)
        last_logged = self._last_logged_snapshots.get(agent.id)

        has_changed = (
            last_logged is None
            or last_logged.dominant_need != snapshot.dominant_need
            or last_logged.primary_goal != snapshot.primary_goal
            or last_logged.active_subgoal != snapshot.active_subgoal
            or last_logged.perceived_obstacle != snapshot.perceived_obstacle
            or last_logged.intended_strategy != snapshot.intended_strategy
            or last_logged.formatted_thought != snapshot.formatted_thought
        )

        if has_changed:
            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=agent.id,
                    event_type="cognitive_snapshot",
                    summary=f"Gedanke von {agent.name}: {snapshot.formatted_thought}",
                    payload=snapshot.to_dict(),
                )
            )
            self._last_logged_snapshots[agent.id] = snapshot

        self._latest_snapshots[agent.id] = snapshot

    async def trigger_plan_decomposition(self, agent: Agent, need_name: str) -> None:
        """Erzeugt einen neuen hierarchischen Plan über den Kognitionsservice."""
        plan = await self._plan_decomposition_service.create_plan_for_need(agent, need_name)
        primary_goal_name = (
            plan.primary_goal
            if isinstance(plan.primary_goal, str)
            else str(plan.primary_goal)
        )
        primary_goal = Goal(name=primary_goal_name)
        self._goal_service.push_goal(agent, primary_goal)

        valid_subgoals_pushed = 0
        for intent in reversed(plan.sub_goals):
            target_pos = (
                Position(intent.target_position[0], intent.target_position[1])
                if intent.target_position
                else None
            )
            action_type_val = (
                intent.action_type.value
                if hasattr(intent.action_type, "value")
                else str(intent.action_type)
            )

            # Ungültige move_to-Aktionen ohne Koordinaten herausfiltern
            if action_type_val == "move_to" and target_pos is None:
                continue

            sub_goal = Goal(
                name=f"SubGoal: {action_type_val}",
                target_position=target_pos,
                target_entity_id=intent.target_entity_id,
                description=intent.description,
            )
            self._goal_service.push_goal(agent, sub_goal)
            valid_subgoals_pushed += 1

        # Fallback: Falls keine Sub-Goals übrig blieben, Exploration erzwingen
        if valid_subgoals_pushed == 0:
            self._goal_service.push_goal(
                agent,
                Goal(
                    name="SubGoal: explore",
                    description="Erkunde Umgebung nach Ressourcen",
                ),
            )


    def _resolve_target_categories_for_agent(self, agent: Agent) -> set[str]:
        primary_goal_name = agent.goals[0].name.lower() if agent.goals else ""
        if any(term in primary_goal_name for term in ("durst", "drink", "trinken", "wasser")):
            return self._precondition_evaluator.RESOURCE_CATEGORIES.get("drinkable", set())
        if any(term in primary_goal_name for term in ("energie", "rest", "ruhe", "schlaf", "erhol")):
            return self._precondition_evaluator.RESOURCE_CATEGORIES.get("rest_area", set())
        return self._precondition_evaluator.RESOURCE_CATEGORIES.get("consumable", set())

    def handle_subgoal_explore(self, agent: Agent, current_goal: Goal) -> None:
        """Evaluiert Ressourcenfunde und steuert Grenzkachel-Exploration."""
        target_categories = self._resolve_target_categories_for_agent(agent)
        discovered = self._precondition_evaluator.find_discovered_entity(
            agent, categories=target_categories
        )

        if discovered:
            self.interrupt_goal_for_replan(
                agent=agent,
                reason=f"Ressource vom Typ '{discovered.entity_type}' entdeckt",
                discovered_fact=discovered,
            )
        else:
            if not agent.has_path and not agent.is_busy:
                target_frontier = self._frontier_explorer.find_nearest_frontier(
                    agent.position, agent.mental_map
                )
                if target_frontier:
                    path = self._pathfinder.find_path(
                        agent.position, target_frontier, agent.mental_map
                    )
                    if path:
                        agent.assign_path(path)
                        current_goal.target_position = target_frontier

            if current_goal.target_position and agent.position == current_goal.target_position:
                # Nächste Grenzkachel ansteuern, falls noch keine Ressource entdeckt wurde
                next_frontier = self._frontier_explorer.find_nearest_frontier(
                    agent.position, agent.mental_map
                )
                if next_frontier:
                    path = self._pathfinder.find_path(
                        agent.position, next_frontier, agent.mental_map
                    )
                    if path:
                        agent.assign_path(path)
                        current_goal.target_position = next_frontier
                        return
                # Nur entfernen, wenn das gesamte Terrain aufgedeckt ist
                self._goal_service.pop_goal(agent)

    def handle_subgoal_consume(
        self, agent: Agent, current_goal: Goal, entities: list[WorldEntity]
    ) -> bool:
        """Führt Konsumaktion deterministisch aus."""
        target_entity = self._find_target_entity_for_goal(current_goal, entities)
        if not target_entity:
            self.interrupt_goal_for_replan(
                agent=agent, reason="Konsumziel nicht auffindbar"
            )
            return False

        current_tick = self._tick_provider()
        consumed = self._action_executor.execute_consume(
            agent=agent,
            target_entity=target_entity,
            all_entities=entities,
            incident_id=f"consume-t{current_tick}-{agent.id}",
        )
        if consumed:
            self._cleanup_satisfied_goal(agent, need_name="hunger")
            return True

        self.interrupt_goal_for_replan(
            agent=agent, reason=f"Konsum von {target_entity.name} fehlgeschlagen"
        )
        return False

    def handle_subgoal_drink(
        self, agent: Agent, current_goal: Goal, entities: list[WorldEntity]
    ) -> bool:
        """Führt Trinkaktion an einer Wasserquelle deterministisch aus."""
        target_entity = self._find_target_entity_for_goal(current_goal, entities)
        if not target_entity:
            return False

        current_tick = self._tick_provider()
        drank = self._action_executor.execute_drink(
            agent=agent,
            target_entity=target_entity,
            all_entities=entities,
            incident_id=f"drink-t{current_tick}-{agent.id}",
        )
        if drank:
            self._cleanup_satisfied_goal(agent, need_name="thirst")
            return True
        return False

    def handle_subgoal_rest(
        self, agent: Agent, current_goal: Goal, entities: list[WorldEntity]
    ) -> bool:
        """Führt Erholungsaktion deterministisch aus."""
        target_entity = self._find_target_entity_for_goal(current_goal, entities)
        current_tick = self._tick_provider()
        rested = self._action_executor.execute_rest(
            agent=agent,
            target_entity=target_entity,
            incident_id=f"rest-t{current_tick}-{agent.id}",
        )
        if rested:
            self._cleanup_satisfied_goal(agent, need_name="energy")
            return True
        return False

    @staticmethod
    def _find_target_entity_for_goal(
        goal: Goal, entities: list[WorldEntity]
    ) -> Optional[WorldEntity]:
        if goal.target_entity_id:
            target = next((e for e in entities if e.id == goal.target_entity_id), None)
            if target:
                return target
        if goal.target_position:
            return next((e for e in entities if e.position == goal.target_position), None)
        return None

    def _cleanup_satisfied_goal(self, agent: Agent, need_name: str) -> None:
        """Entfernt abgeschlossenes Sub-Goal und schließt das Hauptziel bei vollständiger Befriedigung ab."""
        self._goal_service.pop_goal(agent)
        if not self._need_service.is_need_urgent(agent, need_name):
            active_g = agent.active_goal
            if active_g and not any(g.name.startswith("SubGoal:") for g in agent.goals):
                self._goal_service.pop_goal(agent)

    def interrupt_goal_for_replan(
        self,
        agent: Agent,
        reason: str,
        discovered_fact: Optional[EntityFact] = None,
    ) -> None:
        """Bricht aktive Pfade und Ziele kontrolliert ab, um eine Neu-Dekompensation anzustoßen."""
        agent.clear_path()
        agent.goals.clear()

        payload: dict[str, Any] = {"reason": reason}
        if discovered_fact:
            payload["discovered_entity_id"] = discovered_fact.entity_id
            payload["entity_type"] = discovered_fact.entity_type
            if discovered_fact.last_known_position:
                payload["position"] = [
                    discovered_fact.last_known_position.x,
                    discovered_fact.last_known_position.y,
                ]

        current_tick = self._tick_provider()
        self._logger.log(
            SimulationEvent(
                tick=current_tick,
                agent_id=agent.id,
                event_type="goal_interrupted_for_replan",
                summary=f"Agent {agent.name}: Ziel abgebrochen wegen '{reason}'. Re-Planung initiiert.",
                payload=payload,
            )
        )