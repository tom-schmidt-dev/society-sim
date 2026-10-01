from __future__ import annotations

from typing import Any, Callable, Optional

from src.application.services.action_executor import ActionExecutor
from src.application.services.cognitive_snapshot_service import CognitiveSnapshotService
from src.application.services.frontier_explorer import FrontierExplorer
from src.application.services.goal_service import GoalService
from src.application.services.need_service import NeedService
from src.application.services.plan_decomposition_service import PlanDecompositionService
from src.domain.models.agent import Agent
from src.domain.models.agent_cognition import AgentCognitiveSnapshot
from src.domain.models.agent_memory import EntityFact
from src.domain.models.events import SimulationEvent
from src.domain.models.goal import Goal
from src.domain.models.position import Position
from src.domain.models.world_entity import WorldEntity
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

    @property
    def latest_snapshots(self) -> list[AgentCognitiveSnapshot]:
        """Gibt die zuletzt erfassten Kognitions-Snapshots aller Agenten zurück."""
        return list(self._latest_snapshots.values())

    async def process_agent_needs_and_cognition(
        self, agents: list[Agent], entities: list[WorldEntity]
    ) -> None:
        """Überwacht Bedürfnisse, triggert Plandekomposition und führt Sub-Goals aus."""
        for agent in agents:
            satisfied_in_tick = False

            # 1. Plandekomposition bei akutem dominanten Bedürfnis & leerem Zielstack
            if not agent.is_busy and not agent.goals:
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

                if dominant_need:
                    await self.trigger_plan_decomposition(agent, dominant_need)

            # 2. Ausführung & Evaluation aktiver Sub-Goals
            current_goal = agent.active_goal
            if current_goal and not agent.is_busy:
                if current_goal.name == "SubGoal: move_to":
                    self.handle_subgoal_move_to(agent, current_goal)
                elif current_goal.name.startswith("SubGoal: explore"):
                    self.handle_subgoal_explore(agent, current_goal)
                elif current_goal.name == "SubGoal: consume":
                    satisfied_in_tick = self.handle_subgoal_consume(agent, current_goal, entities)
                elif current_goal.name == "SubGoal: drink":
                    satisfied_in_tick = self.handle_subgoal_drink(agent, current_goal, entities)
                elif current_goal.name == "SubGoal: rest":
                    satisfied_in_tick = self.handle_subgoal_rest(agent, current_goal, entities)

            # 3. Zyklischer Vitalwertzuwachs pro Takt
            if not satisfied_in_tick:
                self._need_service.update_needs(agent)

            # 4. Transparenter Kognitions-Snapshot & selektives Event-Logging
            self._capture_and_log_snapshot(agent, entities)

    def _capture_and_log_snapshot(self, agent: Agent, entities: list[WorldEntity]) -> None:
        current_tick = self._tick_provider()
        snapshot = self._snapshot_service.create_snapshot(agent, current_tick, entities)
        prev = self._latest_snapshots.get(agent.id)

        # Logge bei Zustands- und Strategiewechseln oder initialem Erfassen
        if (
            prev is None
            or prev.intended_strategy != snapshot.intended_strategy
            or prev.primary_goal != snapshot.primary_goal
            or prev.active_subgoal != snapshot.active_subgoal
            or prev.perceived_obstacle != snapshot.perceived_obstacle
        ):
            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=agent.id,
                    event_type="cognitive_snapshot",
                    summary=f"Gedanke von {agent.name}: {snapshot.formatted_thought}",
                    payload=snapshot.to_dict(),
                )
            )

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

        for intent in reversed(plan.sub_goals):
            target_pos = (
                Position(intent.target_position[0], intent.target_position[1])
                if intent.target_position
                else None
            )
            sub_goal = Goal(
                name=f"SubGoal: {intent.action_type.value}",
                target_position=target_pos,
                target_entity_id=intent.target_entity_id,
                description=intent.description,
            )
            self._goal_service.push_goal(agent, sub_goal)

    def handle_subgoal_move_to(self, agent: Agent, current_goal: Goal) -> None:
        """Verwaltet Pfadzuweisung für direkte Navigations-Teilziele."""
        if current_goal.target_position and not agent.has_path:
            if agent.position != current_goal.target_position:
                path = self._pathfinder.find_path(
                    agent.position,
                    current_goal.target_position,
                    agent.mental_map,
                )
                if path:
                    agent.assign_path(path)

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
                self._goal_service.pop_goal(agent)

    def handle_subgoal_consume(
        self, agent: Agent, current_goal: Goal, entities: list[WorldEntity]
    ) -> bool:
        """Führt Konsumaktion deterministisch aus."""
        target_entity = self._find_target_entity_for_goal(current_goal, entities)
        if not target_entity:
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