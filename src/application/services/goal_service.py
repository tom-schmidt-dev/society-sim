from __future__ import annotations

import uuid
from typing import Callable, Optional
from src.domain.models.agent import Agent
from src.domain.models.events import SimulationEvent
from src.domain.models.goal import Goal
from src.domain.models.world import WorldGrid
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder


class GoalService:
    def __init__(
            self,
            logger: IEventLogger,
            cognition_provider: ICognitionProvider,
            pathfinder: IPathfinder,
            tick_provider: Optional[Callable[[], int]] = None,
    ) -> None:
        self._logger: IEventLogger = logger
        self._cognition_provider: ICognitionProvider = cognition_provider
        self._pathfinder: IPathfinder = pathfinder
        self._tick_provider: Callable[[], int] = tick_provider or (lambda: 0)

    def push_goal(
            self,
            agent: Agent,
            goal: Goal,
            incident_id: Optional[str] = None,
    ) -> None:
        """Fügt dem Goal-Stack ein Ziel hinzu und erfasst das Ereignis mit dem aktuellen Tick."""
        agent.push_goal(goal)
        self._logger.log(
            SimulationEvent(
                tick=self._tick_provider(),
                agent_id=agent.id,
                event_type="goal_pushed",
                summary=f"Agent {agent.name}: Neues Ziel '{goal.name}' gesetzt (Stack-Tiefe: {len(agent.goals)}).",
                payload={
                    "incident_id": incident_id,
                    "goal": goal.to_dict(),
                    "stack_depth": len(agent.goals),
                },
            )
        )

    def pop_goal(
            self,
            agent: Agent,
            target_goal: Optional[Goal] = None,
            incident_id: Optional[str] = None,
    ) -> Optional[Goal]:
        """Entfernt das spezifizierte (oder oberste) Ziel und erfasst das Ereignis mit dem aktuellen Tick."""
        completed = agent.pop_goal(target_goal=target_goal)
        if completed:
            self._logger.log(
                SimulationEvent(
                    tick=self._tick_provider(),
                    agent_id=agent.id,
                    event_type="goal_popped",
                    summary=f"Agent {agent.name}: Ziel '{completed.name}' abgeschlossen. Aktives Ziel: '{agent.destination_name}' (Stack-Tiefe: {len(agent.goals)}).",
                    payload={
                        "incident_id": incident_id,
                        "completed_goal": completed.to_dict(),
                        "active_goal": agent.active_goal.to_dict() if agent.active_goal else None,
                        "stack_depth": len(agent.goals),
                    },
                )
            )
        return completed

    def process_timed_goal(self, agent: Agent) -> bool:
        """Dekrementiert befristete Halteziele. Gibt True zurück, wenn der Agent in diesem Tick pausiert."""
        if agent.active_goal and agent.active_goal.remaining_ticks is not None:
            if agent.active_goal.tick():
                self.pop_goal(agent)
            return True
        return False

    async def evaluate_sub_goal_completion(
            self,
            agent: Agent,
            grid: WorldGrid,
            recent_dialogues: list[str],
    ) -> None:
        """Evaluiert kognitiv den Abschluss eines Teilziels und loggt mit dem tatsächlichen Abschluss-Tick."""
        start_tick = self._tick_provider()
        eval_incident_id = f"eval-t{start_tick}-{agent.id}-{uuid.uuid4().hex[:6]}"
        target_sub_goal = agent.active_goal
        if not target_sub_goal:
            agent.is_thinking = False
            return
        context = {
            "agent_id": agent.id,
            "name": agent.name,
            "current_x": agent.position.x,
            "current_y": agent.position.y,
            "active_goal": agent.active_goal.to_dict() if agent.active_goal else None,
            "goal_stack": [g.to_dict() for g in agent.goals],
            "recent_dialogues": recent_dialogues,
        }
        try:
            eval_res = await self._cognition_provider.evaluate_goal_status(context)
            # Loggt den Tick NACH Abschluss des LLM-Aufrufs
            current_eval_tick = self._tick_provider()
            self._logger.log(
                SimulationEvent(
                    tick=current_eval_tick,
                    agent_id=agent.id,
                    event_type="goal_evaluated",
                    summary=f"Agent {agent.name} evaluiert Zielstatus: is_completed={eval_res.is_completed}.",
                    payload={
                        "incident_id": eval_incident_id,
                        "is_completed": eval_res.is_completed,
                        "reason": eval_res.reason,
                        "thought": eval_res.thought,
                    },
                )
            )
            if eval_res.is_completed:
                self.pop_goal(agent, target_goal=target_sub_goal, incident_id=eval_incident_id)
                active_goal = agent.active_goal
                if active_goal and active_goal.target_position:
                    # P1: Pfadberechnung auf Basis der mentalen Karte des Agenten
                    new_path = self._pathfinder.find_path(
                        agent.position, active_goal.target_position, agent.mental_map
                    )
                    if new_path:
                        agent.path = new_path
        except Exception as err:
            self._logger.log(
                SimulationEvent(
                    tick=self._tick_provider(),
                    agent_id=agent.id,
                    event_type="goal_eval_failed",
                    summary=f"Fehler bei Zielevaluation von {agent.name}: {err}",
                    payload={"incident_id": eval_incident_id, "error": str(err)},
                )
            )
        finally:
            agent.is_thinking = False