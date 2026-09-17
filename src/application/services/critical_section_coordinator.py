from __future__ import annotations

from typing import Callable, Optional
from src.domain.models.agent import Agent
from src.domain.models.critical_section import CriticalSection, CriticalSectionRequest
from src.domain.models.events import SimulationEvent
from src.domain.models.goal import ExecutionPriority
from src.domain.ports.event_logger import IEventLogger


class CriticalSectionCoordinator:
    """Verwaltet kritische Bereiche deterministisch mit Aging-Boost und Verwaist-Freigaben."""

    def __init__(
        self,
        logger: IEventLogger,
        tick_provider: Optional[Callable[[], int]] = None,
        aging_interval_ticks: int = 3,
        max_lock_duration_ticks: int = 6,
    ) -> None:
        self._logger = logger
        self._tick_provider = tick_provider or (lambda: 0)
        self._aging_interval_ticks = aging_interval_ticks
        self._max_lock_duration_ticks = max_lock_duration_ticks
        self._sections: dict[str, CriticalSection] = {}
        self._lock_acquired_ticks: dict[str, int] = {}

    def get_or_create_section(self, resource_key: str) -> CriticalSection:
        if resource_key not in self._sections:
            self._sections[resource_key] = CriticalSection(resource_key=resource_key)
        return self._sections[resource_key]

    def acquire_or_queue(
        self,
        agent: Agent,
        resource_key: str,
        priority: ExecutionPriority,
        proposed_order: Optional[int] = None,
    ) -> bool:
        current_tick = self._tick_provider()
        self._check_and_recover_timed_out_locks(current_tick)
        section = self.get_or_create_section(resource_key)

        if section.holder_agent_id == agent.id:
            return True

        if section.holder_agent_id is None:
            section.holder_agent_id = agent.id
            self._lock_acquired_ticks[resource_key] = current_tick
            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=agent.id,
                    event_type="critical_section_acquired",
                    summary=f"Agent {agent.name} hat Zugriff auf '{resource_key}' erhalten.",
                    payload={"resource_key": resource_key, "priority": priority.name},
                )
            )
            return True

        if not any(req.agent_id == agent.id for req in section.wait_queue):
            req = CriticalSectionRequest(
                agent_id=agent.id,
                priority=priority,
                tick=current_tick,
                proposed_order=proposed_order,
            )
            section.wait_queue.append(req)
            self._sort_wait_queue(section, current_tick)

            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=agent.id,
                    event_type="critical_section_queued",
                    summary=f"Agent {agent.name} wartet auf '{resource_key}'.",
                    payload={
                        "resource_key": resource_key,
                        "priority": priority.name,
                        "queue_length": len(section.wait_queue),
                    },
                )
            )

        return False

    def release(
        self,
        agent_id: str,
        resource_key: str,
        mark_completed: bool = False,
    ) -> Optional[str]:
        current_tick = self._tick_provider()
        section = self._sections.get(resource_key)
        if not section or section.holder_agent_id != agent_id:
            return None

        if mark_completed:
            section.is_completed = True
            section.completed_by = agent_id
            section.completed_tick = current_tick

        section.holder_agent_id = None
        self._lock_acquired_ticks.pop(resource_key, None)
        next_agent_id: Optional[str] = None

        if section.wait_queue:
            self._sort_wait_queue(section, current_tick)
            next_req = section.wait_queue.pop(0)
            section.holder_agent_id = next_req.agent_id
            self._lock_acquired_ticks[resource_key] = current_tick
            next_agent_id = next_req.agent_id

        self._logger.log(
            SimulationEvent(
                tick=current_tick,
                agent_id=agent_id,
                event_type="critical_section_released",
                summary=f"Kritischer Bereich '{resource_key}' freigegeben. Nächster: {next_agent_id}.",
                payload={
                    "resource_key": resource_key,
                    "is_completed": section.is_completed,
                    "next_agent_id": next_agent_id,
                },
            )
        )
        return next_agent_id

    def release_all_for_agent(self, agent_id: str) -> None:
        """Entfernt einen Agenten bei Abbruch ('abort') aus allen Locks und Warteschlangen."""
        current_tick = self._tick_provider()
        for resource_key, section in list(self._sections.items()):
            section.wait_queue = [req for req in section.wait_queue if req.agent_id != agent_id]
            if section.holder_agent_id == agent_id:
                self.release(agent_id, resource_key, mark_completed=False)
                self._logger.log(
                    SimulationEvent(
                        tick=current_tick,
                        agent_id=agent_id,
                        event_type="critical_section_abandoned",
                        summary=f"Verwaister Lock auf '{resource_key}' durch Abbruch von Agent {agent_id} freigegeben.",
                        payload={"resource_key": resource_key},
                    )
                )

    def _check_and_recover_timed_out_locks(self, current_tick: int) -> None:
        """Circuit-Breaker: Erzwingt Freigabe, wenn ein Lock die Maximaldauer überschreitet."""
        for resource_key, acquired_tick in list(self._lock_acquired_ticks.items()):
            if current_tick - acquired_tick > self._max_lock_duration_ticks:
                section = self._sections.get(resource_key)
                if section and section.holder_agent_id:
                    timed_out_holder = section.holder_agent_id
                    self._logger.log(
                        SimulationEvent(
                            tick=current_tick,
                            agent_id=timed_out_holder,
                            event_type="critical_section_timeout",
                            summary=f"Lock auf '{resource_key}' wegen Zeitüberschreitung entzogen.",
                            payload={"holder_id": timed_out_holder, "held_ticks": current_tick - acquired_tick},
                        )
                    )
                    self.release(timed_out_holder, resource_key, mark_completed=False)

    def _sort_wait_queue(self, section: CriticalSection, current_tick: int) -> None:
        """Sortiert mit dynamic priority boost (Aging): Wartende steigen im Rang auf."""
        def effective_sort_key(req: CriticalSectionRequest) -> tuple[int, int, int]:
            waiting_ticks = max(0, current_tick - req.tick)
            prio_boost = waiting_ticks // self._aging_interval_ticks
            boosted_priority_val = max(1, req.priority.value - prio_boost)
            order_val = req.proposed_order if req.proposed_order is not None else 9999
            return (boosted_priority_val, order_val, req.tick)

        section.wait_queue.sort(key=effective_sort_key)

    def is_completed(self, resource_key: str) -> bool:
        section = self._sections.get(resource_key)
        return bool(section and section.is_completed)

    def is_holder(self, resource_key: str, agent_id: str) -> bool:
        section = self._sections.get(resource_key)
        return bool(section and section.holder_agent_id == agent_id)