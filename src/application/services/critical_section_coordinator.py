from __future__ import annotations

from typing import Callable, Optional
from src.domain.models.agent import Agent
from src.domain.models.critical_section import CriticalSection, CriticalSectionRequest
from src.domain.models.events import SimulationEvent
from src.domain.models.goal import ExecutionPriority
from src.domain.ports.event_logger import IEventLogger


class CriticalSectionCoordinator:
    """Verwaltet kritische Bereiche deterministisch mit Prioritätswarteschlangen und Notwendigkeitsabsicherung."""

    def __init__(
        self,
        logger: IEventLogger,
        tick_provider: Optional[Callable[[], int]] = None,
    ) -> None:
        self._logger = logger
        self._tick_provider = tick_provider or (lambda: 0)
        self._sections: dict[str, CriticalSection] = {}

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
        """Versucht, den kritischen Bereich zu belegen.

        Gibt True zurück, wenn der Lock gewährt wurde, andernfalls False (eingereiht).
        """
        current_tick = self._tick_provider()
        section = self.get_or_create_section(resource_key)

        if section.holder_agent_id == agent.id:
            return True

        if section.holder_agent_id is None:
            section.holder_agent_id = agent.id
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
            self._sort_wait_queue(section)

            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=agent.id,
                    event_type="critical_section_queued",
                    summary=f"Agent {agent.name} wartet auf '{resource_key}' (Rang: {section.wait_queue.index(req) + 1}).",
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
        """Gibt den kritischen Bereich frei und liefert die Agent-ID des nächsten Berechtigten."""
        current_tick = self._tick_provider()
        section = self._sections.get(resource_key)
        if not section or section.holder_agent_id != agent_id:
            return None

        if mark_completed:
            section.is_completed = True
            section.completed_by = agent_id
            section.completed_tick = current_tick

        section.holder_agent_id = None
        next_agent_id: Optional[str] = None

        if section.wait_queue:
            next_req = section.wait_queue.pop(0)
            section.holder_agent_id = next_req.agent_id
            next_agent_id = next_req.agent_id

        self._logger.log(
            SimulationEvent(
                tick=current_tick,
                agent_id=agent_id,
                event_type="critical_section_released",
                summary=f"Kritischer Bereich '{resource_key}' durch Agent {agent_id} freigegeben. Nächster: {next_agent_id}.",
                payload={
                    "resource_key": resource_key,
                    "is_completed": section.is_completed,
                    "next_agent_id": next_agent_id,
                },
            )
        )
        return next_agent_id

    def is_completed(self, resource_key: str) -> bool:
        section = self._sections.get(resource_key)
        return bool(section and section.is_completed)

    def is_holder(self, resource_key: str, agent_id: str) -> bool:
        section = self._sections.get(resource_key)
        return bool(section and section.holder_agent_id == agent_id)

    def get_holder(self, resource_key: str) -> Optional[str]:
        section = self._sections.get(resource_key)
        return section.holder_agent_id if section else None

    def _sort_wait_queue(self, section: CriticalSection) -> None:
        """Sortiert die Warteschlange deterministisch: URGENT hat immer Vorrang vor COOPERATIVE und ROUTINE."""
        def sort_key(req: CriticalSectionRequest) -> tuple[int, int, int]:
            prio_val = req.priority.value
            order_val = req.proposed_order if req.proposed_order is not None else 9999
            return (prio_val, order_val, req.tick)

        section.wait_queue.sort(key=sort_key)