from __future__ import annotations

from typing import Any, Optional

from src.application.services.lifecycle.daily_event_buffer import DailyEventBuffer
from src.domain.models.agent.agent import Agent
from src.domain.models.planning.events import SimulationEvent
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.vector_memory_store import IVectorMemoryStore


class MemoryConsolidationService:
    """
    Synthetisiert Tageserlebnisse von Agenten während der Schlafphase
    und persistiert sie als semantisch indizierte Erinnerungen im IVectorMemoryStore.
    """

    def __init__(
        self,
        vector_store: IVectorMemoryStore,
        event_buffer: DailyEventBuffer,
        logger: Optional[IEventLogger] = None,
    ) -> None:
        self._vector_store = vector_store
        self._event_buffer = event_buffer
        self._logger = logger

    def consolidate_agent(self, agent: Agent, day_number: int, tick: int) -> list[str]:
        """Konsolidiert die Tageserlebnisse eines einzelnen Agenten."""
        raw_events = self._event_buffer.get_events_for_agent(agent.id)
        if not raw_events:
            return []

        memories: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for event in raw_events:
            synthesized = self._synthesize_event(event, day_number)
            if synthesized:
                text, meta = synthesized
                memories.append(text)
                metadatas.append(meta)

        if memories:
            self._vector_store.add_memories(
                agent_id=agent.id,
                memories=memories,
                metadatas=metadatas,
            )

            if self._logger:
                self._logger.log(
                    SimulationEvent(
                        tick=tick,
                        agent_id=agent.id,
                        event_type="memory_consolidated",
                        summary=f"Agent {agent.name}: {len(memories)} Erinnerungen für Tag {day_number} konsolidiert.",
                        payload={
                            "day": day_number,
                            "count": len(memories),
                            "memories": memories,
                        },
                    )
                )

        self._event_buffer.clear_agent(agent.id)
        return memories

    def consolidate_all(
        self, agents: list[Agent], day_number: int, tick: int
    ) -> dict[str, list[str]]:
        """Konsolidiert die Erinnerungen aller schlafenden Agenten."""
        results: dict[str, list[str]] = {}
        for agent in agents:
            results[agent.id] = self.consolidate_agent(agent, day_number, tick)
        return results

    def _synthesize_event(
        self, event: SimulationEvent, day_number: int
    ) -> Optional[tuple[str, dict[str, Any]]]:
        """Extrahiert deterministisch Text und Metadaten aus einem Ereignis."""
        payload = event.payload or {}
        event_type = event.event_type

        # 1. Ressourcenkonsum (unterstützt beide Event-Namenskonventionen)
        if event_type in (
                "resource_consumed",
                "drink_executed",
                "rest_executed",
                "entity_consumed",
                "entity_drank",
                "agent_rested",
        ):
            resource_type = (
                    payload.get("resource_type")
                    or payload.get("target_entity_name")
                    or payload.get("target_type")
                    or "Ressource"
            )
            pos = payload.get("position")
            pos_str = f" bei ({pos[0]}, {pos[1]})" if isinstance(pos, (list, tuple)) and len(pos) == 2 else ""
            text = f"Tag {day_number}: {resource_type}{pos_str} erfolgreich genutzt."
            meta = {
                "category": "resource",
                "tick": event.tick,
                "sentiment": 1.0,
            }
            return text, meta

        # 2. Ressourcenfunde & Exploration
        if event_type == "goal_interrupted_for_replan" and "discovered_entity_id" in payload:
            entity_type = payload.get("entity_type", "Ressource")
            pos = payload.get("position")
            pos_str = f" bei ({pos[0]}, {pos[1]})" if isinstance(pos, (list, tuple)) and len(pos) == 2 else ""
            text = f"Tag {day_number}: Neue Quelle entdeckt ({entity_type}){pos_str}."
            meta = {
                "category": "resource",
                "tick": event.tick,
                "sentiment": 0.8,
            }
            return text, meta

        # 3. Soziale Interaktionen & Dialoge
        if event_type in ("dialogue_resolved", "interaction_completed"):
            partner_id = payload.get("partner_id") or payload.get("interlocutor_id")
            outcome = payload.get("outcome", "neutral")
            sentiment = 1.0 if outcome == "cooperative" else (-0.8 if outcome == "conflict" else 0.0)
            text = f"Tag {day_number}: Begegnung mit {partner_id or 'Mitbewohner'} verlief {outcome}."
            meta = {
                "category": "social",
                "tick": event.tick,
                "interlocutor_id": str(partner_id) if partner_id else None,
                "sentiment": sentiment,
            }
            return text, meta

        # 4. Kooperative Ausweichmanöver
        if event_type == "evasion_hold_started":
            yield_for = payload.get("yield_for_agent_id", "Gegenverkehr")
            text = f"Tag {day_number}: In Nische ausgewichen und Vorfahrt für {yield_for} gewährt."
            meta = {
                "category": "social",
                "tick": event.tick,
                "interlocutor_id": str(yield_for) if yield_for else None,
                "sentiment": 0.5,
            }
            return text, meta

        # 5. Kognitions-Snapshots (Gedanken, Strategien, Blockaden)
        if event_type == "cognitive_snapshot":
            obstacle = payload.get("perceived_obstacle")
            strategy = payload.get("intended_strategy", "Reflexion")
            thought = payload.get("formatted_thought") or event.summary

            if obstacle:
                text = f"Tag {day_number}: Fortschritt behindert durch {obstacle}. Strategie: {strategy}."
                meta = {
                    "category": "conflict",
                    "tick": event.tick,
                    "sentiment": -0.5,
                }
                return text, meta

            primary_goal = payload.get("primary_goal")
            active_subgoal = payload.get("active_subgoal")
            target_pos = payload.get("target_position")
            target_str = f" mit Ziel {target_pos}" if target_pos else ""

            if active_subgoal:
                text = f"Tag {day_number}: Teilziel '{active_subgoal}' verfolgt{target_str}. Gedanke: {thought}"
            elif primary_goal:
                text = f"Tag {day_number}: Ziel '{primary_goal}' verfolgt{target_str}. Gedanke: {thought}"
            else:
                text = f"Tag {day_number}: Zustand '{strategy}'. Gedanke: {thought}"

            meta = {
                "category": "cognition",
                "tick": event.tick,
                "sentiment": 0.1,
            }
            return text, meta

        # 6. Allgemeine Handlungs- und Statusereignisse
        if event_type in ("path_assigned", "wait_for_peer_started"):
            text = f"Tag {day_number}: {event.summary}"
            meta = {
                "category": "general",
                "tick": event.tick,
                "sentiment": 0.0,
            }
            return text, meta

        return None