from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
from src.application.services.evasion_finder import EvasionFinder
from src.domain.models.agent import Agent
from src.domain.models.events import SimulationEvent
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.services.perception_service import PerceptionService


@dataclass(frozen=True, slots=True)
class TargetSearchResult:
    phase: Literal["visual", "projected", "frontier", "unreachable"]
    target_position: Optional[Position]
    path: list[Position]


class TargetSearchService:
    def __init__(
        self,
        pathfinder: IPathfinder,
        perception_service: PerceptionService,
        evasion_finder: EvasionFinder,
        logger: IEventLogger,
    ) -> None:
        self._pathfinder: IPathfinder = pathfinder
        self._perception_service: PerceptionService = perception_service
        self._evasion_finder: EvasionFinder = evasion_finder
        self._logger: IEventLogger = logger

    def search_target(
        self,
        agent: Agent,
        target_entity_id: str,
        grid: WorldGrid,
        entities: list[WorldEntity],
        current_tick: int,
    ) -> TargetSearchResult:
        """Führt die dreistufige epistemische Suchkaskade mit Zielabweichungsprüfung aus."""
        # Stufe 1: Direkte Wahrnehmung
        visible_entities = self._perception_service.get_visible_entities(agent, entities)
        target_entity = next((e for e in visible_entities if e.id == target_entity_id), None)

        if target_entity:
            # Abweichungsprüfung gegen bisherigen Pfad
            if agent.has_path:
                current_target_endpoint = agent.path[-1]
                deviation = current_target_endpoint.manhattan_distance(target_entity.position)
                if deviation > 2:
                    self._logger.log(
                        SimulationEvent(
                            tick=current_tick,
                            agent_id=agent.id,
                            event_type="target_trajectory_invalidated",
                            summary=f"Agent {agent.name}: Ziel {target_entity.name} weicht um {deviation} Kacheln vom Pfadziel ab. Pfad wird verworfen.",
                            payload={
                                "target_id": target_entity_id,
                                "old_endpoint": [current_target_endpoint.x, current_target_endpoint.y],
                                "actual_pos": [target_entity.position.x, target_entity.position.y],
                                "deviation": deviation,
                            },
                        )
                    )
                    agent.clear_path()

            path = self._pathfinder.find_path(agent.position, target_entity.position, agent.mental_map)
            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=agent.id,
                    event_type="search_phase_transition",
                    summary=f"Agent {agent.name}: Ziel {target_entity.name} direkt wahrgenommen.",
                    payload={
                        "phase": "visual",
                        "target_id": target_entity_id,
                        "pos": [target_entity.position.x, target_entity.position.y],
                    },
                )
            )
            return TargetSearchResult(phase="visual", target_position=target_entity.position, path=path)

        # Stufe 2: Extrapolierte Schätzung aus Gedächtnis
        visible_positions = set(self._perception_service.get_visible_positions(agent.position, grid))
        projected_fact = agent.memory.get_projected_fact(
            entity_id=target_entity_id,
            current_tick=current_tick,
            mental_map=agent.mental_map,
            visible_positions=visible_positions,
        )
        if projected_fact and projected_fact.projected_position and projected_fact.confidence > 0.0:
            proj_pos = projected_fact.projected_position
            path = self._pathfinder.find_path(agent.position, proj_pos, agent.mental_map)
            if path:
                self._logger.log(
                    SimulationEvent(
                        tick=current_tick,
                        agent_id=agent.id,
                        event_type="search_phase_transition",
                        summary=f"Agent {agent.name}: Ziel auf Schätzposition ({proj_pos.x}, {proj_pos.y}) extrapoliert (Konfidenz: {projected_fact.confidence:.2f}).",
                        payload={
                            "phase": "projected",
                            "target_id": target_entity_id,
                            "pos": [proj_pos.x, proj_pos.y],
                            "confidence": projected_fact.confidence,
                        },
                    )
                )
                return TargetSearchResult(phase="projected", target_position=proj_pos, path=path)

        # Stufe 3: Sektorbasierte Frontier-Exploration im Bewegungsvektor
        fact = agent.memory.known_entities.get(target_entity_id)
        search_dir: Optional[tuple[float, float]] = None
        if fact:
            if fact.last_observed_velocity != (0.0, 0.0):
                search_dir = fact.last_observed_velocity
            elif fact.smoothed_velocity != (0.0, 0.0):
                search_dir = fact.smoothed_velocity

        occupied = {e.position for e in entities if e.id != agent.id}
        frontier_tile, came_from = self._evasion_finder._find_nearest_frontier(
            start=agent.position,
            grid=agent.mental_map,
            frontier_forbidden=occupied,
            direction_vector=search_dir,
        )
        if frontier_tile:
            path = self._evasion_finder._reconstruct_path(agent.position, frontier_tile, came_from)
            self._logger.log(
                SimulationEvent(
                    tick=current_tick,
                    agent_id=agent.id,
                    event_type="search_phase_transition",
                    summary=f"Agent {agent.name}: Ziel ungesehen. Erkundet Sektor in Richtung Grenzkachel ({frontier_tile.x}, {frontier_tile.y}).",
                    payload={
                        "phase": "frontier",
                        "target_id": target_entity_id,
                        "frontier_pos": [frontier_tile.x, frontier_tile.y],
                        "direction": list(search_dir) if search_dir else None,
                    },
                )
            )
            return TargetSearchResult(phase="frontier", target_position=frontier_tile, path=path)

        return TargetSearchResult(phase="unreachable", target_position=None, path=[])