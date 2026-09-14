from __future__ import annotations

import asyncio
from typing import Any, Optional
from src.domain.models.agent import Agent
from src.domain.models.events import SimulationEvent
from src.domain.models.goal import Goal
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter
from src.domain.services.evasion_finder import EvasionFinder
from src.domain.services.perception_service import PerceptionService
from src.application.services.conflict_coordinator import ConflictCoordinator
from src.application.services.goal_service import GoalService
from abc import ABC, abstractmethod

class SimulationEngine:
    def __init__(
            self,
            grid: WorldGrid,
            pathfinder: IPathfinder,
            presenter: IPresenter,
            logger: IEventLogger,
            cognition_provider: ICognitionProvider,
            tick_interval: float = 0.3,
            goal_service: Optional[GoalService] = None,
            conflict_coordinator: Optional[ConflictCoordinator] = None,
            perception_service: Optional[PerceptionService] = None,
    ) -> None:
        self._entities: list[WorldEntity] = []
        self._grid: WorldGrid = grid
        self._pathfinder: IPathfinder = pathfinder
        self._presenter: IPresenter = presenter
        self._logger: IEventLogger = logger
        self._tick_interval: float = tick_interval

        self._perception_service = perception_service or PerceptionService(default_radius=3)
        self._goal_service = goal_service or GoalService(
            logger, cognition_provider, pathfinder, tick_provider=lambda: self._current_tick
        )
        self._evasion_finder = EvasionFinder(pathfinder)
        self._conflict_coordinator = conflict_coordinator or ConflictCoordinator(
            self._grid,
            logger,
            cognition_provider,
            pathfinder,
            self._goal_service,
            self._evasion_finder,
            tick_provider=lambda: self._current_tick,
        )

        self._agents: list[Agent] = []
        self._current_tick: int = 0
        self._is_running: bool = False
        self._background_tasks: set[asyncio.Task[Any]] = set()

    def register_entity(self, entity: WorldEntity) -> None:
        if not self._grid.is_walkable(entity.position):
            raise ValueError(f"Position {entity.position} für Objekt {entity.name} ist blockiert.")
        self._entities.append(entity)
        self._logger.log(
            SimulationEvent(
                tick=self._current_tick,
                agent_id=entity.id,
                event_type="entity_registered",
                summary=f"Objekt '{entity.name}' an Position ({entity.position.x}, {entity.position.y}) platziert.",
                payload={"x": entity.position.x, "y": entity.position.y, "is_conversational": entity.is_conversational},
            )
        )

    def register_agent(self, agent: Agent) -> None:
        if not self._grid.is_walkable(agent.position):
            raise ValueError(f"Startposition {agent.position} für Agent {agent.name} blockiert.")
        if agent not in self._entities:
            self._entities.append(agent)
        self._agents.append(agent)
        # P1: Grenzen der mentalen Karte auf Grid-Größe synchronisieren
        agent.mental_map.set_bounds(self._grid.width, self._grid.height)
        self._logger.log(
            SimulationEvent(
                tick=self._current_tick,
                agent_id=agent.id,
                event_type="agent_registered",
                summary=f"Agent {agent.name} ist an Position ({agent.position.x}, {agent.position.y}) in die Welt eingetreten.",
                payload={"x": agent.position.x, "y": agent.position.y, "energy": agent.energy},
            )
        )

    def set_agent_target(
        self,
        agent_id: str,
        target: Position,
        destination_name: str = "Ziel",
    ) -> None:
        agent = next((a for a in self._agents if a.id == agent_id), None)
        if not agent:
            raise ValueError(f"Agent mit ID '{agent_id}' nicht gefunden.")

        agent.goals.clear()
        self._goal_service.push_goal(
            agent,
            Goal(name=destination_name, target_position=target)
        )

        self._update_agent_perception(agent)
        path = self._pathfinder.find_path(agent.position, target, agent.mental_map)
        if path:
            agent.path = path
            self._logger.log(
                SimulationEvent(
                    tick=self._current_tick,
                    agent_id=agent.id,
                    event_type="path_planned",
                    summary=f"Agent {agent.name} hat einen Pfad nach {destination_name} mit {len(path)} Schritten berechnet.",
                    payload={
                        "target": {"x": target.x, "y": target.y},
                        "step_count": len(path),
                    },
                )
            )

    def _update_agent_perception(self, agent: Agent) -> list[WorldEntity]:
        visible_positions = self._perception_service.get_visible_positions(agent.position, self._grid)
        for pos in visible_positions:
            agent.mental_map.update_tile(pos, self._grid.is_walkable(pos), self._current_tick)

        visible_entities = self._perception_service.get_visible_entities(agent, self._entities)
        for visible_entity in visible_entities:
            agent.memory.update_entity_perception(
                visible_entity.id, visible_entity.name, visible_entity.position
            )
            # Keine Vorab-Eintragung ungesehener Eigenschaften:
            # Nur Kacheln sperren, deren Typ über das Gedächtnis bereits gesichert als unpassierbar bekannt ist
            if agent.memory.get_assumed_walkable(visible_entity.id) is False:
                agent.mental_map.mark_obstacle(visible_entity.position, self._current_tick)

        return visible_entities

    def _collect_known_positions(self) -> set[Position]:
        discovered: set[Position] = set()
        for agent in self._agents:
            discovered.update(agent.mental_map.tiles.keys())
        return discovered

    def process_tick(self) -> None:
        self._current_tick += 1

        for agent in self._agents:
            if agent.energy > 0:
                agent.energy -= 1

            if self._goal_service.process_timed_goal(agent):
                continue

            if agent.is_busy:
                continue

            visible_entities = self._update_agent_perception(agent)

            # P1: Pfadreparatur, falls neu aufgedeckte Hindernisse die geplante Route schneiden
            if agent.has_path and not all(agent.mental_map.is_walkable(p) for p in agent.path):
                active_goal = agent.active_goal
                target_pos = (
                    active_goal.target_position
                    if active_goal and active_goal.target_position
                    else None
                )
                if target_pos:
                    new_path = self._pathfinder.find_path(agent.position, target_pos, agent.mental_map)
                    if new_path:
                        agent.path = new_path

            if agent.inbox and not agent.is_thinking:
                agent.is_thinking = True
                task = asyncio.create_task(
                    self._conflict_coordinator.handle_incoming_dialogue(
                        agent, self._grid, self._entities
                    )
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                continue

            if agent.has_path:
                next_pos = agent.path[0]
                blocking_entity = next(
                    (other for other in visible_entities if other.position == next_pos),
                    None,
                )

                if blocking_entity:
                    agent.is_thinking = True
                    task = asyncio.create_task(
                        self._conflict_coordinator.resolve_blockage(
                            agent, blocking_entity, next_pos, self._grid, self._entities
                        )
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                else:
                    prev_pos = agent.position
                    agent.step()
                    self._logger.log(
                        SimulationEvent(
                            tick=self._current_tick,
                            agent_id=agent.id,
                            event_type="agent_moved",
                            summary=f"Agent {agent.name} hat sich von ({prev_pos.x}, {prev_pos.y}) nach ({agent.position.x}, {agent.position.y}) bewegt.",
                            payload={
                                "from": {"x": prev_pos.x, "y": prev_pos.y},
                                "to": {"x": agent.position.x, "y": agent.position.y},
                                "remaining_steps": len(agent.path),
                                "energy": agent.energy,
                            },
                        )
                    )

                    if not agent.has_path and len(agent.goals) > 1:
                        agent.is_thinking = True
                        task = asyncio.create_task(
                            self._goal_service.evaluate_sub_goal_completion(
                                agent, self._grid, self._conflict_coordinator.dialogues
                            )
                        )
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)

    async def run(self, max_ticks: int = 20) -> None:
        self._is_running = True

        for agent in self._agents:
            self._update_agent_perception(agent)

        known_tiles = self._collect_known_positions()
        self._presenter.render(
            self._grid, self._entities, self._current_tick, self._conflict_coordinator.dialogues, known_positions=known_tiles
        )

        while self._is_running and self._current_tick < max_ticks:
            await asyncio.sleep(self._tick_interval)
            self.process_tick()
            known_tiles = self._collect_known_positions()
            self._presenter.render(
                self._grid, self._entities, self._current_tick, self._conflict_coordinator.dialogues, known_positions=known_tiles
            )

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._is_running = False