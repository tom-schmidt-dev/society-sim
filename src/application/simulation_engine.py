from __future__ import annotations

import asyncio
from typing import Any, Optional
from src.application.services.action_executor import ActionExecutor
from src.application.services.conflict_coordinator import ConflictCoordinator
from src.application.services.dialogue_coordinator import DialogueCoordinator
from src.application.services.dialogue_history import DialogueHistory
from src.application.services.dialogue_session_manager import DialogueSessionManager
from src.application.services.evasion_finder import EvasionFinder
from src.application.services.goal_service import GoalService
from src.domain.models.agent import Agent
from src.domain.models.events import SimulationEvent
from src.domain.models.goal import Goal
from src.domain.models.position import Position
from src.domain.models.world import WorldGrid
from src.domain.models.world_entity import WorldEntity
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.conflict_coordinator import IConflictCoordinator
from src.domain.ports.dialogue_coordinator import IDialogueCoordinator
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter
from src.domain.services.perception_service import PerceptionService


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
        conflict_coordinator: Optional[IConflictCoordinator] = None,
        dialogue_coordinator: Optional[IDialogueCoordinator] = None,
        dialogue_history: Optional[DialogueHistory] = None,
        perception_service: Optional[PerceptionService] = None,
    ) -> None:
        self._entities: list[WorldEntity] = []
        self._grid: WorldGrid = grid
        self._pathfinder: IPathfinder = pathfinder
        self._presenter: IPresenter = presenter
        self._logger: IEventLogger = logger
        self._tick_interval: float = tick_interval

        self._agents: list[Agent] = []
        self._current_tick: int = 0
        self._is_running: bool = False
        self._background_tasks: set[asyncio.Task[Any]] = set()

        self._dialogue_history = dialogue_history or DialogueHistory()
        self._perception_service = perception_service or PerceptionService(default_radius=3)
        self._goal_service = goal_service or GoalService(
            logger, cognition_provider, pathfinder, tick_provider=lambda: self._current_tick
        )
        self._evasion_finder = EvasionFinder(pathfinder)
        self._session_manager = DialogueSessionManager(max_dialogue_turns=2)

        self._action_executor = ActionExecutor(
            grid=self._grid,
            logger=self._logger,
            dialogue_history=self._dialogue_history,
            goal_service=self._goal_service,
            pathfinder=self._pathfinder,
            tick_provider=lambda: self._current_tick,
        )

        self._conflict_coordinator = conflict_coordinator or ConflictCoordinator(
            logger=self._logger,
            cognition_provider=cognition_provider,
            pathfinder=self._pathfinder,
            goal_service=self._goal_service,
            evasion_finder=self._evasion_finder,
            action_executor=self._action_executor,
            session_manager=self._session_manager,
            dialogue_history=self._dialogue_history,
            tick_provider=lambda: self._current_tick,
        )

        self._dialogue_coordinator = dialogue_coordinator or DialogueCoordinator(
            logger=self._logger,
            cognition_provider=cognition_provider,
            pathfinder=self._pathfinder,
            goal_service=self._goal_service,
            evasion_finder=self._evasion_finder,
            action_executor=self._action_executor,
            session_manager=self._session_manager,
            dialogue_history=self._dialogue_history,
            tick_provider=lambda: self._current_tick,
        )

    @property
    def current_tick(self) -> int:
        return self._current_tick

    @property
    def dialogue_history(self) -> DialogueHistory:
        return self._dialogue_history

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
        self._goal_service.push_goal(agent, Goal(name=destination_name, target_position=target))

        self._update_agent_perception(agent)
        path = self._pathfinder.find_path(agent.position, target, agent.mental_map)
        if path:
            agent.assign_path(path)
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
            tile_type = self._grid.get_tile_type(pos)
            # Sichtbare statische Wände direkt als Hindernis erfassen
            if not self._grid.is_walkable(pos) or agent.memory.get_type_assumed_walkable(tile_type) is False:
                agent.mental_map.mark_obstacle(pos, self._current_tick)
            else:
                agent.mental_map.update_tile(pos, is_walkable=True, tick=self._current_tick)

        visible_entities = self._perception_service.get_visible_entities(agent, self._entities)
        for visible_entity in visible_entities:
            agent.memory.update_entity_perception(
                entity_id=visible_entity.id,
                name=visible_entity.name,
                pos=visible_entity.position,
                tick=self._current_tick,
            )
            # Prinzip 3: Erst markieren, wenn physisch unpassierbar UND Sprache geklärt ist
            if (
                agent.memory.is_epistemically_exhausted(visible_entity.id)
                and agent.memory.get_assumed_walkable(visible_entity.id) is False
            ):
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

            # 1. Posteingang hat Vorrang: Konversation auch während Haltezielen ausführen
            if agent.inbox and not agent.is_thinking:
                agent.is_thinking = True
                task = asyncio.create_task(
                    self._dialogue_coordinator.handle_incoming_dialogue(
                        agent, self._entities
                    )
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                continue

            # 2. Befristete Halteziele dekrementieren
            if self._goal_service.process_timed_goal(agent):
                continue

            # 3. Physische Blockade/Beschäftigung stoppt nur die Fortbewegung
            if agent.is_busy:
                continue

            visible_entities = self._update_agent_perception(agent)

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
                        agent.assign_path(new_path)

            if agent.has_path:
                next_pos = agent.path[0]
                blocking_entity = next(
                    (other for other in visible_entities if other.position == next_pos),
                    None,
                )

                all_context_entities = list(self._entities)
                if not blocking_entity and not self._grid.is_walkable(next_pos):
                    tile_type = self._grid.get_tile_type(next_pos)
                    tile_name = (
                        "Mauer"
                        if tile_type == "wall"
                        else ("Weltgrenze" if tile_type == "boundary" else tile_type.capitalize())
                    )
                    blocking_entity = WorldEntity(
                        id=f"{tile_type}_{next_pos.x}_{next_pos.y}",
                        name=tile_name,
                        position=next_pos,
                        entity_type=tile_type,
                        is_conversational=False,
                        is_passable=False,
                    )
                    all_context_entities.append(blocking_entity)

                if blocking_entity:
                    agent.is_thinking = True
                    task = asyncio.create_task(
                        self._conflict_coordinator.resolve_blockage(
                            agent, blocking_entity, next_pos, all_context_entities
                        )
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                else:
                    prev_pos = agent.position
                    agent.step()

                    current_tile_type = self._grid.get_tile_type(agent.position)
                    if agent.memory.get_type_assumed_walkable(current_tile_type) is None:
                        agent.memory.record_type_walkability(current_tile_type, is_walkable=True, pos=agent.position)
                        self._logger.log(
                            SimulationEvent(
                                tick=self._current_tick,
                                agent_id=agent.id,
                                event_type="tile_type_learned",
                                summary=f"Agent {agent.name} erfährt durch Bewegung: Kacheltyp '{current_tile_type}' ist passierbar.",
                                payload={"tile_type": current_tile_type, "is_walkable": True},
                            )
                        )

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
                                agent, self._grid, self._dialogue_history.get_recent_formatted(limit=8)
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
            self._grid,
            self._entities,
            self._current_tick,
            self._dialogue_history.get_recent_formatted(limit=8),
            known_positions=known_tiles,
        )

        while self._is_running and self._current_tick < max_ticks:
            await asyncio.sleep(self._tick_interval)
            self.process_tick()
            known_tiles = self._collect_known_positions()
            self._presenter.render(
                self._grid,
                self._entities,
                self._current_tick,
                self._dialogue_history.get_recent_formatted(limit=8),
                known_positions=known_tiles,
            )

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._is_running = False