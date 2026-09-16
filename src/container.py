from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
from src.application.services.dialogue_history import DialogueHistory
from src.application.services.evasion_finder import EvasionFinder
from src.application.services.target_search_service import TargetSearchService
from src.application.simulation_engine import SimulationEngine
from src.domain.models.world import WorldGrid
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter
from src.domain.services.perception_service import PerceptionService
from src.infrastructure.cognition.instructor_adapter import InstructorCognitionAdapter
from src.infrastructure.logging.jsonl_logger import JsonlEventLogger
from src.infrastructure.pathfinding.astar import AStarPathfinder
from src.infrastructure.presentation.console_presenter import ConsolePresenter


@dataclass
class ApplicationContainer:
    engine: SimulationEngine
    grid: WorldGrid
    logger: IEventLogger
    cognition: ICognitionProvider
    dialogue_history: DialogueHistory

    @classmethod
    def build(
        cls,
        width: int = 10,
        height: int = 10,
        tick_interval: float = 0.3,
        auditory_radius: int = 3,
        log_file: str = "logs/simulation_events.jsonl",
        blockage_strategy: Literal["action_masking", "reflection"] = "action_masking",
        cognition_provider: Optional[ICognitionProvider] = None,
    ) -> ApplicationContainer:
        grid = WorldGrid(width=width, height=height)
        pathfinder: IPathfinder = AStarPathfinder()
        presenter: IPresenter = ConsolePresenter()
        logger: IEventLogger = JsonlEventLogger()
        perception_service = PerceptionService(default_radius=3)
        dialogue_history = DialogueHistory()
        evasion_finder = EvasionFinder(pathfinder)

        cognition = cognition_provider or InstructorCognitionAdapter(
            model_name="ollama/llama3-8b-q8",
            api_base="http://localhost:11434",
            temperature=0.2,
            blockage_strategy=blockage_strategy,
        )

        target_search_service = TargetSearchService(
            pathfinder=pathfinder,
            perception_service=perception_service,
            evasion_finder=evasion_finder,
            logger=logger,
        )

        engine = SimulationEngine(
            grid=grid,
            pathfinder=pathfinder,
            presenter=presenter,
            logger=logger,
            cognition_provider=cognition,
            tick_interval=tick_interval,
            perception_service=perception_service,
            dialogue_history=dialogue_history,
            target_search_service=target_search_service,
            auditory_radius=auditory_radius,
        )
        return cls(
            engine=engine,
            grid=grid,
            logger=logger,
            cognition=cognition,
            dialogue_history=dialogue_history,
        )