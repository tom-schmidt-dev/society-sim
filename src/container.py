from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
from src.domain.models.world import WorldGrid
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.services.perception_service import PerceptionService
from src.infrastructure.pathfinding.astar import AStarPathfinder
from src.infrastructure.presentation.console_presenter import ConsolePresenter
from src.infrastructure.logging.jsonl_logger import JsonlEventLogger
from src.infrastructure.cognition.instructor_adapter import InstructorCognitionAdapter
from src.application.simulation_engine import SimulationEngine


@dataclass
class ApplicationContainer:
    engine: SimulationEngine
    grid: WorldGrid
    logger: IEventLogger
    cognition: ICognitionProvider

    @classmethod
    def build(
            cls,
            width: int = 10,
            height: int = 10,
            tick_interval: float = 0.3,
            log_file: str = "logs/simulation_events.jsonl",
            blockage_strategy: Literal["action_masking", "reflection"] = "action_masking",
            cognition_provider: Optional[ICognitionProvider] = None,
    ) -> ApplicationContainer:
        grid = WorldGrid(width=width, height=height)
        pathfinder: IPathfinder = AStarPathfinder()
        presenter: IPresenter = ConsolePresenter()
        logger: IEventLogger = JsonlEventLogger()
        perception_service = PerceptionService(default_radius=3)

        cognition = cognition_provider or InstructorCognitionAdapter(
            model_name="ollama/llama3-8b-q8",
            api_base="http://localhost:11434",
            temperature=0.2,
            blockage_strategy=blockage_strategy,
        )

        engine = SimulationEngine(
            grid=grid,
            pathfinder=pathfinder,
            presenter=presenter,
            logger=logger,
            cognition_provider=cognition,
            tick_interval=tick_interval,
            perception_service=perception_service,
        )
        return cls(engine=engine, grid=grid, logger=logger, cognition=cognition)