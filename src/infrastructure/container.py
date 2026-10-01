from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from src.application.services.convoy_arbitrator import ConvoyArbitrator
from src.application.services.convoy_coordinator import ConvoyCoordinator
from src.application.services.daily_event_buffer import DailyEventBuffer
from src.application.services.day_night_service import DayNightService
from src.application.services.evasion_finder import EvasionFinder
from src.application.services.memory_consolidation_service import MemoryConsolidationService
from src.application.services.movement_sync_service import MovementSyncService
from src.application.services.multi_agent_niche_packer import MultiAgentNichePacker
from src.application.services.need_service import NeedService
from src.domain.services.perception_service import PerceptionService
from src.application.services.target_search_service import TargetSearchService
from src.application.simulation_engine import SimulationEngine
from src.application.services.dialogue_history import DialogueHistory
from src.domain.models.world import WorldGrid
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter
from src.domain.ports.vector_memory_store import IVectorMemoryStore
from src.infrastructure.pathfinding.astar import AStarPathfinder
from src.infrastructure.adapters.chroma_memory_adapter import ChromaMemoryAdapter
from src.infrastructure.cognition.instructor_adapter import InstructorCognitionAdapter
from src.infrastructure.logging.buffering_event_logger import BufferingEventLogger
from src.infrastructure.logging.jsonl_logger import JsonlEventLogger
from src.infrastructure.presentation.console_presenter import ConsolePresenter


@dataclass
class ApplicationContainer:
    engine: SimulationEngine
    grid: WorldGrid
    logger: IEventLogger
    cognition: ICognitionProvider
    dialogue_history: DialogueHistory
    convoy_coordinator: ConvoyCoordinator
    niche_packer: MultiAgentNichePacker
    movement_sync_service: MovementSyncService
    convoy_arbitrator: ConvoyArbitrator
    need_service: NeedService
    day_night_service: DayNightService
    vector_store: IVectorMemoryStore
    memory_consolidation_service: MemoryConsolidationService

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
        need_service: Optional[NeedService] = None,
        day_night_service: Optional[DayNightService] = None,
        vector_store: Optional[IVectorMemoryStore] = None,
        memory_consolidation_service: Optional[MemoryConsolidationService] = None,
    ) -> ApplicationContainer:
        grid = WorldGrid(width=width, height=height)
        pathfinder: IPathfinder = AStarPathfinder()
        presenter: IPresenter = ConsolePresenter()

        daily_event_buffer = DailyEventBuffer()
        raw_logger: IEventLogger = JsonlEventLogger()
        logger: IEventLogger = BufferingEventLogger(
            inner_logger=raw_logger, buffer=daily_event_buffer
        )

        perception_service = PerceptionService(default_radius=3)
        dialogue_history = DialogueHistory()
        evasion_finder = EvasionFinder(pathfinder)
        convoy_coordinator = ConvoyCoordinator(pathfinder=pathfinder, logger=logger)
        niche_packer = MultiAgentNichePacker()
        movement_sync_service = MovementSyncService(logger=logger)
        convoy_arbitrator = ConvoyArbitrator()
        need_service = need_service or NeedService()
        day_night_service = day_night_service or DayNightService(logger=logger)

        vector_store = vector_store or ChromaMemoryAdapter()
        memory_consolidation_service = memory_consolidation_service or MemoryConsolidationService(
            vector_store=vector_store,
            event_buffer=daily_event_buffer,
            logger=logger,
        )

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
            movement_sync_service=movement_sync_service,
            convoy_coordinator=convoy_coordinator,
            niche_packer=niche_packer,
            convoy_arbitrator=convoy_arbitrator,
            need_service=need_service,
            day_night_service=day_night_service,
            memory_consolidation_service=memory_consolidation_service,
        )

        return cls(
            engine=engine,
            grid=grid,
            logger=logger,
            cognition=cognition,
            dialogue_history=dialogue_history,
            convoy_coordinator=convoy_coordinator,
            niche_packer=niche_packer,
            movement_sync_service=movement_sync_service,
            convoy_arbitrator=convoy_arbitrator,
            need_service=need_service,
            day_night_service=day_night_service,
            vector_store=vector_store,
            memory_consolidation_service=memory_consolidation_service,
        )