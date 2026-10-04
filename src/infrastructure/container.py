from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from src.application.services.coordination.convoy_arbitrator import ConvoyArbitrator
from src.application.services.coordination.convoy_coordinator import ConvoyCoordinator
from src.application.services.lifecycle.daily_event_buffer import DailyEventBuffer
from src.application.services.lifecycle.day_night_service import DayNightService
from src.application.services.movement.evasion_finder import EvasionFinder
from src.application.services.lifecycle.memory_consolidation_service import MemoryConsolidationService
from src.application.services.movement.movement_sync_service import MovementSyncService
from src.application.services.movement.multi_agent_niche_packer import MultiAgentNichePacker
from src.application.services.lifecycle.need_service import NeedService
from src.domain.services.perception_service import PerceptionService
from src.application.services.movement.target_search_service import TargetSearchService
from src.application.simulation_engine import SimulationEngine
from src.application.services.coordination.dialogue_history import DialogueHistory
from src.domain.models.world.world import WorldGrid
from src.domain.ports.cognition_provider import ICognitionProvider
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.interaction_dispatcher import IInteractionDispatcher
from src.domain.ports.pathfinder import IPathfinder
from src.domain.ports.presenter import IPresenter
from src.domain.ports.vector_memory_store import IVectorMemoryStore
from src.infrastructure.pathfinding.astar import AStarPathfinder
from src.infrastructure.adapters.chroma_memory_adapter import ChromaMemoryAdapter
from src.infrastructure.cognition.adapters.instructor_adapter import InstructorCognitionAdapter
from src.infrastructure.logging.buffering_event_logger import BufferingEventLogger
from src.infrastructure.logging.jsonl_logger import JsonlEventLogger
from src.infrastructure.presentation.console.console_presenter import ConsolePresenter
from src.infrastructure.cognition.health.ollama_health_checker import OllamaHealthChecker
from src.application.services.rendering.frame_buffer_service import FrameBufferService


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
    frame_buffer: FrameBufferService
    interaction_dispatcher: IInteractionDispatcher

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
            frame_buffer: Optional[FrameBufferService] = None,
            enable_deterministic_corridor: bool = True,
            enable_day_night: bool = True,
    ) -> ApplicationContainer:

        grid = WorldGrid(width=width, height=height)
        pathfinder: IPathfinder = AStarPathfinder()

        frame_buffer = frame_buffer or FrameBufferService()
        presenter: IPresenter = ConsolePresenter(frame_buffer=frame_buffer)

        # 1. Logger & Basis-Dienste
        daily_event_buffer = DailyEventBuffer()
        raw_logger: IEventLogger = JsonlEventLogger(file_path=log_file)
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

        if day_night_service is not None and getattr(day_night_service, "_logger", None) is None:
            day_night_service._logger = logger
        day_night_service = day_night_service or DayNightService(logger=logger)

        # 2. Vektorspeicher & Konsolidierung
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

        # 3. Kognitionsadapter & SimulationEngine
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
            vector_memory_store=vector_store,
            enable_deterministic_corridor=enable_deterministic_corridor,
            enable_day_night=enable_day_night,
        )


        # 1. Pre-Flight Health Check für Inferenz
        api_base = "http://localhost:11434"
        configured_model = "ollama/llama3-8b-q8"
        if cognition_provider is None:
            OllamaHealthChecker.check_status(api_base=api_base, model_name=configured_model)

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
            frame_buffer=frame_buffer,
            interaction_dispatcher=engine.interaction_dispatcher,
        )