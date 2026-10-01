from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from src.application.services.daily_event_buffer import DailyEventBuffer
from src.application.services.memory_consolidation_service import MemoryConsolidationService
from src.domain.models.agent import Agent
from src.domain.models.events import SimulationEvent
from src.domain.models.position import Position
from src.domain.ports.event_logger import IEventLogger
from src.domain.ports.vector_memory_store import IVectorMemoryStore


class TestMemoryConsolidationService:
    @pytest.fixture
    def mock_vector_store(self) -> MagicMock:
        return MagicMock(spec=IVectorMemoryStore)

    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        return MagicMock(spec=IEventLogger)

    @pytest.fixture
    def event_buffer(self) -> DailyEventBuffer:
        return DailyEventBuffer()

    @pytest.fixture
    def service(
        self,
        mock_vector_store: MagicMock,
        event_buffer: DailyEventBuffer,
        mock_logger: MagicMock,
    ) -> MemoryConsolidationService:
        return MemoryConsolidationService(
            vector_store=mock_vector_store,
            event_buffer=event_buffer,
            logger=mock_logger,
        )

    def test_consolidate_empty_events_returns_empty(
        self, service: MemoryConsolidationService, mock_vector_store: MagicMock
    ) -> None:
        agent = Agent(id="a1", name="Alice", position=Position(0, 0))
        res = service.consolidate_agent(agent, day_number=1, tick=100)

        assert res == []
        mock_vector_store.add_memories.assert_not_called()

    def test_consolidate_resource_and_social_events(
        self,
        service: MemoryConsolidationService,
        event_buffer: DailyEventBuffer,
        mock_vector_store: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        agent = Agent(id="a1", name="Alice", position=Position(0, 0))

        # Ereignisse im Puffer hinterlegen
        event_buffer.record_event(
            agent.id,
            SimulationEvent(
                tick=25,
                agent_id=agent.id,
                event_type="resource_consumed",
                summary="Wasser getrunken",
                payload={"resource_type": "Wasserquelle", "position": [3, 4]},
            ),
        )
        event_buffer.record_event(
            agent.id,
            SimulationEvent(
                tick=60,
                agent_id=agent.id,
                event_type="dialogue_resolved",
                summary="Verhandlung mit Bob",
                payload={"partner_id": "bob", "outcome": "cooperative"},
            ),
        )

        res = service.consolidate_agent(agent, day_number=1, tick=100)

        assert len(res) == 2
        assert "Wasserquelle bei (3, 4) erfolgreich genutzt" in res[0]
        assert "Begegnung mit bob verlief cooperative" in res[1]

        mock_vector_store.add_memories.assert_called_once()
        call_args = mock_vector_store.add_memories.call_args[1]
        assert call_args["agent_id"] == "a1"
        assert len(call_args["memories"]) == 2
        assert call_args["metadatas"][0]["category"] == "resource"
        assert call_args["metadatas"][1]["category"] == "social"
        assert call_args["metadatas"][1]["interlocutor_id"] == "bob"

        # Puffer für Agent muss nach Konsolidierung geleert sein
        assert event_buffer.get_events_for_agent(agent.id) == []
        mock_logger.log.assert_called_once()

    def test_consolidate_all_multiple_agents(
        self,
        service: MemoryConsolidationService,
        event_buffer: DailyEventBuffer,
        mock_vector_store: MagicMock,
    ) -> None:
        alice = Agent(id="a1", name="Alice", position=Position(0, 0))
        bob = Agent(id="b1", name="Bob", position=Position(1, 1))

        event_buffer.record_event(
            alice.id,
            SimulationEvent(
                tick=10,
                agent_id=alice.id,
                event_type="evasion_hold_started",
                summary="Ausgewichen",
                payload={"yield_for_agent_id": "b1"},
            ),
        )

        res = service.consolidate_all([alice, bob], day_number=1, tick=100)

        assert len(res["a1"]) == 1
        assert len(res["b1"]) == 0
        assert mock_vector_store.add_memories.call_count == 1