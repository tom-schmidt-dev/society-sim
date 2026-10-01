from __future__ import annotations

import chromadb
import pytest
import uuid

from src.infrastructure.adapters.chroma_memory_adapter import ChromaMemoryAdapter


class TestChromaMemoryAdapter:
    @pytest.fixture
    def memory_store(self) -> ChromaMemoryAdapter:
        # Ephemerer In-Memory-Client mit isolierter Collection pro Test
        client = chromadb.EphemeralClient()
        return ChromaMemoryAdapter(
            collection_name=f"test_{uuid.uuid4().hex[:8]}", client=client
        )

    def test_add_and_retrieve_own_memories(self, memory_store: ChromaMemoryAdapter) -> None:
        agent_id = "alice"
        memories = [
            "Takt 10: Habe Wasser am Brunnen getrunken.",
            "Takt 15: Bob hat mir den Weg versperrt.",
        ]
        metadatas = [
            {"tick": 10, "category": "resource"},
            {"tick": 15, "category": "social", "interlocutor_id": "bob"},
        ]

        memory_store.add_memories(agent_id, memories, metadatas)

        results = memory_store.retrieve_relevant(
            agent_id=agent_id,
            query="Wo finde ich Wasser?",
            limit=1,
        )

        assert len(results) == 1
        assert "Wasser" in results[0]

    def test_strict_isolation_between_agents(self, memory_store: ChromaMemoryAdapter) -> None:
        # Alice speichert ein Geheimnis
        memory_store.add_memories(
            agent_id="alice",
            memories=["Geheimer Vorrat bei Koordinate (10, 20) versteckt."],
            metadatas=[{"category": "resource"}],
        )

        # Bob sucht nach Ressourcen
        bob_results = memory_store.retrieve_relevant(
            agent_id="bob",
            query="Geheimer Vorrat",
            limit=5,
        )

        # Bob darf keine Einträge von Alice sehen
        assert len(bob_results) == 0

    def test_retrieve_with_interlocutor_filter(self, memory_store: ChromaMemoryAdapter) -> None:
        agent_id = "alice"
        memories = [
            "Bob hat kooperativ Platz gemacht.",
            "Charlie hat Verhandlung abgelehnt.",
        ]
        metadatas = [
            {"interlocutor_id": "bob", "sentiment": 1.0},
            {"interlocutor_id": "charlie", "sentiment": -1.0},
        ]

        memory_store.add_memories(agent_id, memories, metadatas)

        # Abfrage gefiltert nach Bob
        results_bob = memory_store.retrieve_relevant(
            agent_id=agent_id,
            query="Verhalten von Mitbewohnern",
            limit=2,
            metadata_filter={"interlocutor_id": "bob"},
        )

        assert len(results_bob) == 1
        assert "Bob" in results_bob[0]
        assert "Charlie" not in results_bob[0]

    def test_empty_results_when_no_match(self, memory_store: ChromaMemoryAdapter) -> None:
        results = memory_store.retrieve_relevant(
            agent_id="nobody",
            query="Irgendetwas",
            limit=3,
        )
        assert results == []