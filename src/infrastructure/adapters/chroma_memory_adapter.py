from __future__ import annotations

import uuid
from typing import Any, Optional

import chromadb
from chromadb.api.models.Collection import Collection

from src.domain.ports.vector_memory_store import IVectorMemoryStore


class ChromaMemoryAdapter(IVectorMemoryStore):
    """
    Infrastruktur-Adapter für ChromaDB.
    Verwaltet alle Agenten-Erinnerungen in einer einzigen geteilten Collection
    und erzwingt strikte Mandantentrennung über 'owner_agent_id'.
    """

    def __init__(
        self,
        collection_name: str = "society_sim_memories",
        client: Optional[chromadb.ClientAPI] = None,
    ) -> None:
        self._client = client or chromadb.Client()
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Gemeinsamer episodischer Vektorspeicher aller Agenten"},
        )

    def add_memories(
        self,
        agent_id: str,
        memories: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        if not memories:
            return

        ids = [f"{agent_id}_{uuid.uuid4().hex[:8]}" for _ in memories]
        resolved_metadatas: list[dict[str, Any]] = []

        for idx, text in enumerate(memories):
            meta = dict(metadatas[idx]) if metadatas and idx < len(metadatas) else {}
            # Primärfilter zwingend setzen
            meta["owner_agent_id"] = agent_id
            resolved_metadatas.append(meta)

        self._collection.add(
            ids=ids,
            documents=memories,
            metadatas=resolved_metadatas,
        )

    def retrieve_relevant(
        self,
        agent_id: str,
        query: str,
        limit: int = 3,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        # Primärbedingung: Nur Daten des anfragenden Agenten
        owner_condition: dict[str, Any] = {"owner_agent_id": {"$eq": agent_id}}

        if metadata_filter:
            # Kombiniere Zusatzfilter via $and
            conditions: list[dict[str, Any]] = [owner_condition]
            for key, val in metadata_filter.items():
                if isinstance(val, dict):
                    conditions.append({key: val})
                else:
                    conditions.append({key: {"$eq": val}})
            final_where: dict[str, Any] = {"$and": conditions}
        else:
            final_where = owner_condition

        results = self._collection.query(
            query_texts=[query],
            n_results=limit,
            where=final_where,
        )

        documents = results.get("documents")
        if documents and len(documents) > 0:
            return documents[0]
        return []