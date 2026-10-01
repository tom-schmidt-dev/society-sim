from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class IVectorMemoryStore(ABC):
    """Port für den Zugriff auf semantisch indizierte episodische Gedächtniseinträge."""

    @abstractmethod
    def add_memories(
        self,
        agent_id: str,
        memories: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """
        Fügt Gedächtnissätze eines Agenten dem Vektorspeicher hinzu.
        Ergänzt automatisch 'owner_agent_id' in den Metadaten.
        """
        pass

    @abstractmethod
    def retrieve_relevant(
        self,
        agent_id: str,
        query: str,
        limit: int = 3,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        """
        Ruft die semantisch ähnlichsten Erinnerungen für den angegebenen Agenten ab.
        Stellt sicher, dass ausschließlich Erinnerungen mit owner_agent_id == agent_id zurückgegeben werden.
        """
        pass