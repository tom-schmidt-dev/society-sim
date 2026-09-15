from __future__ import annotations

import asyncio
from typing import Optional


class DialogueSessionManager:
    def __init__(self, max_dialogue_turns: int = 2) -> None:
        self._max_dialogue_turns: int = max_dialogue_turns
        self._conversation_locks: dict[frozenset[str], asyncio.Lock] = {}
        self._dialogue_turns: dict[frozenset[str], int] = {}

    @property
    def max_dialogue_turns(self) -> int:
        return self._max_dialogue_turns

    def _get_pair_key(self, agent_a_id: str, agent_b_id: Optional[str] = None) -> frozenset[str]:
        """Erzeugt einen symmetrischen Schlüssel für das Entitätenpaar."""
        if agent_b_id:
            return frozenset({agent_a_id, agent_b_id})
        return frozenset({agent_a_id})

    def get_lock(self, agent_a_id: str, agent_b_id: Optional[str] = None) -> asyncio.Lock:
        """Gibt das asyncio.Lock für das gegebene Entitätenpaar zurück (erzeugt es bei Bedarf)."""
        pair_key = self._get_pair_key(agent_a_id, agent_b_id)
        if pair_key not in self._conversation_locks:
            self._conversation_locks[pair_key] = asyncio.Lock()
        return self._conversation_locks[pair_key]

    def increment_turn(self, agent_a_id: str, agent_b_id: Optional[str] = None) -> int:
        """Erhöht den Rundenzähler für das Paar und gibt die neue Rundennummer zurück."""
        pair_key = self._get_pair_key(agent_a_id, agent_b_id)
        current = self._dialogue_turns.get(pair_key, 0) + 1
        self._dialogue_turns[pair_key] = current
        return current

    def get_turn_count(self, agent_a_id: str, agent_b_id: Optional[str] = None) -> int:
        """Ermittelt die aktuell erreichte Rundenzahl für das Paar."""
        pair_key = self._get_pair_key(agent_a_id, agent_b_id)
        return self._dialogue_turns.get(pair_key, 0)

    def is_turn_limit_exceeded(self, agent_a_id: str, agent_b_id: Optional[str] = None) -> bool:
        """Prüft, ob die maximal erlaubte Rundenanzahl überschritten wurde."""
        return self.get_turn_count(agent_a_id, agent_b_id) > self._max_dialogue_turns

    def reset_session(self, agent_a_id: str, agent_b_id: Optional[str] = None) -> None:
        """Setzt den Rundenzähler nach Beendigung des Dialogs zurück."""
        pair_key = self._get_pair_key(agent_a_id, agent_b_id)
        self._dialogue_turns.pop(pair_key, None)