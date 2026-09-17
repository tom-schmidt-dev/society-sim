from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from src.domain.models.events import SimulationEvent
from src.domain.ports.event_logger import IEventLogger


class DialogueSessionManager:
    def __init__(
        self,
        max_dialogue_turns: int = 2,
        default_lock_timeout: float = 1.5,
        logger: Optional[IEventLogger] = None,
    ) -> None:
        self._max_dialogue_turns: int = max_dialogue_turns
        self._default_lock_timeout: float = default_lock_timeout
        self._logger: Optional[IEventLogger] = logger
        self._conversation_locks: dict[tuple[str, ...], asyncio.Lock] = {}
        self._dialogue_turns: dict[tuple[str, ...], int] = {}

    @property
    def max_dialogue_turns(self) -> int:
        return self._max_dialogue_turns

    @staticmethod
    def _get_pair_key(agent_a_id: str, agent_b_id: Optional[str] = None) -> tuple[str, ...]:
        """Kanonische Sortierung zur Vermeidung zirkulärer Deadlocks."""
        if agent_b_id:
            return tuple(sorted([agent_a_id, agent_b_id]))
        return (agent_a_id,)

    def get_lock(self, agent_a_id: str, agent_b_id: Optional[str] = None) -> asyncio.Lock:
        pair_key = self._get_pair_key(agent_a_id, agent_b_id)
        if pair_key not in self._conversation_locks:
            self._conversation_locks[pair_key] = asyncio.Lock()
        return self._conversation_locks[pair_key]

    @asynccontextmanager
    async def acquire_session(
        self,
        agent_a_id: str,
        agent_b_id: Optional[str] = None,
        timeout: Optional[float] = None,
        tick: int = 0,
    ) -> AsyncIterator[bool]:
        """Asynchroner Lock mit kanonischer Sortierung und Timeout-Circuit-Breaker."""
        pair_key = self._get_pair_key(agent_a_id, agent_b_id)
        if pair_key not in self._conversation_locks:
            self._conversation_locks[pair_key] = asyncio.Lock()

        lock = self._conversation_locks[pair_key]
        effective_timeout = timeout if timeout is not None else self._default_lock_timeout
        acquired = False

        try:
            await asyncio.wait_for(lock.acquire(), timeout=effective_timeout)
            acquired = True
            yield True
        except asyncio.TimeoutError:
            if self._logger:
                self._logger.log(
                    SimulationEvent(
                        tick=tick,
                        agent_id=agent_a_id,
                        event_type="dialogue_timeout",
                        summary=f"Dialog-Lock-Timeout zwischen {agent_a_id} und {agent_b_id} nach {effective_timeout}s.",
                        payload={
                            "agent_a": agent_a_id,
                            "agent_b": agent_b_id,
                            "timeout_seconds": effective_timeout,
                        },
                    )
                )
            yield False
        finally:
            if acquired and lock.locked():
                lock.release()

    def increment_turn(self, agent_a_id: str, agent_b_id: Optional[str] = None) -> int:
        pair_key = self._get_pair_key(agent_a_id, agent_b_id)
        current = self._dialogue_turns.get(pair_key, 0) + 1
        self._dialogue_turns[pair_key] = current
        return current

    def get_turn_count(self, agent_a_id: str, agent_b_id: Optional[str] = None) -> int:
        pair_key = self._get_pair_key(agent_a_id, agent_b_id)
        return self._dialogue_turns.get(pair_key, 0)

    def is_turn_limit_exceeded(self, agent_a_id: str, agent_b_id: Optional[str] = None) -> bool:
        return self.get_turn_count(agent_a_id, agent_b_id) > self._max_dialogue_turns

    def reset_session(self, agent_a_id: str, agent_b_id: Optional[str] = None) -> None:
        pair_key = self._get_pair_key(agent_a_id, agent_b_id)
        self._dialogue_turns.pop(pair_key, None)