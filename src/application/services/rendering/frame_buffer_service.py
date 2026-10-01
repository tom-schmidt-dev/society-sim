from __future__ import annotations

import queue
from typing import Optional

from src.domain.models.rendering.render_frame import RenderFrame


class FrameBufferService:
    """Thread-sicherer FIFO-Puffer zur Glättung und Entkopplung von Simulation und Anzeige."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._queue: queue.Queue[RenderFrame] = queue.Queue(maxsize=maxsize)

    def push_frame(
        self,
        frame: RenderFrame,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        """Fügt einen Frame ein. Gibt False zurück, falls der Puffer voll ist."""
        try:
            self._queue.put(frame, block=block, timeout=timeout)
            return True
        except queue.Full:
            return False

    def pop_frame(self, timeout: Optional[float] = None) -> Optional[RenderFrame]:
        """Liest den ältesten Frame aus. Gibt None zurück, falls der Puffer leer ist."""
        try:
            if timeout is None:
                return self._queue.get_nowait()
            return self._queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        """Leert den Puffer atomar."""
        with self._queue.mutex:
            self._queue.queue.clear()

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()