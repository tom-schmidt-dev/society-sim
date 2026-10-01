from __future__ import annotations

from src.application.services.rendering.frame_buffer_service import FrameBufferService
from src.domain.models.rendering.render_frame import RenderFrame


def _make_frame(tick: int) -> RenderFrame:
    return RenderFrame(
        tick=tick,
        timestamp=float(tick),
        grid_matrix=[[" . "]],
        entities=[],
    )


def test_push_and_pop_fifo() -> None:
    buffer = FrameBufferService()
    assert buffer.is_empty is True
    assert buffer.size == 0

    buffer.push_frame(_make_frame(1))
    buffer.push_frame(_make_frame(2))

    assert buffer.size == 2
    assert buffer.is_empty is False

    f1 = buffer.pop_frame()
    assert f1 is not None
    assert f1.tick == 1

    f2 = buffer.pop_frame()
    assert f2 is not None
    assert f2.tick == 2

    assert buffer.is_empty is True


def test_pop_empty_queue() -> None:
    buffer = FrameBufferService()
    assert buffer.pop_frame() is None
    assert buffer.pop_frame(timeout=0.01) is None


def test_clear_buffer() -> None:
    buffer = FrameBufferService()
    buffer.push_frame(_make_frame(1))
    buffer.push_frame(_make_frame(2))
    assert buffer.size == 2

    buffer.clear()
    assert buffer.size == 0
    assert buffer.is_empty is True


def test_buffer_full_handling() -> None:
    buffer = FrameBufferService(maxsize=2)
    assert buffer.push_frame(_make_frame(1)) is True
    assert buffer.push_frame(_make_frame(2)) is True
    # Bei block=False schlägt das Einfügen fehl
    assert buffer.push_frame(_make_frame(3), block=False) is False
    assert buffer.size == 2