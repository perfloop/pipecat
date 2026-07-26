#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Focused pytest-benchmark coverage for markerless turn-completion responses."""

import asyncio
from dataclasses import dataclass

from pipecat.frames.frames import Frame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionLLMServiceMixin

_LARGE_RESPONSE_BYTES = 64 * 1024
_SMALL_RESPONSE_BYTES = 32
_CHUNK_BYTES = 32


class _FirstTextBarrierProcessor(UserTurnCompletionLLMServiceMixin, FrameProcessor):
    """Capture the first downstream text frame and hold the producer at that boundary."""

    def __init__(self):
        super().__init__()
        self.first_text_observed = asyncio.Event()
        self.release_first_text = asyncio.Event()
        self.text_frames: list[LLMTextFrame] = []

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        if not isinstance(frame, LLMTextFrame):
            return

        self.text_frames.append(frame)
        if len(self.text_frames) == 1:
            self.first_text_observed.set()
            # Do not let the producer execute any work after its first emitted
            # text frame until pytest-benchmark has stopped its timer.
            await self.release_first_text.wait()


async def _drive_markerless_response(
    processor: _FirstTextBarrierProcessor, chunks: tuple[str, ...]
) -> None:
    for chunk in chunks:
        await processor._push_turn_text(chunk)
    # The legacy implementation emits its first text frame at response end.
    # The barrier above holds this coroutine at that frame; completing the
    # reset happens in teardown, outside the measured callback.
    await processor._turn_reset()


@dataclass
class _BenchmarkState:
    """Fresh per-round state that is intentionally prepared outside timing."""

    loop: asyncio.AbstractEventLoop
    processor: _FirstTextBarrierProcessor
    task: asyncio.Task[None]
    expected_text: str


def _markerless_chunks(nonce: int, response_bytes: int) -> tuple[str, tuple[str, ...]]:
    """Create a runtime-varied, marker-free response outside timing."""
    pattern = f"markerless-{nonce:x}-"
    text = (pattern * ((response_bytes // len(pattern)) + 1))[:response_bytes]
    chunks = tuple(
        text[offset : offset + _CHUNK_BYTES] for offset in range(0, len(text), _CHUNK_BYTES)
    )
    return text, chunks


def _setup(response_bytes: int) -> tuple[tuple[_BenchmarkState], dict[str, object]]:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    expected_text, chunks = _markerless_chunks(id(loop), response_bytes)
    processor = _FirstTextBarrierProcessor()
    task = loop.create_task(_drive_markerless_response(processor, chunks))
    return (_BenchmarkState(loop, processor, task, expected_text),), {}


def _wait_for_first_text(state: _BenchmarkState) -> str:
    """Run exactly until the downstream observer sees the first text frame."""
    state.loop.run_until_complete(state.processor.first_text_observed.wait())
    return state.processor.text_frames[0].text


def _teardown(state: _BenchmarkState) -> None:
    """Finish, validate, and dispose of one response after timing has stopped."""
    try:
        assert state.processor.text_frames
        assert state.expected_text.startswith(state.processor.text_frames[0].text)

        state.processor.release_first_text.set()
        state.loop.run_until_complete(state.task)

        assert "".join(frame.text for frame in state.processor.text_frames) == state.expected_text
        assert state.processor._turn_text_buffer == ""
        assert state.processor._turn_marker is None
    finally:
        if not state.task.done():
            state.task.cancel()
            state.loop.run_until_complete(asyncio.gather(state.task, return_exceptions=True))
        asyncio.set_event_loop(None)
        state.loop.close()


def _benchmark_first_text(benchmark, response_bytes: int) -> str:
    """Measure response-start to first text for one markerless response size."""
    return benchmark.pedantic(
        _wait_for_first_text,
        setup=lambda: _setup(response_bytes),
        teardown=_teardown,
        rounds=3,
        warmup_rounds=1,
        iterations=1,
    )


def test_turn_completion_markerless_64k_first_text(benchmark):
    """Measure response-start to first text for a markerless 64 KiB stream."""
    assert _benchmark_first_text(benchmark, _LARGE_RESPONSE_BYTES)


def test_turn_completion_markerless_32b_first_text(benchmark):
    """Guard markerless response latency below the bounded-prefix limit."""
    assert _benchmark_first_text(benchmark, _SMALL_RESPONSE_BYTES)
