#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Queued FrameProcessor benchmark coverage for markerless turn-completion responses."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import Frame, LLMTextFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.benchmarks.test_user_turn_completion_benchmark import (
    _LARGE_RESPONSE_BYTES,
    _FirstTextBarrierProcessor,
    _markerless_chunks,
)


class _QueuedFirstTextBarrierProcessor(_FirstTextBarrierProcessor):
    """Drive markerless text through the queued FrameProcessor task path."""

    def __init__(self, chunks: tuple[str, ...]):
        super().__init__()
        self._chunks = chunks
        self.response_finished = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, LLMTextFrame):
            for chunk in self._chunks:
                await self._push_turn_text(chunk)
            await self._turn_reset()
            self.response_finished.set()
            return

        await super().process_frame(frame, direction)


@dataclass
class _QueuedBenchmarkState:
    """Fresh state for one queued FrameProcessor benchmark round."""

    loop: asyncio.AbstractEventLoop
    processor: _QueuedFirstTextBarrierProcessor
    expected_text: str


def _setup(response_bytes: int) -> tuple[tuple[_QueuedBenchmarkState], dict[str, object]]:
    """Prepare a processor whose non-system frames use the queued task handler."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    expected_text, chunks = _markerless_chunks(id(loop), response_bytes)
    processor = _QueuedFirstTextBarrierProcessor(chunks)
    task_manager = TaskManager(loop=loop)
    loop.run_until_complete(
        processor.setup(
            FrameProcessorSetup(
                clock=SystemClock(),
                task_manager=task_manager,
                pipeline_worker=SimpleNamespace(app_resources=None),
            )
        )
    )
    # StartFrame creates FrameProcessor.__process_frame_task_handler. The timed
    # callback below queues the non-system LLMTextFrame that this task consumes.
    loop.run_until_complete(processor.process_frame(StartFrame(), FrameDirection.DOWNSTREAM))
    return (_QueuedBenchmarkState(loop, processor, expected_text),), {}


def _queue_and_wait_for_first_text(state: _QueuedBenchmarkState) -> str:
    """Queue one response and run through the first observed text frame."""
    state.loop.run_until_complete(state.processor.queue_frame(LLMTextFrame(text="")))
    state.loop.run_until_complete(state.processor.first_text_observed.wait())
    return state.processor.text_frames[0].text


def _teardown(state: _QueuedBenchmarkState) -> None:
    """Finish and validate one queued response after timing has stopped."""
    try:
        assert state.processor.text_frames
        assert state.expected_text.startswith(state.processor.text_frames[0].text)

        state.processor.release_first_text.set()
        state.loop.run_until_complete(
            asyncio.wait_for(state.processor.response_finished.wait(), timeout=1)
        )

        assert "".join(frame.text for frame in state.processor.text_frames) == state.expected_text
        assert state.processor._turn_text_buffer == ""
        assert state.processor._turn_marker is None
    finally:
        state.loop.run_until_complete(state.processor.cleanup())
        asyncio.set_event_loop(None)
        state.loop.close()


def _benchmark_first_text(benchmark) -> str:
    """Measure queued response-start to first text for one markerless 64 KiB response."""
    return benchmark.pedantic(
        _queue_and_wait_for_first_text,
        setup=lambda: _setup(_LARGE_RESPONSE_BYTES),
        teardown=_teardown,
        rounds=3,
        warmup_rounds=1,
        iterations=1,
    )


def test_turn_completion_markerless_64k_queued_first_text(benchmark):
    """Guard queued FrameProcessor handling for a markerless 64 KiB stream."""
    assert _benchmark_first_text(benchmark)
