#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Emit one real VADController.process_frame sample in Perfloop JSONL format."""

import argparse
import asyncio
import json
import math
import time
from typing import Any

import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.audio.vad.vad_controller import VADController
from pipecat.frames.frames import InputAudioRawFrame, StartFrame

_METRIC = "ns/frame"
_TIMING_CALLS = 1024
_WARMUP_CALLS = 32


def _audio_buffers(frame_count: int) -> list[bytes]:
    """Build complete, varied PCM frames before measurement begins."""
    positions = np.arange(frame_count, dtype=np.float32)
    sine = np.rint(np.sin(positions * (2 * np.pi * 11 / frame_count)) * 18000).astype(np.int16)
    alternating = np.where(np.arange(frame_count) % 2, 12000, -12000).astype(np.int16)
    varied = ((np.arange(frame_count, dtype=np.int32) * 1103 + 97) % 65536 - 32768).astype(np.int16)
    return [
        np.zeros(frame_count, dtype=np.int16).tobytes(),
        sine.tobytes(),
        alternating.tobytes(),
        varied.tobytes(),
    ]


def _frames(sample_rate: int, frame_count: int) -> list[InputAudioRawFrame]:
    """Create immutable complete frames outside the timed controller calls."""
    return [
        InputAudioRawFrame(audio=audio, sample_rate=sample_rate, num_channels=1)
        for audio in _audio_buffers(frame_count)
    ]


async def _run_calls(
    controller: VADController, frames: list[InputAudioRawFrame], calls: int
) -> VADState:
    """Await complete frames through the real controller and retain its state."""
    for index in range(calls):
        await controller.process_frame(frames[index % len(frames)])
    return controller._vad_state


def _assert_inference_state(
    analyzer: SileroVADAnalyzer, state: VADState, sample_rate: int, frame_count: int
) -> None:
    """Validate that controller calls reached the real recurrent ONNX model."""
    model: Any = analyzer._model
    model_state = np.asarray(model._state)
    context = np.asarray(model._context)
    context_size = 64 if sample_rate == 16000 else 32
    if not isinstance(state, VADState):
        raise RuntimeError("VADController did not retain a VAD state")
    if analyzer._vad_buffer:
        raise RuntimeError("VADController left a complete frame buffered")
    if model_state.shape != (2, 1, 128):
        raise RuntimeError(f"unexpected Silero state shape: {model_state.shape}")
    if context.shape != (1, context_size):
        raise RuntimeError(f"unexpected Silero context shape: {context.shape}")
    if model._last_sr != sample_rate or model._last_batch_size != 1:
        raise RuntimeError("Silero recurrent metadata did not match the controller stream")
    if frame_count != analyzer.num_frames_required():
        raise RuntimeError("Silero analyzer did not retain the selected frame shape")
    if not bool(np.all(np.isfinite(np.asarray(model_state)))):
        raise RuntimeError("Silero recurrent state became non-finite")
    if not bool(np.all(np.isfinite(np.asarray(context)))):
        raise RuntimeError("Silero recurrent context became non-finite")


async def _main(sample_rate: int) -> None:
    """Warm and time steady controller dispatch over complete VAD windows."""
    analyzer = SileroVADAnalyzer()
    controller = VADController(analyzer, audio_idle_timeout=0)
    try:
        await controller.process_frame(
            StartFrame(audio_in_sample_rate=sample_rate, audio_out_sample_rate=sample_rate)
        )
        frame_count = analyzer.num_frames_required()
        frames = _frames(sample_rate, frame_count)

        await _run_calls(controller, frames, _WARMUP_CALLS)
        analyzer._last_reset_time = time.time()
        start = time.perf_counter_ns()
        state = await _run_calls(controller, frames, _TIMING_CALLS)
        elapsed = time.perf_counter_ns() - start
        _assert_inference_state(analyzer, state, sample_rate, frame_count)
        print(
            json.dumps({"metric": _METRIC, "value": elapsed / _TIMING_CALLS}, separators=(",", ":"))
        )
    finally:
        await controller.cleanup()


def main() -> None:
    """Parse the supported real-time VAD shape and emit one guard sample."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", choices=(8000, 16000), required=True, type=int)
    args = parser.parse_args()
    asyncio.run(_main(args.sample_rate))


if __name__ == "__main__":
    main()
