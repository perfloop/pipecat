#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Emit one real SileroVADAnalyzer sample in Perfloop JSONL format."""

import argparse
import json
import math
import time
import tracemalloc
from typing import Any

import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer

_ALLOCATION_CALLS = 16
_METRICS = ("ns/frame", "tracemalloc_peak_bytes/frame")
_TIMING_CALLS = 1024
_WARMUP_CALLS = 32


def _frames(frame_count: int) -> list[bytes]:
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


def _confidence(analyzer: SileroVADAnalyzer, frame: bytes) -> float:
    """Run and consume one confidence result so inference cannot be discarded."""
    confidence = float(np.asarray(analyzer.voice_confidence(frame)).reshape(-1)[0])
    if not math.isfinite(confidence):
        raise RuntimeError("Silero VAD returned a non-finite confidence")
    return confidence


def _run_calls(analyzer: SileroVADAnalyzer, frames: list[bytes], calls: int) -> float:
    """Run a rotating frame sequence and retain a scalar sink of all results."""
    confidence_sum = 0.0
    for index in range(calls):
        confidence_sum += _confidence(analyzer, frames[index % len(frames)])
    if not math.isfinite(confidence_sum):
        raise RuntimeError("Silero VAD confidence sink is non-finite")
    return confidence_sum


def _assert_inference_state(
    analyzer: SileroVADAnalyzer, sample_rate: int, frame_count: int
) -> None:
    """Validate the real model's recurrent state after the measured calls."""
    model: Any = analyzer._model
    state = np.asarray(model._state)
    context = np.asarray(model._context)
    context_size = 64 if sample_rate == 16000 else 32
    if state.shape != (2, 1, 128):
        raise RuntimeError(f"unexpected Silero state shape: {state.shape}")
    if context.shape != (1, context_size):
        raise RuntimeError(f"unexpected Silero context shape: {context.shape}")
    if model._last_sr != sample_rate or model._last_batch_size != 1:
        raise RuntimeError("Silero recurrent metadata did not match the measured stream")
    if frame_count != analyzer.num_frames_required():
        raise RuntimeError("Silero analyzer did not retain the selected frame shape")
    if not bool(np.all(np.isfinite(np.asarray(state)))):
        raise RuntimeError("Silero recurrent state became non-finite")
    if not bool(np.all(np.isfinite(np.asarray(context)))):
        raise RuntimeError("Silero recurrent context became non-finite")


def _timed_ns_per_frame(analyzer: SileroVADAnalyzer, frames: list[bytes]) -> float:
    """Measure the warmed real owner over complete frames."""
    start = time.perf_counter_ns()
    confidence_sum = _run_calls(analyzer, frames, _TIMING_CALLS)
    elapsed = time.perf_counter_ns() - start
    if confidence_sum < 0.0:
        raise RuntimeError("unreachable confidence sink guard")
    return elapsed / _TIMING_CALLS


def _peak_tracemalloc_bytes_per_frame(analyzer: SileroVADAnalyzer, frames: list[bytes]) -> int:
    """Measure the maximum Python-tracked transient allocation for one warmed call."""
    maximum_transient = 0
    confidence_sum = 0.0
    tracemalloc.start()
    try:
        for index in range(_ALLOCATION_CALLS):
            tracemalloc.reset_peak()
            before, _ = tracemalloc.get_traced_memory()
            confidence_sum += _confidence(analyzer, frames[index % len(frames)])
            _, peak = tracemalloc.get_traced_memory()
            maximum_transient = max(maximum_transient, peak - before)
    finally:
        tracemalloc.stop()
    if not math.isfinite(confidence_sum):
        raise RuntimeError("Silero VAD allocation sink is non-finite")
    return maximum_transient


def _emit(metric: str, value: float | int) -> None:
    """Write one evidence row for a metric."""
    print(json.dumps({"metric": metric, "value": value}, separators=(",", ":")))


def main() -> None:
    """Run one warmed analyzer sample at one supported frame shape."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", choices=(8000, 16000), required=True, type=int)
    parser.add_argument("--metric", choices=_METRICS, required=True)
    args = parser.parse_args()

    analyzer = SileroVADAnalyzer()
    analyzer.set_sample_rate(args.sample_rate)
    frame_count = analyzer.num_frames_required()
    frames = _frames(frame_count)

    _run_calls(analyzer, frames, _WARMUP_CALLS)
    analyzer._last_reset_time = time.time()
    ns_per_frame = _timed_ns_per_frame(analyzer, frames)
    peak_bytes = _peak_tracemalloc_bytes_per_frame(analyzer, frames)
    _assert_inference_state(analyzer, args.sample_rate, frame_count)

    if args.metric == "ns/frame":
        _emit("ns/frame", ns_per_frame)
        _emit("tracemalloc_peak_bytes/frame", peak_bytes)
    else:
        _emit("tracemalloc_peak_bytes/frame", peak_bytes)
        _emit("ns/frame", ns_per_frame)


if __name__ == "__main__":
    main()
