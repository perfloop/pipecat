#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Emit one real SileroVADAnalyzer sample in Perfloop JSONL format."""

import argparse
import inspect
import json
import math
import sys
import tracemalloc
from typing import Any

import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer

_ALLOCATION_CALLS = 16
_METRICS = (
    "tracemalloc_peak_bytes/frame",
    "tracemalloc_conversion_bytes/frame",
)
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


def _conversion_line() -> int:
    """Locate the source boundary immediately before Silero model invocation."""
    lines, first_line = inspect.getsourcelines(SileroVADAnalyzer.voice_confidence)
    for offset, line in enumerate(lines):
        if "new_confidence = self._model(" in line:
            return first_line + offset
    raise RuntimeError("unable to locate the Silero model-call boundary")


def _conversion_tracemalloc_bytes_per_frame(
    analyzer: SileroVADAnalyzer, frames: list[bytes]
) -> int:
    """Measure Python-tracked conversion storage before real inference begins."""
    target_code = SileroVADAnalyzer.voice_confidence.__code__
    target_line = _conversion_line()
    maximum_transient = 0
    confidence_sum = 0.0

    for index in range(_ALLOCATION_CALLS):
        captured_bytes = [0]
        captured_boundary = [False]

        def trace(frame: Any, event: str, _argument: Any) -> Any:
            if event == "line" and frame.f_code is target_code and frame.f_lineno == target_line:
                captured_bytes[0] = tracemalloc.get_traced_memory()[0]
                captured_boundary[0] = True
            return trace

        previous_trace = sys.gettrace()
        sys.settrace(trace)
        tracemalloc.start()
        try:
            before, _ = tracemalloc.get_traced_memory()
            confidence_sum += _confidence(analyzer, frames[index % len(frames)])
            if not captured_boundary[0]:
                raise RuntimeError("tracemalloc did not observe the Silero conversion boundary")
            maximum_transient = max(maximum_transient, captured_bytes[0] - before)
        finally:
            tracemalloc.stop()
            sys.settrace(previous_trace)

    if not math.isfinite(confidence_sum):
        raise RuntimeError("Silero VAD conversion-allocation sink is non-finite")
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
    analyzer._last_reset_time = float("inf")
    peak_bytes = _peak_tracemalloc_bytes_per_frame(analyzer, frames)
    conversion_bytes = _conversion_tracemalloc_bytes_per_frame(analyzer, frames)
    _assert_inference_state(analyzer, args.sample_rate, frame_count)

    values = {
        "tracemalloc_peak_bytes/frame": peak_bytes,
        "tracemalloc_conversion_bytes/frame": conversion_bytes,
    }
    _emit(args.metric, values[args.metric])
    for metric, value in values.items():
        if metric != args.metric:
            _emit(metric, value)


if __name__ == "__main__":
    main()
