#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Emit a real multi-rate Silero VAD allocation sample in Perfloop JSONL."""

import argparse
import inspect
import json
import math
import random
import sys
import tracemalloc
from typing import Any

import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer

_ALLOCATION_CALLS = 31
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


def _new_warmed_analyzer(sample_rate: int) -> tuple[SileroVADAnalyzer, list[bytes]]:
    """Create one steady-state real analyzer for one supported participant rate."""
    analyzer = SileroVADAnalyzer()
    analyzer.set_sample_rate(sample_rate)
    frames = _frames(analyzer.num_frames_required())
    _run_calls(analyzer, frames, _WARMUP_CALLS)
    analyzer._last_reset_time = float("inf")
    return analyzer, frames


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


def _participant_rates(dominant_sample_rate: int) -> list[int]:
    """Create a heterogeneous, two-rate participant workload for one sample."""
    other_sample_rate = 16000 if dominant_sample_rate == 8000 else 8000
    chooser = random.SystemRandom()
    return [
        dominant_sample_rate,
        other_sample_rate,
        *chooser.choices(
            [dominant_sample_rate, other_sample_rate],
            weights=[3, 1],
            k=_ALLOCATION_CALLS - 2,
        ),
    ]


def _next_frame(
    frames_by_rate: dict[int, list[bytes]], frame_indices: dict[int, int], sample_rate: int
) -> bytes:
    """Choose a varied complete frame for one participant's fixed-rate stream."""
    frames = frames_by_rate[sample_rate]
    frame = frames[frame_indices[sample_rate] % len(frames)]
    frame_indices[sample_rate] += 1
    return frame


def _conversion_line() -> int:
    """Locate the source boundary immediately before Silero model invocation."""
    lines, first_line = inspect.getsourcelines(SileroVADAnalyzer.voice_confidence)
    for offset, line in enumerate(lines):
        if "new_confidence = self._model(" in line:
            return first_line + offset
    raise RuntimeError("unable to locate the Silero model-call boundary")


def _mixed_peak_tracemalloc_bytes_per_frame(
    analyzers: dict[int, SileroVADAnalyzer],
    frames_by_rate: dict[int, list[bytes]],
    participant_rates: list[int],
) -> float:
    """Average Python-tracked transient storage over isolated real frame calls."""
    frame_indices = {8000: 0, 16000: 0}
    transient_bytes = 0
    confidence_sum = 0.0
    tracemalloc.start()
    try:
        for sample_rate in participant_rates:
            tracemalloc.reset_peak()
            before, _ = tracemalloc.get_traced_memory()
            confidence_sum += _confidence(
                analyzers[sample_rate], _next_frame(frames_by_rate, frame_indices, sample_rate)
            )
            _, peak = tracemalloc.get_traced_memory()
            transient_bytes += peak - before
    finally:
        tracemalloc.stop()
    if not math.isfinite(confidence_sum):
        raise RuntimeError("Silero VAD allocation sink is non-finite")
    return transient_bytes / len(participant_rates)


def _mixed_conversion_tracemalloc_bytes_per_frame(
    analyzers: dict[int, SileroVADAnalyzer],
    frames_by_rate: dict[int, list[bytes]],
    participant_rates: list[int],
) -> float:
    """Average conversion storage at the real model boundary across participants."""
    target_code = SileroVADAnalyzer.voice_confidence.__code__
    target_line = _conversion_line()
    frame_indices = {8000: 0, 16000: 0}
    conversion_bytes = 0
    confidence_sum = 0.0

    for sample_rate in participant_rates:
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
            confidence_sum += _confidence(
                analyzers[sample_rate], _next_frame(frames_by_rate, frame_indices, sample_rate)
            )
            if not captured_boundary[0]:
                raise RuntimeError("tracemalloc did not observe the Silero conversion boundary")
            conversion_bytes += captured_bytes[0] - before
        finally:
            tracemalloc.stop()
            sys.settrace(previous_trace)

    if not math.isfinite(confidence_sum):
        raise RuntimeError("Silero VAD conversion-allocation sink is non-finite")
    return conversion_bytes / len(participant_rates)


def _emit(metric: str, value: float) -> None:
    """Write one evidence row for a metric."""
    print(json.dumps({"metric": metric, "value": value}, separators=(",", ":")))


def main() -> None:
    """Run one heterogeneous two-rate real VAD allocation sample."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dominant-sample-rate", choices=(8000, 16000), required=True, type=int)
    parser.add_argument("--metric", choices=_METRICS, required=True)
    args = parser.parse_args()

    analyzers_and_frames = {
        sample_rate: _new_warmed_analyzer(sample_rate) for sample_rate in (8000, 16000)
    }
    analyzers = {sample_rate: item[0] for sample_rate, item in analyzers_and_frames.items()}
    frames_by_rate = {sample_rate: item[1] for sample_rate, item in analyzers_and_frames.items()}
    participant_rates = _participant_rates(args.dominant_sample_rate)

    peak_bytes = _mixed_peak_tracemalloc_bytes_per_frame(
        analyzers, frames_by_rate, participant_rates
    )
    conversion_bytes = _mixed_conversion_tracemalloc_bytes_per_frame(
        analyzers, frames_by_rate, participant_rates
    )
    for sample_rate, analyzer in analyzers.items():
        _assert_inference_state(analyzer, sample_rate, len(frames_by_rate[sample_rate][0]) // 2)

    values = {
        "tracemalloc_peak_bytes/frame": peak_bytes,
        "tracemalloc_conversion_bytes/frame": conversion_bytes,
        "dominant_rate_share": participant_rates.count(args.dominant_sample_rate)
        / len(participant_rates),
    }
    _emit(args.metric, values[args.metric])
    for metric, value in values.items():
        if metric != args.metric:
            _emit(metric, value)


if __name__ == "__main__":
    main()
