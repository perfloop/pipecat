#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Emit one real Silero voice_confidence measurement as proof JSONL."""

import argparse
import gc
import json
import math
import threading
import time
import tracemalloc

import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer

_WARMUP_CALLS = 32
_TIMED_CALLS = 1024
_ALLOCATION_CALLS = 16
_METRICS = ("ns/frame", "tracemalloc_peak_bytes/frame", "retained_input_bytes")


def _frame_size(sample_rate: int) -> int:
    return 512 if sample_rate == 16000 else 256


def _frames(sample_rate: int) -> list[bytes]:
    """Build runtime audio inputs with distinct but complete PCM frame contents."""
    samples = _frame_size(sample_rate)
    index = np.arange(samples, dtype=np.float32)
    silence = np.zeros(samples, dtype=np.int16)
    speech_like = np.rint(
        0.55 * 32767 * np.sin(2 * np.pi * 220 * index / sample_rate)
        + 0.20 * 32767 * np.sin(2 * np.pi * 440 * index / sample_rate)
    ).astype(np.int16)
    varied = ((np.arange(samples, dtype=np.int32) * 1103 + 997) % 65536 - 32768).astype(np.int16)
    alternating = np.where(np.arange(samples) % 2, 24000, -24000).astype(np.int16)
    return [frame.tobytes() for frame in (silence, speech_like, varied, alternating)]


def _confidence(analyzer: SileroVADAnalyzer, frame: bytes) -> float:
    value = float(np.ravel(analyzer.voice_confidence(frame))[0])
    if not math.isfinite(value):
        raise RuntimeError("Silero VAD returned a non-finite confidence")
    return value


def _assert_model_state(analyzer: SileroVADAnalyzer, sample_rate: int):
    """Ensure the measured calls executed the real stateful ONNX path."""
    model = analyzer._model
    state = np.asarray(model._state)
    context = np.asarray(model._context)
    context_size = 64 if sample_rate == 16000 else 32
    if model._last_sr != sample_rate or model._last_batch_size != 1:
        raise RuntimeError("Silero VAD did not retain the measured sample-rate and batch state")
    if state.shape != (2, 1, 128) or context.shape != (1, context_size):
        raise RuntimeError("Silero VAD recurrent state has an unexpected shape")
    if not np.isfinite(state).all() or not np.isfinite(context).all():
        raise RuntimeError("Silero VAD recurrent state contains non-finite values")


def _new_warm_analyzer(sample_rate: int, frames: list[bytes]) -> SileroVADAnalyzer:
    analyzer = SileroVADAnalyzer()
    analyzer.set_sample_rate(sample_rate)
    sink = 0.0
    for index in range(_WARMUP_CALLS):
        sink += _confidence(analyzer, frames[index % len(frames)])
    if not math.isfinite(sink):
        raise RuntimeError("Silero VAD warmup result was not consumed")
    _assert_model_state(analyzer, sample_rate)
    return analyzer


def _measure_ns_per_frame(
    analyzer: SileroVADAnalyzer, sample_rate: int, frames: list[bytes]
) -> float:
    sink = 0.0
    start = time.perf_counter_ns()
    for index in range(_TIMED_CALLS):
        sink += _confidence(analyzer, frames[index % len(frames)])
    elapsed = time.perf_counter_ns() - start
    if not math.isfinite(sink):
        raise RuntimeError("Silero VAD timing result was not consumed")
    _assert_model_state(analyzer, sample_rate)
    return elapsed / _TIMED_CALLS


def _measure_transient_peak_bytes(
    analyzer: SileroVADAnalyzer, sample_rate: int, frames: list[bytes]
) -> int:
    """Measure the maximum per-call tracemalloc peak after stateful warmup."""
    if threading.active_count() != 1:
        raise RuntimeError("tracemalloc allocation sampling requires one Python thread")

    gc.collect()
    tracemalloc.stop()
    tracemalloc.start()
    peaks: list[int] = []
    sink = 0.0
    try:
        for index in range(_ALLOCATION_CALLS):
            before_current, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            sink += _confidence(analyzer, frames[index % len(frames)])
            _, peak = tracemalloc.get_traced_memory()
            peaks.append(max(0, peak - before_current))
    finally:
        tracemalloc.stop()

    if not math.isfinite(sink):
        raise RuntimeError("Silero VAD allocation result was not consumed")
    _assert_model_state(analyzer, sample_rate)
    return max(peaks)


def _retained_input_bytes(analyzer: SileroVADAnalyzer) -> int:
    """Report the analyzer-owned float32 input storage, if the implementation owns it."""
    reusable = getattr(analyzer, "_audio_float32", None)
    return int(reusable.nbytes) if isinstance(reusable, np.ndarray) else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", type=int, choices=(8000, 16000), required=True)
    parser.add_argument("--metric", choices=_METRICS, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    frames = _frames(args.sample_rate)
    analyzer = _new_warm_analyzer(args.sample_rate, frames)

    if args.metric == "ns/frame":
        value = _measure_ns_per_frame(analyzer, args.sample_rate, frames)
    elif args.metric == "tracemalloc_peak_bytes/frame":
        value = _measure_transient_peak_bytes(analyzer, args.sample_rate, frames)
    else:
        value = _retained_input_bytes(analyzer)

    print(json.dumps({"metric": args.metric, "value": value}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
