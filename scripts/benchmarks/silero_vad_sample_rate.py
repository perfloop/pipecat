#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Emit one configured-Silero sample-rate transition measurement as proof JSONL."""

import argparse
import gc
import json
import math
import statistics
import threading
import time
import tracemalloc
from pathlib import Path

import numpy as np
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer

_WARMUP_TRANSITIONS = 32
_TIMED_TRANSITIONS = 64
_TIMING_TRIALS = 17
_METRICS = ("allocs/op", "setup_ns/op", "retained_B/analyzer")
_RATES = (8000, 16000)


def _assert_configured(analyzer: SileroVADAnalyzer, sample_rate: int):
    """Verify that the timed configuration transition completed."""
    if analyzer.sample_rate != sample_rate:
        raise RuntimeError("Silero VAD did not retain the measured sample rate")
    expected_frames = 256 if sample_rate == 8000 else 512
    if analyzer.num_frames_required() != expected_frames:
        raise RuntimeError("Silero VAD did not update its required frame size")


def _confidence(analyzer: SileroVADAnalyzer, frame: bytes) -> float:
    """Consume one real confidence result."""
    value = float(np.ravel(analyzer.voice_confidence(frame))[0])
    if not math.isfinite(value):
        raise RuntimeError("Silero VAD returned a non-finite confidence")
    return value


def _assert_inference_state(analyzer: SileroVADAnalyzer):
    """Ensure the transition benchmark called the real 16 kHz model path."""
    model = analyzer._model
    if model._last_sr != 16000 or model._last_batch_size != 1:
        raise RuntimeError("Silero VAD did not retain the transitioned model state")
    if np.asarray(model._context).shape != (1, 64) or not np.isfinite(model._state).all():
        raise RuntimeError("Silero VAD transitioned model state is invalid")


def _frame(sample_rate: int) -> bytes:
    """Build one complete deterministic PCM frame for a supported sample rate."""
    samples = 256 if sample_rate == 8000 else 512
    values = ((np.arange(samples, dtype=np.int32) * 1103 + 997) % 65536 - 32768).astype(np.int16)
    return values.tobytes()


def _measure_first_frame_allocs_per_op() -> float:
    """Count full-frame allocations in the first 16 kHz call after a rate switch."""
    if threading.active_count() != 1:
        raise RuntimeError("allocation sampling requires one Python thread")

    import memray

    analyzer = SileroVADAnalyzer()
    analyzer.set_sample_rate(8000)
    _confidence(analyzer, _frame(8000))
    analyzer.set_sample_rate(16000)

    trace_path = Path(".perfloop-silero-vad-transition-memray.bin")
    trace_path.unlink(missing_ok=True)
    try:
        with memray.Tracker(str(trace_path), trace_python_allocators=True):
            _confidence(analyzer, _frame(16000))

        allocations = 0
        for record in memray.FileReader(trace_path).get_allocation_records():
            if record.size != 512 * np.dtype(np.float32).itemsize:
                continue
            try:
                stack = record.stack_trace()
            except NotImplementedError:
                continue
            if any(
                function == "voice_confidence"
                and filename.endswith("src/pipecat/audio/vad/silero.py")
                for function, filename, _line in stack
            ):
                allocations += record.n_allocations
    finally:
        trace_path.unlink(missing_ok=True)

    _assert_inference_state(analyzer)
    return float(allocations)


def _measure_setup_ns_per_op() -> float:
    """Measure steady alternating supported-rate configuration transitions."""
    analyzer = SileroVADAnalyzer()
    for index in range(_WARMUP_TRANSITIONS):
        analyzer.set_sample_rate(_RATES[index % len(_RATES)])

    samples: list[float] = []
    for trial in range(_TIMING_TRIALS):
        start = time.perf_counter_ns()
        for index in range(_TIMED_TRANSITIONS):
            analyzer.set_sample_rate(_RATES[(trial + index) % len(_RATES)])
        elapsed = time.perf_counter_ns() - start
        samples.append(elapsed / _TIMED_TRANSITIONS)

    final_rate = _RATES[(_TIMING_TRIALS - 1 + _TIMED_TRANSITIONS - 1) % len(_RATES)]
    _assert_configured(analyzer, final_rate)
    return statistics.median(samples)


def _measure_retained_bytes_per_analyzer() -> int:
    """Measure live Python-tracked bytes retained after initial 16 kHz setup."""
    if threading.active_count() != 1:
        raise RuntimeError("retained-memory sampling requires one Python thread")

    analyzer = SileroVADAnalyzer()
    gc.collect()
    tracemalloc.stop()
    tracemalloc.start()
    try:
        analyzer.set_sample_rate(16000)
        gc.collect()
        current, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    _assert_configured(analyzer, 16000)
    return current


def main() -> int:
    # set_sample_rate logs each parameter refresh; logging is not setup work.
    logger.disable("pipecat")
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=_METRICS, required=True)
    args = parser.parse_args()

    if args.metric == "allocs/op":
        value = _measure_first_frame_allocs_per_op()
    elif args.metric == "setup_ns/op":
        value = _measure_setup_ns_per_op()
    else:
        value = _measure_retained_bytes_per_analyzer()

    print(json.dumps({"metric": args.metric, "value": value}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
