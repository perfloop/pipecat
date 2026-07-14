#!/usr/bin/env python3
"""Measure the steady-state Silero VAD int16-to-float32 conversion boundary."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
import tracemalloc
from collections.abc import Callable
from typing import cast
from unittest.mock import patch

import numpy as np
from loguru import logger

import pipecat.audio.vad.silero as silero

# Keep the benchmark's stdout to exactly its proof JSON object. Pipecat's normal
# startup logs remain useful in applications, but are not benchmark data.
logger.disable("pipecat")


METRIC = "silero_vad_conversion_ns_per_frame"
SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
INPUT_VARIANTS = 64
WARMUP_ITERATIONS = 10_000
SAMPLE_ITERATIONS = 1_000_000
PROFILE_ITERATIONS = 17


class _ProbeModel:
    """Consume the converted samples without including ONNX inference in timing."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self._calls = 0
        self._captured: np.ndarray | None = None
        self._result = np.zeros(1, dtype=np.float32)

    @property
    def calls(self) -> int:
        """Return the number of converted frames consumed."""
        return self._calls

    @property
    def captured(self) -> np.ndarray | None:
        """Return the first converted frame captured before measurement."""
        return self._captured

    def __call__(self, audio_float32: np.ndarray, sample_rate: int) -> np.ndarray:
        """Consume a runtime-varying sample and return a stable confidence array."""
        if (
            sample_rate != SAMPLE_RATE
            or audio_float32.dtype != np.float32
            or audio_float32.ndim != 1
            or audio_float32.size != FRAME_SAMPLES
        ):
            raise ValueError("unexpected Silero VAD conversion input")

        if self._captured is None:
            self._captured = audio_float32.copy()

        self._result[0] = audio_float32[self._calls % audio_float32.size]
        self._calls += 1
        return self._result

    def reset_states(self) -> None:
        """Match the model reset API without adding work to the measured boundary."""


def _input_buffers() -> list[bytes]:
    """Build varied complete VAD frames before the timed region."""
    base = np.arange(-256, 256, dtype=np.int32)
    return [
        ((base * 97 + variant * 251) % 65536 - 32768).astype(np.int16).tobytes()
        for variant in range(INPUT_VARIANTS)
    ]


def _new_analyzer() -> tuple[silero.SileroVADAnalyzer, _ProbeModel]:
    """Create a VAD analyzer whose model validates and consumes converted input."""
    with patch.object(silero, "SileroOnnxModel", _ProbeModel):
        analyzer = silero.SileroVADAnalyzer()

    analyzer.set_sample_rate(SAMPLE_RATE)
    # The benchmark is explicitly steady state; periodic model reset is outside
    # the int16-to-float32 conversion boundary measured here.
    analyzer._last_reset_time = float("inf")
    return analyzer, cast(_ProbeModel, analyzer._model)


def _verify_conversion(
    analyzer: silero.SileroVADAnalyzer, model: _ProbeModel, buffer: bytes
) -> None:
    """Verify the measured path supplies the model normalized float32 samples."""
    analyzer.voice_confidence(buffer)
    expected = np.frombuffer(buffer, np.int16).astype(np.float32) / np.float32(32768.0)
    if model.captured is None:
        raise RuntimeError("the probe model did not consume a converted frame")
    np.testing.assert_array_equal(model.captured, expected)


def _timed_sample() -> float:
    """Return one GC-enabled, steady-state conversion timing sample in ns/frame."""
    analyzer, model = _new_analyzer()
    buffers = _input_buffers()
    _verify_conversion(analyzer, model, buffers[0])

    # Production runs with GC enabled. Collecting before warmup only removes
    # startup residue; no collection is disabled or forced while timing.
    gc.enable()
    gc.collect()
    for index in range(WARMUP_ITERATIONS):
        analyzer.voice_confidence(buffers[index % len(buffers)])

    sink = 0.0
    start = time.perf_counter_ns()
    for index in range(SAMPLE_ITERATIONS):
        sink += float(analyzer.voice_confidence(buffers[index % len(buffers)]))
    elapsed = time.perf_counter_ns() - start

    expected_calls = 1 + WARMUP_ITERATIONS + SAMPLE_ITERATIONS
    if model.calls != expected_calls or not math.isfinite(sink):
        raise RuntimeError("the benchmark did not consume every converted frame")

    return elapsed / SAMPLE_ITERATIONS


def _peak_temporary_bytes(operation: Callable[[int], float]) -> int:
    """Return median temporary traced allocation over repeated operations."""
    gc.collect()
    tracemalloc.start()
    peaks: list[int] = []
    sink = 0.0
    for index in range(PROFILE_ITERATIONS):
        tracemalloc.reset_peak()
        sink += operation(index)
        current, peak = tracemalloc.get_traced_memory()
        peaks.append(peak - current)
    tracemalloc.stop()

    if not math.isfinite(sink):
        raise RuntimeError("the allocation profile did not consume its result")
    return int(statistics.median(peaks))


def _profile() -> None:
    """Profile source conversion against naive and reusable-workspace controls."""
    analyzer, model = _new_analyzer()
    buffers = _input_buffers()
    _verify_conversion(analyzer, model, buffers[0])
    workspace = np.empty(FRAME_SAMPLES, dtype=np.float32)

    def source_operation(index: int) -> float:
        return float(analyzer.voice_confidence(buffers[index % len(buffers)]))

    def naive_control(index: int) -> float:
        audio_int16 = np.frombuffer(buffers[index % len(buffers)], np.int16)
        audio_float32 = audio_int16.astype(np.float32) / np.float32(32768.0)
        return float(audio_float32[index % audio_float32.size])

    def workspace_control(index: int) -> float:
        audio_int16 = np.frombuffer(buffers[index % len(buffers)], np.int16)
        np.copyto(workspace, audio_int16, casting="unsafe")
        np.divide(workspace, np.float32(32768.0), out=workspace)
        return float(workspace[index % workspace.size])

    source_peak = _peak_temporary_bytes(source_operation)
    naive_peak = _peak_temporary_bytes(naive_control)
    workspace_peak = _peak_temporary_bytes(workspace_control)

    if model.calls != 1 + PROFILE_ITERATIONS:
        raise RuntimeError("the source allocation profile did not reach the model")
    if naive_peak <= workspace_peak:
        raise RuntimeError("tracemalloc did not distinguish the allocation controls")
    if source_peak > naive_peak * 2:
        raise RuntimeError("the source profile exceeded its naive conversion control")

    print(
        "silero-vad-allocation-profile "
        f"source_peak_temporary_bytes={source_peak} "
        f"naive_control_peak_temporary_bytes={naive_peak} "
        f"workspace_control_peak_temporary_bytes={workspace_peak}"
    )


def main() -> None:
    """Run one proof sample or the controlled allocation-profile check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.profile:
        _profile()
        return

    print(json.dumps({"metric": METRIC, "value": _timed_sample()}))


if __name__ == "__main__":
    main()
