#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Profile allocation volume at the Silero PCM conversion boundary.

The ONNX model is replaced after analyzer setup so the profile isolates
int16-to-float32 preparation. The model double consumes every converted sample
and temporarily retains the array references during the traced batch. Retention
is measurement instrumentation, not a model of production lifetime: it makes
one conversion allocation per call observable in a tracemalloc snapshot. The
model reads each value before the next call can reuse the destination buffer.
"""

import argparse
import json
import time
import tracemalloc

import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer

FRAME_SAMPLES = 512
SAMPLE_RATE = 16000
WARMUP_CALLS = 64
MIN_PROFILE_CALLS = 2048
PROFILE_CALL_SPAN = 128
SELECTOR = "silero_vad_conversion/16khz-512-2048to2175frames"
METRIC = "tracemalloc_retained_bytes_per_batch"


class _RecordingModel:
    """A synchronous model double that consumes and records converted frames."""

    def __init__(self):
        """Initialize reusable output storage and instrumentation state."""
        self._result = np.zeros(1, dtype=np.float32)
        self.inputs: list[np.ndarray] = []
        self.calls = 0
        self.checksum = 0.0
        self.validated_input = False

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Consume the current frame before retaining its identity for profiling."""
        if not self.validated_input:
            if audio.dtype != np.float32:
                raise AssertionError(f"expected float32 model input, got {audio.dtype}")
            if audio.shape != (FRAME_SAMPLES,):
                raise AssertionError(f"expected {FRAME_SAMPLES} samples, got {audio.shape}")
            if sample_rate != SAMPLE_RATE:
                raise AssertionError(f"expected {SAMPLE_RATE} Hz, got {sample_rate} Hz")
            self.validated_input = True

        value = float(audio[self.calls & (FRAME_SAMPLES - 1)])
        self.checksum += value
        self.inputs.append(audio)
        self._result[0] = value
        self.calls += 1
        return self._result

    def reset_states(self):
        """Match the reset hook used by SileroVADAnalyzer."""


def _runtime_varied_frames() -> tuple[bytes, ...]:
    """Create nonconstant PCM frames before profiling begins."""
    seed = time.monotonic_ns() & 0xFFFF
    offsets = (0, 7919, 16127, 28657)
    frames = []
    for offset in offsets:
        values = (np.arange(FRAME_SAMPLES, dtype=np.int32) + seed + offset) % 65536 - 32768
        frames.append(values.astype(np.int16).tobytes())
    return tuple(frames)


def _profile_call_count() -> int:
    """Choose a runtime-varying repeated-frame batch length for one sample."""
    return MIN_PROFILE_CALLS + (time.monotonic_ns() % PROFILE_CALL_SPAN)


def _make_analyzer() -> tuple[SileroVADAnalyzer, _RecordingModel]:
    """Set up the real analyzer lifecycle while isolating conversion allocations."""
    analyzer = SileroVADAnalyzer(sample_rate=SAMPLE_RATE)
    analyzer.set_sample_rate(SAMPLE_RATE)
    model = _RecordingModel()
    analyzer._model = model
    analyzer._last_reset_time = time.time()
    return analyzer, model


def _consume_frames(analyzer: SileroVADAnalyzer, frames: tuple[bytes, ...], calls: int) -> float:
    """Run a runtime-varied frame sequence and retain a data-dependent result sink."""
    result_sink = 0.0
    for index in range(calls):
        result_sink += float(analyzer.voice_confidence(frames[index % len(frames)]))
    return result_sink


def measure_retained_conversion_bytes() -> tuple[int, int, int]:
    """Measure one repeated-frame allocation batch through real voice_confidence."""
    analyzer, model = _make_analyzer()
    frames = _runtime_varied_frames()
    _consume_frames(analyzer, frames, WARMUP_CALLS)
    model.inputs.clear()

    profile_calls = _profile_call_count()
    tracemalloc.start()
    try:
        before_current, _ = tracemalloc.get_traced_memory()
        result_sink = _consume_frames(analyzer, frames, profile_calls)
        after_current, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    if model.calls != WARMUP_CALLS + profile_calls:
        raise AssertionError(
            f"model consumed {model.calls} frames instead of {WARMUP_CALLS + profile_calls}"
        )
    if len(model.inputs) != profile_calls:
        raise AssertionError(
            f"model retained {len(model.inputs)} frames instead of {profile_calls}"
        )
    if not model.validated_input or not np.isfinite(result_sink) or not np.isfinite(model.checksum):
        raise AssertionError("model result was not consumed as a finite value")
    if after_current < before_current:
        raise AssertionError(
            "tracemalloc current allocation fell during the retained profile batch"
        )

    unique_inputs = len({id(audio) for audio in model.inputs})
    return after_current - before_current, profile_calls, unique_inputs


def verify_allocation_profile() -> None:
    """Verify that tracemalloc profiles a real Silero conversion batch."""
    retained_bytes, profile_calls, unique_inputs = measure_retained_conversion_bytes()
    if retained_bytes <= 0:
        raise AssertionError("tracemalloc did not observe retained conversion allocations")
    if unique_inputs <= 0:
        raise AssertionError("model did not observe a conversion input")
    print(
        "allocation-profile probe passed: "
        f"real_voice_confidence_retained_bytes={retained_bytes} "
        f"profile_calls={profile_calls} unique_inputs={unique_inputs}"
    )


def main() -> None:
    """Emit one proof JSONL allocation sample or run the real-path profile check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-allocation-profile", action="store_true")
    args = parser.parse_args()

    if args.verify_allocation_profile:
        verify_allocation_profile()
        return

    value, _, _ = measure_retained_conversion_bytes()
    print(json.dumps({"selector": SELECTOR, "metric": METRIC, "value": value}))


if __name__ == "__main__":
    main()
