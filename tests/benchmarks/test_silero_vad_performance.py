#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Native pytest-benchmark coverage for the real stateful Silero VAD path."""

import asyncio
import math
import random
from typing import Any

import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.audio.vad.vad_controller import VADController
from pipecat.frames.frames import InputAudioRawFrame, StartFrame


def _audio_buffers(frame_count: int) -> list[bytes]:
    """Build complete, varied PCM frames outside the benchmark operation."""
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


def _confidence(result: Any) -> float:
    """Consume the model result returned by a real analyzer call."""
    confidence = float(np.asarray(result).reshape(-1)[0])
    if not math.isfinite(confidence):
        raise RuntimeError("Silero VAD returned a non-finite confidence")
    return confidence


def _new_analyzer(sample_rate: int) -> SileroVADAnalyzer:
    """Create a warmed analyzer with the periodic reset outside the steady-state run."""
    analyzer = SileroVADAnalyzer()
    analyzer.set_sample_rate(sample_rate)
    analyzer._last_reset_time = float("inf")
    for audio in _audio_buffers(analyzer.num_frames_required()):
        _confidence(analyzer.voice_confidence(audio))
    return analyzer


def _assert_inference_state(analyzer: SileroVADAnalyzer, sample_rate: int) -> None:
    """Assert that the benchmark consumed real recurrent inference results."""
    model: Any = analyzer._model
    state = np.asarray(model._state)
    context = np.asarray(model._context)
    context_size = 64 if sample_rate == 16000 else 32
    if state.shape != (2, 1, 128):
        raise RuntimeError(f"unexpected Silero state shape: {state.shape}")
    if context.shape != (1, context_size):
        raise RuntimeError(f"unexpected Silero context shape: {context.shape}")
    if model._last_sr != sample_rate or model._last_batch_size != 1:
        raise RuntimeError("Silero recurrent metadata did not match the benchmark stream")
    if not bool(np.all(np.isfinite(np.asarray(state)))):
        raise RuntimeError("Silero recurrent state became non-finite")
    if not bool(np.all(np.isfinite(np.asarray(context)))):
        raise RuntimeError("Silero recurrent context became non-finite")


def _participant_rates(dominant_sample_rate: int) -> list[int]:
    """Create a heterogeneous two-rate participant mix for one native sample."""
    other_sample_rate = 16000 if dominant_sample_rate == 8000 else 8000
    chooser = random.SystemRandom()
    return [
        dominant_sample_rate,
        other_sample_rate,
        *chooser.choices([dominant_sample_rate, other_sample_rate], weights=[3, 1], k=29),
    ]


def _benchmark_mixed_rates(benchmark: Any, dominant_sample_rate: int) -> None:
    """Benchmark real owner calls across a changing multi-participant rate mix."""
    analyzers = {sample_rate: _new_analyzer(sample_rate) for sample_rate in (8000, 16000)}
    frames_by_rate = {
        sample_rate: _audio_buffers(analyzer.num_frames_required())
        for sample_rate, analyzer in analyzers.items()
    }
    participant_rates = _participant_rates(dominant_sample_rate)
    frame_indices = {8000: 0, 16000: 0}
    frame_index = 0

    def operation() -> float:
        nonlocal frame_index
        sample_rate = participant_rates[frame_index % len(participant_rates)]
        frames = frames_by_rate[sample_rate]
        frame = frames[frame_indices[sample_rate] % len(frames)]
        frame_indices[sample_rate] += 1
        frame_index += 1
        return _confidence(analyzers[sample_rate].voice_confidence(frame))

    confidence = benchmark(operation)
    if not math.isfinite(confidence):
        raise RuntimeError("pytest-benchmark did not retain the confidence result")
    for sample_rate, analyzer in analyzers.items():
        _assert_inference_state(analyzer, sample_rate)


def test_silero_voice_confidence_8khz_dominant(benchmark: Any) -> None:
    """Benchmark real owner calls for an 8 kHz-dominant participant mix."""
    _benchmark_mixed_rates(benchmark, 8000)


def test_silero_voice_confidence_16khz_dominant(benchmark: Any) -> None:
    """Benchmark real owner calls for a 16 kHz-dominant participant mix."""
    _benchmark_mixed_rates(benchmark, 16000)


def test_vad_controller_process_frame_16khz_512(benchmark: Any) -> None:
    """Benchmark the real VADController operation enclosing the 16 kHz owner."""
    loop = asyncio.new_event_loop()
    analyzer = _new_analyzer(16000)
    controller = VADController(analyzer, audio_idle_timeout=0)
    frames = [
        InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)
        for audio in _audio_buffers(analyzer.num_frames_required())
    ]
    frame_index = 0

    try:
        loop.run_until_complete(
            controller.process_frame(
                StartFrame(audio_in_sample_rate=16000, audio_out_sample_rate=16000)
            )
        )

        def operation() -> VADState:
            nonlocal frame_index
            loop.run_until_complete(controller.process_frame(frames[frame_index % len(frames)]))
            frame_index += 1
            return controller._vad_state

        state = benchmark(operation)
        if not isinstance(state, VADState):
            raise RuntimeError("pytest-benchmark did not retain the VAD controller state")
        if analyzer._vad_buffer:
            raise RuntimeError("VADController left a complete frame buffered")
        _assert_inference_state(analyzer, 16000)
    finally:
        loop.run_until_complete(controller.cleanup())
        loop.close()
