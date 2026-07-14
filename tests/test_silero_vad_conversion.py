#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Behavioral tests for Silero VAD's PCM-to-float conversion boundary."""

import unittest

import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer


class _CapturingModel:
    """Capture model inputs while preserving the analyzer model interface."""

    def __init__(self):
        """Initialize the captured-input collection."""
        self.inputs: list[tuple[np.ndarray, int]] = []

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Record the values consumed by the model and return a valid confidence array."""
        self.inputs.append((audio.copy(), sample_rate))
        return np.array([audio[0]], dtype=np.float32)

    def reset_states(self):
        """Match the reset hook used by SileroVADAnalyzer."""


class TestSileroVADConversion(unittest.TestCase):
    """Verify conversion behavior across consecutive 8 kHz VAD frames."""

    def test_voice_confidence_converts_each_8khz_frame_before_model_consumption(self):
        """Each 8 kHz frame must reach the model with its own normalized samples."""
        analyzer = SileroVADAnalyzer(sample_rate=8000)
        analyzer.set_sample_rate(8000)
        model = _CapturingModel()
        analyzer._model = model

        first_samples = np.arange(-128, 128, dtype=np.int16)
        second_samples = first_samples[::-1].copy()

        first_confidence = analyzer.voice_confidence(first_samples.tobytes())
        second_confidence = analyzer.voice_confidence(second_samples.tobytes())

        expected_first = first_samples.astype(np.float32) / np.float32(32768.0)
        expected_second = second_samples.astype(np.float32) / np.float32(32768.0)
        self.assertEqual(len(model.inputs), 2)
        self.assertEqual(model.inputs[0][1], 8000)
        self.assertEqual(model.inputs[1][1], 8000)
        np.testing.assert_array_equal(model.inputs[0][0], expected_first)
        np.testing.assert_array_equal(model.inputs[1][0], expected_second)
        self.assertEqual(float(first_confidence), float(expected_first[0]))
        self.assertEqual(float(second_confidence), float(expected_second[0]))
