#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from typing import cast
from unittest.mock import patch

import numpy as np

import pipecat.audio.vad.silero as silero
from pipecat.audio.vad.silero import SileroVADAnalyzer


class _CapturingModel:
    def __init__(self, *_args, **_kwargs):
        self.inputs = []
        self.sample_rates = []
        self._result = np.array([0.25], dtype=np.float32)

    def __call__(self, audio_float32, sample_rate):
        self.inputs.append(audio_float32.copy())
        self.sample_rates.append(sample_rate)
        return self._result

    def reset_states(self):
        pass


class TestSileroVADConversion(unittest.TestCase):
    def _new_analyzer(self, sample_rate: int) -> tuple[SileroVADAnalyzer, _CapturingModel]:
        with patch.object(silero, "SileroOnnxModel", _CapturingModel):
            analyzer = SileroVADAnalyzer()

        analyzer.set_sample_rate(sample_rate)
        analyzer._last_reset_time = float("inf")
        return analyzer, cast(_CapturingModel, analyzer._model)

    def test_full_frames_convert_for_both_supported_sample_rates(self):
        for sample_rate, frame_samples in ((8000, 256), (16000, 512)):
            with self.subTest(sample_rate=sample_rate):
                analyzer, model = self._new_analyzer(sample_rate)
                first = np.arange(frame_samples, dtype=np.int16) - frame_samples // 2
                second = -first

                analyzer.voice_confidence(first.tobytes())
                analyzer.voice_confidence(second.tobytes())

                self.assertEqual(model.sample_rates, [sample_rate, sample_rate])
                np.testing.assert_array_equal(
                    model.inputs[0], first.astype(np.float32) / np.float32(32768.0)
                )
                np.testing.assert_array_equal(
                    model.inputs[1], second.astype(np.float32) / np.float32(32768.0)
                )

    def test_short_buffer_conversion_is_preserved(self):
        analyzer, model = self._new_analyzer(16000)
        samples = np.array([0, 16384, -32768, 32767], dtype=np.int16)

        analyzer.voice_confidence(samples.tobytes())

        self.assertEqual(model.sample_rates, [16000])
        np.testing.assert_array_equal(
            model.inputs[0], samples.astype(np.float32) / np.float32(32768.0)
        )


if __name__ == "__main__":
    unittest.main()
