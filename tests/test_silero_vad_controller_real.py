#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Integration coverage for the real Silero VAD controller path."""

import time
import unittest
from typing import Any

import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.audio.vad.vad_controller import VADController
from pipecat.frames.frames import InputAudioRawFrame, StartFrame

_SHAPES = ((8000, 256), (16000, 512))


def _audio_buffers(frame_count: int) -> list[bytes]:
    """Build complete deterministic PCM frames for the real controller path."""
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


class TestSileroVADControllerRealPath(unittest.IsolatedAsyncioTestCase):
    """Ensure VADController dispatch reaches the real stateful Silero analyzer."""

    async def test_process_frame_runs_complete_real_silero_windows(self) -> None:
        for sample_rate, frame_count in _SHAPES:
            with self.subTest(sample_rate=sample_rate):
                analyzer = SileroVADAnalyzer()
                controller = VADController(analyzer, audio_idle_timeout=0)
                try:
                    await controller.process_frame(
                        StartFrame(
                            audio_in_sample_rate=sample_rate,
                            audio_out_sample_rate=sample_rate,
                        )
                    )
                    analyzer._last_reset_time = time.time()

                    for audio in _audio_buffers(frame_count):
                        await controller.process_frame(
                            InputAudioRawFrame(
                                audio=audio,
                                sample_rate=sample_rate,
                                num_channels=1,
                            )
                        )

                    model: Any = analyzer._model
                    state = np.asarray(model._state)
                    context = np.asarray(model._context)
                    context_size = 64 if sample_rate == 16000 else 32
                    self.assertEqual(analyzer._vad_buffer, b"")
                    self.assertIsInstance(controller._vad_state, VADState)
                    self.assertEqual(state.shape, (2, 1, 128))
                    self.assertEqual(context.shape, (1, context_size))
                    self.assertEqual(model._last_sr, sample_rate)
                    self.assertEqual(model._last_batch_size, 1)
                    self.assertTrue(bool(np.all(np.isfinite(np.asarray(state)))))
                    self.assertTrue(bool(np.all(np.isfinite(np.asarray(context)))))
                finally:
                    await controller.cleanup()


if __name__ == "__main__":
    unittest.main()
