#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from decimal import Decimal
from fractions import Fraction
from unittest.mock import AsyncMock, patch

import numpy as np

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.frames.frames import (
    InputAudioRawFrame,
    SpeechControlParamsFrame,
    UserSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.tests.utils import run_test


class MockVADAnalyzer(VADAnalyzer):
    """A mock VAD analyzer that returns states from a predefined sequence."""

    def __init__(self, states: list[VADState]):
        super().__init__(sample_rate=16000)
        self._states = list(states)
        self._call_index = 0

    def num_frames_required(self) -> int:
        return 512

    def voice_confidence(self, buffer: bytes) -> float:
        return 0.9

    async def analyze_audio(self, buffer: bytes) -> VADState:
        if self._call_index < len(self._states):
            state = self._states[self._call_index]
            self._call_index += 1
            return state
        return VADState.QUIET


class TestVADProcessor(unittest.IsolatedAsyncioTestCase):
    def _make_audio_frame(self):
        return InputAudioRawFrame(audio=b"\x00" * 1024, sample_rate=16000, num_channels=1)

    def test_rejects_invalid_speech_activity_period_at_construction(self):
        """Test that invalid activity periods fail during processor construction."""
        for speech_activity_period in (
            "0.2",
            float("nan"),
            float("inf"),
            float("-inf"),
            Decimal("NaN"),
            Decimal("Infinity"),
            True,
        ):
            with self.subTest(speech_activity_period=speech_activity_period):
                with self.assertRaisesRegex(ValueError, "finite real number"):
                    VADProcessor(
                        vad_analyzer=MockVADAnalyzer([VADState.SPEAKING]),
                        speech_activity_period=speech_activity_period,
                    )

    def test_accepts_real_speech_activity_period_at_construction(self):
        """Test that the processor forwards supported real periods to its controller."""

        class IntSubclass(int):
            pass

        for speech_activity_period in (
            np.float64(0.2),
            np.float32(0.2),
            np.int64(1),
            Decimal("0.2"),
            Decimal("1e999999"),
            Fraction(1, 5),
            Fraction(10**1000, 3),
            IntSubclass(1),
        ):
            with self.subTest(speech_activity_period=speech_activity_period):
                VADProcessor(
                    vad_analyzer=MockVADAnalyzer([VADState.SPEAKING]),
                    speech_activity_period=speech_activity_period,
                )

    async def test_forwards_audio_frames(self):
        """Test that audio frames are forwarded downstream."""
        analyzer = MockVADAnalyzer([VADState.QUIET])
        processor = VADProcessor(vad_analyzer=analyzer)

        await run_test(
            processor,
            frames_to_send=[self._make_audio_frame()],
            expected_down_frames=[SpeechControlParamsFrame, InputAudioRawFrame],
        )

    async def test_default_period_throttles_user_speaking_frames(self):
        """Test that the default period emits one activity frame for repeated speech."""
        processor = VADProcessor(
            vad_analyzer=MockVADAnalyzer([VADState.SPEAKING, VADState.SPEAKING])
        )
        controller = processor._vad_controller

        with (
            patch.object(processor, "broadcast_frame", new_callable=AsyncMock) as broadcast_frame,
            patch(
                "pipecat.audio.vad.vad_controller.time.monotonic",
                side_effect=[0.01, 0.01, 0.11, 0.11],
            ),
        ):
            await controller.process_frame(self._make_audio_frame())
            await controller.process_frame(self._make_audio_frame())

        activity_calls = [
            call for call in broadcast_frame.await_args_list if call.args[0] is UserSpeakingFrame
        ]
        self.assertEqual(len(activity_calls), 1)

    async def test_pushes_started_speaking_frame(self):
        """Test that VADUserStartedSpeakingFrame is pushed when speech starts."""
        analyzer = MockVADAnalyzer([VADState.QUIET, VADState.SPEAKING])
        processor = VADProcessor(vad_analyzer=analyzer)

        # Audio frames are forwarded first, then VAD processes and broadcasts VAD frames
        await run_test(
            processor,
            frames_to_send=[self._make_audio_frame(), self._make_audio_frame()],
            expected_down_frames=[
                SpeechControlParamsFrame,
                InputAudioRawFrame,
                InputAudioRawFrame,
                VADUserStartedSpeakingFrame,
                UserSpeakingFrame,
            ],
        )

    async def test_pushes_stopped_speaking_frame(self):
        """Test that VADUserStoppedSpeakingFrame is pushed when speech stops."""
        analyzer = MockVADAnalyzer([VADState.SPEAKING, VADState.QUIET])
        processor = VADProcessor(vad_analyzer=analyzer)

        # Audio frames are forwarded first, then VAD processes and broadcasts VAD frames
        await run_test(
            processor,
            frames_to_send=[self._make_audio_frame(), self._make_audio_frame()],
            expected_down_frames=[
                SpeechControlParamsFrame,
                InputAudioRawFrame,
                VADUserStartedSpeakingFrame,
                UserSpeakingFrame,
                InputAudioRawFrame,
                VADUserStoppedSpeakingFrame,
            ],
        )

    async def test_pushes_one_user_speaking_frame_with_large_integer_period(self):
        """Test that a large integer period pushes only the first SPEAKING activity."""
        processor = VADProcessor(
            vad_analyzer=MockVADAnalyzer([VADState.SPEAKING, VADState.SPEAKING]),
            speech_activity_period=1 << 100000,
        )

        await run_test(
            processor,
            frames_to_send=[self._make_audio_frame(), self._make_audio_frame()],
            expected_down_frames=[
                SpeechControlParamsFrame,
                InputAudioRawFrame,
                VADUserStartedSpeakingFrame,
                UserSpeakingFrame,
                InputAudioRawFrame,
            ],
        )

    async def test_pushes_user_speaking_frame_for_non_positive_period(self):
        """Test that non-positive periods push UserSpeakingFrame while speaking."""
        for speech_activity_period in (0, -0.2):
            with self.subTest(speech_activity_period=speech_activity_period):
                analyzer = MockVADAnalyzer([VADState.SPEAKING, VADState.SPEAKING])
                processor = VADProcessor(
                    vad_analyzer=analyzer,
                    speech_activity_period=speech_activity_period,
                )

                # Audio frames are forwarded first, then VAD processes and broadcasts VAD frames.
                await run_test(
                    processor,
                    frames_to_send=[self._make_audio_frame(), self._make_audio_frame()],
                    expected_down_frames=[
                        SpeechControlParamsFrame,
                        InputAudioRawFrame,
                        VADUserStartedSpeakingFrame,
                        UserSpeakingFrame,
                        InputAudioRawFrame,
                        UserSpeakingFrame,
                    ],
                )

    async def test_no_vad_frames_on_starting_state(self):
        """Test that STARTING state doesn't push VAD frames."""
        analyzer = MockVADAnalyzer([VADState.STARTING])
        processor = VADProcessor(vad_analyzer=analyzer)

        await run_test(
            processor,
            frames_to_send=[self._make_audio_frame()],
            expected_down_frames=[SpeechControlParamsFrame, InputAudioRawFrame],
        )

    async def test_no_vad_frames_on_stopping_state(self):
        """Test that STOPPING state doesn't push VAD frames."""
        analyzer = MockVADAnalyzer([VADState.STOPPING])
        processor = VADProcessor(vad_analyzer=analyzer)

        await run_test(
            processor,
            frames_to_send=[self._make_audio_frame()],
            expected_down_frames=[SpeechControlParamsFrame, InputAudioRawFrame],
        )

    async def test_no_vad_frames_when_quiet(self):
        """Test that no VAD frames are pushed when staying quiet."""
        analyzer = MockVADAnalyzer([VADState.QUIET, VADState.QUIET])
        processor = VADProcessor(vad_analyzer=analyzer)

        await run_test(
            processor,
            frames_to_send=[self._make_audio_frame(), self._make_audio_frame()],
            expected_down_frames=[SpeechControlParamsFrame, InputAudioRawFrame, InputAudioRawFrame],
        )


if __name__ == "__main__":
    unittest.main()
