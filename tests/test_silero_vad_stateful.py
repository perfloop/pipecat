#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Stateful behavioral tests for the Silero ONNX VAD boundary."""

import gc
import unittest
import weakref
from typing import Any

import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer

_SHAPES = ((8000, 256), (16000, 512))


class _SynchronousSessionProxy:
    """Record the synchronous ONNX session boundary without changing inference."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self.last_input: Any = None
        self.run_async_calls = 0
        self.run_calls = 0

    def run(self, output_names: Any, input_feed: Any, *args: Any, **kwargs: Any) -> Any:
        self.run_calls += 1
        self.last_input = np.asarray(input_feed["input"])
        return self._session.run(output_names, input_feed, *args, **kwargs)

    def run_async(self, *args: Any, **kwargs: Any) -> Any:
        self.run_async_calls += 1
        raise AssertionError(
            "SileroOnnxModel must not use asynchronous inference for its audio input"
        )


class TestSileroVADStatefulBehavior(unittest.TestCase):
    """Exercise confidence conversion and the recurrent ONNX state at both rates."""

    @staticmethod
    def _new_analyzer(sample_rate: int) -> SileroVADAnalyzer:
        analyzer = SileroVADAnalyzer()
        analyzer.set_sample_rate(sample_rate)
        # Keep the time-based reset branch outside deterministic conversion sequences.
        analyzer._last_reset_time = float("inf")
        return analyzer

    @staticmethod
    def _frames(frame_count: int) -> list[bytes]:
        positions = np.arange(frame_count, dtype=np.float32)
        sine = np.rint(np.sin(positions * (2 * np.pi * 11 / frame_count)) * 18000).astype(np.int16)
        alternating = np.where(np.arange(frame_count) % 2, 12000, -12000).astype(np.int16)
        varied = ((np.arange(frame_count, dtype=np.int32) * 1103 + 97) % 65536 - 32768).astype(
            np.int16
        )
        return [
            np.zeros(frame_count, dtype=np.int16).tobytes(),
            sine.tobytes(),
            alternating.tobytes(),
            varied.tobytes(),
        ]

    @staticmethod
    def _confidence(result: Any) -> float:
        return float(np.asarray(result).reshape(-1)[0])

    @staticmethod
    def _reference_audio(buffer: bytes) -> np.ndarray:
        """Use a newly allocated reference conversion rather than analyzer storage."""
        audio = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
        np.divide(audio, np.float32(32768.0), out=audio)
        return audio

    def _reference_confidence(self, analyzer: SileroVADAnalyzer, buffer: bytes) -> float:
        audio = self._reference_audio(buffer)
        return self._confidence(analyzer._model(audio, analyzer.sample_rate))

    @staticmethod
    def _state_snapshot(analyzer: SileroVADAnalyzer) -> tuple[np.ndarray, np.ndarray, int, int]:
        model: Any = analyzer._model
        return (
            np.asarray(model._state).copy(),
            np.asarray(model._context).copy(),
            int(model._last_sr),
            int(model._last_batch_size),
        )

    def _assert_state_matches(
        self, actual: SileroVADAnalyzer, reference: SileroVADAnalyzer
    ) -> None:
        actual_state, actual_context, actual_sr, actual_batch = self._state_snapshot(actual)
        expected_state, expected_context, expected_sr, expected_batch = self._state_snapshot(
            reference
        )
        np.testing.assert_allclose(actual_state, expected_state, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(actual_context, expected_context, rtol=0.0, atol=1e-6)
        self.assertEqual(actual_sr, expected_sr)
        self.assertEqual(actual_batch, expected_batch)

    def _assert_inference_state(
        self, analyzer: SileroVADAnalyzer, sample_rate: int, frame_count: int
    ) -> None:
        model: Any = analyzer._model
        state = np.asarray(model._state)
        context = np.asarray(model._context)
        context_size = 64 if sample_rate == 16000 else 32
        self.assertEqual(state.shape, (2, 1, 128))
        self.assertEqual(context.shape, (1, context_size))
        self.assertEqual(model._last_sr, sample_rate)
        self.assertEqual(model._last_batch_size, 1)
        self.assertEqual(frame_count, analyzer.num_frames_required())
        self.assertTrue(bool(np.all(np.isfinite(np.asarray(state)))))
        self.assertTrue(bool(np.all(np.isfinite(np.asarray(context)))))

    def test_confidence_and_recurrent_state_match_reference_conversion(self) -> None:
        for sample_rate, frame_count in _SHAPES:
            with self.subTest(sample_rate=sample_rate):
                analyzer = self._new_analyzer(sample_rate)
                reference = self._new_analyzer(sample_rate)
                frames = self._frames(frame_count)

                # Include repeated, alternating, silence, and speech-like windows.
                for buffer in [*frames, frames[1], frames[3], frames[0], frames[2]]:
                    expected = self._reference_confidence(reference, buffer)
                    actual = self._confidence(analyzer.voice_confidence(buffer))
                    self.assertAlmostEqual(actual, expected, places=6)
                    self._assert_state_matches(analyzer, reference)
                    self._assert_inference_state(analyzer, sample_rate, frame_count)

    def test_invalid_sizes_return_zero_without_changing_model_state(self) -> None:
        for sample_rate, frame_count in _SHAPES:
            with self.subTest(sample_rate=sample_rate):
                analyzer = self._new_analyzer(sample_rate)
                reference = self._new_analyzer(sample_rate)
                valid, *rest = self._frames(frame_count)
                self.assertTrue(rest)

                expected = self._reference_confidence(reference, valid)
                actual = self._confidence(analyzer.voice_confidence(valid))
                self.assertAlmostEqual(actual, expected, places=6)
                self._assert_state_matches(analyzer, reference)
                before = self._state_snapshot(analyzer)

                for invalid in (valid[:-2], valid + b"\x00\x00"):
                    self.assertEqual(self._confidence(analyzer.voice_confidence(invalid)), 0.0)
                    after = self._state_snapshot(analyzer)
                    np.testing.assert_array_equal(after[0], before[0])
                    np.testing.assert_array_equal(after[1], before[1])
                    self.assertEqual(after[2:], before[2:])

                expected = self._reference_confidence(reference, valid)
                actual = self._confidence(analyzer.voice_confidence(valid))
                self.assertAlmostEqual(actual, expected, places=6)
                self._assert_state_matches(analyzer, reference)

    def test_reset_sample_rate_and_batch_transitions_match_reference(self) -> None:
        analyzer = self._new_analyzer(16000)
        reference = self._new_analyzer(16000)
        first_frame = self._frames(512)[1]

        self.assertAlmostEqual(
            self._confidence(analyzer.voice_confidence(first_frame)),
            self._reference_confidence(reference, first_frame),
            places=6,
        )
        analyzer._model.reset_states()
        reference._model.reset_states()
        self._assert_state_matches(analyzer, reference)

        analyzer.set_sample_rate(8000)
        reference.set_sample_rate(8000)
        second_frame = self._frames(256)[2]
        self.assertAlmostEqual(
            self._confidence(analyzer.voice_confidence(second_frame)),
            self._reference_confidence(reference, second_frame),
            places=6,
        )
        self._assert_state_matches(analyzer, reference)
        self._assert_inference_state(analyzer, 8000, 256)

        actual_model: Any = analyzer._model
        reference_model: Any = reference._model
        batch = np.stack(
            (self._reference_audio(self._frames(256)[1]), self._reference_audio(second_frame))
        )
        actual = actual_model(batch.copy(), 8000)
        expected = reference_model(batch.copy(), 8000)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0.0, atol=1e-6)
        self._assert_state_matches(analyzer, reference)
        self.assertEqual(actual_model._last_batch_size, 2)

    def test_model_run_does_not_retain_or_alias_a_reusable_input(self) -> None:
        sample_rate, frame_count = 16000, 512
        analyzer = self._new_analyzer(sample_rate)
        reference = self._new_analyzer(sample_rate)
        model: Any = analyzer._model
        reference_model: Any = reference._model
        original_session = model.session
        proxy = _SynchronousSessionProxy(original_session)
        model.session = proxy

        try:
            audio = self._reference_audio(self._frames(frame_count)[1])
            source_ref = weakref.ref(audio)
            expected = reference_model(audio.copy(), sample_rate)
            actual = model(audio, sample_rate)
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected), rtol=0.0, atol=1e-6
            )
            self.assertEqual(proxy.run_calls, 1)
            self.assertEqual(proxy.run_async_calls, 0)
            self.assertIsNotNone(proxy.last_input)
            self.assertFalse(np.shares_memory(audio, np.asarray(proxy.last_input)))

            context_before = np.asarray(model._context).copy()
            audio.fill(np.float32(-1.0))
            np.testing.assert_array_equal(np.asarray(model._context), context_before)

            next_audio = self._reference_audio(self._frames(frame_count)[3])
            expected_next = reference_model(next_audio.copy(), sample_rate)
            actual_next = model(next_audio, sample_rate)
            np.testing.assert_allclose(
                np.asarray(actual_next), np.asarray(expected_next), rtol=0.0, atol=1e-6
            )

            del audio
            gc.collect()
            self.assertIsNone(source_ref())
        finally:
            model.session = original_session


if __name__ == "__main__":
    unittest.main()
