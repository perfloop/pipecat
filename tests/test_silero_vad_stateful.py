#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import gc
import time
import unittest
import weakref
from importlib import resources

import numpy as np

from pipecat.audio.vad.silero import SileroOnnxModel, SileroVADAnalyzer


def _frame_size(sample_rate: int) -> int:
    return 512 if sample_rate == 16000 else 256


def _frames(sample_rate: int) -> list[bytes]:
    """Return deterministic silence, speech-like, and varied valid PCM frames."""
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


def _model_path() -> str:
    return str(resources.files("pipecat.audio.vad.data").joinpath("silero_vad.onnx"))


class _SynchronousRunProbe:
    """Record the ONNX input while preserving the synchronous session call."""

    def __init__(self, run):
        self._run = run
        self.called = False
        self.completed = False
        self.in_call = False
        self.input: np.ndarray | None = None
        self.input_snapshot: np.ndarray | None = None

    def __call__(self, output_names, input_feed, run_options=None):
        if self.in_call:
            raise AssertionError("session.run re-entered before the prior call returned")
        self.called = True
        self.in_call = True
        self.input = input_feed["input"]
        self.input_snapshot = self.input.copy()
        try:
            return self._run(output_names, input_feed, run_options)
        finally:
            self.in_call = False
            self.completed = True


class TestSileroVADStateful(unittest.TestCase):
    """Exercise real ONNX inference and the recurrent state owned by Silero VAD."""

    def _analyzer_and_reference(
        self, sample_rate: int
    ) -> tuple[SileroVADAnalyzer, SileroOnnxModel]:
        analyzer = SileroVADAnalyzer()
        analyzer.set_sample_rate(sample_rate)
        # Keep the periodic reset outside the deterministic state comparison.
        analyzer._last_reset_time = time.time()
        return analyzer, SileroOnnxModel(_model_path(), force_onnx_cpu=True)

    def _reference_confidence(self, reference: SileroOnnxModel, buffer: bytes, sample_rate: int):
        audio_int16 = np.frombuffer(buffer, np.int16)
        audio_float32 = audio_int16.astype(np.float32) / np.float32(32768.0)
        return reference(audio_float32, sample_rate)[0]

    def _assert_model_state_equal(self, analyzer: SileroVADAnalyzer, reference: SileroOnnxModel):
        model = analyzer._model
        np.testing.assert_array_equal(model._state, reference._state)
        np.testing.assert_array_equal(model._context, reference._context)
        self.assertEqual(model._last_sr, reference._last_sr)
        self.assertEqual(model._last_batch_size, reference._last_batch_size)

    def _assert_call_matches_reference(
        self,
        analyzer: SileroVADAnalyzer,
        reference: SileroOnnxModel,
        buffer: bytes,
        sample_rate: int,
    ):
        expected = self._reference_confidence(reference, buffer, sample_rate)
        actual = analyzer.voice_confidence(buffer)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        self._assert_model_state_equal(analyzer, reference)

    def test_confidence_and_recurrent_state_match_reference_for_each_rate(self):
        """Real voice_confidence matches an independent conversion/model sequence."""
        for sample_rate in (8000, 16000):
            with self.subTest(sample_rate=sample_rate):
                analyzer, reference = self._analyzer_and_reference(sample_rate)
                silence, speech_like, varied, alternating = _frames(sample_rate)
                for frame in (silence, speech_like, varied, speech_like, alternating, silence):
                    self._assert_call_matches_reference(analyzer, reference, frame, sample_rate)

    def test_invalid_buffers_return_zero_without_changing_recurrent_state(self):
        """Invalid empty and short frames preserve model state at both supported rates."""
        for sample_rate in (8000, 16000):
            with self.subTest(sample_rate=sample_rate):
                analyzer, reference = self._analyzer_and_reference(sample_rate)
                self._assert_call_matches_reference(
                    analyzer, reference, _frames(sample_rate)[1], sample_rate
                )
                model = analyzer._model
                state_before = model._state.copy()
                context_before = model._context.copy()
                last_sr_before = model._last_sr
                last_batch_size_before = model._last_batch_size

                invalid_frames = (
                    b"",
                    np.zeros(_frame_size(sample_rate) - 1, dtype=np.int16).tobytes(),
                )
                for invalid in invalid_frames:
                    self.assertEqual(analyzer.voice_confidence(invalid), 0)
                    np.testing.assert_array_equal(model._state, state_before)
                    np.testing.assert_array_equal(model._context, context_before)
                    self.assertEqual(model._last_sr, last_sr_before)
                    self.assertEqual(model._last_batch_size, last_batch_size_before)

    def test_reset_reuse_and_sample_rate_transitions_match_reference(self):
        """Resetting and alternating valid 8/16 kHz frames preserve recurrent behavior."""
        analyzer = SileroVADAnalyzer()
        reference = SileroOnnxModel(_model_path(), force_onnx_cpu=True)

        for sample_rate in (16000, 8000, 16000):
            analyzer.set_sample_rate(sample_rate)
            analyzer._last_reset_time = time.time()
            self.assertEqual(analyzer.num_frames_required(), _frame_size(sample_rate))
            self._assert_call_matches_reference(
                analyzer, reference, _frames(sample_rate)[2], sample_rate
            )

        analyzer._model.reset_states()
        reference.reset_states()
        analyzer._last_reset_time = time.time()
        self._assert_call_matches_reference(analyzer, reference, _frames(16000)[3], 16000)

    def test_onnx_run_is_synchronous_and_reusable_input_isolated(self):
        """The ONNX boundary cannot retain or mutate a future reusable conversion array."""
        for sample_rate in (8000, 16000):
            with self.subTest(sample_rate=sample_rate):
                analyzer, reference = self._analyzer_and_reference(sample_rate)
                session = analyzer._model.session
                probe = _SynchronousRunProbe(session.run)
                original_run = session.run
                original_run_async = session.run_async

                def fail_if_async(*args, **kwargs):
                    raise AssertionError(
                        "voice_confidence must not dispatch asynchronous ONNX inference"
                    )

                frame, next_frame = _frames(sample_rate)[1:3]
                expected = self._reference_confidence(reference, frame, sample_rate)
                try:
                    session.run = probe
                    session.run_async = fail_if_async
                    actual = analyzer.voice_confidence(frame)
                finally:
                    session.run = original_run
                    session.run_async = original_run_async

                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
                self.assertTrue(probe.called)
                self.assertTrue(probe.completed)
                self.assertFalse(probe.in_call)
                self.assertIsNotNone(probe.input)
                self.assertIsNotNone(probe.input_snapshot)
                np.testing.assert_array_equal(probe.input, probe.input_snapshot)
                self._assert_model_state_equal(analyzer, reference)

                reusable = getattr(analyzer, "_audio_float32", None)
                if reusable is not None:
                    self.assertEqual(reusable.dtype, np.float32)
                    self.assertGreaterEqual(reusable.size, _frame_size(sample_rate))
                    self.assertFalse(np.shares_memory(probe.input, reusable))
                    onnx_input_before_mutation = probe.input.copy()
                    reusable[: _frame_size(sample_rate)] = np.nan
                    np.testing.assert_array_equal(probe.input, onnx_input_before_mutation)

                self._assert_call_matches_reference(analyzer, reference, next_frame, sample_rate)

                if reusable is not None:
                    reusable_ref = weakref.ref(reusable)
                    del analyzer._audio_float32
                    del reusable
                    gc.collect()
                    self.assertIsNone(reusable_ref())


if __name__ == "__main__":
    unittest.main()
