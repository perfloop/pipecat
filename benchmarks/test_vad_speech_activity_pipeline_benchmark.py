# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import json
from pathlib import Path

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.frames.frames import InputAudioRawFrame, UserSpeakingFrame
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.tests.utils import SleepFrame, run_test

SAMPLE_RATE = 16_000
NUM_CHANNELS = 1
BYTES_PER_SAMPLE = 2
FRAME_BYTES = 640
FRAME_COUNT = 10
FRAME_INTERVAL_SECS = FRAME_BYTES / (SAMPLE_RATE * NUM_CHANNELS * BYTES_PER_SAMPLE)
DEFAULT_ACTIVITY_PERIOD_SECS = 0.2
EXPECTED_BYTES_SEEN = sum(FRAME_BYTES + frame_index for frame_index in range(FRAME_COUNT))


class SpeakingAnalyzer(VADAnalyzer):
    def __init__(self):
        super().__init__(sample_rate=SAMPLE_RATE)
        self.bytes_seen = 0

    def num_frames_required(self):
        return 512

    def voice_confidence(self, buffer):
        return 1.0

    async def analyze_audio(self, buffer):
        self.bytes_seen += len(buffer) + buffer[0]
        return VADState.SPEAKING


class CountingDirectVADProcessor(VADProcessor):
    def __init__(self):
        super().__init__(
            vad_analyzer=SpeakingAnalyzer(),
            speech_activity_period=0,
            enable_direct_mode=True,
        )
        self._FrameProcessor__started = True
        self.user_speaking_broadcasts = 0

    async def broadcast_frame(self, frame_cls, **kwargs):
        if frame_cls is UserSpeakingFrame:
            self.user_speaking_broadcasts += 1
        await super().broadcast_frame(frame_cls, **kwargs)


def _audio_frame(frame_index):
    return InputAudioRawFrame(
        audio=bytes([frame_index]) * FRAME_BYTES,
        sample_rate=SAMPLE_RATE,
        num_channels=NUM_CHANNELS,
    )


def _cadenced_audio_frames():
    frames = []
    for frame_index in range(FRAME_COUNT):
        frames.append(_audio_frame(frame_index))
        if frame_index < FRAME_COUNT - 1:
            frames.append(SleepFrame(sleep=FRAME_INTERVAL_SECS))
    return frames


async def _run_connected_stream():
    analyzer = SpeakingAnalyzer()
    processor = VADProcessor(
        vad_analyzer=analyzer,
        speech_activity_period=DEFAULT_ACTIVITY_PERIOD_SECS,
        audio_idle_timeout=0,
    )
    downstream_frames, upstream_frames = await run_test(
        processor,
        frames_to_send=_cadenced_audio_frames(),
    )
    downstream_activity = sum(isinstance(frame, UserSpeakingFrame) for frame in downstream_frames)
    upstream_activity = sum(isinstance(frame, UserSpeakingFrame) for frame in upstream_frames)
    return downstream_activity, upstream_activity, analyzer.bytes_seen


async def _run_unthrottled_direct_stream(processor):
    analyzer = SpeakingAnalyzer()
    controller = processor._vad_controller
    controller._vad_analyzer = analyzer
    controller._vad_state = VADState.QUIET
    controller._speech_activity_time = 0
    processor.user_speaking_broadcasts = 0

    for frame_index in range(FRAME_COUNT):
        await controller.process_frame(_audio_frame(frame_index))

    return processor.user_speaking_broadcasts, analyzer.bytes_seen


def _write_connected_metrics(path, downstream_activity, upstream_activity):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "metric": "connected_downstream_user_speaking_frames_per_200ms",
                        "value": downstream_activity,
                    }
                ),
                json.dumps(
                    {
                        "metric": "connected_upstream_user_speaking_frames_per_200ms",
                        "value": upstream_activity,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _write_direct_metrics(path, broadcasts):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metric": "direct_user_speaking_broadcasts_per_200ms",
                "value": broadcasts,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_connected_default_speech_activity_benchmark(benchmark):
    downstream_activity, upstream_activity, bytes_seen = benchmark(
        lambda: asyncio.run(_run_connected_stream())
    )

    assert downstream_activity == upstream_activity
    assert downstream_activity > 0
    assert bytes_seen == EXPECTED_BYTES_SEEN
    _write_connected_metrics(
        Path("build/perfloop/vad-connected-default.json"),
        downstream_activity,
        upstream_activity,
    )


def test_unthrottled_direct_speech_activity_benchmark(benchmark):
    processor = CountingDirectVADProcessor()
    loop = asyncio.new_event_loop()
    try:
        broadcasts, bytes_seen = benchmark(
            lambda: loop.run_until_complete(_run_unthrottled_direct_stream(processor))
        )
    finally:
        loop.close()

    assert broadcasts == FRAME_COUNT
    assert bytes_seen == EXPECTED_BYTES_SEEN
    _write_direct_metrics(Path("build/perfloop/vad-unthrottled-direct.json"), broadcasts)
