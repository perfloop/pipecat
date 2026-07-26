# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import json
from pathlib import Path

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.frames.frames import InputAudioRawFrame, UserSpeakingFrame
from pipecat.processors.audio.vad_processor import VADProcessor

# Ten 640-byte, 16 kHz mono PCM frames represent 200 ms of input audio.
FRAME_COUNT = 10
SPEECH_ACTIVITY_PERIOD_SECS = 0.2
METRICS_PATH = Path("build/perfloop/vad-speech-activity.json")
FRAMES = tuple(
    InputAudioRawFrame(
        audio=bytes([frame_index]) * 640,
        sample_rate=16000,
        num_channels=1,
    )
    for frame_index in range(FRAME_COUNT)
)
EXPECTED_BYTES_SEEN = sum(len(frame.audio) + frame.audio[0] for frame in FRAMES)


class SpeakingAnalyzer(VADAnalyzer):
    def __init__(self):
        super().__init__(sample_rate=16000)
        self.bytes_seen = 0

    def num_frames_required(self):
        return 512

    def voice_confidence(self, buffer):
        return 1.0

    async def analyze_audio(self, buffer):
        self.bytes_seen += len(buffer) + buffer[0]
        return VADState.SPEAKING


class CountingVADProcessor(VADProcessor):
    def __init__(self):
        super().__init__(
            vad_analyzer=SpeakingAnalyzer(),
            speech_activity_period=SPEECH_ACTIVITY_PERIOD_SECS,
            enable_direct_mode=True,
        )
        # Model a started direct-mode processor so the real handler allocates
        # and broadcasts frames without requiring a background pipeline worker.
        self._FrameProcessor__started = True
        self.user_speaking_broadcasts = 0

    async def broadcast_frame(self, frame_cls, **kwargs):
        if frame_cls is UserSpeakingFrame:
            self.user_speaking_broadcasts += 1
        await super().broadcast_frame(frame_cls, **kwargs)


async def run_sustained_speech(processor):
    analyzer = SpeakingAnalyzer()
    controller = processor._vad_controller
    controller._vad_analyzer = analyzer
    controller._vad_state = VADState.QUIET
    controller._speech_activity_time = 0
    processor.user_speaking_broadcasts = 0

    for frame in FRAMES:
        await controller.process_frame(frame)

    return processor.user_speaking_broadcasts, analyzer.bytes_seen


def test_sustained_speech_activity_benchmark(benchmark):
    processor = CountingVADProcessor()
    loop = asyncio.new_event_loop()
    try:
        broadcasts, bytes_seen = benchmark(
            lambda: loop.run_until_complete(run_sustained_speech(processor))
        )
    finally:
        loop.close()

    assert broadcasts > 0
    assert bytes_seen == EXPECTED_BYTES_SEEN
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "metric": "user_speaking_broadcasts_per_200ms",
                "value": broadcasts,
            }
        )
        + "\n",
        encoding="utf-8",
    )
