import asyncio

from loguru import logger

from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionLLMServiceMixin

MARKERLESS_RESPONSE_SIZE = 65_536
MARKERLESS_CHUNK_SIZE = 32


def _markerless_response(seed: str) -> str:
    unit = f"markerless-{seed}-response-payload "
    return (unit * ((MARKERLESS_RESPONSE_SIZE // len(unit)) + 1))[:MARKERLESS_RESPONSE_SIZE]


MARKERLESS_RESPONSES = tuple(
    (
        response,
        tuple(
            response[offset : offset + MARKERLESS_CHUNK_SIZE]
            for offset in range(0, len(response), MARKERLESS_CHUNK_SIZE)
        ),
    )
    for response in (_markerless_response("alpha"), _markerless_response("beta"))
)

# The benchmark measures parser work until the first downstream text frame, not
# configured log sink I/O. Logging remains exercised by the focused unit tests.
logger.disable("pipecat.turns.user_turn_completion_mixin")


class BenchmarkProcessor(UserTurnCompletionLLMServiceMixin, FrameProcessor):
    def __init__(self):
        super().__init__()
        self._emitted_text: list[str] = []
        self._response_index = 0
        self._checksum = 0

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMTextFrame):
            self._emitted_text.append(frame.text)

    async def process_until_first_text(self):
        response, chunks = MARKERLESS_RESPONSES[self._response_index]
        self._response_index = (self._response_index + 1) % len(MARKERLESS_RESPONSES)
        self._emitted_text.clear()

        for chunk in chunks:
            await self._push_turn_text(chunk)
            if self._emitted_text:
                break

        if not self._emitted_text:
            await self._turn_reset()

        emitted = "".join(self._emitted_text)
        if not emitted or not response.startswith(emitted):
            raise AssertionError("markerless response did not forward its original prefix")
        self._checksum += len(emitted)
        await self._turn_reset()


def test_markerless_response_first_text(benchmark):
    loop = asyncio.new_event_loop()
    processor = BenchmarkProcessor()
    try:
        benchmark(lambda: loop.run_until_complete(processor.process_until_first_text()))
    finally:
        loop.close()
