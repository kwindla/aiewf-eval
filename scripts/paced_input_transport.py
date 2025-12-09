import threading
import time
import queue
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from loguru import logger

from pipecat.frames.frames import (
    InputAudioRawFrame,
    StartFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams


def _linear_resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x
    n_in = x.shape[0]
    n_out = int(round(n_in * (sr_out / sr_in)))
    if n_in == 0 or n_out == 0:
        return np.zeros((0, x.shape[1]), dtype=x.dtype)
    t_in = np.arange(n_in, dtype=np.float64)
    t_out = np.linspace(0.0, n_in - 1, num=n_out, dtype=np.float64)
    y = np.empty((n_out, x.shape[1]), dtype=np.float32)
    for ch in range(x.shape[1]):
        y[:, ch] = np.interp(t_out, t_in, x[:, ch].astype(np.float64)).astype(np.float32)
    # Convert back to int16 range
    y = np.clip(y, -1.0, 1.0)
    y_i16 = (y * 32767.0).astype(np.int16)
    return y_i16


class PacedInputTransport(BaseInputTransport):
    """Very simple paced audio input transport.

    - Accepts whole-audio buffers (e.g., from WAV files).
    - Chops into 20 ms chunks at `params.audio_in_sample_rate`.
    - Feeds InputAudioRawFrame to the pipeline at realtime pace.
    - No VAD, no mixing; relies on BaseInputTransport to forward frames.
    """

    def __init__(self, params: TransportParams, *, chunk_ms: int = 20, pre_roll_ms: int = 0, continuous_silence: bool = False):
        super().__init__(params)
        self._chunk_ms = chunk_ms
        self._pre_roll_ms = max(0, int(pre_roll_ms))
        self._buf_queue: "queue.Queue[Tuple[bytes, int]]" = queue.Queue()
        self._feeder: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._num_channels = params.audio_in_channels or 1
        self._did_preroll = False
        self._continuous_silence = continuous_silence  # Send silence when no audio queued

    # Public API
    def enqueue_wav_file(self, path: str):
        data, sr = sf.read(path, dtype="int16", always_2d=True)
        if data.ndim != 2:
            data = np.atleast_2d(data)
        # Resample if needed
        target_sr = self.sample_rate or (self._params.audio_in_sample_rate or sr)
        if sr != target_sr:
            logger.warning(f"Resampling {path} from {sr} Hz to {target_sr} Hz for input transport")
            # Convert to float32 [-1,1] then resample then back to int16
            data_f = (data.astype(np.float32)) / 32768.0
            data = _linear_resample(data_f, sr, target_sr)
            sr = target_sr

        if data.shape[1] != self._num_channels:
            logger.warning(
                f"Channel mismatch for {path}: have {data.shape[1]}, expected {self._num_channels}. Using file value."
            )
            self._num_channels = data.shape[1]
        logger.debug(
            f"{self}: enqueue_wav_file path={path} frames={data.shape[0]} sr={sr} ch={self._num_channels}"
        )
        self.enqueue_bytes(data.tobytes(), num_channels=self._num_channels, sample_rate=sr)
        logger.debug(f"{self}: enqueue_wav_file queued bytes={len(data.tobytes())} queue_size={self._buf_queue.qsize()}")

    def enqueue_bytes(self, audio: bytes, *, num_channels: int, sample_rate: int):
        if sample_rate != (self.sample_rate or sample_rate):
            logger.warning(
                f"enqueue_bytes sample_rate {sample_rate} differs from transport {self.sample_rate}; proceeding"
            )
        # We only store bytes; channel count kept alongside
        self._buf_queue.put((audio, num_channels))

    # Lifecycle hooks
    async def start(self, frame: StartFrame):
        # Initialize base input transport (sets sample_rate, VAD, etc.)
        await super().start(frame)
        # Ensure our feeder thread and audio queue are created
        await self.set_transport_ready(frame)

    async def set_transport_ready(self, frame):
        await super().set_transport_ready(frame)
        if not self._feeder:
            logger.debug(
                f"{self}: set_transport_ready sr={self.sample_rate} ch={self._num_channels} "
                f"chunk_ms={self._chunk_ms} pre_roll_ms={self._pre_roll_ms}"
            )
            self._feeder = threading.Thread(target=self._feeder_loop, name=f"{self}-feeder", daemon=True)
            self._feeder.start()
        self._ready.set()

    async def stop(self, frame):
        await super().stop(frame)
        self._stop.set()
        if self._feeder and self._feeder.is_alive():
            self._feeder.join(timeout=1.0)

    # Internal
    def _feeder_loop(self):
        # Wait until pipeline signals StartFrame
        self._ready.wait()
        sr = self.sample_rate or (self._params.audio_in_sample_rate or 16000)
        bytes_per_sample = 2  # int16

        # Optional pre-roll of silence to let downstream settle
        if self._pre_roll_ms > 0 and not self._did_preroll:
            samples_per_chunk = int(sr * (self._chunk_ms / 1000.0))
            chunk_bytes = samples_per_chunk * self._num_channels * bytes_per_sample
            num_chunks = max(1, (self._pre_roll_ms + self._chunk_ms - 1) // self._chunk_ms)
            logger.debug(
                f"{self}: feeder preroll start sr={sr} chunk_bytes={chunk_bytes} chunks={num_chunks}"
            )
            start_t = time.monotonic()
            for i in range(num_chunks):
                silence = bytes(chunk_bytes)
                frame = InputAudioRawFrame(
                    audio=silence, sample_rate=sr, num_channels=self._num_channels
                )
                loop = self.get_event_loop()
                loop.call_soon_threadsafe(lambda f=frame: self.create_task(self.push_audio_frame(f)))
                # Pace to real time
                next_time = start_t + ((i + 1) * self._chunk_ms) / 1000.0
                sleep_for = next_time - time.monotonic()
                if sleep_for > 0:
                    time.sleep(min(sleep_for, 0.05))
            self._did_preroll = True
        # Main loop: send audio or silence
        silence_chunk_bytes = int(sr * (self._chunk_ms / 1000.0)) * self._num_channels * bytes_per_sample
        silence_chunk = bytes(silence_chunk_bytes)
        last_chunk_time = time.monotonic()

        # Verify silence is actually zeros
        if silence_chunk_bytes > 0:
            logger.debug(
                f"{self}: Generated silence chunk: {silence_chunk_bytes} bytes, "
                f"first 10 bytes: {list(silence_chunk[:10])}"
            )

        while not self._stop.is_set():
            try:
                audio_bytes, num_channels = self._buf_queue.get(timeout=0.02)
                has_audio = True
            except queue.Empty:
                # Log occasionally if we're only sending silence
                # to debug stalls (every ~2s)
                if int(time.monotonic() * 10) % 200 == 0:
                    logger.debug(f"{self}: _feeder_loop no audio in queue; sending silence")
                has_audio = False
                # If continuous_silence mode, send silence chunk
                if self._continuous_silence:
                    audio_bytes = silence_chunk
                    num_channels = self._num_channels
                else:
                    continue

            # Chunking setup
            samples_per_chunk = int(sr * (self._chunk_ms / 1000.0))
            chunk_bytes = samples_per_chunk * num_channels * bytes_per_sample
            total = len(audio_bytes)
            offset = 0
            start_t = time.monotonic()
            chunk_idx = 0

            if has_audio:
                # Log actual audio with sample of first bytes to verify no WAV header
                logger.info(
                    f"{self}: 🎤 SENDING REAL AUDIO: {total} bytes, sr={sr}, ch={num_channels}, "
                    f"first 20 bytes: {list(audio_bytes[:20])}"
                )
            else:
                # Log silence sending (only log once per silence period to avoid spam)
                if chunk_idx == 0:
                    logger.debug(
                        f"{self}: 🔇 Sending silence chunk ({silence_chunk_bytes} bytes)"
                    )

            while offset < total and not self._stop.is_set():
                end = min(offset + chunk_bytes, total)
                chunk = audio_bytes[offset:end]
                offset = end

                frame = InputAudioRawFrame(audio=chunk, sample_rate=sr, num_channels=num_channels)
                loop = self.get_event_loop()
                loop.call_soon_threadsafe(lambda f=frame: self.create_task(self.push_audio_frame(f)))

                # Pace to real time for 20ms per chunk
                chunk_idx += 1
                next_time = start_t + (chunk_idx * self._chunk_ms) / 1000.0
                sleep_for = next_time - time.monotonic()
                if sleep_for > 0:
                    time.sleep(min(sleep_for, 0.05))

            # Only mark task as done if we actually got audio from the queue
            if has_audio:
                self._buf_queue.task_done()
                logger.info(
                    f"{self}: ✅ FINISHED SENDING AUDIO ({total} bytes in {chunk_idx} chunks), "
                    f"resuming silence"
                )

            # NOTE: We no longer send UserStoppedSpeakingFrame here.
            # For realtime/live models with server-side VAD, we rely on the server
            # to detect end of speech. We send continuous audio (including silence)
            # so the connection never appears idle.
