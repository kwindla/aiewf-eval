import threading
import time
import queue
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from loguru import logger

from pipecat.frames.frames import (
    EmulateUserStartedSpeakingFrame,
    EmulateUserStoppedSpeakingFrame,
    Frame,
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
    - Supports wait_for_ready mode where audio is held until signal_ready() is called.
    """

    def __init__(
        self,
        params: TransportParams,
        *,
        chunk_ms: int = 20,
        pre_roll_ms: int = 0,
        continuous_silence: bool = False,
        wait_for_ready: bool = False,
        emit_user_stopped_speaking: bool = False,
    ):
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
        self._wait_for_ready = wait_for_ready  # If True, wait for signal_ready() before sending audio
        self._emit_user_stopped_speaking = emit_user_stopped_speaking  # Emit UserStoppedSpeakingFrame after each file
        self._llm_ready = threading.Event()  # Signaled when downstream LLM is ready to receive audio
        if not wait_for_ready:
            self._llm_ready.set()  # If not waiting, consider LLM ready immediately

        # Recording synchronization: feeder waits for this before sending any frames
        # This ensures user audio is perfectly aligned with recording wall-clock time
        self._recording_baseline_event = threading.Event()
        self._recording_baseline_monotonic: float = 0.0  # time.monotonic() when recording started

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

    def signal_ready(self):
        """Signal that the downstream LLM is ready to receive audio.

        Call this after the LLM has established its session/prompt with the server.
        Until this is called (when wait_for_ready=True), no audio frames will be sent.
        """
        if not self._llm_ready.is_set():
            logger.info(f"{self}: LLM signaled ready, starting audio transmission")
            self._llm_ready.set()

    def set_recording_baseline(self):
        """Set the recording baseline time for wall-clock synchronized audio.

        Call this when audio recording starts. The feeder will synchronize its
        timing baseline to this moment, ensuring user audio frames are aligned
        with wall-clock time in the recording.

        IMPORTANT: Call this BEFORE start_recording() on AudioBufferProcessor,
        and AFTER reset_recording_baseline() on NullAudioOutputTransport. All
        three components must use the same T=0 baseline.
        """
        self._recording_baseline_monotonic = time.monotonic()
        self._recording_baseline_event.set()
        logger.info(
            f"{self}: Recording baseline set at monotonic={self._recording_baseline_monotonic:.3f}"
        )

    def pause(self):
        """Pause audio transmission by clearing the ready signal.

        Call this during reconnection to prevent audio from being sent to a
        disconnected or reconnecting LLM. Call signal_ready() to resume.
        """
        if self._llm_ready.is_set():
            logger.info(f"{self}: Pausing audio transmission")
            self._llm_ready.clear()
            # Also clear any pending audio in the queue to avoid sending stale audio
            while not self._buf_queue.empty():
                try:
                    self._buf_queue.get_nowait()
                    self._buf_queue.task_done()
                except queue.Empty:
                    break
            logger.info(f"{self}: Audio queue cleared")

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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, filtering out emulated VAD frames.

        This transport is designed for synthetic/pre-recorded audio playback to
        services with server-side VAD (Gemini Live, OpenAI Realtime). There is no
        client-side VAD, so the user aggregator's "emulated VAD" logic incorrectly
        assumes VAD failed to detect speech when it sees transcriptions without
        preceding UserStartedSpeakingFrame.

        We filter EmulateUser*SpeakingFrame to prevent spurious InterruptionFrames
        that would disrupt the LLM's response generation.
        """
        if isinstance(frame, (EmulateUserStartedSpeakingFrame, EmulateUserStoppedSpeakingFrame)):
            logger.debug(f"Filtering emulated VAD frame: {type(frame).__name__}")
            return  # Don't forward to parent's handler

        await super().process_frame(frame, direction)

    # Internal
    def _feeder_loop(self):
        # Wait until pipeline signals StartFrame
        self._ready.wait()

        # If wait_for_ready mode, wait for LLM to signal it's ready before sending any audio
        if self._wait_for_ready and not self._llm_ready.is_set():
            logger.info(f"{self}: Waiting for LLM ready signal before sending audio...")
            self._llm_ready.wait()
            logger.info(f"{self}: LLM ready signal received, proceeding with audio")

        sr = self.sample_rate or (self._params.audio_in_sample_rate or 16000)
        bytes_per_sample = 2  # int16

        # Wait for recording baseline before sending ANY frames (including pre-roll)
        # This ensures user audio is perfectly aligned with recording wall-clock time
        logger.info(f"{self}: Waiting for recording baseline before sending audio...")
        self._recording_baseline_event.wait()
        logger.info(
            f"{self}: Recording baseline received, starting audio at "
            f"monotonic={self._recording_baseline_monotonic:.3f}"
        )

        # Optional pre-roll of silence to let downstream settle
        # Pre-roll is sent at real-time pace starting from recording baseline
        if self._pre_roll_ms > 0 and not self._did_preroll:
            samples_per_chunk = int(sr * (self._chunk_ms / 1000.0))
            chunk_bytes = samples_per_chunk * self._num_channels * bytes_per_sample
            num_chunks = max(1, (self._pre_roll_ms + self._chunk_ms - 1) // self._chunk_ms)
            logger.debug(
                f"{self}: feeder preroll start sr={sr} chunk_bytes={chunk_bytes} chunks={num_chunks}"
            )
            # Use recording baseline as start time for pre-roll
            start_t = self._recording_baseline_monotonic
            for i in range(num_chunks):
                silence = bytes(chunk_bytes)
                frame = InputAudioRawFrame(
                    audio=silence, sample_rate=sr, num_channels=self._num_channels
                )
                # Add timing diagnostic for pre-roll frames
                frame._paced_input_send_time = time.monotonic()
                if i == 0:
                    logger.info(f"{self}: First pre-roll frame created at monotonic={frame._paced_input_send_time:.3f}")
                loop = self.get_event_loop()
                loop.call_soon_threadsafe(lambda f=frame: self.create_task(self.push_audio_frame(f)))
                # Pace to real time from recording baseline
                next_time = start_t + ((i + 1) * self._chunk_ms) / 1000.0
                sleep_for = next_time - time.monotonic()
                if sleep_for > 0:
                    time.sleep(min(sleep_for, 0.05))
            self._did_preroll = True

        # Main loop: send audio or silence
        silence_chunk_bytes = int(sr * (self._chunk_ms / 1000.0)) * self._num_channels * bytes_per_sample
        silence_chunk = bytes(silence_chunk_bytes)
        chunk_interval_sec = self._chunk_ms / 1000.0

        # Global timing: track when the next chunk should be sent
        # Use recording baseline as T=0 for perfect wall-clock alignment
        # Account for any pre-roll that was already sent
        preroll_duration = (self._pre_roll_ms / 1000.0) if self._did_preroll else 0
        next_chunk_time = self._recording_baseline_monotonic + preroll_duration

        # Verify silence is actually zeros
        if silence_chunk_bytes > 0:
            logger.debug(
                f"{self}: Generated silence chunk: {silence_chunk_bytes} bytes, "
                f"first 10 bytes: {list(silence_chunk[:10])}"
            )

        while not self._stop.is_set():
            # Check if we should wait for LLM ready (supports pause/resume during reconnection)
            if self._wait_for_ready and not self._llm_ready.is_set():
                logger.debug(f"{self}: Waiting for LLM ready signal (paused)...")
                self._llm_ready.wait(timeout=0.5)  # Check periodically so we can respond to stop
                if not self._llm_ready.is_set():
                    next_chunk_time = time.monotonic()  # Reset timing after pause
                    continue  # Still not ready, check stop flag and try again

            try:
                # Use non-blocking get when in continuous_silence mode to avoid
                # double-waiting (queue timeout + chunk pacing). When there's no
                # audio queued, we want to immediately send silence at real-time pace.
                timeout = 0 if self._continuous_silence else 0.02
                audio_bytes, num_channels = self._buf_queue.get(timeout=timeout)
                has_audio = True
            except queue.Empty:
                has_audio = False
                # If continuous_silence mode, send silence chunk
                if self._continuous_silence:
                    audio_bytes = silence_chunk
                    num_channels = self._num_channels
                else:
                    # Not in continuous silence mode, wait a bit before checking again
                    time.sleep(0.02)
                    continue

            # Chunking setup
            samples_per_chunk = int(sr * (self._chunk_ms / 1000.0))
            chunk_bytes = samples_per_chunk * num_channels * bytes_per_sample
            total = len(audio_bytes)
            offset = 0

            if has_audio:
                # Log actual audio with sample of first bytes to verify no WAV header
                logger.info(
                    f"{self}: SENDING REAL AUDIO: {total} bytes, sr={sr}, ch={num_channels}, "
                    f"first 20 bytes: {list(audio_bytes[:20])}"
                )

            while offset < total and not self._stop.is_set():
                end = min(offset + chunk_bytes, total)
                chunk = audio_bytes[offset:end]
                offset = end

                # Wait until it's time to send the next chunk (global timing)
                sleep_for = next_chunk_time - time.monotonic()
                if sleep_for > 0:
                    time.sleep(min(sleep_for, 0.05))

                frame = InputAudioRawFrame(audio=chunk, sample_rate=sr, num_channels=num_channels)
                # Add timing diagnostic for first few frames
                if offset <= chunk_bytes * 3:
                    frame._paced_input_send_time = time.monotonic()
                    logger.debug(f"{self}: Frame created at monotonic={frame._paced_input_send_time:.3f}")
                loop = self.get_event_loop()
                loop.call_soon_threadsafe(lambda f=frame: self.create_task(self.push_audio_frame(f)))

                # Schedule next chunk 20ms from the current scheduled time (not from now)
                # This maintains accurate long-term timing regardless of processing overhead
                next_chunk_time += chunk_interval_sec

            # Only mark task as done if we actually got audio from the queue
            if has_audio:
                self._buf_queue.task_done()
                num_chunks = (total + chunk_bytes - 1) // chunk_bytes if chunk_bytes > 0 else 0
                logger.info(
                    f"{self}: FINISHED SENDING AUDIO ({total} bytes in {num_chunks} chunks), "
                    f"resuming silence"
                )

                # Optionally emit UserStoppedSpeakingFrame for services with VAD disabled
                # This triggers manual audio buffer commit and response creation
                if self._emit_user_stopped_speaking:
                    logger.info(f"{self}: Emitting UserStoppedSpeakingFrame")
                    loop = self.get_event_loop()
                    loop.call_soon_threadsafe(
                        lambda: self.create_task(self.push_frame(UserStoppedSpeakingFrame()))
                    )

            # NOTE: By default we do not send UserStoppedSpeakingFrame here.
            # For realtime/live models with server-side VAD, we rely on the server
            # to detect end of speech. We send continuous audio (including silence)
            # so the connection never appears idle.
            # Set emit_user_stopped_speaking=True for services with VAD disabled.
