#!/usr/bin/env python3
"""Detect audio turn boundary tags in conversation.wav files.

This script detects 2kHz sine burst markers that are mixed into audio streams
to mark turn boundaries. Tags appear at the start of each user and bot turn,
enabling alignment verification and silent padding measurement.

Usage:
    python scripts/detect_audio_tags.py runs/<run_id>/conversation.wav
    python scripts/detect_audio_tags.py runs/<run_id>/conversation.wav --freq 2000

The script outputs:
- User (left channel) tag positions in milliseconds
- Bot (right channel) tag positions in milliseconds
"""

import argparse
import sys
import wave
from typing import Optional

import numpy as np
from scipy.fft import fft


def detect_tags(
    wav_path: str,
    channel: int,
    freq_hz: int = 2000,
    window_ms: int = 20,
    hop_ms: int = 5,
    threshold_ratio: float = 9.0,
    min_gap_ms: int = 50,
    min_level_db: float = -40.0,
) -> list[dict]:
    """Detect audio tags at specified frequency in a channel.

    Uses short-time Fourier transform (STFT) to find bursts of energy at the
    target frequency. Returns positions where the target frequency has
    significantly higher energy than the average.

    Args:
        wav_path: Path to stereo WAV file.
        channel: 0=left (user), 1=right (bot).
        freq_hz: Expected tag frequency (default 2kHz).
        window_ms: Analysis window size in milliseconds.
        hop_ms: Hop size between windows in milliseconds.
        threshold_ratio: Energy at target freq must be this many times average.
        min_gap_ms: Minimum gap between detected tags to avoid duplicates.
        min_level_db: Minimum RMS level in dB required for a valid tag (filters noise).

    Returns:
        List of dicts with 'position_ms' and 'energy_ratio' for each tag.
    """
    # Load WAV
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16)
    if n_channels == 2:
        audio = audio.reshape(-1, 2)
        track = audio[:, channel].astype(np.float32) / 32768
    else:
        track = audio.astype(np.float32) / 32768

    # STFT parameters
    window_samples = int(sr * window_ms / 1000)
    hop_samples = int(sr * hop_ms / 1000)

    tags = []
    for i in range(0, len(track) - window_samples, hop_samples):
        window = track[i : i + window_samples]

        # Apply Hanning window and compute FFT
        spectrum = np.abs(fft(window * np.hanning(len(window))))

        # Find bin closest to target frequency
        freq_resolution = sr / len(spectrum)
        target_bin = int(freq_hz / freq_resolution)

        # Check if target frequency has strong energy
        # Use a few bins around target to handle slight frequency shifts
        bin_range = max(1, int(100 / freq_resolution))  # 100Hz range
        energy_at_freq = np.max(spectrum[max(0, target_bin - bin_range) : target_bin + bin_range + 1])
        avg_energy = np.mean(spectrum[1 : len(spectrum) // 2])

        if avg_energy > 0:
            energy_ratio = energy_at_freq / avg_energy
        else:
            energy_ratio = 0

        if energy_ratio > threshold_ratio:
            # Check minimum absolute level to filter out noise
            rms = np.sqrt(np.mean(window**2))
            rms_db = 20 * np.log10(rms + 1e-10)
            if rms_db < min_level_db:
                continue

            pos_ms = int(i * 1000 / sr)
            # Avoid duplicates within min_gap_ms
            if not tags or pos_ms - tags[-1]["position_ms"] > min_gap_ms:
                tags.append({"position_ms": pos_ms, "energy_ratio": energy_ratio, "rms_db": rms_db})

    return tags


def analyze_wav(
    wav_path: str, freq_hz: int = 2000, threshold: float = 9.0, min_level_db: float = -40.0
) -> dict:
    """Analyze a WAV file for audio tags on both channels.

    Args:
        wav_path: Path to stereo WAV file.
        freq_hz: Expected tag frequency.
        threshold: Energy ratio threshold for tag detection.
        min_level_db: Minimum RMS level in dB for valid tags.

    Returns:
        Dict with 'user_tags', 'bot_tags', and 'sample_rate'.
    """
    # Get sample rate
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        duration_ms = int(wf.getnframes() * 1000 / sr)

    result = {
        "sample_rate": sr,
        "num_channels": n_channels,
        "duration_ms": duration_ms,
        "user_tags": [],
        "bot_tags": [],
    }

    if n_channels == 2:
        # Stereo: user on left (0), bot on right (1)
        result["user_tags"] = detect_tags(
            wav_path, 0, freq_hz, threshold_ratio=threshold, min_level_db=min_level_db
        )
        result["bot_tags"] = detect_tags(
            wav_path, 1, freq_hz, threshold_ratio=threshold, min_level_db=min_level_db
        )
    else:
        # Mono: tags are mixed together
        result["user_tags"] = detect_tags(
            wav_path, 0, freq_hz, threshold_ratio=threshold, min_level_db=min_level_db
        )
        result["bot_tags"] = []

    return result


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect audio turn boundary tags in conversation.wav files."
    )
    parser.add_argument("wav_path", help="Path to conversation.wav file")
    parser.add_argument(
        "--freq",
        type=int,
        default=2000,
        help="Expected tag frequency in Hz (default: 2000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=9.0,
        help="Energy ratio threshold for tag detection (default: 9.0)",
    )
    parser.add_argument(
        "--min-level",
        type=float,
        default=-40.0,
        help="Minimum RMS level in dB for valid tags (default: -40.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable",
    )

    args = parser.parse_args(argv)

    try:
        result = analyze_wav(args.wav_path, args.freq, args.threshold, args.min_level)
    except FileNotFoundError:
        print(f"Error: File not found: {args.wav_path}", file=sys.stderr)
        return 1
    except wave.Error as e:
        print(f"Error reading WAV file: {e}", file=sys.stderr)
        return 1

    if args.json:
        import json

        print(json.dumps(result, indent=2))
    else:
        print(f"WAV file: {args.wav_path}")
        print(f"Sample rate: {result['sample_rate']}Hz, Channels: {result['num_channels']}")
        print(f"Duration: {result['duration_ms']}ms ({result['duration_ms']/1000:.1f}s)")
        print(f"Target frequency: {args.freq}Hz")
        print()

        print(f"User (left channel) tags: {len(result['user_tags'])}")
        for i, tag in enumerate(result["user_tags"]):
            rms_str = f", rms: {tag['rms_db']:.1f}dB" if "rms_db" in tag else ""
            print(f"  [{i}] {tag['position_ms']}ms (ratio: {tag['energy_ratio']:.1f}x{rms_str})")

        print()
        print(f"Bot (right channel) tags: {len(result['bot_tags'])}")
        for i, tag in enumerate(result["bot_tags"]):
            rms_str = f", rms: {tag['rms_db']:.1f}dB" if "rms_db" in tag else ""
            print(f"  [{i}] {tag['position_ms']}ms (ratio: {tag['energy_ratio']:.1f}x{rms_str})")

        # Calculate gaps between consecutive bot tags (turn durations)
        if len(result["bot_tags"]) > 1:
            print()
            print("Bot tag intervals (approximate turn gaps):")
            for i in range(1, len(result["bot_tags"])):
                gap = result["bot_tags"][i]["position_ms"] - result["bot_tags"][i - 1]["position_ms"]
                print(f"  Turn {i-1} → {i}: {gap}ms ({gap/1000:.1f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
