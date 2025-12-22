#!/usr/bin/env python3
"""Analyze TTFB (Time to First Byte) from benchmark runs using Silero VAD.

This script calculates user-to-model TTFB using Silero VAD for audio segmentation:
- Uses neural network-based voice activity detection for more accurate speech boundaries
- TTFB = gap between user audio end and bot audio start
- Pairs segments by index (user segment N with bot segment N)

Usage:
    uv run python scripts/analyze_ttfb_silero.py <run_dir>
    uv run python scripts/analyze_ttfb_silero.py runs/aiwf_medium_context/20251220T220327_ultravox-v0.7_5c50ef67

Output:
    Prints TTFB statistics and per-turn breakdown.
    Returns JSON to stdout if --json flag is used.
"""

import argparse
import json
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy import signal

# Silence torch warnings
torch.set_num_threads(1)


@dataclass
class AudioSegment:
    """A segment of detected audio."""
    start_ms: float
    end_ms: float
    channel: str  # "user" or "bot"

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class TurnTiming:
    """Timing for a single user->bot turn."""
    turn_index: int
    user_start_ms: float
    user_end_ms: float
    bot_start_ms: float
    bot_end_ms: float
    gap_ms: float  # TTFB: time between user end and bot start
    has_tool_call: bool = False  # Whether this turn involved a tool call


def load_transcript_tool_calls(run_dir: Path) -> dict[int, list[str]]:
    """Load transcript.jsonl and return a dict mapping turn index to tool call names.

    Args:
        run_dir: Path to the run directory containing transcript.jsonl

    Returns:
        Dict mapping turn index to list of tool call names (empty list if no tool calls)
    """
    transcript_path = run_dir / "transcript.jsonl"
    tool_calls_by_turn: dict[int, list[str]] = {}

    if not transcript_path.exists():
        return tool_calls_by_turn

    with open(transcript_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                turn_idx = entry.get("turn", -1)
                tool_calls = entry.get("tool_calls", [])
                tool_call_names = [tc.get("name", "unknown") for tc in tool_calls if tc]
                tool_calls_by_turn[turn_idx] = tool_call_names
            except json.JSONDecodeError:
                continue

    return tool_calls_by_turn


def load_silero_vad():
    """Load Silero VAD model and utilities."""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=True,
        trust_repo=True,
        verbose=False,
    )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def load_stereo_wav(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load stereo WAV file, return (user_audio, bot_audio, sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        if n_channels != 2:
            raise ValueError(f"Expected stereo (2 channels), got {n_channels}")

        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

        audio = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        return audio[:, 0].astype(np.float32), audio[:, 1].astype(np.float32), sample_rate


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample audio to 16kHz for Silero VAD."""
    if orig_sr == 16000:
        return audio
    n_samples = int(len(audio) * 16000 / orig_sr)
    return signal.resample(audio, n_samples)


def detect_segments_silero(
    audio: np.ndarray,
    sample_rate: int,
    model,
    get_speech_timestamps,
    channel: str,
    min_silence_duration_ms: float = 2000.0,
) -> list[AudioSegment]:
    """Detect speech segments using Silero VAD.

    Args:
        audio: Audio samples (int16 or float32)
        sample_rate: Original sample rate
        model: Silero VAD model
        get_speech_timestamps: Silero utility function
        channel: "user" or "bot"
        min_silence_duration_ms: Merge segments closer than this

    Returns:
        List of AudioSegment objects
    """
    # Resample to 16kHz for Silero
    audio_16k = resample_to_16k(audio, sample_rate)

    # Normalize to float32 [-1, 1]
    if audio_16k.dtype == np.int16:
        audio_16k = audio_16k.astype(np.float32) / 32768.0
    elif audio_16k.dtype == np.float32:
        max_val = np.max(np.abs(audio_16k))
        if max_val > 1.0:
            audio_16k = audio_16k / 32768.0

    # Get speech timestamps from Silero
    timestamps = get_speech_timestamps(
        torch.from_numpy(audio_16k),
        model,
        sampling_rate=16000,
        min_silence_duration_ms=int(min_silence_duration_ms),
        min_speech_duration_ms=100,
    )

    # Convert to AudioSegment objects (in original sample rate timing)
    # Silero returns sample indices at 16kHz, convert to ms
    segments = []
    for ts in timestamps:
        start_ms = ts['start'] / 16.0  # 16 samples per ms at 16kHz
        end_ms = ts['end'] / 16.0
        segments.append(AudioSegment(start_ms=start_ms, end_ms=end_ms, channel=channel))

    return segments


def pair_turns(
    user_segments: list[AudioSegment], bot_segments: list[AudioSegment]
) -> tuple[list[TurnTiming], bool]:
    """Pair user segments with bot segments by index to measure TTFB.

    Uses simple index-based pairing: user segment 0 pairs with bot segment 0, etc.
    This approach is more robust than temporal pairing which can cascade-fail on overlaps.

    If the first bot segment starts before the first user segment ends, we skip it.
    This handles models (like Gemini) that emit an initial greeting before the user speaks.

    TTFB is calculated as (bot_start - user_end). Negative values indicate overlap.

    Returns:
        Tuple of (turns, skipped_initial_bot) where skipped_initial_bot indicates
        if we skipped an initial bot greeting.
    """
    # Check if we need to skip an initial bot greeting
    # Skip first bot segment if it starts before the first user segment ends
    skipped_initial_bot = False
    if user_segments and bot_segments:
        if bot_segments[0].start_ms < user_segments[0].end_ms:
            bot_segments = bot_segments[1:]
            skipped_initial_bot = True

    turns = []
    n_pairs = min(len(user_segments), len(bot_segments))

    for i in range(n_pairs):
        user_seg = user_segments[i]
        bot_seg = bot_segments[i]
        gap_ms = bot_seg.start_ms - user_seg.end_ms

        turns.append(
            TurnTiming(
                turn_index=i,
                user_start_ms=user_seg.start_ms,
                user_end_ms=user_seg.end_ms,
                bot_start_ms=bot_seg.start_ms,
                bot_end_ms=bot_seg.end_ms,
                gap_ms=gap_ms,
            )
        )

    return turns, skipped_initial_bot


def analyze_audio(
    wav_path: Path,
    model,
    get_speech_timestamps,
    min_silence_duration_ms: float = 2000.0,
) -> tuple[list[TurnTiming], list[AudioSegment], list[AudioSegment], bool]:
    """Analyze conversation.wav for turn timing using Silero VAD.

    Args:
        wav_path: Path to stereo conversation.wav
        model: Silero VAD model
        get_speech_timestamps: Silero utility function
        min_silence_duration_ms: Silence gap threshold for merging segments

    Returns:
        Tuple of (turns, user_segments, bot_segments, skipped_initial_bot)
    """
    user_audio, bot_audio, sample_rate = load_stereo_wav(wav_path)

    # Detect segments using Silero VAD
    user_segments = detect_segments_silero(
        user_audio, sample_rate, model, get_speech_timestamps,
        "user", min_silence_duration_ms
    )
    bot_segments = detect_segments_silero(
        bot_audio, sample_rate, model, get_speech_timestamps,
        "bot", min_silence_duration_ms
    )

    # Pair turns
    turns, skipped_initial_bot = pair_turns(user_segments, bot_segments)

    return turns, user_segments, bot_segments, skipped_initial_bot


def compute_ttfb_stats(ttfb_values: list[float], prefix: str = "ttfb") -> dict:
    """Compute TTFB statistics for a list of values.

    Args:
        ttfb_values: List of TTFB values in ms
        prefix: Prefix for the keys in the returned dict

    Returns:
        Dictionary with statistical measures
    """
    if not ttfb_values:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_std": None,
            f"{prefix}_p50": None,
            f"{prefix}_p90": None,
            f"{prefix}_p95": None,
        }

    return {
        f"{prefix}_count": len(ttfb_values),
        f"{prefix}_mean": float(np.mean(ttfb_values)),
        f"{prefix}_median": float(np.median(ttfb_values)),
        f"{prefix}_min": float(np.min(ttfb_values)),
        f"{prefix}_max": float(np.max(ttfb_values)),
        f"{prefix}_std": float(np.std(ttfb_values)),
        f"{prefix}_p50": float(np.percentile(ttfb_values, 50)),
        f"{prefix}_p90": float(np.percentile(ttfb_values, 90)),
        f"{prefix}_p95": float(np.percentile(ttfb_values, 95)),
    }


def analyze_run(run_dir: Path, min_silence_duration_ms: float = 2000.0) -> Optional[dict]:
    """Analyze a single run directory for TTFB metrics using Silero VAD.

    Args:
        run_dir: Path to the run directory.
        min_silence_duration_ms: Silence gap threshold for merging segments.

    Returns:
        Dictionary with TTFB metrics, or None if analysis fails.
    """
    wav_path = run_dir / "conversation.wav"

    if not wav_path.exists():
        print(f"Error: conversation.wav not found in {run_dir}", file=sys.stderr)
        return None

    # Load transcript to identify tool call turns
    tool_calls_by_turn = load_transcript_tool_calls(run_dir)

    # Load Silero VAD
    try:
        model, get_speech_timestamps = load_silero_vad()
    except Exception as e:
        print(f"Error loading Silero VAD: {e}", file=sys.stderr)
        return None

    # Analyze audio
    try:
        turns, user_segments, bot_segments, skipped_initial_bot = analyze_audio(
            wav_path, model, get_speech_timestamps, min_silence_duration_ms
        )
    except Exception as e:
        print(f"Error analyzing audio: {e}", file=sys.stderr)
        return None

    if not turns:
        print(f"Error: No turns detected in audio for {run_dir}", file=sys.stderr)
        return None

    # Annotate turns with tool call info
    for turn in turns:
        tool_calls = tool_calls_by_turn.get(turn.turn_index, [])
        turn.has_tool_call = len(tool_calls) > 0

    # Sanity notes
    sanity_notes = []
    if skipped_initial_bot:
        sanity_notes.append(
            "Skipped initial bot greeting (bot started before first user segment ended)"
        )
    if len(user_segments) != len(bot_segments):
        sanity_notes.append(
            f"Segment count mismatch: {len(user_segments)} user vs {len(bot_segments)} bot segments"
        )

    # Check for overlaps (negative TTFB)
    overlaps = [(i, t.gap_ms) for i, t in enumerate(turns) if t.gap_ms < 0]
    if overlaps:
        sanity_notes.append(f"Overlapping turns detected: {len(overlaps)} turns with negative TTFB")

    # Separate turns by tool call status
    tool_call_turns = [t for t in turns if t.has_tool_call]
    non_tool_call_turns = [t for t in turns if not t.has_tool_call]

    # Calculate statistics for all turns
    ttfb_values = [t.gap_ms for t in turns]
    max_ttfb_turn = int(np.argmax(ttfb_values))

    # Calculate statistics for tool call turns
    tool_call_ttfb = [t.gap_ms for t in tool_call_turns]
    tool_call_indices = [t.turn_index for t in tool_call_turns]

    # Calculate statistics for non-tool call turns
    non_tool_call_ttfb = [t.gap_ms for t in non_tool_call_turns]

    # Extract model name from directory name
    model_name = run_dir.name.split('_', 1)[1] if '_' in run_dir.name else run_dir.name
    if '_' in model_name:
        parts = model_name.rsplit('_', 1)
        if len(parts[1]) == 8 and all(c in '0123456789abcdef' for c in parts[1]):
            model_name = parts[0]

    # Build result with overall and segmented stats
    result = {
        "run_dir": str(run_dir),
        "model": model_name,
        "source": "silero_vad",
        "skipped_initial_bot": skipped_initial_bot,
        "num_turns": len(turns),
        "user_segments": len(user_segments),
        "bot_segments": len(bot_segments),
        "num_overlaps": len(overlaps),
        "ttfb_per_turn": ttfb_values,
        "has_tool_call_per_turn": [t.has_tool_call for t in turns],
        "ttfb_mean": float(np.mean(ttfb_values)),
        "ttfb_median": float(np.median(ttfb_values)),
        "ttfb_min": float(np.min(ttfb_values)),
        "ttfb_max": float(np.max(ttfb_values)),
        "ttfb_max_turn": max_ttfb_turn,
        "ttfb_std": float(np.std(ttfb_values)),
        "ttfb_p50": float(np.percentile(ttfb_values, 50)),
        "ttfb_p90": float(np.percentile(ttfb_values, 90)),
        "ttfb_p95": float(np.percentile(ttfb_values, 95)),
        "sanity_notes": sanity_notes,
        # Tool call turn stats
        "tool_call_turns": tool_call_indices,
    }

    # Add tool call stats
    result.update(compute_ttfb_stats(tool_call_ttfb, "tool_call_ttfb"))

    # Add non-tool call stats
    result.update(compute_ttfb_stats(non_tool_call_ttfb, "non_tool_call_ttfb"))

    return result


def print_stats_block(results: dict, prefix: str, title: str):
    """Print a statistics block for a given prefix."""
    count = results.get(f"{prefix}_count", 0)
    if count == 0:
        print(f"  (no turns)")
        return

    print(f"  Count:      {count:>8} turns")
    print(f"  Mean:       {results[f'{prefix}_mean']:>8.0f}ms")
    print(f"  Median:     {results[f'{prefix}_median']:>8.0f}ms")
    print(f"  Min:        {results[f'{prefix}_min']:>8.0f}ms")
    print(f"  Max:        {results[f'{prefix}_max']:>8.0f}ms")
    print(f"  Std Dev:    {results[f'{prefix}_std']:>8.0f}ms")
    print(f"  P50:        {results[f'{prefix}_p50']:>8.0f}ms")
    print(f"  P90:        {results[f'{prefix}_p90']:>8.0f}ms")
    print(f"  P95:        {results[f'{prefix}_p95']:>8.0f}ms")


def print_results(results: dict, verbose: bool = False):
    """Print TTFB analysis results in human-readable format."""
    print("=" * 70)
    print(f"TTFB ANALYSIS (Silero VAD): {results['model']}")
    print("=" * 70)
    print(f"Run: {results['run_dir']}")
    print(f"Source: {results['source']}")
    print(f"User segments: {results['user_segments']}, Bot segments: {results['bot_segments']}")
    print(f"Turns paired: {results['num_turns']}")

    # Show tool call breakdown
    tool_call_count = results.get('tool_call_ttfb_count', 0)
    non_tool_call_count = results.get('non_tool_call_ttfb_count', 0)
    print(f"Tool call turns: {tool_call_count}, Non-tool call turns: {non_tool_call_count}")

    if results.get('num_overlaps', 0) > 0:
        print(f"Overlaps: {results['num_overlaps']} (negative TTFB)")

    if results.get('sanity_notes'):
        print("\nNotes:")
        for note in results['sanity_notes']:
            print(f"  - {note}")

    if verbose:
        has_tool_call = results.get('has_tool_call_per_turn', [])
        print(f"\n{'Turn':<6} {'TTFB':>10} {'Tool Call':>12}")
        print(f"{'-'*6} {'-'*10} {'-'*12}")
        for i, ttfb in enumerate(results['ttfb_per_turn']):
            marker = " <-- MAX" if i == results['ttfb_max_turn'] else ""
            tool_marker = "YES" if (i < len(has_tool_call) and has_tool_call[i]) else ""
            print(f"{i:<6} {ttfb:>8.0f}ms {tool_marker:>12}{marker}")

    # Overall statistics
    print(f"\n{'='*70}")
    print("OVERALL STATISTICS (All Turns)")
    print(f"{'='*70}")
    print(f"  Count:      {results['num_turns']:>8} turns")
    print(f"  Mean:       {results['ttfb_mean']:>8.0f}ms")
    print(f"  Median:     {results['ttfb_median']:>8.0f}ms")
    print(f"  Min:        {results['ttfb_min']:>8.0f}ms")
    print(f"  Max:        {results['ttfb_max']:>8.0f}ms (turn {results['ttfb_max_turn']})")
    print(f"  Std Dev:    {results['ttfb_std']:>8.0f}ms")
    print(f"  P50:        {results['ttfb_p50']:>8.0f}ms")
    print(f"  P90:        {results['ttfb_p90']:>8.0f}ms")
    print(f"  P95:        {results['ttfb_p95']:>8.0f}ms")

    # Non-tool call statistics
    print(f"\n{'-'*70}")
    print("NON-TOOL CALL TURNS")
    print(f"{'-'*70}")
    print_stats_block(results, "non_tool_call_ttfb", "Non-Tool Call")

    # Tool call statistics
    print(f"\n{'-'*70}")
    print(f"TOOL CALL TURNS (turns: {results.get('tool_call_turns', [])})")
    print(f"{'-'*70}")
    print_stats_block(results, "tool_call_ttfb", "Tool Call")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TTFB using Silero VAD for speech segmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script uses Silero VAD (neural network-based voice activity detection)
to segment conversation.wav and calculate TTFB:
- More accurate speech boundary detection than RMS energy
- TTFB = gap between user audio end and bot audio start
- Segments paired by index (user 0 with bot 0, etc.)
        """,
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the run directory containing conversation.wav"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show per-turn TTFB values"
    )
    parser.add_argument(
        "--min-silence-ms",
        type=float,
        default=2000.0,
        help="Silence gap threshold for merging segments (default: 2000ms)"
    )

    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Error: Run directory not found: {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    results = analyze_run(args.run_dir, min_silence_duration_ms=args.min_silence_ms)

    if results is None:
        sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results, verbose=args.verbose)


if __name__ == "__main__":
    main()
