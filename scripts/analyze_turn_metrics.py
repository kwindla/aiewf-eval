#!/usr/bin/env python3
"""Comprehensive per-turn metrics analysis for benchmark runs.

This script consolidates all timing metrics from multiple sources:
- Server TTFB from transcript.jsonl
- Pipeline TTFB from logs + Silero VAD
- WAV V2V from Silero VAD
- Silent Padding (RMS) from pipeline logs
- Silent Padding (Silero) from WAV analysis

It also performs alignment sanity checks using audio tags to verify that
log positions match WAV file positions.

Usage:
    uv run python scripts/analyze_turn_metrics.py <run_dir>
    uv run python scripts/analyze_turn_metrics.py <run_dir> --json
    uv run python scripts/analyze_turn_metrics.py <run_dir> -v  # verbose per-turn output

Output:
    Per-turn metrics table and summary statistics.
"""

import argparse
import json
import re
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy import signal
from scipy.fft import fft

# Silence torch warnings
torch.set_num_threads(1)


@dataclass
class TurnMetrics:
    """All metrics for a single turn."""
    turn_index: int
    # Server TTFB from transcript.jsonl
    server_ttfb_ms: Optional[int] = None
    # Raw timestamps
    user_start_ms: Optional[float] = None  # Silero user speech start
    user_end_ms: Optional[float] = None  # Silero user speech end
    bot_tag_log_ms: Optional[float] = None  # Bot tag position from logs
    bot_tag_wav_ms: Optional[float] = None  # Bot tag position from WAV detection
    bot_rms_onset_ms: Optional[float] = None  # RMS onset from logs
    bot_silero_start_ms: Optional[float] = None  # Silero bot speech start
    bot_silero_end_ms: Optional[float] = None  # Silero bot speech end
    # Derived metrics
    pipeline_ttfb_ms: Optional[float] = None  # Bot tag - User end
    wav_v2v_ms: Optional[float] = None  # Silero bot start - Silero user end
    silent_pad_rms_ms: Optional[float] = None  # RMS onset - Bot tag
    silent_pad_silero_ms: Optional[float] = None  # Silero start - Bot tag (WAV)
    # Alignment check
    tag_alignment_ms: Optional[float] = None  # Log tag - WAV tag (should be ~13ms)
    # Flags
    has_tool_call: bool = False


@dataclass
class AlignmentStats:
    """Statistics for tag alignment sanity check."""
    bot_tags_log: int = 0
    bot_tags_wav: int = 0
    user_tags_log: int = 0
    user_tags_wav: int = 0
    matched_bot_tags: int = 0
    matched_user_tags: int = 0
    bot_alignment_min_ms: Optional[float] = None
    bot_alignment_max_ms: Optional[float] = None
    bot_alignment_mean_ms: Optional[float] = None
    user_alignment_min_ms: Optional[float] = None
    user_alignment_max_ms: Optional[float] = None
    user_alignment_mean_ms: Optional[float] = None
    alignment_ok: bool = True
    issues: list = field(default_factory=list)


def load_transcript(run_dir: Path) -> dict[int, dict]:
    """Load transcript.jsonl and return dict mapping turn index to entry."""
    transcript_path = run_dir / "transcript.jsonl"
    turns = {}
    if transcript_path.exists():
        with open(transcript_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    turn_idx = entry.get("turn", -1)
                    turns[turn_idx] = entry
    return turns


def parse_bot_tags_from_log(log_path: Path) -> list[dict]:
    """Parse bot turn tag positions from run.log."""
    tags = []
    pattern = r"Bot turn tag: sample_pos=(\d+)ms"
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    tags.append({"sample_pos_ms": int(match.group(1))})
    return tags


def parse_user_tags_from_log(log_path: Path) -> list[dict]:
    """Parse user turn tag positions from run.log."""
    tags = []
    pattern = r"User turn tag: sample_pos=(\d+)ms"
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    tags.append({"sample_pos_ms": int(match.group(1))})
    return tags


def parse_rms_onsets_from_log(log_path: Path) -> list[dict]:
    """Parse bot speech onset (RMS) positions from run.log."""
    onsets = []
    pattern = r"Bot speech onset: T\+\d+ms \(sample_pos=(\d+)ms, silent_padding=(\d+)ms, rms=([0-9.-]+)dB\)"
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    onsets.append({
                        "sample_pos_ms": int(match.group(1)),
                        "silent_padding_ms": int(match.group(2)),
                        "rms_db": float(match.group(3)),
                    })
    return onsets


def detect_tags_in_wav(
    wav_path: Path,
    channel: int,
    freq_hz: int = 2000,
    threshold_ratio: float = 20.0,
    min_level_db: float = -40.0,
) -> list[int]:
    """Detect audio tags in WAV file, return positions in ms."""
    with wave.open(str(wav_path), "rb") as wf:
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
    window_ms = 20
    hop_ms = 5
    window_samples = int(sr * window_ms / 1000)
    hop_samples = int(sr * hop_ms / 1000)
    min_gap_ms = 50

    tags = []
    for i in range(0, len(track) - window_samples, hop_samples):
        window = track[i : i + window_samples]
        spectrum = np.abs(fft(window * np.hanning(len(window))))
        freq_resolution = sr / len(spectrum)
        target_bin = int(freq_hz / freq_resolution)
        bin_range = max(1, int(100 / freq_resolution))
        energy_at_freq = np.max(spectrum[max(0, target_bin - bin_range) : target_bin + bin_range + 1])
        avg_energy = np.mean(spectrum[1 : len(spectrum) // 2])

        if avg_energy > 0:
            energy_ratio = energy_at_freq / avg_energy
        else:
            energy_ratio = 0

        if energy_ratio > threshold_ratio:
            rms = np.sqrt(np.mean(window**2))
            rms_db = 20 * np.log10(rms + 1e-10)
            if rms_db < min_level_db:
                continue
            pos_ms = int(i * 1000 / sr)
            if not tags or pos_ms - tags[-1] > min_gap_ms:
                tags.append(pos_ms)

    return tags


def load_silero_vad():
    """Load Silero VAD model."""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=True,
        trust_repo=True,
        verbose=False,
    )
    return model, utils[0]


def run_silero_vad(
    wav_path: Path,
    model,
    get_speech_timestamps,
    min_silence_ms: int = 2000,
    min_speech_ms: int = 750,
) -> tuple[list[dict], list[dict]]:
    """Run Silero VAD on both channels, return (user_segments, bot_segments)."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).reshape(-1, 2)
    user_audio = audio[:, 0].astype(np.float32) / 32768.0
    bot_audio = audio[:, 1].astype(np.float32) / 32768.0

    # Resample to 16kHz
    target_sr = 16000
    user_16k = signal.resample(user_audio, int(len(user_audio) * target_sr / sr))
    bot_16k = signal.resample(bot_audio, int(len(bot_audio) * target_sr / sr))

    user_segments = get_speech_timestamps(
        torch.from_numpy(user_16k.astype(np.float32)),
        model, sampling_rate=target_sr,
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=min_speech_ms,
        threshold=0.7, speech_pad_ms=0,
    )
    bot_segments = get_speech_timestamps(
        torch.from_numpy(bot_16k.astype(np.float32)),
        model, sampling_rate=target_sr,
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=min_speech_ms,
        threshold=0.7, speech_pad_ms=0,
    )

    # Convert to ms
    user_segs = [{"start_ms": s['start'] / 16.0, "end_ms": s['end'] / 16.0} for s in user_segments]
    bot_segs = [{"start_ms": s['start'] / 16.0, "end_ms": s['end'] / 16.0} for s in bot_segments]

    return user_segs, bot_segs


def match_tags_by_proximity(
    log_tags: list[dict],
    wav_tags: list[int],
    max_distance_ms: int = 100,
) -> tuple[list[tuple[int, int, float]], list[int]]:
    """Match log tags to WAV tags by proximity, filtering false positives.

    Args:
        log_tags: List of dicts with 'sample_pos_ms' from logs
        wav_tags: List of positions in ms from WAV detection
        max_distance_ms: Maximum distance to consider a match

    Returns:
        Tuple of (matches, unmatched_wav_indices) where matches is list of
        (log_idx, wav_idx, diff_ms) tuples.
    """
    matches = []
    used_wav = set()

    for log_idx, log_tag in enumerate(log_tags):
        log_pos = log_tag["sample_pos_ms"]
        best_wav_idx = None
        best_diff = float('inf')

        for wav_idx, wav_pos in enumerate(wav_tags):
            if wav_idx in used_wav:
                continue
            diff = log_pos - wav_pos
            if abs(diff) < abs(best_diff) and abs(diff) <= max_distance_ms:
                best_diff = diff
                best_wav_idx = wav_idx

        if best_wav_idx is not None:
            matches.append((log_idx, best_wav_idx, best_diff))
            used_wav.add(best_wav_idx)

    unmatched = [i for i in range(len(wav_tags)) if i not in used_wav]
    return matches, unmatched


def check_alignment(
    bot_tags_log: list[dict],
    bot_tags_wav: list[int],
    user_tags_log: list[dict],
    user_tags_wav: list[int],
    tolerance_ms: int = 20,
) -> tuple[AlignmentStats, dict[int, int], dict[int, int]]:
    """Check alignment between log tag positions and WAV detected positions.

    Returns:
        Tuple of (stats, bot_log_to_wav_map, user_log_to_wav_map)
        Maps are {log_idx: wav_idx} for matched tags.
    """
    stats = AlignmentStats(
        bot_tags_log=len(bot_tags_log),
        bot_tags_wav=len(bot_tags_wav),
        user_tags_log=len(user_tags_log),
        user_tags_wav=len(user_tags_wav),
    )

    # Match bot tags by proximity
    bot_matches, bot_unmatched = match_tags_by_proximity(bot_tags_log, bot_tags_wav)
    bot_log_to_wav = {m[0]: m[1] for m in bot_matches}
    bot_diffs = [m[2] for m in bot_matches]

    if bot_diffs:
        stats.matched_bot_tags = len(bot_diffs)
        stats.bot_alignment_min_ms = min(bot_diffs)
        stats.bot_alignment_max_ms = max(bot_diffs)
        stats.bot_alignment_mean_ms = sum(bot_diffs) / len(bot_diffs)
        if stats.bot_alignment_max_ms > tolerance_ms or stats.bot_alignment_min_ms < -tolerance_ms:
            stats.alignment_ok = False
            stats.issues.append(f"Bot tag alignment outside ±{tolerance_ms}ms: {stats.bot_alignment_min_ms:.0f} to {stats.bot_alignment_max_ms:.0f}ms")

    if bot_unmatched:
        stats.issues.append(f"WAV detected {len(bot_unmatched)} unmatched bot tags at positions: {[bot_tags_wav[i] for i in bot_unmatched[:5]]}{'...' if len(bot_unmatched) > 5 else ''}")

    if len(bot_matches) < len(bot_tags_log):
        stats.alignment_ok = False
        stats.issues.append(f"Missing bot tags: {len(bot_tags_log) - len(bot_matches)} log tags not found in WAV")

    # Match user tags by proximity
    user_matches, user_unmatched = match_tags_by_proximity(user_tags_log, user_tags_wav)
    user_log_to_wav = {m[0]: m[1] for m in user_matches}
    user_diffs = [m[2] for m in user_matches]

    if user_diffs:
        stats.matched_user_tags = len(user_diffs)
        stats.user_alignment_min_ms = min(user_diffs)
        stats.user_alignment_max_ms = max(user_diffs)
        stats.user_alignment_mean_ms = sum(user_diffs) / len(user_diffs)
        if stats.user_alignment_max_ms > tolerance_ms or stats.user_alignment_min_ms < -tolerance_ms:
            stats.alignment_ok = False
            stats.issues.append(f"User tag alignment outside ±{tolerance_ms}ms: {stats.user_alignment_min_ms:.0f} to {stats.user_alignment_max_ms:.0f}ms")

    if user_unmatched:
        stats.issues.append(f"WAV detected {len(user_unmatched)} unmatched user tags at positions: {[user_tags_wav[i] for i in user_unmatched[:5]]}{'...' if len(user_unmatched) > 5 else ''}")

    if len(user_matches) < len(user_tags_log):
        stats.alignment_ok = False
        stats.issues.append(f"Missing user tags: {len(user_tags_log) - len(user_matches)} log tags not found in WAV")

    return stats, bot_log_to_wav, user_log_to_wav


def analyze_run(run_dir: Path) -> tuple[list[TurnMetrics], AlignmentStats, dict]:
    """Analyze a run directory and return per-turn metrics."""
    wav_path = run_dir / "conversation.wav"
    log_path = run_dir / "run.log"

    if not wav_path.exists():
        raise FileNotFoundError(f"conversation.wav not found in {run_dir}")

    # Load all data sources
    print("Loading transcript...", file=sys.stderr)
    transcript = load_transcript(run_dir)

    print("Parsing log file...", file=sys.stderr)
    bot_tags_log = parse_bot_tags_from_log(log_path)
    user_tags_log = parse_user_tags_from_log(log_path)
    rms_onsets = parse_rms_onsets_from_log(log_path)

    print("Detecting tags in WAV...", file=sys.stderr)
    bot_tags_wav = detect_tags_in_wav(wav_path, channel=1)  # Right channel
    # Only detect user tags if present in log (user tags removed in newer runs)
    user_tags_wav = detect_tags_in_wav(wav_path, channel=0) if user_tags_log else []

    print("Running Silero VAD...", file=sys.stderr)
    model, get_speech_timestamps = load_silero_vad()
    user_segments, bot_segments = run_silero_vad(wav_path, model, get_speech_timestamps)

    # Check alignment (now returns mapping dictionaries)
    print("Checking alignment...", file=sys.stderr)
    alignment, bot_log_to_wav, user_log_to_wav = check_alignment(
        bot_tags_log, bot_tags_wav, user_tags_log, user_tags_wav
    )

    # Build per-turn metrics
    # Use number of bot tags as turn count (each tag = one turn)
    n_turns = len(bot_tags_log)
    turns = []
    prev_bot_seg_end: Optional[float] = None  # Track previous bot segment end for constraining search

    for i in range(n_turns):
        m = TurnMetrics(turn_index=i)

        # Server TTFB from transcript
        if i in transcript:
            m.server_ttfb_ms = transcript[i].get("ttfb_ms")
            m.has_tool_call = len(transcript[i].get("tool_calls", [])) > 0

        # Bot tag from log
        if i < len(bot_tags_log):
            m.bot_tag_log_ms = bot_tags_log[i]["sample_pos_ms"]

        # Bot tag from WAV (use proximity-matched index, excluding false positives)
        if i in bot_log_to_wav:
            wav_idx = bot_log_to_wav[i]
            m.bot_tag_wav_ms = bot_tags_wav[wav_idx]
            m.tag_alignment_ms = m.bot_tag_log_ms - m.bot_tag_wav_ms

        # RMS onset from log
        if i < len(rms_onsets):
            m.bot_rms_onset_ms = rms_onsets[i]["sample_pos_ms"]

        # Bot speech start from Silero - find segment closest to bot tag
        if m.bot_tag_log_ms is not None:
            # Find bot segment that starts closest to the tag position
            # Allow up to 500ms before tag (Silero may detect speech before the tag)
            best_bot_seg = None
            best_diff = float('inf')
            for seg in bot_segments:
                diff = seg["start_ms"] - m.bot_tag_log_ms
                if -500 <= diff < best_diff:
                    best_diff = diff
                    best_bot_seg = seg
            if best_bot_seg is not None:
                m.bot_silero_start_ms = best_bot_seg["start_ms"]
                m.bot_silero_end_ms = best_bot_seg["end_ms"]

        # User speech end from Silero - find user segment that ends just before bot starts
        # Constrain search: user segment must end AFTER previous bot segment ended
        # This prevents picking user segments from earlier turns when Silero misses a segment
        if m.bot_silero_start_ms is not None:
            best_user_seg = None
            best_diff = float('inf')
            for seg in user_segments:
                # Skip segments that ended before/during the previous bot segment
                if prev_bot_seg_end is not None and seg["end_ms"] <= prev_bot_seg_end:
                    continue
                diff = m.bot_silero_start_ms - seg["end_ms"]
                if 0 <= diff < best_diff:
                    best_diff = diff
                    best_user_seg = seg
            if best_user_seg is not None:
                m.user_start_ms = best_user_seg["start_ms"]
                m.user_end_ms = best_user_seg["end_ms"]

        # Calculate derived metrics
        if m.bot_tag_log_ms is not None and m.user_end_ms is not None:
            m.pipeline_ttfb_ms = m.bot_tag_log_ms - m.user_end_ms

        if m.bot_silero_start_ms is not None and m.user_end_ms is not None:
            m.wav_v2v_ms = m.bot_silero_start_ms - m.user_end_ms

        if m.bot_rms_onset_ms is not None and m.bot_tag_log_ms is not None:
            m.silent_pad_rms_ms = m.bot_rms_onset_ms - m.bot_tag_log_ms

        if m.bot_silero_start_ms is not None and m.bot_tag_wav_ms is not None:
            m.silent_pad_silero_ms = m.bot_silero_start_ms - m.bot_tag_wav_ms

        # Update prev_bot_seg_end for next iteration
        if m.bot_silero_end_ms is not None:
            prev_bot_seg_end = m.bot_silero_end_ms

        turns.append(m)

    # Build summary stats
    summary = {
        "run_dir": str(run_dir),
        "num_turns": n_turns,
        "user_segments": len(user_segments),
        "bot_segments": len(bot_segments),
    }

    # Calculate stats for each metric
    for metric_name in ["server_ttfb_ms", "pipeline_ttfb_ms", "wav_v2v_ms", "silent_pad_rms_ms", "silent_pad_silero_ms"]:
        values = [getattr(t, metric_name) for t in turns if getattr(t, metric_name) is not None]
        if values:
            summary[f"{metric_name}_median"] = float(np.median(values))
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_min"] = float(np.min(values))
            summary[f"{metric_name}_max"] = float(np.max(values))

    return turns, alignment, summary


def print_results(turns: list[TurnMetrics], alignment: AlignmentStats, summary: dict, verbose: bool = False):
    """Print results in human-readable format."""
    print("=" * 90)
    print("COMPREHENSIVE TURN METRICS ANALYSIS")
    print("=" * 90)
    print(f"Run: {summary['run_dir']}")
    print(f"Turns: {summary['num_turns']}, User segments: {summary['user_segments']}, Bot segments: {summary['bot_segments']}")

    # Alignment check
    print()
    print("-" * 90)
    print("ALIGNMENT SANITY CHECK")
    print("-" * 90)
    if alignment.alignment_ok:
        print("✅ Alignment OK")
    else:
        print("❌ Alignment issues detected")

    print(f"  Bot tags:  {alignment.bot_tags_log} in log, {alignment.bot_tags_wav} in WAV, {alignment.matched_bot_tags} matched")
    if alignment.bot_alignment_mean_ms is not None:
        print(f"  Bot alignment: {alignment.bot_alignment_min_ms:.0f}ms to {alignment.bot_alignment_max_ms:.0f}ms (mean: {alignment.bot_alignment_mean_ms:.1f}ms)")

    print(f"  User tags: {alignment.user_tags_log} in log, {alignment.user_tags_wav} in WAV, {alignment.matched_user_tags} matched")
    if alignment.user_alignment_mean_ms is not None:
        print(f"  User alignment: {alignment.user_alignment_min_ms:.0f}ms to {alignment.user_alignment_max_ms:.0f}ms (mean: {alignment.user_alignment_mean_ms:.1f}ms)")

    if alignment.issues:
        print("  Notes:")
        for issue in alignment.issues:
            print(f"    - {issue}")

    # Per-turn table (verbose mode)
    if verbose:
        print()
        print("-" * 105)
        print("PER-TURN METRICS")
        print("-" * 105)
        print(f"{'Turn':>4} | {'Start Time':>10} | {'Srvr TTFB':>9} | {'Pipe TTFB':>9} | {'WAV V2V':>9} | {'Pad RMS':>8} | {'Pad VAD':>8} | {'Align':>6}")
        print(f"{'-'*4}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
        for t in turns:
            if t.user_start_ms is not None:
                total_secs = t.user_start_ms / 1000
                mins = int(total_secs // 60)
                secs = int(total_secs % 60)
                ts = f"{mins:02d}:{secs:02d}".rjust(10)
            else:
                ts = "N/A".rjust(10)
            srv = f"{t.server_ttfb_ms:>7}ms" if t.server_ttfb_ms is not None else "     N/A"
            pipe = f"{t.pipeline_ttfb_ms:>7.0f}ms" if t.pipeline_ttfb_ms is not None else "     N/A"
            v2v = f"{t.wav_v2v_ms:>7.0f}ms" if t.wav_v2v_ms is not None else "     N/A"
            rms = f"{t.silent_pad_rms_ms:>6.0f}ms" if t.silent_pad_rms_ms is not None else "    N/A"
            sil = f"{t.silent_pad_silero_ms:>6.0f}ms" if t.silent_pad_silero_ms is not None else "    N/A"
            aln = f"{t.tag_alignment_ms:>4.0f}ms" if t.tag_alignment_ms is not None else "  N/A"
            tool = " [T]" if t.has_tool_call else ""
            print(f"{t.turn_index:>4} | {ts} | {srv} | {pipe} | {v2v} | {rms} | {sil} | {aln}{tool}")

    # Summary statistics
    print()
    print("-" * 90)
    print("SUMMARY STATISTICS")
    print("-" * 90)
    metrics = [
        ("Server TTFB", "server_ttfb_ms"),
        ("Pipeline TTFB", "pipeline_ttfb_ms"),
        ("WAV V2V", "wav_v2v_ms"),
        ("Silent Pad (RMS)", "silent_pad_rms_ms"),
        ("Silent Pad (VAD)", "silent_pad_silero_ms"),
    ]
    print(f"{'Metric':<20} | {'Median':>10} | {'Mean':>10} | {'Min':>10} | {'Max':>10}")
    print(f"{'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for label, key in metrics:
        if f"{key}_median" in summary:
            print(f"{label:<20} | {summary[f'{key}_median']:>8.0f}ms | {summary[f'{key}_mean']:>8.0f}ms | {summary[f'{key}_min']:>8.0f}ms | {summary[f'{key}_max']:>8.0f}ms")
        else:
            print(f"{label:<20} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive per-turn metrics analysis for benchmark runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script consolidates all timing metrics:
- Server TTFB (from transcript.jsonl)
- Pipeline TTFB (bot audio arrival - user speech end)
- WAV V2V (Silero bot start - Silero user end)
- Silent Padding RMS (RMS onset - bot tag)
- Silent Padding Silero (Silero start - WAV tag)

It also verifies log/WAV alignment using audio tags.
        """,
    )
    parser.add_argument("run_dir", type=Path, help="Path to the run directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-turn breakdown")

    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Error: Run directory not found: {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        turns, alignment, summary = analyze_run(args.run_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        output = {
            "summary": summary,
            "alignment": {
                "bot_tags_log": alignment.bot_tags_log,
                "bot_tags_wav": alignment.bot_tags_wav,
                "user_tags_log": alignment.user_tags_log,
                "user_tags_wav": alignment.user_tags_wav,
                "matched_bot_tags": alignment.matched_bot_tags,
                "matched_user_tags": alignment.matched_user_tags,
                "bot_alignment_min_ms": alignment.bot_alignment_min_ms,
                "bot_alignment_max_ms": alignment.bot_alignment_max_ms,
                "bot_alignment_mean_ms": alignment.bot_alignment_mean_ms,
                "user_alignment_min_ms": alignment.user_alignment_min_ms,
                "user_alignment_max_ms": alignment.user_alignment_max_ms,
                "user_alignment_mean_ms": alignment.user_alignment_mean_ms,
                "alignment_ok": alignment.alignment_ok,
                "issues": alignment.issues,
            },
            "turns": [
                {
                    "turn": t.turn_index,
                    "server_ttfb_ms": t.server_ttfb_ms,
                    "pipeline_ttfb_ms": t.pipeline_ttfb_ms,
                    "wav_v2v_ms": t.wav_v2v_ms,
                    "silent_pad_rms_ms": t.silent_pad_rms_ms,
                    "silent_pad_silero_ms": t.silent_pad_silero_ms,
                    "tag_alignment_ms": t.tag_alignment_ms,
                    "has_tool_call": t.has_tool_call,
                    "user_start_ms": t.user_start_ms,
                    "user_end_ms": t.user_end_ms,
                    "bot_tag_log_ms": t.bot_tag_log_ms,
                    "bot_tag_wav_ms": t.bot_tag_wav_ms,
                    "bot_rms_onset_ms": t.bot_rms_onset_ms,
                    "bot_silero_start_ms": t.bot_silero_start_ms,
                    "bot_silero_end_ms": t.bot_silero_end_ms,
                }
                for t in turns
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print_results(turns, alignment, summary, verbose=args.verbose)


if __name__ == "__main__":
    main()
