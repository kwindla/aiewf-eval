"""Turn-taking analysis for audio timing anomaly detection.

This module analyzes audio timing metrics to detect turn-taking failures
that may cause garbled transcriptions or other content issues.

Turn-taking failures are detected when:
1. Missing timing data: pipeline_ttfb_ms is null (no user speech detected)
2. Negative TTFB: pipeline_ttfb_ms < 0 (bot started before user finished)
3. Severe alignment drift: tag_alignment_ms outside ±100ms tolerance
4. Missing bot detection: bot_tag_wav_ms is null (log tag not found in WAV)
5. Anomalous silent padding: silent_pad_silero_ms > 5000ms
6. Audio overlap: user and bot speaking simultaneously (per-turn)
7. Empty response: model returned control tokens only, no actual speech
8. No response: model never responded at all (no TTS within 15s timeout)
9. Reconnection: session timeout forced reconnect mid-turn
10. Greeting timeout: bot started greeting but never completed (30s timeout)

Global issues (not per-turn):
- Audio overlaps detected across all segments
- Unmatched bot segments (orphan responses not associated with any turn)
- Unprompted bot responses (bot speaking without recent user speech)
- Greeting timeout (affects turn 0 timing)
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Thresholds for detecting turn-taking failures
ALIGNMENT_TOLERANCE_MS = 150  # ±150ms (accounts for MediaSender buffering delays)
MAX_SILENT_PAD_MS = 5000  # 5 seconds


@dataclass
class TurnTakingResult:
    """Result of turn-taking analysis for a single turn."""
    turn_index: int
    turn_taking_ok: bool = True
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "turn": self.turn_index,
            "turn_taking": self.turn_taking_ok,
            "issues": self.issues,
        }


@dataclass
class TurnTakingAnalysis:
    """Full turn-taking analysis result."""
    run_dir: str
    overall_ok: bool = True
    failed_turns: list[int] = field(default_factory=list)
    per_turn: dict[int, TurnTakingResult] = field(default_factory=dict)
    error: Optional[str] = None
    # Global issues (not per-turn)
    global_issues: list[str] = field(default_factory=list)
    overlaps: list[dict] = field(default_factory=list)
    unmatched_bot_segments: list[dict] = field(default_factory=list)
    unprompted_bot_segments: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_dir": self.run_dir,
            "turn_taking_ok": self.overall_ok,
            "failed_turns": self.failed_turns,
            "per_turn": {k: v.to_dict() for k, v in self.per_turn.items()},
            "error": self.error,
            "global_issues": self.global_issues,
            "overlaps": self.overlaps,
            "unmatched_bot_segments": self.unmatched_bot_segments,
            "unprompted_bot_segments": self.unprompted_bot_segments,
        }


def analyze_turn_metrics_json(run_dir: Path) -> Optional[dict]:
    """Run analyze_turn_metrics.py and return JSON output.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Parsed JSON output from analyze_turn_metrics.py, or None if failed.
    """
    wav_path = run_dir / "conversation.wav"
    if not wav_path.exists():
        return None

    try:
        result = subprocess.run(
            ["uv", "run", "python", "scripts/analyze_turn_metrics.py", str(run_dir), "--json"],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout for VAD processing
        )
        if result.returncode != 0:
            print(f"analyze_turn_metrics.py failed: {result.stderr}", file=sys.stderr)
            return None

        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        print("analyze_turn_metrics.py timed out", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from analyze_turn_metrics.py: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error running analyze_turn_metrics.py: {e}", file=sys.stderr)
        return None


def detect_turn_taking_issues(turn_data: dict, overlaps: list[dict] = None) -> list[str]:
    """Detect turn-taking issues for a single turn.

    Args:
        turn_data: Dict with turn metrics from analyze_turn_metrics.py
        overlaps: List of overlap dicts from analyze_turn_metrics.py (optional)

    Returns:
        List of issue descriptions (empty if no issues).
    """
    issues = []

    # 1. Missing timing data (no user speech detected)
    pipeline_ttfb = turn_data.get("pipeline_ttfb_ms")
    if pipeline_ttfb is None:
        issues.append("missing_timing_data")

    # 2. Negative TTFB (bot started before user finished)
    if pipeline_ttfb is not None and pipeline_ttfb < 0:
        issues.append(f"negative_ttfb ({pipeline_ttfb:.0f}ms)")

    # 3. Severe alignment drift
    alignment = turn_data.get("tag_alignment_ms")
    if alignment is not None and abs(alignment) > ALIGNMENT_TOLERANCE_MS:
        issues.append(f"alignment_drift ({alignment:.0f}ms)")

    # 4. Missing bot detection in WAV
    bot_tag_wav = turn_data.get("bot_tag_wav_ms")
    if bot_tag_wav is None and turn_data.get("bot_tag_log_ms") is not None:
        issues.append("missing_bot_wav_tag")

    # 5. Anomalous silent padding (VAD detected speech far from tag)
    silent_pad = turn_data.get("silent_pad_silero_ms")
    if silent_pad is not None and silent_pad > MAX_SILENT_PAD_MS:
        issues.append(f"high_silent_pad ({silent_pad:.0f}ms)")

    # 6. Audio overlap affecting this turn
    if overlaps:
        turn_start = turn_data.get("user_start_ms")
        turn_end = turn_data.get("bot_silero_end_ms")
        if turn_start is not None and turn_end is not None:
            for overlap in overlaps:
                overlap_start = overlap.get("overlap_start_ms", 0)
                overlap_end = overlap.get("overlap_end_ms", 0)
                overlap_ms = overlap.get("overlap_ms", 0)
                # Check if overlap falls within this turn's time range
                if overlap_start >= turn_start and overlap_end <= turn_end:
                    issues.append(f"audio_overlap ({overlap_ms:.0f}ms)")
                    break  # Only flag once per turn

    # 7. Empty response (model returned control tokens only, no actual speech)
    # 8. No response (model never responded at all)
    # 9. Reconnection (session timeout forced reconnect mid-turn, but not turn 0)
    retry_reasons = turn_data.get("retry_reasons", [])
    turn_index = turn_data.get("turn", -1)
    if retry_reasons:
        empty_count = sum(1 for r in retry_reasons if r == "empty_response")
        no_response_count = sum(1 for r in retry_reasons if r == "no_response")
        reconnection_count = sum(1 for r in retry_reasons if r == "reconnection")
        if empty_count > 0:
            issues.append(f"empty_response ({empty_count} retries)")
        if no_response_count > 0:
            issues.append(f"no_response ({no_response_count} retries)")
        # Turn 0 reconnections are expected (initial connection setup)
        if reconnection_count > 0 and turn_index != 0:
            issues.append(f"reconnection ({reconnection_count} retries)")

    return issues


def detect_greeting_timeout(run_dir: Path) -> bool:
    """Check if a greeting timeout occurred by parsing run.log.

    A greeting timeout occurs when the bot started speaking its initial
    greeting but never completed within the timeout period (30s).

    Args:
        run_dir: Path to the run directory containing run.log

    Returns:
        True if a greeting timeout was detected, False otherwise.
    """
    log_path = run_dir / "run.log"
    if not log_path.exists():
        return False

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                # Look for the specific TURN_FAILURE log message for greeting timeout
                if "[TURN_FAILURE] Greeting did not complete" in line:
                    return True
    except Exception as e:
        print(f"Error reading run.log for greeting timeout: {e}", file=sys.stderr)

    return False


def analyze_turn_taking(run_dir: Path) -> TurnTakingAnalysis:
    """Analyze turn-taking from audio timing metrics.

    This function runs analyze_turn_metrics.py on the run directory
    and detects which turns have timing anomalies that may indicate
    turn-taking failures.

    Args:
        run_dir: Path to the run directory containing conversation.wav

    Returns:
        TurnTakingAnalysis with per-turn results and overall status.
    """
    result = TurnTakingAnalysis(run_dir=str(run_dir))

    # Check if WAV file exists
    wav_path = run_dir / "conversation.wav"
    if not wav_path.exists():
        result.error = "No conversation.wav file found"
        return result

    # Run timing analysis
    metrics = analyze_turn_metrics_json(run_dir)
    if metrics is None:
        result.error = "Failed to analyze turn metrics"
        return result

    # Extract global issues from summary
    summary = metrics.get("summary", {})
    overlaps = summary.get("overlaps", [])
    unmatched_segs = summary.get("unmatched_bot_segments", [])
    unprompted_segs = summary.get("unprompted_bot_segments", [])

    # Store raw data
    result.overlaps = overlaps
    result.unmatched_bot_segments = unmatched_segs
    result.unprompted_bot_segments = unprompted_segs

    # Build global issues list
    if overlaps:
        total_overlap = sum(o.get("overlap_ms", 0) for o in overlaps)
        result.global_issues.append(f"audio_overlap: {len(overlaps)} instances, {total_overlap:.0f}ms total")
        result.overall_ok = False

    if unmatched_segs:
        result.global_issues.append(f"unmatched_bot_segments: {len(unmatched_segs)} orphan segments")
        result.overall_ok = False

    if unprompted_segs:
        result.global_issues.append(f"unprompted_bot_responses: {len(unprompted_segs)} segments without user trigger")
        result.overall_ok = False

    # Check for greeting timeout (affects turn 0)
    greeting_timeout = detect_greeting_timeout(run_dir)
    if greeting_timeout:
        result.global_issues.append("greeting_timeout: bot started greeting but never completed (30s timeout)")
        result.overall_ok = False

    # Analyze each turn
    turns_data = metrics.get("turns", [])
    for turn_data in turns_data:
        turn_idx = turn_data.get("turn", -1)
        if turn_idx < 0:
            continue

        issues = detect_turn_taking_issues(turn_data, overlaps=overlaps)

        # Add greeting_timeout issue to turn 0 if it occurred
        if turn_idx == 0 and greeting_timeout:
            issues.append("greeting_timeout")

        turn_result = TurnTakingResult(
            turn_index=turn_idx,
            turn_taking_ok=len(issues) == 0,
            issues=issues,
        )
        result.per_turn[turn_idx] = turn_result

        if not turn_result.turn_taking_ok:
            result.failed_turns.append(turn_idx)
            result.overall_ok = False

    result.failed_turns.sort()
    return result


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze turn-taking from audio timing")
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = analyze_turn_taking(args.run_dir)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Run: {result.run_dir}")
        print(f"Overall OK: {result.overall_ok}")
        if result.error:
            print(f"Error: {result.error}")
        if result.global_issues:
            print("Global issues:")
            for issue in result.global_issues:
                print(f"  - {issue}")
        if result.overlaps:
            print(f"Overlaps ({len(result.overlaps)}):")
            for o in result.overlaps:
                print(f"  - {o['overlap_start_ms']:.0f}-{o['overlap_end_ms']:.0f}ms ({o['overlap_ms']:.0f}ms)")
        if result.unmatched_bot_segments:
            print(f"Unmatched bot segments ({len(result.unmatched_bot_segments)}):")
            for seg in result.unmatched_bot_segments:
                print(f"  - {seg['start_ms']:.0f}-{seg['end_ms']:.0f}ms ({seg['duration_ms']:.0f}ms)")
        if result.unprompted_bot_segments:
            print(f"Unprompted bot segments ({len(result.unprompted_bot_segments)}):")
            for seg in result.unprompted_bot_segments:
                gap = seg.get('gap_from_last_user_ms')
                gap_str = f", gap={gap:.0f}ms" if gap is not None else ", no preceding user"
                print(f"  - {seg['start_ms']:.0f}-{seg['end_ms']:.0f}ms ({seg['duration_ms']:.0f}ms{gap_str})")
        if result.failed_turns:
            print(f"Failed turns: {result.failed_turns}")
            for turn_idx in result.failed_turns:
                turn_result = result.per_turn[turn_idx]
                print(f"  Turn {turn_idx}: {', '.join(turn_result.issues)}")


if __name__ == "__main__":
    main()
