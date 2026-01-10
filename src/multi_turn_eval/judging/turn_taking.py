"""Turn-taking analysis for audio timing anomaly detection.

This module analyzes audio timing metrics to detect turn-taking failures
that may cause garbled transcriptions or other content issues.

Turn-taking failures are detected when:
1. Missing timing data: pipeline_ttfb_ms is null (no user speech detected)
2. Negative TTFB: pipeline_ttfb_ms < 0 (bot started before user finished)
3. Severe alignment drift: tag_alignment_ms outside ±100ms tolerance
4. Missing bot detection: bot_tag_wav_ms is null (log tag not found in WAV)
5. Anomalous silent padding: silent_pad_silero_ms > 5000ms
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Thresholds for detecting turn-taking failures
ALIGNMENT_TOLERANCE_MS = 100  # ±100ms
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

    def to_dict(self) -> dict:
        return {
            "run_dir": self.run_dir,
            "turn_taking_ok": self.overall_ok,
            "failed_turns": self.failed_turns,
            "per_turn": {k: v.to_dict() for k, v in self.per_turn.items()},
            "error": self.error,
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


def detect_turn_taking_issues(turn_data: dict) -> list[str]:
    """Detect turn-taking issues for a single turn.

    Args:
        turn_data: Dict with turn metrics from analyze_turn_metrics.py

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

    return issues


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

    # Analyze each turn
    turns_data = metrics.get("turns", [])
    for turn_data in turns_data:
        turn_idx = turn_data.get("turn", -1)
        if turn_idx < 0:
            continue

        issues = detect_turn_taking_issues(turn_data)
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
        if result.failed_turns:
            print(f"Failed turns: {result.failed_turns}")
            for turn_idx in result.failed_turns:
                turn_result = result.per_turn[turn_idx]
                print(f"  Turn {turn_idx}: {', '.join(turn_result.issues)}")


if __name__ == "__main__":
    main()
