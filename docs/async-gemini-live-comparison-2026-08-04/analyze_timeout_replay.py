#!/usr/bin/env python3
"""Compare Gemini Live timeout and replay behavior across frozen cohorts."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CURRENT_PREVIEW = Path(
    "runs/aiwf_medium_context/"
    "20260804T175455_gemini-3.1-flash-live-preview_f79bd81e"
)
BOOTSTRAPS = 200_000
SEED = 20260804

TIMESTAMP_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)")
TIMEOUT_RE = re.compile(r"\[NO_RESPONSE\] turn=(\d+) retry_count=(\d+)")
FORCE_RE = re.compile(r"Force advancing past turn (\d+)")
REQUEUE_RE = re.compile(r"Re-queuing audio for turn (\d+)")
TTFB_RE = re.compile(r" TTFB: ([\d.]+)s")
TTS_STARTED_MARKER = "[LLM→] TTSStartedFrame"


def timestamp(line: str) -> float | None:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()


def dedupe_timestamps(values: list[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        if not result or value - result[-1] > 0.01:
            result.append(value)
    return result


def primary_turn_count(run_dir: Path) -> int:
    transcript = run_dir / "transcript.jsonl"
    if not transcript.is_file():
        return 0
    return sum(
        not json.loads(line).get("recovery_turn")
        for line in transcript.read_text().splitlines()
        if line.strip()
    )


def judged_turns(run_dir: Path) -> list[dict]:
    judged = run_dir / "claude_judged.jsonl"
    if not judged.is_file():
        return []
    result = []
    for line in judged.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("recovery_turn") or not isinstance(row.get("turn"), int):
            continue
        scores = row.get("scores") or {}
        result.append(
            {
                "turn": row["turn"],
                "strict": all(
                    scores.get(key) is True
                    for key in (
                        "tool_use_correct",
                        "instruction_following",
                        "kb_grounding",
                    )
                ),
                **{
                    key: scores.get(key) is True
                    for key in (
                        "tool_use_correct",
                        "instruction_following",
                        "kb_grounding",
                        "turn_taking",
                    )
                },
            }
        )
    return result


def analyze_run(run_dir: Path) -> dict:
    lines = (run_dir / "run.log").read_text(errors="replace").splitlines()
    events = [
        (event_time, index, line)
        for index, line in enumerate(lines)
        if (event_time := timestamp(line)) is not None
    ]

    timeouts = []
    forced = []
    requeues = []
    ttfbs = []
    tts_starts = []
    starts = []
    stops = []
    for event_time, index, line in events:
        if match := TIMEOUT_RE.search(line):
            timeouts.append(
                {
                    "time": event_time,
                    "turn": int(match.group(1)),
                    "retry_count": int(match.group(2)),
                    "line": index + 1,
                }
            )
        if match := FORCE_RE.search(line):
            forced.append(
                {"time": event_time, "turn": int(match.group(1)), "line": index + 1}
            )
        if match := REQUEUE_RE.search(line):
            requeues.append(
                {"time": event_time, "turn": int(match.group(1)), "line": index + 1}
            )
        if match := TTFB_RE.search(line):
            ttfbs.append(
                {"time": event_time, "seconds": float(match.group(1)), "line": index + 1}
            )
        if TTS_STARTED_MARKER in line:
            tts_starts.append({"time": event_time, "line": index + 1})
        if "[VAD] UserStartedSpeaking" in line:
            starts.append(event_time)
        if "[VAD] UserStoppedSpeaking" in line:
            stops.append(event_time)

    starts = dedupe_timestamps(starts)
    stops = dedupe_timestamps(stops)
    replay_windows = []
    for replay in requeues:
        start = next(
            (value for value in starts if replay["time"] <= value <= replay["time"] + 4),
            None,
        )
        if start is None:
            continue
        stop = next((value for value in stops if start < value <= start + 20), None)
        if stop is None:
            continue
        metric_stops_during_replay = [
            metric
            for metric in ttfbs
            if start - 0.05 <= metric["time"] <= stop + 0.05
        ]
        output_starts_during_replay = [
            event
            for event in tts_starts
            if start - 0.05 <= event["time"] <= stop + 0.05
        ]
        replay_windows.append(
            {
                "turn": replay["turn"],
                "requeue_time": replay["time"],
                "user_start_time": start,
                "user_stop_time": stop,
                "requeue_line": replay["line"],
                # InterruptionFrame calls stop_all_metrics(), so a TTFB metric
                # emitted at replay VAD start is a forced metric stop, not proof
                # that the provider returned a response.
                "ttfb_metric_stops": metric_stops_during_replay,
                # TTSStartedFrame is the first user-visible audio output.  If it
                # appears before replay VAD stop, bot output and duplicate user
                # audio actually overlap.
                "output_starts": output_starts_during_replay,
            }
        )

    duration = events[-1][0] - events[0][0] if events else None
    return {
        "run_dir": str(run_dir),
        "turns": primary_turn_count(run_dir),
        "timeout_events": timeouts,
        "forced_advances": forced,
        "requeues": requeues,
        "replay_windows": replay_windows,
        "cancelled_retries": sum(
            "Turn retry cancelled (turn completed normally)" in line for line in lines
        ),
        "reconnections": sum(
            "Gemini reconnected: scheduling" in line for line in lines
        ),
        "judged_turns": judged_turns(run_dir),
        "duration_seconds": duration,
    }


def cohort_summary(name: str, run_dirs: list[Path]) -> dict:
    runs = [analyze_run(ROOT / run_dir) for run_dir in run_dirs]
    turns = sum(run["turns"] for run in runs)
    timeout_events = sum(len(run["timeout_events"]) for run in runs)
    affected_turns = sum(
        len({event["turn"] for event in run["timeout_events"]}) for run in runs
    )
    forced_advances = sum(len(run["forced_advances"]) for run in runs)
    requeues = sum(len(run["requeues"]) for run in runs)
    replay_windows = [window for run in runs for window in run["replay_windows"]]
    replay_output_overlaps = sum(bool(window["output_starts"]) for window in replay_windows)
    replay_forced_metric_stops = sum(
        bool(window["ttfb_metric_stops"]) for window in replay_windows
    )
    by_turn_events: Counter[int] = Counter()
    by_turn_runs: Counter[int] = Counter()
    by_turn_forced: Counter[int] = Counter()
    retry_depth: Counter[int] = Counter()
    judged_rows = []
    for run in runs:
        affected = set()
        for event in run["timeout_events"]:
            by_turn_events[event["turn"]] += 1
            retry_depth[event["retry_count"]] += 1
            affected.add(event["turn"])
        by_turn_runs.update(affected)
        by_turn_forced.update(event["turn"] for event in run["forced_advances"])
        forced_turns = {event["turn"] for event in run["forced_advances"]}
        for row in run["judged_turns"]:
            judged_rows.append(
                {
                    **row,
                    "affected": row["turn"] in affected,
                    "forced": row["turn"] in forced_turns,
                }
            )

    def score_summary(rows: list[dict]) -> dict:
        return {
            "turn_count": len(rows),
            **{
                key: {
                    "passes": sum(row[key] for row in rows),
                    "rate_pct": 100 * sum(row[key] for row in rows) / len(rows)
                    if rows
                    else None,
                }
                for key in (
                    "strict",
                    "tool_use_correct",
                    "instruction_following",
                    "kb_grounding",
                    "turn_taking",
                )
            },
        }

    shared_turns = []
    for turn in range(30):
        affected_rows = [
            row for row in judged_rows if row["turn"] == turn and row["affected"]
        ]
        unaffected_rows = [
            row for row in judged_rows if row["turn"] == turn and not row["affected"]
        ]
        if affected_rows and unaffected_rows:
            shared_turns.append((turn, affected_rows, unaffected_rows))

    turn_adjusted = {"shared_turn_count": len(shared_turns)}
    for key in ("strict", "turn_taking"):
        if not shared_turns:
            turn_adjusted[key] = {
                "affected_rate_pct": None,
                "unaffected_rate_pct": None,
                "difference_percentage_points": None,
            }
            continue
        affected_rate = statistics.mean(
            sum(row[key] for row in affected_rows) / len(affected_rows)
            for _, affected_rows, _ in shared_turns
        )
        unaffected_rate = statistics.mean(
            sum(row[key] for row in unaffected_rows) / len(unaffected_rows)
            for _, _, unaffected_rows in shared_turns
        )
        turn_adjusted[key] = {
            "affected_rate_pct": 100 * affected_rate,
            "unaffected_rate_pct": 100 * unaffected_rate,
            "difference_percentage_points": 100 * (affected_rate - unaffected_rate),
        }

    durations = [run["duration_seconds"] for run in runs if run["duration_seconds"]]
    return {
        "name": name,
        "run_count": len(runs),
        "turn_count": turns,
        "timeout_event_count": timeout_events,
        "timeout_events_per_run": [len(run["timeout_events"]) for run in runs],
        "affected_turn_count": affected_turns,
        "affected_turn_rate_pct": 100 * affected_turns / turns if turns else None,
        "affected_turns_per_run": [
            len({event["turn"] for event in run["timeout_events"]}) for run in runs
        ],
        "first_affected_turn_per_run": [
            min((event["turn"] for event in run["timeout_events"]), default=None)
            for run in runs
        ],
        "forced_advance_count": forced_advances,
        "forced_advances_per_run": [len(run["forced_advances"]) for run in runs],
        "requeue_count": requeues,
        "replay_window_count": len(replay_windows),
        "replay_output_overlap_count": replay_output_overlaps,
        "replay_output_overlap_rate_pct": (
            100 * replay_output_overlaps / len(replay_windows) if replay_windows else None
        ),
        "replay_forced_ttfb_stop_count": replay_forced_metric_stops,
        "cancelled_retry_count": sum(run["cancelled_retries"] for run in runs),
        "reconnection_count": sum(run["reconnections"] for run in runs),
        "duration_seconds_median": statistics.median(durations) if durations else None,
        "duration_seconds_min": min(durations) if durations else None,
        "duration_seconds_max": max(durations) if durations else None,
        "retry_count_field_distribution": dict(sorted(retry_depth.items())),
        "recovery_outcomes": {
            "recovered_after_one_replay": retry_depth[0] - retry_depth[1],
            "recovered_after_two_replays": retry_depth[1] - retry_depth[2],
            "recovered_after_three_replays": retry_depth[2] - retry_depth[3],
            "forced_after_three_replays": retry_depth[3],
        },
        "quality_by_timeout_status": {
            "affected": score_summary([row for row in judged_rows if row["affected"]]),
            "unaffected": score_summary(
                [row for row in judged_rows if not row["affected"]]
            ),
            "forced": score_summary([row for row in judged_rows if row["forced"]]),
        },
        "shared_turn_adjusted_quality": turn_adjusted,
        "by_turn": [
            {
                "turn": turn,
                "affected_runs": by_turn_runs[turn],
                "timeout_events": by_turn_events[turn],
                "forced_advances": by_turn_forced[turn],
            }
            for turn in range(30)
            if by_turn_events[turn] or by_turn_forced[turn]
        ],
        "runs": runs,
    }


def main() -> None:
    comparison = json.loads((HERE / "comparison.json").read_text())
    cohorts = {row["key"]: row for row in comparison["cohorts"]}
    clever_dirs = [Path(path) for path in cohorts["clever"]["run_dirs"]]
    preview_dirs = [Path(path) for path in cohorts["gemini31"]["run_dirs"]]
    preview_dirs.append(CURRENT_PREVIEW)

    clever = cohort_summary("Async Gemini Live", clever_dirs)
    preview = cohort_summary("Gemini 3.1 Flash Live Preview", preview_dirs)

    clever_rates = np.asarray(clever["affected_turns_per_run"], dtype=float) / 30
    preview_rates = np.asarray(preview["affected_turns_per_run"], dtype=float) / 30
    rng = np.random.default_rng(SEED)
    clever_samples = clever_rates[
        rng.integers(0, len(clever_rates), size=(BOOTSTRAPS, len(clever_rates)))
    ].mean(axis=1)
    preview_samples = preview_rates[
        rng.integers(0, len(preview_rates), size=(BOOTSTRAPS, len(preview_rates)))
    ].mean(axis=1)
    difference = (clever_samples - preview_samples) * 100

    result = {
        "analysis_date": "2026-08-04",
        "timeout_seconds": 15,
        "replay_delay_seconds": 2,
        "bootstrap_samples": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "affected_turn_rate_difference_percentage_points": (
            clever["affected_turn_rate_pct"] - preview["affected_turn_rate_pct"]
        ),
        "affected_turn_rate_difference_ci95": [
            float(np.percentile(difference, 2.5)),
            float(np.percentile(difference, 97.5)),
        ],
        "cohorts": [clever, preview],
    }
    output = HERE / "timeout-replay-analysis.json"
    output.write_text(json.dumps(result, indent=2) + "\n")

    compact = {
        row["name"]: {
            key: row[key]
            for key in (
                "run_count",
                "turn_count",
                "timeout_event_count",
                "affected_turn_count",
                "affected_turn_rate_pct",
                "forced_advance_count",
                "requeue_count",
                "replay_window_count",
                "replay_output_overlap_count",
                "replay_output_overlap_rate_pct",
                "replay_forced_ttfb_stop_count",
                "cancelled_retry_count",
                "duration_seconds_median",
            )
        }
        for row in result["cohorts"]
    }
    print(json.dumps(compact, indent=2))
    print(
        "affected-turn rate difference 95% cluster bootstrap CI:",
        result["affected_turn_rate_difference_ci95"],
    )
    print("wrote", output.relative_to(ROOT))


if __name__ == "__main__":
    main()
