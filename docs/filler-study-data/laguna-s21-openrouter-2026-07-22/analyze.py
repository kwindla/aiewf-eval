#!/usr/bin/env python3
"""Final fixed-30-turn analysis for the Laguna S 2.1 30/30 campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STATE = HERE / "state"
MODEL = "laguna_s21"
DISPLAY = "laguna-s-2.1"
REQUESTED_MODEL = "poolside/laguna-s-2.1"
README_LABEL = f"{REQUESTED_MODEL} (thinking off)"
N_TURNS = 30
BOOTSTRAPS = 100_000
SEED = 20260722
EXPECTED_N = {"nofiller": 30, "dots96": 30}
SCHEDULES = (
    (
        HERE / "schedule.tsv",
        "ece7b3e83708f018627c78343c74db97642683f1adc77a4d77526ce80970886e",
    ),
    (
        HERE / "schedule-dots-topup.tsv",
        "6521d0be0ab91bc3f64a631b4635e17de2e38dcfcec536cccb1a50aab0da6491",
    ),
    (
        HERE / "schedule-n30-topup.tsv",
        "7ea9b6e3dfc53d104aca9d91eafdb4487623a8862a62c2aca3ae78b836d259e7",
    ),
)
RUN_COMPLETION_SENTINELS = (
    STATE / "RUNS_COMPLETE",
    STATE / "dots-topup" / "RUNS_COMPLETE",
    STATE / "n30-topup" / "RUNS_COMPLETE",
)


@dataclass(frozen=True)
class Conversation:
    slot: str
    arm: str
    run_dir: Path
    tool: tuple[bool, ...]
    instruction: tuple[bool, ...]
    kb: tuple[bool, ...]
    passed: tuple[bool, ...]
    strict_complete: bool
    ttfat_ms: tuple[float, ...]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def end_session_turns(transcript: Path) -> set[int]:
    turns: set[int] = set()
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        if any(call.get("name") == "end_session" for call in row.get("tool_calls") or []):
            turns.add(int(row.get("turn", -1)))
    return turns


def load_conversation(slot: str, arm: str, run_dir: Path) -> Conversation:
    transcript = run_dir / "transcript.jsonl"
    judged = run_dir / "claude_judged.jsonl"
    summary_path = run_dir / "claude_summary.json"
    run_log = run_dir / "run.log"
    if not all(
        path.is_file() and path.stat().st_size
        for path in (transcript, judged, summary_path, run_log)
    ):
        raise ValueError(f"incomplete included run: {run_dir}")

    log = run_log.read_text()
    for signature in (
        "Using OpenRouter with base_url=https://openrouter.ai/api/v1, reasoning_off=True, max_tokens=8192",
        "Recovery nudges enabled=True",
        "Tool call dedupe enabled=True",
        "Tool result run_llm enabled=False",
        "Text pipeline idle_timeout_secs=45.0",
    ):
        if signature not in log:
            raise ValueError(f"runtime signature {signature!r} missing: {run_dir}")
    filler = "MTE_FILLER_DOTS active: 96 x '.' filler tokens, position=suffix"
    if (arm == "dots96" and log.count(filler) != 1) or (
        arm == "nofiller" and "MTE_FILLER_DOTS active:" in log
    ):
        raise ValueError(f"filler signature mismatch: {run_dir}")

    raw: dict[int, dict] = {}
    thinking_tokens = 0
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        if row.get("model_name") != REQUESTED_MODEL:
            raise ValueError(f"model mismatch: {run_dir}")
        thinking_tokens += int((row.get("tokens") or {}).get("thinking_tokens") or 0)
        turn = row.get("turn")
        if (
            isinstance(turn, int)
            and 0 <= turn < N_TURNS
            and row.get("recovery_turn") is not True
        ):
            if turn in raw:
                raise ValueError(f"duplicate scripted turn {turn}: {run_dir}")
            raw[turn] = row
    if not raw:
        raise ValueError(f"included run has no scripted responses: {run_dir}")
    if thinking_tokens:
        raise ValueError(
            f"reasoning-off run reported {thinking_tokens} thinking tokens: {run_dir}"
        )

    judged_rows = [json.loads(line) for line in judged.read_text().splitlines()]
    if len(judged_rows) != len(raw):
        raise ValueError(f"judgment row count mismatch: {run_dir}")
    final: dict[int, dict] = {}
    for row in judged_rows:
        turn = row.get("turn")
        if not isinstance(turn, int) or turn in final:
            raise ValueError(f"invalid or duplicate judged turn: {run_dir}")
        final[turn] = row
    if set(final) != set(raw):
        raise ValueError(f"judgment coverage mismatch: {run_dir}")
    for row in final.values():
        scores = row.get("scores") or {}
        if not all(
            isinstance(scores.get(key), bool)
            for key in ("tool_use_correct", "instruction_following", "kb_grounding")
        ):
            raise ValueError(f"non-boolean required judgment score: {run_dir}")
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("turns_scored") != len(raw)
        or not summary.get("judge_model")
        or not summary.get("judge_version")
    ):
        raise ValueError(f"invalid judge summary: {run_dir}")

    tool: list[bool] = []
    instruction: list[bool] = []
    kb: list[bool] = []
    passed: list[bool] = []
    ttfat: list[float] = []
    for turn in range(N_TURNS):
        scores = (final.get(turn, {}).get("scores") or {})
        values = tuple(
            scores.get(key) is True
            for key in ("tool_use_correct", "instruction_following", "kb_grounding")
        )
        tool.append(values[0])
        instruction.append(values[1])
        kb.append(values[2])
        passed.append(all(values))
        latency = raw.get(turn, {}).get("ttfb_ms")
        if isinstance(latency, (int, float)) and math.isfinite(latency) and latency >= 0:
            ttfat.append(float(latency))
    return Conversation(
        slot=slot,
        arm=arm,
        run_dir=run_dir,
        tool=tuple(tool),
        instruction=tuple(instruction),
        kb=tuple(kb),
        passed=tuple(passed),
        strict_complete=end_session_turns(transcript) == {29},
        ttfat_ms=tuple(ttfat),
    )


def load_all() -> tuple[
    dict[str, list[Conversation]], dict[str, list[Conversation]]
]:
    missing_sentinels = [
        path.relative_to(HERE) for path in RUN_COMPLETION_SENTINELS if not path.is_file()
    ]
    if missing_sentinels:
        raise ValueError(f"runs are incomplete; missing={missing_sentinels}")
    if not (STATE / "N30_JUDGING_COMPLETE").is_file():
        raise ValueError("final n=30 judging is incomplete")
    schedule_rows: list[dict[str, str]] = []
    n10_slots: set[str] = set()
    for index, (schedule_path, expected_hash) in enumerate(SCHEDULES):
        if sha256(schedule_path) != expected_hash:
            raise ValueError(f"frozen schedule changed: {schedule_path.name}")
        rows = read_tsv(schedule_path)
        schedule_rows.extend(rows)
        if index < 2:
            n10_slots.update(row["slot"] for row in rows)
    schedule = {row["slot"]: row for row in schedule_rows}
    manifest = read_tsv(STATE / "manifest.tsv")
    if len(schedule_rows) != 60 or len(schedule) != 60 or len(manifest) != 60:
        raise ValueError("expected the complete 60-assignment n=30 campaign")
    if len(n10_slots) != 20:
        raise ValueError("frozen n=10 decision pool must contain 20 unique slots")
    expected_arm_counts = {
        arm: sum(row["arm"] == arm for row in schedule_rows) for arm in EXPECTED_N
    }
    if expected_arm_counts != EXPECTED_N:
        raise ValueError(f"schedule arm counts are invalid: {expected_arm_counts}")
    cells: dict[str, list[Conversation]] = defaultdict(list)
    n10_cells: dict[str, list[Conversation]] = defaultdict(list)
    seen_slots: set[str] = set()
    seen_dirs: set[Path] = set()
    for row in manifest:
        slot = row["slot"]
        if slot in seen_slots or slot not in schedule:
            raise ValueError(f"duplicate or unexpected slot: {slot}")
        assignment = schedule[slot]
        if (
            row["model"] != MODEL
            or row["arm"] != assignment["arm"]
            or assignment["model"] != MODEL
            or assignment["requested_model"] != REQUESTED_MODEL
            or assignment["service"] != "openrouter"
        ):
            raise ValueError(f"manifest policy mismatch in {slot}")
        run_dir = Path(row["run_dir"])
        run_dir = (run_dir if run_dir.is_absolute() else ROOT / run_dir).resolve()
        try:
            run_dir.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"run outside repository: {run_dir}") from exc
        if run_dir in seen_dirs:
            raise ValueError(f"duplicate included run: {run_dir}")
        seen_slots.add(slot)
        seen_dirs.add(run_dir)
        conversation = load_conversation(slot, row["arm"], run_dir)
        cells[row["arm"]].append(conversation)
        if slot in n10_slots:
            n10_cells[row["arm"]].append(conversation)
    if seen_slots != set(schedule):
        raise ValueError(f"missing slots: {sorted(set(schedule) - seen_slots)}")
    for arm, expected in EXPECTED_N.items():
        if len(cells[arm]) != expected:
            raise ValueError(f"expected {expected} {arm} runs, found {len(cells[arm])}")
        if len(n10_cells[arm]) != 10:
            raise ValueError(
                f"expected 10 frozen n=10 {arm} runs, found {len(n10_cells[arm])}"
            )
    return cells, n10_cells


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [100 * max(0.0, center - spread), 100 * min(1.0, center + spread)]


def arm_summary(
    runs: list[Conversation], rng: np.random.Generator
) -> tuple[dict, dict[str, np.ndarray]]:
    matrices = {
        key: np.asarray(
            [getattr(run, "passed" if key == "pass" else key) for run in runs],
            dtype=float,
        )
        for key in ("pass", "tool", "instruction", "kb")
    }
    n = len(runs)
    denominator = n * N_TURNS
    indices = rng.integers(0, n, size=(BOOTSTRAPS, n))
    boot: dict[str, np.ndarray] = {}
    result: dict[str, object] = {
        "n_attempts": n,
        "fixed_turn_denominator": denominator,
    }
    for key, matrix in matrices.items():
        pass_count = int(matrix.sum())
        conversation_rate = matrix.mean(axis=1)
        boot[key] = conversation_rate[indices].mean(axis=1) * 100
        prefix = "" if key == "pass" else f"{key}_"
        if key == "pass":
            result.update(
                {
                    "pass_count": pass_count,
                    "pass_rate_pct": 100 * pass_count / denominator,
                    "pass_rate_ci95": [
                        float(np.percentile(boot[key], 2.5)),
                        float(np.percentile(boot[key], 97.5)),
                    ],
                    "any_error_count": denominator - pass_count,
                    "any_error_rate_pct": 100 * (denominator - pass_count) / denominator,
                    "any_error_rate_ci95": [
                        float(np.percentile(100 - boot[key], 2.5)),
                        float(np.percentile(100 - boot[key], 97.5)),
                    ],
                }
            )
        else:
            result.update(
                {
                    f"{prefix}pass_count": pass_count,
                    f"{prefix}pass_rate_pct": 100 * pass_count / denominator,
                    f"{prefix}error_count": denominator - pass_count,
                    f"{prefix}error_rate_pct": 100 * (denominator - pass_count) / denominator,
                    f"{prefix}error_rate_ci95": [
                        float(np.percentile(100 - boot[key], 2.5)),
                        float(np.percentile(100 - boot[key], 97.5)),
                    ],
                }
            )
    complete = sum(run.strict_complete for run in runs)
    latencies = [value for run in runs for value in run.ttfat_ms]
    result.update(
        {
            "strict_complete_count": complete,
            "strict_completion_pct": 100 * complete / n,
            "strict_completion_ci95": wilson(complete, n),
            "ttfat_observations": len(latencies),
            "ttfat_p50_ms": statistics.median(latencies) if latencies else None,
            "ttfat_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "ttfat_max_ms": max(latencies) if latencies else None,
            "thinking_tokens": 0,
            "run_dirs": [str(run.run_dir.relative_to(ROOT)) for run in runs],
        }
    )
    return result, boot


def effect_summary(
    control: dict, dots: dict, control_boot: dict[str, np.ndarray], dots_boot: dict[str, np.ndarray]
) -> dict:
    result: dict[str, object] = {}
    for key in ("pass", "tool", "instruction", "kb"):
        prefix = "pass" if key == "pass" else key
        c_rate = control["pass_rate_pct" if key == "pass" else f"{key}_pass_rate_pct"]
        d_rate = dots["pass_rate_pct" if key == "pass" else f"{key}_pass_rate_pct"]
        delta = dots_boot[key] - control_boot[key]
        result[f"{prefix}_delta_points"] = d_rate - c_rate
        result[f"{prefix}_delta_ci95"] = [
            float(np.percentile(delta, 2.5)),
            float(np.percentile(delta, 97.5)),
        ]
    return result


def n10_promotion_decision(
    control_runs: list[Conversation], dots_runs: list[Conversation], control: dict, dots: dict, effect: dict
) -> dict:
    delta = float(effect["pass_delta_points"])
    ci = effect["pass_delta_ci95"]
    control_n = len(control_runs)
    dots_n = len(dots_runs)
    if control_n != 10 or dots_n != 10:
        raise ValueError(f"n=10 decision requires 10/10 runs, found {control_n}/{dots_n}")
    recurring: list[dict[str, object]] = []
    for turn in range(N_TURNS):
        control_failures = sum(not run.passed[turn] for run in control_runs)
        dots_failures = sum(not run.passed[turn] for run in dots_runs)
        control_rate = control_failures / control_n
        dots_rate = dots_failures / dots_n
        if control_failures >= 3 and dots_rate < control_rate:
            recurring.append(
                {
                    "turn": turn,
                    "direction": "benefit",
                    "control_failures": control_failures,
                    "dots_failures": dots_failures,
                }
            )
        if dots_failures >= 3 and dots_rate > control_rate:
            recurring.append(
                {
                    "turn": turn,
                    "direction": "harm",
                    "control_failures": control_failures,
                    "dots_failures": dots_failures,
                }
            )
    aggregate_direction = "benefit" if delta > 0 else "harm" if delta < 0 else None
    aligned = [row for row in recurring if row["direction"] == aggregate_direction]
    triggers: list[str] = []
    if ci[0] > 0 or ci[1] < 0:
        triggers.append("whole-conversation bootstrap 95% interval excludes zero")
    if abs(delta) >= 3.0 and aligned:
        triggers.append(
            "absolute difference >= 3.0 points with recurring same-turn direction"
        )
    if control["strict_completion_pct"] != dots["strict_completion_pct"]:
        triggers.append("strict-completion rates still differ at n=10")
    action = "promote_both_arms_to_30" if triggers else "stop_at_10"
    return {
        "stage": "dots_n10",
        "initial_dots_n": 6,
        "observed_control_n": control_n,
        "observed_dots_n": dots_n,
        "pass_delta_points": delta,
        "pass_delta_ci95": ci,
        "strict_completion_pct": [
            control["strict_completion_pct"],
            dots["strict_completion_pct"],
        ],
        "recurring_turn_signals": recurring,
        "aggregate_aligned_recurring_turn_signals": aligned,
        "triggers": triggers,
        "action": action,
        "final_n": 30 if triggers else 10,
        "decision_pending": bool(triggers),
    }


def main() -> None:
    cells, n10_cells = load_all()
    control, control_boot = arm_summary(
        cells["nofiller"], np.random.default_rng(SEED)
    )
    dots, dots_boot = arm_summary(
        cells["dots96"], np.random.default_rng(SEED + 1)
    )
    effect = effect_summary(control, dots, control_boot, dots_boot)

    # Reconstruct the sample-size decision from the immutable 10/10 pool only.
    # The final 30/30 outcomes must never be allowed to revise the decision that
    # caused their collection.
    n10_control, n10_control_boot = arm_summary(
        n10_cells["nofiller"], np.random.default_rng(SEED)
    )
    n10_dots, n10_dots_boot = arm_summary(
        n10_cells["dots96"], np.random.default_rng(SEED + 1)
    )
    n10_effect = effect_summary(
        n10_control, n10_dots, n10_control_boot, n10_dots_boot
    )
    promotion = n10_promotion_decision(
        n10_cells["nofiller"],
        n10_cells["dots96"],
        n10_control,
        n10_dots,
        n10_effect,
    )
    if promotion["action"] != "promote_both_arms_to_30":
        raise ValueError(
            "frozen n=10 pool does not reproduce the prespecified promotion"
        )
    decision = {
        "stage": "focused_n30",
        "initial_dots_n": 6,
        "observed_control_n": len(cells["nofiller"]),
        "observed_dots_n": len(cells["dots96"]),
        "pass_delta_points": effect["pass_delta_points"],
        "pass_delta_ci95": effect["pass_delta_ci95"],
        "strict_completion_pct": [
            control["strict_completion_pct"],
            dots["strict_completion_pct"],
        ],
        "action": "focused_followup_complete",
        "final_n": 30,
        "adaptive_expansion_completed": True,
        "no_further_sample_size_decision": True,
        "decision_pending": False,
        "frozen_n10_promotion_decision": promotion,
    }
    payload = {
        "schema_version": 1,
        "artifact_status": "FINAL",
        "protocol": {
            "benchmark": "aiwf_medium_context",
            "turns": N_TURNS,
            "target_per_arm": 30,
            "missing_scripted_turns": "fail",
            "thinking_mode": "disabled",
            "full_thinking_off_guaranteed": True,
            "route": "OpenRouter paid Poolside-hosted BF16",
            "filler": {"arm": "dots96", "glyph": ".", "count": 96, "position": "suffix"},
            "bootstrap_unit": "whole conversation",
            "bootstrap_samples": BOOTSTRAPS,
            "seed": SEED,
        },
        "models": {
            MODEL: {
                "display_name": DISPLAY,
                "readme_label": README_LABEL,
                "provider": "OpenRouter",
                "requested_model": REQUESTED_MODEL,
                "endpoint_provider": "Poolside",
                "quantization": "BF16",
                "report_tier": "focused",
                "arms": {"nofiller": control, "dots96": dots},
                "effect": effect,
                "adaptive_decision": decision,
            }
        },
    }
    output = HERE / "aggregates.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
