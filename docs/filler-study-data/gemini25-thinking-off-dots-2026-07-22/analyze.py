#!/usr/bin/env python3
"""Fixed-denominator analysis for the Gemini 2.5 Flash thinking-off screen."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = "gemini25flash"
DISPLAY = "gemini-2.5-flash"
N_TURNS = 30
BOOTSTRAPS = 100_000
SEED = 20260722
LANES = ("control", "control-topup", "dots", "dots-topup", "focused")


@dataclass(frozen=True)
class Conversation:
    arm: str
    run_dir: Path
    tool: tuple[bool, ...]
    instruction: tuple[bool, ...]
    kb: tuple[bool, ...]
    passed: tuple[bool, ...]
    strict_complete: bool
    ttfat_ms: tuple[float, ...]


def end_session_turns(transcript: Path) -> set[int]:
    turns = set()
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        if any(call.get("name") == "end_session" for call in row.get("tool_calls") or []):
            turns.add(int(row.get("turn", -1)))
    return turns


def load_conversation(arm: str, run_dir: Path) -> Conversation:
    judged = run_dir / "claude_judged.jsonl"
    transcript = run_dir / "transcript.jsonl"
    run_log = run_dir / "run.log"
    summary_path = run_dir / "claude_summary.json"
    if not all(path.is_file() and path.stat().st_size for path in (judged, transcript, run_log, summary_path)):
        raise ValueError(f"incomplete included run: {run_dir}")
    log = run_log.read_text()
    for signature in (
        "Configured gemini-2.5-flash with thinking_budget=0 (disabled)",
        "Recovery nudges enabled=True",
        "Tool call dedupe enabled=True",
        "Tool result run_llm enabled=False",
        "Text pipeline idle_timeout_secs=45.0",
    ):
        if signature not in log:
            raise ValueError(f"runtime signature {signature!r} missing: {run_dir}")
    filler = "MTE_FILLER_DOTS active: 96 x '.' filler tokens, position=suffix"
    if (arm == "dots96" and log.count(filler) != 1) or (arm == "nofiller" and "MTE_FILLER_DOTS active:" in log):
        raise ValueError(f"filler signature mismatch: {run_dir}")

    raw = {}
    thoughts = 0
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        if row.get("model_name") != DISPLAY:
            raise ValueError(f"model mismatch: {run_dir}")
        thoughts += int((row.get("tokens") or {}).get("thinking_tokens") or 0)
        turn = row.get("turn")
        if isinstance(turn, int) and 0 <= turn < N_TURNS and row.get("recovery_turn") is not True:
            raw[turn] = row
    if thoughts != 0:
        raise ValueError(f"thinking-off run reports {thoughts} thought tokens: {run_dir}")
    final = {}
    for line in judged.read_text().splitlines():
        row = json.loads(line)
        turn = row.get("turn")
        if isinstance(turn, int) and 0 <= turn < N_TURNS:
            final[turn] = row
    if set(final) != set(raw):
        raise ValueError(f"judgment coverage mismatch: {run_dir}")
    summary = json.loads(summary_path.read_text())
    if summary.get("turns_scored") != len(raw) or not summary.get("judge_model") or not summary.get("judge_version"):
        raise ValueError(f"invalid judge summary: {run_dir}")

    tool, instruction, kb, passed, ttfat = [], [], [], [], []
    for turn in range(N_TURNS):
        scores = (final.get(turn, {}).get("scores") or {})
        values = tuple(scores.get(key) is True for key in (
            "tool_use_correct", "instruction_following", "kb_grounding"
        ))
        tool.append(values[0])
        instruction.append(values[1])
        kb.append(values[2])
        passed.append(all(values))
        latency = raw.get(turn, {}).get("ttfb_ms")
        if isinstance(latency, (int, float)) and math.isfinite(latency) and latency >= 0:
            ttfat.append(float(latency))
    return Conversation(
        arm=arm, run_dir=run_dir, tool=tuple(tool), instruction=tuple(instruction),
        kb=tuple(kb), passed=tuple(passed), strict_complete=end_session_turns(transcript) == {29},
        ttfat_ms=tuple(ttfat),
    )


def load_all() -> dict[str, list[Conversation]]:
    refs = []
    for lane in LANES:
        state = HERE / "state" / lane
        if not state.exists():
            continue
        if not (state / "COMPLETE").is_file():
            raise ValueError(f"lane exists but is incomplete: {lane}")
        with (state / "manifest.tsv").open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["model"] != MODEL or row["arm"] not in {"nofiller", "dots96"}:
                    raise ValueError(f"unexpected manifest cell in {lane}")
                path = Path(row["run_dir"])
                refs.append((row["arm"], path if path.is_absolute() else ROOT / path))
    cells = defaultdict(list)
    seen = set()
    for arm, run_dir in refs:
        resolved = run_dir.resolve()
        if resolved in seen:
            raise ValueError(f"duplicate included run: {resolved}")
        seen.add(resolved)
        cells[arm].append(load_conversation(arm, resolved))
    nc, nd = len(cells["nofiller"]), len(cells["dots96"])
    if (
        nc not in {10, 30}
        or nd not in {6, 10, 30}
        or (nc == 30 and nd not in {6, 30})
    ):
        raise ValueError(f"unsupported staged sample sizes: {nc}/{nd}")
    return cells


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [100 * max(0.0, center - spread), 100 * min(1.0, center + spread)]


def arm_summary(runs: list[Conversation], rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    matrices = {
        key: np.asarray([getattr(run, "passed" if key == "pass" else key) for run in runs], dtype=float)
        for key in ("pass", "tool", "instruction", "kb")
    }
    n = len(runs)
    idx = rng.integers(0, n, size=(BOOTSTRAPS, n))
    conversation_pass = matrices["pass"].mean(axis=1)
    boot_pass = conversation_pass[idx].mean(axis=1) * 100
    denom = n * N_TURNS
    pass_count = int(matrices["pass"].sum())
    complete = sum(run.strict_complete for run in runs)
    latencies = [value for run in runs for value in run.ttfat_ms]
    result = {
        "n_attempts": n,
        "fixed_turn_denominator": denom,
        "strict_complete_count": complete,
        "strict_completion_pct": 100 * complete / n,
        "strict_completion_ci95": wilson(complete, n),
        "pass_count": pass_count,
        "pass_rate_pct": 100 * pass_count / denom,
        "pass_rate_ci95": [float(np.percentile(boot_pass, 2.5)), float(np.percentile(boot_pass, 97.5))],
        "any_error_count": denom - pass_count,
        "any_error_rate_pct": 100 * (denom - pass_count) / denom,
        "any_error_rate_ci95": [float(np.percentile(100 - boot_pass, 2.5)), float(np.percentile(100 - boot_pass, 97.5))],
        "ttfat_observations": len(latencies),
        "ttfat_p50_ms": statistics.median(latencies) if latencies else None,
        "ttfat_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "ttfat_max_ms": max(latencies) if latencies else None,
        "thought_tokens": 0,
        "run_dirs": [str(run.run_dir.relative_to(ROOT)) for run in runs],
    }
    for key in ("tool", "instruction", "kb"):
        error_count = denom - int(matrices[key].sum())
        conversation_error = 1 - matrices[key].mean(axis=1)
        boot_error = conversation_error[idx].mean(axis=1) * 100
        result[f"{key}_error_count"] = error_count
        result[f"{key}_error_rate_pct"] = 100 * error_count / denom
        result[f"{key}_error_rate_ci95"] = [
            float(np.percentile(boot_error, 2.5)), float(np.percentile(boot_error, 97.5))
        ]
    return result, conversation_pass


def decision(control_runs, dots_runs, control, dots, effect) -> dict:
    delta = effect["pass_delta_points"]
    ci = effect["pass_delta_ci95"]
    nc, nd = control["n_attempts"], dots["n_attempts"]
    recurring = []
    for turn in range(N_TURNS):
        cf = sum(not run.passed[turn] for run in control_runs)
        df = sum(not run.passed[turn] for run in dots_runs)
        if cf >= 3 and df / nd < cf / nc:
            recurring.append({"turn": turn, "direction": "benefit", "control_failures": cf, "dots_failures": df})
        if df >= 3 and df / nd > cf / nc:
            recurring.append({"turn": turn, "direction": "harm", "control_failures": cf, "dots_failures": df})
    direction = "benefit" if delta > 0 else "harm" if delta < 0 else None
    aligned = [row for row in recurring if row["direction"] == direction]
    triggers = []
    if nc == 30 and nd == 6:
        # The original 10/6 screen prospectively stopped. This later,
        # control-only extension improves the public no-filler estimate and
        # must not retrospectively reopen the dot-arm sampling decision.
        action = "control_precision_extension_complete"
    elif nd == 6:
        if abs(delta) >= 2:
            triggers.append("absolute pass-rate difference >= 2.0 points")
        if control["strict_completion_pct"] != dots["strict_completion_pct"]:
            triggers.append("strict-completion rates differ")
        action = "top_up_dots_to_10" if triggers else "stop_at_6"
    elif nd == 10:
        if ci[0] > 0 or ci[1] < 0:
            triggers.append("whole-conversation bootstrap 95% interval excludes zero")
        if abs(delta) >= 3 and aligned:
            triggers.append("absolute difference >= 3.0 points with recurring same-turn direction")
        if control["strict_completion_pct"] != dots["strict_completion_pct"]:
            triggers.append("strict-completion rates still differ at n=10")
        action = "promote_both_arms_to_30" if triggers else "stop_at_10"
    elif nc == nd == 30:
        action = "focused_followup_complete"
    else:
        raise ValueError(f"unsupported decision sample sizes: {nc}/{nd}")
    return {
        "initial_dots_n": 6,
        "observed_control_n": nc,
        "observed_dots_n": nd,
        "pass_delta_points": delta,
        "pass_delta_ci95": ci,
        "strict_completion_pct": [control["strict_completion_pct"], dots["strict_completion_pct"]],
        "recurring_turn_signals": recurring,
        "aggregate_aligned_recurring_turn_signals": aligned,
        "triggers": triggers,
        "action": action,
        "stage1_action": "stop_at_6" if nc == 30 and nd == 6 else action,
        "decision_pending": action in {"top_up_dots_to_10", "promote_both_arms_to_30"},
    }


def main() -> None:
    cells = load_all()
    control, c = arm_summary(cells["nofiller"], np.random.default_rng(SEED))
    dots, d = arm_summary(cells["dots96"], np.random.default_rng(SEED + 1))
    rng = np.random.default_rng(SEED + 10_000)
    ci = (d[rng.integers(0, len(d), size=(BOOTSTRAPS, len(d)))].mean(axis=1)
          - c[rng.integers(0, len(c), size=(BOOTSTRAPS, len(c)))].mean(axis=1)) * 100
    effect = {
        "pass_delta_points": dots["pass_rate_pct"] - control["pass_rate_pct"],
        "pass_delta_ci95": [float(np.percentile(ci, 2.5)), float(np.percentile(ci, 97.5))],
    }
    adaptive = decision(cells["nofiller"], cells["dots96"], control, dots, effect)
    payload = {
        "schema_version": 1,
        "artifact_status": "STAGED_EXPLORATORY" if adaptive["decision_pending"] else "FINAL",
        "protocol": {
            "benchmark": "aiwf_medium_context", "turns": N_TURNS,
            "thinking_mode": "disabled", "thinking_budget": 0,
            "full_thinking_off_guaranteed": True,
            "filler": {"arm": "dots96", "glyph": ".", "count": 96, "position": "suffix"},
            "bootstrap_samples": BOOTSTRAPS, "seed": SEED, "model_order": [MODEL],
        },
        "models": {
            MODEL: {
                "display_name": DISPLAY, "readme_label": f"{DISPLAY} (thinking off)",
                "provider": "Google", "requested_model": DISPLAY,
                "report_tier": "focused" if control["n_attempts"] == dots["n_attempts"] == 30 else "exploratory",
                "arms": {"nofiller": control, "dots96": dots},
                "effect": effect, "adaptive_decision": adaptive,
            }
        },
    }
    (HERE / "aggregates.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
