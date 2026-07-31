#!/usr/bin/env python3
"""Fixed-denominator analysis for the staged Gemini minimal/dot campaign."""

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
MODELS = ("gemini35flash", "gemini35flashlite", "gemini36flash")
DISPLAY = {
    "gemini35flash": ("gemini-3.5-flash", "3.5-flash-05-2026"),
    "gemini35flashlite": ("gemini-3.5-flash-lite", "3.5-flash-lite-07-2026"),
    "gemini36flash": ("gemini-3.6-flash", "3.6-flash-07-2026"),
}
LANES = (
    "g35", "g35control", "g35lite", "g36",
    "g35topup", "g35litetopup", "g36topup",
    "g35focused", "g35litefocused", "g36focused",
)
N_TURNS = 30
BOOTSTRAPS = 100_000
SEED = 20260721


@dataclass(frozen=True)
class Conversation:
    model: str
    arm: str
    run_dir: Path
    tool: tuple[bool, ...]
    instruction: tuple[bool, ...]
    kb: tuple[bool, ...]
    passed: tuple[bool, ...]
    strict_complete: bool
    ttfat_ms: tuple[float, ...]
    thought_tokens: int


def end_session_turns(transcript: Path) -> set[int]:
    turns: set[int] = set()
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        if any(call.get("name") == "end_session" for call in row.get("tool_calls") or []):
            turns.add(int(row.get("turn", -1)))
    return turns


def load_conversation(model: str, arm: str, run_dir: Path) -> Conversation:
    judged = run_dir / "claude_judged.jsonl"
    transcript = run_dir / "transcript.jsonl"
    if not judged.is_file() or not judged.stat().st_size:
        raise ValueError(f"missing judgment: {judged}")
    if not transcript.is_file() or not transcript.stat().st_size:
        raise ValueError(f"missing transcript: {transcript}")
    run_log = run_dir / "run.log"
    if not run_log.is_file():
        raise ValueError(f"missing internal run log: {run_log}")
    log_text = run_log.read_text()
    display, _catalog_version = DISPLAY[model]
    if f"Configured {display} with thinking_level=minimal" not in log_text:
        raise ValueError(f"minimal-thinking signature missing: {run_dir}")
    expected_filler = "MTE_FILLER_DOTS active: 96 x '.' filler tokens, position=suffix"
    any_filler = "MTE_FILLER_DOTS active:" in log_text
    filler_logged = expected_filler in log_text and log_text.count("MTE_FILLER_DOTS active:") == 1
    if (arm == "dots96" and not filler_logged) or (arm == "nofiller" and any_filler):
        raise ValueError(f"filler signature mismatch for {model}/{arm}: {run_dir}")
    for signature in (
        "Recovery nudges enabled=True",
        "Tool call dedupe enabled=True",
        "Tool result run_llm enabled=False",
        "Text pipeline idle_timeout_secs=45.0",
    ):
        if signature not in log_text:
            raise ValueError(f"runtime signature {signature!r} missing: {run_dir}")

    final: dict[int, dict] = {}
    for line in judged.read_text().splitlines():
        row = json.loads(line)
        turn = row.get("turn")
        if isinstance(turn, int) and 0 <= turn < N_TURNS:
            final[turn] = row
    raw: dict[int, dict] = {}
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        if row.get("model_name") != display:
            raise ValueError(f"transcript model mismatch in {run_dir}: {row.get('model_name')}")
        turn = row.get("turn")
        if isinstance(turn, int) and 0 <= turn < N_TURNS and row.get("recovery_turn") is not True:
            raw[turn] = row

    observed = set(raw)
    if set(final) != observed:
        raise ValueError(f"judgment coverage mismatch in {run_dir}: judged={sorted(final)} observed={sorted(observed)}")
    for turn, row in final.items():
        scores = row.get("scores") or {}
        if not all(isinstance(scores.get(key), bool) for key in ("tool_use_correct", "instruction_following", "kb_grounding")):
            raise ValueError(f"invalid judgment booleans in {run_dir} turn {turn}")
    summary_path = run_dir / "claude_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing judge summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("turns_scored") != len(observed) or not summary.get("judge_model") or not summary.get("judge_version"):
        raise ValueError(f"invalid judge summary: {summary_path}")

    tool: list[bool] = []
    instruction: list[bool] = []
    kb: list[bool] = []
    passed: list[bool] = []
    ttfat: list[float] = []
    thoughts = 0
    for turn in range(N_TURNS):
        scores = (final.get(turn, {}).get("scores") or {})
        values = (
            scores.get("tool_use_correct") is True,
            scores.get("instruction_following") is True,
            scores.get("kb_grounding") is True,
        )
        tool.append(values[0])
        instruction.append(values[1])
        kb.append(values[2])
        passed.append(all(values))
        transcript_row = raw.get(turn, {})
        latency = transcript_row.get("ttfb_ms")
        if isinstance(latency, (int, float)) and math.isfinite(latency) and latency >= 0:
            ttfat.append(float(latency))
        tokens = transcript_row.get("tokens") or {}
        reasoning = tokens.get("thinking_tokens")
        if isinstance(reasoning, int) and reasoning > 0:
            thoughts += reasoning

    return Conversation(
        model=model,
        arm=arm,
        run_dir=run_dir,
        tool=tuple(tool),
        instruction=tuple(instruction),
        kb=tuple(kb),
        passed=tuple(passed),
        strict_complete=end_session_turns(transcript) == {29},
        ttfat_ms=tuple(ttfat),
        thought_tokens=thoughts,
    )


def load_all() -> dict[tuple[str, str], list[Conversation]]:
    refs: list[tuple[str, str, Path]] = []
    with (HERE / "existing-included.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["primary"] == "1":
                refs.append((row["model"], row["arm"], ROOT / row["run_dir"]))
    for lane in LANES:
        state = HERE / "state" / lane
        if not (state / "COMPLETE").is_file():
            raise ValueError(f"stage-1 lane is not complete: {lane}")
        with (state / "manifest.tsv").open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                run_dir = Path(row["run_dir"])
                refs.append((row["model"], row["arm"], run_dir if run_dir.is_absolute() else ROOT / run_dir))

    cells: dict[tuple[str, str], list[Conversation]] = defaultdict(list)
    seen: set[Path] = set()
    for model, arm, run_dir in refs:
        resolved = run_dir.resolve()
        if resolved in seen:
            raise ValueError(f"duplicate included run: {resolved}")
        if model not in MODELS or arm not in {"nofiller", "dots96"}:
            raise ValueError(f"unexpected cell: {model}/{arm}")
        seen.add(resolved)
        cells[(model, arm)].append(load_conversation(model, arm, resolved))

    for model in MODELS:
        control_n = len(cells[(model, "nofiller")])
        dots_n = len(cells[(model, "dots96")])
        if control_n not in {10, 30}:
            raise ValueError(f"unexpected control n for {model}: {control_n}")
        if dots_n not in {6, 10, 30}:
            raise ValueError(f"unexpected dots n for {model}: {dots_n}")
        if control_n == 30 and dots_n != 30:
            raise ValueError(f"focused follow-up must be balanced for {model}")
    return cells


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [100 * max(0.0, center - spread), 100 * min(1.0, center + spread)]


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def arm_summary(runs: list[Conversation], rng: np.random.Generator) -> dict:
    matrices = {
        key: np.asarray([getattr(r, "passed" if key == "pass" else key) for r in runs], dtype=float)
        for key in ("pass", "tool", "instruction", "kb")
    }
    n = len(runs)
    sample_idx = rng.integers(0, n, size=(BOOTSTRAPS, n))
    conversation_pass = matrices["pass"].mean(axis=1)
    boot_pass = conversation_pass[sample_idx].mean(axis=1) * 100
    pass_count = int(matrices["pass"].sum())
    denom = n * N_TURNS
    complete = sum(run.strict_complete for run in runs)
    ttfat = [value for run in runs for value in run.ttfat_ms]
    summary = {
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
        "ttfat_observations": len(ttfat),
        "ttfat_p50_ms": statistics.median(ttfat) if ttfat else None,
        "ttfat_p95_ms": percentile(ttfat, 95) if ttfat else None,
        "ttfat_max_ms": max(ttfat) if ttfat else None,
        "thought_tokens": sum(run.thought_tokens for run in runs),
        "run_dirs": [str(run.run_dir.relative_to(ROOT)) for run in runs],
        "_conversation_pass": conversation_pass,
    }
    for key in ("tool", "instruction", "kb"):
        error_count = denom - int(matrices[key].sum())
        conversation_error = 1 - matrices[key].mean(axis=1)
        boot_error = conversation_error[sample_idx].mean(axis=1) * 100
        summary[f"{key}_error_count"] = error_count
        summary[f"{key}_error_rate_pct"] = 100 * error_count / denom
        summary[f"{key}_error_rate_ci95"] = [
            float(np.percentile(boot_error, 2.5)),
            float(np.percentile(boot_error, 97.5)),
        ]
    return summary


def compute_adaptive_decision(
    control_runs: list[Conversation],
    dots_runs: list[Conversation],
    control: dict,
    dots: dict,
    effect: dict,
) -> dict:
    delta = effect["pass_delta_points"]
    ci = effect["pass_delta_ci95"]
    nc = control["n_attempts"]
    nd = dots["n_attempts"]
    control_failures = [sum(not run.passed[turn] for run in control_runs) for turn in range(N_TURNS)]
    dots_failures = [sum(not run.passed[turn] for run in dots_runs) for turn in range(N_TURNS)]
    recurring: list[dict] = []
    for turn, (c_count, d_count) in enumerate(zip(control_failures, dots_failures)):
        c_rate = c_count / nc
        d_rate = d_count / nd
        if c_count >= 3 and d_rate < c_rate:
            recurring.append({"turn": turn, "direction": "benefit", "control_failures": c_count, "dots_failures": d_count})
        if d_count >= 3 and d_rate > c_rate:
            recurring.append({"turn": turn, "direction": "harm", "control_failures": c_count, "dots_failures": d_count})

    aggregate_direction = "benefit" if delta > 0 else "harm" if delta < 0 else None
    aligned_recurring = [row for row in recurring if row["direction"] == aggregate_direction]

    triggers: list[str] = []
    if nd == 6:
        if abs(delta) >= 2.0:
            triggers.append("absolute pass-rate difference >= 2.0 points")
        if control["strict_completion_pct"] != dots["strict_completion_pct"]:
            triggers.append("strict-completion rates differ")
        action = "top_up_dots_to_10" if triggers else "stop_at_6"
        final_n = 10 if triggers else 6
    elif nd == 10:
        if ci[0] > 0 or ci[1] < 0:
            triggers.append("whole-conversation bootstrap 95% interval excludes zero")
        if abs(delta) >= 3.0 and aligned_recurring:
            triggers.append("absolute difference >= 3.0 points with recurring same-turn direction")
        if control["strict_completion_pct"] != dots["strict_completion_pct"]:
            triggers.append("strict-completion rates still differ at n=10")
        action = "promote_both_arms_to_30" if triggers else "stop_at_10"
        final_n = 30 if triggers else 10
    elif nd == 30 and nc == 30:
        action = "focused_followup_complete"
        final_n = 30
    else:
        raise ValueError(f"unsupported staged sample sizes: {nc}/{nd}")
    return {
        "initial_dots_n": 6,
        "observed_control_n": nc,
        "observed_dots_n": nd,
        "pass_delta_points": delta,
        "pass_delta_ci95": ci,
        "strict_completion_pct": [control["strict_completion_pct"], dots["strict_completion_pct"]],
        "recurring_turn_signals": recurring,
        "aggregate_aligned_recurring_turn_signals": aligned_recurring,
        "triggers": triggers,
        "action": action,
        "final_dots_n": final_n,
        "decision_pending": action in {"top_up_dots_to_10", "promote_both_arms_to_30"},
    }


def main() -> None:
    cells = load_all()
    decision_path = HERE / "adaptive-decision.json"
    decisions = json.loads(decision_path.read_text()).get("models", {}) if decision_path.is_file() else {}
    if decisions and set(decisions) != set(MODELS):
        raise ValueError(f"adaptive-decision model mismatch: {sorted(decisions)}")
    output = {
        "schema_version": 1,
        "artifact_status": "FINAL" if decisions and not any(row.get("decision_pending") for row in decisions.values()) else "STAGED_EXPLORATORY",
        "protocol": {
            "benchmark": "aiwf_medium_context",
            "turns": N_TURNS,
            "thinking_mode": "minimal",
            "full_thinking_off_guaranteed": False,
            "filler": {"arm": "dots96", "glyph": ".", "count": 96, "position": "suffix"},
            "bootstrap_samples": BOOTSTRAPS,
            "seed": SEED,
            "model_order": list(MODELS),
        },
        "models": {},
    }
    for model_index, model in enumerate(MODELS):
        summaries: dict[str, dict] = {}
        for arm_index, arm in enumerate(("nofiller", "dots96")):
            summaries[arm] = arm_summary(
                cells[(model, arm)], np.random.default_rng(SEED + 100 * model_index + arm_index)
            )
        control = summaries["nofiller"]
        dots = summaries["dots96"]
        c = control.pop("_conversation_pass")
        d = dots.pop("_conversation_pass")
        rng = np.random.default_rng(SEED + 10_000 + model_index)
        idx_c = rng.integers(0, len(c), size=(BOOTSTRAPS, len(c)))
        idx_d = rng.integers(0, len(d), size=(BOOTSTRAPS, len(d)))
        boot_delta = (d[idx_d].mean(axis=1) - c[idx_c].mean(axis=1)) * 100
        delta = dots["pass_rate_pct"] - control["pass_rate_pct"]
        display, version = DISPLAY[model]
        effect = {
            "pass_delta_points": delta,
            "pass_delta_ci95": [float(np.percentile(boot_delta, 2.5)), float(np.percentile(boot_delta, 97.5))],
        }
        expected_decision = compute_adaptive_decision(
            cells[(model, "nofiller")], cells[(model, "dots96")], control, dots, effect
        )
        decision = decisions.get(model, {
            "initial_dots_n": 6,
            "final_dots_n": dots["n_attempts"],
            "triggers": [],
            "decision_pending": True,
        })
        if not isinstance(decision.get("decision_pending"), bool):
            raise ValueError(f"invalid decision_pending for {model}")
        if decisions and decision.get("observed_dots_n") == dots["n_attempts"]:
            if decision != expected_decision:
                raise ValueError(f"adaptive decision does not match mechanical rule for {model}")
        elif decisions and decision["decision_pending"] is False:
            raise ValueError(f"stale final adaptive decision for {model}")
        output["models"][model] = {
            "display_name": display,
            "readme_label": f"{display} (minimal)",
            "provider": "Google",
            "requested_model": display,
            "catalog_version": version,
            "report_tier": "focused" if control["n_attempts"] == dots["n_attempts"] == 30 else "exploratory",
            "arms": summaries,
            "effect": effect,
            "adaptive_decision": decision,
        }
    (HERE / "aggregates.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
