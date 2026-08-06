#!/usr/bin/env python3
"""Fixed-denominator analysis and mechanical stage decisions for Qwen3.6 dots."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTROL27 = (
    ROOT
    / "ops/baseten-qwen36-27b-vllm"
    / "aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z"
)
N_TURNS = 30
BOOTSTRAPS = 100_000
SEED = 20260728
MODELS = ("qwen36_27b", "qwen36_35b")
MODEL_META = {
    "qwen36_27b": {
        "display_name": "qwen3.6-27b (thinking off)",
        "requested_model": "Qwen/Qwen3.6-27B",
        "checkpoint_precision": "BF16",
        "deployment_id": "wxpnlg5",
        "endpoint": "https://model-w67n482q.api.baseten.co/deployment/wxpnlg5/sync/v1",
    },
    "qwen36_35b": {
        "display_name": "qwen3.6-35b-a3b (thinking off, FP8)",
        "requested_model": "Qwen/Qwen3.6-35B-A3B-FP8",
        "checkpoint_precision": "FP8",
        "deployment_id": "qe20zvr",
        "endpoint": "https://model-qzkm8mpq.api.baseten.co/deployment/qe20zvr/sync/v1",
    },
}


@dataclass(frozen=True)
class Conversation:
    slot: str
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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def end_session_turns(transcript: Path) -> set[int]:
    turns = set()
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        if any(call.get("name") == "end_session" for call in row.get("tool_calls") or []):
            turns.add(int(row.get("turn", -1)))
    return turns


def load_conversation(
    slot: str,
    model: str,
    arm: str,
    run_dir: Path,
    expected_hash: str | None,
) -> Conversation:
    meta = MODEL_META[model]
    transcript = run_dir / "transcript.jsonl"
    judged = run_dir / "claude_judged.jsonl"
    summary_path = run_dir / "claude_summary.json"
    run_log = run_dir / "run.log"
    for path in (transcript, judged, summary_path, run_log):
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing required artifact: {path}")
    if expected_hash and sha256(transcript) != expected_hash:
        raise ValueError(f"transcript hash mismatch: {run_dir}")

    log_text = run_log.read_text(errors="replace")
    config_signature = (
        f"Using vllm-openai with base_url={meta['endpoint']}, "
        f"model={meta['requested_model']}, thinking=False, thinking_budget=None, "
        "T=0.6, top_p=0.95, top_k=None, max_tokens=8192"
    )
    if config_signature not in log_text:
        raise ValueError(f"configuration signature mismatch: {run_dir}")
    expected_filler = "MTE_FILLER_DOTS active: 96 x '.' filler tokens, position=suffix"
    has_any_filler = "MTE_FILLER_DOTS active:" in log_text
    if (arm == "dots96" and expected_filler not in log_text) or (
        arm == "nofiller" and has_any_filler
    ):
        raise ValueError(f"filler signature mismatch: {model}/{arm}/{run_dir}")
    for signature in (
        "Recovery nudges enabled=True",
        "Tool call dedupe enabled=True",
        "Tool result run_llm enabled=False",
        "Text pipeline idle_timeout_secs=45.0",
    ):
        if signature not in log_text:
            raise ValueError(f"runtime signature missing {signature!r}: {run_dir}")

    raw = {}
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        if row.get("model_name") != meta["requested_model"]:
            raise ValueError(f"transcript model mismatch: {run_dir}")
        turn = row.get("turn")
        if (
            isinstance(turn, int)
            and 0 <= turn < N_TURNS
            and row.get("recovery_turn") is not True
        ):
            if turn in raw:
                raise ValueError(f"duplicate scripted turn {turn}: {run_dir}")
            raw[turn] = row

    final = {}
    for line in judged.read_text().splitlines():
        row = json.loads(line)
        turn = row.get("turn")
        if isinstance(turn, int) and turn in raw:
            final[turn] = row
    if set(final) != set(raw):
        raise ValueError(
            f"judgment coverage mismatch: judged={sorted(final)} observed={sorted(raw)}"
        )
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("turns_scored") != len(raw)
        or not summary.get("judge_model")
        or not summary.get("judge_version")
    ):
        raise ValueError(f"invalid judge summary: {summary_path}")

    tool = []
    instruction = []
    kb = []
    passed = []
    ttfat = []
    thinking_tokens = 0
    for turn in range(N_TURNS):
        scores = (final.get(turn, {}).get("scores") or {})
        values = (
            scores.get("tool_use_correct") is True,
            scores.get("instruction_following") is True,
            scores.get("kb_grounding") is True,
        )
        if turn in final and not all(
            isinstance(scores.get(key), bool)
            for key in ("tool_use_correct", "instruction_following", "kb_grounding")
        ):
            raise ValueError(f"invalid judgment schema: {run_dir} turn={turn}")
        tool.append(values[0])
        instruction.append(values[1])
        kb.append(values[2])
        passed.append(all(values))
        raw_row = raw.get(turn, {})
        latency = raw_row.get("ttfb_ms")
        if isinstance(latency, (int, float)) and math.isfinite(latency) and latency >= 0:
            ttfat.append(float(latency))
        tokens = raw_row.get("tokens") or {}
        reasoning = tokens.get("thinking_tokens")
        if isinstance(reasoning, int) and reasoning > 0:
            thinking_tokens += reasoning
        if raw_row.get("assistant_thought"):
            raise ValueError(f"thinking-off transcript contains assistant_thought: {run_dir}")
    if thinking_tokens:
        raise ValueError(f"thinking-off transcript reports thinking tokens: {run_dir}")

    return Conversation(
        slot=slot,
        model=model,
        arm=arm,
        run_dir=run_dir,
        tool=tuple(tool),
        instruction=tuple(instruction),
        kb=tuple(kb),
        passed=tuple(passed),
        strict_complete=end_session_turns(transcript) == {29},
        ttfat_ms=tuple(ttfat),
        thought_tokens=thinking_tokens,
    )


def load_cells() -> tuple[dict[tuple[str, str], list[Conversation]], dict[str, list[str]]]:
    cells = {(model, arm): [] for model in MODELS for arm in ("nofiller", "dots96")}
    decision_slots = {
        "qwen36_27b": [
            row["slot"] for row in read_tsv(HERE / "frozen-qwen27-control-subset.tsv")
        ],
        "qwen36_35b": [
            row["slot"] for row in read_tsv(HERE / "frozen-qwen35-control-subset.tsv")
        ],
    }
    control27_rows = [
        row for row in read_tsv(CONTROL27 / "canonical.tsv") if row["mode"] == "none"
    ]
    if len(control27_rows) != 30:
        raise ValueError(f"27B no-filler control count is {len(control27_rows)}, expected 30")
    for row in control27_rows:
        run_dir = ROOT / row["run_dir"]
        cells[("qwen36_27b", "nofiller")].append(
            load_conversation(row["slot"], "qwen36_27b", "nofiller", run_dir, None)
        )

    state35 = HERE / "state/qwen35-control"
    if not (state35 / "judging/COMPLETE.json").is_file():
        raise ValueError("35B control judgments are incomplete")
    control35_rows = read_tsv(state35 / "canonical.tsv")
    if len(control35_rows) != 30:
        raise ValueError(f"35B no-filler control count is {len(control35_rows)}, expected 30")
    for row in control35_rows:
        run_dir = Path(row["run_dir"])
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        cells[("qwen36_35b", "nofiller")].append(
            load_conversation(
                row["slot"],
                "qwen36_35b",
                "nofiller",
                run_dir,
                row["transcript_sha256"],
            )
        )

    for model, lane in (
        ("qwen36_27b", "qwen27-dots"),
        ("qwen36_35b", "qwen35-dots"),
    ):
        state = HERE / "state" / lane
        if not (state / "judging/COMPLETE.json").is_file():
            raise ValueError(f"treatment judgments are incomplete: {lane}")
        rows = read_tsv(state / "canonical.tsv")
        if len(rows) not in {6, 10, 30}:
            raise ValueError(f"unexpected treatment N for {lane}: {len(rows)}")
        for row in rows:
            run_dir = Path(row["run_dir"])
            if not run_dir.is_absolute():
                run_dir = ROOT / run_dir
            cells[(model, "dots96")].append(
                load_conversation(
                    row["slot"],
                    model,
                    "dots96",
                    run_dir,
                    row["transcript_sha256"],
                )
            )

    for key, runs in cells.items():
        if len({run.run_dir.resolve() for run in runs}) != len(runs):
            raise ValueError(f"duplicate run in cell {key}")
    return cells, decision_slots


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [100 * max(0.0, center - spread), 100 * min(1.0, center + spread)]


def arm_summary(runs: list[Conversation], rng: np.random.Generator) -> dict:
    n = len(runs)
    matrices = {
        key: np.asarray(
            [getattr(run, "passed" if key == "pass" else key) for run in runs],
            dtype=float,
        )
        for key in ("pass", "tool", "instruction", "kb")
    }
    indexes = rng.integers(0, n, size=(BOOTSTRAPS, n))
    conversation_pass = matrices["pass"].mean(axis=1)
    boot_pass = conversation_pass[indexes].mean(axis=1) * 100
    denominator = n * N_TURNS
    pass_count = int(matrices["pass"].sum())
    complete = sum(run.strict_complete for run in runs)
    ttfat = [value for run in runs for value in run.ttfat_ms]
    result = {
        "n_attempts": n,
        "fixed_turn_denominator": denominator,
        "pass_count": pass_count,
        "pass_rate_pct": 100 * pass_count / denominator,
        "pass_rate_ci95": [
            float(np.percentile(boot_pass, 2.5)),
            float(np.percentile(boot_pass, 97.5)),
        ],
        "strict_complete_count": complete,
        "strict_completion_pct": 100 * complete / n,
        "strict_completion_ci95": wilson(complete, n),
        "ttfat_observations": len(ttfat),
        "ttfat_p50_ms": statistics.median(ttfat) if ttfat else None,
        "ttfat_p95_ms": float(np.percentile(ttfat, 95)) if ttfat else None,
        "thought_tokens": sum(run.thought_tokens for run in runs),
        "run_dirs": [str(run.run_dir.relative_to(ROOT)) for run in runs],
        "_conversation_pass": conversation_pass,
    }
    for key in ("tool", "instruction", "kb"):
        error_count = denominator - int(matrices[key].sum())
        result[f"{key}_error_count"] = error_count
        result[f"{key}_error_rate_pct"] = 100 * error_count / denominator
    return result


def effect_summary(control: dict, dots: dict, rng: np.random.Generator) -> dict:
    control_values = control["_conversation_pass"]
    dots_values = dots["_conversation_pass"]
    control_idx = rng.integers(
        0, len(control_values), size=(BOOTSTRAPS, len(control_values))
    )
    dots_idx = rng.integers(0, len(dots_values), size=(BOOTSTRAPS, len(dots_values)))
    boot_delta = (
        dots_values[dots_idx].mean(axis=1) - control_values[control_idx].mean(axis=1)
    ) * 100
    return {
        "pass_delta_points": dots["pass_rate_pct"] - control["pass_rate_pct"],
        "pass_delta_ci95": [
            float(np.percentile(boot_delta, 2.5)),
            float(np.percentile(boot_delta, 97.5)),
        ],
        "strict_completion_delta_points": (
            dots["strict_completion_pct"] - control["strict_completion_pct"]
        ),
    }


def clean_summary(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def adaptive_decision(
    control_runs: list[Conversation],
    dots_runs: list[Conversation],
    control: dict,
    dots: dict,
    effect: dict,
) -> dict:
    nc = len(control_runs)
    nd = len(dots_runs)
    if nc != 10 or nd not in {6, 10, 30}:
        raise ValueError(f"unsupported decision sample sizes: {nc}/{nd}")
    delta = effect["pass_delta_points"]
    control_failures = [
        sum(not run.passed[turn] for run in control_runs) for turn in range(N_TURNS)
    ]
    dots_failures = [
        sum(not run.passed[turn] for run in dots_runs) for turn in range(N_TURNS)
    ]
    recurring = []
    direction = "benefit" if delta > 0 else "harm" if delta < 0 else None
    for turn, (control_count, dots_count) in enumerate(
        zip(control_failures, dots_failures)
    ):
        control_rate = control_count / nc
        dots_rate = dots_count / nd
        if control_count >= 3 and dots_rate < control_rate:
            recurring.append(
                {
                    "turn": turn,
                    "direction": "benefit",
                    "control_failures": control_count,
                    "dots_failures": dots_count,
                }
            )
        if dots_count >= 3 and dots_rate > control_rate:
            recurring.append(
                {
                    "turn": turn,
                    "direction": "harm",
                    "control_failures": control_count,
                    "dots_failures": dots_count,
                }
            )
    aligned = [row for row in recurring if row["direction"] == direction]
    triggers = []
    if nd == 6:
        if abs(delta) >= 2.0:
            triggers.append("absolute pass-rate difference >= 2.0 points")
        if control["strict_completion_pct"] != dots["strict_completion_pct"]:
            triggers.append("strict-completion rates differ")
        action = "top_up_dots_to_10" if triggers else "stop_at_6"
    elif nd == 10:
        ci = effect["pass_delta_ci95"]
        if ci[0] > 0 or ci[1] < 0:
            triggers.append("whole-conversation bootstrap 95% interval excludes zero")
        if abs(delta) >= 3.0 and aligned:
            triggers.append(
                "absolute difference >= 3.0 points with aligned recurring same-turn direction"
            )
        if control["strict_completion_pct"] != dots["strict_completion_pct"]:
            triggers.append("strict-completion rates still differ at n=10")
        action = "promote_dots_to_30" if triggers else "stop_at_10"
    else:
        action = "n30_treatment_complete"
    return {
        "control_decision_n": 10,
        "dots_observed_n": nd,
        "action": action,
        "triggers": triggers,
        "recurring_turn_signals": recurring,
        "aggregate_aligned_recurring_turn_signals": aligned,
        "decision_pending": False,
    }


def main() -> int:
    cells, decision_slots = load_cells()
    rng = np.random.default_rng(SEED)
    payload = {
        "schema_version": 1,
        "artifact_status": "INTERIM",
        "protocol": {
            "benchmark": "aiwf_medium_context",
            "turns": 30,
            "thinking_mode": "disabled",
            "full_thinking_off_guaranteed": True,
            "filler": {"arm": "dots96", "glyph": ".", "count": 96, "position": "suffix"},
            "control_n": 30,
            "decision_control_n": 10,
            "missing_scripted_turns": "fail",
            "bootstrap_unit": "whole conversation",
            "bootstrap_samples": BOOTSTRAPS,
            "interleaved": False,
            "report_tier": "exploratory",
        },
        "models": {},
    }
    decisions = {}
    all_final = True
    for model in MODELS:
        control_runs = cells[(model, "nofiller")]
        dots_runs = cells[(model, "dots96")]
        subset_order = decision_slots[model]
        by_slot = {run.slot: run for run in control_runs}
        if set(subset_order) - set(by_slot):
            raise ValueError(f"decision subset missing slots for {model}")
        decision_controls = [by_slot[slot] for slot in subset_order]
        control_all = arm_summary(control_runs, rng)
        control_decision = arm_summary(decision_controls, rng)
        dots = arm_summary(dots_runs, rng)
        effect_all = effect_summary(control_all, dots, rng)
        effect_decision = effect_summary(control_decision, dots, rng)
        decision = adaptive_decision(
            decision_controls, dots_runs, control_decision, dots, effect_decision
        )
        final = decision["action"] in {
            "stop_at_6",
            "stop_at_10",
            "n30_treatment_complete",
        }
        all_final &= final
        meta = MODEL_META[model]
        payload["models"][model] = {
            **meta,
            "provider": "BaseTen",
            "serving": {
                "vllm": "0.26.0",
                "automatic_prefix_caching": True,
                "mamba_cache_mode": "align",
                "mtp_speculative_tokens": 2,
            },
            "report_tier": "exploratory",
            "noncontemporaneous_reused_control": True,
            "arms": {
                "nofiller": clean_summary(control_all),
                "dots96": clean_summary(dots),
            },
            "decision_control": clean_summary(control_decision),
            "effect": effect_all,
            "decision_effect": effect_decision,
            "adaptive_decision": decision,
        }
        decisions[model] = decision
    payload["artifact_status"] = "FINAL_EXPLORATORY" if all_final else "INTERIM"
    (HERE / "aggregates.json").write_text(json.dumps(payload, indent=2) + "\n")
    (HERE / "adaptive-decision.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_status": payload["artifact_status"],
                "models": decisions,
            },
            indent=2,
        )
        + "\n"
    )
    for model in MODELS:
        result = payload["models"][model]
        print(
            model,
            f"control={result['arms']['nofiller']['pass_rate_pct']:.2f}",
            f"dots={result['arms']['dots96']['pass_rate_pct']:.2f}",
            f"decision_delta={result['decision_effect']['pass_delta_points']:+.2f}",
            f"action={result['adaptive_decision']['action']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
