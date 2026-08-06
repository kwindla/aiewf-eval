#!/usr/bin/env python3
"""Strict fixed-denominator analysis for the focused n=30 dot campaign."""

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
PRIMARY_LANES = ("openai-a", "openai-b", "lilac", "baseten")
QWEN_LANE = "baseten-qwen"
QWEN_MODEL = "qwen3_8b"
MODELS = ("gpt54", "terra", "gpt55", "sol", "gemma431", "inkling", "qwen3_8b", "glm52")
DISPLAY = {
    "gpt54": ("gpt-5.4", "OpenAI"),
    "terra": ("gpt-5.6-terra", "OpenAI"),
    "gpt55": ("gpt-5.5", "OpenAI"),
    "sol": ("gpt-5.6-sol", "OpenAI"),
    "gemma431": ("gemma-4-31b-it", "Lilac"),
    "inkling": ("inkling", "BaseTen"),
    "qwen3_8b": ("qwen3-8b", "BaseTen"),
    "glm52": ("glm-5.2", "BaseTen"),
}
ARMS = ("nofiller", "dots96")
N_TURNS = 30
TARGET = 30
BOOTSTRAPS = 100_000
SEED = 20260720


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


def end_session_turn_set(transcript: Path) -> set[int]:
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

    final: dict[int, dict] = {}
    for line in judged.read_text().splitlines():
        row = json.loads(line)
        turn = row.get("turn")
        if isinstance(turn, int) and 0 <= turn < N_TURNS:
            final[turn] = row

    transcript_final: dict[int, dict] = {}
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        turn = row.get("turn")
        if (
            isinstance(turn, int)
            and 0 <= turn < N_TURNS
            and row.get("recovery_turn") is not True
        ):
            transcript_final[turn] = row

    tool: list[bool] = []
    instruction: list[bool] = []
    kb: list[bool] = []
    passed: list[bool] = []
    ttfat: list[float] = []
    for turn in range(N_TURNS):
        row = final.get(turn, {})
        scores = row.get("scores") or {}
        values = (
            scores.get("tool_use_correct") is True,
            scores.get("instruction_following") is True,
            scores.get("kb_grounding") is True,
        )
        tool.append(values[0])
        instruction.append(values[1])
        kb.append(values[2])
        passed.append(all(values))
        latency = transcript_final.get(turn, {}).get("ttfb_ms")
        if isinstance(latency, (int, float)) and math.isfinite(latency) and latency >= 0:
            ttfat.append(float(latency))
    return Conversation(
        model=model,
        arm=arm,
        run_dir=run_dir,
        tool=tuple(tool),
        instruction=tuple(instruction),
        kb=tuple(kb),
        passed=tuple(passed),
        strict_complete=end_session_turn_set(transcript) == {29},
        ttfat_ms=tuple(ttfat),
    )


def load_all() -> dict[tuple[str, str], list[Conversation]]:
    refs: list[tuple[str, str, Path]] = []
    with (HERE / "existing-included.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            # The primary Qwen comparison is a provider-homogeneous replacement
            # cohort. Historical OpenRouter runs remain in the audit ledger only.
            if row["model"] == QWEN_MODEL:
                continue
            refs.append((row["model"], row["arm"], ROOT / row["run_dir"]))
    for lane in (*PRIMARY_LANES, QWEN_LANE):
        counted = HERE / "state" / lane / "counted.tsv"
        complete = HERE / "state" / lane / "COMPLETE"
        if not complete.is_file():
            raise ValueError(f"lane is not complete: {lane}")
        invalidated: set[tuple[str, str, str]] = set()
        invalidated_path = HERE / "invalidated.tsv"
        if invalidated_path.is_file():
            with invalidated_path.open(newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    invalidated.add((row["lane"], row["slot"], row["attempt"]))
        with counted.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if (lane, row["slot"], row["attempt"]) in invalidated:
                    continue
                if lane == QWEN_LANE and row["model"] != QWEN_MODEL:
                    raise ValueError(f"unexpected model in {QWEN_LANE}: {row['model']}")
                if lane != QWEN_LANE and row["model"] == QWEN_MODEL:
                    raise ValueError(f"Qwen primary attempt found outside {QWEN_LANE}: {lane}")
                run_dir = Path(row["run_dir"])
                refs.append((row["model"], row["arm"], run_dir if run_dir.is_absolute() else ROOT / run_dir))

    seen: set[Path] = set()
    cells: dict[tuple[str, str], list[Conversation]] = defaultdict(list)
    for model, arm, run_dir in refs:
        resolved = run_dir.resolve()
        if resolved in seen:
            raise ValueError(f"duplicate included run: {resolved}")
        seen.add(resolved)
        cells[(model, arm)].append(load_conversation(model, arm, resolved))
    expected_cells = {(model, arm) for model in MODELS for arm in ARMS}
    if set(cells) != expected_cells:
        missing = sorted(expected_cells - set(cells))
        unexpected = sorted(set(cells) - expected_cells)
        raise ValueError(f"cell mismatch; missing={missing}, unexpected={unexpected}")
    for model in MODELS:
        for arm in ARMS:
            n = len(cells[(model, arm)])
            if n != TARGET:
                raise ValueError(f"expected {TARGET} attempts for {model}/{arm}, found {n}")
    if len(seen) != len(MODELS) * len(ARMS) * TARGET:
        raise ValueError(f"expected 480 included conversations, found {len(seen)}")
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
    ttfat = [value for run in runs for value in run.ttfat_ms]
    complete = sum(run.strict_complete for run in runs)
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
        "any_error_rate_ci95": [
            float(np.percentile(100 - boot_pass, 2.5)),
            float(np.percentile(100 - boot_pass, 97.5)),
        ],
        "ttfat_observations": len(ttfat),
        "ttfat_p50_ms": statistics.median(ttfat) if ttfat else None,
        "ttfat_p95_ms": percentile(ttfat, 95) if ttfat else None,
        "ttfat_max_ms": max(ttfat) if ttfat else None,
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


def main() -> None:
    cells = load_all()
    output = {
        "protocol": {
            "target_per_arm": TARGET,
            "turns": N_TURNS,
            "bootstrap_samples": BOOTSTRAPS,
            "seed": SEED,
            "primary_sources": {
                QWEN_MODEL: {
                    "lane": QWEN_LANE,
                    "provider": "BaseTen",
                    "historical_attempts_included": 0,
                    "openrouter_attempts_included": 0,
                }
            },
        },
        "models": {},
    }
    tsv_rows: list[list[object]] = []
    for model_index, model in enumerate(MODELS):
        summaries: dict[str, dict] = {}
        for arm_index, arm in enumerate(ARMS):
            rng = np.random.default_rng(SEED + 100 * model_index + arm_index)
            summaries[arm] = arm_summary(cells[(model, arm)], rng)
        control = summaries["nofiller"]
        dots = summaries["dots96"]
        rng = np.random.default_rng(SEED + 10_000 + model_index)
        c = control.pop("_conversation_pass")
        d = dots.pop("_conversation_pass")
        idx_c = rng.integers(0, len(c), size=(BOOTSTRAPS, len(c)))
        idx_d = rng.integers(0, len(d), size=(BOOTSTRAPS, len(d)))
        boot_delta = (d[idx_d].mean(axis=1) - c[idx_c].mean(axis=1)) * 100
        delta = dots["pass_rate_pct"] - control["pass_rate_pct"]
        effect = {
            "pass_delta_points": delta,
            "pass_delta_ci95": [float(np.percentile(boot_delta, 2.5)), float(np.percentile(boot_delta, 97.5))],
            "any_error_reduction_points": delta,
            "any_error_reduction_ci95": [float(np.percentile(boot_delta, 2.5)), float(np.percentile(boot_delta, 97.5))],
        }
        display, provider = DISPLAY[model]
        output["models"][model] = {"display_name": display, "provider": provider, "arms": summaries, "effect": effect}
        row: list[object] = [display, provider, control["n_attempts"]]
        for arm_summary_row in (control, dots):
            for metric in ("pass_rate", "any_error_rate", "tool_error_rate", "instruction_error_rate", "kb_error_rate"):
                row.extend([
                    arm_summary_row[f"{metric}_pct"],
                    arm_summary_row[f"{metric}_ci95"][0],
                    arm_summary_row[f"{metric}_ci95"][1],
                ])
            row.append(arm_summary_row["strict_completion_pct"])
        row.extend([
            delta,
            effect["pass_delta_ci95"][0],
            effect["pass_delta_ci95"][1],
            control["ttfat_p50_ms"],
            control["ttfat_p95_ms"],
            control["ttfat_max_ms"],
        ])
        tsv_rows.append(row)

    (HERE / "aggregates.json").write_text(json.dumps(output, indent=2) + "\n")
    with (HERE / "aggregates.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        header = ["model", "provider", "n_per_arm"]
        for arm in ARMS:
            for metric in ("pass_rate", "any_error_rate", "tool_error_rate", "instruction_error_rate", "kb_error_rate"):
                header.extend([f"{arm}_{metric}_pct", f"{arm}_{metric}_ci95_low", f"{arm}_{metric}_ci95_high"])
            header.append(f"{arm}_strict_completion_pct")
        header.extend(["delta_points", "delta_ci95_low", "delta_ci95_high", "nofiller_ttfat_p50_ms", "nofiller_ttfat_p95_ms", "nofiller_ttfat_max_ms"])
        writer.writerow(header)
        writer.writerows(tsv_rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
