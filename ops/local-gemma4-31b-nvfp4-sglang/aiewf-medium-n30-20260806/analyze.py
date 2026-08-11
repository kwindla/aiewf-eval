#!/usr/bin/env python3
"""Aggregate the local cohort and compare it with the BaseTen BF16 cohort."""

from __future__ import annotations

import collections
import csv
import importlib.util
import json
import random
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE_CAMPAIGN = (
    ROOT / "ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806"
)
BASE_SCRIPT = BASE_CAMPAIGN / "analyze.py"
BASE_AGGREGATES = BASE_CAMPAIGN / "aggregates.json"
OUTPUT_JSON = HERE / "aggregates.json"
OUTPUT_REPORT = HERE / "REPORT.md"
TARGET = 30
N_TURNS = 30
DENOMINATOR = TARGET * N_TURNS
BOOTSTRAPS = 20_000
SEED = 20260806


def load_base_module():
    spec = importlib.util.spec_from_file_location("gemma4_frozen_analysis", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen analysis helpers: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.ROOT = ROOT
    module.CONFIG = HERE / "configuration.json"
    module.CANONICAL = HERE / "canonical.tsv"
    module.JUDGE_COMPLETE = HERE / "judging/COMPLETE.json"
    module.OUTPUT_JSON = OUTPUT_JSON
    module.OUTPUT_REPORT = OUTPUT_REPORT
    return module


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def difference_ci(local: list[int], base: list[int]) -> list[float]:
    rng = random.Random(SEED + 5090)
    values = []
    for _ in range(BOOTSTRAPS):
        local_rate = sum(local[rng.randrange(TARGET)] for _ in range(TARGET)) / DENOMINATOR
        base_rate = sum(base[rng.randrange(TARGET)] for _ in range(TARGET)) / DENOMINATOR
        values.append((local_rate - base_rate) * 100)
    return [percentile(values, 0.025), percentile(values, 0.975)]


def main() -> int:
    helpers = load_base_module()
    config = json.loads((HERE / "configuration.json").read_text())
    if config.get("target_eligible_runs") != TARGET:
        raise RuntimeError("frozen configuration target mismatch")
    complete = json.loads((HERE / "judging/COMPLETE.json").read_text())
    if complete.get("canonical_runs") != TARGET:
        raise RuntimeError("judging completion marker mismatch")
    canonical = read_tsv(HERE / "canonical.tsv")
    if [int(row["slot"]) for row in canonical] != list(range(1, TARGET + 1)):
        raise RuntimeError("canonical cohort is not exactly slots 1..30")

    conversations = [helpers.load_conversation(row) for row in canonical]
    strict = helpers.metric(conversations, None)
    components = {
        name: helpers.metric(conversations, name) for name in helpers.COMPONENTS
    }
    latencies = [value for row in conversations for value in row["latencies"]]
    latency = {
        "scope": "observed_scripted_turns_only",
        "count": len(latencies),
        "p50_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "max_ms": max(latencies),
    }
    per_turn_errors = []
    for turn in range(N_TURNS):
        errors = sum(not row["strict"][turn] for row in conversations)
        per_turn_errors.append(
            {"turn": turn, "errors": errors, "error_percent": errors / TARGET * 100}
        )
    end_counts = collections.Counter(row["end_kind"] for row in conversations)
    end_turns = collections.Counter(
        str(row["end_turn"]) for row in conversations if row["end_turn"] >= 0
    )
    token_totals = {
        name: sum(row["tokens"][name] for row in conversations)
        for name in conversations[0]["tokens"]
    }

    base = json.loads(BASE_AGGREGATES.read_text())
    local_per_conversation = [sum(row["strict"]) for row in conversations]
    base_per_conversation = [sum(row["strict"]) for row in base["conversations"]]
    delta = strict["rate_percent"] - base["strict"]["rate_percent"]
    comparison = {
        "reference": str(BASE_AGGREGATES.relative_to(ROOT)),
        "local_minus_base_strict_percentage_points": delta,
        "independent_conversation_cluster_bootstrap_95_percent": difference_ci(
            local_per_conversation, base_per_conversation
        ),
        "interpretation_caveat": (
            "Local uses NVFP4 weights, FP8 E4M3 KV, no MTP, and one RTX 5090; "
            "BaseTen uses BF16 weights/KV, NEXTN MTP, and two H100s."
        ),
    }
    turn_labels = {
        12: "Submit second session suggestion",
        14: "Offer vegan dietary request",
        15: "Submit confirmed vegan request",
        17: "Submit mobile-app support request",
        24: "Submit session vote",
    }
    turn_comparison = []
    for turn in sorted(turn_labels):
        local_errors = per_turn_errors[turn]["errors"]
        base_errors = base["per_turn_errors"][turn]["errors"]
        turn_comparison.append(
            {
                "turn": turn,
                "requirement": turn_labels[turn],
                "local_errors": local_errors,
                "base_errors": base_errors,
                "local_minus_base_errors": local_errors - base_errors,
            }
        )
    comparison["per_turn_error_comparison"] = turn_comparison
    payload = {
        "schema_version": 1,
        "model": config["model"],
        "provider": "Local RTX 5090",
        "serving": "SGLang v0.5.15.post1, NVFP4 weights, FP8 E4M3 KV, no MTP",
        "checkpoint": "RedHatAI/gemma-4-31B-it-NVFP4",
        "checkpoint_revision": "edafdf3dcaef23ff76f75b91edd6a4a975a399cf",
        "n_conversations": TARGET,
        "fixed_turn_denominator": DENOMINATOR,
        "full_30_turn_conversations": sum(
            row["observed_turns"] == N_TURNS for row in conversations
        ),
        "strict": strict,
        "components": components,
        "latency": latency,
        "end_session": {"kind_counts": dict(end_counts), "turn_counts": dict(end_turns)},
        "per_turn_errors": per_turn_errors,
        "token_totals": token_totals,
        "comparison_to_baseten_bf16": comparison,
        "conversations": conversations,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    local_ci = strict["conversation_cluster_bootstrap_95_percent"]
    base_ci = base["strict"]["conversation_cluster_bootstrap_95_percent"]
    delta_ci = comparison["independent_conversation_cluster_bootstrap_95_percent"]
    turn_rows = "\n".join(
        f"| {row['turn']} | {row['requirement']} | {row['base_errors']}/30 | "
        f"{row['local_errors']}/30 | {row['local_minus_base_errors']:+d} |"
        for row in turn_comparison
    )
    report = f"""# Gemma 4 31B local NVFP4 campaign

The local NVFP4-weights + FP8-KV cohort scored {strict['correct']}/{DENOMINATOR}
strict turns ({strict['rate_percent']:.1f}%, conversation-cluster bootstrap 95%
CI {local_ci[0]:.1f}–{local_ci[1]:.1f}%).

| Configuration | Strict pass | 95% CI | Tool error | Instruction error | KB error | TTFAT P50 | TTFAT P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BaseTen BF16 + MTP, 2xH100 | {base['strict']['rate_percent']:.1f}% | {base_ci[0]:.1f}–{base_ci[1]:.1f}% | {base['components']['tool_use_correct']['error_percent']:.1f}% | {base['components']['instruction_following']['error_percent']:.1f}% | {base['components']['kb_grounding']['error_percent']:.1f}% | {base['latency']['p50_ms']:.0f}ms | {base['latency']['p95_ms']:.0f}ms |
| Local NVFP4 + FP8 KV, RTX 5090 | {strict['rate_percent']:.1f}% | {local_ci[0]:.1f}–{local_ci[1]:.1f}% | {components['tool_use_correct']['error_percent']:.1f}% | {components['instruction_following']['error_percent']:.1f}% | {components['kb_grounding']['error_percent']:.1f}% | {latency['p50_ms']:.0f}ms | {latency['p95_ms']:.0f}ms |

Local minus BaseTen strict pass is {delta:+.1f} percentage points (independent
conversation-cluster bootstrap 95% CI {delta_ci[0]:+.1f} to {delta_ci[1]:+.1f}
points).

| Turn | Requirement | BaseTen errors | Local errors | Local − BaseTen |
|---:|---|---:|---:|---:|
{turn_rows}

The difference is highly concentrated: 16 of the 20 additional local errors
occur on turn 15, where the one-word confirmation `Yes.` must retrieve the
previously established name and vegan preference and call
`submit_dietary_request`. All 48 local strict failures are paired tool-use and
instruction-following failures; KB grounding is 900/900. Every conversation
completed all 30 turns and called `end_session` on scripted turn 29.

This is an end-to-end deployment comparison, not a weights-only quantization
ablation: the local arm also uses FP8 KV, omits MTP, runs on one RTX 5090, and
uses SGLang v0.5.15.post1; the BaseTen arm uses BF16 weights/KV, NEXTN MTP,
two H100s, and SGLang v0.5.16. Missing future turns count as failures. Latency
uses observed scripted turns; whole conversations are the bootstrap unit.

A subsequent serving follow-up found a compact BF16-KV layout that fits on the
5090 by reserving 16K full-attention slots and 5.6K sliding-window slots. Its
frozen N=30 cohort scored 863/900 strict (95.9%, cluster 95% CI 94.9–96.8%).
BF16 minus FP8 KV was +1.2 percentage points with an independent
conversation-cluster bootstrap 95% interval of -0.3 to +2.8 points. See
`../aiewf-medium-bf16kv-n30-20260806/REPORT.md` for that comparison.
"""
    OUTPUT_REPORT.write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
