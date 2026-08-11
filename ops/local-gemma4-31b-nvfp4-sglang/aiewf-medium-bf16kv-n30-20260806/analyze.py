#!/usr/bin/env python3
"""Aggregate the BF16-KV cohort and compare it with both frozen references."""

from __future__ import annotations

import collections
import csv
import importlib.util
import json
import random
import statistics
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE_CAMPAIGN = ROOT / "ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806"
BASE_HELPERS = BASE_CAMPAIGN / "analyze.py"
BASE_AGGREGATES = BASE_CAMPAIGN / "aggregates.json"
FP8_CAMPAIGN = ROOT / "ops/local-gemma4-31b-nvfp4-sglang/aiewf-medium-n30-20260806"
FP8_AGGREGATES = FP8_CAMPAIGN / "aggregates.json"
OUTPUT_JSON = HERE / "aggregates.json"
OUTPUT_REPORT = HERE / "REPORT.md"
TARGET = 30
N_TURNS = 30
DENOMINATOR = TARGET * N_TURNS
BOOTSTRAPS = 20_000
SEED = 20260806


def load_helpers():
    spec = importlib.util.spec_from_file_location("gemma4_frozen_analysis", BASE_HELPERS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen analysis helpers: {BASE_HELPERS}")
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


def difference_ci(left: list[int], right: list[int], *, salt: int) -> list[float]:
    """Independent whole-conversation bootstrap for a rate difference."""
    if len(left) != TARGET or len(right) != TARGET:
        raise RuntimeError("comparison cohorts must each contain 30 conversations")
    rng = random.Random(SEED + salt)
    values = []
    for _ in range(BOOTSTRAPS):
        left_rate = sum(left[rng.randrange(TARGET)] for _ in range(TARGET)) / DENOMINATOR
        right_rate = sum(right[rng.randrange(TARGET)] for _ in range(TARGET)) / DENOMINATOR
        values.append((left_rate - right_rate) * 100)
    return [percentile(values, 0.025), percentile(values, 0.975)]


def comparison(
    local_rate: float,
    local_per_conversation: list[int],
    reference: dict[str, Any],
    reference_path: Path,
    *,
    salt: int,
    caveat: str,
) -> dict[str, Any]:
    reference_per_conversation = [
        sum(conversation["strict"]) for conversation in reference["conversations"]
    ]
    return {
        "reference": str(reference_path.relative_to(ROOT)),
        "local_minus_reference_strict_percentage_points": (
            local_rate - reference["strict"]["rate_percent"]
        ),
        "independent_conversation_cluster_bootstrap_95_percent": difference_ci(
            local_per_conversation, reference_per_conversation, salt=salt
        ),
        "interpretation_caveat": caveat,
    }


def main() -> int:
    helpers = load_helpers()
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
    fp8 = json.loads(FP8_AGGREGATES.read_text())
    local_per_conversation = [sum(row["strict"]) for row in conversations]
    comparisons = {
        "to_local_fp8_kv": comparison(
            strict["rate_percent"],
            local_per_conversation,
            fp8,
            FP8_AGGREGATES,
            salt=16,
            caveat=(
                "Both arms use the same NVFP4 checkpoint, RTX 5090, SGLang image, "
                "sampling settings, batch-one execution, and no MTP. Besides KV "
                "precision, the BF16 arm uses compact asymmetric KV pool limits."
            ),
        ),
        "to_baseten_bf16_mtp": comparison(
            strict["rate_percent"],
            local_per_conversation,
            base,
            BASE_AGGREGATES,
            salt=5090,
            caveat=(
                "Local uses NVFP4 weights, no MTP, one RTX 5090, and SGLang "
                "v0.5.15.post1; BaseTen uses BF16 weights, NEXTN MTP, two H100s, "
                "and SGLang v0.5.16. Both use BF16 KV."
            ),
        ),
    }
    turn_comparison = []
    for turn in range(N_TURNS):
        row = {
            "turn": turn,
            "baseten_bf16_mtp_errors": base["per_turn_errors"][turn]["errors"],
            "local_fp8_kv_errors": fp8["per_turn_errors"][turn]["errors"],
            "local_bf16_kv_errors": per_turn_errors[turn]["errors"],
        }
        if any(row[key] for key in row if key != "turn"):
            turn_comparison.append(row)

    payload = {
        "schema_version": 1,
        "model": config["model"],
        "provider": "Local RTX 5090",
        "serving": (
            "SGLang v0.5.15.post1, NVFP4 weights, BF16 KV, asymmetric "
            "16K full-attention/5.6K SWA pools, batch one, no MTP"
        ),
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
        "comparisons": comparisons,
        "per_turn_error_comparison": turn_comparison,
        "conversations": conversations,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    local_ci = strict["conversation_cluster_bootstrap_95_percent"]
    fp8_ci = fp8["strict"]["conversation_cluster_bootstrap_95_percent"]
    base_ci = base["strict"]["conversation_cluster_bootstrap_95_percent"]
    fp8_delta = comparisons["to_local_fp8_kv"][
        "local_minus_reference_strict_percentage_points"
    ]
    fp8_delta_ci = comparisons["to_local_fp8_kv"][
        "independent_conversation_cluster_bootstrap_95_percent"
    ]
    base_delta = comparisons["to_baseten_bf16_mtp"][
        "local_minus_reference_strict_percentage_points"
    ]
    base_delta_ci = comparisons["to_baseten_bf16_mtp"][
        "independent_conversation_cluster_bootstrap_95_percent"
    ]
    turn_rows = "\n".join(
        f"| {row['turn']} | {row['baseten_bf16_mtp_errors']}/30 | "
        f"{row['local_fp8_kv_errors']}/30 | {row['local_bf16_kv_errors']}/30 |"
        for row in turn_comparison
    ) or "| — | 0/30 | 0/30 | 0/30 |"
    report = f"""# Gemma 4 31B local NVFP4 + BF16-KV campaign

The frozen batch-one BF16-KV cohort scored {strict['correct']}/{DENOMINATOR}
strict turns ({strict['rate_percent']:.1f}%, conversation-cluster bootstrap 95%
CI {local_ci[0]:.1f}–{local_ci[1]:.1f}%).

| Configuration | Strict pass | 95% CI | Tool error | Instruction error | KB error | TTFAT P50 | TTFAT P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BaseTen BF16 weights/KV + MTP, 2xH100 | {base['strict']['rate_percent']:.1f}% | {base_ci[0]:.1f}–{base_ci[1]:.1f}% | {base['components']['tool_use_correct']['error_percent']:.1f}% | {base['components']['instruction_following']['error_percent']:.1f}% | {base['components']['kb_grounding']['error_percent']:.1f}% | {base['latency']['p50_ms']:.0f}ms | {base['latency']['p95_ms']:.0f}ms |
| Local NVFP4 weights + FP8 KV, RTX 5090 | {fp8['strict']['rate_percent']:.1f}% | {fp8_ci[0]:.1f}–{fp8_ci[1]:.1f}% | {fp8['components']['tool_use_correct']['error_percent']:.1f}% | {fp8['components']['instruction_following']['error_percent']:.1f}% | {fp8['components']['kb_grounding']['error_percent']:.1f}% | {fp8['latency']['p50_ms']:.0f}ms | {fp8['latency']['p95_ms']:.0f}ms |
| Local NVFP4 weights + BF16 KV, RTX 5090 | {strict['rate_percent']:.1f}% | {local_ci[0]:.1f}–{local_ci[1]:.1f}% | {components['tool_use_correct']['error_percent']:.1f}% | {components['instruction_following']['error_percent']:.1f}% | {components['kb_grounding']['error_percent']:.1f}% | {latency['p50_ms']:.0f}ms | {latency['p95_ms']:.0f}ms |

Local BF16 KV minus local FP8 KV is {fp8_delta:+.1f} percentage points
(independent conversation-cluster bootstrap 95% CI {fp8_delta_ci[0]:+.1f} to
{fp8_delta_ci[1]:+.1f}). This is the cleanest available estimate of KV-cache
precision impact: weights, hardware, SGLang image, sampling, batch size, and MTP
setting are held fixed. The compact BF16 arm necessarily uses smaller,
asymmetric static KV pools.

Local BF16 KV minus BaseTen BF16 weights/KV + MTP is {base_delta:+.1f} points
(95% CI {base_delta_ci[0]:+.1f} to {base_delta_ci[1]:+.1f}). This remains an
end-to-end deployment comparison because weight precision, MTP, hardware, and
SGLang version differ.

| Turn | BaseTen BF16 + MTP errors | Local FP8-KV errors | Local BF16-KV errors |
|---:|---:|---:|---:|
{turn_rows}

Missing future turns after a model-caused early exit count as failures. Latency
uses observed scripted turns only. Whole conversations, not individual turns,
are the bootstrap unit.
"""
    OUTPUT_REPORT.write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
