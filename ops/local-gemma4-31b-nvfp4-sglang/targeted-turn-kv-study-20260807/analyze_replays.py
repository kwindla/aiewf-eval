#!/usr/bin/env python3
"""Analyze cap parity and paired BF16-versus-FP8 targeted replay outcomes."""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from study import atomic_write_json, atomic_write_text


BOOTSTRAPS = 100_000
RANDOM_SEED = 20260807
TURNS = 2
MAX_LOOKS = 3
PRIMARY_LOOKS = (2048, 4096, 8192)
SIMULTANEOUS_TAIL_PROBABILITY = 0.05 / (2 * TURNS * MAX_LOOKS)


def rows(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            result.extend(json.loads(line) for line in handle if line.strip())
    return result


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / 2**discordant
    return min(1.0, 2 * probability)


def row_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    completion = row.get("completion") or {}
    return (
        completion.get("semantic_output_sha256"),
        row.get("score", {}).get("success"),
        row.get("score", {}).get("category"),
    )


def select_inferential_rows(raw: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select repeat zero explicitly and reject ambiguous logical cells.

    Repeatability artifacts may contain two or three identical executions of a
    logical `(arm, cache, snapshot, seed)` cell.  Only repeat zero is reusable
    for inference; later repeats remain gate evidence.  Silent dictionary
    overwrite would otherwise make the chosen repetition depend on file order.
    """

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in raw:
        groups[(row["arm"], row.get("cache_mode"), row["snapshot_id"], row["seed"])].append(row)

    selected = []
    repeated_groups = 0
    excluded_gate_repeats = 0
    for key, group in groups.items():
        if len(group) == 1:
            selected.append(group[0])
            continue
        repeated_groups += 1
        repeats = collections.Counter(int(row.get("repeat", 0)) for row in group)
        duplicate_repeats = [repeat for repeat, count in repeats.items() if count != 1]
        if duplicate_repeats:
            raise RuntimeError(f"duplicate repeat indices for logical cell {key}: {duplicate_repeats}")
        if 0 not in repeats:
            raise RuntimeError(f"repeated logical cell {key} has no preregistered repeat=0 row")
        signatures = {row_signature(row) for row in group}
        if len(signatures) != 1:
            raise RuntimeError(f"repeatability mismatch for logical cell {key}")
        selected.append(next(row for row in group if int(row.get("repeat", 0)) == 0))
        excluded_gate_repeats += len(group) - 1
    return selected, {
        "raw_rows": len(raw),
        "inferential_rows": len(selected),
        "repeated_logical_cells": repeated_groups,
        "excluded_gate_repeats": excluded_gate_repeats,
        "reused_repeat": 0,
    }


def normal_cluster_p(values: list[float]) -> float:
    """Two-sided exploratory seed-cluster Wald p-value.

    This fixed-look normal approximation is retained for secondary diagnostics.
    It is not the preregistered sequential confirmatory test.  ``erfc`` avoids
    the catastrophic cancellation of ``2 * (1 - cdf(z))`` in the far tail.
    """

    if len(values) < 2:
        return 1.0
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    if standard_error == 0:
        return 1.0 if mean == 0 else 0.0
    z = abs(mean / standard_error)
    return math.erfc(z / math.sqrt(2))


def paired_summary(selected: list[dict[str, Any]], snapshots: list[str]) -> dict[str, Any]:
    index = {
        (row["arm"], row["snapshot_id"], row["seed"]): int(row["score"]["success"])
        for row in selected
    }
    strata: dict[str, list[tuple[int, int]]] = {}
    seed_pairs: dict[str, dict[int, tuple[int, int]]] = {}
    for snapshot in snapshots:
        bf16_seeds = {seed for arm, name, seed in index if arm == "bf16" and name == snapshot}
        fp8_seeds = {seed for arm, name, seed in index if arm == "fp8" and name == snapshot}
        if bf16_seeds != fp8_seeds:
            raise RuntimeError(f"unpaired seed allocation for snapshot {snapshot}")
        seeds = sorted(bf16_seeds)
        strata[snapshot] = [(index[("bf16", snapshot, seed)], index[("fp8", snapshot, seed)]) for seed in seeds]
        seed_pairs[snapshot] = {
            seed: (index[("bf16", snapshot, seed)], index[("fp8", snapshot, seed)])
            for seed in seeds
        }
    if not strata or any(not pairs for pairs in strata.values()):
        raise RuntimeError("missing paired rows for one or more requested snapshots")
    seed_sets = [set(mapping) for mapping in seed_pairs.values()]
    if any(seed_set != seed_sets[0] for seed_set in seed_sets[1:]):
        raise RuntimeError("prefix strata do not share the frozen seed allocation")

    per_stratum = {
        snapshot: {
            "n": len(pairs),
            "bf16_success_percent": sum(a for a, _ in pairs) / len(pairs) * 100,
            "fp8_success_percent": sum(b for _, b in pairs) / len(pairs) * 100,
            "difference_points": sum(a - b for a, b in pairs) / len(pairs) * 100,
        }
        for snapshot, pairs in strata.items()
    }
    effect = sum(item["difference_points"] for item in per_stratum.values()) / len(per_stratum)
    bf16 = sum(item["bf16_success_percent"] for item in per_stratum.values()) / len(per_stratum)
    fp8 = sum(item["fp8_success_percent"] for item in per_stratum.values()) / len(per_stratum)

    # The same numeric seed drives the position-keyed Gumbel stream in every
    # prefix.  Resample it jointly across all prefix strata so uncertainty does
    # not pretend that 12 transformations of one random stream are independent.
    common_seeds = sorted(seed_sets[0])
    cluster_effects = [
        statistics.fmean(seed_pairs[snapshot][seed][0] - seed_pairs[snapshot][seed][1] for snapshot in snapshots)
        for seed in common_seeds
    ]
    rng = random.Random(RANDOM_SEED + sum(map(ord, "".join(snapshots))))
    cluster_bootstrap = []
    for _ in range(BOOTSTRAPS):
        draw = [cluster_effects[rng.randrange(len(cluster_effects))] for _ in cluster_effects]
        cluster_bootstrap.append(statistics.fmean(draw) * 100)

    sensitivity_rng = random.Random(RANDOM_SEED + 10_000 + sum(map(ord, "".join(snapshots))))
    independent_bootstrap = []
    for _ in range(BOOTSTRAPS):
        stratum_effects = []
        for pairs in strata.values():
            draw = [pairs[sensitivity_rng.randrange(len(pairs))] for _ in pairs]
            stratum_effects.append(statistics.fmean(a - b for a, b in draw) * 100)
        independent_bootstrap.append(statistics.fmean(stratum_effects))

    all_pairs = [pair for pairs in strata.values() for pair in pairs]
    bf16_only = sum(a == 1 and b == 0 for a, b in all_pairs)
    fp8_only = sum(a == 0 and b == 1 for a, b in all_pairs)
    return {
        "snapshot_count": len(strata),
        "pairs": len(all_pairs),
        "seed_clusters": len(cluster_effects),
        "equal_prefix_weighted_bf16_success_percent": bf16,
        "equal_prefix_weighted_fp8_success_percent": fp8,
        "equal_prefix_weighted_difference_points": effect,
        "seed_cluster_bootstrap_95_percent": [
            percentile(cluster_bootstrap, 0.025),
            percentile(cluster_bootstrap, 0.975),
        ],
        "independent_stratum_bootstrap_sensitivity_95_percent": [
            percentile(independent_bootstrap, 0.025),
            percentile(independent_bootstrap, 0.975),
        ],
        "two_turn_three_look_simultaneous_95_percent": [
            percentile(cluster_bootstrap, SIMULTANEOUS_TAIL_PROBABILITY),
            percentile(cluster_bootstrap, 1 - SIMULTANEOUS_TAIL_PROBABILITY),
        ],
        "bf16_only_successes": bf16_only,
        "fp8_only_successes": fp8_only,
        "seed_cluster_effects": {
            str(seed): effect for seed, effect in zip(common_seeds, cluster_effects)
        },
        "strata": per_stratum,
    }


def turn_interaction_summary(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare the turn-12 and turn-15 bank effects on the percentage-point scale."""

    output = {}
    for cache_mode in ("warm", "cold"):
        names = (f"{cache_mode}_turn12_bank", f"{cache_mode}_turn15_bank")
        if not all(name in comparisons for name in names):
            continue
        left = comparisons[names[0]]["seed_cluster_effects"]
        right = comparisons[names[1]]["seed_cluster_effects"]
        if set(left) != set(right):
            raise RuntimeError(f"turn interaction has unequal seed clusters in {cache_mode}")
        effects = [left[seed] - right[seed] for seed in sorted(left, key=int)]
        rng = random.Random(RANDOM_SEED + 20_000 + sum(map(ord, cache_mode)))
        bootstrap = []
        for _ in range(BOOTSTRAPS):
            draw = [effects[rng.randrange(len(effects))] for _ in effects]
            bootstrap.append(statistics.fmean(draw) * 100)
        output[cache_mode] = {
            "scale": "(BF16-FP8 turn12) - (BF16-FP8 turn15), percentage points",
            "seed_clusters": len(effects),
            "interaction_points": statistics.fmean(effects) * 100,
            "seed_cluster_bootstrap_95_percent": [
                percentile(bootstrap, 0.025),
                percentile(bootstrap, 0.975),
            ],
            "fixed_look_seed_cluster_wald_p_exploratory": normal_cluster_p(effects),
        }
    return output


def _expected_primary_stage(
    *, arm: str, snapshot_kind: str, seed: int
) -> str:
    """Return the frozen ABBA stage for one cumulative primary-look cell."""

    if snapshot_kind == "golden_mechanism":
        tiers = ((512, "primary"), (1024, "continue-4096"), (2048, "continue-8192"))
    elif snapshot_kind == "real_prefix_bank":
        tiers = ((128, "primary"), (256, "continue-4096"), (512, "continue-8192"))
    else:
        raise RuntimeError(f"unexpected primary snapshot kind: {snapshot_kind}")

    lower = 0
    for upper, stage in tiers:
        if seed < upper:
            halfway = lower + (upper - lower) // 2
            first_half = seed < halfway
            if first_half:
                block = 1 if arm == "fp8" else 2
            else:
                block = 4 if arm == "fp8" else 3
            return f"{stage}-block{block}"
        lower = upper
    raise RuntimeError(f"primary seed {seed} exceeds the frozen maximum")


def validate_primary_look(
    selected: list[dict[str, Any]],
    snapshot_manifest: dict[str, Any],
    look: int,
) -> dict[str, Any]:
    """Fail closed on the exact cumulative primary allocation.

    The collector audits each block against its plan.  This analysis-level gate
    independently requires the frozen 12-prefix bank, exact cumulative seed
    ranges, warm cache, repeat zero, and the FP8/BF16/BF16/FP8 stage mapping.
    """

    if look not in PRIMARY_LOOKS:
        raise RuntimeError(f"unsupported primary look: {look}")
    entries = snapshot_manifest.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("snapshot manifest has no entries list")
    expected_ids: dict[tuple[int, str], set[str]] = collections.defaultdict(set)
    for entry in entries:
        expected_ids[(int(entry["turn"]), str(entry["kind"]))].add(str(entry["snapshot_id"]))
    for turn in (12, 15):
        golden = expected_ids[(turn, "golden_mechanism")]
        bank = expected_ids[(turn, "real_prefix_bank")]
        if golden != {f"turn{turn}-golden"}:
            raise RuntimeError(f"turn {turn} manifest has unexpected golden snapshots: {sorted(golden)}")
        if len(bank) != 12:
            raise RuntimeError(f"turn {turn} manifest has {len(bank)} bank prefixes, expected 12")

    expected_golden_seeds = set(range(look // 4))
    expected_bank_seeds = set(range(look // 16))
    expected_total = look * 4  # two turns by two arms
    if len(selected) != expected_total:
        raise RuntimeError(
            f"primary look {look} has {len(selected)} inferential rows, expected {expected_total}"
        )

    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    errors = []
    for row in selected:
        arm = row.get("arm")
        cache_mode = row.get("cache_mode")
        repeat = int(row.get("repeat", 0))
        turn = int(row.get("turn", -1))
        snapshot_id = str(row.get("snapshot_id"))
        kind = str(row.get("snapshot_kind"))
        seed = int(row.get("seed", -1))
        if arm not in ("bf16", "fp8"):
            errors.append(f"{row.get('request_id')}: unexpected arm {arm}")
            continue
        if cache_mode != "warm":
            errors.append(f"{row.get('request_id')}: primary row is not warm")
        if repeat != 0:
            errors.append(f"{row.get('request_id')}: primary row is not repeat zero")
        if snapshot_id not in expected_ids.get((turn, kind), set()):
            errors.append(
                f"{row.get('request_id')}: snapshot/turn/kind is outside the frozen manifest"
            )
        try:
            expected_stage = _expected_primary_stage(
                arm=str(arm), snapshot_kind=kind, seed=seed
            )
        except RuntimeError as exc:
            errors.append(f"{row.get('request_id')}: {exc}")
        else:
            if row.get("collection_stage") != expected_stage:
                errors.append(
                    f"{row.get('request_id')}: stage {row.get('collection_stage')} != {expected_stage}"
                )
        groups[(str(arm), turn, snapshot_id)].append(row)

    missing_or_extra = []
    for arm in ("bf16", "fp8"):
        for turn in (12, 15):
            for kind in ("golden_mechanism", "real_prefix_bank"):
                expected_seeds = (
                    expected_golden_seeds if kind == "golden_mechanism" else expected_bank_seeds
                )
                for snapshot_id in sorted(expected_ids[(turn, kind)]):
                    values = groups.get((arm, turn, snapshot_id), [])
                    seeds = [int(row["seed"]) for row in values]
                    if len(seeds) != len(set(seeds)):
                        missing_or_extra.append(f"{arm}/{snapshot_id}: duplicate seeds")
                    actual = set(seeds)
                    if actual != expected_seeds:
                        missing = sorted(expected_seeds - actual)
                        extra = sorted(actual - expected_seeds)
                        missing_or_extra.append(
                            f"{arm}/{snapshot_id}: missing={missing[:10]} extra={extra[:10]}"
                        )
    if errors or missing_or_extra:
        details = errors[:20] + missing_or_extra[:20]
        raise RuntimeError("primary allocation validation failed: " + "; ".join(details))
    return {
        "passed": True,
        "look_cases_per_turn_arm": look,
        "rows": len(selected),
        "arms": 2,
        "turns": 2,
        "golden_seeds_per_turn_arm": len(expected_golden_seeds),
        "bank_prefixes_per_turn": 12,
        "bank_seeds_per_prefix_arm": len(expected_bank_seeds),
        "bank_pairs_per_turn": 12 * len(expected_bank_seeds),
        "seed_clusters_per_turn": len(expected_bank_seeds),
        "cache_mode": "warm",
        "schedule": "FP8/BF16/BF16/FP8",
    }


def primary_stop_decision(
    comparisons: dict[str, dict[str, Any]], look: int
) -> dict[str, Any]:
    """Apply the frozen two-point simultaneous-interval precision rule."""

    if look not in PRIMARY_LOOKS:
        raise RuntimeError(f"unsupported primary look: {look}")
    cells = {}
    for turn in (12, 15):
        name = f"warm_turn{turn}_bank"
        if name not in comparisons:
            raise RuntimeError(f"missing primary comparison {name}")
        item = comparisons[name]
        point = float(item["equal_prefix_weighted_difference_points"])
        low, high = map(float, item["two_turn_three_look_simultaneous_95_percent"])
        half_width = (high - low) / 2
        cells[name] = {
            "estimate_points": point,
            "simultaneous_interval": [low, high],
            "interval_half_width_points": half_width,
            "precision_target_points": 2.0,
            "precision_met": half_width <= 2.0,
        }
    precision_met = all(item["precision_met"] for item in cells.values())
    next_look = {2048: 4096, 4096: 8192}.get(look)
    if precision_met:
        decision = "stop_precision_met"
    elif next_look is not None:
        decision = "continue"
    else:
        decision = "stop_maximum_look"
    return {
        "look_cases_per_turn_arm": look,
        "decision": decision,
        "continue_required": decision == "continue",
        "next_look_cases_per_turn_arm": next_look if decision == "continue" else None,
        "cells": cells,
    }


def outcome_category_summary(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in selected:
        if row.get("arm") not in ("bf16", "fp8"):
            continue
        groups[(row["arm"], row["cache_mode"], int(row["turn"]), row["snapshot_kind"])].append(row)
    return [
        {
            "arm": arm,
            "cache_mode": cache_mode,
            "turn": turn,
            "snapshot_kind": kind,
            "rows": len(values),
            "categories": dict(collections.Counter(row["score"]["category"] for row in values)),
        }
        for (arm, cache_mode, turn, kind), values in sorted(groups.items())
    ]


def cache_did_summary(selected: list[dict[str, Any]], snapshots: list[str]) -> dict[str, Any]:
    index = {
        (row["arm"], row["cache_mode"], row["snapshot_id"], row["seed"]): int(row["score"]["success"])
        for row in selected
        if row["arm"] in ("fp8", "bf16") and row["cache_mode"] in ("warm", "cold")
    }
    strata: dict[str, dict[int, int]] = {}
    for snapshot in snapshots:
        seed_sets = [
            {seed for arm0, cache0, name, seed in index if arm0 == arm and cache0 == cache and name == snapshot}
            for arm in ("bf16", "fp8")
            for cache in ("warm", "cold")
        ]
        if not seed_sets or any(seed_set != seed_sets[0] for seed_set in seed_sets[1:]):
            raise RuntimeError(f"incomplete four-cell seed allocation for snapshot {snapshot}")
        strata[snapshot] = {
            seed: (index[("bf16", "warm", snapshot, seed)] - index[("fp8", "warm", snapshot, seed)])
            - (index[("bf16", "cold", snapshot, seed)] - index[("fp8", "cold", snapshot, seed)])
            for seed in sorted(seed_sets[0])
        }
    if not strata or any(not values for values in strata.values()):
        raise RuntimeError("missing four-cell cache pairs for one or more requested snapshots")
    shared_seeds = [set(values) for values in strata.values()]
    if any(seed_set != shared_seeds[0] for seed_set in shared_seeds[1:]):
        raise RuntimeError("cache prefix strata do not share the frozen seed allocation")
    per_stratum = {
        snapshot: {
            "n": len(values),
            "difference_in_differences_points": statistics.fmean(values.values()) * 100,
        }
        for snapshot, values in strata.items()
    }
    estimate = sum(item["difference_in_differences_points"] for item in per_stratum.values()) / len(per_stratum)
    cluster_effects = [
        statistics.fmean(strata[snapshot][seed] for snapshot in snapshots)
        for seed in sorted(shared_seeds[0])
    ]
    rng = random.Random(RANDOM_SEED + 1 + sum(map(ord, "".join(snapshots))))
    bootstrap = []
    for _ in range(BOOTSTRAPS):
        draw = [cluster_effects[rng.randrange(len(cluster_effects))] for _ in cluster_effects]
        bootstrap.append(statistics.fmean(draw) * 100)
    interval = [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)]
    return {
        "snapshot_count": len(strata),
        "four_cell_pairs": sum(len(values) for values in strata.values()),
        "seed_clusters": len(cluster_effects),
        "difference_in_differences_points": estimate,
        "seed_cluster_bootstrap_95_percent": interval,
        "equivalence_margin_points": [-3.0, 3.0],
        "equivalent_within_margin": interval[0] >= -3.0 and interval[1] <= 3.0,
        "strata": per_stratum,
    }


def holm_adjust(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


def cap_parity(selected: list[dict[str, Any]]) -> dict[str, Any] | None:
    relevant = [row for row in selected if row["arm"] in ("bf16-cap512", "bf16-cap8192", "fp8-cap512", "fp8-cap8192")]
    if not relevant:
        return None
    arms = sorted({row["arm"].split("-cap")[0] for row in relevant})
    result = {}
    for arm in arms:
        index = {
            (row["snapshot_id"], row["seed"], int(row["arm"].split("cap")[1])): row
            for row in relevant
            if row["arm"].startswith(arm + "-cap")
        }
        keys = sorted({(snapshot, seed) for snapshot, seed, cap in index})
        mismatches = []
        truncations = []
        for snapshot, seed in keys:
            low = index.get((snapshot, seed, 512))
            high = index.get((snapshot, seed, 8192))
            if low is None or high is None:
                mismatches.append({"snapshot_id": snapshot, "seed": seed, "reason": "missing cap"})
                continue
            low_sig = (
                (low.get("completion") or {}).get("semantic_output_sha256"),
                low["score"]["success"],
                low["score"]["category"],
            )
            high_sig = (
                (high.get("completion") or {}).get("semantic_output_sha256"),
                high["score"]["success"],
                high["score"]["category"],
            )
            if low_sig != high_sig:
                mismatches.append({"snapshot_id": snapshot, "seed": seed, "reason": "output"})
            if "length" in ((low.get("completion") or {}).get("finish_reasons") or []):
                truncations.append({"snapshot_id": snapshot, "seed": seed})
        result[arm] = {
            "pairs": len(keys),
            "mismatches": mismatches,
            "truncations_at_512": truncations,
            "passed": len(keys) == 128 and not mismatches and not truncations,
        }
    return result


def cache_discordance(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (row["arm"], row["snapshot_id"], row["seed"], row["cache_mode"]): row
        for row in selected
        if row["arm"] in ("fp8", "bf16")
    }
    groups: dict[tuple[str, int, str], list[bool]] = collections.defaultdict(list)
    bases = sorted({key[:3] for key in index})
    for arm, snapshot, seed in bases:
        warm = index.get((arm, snapshot, seed, "warm"))
        cold = index.get((arm, snapshot, seed, "cold"))
        if warm is None or cold is None:
            continue
        turn = int(warm["turn"])
        kind = str(warm["snapshot_kind"])
        groups[(arm, turn, kind)].append(warm["score"]["success"] != cold["score"]["success"])
    return [
        {
            "arm": arm,
            "turn": turn,
            "snapshot_kind": kind,
            "pairs": len(values),
            "outcome_discordance_percent": sum(values) / len(values) * 100,
        }
        for (arm, turn, kind), values in sorted(groups.items())
    ]


def greedy_summary(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in selected:
        if row["arm"] not in ("bf16-greedy", "fp8-greedy"):
            continue
        groups[(row["arm"].removesuffix("-greedy"), int(row["turn"]), row["snapshot_kind"])].append(row)
    return [
        {
            "arm": arm,
            "turn": turn,
            "snapshot_kind": kind,
            "snapshots": len(values),
            "success_percent": sum(row["score"]["success"] for row in values) / len(values) * 100,
            "categories": dict(collections.Counter(row["score"]["category"] for row in values)),
        }
        for (arm, turn, kind), values in sorted(groups.items())
    ]


def geometry_bridge_summary(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (row["arm"], row["snapshot_id"], row["seed"]): int(row["score"]["success"])
        for row in selected
        if row["arm"] in ("fp8", "fp8-historical-geometry") and row["cache_mode"] == "warm"
    }
    output = []
    for turn in (12, 15):
        for kind in ("golden_mechanism", "real_prefix_bank"):
            snapshots = sorted(
                {
                    row["snapshot_id"]
                    for row in selected
                    if row["turn"] == turn and row["snapshot_kind"] == kind
                }
            )
            pairs = []
            for snapshot in snapshots:
                seeds = sorted(
                    {seed for arm, name, seed in index if arm == "fp8" and name == snapshot}
                    & {seed for arm, name, seed in index if arm == "fp8-historical-geometry" and name == snapshot}
                )
                pairs.extend(
                    (
                        index[("fp8", snapshot, seed)],
                        index[("fp8-historical-geometry", snapshot, seed)],
                    )
                    for seed in seeds
                )
            if pairs:
                output.append(
                    {
                        "turn": turn,
                        "snapshot_kind": kind,
                        "pairs": len(pairs),
                        "compact_success_percent": sum(a for a, _ in pairs) / len(pairs) * 100,
                        "historical_success_percent": sum(b for _, b in pairs) / len(pairs) * 100,
                        "compact_minus_historical_points": sum(a - b for a, b in pairs) / len(pairs) * 100,
                    }
                )
    return output


def teacher_forced_summary(paths: list[Path]) -> list[dict[str, Any]]:
    rows = [row for path in paths for row in json.loads(path.read_text())["snapshots"]]
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["cache_mode"], int(row["turn"]), row["snapshot_kind"])].append(row)
    output = []
    for (cache_mode, turn, kind), values in sorted(groups.items()):
        index = {(row["arm"], row["snapshot_id"]): row for row in values}
        snapshots = sorted(
            {snapshot for arm, snapshot in index if arm == "bf16"}
            & {snapshot for arm, snapshot in index if arm == "fp8"}
        )
        if not snapshots:
            continue
        deltas = [
            index[("bf16", snapshot)]["canonical_sequence_logprob_mean"]
            - index[("fp8", snapshot)]["canonical_sequence_logprob_mean"]
            for snapshot in snapshots
        ]
        margin_snapshots = [
            snapshot
            for snapshot in snapshots
            if index[("bf16", snapshot)]["first_expected_minus_alternative_logprob"] is not None
            and index[("fp8", snapshot)]["first_expected_minus_alternative_logprob"] is not None
        ]
        margin_deltas = [
            index[("bf16", snapshot)]["first_expected_minus_alternative_logprob"]
            - index[("fp8", snapshot)]["first_expected_minus_alternative_logprob"]
            for snapshot in margin_snapshots
        ]
        output.append(
            {
                "cache_mode": cache_mode,
                "turn": turn,
                "snapshot_kind": kind,
                "matched_snapshots": len(snapshots),
                "mean_bf16_minus_fp8_per_token_logprob": sum(deltas) / len(deltas),
                "mean_bf16_minus_fp8_first_decision_margin": (
                    sum(margin_deltas) / len(margin_deltas) if margin_deltas else None
                ),
                "snapshot_deltas": dict(zip(snapshots, deltas)),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--teacher-forced", type=Path, action="append", default=[])
    parser.add_argument(
        "--primary-look",
        type=int,
        choices=PRIMARY_LOOKS,
        help="require and analyze an exact cumulative primary look",
    )
    parser.add_argument(
        "--snapshot-manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "snapshot-manifest.json",
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    raw = rows(args.results)
    selected, repeat_selection = select_inferential_rows(raw)
    primary_validation = None
    if args.primary_look is not None:
        primary_validation = validate_primary_look(
            selected,
            json.loads(args.snapshot_manifest.read_text()),
            args.primary_look,
        )

    comparisons = {}
    for cache_mode in ("warm", "cold"):
        cache_rows = [row for row in selected if row.get("cache_mode") == cache_mode and row["arm"] in ("fp8", "bf16")]
        for turn in (12, 15):
            golden = [f"turn{turn}-golden"]
            bank = sorted({row["snapshot_id"] for row in cache_rows if row["turn"] == turn and row["snapshot_kind"] == "real_prefix_bank"})
            if all(any(row["arm"] == arm and row["turn"] == turn for row in cache_rows) for arm in ("fp8", "bf16")):
                if golden[0] in {row["snapshot_id"] for row in cache_rows}:
                    comparisons[f"{cache_mode}_turn{turn}_golden"] = paired_summary(cache_rows, golden)
                if bank:
                    comparisons[f"{cache_mode}_turn{turn}_bank"] = paired_summary(cache_rows, bank)

    cache_did = {}
    for turn in (12, 15):
        golden = [f"turn{turn}-golden"]
        bank = sorted(
            {
                row["snapshot_id"]
                for row in selected
                if row.get("turn") == turn and row.get("snapshot_kind") == "real_prefix_bank"
            }
        )
        try:
            cache_did[f"turn{turn}_golden"] = cache_did_summary(selected, golden)
        except RuntimeError:
            pass
        if bank:
            try:
                cache_did[f"turn{turn}_bank"] = cache_did_summary(selected, bank)
            except RuntimeError:
                pass

    turn_interaction = turn_interaction_summary(comparisons)
    stop_decision = (
        primary_stop_decision(comparisons, args.primary_look)
        if args.primary_look is not None
        else None
    )

    payload = {
        "schema_version": 2,
        "source_files": [str(path) for path in args.results],
        "repeat_selection": repeat_selection,
        "primary_look_validation": primary_validation,
        "primary_stop_decision": stop_decision,
        "cap_parity": cap_parity(selected),
        "paired_comparisons": comparisons,
        "turn_interaction": turn_interaction,
        "primary_confirmatory_inference": {
            "method": "two-turn/three-look simultaneous seed-cluster bootstrap interval",
            "ordinary_fixed_look_p_values_suppressed": True,
            "reason": (
                "The permitted look is data-dependent through the precision stop rule. "
                "The simultaneous interval carries the preregistered sequential and two-turn adjustment; "
                "ordinary fixed-look Holm p-values do not."
            ),
            "protocol_deviation": (
                "Earlier cache-pilot artifacts emitted Holm p-values before the final stopped sample. "
                "Those values are exploratory and are not used here."
            ),
        },
        "cache_difference_in_differences": cache_did,
        "cache_outcome_discordance": cache_discordance(selected),
        "greedy_probe": greedy_summary(selected),
        "fp8_geometry_bridge": geometry_bridge_summary(selected),
        "teacher_forced_probe": teacher_forced_summary(args.teacher_forced),
        "outcome_categories": outcome_category_summary(selected),
    }
    atomic_write_json(args.json_output, payload)

    lines = ["# Targeted KV replay analysis", ""]
    if payload["cap_parity"]:
        lines.extend(["## 512-versus-8192 output-cap gate", "", "| Arm | Pairs | Mismatches | 512 truncations | Pass |", "|---|---:|---:|---:|:---:|"])
        for arm, item in payload["cap_parity"].items():
            lines.append(f"| {arm} | {item['pairs']} | {len(item['mismatches'])} | {len(item['truncations_at_512'])} | {'yes' if item['passed'] else 'no'} |")
        lines.append("")
    if comparisons:
        lines.extend(["## Paired KV effects", "", "Positive differences favor BF16 KV. The real-prefix bank weights every prefix equally. Because each prefix reuses the same position-keyed seed stream, the primary uncertainty calculation jointly resamples seed clusters across prefixes. The simultaneous interval is Bonferroni-safe across two turns and all three permitted sample-size looks and governs both the precision continuation rule and confirmatory inference. Ordinary fixed-look p-values are intentionally omitted.", "", "| Cell | Pairs / seed clusters | BF16 success | FP8 success | Difference (seed-cluster 95% CI) | Two-turn/three-look simultaneous CI |", "|---|---:|---:|---:|---:|---:|"])
        for name, item in comparisons.items():
            lo, hi = item["seed_cluster_bootstrap_95_percent"]
            sim_lo, sim_hi = item["two_turn_three_look_simultaneous_95_percent"]
            lines.append(
                f"| {name} | {item['pairs']} / {item['seed_clusters']} | {item['equal_prefix_weighted_bf16_success_percent']:.1f}% | "
                f"{item['equal_prefix_weighted_fp8_success_percent']:.1f}% | "
                f"{item['equal_prefix_weighted_difference_points']:+.1f} pp ({lo:+.1f}, {hi:+.1f}) | "
                f"({sim_lo:+.1f}, {sim_hi:+.1f}) |"
            )
        lines.append("")
    if primary_validation:
        lines.extend(
            [
                "## Primary-look integrity and stopping decision",
                "",
                f"The exact cumulative {args.primary_look:,}-case-per-turn/arm allocation passed: "
                f"{primary_validation['rows']:,} rows, 12 bank prefixes per turn, "
                f"{primary_validation['seed_clusters_per_turn']} paired seed clusters, warm cache, and the frozen ABBA stage mapping.",
                "",
                "| Cell | Estimate | Simultaneous interval | Interval half-width | ±2 pp target met |",
                "|---|---:|---:|---:|:---:|",
            ]
        )
        for name, item in stop_decision["cells"].items():
            low, high = item["simultaneous_interval"]
            lines.append(
                f"| {name} | {item['estimate_points']:+.1f} pp | ({low:+.1f}, {high:+.1f}) | "
                f"{item['interval_half_width_points']:.2f} pp | "
                f"{'yes' if item['precision_met'] else 'no'} |"
            )
        lines.extend(
            [
                "",
                f"Stopping decision: **{stop_decision['decision']}**. "
                + (
                    f"The next cumulative look is {stop_decision['next_look_cases_per_turn_arm']:,} cases per turn/arm."
                    if stop_decision["continue_required"]
                    else "No continuation is required by the frozen precision rule."
                ),
                "",
                "Confirmatory interpretation uses the simultaneous intervals above. Earlier pilot Holm p-values are exploratory and are not reused.",
                "",
            ]
        )
    if turn_interaction:
        lines.extend(["## Turn interaction", "", "This secondary contrast is `(BF16−FP8 at turn 12) − (BF16−FP8 at turn 15)` on the percentage-point scale. Its p-value is a fixed-look exploratory cluster-Wald approximation, not the confirmatory primary test.", "", "| Cache | Seed clusters | Interaction (seed-cluster 95% CI) | Exploratory fixed-look cluster-Wald p |", "|---|---:|---:|---:|"])
        for cache_mode, item in turn_interaction.items():
            lo, hi = item["seed_cluster_bootstrap_95_percent"]
            lines.append(
                f"| {cache_mode} | {item['seed_clusters']} | {item['interaction_points']:+.1f} pp "
                f"({lo:+.1f}, {hi:+.1f}) | {item['fixed_look_seed_cluster_wald_p_exploratory']:.4g} |"
            )
        lines.append("")
    if payload["cache_outcome_discordance"]:
        lines.extend(["## Warm-versus-cold outcome discordance", "", "| Arm | Turn | Prefix kind | Pairs | Discordance |", "|---|---:|---|---:|---:|"])
        for item in payload["cache_outcome_discordance"]:
            lines.append(f"| {item['arm']} | {item['turn']} | {item['snapshot_kind']} | {item['pairs']} | {item['outcome_discordance_percent']:.1f}% |")
        lines.append("")
    if cache_did:
        lines.extend(["## Cache difference-in-differences", "", "The estimate is `(BF16−FP8) warm − (BF16−FP8) cold`; equivalence requires the full seed-cluster interval to lie inside ±3 points.", "", "| Cell | Four-cell pairs / seed clusters | DiD (seed-cluster 95% CI) | ±3 pp equivalent |", "|---|---:|---:|:---:|"])
        for name, item in cache_did.items():
            lo, hi = item["seed_cluster_bootstrap_95_percent"]
            lines.append(
                f"| {name} | {item['four_cell_pairs']} / {item['seed_clusters']} | {item['difference_in_differences_points']:+.1f} pp "
                f"({lo:+.1f}, {hi:+.1f}) | {'yes' if item['equivalent_within_margin'] else 'no'} |"
            )
        lines.append("")
    if payload["greedy_probe"]:
        lines.extend(["## Greedy mechanism probe", "", "| Arm | Turn | Prefix kind | Snapshots | Success |", "|---|---:|---|---:|---:|"])
        for item in payload["greedy_probe"]:
            lines.append(
                f"| {item['arm']} | {item['turn']} | {item['snapshot_kind']} | "
                f"{item['snapshots']} | {item['success_percent']:.1f}% |"
            )
        lines.append("")
    if payload["fp8_geometry_bridge"]:
        lines.extend(["## FP8 pool-geometry bridge", "", "| Turn | Prefix kind | Pairs | Compact success | Historical success | Compact−historical |", "|---:|---|---:|---:|---:|---:|"])
        for item in payload["fp8_geometry_bridge"]:
            lines.append(
                f"| {item['turn']} | {item['snapshot_kind']} | {item['pairs']} | "
                f"{item['compact_success_percent']:.1f}% | {item['historical_success_percent']:.1f}% | "
                f"{item['compact_minus_historical_points']:+.1f} pp |"
            )
        lines.append("")
    if payload["teacher_forced_probe"]:
        lines.extend(["## Teacher-forced canonical tool sequence", "", "Positive log-probability differences favor BF16 KV for the exact expected tool-call suffix. The decision margin compares `<|tool_call>` with each arm's best first-token alternative.", "", "| Cache | Turn | Prefix kind | Matched snapshots | BF16−FP8 mean logp/token | BF16−FP8 first-decision margin |", "|---|---:|---|---:|---:|---:|"])
        for item in payload["teacher_forced_probe"]:
            margin = item["mean_bf16_minus_fp8_first_decision_margin"]
            margin_text = f"{margin:+.5f}" if margin is not None else "n/a"
            lines.append(
                f"| {item['cache_mode']} | {item['turn']} | {item['snapshot_kind']} | "
                f"{item['matched_snapshots']} | {item['mean_bf16_minus_fp8_per_token_logprob']:+.5f} | "
                f"{margin_text} |"
            )
        lines.append("")
    atomic_write_text(args.markdown_output, "\n".join(lines))
    print(
        json.dumps(
            {
                "comparisons": list(comparisons),
                "cap_parity": payload["cap_parity"],
                "primary_stop_decision": stop_decision,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
