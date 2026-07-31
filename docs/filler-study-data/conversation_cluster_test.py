#!/usr/bin/env python3
"""Conversation-cluster permutation test for two filler-study configurations.

Treatment is assigned per benchmark conversation, while each conversation contributes
up to 30 correlated turn outcomes. This test therefore shuffles whole conversations
between configurations and never permutes turns independently.

Usage:
    conversation_cluster_test.py MANIFEST CONFIG_A CONFIG_B [--turns]

The reported delta is CONFIG_A minus CONFIG_B. ``--turns`` adds a descriptive
per-turn difference table; it does not change the cluster-level test.
"""

import argparse
import json
import random
from pathlib import Path


N_PERMUTATIONS = 10_000
RANDOM_SEED = 20260720


def turn_pass(row: dict) -> bool | None:
    scores = row.get("scores") or {}
    values = [value for value in scores.values() if isinstance(value, bool)]
    return all(values) if values else None


def load_runs(manifest: Path, config: str) -> tuple[list[dict[int, bool]], int]:
    run_dirs = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        label, run_dir = line.split("\t", 1)
        if label == config:
            run_dirs.append(Path(run_dir.strip()))

    runs = []
    for run_dir in run_dirs:
        judged_path = run_dir / "claude_judged.jsonl"
        if not judged_path.exists():
            continue
        by_turn = {}
        for line in judged_path.read_text().splitlines():
            row = json.loads(line)
            passed = turn_pass(row)
            turn = row.get("turn")
            if passed is not None and isinstance(turn, int):
                by_turn[turn] = passed  # A retry's final judgment wins.
        if by_turn:
            runs.append(by_turn)
    return runs, len(run_dirs)


def pooled_rate(runs: list[dict[int, bool]]) -> float:
    observations = [passed for run in runs for passed in run.values()]
    if not observations:
        raise ValueError("configuration has no judged turn outcomes")
    return 100.0 * sum(observations) / len(observations)


def print_turn_differences(
    config_a: str,
    runs_a: list[dict[int, bool]],
    config_b: str,
    runs_b: list[dict[int, bool]],
) -> None:
    common_turns = sorted(
        set.intersection(
            *(set(run) for run in runs_a + runs_b),
        )
    )
    rows = []
    for turn in common_turns:
        rate_a = 100.0 * sum(run[turn] for run in runs_a) / len(runs_a)
        rate_b = 100.0 * sum(run[turn] for run in runs_b) / len(runs_b)
        if rate_a != rate_b:
            rows.append((abs(rate_a - rate_b), turn, rate_a, rate_b))

    print(f"descriptive turn differences ({config_a} vs {config_b}; no per-turn p-values):")
    for _, turn, rate_a, rate_b in sorted(rows, reverse=True):
        print(f"  turn {turn:2d}: {rate_a:5.1f}% vs {rate_b:5.1f}% ({rate_a - rate_b:+.1f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("config_a")
    parser.add_argument("config_b")
    parser.add_argument("--turns", action="store_true")
    args = parser.parse_args()

    runs_a, listed_a = load_runs(args.manifest, args.config_a)
    runs_b, listed_b = load_runs(args.manifest, args.config_b)
    if not runs_a or not runs_b:
        raise SystemExit(
            f"need judged runs in both configs; loaded {len(runs_a)} and {len(runs_b)}"
        )

    observed_delta = pooled_rate(runs_a) - pooled_rate(runs_b)
    pooled_runs = runs_a + runs_b
    count_a = len(runs_a)
    exceedances = 0
    random.seed(RANDOM_SEED)
    for _ in range(N_PERMUTATIONS):
        random.shuffle(pooled_runs)
        permuted_delta = (
            pooled_rate(pooled_runs[:count_a]) - pooled_rate(pooled_runs[count_a:])
        )
        if abs(permuted_delta) >= abs(observed_delta) - 1e-9:
            exceedances += 1

    p_value = (exceedances + 1) / (N_PERMUTATIONS + 1)
    loaded = f"{len(runs_a)}/{len(runs_b)}"
    listed = f"{listed_a}/{listed_b}"
    print(
        f"{args.config_a} vs {args.config_b}: delta={observed_delta:+.1f} "
        f"cluster-p={p_value:.4f} (loaded runs {loaded}; manifest rows {listed})"
    )
    if args.turns:
        print_turn_differences(args.config_a, runs_a, args.config_b, runs_b)


if __name__ == "__main__":
    main()
