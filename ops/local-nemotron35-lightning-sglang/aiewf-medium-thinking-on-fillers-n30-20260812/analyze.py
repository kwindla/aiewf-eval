#!/usr/bin/env python3
"""Compare Nemotron thinking-on dots/dashes with the frozen no-filler arm."""

from __future__ import annotations

import csv
import importlib.util
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = Path(__file__).resolve().parent
CAMPAIGN = PROTOCOL / "artifacts"
CONTROL_PROTOCOL = PROTOCOL.parent / "aiewf-medium-binary-n30-20260811"
BASE_ANALYZER = CONTROL_PROTOCOL / "analyze.py"
ARMS = ("nofiller", "dots96", "dashes96")
BOOTSTRAPS = 20_000
SEED = 350812


def load_base():
    spec = importlib.util.spec_from_file_location("nemotron_binary_analyzer", BASE_ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_ANALYZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [ordered[499], ordered[19499]]


def bootstrap_difference(left: list[int], right: list[int]) -> list[float]:
    rng = random.Random(SEED + sum(left) + sum(right))
    nl = len(left)
    nr = len(right)
    estimates = []
    for _ in range(BOOTSTRAPS):
        lrate = sum(left[rng.randrange(nl)] for _ in range(nl)) / (nl * 30)
        rrate = sum(right[rng.randrange(nr)] for _ in range(nr)) / (nr * 30)
        estimates.append(100 * (lrate - rrate))
    return interval(estimates)


def strict_by_turn(conversations: list[dict]) -> dict[str, int]:
    result = {str(turn): 0 for turn in range(30)}
    for conversation in conversations:
        run_dir = ROOT / conversation["run_dir"]
        for row in BASE.read_jsonl(run_dir / "claude_judged.jsonl"):
            turn = int(row["turn"])
            if not 0 <= turn < 30:
                continue
            scores = row["scores"]
            result[str(turn)] += int(
                scores["tool_use_correct"]
                and scores["instruction_following"]
                and scores["kb_grounding"]
            )
    return result


def main() -> int:
    fresh_entries = read_tsv(CAMPAIGN / "canonical.tsv")
    control_entries = [
        row
        for row in read_tsv(CONTROL_PROTOCOL / "artifacts/canonical.tsv")
        if row["arm"] == "on-unbounded"
    ]
    if len(fresh_entries) != 60 or len(control_entries) != 30:
        raise SystemExit("expected 60 fresh and 30 frozen-control conversations")

    entry_map = {
        "nofiller": control_entries,
        "dots96": [row for row in fresh_entries if row["arm"] == "dots96"],
        "dashes96": [row for row in fresh_entries if row["arm"] == "dashes96"],
    }
    conversations = {
        arm: [BASE.load_conversation(entry) for entry in entries]
        for arm, entries in entry_map.items()
    }
    aggregates = {
        arm: {
            **BASE.aggregate(arm, rows),
            "strict_pass_by_turn": strict_by_turn(rows),
        }
        for arm, rows in conversations.items()
    }
    vectors = {
        arm: [row["strict_passes"] for row in rows]
        for arm, rows in conversations.items()
    }
    effects = {}
    for left, right in (
        ("dots96", "nofiller"),
        ("dashes96", "nofiller"),
        ("dashes96", "dots96"),
    ):
        effects[f"{left}_minus_{right}"] = {
            "difference_points": (
                aggregates[left]["strict_pass_rate_pct"]
                - aggregates[right]["strict_pass_rate_pct"]
            ),
            "conversation_bootstrap_ci95_points": bootstrap_difference(
                vectors[left], vectors[right]
            ),
        }

    payload = {
        "schema_version": 1,
        "model": "nemotron-3.5-lightning",
        "mode": "thinking on, unbounded",
        "design": {
            "conversations_per_arm": 30,
            "fixed_turn_denominator_per_arm": 900,
            "control": "frozen on-unbounded cohort from binary campaign",
            "fresh_arms": "dots96 and dashes96, sequentially interleaved",
            "filler_history": "not persisted",
        },
        "bootstrap": {
            "unit": "conversation",
            "replicates": BOOTSTRAPS,
            "seed_base": SEED,
        },
        "arms": aggregates,
        "effects": effects,
    }
    analysis_dir = CAMPAIGN / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "aggregates.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Nemotron 3.5 Lightning thinking-on filler results",
        "",
        "Each arm contains 30 full assigned conversations and uses a fixed "
        "900-turn denominator. The no-filler arm is frozen from the preceding "
        "binary campaign; dots and dashes were collected fresh and interleaved.",
        "",
        "| Arm | Pass rate | 95% CI | Full conversations | TTFAT P50 / P95 | Raw TTFT P50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = aggregates[arm]
        ci = row["strict_pass_cluster_bootstrap_ci95_pct"]
        lines.append(
            f"| {arm} | {row['strict_pass_count']}/900 "
            f"({row['strict_pass_rate_pct']:.1f}%) | {ci[0]:.1f}–{ci[1]:.1f}% | "
            f"{row['complete_30_conversations']}/30 | "
            f"{row['ttfat_p50_ms']:.0f}/{row['ttfat_p95_ms']:.0f} ms | "
            f"{row['raw_ttft_p50_ms']:.0f} ms |"
        )
    lines.extend(["", "## Effects", ""])
    for key, effect in effects.items():
        ci = effect["conversation_bootstrap_ci95_points"]
        lines.append(
            f"- `{key}`: {effect['difference_points']:+.2f} points "
            f"(whole-conversation bootstrap 95% CI {ci[0]:+.2f} to {ci[1]:+.2f})."
        )
    lines.extend(
        [
            "",
            "## Selected tool-commitment turns",
            "",
            "| Turn | nofiller | dots96 | dashes96 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for turn in (11, 12, 15, 17, 24, 25, 29):
        lines.append(
            f"| {turn} | "
            + " | ".join(
                f"{aggregates[arm]['strict_pass_by_turn'][str(turn)]}/30"
                for arm in ARMS
            )
            + " |"
        )
    (analysis_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
