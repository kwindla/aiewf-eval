#!/usr/bin/env python3
"""Build the descriptive GPT-5.4 filler-by-reasoning comparison artifact.

The two reasoning-effort slices came from separate collections with unequal
sample sizes. This artifact supports a descriptive visualization; it is not a
factorial interaction analysis.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMPAIGN = HERE / "dot-stability-n30-2026-07-20"
MANIFEST = HERE / "gpt54_stack_manifest.tsv"
FINAL_AGGREGATES = CAMPAIGN / "aggregates.json"
OUTPUT = HERE / "gpt54-reasoning-comparison.json"
BOOTSTRAPS = 100_000
SEED = 20260721

sys.path.insert(0, str(CAMPAIGN))
import analyze as primary  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_arm(runs: list[primary.Conversation], seed: int) -> tuple[dict, np.ndarray]:
    summary = primary.arm_summary(runs, np.random.default_rng(seed))
    conversation_pass = summary.pop("_conversation_pass")
    return {
        "pass_rate_pct": summary["pass_rate_pct"],
        "pass_rate_ci95": summary["pass_rate_ci95"],
        "strict_completion_pct": summary["strict_completion_pct"],
        "ttfat_p50_ms": summary["ttfat_p50_ms"],
        "ttfat_p95_ms": summary["ttfat_p95_ms"],
        "ttfat_max_ms": summary["ttfat_max_ms"],
    }, conversation_pass


def load_low_slice() -> tuple[dict, list[str], dict[str, str]]:
    cells: dict[str, list[primary.Conversation]] = defaultdict(list)
    included: list[str] = []
    source_hashes: dict[str, str] = {
        str(MANIFEST.relative_to(ROOT)): sha256(MANIFEST),
    }
    seen: set[Path] = set()
    for line in MANIFEST.read_text().splitlines():
        cell, run_dir_text = line.split("\t")
        if cell not in {"low_nofiller", "low_dots96"}:
            raise ValueError(f"unexpected low-effort cell: {cell}")
        arm = cell.removeprefix("low_")
        run_dir = (ROOT / run_dir_text).resolve()
        if run_dir in seen:
            raise ValueError(f"duplicate low-effort run: {run_dir}")
        seen.add(run_dir)
        cells[arm].append(primary.load_conversation("gpt54_low", arm, run_dir))
        relative = str(run_dir.relative_to(ROOT))
        included.append(relative)
        for name in ("transcript.jsonl", "claude_judged.jsonl"):
            path = run_dir / name
            source_hashes[f"{relative}/{name}"] = sha256(path)

    if set(cells) != {"nofiller", "dots96"}:
        raise ValueError(f"low-effort arm mismatch: {sorted(cells)}")
    if any(len(cells[arm]) != 8 for arm in cells):
        raise ValueError("the low-effort comparison must contain 8 conversations per arm")

    nofiller, control_pass = summarize_arm(cells["nofiller"], SEED)
    dots, dots_pass = summarize_arm(cells["dots96"], SEED + 1)
    rng = np.random.default_rng(SEED + 10_000)
    control_idx = rng.integers(0, len(control_pass), size=(BOOTSTRAPS, len(control_pass)))
    dots_idx = rng.integers(0, len(dots_pass), size=(BOOTSTRAPS, len(dots_pass)))
    boot_delta = (
        dots_pass[dots_idx].mean(axis=1) - control_pass[control_idx].mean(axis=1)
    ) * 100
    effect = dots["pass_rate_pct"] - nofiller["pass_rate_pct"]
    return {
        "n_per_arm": 8,
        "nofiller": nofiller,
        "dots96": dots,
        "effect": {
            "pass_delta_points": effect,
            "pass_delta_ci95": [
                float(np.percentile(boot_delta, 2.5)),
                float(np.percentile(boot_delta, 97.5)),
            ],
        },
    }, included, source_hashes


def load_none_slice() -> tuple[dict, dict[str, str]]:
    payload = json.loads(FINAL_AGGREGATES.read_text())
    model = payload.get("models", {}).get("gpt54", {})
    arms = model.get("arms", {})
    if set(arms) != {"nofiller", "dots96"}:
        raise ValueError("final GPT-5.4 aggregate arm mismatch")
    if any(arms[arm].get("n_attempts") != 30 for arm in arms):
        raise ValueError("final GPT-5.4 comparison is not n=30 per arm")
    effect = model.get("effect", {})
    return {
        "n_per_arm": 30,
        "nofiller": {
            key: arms["nofiller"][key]
            for key in (
                "pass_rate_pct",
                "pass_rate_ci95",
                "strict_completion_pct",
                "ttfat_p50_ms",
                "ttfat_p95_ms",
                "ttfat_max_ms",
            )
        },
        "dots96": {
            key: arms["dots96"][key]
            for key in (
                "pass_rate_pct",
                "pass_rate_ci95",
                "strict_completion_pct",
                "ttfat_p50_ms",
                "ttfat_p95_ms",
                "ttfat_max_ms",
            )
        },
        "effect": {
            "pass_delta_points": effect["pass_delta_points"],
            "pass_delta_ci95": effect["pass_delta_ci95"],
        },
    }, {str(FINAL_AGGREGATES.relative_to(ROOT)): sha256(FINAL_AGGREGATES)}


def main() -> None:
    none, none_hashes = load_none_slice()
    low, included, low_hashes = load_low_slice()
    output = {
        "artifact_status": "DESCRIPTIVE_NOT_INTERACTION_TEST",
        "protocol": {
            "model": "gpt-5.4",
            "reasoning_effort_values": ["none", "low"],
            "filler_values": ["nofiller", "dots96"],
            "turns_per_conversation": 30,
            "bootstrap_samples": BOOTSTRAPS,
            "seed": SEED,
            "scope_note": (
                "Separate collections with unequal sample sizes; effect differences are "
                "descriptive and do not estimate a filler-by-reasoning interaction."
            ),
            "low_included_runs": included,
            "source_sha256": {**none_hashes, **low_hashes},
        },
        "reasoning_effort": {"none": none, "low": low},
        "descriptive_effect_difference_points": (
            low["effect"]["pass_delta_points"] - none["effect"]["pass_delta_points"]
        ),
    }
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(OUTPUT)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
