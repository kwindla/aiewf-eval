#!/usr/bin/env python3
"""Create the frozen, balanced BaseTen replacement schedule for Qwen3-8B."""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "schedule-baseten-qwen.tsv"
SEED = 20260721
BLOCKS = 15


def build_arms() -> list[str]:
    """Return 60 assignments: two of each arm in every block of four."""
    rng = random.Random(SEED)
    for _ in range(100_000):
        arms: list[str] = []
        for _block in range(BLOCKS):
            block = ["nofiller", "nofiller", "dots96", "dots96"]
            rng.shuffle(block)
            arms.extend(block)
        if all(
            not (arms[i] == arms[i + 1] == arms[i + 2] == arms[i + 3])
            for i in range(len(arms) - 3)
        ):
            return arms
    raise RuntimeError("unable to construct a schedule without four-run streaks")


def main() -> None:
    rows = [
        [f"baseten-qwen-{index:03d}", "qwen3_8b", arm, "qwen/qwen3-8b", "vllm-openai"]
        for index, arm in enumerate(build_arms(), start=1)
    ]
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["slot", "model", "arm", "requested_model", "service"])
        writer.writerows(rows)
    print(f"{OUTPUT.name}\t{hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}\t{len(rows)}")


if __name__ == "__main__":
    main()
