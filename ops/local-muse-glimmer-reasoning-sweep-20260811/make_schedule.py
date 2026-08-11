#!/usr/bin/env python3
"""Write the preregistered balanced Muse Glimmer arm schedule."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "schedule.tsv"
ARMS = ("low", "medium", "high", "xhigh")

# Four-treatment Williams design. Seven complete repetitions yield 28 runs per
# arm; the first two rows add two more per arm for the requested N=30.
ROWS = (
    ("low", "medium", "xhigh", "high"),
    ("medium", "high", "low", "xhigh"),
    ("high", "xhigh", "medium", "low"),
    ("xhigh", "low", "high", "medium"),
)


def main() -> int:
    sequences = list(ROWS) * 7 + list(ROWS[:2])
    counts = {arm: 0 for arm in ARMS}
    rows: list[dict[str, int | str]] = []
    ordinal = 0
    for block, sequence in enumerate(sequences, start=1):
        for position, arm in enumerate(sequence, start=1):
            ordinal += 1
            counts[arm] += 1
            rows.append(
                {
                    "ordinal": ordinal,
                    "block": block,
                    "position": position,
                    "arm": arm,
                    "arm_index": counts[arm],
                }
            )
    assert ordinal == 120
    assert counts == {arm: 30 for arm in ARMS}

    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUT}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
