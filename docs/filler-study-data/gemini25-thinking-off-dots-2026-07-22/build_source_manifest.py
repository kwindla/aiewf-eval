#!/usr/bin/env python3
"""Freeze the final Gemini 2.5 Flash run pool and content hashes."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LANES = ("control", "control-topup", "dots", "dots-topup", "focused")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = []
    seen = set()
    for lane in LANES:
        manifest = HERE / "state" / lane / "manifest.tsv"
        if not manifest.is_file():
            continue
        with manifest.open(newline="") as handle:
            for source in csv.DictReader(handle, delimiter="\t"):
                run_dir = (ROOT / source["run_dir"]).resolve()
                if run_dir in seen:
                    raise ValueError(f"duplicate run: {run_dir}")
                seen.add(run_dir)
                files = {
                    name: run_dir / name
                    for name in ("transcript.jsonl", "claude_judged.jsonl", "claude_summary.json", "run.log")
                }
                if not all(path.is_file() and path.stat().st_size for path in files.values()):
                    raise ValueError(f"incomplete run content: {run_dir}")
                rows.append({
                    "lane": lane,
                    "model": source["model"],
                    "arm": source["arm"],
                    "run_dir": str(run_dir.relative_to(ROOT)),
                    **{f"{name.replace('.', '_')}_sha256": digest(path) for name, path in files.items()},
                })
    if not rows:
        raise ValueError("no final run pool exists")
    fields = list(rows[0])
    output = HERE / "source-manifest.tsv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} with {len(rows)} frozen runs")


if __name__ == "__main__":
    main()
