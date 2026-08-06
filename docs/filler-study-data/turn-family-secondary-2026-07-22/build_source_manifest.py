#!/usr/bin/env python3
"""Materialize the exact conversation, transcript, and judgment pool for auditing."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
N30_DIR = HERE.parent / "dot-stability-n30-2026-07-20"
GEMINI_DIR = HERE.parent / "gemini-minimal-dots-2026-07-21"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import campaign analyzer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    sources = (
        ("dot-stability-n30-2026-07-20", load_module("manifest_n30", N30_DIR / "analyze.py")),
        ("gemini-minimal-dots-2026-07-21", load_module("manifest_gemini", GEMINI_DIR / "analyze.py")),
    )
    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for campaign, module in sources:
        cells = module.load_all()
        for model in module.MODELS:
            for arm in ("nofiller", "dots96"):
                runs = cells[(model, arm)]
                if len(runs) != 30:
                    raise ValueError(f"expected 30 runs for {campaign}/{model}/{arm}")
                for run_index, run in enumerate(runs, start=1):
                    run_dir = run.run_dir.resolve()
                    if run_dir in seen:
                        raise ValueError(f"duplicate source run: {run_dir}")
                    seen.add(run_dir)
                    transcript = run_dir / "transcript.jsonl"
                    judgment = run_dir / "claude_judged.jsonl"
                    rows.append({
                        "campaign": campaign,
                        "model": model,
                        "arm": arm,
                        "run_index": run_index,
                        "run_dir": str(run_dir.relative_to(ROOT)),
                        "transcript_sha256": digest(transcript),
                        "judgment_sha256": digest(judgment),
                    })
    if len(rows) != 660 or len(seen) != 660:
        raise ValueError(f"expected 660 unique source conversations, found {len(rows)}")
    with (HERE / "source-manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} source-manifest rows")


if __name__ == "__main__":
    main()
