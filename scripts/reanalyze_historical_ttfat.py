#!/usr/bin/env python3
"""Offline supplementary reanalysis for frozen reports; never launch captures.

Run from the repository root with local raw inputs available. Ordinary campaign
analyzers remain directly executable. This handles reports originally assembled
interactively and a frozen analyzer whose source verification must use history.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/ttfat-reanalysis-2026-10"
SOURCE = "9fcb3a965ba7134b02f08bef668db7d33833f266"


def module(path):
    spec = importlib.util.spec_from_file_location(
        "reanalysis_" + hashlib.sha256(str(path).encode()).hexdigest()[:12], ROOT / path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    sys.path.insert(0, str((ROOT / path).parent))
    spec.loader.exec_module(mod)
    return mod


def historical_manifest(path):
    """Verify every frozen source SHA against an actual historical Git blob.

    Unlike capture preflight, reanalysis must not require the executing analyzer
    to have the old source hash. Never modify the frozen source manifest.
    """
    contents = path.read_text()
    relative = str(path.relative_to(ROOT))
    assert contents == subprocess.check_output(
        ["git", "show", f"origin/main:{relative}"], cwd=ROOT, text=True
    )
    evidence = []
    for line in contents.splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.strip()
        revisions = subprocess.check_output(
            ["git", "log", "--format=%H", "origin/main", "--", name],
            cwd=ROOT,
            text=True,
        ).splitlines()
        for rev in revisions:
            blob = subprocess.run(
                ["git", "show", f"{rev}:{name}"], cwd=ROOT, capture_output=True
            )
            if (
                blob.returncode == 0
                and hashlib.sha256(blob.stdout).hexdigest() == expected
            ):
                evidence.append({"path": name, "sha256": expected, "commit": rev})
                break
        else:
            raise ValueError(f"Cannot verify frozen source {name}: {expected}")
    return contents, evidence


def paired_gemma():
    name = "ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py"
    m = module(name)
    capture_text, capture_evidence = historical_manifest(m.collect.SOURCE_HASH_PATH)
    judge_text, judge_evidence = historical_manifest(m.judge_stage.SOURCE_HASH_PATH)
    # Reanalysis validates the frozen code against Git history, not today's code.
    # All configuration, cohort, transcript, judgment and completion checks remain.
    m.collect.validate_source_hashes = lambda: historical_manifest(
        m.collect.SOURCE_HASH_PATH
    )
    m.judge_stage.source_hash_text = lambda: judge_text
    result, included = m.analyze("full")
    destination = ROOT / "ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis"
    (destination / "aggregates-full.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (destination / "REPORT-full.md").write_text(m.render_markdown(result))
    (OUT / "paired-gemma-provenance.json").write_text(
        json.dumps(
            {
                "capture_sources": capture_evidence,
                "judge_sources": judge_evidence,
                "executed_analyzer": name,
                "executed_sha256": hashlib.sha256(
                    (ROOT / name).read_bytes()
                ).hexdigest(),
                "included_run_ids": [r["run_dir"] for r in included],
            },
            indent=2,
        )
        + "\n"
    )


def transcript_summary(paths):
    from scripts import benchmark_summary as summary
    from scripts.analyze_turn_metrics import analyze_run_transcript_only

    def transcript_metrics(command, **kwargs):
        turns, _ = analyze_run_transcript_only(Path(command[4]))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "turns": [
                        {
                            "turn": t.turn_index,
                            "server_ttfb_ms": t.server_ttfb_ms,
                            "raw_ttfb_ms": t.raw_ttfb_ms,
                            "wav_v2v_ms": None,
                            "silent_pad_silero_ms": None,
                            "has_tool_call": t.has_tool_call,
                        }
                        for t in turns
                    ]
                }
            ),
        )

    # Same analyzer function as the CLI, without one child process per text run.
    original_run = summary.subprocess.run
    summary.subprocess.run = transcript_metrics
    try:
        return summary.aggregate_runs(paths)
    finally:
        summary.subprocess.run = original_run


def muse():
    from scripts import benchmark_summary as summary

    for campaign in [
        "muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z",
        "muse-glimmer-reasoning-strength-n30-20260811",
    ]:
        folder = ROOT / "runs" / campaign
        aggregates = sorted(folder.glob("aggregate*.json"))
        for aggregate in aggregates:
            # Use the original recorded inclusion order, never a new glob cohort.
            records = (
                (folder / "included.tsv").read_text().replace("\r", "").splitlines()
                if (folder / "included.tsv").exists()
                else ["run_dir"]
                + (folder / "included-runs.txt").read_text().splitlines()
            )
            header = records[0].split("\t")
            rows = [dict(zip(header, line.split("\t"))) for line in records[1:] if line]
            arm = aggregate.stem.removeprefix("aggregate-")
            paths = [
                ROOT / r["run_dir"]
                for r in rows
                if aggregate.name == "aggregate.json" or r.get("arm") == arm
            ]
            assert len(paths) == 30, (campaign, arm, len(paths))
            data = {"muse-glimmer-30b": transcript_summary(paths)}
            aggregate.write_text(json.dumps(data, indent=2) + "\n")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                summary.print_ascii_table(data)
            aggregate.with_suffix(".txt").write_text(output.getvalue())


def bakeoff():
    name = "ops/baseten-gemma4-31b-sglang/run_bakeoff.py"
    m = module(name)
    folder = ROOT / "ops/baseten-gemma4-31b-sglang/bakeoff-20260806-normalized"
    runs = json.loads((folder / "attempts.json").read_text())["runs"]
    output = []
    for row in runs:
        path = row.get("run_dir")
        if path:
            path = Path(path)
            if path.is_absolute():
                path = (
                    Path("runs") / path.name
                    if path.parent.name == "runs"
                    else Path("runs/aiwf_medium_context") / path.name
                )
            result = m.summarize_run(ROOT / path)
            output.append({"arm": row["arm"], "run_dir": str(path), "summary": result})
    (OUT / "bakeoff-runs.json").write_text(json.dumps(output, indent=2) + "\n")
    # Pooled turn percentiles, not averages of per-conversation percentiles.
    labels = {
        "sglang_mtp": "SGLang NEXTN/MTP",
        "vllm_no_mtp": "vLLM APC, no MTP",
        "sglang_no_mtp": "SGLang APC, no MTP",
    }
    p = folder / "REPORT.md"
    text = p.read_text()
    for arm, label in labels.items():
        selected = [r for r in output if r["arm"] == arm]
        values = []
        for r in selected:
            for line in (
                (ROOT / r["run_dir"] / "transcript.jsonl").read_text().splitlines()
            ):
                row = json.loads(line)
                if (
                    isinstance(row.get("turn"), int)
                    and 0 < row["turn"] < 30
                    and row.get("ttfb_ms") is not None
                ):
                    values.append(float(row["ttfb_ms"]))
        ordered = sorted(values)
        index = (len(values) - 1) * 0.95
        low = int(index)
        p95 = ordered[low] + (ordered[min(low + 1, len(values) - 1)] - ordered[low]) * (
            index - low
        )
        line = f"| {label} | 3/3 | 90 | {statistics.median(values):.0f}ms | {p95:.0f}ms | {max(values):.0f}ms |"
        text = "\n".join(
            line if old.startswith("| " + label + " |") else old
            for old in text.split("\n")
        )
    text = text.replace(
        "scripted turns with valid runtimes and zero thinking tokens.",
        "scripted turns with valid runtimes and zero thinking tokens. Latency excludes\neach conversation's first scripted response.",
    )
    p.write_text(text)


def campaigns():
    """Replay only analysis entry points; order preserves dependent comparisons."""
    commands = json.loads((OUT / "commands.json").read_text())
    for command in commands:
        if command[0].endswith("run_bakeoff.py"):
            raise ValueError("Capture launcher is not an offline analysis command")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/offline_analysis.py"), *command],
            cwd=ROOT,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task", choices=["campaigns", "paired-gemma", "muse", "bakeoff"]
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {
        "campaigns": campaigns,
        "paired-gemma": paired_gemma,
        "muse": muse,
        "bakeoff": bakeoff,
    }[args.task]()


if __name__ == "__main__":
    main()
