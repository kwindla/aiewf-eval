#!/usr/bin/env python3
"""Audit every text leaderboard latency cell; update only traceable v2 values.

Unresolved historical cohorts are explicitly retained, never selected by matching
scores or by scanning arbitrary runs. Raw-scripted cohorts use linear P95, as in
their frozen analyzers. Existing campaign aggregates retain their own estimators.
"""

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess

from scripts.reanalyze_historical_ttfat import module
from multi_turn_eval.latency_reporting import include_response_latency

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/ttfat-reanalysis-2026-10"
BASE = "ea9ed5a0f0c0dcbf14d0026ef3487c6a8f6ce5f7"


def load(path):
    return json.loads((ROOT / path).read_text())


def raw_stats(paths):
    values = []
    for name in paths:
        p = Path(name)
        if not p.is_absolute():
            p = ROOT / p
        for line in (p / "transcript.jsonl").read_text().splitlines():
            row = json.loads(line)
            t = row.get("turn")
            v = row.get("ttfb_ms")
            if (
                isinstance(t, int)
                and 0 <= t < 30
                and row.get("recovery_turn") is not True
                and include_response_latency(t)
                and isinstance(v, (int, float))
                and math.isfinite(v)
                and v >= 0
            ):
                values.append(v)
    a = sorted(values)
    i = (len(a) - 1) * 0.95
    j = int(i)
    return [
        statistics.median(a),
        a[j] + (a[min(j + 1, len(a) - 1)] - a[j]) * (i - j),
        max(a),
    ], len(a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sources = {}

    def add(label, values, source, scope, paths=None):
        sources[label] = {
            "values_ms": values,
            "source": source,
            "scope": scope,
            "run_paths": paths or [],
        }

    def raw(label, paths, source):
        normalized = [
            "runs/" + str(p).split("/runs/", 1)[-1] if "/runs/" in str(p) else str(p)
            for p in paths
        ]
        values, n = raw_stats(normalized)
        add(label, values, source, "v2 scripted only; linear P95", normalized)
        sources[label]["observations"] = n

    focused = "docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py"
    m = module(focused)
    cells = m.load_all()
    for key, arm, label in [
        ("gpt54", "nofiller", "gpt-5.4 (none)"),
        ("gpt54", "dots96", "gpt-5.4 (none, +96 dots)"),
        ("terra", "nofiller", "gpt-5.6-terra (none)"),
        ("sol", "nofiller", "gpt-5.6-sol (none)"),
        ("gpt55", "nofiller", "gpt-5.5 (none)"),
        ("inkling", "nofiller", "inkling (none)"),
        ("glm52", "nofiller", "glm-5.2 (none)"),
    ]:
        raw(label, [r.run_dir for r in cells[key, arm]], focused)
    for folder in [
        "gemini-minimal-dots-2026-07-21",
        "gemini25-thinking-off-dots-2026-07-22",
        "laguna-s21-openrouter-2026-07-22",
    ]:
        source = "docs/filler-study-data/" + folder + "/aggregates.json"
        for model in load(source)["models"].values():
            arm = model["arms"]["nofiller"]
            raw(model["readme_label"], arm["run_dirs"], source)
    source = "docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json"
    for model in load(source)["models"].values():
        arm = model["arms"]["nofiller"]
        label = model["display_name"].replace(
            " (thinking off, FP8)", " (thinking off, FP8)"
        )
        raw(label, arm["run_dirs"], source)
    for pattern, label in [
        (
            "ops/baseten-kimi-k2.6/aiewf-medium-none*/analysis/aggregates.json",
            "kimi-k2.6 (thinking off)",
        ),
        (
            "ops/baseten-kimi-k2.6/aiewf-medium-thinking*/analysis/aggregates.json",
            "kimi-k2.6 (thinking on)",
        ),
        (
            "ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150*/aggregates.json",
            "gemma-4-31b-it (thinking off)",
        ),
    ]:
        path = next(ROOT.glob(pattern))
        data = json.loads(path.read_text())
        candidates = []

        def search(x):
            if isinstance(x, dict):
                if "p50" in x and "p95" in x and "max" in x:
                    candidates.append([x["p50"], x["p95"], x["max"]])
                if "p50_ms" in x and "p95_ms" in x and "max_ms" in x:
                    candidates.append([x["p50_ms"], x["p95_ms"], x["max_ms"]])
                for k, v in x.items():
                    if "raw" not in k and "conversation" not in k and "cohort" not in k:
                        search(v)

        search(data)
        assert candidates, (label, data.keys())
        add(
            label,
            candidates[0],
            str(path.relative_to(ROOT)),
            "v2 scripted only; campaign estimator",
        )
    source = "runs/muse-glimmer-reasoning-strength-n30-20260811/aggregate-low.json"
    a = next(iter(load(source).values()))["ttfts"]
    add(
        "muse-glimmer-30b (thinking low, GGUF)",
        [statistics.median(a), sorted(a)[math.ceil(0.95 * len(a)) - 1], max(a)],
        source,
        "v2 includes recovery; nearest-rank P95",
    )
    for pattern, label in [
        ("ops/local-qwen38*/**/aggregates.json", "qwen3.8-27b (thinking off, NVFP4)"),
        ("ops/baseten-qwen38*/**/aggregates.json", "qwen3.8-27b (thinking off, FP8)"),
    ]:
        path = next(ROOT.glob(pattern))
        d = json.loads(path.read_text())["content_aware_ttfat_ms"]
        add(
            label,
            [d["p50_ms"], d["p95_ms"], d["max_ms"]],
            str(path.relative_to(ROOT)),
            "existing v2 campaign output; includes recovery",
        )
    source = "ops/local-nemotron35-lightning-sglang/aiewf-medium-binary-n30-20260811/artifacts/analysis/aggregates.json"
    for arm, d in load(source)["arms"].items():
        add(
            "nemotron-3.5-lightning (thinking "
            + ("on" if arm == "on-unbounded" else arm)
            + ", NVFP4)",
            [d["ttfat_p50_ms"], d["ttfat_p95_ms"], d["ttfat_max_ms"]],
            source,
            "existing v2 campaign output; includes recovery",
        )
    source = "ops/baseten-deepseek-v4/aiewf-medium-3arm-n30-20260817/analysis/aggregates.json"
    for d in load(source).values():
        add(
            d["label"],
            [d["ttfat_p50_ms"], d["ttfat_p95_ms"], d["ttfat_max_ms"]],
            source,
            "existing v2 campaign output; includes recovery",
        )
    source = (
        "ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/aggregates-full.json"
    )
    d = load(source)["arms"]["nofiller"]["ttfat_ms_observed_responses_only"]
    add(
        "gemma-4-26b-a4b-it (thinking off)",
        [d["p50"], d["p95"], d["max"]],
        source,
        "v2 fresh paired control; scripted only",
    )
    from scripts.benchmark_summary import aggregate_runs
    from scripts.reanalyze_historical_ttfat import transcript_summary

    for file, label in [
        ("claude-fable-5-low-2026-06-09.txt", "claude-fable-5 (low)"),
        ("claude-fable-5-default-2026-06-09.txt", "claude-fable-5 (default)"),
        ("claude-sonnet-5-disabled-2026-07-01.txt", "claude-sonnet-5"),
    ]:
        source = "docs/ten-run-allowlists/" + file
        paths = [
            line
            for line in (ROOT / source).read_text().splitlines()
            if line and not line.startswith("#")
        ]
        values = transcript_summary([ROOT / p for p in paths])["ttfts"]
        add(
            label,
            [
                statistics.median(values),
                sorted(values)[math.ceil(0.95 * len(values)) - 1],
                max(values),
            ],
            source,
            "v2 includes recovery; nearest-rank P95",
            paths,
        )
    source = "ops/zai-glm-5.3/aiewf-medium-low-n30-20260819/run-dirs.txt"
    paths = (ROOT / source).read_text().splitlines()
    values = transcript_summary([ROOT / p for p in paths])["ttfts"]
    add(
        "glm-5.3 (low reasoning)",
        [
            statistics.median(values),
            sorted(values)[math.ceil(0.95 * len(values)) - 1],
            max(values),
        ],
        source,
        "v2 includes recovery; nearest-rank P95",
        paths,
    )
    text = subprocess.check_output(
        ["git", "show", BASE + ":README.md"], cwd=ROOT, text=True
    )
    lines = text.splitlines()
    report = []
    table_start = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("| Model | Pass Rate |") and "TTFAT P50" in line
    )
    table_end = next(
        i for i in range(table_start + 1, len(lines)) if not lines[i].strip()
    )
    for i, line in enumerate(lines):
        if not table_start < i < table_end or not line.startswith("| "):
            continue
        fields = [x.strip() for x in line.strip("|").split("|")]
        if len(fields) != 10 or not all("ms" in x for x in fields[6:9]):
            continue
        label = fields[0].replace("**", "")
        before = [int(x.replace("**", "").removesuffix("ms")) for x in fields[6:9]]
        entry = {"model": label, "before_ms": before}
        if label in sources:
            source = sources[label]
            after = [round(x) for x in source["values_ms"]]
            entry.update(source)
            entry["after_ms"] = after
            entry["status"] = "verified" if before == after else "updated"
            for j, v in enumerate(after, 6):
                fields[j] = (
                    ("**" if fields[j].startswith("**") else "")
                    + str(v)
                    + "ms"
                    + ("**" if fields[j].endswith("**") else "")
                )
            lines[i] = "| " + " | ".join(fields) + " |"
        else:
            entry["after_ms"] = before
            entry["status"] = (
                "legacy exception"
                if label.startswith(
                    ("nemotron-3-ultra", "nemotron-3-super", "nemotron-3-nano")
                )
                else "unresolved cohort provenance"
            )
            entry["source"] = None
        report.append(entry)
    OUT.mkdir(exist_ok=True)
    (OUT / "readme-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.write:
        note = "The [October latency audit](docs/ttfat-reanalysis-2026-10/REPORT.md) verifies 30 rows against v2 cohorts or outputs. Thirteen older rows retain published latency values pending exact cohort provenance; they are not yet certified as first-response-excluded. The four legacy Nemotron exceptions remain as documented."
        rendered = "\n".join(lines) + "\n"
        rendered = rendered.replace(
            "\n\nThis table emphasizes", "\n\n" + note + "\n\nThis table emphasizes", 1
        )
        (ROOT / "README.md").write_text(rendered)
        standalone = ROOT / "leaderboard-medium-context.md"
        old = subprocess.check_output(
            ["git", "show", BASE + ":leaderboard-medium-context.md"],
            cwd=ROOT,
            text=True,
        )
        start = old.index("| Model | Pass Rate |")
        end = old.index("\n\n", start)
        table = "\n".join(lines[table_start:table_end]).replace("**", "")
        standalone.write_text(old[:start] + table + "\n\n" + note + old[end:])
    print(
        {
            status: sum(x["status"] == status for x in report)
            for status in sorted({x["status"] for x in report})
        }
    )


if __name__ == "__main__":
    main()
