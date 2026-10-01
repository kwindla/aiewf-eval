"""Build the before/after ledger and source-reproduction audit from Git and outputs."""

import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/ttfat-reanalysis-2026-10"
BASE = "ea9ed5a0f0c0dcbf14d0026ef3487c6a8f6ce5f7"
SOURCE = "9fcb3a965ba7134b02f08bef668db7d33833f266"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            r
            for k in sorted(set(a) | set(b))
            for r in walk(a.get(k), b.get(k), path + "/" + k)
        ]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) == len(b) and all(isinstance(x, dict) for x in a + b):
            return [
                r
                for i, (x, y) in enumerate(zip(a, b))
                for r in walk(x, y, path + "/" + str(i))
            ]
        return [] if a == b else [(path, a, b)]
    return [] if a == b else [(path, a, b)]


def analyzer(f):
    if f.startswith("runs/muse-glimmer-reasoning"):
        return (
            "ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py"
            if Path(f).name in ("REPORT.md", "analysis.json")
            else "scripts/benchmark_summary.py"
        )
    if f.startswith("runs/muse-glimmer-card"):
        return "scripts/reanalyze_historical_ttfat.py muse / scripts/render_ttfat_reanalysis_prose.py"
    if "bakeoff-" in f:
        return "ops/baseten-gemma4-31b-sglang/run_bakeoff.py:summarize_run"
    if "dots-20260731" in f:
        return "ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py"
    if Path(f).name in ("COMPARISON.md", "comparison.json"):
        return str(Path(f).with_name("compare_arms.py"))
    p = Path(f).parent
    while p != Path("."):
        if (ROOT / p / "analyze.py").exists():
            return str(p / "analyze.py")
        p = p.parent
    return "scripts/render_ttfat_reanalysis_prose.py"


def main():
    files = (
        git("diff-tree", "--no-commit-id", "--name-only", "-r", SOURCE)
        .decode()
        .splitlines()
    )
    files = [f for f in files if not f.endswith(".py")]
    artifacts = []
    numbers = []
    for f in files:
        old = git("show", BASE + ":" + f)
        expected = git("show", SOURCE + ":" + f)
        new = (ROOT / f).read_bytes()
        item = {
            "path": f,
            "analyzer": analyzer(f),
            "before_sha256": sha(old),
            "source_sha256": sha(expected),
            "after_sha256": sha(new),
            "byte_identical_to_source": new == expected,
        }
        if f.endswith(".json"):
            a = json.loads(old)
            b = json.loads(new)
            e = json.loads(expected)
            delta = walk(a, b)
            differences = walk(e, b)
            item["source_differences"] = [p for p, x, y in differences]
            item["input_run_paths"] = sorted(
                set(re.findall(r"runs/[A-Za-z0-9_./+-]+", new.decode()))
            )
            for p, x, y in delta:
                if (
                    isinstance(x, (int, float))
                    and not isinstance(x, bool)
                    and isinstance(y, (int, float))
                    and not isinstance(y, bool)
                ):
                    numbers.append(
                        {
                            "artifact": f,
                            "metric": p,
                            "before": x,
                            "after": y,
                            "analyzer": analyzer(f),
                        }
                    )
        elif new != expected:
            item["source_diff"] = "".join(
                difflib.unified_diff(
                    expected.decode().splitlines(True),
                    new.decode().splitlines(True),
                    fromfile="9fcb3a9/" + f,
                    tofile="regenerated/" + f,
                )
            )
        artifacts.append(item)
    # Additional hand-assembled report tables: every changed numeric cell.
    for f in files:
        if not f.endswith(".md"):
            continue
        old = git("show", BASE + ":" + f).decode().splitlines()
        new = (ROOT / f).read_text().splitlines()
        rows = {
            line.split("|")[1].strip(): line
            for line in new
            if line.startswith("|") and len(line.split("|")) > 2
        }
        for line in old:
            if not line.startswith("|"):
                continue
            key = line.split("|")[1].strip()
            if key not in rows or line == rows[key]:
                continue
            left = line.split("|")
            right = rows[key].split("|")
            if len(left) != len(right):
                continue
            for i, (x, y) in enumerate(zip(left, right)):
                xs = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", x)
                ys = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", y)
                if len(xs) == len(ys):
                    for j, (a, b) in enumerate(zip(xs, ys)):
                        if a != b:
                            numbers.append(
                                {
                                    "artifact": f,
                                    "metric": key + f" / cell {i}, number {j + 1}",
                                    "before": a,
                                    "after": b,
                                    "analyzer": analyzer(f),
                                }
                            )
    audit = json.loads((OUT / "readme-audit.json").read_text())
    for r in audit:
        for i, metric in enumerate(["P50", "P95", "max"]):
            if r["before_ms"][i] != r["after_ms"][i]:
                numbers.append(
                    {
                        "artifact": "README.md",
                        "metric": r["model"] + " / TTFAT " + metric,
                        "before": r["before_ms"][i],
                        "after": r["after_ms"][i],
                        "analyzer": "scripts/audit_readme_ttfat.py; source: "
                        + r["source"],
                    }
                )
    manifest = {
        "baseline": BASE,
        "source": SOURCE,
        "metric_definition": "ttfat-first-response-excluded-v2",
        "artifacts": artifacts,
        "changed_numbers": numbers,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Historical TTFAT reanalysis — October 2026",
        "",
        "This reanalysis excludes each conversation’s first scripted response. Accuracy, cohort membership, recovery eligibility, and percentile formulas retain each campaign’s definitions. Baseline: `ea9ed5a`; reproduction target: `9fcb3a9`. No captures or judging were rerun.",
        "",
        "**Coverage limitation:** all 47 README text rows (141 latency cells) were audited. Thirty rows have traceable v2 values; four retain the existing documented Nemotron exceptions; thirteen older rows lack an established exact published cohort in this audit and are unchanged. These thirteen are not certified as v2. See the row ledger below.",
        "",
        "## Reproduction",
        "",
        "The 14 campaign Python changes in the source comprise 12 analysis files and two campaign test files. The capture-capable bakeoff module is imported only for its offline summarizer. All twelve analysis paths were exercised, including dependent comparison/report generation.",
        "",
        "Numeric JSON aggregates reproduce the source. Non-byte-identical JSON differences are timestamps, executing-source and derived artifact hashes, and equivalent paired-Gemma method metadata. Source Kimi COMPLETE.json report hashes were stale; regeneration repairs them. Text differences are rendering whitespace and explicit v2 scope wording; exact per-file diffs are in [manifest.json](manifest.json).",
        "",
        "The paired-Gemma direct CLI refuses the changed source hash. Its offline reanalysis path verifies every original source SHA against historical Git blobs, retains cohort/configuration/transcript/judgment checks, and records the current analyzer separately in [paired-gemma-provenance.json](paired-gemma-provenance.json). Frozen source manifests are unchanged.",
        "",
        "The shared helper `multi_turn_eval.latency_reporting.include_response_latency` implements only turn-zero selection. Scripted-only campaigns keep recovery excluded; Muse and the generic summaries retain recovery. The Qwen/Kimi/pooled-Gemma validation and percentile methods remain explicit.",
        "",
        "| Output | Matches 9fcb3a9 bytes | Analyzer |",
        "|---|---|---|",
    ]
    for a in artifacts:
        lines.append(
            f"| `{a['path']}` | {'yes' if a['byte_identical_to_source'] else 'no; see manifest diff'} | `{a['analyzer']}` |"
        )
    lines += [
        "",
        "## README cell audit",
        "",
        "Each triplet is P50 / P95 / max in milliseconds. Run IDs and source paths are recorded in [readme-audit.json](readme-audit.json). Existing v2 outputs retain their campaign estimator; raw cohort audits use the frozen scripted-only linear-P95 convention.",
        "",
        "| Model | Before | After | Status | Source |",
        "|---|---|---|---|---|",
    ]
    for r in audit:
        lines.append(
            "| "
            + r["model"]
            + " | "
            + " / ".join(map(str, r["before_ms"]))
            + " | "
            + " / ".join(map(str, r["after_ms"]))
            + " | "
            + r["status"]
            + " | "
            + (
                "`" + r["source"] + "`"
                if r.get("source")
                else "exact cohort not established / historical exception"
            )
            + " |"
        )
    lines += [
        "",
        "## Changed published numbers",
        "",
        "One row per changed JSON scalar or rendered table number; repeated publications are listed separately. The artifact and metric path identify the campaign and cell. The artifact’s run-path list in manifest.json, or its adjacent canonical manifest, identifies the input cohort. Raw sample arrays are evidence, not table statistics, and are not expanded into one row per sample.",
        "",
        "| Campaign artifact | Metric | Before (v1) | After (v2) | Analyzer |",
        "|---|---|---:|---:|---|",
    ]
    for r in numbers:
        lines.append(
            f"| `{r['artifact']}` | `{r['metric']}` | {r['before']} | {r['after']} | `{r['analyzer']}` |"
        )
    lines += [
        "",
        "## Prose and filler study",
        "",
        "All generated prose edits are recorded as diffs in [prose-changes.json](prose-changes.json). The main README prose changes Flash high P50 763 → 762 ms and Nemotron thinking-on P95 5787 → 5786 ms to match their existing v2 aggregates. Muse’s 231/1752 and 236/5850 ms quotations already match v2 and are unchanged. Other model-specific historical observations retain their stated cohorts.",
        "",
        "The Qwen filler-study rows now regenerate as 667 and 764 ms, matching the README. The historical aggregate and Markdown/HTML generator are rerun; no numeric cell is hand-edited. Other filler-study latency figures retain their original analysis scope; the main README’s newly corrected values are separately audited here.",
        "",
        "## Reproduce",
        "",
        "Use an isolated worktree based on ttfat-first-response, its project environment, and the raw runs plus campaign state manifests. Input directories must be real worktree directories with read-only file links/copies if an analyzer enforces repository containment; do not symlink the entire runs directory. Inputs required beyond tracked files include the Qwen-dots and focused-dot `state/` trees and the paired-Gemma campaign.log. The original raw runs remain unchanged.",
        "",
        "```bash",
        "uv run python -m scripts.reanalyze_historical_ttfat campaigns",
        "uv run python -m scripts.reanalyze_historical_ttfat paired-gemma",
        "uv run python -m scripts.reanalyze_historical_ttfat muse",
        "uv run python -m scripts.reanalyze_historical_ttfat bakeoff",
        "uv run python -m scripts.audit_readme_ttfat --write",
        "uv run python -m scripts.render_ttfat_reanalysis_prose",
        "uv run python scripts/build_filler_report.py",
        "uv run python -m scripts.build_ttfat_reanalysis_report",
        "```",
        "",
        "Ordinary campaign subprocesses run through offline_analysis.py, which disallows network/subprocess calls and writes outside the worktree. Supplementary commands invoke only analysis functions and local Git reads. Timestamped metadata may change on replay; all result numbers must remain stable.",
        "",
        "## Remaining provenance gap",
        "",
        "The thirteen unresolved README rows are explicitly listed above. Retaining their published values does not certify first-response exclusion. Recover the exact original inclusion manifests before changing those rows; do not substitute a later filler-study control, an arbitrary model glob, or a cohort chosen to reproduce a score. The four Nemotron legacy notes remain unchanged.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        len(artifacts),
        "source output artifacts;",
        len(numbers),
        "changed number entries",
    )


if __name__ == "__main__":
    main()
