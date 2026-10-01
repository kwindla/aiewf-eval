"""Regenerate latency prose/tables from computed values, retaining other wording."""

import difflib
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "ea9ed5a0f0c0dcbf14d0026ef3487c6a8f6ce5f7"


def original(path):
    return subprocess.check_output(
        ["git", "show", BASE + ":" + path], cwd=ROOT, text=True
    )


def load(path):
    return json.loads((ROOT / path).read_text())


def linear(a, q):
    a = sorted(a)
    i = (len(a) - 1) * q
    j = int(i)
    return a[j] + (a[min(j + 1, len(a) - 1)] - a[j]) * (i - j)


def main():
    changes = []

    def write(path, text):
        before = original(path)
        if before != text:
            changes.append({"path": path, "before": before, "after": text})
        (ROOT / path).write_text(text)

    pool = "ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/"
    d = load(pool + "aggregates.json")["latency"]
    p50 = f"{d['p50_ms']:.0f}"
    p95 = f"{d['p95_ms']:.0f}"
    path = pool + "README.md"
    text = (
        original(path)
        .replace("490ms | 718ms", p50 + "ms | " + p95 + "ms")
        .replace(
            "All 150 conversations completed",
            "Latency excludes each conversation’s first scripted response.\n\nAll 150 conversations completed",
        )
    )
    write(path, text)
    path = "ops/baseten-gemma4-31b-sglang/README.md"
    text = original(path).replace(
        "Pooled TTFAT was 490ms\nP50 and 718ms P95.",
        f"Excluding each conversation’s first response, pooled TTFAT was {p50}ms\nP50 and {p95}ms P95.",
    )
    write(path, text)
    path = "ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/README.md"
    data = load(str(Path(path).with_name("aggregates.json")))
    text = original(path).replace("490/718ms", p50 + "/" + p95 + "ms")
    for arm, old in [("fp8_kv", "105/309ms"), ("bf16_kv", "128/336ms")]:
        d = data["arms"][arm]["pooled"]["latency"]
        text = text.replace(old, f"{d['p50_ms']:.0f}/{d['p95_ms']:.0f}ms")
    text = text.replace(
        "Local BF16 minus FP8 KV",
        "Latency excludes each conversation’s first scripted response.\n\nLocal BF16 minus FP8 KV",
    )
    write(path, text)
    folder = "runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z/"
    rows = [
        json.loads(line)
        for run in (ROOT / folder / "included-runs.txt").read_text().splitlines()
        for line in (ROOT / run / "transcript.jsonl").read_text().splitlines()
    ]
    text = original(folder + "REPORT.md")
    stats = {}
    for label, key in [
        ("TTFT", "raw_ttfb_ms"),
        ("TTFAT", "ttfb_ms"),
        ("Full response latency", "latency_ms"),
    ]:
        a = [
            r[key]
            for r in rows
            if isinstance(r.get("turn"), int)
            and 0 < r["turn"] < 30
            and not r.get("recovery_turn")
            and r.get(key) is not None
        ]
        values = [statistics.median(a), linear(a, 0.95), max(a)]
        stats[label] = values
        new = "| " + label + " | " + " | ".join(f"{v:,.0f} ms" for v in values) + " |"
        text = "\n".join(
            new if line.startswith("| " + label + " |") else line
            for line in text.split("\n")
        )
    text = text.replace(
        "the 900 scripted turns and excludes recovery responses.",
        "scripted turns 1–29 and excludes recovery responses and each conversation’s\nfirst scripted response.",
    )
    text = text.replace(
        "while TTFAT P95 increased from 5,791 ms to 6,488 ms",
        f"while v2 TTFAT P95 is {stats['TTFAT'][1]:,.0f} ms (the earlier 5,791 ms included first responses)",
    )
    write(folder + "REPORT.md", text)
    # The quoted source-first-turn diagnostics above deliberately retain their scope.
    path = "README.md"
    text = (ROOT / path).read_text()
    old = text
    audit = load("docs/ttfat-reanalysis-2026-10/readme-audit.json")
    lookup = {r["model"]: r for r in audit}
    d = lookup["deepseek-v4-flash-0731 (high)"]
    text = text.replace(
        "Flash high scored 845/900 (93.9%) at 763ms",
        f"Flash high scored 845/900 (93.9%) at {d['after_ms'][0]}ms",
    )
    d = lookup["nemotron-3.5-lightning (thinking on, NVFP4)"]
    text = text.replace(
        "while TTFAT was 1464ms / 5787ms",
        f"while TTFAT was {d['after_ms'][0]}ms / {d['after_ms'][1]}ms",
    )
    if text != old:
        changes.append({"path": path, "before": old, "after": text})
    (ROOT / path).write_text(text)
    (ROOT / "docs/ttfat-reanalysis-2026-10/prose-changes.json").write_text(
        json.dumps(
            [
                {
                    "path": c["path"],
                    "diff": "".join(
                        difflib.unified_diff(
                            c["before"].splitlines(True),
                            c["after"].splitlines(True),
                            fromfile="before/" + c["path"],
                            tofile="after/" + c["path"],
                        )
                    ),
                }
                for c in changes
            ],
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
