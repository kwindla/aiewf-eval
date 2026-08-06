#!/usr/bin/env python3
"""General filler analysis. Usage: analyze_filler.py <manifest.tsv> <cfg1,cfg2,...>
Manifest lines: <config>\t<run_dir>. Excludes degraded (<20-turn) runs."""
import json, math, collections, sys
from pathlib import Path

MAN = sys.argv[1]
ORDER = sys.argv[2].split(",")

def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs: return None
    k = (len(xs) - 1) * p; f = math.floor(k)
    return xs[f] if f + 1 >= len(xs) else xs[f] + (xs[f + 1] - xs[f]) * (k - f)

def fm(v, s=""):
    return (f"{v:.0f}{s}" if v is not None else "n/a")

by = collections.defaultdict(list)
for line in Path(MAN).read_text().splitlines():
    if not line.strip(): continue
    cfg, d = line.split("\t"); by[cfg].append(d)

print(f"{'config':10} {'runs':7} | {'pass% mean':10} {'pass rng':10} | {'TTFAT P50':9} {'TTFAT P95':9} {'TTFAT max':9} | {'thinkTok P50':12}")
print("-" * 92)
base = None
for cfg in ORDER:
    prs, ttfb, think, used, skip = [], [], [], 0, 0
    for d in by.get(cfg, []):
        n = sum(1 for _ in (Path(d) / "transcript.jsonl").open())
        if n < 20: skip += 1; continue
        used += 1
        sp = Path(d) / "claude_summary.json"
        if sp.exists():
            pr = json.loads(sp.read_text()).get("turn_pass", {}).get("rate")
            if pr is not None: prs.append(pr)
        for line in (Path(d) / "transcript.jsonl").open():
            r = json.loads(line)
            if r.get("ttfb_ms") is not None: ttfb.append(r["ttfb_ms"])
            th = (r.get("tokens") or {}).get("thinking_tokens")
            if th is not None: think.append(th)
    prm = sum(prs) / len(prs) if prs else None
    p50 = pct(ttfb, .5)
    if base is None and prm is not None: base = (prm, p50)
    rng = f"{min(prs):.0f}-{max(prs):.0f}%" if prs else "n/a"
    runs = f"{used}" + (f" (-{skip})" if skip else "")
    delta = ""
    if base and prm is not None and cfg != ORDER[0]:
        delta = f"  ({prm-base[0]:+.1f}pt, {p50-base[1]:+.0f}ms)"
    print(f"{cfg:10} {runs:>7} | {(f'{prm:.1f}%' if prm is not None else 'n/a'):>10} {rng:>10} | "
          f"{fm(p50,'ms'):>9} {fm(pct(ttfb,.95),'ms'):>9} {fm(max(ttfb) if ttfb else None,'ms'):>9} | "
          f"{fm(pct(think,.5)):>12}{delta}")
