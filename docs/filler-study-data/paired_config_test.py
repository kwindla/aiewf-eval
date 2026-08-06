#!/usr/bin/env python3
"""Turn-stratified permutation test between two benchmark configs.

The benchmark script is fixed, so turn index t is the same question in every
run. For each turn stratum we pool config A's and config B's per-run pass
outcomes, permute the config labels within the stratum (respecting counts),
and recompute the overall pass-rate delta; 10k permutations give an exact-ish
two-sided p-value that is valid with unequal run counts.

Usage: paired_config_test.py <manifest.tsv> <cfgA> <cfgB> [--turns]
  --turns  also print the per-turn discordance table (drives the flip dossier)
"""
import json, sys, random
from pathlib import Path
from collections import defaultdict

random.seed(20260720)
N_PERM = 10000

def turn_pass(row):
    s = row.get("scores") or {}
    vals = [v for v in s.values() if isinstance(v, bool)]
    return all(vals) if vals else None

def load_config(man, cfg):
    """turn -> list of pass bools (one per run); last row wins per (run, turn)."""
    runs = [l.split("\t")[1].strip() for l in open(man) if l.split("\t")[0] == cfg]
    per_turn = defaultdict(list)
    summary_rates = []
    for d in runs:
        jp = Path(d) / "claude_judged.jsonl"
        if not jp.exists():
            continue
        by_turn = {}
        for line in jp.open():
            r = json.loads(line)
            p = turn_pass(r)
            if p is not None and isinstance(r.get("turn"), int):
                by_turn[r["turn"]] = p  # retries: last judgment wins
        for t, p in by_turn.items():
            per_turn[t].append(p)
        sp = Path(d) / "claude_summary.json"
        if sp.exists():
            rate = json.load(sp.open()).get("turn_pass", {}).get("rate")
            if rate is not None:
                mine = 100.0 * sum(by_turn.values()) / len(by_turn) if by_turn else None
                summary_rates.append((rate, mine))
    return per_turn, len(runs), summary_rates

def overall(per_turn):
    obs = [p for t in per_turn for p in per_turn[t]]
    return 100.0 * sum(obs) / len(obs) if obs else float("nan")

def main():
    man, cfgA, cfgB = sys.argv[1], sys.argv[2], sys.argv[3]
    show_turns = "--turns" in sys.argv
    A, nA, valA = load_config(man, cfgA)
    B, nB, valB = load_config(man, cfgB)
    # validation: computed vs claude_summary rates (should agree within rounding)
    for name, val in ((cfgA, valA), (cfgB, valB)):
        bad = [(s, m) for s, m in val if m is None or abs(s - m) > 1.0]
        if bad:
            print(f"WARN {name}: {len(bad)} runs where computed rate deviates "
                  f">1pt from claude_summary (first: {bad[0]})")
    turns = sorted(set(A) & set(B))
    if not turns:
        print("no common turns"); return
    rA, rB = overall({t: A[t] for t in turns}), overall({t: B[t] for t in turns})
    obs_delta = rA - rB

    def delta_of(assignA, assignB):
        a = [p for t in turns for p in assignA[t]]
        b = [p for t in turns for p in assignB[t]]
        return 100.0 * (sum(a) / len(a) - sum(b) / len(b))

    count = 0
    for _ in range(N_PERM):
        pa, pb = {}, {}
        for t in turns:
            pool = A[t] + B[t]
            random.shuffle(pool)
            pa[t], pb[t] = pool[: len(A[t])], pool[len(A[t]):]
        if abs(delta_of(pa, pb)) >= abs(obs_delta) - 1e-9:
            count += 1
    p = (count + 1) / (N_PERM + 1)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    print(f"{cfgA} (n={nA}, {rA:.1f}%) vs {cfgB} (n={nB}, {rB:.1f}%): "
          f"delta={obs_delta:+.1f}pt  p={p:.4f} {sig}  [{len(turns)} turns]")
    if show_turns:
        rows = []
        for t in turns:
            fa = 100.0 * sum(A[t]) / len(A[t])
            fb = 100.0 * sum(B[t]) / len(B[t])
            if abs(fa - fb) > 1e-9:
                rows.append((abs(fa - fb), t, fa, fb))
        rows.sort(reverse=True)
        print("  discordant turns (passrate A vs B):")
        for _, t, fa, fb in rows:
            print(f"    turn {t:2d}: {fa:5.1f}% vs {fb:5.1f}%  ({fa-fb:+.1f})")

if __name__ == "__main__":
    main()
