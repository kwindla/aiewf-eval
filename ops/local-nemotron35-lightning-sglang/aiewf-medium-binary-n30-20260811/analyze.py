#!/usr/bin/env python3
"""Produce fixed-denominator per-arm aggregates from the frozen canonical set."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = Path(__file__).resolve().parent
CAMPAIGN = PROTOCOL / "artifacts"
CANONICAL = CAMPAIGN / "canonical.tsv"
ARMS = ("off", "on-unbounded")
CONVERSATIONS_PER_ARM = 30
TURNS_PER_CONVERSATION = 30
DENOMINATOR = CONVERSATIONS_PER_ARM * TURNS_PER_CONVERSATION
SCORE_KEYS = (
    "tool_use_correct",
    "instruction_following",
    "kb_grounding",
    "turn_taking",
)


def fail(message: str) -> None:
    raise SystemExit(message)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_rate_ci(conversation_passes: list[int]) -> list[float]:
    rng = random.Random(350811)
    estimates = []
    for _ in range(10_000):
        sampled = [rng.choice(conversation_passes) for _ in conversation_passes]
        estimates.append(100 * sum(sampled) / DENOMINATOR)
    estimates.sort()
    return [estimates[249], estimates[9749]]


def load_conversation(entry: dict[str, str]) -> dict:
    run_dir = ROOT / entry["run_dir"]
    transcript_path = run_dir / "transcript.jsonl"
    judged_path = run_dir / "claude_judged.jsonl"
    summary_path = run_dir / "claude_summary.json"
    for path in (transcript_path, judged_path, summary_path, run_dir / "run.log"):
        if not path.is_file():
            fail(f"required artifact is missing: {path}")

    transcript = read_jsonl(transcript_path)
    scripted = [
        row
        for row in transcript
        if not row.get("recovery_turn") and 0 <= int(row["turn"]) < 30
    ]
    turns = [int(row["turn"]) for row in scripted]
    if turns != list(range(len(scripted))):
        fail(f"non-contiguous scripted turns in {run_dir}: {turns}")
    if len(scripted) != int(entry["turns"]):
        fail(f"canonical/transcript turn mismatch in {run_dir}")

    judged = [
        row
        for row in read_jsonl(judged_path)
        if not row.get("recovery_turn") and 0 <= int(row["turn"]) < 30
    ]
    if [int(row["turn"]) for row in judged] != turns:
        fail(f"judged/scripted turn mismatch in {run_dir}")

    correct = dict.fromkeys(SCORE_KEYS, 0)
    strict_passes = 0
    for row in judged:
        scores = row.get("scores")
        if not isinstance(scores, dict):
            fail(f"missing score object in {judged_path}")
        for key in SCORE_KEYS:
            value = scores.get(key)
            if not isinstance(value, bool):
                fail(f"missing boolean {key} in {judged_path}")
            correct[key] += int(value)
        strict_passes += int(
            scores["tool_use_correct"]
            and scores["instruction_following"]
            and scores["kb_grounding"]
        )

    # Leaderboard latency excludes the first scripted response of every
    # conversation. That response pays the intentionally cold system-prompt
    # prefill and is not part of the conversational-latency estimand. Recovery
    # responses remain included because they consume real voice-agent time.
    latency_rows = [
        row
        for row in transcript
        if row.get("recovery_turn") or int(row["turn"]) != 0
    ]
    ttfat = [
        int(row["ttfb_ms"])
        for row in latency_rows
        if row.get("ttfb_ms") is not None
    ]
    raw_ttft = [
        int(row["raw_ttfb_ms"])
        for row in latency_rows
        if row.get("raw_ttfb_ms") is not None
    ]
    end_session_turns = [
        int(row["turn"])
        for row in scripted
        if any(call.get("name") == "end_session" for call in row.get("tool_calls", []))
    ]
    return {
        "slot": int(entry["slot"]),
        "arm": entry["arm"],
        "run_dir": entry["run_dir"],
        "classification": entry["classification"],
        "observed_turns": len(scripted),
        "missing_turns": TURNS_PER_CONVERSATION - len(scripted),
        "correct": correct,
        "strict_passes": strict_passes,
        "ttfat_ms": ttfat,
        "raw_ttft_ms": raw_ttft,
        "end_session_turns": end_session_turns,
        "transcript_sha256": sha256(transcript_path),
        "judged_sha256": sha256(judged_path),
        "summary_sha256": sha256(summary_path),
    }


def aggregate(arm: str, conversations: list[dict]) -> dict:
    if len(conversations) != CONVERSATIONS_PER_ARM:
        fail(f"arm {arm} has {len(conversations)} conversations, expected 30")
    observed = sum(row["observed_turns"] for row in conversations)
    missing = DENOMINATOR - observed
    correct = {
        key: sum(row["correct"][key] for row in conversations)
        for key in SCORE_KEYS
    }
    strict = sum(row["strict_passes"] for row in conversations)
    ttfat = [value for row in conversations for value in row["ttfat_ms"]]
    raw_ttft = [value for row in conversations for value in row["raw_ttft_ms"]]
    conversation_passes = [row["strict_passes"] for row in conversations]
    return {
        "arm": arm,
        "conversations": len(conversations),
        "fixed_turn_denominator": DENOMINATOR,
        "observed_scripted_turns": observed,
        "missing_future_turns_scored_as_failures": missing,
        "complete_30_conversations": sum(
            row["classification"] == "complete_30" for row in conversations
        ),
        "fixed_denominator_short_conversations": sum(
            row["classification"] != "complete_30" for row in conversations
        ),
        "strict_pass_count": strict,
        "strict_pass_rate_pct": 100 * strict / DENOMINATOR,
        "strict_pass_cluster_bootstrap_ci95_pct": bootstrap_rate_ci(
            conversation_passes
        ),
        "any_error_rate_pct": 100 * (DENOMINATOR - strict) / DENOMINATOR,
        "tool_error_rate_pct": 100 * (DENOMINATOR - correct["tool_use_correct"]) / DENOMINATOR,
        "instruction_error_rate_pct": 100 * (DENOMINATOR - correct["instruction_following"]) / DENOMINATOR,
        "kb_error_rate_pct": 100 * (DENOMINATOR - correct["kb_grounding"]) / DENOMINATOR,
        "turn_taking_error_rate_pct": 100 * (DENOMINATOR - correct["turn_taking"]) / DENOMINATOR,
        "ttfat_observations": len(ttfat),
        "ttfat_p50_ms": statistics.median(ttfat) if ttfat else None,
        "ttfat_p95_ms": percentile(ttfat, 0.95),
        "ttfat_max_ms": max(ttfat) if ttfat else None,
        "raw_ttft_observations": len(raw_ttft),
        "raw_ttft_p50_ms": statistics.median(raw_ttft) if raw_ttft else None,
        "raw_ttft_p95_ms": percentile(raw_ttft, 0.95),
        "raw_ttft_max_ms": max(raw_ttft) if raw_ttft else None,
    }


def fmt(value: float) -> str:
    return f"{value:.1f}"


def main() -> int:
    canonical = read_tsv(CANONICAL)
    if len(canonical) != 60:
        fail(f"canonical set has {len(canonical)} rows, expected 60")
    conversations = [load_conversation(entry) for entry in canonical]
    arms = {
        arm: aggregate(arm, [row for row in conversations if row["arm"] == arm])
        for arm in ARMS
    }
    payload = {
        "schema_version": 1,
        "model": "nemotron-3.5-lightning",
        "provider": "Local RTX 5090",
        "method": {
            "conversations_per_arm": CONVERSATIONS_PER_ARM,
            "scripted_turns_per_conversation": TURNS_PER_CONVERSATION,
            "fixed_denominator": True,
            "missing_future_turns_fail_all_displayed_accuracy_criteria": True,
            "latency_conditional_on_observed_responses": True,
            "latency_excludes_first_scripted_turn_per_conversation": True,
            "latency_includes_recovery_responses": True,
            "judge_model": "claude-opus-4-5",
            "canonical_sha256": sha256(CANONICAL),
        },
        "arms": arms,
        "conversations": conversations,
    }
    analysis_dir = CAMPAIGN / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "aggregates.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Nemotron 3.5 Lightning AIEWF medium-context results",
        "",
        "Each arm contains 30 canonical conversations and 900 fixed-denominator "
        "scripted turns. Missing future turns after a model-caused early exit fail "
        "all displayed accuracy criteria. Latency excludes the first scripted "
        "response of each conversation and is summarized over the remaining "
        "observed scripted and recovery responses.",
        "",
        "| Mode | Pass Rate | Any Error | Tool Error | Instruction Error | KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max | Raw TTFT P50 | Full 30-turn conversations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in sorted(ARMS, key=lambda name: -arms[name]["strict_pass_rate_pct"]):
        row = arms[arm]
        lines.append(
            f"| {arm} | {fmt(row['strict_pass_rate_pct'])}% | "
            f"{fmt(row['any_error_rate_pct'])}% | {fmt(row['tool_error_rate_pct'])}% | "
            f"{fmt(row['instruction_error_rate_pct'])}% | {fmt(row['kb_error_rate_pct'])}% | "
            f"{row['ttfat_p50_ms']:.0f}ms | {row['ttfat_p95_ms']:.0f}ms | "
            f"{row['ttfat_max_ms']:.0f}ms | {row['raw_ttft_p50_ms']:.0f}ms | "
            f"{row['complete_30_conversations']}/30 |"
        )
    lines.extend(
        [
            "",
            "The two request modes use NVIDIA's recommended temperature 1.0 and "
            "top-p 0.95, no output-token cap, and no thinking-budget cap. Both "
            "send `force_nonempty_content=true`; only `enable_thinking` changes.",
            "",
        ]
    )
    (analysis_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(arms, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
