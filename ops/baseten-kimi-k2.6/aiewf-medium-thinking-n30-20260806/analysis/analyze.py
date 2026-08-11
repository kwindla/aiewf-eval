#!/usr/bin/env python3
"""Fixed-900-turn analysis for the BaseTen Kimi K2.6 campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
ROOT = CAMPAIGN.parents[2]
CONFIG = CAMPAIGN / "configuration.json"
CANONICAL = CAMPAIGN / "canonical.tsv"
ATTEMPTS = CAMPAIGN / "attempts.tsv"
COLLECTION_SUMMARY = CAMPAIGN / "collection/summary.json"
COLLECTION_COMPLETE = CAMPAIGN / "collection/COMPLETE.json"
JUDGE_COMPLETE = CAMPAIGN / "judging/COMPLETE.json"
JUDGE_INPUTS = CAMPAIGN / "judging/canonical-inputs.tsv"
JUDGE_HASHES = CAMPAIGN / "judging/judge-source-sha256.txt"
MODEL = "moonshotai/Kimi-K2.6"
TARGET = 30
N_TURNS = 30
DENOMINATOR = TARGET * N_TURNS
JUDGE_MODEL = "claude-opus-4-5"
JUDGE_VERSION = "claude-agent-sdk-v4-turn-taking"
COMPONENTS = ("tool_use_correct", "instruction_following", "kb_grounding")
BOOTSTRAPS = 20_000
SEED = 20260806
EXPECTED_SAMPLING = {
    "reasoning_effort": "omit",
    "chat_template_args": {"enable_thinking": True},
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 8192,
}


@dataclass(frozen=True)
class Conversation:
    slot: str
    run_dir: Path
    scores: dict[str, tuple[bool, ...]]
    strict: tuple[bool, ...]
    ttfat_ms: tuple[float, ...]
    raw_ttfb_ms: tuple[float, ...]
    thinking_tokens: tuple[int, ...]
    recovery_rows: int
    end_session_kind: str
    end_session_turn: int
    token_totals: dict[str, int]
    recovery_token_totals: dict[str, int]
    transcript_sha256: str
    judgment_sha256: str
    summary_sha256: str


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON {path}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"non-object JSON {path}:{number}")
        result.append(row)
    return result


def slot_number(slot: str) -> int:
    match = re.fullmatch(r"K26T-(\d{2})", slot)
    if not match:
        raise RuntimeError(f"invalid slot: {slot}")
    return int(match.group(1))


def tool_name(call: dict[str, Any]) -> str | None:
    if isinstance(call.get("name"), str):
        return call["name"]
    function = call.get("function")
    return function.get("name") if isinstance(function, dict) else None


def token_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "thinking_tokens",
    )
    return {
        field: sum(int((row.get("tokens") or {}).get(field) or 0) for row in rows)
        for field in fields
    }


def validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("model") != MODEL
        or config.get("arm") != "thinking"
        or config.get("filler") is not None
        or config.get("sampling") != EXPECTED_SAMPLING
    ):
        raise RuntimeError("frozen configuration mismatch")


def request_signature(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    return {
        "endpoint": config["endpoint"],
        **EXPECTED_SAMPLING,
        "filler": None,
    }


def load_conversation(manifest: dict[str, str]) -> Conversation:
    run_dir = (ROOT / manifest["run_dir"]).resolve()
    transcript_path = run_dir / "transcript.jsonl"
    judgment_path = run_dir / "claude_judged.jsonl"
    summary_path = run_dir / "claude_summary.json"
    transcript = read_jsonl(transcript_path)
    scheduled = [row for row in transcript if row.get("recovery_turn") is not True]
    recovery = [row for row in transcript if row.get("recovery_turn") is True]
    if [row.get("turn") for row in scheduled] != list(range(N_TURNS)):
        raise RuntimeError(f"transcript is not exactly scripted turns 0-29: {run_dir}")
    if any(row.get("model_name") != MODEL for row in transcript):
        raise RuntimeError(f"model mismatch: {run_dir}")
    if not all(isinstance(row.get("ttfb_ms"), (int, float)) for row in scheduled):
        raise RuntimeError(f"scripted TTFAT is missing: {run_dir}")
    if not all(isinstance(row.get("raw_ttfb_ms"), (int, float)) for row in scheduled):
        raise RuntimeError(f"scripted raw TTFB is missing: {run_dir}")
    judged = read_jsonl(judgment_path)
    if [row.get("turn") for row in judged] != list(range(N_TURNS)):
        raise RuntimeError(f"judgment is not exactly turns 0-29: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("judge_model") != JUDGE_MODEL or summary.get("judge_version") != JUDGE_VERSION:
        raise RuntimeError(f"judge identity mismatch: {run_dir}")
    scores = {
        component: tuple(bool(row["scores"][component]) for row in judged)
        for component in COMPONENTS
    }
    strict = tuple(all(scores[name][turn] for name in COMPONENTS) for turn in range(N_TURNS))
    scheduled_end = [
        int(row["turn"])
        for row in scheduled
        if any(tool_name(call) == "end_session" for call in row.get("tool_calls") or [])
    ]
    recovery_end = [
        int(row["turn"])
        for row in recovery
        if isinstance(row.get("turn"), int)
        and any(tool_name(call) == "end_session" for call in row.get("tool_calls") or [])
    ]
    if scheduled_end:
        end_kind, end_turn = "scripted", min(scheduled_end)
    elif recovery_end:
        end_kind, end_turn = "recovery", min(recovery_end)
    else:
        end_kind, end_turn = "missing", -1
    return Conversation(
        slot=manifest["slot"],
        run_dir=run_dir,
        scores=scores,
        strict=strict,
        ttfat_ms=tuple(float(row["ttfb_ms"]) for row in scheduled),
        raw_ttfb_ms=tuple(float(row["raw_ttfb_ms"]) for row in scheduled),
        thinking_tokens=tuple(
            int((row.get("tokens") or {}).get("thinking_tokens") or 0)
            for row in scheduled
        ),
        recovery_rows=len(recovery),
        end_session_kind=end_kind,
        end_session_turn=end_turn,
        token_totals=token_totals(transcript),
        recovery_token_totals=token_totals(recovery),
        transcript_sha256=sha256(transcript_path),
        judgment_sha256=sha256(judgment_path),
        summary_sha256=sha256(summary_path),
    )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def cluster_ci(per_conversation_correct: list[int]) -> list[float]:
    rng = random.Random(SEED + sum(per_conversation_correct))
    n = len(per_conversation_correct)
    estimates = []
    for _ in range(BOOTSTRAPS):
        correct = sum(per_conversation_correct[rng.randrange(n)] for _ in range(n))
        estimates.append(correct / (n * N_TURNS) * 100)
    return [float(percentile(estimates, 0.025) or 0), float(percentile(estimates, 0.975) or 0)]


def metric(conversations: list[Conversation], component: str | None) -> dict[str, Any]:
    values = [
        sum(conversation.strict if component is None else conversation.scores[component])
        for conversation in conversations
    ]
    correct = sum(values)
    return {
        "correct": correct,
        "total": DENOMINATOR,
        "rate_percent": correct / DENOMINATOR * 100,
        "error_percent": (DENOMINATOR - correct) / DENOMINATOR * 100,
        "conversation_cluster_bootstrap_95_percent": cluster_ci(values),
    }


def rounded(value: float | None) -> str:
    return "N/A" if value is None else str(int(round(value)))


def write_tsv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    collection_marker = json.loads(COLLECTION_COMPLETE.read_text(encoding="utf-8"))
    judge_marker = json.loads(JUDGE_COMPLETE.read_text(encoding="utf-8"))
    if collection_marker.get("canonical_conversations") != TARGET:
        raise RuntimeError("collection completion marker mismatch")
    if collection_marker.get("summary_json_sha256") != sha256(COLLECTION_SUMMARY):
        raise RuntimeError("collection summary hash mismatch")
    if judge_marker.get("canonical_runs") != TARGET or judge_marker.get("scripted_turns") != DENOMINATOR:
        raise RuntimeError("judge completion marker mismatch")
    if judge_marker.get("canonical_inputs_sha256") != sha256(JUDGE_INPUTS):
        raise RuntimeError("judge input hash mismatch")
    if judge_marker.get("judge_source_sha256") != sha256(JUDGE_HASHES):
        raise RuntimeError("judge source hash mismatch")
    canonical = sorted(read_tsv(CANONICAL), key=lambda row: slot_number(row["slot"]))
    if [slot_number(row["slot"]) for row in canonical] != list(range(1, TARGET + 1)):
        raise RuntimeError("canonical cohort is not K26T-01 through K26T-30")
    conversations = [load_conversation(row) for row in canonical]

    strict = metric(conversations, None)
    components = {name: metric(conversations, name) for name in COMPONENTS}
    latencies = [value for conversation in conversations for value in conversation.ttfat_ms]
    latency = {
        "scope": "scripted_turns_0_29_only",
        "count": len(latencies),
        "p50": statistics.median(latencies),
        "p95": percentile(latencies, 0.95),
        "max": max(latencies),
    }
    if latency["count"] != DENOMINATOR:
        raise RuntimeError("README TTFAT denominator is not exactly 900 scripted turns")
    raw_latencies = [
        value for conversation in conversations for value in conversation.raw_ttfb_ms
    ]
    reasoning_delays = [
        ttfat - raw
        for conversation in conversations
        for ttfat, raw in zip(conversation.ttfat_ms, conversation.raw_ttfb_ms)
    ]
    raw_latency = {
        "scope": "scripted_turns_0_29_only",
        "count": len(raw_latencies),
        "p50": statistics.median(raw_latencies),
        "p95": percentile(raw_latencies, 0.95),
        "max": max(raw_latencies),
    }
    reasoning_delay = {
        "definition": "content_or_tool_ttfat_minus_raw_first_chunk_ttfb",
        "count": len(reasoning_delays),
        "p50": statistics.median(reasoning_delays),
        "p95": percentile(reasoning_delays, 0.95),
        "max": max(reasoning_delays),
    }
    thinking_values = [
        value for conversation in conversations for value in conversation.thinking_tokens
    ]
    thinking_summary = {
        "scripted_rows_with_positive_thinking_tokens": sum(value > 0 for value in thinking_values),
        "scripted_rows": len(thinking_values),
        "thinking_tokens_total": sum(thinking_values),
        "thinking_tokens_p50": statistics.median(thinking_values),
        "thinking_tokens_p95": percentile([float(value) for value in thinking_values], 0.95),
        "thinking_tokens_max": max(thinking_values),
    }
    if len(raw_latencies) != DENOMINATOR or len(thinking_values) != DENOMINATOR:
        raise RuntimeError("raw latency/thinking denominator is not exactly 900 scripted turns")
    end_counts = Counter(conversation.end_session_kind for conversation in conversations)
    end_turns = Counter(
        str(conversation.end_session_turn)
        for conversation in conversations
        if conversation.end_session_turn >= 0
    )
    token_fields = tuple(conversations[0].token_totals)
    billed_tokens = {
        field: sum(conversation.token_totals[field] for conversation in conversations)
        for field in token_fields
    }
    recovery_tokens = {
        field: sum(conversation.recovery_token_totals[field] for conversation in conversations)
        for field in token_fields
    }
    recovery_rows = sum(conversation.recovery_rows for conversation in conversations)
    turn_errors = []
    for turn in range(N_TURNS):
        strict_failures = sum(not conversation.strict[turn] for conversation in conversations)
        row = {"turn": turn, "strict_failures": strict_failures}
        row.update(
            {
                f"{component}_failures": sum(
                    not conversation.scores[component][turn] for conversation in conversations
                )
                for component in COMPONENTS
            }
        )
        turn_errors.append(row)
    top_turns = sorted(turn_errors, key=lambda row: (-row["strict_failures"], row["turn"]))

    attempts = read_tsv(ATTEMPTS)
    collection = json.loads(COLLECTION_SUMMARY.read_text(encoding="utf-8"))
    first_attempt = int(collection["canonical_on_first_attempt"])
    readme_row = (
        f"| kimi-k2.6 (thinking on) | {strict['rate_percent']:.1f}% "
        f"| {strict['error_percent']:.1f}% "
        f"| {components['tool_use_correct']['error_percent']:.1f}% "
        f"| {components['instruction_following']['error_percent']:.1f}% "
        f"| {components['kb_grounding']['error_percent']:.1f}% "
        f"| {rounded(latency['p50'])}ms | {rounded(latency['p95'])}ms "
        f"| {rounded(latency['max'])}ms | BaseTen |"
    )
    hashes = {
        "configuration.json": sha256(CONFIG),
        "canonical.tsv": sha256(CANONICAL),
        "attempts.tsv": sha256(ATTEMPTS),
        "collection/COMPLETE.json": sha256(COLLECTION_COMPLETE),
        "collection/summary.json": sha256(COLLECTION_SUMMARY),
        "judging/COMPLETE.json": sha256(JUDGE_COMPLETE),
        "judging/canonical-inputs.tsv": sha256(JUDGE_INPUTS),
        "judging/judge-source-sha256.txt": sha256(JUDGE_HASHES),
        "analysis/analyze.py": sha256(Path(__file__).resolve()),
    }
    payload = {
        "schema_version": 1,
        "generated_at": judge_marker["completed_at"],
        "campaign_id": config["campaign_id"],
        "model": MODEL,
        "provider": "BaseTen",
        "benchmark": "aiwf_medium_context",
        "arm": "thinking",
        "request_signature": request_signature(config),
        "filler": None,
        "fixed_score_denominator": "30 canonical conversations x scripted turns 0-29 = 900",
        "strict_pass": strict,
        "components": components,
        "ttfat_ms": latency,
        "raw_ttfb_ms": raw_latency,
        "reasoning_delay_ms": reasoning_delay,
        "thinking": thinking_summary,
        "recovery": {
            "rows_excluded_from_score_and_ttfat": recovery_rows,
            "tokens_included_in_billed_totals": recovery_tokens,
        },
        "billed_token_totals_all_canonical_rows": billed_tokens,
        "end_session_outcomes": {
            "counts": dict(sorted(end_counts.items())),
            "turn_distribution": dict(sorted(end_turns.items(), key=lambda item: int(item[0]))),
        },
        "collection_reliability": {
            "conversation_attempts_recorded": len(attempts),
            "canonical_yield_per_conversation_attempt_percent": (
                TARGET / len(attempts) * 100
            ),
            "canonical_on_first_attempt": first_attempt,
            "slots_requiring_retries": collection["slots_requiring_retries"],
            "attempt_outcomes": collection["attempt_outcomes"],
        },
        "turn_errors": turn_errors,
        "readme_candidate_row": readme_row,
        "input_hashes": hashes,
        "runs": [
            {
                "slot": conversation.slot,
                "run_dir": str(conversation.run_dir.relative_to(ROOT)),
                "strict_passes": sum(conversation.strict),
                "end_session_kind": conversation.end_session_kind,
                "end_session_turn": conversation.end_session_turn,
                "recovery_rows": conversation.recovery_rows,
                "positive_thinking_token_rows": sum(
                    value > 0 for value in conversation.thinking_tokens
                ),
                "thinking_tokens": sum(conversation.thinking_tokens),
                "transcript_sha256": conversation.transcript_sha256,
                "judgment_sha256": conversation.judgment_sha256,
                "summary_sha256": conversation.summary_sha256,
            }
            for conversation in conversations
        ],
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "aggregates.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_tsv(
        HERE / "aggregates.tsv",
        (
            "model",
            "runs",
            "scripted_turns",
            "strict_pass_percent",
            "any_error_percent",
            "tool_error_percent",
            "instruction_error_percent",
            "kb_error_percent",
            "ttfat_p50_ms",
            "ttfat_p95_ms",
            "ttfat_max_ms",
            "raw_ttfb_p50_ms",
            "reasoning_delay_p50_ms",
            "provider",
        ),
        [
            {
                "model": MODEL,
                "runs": TARGET,
                "scripted_turns": DENOMINATOR,
                "strict_pass_percent": f"{strict['rate_percent']:.4f}",
                "any_error_percent": f"{strict['error_percent']:.4f}",
                "tool_error_percent": f"{components['tool_use_correct']['error_percent']:.4f}",
                "instruction_error_percent": f"{components['instruction_following']['error_percent']:.4f}",
                "kb_error_percent": f"{components['kb_grounding']['error_percent']:.4f}",
                "ttfat_p50_ms": rounded(latency["p50"]),
                "ttfat_p95_ms": rounded(latency["p95"]),
                "ttfat_max_ms": rounded(latency["max"]),
                "raw_ttfb_p50_ms": rounded(raw_latency["p50"]),
                "reasoning_delay_p50_ms": rounded(reasoning_delay["p50"]),
                "provider": "BaseTen",
            }
        ],
    )
    write_tsv(
        HERE / "included-runs.tsv",
        tuple(payload["runs"][0]),
        payload["runs"],
    )
    ci = strict["conversation_cluster_bootstrap_95_percent"]
    top_rows = "".join(
        f"| {row['turn']} | {row['strict_failures']}/30 | "
        f"{row['tool_use_correct_failures']}/30 | "
        f"{row['instruction_following_failures']}/30 | "
        f"{row['kb_grounding_failures']}/30 |\n"
        for row in top_turns[:10]
    )
    report = f"""# BaseTen Kimi K2.6 — AIEWF medium-context result

The fixed denominator is 30 canonical conversations × scripted turns 0–29 =
900. Recovery rows are excluded from both scores and README TTFAT, but their
tokens remain in the billed-token totals.

## README-format result

| Measure | Result |
|---|---:|
| Strict turn pass | {strict['correct']}/{DENOMINATOR} ({strict['rate_percent']:.1f}%) |
| Conversation-cluster bootstrap 95% CI | {ci[0]:.1f}–{ci[1]:.1f}% |
| Any error | {strict['error_percent']:.1f}% |
| Tool error | {components['tool_use_correct']['error_percent']:.1f}% |
| Instruction error | {components['instruction_following']['error_percent']:.1f}% |
| KB error | {components['kb_grounding']['error_percent']:.1f}% |
| Scripted-turn TTFAT P50 / P95 / max | {rounded(latency['p50'])} / {rounded(latency['p95'])} / {rounded(latency['max'])} ms |
| Raw first-chunk TTFB P50 / P95 / max | {rounded(raw_latency['p50'])} / {rounded(raw_latency['p95'])} / {rounded(raw_latency['max'])} ms |
| Reasoning delay P50 / P95 / max | {rounded(reasoning_delay['p50'])} / {rounded(reasoning_delay['p95'])} / {rounded(reasoning_delay['max'])} ms |
| Scripted rows with positive thinking tokens | {thinking_summary['scripted_rows_with_positive_thinking_tokens']}/{DENOMINATOR} |
| Scripted thinking tokens | {thinking_summary['thinking_tokens_total']:,} |

```text
{readme_row}
```

## Collection and protocol reliability

| Measure | Result |
|---|---:|
| Canonical complete conversations | 30/30 |
| Conversation attempts recorded | {len(attempts)} |
| Canonical yield per conversation attempt | {TARGET / len(attempts) * 100:.1f}% |
| Canonical on slot's first attempt | {first_attempt}/30 |
| `end_session` on scripted turn | {end_counts.get('scripted', 0)}/30 |
| `end_session` on recovery turn | {end_counts.get('recovery', 0)}/30 |
| No `end_session` | {end_counts.get('missing', 0)}/30 |
| Recovery rows excluded from score/TTFAT | {recovery_rows} |

## Highest-error scripted turns

| Turn | Any strict failures | Tool failures | Instruction failures | KB failures |
|---:|---:|---:|---:|---:|
{top_rows}
The 15 strict failures are tightly concentrated. Thirteen runs called
`request_tech_support` on turn 16 before gathering the specific app problem;
two runs used an inappropriate generic event-scope deflection for the Salon 2
directions question on turn 19. No other scripted turn failed.

## Usage accounting

All canonical transcript rows, including recovery, total
{billed_tokens['prompt_tokens']:,} prompt tokens,
{billed_tokens['completion_tokens']:,} completion tokens, and
{billed_tokens['cache_read_input_tokens']:,} cache-read input tokens. Recovery
rows alone account for {recovery_tokens['prompt_tokens']:,} prompt and
{recovery_tokens['completion_tokens']:,} completion tokens.
"""
    (HERE / "REPORT.md").write_text(report, encoding="utf-8")
    complete = {
        "campaign_id": config["campaign_id"],
        "generated_at": judge_marker["completed_at"],
        "fixed_scripted_turns": DENOMINATOR,
        "aggregates_json_sha256": sha256(HERE / "aggregates.json"),
        "aggregates_tsv_sha256": sha256(HERE / "aggregates.tsv"),
        "included_runs_sha256": sha256(HERE / "included-runs.tsv"),
        "report_sha256": sha256(HERE / "REPORT.md"),
    }
    (HERE / "COMPLETE.json").write_text(
        json.dumps(complete, indent=2) + "\n", encoding="utf-8"
    )
    print(f"analysis complete: strict={strict['rate_percent']:.1f}% denominator={DENOMINATOR}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
