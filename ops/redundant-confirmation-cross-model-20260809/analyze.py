#!/usr/bin/env python3
"""Descriptive Turn-12 redundant-confirmation census across AIWF runs.

The script deliberately keeps exact reported model names separate.  It emits
both a broad inventory and a stricter standard-prompt/no-filler/complete-run
view; neither should be interpreted as a randomized cross-model comparison.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/aiwf_medium_context"
OUT = Path(__file__).resolve().parent / "results"
TARGET = "Oh, one more suggestion. How about a session on state machine abstractions for complex workflows?."
SOURCE_CUTOFF_DATE = "20260809"

FALSE_COMPLETION = re.compile(
    r"\b(i(?:'ve| have)|we(?:'ve| have))\s+(?:now\s+)?(?:submitted|added|recorded)|"
    r"\b(?:has been|is now)\s+(?:submitted|added|recorded)|\btaken care of\b",
    re.I,
)
QUESTION_OR_CONFIRM = re.compile(
    r"\b(?:confirm|confirmation|would you like|shall i|should i|want me to|go ahead|"
    r"please say|please confirm|one moment)\b|\?",
    re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def norm(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def correct_turn11(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    calls = row.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("name") != "submit_session_suggestion":
        return False
    args = calls[0].get("args") or {}
    suggestion = norm(args.get("suggestion_text"))
    return (
        isinstance(args, dict)
        and norm(args.get("name")) == "jennifer smith"
        and all(token in suggestion.split() for token in ("open", "telemetry", "tracing"))
    )


def classify_turn12(row: dict[str, Any]) -> str:
    calls = row.get("tool_calls") or []
    if len(calls) > 1:
        return "duplicate_or_multiple_tool_calls"
    if len(calls) == 1:
        call = calls[0]
        args = call.get("args") or {}
        if call.get("name") != "submit_session_suggestion":
            return "wrong_tool"
        suggestion = norm(args.get("suggestion_text"))
        if (
            isinstance(args, dict)
            and {"name", "suggestion_text"}.issubset(args)
            and norm(args.get("name")) == "jennifer smith"
            and all(token in suggestion.split() for token in (
                "state", "machine", "abstractions", "complex", "workflows"
            ))
        ):
            return "correct_tool_and_arguments"
        return "correct_tool_wrong_or_missing_argument"
    text = str(row.get("assistant_text") or "")
    if FALSE_COMPLETION.search(text):
        return "no_tool_false_claim_of_completion"
    if QUESTION_OR_CONFIRM.search(text):
        return "no_tool_redundant_confirmation_or_question"
    return "no_tool_other"


def redundant_subtype(text: str) -> str:
    lowered = " ".join(text.split()).lower()
    if re.search(
        r"i(?:'ve| have).{0,45}(?:sent|submitted|added|recorded)|"
        r"gone ahead and|successfully submitted",
        lowered,
    ) and re.search(r"anything else|\?", lowered):
        return "post_action_claim_then_followup"
    if re.search(
        r"under your name|use your name|also from you|coming from you|"
        r"for someone else|is this suggestion.*from you|is this.*also from you|"
        r"your name.*(?:again|still)|name.*jennifer",
        lowered,
    ):
        return "identity_or_ownership_reconfirmation"
    if re.search(
        r"you(?:'d| would) like (?:to suggest|a session|this suggestion)|"
        r"session on .*\b(?:correct|right)\b|suggestion.*\b(?:correct|right)\b|"
        r"is that (?:the|your) suggestion",
        lowered,
    ):
        return "content_or_intent_reconfirmation"
    if re.search(
        r"should i submit|shall i|want me to|would you like me to|go ahead|"
        r"would you like (?:for )?me to|may i submit|can i submit",
        lowered,
    ):
        return "authorization_reconfirmation"
    return "other_question"


def infer_signature(log: str) -> tuple[str, str, str]:
    # Do not let provider/model names inside the full system prompt masquerade
    # as serving provenance.
    header = log.split("Generating chat", 1)[0]
    filler = "none"
    match = re.search(r"MTE_FILLER_(DOTS|DASHES).*?appending\s+(\d+)", log, re.I)
    if match:
        filler = f"{match.group(1).lower()}{match.group(2)}"
    elif "MTE_FILLER" in log:
        filler = "other_filler"

    reasoning = "unspecified"
    patterns = [
        r"reasoning\.effort=([a-z0-9_-]+)",
        r"reasoning_effort=([a-z0-9_-]+)",
        r"thinking=(True|False)",
        r"thinking_budget=(-?\d+)",
    ]
    for pattern in patterns:
        found = re.search(pattern, header, re.I)
        if found:
            value = found.group(1).lower()
            reasoning = {"false": "off", "true": "on"}.get(value, value)
            break

    service = "unspecified"
    service_patterns = [
        ("openai-responses", r"OpenAIResponsesLLMService|Responses API"),
        ("vllm-openai", r"Using vllm-openai"),
        ("baseten", r"Using BaseTen"),
        ("lilac", r"Using Lilac"),
        ("openrouter", r"Using OpenRouter"),
        ("anthropic", r"Anthropic"),
        ("gemini", r"GeminiLive|GoogleLive|gemini"),
        ("realtime", r"Realtime"),
    ]
    for name, pattern in service_patterns:
        if re.search(pattern, header, re.I):
            service = name
            break
    return filler, reasoning, service


def wilson(successes: int, n: int) -> tuple[float, float]:
    if not n:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (center - half), 100 * (center + half))


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        value = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = value
        i = j + 1
    return out


def correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - xm) ** 2 for x in xs) * sum((y - ym) ** 2 for y in ys))
    return num / den if den else float("nan")


def summarize(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group_key, items in groups.items():
        counts = Counter(item["category"] for item in items)
        n = len(items)
        redundant = counts["no_tool_redundant_confirmation_or_question"]
        correct = counts["correct_tool_and_arguments"]
        failures = n - correct
        low, high = wilson(redundant, n)
        entry = {key: value for key, value in zip(keys, group_key)}
        entry.update(
            n=n,
            eligible_n=sum(item["eligible_turn11"] for item in items),
            correct=correct,
            correct_pct=100 * correct / n,
            redundant=redundant,
            redundant_pct=100 * redundant / n,
            redundant_ci_low=low,
            redundant_ci_high=high,
            redundant_share_of_failures_pct=100 * redundant / failures if failures else 0.0,
            false_completion=counts["no_tool_false_claim_of_completion"],
            no_tool_other=counts["no_tool_other"],
            other_failures=failures - redundant - counts["no_tool_false_claim_of_completion"] - counts["no_tool_other"],
            mean_judged_pass_pct=sum(item["judged_pass_pct"] for item in items) / n,
        )
        result.append(entry)
    return sorted(result, key=lambda row: (-row["n"], *(str(row[key]) for key in keys)))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    for transcript in sorted(RUNS.glob("*/transcript.jsonl")):
        run_dir = transcript.parent
        timestamp = re.match(r"^(\d{8})T", run_dir.name)
        if timestamp and timestamp.group(1) > SOURCE_CUTOFF_DATE:
            skipped["after_source_cutoff"] += 1
            continue
        raw = read_jsonl(transcript)
        by_turn = {row.get("turn"): row for row in raw if isinstance(row.get("turn"), int)}
        target = by_turn.get(12)
        if not target or target.get("user_text") != TARGET:
            skipped["no_canonical_turn12"] += 1
            continue
        original_turns = {turn for turn in by_turn if 0 <= turn < 30}
        complete = original_turns == set(range(30))
        log_path = run_dir / "run.log"
        # Configuration signatures are emitted at startup.  Some later DEBUG
        # lines contain the full prompt on a single multi-megabyte line, so a
        # bounded prefix is both sufficient and dramatically faster.
        if log_path.exists():
            with log_path.open("rb") as handle:
                log = handle.read(262_144).decode(errors="replace")
        else:
            log = ""
        filler, reasoning, service = infer_signature(log)
        judged_path = run_dir / "claude_judged.jsonl"
        judged_rows = read_jsonl(judged_path) if judged_path.exists() else []
        judged_original = [r for r in judged_rows if isinstance(r.get("turn"), int) and 0 <= r["turn"] < 30]
        passed = 0
        for judged in judged_original:
            scores = judged.get("scores") or {}
            passed += int(bool(scores) and all(scores.get(k) is True for k in (
                "turn_taking", "tool_use_correct", "instruction_following", "kb_grounding"
            )))
        category = classify_turn12(target)
        text = str(target.get("assistant_text") or "")
        rows.append(
            {
                "run_dir": str(run_dir.relative_to(ROOT)),
                "model": str(target.get("model_name") or "unknown"),
                "reasoning": reasoning,
                "filler": filler,
                "service": service,
                "complete_30": complete,
                "judged_turns": len(judged_original),
                "judged_pass_pct": 100 * passed / len(judged_original) if judged_original else float("nan"),
                "eligible_turn11": correct_turn11(by_turn.get(11)),
                "category": category,
                "redundant_subtype": redundant_subtype(text) if category == "no_tool_redundant_confirmation_or_question" else "",
                "assistant_text": " ".join(text.split()),
            }
        )

    standard = [
        row for row in rows
        if row["complete_30"] and row["judged_turns"] == 30 and row["filler"] == "none"
    ]
    complete_judged = [
        row for row in rows if row["complete_30"] and row["judged_turns"] == 30
    ]
    eligible = [row for row in standard if row["eligible_turn11"]]
    model_summary = summarize(standard, ("model",))
    eligible_model_summary = summarize(eligible, ("model",))
    config_summary = summarize(standard, ("model", "reasoning", "service"))
    intervention_summary = summarize(
        complete_judged, ("model", "reasoning", "service", "filler")
    )

    subtype_rows = []
    for (model, subtype), count in sorted(Counter(
        (row["model"], row["redundant_subtype"]) for row in standard if row["redundant_subtype"]
    ).items()):
        subtype_rows.append({"model": model, "subtype": subtype, "count": count})

    qualifying = [row for row in model_summary if row["n"] >= 10]
    quality_x = [row["mean_judged_pass_pct"] for row in qualifying]
    redundant_all = [row["redundant_pct"] for row in qualifying]
    redundant_share = [row["redundant_share_of_failures_pct"] for row in qualifying]
    correlations = {
        "models_n_ge_10": len(qualifying),
        "spearman_quality_vs_redundant_all_turn12_rate": correlation(rank(quality_x), rank(redundant_all)),
        "spearman_quality_vs_redundant_share_of_turn12_failures": correlation(rank(quality_x), rank(redundant_share)),
    }

    write_csv(OUT / "all-turn12-rows.csv", rows)
    write_csv(OUT / "model-summary-standard.csv", model_summary)
    write_csv(OUT / "eligible-model-summary-standard.csv", eligible_model_summary)
    write_csv(OUT / "configuration-summary-standard.csv", config_summary)
    write_csv(OUT / "configuration-summary-with-fillers.csv", intervention_summary)
    write_csv(OUT / "redundant-subtypes-standard.csv", subtype_rows)
    payload = {
        "schema_version": 1,
        "scope": {
            "source_cutoff_date": SOURCE_CUTOFF_DATE,
            "all_canonical_turn12_rows": len(rows),
            "standard_complete_judged_no_filler_rows": len(standard),
            "eligible_turn11_subset_rows": len(eligible),
            "exact_reported_models_standard": len(model_summary),
            "skipped": dict(skipped),
        },
        "correlations": correlations,
        "standard_aggregate_categories": dict(Counter(row["category"] for row in standard)),
        "eligible_aggregate_categories": dict(Counter(row["category"] for row in eligible)),
    }
    (OUT / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
