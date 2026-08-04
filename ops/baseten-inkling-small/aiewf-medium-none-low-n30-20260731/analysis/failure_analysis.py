#!/usr/bin/env python3
"""Attribute raw completion failures in the frozen Inkling Small campaign.

The only membership source is ``../canonical.tsv``.  The default invocation is
read-only; ``--write`` atomically emits ``FAILURE-ANALYSIS.json`` and a Markdown
companion.  This analysis reads transcripts and run logs only.  It never reads
or modifies Claude judge outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
ROOT = CAMPAIGN.parents[2]
CANONICAL_PATH = CAMPAIGN / "canonical.tsv"
SCHEDULE_PATH = CAMPAIGN / "frozen-order.tsv"
ATTEMPTS_PATH = CAMPAIGN / "attempts.tsv"
JSON_OUTPUT = HERE / "FAILURE-ANALYSIS.json"
MARKDOWN_OUTPUT = HERE / "FAILURE-ANALYSIS.md"

ARMS = ("none", "low")
N_TURNS = 30
TARGET_PER_ARM = 30
FOCUS_TURNS = (13, 14, 15, 16, 17, 28, 29)
MODEL = "thinkingmachines/inkling-small"

CAUSES = (
    "strict_complete",
    "model_abort",
    "recovery_end_session",
    "baseten_429_idle",
    "unattributed_short",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError as exc:
        fail(f"path escapes repository root: {path}")
        raise AssertionError from exc


def resolve_run_dir(value: str) -> Path:
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        fail(f"canonical run path escapes repository root: {value}")
        raise AssertionError from exc
    return resolved


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        fail(f"missing or empty TSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
        fail(f"missing or empty JSONL: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON at {path}:{line_number}: {exc}")
        if not isinstance(row, dict):
            fail(f"non-object JSON row at {path}:{line_number}")
        rows.append(row)
    if not rows:
        fail(f"JSONL contains no rows: {path}")
    return rows


def tool_name(call: Any) -> str | None:
    if not isinstance(call, dict):
        return None
    if isinstance(call.get("name"), str):
        return call["name"]
    function = call.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return None


def tool_names(row: dict[str, Any]) -> tuple[str, ...]:
    result: list[str] = []
    for call in row.get("tool_calls") or []:
        name = tool_name(call)
        result.append(name if name is not None else "<unnamed>")
    return tuple(result)


def response_present(row: dict[str, Any]) -> bool:
    text = row.get("assistant_text")
    return bool((isinstance(text, str) and text.strip()) or tool_names(row))


def response_pattern(row: dict[str, Any] | None) -> str:
    if row is None:
        return "missing"
    names = tool_names(row)
    has_text = bool(
        isinstance(row.get("assistant_text"), str)
        and row["assistant_text"].strip()
    )
    if names:
        prefix = "text_and_tool" if has_text else "tool_only"
        return f"{prefix}:{'+'.join(names)}"
    return "text_only" if has_text else "empty"


def scheduled_map(rows: Iterable[dict[str, Any]], *, slot: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("recovery_turn") is True:
            continue
        turn = row.get("turn")
        if not isinstance(turn, int) or not 0 <= turn < N_TURNS:
            fail(f"{slot}: invalid scheduled turn {turn!r}")
        if turn in result:
            fail(f"{slot}: duplicate scheduled turn {turn}")
        if row.get("model_name") != MODEL:
            fail(f"{slot}: unexpected model {row.get('model_name')!r}")
        result[turn] = row
    observed = sorted(result)
    if observed != list(range(len(observed))):
        fail(f"{slot}: scheduled turns are not a contiguous prefix: {observed}")
    if not result:
        fail(f"{slot}: canonical transcript has no scheduled response")
    return result


def end_session_turns(
    rows: Iterable[dict[str, Any]], *, recovery: bool
) -> tuple[int, ...]:
    result: list[int] = []
    for row in rows:
        if (row.get("recovery_turn") is True) != recovery:
            continue
        if "end_session" in tool_names(row) and isinstance(row.get("turn"), int):
            result.append(row["turn"])
    return tuple(sorted(result))


def classify_run(
    *,
    scheduled: dict[int, dict[str, Any]],
    all_rows: list[dict[str, Any]],
    run_log_text: str,
) -> str:
    """Classify one run from raw transcript and serving evidence."""
    observed = tuple(sorted(scheduled))
    scheduled_ends = end_session_turns(all_rows, recovery=False)
    recovery_ends = end_session_turns(all_rows, recovery=True)
    if observed == tuple(range(N_TURNS)) and scheduled_ends == (29,):
        return "strict_complete"
    if scheduled_ends:
        return "model_abort"
    if recovery_ends:
        return "recovery_end_session"
    if len(scheduled) < N_TURNS and terminal_baseten_429_idle(run_log_text):
        return "baseten_429_idle"
    return "unattributed_short"


def terminal_baseten_429_idle(run_log_text: str) -> bool:
    """Return whether the final attempted generation failed 429 then idled."""
    lines = run_log_text.splitlines()
    if not any("Using BaseTen" in line for line in lines):
        return False

    progress_markers = ("queue_turn:", "Recorded turn ", " TTFB:", "EOT (")
    progress = [
        index
        for index, line in enumerate(lines)
        if any(marker in line for marker in progress_markers)
    ]
    errors = [
        index
        for index, line in enumerate(lines)
        if "Error code: 429" in line and "Rate limit exceeded" in line
    ]
    idles = [
        index
        for index, line in enumerate(lines)
        if "Idle timeout detected." in line
    ]
    if not progress or not errors or not idles:
        return False

    last_progress = max(progress)
    terminal_errors = [index for index in errors if index > last_progress]
    if not terminal_errors:
        return False
    return max(terminal_errors) < max(idles)


def validate_canonical(
    rows: list[dict[str, str]],
    *,
    schedule: list[dict[str, str]],
    attempts: list[dict[str, str]],
) -> None:
    required = {
        "slot",
        "pair",
        "arm",
        "attempt",
        "run_dir",
        "scheduled_rows",
        "response_turns",
        "end_session_turn",
        "classification",
    }
    if not rows or not required.issubset(rows[0]):
        fail(f"canonical.tsv lacks required fields: {sorted(required)}")
    if len(rows) != 2 * TARGET_PER_ARM:
        fail(f"expected 60 canonical rows, found {len(rows)}")
    slots = [row["slot"] for row in rows]
    if slots != [f"IS-{index:02d}" for index in range(1, 61)]:
        fail("canonical slots are not exactly IS-01 through IS-60")
    if len({row["run_dir"] for row in rows}) != len(rows):
        fail("canonical.tsv contains duplicate run directories")
    counts = Counter(row["arm"] for row in rows)
    if counts != Counter({"none": TARGET_PER_ARM, "low": TARGET_PER_ARM}):
        fail(f"canonical arms are not 30/30: {dict(counts)}")

    schedule_required = {"slot", "pair", "arm"}
    if not schedule or not schedule_required.issubset(schedule[0]):
        fail(f"frozen-order.tsv lacks required fields: {sorted(schedule_required)}")
    if len(schedule) != len(rows):
        fail(f"frozen schedule has {len(schedule)} rows; expected {len(rows)}")
    for row, assignment in zip(rows, schedule):
        for field in ("slot", "pair", "arm"):
            if row[field] != assignment[field]:
                fail(
                    f"canonical {field} mismatch against frozen schedule at "
                    f"{row['slot']}"
                )

    attempt_identity_fields = (
        "slot",
        "pair",
        "arm",
        "attempt",
        "run_dir",
        "scheduled_rows",
        "response_turns",
        "end_session_turn",
        "classification",
    )
    if not attempts or not set(attempt_identity_fields).issubset(attempts[0]):
        fail(
            "attempts.tsv lacks required fields: "
            f"{sorted(attempt_identity_fields)}"
        )
    attempt_identities = Counter(
        tuple(attempt[field] for field in attempt_identity_fields)
        for attempt in attempts
    )
    for row in rows:
        identity = tuple(row[field] for field in attempt_identity_fields)
        if attempt_identities[identity] != 1:
            fail(
                f"canonical run identity for {row['slot']} has "
                f"{attempt_identities[identity]} exact attempts.tsv matches"
            )


def summarize_turn(runs: list[dict[str, Any]], turn: int) -> dict[str, Any]:
    observed_rows = [run["scheduled"].get(turn) for run in runs]
    present = [row for row in observed_rows if row is not None]
    patterns = Counter(response_pattern(row) for row in observed_rows)
    calls = Counter(
        name
        for row in present
        for name in tool_names(row)
    )
    return {
        "assigned_conversations": len(runs),
        "observed_responses": len(present),
        "missing_responses": len(runs) - len(present),
        "response_patterns": dict(sorted(patterns.items())),
        "tool_calls": dict(sorted(calls.items())),
        "end_session_calls": calls.get("end_session", 0),
    }


def summarize_transition(
    runs: list[dict[str, Any]], first: int, second: int
) -> dict[str, int]:
    patterns = Counter(
        f"t{first}={response_pattern(run['scheduled'].get(first))} -> "
        f"t{second}={response_pattern(run['scheduled'].get(second))}"
        for run in runs
    )
    return dict(sorted(patterns.items()))


def run_record(row: dict[str, str]) -> dict[str, Any]:
    slot = row["slot"]
    run_dir = resolve_run_dir(row["run_dir"])
    transcript_path = run_dir / "transcript.jsonl"
    run_log_path = run_dir / "run.log"
    if not run_log_path.is_file() or not run_log_path.stat().st_size:
        fail(f"{slot}: missing or empty run.log")
    all_rows = read_jsonl(transcript_path)
    scheduled = scheduled_map(all_rows, slot=slot)
    response_turns = sum(response_present(value) for value in scheduled.values())
    if int(row["scheduled_rows"]) != len(scheduled):
        fail(f"{slot}: canonical scheduled_rows mismatch")
    if int(row["response_turns"]) != response_turns:
        fail(f"{slot}: canonical response_turns mismatch")

    scheduled_ends = end_session_turns(all_rows, recovery=False)
    recovery_ends = end_session_turns(all_rows, recovery=True)
    recorded_end = max(scheduled_ends, default=-1)
    if recovery_ends and recorded_end < 0:
        recorded_end = max(recovery_ends)
    if int(row["end_session_turn"]) != recorded_end:
        fail(f"{slot}: canonical end_session_turn mismatch")

    log_text = run_log_path.read_text(encoding="utf-8", errors="replace")
    cause = classify_run(
        scheduled=scheduled,
        all_rows=all_rows,
        run_log_text=log_text,
    )
    expected_manifest_class = {
        "strict_complete": "strict_complete",
        "model_abort": "model_abort",
        "recovery_end_session": "recovery_end_session",
        "baseten_429_idle": "incomplete_no_end_session",
        "unattributed_short": "incomplete_no_end_session",
    }[cause]
    if row["classification"] != expected_manifest_class:
        fail(
            f"{slot}: raw cause {cause} disagrees with canonical "
            f"classification {row['classification']}"
        )
    recovery_rows = [value for value in all_rows if value.get("recovery_turn") is True]
    return {
        "slot": slot,
        "arm": row["arm"],
        "run_dir": relative(run_dir),
        "scheduled": scheduled,
        "scheduled_turns": len(scheduled),
        "missing_turns": N_TURNS - len(scheduled),
        "cause": cause,
        "scheduled_end_session_turns": list(scheduled_ends),
        "recovery_end_session_turns": list(recovery_ends),
        "recovery_rows": recovery_rows,
        "transcript_path": transcript_path,
        "run_log_path": run_log_path,
    }


def arm_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    cause_counts = Counter(run["cause"] for run in runs)
    missing_by_cause = {
        cause: sum(run["missing_turns"] for run in runs if run["cause"] == cause)
        for cause in CAUSES
    }
    return {
        "conversations": len(runs),
        "fixed_turn_denominator": len(runs) * N_TURNS,
        "observed_scheduled_turns": sum(run["scheduled_turns"] for run in runs),
        "missing_scheduled_turns": sum(run["missing_turns"] for run in runs),
        "conversation_causes": {
            cause: {
                "count": cause_counts.get(cause, 0),
                "percent": 100 * cause_counts.get(cause, 0) / len(runs),
            }
            for cause in CAUSES
        },
        "missing_turns_by_cause": missing_by_cause,
        "focus_turns": {
            str(turn): summarize_turn(runs, turn) for turn in FOCUS_TURNS
        },
        "transitions": {
            "t14_to_t15": summarize_transition(runs, 14, 15),
            "t16_to_t17": summarize_transition(runs, 16, 17),
        },
    }


def recovery_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        (run, row)
        for run in runs
        for row in run["recovery_rows"]
        if row.get("turn") == 30
    ]
    patterns = Counter(response_pattern(row) for _, row in rows)
    return {
        "definition": "Synthetic recovery rows with transcript turn ID 30; not a scored scheduled turn.",
        "conversations_with_turn_30": len({run["slot"] for run, _ in rows}),
        "response_patterns": dict(sorted(patterns.items())),
        "end_session_calls": sum("end_session" in tool_names(row) for _, row in rows),
    }


def analyze() -> dict[str, Any]:
    canonical = read_tsv(CANONICAL_PATH)
    schedule = read_tsv(SCHEDULE_PATH)
    attempts = read_tsv(ATTEMPTS_PATH)
    validate_canonical(canonical, schedule=schedule, attempts=attempts)
    runs = [run_record(row) for row in canonical]
    by_arm = {
        arm: [run for run in runs if run["arm"] == arm]
        for arm in ARMS
    }
    unknown = [run["slot"] for run in runs if run["cause"] == "unattributed_short"]
    if unknown:
        fail(f"unattributed canonical short runs require review: {unknown}")

    input_runs = {
        run["slot"]: {
            "run_dir": run["run_dir"],
            "transcript": relative(run["transcript_path"]),
            "transcript_sha256": sha256(run["transcript_path"]),
            "run_log": relative(run["run_log_path"]),
            "run_log_sha256": sha256(run["run_log_path"]),
        }
        for run in runs
    }
    return {
        "schema_version": 1,
        "artifact_status": "RAW_CAUSE_ATTRIBUTION",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "campaign": "aiewf-medium-inkling-small-baseten-none-low-n30-20260731",
        "model": MODEL,
        "method": {
            "membership_source": relative(CANONICAL_PATH),
            "membership_validation_sources": [
                relative(SCHEDULE_PATH),
                relative(ATTEMPTS_PATH),
            ],
            "judge_dependency": False,
            "scheduled_turns_per_conversation": N_TURNS,
            "classification_precedence": list(CAUSES),
            "baseten_429_idle_definition": (
                "A short run with no end_session whose run.log contains the BaseTen "
                "signature, HTTP 429 rate-limit error, and idle-timeout cancellation."
            ),
            "recovery_turns_are_not_scheduled": True,
        },
        "inputs": {
            "canonical": {
                "path": relative(CANONICAL_PATH),
                "sha256": sha256(CANONICAL_PATH),
            },
            "frozen_order": {
                "path": relative(SCHEDULE_PATH),
                "sha256": sha256(SCHEDULE_PATH),
            },
            "attempts": {
                "path": relative(ATTEMPTS_PATH),
                "sha256": sha256(ATTEMPTS_PATH),
            },
            "analyzer": {
                "path": relative(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "runs": input_runs,
        },
        "arms": {arm: arm_summary(by_arm[arm]) for arm in ARMS},
        "synthetic_recovery_turn_30": {
            arm: recovery_summary(by_arm[arm]) for arm in ARMS
        },
        "run_causes": [
            {
                "slot": run["slot"],
                "arm": run["arm"],
                "run_dir": run["run_dir"],
                "cause": run["cause"],
                "scheduled_turns": run["scheduled_turns"],
                "missing_turns": run["missing_turns"],
                "scheduled_end_session_turns": run["scheduled_end_session_turns"],
                "recovery_end_session_turns": run["recovery_end_session_turns"],
            }
            for run in runs
        ],
    }


def pct(value: float) -> str:
    return f"{value:.1f}%"


def pattern_list(value: dict[str, int]) -> str:
    return "; ".join(f"`{key}` {count}" for key, count in value.items()) or "—"


def render_markdown(result: dict[str, Any]) -> str:
    arms = result["arms"]
    lines = [
        "# Inkling Small raw completion and failure attribution",
        "",
        (
            "This additive analysis uses only the 60 runs named by `canonical.tsv` and "
            "their raw transcripts/run logs. It has no Claude-judge dependency."
        ),
        "",
        "## Conversation outcomes",
        "",
        "| arm | strict complete | model abort | recovery end | BaseTen 429 + idle | observed / fixed turns |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        value = arms[arm]
        causes = value["conversation_causes"]
        lines.append(
            f"| {arm} | {causes['strict_complete']['count']}/30 "
            f"({pct(causes['strict_complete']['percent'])}) | "
            f"{causes['model_abort']['count']}/30 "
            f"({pct(causes['model_abort']['percent'])}) | "
            f"{causes['recovery_end_session']['count']}/30 "
            f"({pct(causes['recovery_end_session']['percent'])}) | "
            f"{causes['baseten_429_idle']['count']}/30 "
            f"({pct(causes['baseten_429_idle']['percent'])}) | "
            f"{value['observed_scheduled_turns']} / {value['fixed_turn_denominator']} |"
        )
    lines.extend(
        [
            "",
            "## Missing scheduled turns by immediate cause",
            "",
            "| arm | model abort | recovery end | BaseTen 429 + idle | total missing |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        value = arms[arm]
        missing = value["missing_turns_by_cause"]
        lines.append(
            f"| {arm} | {missing['model_abort']} | "
            f"{missing['recovery_end_session']} | {missing['baseten_429_idle']} | "
            f"{value['missing_scheduled_turns']} |"
        )
    lines.extend(["", "## Focus-turn response patterns", ""])
    for turn in FOCUS_TURNS:
        lines.extend(
            [
                f"### Turn {turn}",
                "",
                "| arm | observed | missing | exact response patterns |",
                "|---|---:|---:|---|",
            ]
        )
        for arm in ARMS:
            value = arms[arm]["focus_turns"][str(turn)]
            lines.append(
                f"| {arm} | {value['observed_responses']} | "
                f"{value['missing_responses']} | {pattern_list(value['response_patterns'])} |"
            )
        lines.append("")
    lines.extend(["## Transition cross-tabs", ""])
    for transition in ("t14_to_t15", "t16_to_t17"):
        lines.append(f"### {transition.replace('_', ' ')}")
        lines.append("")
        for arm in ARMS:
            lines.append(f"- `{arm}`: {pattern_list(arms[arm]['transitions'][transition])}.")
        lines.append("")

    none = arms["none"]
    low = arms["low"]
    low_t15 = low["focus_turns"]["15"]
    lines.extend(
        [
            "## Interpretation",
            "",
            (
                f"`none` strictly completed {none['conversation_causes']['strict_complete']['count']}/30 "
                f"conversations; `low` completed {low['conversation_causes']['strict_complete']['count']}/30. "
                f"All {none['conversation_causes']['baseten_429_idle']['count'] + low['conversation_causes']['baseten_429_idle']['count']} "
                "unended short runs were BaseTen 429-plus-idle serving failures, not generated terminal calls."
            ),
            "",
            (
                f"At turn 15, `low` generated {low_t15['end_session_calls']} direct `end_session` calls. "
                f"It also produced {low['conversation_causes']['recovery_end_session']['count']} recovery-terminal "
                "conversations, while `none` produced none. This localizes the low-effort completion collapse "
                "to generated closing behavior around required tool/recovery boundaries rather than the intended turn-29 close."
            ),
            "",
            (
                "Missing future turns remain fixed-denominator benchmark failures, but this cause table should be "
                "used whenever those failures are attributed to model behavior versus serving. Synthetic recovery "
                "turn 30 is not a scored scheduled turn."
            ),
            "",
            "## Reproducibility",
            "",
            (
                "`FAILURE-ANALYSIS.json` records the SHA-256 of `canonical.tsv`, this analyzer, and every "
                "included transcript and run log. All paths are repository-relative."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_outputs(result: dict[str, Any]) -> None:
    atomic_write(JSON_OUTPUT, json.dumps(result, indent=2) + "\n")
    atomic_write(MARKDOWN_OUTPUT, render_markdown(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="atomically write FAILURE-ANALYSIS.json and FAILURE-ANALYSIS.md",
    )
    args = parser.parse_args()
    result = analyze()
    summary = {
        "status": "written" if args.write else "read_only",
        "membership_source": result["method"]["membership_source"],
        "judge_dependency": result["method"]["judge_dependency"],
        "arms": {
            arm: {
                "conversation_causes": result["arms"][arm]["conversation_causes"],
                "missing_turns_by_cause": result["arms"][arm]["missing_turns_by_cause"],
            }
            for arm in ARMS
        },
    }
    if args.write:
        write_outputs(result)
        summary["outputs"] = [relative(JSON_OUTPUT), relative(MARKDOWN_OUTPUT)]
    else:
        summary["note"] = "No files were written. Pass --write to emit the two additive artifacts."
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
