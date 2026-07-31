#!/usr/bin/env python3
"""Verify final Laguna aggregate and its frozen 60-run source pool."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STATE = HERE / "state"
MARKDOWN = ROOT / "docs/filler-token-latent-scratchpad-study.md"
HTML = ROOT / "docs/filler-token-latent-scratchpad-study.html"
MODEL = "laguna_s21"
REQUESTED_MODEL = "poolside/laguna-s-2.1"
N_TURNS = 30
EXPECTED_N = {"nofiller": 30, "dots96": 30}
SCHEDULES = (
    (
        HERE / "schedule.tsv",
        "ece7b3e83708f018627c78343c74db97642683f1adc77a4d77526ce80970886e",
        {"nofiller": 10, "dots96": 6},
    ),
    (
        HERE / "schedule-dots-topup.tsv",
        "6521d0be0ab91bc3f64a631b4635e17de2e38dcfcec536cccb1a50aab0da6491",
        {"nofiller": 0, "dots96": 4},
    ),
    (
        HERE / "schedule-n30-topup.tsv",
        "7ea9b6e3dfc53d104aca9d91eafdb4487623a8862a62c2aca3ae78b836d259e7",
        {"nofiller": 20, "dots96": 20},
    ),
)
RUN_SENTINELS = (
    STATE / "RUNS_COMPLETE",
    STATE / "dots-topup" / "RUNS_COMPLETE",
    STATE / "n30-topup" / "RUNS_COMPLETE",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-9)


def finite_interval(value: object, low: float, high: float) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
        and low <= value[0] <= value[1] <= high
    )


def source_metrics(run_dir: Path, arm: str) -> dict[str, object]:
    transcript = run_dir / "transcript.jsonl"
    judged = run_dir / "claude_judged.jsonl"
    summary_path = run_dir / "claude_summary.json"
    run_log = run_dir / "run.log"
    for path in (transcript, judged, summary_path, run_log):
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing source artifact: {path}")

    log = run_log.read_text()
    for signature in (
        "Using OpenRouter with base_url=https://openrouter.ai/api/v1, reasoning_off=True, max_tokens=8192",
        "Recovery nudges enabled=True",
        "Tool call dedupe enabled=True",
        "Tool result run_llm enabled=False",
        "Text pipeline idle_timeout_secs=45.0",
    ):
        if signature not in log:
            raise ValueError(f"runtime signature {signature!r} missing: {run_dir}")
    filler = "MTE_FILLER_DOTS active: 96 x '.' filler tokens, position=suffix"
    if (arm == "dots96" and log.count(filler) != 1) or (
        arm == "nofiller" and "MTE_FILLER_DOTS active:" in log
    ):
        raise ValueError(f"filler signature mismatch: {run_dir}")

    raw: dict[int, dict] = {}
    end_session_turns: set[int] = set()
    thinking_tokens = 0
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        if row.get("model_name") != REQUESTED_MODEL:
            raise ValueError(f"model mismatch: {run_dir}")
        thinking_tokens += int((row.get("tokens") or {}).get("thinking_tokens") or 0)
        if any(
            call.get("name") == "end_session" for call in row.get("tool_calls") or []
        ):
            end_session_turns.add(int(row.get("turn", -1)))
        turn = row.get("turn")
        if (
            isinstance(turn, int)
            and 0 <= turn < N_TURNS
            and row.get("recovery_turn") is not True
        ):
            if turn in raw:
                raise ValueError(f"duplicate scripted turn {turn}: {run_dir}")
            raw[turn] = row
    if not raw or thinking_tokens:
        raise ValueError(f"invalid reasoning-off transcript: {run_dir}")

    final: dict[int, dict] = {}
    for line in judged.read_text().splitlines():
        row = json.loads(line)
        turn = row.get("turn")
        if not isinstance(turn, int) or turn in final:
            raise ValueError(f"invalid or duplicate judgment turn: {run_dir}")
        final[turn] = row
    if set(final) != set(raw):
        raise ValueError(f"judgment coverage mismatch: {run_dir}")
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("turns_scored") != len(raw)
        or not summary.get("judge_model")
        or not summary.get("judge_version")
    ):
        raise ValueError(f"judge summary mismatch: {run_dir}")

    pass_counts = {key: 0 for key in ("pass", "tool", "instruction", "kb")}
    latencies: list[float] = []
    for turn in range(N_TURNS):
        scores = (final.get(turn, {}).get("scores") or {})
        values = {
            "tool": scores.get("tool_use_correct") is True,
            "instruction": scores.get("instruction_following") is True,
            "kb": scores.get("kb_grounding") is True,
        }
        if turn in final and not all(
            isinstance(scores.get(key), bool)
            for key in ("tool_use_correct", "instruction_following", "kb_grounding")
        ):
            raise ValueError(f"non-boolean required score: {run_dir}")
        for key, passed in values.items():
            pass_counts[key] += int(passed)
        pass_counts["pass"] += int(all(values.values()))
        latency = raw.get(turn, {}).get("ttfb_ms")
        if isinstance(latency, (int, float)) and math.isfinite(latency) and latency >= 0:
            latencies.append(float(latency))
    return {
        "pass_counts": pass_counts,
        "strict_complete": end_session_turns == {29},
        "latencies": latencies,
    }


def main() -> None:
    for sentinel in (*RUN_SENTINELS, STATE / "N30_JUDGING_COMPLETE"):
        if not sentinel.is_file():
            raise ValueError(f"completion sentinel absent: {sentinel.relative_to(HERE)}")

    schedule_rows: list[dict[str, str]] = []
    stage_slots: list[set[str]] = []
    for path, expected_hash, expected_counts in SCHEDULES:
        if digest(path) != expected_hash:
            raise ValueError(f"frozen schedule changed: {path.name}")
        rows = read_tsv(path)
        counts = {
            arm: sum(row.get("arm") == arm for row in rows) for arm in EXPECTED_N
        }
        if counts != expected_counts:
            raise ValueError(f"stage counts changed in {path.name}: {counts}")
        schedule_rows.extend(rows)
        stage_slots.append({row["slot"] for row in rows})
    schedule = {row["slot"]: row for row in schedule_rows}
    if len(schedule_rows) != 60 or len(schedule) != 60:
        raise ValueError("schedule union is not exactly 60 unique assignments")
    if any(
        row.get("model") != MODEL
        or row.get("requested_model") != REQUESTED_MODEL
        or row.get("service") != "openrouter"
        or row.get("arm") not in EXPECTED_N
        for row in schedule_rows
    ):
        raise ValueError("schedule policy mismatch")

    for lane, expected_slots in (
        ("dots-topup", stage_slots[1]),
        ("n30-topup", stage_slots[2]),
    ):
        rows = read_tsv(STATE / lane / "manifest.tsv")
        if {row["slot"] for row in rows} != expected_slots or len(rows) != len(
            expected_slots
        ):
            raise ValueError(f"{lane} manifest differs from its frozen schedule")

    manifest = read_tsv(STATE / "manifest.tsv")
    if len(manifest) != 60:
        raise ValueError("master manifest does not contain exactly 60 rows")
    seen_slots: set[str] = set()
    seen_dirs: set[Path] = set()
    source_by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    aggregate_dirs_by_arm: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        slot = row.get("slot", "")
        if slot in seen_slots or slot not in schedule:
            raise ValueError(f"duplicate or unexpected master-manifest slot: {slot!r}")
        assignment = schedule[slot]
        if (
            row.get("model") != MODEL
            or row.get("arm") != assignment["arm"]
            or row.get("classification")
            not in {"strict_complete", "model_abort", "incomplete_no_end_session"}
        ):
            raise ValueError(f"master-manifest policy mismatch in {slot}")
        raw_path = Path(row["run_dir"])
        run_dir = (raw_path if raw_path.is_absolute() else ROOT / raw_path).resolve()
        try:
            relative = run_dir.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"run outside repository: {run_dir}") from exc
        if run_dir in seen_dirs:
            raise ValueError(f"duplicate source run: {run_dir}")
        seen_slots.add(slot)
        seen_dirs.add(run_dir)
        aggregate_dirs_by_arm[row["arm"]].add(str(relative))
        source_by_arm[row["arm"]].append(source_metrics(run_dir, row["arm"]))
    if seen_slots != set(schedule):
        raise ValueError("master manifest is not the exact schedule union")

    payload = json.loads((HERE / "aggregates.json").read_text())
    protocol = payload.get("protocol", {})
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_status") != "FINAL"
        or protocol.get("benchmark") != "aiwf_medium_context"
        or protocol.get("turns") != 30
        or protocol.get("target_per_arm") != 30
        or protocol.get("missing_scripted_turns") != "fail"
        or protocol.get("thinking_mode") != "disabled"
        or protocol.get("full_thinking_off_guaranteed") is not True
        or protocol.get("route") != "OpenRouter paid Poolside-hosted BF16"
        or protocol.get("filler")
        != {"arm": "dots96", "glyph": ".", "count": 96, "position": "suffix"}
        or protocol.get("bootstrap_unit") != "whole conversation"
        or protocol.get("bootstrap_samples") != 100_000
        or protocol.get("seed") != 20260722
    ):
        raise ValueError("final aggregate protocol mismatch")
    model = payload.get("models", {}).get(MODEL, {})
    if (
        set(payload.get("models", {})) != {MODEL}
        or model.get("requested_model") != REQUESTED_MODEL
        or model.get("provider") != "OpenRouter"
        or model.get("endpoint_provider") != "Poolside"
        or model.get("quantization") != "BF16"
        or model.get("report_tier") != "focused"
    ):
        raise ValueError("model provenance or report tier mismatch")

    arms = model.get("arms", {})
    for arm, expected_n in EXPECTED_N.items():
        result = arms.get(arm, {})
        denominator = expected_n * N_TURNS
        if (
            result.get("n_attempts") != expected_n
            or result.get("fixed_turn_denominator") != denominator
            or result.get("thinking_tokens") != 0
            or set(result.get("run_dirs", [])) != aggregate_dirs_by_arm[arm]
            or len(result.get("run_dirs", [])) != expected_n
        ):
            raise ValueError(f"aggregate source pool mismatch for {arm}")
        source = source_by_arm[arm]
        pass_counts = {
            key: sum(item["pass_counts"][key] for item in source)
            for key in ("pass", "tool", "instruction", "kb")
        }
        for key in ("pass", "tool", "instruction", "kb"):
            pass_key = "pass_count" if key == "pass" else f"{key}_pass_count"
            error_key = "any_error_count" if key == "pass" else f"{key}_error_count"
            rate_key = "pass_rate_pct" if key == "pass" else f"{key}_pass_rate_pct"
            error_rate_key = (
                "any_error_rate_pct" if key == "pass" else f"{key}_error_rate_pct"
            )
            if (
                result.get(pass_key) != pass_counts[key]
                or result.get(error_key) != denominator - pass_counts[key]
                or not close(result[rate_key], 100 * pass_counts[key] / denominator)
                or not close(
                    result[error_rate_key],
                    100 * (denominator - pass_counts[key]) / denominator,
                )
            ):
                raise ValueError(f"fixed-denominator {key} mismatch for {arm}")
            ci_key = "pass_rate_ci95" if key == "pass" else f"{key}_error_rate_ci95"
            if not finite_interval(result.get(ci_key), 0, 100):
                raise ValueError(f"invalid {ci_key} for {arm}")
        if not finite_interval(result.get("any_error_rate_ci95"), 0, 100):
            raise ValueError(f"invalid any-error interval for {arm}")

        strict_count = sum(bool(item["strict_complete"]) for item in source)
        if (
            result.get("strict_complete_count") != strict_count
            or not close(result["strict_completion_pct"], 100 * strict_count / expected_n)
            or not finite_interval(result.get("strict_completion_ci95"), 0, 100)
        ):
            raise ValueError(f"strict-completion mismatch for {arm}")
        latencies = [value for item in source for value in item["latencies"]]
        if (
            not latencies
            or result.get("ttfat_observations") != len(latencies)
            or not close(result["ttfat_p50_ms"], statistics.median(latencies))
            or not close(result["ttfat_p95_ms"], float(np.percentile(latencies, 95)))
            or not close(result["ttfat_max_ms"], max(latencies))
            or not (
                0
                <= result["ttfat_p50_ms"]
                <= result["ttfat_p95_ms"]
                <= result["ttfat_max_ms"]
            )
        ):
            raise ValueError(f"TTFAT mismatch for {arm}")

    control = arms["nofiller"]
    dots = arms["dots96"]
    effect = model.get("effect", {})
    for key in ("pass", "tool", "instruction", "kb"):
        rate_key = "pass_rate_pct" if key == "pass" else f"{key}_pass_rate_pct"
        delta_key = f"{key}_delta_points"
        ci_key = f"{key}_delta_ci95"
        if (
            not close(effect[delta_key], dots[rate_key] - control[rate_key])
            or not finite_interval(effect.get(ci_key), -100, 100)
        ):
            raise ValueError(f"effect mismatch for {key}")

    decision = model.get("adaptive_decision", {})
    promotion = decision.get("frozen_n10_promotion_decision", {})
    if (
        decision.get("stage") != "focused_n30"
        or decision.get("observed_control_n") != 30
        or decision.get("observed_dots_n") != 30
        or decision.get("action") != "focused_followup_complete"
        or decision.get("final_n") != 30
        or decision.get("adaptive_expansion_completed") is not True
        or decision.get("no_further_sample_size_decision") is not True
        or decision.get("decision_pending") is not False
        or not close(decision["pass_delta_points"], effect["pass_delta_points"])
        or decision.get("pass_delta_ci95") != effect.get("pass_delta_ci95")
        or decision.get("strict_completion_pct")
        != [control["strict_completion_pct"], dots["strict_completion_pct"]]
        or promotion.get("stage") != "dots_n10"
        or promotion.get("observed_control_n") != 10
        or promotion.get("observed_dots_n") != 10
        or promotion.get("action") != "promote_both_arms_to_30"
        or promotion.get("final_n") != 30
        or promotion.get("decision_pending") is not True
        or not promotion.get("triggers")
        or not finite_interval(promotion.get("pass_delta_ci95"), -100, 100)
    ):
        raise ValueError("adaptive-expansion provenance mismatch")

    markdown = MARKDOWN.read_text()
    if "**Scope:** Twenty-four standard filler comparisons" not in markdown:
        raise ValueError("Markdown report scope was not updated to 24 models")
    primary = markdown.split("<!-- N30_PRIMARY_START -->", 1)[1].split(
        "<!-- N30_PRIMARY_END -->", 1
    )[0]
    table_lines = [line for line in primary.splitlines() if line.startswith("|")]
    if len(table_lines) != 26:
        raise ValueError(
            f"Markdown primary table should have 24 model rows, found {len(table_lines) - 2}"
        )
    expected_markdown_row = (
        "| laguna-s-2.1 | OpenRouter | 85.6 | 83.3 | "
        "−2.2 [−8.3, +5.1] | 13% → 13% | 295 | 30 / 30 | uncertain |"
    )
    if primary.count("| laguna-s-2.1 |") != 1 or expected_markdown_row not in primary:
        raise ValueError("Markdown Laguna S 2.1 row mismatch")
    for required in (
        "paid OpenRouter route to Poolside-hosted BF16 weights",
        "both arms use `reasoning.enabled=false`",
        "separate 30/30 campaign aggregate",
    ):
        if required not in primary:
            raise ValueError(f"Markdown Laguna provenance is missing: {required}")

    html = HTML.read_text()
    if (
        "a 24-Model Exploratory Study" not in html
        or "Twenty-four-model exploratory screen" not in html
    ):
        raise ValueError("HTML report model count was not updated to 24")
    section = html.split('<section id="primary-screen">', 1)[1].split(
        "</section>", 1
    )[0]
    if section.count("<tr><td>") != 24:
        raise ValueError(
            f"HTML primary table should have 24 rows, found {section.count('<tr><td>')}"
        )
    expected_html_row = (
        '<tr><td>laguna-s-2.1</td><td class="mut">OpenRouter</td>'
        '<td class="r">85.6</td><td class="r">83.3</td>'
        '<td class="r em">−2.2 <span class="mut">[−8.3, +5.1]</span></td>'
        '<td class="r mut">13% → 13%</td><td class="r mut">295</td>'
        '<td class="r mut">30 / 30</td>'
        '<td><span class="chip chip-null">uncertain</span></td></tr>'
    )
    if section.count("<tr><td>laguna-s-2.1</td>") != 1 or expected_html_row not in section:
        raise ValueError("HTML Laguna S 2.1 table row mismatch")
    figure = section.split("<figure>", 1)[1].split("</figure>", 1)[0]
    laguna_figure = figure.split('class="lbl">laguna-s-2.1</text>', 1)[1].split(
        'class="lbl">qwen3.6-27b</text>', 1
    )[0]
    if (
        laguna_figure.count("<line ") != 4
        or 'stroke="var(--nul)"' not in laguna_figure
        or 'opacity="0.45"' not in laguna_figure
        or '<tspan class="provider"> · OpenRouter</tspan>' not in laguna_figure
        or '−2.2<tspan class="pval"> · 295 ms</tspan>' not in laguna_figure
    ):
        raise ValueError("HTML Laguna chart row is missing its focused whisker or labels")
    for required in (
        "paid OpenRouter route to Poolside-hosted BF16 weights",
        "<code>reasoning.enabled=false</code>",
        "separate 30/30 campaign",
    ):
        if required not in section:
            raise ValueError(f"HTML Laguna provenance is missing: {required}")
    mechanism = html.split('<section id="mechanism">', 1)[1].split("</section>", 1)[0]
    if (
        "laguna-s-2.1" in mechanism
        or mechanism.count('class="turn-cell') != 660
        or mechanism.count('class="family-contribution-cell') != 55
        or mechanism.count('class="family-contribution-total') != 11
    ):
        raise ValueError("frozen 11-model mechanism cohort changed")
    print("Laguna final 30/30 aggregate, source pool, and 24-row report verified")


if __name__ == "__main__":
    main()
