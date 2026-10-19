"""Recalculate selected-cohort metrics; final aggregation requires an explicit disposition for every input."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

import campaign

HERE, ROOT = campaign.HERE, campaign.ROOT
DIMENSIONS = ("tool_use_correct", "instruction_following", "kb_grounding")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_archives():
    restored = json.loads((HERE / "restored-sources.json").read_text())
    for archive in restored["archives"]:
        path = Path(archive["path"])
        assert path.stat().st_size == archive["size_bytes"], path
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == archive["sha256"], path
    for member in restored["members"]:
        assert digest((ROOT / member["restored_path"]).read_bytes()) == member["sha256"]


def read_input(item):
    path = ROOT / item["input"]
    raw = path.read_bytes()
    assert digest(raw) == item["transcript_sha256"], path
    assert Path(item["source"]).read_bytes() == raw, item["source"]
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def statistics_ms(values):
    if not values:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p50_ms": None,
                "p95_ms": None, "min_ms": None, "max_ms": None}
    return {
        "n": len(values), "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values), "p50_ms": statistics.median(values),
        "p95_ms": float(np.percentile(values, 95, method="linear")),
        "min_ms": min(values), "max_ms": max(values),
    }


def text_latency(cohort):
    positive, signed, observations = [], [], []
    excluded_first, missing, nonpositive = 0, 0, 0
    for item in cohort["inputs"]:
        rows = read_input(item)
        for row in rows:
            turn, recovery = row.get("turn"), row.get("recovery_turn") is True
            if turn == 0 and not recovery:
                excluded_first += 1
                continue
            value = row.get("ttfb_ms")
            observations.append({"capture_id": item["capture_id"], "turn": turn,
                                 "recovery": recovery, "value_ms": value})
            if finite_number(value):
                signed.append(value)
                if value > 0:
                    positive.append(value)
                else:
                    nonpositive += 1
            else:
                missing += 1
    return {
        "definition": "Recorded service TTFAT/TTFB, scripted index zero excluded, recovery responses included; positive finite measurements only; linear P95",
        "positive": statistics_ms(positive), "signed": statistics_ms(signed),
        "missing_n": missing, "nonpositive_n": nonpositive,
        "excluded_first_scripted_n": excluded_first,
        "eligible_response_n": len(observations),
    }, observations


def voice_latency(cohort):
    positive, signed, tool, non_tool, silence = [], [], [], [], []
    observations = []
    nonpositive = 0
    expected = 30 * cohort["actual_n"]
    for item in cohort["inputs"]:
        read_input(item)
        path = ROOT / item["latency_metrics_snapshot"]
        raw = path.read_bytes()
        assert digest(raw) == item["latency_metrics_sha256"], path
        assert Path(item["latency_metrics_source"]).read_bytes() == raw
        assert digest(Path(item["original_quality_input_source"]).read_bytes()) == item["original_quality_input_sha256"]
        data = json.loads(raw)
        turns = data["turns"]
        turns = list(turns.values()) if isinstance(turns, dict) else turns
        seen = set()
        for row in turns:
            turn = row.get("turn", row.get("turn_index"))
            assert type(turn) is int and 0 <= turn < 30 and turn not in seen, path
            seen.add(turn)
            value = row.get("wav_v2v_ms")
            observations.append({"capture_id": item["capture_id"], "turn": turn,
                                 "has_tool_call": bool(row.get("has_tool_call")), "value_ms": value,
                                 "silence_ms": row.get("silent_pad_silero_ms")})
            if finite_number(value):
                signed.append(value)
                if value > 0:
                    positive.append(value)
                    (tool if row.get("has_tool_call") else non_tool).append(value)
                else:
                    nonpositive += 1
            pad = row.get("silent_pad_silero_ms")
            if finite_number(pad) and pad >= 0:
                silence.append(pad)
    missing = expected - len(signed)
    assert missing >= 0
    return {
        "definition": "All 30 scripted turns; actual first recorded speech including acknowledgments; positive finite V2V; original Silero detector/alignment; no zero imputation",
        "first_speech_positive": statistics_ms(positive),
        "first_speech_signed": statistics_ms(signed),
        "tool": statistics_ms(tool), "non_tool": statistics_ms(non_tool),
        "silence": statistics_ms(silence), "nonpositive_n": nonpositive,
        "missing_latency_n": missing,
    }, observations


def verify_voice_latency_against_historical(cohort, calculated):
    baseline = json.loads((ROOT / "docs/speech-results-2026-09/aggregate.json").read_text())
    key = cohort["key"].removeprefix("voice-")
    old = next(model for model in baseline["models"] if model["key"] == key)
    for group in ("first_speech_positive", "first_speech_signed", "tool", "non_tool", "silence"):
        for field in ("n", "mean_ms", "median_ms", "max_ms"):
            before, after = old[group][field], calculated[group][field]
            assert (before is None and after is None) or math.isclose(before, after, rel_tol=1e-10, abs_tol=1e-8), (key, group, field, before, after)
    assert old["nonpositive_n"] == calculated["nonpositive_n"]
    assert old["missing_latency_n"] == calculated["missing_latency_n"]


def confidence_interval(key, conversation_passes):
    scores = np.asarray(conversation_passes, dtype=float) / 30 * 100
    rng = np.random.default_rng(20261017 + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big"))
    samples = rng.choice(scores, size=(20000, len(scores)), replace=True).mean(axis=1)
    return [float(v) for v in np.percentile(samples, [2.5, 97.5])]


def quality(cohort):
    counts = Counter()
    passes_per_conversation = []
    cost, durations, usage = 0.0, [], Counter()
    unknown_cost = 0
    blocked = []
    audio_passes = 0
    for item in cohort["inputs"]:
        if cohort["mode"] == "voice" and item["turn_taking_scored"]:
            audio_passes += sum(v["turn_taking"] for v in item["turn_taking_data"].values())
        if campaign.validate_blocked(cohort, item):
            blocked.append(item["job_id"])
            continue
        assert campaign.validate_existing(cohort, item), item["job_id"]
        result = json.loads(campaign.result_path(item).read_text())
        if cohort["mode"] == "voice" and not item["turn_taking_scored"]:
            campaign.mark_turn_taking_unscored(ROOT / result["raw_output"], result)
        meta = result["metadata"]
        score = 0
        for turn in range(30):
            judgment = result["judgments"][str(turn)]
            values = judgment["scores"]
            assert all(type(values[dim]) is bool for dim in DIMENSIONS)
            for dimension in DIMENSIONS:
                counts[dimension] += values[dimension]
            strict = all(values[dimension] for dimension in DIMENSIONS)
            counts["strict"] += strict
            score += strict
            if cohort["mode"] == "voice":
                if item["turn_taking_scored"]:
                    expected = item["turn_taking_data"][str(turn)]["turn_taking"]
                    assert values["turn_taking"] is expected
                    # Audio coverage is independent of quality-judge availability.
                else:
                    assert values["turn_taking"] is None
        passes_per_conversation.append(score)
        if finite_number(meta.get("judge_cost_usd")):
            cost += meta["judge_cost_usd"]
        else:
            unknown_cost += 1
        if finite_number(meta.get("judge_duration_ms")):
            durations.append(meta["judge_duration_ms"])
        for key, value in (meta.get("judge_usage") or {}).items():
            if type(value) is int:
                usage[key] += value
    total = 30 * cohort["actual_n"]
    counts = {dim: counts[dim] for dim in (*DIMENSIONS, "strict")}
    if cohort["mode"] == "voice":
        counts["turn_taking"] = audio_passes
    if cohort["mode"] == "voice" and not cohort["inputs"][0]["turn_taking_scored"]:
        counts["turn_taking"] = None
    return {
        "scores": counts, "accuracy_denominator": total,
        "quality_complete": not blocked,
        "graded_conversations": cohort["actual_n"] - len(blocked),
        "provider_blocked_conversations": len(blocked), "provider_blocked_jobs": blocked,
        "graded_turns": total - 30 * len(blocked), "unknown_turns": 30 * len(blocked),
        "strict_percent": None if blocked else 100 * counts["strict"] / total,
        "strict_bounds_percent": [100 * counts["strict"] / total,
                                   100 * (counts["strict"] + 30 * len(blocked)) / total],
        "strict_ci95": None if blocked else confidence_interval(cohort["key"], passes_per_conversation),
        "dimension_error_percent": {dim: None if blocked else 100 * (total - counts[dim]) / total for dim in DIMENSIONS},
        "conversation_strict_passes": passes_per_conversation,
        "successful_judge_cost_usd": cost, "unknown_cost_jobs": unknown_cost,
        "judge_duration_ms": statistics_ms(durations), "judge_usage": dict(usage),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    cohorts = campaign.cohorts()
    assert Counter(c["mode"] for c in cohorts) == {"text": 43, "voice": 9}
    assert sum(c["actual_n"] for c in cohorts) == 1316
    if args.final:
        assert all(campaign.resolved(c, item) for c in cohorts for item in c["inputs"])
        verify_archives()
    text, voice, observations = [], [], {}
    for cohort in cohorts:
        common = {
            "key": cohort["key"], "original_label": cohort["label"],
            "label": cohort.get("display_label", cohort["label"]), "mode": cohort["mode"],
            "complete": cohort["actual_n"], "target": cohort["target"], "shortfall": cohort["shortfall"],
            "manifest": f"cohorts/{cohort['key']}.json", "provider": cohort.get("provider"),
        }
        if cohort["mode"] == "text":
            latency, raw = text_latency(cohort)
            common["latency"] = latency
            common["available_complete"] = cohort["candidate_audit"]["eligible_complete"]
            common["configuration"] = cohort.get("matching_rule", cohort.get("configuration"))
            text.append(common)
        else:
            latency, raw = voice_latency(cohort)
            verify_voice_latency_against_historical(cohort, latency)
            common.update(latency)
            common.update(cohort["failure_accounting"])
            common["key"] = cohort["key"].removeprefix("voice-")
            common["cohort_key"] = cohort["key"]
            common["turn_taking_scored"] = cohort["inputs"][0]["turn_taking_scored"]
            common["additional_experiment"] = cohort.get("additional_experiment", False)
            common["configuration"] = cohort["configuration"]
            voice.append(common)
        observations[cohort["key"]] = raw
        if args.final:
            common.update(quality(cohort))
    policy = {
        "judge_model": campaign.MODEL, "judge_effort": campaign.EFFORT,
        "rubric_sha256": {mode: digest(text.encode()) for mode, text in campaign.RUBRICS.items()},
        "text_target_n": 30, "voice_target_n": 25,
        "accuracy": "conditional on complete conversations; no incomplete prefixes or unobserved future turns included",
        "strict": "tool_use_correct AND instruction_following AND kb_grounding; turn-taking reported separately",
        "confidence_interval": "20000 whole-conversation bootstrap resamples; excludes judge uncertainty and differences in capture date/SDK/serving",
        "text_latency": "first scripted response excluded; recorded recovery responses included; positive finite service timings; linear P95",
        "voice_latency": "all turns; positive measurable actual first speech; signed and missing diagnostics preserved",
        "source_preservation": "selected original sources, immutable input snapshots, original voice quality records, and original per-turn audio JSON verified byte-for-byte",
        "provider_rejections": "explicit unscored dispositions; any incomplete quality cohort has null headline accuracy/error rates/CIs; source membership and latency retained",
        "cost": "SDK-reported successful-request costs; operational attempts without usage metadata are not assigned an invented cost",
    }
    output = {
        "schema_version": 1, "as_of": datetime.now(timezone.utc).isoformat(),
        "complete": args.final and all(m["quality_complete"] for m in text + voice),
        "processing_complete": args.final, "policy": policy,
        "complete_means": "every selected conversation has a valid quality grade; processing_complete tracks terminal dispositions separately",
        "all_quality_complete": args.final and all(m["quality_complete"] for m in text + voice),
        "text_models": text, "voice_models": voice,
        "counts": {"text_configurations": 43, "text_conversations": 1091,
                   "voice_configurations": 9, "voice_conversations": 225, "scripted_turns": 39480,
                   **({"graded_conversations": sum(m["graded_conversations"] for m in text + voice),
                       "provider_blocked_conversations": sum(m["provider_blocked_conversations"] for m in text + voice)} if args.final else {})},
    }
    if args.final:
        superseded = [json.loads(path.read_text()) for path in
                      (HERE / "reasoning-cleanup" / "prior-results").glob("*.json")]
        published_cost = sum(m["successful_judge_cost_usd"] for m in text + voice)
        prior_cost = sum(r["metadata"].get("judge_cost_usd", 0) or 0 for r in superseded)
        output["cost_accounting"] = {
            "published_grade_cost_usd": published_cost,
            "superseded_grade_cost_usd": prior_cost,
            "total_successful_request_cost_usd": published_cost + prior_cost,
            "superseded_successful_requests": len(superseded),
            "superseded_unknown_cost_jobs": sum(not finite_number(r["metadata"].get("judge_cost_usd")) for r in superseded),
            "superseded_text_cost_usd": prior_cost,
            "definition": "All successful requests including archived grades replaced after reasoning cleanup; failed requests without usage remain unestimated",
        }
        output["policy"]["judge_input_normalization"] = campaign.REASONING_NORMALIZATION
    campaign.dump(HERE / "latency-observations.json", observations)
    campaign.dump(HERE / ("aggregate.json" if args.final else "latency-recalculation.json"), output)
    print(json.dumps({"final": args.final, "text": len(text), "voice": len(voice),
                      "text_latency_observations": sum(r["latency"]["positive"]["n"] for r in text),
                      "voice_latency_observations": sum(r["first_speech_positive"]["n"] for r in voice)}, indent=2))


if __name__ == "__main__":
    main()
