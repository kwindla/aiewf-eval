"""Rebuild the calibration report and matched Gemini comparison from saved grades."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DIMS = ("tool_use_correct", "instruction_following", "kb_grounding")


def read(path):
    return json.loads(path.read_text())


def build_calibration():
    configuration = read(HERE / "configuration.json")
    summary = read(HERE / "calibration-summary.json")
    results = [read(p) for p in sorted((HERE / "results").glob("*.json"))]
    assert len(results) == 32
    for case in ("A", "B", "C", "D"):
        assert len({r["metadata"]["judge_input_sha256"] for r in results if r["job"]["case"] == case}) == 1
    lines = [
        "# Opus 5.5 medium text judge calibration", "",
        "The protocol and intended labels were frozen before any calibration grade returned. "
        "Four blind 30-turn conversations were graded twice per model and rubric. "
        "There are 24 labeled target turns, with only unambiguous dimensions labeled. "
        "These deliberately selected cases diagnose the disputed behaviors; they do not "
        "estimate general judge accuracy. The archived judge effort was not recorded; "
        "the legacy-rubric arm controls effort at medium for both models rather than "
        "recreating an undocumented historical default.", "",
        "| Rubric | Judge (medium) | Labeled dimension agreement | Clear-failure agreement |",
        "|---|---|---:|---:|",
    ]
    for key, value in summary.items():
        rubric, model = key.split("/")
        lines.append(f"| {rubric} | {model} | {value['agreement']}/{value['total']} | "
                     f"{value['negative_agreement']}/{value['negative_total']} |")
    lines += ["", "## Repetitions and disagreements", ""]
    for rubric in ("legacy", "candidate"):
        for model in ("claude-opus-5", "claude-opus-5-5"):
            matched = [r for r in results if r["job"]["rubric"] == rubric and r["job"]["model"] == model]
            changed = []
            for case in ("A", "B", "C", "D"):
                pair = sorted([r for r in matched if r["job"]["case"] == case], key=lambda r: r["job"]["repeat"])
                for turn in range(30):
                    for dim in DIMS:
                        if pair[0]["judgments"][str(turn)]["scores"][dim] != pair[1]["judgments"][str(turn)]["scores"][dim]:
                            changed.append(f"{case}/{turn}/{dim}")
            lines.append(f"- {rubric}, {model}: {len(changed)}/360 full-conversation dimension scores "
                         f"differ between repetitions. " + (", ".join(changed) if changed else "All agree."))
    lines += ["", "Every labeled disagreement follows (case / zero-based turn / repetition).", ""]
    for key, value in summary.items():
        for row in value["disagreements"]:
            lines.append(f"- **{key}, {row['case']}/{row['turn']}/r{row['repeat']}, {row['dimension']}**: "
                         f"expected {row['expected']}, got {row['actual']}. {row['reasoning']}")
    lines += ["", "## Full conversation scores", "",
              "These totals include unlabeled turns and are reported without selecting a favorable repetition.", "",
              "| Rubric | Judge | Case | Repetition | Strict passes / 30 |", "|---|---|---|---:|---:|"]
    for result in sorted(results, key=lambda r: (r['job']['rubric'], r['job']['model'], r['job']['case'], r['job']['repeat'])):
        job = result["job"]
        passes = sum(all(j["scores"][d] for d in DIMS) for j in result["judgments"].values())
        lines.append(f"| {job['rubric']} | {job['model']} | {job['case']} | {job['repeat']} | {passes}/30 |")
    costs = sum(r["metadata"].get("judge_cost_usd") or 0 for r in results)
    lines += ["", "## Judge runtime and cost", "",
              "| Rubric | Judge | Median SDK duration | Cost for 8 calls |",
              "|---|---|---:|---:|"]
    for rubric in ("legacy", "candidate"):
        for model in ("claude-opus-5", "claude-opus-5-5"):
            selected = [r for r in results if r["job"]["rubric"] == rubric and r["job"]["model"] == model]
            seconds = np.median([r['metadata']['judge_duration_ms'] / 1000 for r in selected])
            cost = sum(r['metadata'].get('judge_cost_usd') or 0 for r in selected)
            lines.append(f"| {rubric} | {model} | {seconds:.1f}s | ${cost:.4f} |")
    lines += ["", "## Configuration and operational notes", "",
              f"Successful calibration calls cost ${costs:.4f} according to SDK usage metadata. "
              "The first two requests were rejected before grading because SDK 0.1.73 bundled "
              "Claude Code 2.1.128; Opus 5.5 requires at least 2.1.280. The SDK was pinned to "
              "0.2.168, bundling 2.1.301, before any successful grade. "
              "All errors are retained in `attempts.jsonl`; original attempt directories were preserved.", "",
              f"Candidate revision: `{configuration['candidate_revision']}`; system-prompt SHA256 "
              f"`{configuration['rubric_sha256']['candidate']}`. Explicit medium effort and adaptive "
              "thinking were used for both judges; tools and ambient setting sources were disabled. "
              "Actual response model IDs, effort, SDK version, user/system prompt hashes, and costs "
              "are saved with every result.", "",
              "The candidate follows the text assistant's actual governing policy. It accepts "
              "reasonable clarification after an attendance-date correction and policy-compliant "
              "venue refusals; it still rejects invented navigation, wrong dates/times, missed tools, "
              "and unsupported weather answers. Empty assistant text with a valid tool call remains "
              "accepted, matching the text capture harness. This is a rubric correction informed by "
              "the disputed examples, not an independent holdout evaluation. The speech rubric is unchanged.", ""]
    if (HERE / "adoption.json").exists():
        lines += ["## Decision", "", read(HERE / "adoption.json")["rationale"], ""]
    (HERE / "REPORT.md").write_text("\n".join(lines))


def build_comparison():
    manifest = read(HERE / "regrade-manifest.json")
    decision = read(HERE / "adoption.json")
    generation = read(HERE / "generation-inputs.json")
    for model, inputs in generation.items():
        archived = ROOT / f"ops/google-gemini{model.replace('.', '')}-flash/artifacts/source-sha256.txt"
        hashes = {name: value for value, name in (line.split(maxsplit=1) for line in archived.read_text().splitlines())}
        for relative, expected in inputs.items():
            assert hashes[relative] == expected
    per_model = {}
    failures = []
    for entry in manifest:
        source = Path(entry["source"])
        for filename, field in (("transcript.jsonl", "transcript_sha256"),
                                ("claude_summary.json", "original_summary_sha256"),
                                ("claude_judged.jsonl", "original_judged_sha256")):
            assert hashlib.sha256((source / filename).read_bytes()).hexdigest() == entry[field]
        result = read(ROOT / entry["output"])
        meta = result["metadata"]
        assert meta["judge_model"] == decision["model"]
        assert meta["judge_response_models"] == [decision["model"]]
        assert meta["judge_effort"] == decision["effort"]
        assert meta["judge_prompt_sha256"] == decision["rubric_sha256"]
        assert meta["judge_sdk_version"] == "0.2.168"
        assert meta["judge_tools"] == [] and meta["judge_setting_sources"] == []
        assert meta["judge_thinking"] == "adaptive"
        assert len(result["judgments"]) == 30
        old = {str(r["turn"]): r for r in map(json.loads, (source / "claude_judged.jsonl").read_text().splitlines())}
        bucket = per_model.setdefault(entry["source_model"], {"strict": [], "old_strict": [], "dimensions": dict.fromkeys(DIMS, 0),
                                      "changed_pass_to_fail": 0, "changed_fail_to_pass": 0, "cost": 0.0,
                                      "old_failures_by_turn": {}, "new_failures_by_turn": {}})
        passes = 0
        old_passes = 0
        for turn, judgment in result["judgments"].items():
            strict = all(judgment["scores"][d] for d in DIMS)
            old_strict = all(old[turn]["scores"][d] for d in DIMS)
            passes += strict
            old_passes += old_strict
            bucket["changed_pass_to_fail"] += old_strict and not strict
            bucket["changed_fail_to_pass"] += strict and not old_strict
            if not old_strict:
                bucket["old_failures_by_turn"][turn] = bucket["old_failures_by_turn"].get(turn, 0) + 1
            if not strict:
                bucket["new_failures_by_turn"][turn] = bucket["new_failures_by_turn"].get(turn, 0) + 1
            for dim in DIMS:
                bucket["dimensions"][dim] += judgment["scores"][dim]
            if not strict:
                failures.append({"model": entry["source_model"], "slot": entry["slot"], "turn": int(turn),
                                 "scores": judgment["scores"], "reasoning": judgment["reasoning"],
                                 "response": old[turn]["assistant_text"]})
        bucket["strict"].append(passes)
        bucket["old_strict"].append(old_passes)
        bucket["cost"] += meta.get("judge_cost_usd") or 0
    assert all(len(v["strict"]) == 30 for v in per_model.values())
    rng = np.random.default_rng(551017)
    boots = {}
    for model, bucket in per_model.items():
        sample = np.asarray(bucket["strict"]) / 30 * 100
        boot = sample[rng.integers(0, 30, size=(20000, 30))].mean(axis=1)
        boots[model] = boot
        bucket["strict_count"] = sum(bucket["strict"])
        bucket["strict_rate"] = bucket["strict_count"] / 900 * 100
        bucket["ci95"] = np.quantile(boot, [.025, .975]).tolist()
    difference = boots["gemini-3.8-flash"] - boots["gemini-3.7-flash"]
    comparison = {"judge": decision, "models": per_model,
                  "difference_38_minus_37_pp": per_model["gemini-3.8-flash"]["strict_rate"] - per_model["gemini-3.7-flash"]["strict_rate"],
                  "difference_ci95_pp": np.quantile(difference, [.025, .975]).tolist(), "failures": failures}
    notes = read(HERE / "review-notes.json") if (HERE / "review-notes.json").exists() else []
    comparison["review_notes"] = notes
    (HERE / "matched-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    lines = ["# Gemini Flash comparison with a matched judge", "",
             "All 60 original 30-turn transcripts were regraded once with **Opus 5.5 medium** "
             f"and the frozen `{decision['revision']}` text rubric. Original transcripts and judgments "
             "were preserved and their hashes verified after regrading. No Gemini generation was repeated.", "",
             "| Model (low) | Archived strict score | Matched strict score | Tool use | Instructions | KB grounding | 95% cluster interval |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for model, bucket in per_model.items():
        ci = bucket['ci95']
        dims = bucket['dimensions']
        lines.append(f"| {model} | {sum(bucket['old_strict'])}/900 | {bucket['strict_count']}/900 ({bucket['strict_rate']:.1f}%) | "
                     f"{dims['tool_use_correct']}/900 | {dims['instruction_following']}/900 | {dims['kb_grounding']}/900 | {ci[0]:.1f}–{ci[1]:.1f}% |")
    ci = comparison["difference_ci95_pp"]
    lines += ["", f"Gemini 3.8 minus 3.7: **{comparison['difference_38_minus_37_pp']:+.2f} percentage points**, "
              f"independent conversation-cluster bootstrap 95% interval {ci[0]:+.2f} to {ci[1]:+.2f} points "
              "(20,000 resamples, seed 551017). These intervals describe conversation variation conditional "
              "on one frozen judge pass; they do not include judge uncertainty. Both cohorts are complete "
              "and use a fixed 900-turn denominator.", "",
              "## Changes from archived grades", ""]
    for model, bucket in per_model.items():
        lines.append(f"- {model}: {bucket['changed_fail_to_pass']} fail→pass, {bucket['changed_pass_to_fail']} pass→fail.")
        old = ", ".join(f"turn {t}: {n}" for t, n in sorted(bucket['old_failures_by_turn'].items(), key=lambda x: int(x[0]))) or "none"
        new = ", ".join(f"turn {t}: {n}" for t, n in sorted(bucket['new_failures_by_turn'].items(), key=lambda x: int(x[0]))) or "none"
        lines.append(f"  Archived strict failures by zero-based turn: {old}. Matched failures: {new}.")
    lines += ["", "## Remaining strict failures", ""]
    if not failures:
        lines.append("None.")
    for failure in failures:
        bad = ", ".join(d for d in DIMS if not failure['scores'][d])
        lines.append(f"- **{failure['model']}, slot {failure['slot']}, turn {failure['turn']}** ({bad}): {failure['reasoning']}")
        quote = "\n".join("> " + line.rstrip() if line.strip() else ">" for line in failure['response'].strip().splitlines())
        lines += ["", quote, ""]
    if notes:
        lines += ["## Source audit of remaining flags", ""]
        for note in notes:
            assert any((f['model'], f['slot'], f['turn']) == (note['model'], note['slot'], note['turn']) for f in failures)
            lines.append(f"- **{note['model']}, slot {note['slot']}, turn {note['turn']}**: {note['assessment']}")
    costs = sum(v["cost"] for v in per_model.values())
    lines += ["", "## Interpretation and provenance", "",
              "The archived scores mixed Opus 4.5 (Gemini 3.7) and Opus 5 (Gemini 3.8). "
              "The new comparison fixes both judge model and rubric. The full knowledge base is now "
              "available to the judge, so factual alternatives and factual mistakes can be checked "
              "against the actual source. Acceptance of a reasonable clarification or policy-compliant "
              "refusal is deliberate; invented facts and missed authorized tools remain failures.", "",
              "Generation latency is unchanged: Gemini 3.7 measured 817ms P50 / 1260ms P95 warm TTFAT; "
              "Gemini 3.8 measured 1411ms / 1968ms. These campaigns ran on different dates and SDK versions, "
              "so this is not a controlled serving-speed experiment. The generation system prompt, "
              "knowledge base, scripted turns, and tool schemas are byte-identical between the two "
              "campaign worktrees and match both archived campaign source-hash manifests. "
              "Their hashes are recorded in `generation-inputs.json`.", "",
              f"Regrading cost ${costs:.4f} from SDK usage metadata. "
              "`regrade-manifest.json` maps all inputs and original-output hashes to saved per-turn results "
              "in `regraded/`. Original raw directories and extra SDK outputs remain local under `runs/`. "
              "The original campaign reports remain archived. Other leaderboard entries retain their "
              "historical grading configurations; the speech rubric is unchanged.", "",
              f"Judge `{decision['model']}`, effort `{decision['effort']}`, prompt SHA256 `{decision['rubric_sha256']}`.", ""]
    (HERE / "MATCHED-COMPARISON.md").write_text("\n".join(lines))


if __name__ == "__main__":
    build_calibration()
    if (HERE / "adoption.json").exists() and len(list((HERE / "regraded").glob("*.json"))) == 60:
        build_comparison()
