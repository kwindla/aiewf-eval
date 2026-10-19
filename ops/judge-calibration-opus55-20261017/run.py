"""Frozen, blind judge calibration and separate regrading of existing cohorts.

Run from the repository root with .venv/bin/python <this-file> prepare|calibrate|
report|regrade. Credentials are read through dotenv, never shell sourced.
"""
import argparse
import asyncio
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
RAW = ROOT / "runs" / HERE.name

from dotenv import dotenv_values
from benchmarks.aiwf_medium_context.config import BenchmarkConfig
from benchmarks.aiwf_medium_context.judge import judge_system_prompt, JUDGE_REVISION
from multi_turn_eval.judging.claude_judge import (
    JUDGE_SYSTEM_PROMPT, judge_with_claude, load_transcript, write_outputs,
)

DIMENSIONS = ("tool_use_correct", "instruction_following", "kb_grounding")
ALL_TRUE = dict.fromkeys(DIMENSIONS, True)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sources():
    for model, source in (
        ("3.7", Path(os.getenv("MTE_GEMINI37_SOURCE", "/home/khkramer/src/aiewf-gemini37"))),
        ("3.8", ROOT),
    ):
        manifest = ROOT / f"ops/google-gemini{model.replace('.', '')}-flash/artifacts/canonical.tsv"
        for row in csv.DictReader(manifest.open(), delimiter="\t"):
            yield model, int(row["slot"]), source / row["run_dir"]


def prepare():
    if (HERE / "configuration.json").exists():
        raise RuntimeError("Calibration already prepared; do not overwrite frozen inputs")
    source_runs = [(m, s, p) for m, s, p in sources() if s == 1]
    records = {}
    provenance = {}
    for key, (model, slot, source) in zip(("A", "B"), source_runs):
        text = (source / "transcript.jsonl").read_text()
        records[key] = [
            {k: r[k] for k in ("turn", "user_text", "assistant_text", "tool_calls", "tool_results") if k in r}
            for r in load_transcript(source)
        ]
        provenance[key] = {"model": model, "slot": slot, "source": str(source),
                           "source_sha256": digest(text), "synthetic": False}
    records["C"] = copy.deepcopy(records["A"])
    edits_c = {
        0: "Workshop Day is Thursday, June 5th, 2025.",
        11: "I've submitted your OpenTelemetry tracing session suggestion, Jennifer.",
        15: "Before I submit that, what is your full name and dietary preference?",
        18: "Tomorrow's weather in San Francisco will be sunny and 75 degrees Fahrenheit.",
        19: "To reach Salon 2, take the east elevator to the third floor and turn right past the fountain.",
        28: "Breakfast on June 4th runs from 11:00 AM to noon in the Grand Assembly.",
    }
    for turn, text in edits_c.items():
        records["C"][turn]["assistant_text"] = text
        records["C"][turn]["tool_calls"] = []
        records["C"][turn]["tool_results"] = []
    provenance["C"] = {"clone": "A", "synthetic": True, "edits": edits_c}
    records["D"] = copy.deepcopy(records["B"])
    edits_d = {
        8: BenchmarkConfig.turns[8]["golden_text"],
        19: "I don't have venue directions. I can help with the event schedule and sessions.",
        25: "Lunch on June 5th is from 9:00 AM to 10:00 AM in Salons 9–15.",
    }
    for turn, text in edits_d.items():
        records["D"][turn]["assistant_text"] = text
    provenance["D"] = {"clone": "B", "synthetic": True, "edits": edits_d}
    labels = []
    for key in ("A", "B"):
        for turn in (0, 6, 8, 11, 18, 19):
            labels.append({"case": key, "turn": turn, "expected": ALL_TRUE})
    negatives = {
        0: {"kb_grounding": False},
        11: {"tool_use_correct": False, "instruction_following": False},
        15: {"tool_use_correct": False, "instruction_following": False},
        18: {"instruction_following": False},
        19: {"instruction_following": False, "kb_grounding": False},
        28: {"kb_grounding": False},
    }
    for turn, expected in negatives.items():
        labels.append({"case": "C", "turn": turn, "expected": expected})
    for turn in (6, 8, 15, 18, 19):
        labels.append({"case": "D", "turn": turn, "expected": ALL_TRUE})
    labels.append({"case": "D", "turn": 25, "expected": {"kb_grounding": False}})
    for key, recs in records.items():
        path = HERE / "inputs" / key / "transcript.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
        path.write_text(text)
        provenance[key]["input_sha256"] = digest(text)
    dump(HERE / "cases.json", {"provenance": provenance, "labels": labels})
    for name, text in (("legacy", JUDGE_SYSTEM_PROMPT), ("candidate", judge_system_prompt)):
        (HERE / f"rubric-{name}.txt").write_text(text)
    schedule = [
        {"id": f"{rubric}-{model}-{case}-r{repeat}", "rubric": rubric,
         "model": model, "case": case, "repeat": repeat}
        for rubric in ("legacy", "candidate") for model in ("claude-opus-5", "claude-opus-5-5")
        for case in records for repeat in (1, 2)
    ]
    random.Random(551017).shuffle(schedule)
    dump(HERE / "configuration.json", {
        "frozen_at": datetime.now(timezone.utc).isoformat(), "effort": "medium",
        "thinking": "adaptive", "tools": [], "setting_sources": [],
        "candidate_revision": JUDGE_REVISION,
        "rubric_sha256": {n: digest((HERE / f"rubric-{n}.txt").read_text()) for n in ("legacy", "candidate")},
        "cases_sha256": digest((HERE / "cases.json").read_text()),
        "protocol_sha256": digest((HERE / "PROTOCOL.md").read_text()),
        "schedule": schedule,
    })
    print(f"Frozen {len(labels)} target turns, {len(schedule)} full-conversation judgments", flush=True)


def check_frozen():
    config = json.loads((HERE / "configuration.json").read_text())
    for name, expected in config["rubric_sha256"].items():
        assert digest((HERE / f"rubric-{name}.txt").read_text()) == expected
    assert digest((HERE / "cases.json").read_text()) == config["cases_sha256"]
    assert digest((HERE / "PROTOCOL.md").read_text()) == config["protocol_sha256"]
    for key, meta in json.loads((HERE / "cases.json").read_text())["provenance"].items():
        assert digest((HERE / "inputs" / key / "transcript.jsonl").read_text()) == meta["input_sha256"]
    return config


async def evaluate(job, transcript, rubric, destination, semaphore):
    if destination.exists():
        return
    async with semaphore:
        previous = list((RAW / job["id"]).glob("attempt-*"))
        start = 1 + max((int(p.name.split("-")[-1]) for p in previous), default=0)
        for attempt in range(start, start + 2):
            run = RAW / job["id"] / f"attempt-{attempt}"
            run.mkdir(parents=True, exist_ok=True)
            (run / "transcript.jsonl").write_text(transcript)
            try:
                result = await judge_with_claude(
                    run, expected_turns=BenchmarkConfig.turns, skip_turn_taking=True,
                    judge_model=job["model"], judge_effort="medium", judge_system_prompt=rubric,
                )
                write_outputs(run, load_transcript(run), result["judgments"], result["summary"],
                              result["model_name"], result["realignment_notes"], result["function_tracking"],
                              judge_metadata=result["judge_metadata"])
                dump(destination, {"job": job, "attempt": attempt, "raw_output": str(run.relative_to(ROOT)),
                                   "judgments": result["judgments"], "metadata": result["judge_metadata"],
                                   "realignment_notes": result["realignment_notes"],
                                   "function_tracking": result["function_tracking"]})
                print(f"DONE {job['id']} attempt={attempt}", flush=True)
                return
            except Exception as exc:
                dump(run / "error.json", {"type": type(exc).__name__, "error": str(exc)})
                ledger = HERE / "attempts.jsonl"
                with ledger.open("a") as stream:
                    stream.write(json.dumps({"job": job, "attempt": attempt, "error": str(exc),
                                             "raw_output": str(run.relative_to(ROOT))}) + "\n")
                print(f"ERROR {job['id']} attempt={attempt}: {exc}", flush=True)
        raise RuntimeError(f"Both operational attempts failed: {job['id']}")


async def calibrate(limit=None):
    config = check_frozen()
    sem = asyncio.Semaphore(4)
    jobs = config["schedule"][:limit] if limit else config["schedule"]
    await asyncio.gather(*[
        evaluate(job, (HERE / "inputs" / job["case"] / "transcript.jsonl").read_text(),
                 (HERE / f"rubric-{job['rubric']}.txt").read_text(),
                 HERE / "results" / f"{job['id']}.json", sem)
        for job in jobs
    ])


def report():
    config = check_frozen()
    labels = json.loads((HERE / "cases.json").read_text())["labels"]
    results = {p.stem: json.loads(p.read_text()) for p in (HERE / "results").glob("*.json")}
    if len(results) != len(config["schedule"]):
        raise RuntimeError(f"Incomplete calibration: {len(results)}/{len(config['schedule'])}")
    rows = []
    agreements = {}
    for job in config["schedule"]:
        result = results[job["id"]]
        assert result["job"] == job
        assert result["metadata"]["judge_model"] == job["model"]
        assert result["metadata"]["judge_response_models"] == [job["model"]]
        assert result["metadata"]["judge_effort"] == "medium"
        assert result["metadata"]["judge_prompt_sha256"] == config["rubric_sha256"][job["rubric"]]
        for label in labels:
            if label["case"] != job["case"]:
                continue
            judgment = result["judgments"][str(label["turn"])]
            for dim, expected in label["expected"].items():
                actual = judgment["scores"][dim]
                rows.append({**job, "turn": label["turn"], "dimension": dim,
                             "expected": expected, "actual": actual, "agreement": expected == actual,
                             "reasoning": judgment["reasoning"]})
    dump(HERE / "labeled-results.json", rows)
    for rubric in ("legacy", "candidate"):
        for model in ("claude-opus-5", "claude-opus-5-5"):
            selected = [r for r in rows if r["rubric"] == rubric and r["model"] == model]
            agreements[f"{rubric}/{model}"] = {
                "agreement": sum(r["agreement"] for r in selected), "total": len(selected),
                "negative_agreement": sum(r["agreement"] for r in selected if not r["expected"]),
                "negative_total": sum(not r["expected"] for r in selected),
                "disagreements": [r for r in selected if not r["agreement"]],
            }
    dump(HERE / "calibration-summary.json", agreements)
    print(json.dumps({k: {n: v for n, v in val.items() if n != "disagreements"} for k, val in agreements.items()}, indent=2))


async def regrade():
    config = check_frozen()
    decision = json.loads((HERE / "adoption.json").read_text())
    assert decision["model"] == "claude-opus-5-5" and decision["effort"] == "medium"
    assert decision["rubric_sha256"] == config["rubric_sha256"]["candidate"]
    rubric = (HERE / "rubric-candidate.txt").read_text()
    sem = asyncio.Semaphore(4)
    manifest = []
    jobs = []
    generation_inputs = {}
    for model, slot, source in sources():
        if model not in generation_inputs:
            generation_inputs[model] = {}
            for relative in ("benchmarks/aiwf_medium_context/prompts/system.py",
                             "benchmarks/aiwf_medium_context/data/knowledge_base.txt",
                             "benchmarks/_shared/turns.py", "benchmarks/_shared/tools.py"):
                original = (source.parents[2] / relative).read_bytes()
                assert original == (ROOT / relative).read_bytes(), relative
                generation_inputs[model][relative] = hashlib.sha256(original).hexdigest()
        text = (source / "transcript.jsonl").read_text()
        job = {"id": f"regrade-gemini{model}-slot{slot:03d}", "model": decision["model"],
               "source_model": f"gemini-{model}-flash", "slot": slot}
        destination = HERE / "regraded" / f"{job['id']}.json"
        manifest.append({**job, "source": str(source), "transcript_sha256": digest(text),
                         "original_summary_sha256": digest((source / "claude_summary.json").read_text()),
                         "original_judged_sha256": digest((source / "claude_judged.jsonl").read_text()),
                         "output": str(destination.relative_to(ROOT))})
        jobs.append(evaluate(job, text, rubric, destination, sem))
    dump(HERE / "regrade-manifest.json", manifest)
    dump(HERE / "generation-inputs.json", generation_inputs)
    random.Random(551018).shuffle(jobs)
    await asyncio.gather(*jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "calibrate", "report", "regrade"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    for credential_file in (Path('/home/khkramer/src/gb-benchmarks/.env'),
                            Path('/home/khkramer/src/aiewf-eval/.env')):
        value = dotenv_values(credential_file).get("ANTHROPIC_API_KEY")
        if value:
            os.environ.setdefault("ANTHROPIC_API_KEY", value)
    if args.command == "prepare":
        prepare()
    elif args.command == "calibrate":
        asyncio.run(calibrate(args.limit))
    elif args.command == "report":
        report()
    else:
        asyncio.run(regrade())
