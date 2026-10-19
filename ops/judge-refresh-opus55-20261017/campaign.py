"""Resumable blind Opus 5.5 medium refresh of separately frozen cohorts."""

import argparse
import asyncio
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RAW = ROOT / "runs" / HERE.name
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values
from benchmarks.aiwf_medium_context.config import BenchmarkConfig
from benchmarks.aiwf_medium_context.judge import judge_system_prompt as TEXT_RUBRIC
from benchmarks.aiwf_medium_context_speech_v2.judge import judge_system_prompt as VOICE_RUBRIC
from multi_turn_eval.judging.claude_judge import (
    REASONING_NORMALIZATION, judge_with_claude, load_transcript,
    prepare_judge_records, write_outputs,
)

MODEL = "claude-opus-5-5"
EFFORT = "medium"
RUBRICS = {"text": TEXT_RUBRIC, "voice": VOICE_RUBRIC}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def log(event, **fields):
    record = {"at": now(), "event": event, **fields}
    with (HERE / "attempts.jsonl").open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def freeze(spec):
    """Freeze one cohort before any of its judgments are requested."""
    destination = HERE / "cohorts" / f"{spec['key']}.json"
    if destination.exists():
        old = json.loads(destination.read_text())
        assert [(x["capture_id"], x["transcript_sha256"]) for x in old["inputs"]] == [
            (x["capture_id"], x["transcript_sha256"]) for x in spec["inputs"]
        ], f"Refusing to change frozen membership: {spec['key']}"
        return old
    if not spec["inputs"] or len(spec["inputs"]) > spec["target"]:
        raise ValueError("Invalid selected cohort size")
    assert spec["mode"] in RUBRICS
    assert len({x["transcript_sha256"] for x in spec["inputs"]}) == len(spec["inputs"])
    for slot, item in enumerate(spec["inputs"], 1):
        source = Path(item["source"])
        data = source.read_bytes()
        assert sha(data) == item["transcript_sha256"], source
        snapshot = RAW / "inputs" / spec["key"] / f"{slot:03d}" / "transcript.jsonl"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            assert snapshot.read_bytes() == data
        else:
            snapshot.write_bytes(data)
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        scripted = [r for r in rows if not r.get("recovery_turn")]
        assert sorted(r["turn"] for r in scripted) == list(range(30)), source
        assert all(r.get("assistant_text") or r.get("tool_calls") for r in scripted), source
        item["input"] = str(snapshot.relative_to(ROOT))
        item["job_id"] = f"{spec['key']}-{slot:03d}-{item['transcript_sha256'][:12]}"
    spec.update(
        frozen_at=now(), judge_model=MODEL, judge_effort=EFFORT,
        judge_prompt_sha256=sha(RUBRICS[spec["mode"]].encode()),
        actual_n=len(spec["inputs"]), shortfall=spec["target"] - len(spec["inputs"]),
    )
    dump(destination, spec)
    return spec


def cohorts(mode=None):
    return [
        data for path in sorted((HERE / "cohorts").glob("*.json"))
        if (data := json.loads(path.read_text()))["mode"] == mode or mode is None
    ]


def result_path(item):
    return HERE / "results" / f"{item['job_id']}.json"


def blocked_path(item):
    return HERE / "provider-blocked" / f"{item['job_id']}.json"


def normalization_metadata(item):
    raw = (ROOT / item["input"]).read_bytes()
    assert sha(raw) == item["transcript_sha256"]
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    _, changes = prepare_judge_records(records)
    return ({"judge_input_normalization": REASONING_NORMALIZATION,
             "judge_normalized_turns": changes} if changes else {})


def validate_normalization(item, metadata):
    expected = normalization_metadata(item)
    for key in ("judge_input_normalization", "judge_normalized_turns"):
        assert metadata.get(key) == expected.get(key), f"Judge input normalization mismatch: {item['job_id']}"


def validate_blocked(spec, item):
    path = blocked_path(item)
    if not path.exists():
        return False
    assert not result_path(item).exists(), "A case cannot be both graded and blocked"
    data = json.loads(path.read_text())
    assert data["transcript_sha256"] == item["transcript_sha256"]
    assert data["judge_model"] == MODEL and data["judge_effort"] == EFFORT
    assert data["judge_prompt_sha256"] == spec["judge_prompt_sha256"]
    assert data["status"] == "provider_safeguard_rejection"
    validate_normalization(item, data)
    return True


def resolved(spec, item):
    return validate_existing(spec, item) or validate_blocked(spec, item)


def preserve_provider_block(spec, item, attempt, error, clarification=None):
    dump(blocked_path(item), {
        "at": now(), "job_id": item["job_id"], "cohort": spec["key"],
        "status": "provider_safeguard_rejection", "attempt": attempt,
        "transcript_sha256": item["transcript_sha256"],
        "judge_model": MODEL, "judge_effort": EFFORT,
        "judge_prompt_sha256": spec["judge_prompt_sha256"],
        "error": error, "task_clarification": clarification,
        "quality_judgment": None,
        **normalization_metadata(item),
    })
    log("provider_blocked", job=item["job_id"], attempt=attempt)


def write_mode_completion(mode, selected):
    """Preserve the original phase boundary when a finished campaign is resumed."""
    blocked = sum(validate_blocked(c, item) for c in selected for item in c["inputs"])
    jobs = sum(c["actual_n"] for c in selected)
    fields = {"cohorts": len(selected), "jobs": jobs, "graded_jobs": jobs - blocked,
              "provider_blocked_jobs": blocked, "all_quality_complete": blocked == 0}
    path = HERE / f"{mode}-COMPLETE.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if any(previous.get(key) != value for key, value in fields.items()):
            raise RuntimeError("Completed campaign dispositions changed; preserve the original and create an explicit revision")
        return previous
    result = {"at": now(), **fields,
              "complete_means": "all selected inputs have an explicit terminal disposition"}
    dump(path, result)
    return result


def validate_existing(spec, item):
    destination = result_path(item)
    if not destination.exists():
        return False
    result = json.loads(destination.read_text())
    assert result["transcript_sha256"] == item["transcript_sha256"]
    assert result["metadata"]["judge_model"] == MODEL
    assert result["metadata"]["judge_effort"] == EFFORT
    assert result["metadata"]["judge_response_models"] == [MODEL]
    assert result["metadata"]["judge_prompt_sha256"] == spec["judge_prompt_sha256"]
    assert set(result["judgments"]) == {str(n) for n in range(30)}
    validate_normalization(item, result["metadata"])
    return True


def credit_failure(message):
    return bool(re.search(
        r"credit.balance|insufficient.credit|not.enough.credit|out.of.credit|payment.required|insufficient.funds|purchase.*credit|billing.*(?:limit|balance)|balance.*(?:low|insufficient)",
        message, re.I,
    ))


def safe_error(exc):
    return re.sub(r"sk-(?:ant-)?[A-Za-z0-9_-]+", "[REDACTED]", str(exc))[:4000]


class CreditCooldown:
    def __init__(self):
        self.until = 0.0

    async def wait(self):
        while self.until > time.monotonic():
            await asyncio.sleep(min(30, self.until - time.monotonic()))

    def suspend(self):
        self.until = max(self.until, time.monotonic() + 90)


def mark_turn_taking_unscored(run, result):
    """Keep absent audio judgments null in JSON and in the human-readable report."""
    summary_file = run / "claude_summary.json"
    summary = json.loads(summary_file.read_text())
    summary["claude_passes"]["turn_taking"] = None
    summary["turn_taking_affected_instruction"] = None
    dump(summary_file, summary)
    total = len(result["judgments"])
    dimensions = ("tool_use_correct", "instruction_following", "kb_grounding")
    passed = sum(all(j["scores"][dim] for dim in dimensions) for j in result["judgments"].values())
    lines = [
        "# Spoken-transcript quality evaluation", "",
        f"**Judge**: {MODEL}", f"**Judge effort**: {EFFORT}",
        f"**Turns**: {total}", "**Turn-taking**: unscored (null)",
        f"**Strict quality passes**: {passed}/{total}", "",
        "Audio timing is measured separately. Missing turn-taking judgments are not failures or passes.", "",
    ]
    for turn, judgment in result["judgments"].items():
        failed = [dim for dim in dimensions if not judgment["scores"][dim]]
        if failed:
            lines += [f"## Turn {turn}", "", ", ".join(failed), "", judgment["reasoning"], ""]
    (run / "claude_analysis.md").write_text("\n".join(lines) + "\n")


async def evaluate(spec, item, cooldown):
    assert sha(RUBRICS[spec["mode"]].encode()) == spec["judge_prompt_sha256"]
    data = (ROOT / item["input"]).read_bytes()
    assert sha(data) == item["transcript_sha256"]
    job_dir = RAW / "judging" / item["job_id"]
    previous = [int(p.name.split("-")[-1]) for p in job_dir.glob("attempt-*")]
    attempt = max(previous, default=0)
    operational_failures = 0
    clarification_file = HERE / "task-clarifications.json"
    clarifications = json.loads(clarification_file.read_text()) if clarification_file.exists() else {}
    clarification = clarifications.get(item["job_id"], {}).get("text")
    if normalization_metadata(item):
        # The old note concerned leaked analysis, which is absent after cleanup.
        clarification = None
    while operational_failures < 4:
        await cooldown.wait()
        attempt += 1
        run = job_dir / f"attempt-{attempt}"
        run.mkdir(parents=True)
        (run / "transcript.jsonl").write_bytes(data)
        log("start", job=item["job_id"], attempt=attempt)
        if clarification:
            log("task_clarification", job=item["job_id"], attempt=attempt,
                sha256=sha(clarification.encode()))
        try:
            audio = item.get("turn_taking_data")
            audio = {int(k): v for k, v in audio.items()} if audio is not None else None
            result = await asyncio.wait_for(judge_with_claude(
                run, expected_turns=BenchmarkConfig.turns,
                skip_turn_taking=audio is None, precomputed_turn_taking=audio,
                judge_model=MODEL, judge_effort=EFFORT,
                judge_system_prompt=RUBRICS[spec["mode"]],
                task_clarification=clarification,
            ), timeout=600)
            metadata = result["judge_metadata"]
            assert metadata["judge_response_models"] == [MODEL], metadata["judge_response_models"]
            if spec["mode"] == "voice" and not item.get("turn_taking_scored", True):
                for judgment in result["judgments"].values():
                    judgment["scores"]["turn_taking"] = None
            metadata["turn_taking_scored"] = spec["mode"] == "voice" and item.get("turn_taking_scored", True)
            write_outputs(
                run, load_transcript(run), result["judgments"], result["summary"],
                result["model_name"], result["realignment_notes"], result["function_tracking"],
                judge_metadata=metadata,
            )
            if spec["mode"] == "voice" and not item.get("turn_taking_scored", True):
                mark_turn_taking_unscored(run, result)
            dump(result_path(item), {
                "job_id": item["job_id"], "cohort": spec["key"], "attempt": attempt,
                "transcript_sha256": item["transcript_sha256"],
                "raw_output": str(run.relative_to(ROOT)), "judgments": result["judgments"],
                "metadata": metadata, "realignment_notes": result["realignment_notes"],
                "function_tracking": result["function_tracking"],
            })
            log("done", job=item["job_id"], attempt=attempt, cost_usd=metadata.get("judge_cost_usd"))
            return
        except Exception as exc:
            error = safe_error(exc)
            credit = credit_failure(error)
            dump(run / "error.json", {"at": now(), "type": type(exc).__name__, "error": error, "credit_failure": credit})
            log("error", job=item["job_id"], attempt=attempt, error=error, credit_failure=credit)
            if "reasoning_extraction" in error:
                preserve_provider_block(spec, item, attempt, error, clarification)
                return
            if credit:
                cooldown.suspend()
                log("credit_cooldown", seconds=90)
            else:
                operational_failures += 1
                if operational_failures < 4:
                    await asyncio.sleep(min(120, 15 * 2 ** (operational_failures - 1)))
    raise RuntimeError(f"Four operational attempts failed: {item['job_id']}")


async def run(mode, concurrency, watch, expected_cohorts):
    for file in (Path("/home/khkramer/src/gb-benchmarks/.env"), Path("/home/khkramer/src/aiewf-eval/.env")):
        key = dotenv_values(file).get("ANTHROPIC_API_KEY")
        if key:
            os.environ.setdefault("ANTHROPIC_API_KEY", key)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is unavailable")
    if mode == "voice":
        while not (HERE / "text-COMPLETE.json").exists():
            await asyncio.sleep(20)
        text = cohorts("text")
        assert len(text) == 43 and all(resolved(c, item) for c in text for item in c["inputs"])
    cooldown = CreditCooldown()
    while True:
        selected = cohorts(mode)
        pending = [(c, item) for c in selected for item in c["inputs"] if not resolved(c, item)]
        if not pending:
            if not watch or len(selected) >= expected_cohorts:
                write_mode_completion(mode, selected)
                log("mode_complete", mode=mode, cohorts=len(selected))
                return
            await asyncio.sleep(20)
            continue
        batch = pending[:concurrency]
        results = await asyncio.gather(*(evaluate(c, item, cooldown) for c, item in batch), return_exceptions=True)
        failures = [safe_error(r) for r in results if isinstance(r, BaseException)]
        if failures:
            dump(HERE / f"{mode}-operational-failures.json", {"at": now(), "failures": failures})
            raise RuntimeError("Operational failures require inspection; successful jobs are preserved")


def status():
    rows = []
    cost = 0.0
    for c in cohorts():
        done = 0
        for item in c["inputs"]:
            if validate_existing(c, item):
                done += 1
                cost += json.loads(result_path(item).read_text())["metadata"].get("judge_cost_usd", 0) or 0
        rows.append({"key": c["key"], "mode": c["mode"], "done": done, "provider_blocked": sum(validate_blocked(c, i) for i in c["inputs"]), "n": c["actual_n"], "shortfall": c["shortfall"]})
    superseded = sum(
        json.loads(path.read_text())["metadata"].get("judge_cost_usd", 0) or 0
        for path in (HERE / "reasoning-cleanup" / "prior-results").glob("*.json")
    )
    print(json.dumps({"at": now(), "cohorts": rows,
                      "published_grade_cost_usd": cost,
                      "superseded_grade_cost_usd": superseded,
                      "successful_judge_cost_usd": cost + superseded}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("freeze")
    add.add_argument("spec", type=Path)
    worker = sub.add_parser("run")
    worker.add_argument("--mode", choices=["text", "voice"], required=True)
    worker.add_argument("--concurrency", type=int, default=8)
    worker.add_argument("--watch", action="store_true")
    worker.add_argument("--expected-cohorts", type=int, required=True)
    sub.add_parser("status")
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    if args.command == "freeze":
        freeze(json.loads(args.spec.read_text()))
    elif args.command == "status":
        status()
    else:
        with (HERE / f"{args.mode}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            asyncio.run(run(args.mode, args.concurrency, args.watch, args.expected_cohorts))
