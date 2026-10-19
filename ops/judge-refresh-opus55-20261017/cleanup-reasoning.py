"""Audit frozen inputs, archive superseded grades, and rejudge cleaned answers."""

import asyncio
import fcntl
import json
import shutil

import campaign

REVISION = campaign.HERE / "reasoning-cleanup"


async def main():
    selected = campaign.cohorts()
    jobs = []
    recovery_only = []
    for spec in selected:
        for item in spec["inputs"]:
            metadata = campaign.normalization_metadata(item)
            if metadata:
                jobs.append({"job_id": item["job_id"], "cohort": spec["key"],
                             "mode": spec["mode"],
                             "transcript_sha256": item["transcript_sha256"],
                             **metadata})
            else:
                records = campaign.load_transcript((campaign.ROOT / item["input"]).parent)
                if any("</think>" in r["assistant_text"] for r in records if r.get("recovery_turn")):
                    recovery_only.append(item["job_id"])
    assert len(jobs) == 29 and all(job["mode"] == "text" for job in jobs)
    manifest = REVISION / "manifest.json"
    if not manifest.exists():
        REVISION.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            for source_dir, archive_dir in (("results", "prior-results"),
                                            ("provider-blocked", "prior-blocked")):
                source = campaign.HERE / source_dir / f"{job['job_id']}.json"
                if source.exists():
                    target = REVISION / archive_dir / source.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        for name in ("text-COMPLETE.json", "VERIFICATION.json", "aggregate.json"):
            shutil.copy2(campaign.HERE / name, REVISION / name)
        campaign.dump(manifest, {
            "started_at": campaign.now(),
            "reason": "User requested stripping leaked reasoning before judging, without parser root-cause work",
            "normalization": campaign.REASONING_NORMALIZATION,
            "cohorts_audited": len(selected),
            "conversations_audited": sum(c["actual_n"] for c in selected),
            "changed_conversations": len(jobs),
            "changed_scripted_turns": sum(len(job["judge_normalized_turns"]) for job in jobs),
            "recovery_only_unchanged_jobs": recovery_only,
            "source_cohorts_rubrics_model_effort_and_latency_unchanged": True,
            "old_task_clarification_omitted": True,
            "jobs": jobs,
        })
    record = json.loads(manifest.read_text())
    assert record["jobs"] == jobs
    if record.get("finished_at"):
        assert all(campaign.resolved(c, i) for c in selected for i in c["inputs"])
        return
    # Retain the source-hashed originals in the revision before replacing any
    # disposition. On resume, keep valid normalized results already received.
    for job in jobs:
        for source_dir in ("results", "provider-blocked"):
            source = campaign.HERE / source_dir / f"{job['job_id']}.json"
            if source.exists():
                data = json.loads(source.read_text())
                meta = data.get("metadata", data)
                if meta.get("judge_input_normalization") != record["normalization"]:
                    archive_dir = "prior-results" if source_dir == "results" else "prior-blocked"
                    assert (REVISION / archive_dir / source.name).read_bytes() == source.read_bytes()
                    source.unlink()
    (campaign.HERE / "text-COMPLETE.json").unlink(missing_ok=True)
    await campaign.run("text", concurrency=4, watch=False, expected_cohorts=43)
    record["outcomes"] = []
    lookup = {i["job_id"]: (c, i) for c in selected for i in c["inputs"]}
    for job in jobs:
        spec, item = lookup[job["job_id"]]
        graded = campaign.validate_existing(spec, item)
        data = json.loads((campaign.result_path(item) if graded else campaign.blocked_path(item)).read_text())
        record["outcomes"].append({"job_id": item["job_id"], "attempt": data["attempt"],
                                    "status": "graded" if graded else "provider_safeguard_rejection"})
    record["finished_at"] = campaign.now()
    campaign.dump(manifest, record)
    print(json.dumps({"changed_conversations": len(jobs), "outcomes": record["outcomes"]}, indent=2))


if __name__ == "__main__":
    with (campaign.HERE / "text.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(main())
