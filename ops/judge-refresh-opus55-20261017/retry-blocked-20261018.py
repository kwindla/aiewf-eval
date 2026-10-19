"""User-requested retry of the three provider-rejected inputs, preserving history."""

import asyncio
import fcntl
import json
import os
import shutil

import campaign


async def main():
    revision = campaign.HERE / "retries" / "20261018"
    if revision.exists():
        raise RuntimeError("Retry already started; inspect saved attempts before resuming")
    pending = [
        (spec, item)
        for spec in campaign.cohorts("text")
        for item in spec["inputs"]
        if campaign.validate_blocked(spec, item)
    ]
    assert len(pending) == 3
    for source in (
        campaign.Path("/home/khkramer/src/gb-benchmarks/.env"),
        campaign.Path("/home/khkramer/src/aiewf-eval/.env"),
    ):
        key = campaign.dotenv_values(source).get("ANTHROPIC_API_KEY")
        if key:
            os.environ.setdefault("ANTHROPIC_API_KEY", key)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is unavailable")
    revision.mkdir(parents=True)
    for spec, item in pending:
        shutil.copy2(campaign.blocked_path(item), revision / campaign.blocked_path(item).name)
    for name in ("text-COMPLETE.json", "VERIFICATION.json"):
        shutil.copy2(campaign.HERE / name, revision / name)
    record = {
        "started_at": campaign.now(),
        "reason": "User requested retry of all three provider-rejected conversations",
        "judge_model": campaign.MODEL,
        "judge_effort": campaign.EFFORT,
        "source_and_rubric_unchanged": True,
        "clarifications": "Existing per-input clarification retained unchanged, if present",
        "jobs": [item["job_id"] for _, item in pending],
    }
    campaign.dump(revision / "retry.json", record)
    cooldown = campaign.CreditCooldown()
    errors = await asyncio.gather(
        *(campaign.evaluate(spec, item, cooldown) for spec, item in pending),
        return_exceptions=True,
    )
    outcomes = []
    for (spec, item), error in zip(pending, errors):
        if campaign.validate_existing(spec, item):
            campaign.blocked_path(item).unlink()
            result = json.loads(campaign.result_path(item).read_text())
            outcome = {"status": "graded", "attempt": result["attempt"],
                       "cost_usd": result["metadata"].get("judge_cost_usd")}
        elif isinstance(error, BaseException):
            outcome = {"status": "operational_failure", "error": campaign.safe_error(error)}
        else:
            assert campaign.validate_blocked(spec, item)
            blocked = json.loads(campaign.blocked_path(item).read_text())
            outcome = {"status": "provider_safeguard_rejection", "attempt": blocked["attempt"]}
        outcomes.append({"job_id": item["job_id"], **outcome})
    record.update(finished_at=campaign.now(), outcomes=outcomes)
    campaign.dump(revision / "retry.json", record)
    if any(item["status"] == "graded" for item in outcomes):
        (campaign.HERE / "text-COMPLETE.json").unlink()
        campaign.write_mode_completion("text", campaign.cohorts("text"))
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    with (campaign.HERE / "text.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(main())
