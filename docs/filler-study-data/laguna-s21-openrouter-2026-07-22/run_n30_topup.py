#!/usr/bin/env python3
"""Resumable balanced top-up from n=10 to n=30 in each Laguna arm."""

from __future__ import annotations

import fcntl
import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("laguna_run_campaign", HERE / "run_campaign.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load run_campaign.py")
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def main() -> None:
    schedule = HERE / "schedule-n30-topup.tsv"
    state = HERE / "state" / "n30-topup"
    logs = state / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    attempts_path = state / "attempts.tsv"
    counted_path = state / "counted.tsv"
    manifest_path = state / "manifest.tsv"
    master_manifest = HERE / "state" / "manifest.tsv"
    driver_log = state / "driver.log"

    with (state / "driver.lock").open("a") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another Laguna n=30 top-up driver is active") from exc

        def log(message: str) -> None:
            line = f"[{campaign.now()}] {message}"
            print(line, flush=True)
            with driver_log.open("a") as handle:
                handle.write(line + "\n")

        assignments = campaign.read_tsv(schedule)
        if len(assignments) != 40:
            raise RuntimeError("n=30 top-up schedule must contain exactly 40 assignments")
        for index, assignment in enumerate(assignments):
            slot = assignment["slot"]
            arm = assignment["arm"]
            expected_arm = "nofiller" if index % 2 == 0 else "dots96"
            expected_number = 11 + index // 2
            expected_slot = f"LS21-{'N' if expected_arm == 'nofiller' else 'D'}{expected_number:02d}"
            if (
                assignment["model"] != campaign.MODEL_KEY
                or assignment["requested_model"] != campaign.MODEL
                or assignment["service"] != "openrouter"
                or arm != expected_arm
                or slot != expected_slot
            ):
                raise RuntimeError(f"n=30 top-up schedule policy failure in {slot}")

        counted = {row["slot"] for row in campaign.read_tsv(counted_path)}
        master_rows = campaign.read_tsv(master_manifest)
        master = {row["slot"]: row for row in master_rows}
        if len(master) != len(master_rows):
            raise RuntimeError("duplicate slot in master manifest")

        for assignment in assignments:
            slot = assignment["slot"]
            arm = assignment["arm"]
            if slot in counted:
                if slot not in master:
                    raise RuntimeError(f"counted top-up slot missing from master manifest: {slot}")
                continue
            if slot in master:
                raise RuntimeError(f"uncounted top-up slot already in master manifest: {slot}")

            prior = [
                row
                for row in campaign.read_tsv(attempts_path)
                if row["slot"] == slot and not row["classification"].startswith("infra_")
            ]
            candidate: Path | None = None
            classification = ""
            attempt = len(
                [row for row in campaign.read_tsv(attempts_path) if row["slot"] == slot]
            )
            if prior:
                candidate = Path(prior[-1]["run_dir"])
                classification = prior[-1]["classification"]
                log(f"adopting slot={slot} run={candidate}")

            while candidate is None:
                attempt += 1
                if attempt > campaign.MAX_ATTEMPTS:
                    raise RuntimeError(f"replacement limit reached: {slot}")
                run_output = logs / f"{slot}-attempt-{attempt}.log"
                started = campaign.now()
                log(f"run slot={slot} attempt={attempt} arm={arm}")
                rc, output, run_dir = campaign.run_attempt(arm, run_output)
                rows = 0
                es_turn = -1
                if run_dir is not None and (run_dir / "transcript.jsonl").is_file():
                    rows = len((run_dir / "transcript.jsonl").read_text().splitlines())
                    es_turn = campaign.end_session_turn(run_dir)
                infra = es_turn < 0 and bool(campaign.INFRA_RE.search(output))
                if infra:
                    classification = (
                        "infra_zero_response_replaced"
                        if rows == 0
                        else "infra_partial_response_replaced"
                    )
                elif rows == 0:
                    classification = "zero_response_unclassified"
                elif es_turn == 29:
                    classification = "strict_complete"
                elif es_turn >= 0:
                    classification = "model_abort"
                else:
                    classification = "incomplete_no_end_session"
                campaign.append_tsv(
                    attempts_path,
                    [
                        "slot",
                        "model",
                        "arm",
                        "attempt",
                        "start_utc",
                        "end_utc",
                        "run_rc",
                        "run_dir",
                        "transcript_rows",
                        "end_session_turn",
                        "classification",
                        "log",
                    ],
                    {
                        "slot": slot,
                        "model": campaign.MODEL_KEY,
                        "arm": arm,
                        "attempt": attempt,
                        "start_utc": started,
                        "end_utc": campaign.now(),
                        "run_rc": rc,
                        "run_dir": run_dir or "NA",
                        "transcript_rows": rows,
                        "end_session_turn": es_turn,
                        "classification": classification,
                        "log": run_output,
                    },
                )
                log(
                    f"attempt slot={slot} rc={rc} rows={rows} "
                    f"end_session={es_turn} class={classification}"
                )
                if classification.startswith("infra_"):
                    continue
                if classification == "zero_response_unclassified" or run_dir is None:
                    raise RuntimeError(f"unclassified zero-response: {slot}")
                candidate = run_dir

            campaign.validate_run(candidate, arm)
            row = {
                "slot": slot,
                "model": campaign.MODEL_KEY,
                "arm": arm,
                "run_dir": candidate.relative_to(ROOT),
                "classification": classification,
            }
            campaign.append_tsv(
                counted_path,
                ["slot", "model", "arm", "attempt", "run_dir", "classification"],
                {**row, "attempt": attempt, "run_dir": candidate},
            )
            campaign.append_tsv(
                manifest_path,
                ["slot", "model", "arm", "run_dir", "classification"],
                row,
            )
            if slot in master:
                raise RuntimeError(f"refusing duplicate master-manifest append: {slot}")
            campaign.append_tsv(
                master_manifest,
                ["slot", "model", "arm", "run_dir", "classification"],
                row,
            )
            counted.add(slot)
            master[slot] = row
            log(f"counted slot={slot} class={classification} run={candidate}")

        (state / "RUNS_COMPLETE").touch()
        log("N30_TOPUP_RUNS_COMPLETE")


if __name__ == "__main__":
    main()
