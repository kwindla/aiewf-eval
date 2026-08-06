#!/usr/bin/env python3
"""Read-only check or freeze an adaptive dots-stage decision.

This helper makes no provider requests. ``--execute`` appends one durable stage
decision after confirming that the preceding dot stage and its local analysis
artifact are complete and unchanged.
"""

from __future__ import annotations

import argparse
import fcntl
from pathlib import Path

import collect


TRANSITIONS = {(6, 10), (10, 30)}


def validate_transition(
    from_stage: int,
    to_stage: int,
    analysis: str | None,
    rationale: str | None,
    *,
    execute: bool,
) -> tuple[list[dict[str, str]], Path | None]:
    if (from_stage, to_stage) not in TRANSITIONS:
        raise RuntimeError("allowed transitions are 6->10 and 10->30")
    collect.validate_configuration()
    schedule = collect.validate_schedule()
    collect.validate_source_hashes()
    canonical = collect.validate_manifests(schedule)
    if len(canonical) != from_stage:
        raise RuntimeError(
            f"stage decision requires exactly {from_stage} canonical dots runs; "
            f"found {len(canonical)}"
        )
    marker = f"DOT_STAGE_DONE stage={from_stage} canonical={from_stage} control=30"
    if marker not in collect.CAMPAIGN_LOG.read_text(encoding="utf-8"):
        raise RuntimeError(f"campaign.log lacks completed-stage marker: {marker}")
    if collect.validate_or_freeze_control(execute=False) != 30:
        raise RuntimeError("the 30-run primary control is not frozen")

    decisions = [
        row for row in collect.read_tsv(collect.STAGE_DECISIONS)
        if int(row["completed_stage"]) == from_stage
        and int(row["requested_stage"]) == to_stage
    ]
    if len(decisions) > 1:
        raise RuntimeError("duplicate stage decisions exist for this transition")

    artifact: Path | None = None
    if analysis:
        artifact = collect.resolve_repo_path(analysis)
        expected_artifact = (collect.HERE / f"analysis/stage-{from_stage}.json").resolve()
        if artifact != expected_artifact:
            raise RuntimeError(
                f"stage {from_stage} gate requires the canonical analysis artifact: "
                f"{expected_artifact.relative_to(collect.ROOT)}"
            )
        if not artifact.is_file() or not artifact.stat().st_size:
            raise RuntimeError(f"analysis artifact is missing or empty: {artifact}")
    if execute and artifact is None:
        raise RuntimeError("--execute requires --analysis")
    if execute and not (rationale or "").strip():
        raise RuntimeError("--execute requires a non-empty --rationale")

    if decisions:
        row = decisions[0]
        frozen_artifact = collect.resolve_repo_path(row["analysis_artifact"])
        if not frozen_artifact.is_file():
            raise RuntimeError(f"frozen analysis artifact is missing: {frozen_artifact}")
        if collect.sha256(frozen_artifact) != row["analysis_sha256"]:
            raise RuntimeError("frozen analysis artifact changed")
        if execute:
            proposed = {
                "decision": "extend",
                "analysis_artifact": str(artifact.relative_to(collect.ROOT)),
                "analysis_sha256": collect.sha256(artifact),
                "rationale": (rationale or "").strip(),
            }
            if any(row[key] != value for key, value in proposed.items()):
                raise RuntimeError("a different decision is already frozen for this transition")
    return decisions, artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-stage", type=int, required=True)
    parser.add_argument("--to-stage", type=int, required=True)
    parser.add_argument("--analysis")
    parser.add_argument("--rationale")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    decisions, artifact = validate_transition(
        args.from_stage,
        args.to_stage,
        args.analysis,
        args.rationale,
        execute=False,
    )
    if not args.execute:
        state = "already frozen" if decisions else "ready to freeze"
        print(
            f"read-only gate preflight: {args.from_stage}->{args.to_stage} {state}; "
            "no file or provider request was made"
        )
        return

    with collect.LOCK.open("a") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another collector or stage-gate process owns the lock") from exc
        decisions, artifact = validate_transition(
            args.from_stage,
            args.to_stage,
            args.analysis,
            args.rationale,
            execute=True,
        )
        if decisions:
            print("identical stage decision is already frozen; no change made")
            return
        assert artifact is not None
        collect.append_tsv(
            collect.STAGE_DECISIONS,
            collect.DECISION_FIELDS,
            {
                "completed_stage": args.from_stage,
                "requested_stage": args.to_stage,
                "decision": "extend",
                "decided_at": collect.now(),
                "analysis_artifact": str(artifact.relative_to(collect.ROOT)),
                "analysis_sha256": collect.sha256(artifact),
                "rationale": args.rationale.strip(),
            },
        )
        print(
            f"frozen extend decision {args.from_stage}->{args.to_stage}: "
            f"{artifact.relative_to(collect.ROOT)}"
        )


if __name__ == "__main__":
    main()
