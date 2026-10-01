"""Lossless experimental judge view with fragment provenance and backend diagnostics.

Does not synthesize answers or automatically move speech across user requests.
Ambiguous interval ordering blocks approval pending audio review.
"""

import argparse, hashlib, json
from pathlib import Path


def read(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l]


def reconstruct(run):
    events = read(run / "live-events.jsonl")
    original = read(run / "transcript.jsonl")
    out = run / "reconstructed"
    out.mkdir(exist_ok=True)
    mapped = []
    issues = []
    backend = []
    used = set()
    records = []
    for row in original:
        turn = row["turn"]
        fragments = [
            (i, e)
            for i, e in enumerate(events)
            if e.get("turn") == turn
            and e.get("event", {}).get("type") == "session.output_transcript.delta"
        ]
        # Keep event receipt order; intervals audit whether any manual regrouping is needed.
        times = [e["event"]["start_ms"] for _, e in fragments]
        if times != sorted(times):
            issues.append(
                dict(
                    turn=turn,
                    reason="Nonmonotonic spoken fragments; inspect audio before reordering",
                )
            )
        text = "".join(e["event"]["delta"] for _, e in fragments).strip()
        if text != row["assistant_text"]:
            issues.append(
                dict(
                    turn=turn,
                    reason="Raw output includes fragments outside committed transcript",
                    committed=row["assistant_text"],
                    all_fragments=text,
                )
            )
        r = dict(row)
        r["assistant_text"] = text
        records.append(r)
        ids = [e["event"]["event_id"] for _, e in fragments]
        if used.intersection(ids):
            raise ValueError("Duplicate output event ID")
        used.update(ids)
        mapped.append(
            dict(
                turn=turn,
                original_text=row["assistant_text"],
                reconstructed_text=text,
                fragments=[
                    dict(line=i + 1, receipt_t=e["t"], **e["event"])
                    for i, e in fragments
                ],
                changed=text != row["assistant_text"],
            )
        )
        messages = []
        calls = []
        delegations = []
        for i, e in enumerate(events):
            if e.get("turn") != turn:
                continue
            event = e.get("event", {})
            if event.get("type") == "session.delegation.created":
                delegations.append(event["delegation"])
            inner = event.get("event", {})
            if (
                event.get("type") == "response.event"
                and inner.get("type") == "response.output_item.done"
            ):
                item = inner["item"]
                entry = dict(
                    line=i + 1,
                    receipt_t=e["t"],
                    delegation_id=event.get("delegation_id"),
                    item=item,
                )
                if item["type"] == "message":
                    messages.append(entry)
                elif item["type"] == "function_call":
                    calls.append(entry)
        backend.append(
            dict(
                turn=turn,
                delegations=delegations,
                messages=messages,
                calls=calls,
                tool_results=[
                    e
                    for e in events
                    if e.get("turn") == turn and e["type"] == "tool_result"
                ],
                spoken_text=text,
            )
        )
    # Cross-exchange timestamp inversions are not repaired by receipt-time turn labels.
    previous_end = None
    for m in mapped:
        fs = m["fragments"]
        if fs and previous_end is not None and fs[0]["start_ms"] < previous_end:
            issues.append(
                dict(
                    turn=m["turn"],
                    reason="Cross-exchange output timestamp overlap; audio review required",
                )
            )
        if fs:
            previous_end = max(f["end_ms"] for f in fs)
    committed = {r["turn"] for r in original}
    orphaned = [
        dict(line=i + 1, event=e)
        for i, e in enumerate(events)
        if e.get("event", {}).get("type") == "session.output_transcript.delta"
        and e.get("turn") not in committed
    ]
    (out / "transcript.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    (out / "mapping.json").write_text(
        json.dumps(
            dict(
                version="lossless-exchanges-v1",
                source_sha256=hashlib.sha256(
                    (run / "transcript.jsonl").read_bytes()
                ).hexdigest(),
                events_sha256=hashlib.sha256(
                    (run / "live-events.jsonl").read_bytes()
                ).hexdigest(),
                exchanges=mapped,
                unassigned_or_startup_fragments=orphaned,
                issues=issues,
                approved_for_judging=not issues,
            ),
            indent=2,
        )
        + "\n"
    )
    (out / "backend-diagnostics.json").write_text(json.dumps(backend, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                run=str(run),
                exchanges=len(records),
                changed=sum(x["changed"] for x in mapped),
                issues=issues,
            )
        )
    )
    return not issues
