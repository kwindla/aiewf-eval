#!/usr/bin/env python3
"""Validate the frozen mechanical scorer against 900 judged target turns."""

from __future__ import annotations

import collections
import json
from typing import Any

from scorer import score_message, transcript_message
from study import HERE, ROOT, atomic_write_json, atomic_write_text, iter_historical_target_rows


def main() -> int:
    rows: list[dict[str, Any]] = []
    disagreements = []
    categories: collections.Counter[str] = collections.Counter()
    by_source: collections.Counter[tuple[str, bool]] = collections.Counter()
    by_turn: collections.Counter[tuple[int, bool]] = collections.Counter()

    for item in iter_historical_target_rows():
        scored = score_message(item["turn"], transcript_message(item["transcript"]))
        judged = bool(item["judged"]["scores"]["tool_use_correct"])
        agrees = scored["success"] == judged
        row = {
            "source": item["source"],
            "cohort": item["cohort"],
            "slot": item["slot"],
            "run_dir": str(item["run_dir"].relative_to(ROOT)),
            "turn": item["turn"],
            "mechanical_success": scored["success"],
            "mechanical_category": scored["category"],
            "judge_tool_use_correct": judged,
            "agrees": agrees,
        }
        rows.append(row)
        categories[scored["category"]] += 1
        by_source[(item["source"], agrees)] += 1
        by_turn[(item["turn"], agrees)] += 1
        if not agrees:
            disagreements.append(
                {
                    **row,
                    "assistant_text": item["transcript"].get("assistant_text"),
                    "tool_calls": item["transcript"].get("tool_calls"),
                    "judge_reasoning": item["judged"].get("claude_reasoning"),
                }
            )

    total = len(rows)
    agreements = total - len(disagreements)
    payload = {
        "schema_version": 1,
        "total": total,
        "agreements": agreements,
        "disagreements": len(disagreements),
        "agreement_percent": agreements / total * 100,
        "required_percent": 99.5,
        "gate_passed": agreements / total >= 0.995,
        "categories": dict(categories),
        "by_source": {
            source: {
                "agreement": by_source[(source, True)],
                "disagreement": by_source[(source, False)],
            }
            for source in sorted({row["source"] for row in rows})
        },
        "by_turn": {
            str(turn): {
                "agreement": by_turn[(turn, True)],
                "disagreement": by_turn[(turn, False)],
            }
            for turn in sorted({row["turn"] for row in rows})
        },
        "disagreement_rows": disagreements,
    }
    atomic_write_json(HERE / "scorer-validation.json", payload)
    report = f"""# Mechanical scorer validation

The frozen scorer agreed with the historical Claude tool-use judgment on
**{agreements}/{total} target turns ({payload['agreement_percent']:.3f}%)**.
The preregistered 99.5% gate **{'passed' if payload['gate_passed'] else 'did not pass'}**.

The validation set is all turns 12 and 15 from 150 BaseTen BF16, 150 local
FP8-KV, and 150 local BF16-KV completed conversations. The direct replay's
primary outcome is stricter about multiple calls: exactly one correct call is
required. Historical transcripts already contain the benchmark's streamed-call
normalization and deduplication, so the validation tests parity after that seam.

## Categories

```json
{json.dumps(payload['categories'], indent=2, sort_keys=True)}
```

## Disagreements

```json
{json.dumps(disagreements, indent=2, ensure_ascii=False)}
```
"""
    atomic_write_text(HERE / "scorer-validation.md", report)
    print(json.dumps({key: payload[key] for key in ("total", "agreements", "disagreements", "agreement_percent", "gate_passed")}, indent=2))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
