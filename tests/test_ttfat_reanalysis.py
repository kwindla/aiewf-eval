import pytest
import json
from pathlib import Path

from multi_turn_eval.latency_reporting import include_response_latency

ROOT = Path(__file__).resolve().parents[1]


def test_first_response_selection_keeps_missing_and_recovery_semantics():
    assert not include_response_latency(0)
    assert include_response_latency(1)
    assert include_response_latency(30)
    assert include_response_latency(None)
    assert include_response_latency(0, recovery=True)


@pytest.mark.local_evidence
def test_historical_latency_cells_match_immutable_audit():
    audit = json.loads(
        (ROOT / "docs/ttfat-reanalysis-2026-10/readme-audit.json").read_text()
    )
    text = (ROOT / "ops/judge-refresh-opus55-20261017/historical/README.md").read_text()
    archive = (ROOT / "docs/groq-gpt-oss-120b-archived-2026-10-17.md").read_text()
    archive += (ROOT / "docs/text-rows-archived-2026-10-17.md").read_text()
    assert len(audit) == 47
    for row in audit:
        selected = archive if row["model"] in {
            "gpt-oss-120b (groq)",
            "claude-fable-5 (low)",
            "claude-fable-5 (default)",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-sonnet-5",
        } else text
        line = next(
            line
            for line in selected.splitlines()
            if line.startswith("| ")
            and line.split("|")[1].strip().replace("**", "") == row["model"]
        )
        cells = [x.strip().replace("**", "") for x in line.strip("|").split("|")]
        assert [int(x.removesuffix("ms")) for x in cells[6:9]] == row["after_ms"]
        if row["status"] in ("updated", "verified"):
            assert row["source"]
            assert [round(x) for x in row["values_ms"]] == row["after_ms"]
        else:
            assert row["before_ms"] == row["after_ms"]
