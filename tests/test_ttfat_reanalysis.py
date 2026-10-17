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


def test_readme_latency_cells_match_audit_including_explicit_exceptions():
    audit = json.loads(
        (ROOT / "docs/ttfat-reanalysis-2026-10/readme-audit.json").read_text()
    )
    text = (ROOT / "README.md").read_text()
    assert len(audit) == 47
    for row in audit:
        line = next(
            line
            for line in text.splitlines()
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
