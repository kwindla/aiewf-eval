"""Publication checks: counts, scoring scope, prompt identity, and displayed values."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig as Standard
from benchmarks.aiwf_medium_context_speech_live_optimized.config import (
    BenchmarkConfig as Optimized,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/speech-results-2026-09"
AGG = json.loads((DATA / "aggregate.json").read_text())
MANIFEST = json.loads((DATA / "manifest.json").read_text())


def ms(value):
    return "—" if value is None else f"{value:.0f}ms"


def test_readme_speech_tables_match_aggregate_exactly():
    section = (
        (ROOT / "README.md")
        .read_text()
        .split("Speech-to-speech models:\n")[1]
        .split("### Sample Recordings")[0]
    )
    found = [
        line
        for line in section.splitlines()
        if line.startswith("| ") and not line.startswith("| Model")
    ]
    expected = []
    for m in AGG["models"]:
        denominator = m["accuracy_denominator"]
        failures = f"{m['failed_conversations']}/{m['eligible_attempts']} ({m['failure_rate_percent']:.1f}%)"
        cells = [
            m["label"],
            str(m["complete"]),
            failures,
            f"{m['strict_percent']:.1f}%",
        ]
        if m["key"] == "gemini38medium_new":
            cells += [
                ms(m["first_speech_positive"]["mean_ms"]),
                ms(m["first_speech_positive"]["median_ms"]),
                f"{m['first_speech_positive']['n']}/{denominator}",
            ]
        else:
            cells += [
                f"{m['scores'][key]}/{denominator}"
                for key in ["tool_use_correct", "instruction_following", "kb_grounding"]
            ]
            cells += [
                f"{m['scores']['turn_taking']}/{denominator}"
                if m["turn_taking_scored"]
                else "—"
            ]
            cells += [
                ms(m["non_tool"]["median_ms"]),
                ms(m["non_tool"]["max_ms"]),
                ms(m["tool"]["mean_ms"]),
                ms(m["silence"]["mean_ms"]),
            ]
        expected.append("| " + " | ".join(cells) + " |")
    assert found == expected


def test_attempt_ledger_and_metric_denominators():
    selected = []
    for m in AGG["models"]:
        attempts = [r for r in MANIFEST["attempts"] if r["cohort"] == m["key"]]
        counts = Counter(r["disposition"] for r in attempts)
        assert counts["scored"] == m["complete"] == 25
        assert counts["failed"] == m["failed_conversations"]
        assert counts["harness_excluded"] == m["harness_excluded"]
        assert counts["scored"] + counts["failed"] == m["eligible_attempts"]
        assert m["accuracy_denominator"] == m["complete"] * 30
        assert (
            abs(
                m["strict_percent"]
                - 100 * m["scores"]["strict"] / m["accuracy_denominator"]
            )
            < 1e-10
        )
        assert (
            abs(
                m["failure_rate_percent"]
                - 100 * counts["failed"] / m["eligible_attempts"]
            )
            < 1e-10
        )
        assert (
            m["first_speech_positive"]["n"]
            + m["nonpositive_n"]
            + m["missing_latency_n"]
            == m["accuracy_denominator"]
        )
        for r in attempts:
            if r["disposition"] == "scored":
                assert r["completed_turns"] == 30
                selected.append(r["run_path"])
        for f in m["failure_details"]:
            assert f["turn_number"] == f["turn_index"] + 1
            match = next(r for r in attempts if r["run_path"] == f["run_dir"])
            assert match["disposition"] == "failed"
            assert match["failure_turn_number"] == f["turn_number"]
    assert len(selected) == len(set(selected)) == 225
    assert len(MANIFEST["attempts"]) == 245


def test_reconstructed_judges_do_not_claim_turn_taking_scores():
    for m in AGG["models"]:
        if m["key"] in ("live_optimized", "gemini38medium_new"):
            assert m["turn_taking_scored"] is False
            assert m["scores"]["turn_taking"] is None
        else:
            assert m["turn_taking_scored"] is True
            assert isinstance(m["scores"]["turn_taking"], int)
    assert all(m["key"] != "gemini38medium_old" for m in AGG["models"])


def test_published_prompt_hashes_match_benchmark_definitions():
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    assert (
        digest(Standard.system_instruction) == AGG["policy"]["standard_prompt_sha256"]
    )
    assert digest(Standard.judge_system_prompt) == AGG["policy"]["judge_prompt_sha256"]
    optimized = next(m for m in AGG["models"] if m["key"] == "live_optimized")[
        "configuration"
    ]
    assert digest(Optimized.frontend_system_instruction) == optimized["frontend_sha256"]
    assert digest(Optimized.backend_system_instruction) == optimized["backend_sha256"]


def test_raw_inventory_does_not_claim_archival():
    inventory = MANIFEST["referenced_runs"]
    assert len({r["run_path"] for r in inventory}) == len(inventory)
    assert all(r["archived"] is False for r in inventory)
    assert MANIFEST["raw_runs"]["archived"] is False
    assert sum(r["reference_scope"] == "committed_campaign" for r in inventory) == 398
    assert (
        sum(r["reference_scope"] == "additional_disk_diagnostic" for r in inventory)
        == 57
    )
    assert {r["run_path"] for r in MANIFEST["attempts"]} <= {
        r["run_path"] for r in inventory
    }
    for r in inventory:
        assert (
            r["run_path"].startswith("runs/")
            and r["run_id"] == Path(r["run_path"]).name
        )
    for artifact in MANIFEST["artifacts"]:
        assert artifact["path"].startswith(("git/", "disk/"))
        assert re.fullmatch("[0-9a-f]{64}", artifact["sha256"])


def test_additional_readme_numeric_claims_trace_to_aggregate():
    text = (ROOT / "README.md").read_text()
    by = {m["key"]: m for m in AGG["models"]}
    opt = by["live_optimized"]
    assert (
        f"mean is {opt['first_speech_positive']['mean_ms']:.0f}ms, median {opt['first_speech_positive']['median_ms']:.0f}ms, with {opt['first_speech_positive']['n']}/{opt['accuracy_denominator']}"
        in text
    )
    regular = by["gemini38regular"]
    assert (
        f"{regular['scores']['strict']}/{regular['accuracy_denominator']} strict passes"
        in text
    )
    assert (
        f"mean {regular['first_speech_positive']['mean_ms']:.0f}ms, median {regular['first_speech_positive']['median_ms']:.0f}ms, {regular['first_speech_positive']['n']}/{regular['accuracy_denominator']}"
        in text
    )
    note = AGG["readme_numeric_notes"]["regular_gemini"]
    assert (
        f"{note['retained_before_duplicate_fix']} earlier completed conversations"
        in text
    )
    assert (
        f"{regular['failed_conversations']}/{regular['eligible_attempts']} eligible attempts, not {regular['failed_conversations'] + regular['harness_excluded']}/{note['total_captures']}"
        in text
    )
    old = AGG["readme_numeric_notes"]["historical_live_validation"]
    assert f"scored {old['single_strict_passes']}/{old['single_turns']}" in text
    assert f"plus {old['added_conversations']} additions" in text
    assert (
        f"{old['diagnostic_complete']}/{old['diagnostic_attempts']} reaching all 30 transcript turns"
        in text
    )
    historical = AGG["readme_numeric_notes"]["historical_gemini31"]
    assert (
        f"{historical['no_response_free']}/{historical['attempts']} no-response-free"
        in text
    )
    assert f"WebSocket {historical['websocket_reconnect_code']}" in text


def test_result_document_links_resolve():
    # Only the newly published speech section, not unrelated historical README links.
    readme = (ROOT / "README.md").read_text()
    speech = readme.split("Speech-to-speech models:\n")[1].split("## Quick Start")[0]
    documents = [(ROOT / "README.md", speech)] + [
        (p, p.read_text()) for p in DATA.glob("*.md")
    ]
    for origin, text in documents:
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://")):
                continue
            path, _, anchor = target.partition("#")
            resolved = (origin.parent / path).resolve() if path else origin
            assert resolved.exists(), (origin, target)
            if anchor:
                headings = re.findall(r"^#+\s+(.+)$", resolved.read_text(), re.M)
                slugs = [
                    re.sub(r"[^\w\s-]", "", h.lower()).replace(" ", "-")
                    for h in headings
                ]
                assert anchor in slugs, (origin, target)
