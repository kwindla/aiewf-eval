"""Validate all current public cells without the maintainer's campaign evidence."""

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "docs/model-evaluations-2026-10"
DIMS = ("tool_use_correct", "instruction_following", "kb_grounding")


def summary(name):
    return json.loads((BUNDLE / name / "summary.json").read_text())["data"]


def pct(value):
    return "—" if value is None else f"{value:.1f}%"


def ms(value):
    return "—" if value is None else f"{value:.0f}ms"


def table(text, header):
    section = text[text.index(header):].split("\n\n", 1)[0]
    return {
        cells[0]: cells
        for line in section.splitlines()[2:]
        if line.startswith("| ")
        for cells in [[cell.strip() for cell in line.strip("|").split("|")]]
    }


def test_every_ranked_text_cell_matches_the_committed_summary():
    models = summary("judge-recovery-context-20261019")["text_models"]
    ranked = [model for model in models if model["complete"] > 0]
    assert len(ranked) == 50
    for filename in ["README.md", "leaderboard-medium-context.md"]:
        rows = table((ROOT / filename).read_text(), "| Model | Pass Rate | Any Error |")
        assert set(rows) == {m["label"] for m in ranked}
        for m in ranked:
            expected = [m["label"], pct(m["strict_percent"]), pct(100 - m["strict_percent"])]
            expected += [pct(m["dimension_error_percent"][d]) for d in DIMS]
            expected += [ms(m["latency"]["positive"][k]) for k in ("p50_ms", "p95_ms", "max_ms")]
            expected += [m["provider"], str(m["complete"])]
            assert rows[m["label"]] == expected


def test_every_voice_cell_matches_the_committed_summary():
    models = summary("judge-recovery-context-20261019")["voice_models"]
    models += summary("nova-2-5-sonic-20261019/user-context-v2")["voice_models"]
    text = (ROOT / "README.md").read_text()
    rows = table(text, "| Model | Selected | Failed conversations | Pass Rate |")
    ordinary = [m for m in models if not m.get("additional_experiment")]
    assert len(rows) == len(ordinary) == 11
    for m in ordinary:
        den = 30 * m["complete"]
        counts = m["complete_only"]["dimension_passes"]
        turn = m.get("turn_taking_passes", m.get("scores", {}).get("turn_taking"))
        if turn is None:
            turn = m.get("scores", {}).get("turn_taking")
        expected = [m["label"], str(m["complete"]),
                    f"{m['failed_conversations']}/{m['eligible_attempts']} ({m['failure_rate_percent']:.1f}%)",
                    pct(m["strict_percent"])]
        expected += [f"{counts[d]}/{den}" for d in DIMS]
        expected += [f"{turn}/{den}" if m["turn_taking_scored"] else "—"]
        expected += [ms(m["non_tool"]["median_ms"]), ms(m["non_tool"]["max_ms"]),
                     ms(m["tool"]["mean_ms"]), ms(m["silence"]["mean_ms"])]
        assert rows[m["label"]] == expected
    extra = table(text, "| Model / configuration | Selected | Failed conversations |")
    for m in models:
        if m.get("additional_experiment"):
            assert extra[m["label"]] == [m["label"], str(m["complete"]),
                f"{m['failed_conversations']}/{m['eligible_attempts']} ({m['failure_rate_percent']:.1f}%)",
                pct(m["strict_percent"]), ms(m["first_speech_positive"]["mean_ms"]),
                ms(m["first_speech_positive"]["median_ms"]), f"{m['first_speech_positive']['n']}/750"]


def test_publication_copies_match_the_archive_manifest():
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    assert manifest["file_count"] == 9212
    assert len(manifest["archive_sha256"]) == 64
    assert manifest["raw_runs"]["included"] is False
    for source, entry in manifest["public_entrypoints"].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == entry["published_sha256"]
    for source, entry in manifest["publication_sources"].items():
        path = ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["published_sha256"]
        if path.suffix == ".json":
            value = json.loads(path.read_text())
            assert value["source_artifact"] == source
            assert value["source_sha256"] == entry["source_sha256"]
            assert not any(isinstance(value["data"].get(key), (list, dict))
                           for key in ("cases", "ledger", "conversations", "attempts"))


def test_publication_links_resolve_without_a_local_archive():
    paths = list(BUNDLE.rglob("*.md"))
    paths += [ROOT / name for name in ["README.md", "leaderboard-medium-context.md",
              "docs/text-cohort-sizes-2026-10-17.md", "docs/text-rows-archived-2026-10-17.md",
              "docs/groq-gpt-oss-120b-archived-2026-10-17.md"]]
    for path in paths:
        for url in re.findall(r"\[[^\]]+\]\(([^\s)]+)\)", path.read_text()):
            if re.match(r"^(?:https?:|mailto:|#)", url):
                continue
            target = unquote(url.split("#", 1)[0].strip("<>"))
            assert (path.parent / target).exists(), (path.relative_to(ROOT), url)


@pytest.mark.local_evidence
def test_publication_is_reproduced_from_verified_original_bytes():
    from scripts.generate_campaign_publication import EvidenceArchive, default_archive, render
    from scripts.campaign_publication_links import public_links
    archive = EvidenceArchive(default_archive())
    try:
        assert "ops/judge-refresh-opus55-20261017/historical/README.md" in archive.members
        for source, entry in archive.metadata["publication_sources"].items():
            raw = archive.read(source)
            assert hashlib.sha256(raw).hexdigest() == entry["source_sha256"]
            assert (ROOT/source).read_bytes() == raw
            assert render(source, raw, entry["path"]) == (ROOT/entry["path"]).read_bytes()
        for source in archive.metadata["public_entrypoints"]:
            raw = archive.read(source)
            expected = public_links(raw.decode(), source).encode() if source.endswith('.md') else raw
            assert (ROOT/source).read_bytes() == expected
        # Prove that the tiny CI fixture came from the frozen legacy formatter.
        import importlib.util
        source = "ops/judge-recovery-context-20261019/legacy/src/multi_turn_eval/judging/claude_judge.py"
        assert (ROOT/source).read_bytes() == archive.read(source)
        spec = importlib.util.spec_from_file_location("multi_turn_eval.judging.fixture_provenance", ROOT/source)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        rows = [dict(turn=n, user_text="Please submit it.", assistant_text="Done.", tool_calls=[]) for n in [0, 1]]
        assert legacy.format_turns_for_claude(rows, [{}, {}]) == (ROOT/"tests/fixtures/judge-no-recovery.txt").read_text()
    finally:
        archive.close()


def test_restore_preserves_current_code_and_rejects_conflicting_evidence(tmp_path, monkeypatch):
    from scripts import generate_campaign_publication as generator
    from types import SimpleNamespace
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    monkeypatch.setattr(generator.subprocess, "check_output", lambda *a, **k: b"ops/method.py\0README.md\0")
    (tmp_path/"ops").mkdir()
    (tmp_path/"ops/method.py").write_bytes(b"current code")
    (tmp_path/"README.md").write_bytes(b"current publication")
    archive = SimpleNamespace(members={"ops/method.py": {}, "ops/data.json": {}, "README.md": {}}, read=lambda name: b"old evidence")
    assert generator.restore(archive) == 1
    assert (tmp_path/"ops/method.py").read_bytes() == b"current code"
    assert (tmp_path/"README.md").read_bytes() == b"current publication"
    assert (tmp_path/"ops/data.json").read_bytes() == b"old evidence"
    (tmp_path/"ops/data.json").write_bytes(b"new evidence")
    with pytest.raises(ValueError, match="Existing evidence differs"):
        generator.restore(archive)
