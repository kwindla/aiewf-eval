import json
from types import SimpleNamespace

from scripts import aggregate_existing_runs, benchmark_summary, run_comprehensive_eval


def _write_transcript(run_dir):
    rows = [
        {"turn": 0, "ttfb_ms": 900},
        {"turn": 1, "ttfb_ms": 60},
        {"turn": 30, "recovery_turn": True, "ttfb_ms": 70},
    ]
    (run_dir / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_transcript_aggregators_exclude_first_response_but_keep_recovery(tmp_path):
    _write_transcript(tmp_path)

    expected = [60, 70]
    assert aggregate_existing_runs.load_ttfb_from_transcript(tmp_path) == expected
    assert run_comprehensive_eval.load_ttfb_from_transcript(tmp_path) == expected


def test_benchmark_summary_excludes_first_response_latency(tmp_path, monkeypatch):
    judged = {
        "turn": 0,
        "scores": {
            "tool_use_correct": True,
            "instruction_following": True,
            "kb_grounding": True,
        },
    }
    (tmp_path / "claude_judged.jsonl").write_text(
        json.dumps(judged) + "\n", encoding="utf-8"
    )
    metrics = {
        "turns": [
            {"turn": 0, "server_ttfb_ms": 900, "raw_ttfb_ms": 850},
            {"turn": 1, "server_ttfb_ms": 60, "raw_ttfb_ms": 55},
            {"turn": 30, "server_ttfb_ms": 70, "raw_ttfb_ms": 65},
        ]
    }
    monkeypatch.setattr(
        benchmark_summary.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(metrics)
        ),
    )

    data = benchmark_summary.get_run_data(tmp_path)

    assert data["ttfts"] == [60, 70]
    assert data["raw_ttfts"] == [55, 65]
