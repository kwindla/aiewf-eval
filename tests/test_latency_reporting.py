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
            {
                "turn": 0,
                "server_ttfb_ms": 900,
                "raw_ttfb_ms": 850,
                "wav_v2v_ms": 1200,
                "silent_pad_silero_ms": 80,
            },
            {
                "turn": 1,
                "server_ttfb_ms": 60,
                "raw_ttfb_ms": 55,
                "wav_v2v_ms": 400,
                "silent_pad_silero_ms": 40,
                "has_tool_call": True,
            },
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

    # Token-latency filtering must not drop first-turn accuracy or audio metrics.
    assert data["turn_pass"] == {"correct": 1, "total": 1}
    assert data["scores"]["tool_use_correct"] == {"correct": 1, "total": 1}
    assert data["non_tool_v2vs"] == [1200]
    assert data["tool_v2vs"] == [400]
    assert data["silence_pads"] == [80, 40]


def test_transcript_selection_uses_turn_index_and_preserves_recovery(tmp_path):
    rows = [
        {"turn": "0", "ttfb_ms": 900},
        {"turn": 0, "recovery_turn": True, "ttfb_ms": 70},
        {"turn": 1, "ttfb_ms": None},
        {"turn": 2, "ttfb_ms": 60},
        {"ttfb_ms": 80},
    ]
    (tmp_path / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    assert aggregate_existing_runs.load_ttfb_from_transcript(tmp_path) == [70, 60, 80]
    assert run_comprehensive_eval.load_ttfb_from_transcript(tmp_path) == [70, 60, 80]


def test_summary_excludes_first_response_for_each_conversation(tmp_path, monkeypatch):
    runs = [tmp_path / "first", tmp_path / "second"]
    for run in runs:
        run.mkdir()
        (run / "claude_judged.jsonl").write_text(
            json.dumps(
                {
                    "turn": 0,
                    "scores": {
                        "tool_use_correct": True,
                        "instruction_following": True,
                        "kb_grounding": True,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    metrics = [
        {"turn": 0, "server_ttfb_ms": 900, "raw_ttfb_ms": 850},
        {"turn": 1, "server_ttfb_ms": 60, "raw_ttfb_ms": 55},
        {"turn": 2, "server_ttfb_ms": None, "raw_ttfb_ms": None},
        {"turn": 30, "server_ttfb_ms": 70, "raw_ttfb_ms": 65},
    ]
    monkeypatch.setattr(
        benchmark_summary.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps({"turns": metrics})
        ),
    )
    data = benchmark_summary.aggregate_runs(runs)
    assert data["ttfts"] == [60, 70, 60, 70]
    assert data["raw_ttfts"] == [55, 65, 55, 65]
    assert data["scores"]["turn_pass"] == {"correct": 2, "total": 2}


def test_missing_first_latency_does_not_exclude_next_response(tmp_path, monkeypatch):
    rows = [{"turn": 0}, {"turn": 1, "ttfb_ms": 60}]
    (tmp_path / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    assert aggregate_existing_runs.load_ttfb_from_transcript(tmp_path) == [60]
    assert run_comprehensive_eval.load_ttfb_from_transcript(tmp_path) == [60]
    (tmp_path / "claude_judged.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        benchmark_summary.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "turns": [
                        {"turn": 0},
                        {"turn": 1, "server_ttfb_ms": 60},
                        {"server_ttfb_ms": 80},
                    ]
                }
            ),
        ),
    )
    data = benchmark_summary.get_run_data(tmp_path)
    assert data["ttfts"] == [60, 80]
    assert data["raw_ttfts"] == []
