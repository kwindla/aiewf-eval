#!/usr/bin/env python3
"""Offline validation for the frozen Gemma 4 26B A4B dots bundle."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("gemma4_dots_collect", HERE / "collect.py")
assert SPEC is not None and SPEC.loader is not None
collect = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collect
SPEC.loader.exec_module(collect)


def load_workflow_modules():
    sys.modules["collect"] = collect
    judge_spec = importlib.util.spec_from_file_location(
        "gemma4_dots_judge", HERE / "judge_stage.py"
    )
    assert judge_spec is not None and judge_spec.loader is not None
    judge = importlib.util.module_from_spec(judge_spec)
    sys.modules["judge_stage"] = judge
    judge_spec.loader.exec_module(judge)

    analysis_spec = importlib.util.spec_from_file_location(
        "gemma4_dots_analysis", HERE / "analyze_stage.py"
    )
    assert analysis_spec is not None and analysis_spec.loader is not None
    analysis = importlib.util.module_from_spec(analysis_spec)
    analysis_spec.loader.exec_module(analysis)
    return judge, analysis


def transcript_row(
    turn: int,
    *,
    end_session: bool = False,
    recovery: bool = False,
) -> dict[str, object]:
    calls: list[dict[str, str]] = []
    if end_session:
        calls.append({"name": "end_session", "arguments": "{}"})
    return {
        "model_name": collect.MODEL,
        "turn": turn,
        "recovery_turn": recovery,
        "assistant_text": "ok",
        "tool_calls": calls,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class BundleTests(unittest.TestCase):
    def test_frozen_configuration_schedule_and_hashes(self) -> None:
        config = collect.validate_configuration()
        schedule = collect.validate_schedule()
        collect.validate_headers()
        collect.validate_source_hashes()
        self.assertEqual(config["initial_target_per_arm"], 10)
        self.assertEqual(config["promoted_target_per_arm"], 30)
        self.assertEqual(len(schedule), 60)
        self.assertEqual(schedule[:20], collect.selected_schedule(schedule, "initial"))
        self.assertEqual(schedule, collect.selected_schedule(schedule, "full"))

    def test_exact_arm_environment_and_no_secret_leakage(self) -> None:
        leaked = {
            "MTE_FILLER_DOTS": "17",
            "MTE_FILLER_TOKEN": "!",
            "MTE_FILLER_POSITION": "prefix",
            "MTE_VLLM_THINKING": "1",
            "VLLM_BASE_URL": "https://wrong.invalid/v1",
            "VLLM_API_KEY": "old-secret",
            "BASETEN_API_KEY": "provider-secret",
        }
        with patch.dict(os.environ, leaked, clear=False):
            control = collect.child_environment("new-secret", "nofiller")
            dots = collect.child_environment("new-secret", "dots96")

        self.assertNotIn("MTE_FILLER_DOTS", control)
        self.assertNotIn("MTE_FILLER_TOKEN", control)
        self.assertNotIn("MTE_FILLER_POSITION", control)
        self.assertNotIn("BASETEN_API_KEY", control)
        self.assertEqual(control["VLLM_BASE_URL"], collect.ENDPOINT)
        self.assertEqual(control["VLLM_API_KEY"], "new-secret")
        self.assertEqual(control["MTE_VLLM_THINKING"], "0")
        self.assertEqual(dots["MTE_FILLER_DOTS"], "96")
        self.assertEqual(dots["MTE_FILLER_TOKEN"], ".")
        self.assertEqual(dots["MTE_FILLER_POSITION"], "suffix")

    def test_transcript_classification_keeps_short_model_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            strict = directory / "strict"
            strict.mkdir()
            write_jsonl(
                strict / "transcript.jsonl",
                [
                    transcript_row(turn, end_session=turn == 29)
                    for turn in range(30)
                ],
            )
            result = collect.inspect_transcript(strict)
            self.assertEqual(result["classification"], "strict_complete")
            self.assertEqual(result["scheduled_rows"], 30)
            self.assertEqual(result["end_session_turn"], 29)

            aborted = directory / "aborted"
            aborted.mkdir()
            write_jsonl(
                aborted / "transcript.jsonl",
                [
                    transcript_row(turn, end_session=turn == 4)
                    for turn in range(5)
                ],
            )
            result = collect.inspect_transcript(aborted)
            self.assertEqual(result["classification"], "model_abort")
            self.assertEqual(result["response_turns"], 5)

            recovered = directory / "recovered"
            recovered.mkdir()
            rows = [transcript_row(turn) for turn in range(7)]
            rows.append(transcript_row(30, end_session=True, recovery=True))
            write_jsonl(recovered / "transcript.jsonl", rows)
            result = collect.inspect_transcript(recovered)
            self.assertEqual(result["classification"], "recovery_end_session")
            self.assertEqual(result["scheduled_rows"], 7)
            self.assertEqual(result["end_session_turn"], 30)

    def test_zero_response_is_not_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_jsonl(
                directory / "transcript.jsonl",
                [
                    {
                        "model_name": collect.MODEL,
                        "turn": 0,
                        "assistant_text": "",
                        "tool_calls": [],
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "no valid model response"):
                collect.inspect_transcript(directory)
        self.assertIsNotNone(
            collect.INFRASTRUCTURE_ZERO_RESPONSE.search(
                "APIConnectionError: upstream unavailable"
            )
        )

    def test_runtime_provenance_distinguishes_exact_dots_arm(self) -> None:
        common = "\n".join(
            (
                "Recovery nudges enabled=True",
                "Tool call dedupe enabled=True",
                "Tool result run_llm enabled=False",
                "Using vllm-openai with "
                f"base_url={collect.ENDPOINT}, model={collect.MODEL}, "
                "thinking=False, thinking_budget=None, T=1.0, "
                "top_p=0.95, top_k=64, max_tokens=8192",
                "Text pipeline idle_timeout_secs=45.0",
            )
        )
        marker = (
            "MTE_FILLER_DOTS active: 96 x '.' filler tokens, "
            "position=suffix (history left filler-free)"
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "run.log").write_text(common + "\n", encoding="utf-8")
            collect.validate_run_provenance(directory, "nofiller")
            with self.assertRaisesRegex(ValueError, "activation evidence"):
                collect.validate_run_provenance(directory, "dots96")
            (directory / "run.log").write_text(
                common + "\n" + marker + "\n", encoding="utf-8"
            )
            collect.validate_run_provenance(directory, "dots96")
            with self.assertRaisesRegex(ValueError, "leaked"):
                collect.validate_run_provenance(directory, "nofiller")

    def test_promotion_record_requires_a_predeclared_trigger_and_hashes(self) -> None:
        valid = {
            "campaign_id": collect.CAMPAIGN_ID,
            "decision_after_n_per_arm": 10,
            "promote_to_n30": True,
            "triggered_rules": ["completion_differs"],
            "aggregates_sha256": "a" * 64,
            "included_runs_sha256": "b" * 64,
            "decided_at": "2026-07-31T12:00:00-07:00",
        }
        with tempfile.TemporaryDirectory() as temporary:
            decision = Path(temporary) / "decision.json"
            decision.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(
                collect.validate_promotion_decision(decision), valid
            )
            valid["triggered_rules"] = ["unregistered_rule"]
            decision.write_text(json.dumps(valid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "valid triggered rule"):
                collect.validate_promotion_decision(decision)

    def test_autoscaling_warmup_and_teardown_are_exact(self) -> None:
        self.assertEqual(collect.ACTIVE_AUTOSCALING["min_replica"], 1)
        self.assertEqual(collect.ACTIVE_AUTOSCALING["max_replica"], 1)
        self.assertEqual(collect.TEARDOWN_AUTOSCALING["min_replica"], 0)
        self.assertEqual(collect.TEARDOWN_AUTOSCALING["max_replica"], 1)
        self.assertEqual(collect.ACTIVE_AUTOSCALING["concurrency_target"], 1)

    def test_model_failure_still_requests_and_confirms_teardown(self) -> None:
        assignment = collect.validate_schedule()[0]
        calls: list[tuple[str, object]] = []

        def record_scale(
            _key: str, settings: dict[str, object], *, event: str
        ) -> None:
            calls.append((event, dict(settings)))

        def record_wait(_key: str, *, active: bool) -> None:
            calls.append(("WAIT", active))

        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(collect, "LOG_DIR", Path(temporary) / "logs"),
                patch.object(collect, "validate_manifests", return_value=[]),
                patch.object(collect, "selected_schedule", return_value=[assignment]),
                patch.object(collect, "append_log"),
                patch.object(collect, "set_autoscaling", side_effect=record_scale),
                patch.object(collect, "wait_for_state", side_effect=record_wait),
                patch.object(
                    collect,
                    "run_model_attempt",
                    side_effect=RuntimeError("synthetic model failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic model failure"):
                    collect.execute_collection(
                        config=collect.validate_configuration(),
                        schedule=[assignment],
                        stage="initial",
                        api_key="not-a-real-key",
                    )

        self.assertEqual(
            calls,
            [
                ("SCALE_UP", collect.ACTIVE_AUTOSCALING),
                ("WAIT", True),
                ("TEARDOWN", collect.TEARDOWN_AUTOSCALING),
                ("WAIT", False),
            ],
        )

    def test_default_cli_is_read_only(self) -> None:
        stdout = io.StringIO()
        with (
            patch.object(sys, "argv", [str(HERE / "collect.py")]),
            patch.object(
                collect,
                "api_json",
                side_effect=AssertionError("network must not be called"),
            ) as api_mock,
            patch.object(
                collect.subprocess,
                "Popen",
                side_effect=AssertionError("model must not be called"),
            ) as process_mock,
            contextlib.redirect_stdout(stdout),
        ):
            self.assertEqual(collect.main(), 0)
        api_mock.assert_not_called()
        process_mock.assert_not_called()
        self.assertIn("Read-only preflight only", stdout.getvalue())

    def test_judge_identity_worker_cap_and_secure_child_environment(self) -> None:
        judge, _ = load_workflow_modules()
        self.assertEqual(
            judge.actual_judge_identity(),
            ("claude-opus-4-5", "claude-agent-sdk-v4-turn-taking"),
        )
        leaked = {
            "BASETEN_API_KEY": "must-not-leak",
            "VLLM_API_KEY": "must-not-leak",
            "OPENAI_API_KEY": "must-not-leak",
            "GOOGLE_API_KEY": "must-not-leak",
            "MTE_VLLM_THINKING": "1",
        }
        with patch.dict(os.environ, leaked, clear=False):
            env = judge.judge_environment("anthropic-only")
        self.assertEqual(env["ANTHROPIC_API_KEY"], "anthropic-only")
        self.assertEqual(env["PYTHON_DOTENV_DISABLED"], "1")
        self.assertEqual(
            [name for name in env if name.endswith("_API_KEY")],
            ["ANTHROPIC_API_KEY"],
        )
        self.assertFalse(any(name.startswith("MTE_") for name in env))

    def test_judge_default_cli_never_executes_a_child(self) -> None:
        judge, _ = load_workflow_modules()
        stdout = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [str(HERE / "judge_stage.py"), "--stage", "initial"],
            ),
            patch.object(judge, "load_stage_entries", return_value=[]),
            patch.object(
                judge.subprocess,
                "run",
                side_effect=AssertionError("judge child must not run"),
            ) as child_mock,
            contextlib.redirect_stdout(stdout),
        ):
            self.assertEqual(judge.main(), 0)
        child_mock.assert_not_called()
        self.assertIn("Read-only preflight only", stdout.getvalue())

    @staticmethod
    def synthetic_conversation(
        pair: int,
        arm: str,
        *,
        failures: tuple[int, ...] = (),
        complete: bool = True,
    ) -> dict[str, object]:
        failure_set = set(failures)
        passes = [int(turn not in failure_set) for turn in range(30)]
        errors = [1 - value for value in passes]
        return {
            "pair": pair,
            "slot": f"synthetic-{pair}-{arm}",
            "arm": arm,
            "run_dir": "synthetic",
            "classification": "strict_complete" if complete else "model_abort",
            "complete": int(complete),
            "observed_turns": 30 - len(failures),
            "missing_turns": len(failures),
            "metrics": {
                "strict_pass": passes,
                "any_error": errors,
                "tool_error": errors,
                "instruction_error": errors,
                "kb_error": errors,
            },
            "turn_taking_errors": errors,
            "latencies": [500.0],
            "transcript_sha256": "a" * 64,
            "judgment_sha256": "b" * 64,
            "summary_sha256": "c" * 64,
        }

    def test_paired_fixed_denominator_bootstrap_and_concentration(self) -> None:
        _, analysis = load_workflow_modules()
        controls = [
            self.synthetic_conversation(pair, "nofiller")
            for pair in range(1, 11)
        ]
        dots = [
            self.synthetic_conversation(pair, "dots96", failures=(12, 13, 14))
            for pair in range(1, 11)
        ]
        summary = analysis.summarize_arm(
            dots, iterations=1_000, seed=123
        )
        first = analysis.paired_bootstrap_effect(
            controls,
            dots,
            "strict_pass",
            iterations=1_000,
            seed=456,
        )
        second = analysis.paired_bootstrap_effect(
            controls,
            dots,
            "strict_pass",
            iterations=1_000,
            seed=456,
        )
        self.assertEqual(summary["fixed_turn_denominator"], 300)
        self.assertEqual(summary["metrics"]["strict_pass"]["count"], 270)
        self.assertEqual(
            summary["error_concentration"]["any_error"]
            ["top_3_turn_error_share_percent"],
            100.0,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["dots_minus_control_points"], -10.0)
        self.assertAlmostEqual(first["paired_bootstrap_95_low"], -10.0)
        self.assertAlmostEqual(first["paired_bootstrap_95_high"], -10.0)

    def test_built_conversation_satisfies_included_manifest_schema(self) -> None:
        _, analysis = load_workflow_modules()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            transcript = run_dir / "transcript.jsonl"
            write_jsonl(
                transcript,
                [
                    {
                        **transcript_row(turn, end_session=turn == 29),
                        "ttfb_ms": 500.0,
                    }
                    for turn in range(30)
                ],
            )
            write_jsonl(
                run_dir / "claude_judged.jsonl",
                [
                    {
                        "turn": turn,
                        "scores": {
                            "tool_use_correct": True,
                            "instruction_following": True,
                            "kb_grounding": True,
                            "turn_taking": True,
                        },
                    }
                    for turn in range(30)
                ],
            )
            entry = {
                "pair": 1,
                "slot": "synthetic-1",
                "arm": "nofiller",
                "run_dir": run_dir,
                "run_dir_text": "synthetic-run",
                "classification": "strict_complete",
                "turns": list(range(30)),
                "transcript": transcript,
                "transcript_sha256": "a" * 64,
            }
            with (
                patch.object(
                    analysis.judge_stage,
                    "validate_outputs",
                    return_value=(True, ""),
                ),
                patch.object(
                    analysis.judge_stage,
                    "sha256",
                    return_value="b" * 64,
                ),
            ):
                conversation = analysis.build_conversation(entry)

        self.assertEqual(conversation["strict_passes"], 30)
        self.assertTrue(
            set(analysis.INCLUDED_FIELDS).issubset(conversation),
            set(analysis.INCLUDED_FIELDS) - set(conversation),
        )

    def test_promotion_rule_uses_exact_registered_trigger_names(self) -> None:
        _, analysis = load_workflow_modules()
        controls = [
            self.synthetic_conversation(pair, "nofiller")
            for pair in range(1, 11)
        ]
        dots = [
            self.synthetic_conversation(pair, "dots96", failures=(12,))
            for pair in range(1, 11)
        ]
        control_summary = analysis.summarize_arm(
            controls, iterations=200, seed=1
        )
        dots_summary = analysis.summarize_arm(
            dots, iterations=200, seed=2
        )
        effects = {
            "strict_pass": {
                "dots_minus_control_points": -3.333333333333333,
                "paired_bootstrap_95_low": -3.333333333333333,
                "paired_bootstrap_95_high": -3.333333333333333,
            }
        }
        decision = analysis.evaluate_promotion(
            "initial", effects, control_summary, dots_summary
        )
        self.assertTrue(decision["promote_to_n30"])
        self.assertEqual(
            set(decision["triggered_rules"]),
            {
                "ci_excludes_zero",
                "absolute_effect_ge_3_and_aligned_same_turn_recurs_ge_3",
            },
        )
        self.assertFalse(decision["collection_launched"])

    def test_reviewed_promotion_is_hash_linked_and_collector_compatible(self) -> None:
        _, analysis = load_workflow_modules()
        result = {
            "stage": "initial",
            "promotion_evaluation": {
                "evaluated": True,
                "promote_to_n30": True,
                "triggered_rules": ["ci_excludes_zero"],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            analysis_dir = Path(temporary)
            with (
                patch.object(analysis, "ANALYSIS_DIR", analysis_dir),
                patch.object(analysis, "ROOT", analysis_dir),
            ):
                paths = analysis.stage_paths("initial")
                paths["json"].write_text("{}\n", encoding="utf-8")
                paths["included"].write_text("slot\n", encoding="utf-8")
                decision_path = analysis.write_reviewed_promotion(
                    result, paths, "Offline Reviewer"
                )
                self.assertIsNotNone(decision_path)
                payload = collect.validate_promotion_decision(decision_path)
                self.assertEqual(payload["reviewed_by"], "Offline Reviewer")
                self.assertEqual(
                    payload["aggregates_sha256"],
                    analysis.judge_stage.sha256(paths["json"]),
                )
                self.assertFalse((analysis_dir / ".collection.lock").exists())

    def test_report_renders_denominator_pair_ci_and_error_concentration(self) -> None:
        _, analysis = load_workflow_modules()
        controls = [
            self.synthetic_conversation(pair, "nofiller")
            for pair in range(1, 11)
        ]
        dots = [
            self.synthetic_conversation(pair, "dots96", failures=(12,))
            for pair in range(1, 11)
        ]
        control_summary = analysis.summarize_arm(
            controls, iterations=200, seed=10
        )
        dots_summary = analysis.summarize_arm(
            dots, iterations=200, seed=20
        )
        effect = analysis.paired_bootstrap_effect(
            controls,
            dots,
            "strict_pass",
            iterations=200,
            seed=30,
        )
        promotion = analysis.evaluate_promotion(
            "initial",
            {"strict_pass": effect},
            control_summary,
            dots_summary,
        )
        report = analysis.render_markdown(
            {
                "stage": "initial",
                "arms": {"nofiller": control_summary, "dots96": dots_summary},
                "effects": {"strict_pass": effect},
                "promotion_evaluation": promotion,
            }
        )
        self.assertIn("Missing future turns are errors", report)
        self.assertIn("paired bootstrap 95% CI", report)
        self.assertIn("top-3 turn share", report)
        self.assertIn("collection", report.lower())

    def test_analysis_default_cli_does_not_write_or_launch_collection(self) -> None:
        _, analysis = load_workflow_modules()
        result = {
            "effects": {
                "strict_pass": {
                    "dots_minus_control_points": 0.0,
                    "paired_bootstrap_95_low": -1.0,
                    "paired_bootstrap_95_high": 1.0,
                }
            },
            "promotion_evaluation": {"promote_to_n30": False},
        }
        stdout = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [str(HERE / "analyze_stage.py"), "--stage", "initial"],
            ),
            patch.object(analysis, "analyze", return_value=(result, [{}] * 20)),
            patch.object(
                analysis,
                "write_outputs",
                side_effect=AssertionError("read-only analysis must not write"),
            ) as write_mock,
            patch.object(
                analysis.collect,
                "main",
                side_effect=AssertionError("collector must not run"),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            self.assertEqual(analysis.main(), 0)
        write_mock.assert_not_called()
        self.assertIn("Read-only analysis only", stdout.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
