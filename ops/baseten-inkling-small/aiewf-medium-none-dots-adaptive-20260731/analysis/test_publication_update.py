#!/usr/bin/env python3
"""Offline tests for the Inkling Small publication handoff."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "publication_update.py"
SPEC = importlib.util.spec_from_file_location("inkling_small_publication_update", MODULE_PATH)
assert SPEC and SPEC.loader
publication = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = publication
SPEC.loader.exec_module(publication)


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm(pass_rate: float, p50: float, completion: float = 100.0) -> dict:
    return {
        "n_conversations": 30,
        "fixed_turn_denominator": 900,
        "strict_pass_rate_pct": pass_rate,
        "any_error_rate_pct": 100.0 - pass_rate,
        "tool_use_correct_error_rate_pct": 2.0,
        "instruction_following_error_rate_pct": 5.0,
        "kb_grounding_error_rate_pct": 4.0,
        "strict_protocol_completion_pct": completion,
        "ttfat_ms": {"p50": p50, "p95": p50 + 500, "max": p50 + 1500},
    }


class Fixture:
    def __init__(self, root: Path, stage: int = 6) -> None:
        self.root = root
        self.primary_campaign = root / (
            "ops/baseten-inkling-small/"
            "aiewf-medium-none-low-n30-20260731"
        )
        self.primary_path = self.primary_campaign / "analysis/aggregates.json"
        self.failure_path = self.primary_campaign / "analysis/FAILURE-ANALYSIS.json"
        self.judge_audit_path = self.primary_campaign / "analysis/JUDGE-AUDIT.json"
        self.dots_campaign = root / (
            "ops/baseten-inkling-small/"
            "aiewf-medium-none-dots-adaptive-20260731"
        )
        self.dots_analysis = self.dots_campaign / "analysis"
        self.dots_path = self.dots_analysis / f"stage-{stage}.json"
        self.stage = stage
        self._build_primary()
        self._build_failure_analysis()
        self._build_judge_audit()
        self._build_dots()

    def _build_primary(self) -> None:
        hashed = [
            "configuration.json",
            "canonical.tsv",
            "analysis/analyze.py",
            "judging/COMPLETE.json",
        ]
        for relative in hashed:
            write(self.primary_campaign / relative, f"fixture:{relative}\n")
        payload = {
            "schema_version": 1,
            "artifact_status": "FINAL",
            "protocol": {
                "campaign_id": publication.CAMPAIGN_ID,
                "benchmark": "aiwf_medium_context",
                "model": publication.MODEL,
                "provider": publication.PROVIDER_SOURCE,
                "endpoint": "https://inference.baseten.co/v1",
                "conversations_per_arm": 30,
                "scheduled_turns_per_conversation": 30,
                "fixed_turn_denominator_per_arm": 900,
                "strict_pass_definition": (
                    "tool_use_correct AND instruction_following AND kb_grounding"
                ),
                "arm_ci_method": "whole-conversation nonparametric bootstrap",
            },
            "input_hashes": {
                relative: digest(self.primary_campaign / relative)
                for relative in hashed
            },
            "arms": {"none": arm(91.0, 725.4), "low": arm(88.0, 940.2)},
        }
        write(self.primary_path, json.dumps(payload, indent=2) + "\n")

    def _build_failure_analysis(self) -> None:
        analyzer = self.primary_campaign / "analysis/failure_analysis.py"
        write(analyzer, "# fixture failure analyzer\n")
        run_inputs = {}
        for index in range(1, 61):
            slot = f"IS-{index:02d}"
            run_dir = self.root / f"runs/aiwf_medium_context/fixture-{index:02d}"
            transcript = run_dir / "transcript.jsonl"
            run_log = run_dir / "run.log"
            write(transcript, f'{{"turn": 0, "slot": "{slot}"}}\n')
            write(run_log, f"fixture run log {slot}\n")
            run_inputs[slot] = {
                "run_dir": str(run_dir.relative_to(self.root)),
                "transcript": str(transcript.relative_to(self.root)),
                "transcript_sha256": digest(transcript),
                "run_log": str(run_log.relative_to(self.root)),
                "run_log_sha256": digest(run_log),
            }
        payload = {
            "schema_version": 1,
            "artifact_status": "RAW_CAUSE_ATTRIBUTION",
            "campaign": publication.CAMPAIGN_ID,
            "model": publication.MODEL,
            "method": {
                "judge_dependency": False,
                "scheduled_turns_per_conversation": 30,
                "recovery_turns_are_not_scheduled": True,
                "baseten_429_idle_definition": (
                    "A short run with BaseTen HTTP 429 and idle timeout."
                ),
            },
            "inputs": {
                "canonical": {
                    "path": str(
                        (self.primary_campaign / "canonical.tsv").relative_to(
                            self.root
                        )
                    ),
                    "sha256": digest(self.primary_campaign / "canonical.tsv"),
                },
                "analyzer": {
                    "path": str(analyzer.relative_to(self.root)),
                    "sha256": digest(analyzer),
                },
                "runs": run_inputs,
            },
            "arms": {
                arm_name: {
                    "conversations": 30,
                    "fixed_turn_denominator": 900,
                    "conversation_causes": {
                        "baseten_429_idle": {"count": count},
                        "unattributed_short": {"count": 0},
                    },
                }
                for arm_name, count in (("none", 12), ("low", 10))
            },
        }
        write(self.failure_path, json.dumps(payload, indent=2) + "\n")

    def _build_judge_audit(self) -> None:
        required = {
            "analysis/aggregates.json": self.primary_path,
            "judging/COMPLETE.json": self.primary_campaign
            / "judging/COMPLETE.json",
            "canonical.tsv": self.primary_campaign / "canonical.tsv",
            "judging/canonical-inputs.tsv": self.primary_campaign
            / "judging/canonical-inputs.tsv",
            "judging/judge-source-sha256.txt": self.primary_campaign
            / "judging/judge-source-sha256.txt",
            "analysis/analyze.py": self.primary_campaign / "analysis/analyze.py",
            "analysis/judge_audit.py": self.primary_campaign
            / "analysis/judge_audit.py",
        }
        for relative, path in required.items():
            if not path.exists():
                write(path, f"fixture:{relative}\n")

        base = {
            "tool_use_correct": True,
            "instruction_following": True,
            "kb_grounding": True,
        }
        changes = []
        for index, (slot, turn) in enumerate(
            (("IS-18", 16), ("IS-18", 17), ("IS-47", 16), ("IS-47", 17))
        ):
            official = dict(base)
            if index == 2:
                official["instruction_following"] = False
            alternate = dict(official)
            alternate["tool_use_correct"] = False
            changes.append(
                {
                    "slot": slot,
                    "arm": "none",
                    "turn": turn,
                    "official_scores": official,
                    "counterfactual_scores": alternate,
                }
            )

        def metric(official: int, delta: int) -> dict:
            counterfactual = official + delta
            return {
                "official_count": official,
                "official_rate_pct": 100 * official / 900,
                "counterfactual_count": counterfactual,
                "counterfactual_rate_pct": 100 * counterfactual / 900,
                "delta_count": delta,
                "delta_percentage_points": 100 * delta / 900,
            }

        payload = {
            "schema_version": 1,
            "artifact_status": "POST_HOC_SENSITIVITY_AUDIT",
            "campaign": publication.CAMPAIGN_ID,
            "model": publication.MODEL,
            "official_artifacts_unchanged": True,
            "policy": {
                "official_judge_model": "claude-opus-4-5",
                "official_judge_version": "claude-agent-sdk-v4-turn-taking",
                "counterfactual": {
                    "status": "post-hoc sensitivity policy, not an official relabeling"
                },
            },
            "input_hashes": {
                relative: digest(path) for relative, path in required.items()
            },
            "label_changes": changes,
            "arms": {
                "none": {
                    "fixed_turn_denominator": 900,
                    "metrics": {
                        "strict_pass": metric(700, -3),
                        "any_error": metric(200, 3),
                        "tool_error": metric(80, 4),
                    },
                },
                "low": {
                    "fixed_turn_denominator": 900,
                    "metrics": {
                        "strict_pass": metric(400, 0),
                        "any_error": metric(500, 0),
                        "tool_error": metric(300, 0),
                    },
                },
            },
        }
        write(self.judge_audit_path, json.dumps(payload, indent=2) + "\n")

    def _build_dots(self) -> None:
        inputs = {
            "control_inputs": self.dots_campaign / "control-inputs.tsv",
            "dots_judge_inputs": self.dots_campaign / "judging/canonical-inputs.tsv",
            "dots_judge_complete": self.dots_campaign
            / f"judging/COMPLETE-stage-{self.stage}.json",
        }
        for name, path in inputs.items():
            write(path, f"fixture:{name}\n")
        dots_pass = 94.0
        control = {
            "conversations": 30,
            "fixed_turn_denominator": 900,
            "rates_percent": {
                "strict_pass": 91.0,
                "any_error": 9.0,
                "tool_error": 2.0,
                "instruction_error": 5.0,
                "kb_error": 4.0,
            },
            "ttfat_ms_observed_responses_only": {"p50": 725.4},
            "strict_completion": {"count": 30, "total": 30, "percent": 100.0},
        }
        dots = {
            "conversations": self.stage,
            "fixed_turn_denominator": self.stage * 30,
            "rates_percent": {
                "strict_pass": dots_pass,
                "any_error": 6.0,
                "tool_error": 1.0,
                "instruction_error": 4.0,
                "kb_error": 2.0,
            },
            "ttfat_ms_observed_responses_only": {"p50": 760.0},
            "strict_completion": {
                "count": self.stage,
                "total": self.stage,
                "percent": 100.0,
            },
        }
        recommendations = {6: "extend_to_10", 10: "extend_to_30", 30: "terminal_at_30"}
        payload = {
            "schema_version": 1,
            "campaign_id": publication.DOTS_CAMPAIGN_ID,
            "stage": self.stage,
            "model": publication.MODEL,
            "provider": publication.PROVIDER_SOURCE,
            "configuration": {
                "control": "frozen primary none arm",
                "treatment": "+96 suffix dots",
                "reasoning_effort": "none",
                "fixed_turns_per_conversation": 30,
            },
            "method": {
                "fixed_denominator": True,
                "missing_future_turns_fail_all_displayed_criteria": True,
                "bootstrap_unit": "whole conversation",
                "bootstrap_iterations": 100_000,
            },
            "inputs": {
                **{
                    key: str(path.relative_to(self.root))
                    for key, path in inputs.items()
                },
                **{
                    f"{key}_sha256": digest(path)
                    for key, path in inputs.items()
                },
            },
            "arms": {"control_none": control, "dots96": dots},
            "effects": {
                "strict_pass": {
                    "dots_minus_control_points": dots_pass - 91.0,
                    "ci95_low": -1.0,
                    "ci95_high": 7.0,
                }
            },
            "adaptive_decision": {
                "evaluated_stage": self.stage,
                "recommendation": recommendations[self.stage],
                "gate_executed": False,
            },
        }
        write(self.dots_path, json.dumps(payload, indent=2) + "\n")


def readme_fixture() -> str:
    return (
        "# Fixture\n\n"
        + publication.README_HEADER
        + "\n"
        + publication.README_SEPARATOR
        + "\n"
        + "| **alpha** | **95.0%** | **5.0%** | **1.0%** | **2.0%** | **3.0%** | **500ms** | **800ms** | **1200ms** | **OpenAI** |\n"
        + "| omega | 80.0% | 20.0% | 5.0% | 10.0% | 8.0% | 1000ms | 2000ms | 3000ms | Other |\n\n"
        + publication.LEGACY_INKLING_PROSE
        + "\n"
    )


def table_row(cells: list[str]) -> str:
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def html_fixture(rows: list[list[str]], labels: list[str], prose: str = "") -> str:
    header = ["Model", "Provider", "Base", "Dots", "Delta", "Completion", "TTFAT", "Runs", "Status"]
    return (
        '<html><section id="primary-screen">'
        '<figure><svg width="100">'
        + "".join(f'<text x="0" y="1" class="lbl">{label}</text>' for label in labels)
        + "</svg></figure><table><tr>"
        + "".join(f"<th>{cell}</th>" for cell in header)
        + "</tr>"
        + "".join(table_row(row) for row in rows)
        + (
            '</table><p class="measure"><b>Run-pool provenance.</b>'
            f"{prose}</p></section></html>"
        )
    )


class PublicationUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = Fixture(self.root)
        self.data = publication.load_publication_data(
            root=self.root,
            primary_path=self.fixture.primary_path,
            dots_analysis=self.fixture.dots_analysis,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_loads_final_inputs_and_uses_none_ttfat(self) -> None:
        self.assertEqual(self.data.dots_stage, 6)
        self.assertEqual(self.data.normalized["provider"], "BaseTen")
        self.assertEqual(self.data.normalized["screen_row"]["none_ttfat_p50_ms"], 725.4)
        self.assertEqual(self.data.normalized["screen_row"]["included_runs"], [30, 6])
        robustness = self.data.normalized["robustness"]
        self.assertEqual(
            robustness["primary_effort_campaign"][
                "baseten_429_idle_short_runs"
            ],
            {"none": 12, "low": 10, "total": 22},
        )
        self.assertEqual(
            robustness["judge_sensitivity"][
                "changed_tool_use_correct_labels"
            ],
            4,
        )
        self.assertLessEqual(
            robustness["judge_sensitivity"][
                "max_abs_arm_rate_change_percentage_points"
            ],
            0.5,
        )
        self.assertEqual(len(self.data.normalized["source_artifacts"]), 4)

    def test_selects_highest_reached_stage(self) -> None:
        Fixture(self.root, stage=30)
        data = publication.load_publication_data(
            root=self.root,
            primary_path=self.fixture.primary_path,
            dots_analysis=self.fixture.dots_analysis,
        )
        self.assertEqual(data.dots_stage, 30)

    def test_missing_stage_and_cross_artifact_drift_fail_closed(self) -> None:
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(RuntimeError, "reached dots-stage"):
            publication.load_publication_data(
                root=self.root,
                primary_path=self.fixture.primary_path,
                dots_analysis=empty,
            )
        payload = json.loads(self.fixture.dots_path.read_text())
        payload["arms"]["control_none"]["ttfat_ms_observed_responses_only"]["p50"] = 999
        write(self.fixture.dots_path, json.dumps(payload))
        with self.assertRaisesRegex(RuntimeError, "TTFAT no longer matches"):
            publication.load_publication_data(
                root=self.root,
                primary_path=self.fixture.primary_path,
                dots_analysis=self.fixture.dots_analysis,
            )

    def test_readme_transform_is_exact_sorted_and_idempotent(self) -> None:
        once = publication.update_readme_text(readme_fixture(), self.data)
        twice = publication.update_readme_text(once, self.data)
        self.assertEqual(once, twice)
        self.assertEqual(once.count("| inkling-small (none) |"), 1)
        self.assertEqual(once.count("| inkling-small (low) |"), 1)
        self.assertIn("| inkling-small (none) | 91.0%", once)
        self.assertIn("| 725ms | 1225ms | 2225ms | BaseTen |", once)
        self.assertNotIn("22/60", once)
        self.assertNotIn("BaseTen HTTP 429", once)
        self.assertNotIn("disputed `tool_use_correct` labels", once)
        self.assertNotIn("30 frozen temporal pairs", once)
        self.assertNotIn("900-turn denominators", once)
        self.assertNotIn("30-conversation", once)
        self.assertNotIn("n=6", once)
        publication.validate_readme_text(once, self.data)

    def test_current_readme_is_compatible_without_writing(self) -> None:
        source = publication.README_PATH.read_text(encoding="utf-8")
        proposed = publication.update_readme_text(source, self.data)
        self.assertNotEqual(source, proposed)
        self.assertEqual(publication.update_readme_text(proposed, self.data), proposed)
        self.assertEqual(publication.README_PATH.read_text(encoding="utf-8"), source)

    def test_canonical_generator_transform_is_compileable_and_idempotent(self) -> None:
        source = publication.GENERATOR_PATH.read_text(encoding="utf-8")
        transformed = publication.transform_generator(source)
        self.assertEqual(publication.transform_generator(transformed), transformed)
        compile(transformed, "synthetic-generator.py", "exec")
        self.assertEqual(transformed.count(publication.GENERATOR_DATA_START), 1)
        self.assertEqual(transformed.count(publication.GENERATOR_DETAIL_START), 1)
        self.assertIn("INKLING_SMALL_ROBUSTNESS", transformed)
        self.assertIn("changed_tool_use_correct_labels", transformed)
        self.assertIn("no more than", transformed)
        before_chart = source[source.index("def fig_dumbbell():") : source.index("def fig_dose():")]
        after_chart = transformed[
            transformed.index("def fig_dumbbell():") : transformed.index("def fig_dose():")
        ]
        self.assertEqual(before_chart, after_chart)

    def test_readme_provider_verifier_transform_is_additive_and_idempotent(self) -> None:
        source = publication.README_VERIFIER_PATH.read_text(encoding="utf-8")
        transformed = publication.transform_readme_verifier(source)
        self.assertEqual(publication.transform_readme_verifier(transformed), transformed)
        compile(transformed, "synthetic-readme-verifier.py", "exec")
        self.assertEqual(transformed.count('"inkling-small (none)": "BaseTen"'), 1)
        self.assertEqual(transformed.count('"inkling-small (low)": "BaseTen"'), 1)
        self.assertIn('"inkling (none)": "BaseTen"', transformed)

    def test_html_validation_preserves_existing_rows_and_chart(self) -> None:
        old_rows = [
            ["alpha", "OpenAI", "95.0", "96.0", "+1.0", "100 / 100", "500", "30 / 30", "increase"],
            ["omega", "Other", "80.0", "79.0", "−1.0", "100 / 100", "1000", "10 / 10", "no effect"],
        ]
        before = html_fixture(old_rows, ["alpha", "omega"])
        screen = self.data.normalized["screen_row"]
        inkling = [
            "inkling-small",
            "BaseTen",
            f"{screen['no_filler_pass_rate_pct']:.1f}",
            f"{screen['dots_pass_rate_pct']:.1f}",
            "+3.0",
            "100 / 100",
            str(round(screen["none_ttfat_p50_ms"])),
            "30 / 6",
            "suggestive",
        ]
        after = (
            "<p>Inkling Small adds a separate fixed-denominator comparison.</p>"
            + html_fixture(
                old_rows + [inkling],
                ["alpha", "omega", "inkling-small"],
                (
                    "22/60 retained attempts ended short. The 4 disputed "
                    "<code>tool_use_correct</code> labels shifted rates by no more "
                    "than 0.5 percentage points."
                ),
            )
        )
        publication.validate_html_update(before, after, self.data)
        with self.assertRaisesRegex(RuntimeError, "methodology prose"):
            publication.validate_html_update(
                before,
                after.replace(
                    "Inkling Small adds a separate fixed-denominator comparison.",
                    "Inkling Small uses a separate comparison.",
                ),
                self.data,
            )
        with self.assertRaisesRegex(RuntimeError, "preserve every existing model row"):
            publication.validate_html_update(
                before,
                (
                    "<p>Inkling Small adds a separate fixed-denominator comparison.</p>"
                    + html_fixture(
                        [old_rows[0], inkling],
                        ["alpha", "omega", "inkling-small"],
                        (
                            "22/60 retained attempts ended short. The 4 disputed "
                            "<code>tool_use_correct</code> labels shifted rates by no more "
                            "than 0.5 percentage points."
                        ),
                    )
                ),
                self.data,
            )

    def test_markdown_validation_requires_report_only_robustness_note(self) -> None:
        text = (
            "prefix\n<!-- N30_PRIMARY_START -->\n"
            "The original 17 rows retain their exploratory-screen order. "
            "22/60 retained attempts ended short. The 4 disputed "
            "`tool_use_correct` labels shifted rates by no more than 0.5 "
            "percentage points.\n\nFlash Lite attempt-policy sensitivity:\n"
            "<!-- N30_PRIMARY_END -->\nsuffix\n"
        )
        publication.validate_markdown_update(text)
        with self.assertRaisesRegex(RuntimeError, "robustness disclosure"):
            publication.validate_markdown_update(
                text.replace("22/60 retained attempts ended", "22 attempts ended")
            )

    def test_robustness_artifacts_fail_closed_on_drift(self) -> None:
        failure = json.loads(self.fixture.failure_path.read_text())
        failure["arms"]["none"]["conversation_causes"]["baseten_429_idle"][
            "count"
        ] = 11
        write(self.fixture.failure_path, json.dumps(failure))
        with self.assertRaisesRegex(RuntimeError, "frozen publication value"):
            publication.load_publication_data(
                root=self.root,
                primary_path=self.fixture.primary_path,
                dots_analysis=self.fixture.dots_analysis,
            )

        self.fixture._build_failure_analysis()
        audit = json.loads(self.fixture.judge_audit_path.read_text())
        audit["arms"]["none"]["metrics"]["tool_error"][
            "delta_percentage_points"
        ] = 0.6
        write(self.fixture.judge_audit_path, json.dumps(audit))
        with self.assertRaisesRegex(RuntimeError, "delta_percentage_points mismatch"):
            publication.load_publication_data(
                root=self.root,
                primary_path=self.fixture.primary_path,
                dots_analysis=self.fixture.dots_analysis,
            )

    def test_generator_environment_excludes_credentials(self) -> None:
        with patch.dict(
            os.environ,
            {"PATH": "/bin", "BASETEN_API_KEY": "secret", "OPENAI_API_KEY": "secret"},
            clear=True,
        ):
            self.assertEqual(publication.local_generator_environment(), {"PATH": "/bin"})


if __name__ == "__main__":
    unittest.main()
