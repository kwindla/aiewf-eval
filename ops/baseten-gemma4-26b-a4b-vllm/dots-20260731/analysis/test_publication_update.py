#!/usr/bin/env python3
"""Offline fixture tests for the Gemma publication updater."""

from __future__ import annotations

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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


publication = load_module("gemma26_publication_update", HERE / "publication_update.py")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(count: int, denominator: int) -> dict:
    rate = count / denominator * 100
    return {
        "count": count,
        "total": denominator,
        "rate_percent": rate,
        "whole_conversation_bootstrap_95": [max(0.0, rate - 2.0), min(100.0, rate + 2.0)],
    }


def arm(n: int, pass_count: int, p50: float) -> dict:
    denominator = n * 30
    return {
        "conversations": n,
        "fixed_turn_denominator": denominator,
        "observed_turns": denominator,
        "missing_turns_counted_as_failures": 0,
        "metrics": {
            "strict_pass": metric(pass_count, denominator),
            "any_error": metric(denominator - pass_count, denominator),
            "tool_error": metric(12 if n == 10 else 36, denominator),
            "instruction_error": metric(18 if n == 10 else 54, denominator),
            "kb_error": metric(3 if n == 10 else 9, denominator),
        },
        "strict_completion": {
            "count": n,
            "total": n,
            "rate_percent": 100.0,
            "whole_conversation_bootstrap_95": [100.0, 100.0],
        },
        "ttfat_ms_observed_responses_only": {
            "count": denominator,
            "p50": p50,
            "p95": p50 + 100,
            "max": p50 + 1000,
        },
    }


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.campaign = root / "ops/baseten-gemma4-26b-a4b-vllm/dots-20260731"
        self.analysis = self.campaign / "analysis"
        for relative in (
            "configuration.json",
            "frozen-order.tsv",
            "judging/judge-source-sha256.txt",
            "analyze_stage.py",
        ):
            write(self.campaign / relative, f"fixture:{relative}\n")
        write(
            self.campaign / "canonical.tsv",
            "slot\n" + "".join(f"G4D-{index:02d}\n" for index in range(1, 21)),
        )

    def _hashes(self, stage: str) -> dict:
        paths = {
            "configuration": self.campaign / "configuration.json",
            "frozen_order": self.campaign / "frozen-order.tsv",
            "canonical": self.campaign / "canonical.tsv",
            "judge_inputs": self.campaign / f"judging/canonical-inputs-{stage}.tsv",
            "judge_complete": self.campaign / f"judging/COMPLETE-{stage}.json",
            "judge_source": self.campaign / "judging/judge-source-sha256.txt",
            "analysis_source": self.campaign / "analyze_stage.py",
        }
        for name in ("judge_inputs", "judge_complete"):
            write(paths[name], f"fixture:{stage}:{name}\n")
        return {name: digest(path) for name, path in paths.items()}

    def add_stage(self, stage: str, *, promote: bool = False) -> tuple[Path, Path, Path]:
        n = 30 if stage == "full" else 10
        if stage == "full":
            canonical = self.campaign / "canonical.tsv"
            current = canonical.read_text(encoding="utf-8")
            if current.count("\n") == 21:
                write(
                    canonical,
                    current
                    + "".join(
                        f"G4D-{index:02d}\n" for index in range(21, 61)
                    ),
                )
        denominator = n * 30
        control_count = int(denominator * 0.84)
        dots_count = control_count + (9 if stage == "full" else (12 if promote else 0))
        delta = (dots_count - control_count) / denominator * 100
        if stage == "full":
            promotion = {
                "evaluated": False,
                "terminal_stage": True,
                "triggered_rules": [],
                "promote_to_n30": False,
                "note": "The full 30-pair stage is terminal; no promotion rule applies.",
            }
        else:
            promotion = {
                "evaluated": True,
                "terminal_stage": False,
                "rules": {"ci_excludes_zero": promote},
                "triggered_rules": ["ci_excludes_zero"] if promote else [],
                "promote_to_n30": promote,
                "aligned_recurring_turns": [],
                "collection_launched": False,
                "note": "fixture",
            }
        payload = {
            "schema_version": 1,
            "campaign_id": publication.CAMPAIGN_ID,
            "stage": stage,
            "model": publication.MODEL,
            "provider": publication.PROVIDER,
            "configuration": {
                "control": "fresh contemporaneous nofiller",
                "treatment": "+96 space-separated suffix dots, request-only",
                "thinking_enabled": False,
                "fixed_turns_per_conversation": 30,
                "temporal_pairing": True,
            },
            "method": {
                "fixed_denominator": True,
                "missing_future_turns_fail_all_displayed_accuracy_criteria": True,
                "arm_interval_unit": "whole conversation",
                "effect_interval_unit": "frozen temporal pair",
                "effect_bootstrap_design": "paired bootstrap",
                "bootstrap_iterations": 100_000,
            },
            "input_hashes": self._hashes(stage),
            "arms": {
                "nofiller": arm(n, control_count, 610.4),
                "dots96": arm(n, dots_count, 615.0),
            },
            "effects": {
                "strict_pass": {
                    "dots_minus_control_points": delta,
                    "paired_bootstrap_95_low": 0.5 if promote or stage == "full" else -2.0,
                    "paired_bootstrap_95_high": 7.0 if promote or stage == "full" else 2.0,
                }
            },
            "promotion_evaluation": promotion,
        }
        aggregate, included, report = publication.stage_files(self.analysis, stage)
        write(included, "slot\tpair\tarm\nfixture\n")
        write(report, f"# fixture {stage}\n")
        write(aggregate, json.dumps(payload, indent=2) + "\n")
        return aggregate, included, report

    def add_promotion(self, initial: tuple[Path, Path, Path]) -> Path:
        aggregate, included, _report = initial
        payload = {
            "campaign_id": publication.CAMPAIGN_ID,
            "decision_after_n_per_arm": 10,
            "promote_to_n30": True,
            "triggered_rules": ["ci_excludes_zero"],
            "aggregates_sha256": digest(aggregate),
            "included_runs_sha256": digest(included),
            "decided_at": "2026-07-31T12:00:00+00:00",
            "reviewed_by": "Fixture Reviewer",
            "aggregates_path": str(aggregate.relative_to(self.root)),
            "included_runs_path": str(included.relative_to(self.root)),
        }
        path = self.analysis / "promotion-decision-initial.json"
        write(path, json.dumps(payload, indent=2) + "\n")
        return path

    def add_review(
        self,
        stage: str,
        files: tuple[Path, Path, Path],
        promotion: Path | None = None,
    ) -> Path:
        aggregate, included, report = files
        artifacts = {
            name: {"path": str(path.relative_to(self.root)), "sha256": digest(path)}
            for name, path in (
                ("aggregates", aggregate),
                ("included_runs", included),
                ("report", report),
            )
        }
        if promotion is not None:
            artifacts["promotion_decision"] = {
                "path": str(promotion.relative_to(self.root)),
                "sha256": digest(promotion),
            }
        payload = {
            "schema_version": 1,
            "artifact_status": "FINAL_PUBLICATION_REVIEW",
            "campaign_id": publication.CAMPAIGN_ID,
            "model": publication.MODEL,
            "provider": publication.PROVIDER,
            "selected_stage": stage,
            "action": "publish_full_terminal" if stage == "full" else "stop_at_initial",
            "reviewed_by": "Fixture Reviewer",
            "reviewed_at": "2026-07-31T13:00:00+00:00",
            "artifacts": artifacts,
        }
        path = self.analysis / "publication-review.json"
        write(path, json.dumps(payload, indent=2) + "\n")
        return path


def readme_fixture() -> str:
    return (
        "# Fixture\n\n"
        + publication.README_HEADER
        + "\n"
        + publication.README_SEPARATOR
        + "\n"
        + "| **alpha** | **95.0%** | 5.0% | 1.0% | 2.0% | 3.0% | 500ms | 800ms | 1200ms | OpenAI |\n"
        + "| gemma-4-26b-a4b-it (thinking off) | 81.4% | 18.6% | 13.0% | 18.6% | 0.0% | 597ms | 670ms | 4583ms | BaseTen |\n"
        + "| omega | 70.0% | 30.0% | 10.0% | 20.0% | 5.0% | 1000ms | 2000ms | 3000ms | Other |\n\n"
        + "After table.\n"
    )


def table_row(cells: list[str]) -> str:
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def html_fixture(rows: list[list[str]], labels: list[str], count: int, prose: str = "") -> str:
    header = ["Model", "Provider", "Base", "Dots", "Delta", "Completion", "TTFAT", "Runs", "Status"]
    word = {24: "Twenty-four", 25: "Twenty-five", 26: "Twenty-six"}[count]
    return (
        '<html><section id="primary-screen">'
        + f"<h2>{word}-model exploratory screen</h2>"
        + '<figure><svg width="100">'
        + "".join(f'<text x="0" y="1" class="lbl">{label}</text>' for label in labels)
        + "</svg></figure><table><tr>"
        + "".join(f"<th>{cell}</th>" for cell in header)
        + "</tr>"
        + "".join(table_row(row) for row in rows)
        + f"</table><p>{prose}</p></section></html>"
    )


def markdown_fixture(names: list[str], count: int) -> str:
    word = {24: "Twenty-four", 25: "Twenty-five", 26: "Twenty-six"}[count]
    rows = "\n".join(
        f"| {name} | Provider | 1 | 2 | +1 | 100% → 100% | 500 | 10 / 10 | status |"
        for name in names
    )
    return (
        f"**Scope:** {word} standard filler comparisons\n\n"
        "<!-- N30_PRIMARY_START -->\n"
        "| model | endpoint | no filler | +96 dots | delta | completion | TTFAT | runs | status |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---|\n"
        + rows
        + "\n<!-- N30_PRIMARY_END -->\n"
    )


class PublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = Fixture(self.root)
        files = self.fixture.add_stage("initial", promote=False)
        review = self.fixture.add_review("initial", files)
        self.data = publication.load_publication_data(
            root=self.root, analysis=self.fixture.analysis, review_path=review
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_reviewed_initial_stop_loads_and_uses_fresh_control(self) -> None:
        self.assertEqual(self.data.stage, "initial")
        self.assertEqual(self.data.n, 10)
        screen = self.data.normalized["screen_row"]
        self.assertEqual(screen["included_runs"], [10, 10])
        self.assertEqual(screen["no_filler_ttfat_p50_ms"], 610.4)
        self.assertTrue(screen["temporally_paired"])

    def test_triggered_initial_without_full_fails_closed(self) -> None:
        other = Fixture(self.root / "other")
        files = other.add_stage("initial", promote=True)
        review = other.add_review("initial", files)
        with self.assertRaisesRegex(RuntimeError, "terminal full stage"):
            publication.load_publication_data(
                root=other.root, analysis=other.analysis, review_path=review
            )

    def test_reviewed_full_stage_selects_highest_result(self) -> None:
        other = Fixture(self.root / "full")
        initial = other.add_stage("initial", promote=True)
        promotion = other.add_promotion(initial)
        full = other.add_stage("full")
        review = other.add_review("full", full, promotion)
        data = publication.load_publication_data(
            root=other.root, analysis=other.analysis, review_path=review
        )
        self.assertEqual((data.stage, data.n), ("full", 30))
        self.assertTrue(data.normalized["screen_row"]["focused"])

    def test_initial_canonical_prefix_is_bound_after_full_append(self) -> None:
        canonical = self.fixture.campaign / "canonical.tsv"
        expected = publication.sha256_tsv_prefix(canonical, 20)
        with canonical.open("a", encoding="utf-8") as handle:
            handle.write("G4D-21\n")
        self.assertEqual(publication.sha256_tsv_prefix(canonical, 20), expected)
        lines = canonical.read_text(encoding="utf-8").splitlines()
        lines[1] = "G4D-TAMPERED"
        write(canonical, "\n".join(lines) + "\n")
        self.assertNotEqual(publication.sha256_tsv_prefix(canonical, 20), expected)

    def test_review_hash_drift_fails_closed(self) -> None:
        write(self.data.report_path, "changed after review\n")
        with self.assertRaisesRegex(RuntimeError, "review report hash does not match"):
            publication.load_publication_data(
                root=self.root,
                analysis=self.fixture.analysis,
                review_path=self.fixture.analysis / "publication-review.json",
            )

    def test_readme_replacement_is_sorted_exact_and_idempotent(self) -> None:
        once = publication.update_readme(readme_fixture(), self.data)
        twice = publication.update_readme(once, self.data)
        self.assertEqual(once, twice)
        self.assertEqual(once.count("| gemma-4-26b-a4b-it (thinking off) |"), 1)
        self.assertIn("| 84.0% | 16.0% | 4.0% | 6.0% | 1.0% | 610ms | 710ms | 1610ms | BaseTen |", once)
        publication.validate_readme(once, self.data)

    def test_current_readme_transform_is_pure(self) -> None:
        source = publication.README_PATH.read_text(encoding="utf-8")
        proposed = publication.update_readme(source, self.data)
        self.assertEqual(publication.update_readme(proposed, self.data), proposed)
        self.assertEqual(publication.README_PATH.read_text(encoding="utf-8"), source)

    def test_generator_supports_current_optional_inkling_state(self) -> None:
        source = publication.GENERATOR_PATH.read_text(encoding="utf-8")
        transformed, count = publication.transform_generator(source)
        expected_count = (
            26 if "# INKLING_SMALL_PUBLICATION_DATA_START" in source else 25
        )
        self.assertEqual(count, expected_count)
        self.assertEqual(
            publication.transform_generator(transformed),
            (transformed, expected_count),
        )
        before_chart = source[source.index("def fig_dumbbell():") : source.index("def fig_dose():")]
        after_chart = transformed[
            transformed.index("def fig_dumbbell():") : transformed.index("def fig_dose():")
        ]
        self.assertEqual(before_chart, after_chart)

        if expected_count == 26:
            self.assertIn('        26: "Twenty-six",', transformed)

    def test_current_verifier_transforms_are_idempotent_for_25_and_26(self) -> None:
        for count in (25, 26):
            for path in publication.VERIFIER_PATHS:
                source = path.read_text(encoding="utf-8")
                transformed = publication.transform_verifier(source, count, path)
                self.assertEqual(
                    publication.transform_verifier(transformed, count, path), transformed
                )
                compile(transformed, str(path), "exec")
                self.assertNotIn("updated to 24", transformed)

    def test_html_and_markdown_validation_preserve_all_prior_rows(self) -> None:
        old_names = [f"model-{index}" for index in range(24)]
        old_rows = [
            [name, "Provider", "1.0", "2.0", "+1.0", "100% → 100%", "500", "10 / 10", "status"]
            for name in old_names
        ]
        screen = self.data.normalized["screen_row"]
        gemma = [
            publication.REPORT_NAME,
            "BaseTen",
            f"{screen['no_filler_pass_rate_pct']:.1f}",
            f"{screen['dots_pass_rate_pct']:.1f}",
            "+0.0",
            "100% → 100%",
            str(round(screen["no_filler_ttfat_p50_ms"])),
            "10 / 10",
            "no detectable effect",
        ]
        before_html = html_fixture(old_rows, old_names, 24)
        after_html = (
            "<p>Gemma 4 26B A4B adds a separate fixed-denominator comparison.</p>"
            + html_fixture(
                old_rows + [gemma],
                old_names + [publication.REPORT_NAME],
                25,
            )
        )
        publication.validate_html(before_html, after_html, self.data, 25)
        before_markdown = markdown_fixture(old_names, 24)
        after_markdown = markdown_fixture(old_names + [publication.REPORT_NAME], 25)
        publication.validate_markdown(before_markdown, after_markdown, 25)
        with self.assertRaisesRegex(RuntimeError, "preserve every existing Section 3 row"):
            publication.validate_html(
                before_html,
                (
                    "<p>Gemma 4 26B A4B adds a separate fixed-denominator comparison.</p>"
                    + html_fixture(
                        old_rows[1:] + [gemma],
                        old_names + [publication.REPORT_NAME],
                        25,
                    )
                ),
                self.data,
                25,
            )

    def test_generator_environment_excludes_provider_keys(self) -> None:
        with patch.dict(
            os.environ,
            {"PATH": "/bin", "BASETEN_API_KEY": "secret", "ANTHROPIC_API_KEY": "secret"},
            clear=True,
        ):
            self.assertEqual(publication.generator_environment(), {"PATH": "/bin"})


if __name__ == "__main__":
    unittest.main()
