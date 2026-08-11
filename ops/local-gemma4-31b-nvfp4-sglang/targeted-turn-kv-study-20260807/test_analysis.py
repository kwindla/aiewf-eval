#!/usr/bin/env python3

import unittest

from analyze_replays import (
    SIMULTANEOUS_TAIL_PROBABILITY,
    _expected_primary_stage,
    exact_mcnemar,
    paired_summary,
    primary_stop_decision,
    select_inferential_rows,
    turn_interaction_summary,
    validate_primary_look,
)
from audit import semantic_output_sha256


def completion(call_id):
    return {
        "message": {
            "role": "assistant",
            "content": "done",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "submit", "arguments": "{}"},
                }
            ],
        },
        "reasoning_content": "",
        "finish_reasons": ["tool_calls"],
    }


class AnalysisTest(unittest.TestCase):
    def test_transport_call_id_is_not_semantic_output(self):
        self.assertEqual(
            semantic_output_sha256(completion("call_one")),
            semantic_output_sha256(completion("call_two")),
        )

    def test_exact_mcnemar(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertEqual(exact_mcnemar(1, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar(8, 0), 2 / 256)

    def test_equal_prefix_weighting(self):
        selected = []
        for arm, outcomes in (
            ("bf16", {"a": [1, 1], "b": [0, 0]}),
            ("fp8", {"a": [0, 0], "b": [0, 0]}),
        ):
            for snapshot, values in outcomes.items():
                for seed, value in enumerate(values):
                    selected.append(
                        {
                            "arm": arm,
                            "snapshot_id": snapshot,
                            "seed": seed,
                            "score": {"success": bool(value)},
                        }
                    )
        result = paired_summary(selected, ["a", "b"])
        self.assertEqual(result["pairs"], 4)
        self.assertEqual(result["seed_clusters"], 2)
        self.assertEqual(result["equal_prefix_weighted_difference_points"], 50.0)

    def test_repeat_zero_is_selected_explicitly(self):
        rows = []
        for repeat in (2, 0, 1):
            rows.append(
                {
                    "arm": "fp8",
                    "cache_mode": "cold",
                    "snapshot_id": "a",
                    "seed": 7,
                    "repeat": repeat,
                    "score": {"success": True, "category": "ok"},
                    "completion": {"semantic_output_sha256": "same"},
                }
            )
        selected, summary = select_inferential_rows(rows)
        self.assertEqual([row["repeat"] for row in selected], [0])
        self.assertEqual(summary["excluded_gate_repeats"], 2)

    def test_repeat_disagreement_is_rejected(self):
        rows = [
            {
                "arm": "fp8",
                "cache_mode": "cold",
                "snapshot_id": "a",
                "seed": 7,
                "repeat": repeat,
                "score": {"success": repeat == 0, "category": str(repeat)},
                "completion": {"semantic_output_sha256": str(repeat)},
            }
            for repeat in (0, 1)
        ]
        with self.assertRaises(RuntimeError):
            select_inferential_rows(rows)

    def test_turn_interaction_uses_seed_clusters(self):
        comparisons = {
            "warm_turn12_bank": {"seed_cluster_effects": {"0": 1.0, "1": 0.0}},
            "warm_turn15_bank": {"seed_cluster_effects": {"0": 0.0, "1": 0.0}},
        }
        result = turn_interaction_summary(comparisons)["warm"]
        self.assertEqual(result["seed_clusters"], 2)
        self.assertEqual(result["interaction_points"], 50.0)

    def _primary_fixture(self, look=2048):
        manifest = {"entries": []}
        for turn in (12, 15):
            manifest["entries"].append(
                {
                    "snapshot_id": f"turn{turn}-golden",
                    "turn": turn,
                    "kind": "golden_mechanism",
                }
            )
            for index in range(12):
                manifest["entries"].append(
                    {
                        "snapshot_id": f"turn{turn}-bank-{index:02d}",
                        "turn": turn,
                        "kind": "real_prefix_bank",
                    }
                )
        rows = []
        tier_limits = {
            "golden_mechanism": ((512, "primary"), (1024, "continue-4096"), (2048, "continue-8192")),
            "real_prefix_bank": ((128, "primary"), (256, "continue-4096"), (512, "continue-8192")),
        }
        for arm in ("fp8", "bf16"):
            for entry in manifest["entries"]:
                count = look // 4 if entry["kind"] == "golden_mechanism" else look // 16
                for seed in range(count):
                    lower = 0
                    stage_name = None
                    upper = None
                    for tier_upper, tier_name in tier_limits[entry["kind"]]:
                        if seed < tier_upper:
                            stage_name = tier_name
                            upper = tier_upper
                            break
                        lower = tier_upper
                    self.assertIsNotNone(stage_name)
                    first_half = seed < lower + (upper - lower) // 2
                    block = (
                        (1 if arm == "fp8" else 2)
                        if first_half
                        else (4 if arm == "fp8" else 3)
                    )
                    rows.append(
                        {
                            "request_id": f"{arm}:{entry['snapshot_id']}:{seed}",
                            "arm": arm,
                            "cache_mode": "warm",
                            "collection_stage": f"{stage_name}-block{block}",
                            "snapshot_id": entry["snapshot_id"],
                            "snapshot_kind": entry["kind"],
                            "turn": entry["turn"],
                            "seed": seed,
                            "repeat": 0,
                        }
                    )
        return rows, manifest

    def test_primary_look_requires_exact_allocation_and_abba(self):
        for look in (2048, 4096, 8192):
            with self.subTest(look=look):
                rows, manifest = self._primary_fixture(look)
                result = validate_primary_look(rows, manifest, look)
                self.assertTrue(result["passed"])
                self.assertEqual(result["rows"], look * 4)
                self.assertEqual(result["bank_pairs_per_turn"], look * 3 // 4)
                self.assertEqual(result["seed_clusters_per_turn"], look // 16)

    def test_primary_stage_boundaries(self):
        cases = [
            ("golden_mechanism", 0, "fp8", "primary-block1"),
            ("golden_mechanism", 255, "bf16", "primary-block2"),
            ("golden_mechanism", 256, "bf16", "primary-block3"),
            ("golden_mechanism", 511, "fp8", "primary-block4"),
            ("golden_mechanism", 512, "fp8", "continue-4096-block1"),
            ("golden_mechanism", 767, "bf16", "continue-4096-block2"),
            ("golden_mechanism", 768, "bf16", "continue-4096-block3"),
            ("golden_mechanism", 1023, "fp8", "continue-4096-block4"),
            ("golden_mechanism", 1024, "fp8", "continue-8192-block1"),
            ("golden_mechanism", 1535, "bf16", "continue-8192-block2"),
            ("golden_mechanism", 1536, "bf16", "continue-8192-block3"),
            ("golden_mechanism", 2047, "fp8", "continue-8192-block4"),
            ("real_prefix_bank", 0, "fp8", "primary-block1"),
            ("real_prefix_bank", 63, "bf16", "primary-block2"),
            ("real_prefix_bank", 64, "bf16", "primary-block3"),
            ("real_prefix_bank", 127, "fp8", "primary-block4"),
            ("real_prefix_bank", 128, "fp8", "continue-4096-block1"),
            ("real_prefix_bank", 191, "bf16", "continue-4096-block2"),
            ("real_prefix_bank", 192, "bf16", "continue-4096-block3"),
            ("real_prefix_bank", 255, "fp8", "continue-4096-block4"),
            ("real_prefix_bank", 256, "fp8", "continue-8192-block1"),
            ("real_prefix_bank", 383, "bf16", "continue-8192-block2"),
            ("real_prefix_bank", 384, "bf16", "continue-8192-block3"),
            ("real_prefix_bank", 511, "fp8", "continue-8192-block4"),
        ]
        for kind, seed, arm, expected in cases:
            with self.subTest(kind=kind, seed=seed, arm=arm):
                self.assertEqual(
                    _expected_primary_stage(arm=arm, snapshot_kind=kind, seed=seed),
                    expected,
                )

    def test_primary_look_rejects_identical_cross_arm_omission(self):
        rows, manifest = self._primary_fixture()
        target = next(
            row
            for row in rows
            if row["arm"] == "fp8"
            and row["snapshot_id"] == "turn12-bank-00"
            and row["seed"] == 127
        )
        target["snapshot_id"] = "turn12-bank-01"
        with self.assertRaises(RuntimeError):
            validate_primary_look(rows, manifest, 2048)

    def test_primary_look_rejects_wrong_macro_stage(self):
        rows, manifest = self._primary_fixture()
        rows[0]["collection_stage"] = "primary-block4"
        with self.assertRaises(RuntimeError):
            validate_primary_look(rows, manifest, 2048)

    def test_primary_look_rejects_wrong_continuation_stage(self):
        rows, manifest = self._primary_fixture(4096)
        target = next(
            row
            for row in rows
            if row["arm"] == "fp8"
            and row["snapshot_kind"] == "real_prefix_bank"
            and row["seed"] == 128
        )
        target["collection_stage"] = "primary-block1"
        with self.assertRaises(RuntimeError):
            validate_primary_look(rows, manifest, 4096)

    def test_primary_comparison_has_no_p_value_fields(self):
        selected = []
        for arm, values in (("bf16", [1, 0]), ("fp8", [0, 0])):
            for seed, value in enumerate(values):
                selected.append(
                    {
                        "arm": arm,
                        "snapshot_id": "a",
                        "seed": seed,
                        "score": {"success": bool(value)},
                    }
                )
        result = paired_summary(selected, ["a"])
        self.assertNotIn("exact_mcnemar_p", result)
        self.assertNotIn("seed_cluster_robust_score_p", result)
        self.assertNotIn("fixed_look_seed_cluster_wald_p_exploratory", result)

    def test_simultaneous_tail_probability(self):
        self.assertEqual(SIMULTANEOUS_TAIL_PROBABILITY, 1 / 240)

    def test_primary_stop_decision_uses_conventional_interval_half_width(self):
        comparisons = {
            "warm_turn12_bank": {
                "equal_prefix_weighted_difference_points": 0.0,
                "two_turn_three_look_simultaneous_95_percent": [-3.0, 1.0],
            },
            "warm_turn15_bank": {
                "equal_prefix_weighted_difference_points": 0.0,
                "two_turn_three_look_simultaneous_95_percent": [-2.1, 2.1],
            },
        }
        result = primary_stop_decision(comparisons, 2048)
        self.assertEqual(result["decision"], "continue")
        self.assertEqual(result["next_look_cases_per_turn_arm"], 4096)
        self.assertTrue(result["cells"]["warm_turn12_bank"]["precision_met"])
        self.assertEqual(
            result["cells"]["warm_turn12_bank"]["interval_half_width_points"], 2.0
        )
        self.assertFalse(result["cells"]["warm_turn15_bank"]["precision_met"])

    def test_primary_stop_and_maximum_look_decisions(self):
        precise = {
            f"warm_turn{turn}_bank": {
                "equal_prefix_weighted_difference_points": 0.0,
                "two_turn_three_look_simultaneous_95_percent": [-2.0, 2.0],
            }
            for turn in (12, 15)
        }
        self.assertEqual(
            primary_stop_decision(precise, 2048)["decision"], "stop_precision_met"
        )
        imprecise = {
            f"warm_turn{turn}_bank": {
                "equal_prefix_weighted_difference_points": 0.0,
                "two_turn_three_look_simultaneous_95_percent": [-2.01, 2.01],
            }
            for turn in (12, 15)
        }
        maximum = primary_stop_decision(imprecise, 8192)
        self.assertEqual(maximum["decision"], "stop_maximum_look")
        self.assertFalse(maximum["continue_required"])


if __name__ == "__main__":
    unittest.main()
