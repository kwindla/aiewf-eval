#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from audit import audit_results, completion_errors, strict_repeatability_gate
from collection_provenance import (
    HF_CACHE_DIR,
    HERE,
    PINNED_IMAGE,
    PINNED_IMAGE_ID,
    SGLANG_JIT_CACHE_DIR,
    ProvenanceError,
    expected_server_command,
    expected_server_info,
    sha256_json,
    validate_collection_plan,
    validate_server_documents,
)
from scorer import score_message


def server_documents(arm="fp8"):
    image = {"Config": {"Env": ["BASE_IMAGE_VALUE=1"]}}
    environment = [
        "PYTHONPATH=/study-shims",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "SGLANG_HONOR_REQUEST_SEED_WITHOUT_BATCH_INVARIANCE=1",
        "BASE_IMAGE_VALUE=1",
    ]
    inspect = {
        "Id": "container-instance-1",
        "Name": f"/aiewf-gemma-kv-target-{arm}-compact-seeded",
        "Image": PINNED_IMAGE_ID,
        "Created": "2026-08-07T00:00:00Z",
        "State": {"Running": True, "StartedAt": "2026-08-07T00:00:01Z"},
        "Config": {
            "Image": PINNED_IMAGE,
            "Cmd": expected_server_command(arm, "compact", "seeded"),
            "Env": environment,
        },
        "Mounts": [
            {"Source": str(HF_CACHE_DIR), "Destination": "/root/.cache/huggingface", "RW": True},
            {
                "Source": str(SGLANG_JIT_CACHE_DIR),
                "Destination": "/root/.cache/tvm-ffi",
                "RW": True,
            },
            {"Source": str(HERE / "shims"), "Destination": "/study-shims", "RW": False},
        ],
        "HostConfig": {
            "NetworkMode": "host",
            "IpcMode": "host",
            "ShmSize": 32 * 1024**3,
            "DeviceRequests": [{"Capabilities": [["gpu"]]}],
        },
    }
    info = expected_server_info(arm, "compact", "seeded")
    return inspect, image, info


def completion(message=None):
    message = message or {"role": "assistant", "content": "No action.", "tool_calls": []}
    value = {
        "message": message,
        "reasoning_content": "",
        "usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 95}},
        "cached_tokens": 95,
        "first_sse_ms": 1.0,
        "ttfat_ms": 2.0,
        "finish_reasons": ["stop"],
        "raw_events": [],
    }
    semantic_message = {
        "role": message["role"],
        "content": message["content"],
        "tool_calls": [
            {"type": call.get("type"), "function": call.get("function")}
            for call in message.get("tool_calls") or []
        ],
    }
    value["raw_events_sha256"] = sha256_json([])
    value["output_sha256"] = sha256_json(
        {"message": message, "reasoning_content": "", "finish_reasons": ["stop"]}
    )
    value["semantic_output_sha256"] = sha256_json(
        {"message": semantic_message, "reasoning_content": "", "finish_reasons": ["stop"]}
    )
    return value


def two_case_plan():
    config = {
        "stage": "unit",
        "source_sha256": {},
        "server_instance_id": "instance",
        "server_config_sha256": "server-config",
    }
    config_hash = sha256_json(config)
    cases = []
    for seed in (0, 1):
        cases.append(
            {
                "request_id": f"request-{seed}",
                "arm": "fp8",
                "treatment_arm": "fp8",
                "cache_mode": "warm",
                "snapshot_id": "turn12-golden",
                "snapshot_kind": "golden_mechanism",
                "turn": 12,
                "seed": seed,
                "repeat": 0,
                "max_tokens": 512,
                "temperature_override": None,
                "request_sha256": f"request-hash-{seed}",
                "base_request_sha256": "base-hash",
                "input_ids_sha256": "input-hash",
            }
        )
    core = {
        "schema_version": 2,
        "config_sha256": config_hash,
        "config": config,
        "server": {"server_binding_sha256": "binding"},
        "expected_case_count": 2,
        "cases": cases,
    }
    return {**core, "plan_sha256": sha256_json(core), "created_utc": "ignored"}


def row_for_case(case, plan):
    output = completion()
    return {
        "schema_version": 2,
        **case,
        "collection_stage": plan["config"]["stage"],
        "collection_plan_sha256": plan["plan_sha256"],
        "collection_config_sha256": plan["config_sha256"],
        "server_instance_id": plan["config"]["server_instance_id"],
        "server_config_sha256": plan["config"]["server_config_sha256"],
        "started_unix": 1.0,
        "error": None,
        "score": score_message(12, output["message"]),
        "completion": output,
        "warm_cache_gate_passed": True,
    }


class ServerProvenanceTest(unittest.TestCase):
    def test_exact_fp8_server_passes(self):
        inspect, image, info = server_documents("fp8")
        result = validate_server_documents(
            arm="fp8",
            geometry="compact",
            sampling="seeded",
            port=30000,
            requested_container="aiewf-gemma-kv-target-fp8-compact-seeded",
            inspect=inspect,
            image_inspect=image,
            server_info=info,
        )
        self.assertEqual(result["arm"], "fp8")

    def test_fp8_live_server_cannot_be_labeled_bf16(self):
        inspect, image, info = server_documents("fp8")
        with self.assertRaises(ProvenanceError):
            validate_server_documents(
                arm="bf16",
                geometry="compact",
                sampling="seeded",
                port=30000,
                requested_container="aiewf-gemma-kv-target-bf16-compact-seeded",
                inspect=inspect,
                image_inspect=image,
                server_info=info,
            )

    def test_runtime_telemetry_does_not_change_server_binding(self):
        inspect, image, info = server_documents("bf16")
        info["internal_states"] = [{"last_gen_throughput": 10.0}]
        first = validate_server_documents(
            arm="bf16",
            geometry="compact",
            sampling="seeded",
            port=30000,
            requested_container="aiewf-gemma-kv-target-bf16-compact-seeded",
            inspect=inspect,
            image_inspect=image,
            server_info=info,
        )
        info["internal_states"] = [{"last_gen_throughput": 20.0}]
        second = validate_server_documents(
            arm="bf16",
            geometry="compact",
            sampling="seeded",
            port=30000,
            requested_container="aiewf-gemma-kv-target-bf16-compact-seeded",
            inspect=inspect,
            image_inspect=image,
            server_info=info,
        )
        self.assertEqual(first["server_binding_sha256"], second["server_binding_sha256"])
        self.assertNotEqual(first["diagnostics"], second["diagnostics"])


class PlanAuditTest(unittest.TestCase):
    def test_incomplete_plan_fails_closed(self):
        plan = two_case_plan()
        validate_collection_plan(plan)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.jsonl"
            path.write_text(json.dumps(row_for_case(plan["cases"][0], plan)) + "\n")
            result = audit_results([path], plan)
        self.assertFalse(result["passed"])
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["expected_rows"], 2)
        self.assertEqual(result["missing_request_ids"], ["request-1"])

    def test_missing_usage_and_length_are_rejected(self):
        plan = two_case_plan()
        row = row_for_case(plan["cases"][0], plan)
        row["completion"]["usage"] = None
        row["completion"]["finish_reasons"] = ["length"]
        errors = completion_errors(row)
        self.assertTrue(any("missing usage" in error for error in errors))
        self.assertTrue(any("truncated" in error for error in errors))

    def test_strict_repeatability_rejects_partial_input(self):
        plan = two_case_plan()
        row = row_for_case(plan["cases"][0], plan)
        row.update(
            {
                "request_id": "one-row",
                "snapshot_id": "turn12-golden",
                "snapshot_kind": "golden_mechanism",
                "seed": 0,
                "repeat": 0,
            }
        )
        result = strict_repeatability_gate(
            [row], arm="fp8", cache_mode="warm", expected_repeats=[0, 1]
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["expected_rows"], 200)
        self.assertEqual(result["observed_rows"], 1)


if __name__ == "__main__":
    unittest.main()
