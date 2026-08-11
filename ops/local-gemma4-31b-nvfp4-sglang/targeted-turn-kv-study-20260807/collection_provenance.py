#!/usr/bin/env python3
"""Fail-closed collection plans and live-server provenance for KV replays."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import requests

from study import (
    CHECKPOINT,
    CHECKPOINT_REVISION,
    HERE,
    MODEL,
    atomic_write_json,
    canonical_json,
    sha256_file,
    sha256_json,
)


PINNED_IMAGE = (
    "lmsysorg/sglang@sha256:"
    "00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2"
)
PINNED_IMAGE_ID = "sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2"
HF_CACHE_DIR = Path("/home/khkramer/.cache/huggingface")
SGLANG_JIT_CACHE_DIR = Path("/home/khkramer/.cache/sglang-tvm-ffi-gemma4-nvfp4")

SOURCE_FILES = (
    "PREREGISTRATION.md",
    "snapshot-manifest.json",
    "seed-manifest.json",
    "macro-schedule.tsv",
    "study.py",
    "collection_provenance.py",
    "replay.py",
    "run_stage.py",
    "scorer.py",
    "audit.py",
    "capture_runtime.py",
    "server.sh",
    "shims/sitecustomize.py",
)


class ProvenanceError(RuntimeError):
    """A live configuration or frozen artifact differs from the requested study."""


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _command(*values: str) -> str:
    return subprocess.run(values, check=True, text=True, capture_output=True).stdout


def _environment_map(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise ProvenanceError(f"invalid environment entry: {value!r}")
        if key in result:
            raise ProvenanceError(f"duplicate environment key: {key}")
        result[key] = item
    return result


def container_name(arm: str, geometry: str, sampling: str) -> str:
    return f"aiewf-gemma-kv-target-{arm}-{geometry}-{sampling}"


def expected_server_command(
    arm: str,
    geometry: str,
    sampling: str,
    *,
    port: int = 30000,
) -> list[str]:
    if arm not in ("fp8", "bf16"):
        raise ProvenanceError(f"unsupported arm: {arm}")
    if geometry not in ("compact", "historical"):
        raise ProvenanceError(f"unsupported geometry: {geometry}")
    if sampling not in ("seeded", "native", "batch-invariant"):
        raise ProvenanceError(f"unsupported sampling mode: {sampling}")
    if geometry == "historical" and arm != "fp8":
        raise ProvenanceError("historical geometry is defined only for FP8")

    kv_dtype = "fp8_e4m3" if arm == "fp8" else "bfloat16"
    sampling_args = ["--attention-backend", "triton"]
    if sampling == "seeded":
        sampling_args.extend(("--sampling-backend", "pytorch"))
    elif sampling == "batch-invariant":
        sampling_args.append("--enable-deterministic-inference")
    geometry_args = ["--max-total-tokens", "16000", "--swa-full-tokens-ratio", "0.35"]
    if geometry == "historical":
        geometry_args = ["--swa-full-tokens-ratio", "1.0"]
    return [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        CHECKPOINT,
        "--revision",
        CHECKPOINT_REVISION,
        "--served-model-name",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--quantization",
        "compressed-tensors",
        "--fp4-gemm-backend",
        "cutlass",
        "--kv-cache-dtype",
        kv_dtype,
        *sampling_args,
        "--disable-cuda-graph",
        "--skip-server-warmup",
        "--context-length",
        "32768",
        *geometry_args,
        "--max-running-requests",
        "1",
        "--chunked-prefill-size",
        "2048",
        "--max-prefill-tokens",
        "2048",
        "--mem-fraction-static",
        "0.90",
        "--sampling-defaults",
        "openai",
        "--stream-interval",
        "1",
        "--reasoning-parser",
        "gemma4",
        "--tool-call-parser",
        "gemma4",
        "--enable-cache-report",
        "--enable-metrics",
    ]


def expected_environment(
    image_inspect: dict[str, Any], sampling: str
) -> dict[str, str]:
    expected = _environment_map(image_inspect["Config"].get("Env") or [])
    expected.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    if sampling == "seeded":
        expected.update(
            {
                "PYTHONPATH": "/study-shims",
                "SGLANG_HONOR_REQUEST_SEED_WITHOUT_BATCH_INVARIANCE": "1",
            }
        )
    return expected


def expected_mounts(sampling: str) -> set[tuple[str, str, bool]]:
    result = {
        (str(HF_CACHE_DIR), "/root/.cache/huggingface", True),
        (str(SGLANG_JIT_CACHE_DIR), "/root/.cache/tvm-ffi", True),
    }
    if sampling == "seeded":
        result.add((str(HERE / "shims"), "/study-shims", False))
    return result


def selected_server_info(server_info: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "model_path",
        "revision",
        "served_model_name",
        "kv_cache_dtype",
        "dtype",
        "quantization",
        "fp4_gemm_runner_backend",
        "attention_backend",
        "sampling_backend",
        "max_total_num_tokens",
        "swa_full_tokens_ratio",
        "max_running_requests",
        "disable_cuda_graph",
        "context_length",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "mem_fraction_static",
        "sampling_defaults",
        "enable_deterministic_inference",
        "disable_radix_cache",
        "enable_cache_report",
        "enable_metrics",
        "reasoning_parser",
        "tool_call_parser",
    )
    return {key: server_info.get(key) for key in keys}


def expected_server_info(arm: str, geometry: str, sampling: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model_path": CHECKPOINT,
        "revision": CHECKPOINT_REVISION,
        "served_model_name": MODEL,
        "kv_cache_dtype": "fp8_e4m3" if arm == "fp8" else "bfloat16",
        "dtype": "bfloat16",
        "quantization": "compressed-tensors",
        "fp4_gemm_runner_backend": "cutlass",
        "attention_backend": "triton",
        "swa_full_tokens_ratio": 1.0 if geometry == "historical" else 0.35,
        "max_running_requests": 1,
        "disable_cuda_graph": True,
        "context_length": 32768,
        "chunked_prefill_size": 2048,
        "max_prefill_tokens": 2048,
        "mem_fraction_static": 0.9,
        "sampling_defaults": "openai",
        "enable_deterministic_inference": sampling == "batch-invariant",
        "disable_radix_cache": False,
        "enable_cache_report": True,
        "enable_metrics": True,
        "reasoning_parser": "gemma4",
        "tool_call_parser": "gemma4",
    }
    if geometry == "compact":
        result["max_total_num_tokens"] = 16000
    if sampling == "seeded":
        result["sampling_backend"] = "pytorch"
    return result


def validate_server_documents(
    *,
    arm: str,
    geometry: str,
    sampling: str,
    port: int,
    requested_container: str,
    inspect: dict[str, Any],
    image_inspect: dict[str, Any],
    server_info: dict[str, Any],
) -> dict[str, Any]:
    """Validate immutable Docker state and the live SGLang configuration."""

    errors: list[str] = []
    actual_name = str(inspect.get("Name") or "").removeprefix("/")
    expected_name = container_name(arm, geometry, sampling)
    if requested_container != expected_name:
        errors.append(f"requested container {requested_container!r} != {expected_name!r}")
    if actual_name != requested_container:
        errors.append(f"live container name {actual_name!r} != {requested_container!r}")
    if not (inspect.get("State") or {}).get("Running"):
        errors.append("container is not running")
    if inspect.get("Image") != PINNED_IMAGE_ID:
        errors.append(f"container image id {inspect.get('Image')!r} != pinned id")
    if (inspect.get("Config") or {}).get("Image") != PINNED_IMAGE:
        errors.append("container image reference is not the pinned digest")

    command = (inspect.get("Config") or {}).get("Cmd") or []
    wanted_command = expected_server_command(arm, geometry, sampling, port=port)
    if command != wanted_command:
        errors.append("container command differs from the exact frozen command")

    try:
        actual_environment = _environment_map((inspect.get("Config") or {}).get("Env") or [])
        wanted_environment = expected_environment(image_inspect, sampling)
        if actual_environment != wanted_environment:
            errors.append("container environment differs from pinned image plus frozen overrides")
    except ProvenanceError as exc:
        actual_environment = {}
        wanted_environment = {}
        errors.append(str(exc))

    mounts = {
        (str(item.get("Source")), str(item.get("Destination")), bool(item.get("RW")))
        for item in inspect.get("Mounts") or []
    }
    wanted_mounts = expected_mounts(sampling)
    if mounts != wanted_mounts:
        errors.append(f"container mounts differ: actual={sorted(mounts)!r}")

    host_config = inspect.get("HostConfig") or {}
    if host_config.get("NetworkMode") != "host":
        errors.append("container network mode is not host")
    if host_config.get("IpcMode") != "host":
        errors.append("container IPC mode is not host")
    if int(host_config.get("ShmSize") or 0) != 32 * 1024**3:
        errors.append("container shared-memory size is not 32 GiB")
    device_requests = host_config.get("DeviceRequests") or []
    if not any("gpu" in (item.get("Capabilities") or [[]])[0] for item in device_requests):
        errors.append("container has no Docker GPU device request")

    live_selected = selected_server_info(server_info)
    for key, wanted in expected_server_info(arm, geometry, sampling).items():
        if live_selected.get(key) != wanted:
            errors.append(
                f"/get_server_info {key}={live_selected.get(key)!r}, expected {wanted!r}"
            )

    if errors:
        raise ProvenanceError("; ".join(errors))

    config = {
        "image_reference": PINNED_IMAGE,
        "image_id": inspect["Image"],
        "command": command,
        "environment": actual_environment,
        "mounts": sorted(mounts),
        "network_mode": host_config.get("NetworkMode"),
        "ipc_mode": host_config.get("IpcMode"),
        "shm_size": host_config.get("ShmSize"),
        "server_info": live_selected,
        "image_environment_sha256": sha256_json(wanted_environment),
    }
    config_sha256 = sha256_json(config)
    binding = {
        "container_name": actual_name,
        "container_id": inspect["Id"],
        "container_created": inspect.get("Created"),
        "container_started_at": (inspect.get("State") or {}).get("StartedAt"),
        "server_config_sha256": config_sha256,
    }
    return {
        "schema_version": 1,
        "captured_utc": utc_now(),
        "arm": arm,
        "geometry": geometry,
        "sampling": sampling,
        "port": port,
        "binding": binding,
        "config": config,
        # SGLang includes runtime telemetry such as last_gen_throughput under
        # internal_states. Preserve the full-document hash diagnostically, but
        # do not mistake expected telemetry changes for configuration drift.
        "diagnostics": {"full_server_info_sha256": sha256_json(server_info)},
        "server_binding_sha256": sha256_json({"binding": binding, "config": config}),
    }


def capture_live_server(
    *,
    arm: str,
    geometry: str,
    sampling: str,
    endpoint: str,
    requested_container: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    port_text = endpoint.split(":")[-1].split("/", 1)[0]
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ProvenanceError(f"cannot determine port from endpoint {endpoint!r}") from exc
    name = requested_container or container_name(arm, geometry, sampling)
    inspect = json.loads(_command("docker", "inspect", name))[0]
    image_inspect = json.loads(_command("docker", "image", "inspect", PINNED_IMAGE))[0]
    http = session or requests.Session()
    server_url = endpoint.removesuffix("/v1").rstrip("/") + "/get_server_info"
    response = http.get(server_url, timeout=10)
    response.raise_for_status()
    server_info = response.json()
    return validate_server_documents(
        arm=arm,
        geometry=geometry,
        sampling=sampling,
        port=port,
        requested_container=name,
        inspect=inspect,
        image_inspect=image_inspect,
        server_info=server_info,
    )


def verified_snapshots(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    result: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("entries") or []:
        snapshot_id = entry["snapshot_id"]
        if snapshot_id in result:
            raise ProvenanceError(f"duplicate snapshot manifest entry: {snapshot_id}")
        snapshot = json.loads((manifest_path.parent / entry["path"]).read_text())
        if snapshot.get("snapshot_id") != snapshot_id:
            raise ProvenanceError(f"snapshot id mismatch for {snapshot_id}")
        if snapshot.get("turn") != entry.get("turn") or snapshot.get("kind") != entry.get("kind"):
            raise ProvenanceError(f"snapshot metadata mismatch for {snapshot_id}")
        request_hash = sha256_json(snapshot.get("request"))
        message_hash = sha256_json((snapshot.get("request") or {}).get("messages"))
        if request_hash != snapshot.get("request_sha256") or request_hash != entry.get("request_sha256"):
            raise ProvenanceError(f"request hash mismatch for {snapshot_id}")
        if message_hash != snapshot.get("messages_sha256") or message_hash != entry.get("messages_sha256"):
            raise ProvenanceError(f"message hash mismatch for {snapshot_id}")
        result[snapshot_id] = snapshot
    if not result:
        raise ProvenanceError("snapshot manifest is empty")
    return result


def verified_token_ids(
    token_manifest_path: Path,
    snapshots: dict[str, dict[str, Any]],
    *,
    arm: str,
) -> dict[str, list[int]]:
    payload = json.loads(token_manifest_path.read_text())
    manifest_arm = str(payload.get("arm") or "")
    if not manifest_arm.startswith(arm):
        raise ProvenanceError(
            f"token manifest arm {manifest_arm!r} is incompatible with requested arm {arm!r}"
        )
    result: dict[str, list[int]] = {}
    for row in payload.get("snapshots") or []:
        snapshot_id = row["snapshot_id"]
        if snapshot_id in result:
            raise ProvenanceError(f"duplicate token manifest entry: {snapshot_id}")
        if snapshot_id not in snapshots:
            raise ProvenanceError(f"unknown token-manifest snapshot: {snapshot_id}")
        ids = row.get("prompt_token_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(value, int) for value in ids):
            raise ProvenanceError(f"invalid prompt token ids for {snapshot_id}")
        if len(ids) != int(row.get("prompt_tokens") or -1):
            raise ProvenanceError(f"prompt token count mismatch for {snapshot_id}")
        if sha256_json(ids) != row.get("prompt_token_ids_sha256"):
            raise ProvenanceError(f"prompt token hash mismatch for {snapshot_id}")
        snapshot = snapshots[snapshot_id]
        if row.get("request_sha256") != snapshot.get("request_sha256"):
            raise ProvenanceError(f"token/request hash mismatch for {snapshot_id}")
        if int(row.get("turn")) != int(snapshot.get("turn")):
            raise ProvenanceError(f"token/turn mismatch for {snapshot_id}")
        result[snapshot_id] = ids
    if set(result) != set(snapshots):
        missing = sorted(set(snapshots) - set(result))
        extra = sorted(set(result) - set(snapshots))
        raise ProvenanceError(f"token manifest coverage differs: missing={missing}, extra={extra}")
    return result


def source_hashes(token_manifest_path: Path) -> dict[str, str]:
    paths = {name: HERE / name for name in SOURCE_FILES}
    paths[str(token_manifest_path)] = token_manifest_path
    result = {}
    for name, path in paths.items():
        if not path.exists():
            raise ProvenanceError(f"required source artifact is absent: {path}")
        result[name] = sha256_file(path)
    return result


def request_body(
    snapshot: dict[str, Any],
    *,
    seed: int,
    cache_mode: str,
    request_id: str,
    max_tokens: int,
    prompt_token_ids: list[int],
    temperature: float | None,
) -> dict[str, Any]:
    body = dict(snapshot["request"])
    body["seed"] = seed
    body["max_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature
    body["return_cached_tokens_details"] = True
    if cache_mode == "warm":
        body["cache_salt"] = f"gemma-kv-target-v1:{snapshot['snapshot_id']}"
    elif cache_mode == "cold":
        body["cache_salt"] = f"gemma-kv-target-v1:cold:{request_id}"
    elif cache_mode != "unsalted":
        raise ProvenanceError(f"unsupported cache mode: {cache_mode}")
    body["input_ids"] = prompt_token_ids
    return body


def future_request_id(
    config_sha256: str,
    *,
    stage: str,
    arm: str,
    cache_mode: str,
    snapshot_id: str,
    seed: int,
    repeat: int,
) -> str:
    return (
        f"v2:{config_sha256[:16]}:{stage}:{arm}:{cache_mode}:"
        f"{snapshot_id}:{seed}:{repeat}"
    )


def make_collection_plan(
    *,
    stage: str,
    treatment_arm: str,
    geometry: str,
    sampling: str,
    cache_mode: str,
    max_tokens: int,
    temperature: float | None,
    snapshot_manifest_path: Path,
    seed_manifest_path: Path,
    token_manifest_path: Path,
    server: dict[str, Any],
    case_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    snapshots = verified_snapshots(snapshot_manifest_path)
    tokens = verified_token_ids(token_manifest_path, snapshots, arm=treatment_arm)
    sources = source_hashes(token_manifest_path)
    frozen_server = {key: value for key, value in server.items() if key != "captured_utc"}
    config = {
        "stage": stage,
        "treatment_arm": treatment_arm,
        "geometry": geometry,
        "sampling": sampling,
        "cache_mode": cache_mode,
        "max_tokens": max_tokens,
        "temperature_override": temperature,
        "snapshot_manifest": str(snapshot_manifest_path),
        "snapshot_manifest_sha256": sha256_file(snapshot_manifest_path),
        "seed_manifest": str(seed_manifest_path),
        "seed_manifest_sha256": sha256_file(seed_manifest_path),
        "token_manifest": str(token_manifest_path),
        "token_manifest_sha256": sha256_file(token_manifest_path),
        "source_sha256": sources,
        "server_binding_sha256": server["server_binding_sha256"],
        "server_instance_id": server["binding"]["container_id"],
        "server_config_sha256": server["binding"]["server_config_sha256"],
    }
    config_sha256 = sha256_json(config)
    cases = []
    request_ids: set[str] = set()
    logical_keys: set[tuple[Any, ...]] = set()
    for spec in case_specs:
        snapshot_id = str(spec["snapshot_id"])
        if snapshot_id not in snapshots:
            raise ProvenanceError(f"collection case has unknown snapshot: {snapshot_id}")
        result_arm = str(spec.get("arm") or treatment_arm)
        case_cache = str(spec.get("cache_mode") or cache_mode)
        seed = int(spec["seed"])
        repeat = int(spec.get("repeat", 0))
        case_max_tokens = int(spec.get("max_tokens", max_tokens))
        case_temperature = spec.get("temperature", temperature)
        request_id = future_request_id(
            config_sha256,
            stage=stage,
            arm=result_arm,
            cache_mode=case_cache,
            snapshot_id=snapshot_id,
            seed=seed,
            repeat=repeat,
        )
        body = request_body(
            snapshots[snapshot_id],
            seed=seed,
            cache_mode=case_cache,
            request_id=request_id,
            max_tokens=case_max_tokens,
            prompt_token_ids=tokens[snapshot_id],
            temperature=case_temperature,
        )
        logical_key = (result_arm, case_cache, snapshot_id, seed, repeat)
        if logical_key in logical_keys:
            raise ProvenanceError(f"duplicate logical collection case: {logical_key}")
        logical_keys.add(logical_key)
        if request_id in request_ids:
            raise ProvenanceError(f"duplicate collection request id: {request_id}")
        request_ids.add(request_id)
        snapshot = snapshots[snapshot_id]
        cases.append(
            {
                "request_id": request_id,
                "arm": result_arm,
                "treatment_arm": treatment_arm,
                "cache_mode": case_cache,
                "snapshot_id": snapshot_id,
                "snapshot_kind": snapshot["kind"],
                "turn": snapshot["turn"],
                "seed": seed,
                "repeat": repeat,
                "max_tokens": case_max_tokens,
                "temperature_override": case_temperature,
                "request_sha256": sha256_json(body),
                "base_request_sha256": snapshot["request_sha256"],
                "input_ids_sha256": sha256_json(tokens[snapshot_id]),
            }
        )
    plan_core = {
        "schema_version": 2,
        "config_sha256": config_sha256,
        "config": config,
        "server": frozen_server,
        "expected_case_count": len(cases),
        "cases": cases,
    }
    return {**plan_core, "plan_sha256": sha256_json(plan_core), "created_utc": utc_now()}


def validate_collection_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != 2:
        raise ProvenanceError("unsupported collection plan schema")
    config = plan.get("config") or {}
    if sha256_json(config) != plan.get("config_sha256"):
        raise ProvenanceError("collection plan config hash mismatch")
    cases = plan.get("cases") or []
    if len(cases) != int(plan.get("expected_case_count") or -1):
        raise ProvenanceError("collection plan case count mismatch")
    request_ids = [row.get("request_id") for row in cases]
    logical_keys = [
        (row.get("arm"), row.get("cache_mode"), row.get("snapshot_id"), row.get("seed"), row.get("repeat"))
        for row in cases
    ]
    if len(set(request_ids)) != len(request_ids):
        raise ProvenanceError("collection plan has duplicate request ids")
    if len(set(logical_keys)) != len(logical_keys):
        raise ProvenanceError("collection plan has duplicate logical cells")
    plan_core = {
        key: plan[key]
        for key in ("schema_version", "config_sha256", "config", "server", "expected_case_count", "cases")
    }
    if sha256_json(plan_core) != plan.get("plan_sha256"):
        raise ProvenanceError("collection plan hash mismatch")


def write_or_validate_plan(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    validate_collection_plan(plan)
    if path.exists():
        existing = json.loads(path.read_text())
        validate_collection_plan(existing)
        if existing.get("plan_sha256") != plan.get("plan_sha256"):
            raise ProvenanceError(
                f"existing plan {path} differs from current collection configuration"
            )
        return existing
    atomic_write_json(path, plan)
    return plan


def validate_live_server_against_plan(plan: dict[str, Any], live: dict[str, Any]) -> None:
    validate_collection_plan(plan)
    if live.get("server_binding_sha256") != (plan.get("server") or {}).get(
        "server_binding_sha256"
    ):
        raise ProvenanceError("live server instance/config differs from frozen collection plan")


def validate_current_sources(plan: dict[str, Any]) -> None:
    validate_collection_plan(plan)
    errors = []
    for name, expected in (plan.get("config") or {}).get("source_sha256", {}).items():
        path = Path(name)
        if not path.is_absolute():
            path = HERE / path
        if not path.exists():
            errors.append(f"source artifact is absent: {path}")
        elif sha256_file(path) != expected:
            errors.append(f"source artifact changed after plan freeze: {path}")
    if errors:
        raise ProvenanceError("; ".join(errors))


def plan_case_index(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validate_collection_plan(plan)
    return {row["request_id"]: row for row in plan["cases"]}


def validate_row_identity(row: dict[str, Any], case: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    errors = []
    expected = {
        "request_id": case["request_id"],
        "arm": case["arm"],
        "treatment_arm": case["treatment_arm"],
        "cache_mode": case["cache_mode"],
        "snapshot_id": case["snapshot_id"],
        "snapshot_kind": case["snapshot_kind"],
        "turn": case["turn"],
        "seed": case["seed"],
        "repeat": case["repeat"],
        "max_tokens": case["max_tokens"],
        "temperature_override": case["temperature_override"],
        "request_sha256": case["request_sha256"],
        "base_request_sha256": case["base_request_sha256"],
        "input_ids_sha256": case["input_ids_sha256"],
        "collection_stage": plan["config"]["stage"],
        "collection_plan_sha256": plan["plan_sha256"],
        "collection_config_sha256": plan["config_sha256"],
        "server_instance_id": plan["config"]["server_instance_id"],
        "server_config_sha256": plan["config"]["server_config_sha256"],
    }
    for key, value in expected.items():
        if row.get(key) != value:
            errors.append(f"{case['request_id']}: {key}={row.get(key)!r}, expected {value!r}")
    return errors
