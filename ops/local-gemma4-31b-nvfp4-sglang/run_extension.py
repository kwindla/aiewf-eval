#!/usr/bin/env python3
"""Start one local KV configuration, resume its N=120 extension, and stop it."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COLLECTOR = ROOT / "ops/aiewf-campaign-template/collect.py"
CONFIGURATIONS = {
    "fp8": {
        "campaign": HERE / "aiewf-medium-fp8kv-n120-extension-20260807",
        "start": HERE / "start_server.sh",
        "stop": HERE / "stop_server.sh",
        "health": "http://127.0.0.1:30000/health",
        "container": "aiewf-gemma4-31b-nvfp4",
    },
    "bf16": {
        "campaign": HERE / "aiewf-medium-bf16kv-n120-extension-20260807",
        "start": HERE / "start_server_bf16_kv.sh",
        "stop": HERE / "stop_server_bf16_kv.sh",
        "health": "http://127.0.0.1:30001/health",
        "container": "aiewf-gemma4-31b-nvfp4-bf16kv",
    },
}


def wait_ready(settings: dict[str, object]) -> None:
    deadline = time.monotonic() + 20 * 60
    while time.monotonic() < deadline:
        try:
            if requests.get(str(settings["health"]), timeout=5).ok:
                return
        except requests.RequestException:
            pass
        inspect = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Running}}",
                str(settings["container"]),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if inspect.returncode != 0 or inspect.stdout.strip() != "true":
            logs = subprocess.run(
                ["docker", "logs", "--tail", "200", str(settings["container"])],
                capture_output=True,
                text=True,
                check=False,
            )
            raise RuntimeError(
                "local server exited during startup:\n" + logs.stdout + logs.stderr
            )
        time.sleep(5)
    raise TimeoutError("local SGLang server did not become ready within 20 minutes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-cache", choices=tuple(CONFIGURATIONS), required=True)
    args = parser.parse_args()
    settings = CONFIGURATIONS[args.kv_cache]
    config = Path(settings["campaign"]) / "configuration.json"
    started = False
    try:
        subprocess.run([str(settings["start"])], cwd=ROOT, check=True)
        started = True
        print(f"Waiting for local {args.kv_cache.upper()}-KV server", flush=True)
        wait_ready(settings)
        print(
            f"Server ready; starting/resuming {args.kv_cache.upper()}-KV N=120 extension",
            flush=True,
        )
        env = os.environ.copy()
        env["LOCAL_VLLM_API_KEY"] = "EMPTY"
        result = subprocess.run(
            [sys.executable, str(COLLECTOR), "--config", str(config), "--execute"],
            cwd=ROOT,
            env=env,
            check=False,
        )
        return result.returncode
    finally:
        if started:
            print(f"Stopping local {args.kv_cache.upper()}-KV server", flush=True)
            subprocess.run([str(settings["stop"])], cwd=ROOT, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
