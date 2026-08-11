#!/usr/bin/env python3
"""Start the BF16-KV server, collect/resume N=30, and stop it."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = HERE / "configuration.json"
COLLECTOR = ROOT / "ops/aiewf-campaign-template/collect.py"
START = HERE.parent / "start_server_bf16_kv.sh"
STOP = HERE.parent / "stop_server_bf16_kv.sh"
HEALTH = "http://127.0.0.1:30001/health"
CONTAINER = "aiewf-gemma4-31b-nvfp4-bf16kv"


def wait_ready() -> None:
    deadline = time.monotonic() + 20 * 60
    while time.monotonic() < deadline:
        try:
            if requests.get(HEALTH, timeout=5).ok:
                return
        except requests.RequestException:
            pass
        inspect = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", CONTAINER],
            capture_output=True,
            text=True,
            check=False,
        )
        if inspect.returncode != 0 or inspect.stdout.strip() != "true":
            logs = subprocess.run(
                ["docker", "logs", "--tail", "200", CONTAINER],
                capture_output=True,
                text=True,
                check=False,
            )
            raise RuntimeError("BF16-KV server exited during startup:\n" + logs.stdout + logs.stderr)
        time.sleep(5)
    raise TimeoutError("BF16-KV server did not become ready within 20 minutes")


def main() -> int:
    started = False
    try:
        subprocess.run([str(START)], cwd=ROOT, check=True)
        started = True
        print("Waiting for local BF16-KV SGLang server", flush=True)
        wait_ready()
        print("Server ready; starting/resuming frozen N=30 cohort", flush=True)
        env = os.environ.copy()
        env["LOCAL_VLLM_API_KEY"] = "EMPTY"
        result = subprocess.run(
            [sys.executable, str(COLLECTOR), "--config", str(CONFIG), "--execute"],
            cwd=ROOT,
            env=env,
            check=False,
        )
        return result.returncode
    finally:
        if started:
            print("Stopping local BF16-KV SGLang server", flush=True)
            subprocess.run([str(STOP)], cwd=ROOT, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
