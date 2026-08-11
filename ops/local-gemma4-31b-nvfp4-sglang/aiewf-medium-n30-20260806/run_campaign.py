#!/usr/bin/env python3
"""Start the frozen local server, collect/resume the cohort, and stop it."""

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
START = HERE.parent / "start_server.sh"
STOP = HERE.parent / "stop_server.sh"
HEALTH = "http://127.0.0.1:30000/health"
CONTAINER = "aiewf-gemma4-31b-nvfp4"


def wait_ready() -> None:
    deadline = time.monotonic() + 20 * 60
    while time.monotonic() < deadline:
        try:
            response = requests.get(HEALTH, timeout=5)
            if response.ok:
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
            raise RuntimeError("local server exited during startup:\n" + logs.stdout + logs.stderr)
        time.sleep(5)
    raise TimeoutError("local SGLang server did not become ready within 20 minutes")


def main() -> int:
    started = False
    try:
        subprocess.run([str(START)], cwd=ROOT, check=True)
        started = True
        print("Waiting for local SGLang server", flush=True)
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
            print("Stopping local SGLang server", flush=True)
            subprocess.run([str(STOP)], cwd=ROOT, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
