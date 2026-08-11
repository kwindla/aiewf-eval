#!/usr/bin/env python3
"""Wake the frozen BaseTen deployment, run/resume collection, and tear down."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = HERE / "configuration.json"
COLLECTOR = ROOT / "ops/aiewf-campaign-template/collect.py"
CONTROL = "https://api.baseten.co/v1/models/q951m16w/deployments/q862ez8"


def load_key() -> str:
    for path in (ROOT / ".env", ROOT.parent / "gb-benchmarks/.env"):
        if path.is_file():
            key = dotenv_values(path).get("BASETEN_API_KEY")
            if key:
                return str(key)
    raise RuntimeError("BASETEN_API_KEY not found")


def headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Api-Key {key}", "Content-Type": "application/json"}


def set_minimum(key: str, minimum: int) -> None:
    response = requests.patch(
        CONTROL + "/autoscaling_settings",
        headers=headers(key),
        json={
            "min_replica": minimum,
            "max_replica": 1,
            "autoscaling_window": 60,
            "scale_down_delay": 120,
            "concurrency_target": 1,
            "target_utilization_percentage": 70,
        },
        timeout=30,
    )
    response.raise_for_status()


def wait_state(key: str, *, active: bool) -> dict[str, Any]:
    deadline = time.monotonic() + (30 * 60 if active else 15 * 60)
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = requests.get(CONTROL, headers=headers(key), timeout=30)
        response.raise_for_status()
        last = response.json()
        status = last.get("status")
        replicas = int(last.get("active_replica_count") or 0)
        if active and status == "ACTIVE" and replicas >= 1:
            return last
        if not active and status == "SCALED_TO_ZERO" and replicas == 0:
            return last
        time.sleep(10)
    target = "ACTIVE" if active else "SCALED_TO_ZERO"
    raise TimeoutError(f"deployment did not reach {target}: {last}")


def main() -> int:
    key = load_key()
    try:
        print("Waking BaseTen SGLang NEXTN/MTP deployment", flush=True)
        set_minimum(key, 1)
        wait_state(key, active=True)
        print("Deployment active; starting/resuming frozen N=30 cohort", flush=True)
        result = subprocess.run(
            [
                sys.executable,
                str(COLLECTOR),
                "--config",
                str(CONFIG),
                "--execute",
            ],
            cwd=ROOT,
            check=False,
        )
        return result.returncode
    finally:
        print("Returning deployment to min_replica=0", flush=True)
        set_minimum(key, 0)
        wait_state(key, active=False)
        print("Deployment confirmed SCALED_TO_ZERO", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
