#!/usr/bin/env python3
"""Resumably judge the local cohort with the frozen Gemma campaign judge."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = (
    ROOT
    / "ops/baseten-gemma4-31b-sglang"
    / "aiewf-medium-mtp-n30-20260806/judge_campaign.py"
)


def load_judge_module():
    spec = importlib.util.spec_from_file_location("gemma4_frozen_judge", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen judge: {BASE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.ROOT = ROOT
    module.CANONICAL = HERE / "canonical.tsv"
    module.JUDGING = HERE / "judging"
    module.INPUTS = module.JUDGING / "canonical-inputs.tsv"
    module.ATTEMPTS = module.JUDGING / "judge-attempts.tsv"
    module.LOGS = module.JUDGING / "logs"
    module.COMPLETE = module.JUDGING / "COMPLETE.json"
    return module


if __name__ == "__main__":
    raise SystemExit(load_judge_module().main())
