#!/usr/bin/env python3
"""Judge one local Gemma N=120 extension with the frozen campaign judge."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/judge_campaign.py"
CAMPAIGNS = {
    "fp8": HERE / "aiewf-medium-fp8kv-n120-extension-20260807",
    "bf16": HERE / "aiewf-medium-bf16kv-n120-extension-20260807",
}


def load_judge_module(campaign: Path):
    spec = importlib.util.spec_from_file_location("gemma4_extension_judge", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen judge: {BASE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = campaign
    module.ROOT = ROOT
    module.CANONICAL = campaign / "canonical.tsv"
    module.JUDGING = campaign / "judging"
    module.INPUTS = module.JUDGING / "canonical-inputs.tsv"
    module.ATTEMPTS = module.JUDGING / "judge-attempts.tsv"
    module.LOGS = module.JUDGING / "logs"
    module.COMPLETE = module.JUDGING / "COMPLETE.json"
    module.TARGET = 120
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--kv-cache", choices=tuple(CAMPAIGNS), required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return load_judge_module(CAMPAIGNS[args.kv_cache]).main()


if __name__ == "__main__":
    raise SystemExit(main())
