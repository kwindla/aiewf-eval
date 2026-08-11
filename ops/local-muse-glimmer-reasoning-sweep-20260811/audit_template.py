#!/usr/bin/env python3
"""Audit Muse Glimmer's actual server-side template renders before collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from benchmarks.aiwf_medium_context.config import BenchmarkConfig


SUPPORTED = ("low", "medium", "high", "xhigh")


def render(base_url: str, body: dict[str, Any]) -> str:
    response = requests.post(
        base_url.rstrip("/") + "/apply-template", json=body, timeout=30
    )
    response.raise_for_status()
    payload = response.json()
    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        raise RuntimeError(f"unexpected /apply-template response: {payload}")
    return prompt


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    system_instruction = BenchmarkConfig.system_instruction
    base: dict[str, Any] = {
        "messages": [{"role": "system", "content": system_instruction}],
        "add_generation_prompt": True,
    }
    variants: dict[str, dict[str, Any]] = {
        "default": {},
        "reasoning_effort_none": {"reasoning_effort": "none"},
        "enable_thinking_false": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
        **{
            f"strength_{strength}": {
                "chat_template_kwargs": {"reasoning_strength": strength}
            }
            for strength in (*SUPPORTED, "none", "minimal")
        },
    }
    prompts = {
        name: render(args.base_url, {**base, **overrides})
        for name, overrides in variants.items()
    }

    default = prompts["default"]
    assert prompts["reasoning_effort_none"] == default
    assert prompts["enable_thinking_false"] == default
    assert prompts["strength_high"] == default
    for strength in (*SUPPORTED, "none", "minimal"):
        prompt = prompts[f"strength_{strength}"]
        assert system_instruction in prompt
        assert prompt.count(system_instruction) == 1
        assert f"Reasoning strength: {strength}." in prompt

    output = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "supported_strengths": list(SUPPORTED),
        "unsupported_render_controls": ["none", "minimal"],
        "system_instruction_length": len(system_instruction),
        "system_instruction_sha256": sha256(system_instruction),
        "assertions": {
            "system_instruction_preserved_once": True,
            "default_is_high": True,
            "reasoning_effort_none_is_render_noop": True,
            "enable_thinking_false_is_render_noop": True,
            "unsupported_values_are_unvalidated_labels_only": True,
        },
        "renders": {
            name: {
                "sha256": sha256(prompt),
                "length": len(prompt),
                "reasoning_lines": [
                    line
                    for line in prompt.splitlines()
                    if line.startswith("Reasoning strength:")
                ],
            }
            for name, prompt in prompts.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
