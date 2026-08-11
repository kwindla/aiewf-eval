"""Focused offline tests for the date-versioned cost estimator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def load_estimator():
    path = HERE / "estimate_costs.py"
    spec = importlib.util.spec_from_file_location("freshness_cost_estimator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def estimator():
    return load_estimator()


def usage(**overrides):
    row = {
        "key": "example",
        "label": "example",
        "provider": "Example",
        "model": "example/model",
        "run_dir": "runs/example",
        "status": "complete",
        "prompt_tokens": 1000,
        "cache_read_input_tokens": 400,
        "cache_creation_input_tokens": 0,
        "completion_tokens": 100,
        "estimated_speech_minutes_150wpm": 2.0,
    }
    row.update(overrides)
    return row


def test_anthropic_prompt_is_not_reduced_by_separate_cache_counters(estimator):
    row = usage(
        prompt_tokens=100,
        cache_read_input_tokens=200,
        cache_creation_input_tokens=300,
        completion_tokens=40,
    )
    price = {
        "billing_mode": "tokens",
        "token_accounting": "prompt_excludes_cache_reads_and_writes",
        "rates_usd_per_million_tokens": {
            "input": 1,
            "cached_input": 0.1,
            "cache_write_5m": 1.25,
            "output": 5,
        },
        "source": "https://example.invalid",
    }

    result = estimator.estimate_row(row, price)

    assert result["billed_token_counts"] == {
        "uncached_input": 100,
        "cached_input": 200,
        "cache_write_5m": 300,
        "output": 40,
    }
    assert result["estimated_sample_cost_usd"] == pytest.approx(0.000695)


def test_inclusive_prompt_subtracts_cache_hits_before_uncached_rate(estimator):
    price = {
        "billing_mode": "tokens",
        "token_accounting": "prompt_includes_cache_reads",
        "rates_usd_per_million_tokens": {
            "input": 2,
            "cached_input": 0.5,
            "output": 4,
        },
        "source": "https://example.invalid",
    }

    result = estimator.estimate_row(usage(), price)

    assert result["billed_token_counts"]["uncached_input"] == 600
    assert result["estimated_sample_cost_usd"] == pytest.approx(0.0018)
    assert result["estimated_cost_per_conversation_minute_usd"] == pytest.approx(
        0.0009
    )


def test_dedicated_gpu_is_utilization_priced_not_token_priced(estimator):
    price = {
        "billing_mode": "active_gpu_conversation_minute",
        "active_compute_usd_per_minute": 0.10833,
        "utilization_assumption": {
            "accelerator": "1x H100",
            "active_conversations": 1,
            "scope": "active estimated conversation minutes only",
            "excluded": "cold-start and idle scale-down time",
        },
        "source": "https://example.invalid",
    }

    result = estimator.estimate_row(usage(prompt_tokens=999999999), price)

    assert result["billed_token_counts"]["uncached_input"] is None
    assert result["estimated_sample_cost_usd"] == pytest.approx(0.21666)
    assert result["estimated_cost_per_conversation_minute_usd"] == pytest.approx(
        0.10833
    )
    assert result["utilization_assumption"]["active_conversations"] == 1


def test_current_usage_is_fully_priced_and_contains_no_pro_entry(estimator):
    usage_rows = json.loads((HERE / "usage-results.json").read_text())
    pricing = json.loads((HERE / "pricing-2026-08-06.json").read_text())

    result = estimator.estimate_all(usage_rows, pricing)

    expected = sum(row["status"] == "complete" for row in usage_rows)
    assert result["included_completed_conversations"] == expected
    assert len(result["estimates"]) == expected
    assert not any(
        "pro" in key.casefold().replace("-", "_").split("_")
        for key in pricing["models"]
    )


def test_missing_complete_row_price_fails_loudly(estimator):
    pricing = {
        "usage_sample_date": "2026-08-06",
        "pricing_as_of": "2026-08-06",
        "currency": "USD",
        "conversation_minutes_method": "test",
        "models": {},
    }

    with pytest.raises(ValueError, match="missing pricing.*example"):
        estimator.estimate_all([usage()], pricing)
