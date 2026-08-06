#!/usr/bin/env python3
"""Generate the filler-token study HTML report (self-contained, theme-aware).

Design plan (artifact-design):
- Color: paper #FAF9F6 / ink #22252A / muted #71767D / benefit #2A6F97 /
  harm #B0432A / null #9AA0A6 / hairline #E4E1DB; dark equivalents.
  Semantic color encodes DIRECTION only, identically in every figure.
- Type: Charter/Georgia serif body; ui-monospace for data, labels, axes.
- Layout: single ~70ch column; figures to 860px; captions integrated.
Tufte: dumbbell/slopegraph, range whiskers, inline bars, failure-strip;
grayscale + two semantic hues; no grids beyond hairline range frames.
"""
import html as H
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs/filler-token-latent-scratchpad-study.html"
MARKDOWN_OUT = Path(__file__).resolve().parents[1] / "docs/filler-token-latent-scratchpad-study.md"

# ---------------- data ----------------
MODELS = [  # name, provider, base, filler, delta, p, n, verdict, note, baseline P50 TTFAT (ms)
 ("gpt-5.4","OpenAI",90.3,96.3,"+6.0","0.0072","10 / 10","pos","works",677),
 ("gpt-5.6-terra","OpenAI",91.0,95.4,"+4.4 †","0.053","10 / 8 †","warn","conditional — 76% abort",568),
 ("gpt-5.5","OpenAI",96.3,100.0,"+3.8","0.023","8 / 8","pos","works",882),
 ("gpt-5.6-sol","OpenAI",96.7,100.0,"+3.3","0.0001","10 / 10","pos","works",1016),
 ("gemma-4-31b-it","Lilac",97.0,99.0,"+2.0","0.088","10 / 10","sugg","suggestive",1006),
 ("gpt-oss-120b","BaseTen",84.9,85.3,"+0.4","0.96","6 / 6","null","no detectable effect",405),
 ("qwen3-32b","OpenRouter",90.7,91.0,"+0.3","1.00","10 / 10","null","no detectable effect",832),
 ("gpt-4.1","OpenAI",96.6,96.7,"+0.0","1.00","6 / 5","null","no detectable effect",610),
 ("kimi-k2.6","BaseTen",94.4,94.4,"+0.0","1.00","6 / 6","null","no detectable effect",463),
 ("nemotron-3-ultra","BaseTen",98.9,98.9,"+0.0","1.00","6 / 6","null","no detectable effect",447),
 ("qwen3-14b","OpenRouter",90.7,90.3,"−0.3","1.00","10 / 10","null","no detectable effect",707),
 ("gpt-5.6-luna","OpenAI",89.2,88.5,"−0.6","0.56","8 / 8 †","null","no detectable effect",583),
 ("deepseek-chat-v3.1","OpenRouter",95.3,94.7,"−0.7","0.90","10 / 10","null","no detectable effect",1406),
 ("claude-haiku-4-5","Anthropic",99.3,98.7,"−0.7","0.72","10 / 10","null","no detectable effect",777),
 ("inkling","BaseTen",97.1,95.0,"−2.1","0.31","8 / 8","null","no detect. (trend −)",465),
 ("qwen3-8b","BaseTen",80.0,77.2,"−2.8","0.40","10 / 6","null","no detect. (trend −)",1158),
 ("glm-5.2","BaseTen",99.7,95.3,"−4.3","0.020","10 / 10","neg","negative estimate",881),
]
if len(MODELS) != 17 or len({row[0] for row in MODELS}) != 17:
    raise ValueError("the retained exploratory screen must contain 17 unique rows before Gemini extensions")
GEMINI_PATH = (Path(__file__).resolve().parents[1] /
               "docs/filler-study-data/gemini-minimal-dots-2026-07-21/aggregates.json")
if not GEMINI_PATH.is_file():
    raise RuntimeError(f"Gemini minimal/dot aggregates are required: {GEMINI_PATH}")
GEMINI_PAYLOAD = json.loads(GEMINI_PATH.read_text())
if GEMINI_PAYLOAD.get("artifact_status") != "FINAL":
    raise ValueError("Gemini minimal/dot aggregates are not final")
gemini_protocol = GEMINI_PAYLOAD.get("protocol", {})
if (
    gemini_protocol.get("thinking_mode") != "minimal"
    or gemini_protocol.get("full_thinking_off_guaranteed") is not False
):
    raise ValueError("Gemini extension must use the documented provider-minimal reasoning floor")
gemini_order = gemini_protocol.get("model_order")
if gemini_order != ["gemini35flash", "gemini35flashlite", "gemini36flash"]:
    raise ValueError(f"Gemini extension order mismatch: {gemini_order}")
gemini_models = GEMINI_PAYLOAD.get("models", {})
if set(gemini_models) != set(gemini_order):
    raise ValueError(f"Gemini extension model mismatch: {sorted(gemini_models)}")
GEMINI_ROWS = {}
for gemini_key in gemini_order:
    result = gemini_models[gemini_key]
    control = result.get("arms", {}).get("nofiller", {})
    dots = result.get("arms", {}).get("dots96", {})
    decision = result.get("adaptive_decision", {})
    if decision.get("decision_pending") is not False:
        raise ValueError(f"Gemini adaptive decision is still pending for {gemini_key}")
    if control.get("n_attempts") not in {10, 30} or dots.get("n_attempts") not in {6, 10, 30}:
        raise ValueError(f"Gemini extension sample size mismatch for {gemini_key}")
    if control.get("fixed_turn_denominator") != 30 * control["n_attempts"]:
        raise ValueError(f"Gemini control denominator mismatch for {gemini_key}")
    if dots.get("fixed_turn_denominator") != 30 * dots["n_attempts"]:
        raise ValueError(f"Gemini dots denominator mismatch for {gemini_key}")
    delta_raw = result["effect"]["pass_delta_points"]
    ci = result["effect"]["pass_delta_ci95"]
    if ci[0] > 0:
        row_key, verdict = "pos", "increase"
    elif ci[1] < 0:
        row_key, verdict = "neg", "decrease"
    elif abs(delta_raw) >= 2:
        row_key, verdict = "sugg", "suggestive"
    else:
        row_key, verdict = "null", "no detectable effect"
    row = (
        result["display_name"],
        result["provider"],
        control["pass_rate_pct"],
        dots["pass_rate_pct"],
        f"{delta_raw:+.1f}".replace("-", "−"),
        "",
        f'{control["n_attempts"]} / {dots["n_attempts"]}',
        row_key,
        verdict,
        round(control["ttfat_p50_ms"]),
    )
    MODELS.append(row)
    GEMINI_ROWS[result["display_name"]] = result
if len(MODELS) != 20 or len({row[0] for row in MODELS}) != 20:
    raise ValueError("the extended exploratory screen must contain 20 unique rows")
GEMINI25_PATH = (Path(__file__).resolve().parents[1] /
                 "docs/filler-study-data/gemini25-thinking-off-dots-2026-07-22/aggregates.json")
GEMINI25_RESULT = None
if GEMINI25_PATH.is_file():
    gemini25_payload = json.loads(GEMINI25_PATH.read_text())
    if gemini25_payload.get("artifact_status") == "FINAL":
        protocol25 = gemini25_payload.get("protocol", {})
        if (
            protocol25.get("model_order") != ["gemini25flash"]
            or protocol25.get("thinking_mode") != "disabled"
            or protocol25.get("thinking_budget") != 0
            or protocol25.get("full_thinking_off_guaranteed") is not True
        ):
            raise ValueError("Gemini 2.5 extension is not explicitly thinking-off")
        result25 = gemini25_payload.get("models", {}).get("gemini25flash")
        if not result25 or result25.get("adaptive_decision", {}).get("decision_pending") is not False:
            raise ValueError("Gemini 2.5 adaptive decision is incomplete")
        control25 = result25.get("arms", {}).get("nofiller", {})
        dots25 = result25.get("arms", {}).get("dots96", {})
        if control25.get("n_attempts") not in {10, 30} or dots25.get("n_attempts") not in {6, 10, 30}:
            raise ValueError("Gemini 2.5 extension sample size mismatch")
        if (
            control25.get("fixed_turn_denominator") != 30 * control25["n_attempts"]
            or dots25.get("fixed_turn_denominator") != 30 * dots25["n_attempts"]
        ):
            raise ValueError("Gemini 2.5 fixed denominator mismatch")
        delta25 = result25["effect"]["pass_delta_points"]
        ci25 = result25["effect"]["pass_delta_ci95"]
        if ci25[0] > 0:
            row_key25, verdict25 = "pos", "increase"
        elif ci25[1] < 0:
            row_key25, verdict25 = "neg", "decrease"
        elif abs(delta25) >= 2:
            row_key25, verdict25 = "sugg", "suggestive"
        else:
            row_key25, verdict25 = "null", "no detectable effect"
        MODELS.append((
            result25["display_name"], result25["provider"], control25["pass_rate_pct"],
            dots25["pass_rate_pct"], f"{delta25:+.1f}".replace("-", "−"), "",
            f'{control25["n_attempts"]} / {dots25["n_attempts"]}', row_key25, verdict25,
            round(control25["ttfat_p50_ms"]),
        ))
        GEMINI25_RESULT = result25
LAGUNA_PATH = (Path(__file__).resolve().parents[1] /
               "docs/filler-study-data/laguna-s21-openrouter-2026-07-22/aggregates.json")
if not LAGUNA_PATH.is_file():
    raise RuntimeError(f"final Laguna S 2.1 aggregates are required: {LAGUNA_PATH}")
laguna_payload = json.loads(LAGUNA_PATH.read_text())
laguna_protocol = laguna_payload.get("protocol", {})
if (
    laguna_payload.get("schema_version") != 1
    or laguna_payload.get("artifact_status") != "FINAL"
    or laguna_protocol.get("benchmark") != "aiwf_medium_context"
    or laguna_protocol.get("turns") != 30
    or laguna_protocol.get("target_per_arm") != 30
    or laguna_protocol.get("missing_scripted_turns") != "fail"
    or laguna_protocol.get("thinking_mode") != "disabled"
    or laguna_protocol.get("full_thinking_off_guaranteed") is not True
    or laguna_protocol.get("route") != "OpenRouter paid Poolside-hosted BF16"
    or laguna_protocol.get("filler")
    != {"arm": "dots96", "glyph": ".", "count": 96, "position": "suffix"}
    or laguna_protocol.get("bootstrap_unit") != "whole conversation"
    or laguna_protocol.get("bootstrap_samples") != 100_000
):
    raise ValueError("Laguna S 2.1 final campaign protocol mismatch")
laguna_models = laguna_payload.get("models", {})
if set(laguna_models) != {"laguna_s21"}:
    raise ValueError(f"Laguna S 2.1 model set mismatch: {sorted(laguna_models)}")
LAGUNA_RESULT = laguna_models["laguna_s21"]
laguna_control = LAGUNA_RESULT.get("arms", {}).get("nofiller", {})
laguna_dots = LAGUNA_RESULT.get("arms", {}).get("dots96", {})
laguna_effect = LAGUNA_RESULT.get("effect", {})
if (
    LAGUNA_RESULT.get("display_name") != "laguna-s-2.1"
    or LAGUNA_RESULT.get("provider") != "OpenRouter"
    or LAGUNA_RESULT.get("requested_model") != "poolside/laguna-s-2.1"
    or LAGUNA_RESULT.get("endpoint_provider") != "Poolside"
    or LAGUNA_RESULT.get("quantization") != "BF16"
    or LAGUNA_RESULT.get("report_tier") != "focused"
    or laguna_control.get("n_attempts") != 30
    or laguna_dots.get("n_attempts") != 30
    or laguna_control.get("fixed_turn_denominator") != 900
    or laguna_dots.get("fixed_turn_denominator") != 900
    or laguna_control.get("thinking_tokens") != 0
    or laguna_dots.get("thinking_tokens") != 0
    or abs(
        laguna_effect.get("pass_delta_points", float("nan"))
        - (laguna_dots.get("pass_rate_pct", 0) - laguna_control.get("pass_rate_pct", 0))
    ) > 1e-9
    or not (
        isinstance(laguna_effect.get("pass_delta_ci95"), list)
        and len(laguna_effect["pass_delta_ci95"]) == 2
        and laguna_effect["pass_delta_ci95"][0]
        <= laguna_effect["pass_delta_ci95"][1]
    )
):
    raise ValueError("Laguna S 2.1 final aggregate mismatch")
laguna_delta = laguna_effect["pass_delta_points"]
laguna_ci = laguna_effect["pass_delta_ci95"]
if laguna_ci[0] > 0:
    laguna_key, laguna_verdict = "pos", "increase"
elif laguna_ci[1] < 0:
    laguna_key, laguna_verdict = "neg", "decrease"
else:
    laguna_key, laguna_verdict = "null", "uncertain"
MODELS.append((
    LAGUNA_RESULT["display_name"], LAGUNA_RESULT["provider"],
    laguna_control["pass_rate_pct"], laguna_dots["pass_rate_pct"],
    f"{laguna_delta:+.1f}".replace("-", "−"), "", "30 / 30",
    laguna_key, laguna_verdict, int(laguna_control["ttfat_p50_ms"] + 0.5),
))
QWEN_PATH = (Path(__file__).resolve().parents[1] /
             "docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json")
if not QWEN_PATH.is_file():
    raise RuntimeError(f"final Qwen3.6 filler aggregates are required: {QWEN_PATH}")
QWEN_PAYLOAD = json.loads(QWEN_PATH.read_text())
qwen_protocol = QWEN_PAYLOAD.get("protocol", {})
if (
    QWEN_PAYLOAD.get("schema_version") != 1
    or QWEN_PAYLOAD.get("artifact_status") != "FINAL_EXPLORATORY"
    or qwen_protocol.get("benchmark") != "aiwf_medium_context"
    or qwen_protocol.get("turns") != 30
    or qwen_protocol.get("thinking_mode") != "disabled"
    or qwen_protocol.get("full_thinking_off_guaranteed") is not True
    or qwen_protocol.get("control_n") != 30
    or qwen_protocol.get("decision_control_n") != 10
    or qwen_protocol.get("missing_scripted_turns") != "fail"
    or qwen_protocol.get("bootstrap_unit") != "whole conversation"
    or qwen_protocol.get("bootstrap_samples") != 100_000
    or qwen_protocol.get("interleaved") is not False
    or qwen_protocol.get("report_tier") != "exploratory"
    or qwen_protocol.get("filler")
    != {"arm": "dots96", "glyph": ".", "count": 96, "position": "suffix"}
):
    raise ValueError("Qwen3.6 exploratory filler protocol mismatch")
QWEN_ORDER = ["qwen36_27b", "qwen36_35b"]
QWEN_EXPECTED = {
    "qwen36_27b": {
        "display_name": "qwen3.6-27b (thinking off)",
        "report_name": "qwen3.6-27b",
        "requested_model": "Qwen/Qwen3.6-27B",
        "checkpoint_precision": "BF16",
    },
    "qwen36_35b": {
        "display_name": "qwen3.6-35b-a3b (thinking off, FP8)",
        "report_name": "qwen3.6-35b-a3b (FP8)",
        "requested_model": "Qwen/Qwen3.6-35B-A3B-FP8",
        "checkpoint_precision": "FP8",
    },
}
qwen_models = QWEN_PAYLOAD.get("models", {})
if list(qwen_models) != QWEN_ORDER:
    raise ValueError(f"Qwen3.6 model order mismatch: {list(qwen_models)}")
QWEN_RESULTS = {}
for qwen_key in QWEN_ORDER:
    result = qwen_models[qwen_key]
    expected = QWEN_EXPECTED[qwen_key]
    control = result.get("arms", {}).get("nofiller", {})
    dots = result.get("arms", {}).get("dots96", {})
    effect = result.get("effect", {})
    decision = result.get("adaptive_decision", {})
    serving = result.get("serving", {})
    if (
        result.get("display_name") != expected["display_name"]
        or result.get("requested_model") != expected["requested_model"]
        or result.get("checkpoint_precision") != expected["checkpoint_precision"]
        or result.get("provider") != "BaseTen"
        or result.get("report_tier") != "exploratory"
        or result.get("noncontemporaneous_reused_control") is not True
        or serving.get("vllm") != "0.26.0"
        or serving.get("automatic_prefix_caching") is not True
        or serving.get("mamba_cache_mode") != "align"
        or serving.get("mtp_speculative_tokens") != 2
        or control.get("n_attempts") != 30
        or dots.get("n_attempts") not in {6, 10, 30}
        or control.get("fixed_turn_denominator") != 900
        or dots.get("fixed_turn_denominator") != 30 * dots["n_attempts"]
        or control.get("thought_tokens") != 0
        or dots.get("thought_tokens") != 0
        or decision.get("decision_pending") is not False
        or decision.get("action")
        not in {"stop_at_6", "stop_at_10", "n30_treatment_complete"}
        or abs(
            effect.get("pass_delta_points", float("nan"))
            - (dots.get("pass_rate_pct", 0) - control.get("pass_rate_pct", 0))
        ) > 1e-9
        or not (
            isinstance(effect.get("pass_delta_ci95"), list)
            and len(effect["pass_delta_ci95"]) == 2
            and effect["pass_delta_ci95"][0] <= effect["pass_delta_ci95"][1]
        )
    ):
        raise ValueError(f"Qwen3.6 final aggregate mismatch: {qwen_key}")
    raw_delta = effect["pass_delta_points"]
    ci = effect["pass_delta_ci95"]
    if ci[0] > 0:
        row_key, verdict = "pos", "increase"
    elif ci[1] < 0:
        row_key, verdict = "neg", "decrease"
    elif abs(raw_delta) >= 2:
        row_key, verdict = "sugg", "suggestive"
    else:
        row_key, verdict = "null", "no detectable effect"
    MODELS.append((
        expected["report_name"], result["provider"],
        control["pass_rate_pct"], dots["pass_rate_pct"],
        f"{raw_delta:+.1f}".replace("-", "−"), "",
        f'{control["n_attempts"]} / {dots["n_attempts"]}',
        row_key, verdict, int(control["ttfat_p50_ms"] + 0.5),
    ))
    QWEN_RESULTS[expected["report_name"]] = result
# INKLING_SMALL_PUBLICATION_DATA_START
INKLING_SMALL_PUBLICATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "ops/baseten-inkling-small/aiewf-medium-none-low-n30-20260731/analysis/publication-input.json"
)
if not INKLING_SMALL_PUBLICATION_PATH.is_file():
    raise RuntimeError(
        f"final Inkling Small publication input is required: {INKLING_SMALL_PUBLICATION_PATH}"
    )
INKLING_SMALL_PUBLICATION = json.loads(INKLING_SMALL_PUBLICATION_PATH.read_text())
if (
    INKLING_SMALL_PUBLICATION.get("schema_version") != 1
    or INKLING_SMALL_PUBLICATION.get("artifact_status") != "FINAL_PUBLICATION_INPUT"
    or INKLING_SMALL_PUBLICATION.get("model") != "thinkingmachines/inkling-small"
    or INKLING_SMALL_PUBLICATION.get("report_name") != "inkling-small"
    or INKLING_SMALL_PUBLICATION.get("provider") != "BaseTen"
):
    raise ValueError("Inkling Small publication input identity mismatch")
INKLING_SMALL_SCREEN = INKLING_SMALL_PUBLICATION.get("screen_row", {})
if (
    INKLING_SMALL_SCREEN.get("name") != "inkling-small"
    or INKLING_SMALL_SCREEN.get("provider") != "BaseTen"
    or INKLING_SMALL_SCREEN.get("included_runs", [None])[0] != 30
    or INKLING_SMALL_SCREEN.get("none_ttfat_p50_ms") is None
):
    raise ValueError("Inkling Small screen row is incomplete")
INKLING_SMALL_ROBUSTNESS = INKLING_SMALL_PUBLICATION.get("robustness", {})
INKLING_SMALL_EFFORT_ROBUSTNESS = INKLING_SMALL_ROBUSTNESS.get(
    "primary_effort_campaign", {}
)
INKLING_SMALL_SHORT_RUNS = INKLING_SMALL_EFFORT_ROBUSTNESS.get(
    "baseten_429_idle_short_runs", {}
)
INKLING_SMALL_JUDGE_SENSITIVITY = INKLING_SMALL_ROBUSTNESS.get(
    "judge_sensitivity", {}
)
if (
    INKLING_SMALL_EFFORT_ROBUSTNESS.get("retained_attempts") != 60
    or INKLING_SMALL_SHORT_RUNS.get("none") != 12
    or INKLING_SMALL_SHORT_RUNS.get("low") != 10
    or INKLING_SMALL_SHORT_RUNS.get("total") != 22
    or INKLING_SMALL_EFFORT_ROBUSTNESS.get(
        "fixed_denominator_missing_future_turns_fail"
    ) is not True
    or INKLING_SMALL_EFFORT_ROBUSTNESS.get(
        "serving_failures_not_generated_terminal_calls"
    ) is not True
    or INKLING_SMALL_JUDGE_SENSITIVITY.get(
        "changed_tool_use_correct_labels"
    ) != 4
    or INKLING_SMALL_JUDGE_SENSITIVITY.get(
        "max_abs_arm_rate_change_percentage_points", 1
    ) > 0.5
    or INKLING_SMALL_JUDGE_SENSITIVITY.get(
        "disclosure_bound_percentage_points"
    ) != 0.5
    or INKLING_SMALL_JUDGE_SENSITIVITY.get(
        "official_artifacts_unchanged"
    ) is not True
):
    raise ValueError("Inkling Small robustness disclosure input mismatch")
inkling_small_delta = INKLING_SMALL_SCREEN["dots_minus_control_points"]
MODELS.append((
    "inkling-small",
    "BaseTen",
    INKLING_SMALL_SCREEN["no_filler_pass_rate_pct"],
    INKLING_SMALL_SCREEN["dots_pass_rate_pct"],
    f"{inkling_small_delta:+.1f}".replace("-", "−"),
    "",
    f'{INKLING_SMALL_SCREEN["included_runs"][0]} / {INKLING_SMALL_SCREEN["included_runs"][1]}',
    INKLING_SMALL_SCREEN["key"],
    INKLING_SMALL_SCREEN["interpretation"],
    round(INKLING_SMALL_SCREEN["none_ttfat_p50_ms"]),
))
INKLING_SMALL_METHOD_MARKDOWN = (
    " Inkling Small adds a separate fixed-denominator BaseTen comparison: its 30-run "
    f"`none` control is frozen from the none/low campaign and its later adaptive dot arm "
    f"stopped at {INKLING_SMALL_PUBLICATION['dots_stage']}; the two arms are not interleaved."
)
INKLING_SMALL_LIMITS_MARKDOWN = (
    " The Inkling Small screen is fixed-denominator and attempt-based, but reuses an "
    "earlier control, so deployment-time drift remains a limitation."
)
INKLING_SMALL_PROVENANCE_MARKDOWN = (
    " The Inkling Small row uses BaseTen for both arms, the frozen `none` arm's TTFAT, "
    "and the highest mechanically reached dot-stage artifact. In Inkling Small's primary "
    "30-pair `none`/`low` campaign, "
    f"{INKLING_SMALL_SHORT_RUNS['total']}/"
    f"{INKLING_SMALL_EFFORT_ROBUSTNESS['retained_attempts']} retained attempts ended "
    "short after a BaseTen HTTP 429 followed by the harness idle timeout "
    f"({INKLING_SMALL_SHORT_RUNS['none']} `none`, "
    f"{INKLING_SMALL_SHORT_RUNS['low']} `low`); these were serving failures rather "
    "than generated terminal calls, and fixed-denominator scoring retains them with "
    "missing future turns counted as failures. A post-hoc sensitivity check changing "
    f"the {INKLING_SMALL_JUDGE_SENSITIVITY['changed_tool_use_correct_labels']} disputed "
    "`tool_use_correct` labels shifted any arm-level published rate by no more than "
    f"{INKLING_SMALL_JUDGE_SENSITIVITY['disclosure_bound_percentage_points']:.1f} "
    "percentage points; official judgments remain unchanged."
)
INKLING_SMALL_METHOD_HTML = INKLING_SMALL_METHOD_MARKDOWN.replace("`none`", "<code>none</code>")
INKLING_SMALL_LIMITS_HTML = INKLING_SMALL_LIMITS_MARKDOWN
INKLING_SMALL_PROVENANCE_HTML = INKLING_SMALL_PROVENANCE_MARKDOWN.replace(
    "`none`", "<code>none</code>"
).replace(
    "`low`", "<code>low</code>"
).replace(
    "`tool_use_correct`", "<code>tool_use_correct</code>"
)
# INKLING_SMALL_PUBLICATION_DATA_END
# GEMMA26_PUBLICATION_DATA_START
GEMMA26_PUBLICATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/publication-input.json"
)
if not GEMMA26_PUBLICATION_PATH.is_file():
    raise RuntimeError(f"final Gemma 4 26B publication input is required: {GEMMA26_PUBLICATION_PATH}")
GEMMA26_PUBLICATION = json.loads(GEMMA26_PUBLICATION_PATH.read_text())
if (
    GEMMA26_PUBLICATION.get("schema_version") != 1
    or GEMMA26_PUBLICATION.get("artifact_status") != "FINAL_PUBLICATION_INPUT"
    or GEMMA26_PUBLICATION.get("model") != "google/gemma-4-26B-A4B-it"
    or GEMMA26_PUBLICATION.get("provider") != "BaseTen"
):
    raise ValueError("Gemma 4 26B publication input identity mismatch")
GEMMA26_SCREEN = GEMMA26_PUBLICATION.get("screen_row", {})
if (
    GEMMA26_SCREEN.get("name") != "gemma-4-26b-a4b"
    or GEMMA26_SCREEN.get("provider") != "BaseTen"
    or GEMMA26_SCREEN.get("included_runs", [None, None])[0]
       != GEMMA26_SCREEN.get("included_runs", [None, None])[1]
    or GEMMA26_SCREEN.get("included_runs", [None])[0] not in {10, 30}
    or GEMMA26_SCREEN.get("no_filler_ttfat_p50_ms") is None
):
    raise ValueError("Gemma 4 26B screen row is incomplete")
gemma26_delta = GEMMA26_SCREEN["dots_minus_control_points"]
MODELS.append((
    "gemma-4-26b-a4b", "BaseTen",
    GEMMA26_SCREEN["no_filler_pass_rate_pct"],
    GEMMA26_SCREEN["dots_pass_rate_pct"],
    f"{gemma26_delta:+.1f}".replace("-", "−"), "",
    f'{GEMMA26_SCREEN["included_runs"][0]} / {GEMMA26_SCREEN["included_runs"][1]}',
    GEMMA26_SCREEN["key"], GEMMA26_SCREEN["interpretation"],
    round(GEMMA26_SCREEN["no_filler_ttfat_p50_ms"]),
))
GEMMA26_METHOD_MARKDOWN = (
    " Gemma 4 26B A4B adds a separate fixed-denominator, temporally paired BaseTen "
    f"comparison with {GEMMA26_SCREEN['included_runs'][0]} fresh contemporaneous conversations "
    "per arm and native thinking disabled."
)
GEMMA26_LIMITS_MARKDOWN = (
    " The Gemma 4 26B comparison is attempt-based and paired within its collection window; "
    "it does not reuse the older README control."
)
GEMMA26_PROVENANCE_MARKDOWN = (
    " The Gemma 4 26B row and its README row share the fresh BaseTen no-filler arm; "
    "the screen TTFAT is that row configuration's observed-response P50."
)
GEMMA26_METHOD_HTML = GEMMA26_METHOD_MARKDOWN
GEMMA26_LIMITS_HTML = GEMMA26_LIMITS_MARKDOWN
GEMMA26_PROVENANCE_HTML = GEMMA26_PROVENANCE_MARKDOWN
# GEMMA26_PUBLICATION_DATA_END
EXPECTED_MODEL_COUNT = 25 + int(GEMINI25_RESULT is not None)
if len(MODELS) != EXPECTED_MODEL_COUNT or len({row[0] for row in MODELS}) != EXPECTED_MODEL_COUNT:
    raise ValueError(f"the exploratory screen must contain {EXPECTED_MODEL_COUNT} unique rows")
SENSITIVITY_PATH = (Path(__file__).resolve().parents[1] /
                    "docs/filler-study-data/gemini-minimal-dots-2026-07-21/idle-timeout-sensitivity.json")
if not SENSITIVITY_PATH.is_file():
    raise RuntimeError(f"Gemini idle-timeout sensitivity is required: {SENSITIVITY_PATH}")
SENSITIVITY = json.loads(SENSITIVITY_PATH.read_text())
if SENSITIVITY.get("artifact_status") != "SENSITIVITY_ONLY_NOT_PRIMARY":
    raise ValueError("Gemini idle-timeout result is not marked sensitivity-only")
SENSITIVITY_PRIMARY = SENSITIVITY.get("primary_attempt_based", {})
SENSITIVITY_REPLACEMENT = SENSITIVITY.get("replacement_sensitivity", {})
for sensitivity_arm in (SENSITIVITY_PRIMARY, SENSITIVITY_REPLACEMENT):
    if (
        sensitivity_arm.get("n_attempts") != 30
        or sensitivity_arm.get("fixed_turn_denominator") != 900
    ):
        raise ValueError("Gemini idle-timeout sensitivity must preserve an n=30 fixed denominator")
N30_PATH = (Path(__file__).resolve().parents[1] /
            "docs/filler-study-data/dot-stability-n30-2026-07-20/aggregates.json")
N30_KEY_BY_NAME = {
    "gpt-5.4": "gpt54",
    "gpt-5.6-terra": "terra",
    "gpt-5.5": "gpt55",
    "gpt-5.6-sol": "sol",
    "gemma-4-31b-it": "gemma431",
    "inkling": "inkling",
    "qwen3-8b": "qwen3_8b",
    "glm-5.2": "glm52",
}
FOCUSED = {}


def signed(value):
    return f"{value:+.1f}".replace("-", "−")


if not N30_PATH.is_file():
    raise RuntimeError(f"focused n=30 aggregates are required: {N30_PATH}")
N30_PAYLOAD = json.loads(N30_PATH.read_text())
if N30_PAYLOAD.get("protocol", {}).get("target_per_arm") != 30:
    raise ValueError("focused aggregate target must be 30 per arm")
n30 = N30_PAYLOAD.get("models", {})
if set(n30) != set(N30_KEY_BY_NAME.values()):
    raise ValueError(f"focused aggregate model mismatch: {sorted(n30)}")
refreshed = []
for row in MODELS:
    name, provider, base, filler, delta, p, runs, key, verdict, ttfat = row
    n30_key = N30_KEY_BY_NAME.get(name)
    if n30_key:
        result = n30[n30_key]
        if set(result.get("arms", {})) != {"nofiller", "dots96"}:
            raise ValueError(f"focused aggregate arm mismatch for {name}")
        control = result["arms"]["nofiller"]
        dots = result["arms"]["dots96"]
        if control.get("n_attempts") != 30 or dots.get("n_attempts") != 30:
            raise ValueError(f"focused aggregate is not n=30 for {name}")
        provider = result.get("provider")
        if name == "qwen3-8b":
            source = N30_PAYLOAD.get("protocol", {}).get("primary_sources", {}).get(n30_key, {})
            if (
                provider != "BaseTen"
                or source.get("lane") != "baseten-qwen"
                or source.get("provider") != "BaseTen"
                or source.get("historical_attempts_included") != 0
                or source.get("openrouter_attempts_included") != 0
            ):
                raise ValueError("Qwen aggregate is not the BaseTen-only replacement cohort")
        effect = result["effect"]
        base = control["pass_rate_pct"]
        filler = dots["pass_rate_pct"]
        raw_delta = effect["pass_delta_points"]
        ci = effect["pass_delta_ci95"]
        delta = signed(raw_delta)
        key = "pos" if ci[0] > 0 else "neg" if ci[1] < 0 else "null"
        verdict = "increase" if key == "pos" else "decrease" if key == "neg" else "uncertain"
        runs = "30 / 30"
        ttfat = round(control["ttfat_p50_ms"])
        FOCUSED[name] = {
            "ci": ci,
            "completion": [control["strict_completion_pct"], dots["strict_completion_pct"]],
            "raw_delta": raw_delta,
            "control": control,
            "dots": dots,
        }
    refreshed.append((name, provider, base, filler, delta, p, runs, key, verdict, ttfat))
MODELS = refreshed
for result in GEMINI_ROWS.values():
    if result.get("report_tier") != "focused":
        continue
    control = result["arms"]["nofiller"]
    dots = result["arms"]["dots96"]
    if control.get("n_attempts") != 30 or dots.get("n_attempts") != 30:
        raise ValueError(f"focused Gemini aggregate is not n=30: {result['display_name']}")
    effect = result["effect"]
    FOCUSED[result["display_name"]] = {
        "ci": effect["pass_delta_ci95"],
        "completion": [control["strict_completion_pct"], dots["strict_completion_pct"]],
        "raw_delta": effect["pass_delta_points"],
        "control": control,
        "dots": dots,
    }
if GEMINI25_RESULT is not None and GEMINI25_RESULT.get("report_tier") == "focused":
    control = GEMINI25_RESULT["arms"]["nofiller"]
    dots = GEMINI25_RESULT["arms"]["dots96"]
    if control.get("n_attempts") != 30 or dots.get("n_attempts") != 30:
        raise ValueError("focused Gemini 2.5 aggregate is not n=30")
    effect = GEMINI25_RESULT["effect"]
    FOCUSED[GEMINI25_RESULT["display_name"]] = {
        "ci": effect["pass_delta_ci95"],
        "completion": [control["strict_completion_pct"], dots["strict_completion_pct"]],
        "raw_delta": effect["pass_delta_points"],
        "control": control,
        "dots": dots,
    }
FOCUSED[LAGUNA_RESULT["display_name"]] = {
    "ci": laguna_effect["pass_delta_ci95"],
    "completion": [
        laguna_control["strict_completion_pct"],
        laguna_dots["strict_completion_pct"],
    ],
    "raw_delta": laguna_effect["pass_delta_points"],
    "control": laguna_control,
    "dots": laguna_dots,
}
PROSPECTIVE_DETAILS = {}
if GEMINI25_RESULT is not None:
    PROSPECTIVE_DETAILS[GEMINI25_RESULT["display_name"]] = {
        "completion": [
            GEMINI25_RESULT["arms"]["nofiller"]["strict_completion_pct"],
            GEMINI25_RESULT["arms"]["dots96"]["strict_completion_pct"],
        ]
    }
for report_name, result in QWEN_RESULTS.items():
    PROSPECTIVE_DETAILS[report_name] = {
        "completion": [
            result["arms"]["nofiller"]["strict_completion_pct"],
            result["arms"]["dots96"]["strict_completion_pct"],
        ]
    }
# INKLING_SMALL_PUBLICATION_DETAIL_START
PROSPECTIVE_DETAILS["inkling-small"] = {
    "completion": INKLING_SMALL_SCREEN["strict_completion_pct"]
}
if INKLING_SMALL_SCREEN.get("focused") is True:
    FOCUSED["inkling-small"] = {
        "ci": INKLING_SMALL_SCREEN["ci95"],
        "completion": INKLING_SMALL_SCREEN["strict_completion_pct"],
        "raw_delta": INKLING_SMALL_SCREEN["dots_minus_control_points"],
        "control": {
            "pass_rate_pct": INKLING_SMALL_SCREEN["no_filler_pass_rate_pct"],
            "ttfat_p50_ms": INKLING_SMALL_SCREEN["none_ttfat_p50_ms"],
        },
        "dots": {"pass_rate_pct": INKLING_SMALL_SCREEN["dots_pass_rate_pct"]},
    }
# INKLING_SMALL_PUBLICATION_DETAIL_END
# GEMMA26_PUBLICATION_DETAIL_START
PROSPECTIVE_DETAILS["gemma-4-26b-a4b"] = {
    "completion": GEMMA26_SCREEN["strict_completion_pct"]
}
if GEMMA26_SCREEN.get("focused") is True:
    FOCUSED["gemma-4-26b-a4b"] = {
        "ci": GEMMA26_SCREEN["ci95"],
        "completion": GEMMA26_SCREEN["strict_completion_pct"],
        "raw_delta": GEMMA26_SCREEN["dots_minus_control_points"],
        "control": {
            "pass_rate_pct": GEMMA26_SCREEN["no_filler_pass_rate_pct"],
            "ttfat_p50_ms": GEMMA26_SCREEN["no_filler_ttfat_p50_ms"],
        },
        "dots": {"pass_rate_pct": GEMMA26_SCREEN["dots_pass_rate_pct"]},
    }
# GEMMA26_PUBLICATION_DETAIL_END
TURN_FAMILY_PATH = (Path(__file__).resolve().parents[1] /
                    "docs/filler-study-data/turn-family-secondary-2026-07-22/aggregates.json")
if not TURN_FAMILY_PATH.is_file():
    raise RuntimeError(f"turn-family secondary aggregate is required: {TURN_FAMILY_PATH}")
TURN_FAMILY_PAYLOAD = json.loads(TURN_FAMILY_PATH.read_text())
if TURN_FAMILY_PAYLOAD.get("artifact_status") != "FINAL_EXPLORATORY_SECONDARY":
    raise ValueError("turn-family secondary aggregate is not final and exploratory")
turn_family_protocol = TURN_FAMILY_PAYLOAD.get("protocol", {})
if (
    turn_family_protocol.get("n_per_arm") != 30
    or turn_family_protocol.get("bootstrap_unit") != "whole conversation"
    or turn_family_protocol.get("primary_estimand_unchanged") is not True
):
    raise ValueError("turn-family secondary protocol mismatch")
TURN_FAMILIES = TURN_FAMILY_PAYLOAD.get("taxonomy", {}).get("families", [])
if sorted(turn for family in TURN_FAMILIES for turn in family.get("turns", [])) != list(range(30)):
    raise ValueError("turn-family taxonomy is not exhaustive")
TURN_FAMILY_MODELS = {
    result["display_name"]: result
    for result in TURN_FAMILY_PAYLOAD.get("models", {}).values()
}
TURN_FAMILY_KEY_BY_NAME = {
    result["display_name"]: key
    for key, result in TURN_FAMILY_PAYLOAD.get("models", {}).items()
}
TURN_FAMILY_KEYS = turn_family_protocol.get("model_order", [])
TURN_FAMILY_ORDER = [
    TURN_FAMILY_PAYLOAD["models"][key]["display_name"] for key in TURN_FAMILY_KEYS
]
if (
    len(TURN_FAMILY_ORDER) != 11
    or len(set(TURN_FAMILY_ORDER)) != 11
    or set(TURN_FAMILY_MODELS) != set(TURN_FAMILY_ORDER)
    or not set(TURN_FAMILY_ORDER).issubset(FOCUSED)
):
    raise ValueError("turn-family artifact must remain the frozen 11-model mechanism cohort")
if (
    TURN_FAMILY_PAYLOAD.get("schema_version") != 2
    or set(TURN_FAMILY_KEY_BY_NAME) != set(TURN_FAMILY_ORDER)
    or any(len(result.get("turns", [])) != 30 for result in TURN_FAMILY_MODELS.values())
):
    raise ValueError("turn-family per-turn extension is missing or incomplete")
REASONING_PATH = (Path(__file__).resolve().parents[1] /
                  "docs/filler-study-data/gpt54-reasoning-comparison.json")
if not REASONING_PATH.is_file():
    raise RuntimeError(f"GPT-5.4 reasoning comparison is required: {REASONING_PATH}")
REASONING_PAYLOAD = json.loads(REASONING_PATH.read_text())
if REASONING_PAYLOAD.get("artifact_status") != "DESCRIPTIVE_NOT_INTERACTION_TEST":
    raise ValueError("GPT-5.4 reasoning comparison is not marked descriptive-only")
REASONING = REASONING_PAYLOAD.get("reasoning_effort", {})
if set(REASONING) != {"none", "low"}:
    raise ValueError("GPT-5.4 reasoning comparison must contain none and low")
if REASONING["none"].get("n_per_arm") != 30 or REASONING["low"].get("n_per_arm") != 8:
    raise ValueError("GPT-5.4 reasoning comparison sample-size mismatch")
if set(REASONING["none"]) != {"n_per_arm", "nofiller", "dots96", "effect"}:
    raise ValueError("GPT-5.4 reasoning-off comparison schema mismatch")
if set(REASONING["low"]) != {"n_per_arm", "nofiller", "dots96", "effect"}:
    raise ValueError("GPT-5.4 low-reasoning comparison schema mismatch")
DOSE = [(0,90.3,83,97,677),(24,95.4,93,97,657),(48,91.2,87,97,649),(96,96.3,90,100,658),(192,97.5,90,100,641)]
ABL = [("no filler (baseline)",90.3,"","base"),
       ("96 dots · suffix",96.3,"p=0.0072","pos"),
       ("96 dashes · suffix",97.5,"p=0.0007","pos"),
       ("96×“the” · suffix",94.2,"p=0.055","sugg"),
       ("96 dots · prefix",97.5,"p=0.0007","pos"),
       ("96 dots · system prompt",91.3,"p=0.737","null")]
TURNS_NF = [100,100,100,100,100,100,100,100,100,100,100,100,0,100,100,10,90,60,100,90,100,100,100,100,60,100,100,100,100,100]
TURNS_D96= [100,100,100,100,100,100,100,100,100,100,100,100,70,100,100,70,100,80,100,100,100,100,100,100,80,90,100,100,100,100]
ABORTS = [("gpt-5.4 · any pattern, any position","0 / 66",0.0),
          ("nemotron-super · thinking on · dots or dashes","0 / 6",0.0),
          ("gpt-5.6-terra · dashes","3 / 12",25.0),
          ("gpt-5.6-terra · dots","25 / 33",76.0),
          ("nemotron-super · thinking off · dots","24 / 24",100.0)]

CK = {"pos":"var(--pos)","neg":"var(--neg)","null":"var(--nul)","sugg":"var(--pos)","warn":"var(--pos)","base":"var(--ink)"}

def esc(s): return H.escape(str(s))

# ---------------- fig 1: dumbbell ----------------
def fig_dumbbell():
    x0,x1,w = 250,700,None
    lo,hi = 75,101
    def X(v): return x0+(v-lo)/(hi-lo)*(x1-x0)
    rh=30; top=34
    rows=len(MODELS)
    height=top+rows*rh+34
    s=[f'<svg viewBox="0 0 960 {height}" role="img" aria-label="Filler effect per model">']
    # axis
    for v in (80,85,90,95,100):
        s.append(f'<line x1="{X(v):.0f}" y1="{top-6}" x2="{X(v):.0f}" y2="{top+rows*rh}" class="hair"/>')
        s.append(f'<text x="{X(v):.0f}" y="{top-12}" class="ax" text-anchor="middle">{v}</text>')
    s.append(f'<text x="{x1+8}" y="{top-12}" class="ax" text-anchor="start">pass %</text>')
    y=top+rh//2
    for name,prov,b,f,d,p,n,key,verdict,ttfat in MODELS:
        c=CK[key]; dim=' opacity="0.55"' if key in ("null",) else ''
        faded='0.45' if key=="null" else '1'
        bx=max(x0,min(x1,X(b))); fx=max(x0,min(x1,X(f)))
        s.append(f'<text x="0" y="{y+4}" class="lbl">{esc(name)}</text>')
        s.append(f'<line x1="{bx:.1f}" y1="{y}" x2="{fx:.1f}" y2="{y}" stroke="{c}" stroke-width="2" opacity="{faded}"/>')
        both_left = b < lo and f < lo
        if b < lo:
            b_y = y - 3 if both_left else y
            s.append(f'<circle cx="{x0}" cy="{b_y}" r="4" fill="var(--paper)" stroke="{c}" stroke-width="1.6" opacity="{faded}"/>')
            if not both_left:
                s.append(f'<text x="{x0+11}" y="{y-7}" class="ax" fill="{c}">{b:.1f}% no filler · point left of scale</text>')
        else:
            s.append(f'<circle cx="{bx:.1f}" cy="{y}" r="4" fill="var(--paper)" stroke="{c}" stroke-width="1.6" opacity="{faded}"/>')
        if f < lo:
            f_y = y + 3 if both_left else y
            s.append(f'<circle cx="{x0}" cy="{f_y}" r="4.2" fill="{c}" opacity="{faded}"/>')
            if both_left:
                s.append(f'<text x="{x0+11}" y="{y-7}" class="ax" fill="{c}">{b:.1f} → {f:.1f}% · both points left of scale</text>')
            else:
                s.append(f'<text x="{x0+11}" y="{y-7}" class="ax" fill="{c}">{f:.1f}% dots · point left of scale</text>')
        else:
            s.append(f'<circle cx="{fx:.1f}" cy="{y}" r="4.2" fill="{c}" opacity="{faded}"/>')
        if name in FOCUSED:
            ci=FOCUSED[name]["ci"]
            ci0=max(x0,min(x1,X(b+ci[0]))); ci1=max(x0,min(x1,X(b+ci[1])))
            s.append(f'<line x1="{ci0:.1f}" y1="{y+7}" x2="{ci1:.1f}" y2="{y+7}" stroke="{c}" stroke-width="1.3" opacity="0.8"/>')
            s.append(f'<line x1="{ci0:.1f}" y1="{y+4}" x2="{ci0:.1f}" y2="{y+10}" stroke="{c}" stroke-width="1"/>')
            s.append(f'<line x1="{ci1:.1f}" y1="{y+4}" x2="{ci1:.1f}" y2="{y+10}" stroke="{c}" stroke-width="1"/>')
        star = ""  # Any footnote marker is carried in the displayed delta itself.
        provider_label = "AI Studio" if prov == "Google" else prov
        s.append(f'<text x="{x1+14}" y="{y+4}" class="num" fill="{c}" opacity="{faded}">{esc(d)}{star}'
                 f'<tspan class="pval"> · {ttfat} ms</tspan>'
                 f'<tspan class="provider"> · {esc(provider_label)}</tspan></text>')
        y+=rh
    s.append(f'<circle cx="{x0}" cy="{y+6}" r="4" fill="var(--paper)" stroke="var(--mut)" stroke-width="1.6"/><text x="{x0+10}" y="{y+10}" class="ax">no filler</text>')
    s.append(f'<circle cx="{x0+110}" cy="{y+6}" r="4.2" fill="var(--mut)"/><text x="{x0+120}" y="{y+10}" class="ax">+ 96 dots</text>')
    s.append(f'<text x="{x0+230}" y="{y+10}" class="ax">labels: Δ · row-config P50 TTFAT · provider · focused 95% CI whisker</text>')
    s.append('</svg>')
    return "".join(s)

# ---------------- fig 2: dose-response ----------------
def fig_dose():
    x0,x1=90,780; lo,hi=0,192
    def X(v): return x0+(v-lo)/(hi-lo)*(x1-x0)
    # accuracy panel
    aT,aB=30,150; ylo,yhi=82,101
    def Y(v): return aB-(v-ylo)/(yhi-ylo)*(aB-aT)
    s=['<svg viewBox="0 0 860 260" role="img" aria-label="gpt-5.4 dose response">']
    for v in (85,90,95,100):
        s.append(f'<line x1="{x0-6}" y1="{Y(v):.0f}" x2="{x1}" y2="{Y(v):.0f}" class="hair"/>')
        s.append(f'<text x="{x0-12}" y="{Y(v)+4:.0f}" class="ax" text-anchor="end">{v}</text>')
    s.append(f'<text x="{x0-58}" y="{aT-10}" class="ax">pass %</text>')
    pts=[]
    for d,m,mn,mx,_ in DOSE:
        s.append(f'<line x1="{X(d):.1f}" y1="{Y(mn):.1f}" x2="{X(d):.1f}" y2="{Y(mx):.1f}" stroke="var(--mut)" stroke-width="1.4" opacity="0.6"/>')
        pts.append((X(d),Y(m)))
    path="M"+" L".join(f"{x:.1f} {y:.1f}" for x,y in pts)
    s.append(f'<path d="{path}" fill="none" stroke="var(--pos)" stroke-width="2"/>')
    for (x,y),(d,m,_,_,_) in zip(pts,DOSE):
        fill = "var(--paper)" if d==48 else "var(--pos)"
        stroke=' stroke="var(--pos)" stroke-width="1.6"' if d==48 else ''
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{fill}"{stroke}/>')
        dy = 22 if d==48 else -12
        s.append(f'<text x="{x:.1f}" y="{y+dy:.1f}" class="num" text-anchor="middle" fill="var(--ink)">{m:.1f}</text>')
    s.append(f'<text x="{X(48)+10:.0f}" y="{Y(91.2)+38:.0f}" class="ax">48-dot cell: cluster-p=0.70 vs baseline</text>')
    # latency panel
    lT,lB=190,232; llo,lhi=0,800
    def LY(v): return lB-(v-llo)/(lhi-llo)*(lB-lT)
    s.append(f'<line x1="{x0-6}" y1="{LY(0):.0f}" x2="{x1}" y2="{LY(0):.0f}" class="hair"/>')
    lp=[]
    for d,_,_,_,t in DOSE: lp.append((X(d),LY(t)))
    lpath="M"+" L".join(f"{x:.1f} {y:.1f}" for x,y in lp)
    s.append(f'<path d="{lpath}" fill="none" stroke="var(--mut)" stroke-width="1.6"/>')
    for (x,y),(d,_,_,_,t) in zip(lp,DOSE):
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="var(--mut)"/>')
        s.append(f'<text x="{x:.1f}" y="{y-8:.1f}" class="ax" text-anchor="middle">{t}</text>')
    s.append(f'<text x="{x0-58}" y="{lT+4}" class="ax">TTFAT</text><text x="{x0-58}" y="{lT+18}" class="ax">P50 ms</text>')
    for d,_,_,_,_ in DOSE:
        s.append(f'<text x="{X(d):.1f}" y="254" class="ax" text-anchor="middle">{d}</text>')
    s.append(f'<text x="{x1+8}" y="254" class="ax">dots</text>')
    s.append('</svg>')
    return "".join(s)

# ---------------- fig 3: ablation ----------------
def fig_abl():
    x0,x1=280,720; lo,hi=88,100
    def X(v): return x0+(v-lo)/(hi-lo)*(x1-x0)
    rh=32; top=30; height=top+len(ABL)*rh+30
    s=[f'<svg viewBox="0 0 860 {height}" role="img" aria-label="Pattern and position ablation">']
    for v in (90,94,98):
        s.append(f'<line x1="{X(v):.0f}" y1="{top-6}" x2="{X(v):.0f}" y2="{top+len(ABL)*rh}" class="hair"/>')
        s.append(f'<text x="{X(v):.0f}" y="{top-12}" class="ax" text-anchor="middle">{v}</text>')
    s.append(f'<text x="{x1+8}" y="{top-12}" class="ax">pass %</text>')
    base=ABL[0][1]; y=top+rh//2
    bx=X(base)
    s.append(f'<line x1="{bx:.1f}" y1="{top-2}" x2="{bx:.1f}" y2="{top+len(ABL)*rh}" stroke="var(--mut)" stroke-width="1" stroke-dasharray="2 3"/>')
    for name,v,p,key in ABL:
        c=CK[key]
        s.append(f'<text x="0" y="{y+4}" class="lbl">{esc(name)}</text>')
        if key!="base":
            s.append(f'<line x1="{bx:.1f}" y1="{y}" x2="{X(v):.1f}" y2="{y}" stroke="{c}" stroke-width="2" opacity="{0.5 if key=="null" else 1}"/>')
        s.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="4.2" fill="{c}" opacity="{0.6 if key=="null" else 1}"/>')
        lbl=f'{v:.1f}' + (f'<tspan class="pval">  {esc(p)}</tspan>' if p else '')
        s.append(f'<text x="{x1+14}" y="{y+4}" class="num" fill="var(--ink)">{lbl}</text>')
        y+=rh
    s.append('</svg>')
    return "".join(s)

# ---------------- fig 4: turn strip ----------------
def fig_strip():
    cw,ch,gap=24,24,3; x0=120; top=26
    s=[f'<svg viewBox="0 0 900 120" role="img" aria-label="Per-turn failure rates">']
    for row,(name,vals) in enumerate((("no filler",TURNS_NF),("+ 96 dots",TURNS_D96))):
        y=top+row*(ch+gap)
        s.append(f'<text x="{x0-10}" y="{y+ch-7}" class="lbl" text-anchor="end">{name}</text>')
        for t,v in enumerate(vals):
            fail=(100-v)/100
            x=x0+t*(cw+gap)*0.92
            if fail<=0.001:
                s.append(f'<rect x="{x:.1f}" y="{y}" width="{cw*0.92-1:.1f}" height="{ch}" class="cell"/>')
            else:
                s.append(f'<rect x="{x:.1f}" y="{y}" width="{cw*0.92-1:.1f}" height="{ch}" fill="var(--neg)" opacity="{0.18+0.82*fail:.2f}"/>')
    y2=top+2*(ch+gap)+16
    for t in (0,5,10,15,20,25,29):
        x=x0+t*(cw+gap)*0.92+cw*0.45
        s.append(f'<text x="{x:.1f}" y="{y2}" class="ax" text-anchor="middle">{t}</text>')
    s.append(f'<text x="{x0+30*(cw+gap)*0.92+4:.0f}" y="{y2}" class="ax">turn</text>')
    for t in (12,15):
        x=x0+t*(cw+gap)*0.92+cw*0.45
        s.append(f'<text x="{x:.1f}" y="{top-8}" class="ax" text-anchor="middle">▼</text>')
    s.append('</svg>')
    return "".join(s)

# ---------------- fig 5: reasoning-effort slices ----------------
def fig_reasoning():
    x0,x1=245,680
    pass_lo,pass_hi=88,101
    latency_lo,latency_hi=600,1200
    def PX(v): return x0+(v-pass_lo)/(pass_hi-pass_lo)*(x1-x0)
    def LX(v): return x0+(v-latency_lo)/(latency_hi-latency_lo)*(x1-x0)

    s=['<svg viewBox="0 0 860 315" role="img" aria-label="GPT-5.4 filler effects at reasoning effort none and low">']
    s.append('<text x="0" y="18" class="lbl">PASS RATE</text>')
    for value in (90,92,94,96,98,100):
        x=PX(value)
        s.append(f'<line x1="{x:.1f}" y1="28" x2="{x:.1f}" y2="148" class="hair"/>')
        s.append(f'<text x="{x:.1f}" y="22" class="ax" text-anchor="middle">{value}</text>')
    for effort,y in (("none",65),("low",120)):
        row=REASONING[effort]
        base=row["nofiller"]["pass_rate_pct"]
        dots=row["dots96"]["pass_rate_pct"]
        effect=row["effect"]["pass_delta_points"]
        ci=row["effect"]["pass_delta_ci95"]
        bx,dx=PX(base),PX(dots)
        ci0,ci1=PX(base+ci[0]),PX(base+ci[1])
        s.append(f'<text x="0" y="{y+4}" class="lbl">effort {effort} · n={row["n_per_arm"]}/arm</text>')
        s.append(f'<line x1="{bx:.1f}" y1="{y}" x2="{dx:.1f}" y2="{y}" stroke="var(--pos)" stroke-width="2"/>')
        s.append(f'<circle cx="{bx:.1f}" cy="{y}" r="4.2" fill="var(--paper)" stroke="var(--pos)" stroke-width="1.6"/>')
        s.append(f'<circle cx="{dx:.1f}" cy="{y}" r="4.4" fill="var(--pos)"/>')
        s.append(f'<text x="{bx:.1f}" y="{y-11}" class="ax" text-anchor="middle">{base:.1f} no filler</text>')
        s.append(f'<text x="{dx:.1f}" y="{y-11}" class="ax" text-anchor="middle">{dots:.1f} +dots</text>')
        s.append(f'<line x1="{ci0:.1f}" y1="{y+11}" x2="{ci1:.1f}" y2="{y+11}" stroke="var(--pos)" stroke-width="1.3" opacity=".8"/>')
        s.append(f'<line x1="{ci0:.1f}" y1="{y+7}" x2="{ci0:.1f}" y2="{y+15}" stroke="var(--pos)" stroke-width="1"/>')
        s.append(f'<line x1="{ci1:.1f}" y1="{y+7}" x2="{ci1:.1f}" y2="{y+15}" stroke="var(--pos)" stroke-width="1"/>')
        s.append(f'<text x="{x1+16}" y="{y+4}" class="num" fill="var(--pos)">Δ {signed(effect)} <tspan class="pval">[{signed(ci[0])}, {signed(ci[1])}]</tspan></text>')

    s.append('<text x="0" y="181" class="lbl">TTFAT P50</text>')
    for value in (700,900,1100):
        x=LX(value)
        s.append(f'<line x1="{x:.1f}" y1="190" x2="{x:.1f}" y2="276" class="hair"/>')
        s.append(f'<text x="{x:.1f}" y="184" class="ax" text-anchor="middle">{value} ms</text>')
    for effort,y in (("none",218),("low",263)):
        row=REASONING[effort]
        base=round(row["nofiller"]["ttfat_p50_ms"])
        dots=round(row["dots96"]["ttfat_p50_ms"])
        bx,dx=LX(base),LX(dots)
        latency_delta=dots-base
        s.append(f'<text x="0" y="{y+4}" class="lbl">effort {effort} · n={row["n_per_arm"]}/arm</text>')
        s.append(f'<line x1="{bx:.1f}" y1="{y}" x2="{dx:.1f}" y2="{y}" stroke="var(--mut)" stroke-width="2"/>')
        s.append(f'<circle cx="{bx:.1f}" cy="{y}" r="4" fill="var(--paper)" stroke="var(--mut)" stroke-width="1.6"/>')
        s.append(f'<circle cx="{dx:.1f}" cy="{y}" r="4.2" fill="var(--mut)"/>')
        s.append(f'<text x="{(bx+dx)/2:.1f}" y="{y-11}" class="ax" text-anchor="middle">{base} → {dots} ms</text>')
        s.append(f'<text x="{x1+16}" y="{y+4}" class="num" fill="var(--mut)">+{latency_delta} ms</text>')
    difference=REASONING_PAYLOAD["descriptive_effect_difference_points"]
    s.append(f'<text x="0" y="306" class="ax">descriptive difference between effects: {signed(difference)} points · not an interaction test</text>')
    s.append('</svg>')
    return "".join(s)

# ---------------- tables ----------------
def master_table():
    rows=[]
    for name,prov,b,f,d,p,n,key,verdict,ttfat in MODELS:
        chip=f'<span class="chip chip-{key}">{esc(verdict)}</span>'
        detail=FOCUSED.get(name)
        completion_detail = detail or PROSPECTIVE_DETAILS.get(name)
        delta_cell=esc(d)
        completion='—'
        if detail:
            ci=detail["ci"]
            delta_cell=f'{esc(d)} <span class="mut">[{signed(ci[0])}, {signed(ci[1])}]</span>'
        if completion_detail:
            completion=f'{completion_detail["completion"][0]:.0f}% → {completion_detail["completion"][1]:.0f}%'
        rows.append(f'<tr><td>{esc(name)}</td><td class="mut">{esc(prov)}</td>'
                    f'<td class="r">{b:.1f}</td><td class="r">{f:.1f}</td>'
                    f'<td class="r em">{delta_cell}</td><td class="r mut">{completion}</td>'
                    f'<td class="r mut">{ttfat}</td><td class="r mut">{esc(n)}</td><td>{chip}</td></tr>')
    return ('<table><thead><tr><th>model</th><th>served via</th><th class="r">no filler</th>'
            '<th class="r">+96 dots</th><th class="r">Δ pt [95% CI, focused]</th>'
            '<th class="r">strict completion</th><th class="r">P50 TTFAT ms<br>(row config)</th>'
            '<th class="r">included runs no/dots</th><th>status</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>')


def turn_effect_heatmaps():
    """Aligned, fixed-scale turn views for strict pass and missing-turn contribution."""
    focused_order = list(TURN_FAMILY_ORDER)
    family_by_turn = {
        turn: family
        for family in TURN_FAMILIES
        for turn in family["turns"]
    }
    family_code = {
        "grounded_information": "G",
        "recommendation_protocol": "R",
        "tool_preparation": "P",
        "tool_commitment": "C",
        "interaction_boundary": "B",
    }
    width, x0, cell_w, row_h = 940, 198, 23.2, 21
    cap = 50.0
    panels = (
        ("pass", 28, 90, "+96 dots − no filler strict-pass rate (points)"),
        ("missing", 366, 428, "missing-turn contribution: no filler − +96 dots (points)"),
    )
    parts = [
        f'<svg class="turn-heatmaps" viewBox="0 0 {width} 674" role="img" '
        'aria-label="Per-turn filler effects and aligned missing-turn contributions for eleven models">',
        '<text x="520" y="18" class="ax">blue = benefit · orange = harm · color clipped at ±50 points</text>',
    ]
    for panel, title_y, row_top, label in panels:
        family_y = row_top - 30
        turn_y = row_top - 12
        parts.append(
            f'<g class="turn-heatmap" data-panel="{panel}" data-color-cap="{cap:.1f}">'
            f'<text x="0" y="{title_y}" class="lbl">{esc(label)}</text>'
        )
        for turn in range(30):
            x = x0 + turn * cell_w
            family = family_by_turn[turn]
            parts.append(
                f'<text x="{x + 10.4:.1f}" y="{family_y}" class="family-code" text-anchor="middle">'
                f'{family_code[family["key"]]}</text>'
                f'<text x="{x + 10.4:.1f}" y="{turn_y}" class="ax" text-anchor="middle">{turn}</text>'
            )
        for row_index, name in enumerate(focused_order):
            key = TURN_FAMILY_KEY_BY_NAME[name]
            y = row_top + row_index * row_h
            parts.append(f'<text x="188" y="{y + 13}" class="heatmap-model" text-anchor="end">{esc(name)}</text>')
            turns = TURN_FAMILY_MODELS[name]["turns"]
            for turn_row in turns:
                turn = turn_row["turn"]
                family_key = turn_row["family_key"]
                if panel == "pass":
                    value = turn_row["pass_delta_points"]
                    low, high = turn_row["pass_delta_ci95"]
                    title = (
                        f'{name}; turn {turn}; {family_by_turn[turn]["label"]}; '
                        f'pass: no filler {turn_row["nofiller_pass_rate_pct"]:.1f}%, '
                        f'dots {turn_row["dots96_pass_rate_pct"]:.1f}%, '
                        f'delta {value:+.1f} points; pointwise 95% CI [{low:+.1f}, {high:+.1f}]'
                    )
                else:
                    value = turn_row["aligned_missing_contribution_points"]
                    raw_delta = turn_row["missing_turn_rate_delta_points"]
                    title = (
                        f'{name}; turn {turn}; {family_by_turn[turn]["label"]}; '
                        f'missing: no filler {turn_row["nofiller_missing_turn_rate_pct"]:.1f}%, '
                        f'dots {turn_row["dots96_missing_turn_rate_pct"]:.1f}%; '
                        f'benefit-aligned contribution {value:+.1f} points '
                        f'(dots-minus-control missing delta {raw_delta:+.1f})'
                    )
                x = x0 + turn * cell_w
                if abs(value) < 0.05:
                    fill, opacity, css_class = "var(--chipbg)", 1.0, "zero"
                else:
                    fill = "var(--pos)" if value > 0 else "var(--neg)"
                    opacity = 0.16 + 0.78 * min(abs(value) / cap, 1)
                    css_class = "pos" if value > 0 else "neg"
                parts.append(
                    f'<rect class="turn-cell {css_class}" data-panel="{panel}" '
                    f'data-model-key="{esc(key)}" data-turn="{turn}" data-family="{esc(family_key)}" '
                    f'data-value="{value:.12g}" data-color-cap="{cap:.1f}" '
                    f'x="{x:.1f}" y="{y}" width="21.2" height="17" rx="1" '
                    f'fill="{fill}" opacity="{opacity:.3f}"><title>{esc(title)}</title></rect>'
                )
        parts.append('</g>')
    legend = " · ".join(
        f'{family_code[family["key"]]} {family["short_label"]}' for family in TURN_FAMILIES
    )
    parts.append(f'<text x="0" y="344" class="ax">turn family: {esc(legend)}</text>')
    parts.append('</svg>')
    return "".join(parts)


def family_contribution_matrix():
    focused_order = list(TURN_FAMILY_ORDER)
    cells = [
        '<div class="family-contribution-matrix" role="table" '
        'aria-label="Task-family contributions to each overall filler effect">',
        '<div class="family-contribution-corner" role="columnheader">30-turn contribution</div>',
    ]
    for family in TURN_FAMILIES:
        cells.append(
            f'<div class="family-contribution-head" role="columnheader">{esc(family["short_label"])}'
            f'<span>{len(family["turns"])} / 30 weight</span></div>'
        )
    cells.append('<div class="family-contribution-head total" role="columnheader">Σ overall<span>fig 1 point estimate</span></div>')
    cap = 12.0
    for name in focused_order:
        key = TURN_FAMILY_KEY_BY_NAME[name]
        model = TURN_FAMILY_MODELS[name]
        cells.append(f'<div class="family-contribution-model" role="rowheader">{esc(name)}</div>')
        for family in TURN_FAMILIES:
            result = model["families"][family["key"]]
            value = result["overall_contribution_points"]
            direction = "pos" if value > 0.05 else "neg" if value < -0.05 else "zero"
            strength = 8 + 35 * min(abs(value) / cap, 1)
            title = (
                f'{name}; {family["label"]}; {value:+.1f} points contributed to the '
                f'30-turn overall effect ({result["conditional_delta_points"]:+.1f} within-family '
                f'× {len(family["turns"])}/30)'
            )
            cells.append(
                f'<div class="family-contribution-cell {direction}" role="cell" '
                f'data-estimand="overall-contribution" data-model-key="{esc(key)}" '
                f'data-family="{esc(family["key"])}" data-value="{value:.12g}" '
                f'style="--family-strength:{strength:.1f}%" title="{esc(title)}">'
                f'{signed(value)}</div>'
            )
        overall = model["overall"]["delta_points"]
        cells.append(
            f'<div class="family-contribution-total" role="cell" data-model-key="{esc(key)}" '
            f'data-value="{overall:.12g}"><b>{signed(overall)}</b></div>'
        )
    cells.append('</div>')
    return "".join(cells)


def turn_family_matrix():
    focused_order = list(TURN_FAMILY_ORDER)
    cells = ['<div class="family-matrix" role="table" aria-label="Turn-family filler effects">',
             '<div class="family-corner" role="columnheader">dots − no filler</div>']
    for family in TURN_FAMILIES:
        cells.append(
            f'<div class="family-head" role="columnheader">{esc(family["short_label"])}'
            f'<span>{len(family["turns"])} turns</span></div>'
        )
    for name in focused_order:
        cells.append(f'<div class="family-model" role="rowheader">{esc(name)}</div>')
        model = TURN_FAMILY_MODELS[name]
        for family in TURN_FAMILIES:
            result = model["families"][family["key"]]
            delta = result["conditional_delta_points"]
            low, high = result["conditional_delta_ci95"]
            direction = "pos" if delta > 0.05 else "neg" if delta < -0.05 else "zero"
            strength = 7 + 31 * min(abs(delta) / 50, 1)
            title = (
                f'{name}; {family["label"]}; no filler {result["nofiller_pass_rate_pct"]:.1f}%; '
                f'dots {result["dots96_pass_rate_pct"]:.1f}%; within-family delta {delta:+.1f} points'
            )
            cells.append(
                f'<div class="family-cell {direction}" role="cell" '
                f'data-model="{esc(name)}" data-family="{esc(family["key"])}" '
                f'style="--family-strength:{strength:.1f}%" title="{esc(title)}">'
                f'<b>{signed(delta)}</b><span>[{signed(low)}, {signed(high)}]</span></div>'
            )
    cells.append('</div>')
    return "".join(cells)


def turn_family_findings(markdown=False):
    text = (
        "Effects do not follow a universal family rule: positive and negative tool-commitment "
        "estimates both occur, and some near-zero overall effects contain offsetting family contributions."
    )
    return text if markdown else esc(text)


def turn_family_html_section(turn_heatmaps, contribution_matrix, exact_matrix):
    return f'''<!-- TURN_FAMILY_HTML_START -->
<h3 id="turn-family-effects">Cross-model descriptive decomposition</h3>
<p class="measure">The original pilot suggested a turn-specific mechanism. We therefore applied a
five-family taxonomy to the same 11 focused n=30 comparisons and also show every scripted turn,
without selecting only the largest cells. The taxonomy was frozen from benchmark semantics without
the 11-model family outcomes, but after the pilot and primary overall results were known. The
decomposition is retrospective and exploratory.</p>
<figure id="turn-heatmap"><figcaption class="eyebrow" style="margin-bottom:.6rem">fig 3b · all 30 scripted turns · 11 focused models · 30 attempts per arm</figcaption>
{turn_heatmaps}
<figcaption><b>The upper panel is the strict-pass change; the lower panel isolates the part due to
missing turns.</b> Both use benefit-aligned signs: blue means dots improved pass rate or reduced
missing turns. Color is fixed across both panels and clipped at ±50 points; hover text gives exact
rates and, for pass cells, pointwise intervals. Long suffix bands in the lower panel generally
represent one early exit propagated through later fixed-denominator turns—not many independent
failures.</figcaption></figure>
<figure id="family-contributions"><figcaption class="eyebrow" style="margin-bottom:.6rem">fig 3c · task-family contribution to the 30-turn overall effect</figcaption>
<div class="tblwrap">{contribution_matrix}</div>
<figcaption><b>Each family cell is its within-family effect multiplied by its turn count / 30.</b>
The five signed cells add exactly to the outlined overall point estimate at right. This puts unequal
family sizes on the same scale as fig 1; it is a decomposition, not a separate effect estimate.</figcaption></figure>
<p class="measure"><b>What the decomposition shows.</b> {turn_family_findings()}</p>
<details class="exact-family"><summary>Exact within-family estimates and pointwise intervals</summary>
<div class="tblwrap">{exact_matrix}</div>
<p class="fine">Cells are fixed-denominator +96-dots minus no-filler pass-rate points. Brackets are
pointwise, unadjusted 95% whole-conversation bootstrap intervals. The 55 intervals are not
simultaneous; a zero-width bootstrap interval at a boundary is not population certainty.</p></details>
<p class="fine">These 330 turn cells and 55 family cells are retrospective descriptive
decompositions, not treatment-by-turn or treatment-by-family interaction tests. Turn cells are
single scripted positions, not independent tasks, and their intervals are pointwise and unadjusted.
Turn families differ in size and position; early termination propagates missing failures into later
turns. Fig 1 remains the primary inferential summary.</p>
<!-- TURN_FAMILY_HTML_END -->'''


def abort_table():
    rows=[]
    for name,frac,pct in ABORTS:
        bar=f'<div class="bar"><div class="barfill" style="width:{pct:.0f}%"></div></div>'
        rows.append(f'<tr><td>{esc(name)}</td><td class="r mut">{esc(frac)}</td>'
                    f'<td class="r em">{pct:.0f}%</td><td class="barcell">{bar}</td></tr>')
    return ('<table><thead><tr><th>configuration</th><th class="r">aborts</th>'
            '<th class="r">rate</th><th></th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>')


def markdown_primary_section():
    gemini25_method = (
        " Gemini 2.5 Flash is a separate prospective fixed-denominator extension with "
        "thinking explicitly disabled via `thinking_budget=0`. Its prespecified dot screen "
        "stopped at 10/6; a later control-only precision extension expanded no filler to 30 "
        "for the public benchmark estimate without reopening dot sampling."
        if GEMINI25_RESULT is not None else ""
    )
    gemini25_provenance = (
        " The appended Gemini 2.5 Flash row and its `(thinking off)` README row share a "
        "separate campaign aggregate; the chart's open control point uses all 30 no-filler "
        "conversations while its exploratory dot point remains at six."
        if GEMINI25_RESULT is not None else ""
    )
    laguna_method = (
        " Laguna S 2.1 is a separate frozen prospective 30/30 campaign using the paid "
        "OpenRouter route to Poolside-hosted BF16 weights, with reasoning explicitly disabled."
    )
    laguna_provenance = (
        " The appended Laguna S 2.1 row comes from its separate 30/30 campaign aggregate; "
        "both arms use `reasoning.enabled=false`, and its TTFAT is specific to the paid "
        "OpenRouter/Poolside BF16 route."
    )
    qwen_method = (
        " The two Qwen3.6 rows are separate exploratory fixed-denominator comparisons "
        "with native thinking disabled. Each reuses a frozen 30-conversation no-filler "
        "cohort and applies the prespecified 6→10→30 stopping rule only to the later dot arm; "
        "the arms are not contemporaneous or interleaved."
    )
    qwen_provenance = (
        " The appended Qwen3.6 rows use BaseTen single-H100 vLLM 0.26 APC+MTP deployments: "
        "official BF16 weights for 27B and the official FP8 checkpoint for 35B-A3B. Their "
        "open control points use all 30 reusable no-filler conversations, while each dot "
        "point uses its mechanically selected stopped-stage sample."
    )
    lines = [
        "## Primary screen: 96 trailing dots vs no filler, thinking off or provider-minimal",
        "",
        "| model | endpoint | no filler | +96 dots | Δ pt [95% CI, focused] | strict completion | P50 TTFAT ms (row config) | included runs no/dots | interpretation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, provider, base, filler, delta, _p, runs, _key, verdict, ttfat in MODELS:
        detail = FOCUSED.get(name)
        completion_detail = detail or PROSPECTIVE_DETAILS.get(name)
        delta_cell = delta
        completion = "—"
        if detail:
            low, high = detail["ci"]
            delta_cell = f"{delta} [{signed(low)}, {signed(high)}]"
        if completion_detail:
            completion = f'{completion_detail["completion"][0]:.0f}% → {completion_detail["completion"][1]:.0f}%'
        lines.append(
            f"| {name} | {provider} | {base:.1f} | {filler:.1f} | {delta_cell} | "
            f"{completion} | {ttfat} | {runs} | {verdict} |"
        )
    lines.extend([
        "",
        f"The {len(FOCUSED)} focused rows use exactly 30 eligible conversations and 900 fixed scripted turns per arm. Missing, malformed, or forfeited future turns fail all displayed criteria. Their intervals resample whole conversations. The three appended Gemini 3 rows use prospective fixed-denominator pools at Google's `minimal` reasoning floor; Gemini 3 does not guarantee complete thinking-off.{gemini25_method}{laguna_method}{qwen_method}{INKLING_SMALL_METHOD_MARKDOWN}{GEMMA26_METHOD_MARKDOWN} The other nine standard rows retain their original exploratory available-outcome pools and show no new confidence interval; `†` marks a selected historical estimate.",
        "",
        f"The original 17 rows retain their exploratory-screen order rather than being resorted after the n=30 refresh; the three Gemini 3 extensions are appended in their prespecified requested order, followed by Gemini 2.5 Flash, Laguna S 2.1, and the two Qwen3.6 configurations from their separate campaigns. The original eight focused rows and the corresponding thinking-off rows in `README.md` share the same frozen no-filler aggregates. The Gemini 3 rows and their `(minimal)` README rows likewise share one campaign aggregate.{gemini25_provenance}{laguna_provenance}{qwen_provenance}{INKLING_SMALL_PROVENANCE_MARKDOWN}{GEMMA26_PROVENANCE_MARKDOWN} The Qwen3-8B primary row uses only its dedicated 30-per-arm BaseTen replacement cohort, including its no-filler TTFAT; no OpenRouter attempt is pooled into it. That endpoint serves official BF16 weights with vLLM automatic prefix caching, so its latency is specific to that configuration.",
        "",
        f"Flash Lite attempt-policy sensitivity: one no-filler attempt reached the harness idle timeout after eight turns. Under the frozen attempt-based rule, it remains in the primary pool and its missing future turns fail. Replacing it only for sensitivity analysis with the already-generated complete extra attempt moves the no-filler pass rate from {SENSITIVITY_PRIMARY['pass_rate_pct']:.1f}% to {SENSITIVITY_REPLACEMENT['pass_rate_pct']:.1f}% ({SENSITIVITY['pass_rate_change_points']:+.1f} points) and strict completion from {SENSITIVITY_PRIMARY['strict_completion_pct']:.1f}% to {SENSITIVITY_REPLACEMENT['strict_completion_pct']:.1f}%. The primary estimates remain attempt-based and unchanged.",
        "",
        "Nemotron Super is excluded from this screen because its available comparison holds +96 dots fixed and changes thinking mode, rather than estimating a filler effect. The thinking-off mode was not repaired: all 24 dot-treated attempts called `end_session` at turn 0. The separate thinking-on result was 91.7% over four judged conversations; a 6/6 BaseTen gate established completion only. Three Modal cells without automatic prefix caching (APC) also completed 6/6, while APC+MTP caused a distinct tool-execution collapse.",
        "",
        "Each row reports one pooled turn-level P50 TTFAT from that row's no-filler configuration—not an arm-to-arm timing comparison.",
    ])
    return "\n".join(lines)


def update_markdown_primary():
    start = "<!-- N30_PRIMARY_START -->"
    end = "<!-- N30_PRIMARY_END -->"
    text = MARKDOWN_OUT.read_text()
    # GEMMA26_MARKDOWN_SCOPE_START
    scope_words = {24: "Twenty-four", 25: "Twenty-five", 26: "Twenty-six"}
    expected_scope = scope_words.get(len(MODELS), str(len(MODELS)))
    scope_variants = [
        f"**Scope:** {word} standard filler comparisons"
        for word in scope_words.values()
    ]
    matched_scope = [variant for variant in scope_variants if variant in text]
    if len(matched_scope) != 1:
        raise ValueError("Markdown report scope count is missing or ambiguous")
    text = text.replace(
        matched_scope[0],
        f"**Scope:** {expected_scope} standard filler comparisons",
    )
    # GEMMA26_MARKDOWN_SCOPE_END
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("Markdown primary-screen markers are missing or duplicated")
    before, remainder = text.split(start, 1)
    _old, after = remainder.split(end, 1)
    MARKDOWN_OUT.write_text(
        before + start + "\n" + markdown_primary_section() + "\n" + end + after
    )


def markdown_turn_family_section():
    headers = [f'{family["short_label"]} ({len(family["turns"])})' for family in TURN_FAMILIES]
    lines = [
        "### Cross-model descriptive decomposition",
        "",
        "The HTML report shows two aligned 11×30 heatmaps: the strict-pass change at every scripted turn and the benefit-aligned contribution from missing turns. It also shows each task family's additive contribution to the 30-turn overall effect. No turn was selected because of its observed result.",
        "",
        "A separate reviewer assigned every scripted turn to one behavioral family using only the benchmark specification—not the 11-model filler outcomes. The mapping was frozen before family-level computation, but after the pilot and primary overall results were known. This is retrospective and exploratory, not an outcome-naive preregistration.",
        "",
        "| model | " + " | ".join(headers) + " |",
        "|---|" + "|".join("---:" for _ in headers) + "|",
    ]
    for name in TURN_FAMILY_ORDER:
        values = []
        for family in TURN_FAMILIES:
            row = TURN_FAMILY_MODELS[name]["families"][family["key"]]
            low, high = row["conditional_delta_ci95"]
            values.append(f"{signed(row['conditional_delta_points'])} [{signed(low)}, {signed(high)}]")
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    mapping = "; ".join(
        f'{family["short_label"]}: {", ".join(str(turn) for turn in family["turns"])}'
        for family in TURN_FAMILIES
    )
    lines.extend([
        "",
        "Cells are fixed-denominator within-family dot-minus-control pass-rate points with pointwise, unadjusted 95% whole-conversation bootstrap intervals. The number in each header is the family turn count. No interval has simultaneous 95% coverage across the 55 model-family cells. In the HTML contribution matrix, each cell is this value multiplied by the family turn count / 30, and the five cells sum exactly to the primary overall point estimate.",
        "",
        turn_family_findings(markdown=True),
        "",
        f"Frozen zero-based turn mapping — {mapping}.",
        "",
        "The companion artifact partitions every failure into a missing/post-abort turn or an observed judged failure. Long suffix bands can therefore reflect one early exit propagated through later fixed-denominator turns, not many independent semantic failures. These 330 turn cells and 55 family cells are descriptive decompositions, not treatment-by-turn or treatment-by-family interaction tests. Families differ in size and position. The whole-conversation intervals in fig 1 remain the primary inferential summary.",
    ])
    return "\n".join(lines)


def update_markdown_turn_family():
    start = "<!-- TURN_FAMILY_START -->"
    end = "<!-- TURN_FAMILY_END -->"
    insert = "<!-- TURN_FAMILY_INSERT -->"
    text = MARKDOWN_OUT.read_text()
    block = start + "\n" + markdown_turn_family_section() + "\n" + end
    if text.count(insert) != 1:
        raise ValueError("Markdown turn-family insertion marker is missing or duplicated")
    if start in text or end in text:
        if text.count(start) != 1 or text.count(end) != 1 or text.index(start) > text.index(end):
            raise ValueError("Markdown turn-family markers are missing, reversed, or duplicated")
        before, remainder = text.split(start, 1)
        _old, after = remainder.split(end, 1)
        text = before.rstrip() + "\n\n" + after.lstrip()
    text = text.replace(insert, insert + "\n\n" + block)
    next_heading = text.find("\n## ", text.index(insert) + len(insert))
    if not (
        text.count(start) == text.count(end) == 1
        and text.index(insert) < text.index(start) < text.index(end)
        and (next_heading == -1 or text.index(end) < next_heading)
    ):
        raise ValueError("Markdown turn-family block is not inside the mechanism section")
    MARKDOWN_OUT.write_text(text)

# ---------------- page ----------------
CSS = """
:root{--paper:#FAF9F6;--ink:#22252A;--mut:#71767D;--pos:#2A6F97;--neg:#B0432A;
--nul:#9AA0A6;--hair:#E6E3DC;--card:#F1EFE9;--chipbg:#ECEAE4;}
@media (prefers-color-scheme: dark){:root{--paper:#15171B;--ink:#E7E5E0;--mut:#9BA1A8;
--pos:#7FB3D5;--neg:#E08A6D;--nul:#6E747B;--hair:#2C2F35;--card:#1C1F24;--chipbg:#23262C;}}
:root[data-theme="dark"]{--paper:#15171B;--ink:#E7E5E0;--mut:#9BA1A8;--pos:#7FB3D5;
--neg:#E08A6D;--nul:#6E747B;--hair:#2C2F35;--card:#1C1F24;--chipbg:#23262C;}
:root[data-theme="light"]{--paper:#FAF9F6;--ink:#22252A;--mut:#71767D;--pos:#2A6F97;
--neg:#B0432A;--nul:#9AA0A6;--hair:#E6E3DC;--card:#F1EFE9;--chipbg:#ECEAE4;}
html{background:var(--paper);}
body{margin:0;background:var(--paper);color:var(--ink);
font:17px/1.62 Charter,"Bitstream Charter",Georgia,"Times New Roman",serif;}
.wrap{max-width:860px;margin:0 auto;padding:3rem 1.4rem 5rem;}
.measure{max-width:70ch;}
.eyebrow{font:600 12px/1 ui-monospace,"SF Mono",Menlo,Consolas,monospace;
letter-spacing:.14em;text-transform:uppercase;color:var(--mut);}
h1{font-size:2.1rem;line-height:1.15;margin:.5rem 0 .4rem;text-wrap:balance;font-weight:700;}
.sub{color:var(--mut);font-size:1.02rem;max-width:62ch;margin:0 0 .4rem;}
.meta{font:12.5px/1.7 ui-monospace,"SF Mono",Menlo,monospace;color:var(--mut);
border-top:1px solid var(--hair);border-bottom:1px solid var(--hair);
padding:.55rem 0;margin:1.4rem 0 2.6rem;display:flex;gap:1.6rem;flex-wrap:wrap;}
h2{font-size:1.34rem;margin:3rem 0 .8rem;text-wrap:balance;}
h2 .no{color:var(--mut);font-weight:400;margin-right:.5rem;font-size:1.05rem;
font-family:ui-monospace,Menlo,monospace;}
h3{font-size:1.06rem;margin:1.8rem 0 .5rem;}
p{margin:.75rem 0;}
strong{font-weight:700;}
.pos-t{color:var(--pos)}.neg-t{color:var(--neg)}
figure{margin:1.8rem 0;}
figcaption{font-size:.86rem;color:var(--mut);max-width:70ch;margin-top:.5rem;line-height:1.5;}
figcaption b{color:var(--ink);font-weight:600;}
svg{width:100%;height:auto;display:block;}
svg .ax{font:11px ui-monospace,Menlo,monospace;fill:var(--mut);}
svg .lbl{font:12.5px ui-monospace,Menlo,monospace;fill:var(--ink);}
svg .num{font:12.5px ui-monospace,Menlo,monospace;}
svg .pval{fill:var(--mut);font-size:11px;}
svg .provider{fill:var(--mut);font-size:10.5px;}
svg .hair{stroke:var(--hair);stroke-width:1;}
svg .cell{fill:var(--chipbg);}
table{border-collapse:collapse;width:100%;margin:1.2rem 0;
font:14px/1.55 ui-monospace,"SF Mono",Menlo,monospace;font-variant-numeric:tabular-nums;}
th{font-weight:600;text-align:left;color:var(--mut);font-size:11.5px;
text-transform:uppercase;letter-spacing:.07em;padding:.45rem .7rem .45rem 0;
border-bottom:1px solid var(--ink);}
td{padding:.42rem .7rem .42rem 0;border-bottom:1px solid var(--hair);vertical-align:top;}
td.r,th.r{text-align:right;}
td.mut{color:var(--mut);}
td.em{font-weight:700;}
.chip{display:inline-block;font-size:11px;padding:.1rem .5rem;border-radius:2px;
background:var(--chipbg);white-space:nowrap;}
.chip-pos{color:var(--pos);}.chip-sugg{color:var(--pos);opacity:.75;}
.chip-warn{color:var(--neg);}.chip-neg{color:var(--neg);}.chip-null{color:var(--mut);}
.bar{background:var(--chipbg);height:10px;width:100%;min-width:130px;margin-top:5px;}
.barfill{background:var(--neg);height:10px;}
td.barcell{width:34%;}
.callout{background:var(--card);border-left:3px solid var(--pos);
padding:.9rem 1.1rem;margin:1.4rem 0;font-size:.95rem;}
.callout.warn{border-left-color:var(--neg);}
.exchange{background:var(--card);padding:.9rem 1.1rem;margin:1rem 0;
font:13.5px/1.6 ui-monospace,Menlo,monospace;overflow-x:auto;}
.exchange .who{color:var(--mut);}
.kv{display:grid;grid-template-columns:auto 1fr;gap:.25rem 1.2rem;
font:13.5px/1.6 ui-monospace,Menlo,monospace;margin:1rem 0;}
.kv dt{color:var(--mut);white-space:nowrap;}.kv dd{margin:0;}
ol,ul{padding-left:1.3rem;}li{margin:.35rem 0;}
.fine{font-size:.86rem;color:var(--mut);line-height:1.55;}
a{color:var(--pos);}
.tblwrap{overflow-x:auto;}
.family-matrix{display:grid;grid-template-columns:minmax(135px,1.15fr) repeat(5,minmax(116px,1fr));
gap:3px;min-width:760px;font:11px/1.3 ui-monospace,"SF Mono",Menlo,monospace;
font-variant-numeric:tabular-nums;margin:1rem 0;}
.family-corner,.family-head,.family-model,.family-cell{padding:.42rem .45rem;}
.family-corner,.family-head{color:var(--mut);border-bottom:1px solid var(--ink);align-self:end;}
.family-head span{display:block;font-size:10px;margin-top:.15rem;}
.family-model{color:var(--ink);display:flex;align-items:center;border-bottom:1px solid var(--hair);}
.family-cell{border-bottom:1px solid var(--hair);text-align:right;
background:color-mix(in srgb,var(--family-color) var(--family-strength),var(--paper));}
.family-cell.pos{--family-color:var(--pos);}.family-cell.neg{--family-color:var(--neg);}
.family-cell.zero{--family-color:var(--nul);}
.family-cell b,.family-cell span,.family-cell small{display:block;}
.family-cell span,.family-cell small{color:var(--mut);font-size:10px;font-weight:400;}
.turn-heatmaps .heatmap-model{font:11.5px ui-monospace,"SF Mono",Menlo,monospace;fill:var(--ink);}
.turn-heatmaps .family-code{font:600 10px ui-monospace,"SF Mono",Menlo,monospace;fill:var(--mut);}
.family-contribution-matrix{display:grid;
grid-template-columns:minmax(135px,1.15fr) repeat(5,minmax(92px,1fr)) minmax(92px,.9fr);
gap:3px;min-width:760px;font:11px/1.3 ui-monospace,"SF Mono",Menlo,monospace;
font-variant-numeric:tabular-nums;margin:1rem 0;}
.family-contribution-corner,.family-contribution-head,.family-contribution-model,
.family-contribution-cell,.family-contribution-total{padding:.48rem .45rem;}
.family-contribution-corner,.family-contribution-head{color:var(--mut);border-bottom:1px solid var(--ink);align-self:end;}
.family-contribution-head span{display:block;font-size:9.5px;margin-top:.15rem;}
.family-contribution-head.total{border-left:1px solid var(--mut);padding-left:.7rem;}
.family-contribution-model{color:var(--ink);display:flex;align-items:center;border-bottom:1px solid var(--hair);}
.family-contribution-cell{border-bottom:1px solid var(--hair);text-align:right;
background:color-mix(in srgb,var(--family-color) var(--family-strength),var(--paper));}
.family-contribution-cell.pos{--family-color:var(--pos);}.family-contribution-cell.neg{--family-color:var(--neg);}
.family-contribution-cell.zero{--family-color:var(--nul);}
.family-contribution-total{border-left:1px solid var(--mut);border-bottom:1px solid var(--hair);
text-align:right;padding-left:.7rem;}
.exact-family{margin:1.4rem 0;border-top:1px solid var(--hair);border-bottom:1px solid var(--hair);padding:.65rem 0;}
.exact-family summary{cursor:pointer;font:600 12px/1.5 ui-monospace,"SF Mono",Menlo,monospace;color:var(--mut);}
@media (prefers-reduced-motion: reduce){*{transition:none!important;}}
"""

def build():
    f1,f2,f3,f4,f5 = fig_dumbbell(),fig_dose(),fig_abl(),fig_strip(),fig_reasoning()
    family_matrix = turn_family_matrix()
    turn_heatmaps = turn_effect_heatmaps()
    contribution_matrix = family_contribution_matrix()
    turn_family_section = turn_family_html_section(
        turn_heatmaps, contribution_matrix, family_matrix
    )
    focused_order = [name for name, *_ in MODELS if name in FOCUSED]
    focused_parts = []
    positive = negative = uncertain = 0
    for name in focused_order:
        detail = FOCUSED[name]
        control = detail["control"]["pass_rate_pct"]
        dots = detail["dots"]["pass_rate_pct"]
        delta = detail["raw_delta"]
        low, high = detail["ci"]
        if low > 0:
            positive += 1
        elif high < 0:
            negative += 1
        else:
            uncertain += 1
        focused_parts.append(
            f'{esc(name)} {control:.1f}→{dots:.1f}% '
            f'({signed(delta)} pt, 95% CI {signed(low)} to {signed(high)})'
        )
    focused_effects = "; ".join(focused_parts)
    focused_counts = (
        f"Among the {len(focused_order)} fixed n=30 comparisons, {positive} effect interval"
        f"{'s' if positive != 1 else ''} lie entirely above zero, {negative} "
        f"entirely below zero, and {uncertain} span zero."
    )
    gpt54 = FOCUSED["gpt-5.4"]
    gpt54_control = gpt54["control"]["pass_rate_pct"]
    gpt54_dots = gpt54["dots"]["pass_rate_pct"]
    gpt54_delta = gpt54["raw_delta"]
    gpt54_ttfat = round(gpt54["control"]["ttfat_p50_ms"])
    gemini_control_total = sum(row["arms"]["nofiller"]["n_attempts"] for row in GEMINI_ROWS.values())
    gemini_dots_total = sum(row["arms"]["dots96"]["n_attempts"] for row in GEMINI_ROWS.values())
    gemini25_control_total = GEMINI25_RESULT["arms"]["nofiller"]["n_attempts"] if GEMINI25_RESULT else 0
    gemini25_dots_total = GEMINI25_RESULT["arms"]["dots96"]["n_attempts"] if GEMINI25_RESULT else 0
    model_count_word = {
        21: "Twenty-one",
        22: "Twenty-two",
        23: "Twenty-three",
        24: "Twenty-four",
        25: "Twenty-five",
        26: "Twenty-six",
    }.get(len(MODELS), str(len(MODELS)))
    gemini25_method_html = (
        " A separate Gemini 2.5 Flash extension uses <code>thinking_budget=0</code> "
        "(full thinking-off) and the same adaptive rule. Its dot screen stopped at 10/6; a "
        "later control-only precision extension expanded no filler to 30 without reopening "
        "dot sampling."
        if GEMINI25_RESULT else ""
    )
    gemini25_limits_html = (
        " The Gemini 2.5 Flash extension is also fixed-denominator and attempt-based."
        if GEMINI25_RESULT else ""
    )
    gemini25_provenance_html = (
        " The Gemini 2.5 Flash row and its <code>(thinking off)</code> README row share a "
        "separate campaign with <code>thinking_budget=0</code>; its public no-filler estimate "
        "uses 30 conversations and its exploratory dot arm remains at six."
        if GEMINI25_RESULT else ""
    )
    laguna_method_html = (
        " Laguna S 2.1 uses a separate frozen prospective 30/30 campaign on the paid "
        "OpenRouter route to Poolside-hosted BF16 weights, with reasoning explicitly disabled."
    )
    laguna_limits_html = (
        " The Laguna S 2.1 campaign is likewise fixed-denominator and attempt-based."
    )
    laguna_provenance_html = (
        " The Laguna S 2.1 row comes from its separate 30/30 campaign with "
        "<code>reasoning.enabled=false</code>; its TTFAT is specific to the paid "
        "OpenRouter route to Poolside-hosted BF16 weights."
    )
    qwen_method_html = (
        " The two Qwen3.6 rows are separate exploratory fixed-denominator comparisons "
        "with native thinking disabled. Each reuses a frozen 30-conversation no-filler "
        "cohort and applies the prespecified 6→10→30 rule only to the later dot arm; "
        "the arms are neither contemporaneous nor interleaved."
    )
    qwen_limits_html = (
        " The Qwen3.6 comparisons use fixed denominators and attempt-based eligibility, "
        "but their reused controls and later treatment timing make them exploratory even "
        "if a dot arm reaches 30."
    )
    qwen_provenance_html = (
        " The Qwen3.6 rows use BaseTen single-H100 vLLM 0.26 APC+MTP deployments: "
        "official BF16 weights for 27B and the official FP8 checkpoint for 35B-A3B. "
        "Each open control point uses all 30 reusable no-filler conversations; each dot "
        "point uses the mechanically selected stopped-stage sample."
    )
    page=f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Filler-Token Latent Scratchpad — a {len(MODELS)}-Model Exploratory Study</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
<div class="eyebrow">aiewf-eval · research report</div>
<h1>Filler-Token Latent Scratchpad</h1>
<p class="sub">Appending content-free tokens sometimes improves low-latency accuracy without a
large observed change in median TTFAT. A {len(MODELS)}-model filler screen on a 30-turn voice-agent benchmark,
with dose-response, mechanism, and failure modes.</p>
<div class="meta"><span>benchmark aiwf_medium_context · 30 turns · tool-calling voice agent</span>
<span>{len(FOCUSED)} focused models · 30 conversations/arm</span><span>July 18–28, 2026</span></div>
</header>

<section class="measure">
<h2><span class="no">1</span>Summary</h2>
<ol>
<li><strong>The refreshed screen remains heterogeneous.</strong> The original eight prespecified rows,
plus any prospective extension promoted or completed at the focused tier, use 30 eligible conversations per arm,
fixed 900-turn denominators, and whole-conversation bootstrap intervals; the other rows remain
visible as exploratory context. {focused_counts}
The focused estimates are: {focused_effects}.</li>
<li><strong>The original GPT-5.4 pilot localized its gain to decisiveness-like turns.</strong>
About three-quarters of that pilot's net gain concentrated in two tool-commitment turns where the
unfilled model asked a redundant confirmation question instead of acting. The refreshed n=30
aggregate estimates the overall effect more precisely; the turn-localization figure remains the
prespecified pilot diagnostic rather than a reselected n=30 analysis.</li>
<li><strong>GPT-5.4 also improved with reasoning effort set to low (fig 5).</strong> Holding that model
setting fixed, pass rate rose from 96.2% without filler to <strong>99.6%</strong> with +96 dots
(+40&thinsp;ms). This compares the two filler conditions at one reasoning setting; it does not
measure how filler and reasoning effort interact.</li>
<li><strong>Position and pattern are candidates for follow-up.</strong> Positive estimates appear late in the prompt on
either side of the question (dots: prefix +7.2, suffix +6.0) but shows nothing in the system
prompt (+0.9, p=0.737). Dashes do at least as well as dots (no direct equivalence test); even a
repeated English word trends at half strength (p=0.055). Pattern also changes the observed
<em>hazard</em> in the small follow-up cells (next item).</li>
<li><strong>It can break tool discipline.</strong> Trailing dots are associated with early
termination on some models: nemotron-super aborts 100% of conversations at turn 0;
gpt-5.6-terra 76% (25 of 33 attempts).
The observed Terra abort rate is 25% with dashes in a smaller cell, with survivor accuracy similar — hazard and
benefit may respond to different knobs. The interaction appears model- and configuration-specific;
null estimates are not evidence that the glyphs are harmless.</li>
<li><strong>The separate GPT-5.4 dash confirmation succeeded at a smaller effect.</strong>
In 41 fresh conversations per arm on the pinned snapshot, no filler scored 92.44% and +96 dashes
scored 95.37%: +2.93 points (95% CI +1.09 to +4.77), with strict completion 41/41 and median
TTFAT 678&thinsp;ms in both arms. This confirms that selected dash configuration, not dots or a
general glyph equivalence claim.</li>
</ol>
</section>

<section class="measure">
<h2><span class="no">2</span>Background &amp; method</h2>
<p>The probe follows arXiv 2607.03502 (building on Pfau et al.’s “Let’s Think Dot by
Dot”): the latent-compute hypothesis proposes that content-free filler positions give a frozen
model extra <em>parallel</em> compute without emitting reasoning tokens. This study tests the
resulting behavior and TTFAT, not internal computation. The paper
tested open-weights models on toy tasks; this study tests closed and open APIs on a realistic task:
a 30-turn scripted voice-agent conversation over a conference knowledge base with function calling
(<span class="chip">aiwf_medium_context</span>).</p>
<dl class="kv">
<dt>injection</dt><dd>N space-separated glyphs (a nominal count, not tokenizer-verified) appended to the final
user message of each request; conversation history stays filler-free, so the persisted context never
teaches the pattern</dd>
<dt>latency</dt><dd>TTFAT = first user-visible token, content-aware (reasoning deltas excluded)</dd>
<dt>judge</dt><dd>claude-opus-4-5 rubric judge; test–retest on re-judged runs: 90/90 turns
identical. The judge never sees the filler</dd>
<dt>statistics</dt><dd>each focused row contains 30 eligible attempts per arm and 900 fixed
scripted turns per arm. Missing, malformed, or forfeited future turns fail all displayed
criteria. Confidence intervals resample whole conversations (100,000 bootstrap draws); strict
completion intervals use Wilson's method. The three appended Gemini 3 rows use prospective,
fixed-denominator pools at the provider's <code>minimal</code> reasoning floor; their adaptive
dot-screen and n=30 promotion rules were frozen in advance.{gemini25_method_html}{laguna_method_html}{qwen_method_html}{INKLING_SMALL_METHOD_HTML}{GEMMA26_METHOD_HTML} The nine retained historical standard rows and the
separate dose/pattern follow-ups keep their original exploratory analyses</dd>
<dt>inference scope</dt><dd>Section 3 shows estimates and focused-row confidence intervals without
a multiple-testing headline or p-values. The prospective GPT-5.4 dash validation is now complete:
41 fresh conversations per arm, +2.93 points (95% CI +1.09 to +4.77). It confirms the pinned dash
configuration and is not pooled into the dot screen</dd>
<dt>limits</dt><dd>Filler counts are nominal (“96 tokens”) and not verified against each provider’s tokenizer.
The judge check establishes repeatability (90/90 on re-judge), not validity against human
adjudication. All focused rows and all three Gemini 3 extensions are fixed-denominator,
attempt-based analyses.{gemini25_limits_html}{laguna_limits_html}{qwen_limits_html}{INKLING_SMALL_LIMITS_HTML}{GEMMA26_LIMITS_HTML} The nine retained historical standard rows keep their original
available-outcome pools: missing judgments are omitted and
incomplete runs contribute only observed turns, so they should be read as historical context.
gpt-5.4-mini was excluded from the master
accuracy screen after a dedicated investigation found sparse strict completion at effort=none:
0/17 no filler, 1/17 dots, and 3/14 dashes. A separate effort-medium follow-up is reported only as a
failure-mode result below (docs/gpt-5.4-mini-abort-investigation-2026-07-20.md)</dd>
<dt>scale</dt><dd>{len(FOCUSED)} focused comparisons at 30 eligible conversations per arm. Each
prospective Gemini extension starts at 10 no-filler attempts and six dot attempts, with prespecified top-ups
to 10 or promotion of both arms to 30; Laguna S 2.1 completed its separately frozen adaptive sequence at
30 per arm; each Qwen3.6 comparison reuses 30 frozen no-filler controls and adaptively stops its
later dot arm at 6, 10, or 30; generally 6–10 available judged runs per arm for the nine
retained historical standard rows. Original GPT-5.4
follow-ups reused an earlier baseline and remain exploratory</dd>
</dl>
</section>

<section id="primary-screen">
<h2><span class="no">3</span>{model_count_word}-model exploratory screen</h2>
<figure><figcaption class="eyebrow" style="margin-bottom:.6rem">fig 1 · pass rate, no filler → +96 dots</figcaption>
{f1}
<figcaption><b>{focused_counts}</b> Open circle = the row's low-latency no-filler configuration and filled = +96 dots on
the final user turn.
Rows retain the original exploratory-screen order and were not resorted after the n=30 refresh.
Lower whiskers show dot-minus-control effect intervals for the {len(FOCUSED)} focused rows only; † marks a
retained historical available-outcome estimate.</figcaption></figure>
<div class="tblwrap">{master_table()}</div>
<p class="measure"><b>Focused n=30 estimates.</b> {focused_effects}. The intervals, not the
ordering of point estimates, are the intended precision summary. Cross-model similarities remain
hypothesis-generating because checkpoints, providers, and serving stacks differ.</p>
<p class="measure"><b>Run-pool provenance.</b> Pass rates and TTFATs in this section come from
the screen's frozen manifests. The original eight refreshed rows and their thinking-off rows in
<code>README.md</code> share the same n=30 no-filler aggregates. The three Gemini 3 rows and their
<code>(minimal)</code> README rows share the same prospective campaign aggregate; Google's
<code>minimal</code> floor does not guarantee complete thinking-off.{gemini25_provenance_html}{laguna_provenance_html}{qwen_provenance_html}{INKLING_SMALL_PROVENANCE_HTML}{GEMMA26_PROVENANCE_HTML} The nine retained historical
standard rows keep their original exploratory pools. GPT-5.4 uses the rolling Responses alias with explicit effort
<code>none</code>; Gemma uses the Lilac route with thinking disabled. Qwen3-8B uses only its new
30-per-arm dedicated BaseTen cohort; no historical or campaign OpenRouter attempt is pooled into
that primary row, and its displayed TTFAT comes from the BaseTen no-filler arm. The endpoint uses
official BF16 weights with vLLM automatic prefix caching, so the latency is route-specific.</p>
<p class="measure"><b>Flash Lite attempt-policy sensitivity.</b> One no-filler attempt reached
the harness idle timeout after eight turns. Under the frozen attempt-based rule, it remains in the
primary pool and its missing future turns fail. Replacing it only for sensitivity analysis with the
already-generated complete extra attempt moves the no-filler pass rate from
{SENSITIVITY_PRIMARY['pass_rate_pct']:.1f}% to {SENSITIVITY_REPLACEMENT['pass_rate_pct']:.1f}%
({SENSITIVITY['pass_rate_change_points']:+.1f} points) and strict completion from
{SENSITIVITY_PRIMARY['strict_completion_pct']:.1f}% to
{SENSITIVITY_REPLACEMENT['strict_completion_pct']:.1f}%. The primary estimates remain
attempt-based and unchanged.</p>
<p class="measure"><b>Separate Nemotron-super configuration finding.</b> Nemotron-super is excluded
from fig 1 and its table because the available comparison changes thinking mode while holding dots
fixed, rather than estimating a filler effect. The original thinking-off configuration remains
a real failure mode: all 24 dot-treated attempts emitted a spurious <code>end_session</code> at
turn 0, and its historical 84.9% no-filler baseline is survivor-selected. We did not repair that
decoding mode. Enabling native thinking avoided the original filler abort in a 6/6 BaseTen gate.
Three separate no-APC Modal cells—BF16 with MTP, BF16 without MTP, and NVFP4 without MTP—also
completed 6/6; the matched no-MTP BF16 and NVFP4 cells each scored 172/180 with 35/36 required
tools. Automatic prefix caching (APC) plus MTP caused a distinct tool-execution collapse and is
not recommended. These are configuration mitigations, not a repair.</p>
<p class="measure"><b>Latency note.</b> Each row reports one observed pooled turn-level P50 TTFAT
from that row's no-filler configuration—not an arm-to-arm latency comparison.</p>
</section>

<section>
<h2><span class="no">4</span>Original pilot dose-response — similar observed median latency</h2>
<figure><figcaption class="eyebrow" style="margin-bottom:.6rem">fig 2 · original gpt-5.4 pilot, thinking off · accuracy and median latency vs dot count</figcaption>
{f2}
<figcaption><b>+5 to +7 points at 24, 96, and 192 dots; similar sample medians.</b> Whiskers span
run-level min–max. The 48-dot value has no detectable difference from baseline
(cluster-p=0.70, 48-vs-0 comparison) — with n=8 we cannot say whether it is noise or a real non-monotonicity.
Median TTFAT sits in a 640–680&thinsp;ms band at every dose; the cost of the extra input tokens
appears only in the tail (P95 1990&thinsp;ms at 192 dots vs 1546 baseline). Reference: gpt-5.4
with low thinking scores 96.2–97.0% depending on run pool (contemporaneous n=8 vs earlier
leaderboard aggregate) at ~780–1100&thinsp;ms.</figcaption></figure>
</section>

<section id="mechanism">
<h2><span class="no">5</span>Where effects occur: exploratory turn and task-family analysis</h2>
<p class="measure">The original GPT-5.4 pilot generated a decisiveness-over-context hypothesis.
The cross-model views below test where the observed point estimates sit in this fixed script; they
do not replace the whole-conversation effects and intervals in fig 1.</p>
<figure><figcaption class="eyebrow" style="margin-bottom:.6rem">fig 3a · historical pilot · per-turn failure rate across 10 runs · gpt-5.4</figcaption>
{f4}
<figcaption><b>Most of the +6.0 lives in two turns.</b> Cell shade = failure rate at that turn
(blank = all runs pass). Turns 12 and 15 (▼) — both tool-commitment moments — account for
roughly three-quarters of the net improvement (13 of 18 avoided failures); smaller gains at turns
16–24 and one new failure at turn 25 make up the rest.</figcaption></figure>
<div class="measure">
<p>Both driver turns have the same shape — the user supplies the last missing piece, and the
correct move is to call the tool with already-established arguments:</p>
<div class="exchange"><span class="who">turn 15 · user:</span> “Yes.”
<em>(confirming a vegan dietary request; name known since turn 10)</em><br><br>
<span class="who">no filler (9/10 runs fail):</span> “Sure — what’s your name, and what
dietary preference would you like me to submit?”<br>
<span class="who">+96 dots (7/10 runs pass):</span> <b>submit_dietary_request</b>(name=“Jennifer
Smith”, dietary_preference=“vegan”) ✓</div>
<p>Without filler the model re-asks for information it already has; with filler it acts. The judge
(deterministic on re-test, blind to the filler by construction) scores both sides on the same
rubric. One hypothesis is that filler gives the model additional latent computation to resolve
“do I already have everything I need?” — a question about the <em>conversation
history</em>, not about the final user message. (We describe this as latent compute because the
behavioral signature matches — position- and dose-dependence at fixed output — but API-level
experiments cannot observe the computation itself; punctuation semantics or recency effects are
not fully excluded.)</p>
</div>
{turn_family_section}
<h3>Placement follow-up</h3>
<figure><figcaption class="eyebrow" style="margin-bottom:.6rem">fig 4 · original pattern &amp; position follow-up · gpt-5.4, thinking off, count 96, n=8</figcaption>
{f3}
<figcaption><b>Late placement is the leading hypothesis; glyph equivalence is untested.</b> Ablation cells n=8; the
no-filler and dots-suffix rows reuse the n=10 baselines. Dashes do at least as well as dots; a
repeated English word trends at half strength (p=0.055); dots placed <em>before</em> the question have a similar estimate
(+7.2, cluster-p=0.0007) — but the same 96 dots in the system prompt show nothing (+0.9,
p=0.737). These cells argue against a position-independent prompt-lengthening account and are
consistent with, but do not establish, an effect tied to accumulated conversation context. (System-prompt
placement also differs in role and cache treatment; the position inference is behavioral, not
mechanistic.)</figcaption></figure>
</section>

<section id="reasoning-effort">
<h2><span class="no">6</span>Filler effects at two reasoning-effort settings</h2>
<figure><figcaption class="eyebrow" style="margin-bottom:.6rem">fig 5 · gpt-5.4 · two parallel filler comparisons</figcaption>
{f5}
<figcaption><b>Dots improved pass rate at both measured reasoning settings.</b> Open circles are
no filler; filled circles are +96 dots. Accuracy whiskers show whole-conversation 95% bootstrap
intervals for dot-minus-control effects, translated onto the endpoint scale. The
<code>none</code> comparison is the final n=30-per-arm cohort; <code>low</code> is a separate,
contemporaneous exploratory comparison with n=8 per arm. TTFAT values are pooled turn-level
medians within each displayed arm.</figcaption></figure>
<div class="measure">
<p><b>What the two comparisons show.</b> With the OpenAI API's reasoning-effort setting at
<code>none</code>, GPT-5.4 rose from 90.2% to 95.2% with dots: +5.0 points [95% CI +3.0, +6.9].
With reasoning effort at <code>low</code>, it rose from 96.2% to 99.6%: +3.3 points [95% CI
+1.2, +5.0]. “Low” describes the model's internal reasoning-budget setting, not a low-quality or
incomplete run. The low-effort gain was driven by the same turn-15 tool-commitment behavior
identified in the original pilot.</p>
<p><b>What the comparison does not show.</b> The descriptive difference between the two filler
effects is −1.7 points. The reasoning-off and low-effort slices were collected separately and
have unequal sample sizes; reasoning effort was not randomized in one joint 2×2 experiment.
Therefore, −1.7 is not an interaction estimate and does not tell us whether filler substitutes
for or complements reasoning. The latency panel is also descriptive: dots changed P50 TTFAT by
only +5 ms at effort <code>none</code> and +40 ms at effort <code>low</code>, while the low-effort
configuration itself was roughly 400 ms slower in these collections.</p>
<div class="callout"><strong>Measured result, bounded claim.</strong> The filler effect is not
confined to reasoning-off mode. Determining its relationship to reasoning effort requires a new,
balanced experiment that randomizes reasoning setting and filler condition together.</div>
</div>
</section>

<section>
<h2><span class="no">7</span>The failure mode: filler as a stop sign</h2>
<div class="measure">
<p>On some models, trailing dots are associated with early termination. The model’s first action
is to call <span class="chip">end_session</span> — at turn 0, before answering anything. This was
reproducible in every run of one tested configuration and varied by filler pattern.</p></div>
<div class="tblwrap">{abort_table()}</div>
<div class="measure"><p><b>Dashes had fewer aborts in the earlier hazard study; the benefit question stays open.</b> On
terra, that study observed a 76% abort rate with dots versus 25% with dashes (3/12), while survivor accuracy is similar
in these small cells. But survivor scores are treatment-selected on both sides, so no benefit
comparison is causal. An intention-to-treat view is unforgiving: counting
aborted conversations as total failures, terra + dots yields an expected pass rate near
<b>23%</b> — far below its 91.0% no-filler baseline. On nemotron-super with thinking enabled,
no aborts occurred in six attempts, but filler scores were lower in the small cells we measured
(−7.7 dots n=4, −4.4 dashes n=2); no measured filler cell improved that model.</p>
<p><b>Mini remains completion-limited even with reasoning.</b> In a separate gpt-5.4-mini
effort-medium screen, observed strict completion was 4/20 without filler and 8/10 with 96 dashes
(fixed-table Fisher arithmetic p=0.004111, not calibrated for the outcome-dependent stopping
rule). Accuracy among strict completers rose 86.7% → 93.3% (selected-subset cluster-p=0.0054),
but that estimate is survivor-selected and has no unselected-population interpretation. The
screen needs a fixed-size replication.</p></div>
</section>

<section class="measure">
<h2><span class="no">8</span>What this screen does not support</h2>
<ul>
<li><strong>Baseline headroom</strong> — the displayed rows show no simple monotone relationship
between row-configuration baseline score and filler response (fig 1).</li>
<li><strong>Depth</strong> — dense Qwen3 8B/14B/32B (36→64 layers, same recipe and
tokenizer) did not show a consistent size gradient once n=10.</li>
<li><strong>Architecture class</strong> — positive, negative, and null estimates appear on both
sides of the dense/MoE split; this heterogeneous screen does not isolate architecture.</li>
<li><strong>Toy-task transfer</strong> — gains reported on arithmetic benchmarks do not predict
gains on conversational tool use; the effect is task-dependent as well as model-dependent.</li>
<li><strong>Judge artifacts</strong> — the judge was repeatable on re-test (90/90) and filler-blind
by construction, but this does not establish validity against human adjudication.</li>
<li><strong>Position-independent prompt lengthening</strong> — the system-prompt cell showed no
detectable change, while late user-message placements did; role and cache treatment remain
confounded with position.</li>
</ul>
</section>

<section class="measure">
<h2><span class="no">9</span>Practical guidance</h2>
<div class="callout"><strong>The selected GPT-5.4 dash configuration is confirmed.</strong>
The prospective pinned-snapshot run completed 41 fresh conversations per arm: 92.44% without
filler versus 95.37% with 96 trailing dashes, a +2.93-point effect (95% CI +1.09 to +4.77).
Strict completion was 41/41 and median TTFAT was 678&thinsp;ms in both arms. This validates that
specific model/configuration and does not establish dots–dashes equivalence.</div>
<div class="callout"><strong>If a model responds to filler</strong> (only measurement tells you):
include <strong>~96 dashes</strong> appended to the final user message as a validation cell,
with history kept clean. Dashes performed at least as well as dots in the tested gpt-5.4
configurations (without a direct equivalence test) and had fewer aborts in the terra screen. On gpt-5.4
the refreshed dot comparison is {gpt54_control:.1f}% → {gpt54_dots:.1f}%
({signed(gpt54_delta)} points) with a no-filler row TTFAT of {gpt54_ttfat}&thinsp;ms; the separate
(low, +96 dots) = 99.6% @ ~1.1&thinsp;s. These numbers are gpt-5.4’s; other responsive models
need their own dose/pattern validation.</div>
<div class="callout warn"><strong>Never deploy unmeasured.</strong> The study observed silent
accuracy shifts, majority-rate conversation aborts, and total breakage. A benefit in one tested
model configuration does not establish a benefit in another.</div>
</section>

<section class="measure">
<h2><span class="no">10</span>Provenance</h2>
<p class="fine">The focused refresh contains exactly 480 eligible conversations: 30 per arm for
eight prespecified models, assembled from 174 frozen historical attempts, 246 original non-Qwen
scheduled top-ups, and 60 BaseTen Qwen replacement attempts. The focused refresh spans OpenAI,
BaseTen, and Lilac. The Gemini extension adds {gemini_control_total} no-filler and
{gemini_dots_total} dot-treated eligible attempts through the Google Gemini Developer API. The
separate Gemini 2.5 thinking-off extension adds {gemini25_control_total} no-filler and
{gemini25_dots_total} dot-treated attempts when final. The separate Laguna S 2.1 campaign adds
30 no-filler and 30 dot-treated attempts through the paid OpenRouter route to Poolside-hosted BF16 weights.
The nine
retained historical standard rows keep their original exploratory pools, including the report's
OpenRouter and Anthropic endpoints. The separate GPT-5.4 dash confirmation contains
82 fresh conversations.
Harness: <span class="chip">aiewf-eval</span> multi-turn-eval
(commit 3e9f805 plus MTE_FILLER_TOKEN / MTE_FILLER_POSITION knobs). Judge: claude-opus-4-5 via
claude-agent-sdk. Config→run manifests, the permutation-test tool, and the per-cell analyzer are
archived in <span class="chip">docs/filler-study-data/</span>. Focused rates use fixed 900-turn
denominators, missing/post-abort turns as failures, and whole-conversation bootstrap intervals.
Original dose, placement, and retained historical-screen analyses remain explicitly exploratory.</p>
</section>
</div>
</body>
</html>"""
    OUT.write_text(page, encoding="utf-8")
    update_markdown_primary()
    update_markdown_turn_family()
    print(f"wrote {OUT} ({len(page)} bytes)")

build()
