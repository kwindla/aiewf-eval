"""Inventory original captures, including unjudged and compressed runs.

This is a source audit, not a leaderboard or a judge. Counts at a budget may
span serving stacks: the per-capture configuration must travel with a remap.
Run from the repository root, supplying a newline-separated list of paths from
`rg --files --hidden --no-ignore -g transcript.jsonl /home/khkramer /tmp`.
Archive arguments are read in streaming mode; nothing is extracted to disk.
"""

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.aiwf_medium_context.config import BenchmarkConfig

FAMILIES = {
    "qwen3.8-27b": {"Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B-FP8", "qwen3.8-27b-nvfp4-unsloth"},
    "qwen3.6-27b": {"Qwen/Qwen3.6-27B"},
    "qwen3.6-35b-a3b": {"Qwen/Qwen3.6-35B-A3B-FP8"},
    "kimi-k2.6": {"moonshotai/Kimi-K2.6", "moonshotai-kimi-k2.6", "nvidia/Kimi-K2.6-NVFP4"},
    "gemma-4-31b-it": {"google/gemma-4-31B-it", "google/gemma-4-31b-it", "gemma-4-31b", "gemma-4-31b-it"},
    "gemma-4-26b-a4b-it": {"google/gemma-4-26B-A4B-it"},
    "glm-5.2": {"zai-org/GLM-5.2"},
    "glm-5.3": {"glm-5.3"},
    "inkling": {"thinkingmachines/inkling"},
    "deepseek-v4-flash-0731": {"deepseek-ai/DeepSeek-V4-Flash-0731"},
    "deepseek-v4-pro-0813": {"deepseek-ai/DeepSeek-V4-Pro-0813"},
    "nemotron-3-ultra": {
        "nemotron-3-ultra", "nvidia/nemotron-3-ultra",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    },
    "nemotron-3-super-120b": {
        "nemotron-3-super-120b", "nvidia/nemotron-3-super",
        "nvidia/Nemotron-120B-A12B",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    },
    "nemotron-3-nano-30b": {
        "nemotron-3-nano-30b", "nemotron-3-nano", "nvidia/nemotron-3-nano",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    },
    "gpt-5.4": {"gpt-5.4", "gpt-5.4-2026-03-05"},
}
for name in (
    "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "gpt-5-mini",
    "gpt-5.1", "gpt-5.2", "gpt-5.6-luna", "gpt-5.6-terra",
    "gpt-5.6-sol", "gpt-5.5", "gemini-2.5-flash", "gemini-3.5-flash",
    "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash",
    "gemini-3.8-flash", "nemotron-3.5-lightning", "muse-glimmer-30b",
    "poolside/laguna-s-2.1",
):
    FAMILIES[name] = {name}
ALIAS_TO_FAMILY = {alias: name for name, aliases in FAMILIES.items() for alias in aliases}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def normalize(value):
    return " ".join(value.split())


EXPECTED_USERS = [normalize(t["input"]) for t in BenchmarkConfig.turns]
EXPECTED_PROMPT = sha(BenchmarkConfig.system_instruction.encode())


def configuration(raw):
    text = raw.decode(errors="replace")
    startup = text[:16000]
    settings = {}
    for key in ("thinking_budget", "T", "temperature", "top_p", "top_k", "max_tokens", "max_output_tokens", "thinking", "thinking_off", "reasoning_off", "enable_thinking", "include_thoughts"):
        settings[key] = sorted(set(re.findall(
            r"\b" + key + r"=(None|True|False|true|false|[0-9]+(?:\.[0-9]+)?)", startup
        )))
    settings["reasoning_effort"] = sorted(set(re.findall(
        r"reasoning(?:\.effort|_effort)=(none|minimal|low|medium|high|xhigh)", startup
    )))
    settings["thinking_level"] = sorted(set(re.findall(r"thinking_level=(minimal|low|medium|high)", startup)))
    settings["reasoning_strength"] = sorted(set(re.findall(r"reasoning_strength[\s\"':=]+(low|medium|high|xhigh)", startup)))
    settings["service_tier"] = sorted(set(re.findall(r"service_tier=([a-z]+)", startup)))
    settings["hosts"] = sorted(set(
        urlsplit(url).hostname for url in re.findall(r"base_url=(https?://[^,\s]+)", startup)
    ))
    settings["origins"] = sorted(set(
        f"{parsed.scheme}://{parsed.hostname}" + (f":{parsed.port}" if parsed.port else "")
        for url in re.findall(r"base_url=(https?://[^,\s]+)", startup)
        if (parsed := urlsplit(url)).hostname
    ))
    settings["budget_parameter"] = (
        "native_thinking_token_budget" if "NATIVE thinking_token_budget" in startup
        else "legacy_or_unlogged"
    )
    settings["filler"] = [list(x) for x in sorted(set(re.findall(
        r"MTE_FILLER_DOTS active: ([0-9]+) x (\S+) filler tokens, position=([a-z]+)", text
    )))]
    settings["filler"] += [[n, "'.'", "suffix"] for n in sorted(set(re.findall(
        r"MTE_FILLER_DOTS active: appending ([0-9]+) dots", text
    )))]
    settings["unparsed_filler_marker"] = (
        "MTE_FILLER_DOTS active:" in text and not settings["filler"]
    )
    settings["system_prompt_sha256"] = None
    for line in text.splitlines():
        if "'role': 'system'" not in line:
            continue
        start = line.find("[{")
        if start >= 0:
            try:
                messages = ast.literal_eval(line[start:])
                system = next(m["content"] for m in messages if m.get("role") == "system")
                if isinstance(system, str):
                    settings["system_prompt_sha256"] = sha(system.encode())
            except (ValueError, SyntaxError, StopIteration, KeyError, TypeError):
                pass
        break
    settings["log_prefix_sha256"] = sha(raw)
    return settings


def capture(raw, source):
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not records or not isinstance(records[0], dict):
        return None
    model = records[0].get("model_name")
    if model not in ALIAS_TO_FAMILY:
        return None
    scripted = [r for r in records if not r.get("recovery_turn")]
    turns = [r.get("turn") for r in scripted]
    exact_turns = all(type(t) is int for t in turns) and sorted(turns) == list(range(30))
    complete = exact_turns and all(
        str(r.get("assistant_text") or "").strip() or r.get("tool_calls") for r in scripted
    )
    users_match = exact_turns and [
        normalize(str(r.get("user_text") or ""))
        for r in sorted(scripted, key=lambda r: r["turn"])
    ] == EXPECTED_USERS
    return {
        "model": model, "family": ALIAS_TO_FAMILY[model],
        "capture_id": Path(source.split("::")[-1]).parent.name,
        "transcript_sha256": sha(raw), "complete": bool(complete),
        "scripted_rows": len(scripted), "scripted_users_match": bool(users_match),
        "first_record_timestamp": records[0].get("ts"), "sources": [source],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--archive", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    found = []
    errors = []
    for name in sorted(set(args.paths.read_text().splitlines())):
        path = Path(name)
        if path.name != "transcript.jsonl" or "aiwf_medium_context" not in path.parts:
            continue
        try:
            with path.open() as handle:
                first = next((line for line in handle if line.strip()), "{}")
            if json.loads(first).get("model_name") not in ALIAS_TO_FAMILY:
                continue
            item = capture(path.read_bytes(), str(path))
            if item:
                log = path.parent / "run.log"
                with log.open("rb") if log.exists() else nullcontext(None) as stream:
                    item["configuration"] = configuration(stream.read(524288) if stream else b"")
                found.append(item)
        except (OSError, ValueError, AttributeError) as exc:
            errors.append({"source": name, "error_type": type(exc).__name__})
    for archive in args.archive:
        logs = {}
        archived = []
        with tarfile.open(archive, "r|gz") as stream:
            for member in stream:
                if not member.isfile():
                    continue
                if member.name.endswith("/transcript.jsonl"):
                    try:
                        item = capture(stream.extractfile(member).read(), f"{archive}::{member.name}")
                        if item:
                            archived.append((str(Path(member.name).parent), item))
                    except (ValueError, AttributeError) as exc:
                        errors.append({"source": f"{archive}::{member.name}", "error_type": type(exc).__name__})
                elif member.name.endswith("/run.log"):
                    logs[str(Path(member.name).parent)] = configuration(stream.extractfile(member).read(524288))
        for parent, item in archived:
            item["configuration"] = logs.get(parent, configuration(b""))
            found.append(item)

    unique = {}
    for item in found:
        key = item["transcript_sha256"]
        if key in unique:
            unique[key]["sources"].extend(item["sources"])
        else:
            unique[key] = item
    by_id = defaultdict(set)
    for item in unique.values():
        by_id[(item["model"], item["capture_id"])].add(item["transcript_sha256"])
    conflicts = [list(key) for key, values in by_id.items() if len(values) > 1]
    records = sorted(unique.values(), key=lambda r: (r["family"], r["capture_id"], r["transcript_sha256"]))
    for item in records:
        item["conflicting_capture_id"] = (
            bool(re.match(r"^\d{8}T", item["capture_id"]))
            and len(by_id[(item["model"], item["capture_id"])]) > 1
        )
    conflicts = [key for key in conflicts if re.match(r"^\d{8}T", key[1])]
    summary = {}
    for family in FAMILIES:
        group = [r for r in records if r["family"] == family]
        complete = [r for r in group if r["complete"] and not r["conflicting_capture_id"]]
        summary[family] = {
            "captures": len(group), "complete": len(complete),
            "scripted_users_match": sum(r["scripted_users_match"] for r in complete),
            "complete_by_budget": dict(Counter(
                ",".join(r["configuration"]["thinking_budget"]) or "unlogged" for r in complete
            )),
        }
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_system_prompt_sha256": EXPECTED_PROMPT,
        "discovered_capture_files": len(found), "unique_transcripts": len(records),
        "duplicates": len(found) - len(records), "capture_id_conflicts": conflicts,
        "errors": errors, "summary": summary, "captures": records,
        "scope": "Candidate source audit; configurations and serving stacks are not pooled into leaderboard rows.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"summary": summary, "errors": len(errors), "conflicts": len(conflicts)}, indent=2))


if __name__ == "__main__":
    main()
