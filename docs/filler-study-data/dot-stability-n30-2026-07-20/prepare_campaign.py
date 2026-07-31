#!/usr/bin/env python3
"""Build the frozen historical inclusion ledger and randomized deficit schedules."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "runs/aiwf_medium_context"
TARGET = 30
SEED = 20260720

REGISTRY = {
    "gpt54": ("openai-a", "gpt-5.4", "openai"),
    "gpt55": ("openai-a", "gpt-5.5", "openai"),
    "terra": ("openai-b", "gpt-5.6-terra", "openai"),
    "sol": ("openai-b", "gpt-5.6-sol", "openai"),
    "gemma431": ("lilac", "lilac/gemma-4-31b-it", "lilac"),
    "inkling": ("baseten", "thinkingmachines/inkling", "baseten"),
    "qwen3_8b": ("openrouter", "qwen/qwen3-8b", "openrouter"),
    "glm52": ("baseten", "zai-org/GLM-5.2", "baseten"),
}


def manifest(path: str, configs: set[str]) -> list[Path]:
    rows: list[Path] = []
    with (ROOT / path).open(newline="") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            if fields and fields[0] in configs:
                rows.append(ROOT / fields[-1])
    return rows


def allowlist(path: str) -> list[Path]:
    return [ROOT / line.strip() for line in (ROOT / path).read_text().splitlines() if line.strip()]


def terra_dots() -> list[Path]:
    rows: list[Path] = []
    for run_dir in sorted(RUNS.glob("20260719T*_gpt-5.6-terra_*")):
        log_path = run_dir / "run.log"
        if not log_path.is_file():
            continue
        log = log_path.read_text(errors="replace")
        is_dot = "MTE_FILLER_DOTS active" in log and (
            "appending 96 dots" in log or "96 x '.' filler" in log
        )
        if is_dot and "reasoning.effort=none" in log:
            rows.append(run_dir)
    if len(rows) < TARGET:
        raise ValueError(f"expected at least {TARGET} Terra dot attempts, found {len(rows)}")
    return rows[:TARGET]


def end_session_turn(transcript: Path) -> int:
    best = -1
    if not transcript.is_file():
        return best
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        for call in row.get("tool_calls") or []:
            if call.get("name") == "end_session":
                best = max(best, int(row.get("turn", -1)))
    return best


def historical() -> dict[tuple[str, str], list[tuple[Path, str, str]]]:
    cells: dict[tuple[str, str], list[tuple[Path, str, str]]] = defaultdict(list)

    def add(model: str, arm: str, paths: list[Path], source: str, grade: str) -> None:
        cells[(model, arm)].extend((path, source, grade) for path in paths)

    gpt54 = "docs/filler-study-data/gpt54_filler_manifest.tsv"
    add("gpt54", "nofiller", manifest(gpt54, {"nofiller"}), gpt54, "runtime-signature")
    add("gpt54", "dots96", manifest(gpt54, {"filler96"}), gpt54, "runtime-signature")

    terra_early = "docs/filler-study-data/gpt56_terra_manifest.tsv"
    terra_screen = "docs/filler-study-data/expand_oai_ant_manifest.tsv"
    add("terra", "nofiller", manifest(terra_early, {"none"}), terra_early, "runtime-signature")
    add("terra", "nofiller", manifest(terra_screen, {"terra_nofiller"}), terra_screen, "runtime-signature")
    add("terra", "dots96", terra_dots(), "chronological first 30 Jul-19 dot attempts", "attempt-reconstruction")

    gpt55 = "docs/filler-study-data/gpt55_manifest.tsv"
    add("gpt55", "nofiller", manifest(gpt55, {"gpt55_nofiller"}), gpt55, "runtime-signature")
    add("gpt55", "dots96", manifest(gpt55, {"gpt55_dots96"}), gpt55, "runtime-signature")

    oai = "docs/filler-study-data/expand_oai_ant_manifest.tsv"
    add("sol", "nofiller", manifest(oai, {"sol_nofiller"}), oai, "runtime-signature")
    add("sol", "dots96", manifest(oai, {"sol_filler96"}), oai, "runtime-signature")

    gemma = "docs/filler-study-data/broaden_oai_lilac_manifest.tsv"
    june = "docs/ten-run-allowlists/lilac-gemma-4-31b-it-off-2026-06-15.txt"
    add("gemma431", "nofiller", allowlist(june), june, "runtime-signature")
    add("gemma431", "nofiller", manifest(gemma, {"gemma431_nofiller"}), gemma, "runtime-signature")
    add("gemma431", "dots96", manifest(gemma, {"gemma431_filler96"}), gemma, "runtime-signature")

    inkling = "docs/filler-study-data/inkling_filler_manifest.tsv"
    add("inkling", "nofiller", manifest(inkling, {"nofiller"}), inkling, "runtime-signature")
    ink_dots = manifest(inkling, {"dots96"}) + [
        RUNS / "20260718T221939_thinkingmachines_inkling_f46cecd8",
        RUNS / "20260718T225034_thinkingmachines_inkling_dcd5c571",
    ]
    add("inkling", "dots96", ink_dots, inkling + "+attempt audit", "attempt-reconstruction")

    openrouter = "docs/filler-study-data/expand_openrouter_manifest.tsv"
    add("qwen3_8b", "nofiller", manifest(openrouter, {"qwen3_8b_nofiller"}), openrouter, "runtime-signature")
    add("qwen3_8b", "dots96", manifest(openrouter, {"qwen3_8b_filler96"}), openrouter, "runtime-signature")

    baseten = "docs/filler-study-data/broaden_baseten_manifest.tsv"
    add("glm52", "nofiller", manifest(baseten, {"glm52_nofiller"}), baseten, "runtime-signature")
    add("glm52", "dots96", manifest(baseten, {"glm52_filler96"}), baseten, "runtime-signature")
    return cells


def constrained_arms(nofiller: int, dots: int, seed: int) -> list[str]:
    arms = ["nofiller"] * nofiller + ["dots96"] * dots
    rng = random.Random(seed)
    for _ in range(100_000):
        rng.shuffle(arms)
        if all(not (arms[i] == arms[i + 1] == arms[i + 2] == arms[i + 3]) for i in range(max(0, len(arms) - 3))):
            return list(arms)
    # A one-arm schedule (Terra) cannot satisfy the run-length constraint.
    if nofiller == 0 or dots == 0:
        return list(arms)
    raise RuntimeError("unable to construct constrained schedule")


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    cells = historical()
    seen: set[Path] = set()
    ledger_rows: list[list[object]] = []
    deficits: dict[tuple[str, str], int] = {}

    for model in REGISTRY:
        for arm in ("nofiller", "dots96"):
            entries = cells[(model, arm)]
            if len(entries) > TARGET:
                entries = entries[:TARGET]
            for run_dir, source, grade in entries:
                run_dir = run_dir.resolve()
                if run_dir in seen:
                    raise ValueError(f"duplicate historical run directory: {run_dir}")
                seen.add(run_dir)
                transcript = run_dir / "transcript.jsonl"
                if not transcript.is_file() or transcript.stat().st_size == 0:
                    raise ValueError(f"included run lacks nonempty transcript: {run_dir}")
                rows = len(transcript.read_text().splitlines())
                es_turn = end_session_turn(transcript)
                if es_turn == 29:
                    classification = "strict_complete"
                elif es_turn >= 0:
                    classification = "model_abort"
                else:
                    classification = "incomplete_no_end_session"
                ledger_rows.append([
                    model,
                    arm,
                    str(run_dir.relative_to(ROOT)),
                    rows,
                    es_turn,
                    classification,
                    int((run_dir / "claude_judged.jsonl").is_file()),
                    grade,
                    source,
                ])
            deficits[(model, arm)] = TARGET - len(entries)
            if deficits[(model, arm)] < 0:
                raise AssertionError("negative deficit")

    write_tsv(
        HERE / "existing-included.tsv",
        ["model", "arm", "run_dir", "transcript_rows", "end_session_turn", "classification", "judged", "provenance", "source"],
        ledger_rows,
    )
    write_tsv(
        HERE / "deficits.tsv",
        ["model", "arm", "included", "new_required", "target"],
        [
            [model, arm, TARGET - deficits[(model, arm)], deficits[(model, arm)], TARGET]
            for model in REGISTRY
            for arm in ("nofiller", "dots96")
        ],
    )

    lane_models: dict[str, list[str]] = defaultdict(list)
    for model, (lane, _requested, _service) in REGISTRY.items():
        lane_models[lane].append(model)
    schedule_hashes: list[list[object]] = []
    total_scheduled = 0
    for lane, models in lane_models.items():
        per_model: dict[str, list[str]] = {}
        for offset, model in enumerate(models):
            per_model[model] = constrained_arms(
                deficits[(model, "nofiller")],
                deficits[(model, "dots96")],
                SEED + 1009 * offset + int(hashlib.sha256(model.encode()).hexdigest()[:8], 16),
            )
        rows: list[list[object]] = []
        slot = 1
        while any(per_model.values()):
            for model in models:
                if not per_model[model]:
                    continue
                arm = per_model[model].pop(0)
                _lane, requested, service = REGISTRY[model]
                rows.append([f"{lane}-{slot:03d}", model, arm, requested, service])
                slot += 1
        schedule = HERE / f"schedule-{lane}.tsv"
        write_tsv(schedule, ["slot", "model", "arm", "requested_model", "service"], rows)
        digest = hashlib.sha256(schedule.read_bytes()).hexdigest()
        schedule_hashes.append([schedule.name, digest, len(rows)])
        total_scheduled += len(rows)
    write_tsv(HERE / "schedule-hashes.tsv", ["schedule", "sha256", "rows"], schedule_hashes)

    if total_scheduled != sum(deficits.values()):
        raise AssertionError("scheduled total does not match deficits")
    print(f"historical included={len(ledger_rows)} new scheduled={total_scheduled}")


if __name__ == "__main__":
    main()
