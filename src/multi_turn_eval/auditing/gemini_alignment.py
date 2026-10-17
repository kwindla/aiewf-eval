"""Validate Gemini input-window timestamps against three recorded speech landmarks."""

import json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import correlate, resample_poly
from benchmarks.aiwf_medium_context.config import BenchmarkConfig


def main():
    run = Path(sys.argv[1])
    audio, sr = sf.read(run / "conversation.wav", dtype="float64")
    assert sr == 24000
    audit = json.loads(Path(sys.argv[2]).read_text())
    rows = [
        dict(type="input_start", turn=int(i), t=w["start_ms"] / 1000)
        for i, w in audit["windows"].items()
    ]
    base = 0
    checks = []
    for r in rows:
        if r["type"] != "input_start":
            continue
        source, rate = sf.read(
            BenchmarkConfig.get_audio_path(r["turn"]), dtype="float64"
        )
        if rate != sr:
            source = resample_poly(source, sr, rate)
        expected = int((r["t"] - base) * sr)
        offsets = []
        scores = []
        for part in np.array_split(np.arange(max(0, len(source) - 2400)), 3):
            # Select a loud 100 ms patch inside each third of the utterance.
            choices = part[::240]
            if not len(choices):
                continue
            pos = max(choices, key=lambda p: float(np.sum(source[p : p + 2400] ** 2)))
            patch = source[pos : pos + 2400]
            energy = float(np.sum(patch**2))
            if energy < 1e-5:
                continue
            lo = max(0, expected + pos - 24000)
            hi = min(len(audio), expected + pos + 24000 + len(patch))
            region = audio[lo:hi, 0]
            corr = correlate(region, patch, mode="valid", method="fft")
            cumulative = np.concatenate(([0.0], np.cumsum(region**2)))
            energies = cumulative[len(patch) :] - cumulative[: -len(patch)]
            normalized = corr / np.sqrt(np.maximum(energy * energies, 1e-20))
            at = int(np.argmax(normalized))
            score = float(normalized[at])
            offsets.append((lo + at - pos - expected) / sr * 1000)
            scores.append(score)
        checks.append(
            dict(
                turn=r["turn"],
                offset_ms=offsets,
                min_correlation=min(scores, default=0),
                within_turn_drift_ms=max(offsets, default=0) - min(offsets, default=0),
            )
        )
    issues = [
        c
        for c in checks
        if c["min_correlation"] < 0.95
        or max(map(abs, c["offset_ms"]), default=0) > 250
        or c["within_turn_drift_ms"] > 100
    ]
    result = dict(checks=checks, issues=issues)
    Path(sys.argv[3]).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if issues:
        raise SystemExit("Recorded audio timing failed waveform alignment audit")
