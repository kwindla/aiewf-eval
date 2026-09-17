"""Tests for turn-tag emission, detection, and analyzer anchoring.

These cover the failure that made V2V latency unmeasurable: the transport logged a
tag position that the tagged audio never actually occupied in the recorded WAV, and
the analyzer anchored per-turn timing on that log position anyway — producing
negative pipeline TTFB and 100s+ V2V values while still printing summary medians.
"""

import wave

import numpy as np
import pytest

from multi_turn_eval.transports.null_audio_output import (
    AUDIO_TAG_DURATION_MS,
    AUDIO_TAG_FREQUENCY_HZ,
    NullAudioOutputTransport,
)
from scripts.analyze_turn_metrics import (
    MAX_PLAUSIBLE_V2V_MS,
    detect_tags_in_wav,
    is_plausible_ttfb,
    is_plausible_v2v,
    suppress_summary_statistics,
    tag_coverage,
)

SAMPLE_RATE = 24000


def _tag_samples(rate: int = SAMPLE_RATE) -> int:
    return int(rate * AUDIO_TAG_DURATION_MS / 1000)


# --------------------------------------------------------------------------------------
# Fix 2: the tag must never be silently truncated below detectability
# --------------------------------------------------------------------------------------


def test_tag_is_long_enough_for_detector_window():
    """Tag must exceed the detector's 20ms STFT window, or it can never fill one."""
    assert AUDIO_TAG_DURATION_MS > 20


def test_short_frame_carries_tag_remainder():
    """A frame shorter than the tag must carry the rest into following frames."""
    transport = NullAudioOutputTransport.__new__(NullAudioOutputTransport)
    transport._tag_remainder = None

    tag = transport._generate_audio_tag(SAMPLE_RATE)
    short_frame = np.zeros(len(tag) // 4, dtype=np.int16).tobytes()

    first = transport._mix_tag_into_frame(short_frame, tag)

    # The frame keeps its own length, and the unmixed remainder is retained.
    assert len(first) == len(short_frame)
    assert transport._tag_remainder is not None
    assert len(transport._tag_remainder) == len(tag) - len(tag) // 4


def test_tag_remainder_drains_across_frames():
    """Feeding successive frames drains the remainder and preserves total tag energy."""
    transport = NullAudioOutputTransport.__new__(NullAudioOutputTransport)
    transport._tag_remainder = None

    tag = transport._generate_audio_tag(SAMPLE_RATE)
    chunk = len(tag) // 4
    frame_bytes = np.zeros(chunk, dtype=np.int16).tobytes()

    emitted = []
    emitted.append(transport._mix_tag_into_frame(frame_bytes, tag))
    for _ in range(5):
        emitted.append(transport._continue_tag_if_pending(frame_bytes))

    assert transport._tag_remainder is None, "remainder should fully drain"

    # Frames are silent, so the mixed stream must reproduce the tag sample-for-sample.
    # (Counting non-zero samples would be wrong: a 2kHz sine at 24kHz crosses zero
    # twice every 12 samples.)
    stitched = np.concatenate([np.frombuffer(b, dtype=np.int16) for b in emitted])
    np.testing.assert_array_equal(stitched[: len(tag)], tag)
    assert not stitched[len(tag) :].any(), "no tag energy should leak past the tag length"


# --------------------------------------------------------------------------------------
# Fix 2/3: detection round-trip against a synthetic WAV with known tag positions
# --------------------------------------------------------------------------------------


def _write_stereo_wav(path, bot_track: np.ndarray, rate: int = SAMPLE_RATE):
    user = np.zeros_like(bot_track)
    interleaved = np.empty(bot_track.size * 2, dtype=np.int16)
    interleaved[0::2] = user
    interleaved[1::2] = bot_track
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(interleaved.tobytes())


def _make_tag(rate: int = SAMPLE_RATE) -> np.ndarray:
    n = _tag_samples(rate)
    t = np.arange(n) / rate
    amp = 32768 * (10 ** (-12 / 20))
    return (amp * np.sin(2 * np.pi * AUDIO_TAG_FREQUENCY_HZ * t)).astype(np.int16)


def test_detect_tags_roundtrip_in_silence(tmp_path):
    """Tags mixed into silence are recovered at their known positions."""
    track = np.zeros(SAMPLE_RATE * 20, dtype=np.int16)
    positions_ms = [1000, 5000, 12000, 18000]
    tag = _make_tag()
    for pos in positions_ms:
        start = int(pos / 1000 * SAMPLE_RATE)
        track[start : start + len(tag)] = tag

    wav = tmp_path / "conversation.wav"
    _write_stereo_wav(wav, track)

    found = detect_tags_in_wav(wav, channel=1)

    assert len(found) == len(positions_ms)
    for expected, actual in zip(positions_ms, sorted(found)):
        assert abs(actual - expected) <= 25


def test_detect_tags_roundtrip_over_speech_like_audio(tmp_path):
    """Tags remain detectable when mixed into loud broadband audio, not just silence."""
    rng = np.random.default_rng(1234)
    speech = (rng.normal(0, 0.05, SAMPLE_RATE * 12) * 32767).astype(np.int16)
    tag = _make_tag()
    positions_ms = [2000, 7000]
    track = speech.copy()
    for pos in positions_ms:
        start = int(pos / 1000 * SAMPLE_RATE)
        mixed = track[start : start + len(tag)].astype(np.int32) + tag.astype(np.int32)
        track[start : start + len(tag)] = np.clip(mixed, -32768, 32767)

    wav = tmp_path / "conversation.wav"
    _write_stereo_wav(wav, track)

    found = sorted(detect_tags_in_wav(wav, channel=1))

    for expected in positions_ms:
        assert any(abs(f - expected) <= 25 for f in found), f"tag at {expected}ms not detected"


# --------------------------------------------------------------------------------------
# Fix 3: impossible values must never be reported as data
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1.0, -3549.0, -0.5])
def test_negative_pipeline_ttfb_is_rejected(bad):
    """A bot tag cannot precede the user's speech ending."""
    assert not is_plausible_ttfb(bad)


@pytest.mark.parametrize("good", [0.0, 131.0, 1085.0, 8568.0])
def test_plausible_pipeline_ttfb_is_accepted(good):
    assert is_plausible_ttfb(good)


@pytest.mark.parametrize("bad", [-100.0, 35571.0, 132024.0, 181327.0])
def test_absurd_v2v_is_rejected(bad):
    """Values seen in the corrupt runs must not survive the sanity gate."""
    assert not is_plausible_v2v(bad)


@pytest.mark.parametrize("good", [192.0, 832.0, 1088.0, 1600.0])
def test_plausible_v2v_is_accepted(good):
    """The anchored measurements from real runs must still pass."""
    assert is_plausible_v2v(good)


def test_v2v_ceiling_is_the_documented_bound():
    assert MAX_PLAUSIBLE_V2V_MS == 10000


# --------------------------------------------------------------------------------------
# Fix 3: coverage accounting drives the hard-fail gate
# --------------------------------------------------------------------------------------


class _FakeTurn:
    def __init__(self, anchored: bool):
        self.bot_tag_wav_ms = 1234.0 if anchored else None


def test_tag_coverage_counts_only_wav_anchored_turns():
    turns = [_FakeTurn(True), _FakeTurn(False), _FakeTurn(True), _FakeTurn(False)]
    anchored, total, pct = tag_coverage(turns)
    assert (anchored, total) == (2, 4)
    assert pct == pytest.approx(50.0)


def test_tag_coverage_handles_empty_run():
    anchored, total, pct = tag_coverage([])
    assert (anchored, total, pct) == (0, 0, 0.0)


def test_low_coverage_control_run_is_below_gate():
    """A control run with 11 of 24 turns anchored must not clear the 80% gate."""
    turns = [_FakeTurn(i < 11) for i in range(24)]
    _, _, pct = tag_coverage(turns)
    assert pct < 80.0


def test_json_output_strips_stats_when_coverage_fails():
    """--json must withhold aggregates too, not just the printed table.

    A campaign script reading --json would otherwise consume a median computed over
    whichever turns happened to anchor and treat it as the run's latency.
    """
    summary = {
        "tag_coverage_pct": 62.0,
        "tag_coverage_ok": False,
        "anchored_turns": 13,
        "coverage_denominator": 21,
        "wav_v2v_ms_median": 800.0,
        "wav_v2v_ms_mean": 2149.0,
        "wav_v2v_ms_min": 608.0,
        "wav_v2v_ms_max": 9664.0,
        "server_ttfb_ms_median": 1028.0,
        "num_turns": 13,
    }

    cleaned = suppress_summary_statistics(summary)

    assert not [k for k in cleaned if k.endswith(("_median", "_mean", "_min", "_max", "_p95"))]
    assert cleaned["statistics_suppressed"] is True
    assert "62%" in cleaned["statistics_suppressed_reason"]
    # Coverage accounting itself must survive — it is the diagnosis.
    assert cleaned["anchored_turns"] == 13
    assert cleaned["coverage_denominator"] == 21


def test_coverage_denominator_is_run_turns_not_tagged_turns():
    """Untagged turns must count against coverage, not be excluded from it.

    A stalled turn typically loses its tag entirely. Scoring anchored turns against
    only the *tagged* ones reported 13/13 = 100% for a run that actually produced 21
    turns — hiding the 8 slowest and passing the gate on a biased sample.
    """
    # 13 tagged turns, all anchored, in a run whose transcript holds 21 turns.
    turns = [_FakeTurn(True) for _ in range(13)]
    anchored, tagged_total, tagged_pct = tag_coverage(turns)
    assert (anchored, tagged_total) == (13, 13)
    assert tagged_pct == pytest.approx(100.0)

    run_turns = 21
    denominator = max(run_turns, tagged_total)
    true_pct = anchored / denominator * 100
    assert true_pct < 80.0, "a run missing 8 of 21 turns must not pass the coverage gate"
