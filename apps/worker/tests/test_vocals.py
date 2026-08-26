"""Vocal-activity spans — the 27 Aug lip-sync fix's arithmetic layer."""

from __future__ import annotations

from worker.media.vocals import spans_from_envelope, vocal_fraction


def test_spans_bridge_breaths_and_drop_leakage() -> None:
    hop = 0.05
    # 3s silence, 4s singing with a short breath in it, 3s silence,
    # one isolated 0.2s leaked hit.
    envelope = (
        [0.001] * 60
        + [0.2] * 40 + [0.001] * 20 + [0.2] * 40   # sung, 1s breath bridged
        + [0.001] * 60
        + [0.2] * 4                                  # leakage, dropped
        + [0.001] * 16
    )
    spans = spans_from_envelope(envelope, hop_seconds=hop)
    assert len(spans) == 1
    start, end = spans[0]
    assert 2.9 <= start <= 3.1
    assert 7.9 <= end <= 8.2  # breath bridged, leaked hit not a span


def test_vocal_fraction_windows() -> None:
    spans = [(3.0, 8.0)]
    assert vocal_fraction(spans, 0.0, 3.0) == 0.0
    assert vocal_fraction(spans, 4.0, 6.0) == 1.0
    assert 0.45 <= vocal_fraction(spans, 0.0, 10.0) <= 0.55
    assert vocal_fraction(spans, 10.0, 10.0) == 0.0


def test_empty_envelope_is_no_spans() -> None:
    assert spans_from_envelope([]) == []
