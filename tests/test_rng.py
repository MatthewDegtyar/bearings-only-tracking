"""Tests for the reproducibility contract."""

from __future__ import annotations

import numpy as np
import pytest

from kf2.rng import Stream, stream_rng


def _draw(seed, run, stream, n=64):
    return stream_rng(seed, run, stream).standard_normal(n)


def test_same_triple_gives_the_same_stream():
    assert np.array_equal(_draw(7, 3, Stream.PROCESS), _draw(7, 3, Stream.PROCESS))


def test_streams_differ_across_seed_run_and_source():
    base = _draw(7, 3, Stream.PROCESS)
    assert not np.array_equal(base, _draw(8, 3, Stream.PROCESS))
    assert not np.array_equal(base, _draw(7, 4, Stream.PROCESS))
    assert not np.array_equal(base, _draw(7, 3, Stream.MEASUREMENT))


def test_streams_are_uncorrelated():
    """Named streams must behave as independent sources, not offsets of one."""
    a = _draw(20260730, 0, Stream.PROCESS, 20000)
    b = _draw(20260730, 0, Stream.MEASUREMENT, 20000)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.03


def test_run_order_does_not_matter():
    """Runs must be executable in any order, in parallel, or individually.

    A generator that consumed one global stream sequentially would fail this,
    and would make a parallel sweep produce different data from a serial one.
    """
    forward = [_draw(11, r, Stream.PROCESS, 8) for r in range(6)]
    backward = [_draw(11, r, Stream.PROCESS, 8) for r in reversed(range(6))]
    assert all(np.array_equal(a, b) for a, b in zip(forward, reversed(backward)))


def test_adding_a_stream_does_not_disturb_existing_ones():
    """Phase 3 will add clutter and detection draws. Results recorded now must
    still reproduce afterwards, so streams are keyed, never sequential."""
    before = _draw(5, 1, Stream.PROCESS)
    _ = _draw(5, 1, Stream.CLUTTER, 10_000)
    assert np.array_equal(before, _draw(5, 1, Stream.PROCESS))


def test_stream_ids_are_stable():
    """Renumbering breaks every recorded (seed, scenario) pair."""
    assert (Stream.PROCESS, Stream.MEASUREMENT, Stream.INIT) == (1, 2, 3)
    assert (Stream.CLUTTER, Stream.DETECTION) == (4, 5)


def test_negative_run_index_is_rejected():
    with pytest.raises(ValueError):
        stream_rng(1, -1, Stream.PROCESS)


def test_draws_have_the_right_distribution():
    x = _draw(99, 0, Stream.PROCESS, 200000)
    assert x.mean() == pytest.approx(0.0, abs=0.01)
    assert x.std() == pytest.approx(1.0, abs=0.01)
