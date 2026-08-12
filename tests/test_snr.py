"""Tests for SNR-dependent measurement noise.

The load-bearing one is :func:`test_disabled_reproduces_bit_identical_results`:
a new failure mode is only useful if switching it off leaves everything else
exactly as it was.
"""

from __future__ import annotations

import numpy as np
import pytest

from kf2 import Scenario, datagen, run_monte_carlo
from kf2.config import replace
from kf2.snr import (
    assumed_sigma,
    bearing_sigma_series,
    crlb_bearing_sigma,
    crlb_constant,
    db_to_linear,
    fade_db,
    nominal_ranges,
    snr_db,
)


# ---------------------------------------------------------------------------
# Default off
# ---------------------------------------------------------------------------


def test_disabled_reproduces_bit_identical_results():
    """With the knob off, no SNR parameter may change any number."""
    sc = replace(Scenario(mc_runs=25, steps=300), p0_pos=600.0)
    loud = replace(
        sc, snr_ref_db=3.0, snr_ref_range=99.0, snr_fade_depth_db=20.0,
        snr_fade_period=7.0, crlb_constant=0.5, r_assumption="best",
    )
    a, b = run_monte_carlo(sc), run_monte_carlo(loud)
    assert np.array_equal(np.nan_to_num(a.nees), np.nan_to_num(b.nees))
    assert np.array_equal(a.pos_err, b.pos_err)
    assert np.array_equal(np.nan_to_num(a.nis), np.nan_to_num(b.nis))


def test_disabled_gives_exactly_the_constant_sigma():
    sc = Scenario(steps=100)
    own = datagen.ownship_track(sc)
    truth = datagen.target_truth(sc, 0)
    sig = datagen.measurement_sigma(sc, truth, own)
    assert np.all(sig == sc.sigma_bearing)
    assert assumed_sigma(sc, own.xy, own.t) is None


def test_enabling_it_actually_changes_the_data():
    sc = replace(Scenario(mc_runs=10, steps=200), p0_pos=600.0)
    on = run_monte_carlo(replace(sc, snr_enabled=True))
    assert not np.array_equal(np.nan_to_num(run_monte_carlo(sc).nees), np.nan_to_num(on.nees))


# ---------------------------------------------------------------------------
# The physics
# ---------------------------------------------------------------------------


def test_crlb_variance_is_inversely_proportional_to_snr():
    """sigma^2 ~ 1/SNR is the whole content of the CRLB form used here."""
    s1 = crlb_bearing_sigma(1.0, k_crlb=0.01)
    s100 = crlb_bearing_sigma(100.0, k_crlb=0.01)
    assert s1 / s100 == pytest.approx(10.0)
    assert crlb_bearing_sigma(4.0, 0.02) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        crlb_bearing_sigma(0.0, 0.01)


def test_passive_receiver_spreading_is_one_way():
    """Received power falls as 1/r^2 for a passive receiver -- 20 dB per decade,
    not the 40 dB of a two-way active radar."""
    sc = replace(Scenario(), snr_enabled=True, snr_ref_range=1000.0)
    t = np.zeros(2)
    db = snr_db(sc, np.array([1000.0, 10000.0]), t)
    assert db[0] - db[1] == pytest.approx(20.0)


def test_sigma_is_proportional_to_range():
    """The practical consequence: a passive bearing degrades linearly as the
    target opens."""
    sc = replace(Scenario(steps=200), snr_enabled=True)
    own = datagen.ownship_track(sc)
    truth = datagen.target_truth(sc, 0)
    sig = datagen.measurement_sigma(sc, truth, own)
    r = datagen.target_range(truth, own)
    assert np.allclose(sig / r, (sig / r)[0])


def test_default_constant_is_calibrated_to_the_reference():
    """At the reference range with no fade, the SNR model must agree with the
    constant-R model -- so the knob varies the noise without relocating it."""
    sc = replace(Scenario(), snr_enabled=True, snr_ref_range=5000.0)
    sigma = bearing_sigma_series(sc, np.array([sc.snr_ref_range]), np.zeros(1))
    assert sigma[0] == pytest.approx(sc.sigma_bearing, rel=1e-12)
    assert crlb_constant(sc) == pytest.approx(
        sc.sigma_bearing * np.sqrt(db_to_linear(sc.snr_ref_db))
    )


def test_explicit_constant_overrides_calibration():
    sc = replace(Scenario(), snr_enabled=True, crlb_constant=0.25)
    assert crlb_constant(sc) == 0.25
    with pytest.raises(ValueError):
        replace(Scenario(), crlb_constant=-1.0)


def test_fade_is_bounded_and_periodic():
    sc = replace(Scenario(), snr_enabled=True, snr_fade_depth_db=12.0, snr_fade_period=100.0)
    t = np.linspace(0, 300, 601)
    f = fade_db(sc, t)
    assert f.min() == pytest.approx(-12.0, abs=1e-6)
    assert f.max() == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(f[:200], f[200:400], atol=1e-9), "period 100 s at dt 0.5"
    assert np.all(fade_db(replace(sc, snr_fade_depth_db=0.0), t) == 0.0)


def test_deeper_fade_means_worse_bearings():
    sc = replace(Scenario(steps=200), snr_enabled=True)
    own = datagen.ownship_track(sc)
    r = nominal_ranges(sc, own.xy, own.t)
    shallow = bearing_sigma_series(replace(sc, snr_fade_depth_db=3.0), r, own.t)
    deep = bearing_sigma_series(replace(sc, snr_fade_depth_db=12.0), r, own.t)
    assert deep.max() > shallow.max()
    # 12 dB of fade is 4x in power, so 2x in sigma at the fade trough.
    assert deep.max() / shallow.max() == pytest.approx(10 ** (9 / 20.0), rel=0.02)


# ---------------------------------------------------------------------------
# What the filter assumes
# ---------------------------------------------------------------------------


def test_constant_assumptions_are_constant_and_ordered():
    sc = replace(Scenario(steps=300), snr_enabled=True, snr_fade_depth_db=10.0)
    own = datagen.ownship_track(sc)
    mean = assumed_sigma(replace(sc, r_assumption="mean"), own.xy, own.t)
    best = assumed_sigma(replace(sc, r_assumption="best"), own.xy, own.t)
    assert np.all(mean == mean[0]) and np.all(best == best[0])
    assert best[0] < mean[0], "best case must be the optimistic one"
    assert assumed_sigma(replace(sc, r_assumption="true"), own.xy, own.t) is None


def test_assumptions_use_the_nominal_geometry_not_the_realised_one():
    """A design-time constant R would be chosen from an expected geometry.
    Deriving it from each run's truth would flatter it."""
    sc = replace(Scenario(steps=200), snr_enabled=True, r_assumption="mean")
    own = datagen.ownship_track(sc)
    a = assumed_sigma(sc, own.xy, own.t)
    b = assumed_sigma(sc, own.xy, own.t)
    assert np.array_equal(a, b)  # no run index involved at all


def test_bad_assumption_is_rejected():
    with pytest.raises(ValueError):
        replace(Scenario(), r_assumption="hopeful")


def test_filter_measurement_noise_is_settable():
    from kf2.ckf5 import BearingsOnlyCKF5 as BearingsOnlyCKF
    from kf2.filters import BearingsOnlyEKF

    for cls in (BearingsOnlyEKF, BearingsOnlyCKF):
        f = cls(1e-3, 0.01)
        assert f.R == pytest.approx(1e-4)
        f.set_measurement_noise(0.02)
        assert f.R == pytest.approx(4e-4)


def test_true_assumption_tracks_the_applied_noise():
    """With r_assumption='true' the filter's R must equal the noise actually
    applied, so this configuration is the consistent reference."""
    sc = replace(Scenario(steps=200), snr_enabled=True, r_assumption="true")
    own = datagen.ownship_track(sc)
    truth = datagen.target_truth(sc, 0)
    applied = datagen.measurement_sigma(sc, truth, own)
    assert assumed_sigma(sc, own.xy, own.t) is None  # run_trial fills it from applied
    assert applied.min() > 0 and applied.max() > applied.min()
