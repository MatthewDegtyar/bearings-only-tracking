"""Tests for data generation -- the 'no inverse crime' side of the boundary."""

from __future__ import annotations

import numpy as np
import pytest

from kf2.config import DEG, Scenario, replace
from kf2.datagen import (
    Detections,
    _integrate_em,
    _integrate_em_reference,
    _velocity_increments,
    bearing,
    em_step_covariance,
    generate_detections,
    initial_covariance,
    initial_estimate,
    ownship_heading,
    ownship_track,
    target_truth,
    wrap_pi,
)
from kf2.filters import cwna_process_noise
from kf2.rng import Stream, stream_rng


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------


def test_wrap_pi_basics():
    assert wrap_pi(0.1) == pytest.approx(0.1)
    assert wrap_pi(2 * np.pi + 0.1) == pytest.approx(0.1)
    assert wrap_pi(-2 * np.pi - 0.1) == pytest.approx(-0.1)
    assert np.all(np.abs(wrap_pi(np.linspace(-20, 20, 501))) <= np.pi + 1e-12)


def test_wrap_pi_across_the_branch_cut():
    """The failure this guards: a target at the cut must give a small innovation."""
    z, h = -np.pi + 0.01, np.pi - 0.01
    assert wrap_pi(z - h) == pytest.approx(0.02)


def test_bearing_matches_atan2_and_broadcasts():
    assert bearing(np.array([1.0, 1.0]), np.array([0.0, 0.0])) == pytest.approx(np.pi / 4)
    tgt = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    own = np.zeros((3, 2))
    assert np.allclose(bearing(tgt, own), [0.0, np.pi / 2, np.pi])


# ---------------------------------------------------------------------------
# Ownship
# ---------------------------------------------------------------------------


def test_ownship_speed_is_constant_and_heading_follows_the_law():
    sc = Scenario(steps=200)
    own = ownship_track(sc)
    assert np.allclose(np.hypot(own.vx, own.vy), sc.own_speed)
    assert np.allclose(own.psi, ownship_heading(sc, own.t))
    assert own.x[0] == pytest.approx(sc.own_x0)
    assert own.y[0] == pytest.approx(sc.own_y0)


def test_zero_manoeuvre_amplitude_is_a_straight_line():
    """amp = 0 is the unobservable end of the Phase 2 sweep."""
    sc = Scenario(steps=200, own_manoeuvre_amp_deg=0.0, own_psi0_deg=30.0)
    own = ownship_track(sc)
    assert np.allclose(own.psi, 30.0 * DEG)
    # Position must lie on a straight line through the start.
    d = np.column_stack((own.x - own.x[0], own.y - own.y[0]))
    cross = d[:, 0] * np.sin(30.0 * DEG) - d[:, 1] * np.cos(30.0 * DEG)
    assert np.allclose(cross, 0.0, atol=1e-9)
    # ...and travel at the right speed.
    assert np.hypot(*d[-1]) == pytest.approx(sc.own_speed * sc.duration, rel=1e-9)


def test_ownship_is_identical_across_runs():
    sc = Scenario(steps=50)
    a, b = ownship_track(sc), ownship_track(sc)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)


# ---------------------------------------------------------------------------
# Target truth: the Euler-Maruyama integrator
# ---------------------------------------------------------------------------


def test_vectorised_integrator_equals_the_explicit_substep_loop():
    """The vectorised form collapses the substep loop algebraically.

    This is the check that the collapse is right; everything downstream assumes
    it. Both consume the identical noise array, so equality is exact up to
    floating-point summation order.
    """
    sc = Scenario(steps=40, truth_substeps=17, q=2e-3)
    rng = stream_rng(sc.seed, 0, Stream.PROCESS)
    dv = _velocity_increments(sc, rng)
    x0 = np.array([sc.tgt_x0, sc.tgt_y0, sc.tgt_vx0, sc.tgt_vy0])

    fast = _integrate_em(sc, dv, x0)
    slow = _integrate_em_reference(sc, dv, x0)
    assert np.allclose(fast, slow, rtol=0, atol=1e-8)


def test_em_step_covariance_closed_form_matches_the_weight_sums():
    """Verify the (n-1)n(2n-1)/6 collapse against explicit sums over substeps."""
    q, dt, n = 1e-3, 1.0, 50
    h = dt / n
    var_dv = q * h
    w = (n - 1 - np.arange(n)).astype(float)

    Q = em_step_covariance(q, dt, n)
    assert Q[0, 0] == pytest.approx(h**2 * var_dv * np.sum(w**2), rel=1e-12)
    assert Q[0, 2] == pytest.approx(h * var_dv * np.sum(w), rel=1e-12)
    assert Q[2, 2] == pytest.approx(n * var_dv, rel=1e-12)
    assert np.allclose(Q, Q.T)


def test_em_covariance_is_realised_by_the_generator():
    """Monte Carlo check that the integrator actually produces that covariance."""
    sc = Scenario(steps=1, truth_substeps=50, q=1e-3, tgt_vx0=0.0, tgt_vy0=0.0)
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(4242)

    n_trials = 20000
    samples = np.empty((n_trials, 4))
    for i in range(n_trials):
        samples[i] = _integrate_em(sc, _velocity_increments(sc, rng), x0)[1]

    emp = np.cov(samples, rowvar=False)
    ref = em_step_covariance(sc.q, sc.dt, sc.truth_substeps)
    scale = np.sqrt(np.outer(np.diag(ref), np.diag(ref)))
    # Relative MC error on a variance from 2e4 samples is sqrt(2/n) = 1%.
    assert np.allclose(emp / scale, ref / scale, atol=0.05)


def test_the_intentional_mismatch_with_the_filter_is_pinned():
    """The whole content of 'no inverse crime' at this substep count.

    If someone makes the truth model share the filter's Q these become 1.0.
    """
    q, dt = 1e-3, 1.0
    Q_em = em_step_covariance(q, dt, 50)
    Q_filter = cwna_process_noise(q, dt)
    assert Q_em[0, 0] / Q_filter[0, 0] == pytest.approx(0.9702, abs=1e-3)
    assert Q_em[0, 2] / Q_filter[0, 2] == pytest.approx(0.9800, abs=1e-3)
    assert Q_em[2, 2] / Q_filter[2, 2] == pytest.approx(1.0, rel=1e-12)
    assert Q_em[0, 0] < Q_filter[0, 0], "mismatch is one-sided: truth is quieter"


def test_the_mismatch_vanishes_as_substeps_grow():
    """It must be a discretisation artefact, not a bug."""
    q, dt = 1e-3, 1.0
    Q_filter = cwna_process_noise(q, dt)
    ratios = [em_step_covariance(q, dt, n)[0, 0] / Q_filter[0, 0] for n in (50, 200, 800, 20000)]
    assert all(a < b for a, b in zip(ratios, ratios[1:])), "must converge monotonically"
    assert ratios[-1] == pytest.approx(1.0, abs=1e-4)


def test_target_truth_is_reproducible_and_run_dependent():
    sc = Scenario(steps=30)
    assert np.array_equal(target_truth(sc, 3), target_truth(sc, 3))
    assert not np.array_equal(target_truth(sc, 3), target_truth(sc, 4))
    assert target_truth(sc, 0).shape == (sc.steps + 1, 4)
    assert np.allclose(target_truth(sc, 0)[0], [sc.tgt_x0, sc.tgt_y0, sc.tgt_vx0, sc.tgt_vy0])


def test_zero_process_noise_gives_exact_constant_velocity():
    sc = Scenario(steps=100, q=0.0)
    truth = target_truth(sc, 0)
    t = np.arange(sc.steps + 1) * sc.dt
    assert np.allclose(truth[:, 0], sc.tgt_x0 + sc.tgt_vx0 * t)
    assert np.allclose(truth[:, 1], sc.tgt_y0 + sc.tgt_vy0 * t)
    assert np.allclose(truth[:, 2], sc.tgt_vx0)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_initial_estimate_is_drawn_from_the_prior():
    """Seeding at truth would make NEES start at zero and report unearned consistency."""
    sc = Scenario(steps=5, p0_pos=300.0, p0_vel=1.5)
    truth0 = np.array([sc.tgt_x0, sc.tgt_y0, sc.tgt_vx0, sc.tgt_vy0])
    errs = np.array([initial_estimate(replace(sc, seed=s), truth0, 0) - truth0 for s in range(4000)])
    emp = np.cov(errs, rowvar=False)
    ref = initial_covariance(sc)
    assert np.allclose(np.diag(emp), np.diag(ref), rtol=0.10)
    assert np.abs(errs.mean(axis=0)).max() < 0.1 * np.sqrt(np.diag(ref)).max()


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------


def _detections_for(sc: Scenario, run: int = 0) -> Detections:
    own = ownship_track(sc)
    return generate_detections(sc, target_truth(sc, run), own, run)


def test_phase1_scan_holds_exactly_one_bearing():
    sc = Scenario(steps=50, pd=1.0, clutter_rate=0.0)
    det = _detections_for(sc)
    assert det.per_step[0].size == 0, "k=0 carries no measurement by convention"
    assert all(s.size == 1 for s in det.per_step[1:])
    assert all(i == 0 for i in det.truth_index[1:])


def test_measurement_noise_matches_sigma():
    sc = Scenario(steps=4000, sigma_bearing_deg=0.5, q=0.0)
    own = ownship_track(sc)
    truth = target_truth(sc, 0)
    det = generate_detections(sc, truth, own, 0)
    true_b = bearing(truth[:, :2], own.xy)
    resid = np.array([wrap_pi(det.per_step[k][0] - true_b[k]) for k in range(1, sc.steps + 1)])
    assert resid.std() == pytest.approx(sc.sigma_bearing, rel=0.05)
    assert abs(resid.mean()) < 4 * sc.sigma_bearing / np.sqrt(resid.size)


def test_missed_detections_appear_at_the_right_rate():
    sc = Scenario(steps=4000, pd=0.7)
    det = _detections_for(sc)
    rate = det.detected[1:].mean()
    assert rate == pytest.approx(0.7, abs=0.03)
    missed = [k for k in range(1, sc.steps + 1) if det.truth_index[k] is None]
    assert all(det.per_step[k].size == 0 for k in missed), "no clutter configured"


def test_clutter_arrives_at_the_poisson_rate_and_target_is_still_present():
    sc = Scenario(steps=4000, clutter_rate=2.0)
    det = _detections_for(sc)
    counts = np.array([det.per_step[k].size - 1 for k in range(1, sc.steps + 1)])
    assert counts.mean() == pytest.approx(2.0, abs=0.1)
    assert counts.var() == pytest.approx(2.0, abs=0.2), "Poisson: variance equals mean"
    assert all(i is not None for i in det.truth_index[1:]), "pd = 1"


def test_target_is_not_always_first_in_the_scan():
    """Ordering must not be a free association cue."""
    sc = Scenario(steps=2000, clutter_rate=3.0)
    det = _detections_for(sc)
    positions = np.array([det.truth_index[k] for k in range(1, sc.steps + 1)])
    assert positions.max() > 0 and (positions > 0).mean() > 0.4


def test_changing_pd_does_not_disturb_the_measurement_stream():
    """Common random numbers: scenarios must stay paired when a knob moves.

    The measurement noise is drawn for every step regardless of detection, so
    turning pd down changes which measurements exist, never their values.
    """
    a = Scenario(steps=200, pd=1.0)
    b = replace(a, pd=0.6)
    own = ownship_track(a)
    truth = target_truth(a, 0)
    det_a = generate_detections(a, truth, own, 0)
    det_b = generate_detections(b, truth, own, 0)

    common = [k for k in range(1, a.steps + 1) if det_b.truth_index[k] is not None]
    assert len(common) > 50
    for k in common:
        assert det_b.per_step[k][det_b.truth_index[k]] == pytest.approx(det_a.per_step[k][0])
