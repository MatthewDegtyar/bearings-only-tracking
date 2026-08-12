"""Tests for validation gating and association."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from kf2.filters import BearingsOnlyEKF
from kf2.gating import associate, gate_threshold, truncated_chi2_moments


def _tracked_filter():
    """A filter with a modest, well-conditioned covariance for gating tests."""
    ekf = BearingsOnlyEKF(q=1e-3, sigma_bearing=0.5 * np.pi / 180)
    ekf.initialise(np.array([5000.0, 0.0, 0.0, 0.0]), np.diag([100.0**2, 100.0**2, 1.0, 1.0]))
    return ekf


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------


def test_gate_threshold_matches_the_chi_square_table():
    assert gate_threshold(0.997, 1) == pytest.approx(8.8074, abs=1e-3)
    assert gate_threshold(0.95, 1) == pytest.approx(3.8415, abs=1e-3)
    assert gate_threshold(0.95, 2) == pytest.approx(5.9915, abs=1e-3)


def test_gate_threshold_rejects_impossible_probabilities():
    for p in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            gate_threshold(p)


# ---------------------------------------------------------------------------
# Truncated moments -- the Phase 3 correction
# ---------------------------------------------------------------------------


def test_truncated_moments_tend_to_the_untruncated_ones():
    mean, var = truncated_chi2_moments(1e6, dim=1)
    assert mean == pytest.approx(1.0, abs=1e-6)
    assert var == pytest.approx(2.0, abs=1e-4)
    mean4, var4 = truncated_chi2_moments(1e6, dim=4)
    assert mean4 == pytest.approx(4.0, abs=1e-5)
    assert var4 == pytest.approx(8.0, abs=1e-3)


def test_gating_biases_nis_downwards():
    """The reason this component exists.

    A gate rejects exactly the large innovations that would reveal
    overconfidence, so the accepted sample reads healthier than chi-square, and
    the naive reference of 1.0 is wrong under gating. The size of the bias is
    strongly gate-dependent, which is the practical point -- it is negligible at
    a loose gate and dominant at a tight one:

        gate 0.9999  E[NIS | gated] = 0.998   ( 0.2% low)
        gate 0.997   E[NIS | gated] = 0.971   ( 2.9% low)
        gate 0.99    E[NIS | gated] = 0.925   ( 7.5% low)
        gate 0.95    E[NIS | gated] = 0.759   (24.1% low)
        gate 0.90    E[NIS | gated] = 0.623   (37.7% low)

    So a filter that tightens its gate makes its own NIS look better. That is
    the self-reinforcing loop Phase 3 has to break.
    """
    means = [truncated_chi2_moments(gate_threshold(p, 1), 1)[0] for p in (0.90, 0.95, 0.99, 0.997)]
    assert all(m < 1.0 for m in means)
    assert all(a < b for a, b in zip(means, means[1:])), "tighter gate, larger bias"
    assert means[-1] == pytest.approx(0.9709, abs=1e-3)
    assert means[0] == pytest.approx(0.6230, abs=1e-3)
    assert truncated_chi2_moments(gate_threshold(0.997, 1), 1)[1] == pytest.approx(1.714, abs=1e-2)


@pytest.mark.parametrize("dim,prob", [(1, 0.95), (1, 0.997), (2, 0.99), (4, 0.95)])
def test_truncated_moments_match_monte_carlo(dim, prob):
    rng = np.random.default_rng(3)
    T = gate_threshold(prob, dim)
    x = rng.chisquare(dim, 400000)
    kept = x[x <= T]
    mean, var = truncated_chi2_moments(T, dim)
    assert kept.mean() == pytest.approx(mean, rel=0.01)
    assert kept.var() == pytest.approx(var, rel=0.03)


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------


def test_empty_scan_is_a_missed_update_not_an_error():
    a = associate(_tracked_filter(), np.zeros(2), np.empty(0), gate_threshold(0.997))
    assert a.accepted is False and a.z is None and a.nis is None
    assert a.n_candidates == 0 and a.n_in_gate == 0


def test_picks_the_nearest_candidate_in_normalised_distance():
    ekf = _tracked_filter()
    own = np.zeros(2)
    truth_b = ekf.predicted_bearing(own)
    sigma = np.sqrt(ekf.innovation(truth_b, own).S)
    scan = np.array([truth_b + 2.0 * sigma, truth_b + 0.2 * sigma, truth_b - 1.5 * sigma])

    a = associate(ekf, own, scan, gate_threshold(0.997))
    assert a.accepted and a.picked_index == 1
    assert a.n_candidates == 3 and a.n_in_gate == 3
    assert a.z == pytest.approx(scan[1])


def test_candidates_outside_the_gate_are_rejected():
    ekf = _tracked_filter()
    own = np.zeros(2)
    truth_b = ekf.predicted_bearing(own)
    sigma = np.sqrt(ekf.innovation(truth_b, own).S)
    threshold = gate_threshold(0.997)
    far = truth_b + 10.0 * sigma  # normalised distance 100 >> 8.8

    assert not associate(ekf, own, np.array([far]), threshold).accepted
    a = associate(ekf, own, np.array([far, truth_b + 0.5 * sigma]), threshold)
    assert a.accepted and a.n_candidates == 2 and a.n_in_gate == 1


def test_gate_acceptance_rate_matches_the_gate_probability():
    """A correctly sized gate accepts the target at its nominal rate."""
    ekf = _tracked_filter()
    own = np.zeros(2)
    truth_b = ekf.predicted_bearing(own)
    sigma = np.sqrt(ekf.innovation(truth_b, own).S)

    rng = np.random.default_rng(9)
    draws = truth_b + sigma * rng.standard_normal(20000)
    for prob in (0.95, 0.997):
        T = gate_threshold(prob, 1)
        accepted = np.mean([associate(ekf, own, np.array([z]), T).accepted for z in draws])
        assert accepted == pytest.approx(prob, abs=0.01)


def test_association_does_not_mutate_the_filter():
    ekf = _tracked_filter()
    x, P = ekf.state, ekf.covariance
    associate(ekf, np.zeros(2), np.array([0.0, 0.01, -0.01]), gate_threshold(0.997))
    assert np.array_equal(ekf.state, x) and np.array_equal(ekf.covariance, P)


def test_nis_of_the_accepted_measurement_is_inside_the_gate():
    ekf = _tracked_filter()
    own = np.zeros(2)
    T = gate_threshold(0.997)
    rng = np.random.default_rng(5)
    truth_b = ekf.predicted_bearing(own)
    sigma = np.sqrt(ekf.innovation(truth_b, own).S)
    for _ in range(200):
        scan = truth_b + sigma * rng.standard_normal(4)
        a = associate(ekf, own, scan, T)
        if a.accepted:
            assert a.nis <= T
