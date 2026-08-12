"""Tests for the bearings-only EKF."""

from __future__ import annotations

import numpy as np
import pytest

from kf2.config import Scenario
from kf2.datagen import (
    bearing,
    generate_detections,
    initial_covariance,
    initial_estimate,
    ownship_track,
    target_truth,
    wrap_pi,
)
from kf2.filters import BearingsOnlyEKF, cv_transition, cwna_process_noise


def _filter_at(x, P=None, q=1e-3, sigma=0.5 * np.pi / 180):
    ekf = BearingsOnlyEKF(q, sigma)
    ekf.initialise(np.asarray(x, float), np.eye(4) if P is None else P)
    return ekf


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_cv_transition_moves_position_by_velocity():
    F = cv_transition(2.0)
    x = np.array([0.0, 0.0, 3.0, -1.0])
    assert np.allclose(F @ x, [6.0, -2.0, 3.0, -1.0])


def test_process_noise_is_psd_and_scales_correctly():
    Q = cwna_process_noise(1e-3, 1.0)
    assert np.allclose(Q, Q.T)
    assert np.linalg.eigvalsh(Q).min() > 0
    # Position variance is cubic in dt, velocity variance linear.
    assert cwna_process_noise(1e-3, 2.0)[0, 0] / Q[0, 0] == pytest.approx(8.0)
    assert cwna_process_noise(1e-3, 2.0)[2, 2] / Q[2, 2] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Measurement model
# ---------------------------------------------------------------------------


def test_jacobian_matches_central_differences():
    x = np.array([5000.0, 3000.0, -3.0, 1.5])
    own = np.array([100.0, -50.0])
    ekf = _filter_at(x)
    H = ekf.jacobian(own)

    def h(state):
        d = state[:2] - own
        return np.arctan2(d[1], d[0])

    eps = 1e-4
    for i in range(4):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        assert H[0, i] == pytest.approx((h(xp) - h(xm)) / (2 * eps), abs=1e-9)


def test_bearing_is_blind_to_velocity_at_a_single_instant():
    """The structural reason range is unobservable without an ownship manoeuvre."""
    H = _filter_at([5000.0, 3000.0, -3.0, 1.5]).jacobian(np.zeros(2))
    assert H[0, 2] == 0.0 and H[0, 3] == 0.0


def test_innovation_wraps_across_the_branch_cut():
    """Target just past +pi, estimate just short of it: the innovation is small."""
    own = np.zeros(2)
    ekf = _filter_at([-1000.0, 1.0, 0.0, 0.0])  # predicted bearing just under +pi
    z = -np.pi + 0.001  # measurement just past the cut
    inn = ekf.innovation(z, own)
    assert abs(inn.nu) < 0.01, "unwrapped this would be about 2 pi"


def test_innovation_does_not_mutate_the_filter():
    ekf = _filter_at([5000.0, 3000.0, -3.0, 1.5])
    x, P = ekf.state, ekf.covariance
    ekf.innovation(0.5, np.zeros(2))
    assert np.array_equal(ekf.state, x) and np.array_equal(ekf.covariance, P)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


def test_update_reduces_uncertainty_across_the_line_of_sight():
    P0 = np.diag([300.0**2, 300.0**2, 1.5**2, 1.5**2])
    ekf = _filter_at([5000.0, 0.0, 0.0, 0.0], P0)
    before = ekf.covariance
    ekf.update(ekf.predicted_bearing(np.zeros(2)), np.zeros(2))
    after = ekf.covariance
    # Target due east: bearing constrains the cross-range (y) direction only.
    assert after[1, 1] < 0.5 * before[1, 1]
    assert after[0, 0] == pytest.approx(before[0, 0], rel=1e-9), "range is untouched"
    assert np.trace(after) < np.trace(before)


def test_zero_innovation_leaves_the_state_alone():
    ekf = _filter_at([5000.0, 3000.0, -3.0, 1.5], np.diag([1e4, 1e4, 1.0, 1.0]))
    x = ekf.state
    ekf.update(ekf.predicted_bearing(np.zeros(2)), np.zeros(2))
    assert np.allclose(ekf.state, x)


def test_covariance_stays_symmetric_and_positive_definite_over_a_run():
    sc = Scenario(steps=600)
    own = ownship_track(sc)
    truth = target_truth(sc, 0)
    det = generate_detections(sc, truth, own, 0)

    ekf = BearingsOnlyEKF(sc.filter_q, sc.sigma_bearing)
    ekf.initialise(initial_estimate(sc, truth[0], 0), initial_covariance(sc))

    min_eig = np.inf
    for k in range(1, sc.steps + 1):
        ekf.predict(sc.dt)
        ekf.update(float(det.per_step[k][0]), own.xy[k])
        P = ekf.covariance
        # FilterPy does not force exact symmetry the way the earlier hand-rolled
        # filter did, so the assertion is on the property that matters rather
        # than on bitwise equality. Measured over 600 steps at p0 = 900 m the
        # relative asymmetry stays at 2e-15, which is last-bit rounding of the
        # Joseph form and not a drift.
        asym = np.abs(P - P.T).max() / np.abs(P).max()
        assert asym < 1e-12, f"covariance asymmetry {asym:.2e} exceeds rounding"
        min_eig = min(min_eig, np.linalg.eigvalsh(0.5 * (P + P.T)).min())
    assert min_eig > 0.0


def test_filter_tracks_better_than_the_prior():
    """A basic sanity floor: the filter must actually use the measurements."""
    sc = Scenario(steps=400)
    own = ownship_track(sc)
    truth = target_truth(sc, 0)
    det = generate_detections(sc, truth, own, 0)

    ekf = BearingsOnlyEKF(sc.filter_q, sc.sigma_bearing)
    ekf.initialise(initial_estimate(sc, truth[0], 0), initial_covariance(sc))
    start = np.hypot(*(truth[0][:2] - ekf.state[:2]))
    for k in range(1, sc.steps + 1):
        ekf.predict(sc.dt)
        ekf.update(float(det.per_step[k][0]), own.xy[k])
    end = np.hypot(*(truth[-1][:2] - ekf.state[:2]))
    assert end < start


def test_predict_is_the_textbook_linear_propagation():
    P0 = np.diag([100.0, 200.0, 3.0, 4.0])
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    ekf = _filter_at(x0, P0, q=2e-3)
    ekf.predict(0.5)
    F, Q = cv_transition(0.5), cwna_process_noise(2e-3, 0.5)
    assert np.allclose(ekf.state, F @ x0)
    assert np.allclose(ekf.covariance, F @ P0 @ F.T + Q)


def test_update_rejects_a_degenerate_innovation_covariance():
    ekf = BearingsOnlyEKF(q=0.0, sigma_bearing=0.0)
    ekf.initialise(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros((4, 4)))
    with pytest.raises(FloatingPointError):
        ekf.update(0.0, np.zeros(2))
