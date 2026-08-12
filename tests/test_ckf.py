"""Tests for the cubature filter -- the fix.

Two things are being asserted: that it is a correct filter (agrees with the EKF
wherever linearisation is valid, keeps a usable covariance), and that it actually
repairs the mechanism it was introduced for.
"""

from __future__ import annotations

import numpy as np
import pytest

from kf2 import Scenario, datagen, evaluate, run_monte_carlo, run_trial
from kf2.ckf5 import BearingsOnlyCKF5 as BearingsOnlyCKF, _cubature_points
from kf2.config import replace
from kf2.filters import BearingsOnlyEKF


def _pair(P=None):
    x = np.array([5000.0, 3000.0, -3.0, 1.5])
    P = np.diag([100.0**2, 100.0**2, 1.0, 1.0]) if P is None else P
    a, b = BearingsOnlyEKF(1e-3, 0.5 * np.pi / 180), BearingsOnlyCKF(1e-3, 0.5 * np.pi / 180)
    a.initialise(x, P)
    b.initialise(x, P)
    return a, b


# ---------------------------------------------------------------------------
# Cubature rule
# ---------------------------------------------------------------------------


def test_cubature_points_reproduce_the_moments_they_encode():
    """The 2n points with equal weights must have exactly the mean and
    covariance they were built from -- that is the whole contract."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    P = A @ A.T + 4 * np.eye(4)
    x = rng.standard_normal(4) * 10

    for degree, n_pts in ((3, 8), (5, 25)):
        pts, w = _cubature_points(x, P, degree)
        assert pts.shape == (n_pts, 4), degree
        assert w.sum() == pytest.approx(1.0)
        assert np.allclose(w @ pts, x), degree
        d = pts - x
        assert np.allclose((w[:, None] * d).T @ d, P), degree


def test_weights_are_positive_by_construction():
    """The reason for cubature over the unscented transform: no negative centre
    weight, so no route to a non-PSD covariance from the weighting alone."""
    for degree in (3, 5):
        _, w = _cubature_points(np.zeros(4), np.eye(4), degree)
        assert (w > 0).all(), f"degree {degree} has a non-positive weight"


def test_degree_five_integrates_the_fourth_order_term_degree_three_misses():
    """The whole reason degree 5 is the default.

    The bearing's curvature enters the innovation *variance* at fourth order, so
    a rule that does not integrate fourth moments gets it wrong by an amount that
    depends on where its points sit -- which is what made the fix frame-dependent.
    """
    from kf2.ckf5 import RULES

    for degree, e4 in ((3, 4.0), (5, 3.0)):
        z, w = RULES[degree]
        assert float(np.sum(w * z[:, 0] ** 4)) == pytest.approx(e4, abs=1e-9)
    # ...and degree 5 stops at degree 6, as it must.
    z, w = RULES[5]
    assert float(np.sum(w * z[:, 0] ** 6)) == pytest.approx(9.0, abs=1e-9)  # true 15


# ---------------------------------------------------------------------------
# Agreement with the EKF where linearisation is valid
# ---------------------------------------------------------------------------


def test_matches_the_ekf_when_uncertainty_is_small():
    """With a tight covariance the bearing is locally linear, so the two filters
    must agree closely. Disagreement here would mean a bug, not a method."""
    P = np.diag([1.0, 1.0, 0.01, 0.01])
    ekf, ckf = _pair(P)
    own = np.zeros(2)
    z = ekf.predicted_bearing(own) + 1e-3

    ie, ic = ekf.innovation(z, own), ckf.innovation(z, own)
    assert ic.nu == pytest.approx(ie.nu, rel=1e-9)
    assert ic.S == pytest.approx(ie.S, rel=1e-6)

    ekf.update(z, own)
    ckf.update(z, own)
    assert np.allclose(ckf.state, ekf.state, rtol=1e-5)
    assert np.allclose(ckf.covariance, ekf.covariance, rtol=1e-4, atol=1e-9)


def test_diverges_from_the_ekf_when_uncertainty_is_large():
    """...and must not agree when the linearisation is invalid, or it would be
    an expensive way to reproduce the bug."""
    P = np.diag([1500.0**2, 1500.0**2, 4.0, 4.0])
    ekf, ckf = _pair(P)
    own = np.zeros(2)
    z = ekf.predicted_bearing(own) + 0.02
    assert ckf.innovation(z, own).S != pytest.approx(ekf.innovation(z, own).S, rel=1e-3)


def test_prediction_is_identical_to_the_ekf():
    """Dynamics are exactly linear, so only the update differs. Keeping
    prediction shared is what makes the comparison one-variable."""
    ekf, ckf = _pair()
    ekf.predict(1.0)
    ckf.predict(1.0)
    assert np.allclose(ekf.state, ckf.state)
    assert np.allclose(ekf.covariance, ckf.covariance)


# ---------------------------------------------------------------------------
# Numerical behaviour
# ---------------------------------------------------------------------------


def test_covariance_stays_symmetric_and_positive_definite_over_a_run():
    sc = replace(Scenario(steps=600), p0_pos=1000.0)
    own = datagen.ownship_track(sc)
    truth = datagen.target_truth(sc, 0)
    det = datagen.generate_detections(sc, truth, own, 0)

    f = BearingsOnlyCKF(sc.filter_q, sc.sigma_bearing)
    f.initialise(datagen.initial_estimate(sc, truth[0], 0), datagen.initial_covariance(sc))
    worst = np.inf
    for k in range(1, sc.steps + 1):
        f.predict(sc.dt)
        f.update(float(det.per_step[k][0]), own.xy[k])
        P = f.covariance
        assert np.array_equal(P, P.T)
        worst = min(worst, np.linalg.eigvalsh(P).min())
    assert worst > 0.0


def test_bearings_near_the_branch_cut_are_averaged_circularly():
    """Sigma points straddling +/-pi must not average to zero bearing."""
    ckf = BearingsOnlyCKF(1e-3, 0.5 * np.pi / 180)
    ckf.initialise(np.array([-6000.0, 0.0, 0.0, 0.0]), np.diag([300.0**2, 300.0**2, 1.0, 1.0]))
    own = np.zeros(2)
    z_hat, Pzz, _ = ckf._moments(own)
    assert abs(abs(z_hat) - np.pi) < 0.05, "predicted bearing must stay near +/-pi"
    assert 0 < Pzz < 1.0


# ---------------------------------------------------------------------------
# Does it fix the thing it was introduced for?
# ---------------------------------------------------------------------------


def test_interchangeable_with_the_ekf_in_the_pipeline():
    a = run_trial(replace(Scenario(steps=200), p0_pos=600.0), 0, estimator="ckf")
    b = run_trial(replace(Scenario(steps=200), p0_pos=600.0), 0, estimator="ekf")
    assert a.nees.shape == b.nees.shape
    assert np.isfinite(a.nees).all()
    # Paired: identical measurement stream, so any difference is the estimator.
    assert not np.array_equal(a.pos_err, b.pos_err)


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="unknown estimator"):
        run_trial(Scenario(steps=10), 0, estimator="wishful")


def test_ckf_reduces_nees_inflation_where_the_ekf_is_inconsistent():
    """The result the fix exists for.

    At p0_pos = 1000 m the EKF is badly overconfident. The CKF, on byte-identical
    measurements, must be materially closer to 4 -- while NIS, which never saw
    the problem, stays put.
    """
    sc = replace(Scenario(mc_runs=150, steps=600, gate_windows=4), p0_pos=1000.0)
    ekf = evaluate(run_monte_carlo(sc, estimator="ekf"))
    ckf = evaluate(run_monte_carlo(sc, estimator="ckf"))
    pick = lambda r, p: next(c for c in r.consistency.criteria if c.name.startswith(p))  # noqa: E731

    n_e, n_c = pick(ekf, "NEES bias").statistic, pick(ckf, "NEES bias").statistic
    s_e, s_c = pick(ekf, "NIS bias").statistic, pick(ckf, "NIS bias").statistic

    assert n_e > 6.0, f"the EKF should be badly inconsistent here, got {n_e:.2f}"
    assert n_c < 0.7 * n_e, f"CKF must materially reduce it: {n_e:.2f} -> {n_c:.2f}"
    # NIS is blind to all of this -- both estimators sit near their reference.
    assert abs(s_e - 1.0) < 0.05 and abs(s_c - 1.0) < 0.05


def test_ckf_does_not_break_the_benign_baseline():
    """A fix that damages the case that was already fine is not a fix."""
    sc = replace(Scenario(mc_runs=150, steps=300, gate_windows=4))
    assert evaluate(run_monte_carlo(sc, estimator="ckf")).passed


# ---------------------------------------------------------------------------
# Oracle diagnostic
# ---------------------------------------------------------------------------


def test_oracle_is_registered_and_uses_truth():
    """The oracle linearises at ground truth. Not implementable -- it exists to
    bound what is achievable, and lives in the harness so that it and the
    headline tables can never be generated at different settings."""
    from kf2.montecarlo import ESTIMATORS, ORACLE_ESTIMATORS

    assert "oracle" in ESTIMATORS and "oracle" in ORACLE_ESTIMATORS
    sc = replace(Scenario(mc_runs=40, steps=300), p0_pos=1000.0)
    oracle = run_monte_carlo(sc, estimator="oracle")
    ekf = run_monte_carlo(sc, estimator="ekf")
    # Same measurements, different Jacobian evaluation point.
    assert not np.array_equal(oracle.pos_err, ekf.pos_err)
    assert np.isfinite(oracle.nees).all()


def test_oracle_beats_the_ekf_where_the_point_is_the_mechanism():
    """At 600 m the linearisation point is the whole story, so the oracle should
    be close to nominal while the EKF is not."""
    sc = replace(Scenario(mc_runs=200, steps=600), p0_pos=600.0)
    o = np.nanmean(np.nanmean(run_monte_carlo(sc, estimator="oracle").nees, axis=1))
    e = np.nanmean(np.nanmean(run_monte_carlo(sc, estimator="ekf").nees, axis=1))
    assert o < e
    assert abs(o - 4.0) < 0.5, f"oracle should be near nominal at 600 m, got {o:.3f}"


# ---------------------------------------------------------------------------
# Iterated cubature
# ---------------------------------------------------------------------------


def test_one_iteration_reduces_exactly_to_plain_cubature():
    """The algebra must collapse: with x_0 = prior, A P- A' + Om + R = Pzz and
    K = Pxz / Pzz. If this drifts, the iterated filter is a different object
    from the one it claims to generalise."""
    from kf2.ckf5 import IteratedCKF

    x = np.array([5000.0, 3000.0, -3.0, 1.5])
    P = np.diag([400.0**2, 400.0**2, 2.0, 2.0])
    plain, once = BearingsOnlyCKF(1e-3, 0.0087), IteratedCKF(1e-3, 0.0087, iterations=1)
    plain.initialise(x, P)
    once.initialise(x, P)
    own = np.zeros(2)
    z = plain.predicted_bearing(own) + 0.01

    a, b = plain.update(z, own), once.update(z, own)
    assert b.nu == pytest.approx(a.nu, rel=1e-10)
    assert b.S == pytest.approx(a.S, rel=1e-9)
    assert np.allclose(once.state, plain.state, rtol=1e-9)
    assert np.allclose(once.covariance, plain.covariance, rtol=1e-9)


def test_iteration_count_is_validated_and_scenario_driven():
    from kf2.ckf5 import IteratedCKF
    from kf2.montecarlo import ESTIMATORS

    with pytest.raises(ValueError):
        IteratedCKF(1e-3, 0.0087, iterations=0)
    sc = replace(Scenario(), ckf_iterations=7)
    assert ESTIMATORS["ickf"](sc).iterations == 7
    with pytest.raises(ValueError):
        replace(Scenario(), ckf_iterations=0)


def test_iterated_covariance_stays_usable():
    """Iterating must not shrink the covariance once per pass -- the classic bug
    when the prior is re-applied cumulatively instead of held fixed."""
    from kf2.ckf5 import IteratedCKF

    x = np.array([6000.0, 4000.0, -3.0, 1.5])
    P = np.diag([800.0**2, 800.0**2, 2.0, 2.0])
    own = np.zeros(2)
    traces = []
    for iters in (1, 2, 5):
        f = IteratedCKF(1e-3, 0.0087, iterations=iters, tol=0.0)
        f.initialise(x, P)
        f.update(f.predicted_bearing(own) + 0.02, own)
        assert np.linalg.eigvalsh(f.covariance).min() > 0
        traces.append(np.trace(f.covariance))
    # More passes must not monotonically collapse the covariance.
    assert traces[-1] > 0.25 * traces[0], traces


def test_iterated_is_interchangeable_in_the_pipeline():
    sc = replace(Scenario(mc_runs=20, steps=200), p0_pos=800.0)
    mc = run_monte_carlo(sc, estimator="ickf")
    assert np.isfinite(mc.nees).all()
    assert mc.estimator == "ickf"


def test_iterating_degrades_rather_than_improves():
    """A negative result, pinned so it cannot be quietly re-litigated.

    Iterating the sigma-point update converges to a fixed point worse than the
    single pass: along the weakly observable range direction the posterior has
    been displaced by a measurement carrying no information there, so the
    linearisation point ends up somewhere the data never justified. The damage
    appears entirely at the first extra pass.
    """
    sc = replace(Scenario(mc_runs=80, steps=600), p0_pos=1000.0, ckf_iteration_tol=0.0)
    from kf2.evaluation import survivor_mask

    def loss_and_nees(iters):
        mc = run_monte_carlo(replace(sc, ckf_iterations=iters), estimator="ickf")
        alive = survivor_mask(
            mc.pos_err,
            settle_index=sc.settle_index,
            divergence_pos_error=sc.divergence_pos_error,
        )
        return 1.0 - alive.mean(), float(np.nanmean(np.nanmean(mc.nees, axis=1)[alive]))

    loss1, nees1 = loss_and_nees(1)
    loss3, nees3 = loss_and_nees(3)
    assert loss3 > 3 * loss1, f"iterating must be seen to destabilise: {loss1:.3f} -> {loss3:.3f}"
    assert nees3 > nees1


def test_sample_from_mode_does_not_explain_the_failure():
    """Rules out the obvious hypothesis: that a shrinking sampling covariance
    drives a confidence spiral. Holding the spread at the prior changes almost
    nothing, so the failure is about where the point goes, not how wide the
    sampling is."""
    sc = replace(Scenario(mc_runs=60, steps=600), p0_pos=1000.0)
    from kf2.evaluation import survivor_mask

    losses = []
    for mode in ("posterior", "prior"):
        mc = run_monte_carlo(replace(sc, ckf_sample_from=mode), estimator="ickf")
        alive = survivor_mask(
            mc.pos_err,
            settle_index=sc.settle_index,
            divergence_pos_error=sc.divergence_pos_error,
        )
        losses.append(1.0 - alive.mean())
    assert abs(losses[0] - losses[1]) < 0.10, f"modes should behave alike: {losses}"
    assert min(losses) > 0.15, "both modes should destabilise"
