"""Control experiment: is the harness itself unbiased?

Everything except the measurement model is shared with the real pipeline -- the
same truth generator, the same consistent initialisation, the same
constant-velocity prediction, the same Joseph-form update, the same NEES
computation. Only the measurement is swapped for a *linear* one, where the Kalman
filter is exactly optimal and NEES must be exactly ``dim``.

This is the test that separates "the EKF is nonlinear" from "the harness has a
bug". Without it, the residual NEES bias in the bearings-only baseline is
unattributable, and the plan's rule -- a gate failure is a bug, not a phenomenon
-- cannot be applied.

**An earlier version of this file re-implemented the Kalman filter inline.** It
imported only the F and Q matrices and did its own predict, update and NEES. So
it was blind by construction to defects in exactly the components it claimed to
exonerate: sabotaging ``BearingsOnlyEKF.predict`` and ``.update`` left it reading
4.0017, unchanged to the last digit. It now subclasses the real filter and
overrides only the measurement model, and
:func:`test_control_detects_a_sabotaged_prediction` asserts that it is actually
sensitive to what it validates.
"""

from __future__ import annotations

import numpy as np
import pytest

from kf2 import datagen
from kf2.config import Scenario
from kf2.filters import BearingsOnlyEKF, cwna_process_noise
from kf2.evaluation import nees_of
from kf2.rng import Stream, stream_rng

#: Maps position to a measurement of order 0.5 rad, so the innovation wrapping in
#: the shared update path is an identity here rather than a distortion.
SCALE = 1.0e-4


class LinearScalarEKF(BearingsOnlyEKF):
    """The real filter with the bearing replaced by an exactly linear scalar.

    ``h(x) = SCALE * x[axis]``, with ``axis`` alternating between px and py on
    successive steps. The alternation keeps the scalar-measurement interface the
    rest of the pipeline assumes while leaving the system fully observable -- a
    fixed axis would leave the other position component unobserved, which is
    valid but much noisier and would blunt the control.

    Inherits ``predict``, ``update`` (Joseph form) and ``_symmetrise`` unchanged,
    so a defect in any of them shows up here. Only the measurement function and
    its Jacobian are overridden, and the Jacobian is exact rather than
    approximate -- which is the point: with a linear measurement the Kalman
    filter is optimal and NEES must be exactly 4.
    """

    axis = 1

    def measurement(self, x: np.ndarray, own_xy: np.ndarray) -> float:
        return float(SCALE * np.asarray(x).ravel()[self.axis])

    def jacobian(self, own_xy: np.ndarray, x_lin: np.ndarray | None = None) -> np.ndarray:
        H = np.zeros((1, 4))
        H[0, self.axis] = SCALE
        return H


def _linear_measurement_per_run(sc: Scenario, sigma_pos: float) -> np.ndarray:
    """Per-*run* mean NEES. These are the independent samples.

    Averaging over time steps first matters: NEES values within a run are almost
    perfectly correlated, so treating the 400 x 601 grid as independent
    understates the standard error by ~8x. The across-run SD of a per-run mean is
    ~0.93, so at 400 runs the 95% half-width is about 2.3% -- which is *larger*
    than the oracle floor this project quotes. Any assertion here has to use that
    interval rather than a tolerance picked by eye.
    """
    per_step = _linear_measurement_nees(sc, sigma_pos, per_run=True)
    return per_step


def _linear_measurement_nees(sc: Scenario, sigma_pos: float, per_run: bool = False) -> np.ndarray:
    """Per-step average NEES for the linear-measurement filter."""
    sigma_z = sigma_pos * SCALE
    total = np.zeros(sc.steps + 1)
    run_means = np.zeros(sc.mc_runs)
    own_xy = np.zeros(2)  # unused by the linear measurement, kept for the interface

    for run in range(sc.mc_runs):
        truth = datagen.target_truth(sc, run)
        f = LinearScalarEKF(sc.filter_q, sigma_z)
        f.initialise(datagen.initial_estimate(sc, truth[0], run), datagen.initial_covariance(sc))

        rng = stream_rng(sc.seed, run, Stream.MEASUREMENT)
        noise = sigma_z * rng.standard_normal(sc.steps + 1)

        this = np.empty(sc.steps + 1)
        this[0] = nees_of(truth[0] - f.state, f.covariance)
        for k in range(1, sc.steps + 1):
            f.predict(sc.dt)
            f.axis = k % 2  # alternate px / py
            f.update(float(SCALE * truth[k, f.axis] + noise[k]), own_xy)
            this[k] = nees_of(truth[k] - f.state, f.covariance)
        total += this
        run_means[run] = this.mean()
    return run_means if per_run else total / sc.mc_runs


SMALL = Scenario(mc_runs=120, steps=400)


def test_linear_measurement_filter_is_exactly_consistent():
    """With a linear measurement, NEES must sit at 4.

    Asserted as a confidence interval over per-run means, not as a hand-picked
    tolerance: a review showed the earlier +/-0.06 was ~8x tighter than the real
    sampling error and passed only by luck. The interval also tells the reader
    what this control can and cannot resolve.
    """
    v = _linear_measurement_per_run(Scenario(mc_runs=1200, steps=600), sigma_pos=60.0)
    se = v.std(ddof=1) / np.sqrt(v.size)
    lo, hi = v.mean() - 1.96 * se, v.mean() + 1.96 * se
    assert lo <= 4.0 <= hi, f"control excludes 4: {v.mean():.4f} CI [{lo:.4f}, {hi:.4f}]"
    # Resolution of this control, stated rather than assumed.
    assert 1.96 * se / 4.0 < 0.02, "control should resolve better than 2% at this size"


def test_control_detects_a_sabotaged_prediction(monkeypatch):
    """The control must be *sensitive* to what it claims to validate.

    This is the test the earlier inline implementation could not have passed. A
    15% error in the process-noise assembly is a real model defect; the control
    has to see it, or its exoneration of the shared components means nothing.
    """
    baseline = _linear_measurement_nees(SMALL, 60.0).mean()

    def inflated_q(q, dt):
        return 1.15 * cwna_process_noise(q, dt)

    monkeypatch.setattr("kf2.filters.cwna_process_noise", inflated_q)
    sabotaged = _linear_measurement_nees(SMALL, 60.0).mean()

    assert abs(sabotaged - baseline) > 0.02, (
        f"control is blind to a 15% Q error: {baseline:.4f} -> {sabotaged:.4f}"
    )
    assert sabotaged < baseline, "over-modelled process noise must make NEES conservative"


def test_control_detects_a_sabotaged_update(monkeypatch):
    """Same, for the measurement update: drop the K R K' term from the Joseph
    form, which makes the posterior covariance too small."""
    baseline = _linear_measurement_nees(SMALL, 60.0).mean()

    def broken_update(self, z, own_xy, x_lin=None):
        H = self.jacobian(own_xy, x_lin)
        inn = self.innovation(z, own_xy, x_lin)
        P = self.covariance
        K = (P @ H.T) / inn.S
        self._f.x = self.state + (K * inn.nu).reshape(4)
        IKH = np.eye(4) - K @ H
        broken = IKH @ P @ IKH.T          # the K R K' term deliberately dropped
        self._f.P = 0.5 * (broken + broken.T)
        return inn

    monkeypatch.setattr(BearingsOnlyEKF, "update", broken_update)
    sabotaged = _linear_measurement_nees(SMALL, 60.0).mean()

    assert sabotaged > baseline + 0.05, (
        f"control is blind to a dropped K R K' term: {baseline:.4f} -> {sabotaged:.4f}"
    )


def test_control_shares_the_real_filter():
    """Structural check: the control must not fork the implementation again."""
    assert issubclass(LinearScalarEKF, BearingsOnlyEKF)
    for method in ("predict", "update", "innovation"):
        assert method not in LinearScalarEKF.__dict__, (
            f"LinearScalarEKF overrides {method}; it must inherit the real one"
        )


def test_initial_nees_is_exactly_dim_by_construction():
    """k=0 is a pure check on the sampler: the estimate is drawn from P0."""
    sc = Scenario(mc_runs=3000, steps=1)
    truth0 = np.array([sc.tgt_x0, sc.tgt_y0, sc.tgt_vx0, sc.tgt_vy0])
    P0 = datagen.initial_covariance(sc)
    nees = [
        nees_of(truth0 - datagen.initial_estimate(sc, truth0, run), P0)
        for run in range(sc.mc_runs)
    ]
    assert np.mean(nees) == pytest.approx(4.0, abs=0.12)
    assert np.var(nees) == pytest.approx(8.0, rel=0.15), "chi2(4) has variance 2k"


def test_process_noise_mismatch_is_below_the_resolution_floor():
    """Justifies ``truth_substeps = 50``.

    The Euler-Maruyama truth is analytically 2.98% quieter in position variance
    than the filter's Q. Raising the substep count 40x -- which all but removes
    the mismatch -- must not move consistency by more than the harness can
    resolve, or the mismatch would be confounded with whatever the sweep measures.
    """
    coarse = _linear_measurement_nees(Scenario(mc_runs=400, steps=600), 60.0).mean()
    fine = _linear_measurement_nees(
        Scenario(mc_runs=400, steps=600, truth_substeps=2000), 60.0
    ).mean()
    assert abs(coarse - fine) < 0.15, (coarse, fine)
    assert coarse == pytest.approx(4.0, abs=0.1)
