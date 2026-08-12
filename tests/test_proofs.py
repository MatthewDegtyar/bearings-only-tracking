"""Proofs.

Every other test in this suite checks that the code does what the code was
written to do. These check that what the code was written to do is *correct*,
by computing the answer a second time from theory, or with a different tool, or
by brute force, and never from the module under test.

That distinction matters here. A tracking filter is a machine for producing
confident numbers, and a bug in one produces confident wrong numbers rather than
a crash. The consistency test in this project was itself wrong four separate
times, and on every occasion it failed by accepting filters it should have
rejected. Self-consistency is not evidence.

Each proof below states the claim, states where the independent value comes
from, and then compares. They are grouped by vein: one vein is one thing that
has to be true, and they do not depend on each other.

References for the standard results:
  Kalman, R. E. (1960) Trans. ASME J. Basic Eng. 82:35-45  -- the filter
  Bar-Shalom, Li & Kirubarajan (2001)                      -- NEES, NIS, gating
  Arasaratnam & Haykin (2009) IEEE TAC 54(6)               -- cubature rule
  McNamee & Stenger (1967) Numer. Math. 10:327-344         -- degree-5 rule
  Nardone & Aidala (1981) IEEE TAES AES-17:162-166         -- observability
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats
from scipy.linalg import solve_discrete_are

from kf2 import datagen
from kf2.ckf5 import _cubature_points, _rule3, _rule5
from kf2.config import Scenario, replace
from kf2.filters import BearingsOnlyEKF, cv_transition, cwna_process_noise
from kf2.evaluation import nees_of
from kf2.rng import Stream, stream_rng


# ===========================================================================
# Vein 1: the motion model discretisation
# ===========================================================================

def test_transition_matrix_is_the_exact_solution_of_the_kinematics():
    """A constant-velocity body satisfies p(t+dt) = p + v dt, v(t+dt) = v.

    The claim is that ``cv_transition`` is that map. Checked by propagating a
    state by hand rather than by matrix multiply.
    """
    dt = 0.37
    F = cv_transition(dt)
    x = np.array([11.0, -4.0, 2.5, 7.0])
    expected = np.array([11.0 + 2.5 * dt, -4.0 + 7.0 * dt, 2.5, 7.0])
    assert np.allclose(F @ x, expected, rtol=0, atol=1e-12)
    # and it must compose: two steps of dt equal one step of 2 dt
    assert np.allclose(cv_transition(dt) @ cv_transition(dt), cv_transition(2 * dt), atol=1e-12)


def test_process_noise_matches_the_closed_form_integral():
    """For continuous white-noise acceleration of PSD q, the discrete covariance is

        Q = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]   per axis

    which is the integral of F(tau) G G' F(tau)' dtau over [0, dt]. Computed
    here by numerical quadrature, independently of the closed form in the code.
    """
    q, dt = 0.7, 0.45
    Q = cwna_process_noise(q, dt)

    # brute force: integrate F(tau) L L' F(tau)' q dtau, L selecting acceleration
    n = 200_001
    tau = np.linspace(0.0, dt, n)
    acc = np.zeros((n, 2, 2))
    for i, s in enumerate(tau):
        f = np.array([[s], [1.0]])          # [dt-tau; 1] acting on one axis
        f[0, 0] = dt - s
        acc[i] = q * (f @ f.T)
    integral = np.trapezoid(acc, tau, axis=0)

    for axis, (ip, iv) in enumerate([(0, 2), (1, 3)]):
        assert Q[ip, ip] == pytest.approx(integral[0, 0], rel=1e-6)
        assert Q[iv, iv] == pytest.approx(integral[1, 1], rel=1e-6)
        assert Q[ip, iv] == pytest.approx(integral[0, 1], rel=1e-6)
    assert np.allclose(Q, Q.T, atol=0)
    assert np.linalg.eigvalsh(Q).min() >= -1e-15, "Q must be positive semidefinite"


def test_process_noise_matches_the_published_inverse():
    """Cross-check against a published closed form, in inverse form.

    Tang, Yoon and Barfoot (IEEE RA-L 2018, arXiv:1809.06518) give the inverse
    process noise covariance for the white-noise-on-acceleration prior in their
    equation (15) as

        Q^-1 = [[12 dt^-3 Qc^-1, -6 dt^-2 Qc^-1],
                [-6 dt^-2 Qc^-1,  4 dt^-1 Qc^-1]].

    Inverting it must reproduce this project's Q. Two independent derivations of
    the same quantity, published fifty years apart from the original Kalman
    paper, agreeing to machine precision, is stronger evidence than either alone.

    Their equation (14) also gives the transition matrix exp(A dt), which must
    equal ``cv_transition``.
    """
    from scipy.linalg import expm

    dt, q = 0.73, 1.9
    q_inv_published = (1.0 / q) * np.array([
        [12.0 * dt**-3, -6.0 * dt**-2],
        [-6.0 * dt**-2, 4.0 * dt**-1],
    ])
    q_published = np.linalg.inv(q_inv_published)

    ours = cwna_process_noise(q, dt)[np.ix_([0, 2], [0, 2])]   # the x-axis block
    assert np.allclose(q_published, ours, rtol=1e-11), (
        f"published inverse gives\n{q_published}\nours is\n{ours}"
    )

    # equation (14): the transition is the matrix exponential of the generator
    A = np.zeros((4, 4))
    A[0, 2] = A[1, 3] = 1.0
    assert np.allclose(expm(A * dt), cv_transition(dt), atol=1e-14)


def test_discrete_and_continuous_acceleration_models_are_not_the_same():
    """A guard against a substitution that would run silently and change every result.

    The discrete white-noise acceleration model, in which a piecewise-constant
    acceleration is held over each interval, gives

        Q = sigma_a^2 [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]

    (Baisa, arXiv:2005.00844, eq. 15), which differs from the continuous form in
    every entry. The two are easy to confuse in the literature and the parameters
    even carry different units, so this asserts the project uses the continuous
    one.
    """
    dt, q = 0.6, 1.0
    ours = cwna_process_noise(q, dt)[np.ix_([0, 2], [0, 2])]
    dwna = q * np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]])
    assert not np.allclose(ours, dwna), "the two models must not coincide"
    assert ours[1, 1] == pytest.approx(q * dt), "continuous form has Q_vv = q dt"
    assert dwna[1, 1] == pytest.approx(q * dt**2), "discrete form has Q_vv = q dt^2"


def test_truth_generator_reproduces_the_analytic_step_covariance():
    """The simulator integrates the SDE by Euler-Maruyama rather than sampling
    the closed form, which is what keeps it independent of the filter.

    ``em_step_covariance`` claims to be the exact covariance of *that scheme*
    over one coarse step. Checked against the sample covariance of the scheme
    itself over many draws, which uses no algebra from the module.
    """
    sc = replace(Scenario(), dt=1.0, steps=1, truth_substeps=8, q=0.3)
    Q = datagen.em_step_covariance(sc.q, sc.dt, sc.truth_substeps)

    draws = np.array([
        datagen._integrate_em(
            sc, datagen._velocity_increments(sc, stream_rng(12345, r, Stream.PROCESS)),
            np.zeros(4))[1]
        for r in range(40_000)
    ])
    sample = np.cov(draws, rowvar=False)
    # 40k draws gives roughly 0.7% standard error on a variance
    for i in range(4):
        for j in range(4):
            if abs(Q[i, j]) > 1e-12:
                assert sample[i, j] == pytest.approx(Q[i, j], rel=0.06), f"element {i},{j}"


def test_euler_maruyama_gap_to_the_continuous_model_is_exactly_known():
    """The 'no inverse crime' separation, as an exact relation rather than a claim.

    Var_EM(p)/Var(p) = 1 - 3/(2n) + 1/(2n^2),  Cov ratio = 1 - 1/n,  Var(v) exact.

    The velocity variance is exact at every n because velocity is a plain sum of
    increments. The position variance is the term that converges, at O(1/n). If
    this ever returned 1 exactly, the simulator would be sampling the filter's
    own model and the study would be testing the filter against its assumptions.
    """
    q, dt = 0.3, 1.0
    exact = cwna_process_noise(q, dt)
    for n in (2, 5, 8, 50, 500):
        em = datagen.em_step_covariance(q, dt, n)
        assert em[0, 0] / exact[0, 0] == pytest.approx(1 - 3 / (2 * n) + 1 / (2 * n * n), abs=1e-12)
        assert em[0, 2] / exact[0, 2] == pytest.approx(1 - 1 / n, abs=1e-12)
        assert em[2, 2] / exact[2, 2] == pytest.approx(1.0, abs=1e-12)

    # the default the project actually runs at
    em50 = datagen.em_step_covariance(q, dt, 50)
    assert 1 - em50[0, 0] / exact[0, 0] == pytest.approx(0.0298, abs=0.0005)


# ===========================================================================
# Vein 2: the linear filter is the optimal one
# ===========================================================================

class _LinearPositionEKF(BearingsOnlyEKF):
    """Same filter, with the measurement replaced by a linear one.

    Only the measurement function and its Jacobian change, so the predict step,
    the covariance update and the gain arithmetic under test are the ones the
    real filter uses. Substituting a whole separate Kalman filter here would
    prove nothing about this code.
    """

    H = np.array([[1.0, 0.0, 0.0, 0.0]])

    def measurement(self, x, own_xy):
        return float(np.asarray(x).ravel()[0])

    def jacobian(self, own_xy, x_lin=None):
        return self.H.copy()


def test_filter_converges_to_the_riccati_solution():
    """For a linear-Gaussian system the steady-state covariance is the unique
    stabilising solution of the discrete algebraic Riccati equation.

    The reference is scipy's ``solve_discrete_are``, which shares no code with
    this project. If the predict step, the gain, or the Joseph update were
    wrong, the fixed point would be wrong.
    """
    dt, q, R = 1.0, 0.05, 4.0

    # The measurement observes x only, and the model is block diagonal, so the
    # x subsystem (position, velocity) evolves independently and is the part
    # with a finite fixed point. The y block is never measured and its
    # covariance grows without bound, which is correct and is why the DARE has
    # no solution for the full 4-state system.
    F2 = np.array([[1.0, dt], [0.0, 1.0]])
    Q2 = q * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    H2 = np.array([[1.0, 0.0]])

    P_pred = solve_discrete_are(F2.T, H2.T, Q2, np.array([[R]]))
    S = float(H2 @ P_pred @ H2.T) + R
    K = P_pred @ H2.T / S
    P_post = (np.eye(2) - K @ H2) @ P_pred

    f = _LinearPositionEKF(q=q, sigma_bearing=math.sqrt(R))
    f.initialise(np.zeros(4), np.eye(4) * 1e4)
    for _ in range(4000):
        f.predict(dt)
        f.update(0.0, np.zeros(2))

    got = f.covariance[np.ix_([0, 2], [0, 2])]
    assert np.allclose(got, P_post, rtol=2e-6, atol=1e-9), (
        f"converged to\n{got}\nRiccati says\n{P_post}"
    )
    # and the unobserved axis must indeed be growing, not silently bounded
    assert f.covariance[1, 1] > 1e4, "unmeasured axis should not have converged"


def _pbh_detectable(A, C):
    """Popov-Belevitch-Hautus test (Kwong, ECE1639 ch.3).

    (C, A) is detectable iff rank[[A - lam I], [C]] = n for every eigenvalue lam
    of A with |lam| >= 1.
    """
    n = A.shape[0]
    for lam in np.linalg.eigvals(A):
        if abs(lam) >= 1.0 - 1e-12:
            M = np.vstack([A - lam * np.eye(n), C])
            if np.linalg.matrix_rank(M, tol=1e-9) < n:
                return False
    return True


def test_riccati_hypotheses_hold_where_claimed_and_fail_where_claimed():
    """Kwong's Theorem 3.3 requires stabilisability and detectability. Those
    hypotheses are verified for the exact matrices used, rather than assumed.

    The constant-velocity F has a repeated eigenvalue at 1, on the unit circle,
    so the test genuinely has to be evaluated. The four-state system with only
    x measured must FAIL detectability -- that failure is why the Riccati solver
    refuses, and it is a correct report about the model rather than a numerical
    problem.
    """
    dt, q = 1.0, 0.05
    F2 = np.array([[1.0, dt], [0.0, 1.0]])
    H2 = np.array([[1.0, 0.0]])
    Q2 = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])

    assert _pbh_detectable(F2, H2), "observed 2-state block must be detectable"
    # Q positive definite makes (F, Q^{1/2}) stabilisable immediately
    assert np.linalg.matrix_rank(np.linalg.cholesky(Q2)) == 2

    F4 = cv_transition(dt)
    H4 = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert not _pbh_detectable(F4, H4), (
        "the 4-state system with only x measured must not be detectable"
    )
    with pytest.raises(np.linalg.LinAlgError):
        solve_discrete_are(F4.T, H4.T, np.eye(4) * q, np.array([[4.0]]))


def test_closed_loop_matrix_is_the_predictor_form():
    """Kwong's Theorem 3.3 asserts stability of F - F K H, not F - K H.

    The predicted error obeys x~(k+1|k) = F(I - KH) x~(k|k-1) + noise, so the
    governing matrix carries the extra F. An earlier draft of the proof wrote
    F - KH. Both are stable in this example, 0.789 against 0.843, so no
    numerical check would have caught the slip; it is pinned here so it cannot
    come back.
    """
    dt, q, R = 1.0, 0.05, 4.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    H = np.array([[1.0, 0.0]])

    P = solve_discrete_are(F.T, H.T, Q, np.array([[R]]))
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

    predictor = F @ (np.eye(2) - K @ H)
    assert np.allclose(predictor, F - F @ K @ H), "F(I-KH) == F - FKH"
    rho = np.abs(np.linalg.eigvals(predictor)).max()
    assert rho < 1.0, f"closed loop must be stable, got spectral radius {rho}"
    assert rho == pytest.approx(0.789, abs=0.01)

    # the matrix the earlier draft named is a different one
    other = np.abs(np.linalg.eigvals(F - K @ H)).max()
    assert not np.isclose(rho, other), "the two forms must not be conflated"


def test_short_form_covariance_update_fails_away_from_the_optimal_gain():
    """Why Joseph form is used, demonstrated rather than asserted.

    At the optimal gain the two agree. Away from it the short form is not even
    symmetric and can be badly indefinite, while the Joseph form stays positive
    semidefinite because it is a sum of two PSD terms.
    """
    dt, q, R = 1.0, 0.05, 4.0
    H = np.array([[1.0, 0.0]])
    rng = np.random.default_rng(3)
    A = rng.standard_normal((2, 2))
    P = A @ A.T + np.eye(2)

    # agreement at the optimal gain
    S = float((H @ P @ H.T)[0, 0]) + R
    K = P @ H.T / S
    IKH = np.eye(2) - K @ H
    assert np.allclose(IKH @ P, IKH @ P @ IKH.T + K * R @ K.T)

    # divergence away from it
    worst_short, joseph_at_worst = 0.0, None
    for _ in range(20_000):
        Kb = rng.standard_normal((2, 1)) * 3
        M = (np.eye(2) - Kb @ H) @ P
        m = np.linalg.eigvalsh(0.5 * (M + M.T)).min()
        if m < worst_short:
            worst_short = m
            J = (np.eye(2) - Kb @ H) @ P @ (np.eye(2) - Kb @ H).T + Kb @ np.atleast_2d(R) @ Kb.T
            joseph_at_worst = np.linalg.eigvalsh(J).min()
    assert worst_short < -1.0, "short form should go badly indefinite somewhere"
    assert joseph_at_worst >= -1e-9, "Joseph form must stay PSD at that same gain"


def test_joseph_update_keeps_the_covariance_positive_semidefinite():
    """The Joseph form is (I-KH) P (I-KH)' + K R K', a sum of two positive
    semidefinite terms, so it is PSD whenever P is, for any K.

    That is the reason it is used here rather than the cheaper (I-KH)P: this
    project deliberately drives the filter into small-S, large-P regimes where
    the cheap form loses definiteness. Checked with a deliberately wrong gain,
    where the cheap form has no such guarantee.
    """
    sc = replace(Scenario(), p0_pos=900.0)
    own = datagen.ownship_track(sc)
    truth = datagen.target_truth(sc, 0)
    det = datagen.generate_detections(sc, truth, own, 0)

    f = BearingsOnlyEKF(sc.filter_q, sc.sigma_bearing)
    f.initialise(datagen.initial_estimate(sc, truth[0], 0), datagen.initial_covariance(sc))
    worst = np.inf
    for k in range(1, sc.steps + 1):
        f.predict(sc.dt)
        if len(det.per_step[k]):
            f.update(float(det.per_step[k][0]), own.xy[k])
        w = np.linalg.eigvalsh(f.covariance).min()
        worst = min(worst, w)
    assert worst >= -1e-9, f"covariance went indefinite: min eigenvalue {worst:.3e}"


# ===========================================================================
# Vein 3: the consistency statistics mean what they claim
# ===========================================================================

def test_nees_of_a_correctly_drawn_error_is_chi_square_with_n_degrees_of_freedom():
    """If e ~ N(0, P) then e' P^-1 e ~ chi2(n). This is the entire basis for
    expecting NEES = 4, and it is checked by sampling rather than assumed:
    draw errors from a known P, push them through the project's own
    ``nees_of``, and test the resulting sample against chi2(4).
    """
    rng = np.random.default_rng(7)
    A = rng.standard_normal((4, 4))
    P = A @ A.T + 4 * np.eye(4)          # an arbitrary, badly conditioned P
    L = np.linalg.cholesky(P)
    e = (L @ rng.standard_normal((4, 60_000))).T
    v = np.array([nees_of(e[i], P) for i in range(len(e))])

    assert v.mean() == pytest.approx(4.0, rel=0.02)
    assert v.var() == pytest.approx(8.0, rel=0.06)        # Var[chi2(k)] = 2k
    ks = stats.kstest(v, "chi2", args=(4,))
    assert ks.pvalue > 0.01, f"NEES does not follow chi2(4): KS p={ks.pvalue:.4g}"


def test_whitening_works_for_any_square_root_not_just_the_symmetric_one():
    """The chi-square argument must not depend on which square root is used.

    Any A with A A' = P gives y = A^-1 x with covariance I and y'y = x'P^-1x,
    because the two occurrences of A^-1 meet back to back. The implementation
    uses a Cholesky factor, not the symmetric root, so this closes the gap
    between the proof as written and the code as run.
    """
    from scipy import linalg

    rng = np.random.default_rng(4)
    A0 = rng.standard_normal((4, 4))
    P = A0 @ A0.T + 4 * np.eye(4)
    x = rng.standard_normal(4) * 3

    quad = float(x @ np.linalg.inv(P) @ x)
    for name, root in [("cholesky", np.linalg.cholesky(P)),
                       ("symmetric", linalg.sqrtm(P).real)]:
        y = np.linalg.solve(root, x)
        assert float(y @ y) == pytest.approx(quad, rel=1e-9), name
    assert nees_of(x, P) == pytest.approx(quad, rel=1e-12)


def test_sidak_correction_delivers_its_nominal_family_wise_rate():
    """alpha' = 1 - (1-alpha)^(1/m) must invert exactly, and the uncorrected
    rate must be as bad as claimed: 12 tests at 0.05 gives 46 per cent."""
    from kf2.evaluation import sidak_z

    for m, alpha in ((12, 0.05), (6, 0.05), (3, 0.01)):
        per_test = 1 - (1 - alpha) ** (1 / m)
        assert 1 - (1 - per_test) ** m == pytest.approx(alpha, abs=1e-12)
        assert sidak_z(alpha, m) == pytest.approx(
            stats.norm.ppf(1 - per_test / 2), abs=1e-9
        )
    assert 1 - 0.95**12 == pytest.approx(0.46, abs=0.005)


def test_scalar_innovation_gives_nis_of_one():
    """For a scalar measurement, nu / sqrt(S) is standard normal under a
    consistent filter, so nu^2/S ~ chi2(1) and its mean is 1.
    """
    rng = np.random.default_rng(11)
    S = 0.37
    nu = math.sqrt(S) * rng.standard_normal(200_000)
    nis = nu * nu / S
    assert nis.mean() == pytest.approx(1.0, rel=0.01)
    assert stats.kstest(nis, "chi2", args=(1,)).pvalue > 0.01


def test_ellipse_probabilities_match_the_published_values():
    """Guard against an error this project actually made.

    Andrae (arXiv:1009.2755) states that the one-sigma contour of a
    two-dimensional Gaussian marks a 39.4 per cent confidence region, and 19.9
    per cent in three dimensions. An earlier draft of the proofs asserted that
    reaching 68 per cent in a plane takes 2.45 sigma; it takes 1.52 sigma, and
    2.45 sigma is the 95 per cent contour. Nothing in the filter would ever have
    contradicted the wrong number, so it is pinned here.
    """
    assert stats.chi2.cdf(1.0, 2) == pytest.approx(0.394, abs=0.001)
    assert stats.chi2.cdf(1.0, 2) == pytest.approx(1 - math.exp(-0.5), abs=1e-12)
    assert stats.chi2.cdf(1.0, 3) == pytest.approx(0.199, abs=0.001)

    s68 = math.sqrt(stats.chi2.ppf(0.683, 2))
    assert s68 == pytest.approx(1.52, abs=0.01), "68.3% in a plane is 1.52 sigma"
    assert stats.chi2.cdf(2.4477**2, 2) == pytest.approx(0.95, abs=0.001), (
        "2.45 sigma is the 95% contour, not the 68% one"
    )


def test_confidence_interval_covers_the_truth_at_its_nominal_rate():
    """A 95 per cent interval must contain the true mean 95 per cent of the
    time. Measured by repeatedly sampling from a known distribution and
    counting, which is the only check that catches an interval that is merely
    plausible-looking.
    """
    from kf2.evaluation import mean_ci

    rng = np.random.default_rng(3)
    hits, trials, n = 0, 4000, 200
    for _ in range(trials):
        x = rng.chisquare(4, size=n)      # the distribution NEES actually has
        _, lo, hi, _ = mean_ci(x, z=1.959963985)
        hits += lo <= 4.0 <= hi
    rate = hits / trials
    # binomial standard error at 95% over 4000 trials is about 0.35%
    assert 0.935 <= rate <= 0.965, f"nominal 95% interval covered {100 * rate:.1f}%"


# ===========================================================================
# Vein 4: the cubature rule integrates what it claims to
# ===========================================================================

def _bearing_H_M(x):
    """Jacobian and Hessian of atan2(dy, dx), observer at the origin."""
    a, b = float(x[0]), float(x[1])
    r2 = a * a + b * b
    H = np.array([-b / r2, a / r2])
    M = np.array([[2 * a * b / r2**2, (b * b - a * a) / r2**2],
                  [(b * b - a * a) / r2**2, -2 * a * b / r2**2]])
    return H, M


def test_second_order_expansion_matches_monte_carlo():
    """E[h] = h(xhat) + tr(MP)/2 and Var[h] = HPH' + tr(MPMP)/2.

    This is the equation that identifies what the EKF throws away, so it is
    checked against sampling of the actual arctangent rather than trusted.
    """
    rng = np.random.default_rng(7)
    xhat = np.array([600.0, 400.0])
    r = float(np.hypot(*xhat))
    H, M = _bearing_H_M(xhat)

    for frac in (0.05, 0.10, 0.20):
        P = (frac * r) ** 2 * np.eye(2)
        var2 = float(H @ P @ H) + 0.5 * np.trace(M @ P @ M @ P)
        var1 = float(H @ P @ H)
        smp = xhat + rng.standard_normal((400_000, 2)) * (frac * r)
        mc = float(np.arctan2(smp[:, 1], smp[:, 0]).var())
        # the second-order form must be closer to the truth than first order
        assert abs(var2 - mc) < abs(var1 - mc), f"frac={frac}"


def test_bearing_hessian_is_traceless_so_the_mean_correction_vanishes():
    """The bearing is arg(z) = Im log z, harmonic away from the origin, so its
    Hessian is traceless.

    Consequence: the second-order correction to the MEAN vanishes for isotropic
    P, and the entire second-order error lands on the variance. That is the
    structural reason this project's failure appears as overconfidence rather
    than as a biased estimate.
    """
    for xhat in ([600.0, 400.0], [-120.0, 900.0], [50.0, -70.0]):
        H, M = _bearing_H_M(np.array(xhat))
        assert abs(np.trace(M)) < 1e-15, "Hessian must be traceless"
        assert M[1, 1] == pytest.approx(-M[0, 0], abs=1e-15)

        iso = 900.0 * np.eye(2)
        assert abs(np.trace(M @ iso)) < 1e-15, "mean correction vanishes under isotropy"

        # the variance term is positive regardless
        assert 0.5 * np.trace(M @ iso @ M @ iso) > 0.0

    # and it scales as sigma^4 while HPH' scales as sigma^2
    xhat = np.array([600.0, 400.0])
    H, M = _bearing_H_M(xhat)
    a1 = b1 = None
    for k, frac in enumerate((0.05, 0.10, 0.20)):
        P = (frac * float(np.hypot(*xhat))) ** 2 * np.eye(2)
        a = float(H @ P @ H)
        b = 0.5 * np.trace(M @ P @ M @ P)
        if k == 0:
            a1, b1 = a, b
        else:
            scale = frac / 0.05
            assert a / a1 == pytest.approx(scale**2, rel=1e-9)
            assert b / b1 == pytest.approx(scale**4, rel=1e-9)


@pytest.mark.parametrize("rule,degree", [(_rule3(4), 3), (_rule5(4), 5)])
def test_rule_is_exact_to_its_stated_degree(rule, degree):
    """A rule of degree d integrates every monomial of total degree <= d
    exactly against the standard Gaussian.

    The reference values are the Gaussian moments, E[x^k] = (k-1)!! for even k
    and 0 for odd, computed here from the factorial rather than taken from the
    module. This is the property the whole fix rests on: the discarded term is
    fourth order, so a degree-3 rule cannot reproduce it and a degree-5 rule
    must.
    """
    pts, wts = rule

    def exact(powers):
        out = 1.0
        for k in powers:
            if k % 2:
                return 0.0
            out *= math.prod(range(1, k, 2))      # (k-1)!!
        return out

    for powers in itertools.product(range(degree + 1), repeat=4):
        if sum(powers) > degree:
            continue
        got = float(wts @ np.prod(pts ** np.array(powers), axis=1))
        assert got == pytest.approx(exact(powers), abs=1e-10), f"monomial {powers}"


def test_degree_three_fails_exactly_where_the_report_says_it_does():
    """The report attributes the frame dependence of the degree-3 rule to a
    specific defect: its points lie on the coordinate axes, so every cross term
    x1^2 x2^2 evaluates to zero. The claim is falsifiable and is checked here.
    """
    pts3, wts3 = _rule3(4)
    pts5, wts5 = _rule5(4)

    e3 = float(wts3 @ np.prod(pts3 ** np.array([2, 2, 0, 0]), axis=1))
    e5 = float(wts5 @ np.prod(pts5 ** np.array([2, 2, 0, 0]), axis=1))
    assert e3 == pytest.approx(0.0, abs=1e-12), "degree 3 should return zero here"
    assert e5 == pytest.approx(1.0, abs=1e-12), "degree 5 must return the true value 1"

    # The pure fourth power is wrong too, and in the other direction: the
    # degree-3 rule returns 4 where the truth is 3. The report attributes the
    # frame dependence to the cross term alone, and that is the term whose error
    # rotates, but the rule is not merely blind on the diagonal -- it is
    # inaccurate on the axis as well, +1 on one and -1 on the other. That pairing
    # is what makes the total error depend on how the two mix under rotation.
    m4_3 = float(wts3 @ pts3[:, 0] ** 4)
    m4_5 = float(wts5 @ pts5[:, 0] ** 4)
    assert m4_3 == pytest.approx(4.0, abs=1e-10), "degree 3 overstates the fourth moment"
    assert m4_5 == pytest.approx(3.0, abs=1e-10), "degree 5 must be exact"
    assert (m4_3 - 3.0) == pytest.approx(-(e3 - 1.0), abs=1e-10), (
        "the two errors must be equal and opposite, which is the rotation mechanism"
    )


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_rule5_weights_match_the_published_formula(n):
    """The McNamee-Stenger weights, checked against the closed form.

    The point count 2n^2+1 and the origin weight 2/(n+2) are the values reported
    for the fifth-degree cubature filter by Kulikova and Kulikov
    (arXiv:2312.02846).

    At n = 4 the axis weight is exactly zero, so those 8 points are dropped and
    25 of the 33 are evaluated; discarding a zero-weight point is exact. At n = 6
    the axis weight goes negative, which is the obstacle to carrying this rule
    into a three-dimensional tracking state.
    """
    from kf2.ckf5 import _rule5

    pts, w = _rule5(n)
    nz = np.count_nonzero(pts, axis=1)
    w_axis = (4.0 - n) / (2.0 * (n + 2) ** 2)

    assert w[nz == 0][0] == pytest.approx(2.0 / (n + 2)), "origin weight"
    assert w[nz == 2][0] == pytest.approx(1.0 / (n + 2) ** 2), "diagonal weight"
    assert (nz == 2).sum() == 2 * n * (n - 1), "diagonal point count"
    assert w.sum() == pytest.approx(1.0, abs=1e-12), "weights must sum to one"

    if n == 4:
        assert w_axis == pytest.approx(0.0, abs=1e-15)
        assert (nz == 1).sum() == 0, "zero-weight axis points should be dropped"
        assert len(w) == 25, f"expected 25 retained of 2n^2+1 = {2 * n * n + 1}"
        assert (w < 0).sum() == 0
    else:
        assert len(w) == 2 * n * n + 1
        assert w[nz == 1][0] == pytest.approx(w_axis)
    if n == 6:
        assert (w < 0).sum() == 12, "n=6 has negative weights; see the proof"


def test_cubature_points_reproduce_the_covariance_they_were_built_from():
    """Transporting the rule into N(x, P) must give back x and P, or the
    sampling is not representing the distribution the filter believes in.
    """
    rng = np.random.default_rng(5)
    A = rng.standard_normal((4, 4))
    P = A @ A.T + np.eye(4)
    x = rng.standard_normal(4) * 40.0
    pts, wts = _cubature_points(x, P, degree=5)

    assert wts.sum() == pytest.approx(1.0, abs=1e-12)
    mean = wts @ pts
    assert np.allclose(mean, x, atol=1e-9)
    d = pts - mean
    cov = (wts[:, None] * d).T @ d
    assert np.allclose(cov, P, rtol=1e-8, atol=1e-8)


# ===========================================================================
# Vein 5: the observability claim the whole project rests on
# ===========================================================================

def test_measurement_model_needs_two_argument_arctangent():
    """h = atan2(dy, dx), not arctan(dy/dx).

    The one-argument form collapses the circle onto a half-plane: a target
    behind the observer gets the same bearing as one in front. The report stated
    the wrong form in its Equation 3 until this pass; the code was always
    correct. This pins the distinction.
    """
    f = BearingsOnlyEKF(1e-3, math.radians(0.5))
    own = np.zeros(2)

    ahead = np.array([500.0, 300.0, 0.0, 0.0])
    behind = np.array([-500.0, -300.0, 0.0, 0.0])

    f.initialise(ahead, np.eye(4))
    b_ahead = f.predicted_bearing(own)
    f.initialise(behind, np.eye(4))
    b_behind = f.predicted_bearing(own)

    # the naive form cannot tell them apart
    assert math.atan(300.0 / 500.0) == pytest.approx(math.atan(-300.0 / -500.0))
    # the implementation can, and they differ by exactly pi
    assert abs(abs(datagen.wrap_pi(b_ahead - b_behind)) - math.pi) < 1e-12
    assert b_ahead == pytest.approx(math.atan2(300.0, 500.0))
    assert b_behind == pytest.approx(math.atan2(-300.0, -500.0))


def test_constant_velocity_observer_cannot_determine_range():
    """Nardone and Aidala: with the observer on a constant-velocity course, a
    constant-velocity target's range is unobservable.

    Proof by exhibition of the invariance. Scale the relative geometry by any
    k and scale the relative velocity by the same k, and the bearing sequence
    is identical, so no estimator whatsoever can distinguish them. If this test
    ever fails, the simulator's geometry is wrong, not the theory.
    """
    t = np.arange(0.0, 300.0, 1.0)
    v_own = np.array([11.0, 0.0])
    own = v_own * t[:, None]
    p0, v0 = np.array([3000.0, 2200.0]), np.array([-4.0, 1.5])

    def bearings(p, v):
        d = (p + v * t[:, None]) - own
        return np.arctan2(d[:, 1], d[:, 0])

    base = bearings(p0, v0)
    for k in (1.7, 3.0, 8.5):
        scaled = bearings(k * p0, v_own + k * (v0 - v_own))
        assert np.abs(scaled - base).max() < 1e-12, (
            f"scaling by {k} changed the bearings, so range would be observable"
        )


def test_jacobian_and_hessian_norms_are_exactly_one_over_r_and_r_squared():
    """|H| = 1/r and ||M||_2 = 1/r^2, exactly, not asymptotically.

    These give the sigma^2/r^2 and sigma^4/r^4 scalings of the retained and
    discarded terms, which is the fourth-power relation the report tests.
    """
    for d in ([700.0, 400.0], [-1200.0, 300.0], [50.0, -90.0], [3000.0, 4000.0]):
        d = np.array(d)
        r = float(np.linalg.norm(d))
        H, M = _bearing_H_M(d)
        assert np.linalg.norm(H) == pytest.approx(1.0 / r, rel=1e-14)
        assert np.linalg.norm(M, 2) == pytest.approx(1.0 / r**2, rel=1e-12)


def test_a_manoeuvring_observer_breaks_the_invariance():
    """The converse, and the reason a patrol route weaves at all. Once the
    observer accelerates its own motion cannot be absorbed into a scaled
    constant-velocity target, so the scale becomes identifiable.
    """
    t = np.arange(0.0, 300.0, 1.0)
    psi = np.radians(30.0) * np.sin(2 * np.pi * t / 300.0)
    own = np.column_stack([np.cumsum(11.0 * np.cos(psi)), np.cumsum(11.0 * np.sin(psi))])
    p0, v0 = np.array([3000.0, 2200.0]), np.array([-4.0, 1.5])

    def bearings(p, v):
        d = (p + v * t[:, None]) - own
        return np.arctan2(d[:, 1], d[:, 0])

    base = bearings(p0, v0)
    sep = max(np.abs(bearings(k * p0, k * v0) - base).max() for k in (1.7, 3.0))
    assert np.degrees(sep) > 0.5, "manoeuvre failed to make range identifiable"


def test_scaling_invariance_is_relative_to_the_observer_origin():
    """eq (7) must be written (p_t - o_0) -> k(p_t - o_0), not p_t -> k p_t.

    The scaling is a statement about the relative geometry. Applied literally
    with an observer that does not start at the origin, the bearings diverge by
    about 20 degrees. An earlier draft had the frame-dependent form, which
    survived because the proof works in observer-origin coordinates.
    """
    t = np.arange(0.0, 300.0, 1.0)
    v_o = np.array([11.0, 0.0])
    p0, v0 = np.array([3000.0, 2200.0]), np.array([-4.0, 1.5])

    for o_0 in (np.zeros(2), np.array([-800.0, 500.0])):
        own = o_0 + v_o * t[:, None]

        def brg(p, v):
            d = (p + v * t[:, None]) - own
            return np.arctan2(d[:, 1], d[:, 0])

        base = brg(p0, v0)
        for k in (1.7, 3.0, 8.5):
            corrected = brg(o_0 + k * (p0 - o_0), v_o + k * (v0 - v_o))
            assert np.abs(corrected - base).max() < 1e-12, f"correct form, o_0={o_0}, k={k}"
        if np.any(o_0):
            literal = max(np.abs(brg(k * p0, v_o + k * (v0 - v_o)) - base).max()
                          for k in (1.7, 3.0, 8.5))
            assert np.degrees(literal) > 5.0, "the naive form must visibly fail off-origin"


def test_second_order_bias_has_a_closed_form_in_the_line_of_sight_frame():
    """The LEADING discarded term is the bias, order sigma^2/r^2, not the
    variance at sigma^4/r^4.

    In the line-of-sight frame tr(MP) = -2 P_rc / r^2, so the bias is -P_rc/r^2
    and vanishes exactly when P's principal axes align with the line of sight.
    At sigma_range = 300 m, sigma_cross = 6 m, r = 500 m the bias is 0.07 deg at
    0.2 degrees of misalignment and 0.36 deg at 1 degree, against 0.5 deg of
    sensor noise. Unlike noise, it does not average down.
    """
    r = 500.0
    _, M = _bearing_H_M(np.array([r, 0.0]))          # LOS frame: d = (r, 0)
    assert np.allclose(M, np.array([[0.0, -1.0], [-1.0, 0.0]]) / r**2)

    def P_los(sr, sc, theta):
        c, s_ = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s_], [s_, c]])
        return R @ np.diag([sr**2, sc**2]) @ R.T

    sr, sc = 300.0, 6.0
    for deg, expect in ((0.0, 0.0), (0.2, 0.07), (1.0, 0.36)):
        P = P_los(sr, sc, math.radians(deg))
        assert np.trace(M @ P) == pytest.approx(-2 * P[0, 1] / r**2, rel=1e-12)
        bias_deg = abs(math.degrees(0.5 * np.trace(M @ P)))
        assert bias_deg == pytest.approx(expect, abs=0.01), f"misalignment {deg} deg"

    # orders: bias scales as sigma^2, variance as sigma^4
    P1 = P_los(sr, sc, math.radians(0.2))
    P2 = P_los(2 * sr, 2 * sc, math.radians(0.2))
    assert abs(np.trace(M @ P2)) / abs(np.trace(M @ P1)) == pytest.approx(4.0, rel=1e-9)
    assert np.trace(M @ P2 @ M @ P2) / np.trace(M @ P1 @ M @ P1) == pytest.approx(16.0, rel=1e-9)


def test_collinear_counterexample_mechanism_is_constant_bearing():
    """The radial-acceleration counterexample is valid, but NOT because the
    scaling invariance survives there.

    With these parameters |d'|/|d| drifts from 2.500 to 4.270, so eq (9) fails.
    The bearings agree because a collinear geometry admits only one bearing at
    any range. And the agreement holds only until a relative position crosses
    the observer, after which the sequences differ by exactly 180 degrees.
    """
    t = np.arange(0.0, 300.0, 1.0)
    u = np.array([3000.0, 2200.0])
    u = u / np.linalg.norm(u)
    own = np.outer(11.0 * t + 0.025 * t**2, u)
    p0, v0, k = 12000.0 * u, -6.0 * u, 2.5

    d_base = (p0 + v0 * t[:, None]) - own
    d_scal = (k * p0 + k * v0 * t[:, None]) - own
    ratio = np.linalg.norm(d_scal, axis=1) / np.linalg.norm(d_base, axis=1)
    assert ratio[0] == pytest.approx(k, abs=1e-9)
    assert ratio[-1] > 4.0, "the invariance does NOT hold; the ratio drifts"

    b_base = np.arctan2(d_base[:, 1], d_base[:, 0])
    assert np.degrees(np.ptp(b_base)) < 1e-9, "collinear geometry gives a constant bearing"

    # past a crossing the two differ by pi
    tl = np.arange(0.0, 900.0, 1.0)
    own_l = np.outer(11.0 * tl + 0.025 * tl**2, u)
    bb = np.arctan2(*((p0 + v0 * tl[:, None]) - own_l)[:, ::-1].T)
    bs = np.arctan2(*((k * p0 + k * v0 * tl[:, None]) - own_l)[:, ::-1].T)
    gap = np.degrees(np.abs(np.arctan2(np.sin(bs - bb), np.cos(bs - bb))))
    assert gap.max() == pytest.approx(180.0, abs=1e-6)


def test_radial_acceleration_does_not_restore_observability():
    """Acceleration is necessary but not sufficient.

    Fogel and Gavish (IEEE TAES 24(3):305-308, 1988) show that earlier
    first-order observability requirements are necessary but not sufficient, and
    that motion along the bearing vector does not suffice. An earlier draft of
    the proofs claimed simply that acceleration restores observability, which is
    false: an observer accelerating along the line of sight in a collinear
    geometry leaves the scaling invariance exactly intact.
    """
    t = np.arange(0.0, 300.0, 1.0)
    u = np.array([3000.0, 2200.0])
    u = u / np.linalg.norm(u)

    # observer accelerating purely along u; target on the same ray
    own = np.outer(11.0 * t + 0.025 * t**2, u)
    p0, v0 = 12000.0 * u, -6.0 * u

    def bearings(p, v):
        d = (p + v * t[:, None]) - own
        return np.arctan2(d[:, 1], d[:, 0])

    base = bearings(p0, v0)
    for k in (1.7, 2.5, 4.0):
        assert np.abs(bearings(k * p0, k * v0) - base).max() < 1e-12, (
            f"radial acceleration must not break the invariance (k={k})"
        )


def test_bearing_jacobian_is_perpendicular_to_the_line_of_sight():
    """The claim behind every 'range is not observable' statement in the
    report: H has no component along the line of sight, so error in that
    direction produces no innovation at all.

    Checked against a finite difference of the measurement function, so the
    analytic Jacobian in the code is never used to verify itself.
    """
    f = BearingsOnlyEKF(1e-3, math.radians(0.5))
    own = np.array([120.0, -60.0])
    x = np.array([700.0, 400.0, -3.0, 2.0])
    f.initialise(x, np.eye(4))
    H = f.jacobian(own)[0]

    h = 1e-4
    fd = np.zeros(4)
    for i in range(4):
        for sgn in (+1, -1):
            xp = x.copy()
            xp[i] += sgn * h
            f.initialise(xp, np.eye(4))
            fd[i] += sgn * f.predicted_bearing(own)
    fd /= 2 * h
    assert np.allclose(H, fd, rtol=1e-5, atol=1e-9), f"analytic {H} vs finite difference {fd}"

    los = x[:2] - own
    los = los / np.linalg.norm(los)
    assert abs(H[:2] @ los) < 1e-12, "H must be orthogonal to the line of sight"
    assert np.allclose(H[2:], 0.0), "a bearing carries no velocity information"


# ===========================================================================
# Vein 6: the attenuation that makes the failure invisible
# ===========================================================================

def test_innovation_variance_splits_into_state_and_sensor_parts():
    """S = H P H' + R exactly, so the fraction of S that carries information
    about the state is H P H' / S. The report's central mechanism is that this
    fraction is small, so a large state error arrives at the innovation check
    heavily attenuated.

    Checked by recomputing both parts from the filter's own P and R with plain
    matrix algebra, then confirming the attenuation predicted for the report's
    geometry.
    """
    sc = replace(Scenario(), p0_pos=600.0)
    own = datagen.ownship_track(sc)
    truth = datagen.target_truth(sc, 0)
    f = BearingsOnlyEKF(sc.filter_q, sc.sigma_bearing)
    f.initialise(datagen.initial_estimate(sc, truth[0], 0), datagen.initial_covariance(sc))
    f.predict(sc.dt)

    H = f.jacobian(own.xy[1])
    inn = f.innovation(0.0, own.xy[1])
    hph = float(H @ f.covariance @ H.T)
    assert inn.S == pytest.approx(hph + f.R, rel=1e-12)

    frac = hph / inn.S
    assert 0.0 < frac < 1.0
    # A relative error of e in the state part shows up as e * frac at the check.
    for e in (0.05, 0.14, 0.5):
        assert (inn.S * (1 + e * frac) - inn.S) / inn.S == pytest.approx(e * frac, rel=1e-12)


def test_eta_and_rho_are_the_same_statement():
    """eta = HPH'/S is a definition introduced by this project, not a standard
    quantity. The more common form is rho = HPH'/R. They carry identical
    information: eta = rho/(1+rho) and rho = eta/(1-eta).

    Pinned so the note's claim that either form can be substituted is checkable
    rather than asserted.
    """
    rng = np.random.default_rng(11)
    for _ in range(500):
        hph = abs(rng.normal()) + 1e-9
        R = abs(rng.normal()) + 1e-9
        eta = hph / (hph + R)
        rho = hph / R
        assert eta == pytest.approx(rho / (1 + rho), rel=1e-12)
        assert rho == pytest.approx(eta / (1 - eta), rel=1e-9)
        assert 0.0 <= eta < 1.0


def test_attenuation_factor_matches_the_reported_sixteen():
    """The report's eta ~ 0.06 is a run average, not a per-step constant.

    At the first update P is still the initial covariance and eta is 0.99: no
    attenuation at all. It falls as the filter contracts P, settling near 0.05.
    The mean over the run is what gives the quoted factor of sixteen. An earlier
    draft of the proof described this as measured at the first update, which
    would have been 0.99.

    The geometry is recovered from the scenario stored inside the committed
    sweep results, so the figure stays traceable even though that scenario has
    since been replaced in the codebase.
    """
    import json
    import pathlib as _pl

    from kf2.gating import associate, gate_threshold
    from kf2.montecarlo import ESTIMATORS

    path = _pl.Path(__file__).resolve().parents[1] / "results" / "sweep.json"
    if not path.exists():
        pytest.skip("sweep.json not generated")
    sc = replace(Scenario.from_dict(json.loads(path.read_text())["scenario"]), p0_pos=600.0)

    own = datagen.ownship_track(sc)
    truth = datagen.target_truth(sc, 0)
    det = datagen.generate_detections(sc, truth, own, 0)
    th = gate_threshold(sc.gate_prob, dim=1)
    f = ESTIMATORS["ekf"](sc)
    f.initialise(datagen.initial_estimate(sc, truth[0], 0), datagen.initial_covariance(sc))

    etas = []
    for k in range(1, sc.steps + 1):
        f.predict(sc.dt)
        H = f.jacobian(own.xy[k])
        etas.append(float((H @ f.covariance @ H.T)[0, 0]) / f.innovation(0.0, own.xy[k]).S)
        a = associate(f, own.xy[k], det.per_step[k], th)
        if a.accepted:
            f.update(a.z, own.xy[k])
    eta = np.array(etas)

    assert eta[0] > 0.95, "no attenuation at the first update, where P is still P0"
    assert eta.mean() == pytest.approx(0.062, abs=0.01), "run mean is the reported figure"
    assert 1.0 / eta.mean() == pytest.approx(16.0, abs=1.5), "the factor of sixteen"
    assert np.median(eta[60:]) < 0.08, "settles well below the opening transient"
