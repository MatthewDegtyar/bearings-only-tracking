"""A degree-5 cubature variant. NOT a library implementation.

FilterPy's CubatureKalmanFilter, used in :mod:`kf2.filters`, implements the
third-degree spherical-radial rule, which is the published cubature Kalman
filter of Arasaratnam and Haykin. This module is a departure from it: the same
sampling update carried out with a fifth-degree fully symmetric rule.

It lives in its own module, and is named ``ckf5`` rather than ``ckf``, so that
nothing in the study can quietly compare a library filter against a modified one
while calling both "the CKF". The primary comparison in the report is FilterPy's
EKF against FilterPy's CKF. This variant is reported separately as an extension.

Why it exists: the third-degree rule places all its points on the coordinate
axes, so it integrates no cross terms, and its answer on this problem turned out
to depend on the orientation of the coordinate frame. Measured recovery varied
between 26 and 68 per cent under rotations of a physically unchanged problem. A
fifth-degree rule integrates the fourth moments the discarded term needs and
returned a single value at every rotation. The construction is McNamee and
Stenger's, applied to filtering by Jia, Xin and Cheng.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from .datagen import wrap_pi
from .filters import Innovation, cv_transition, cwna_process_noise


class BearingsOnlyCKF5:
    """Duck-type compatible with :class:`kf2.ekf.BearingsOnlyEKF`.

    Same interface, so gating, association and the Monte Carlo driver are
    unchanged and the two estimators can be run on byte-identical measurement
    streams.
    """

    n = 4

    def __init__(self, q: float, sigma_bearing: float, degree: int = 5):
        if degree not in RULES:
            raise ValueError(f"degree must be one of {sorted(RULES)}")
        self.degree = int(degree)
        self.q = float(q)
        self.R = float(sigma_bearing) ** 2
        self._x = np.zeros(4)
        self._P = np.eye(4)

    # --- same surface as the EKF -----------------------------------------
    def initialise(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self._x = np.array(x0, dtype=float).reshape(4)
        self._P = _symmetrise(np.array(P0, dtype=float).reshape(4, 4))

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()


    def set_measurement_noise(self, sigma_bearing: float) -> None:
        """Set the measurement noise the filter *assumes* for the next update.

        Used by the SNR model. Separate from the noise actually applied to the
        data, which lives in kf2.datagen -- the whole point of the experiment is
        that those two can disagree.
        """
        self.R = float(sigma_bearing) ** 2

    def predict(self, dt: float) -> None:
        F = cv_transition(dt)
        self._x = F @ self._x
        self._P = _symmetrise(F @ self._P @ F.T + cwna_process_noise(self.q, dt))

    def predicted_bearing(self, own_xy: np.ndarray) -> float:
        d = self._x[:2] - np.asarray(own_xy)
        return float(np.arctan2(d[1], d[0]))

    # --- the cubature measurement update ---------------------------------
    def _moments(
        self,
        own_xy: np.ndarray,
        x: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> tuple[float, float, np.ndarray]:
        """Predicted bearing, its variance, and the cross-covariance.

        Computed with respect to ``N(x, P)``, defaulting to the filter's own
        state. The override exists for the iterated variant, which needs these
        moments about a *moved* sampling point while the prior stays fixed.

        Bearings are circular, so the points are averaged *relative to* the
        bearing of the mean state and the offset added back. Averaging raw
        angles would break for any track straddling the +/-pi cut.
        """
        x = self._x if x is None else np.asarray(x, dtype=float)
        P = self._P if P is None else np.asarray(P, dtype=float)
        pts, w = _cubature_points(x, P, self.degree)
        own = np.asarray(own_xy, dtype=float)
        d = pts[:, :2] - own
        ref = float(np.arctan2(x[1] - own[1], x[0] - own[0]))
        dz = wrap_pi(np.arctan2(d[:, 1], d[:, 0]) - ref)

        dz_bar = float(np.sum(w * dz))
        z_hat = ref + dz_bar
        dzc = dz - dz_bar
        dxc = pts - x

        Pzz = float(np.sum(w * dzc * dzc)) + self.R
        Pxz = np.sum(w[:, None] * dxc * dzc[:, None], axis=0)
        return z_hat, Pzz, Pxz

    def innovation(
        self, z: float, own_xy: np.ndarray, x_lin: np.ndarray | None = None
    ) -> Innovation:
        """Innovation and its covariance, without applying the update.

        ``x_lin`` is accepted and ignored: there is no linearisation point to
        override. It exists so the estimators are interchangeable.
        """
        z_hat, Pzz, _ = self._moments(own_xy)
        return Innovation(nu=float(wrap_pi(z - z_hat)), S=Pzz)

    def update(self, z: float, own_xy: np.ndarray, x_lin: np.ndarray | None = None) -> Innovation:
        z_hat, Pzz, Pxz = self._moments(own_xy)
        if not np.isfinite(Pzz) or Pzz <= 0.0:
            raise FloatingPointError("non-positive innovation covariance")

        nu = float(wrap_pi(z - z_hat))
        K = Pxz / Pzz
        self._x = self._x + K * nu
        self._P = _symmetrise(self._P - np.outer(K, K) * Pzz)
        return Innovation(nu=nu, S=Pzz)


def _rule3(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Third-degree spherical-radial rule: 2n points on the axes, equal weights."""
    z = np.vstack((np.eye(n), -np.eye(n))) * math.sqrt(n)
    return z, np.full(2 * n, 1.0 / (2 * n))


def _rule5(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Fifth-degree rule (McNamee-Stenger) for the Gaussian weight.

    Origin, 2n axis points at +/-sqrt(n+2), and 2n(n-1) diagonal points at
    +/-sqrt((n+2)/2) on each coordinate pair. For n = 4 the axis weight is
    exactly zero, leaving 25 effective points.
    """
    pts = [np.zeros(n)]
    wts = [2.0 / (n + 2)]
    w_axis = (4.0 - n) / (2.0 * (n + 2) ** 2)
    w_diag = 1.0 / (n + 2) ** 2
    for i in range(n):
        for sign in (1.0, -1.0):
            e = np.zeros(n)
            e[i] = sign * math.sqrt(n + 2.0)
            pts.append(e)
            wts.append(w_axis)
    for i, j in itertools.combinations(range(n), 2):
        for si, sj in itertools.product((1.0, -1.0), repeat=2):
            e = np.zeros(n)
            e[i] = si * math.sqrt((n + 2) / 2.0)
            e[j] = sj * math.sqrt((n + 2) / 2.0)
            pts.append(e)
            wts.append(w_diag)
    z, w = np.array(pts), np.array(wts)
    keep = np.abs(w) > 1e-15
    return z[keep], w[keep]


#: Standardised point sets and weights, by polynomial degree of exactness.
RULES = {3: _rule3(4), 5: _rule5(4)}


def _cubature_points(x: np.ndarray, P: np.ndarray, degree: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Points and weights for ``N(x, P)``, exact to the given polynomial degree.

    The square root transports the standardised rule into the covariance
    ellipsoid, so the columns of the factor set where the points land.
    """
    z, w = RULES[degree]
    return x + z @ _safe_cholesky(P).T, w


def _safe_cholesky(P: np.ndarray) -> np.ndarray:
    """Cholesky with an eigenvalue-clipping fallback.

    The sigma-point covariance update ``P - K Pzz K'`` has no Joseph form and can
    lose positive-definiteness in the last bits. Clipping is a numerical
    safeguard, not a fix for a broken filter -- if it ever triggers on a real
    scenario that is a finding, so it is loud.
    """
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(_symmetrise(P))
        floor = max(np.max(w), 0.0) * 1e-12
        if floor <= 0.0:
            raise
        return np.linalg.cholesky(V @ np.diag(np.clip(w, floor, None)) @ V.T)


def _symmetrise(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + P.T)


class IteratedCKF(BearingsOnlyCKF5):
    """Iterated cubature update (iterated posterior linearisation).

    Motivation. The oracle diagnostic shows two mechanisms: at moderate prior
    uncertainty the linearisation *point* is wrong, at wide uncertainty
    linearisation itself is. Plain cubature addresses the second. Nothing
    addresses the first, which is why the cubature residual is largest where the
    oracle says most is achievable.

    An iterated *EKF* was tried and was worse -- but IEKF iterates a first-order
    linearisation, which the diagnostic says is the wrong object at wide priors.
    Iterating a sigma-point update is a different thing: each pass re-samples the
    nonlinearity about a better point.

    Formulation. The prior ``(x-, P-)`` stays fixed across iterations; only the
    sampling distribution moves. Each pass statistically linearises the bearing
    about ``N(x_i, P_i)``::

        A = Pxz' P_i^-1        b = z_hat - A x_i      Om = Pzz - R - A P_i A'

    then applies one Kalman update to the *prior* with that linearisation. The
    prior is never re-applied cumulatively, which would shrink the covariance
    once per pass.

    With ``iterations = 1`` this reduces algebraically to the plain cubature
    update -- ``S = Pzz``, ``K = Pxz / Pzz`` -- which the tests assert exactly.
    """

    #: Which covariance the sigma points are drawn from on iterations after the
    #: first. ``"posterior"`` is the textbook IPLF: sample from N(x_i, P_i).
    #: ``"prior"`` moves only the sampling *point*, holding the spread at P-.
    #:
    #: The distinction turns out to matter. Sampling from the shrinking posterior
    #: is self-reinforcing: a tighter sampling distribution makes the bearing look
    #: more linear, so the linearisation-error term Omega falls, so S falls, so
    #: the gain rises and P shrinks further. Each pass makes the filter more
    #: confident on evidence it has already used.
    SAMPLE_MODES = ("posterior", "prior")

    def __init__(
        self,
        q: float,
        sigma_bearing: float,
        iterations: int = 3,
        tol: float = 1e-3,
        sample_from: str = "posterior",
        degree: int = 5,
    ):
        super().__init__(q, sigma_bearing, degree)
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if sample_from not in self.SAMPLE_MODES:
            raise ValueError(f"sample_from must be one of {self.SAMPLE_MODES}")
        self.iterations = int(iterations)
        self.tol = float(tol)
        self.sample_from = sample_from

    def _linearise(self, own_xy, x_i, P_i):
        """Statistical linearisation of the bearing about N(x_i, P_i)."""
        z_hat, Pzz, Pxz = self._moments(own_xy, x_i, P_i)
        A = np.linalg.solve(P_i, Pxz)  # A' ; A is (1,4) as a row
        omega = Pzz - self.R - float(A @ P_i @ A)
        return z_hat, A, max(omega, 0.0)

    def update(self, z: float, own_xy: np.ndarray, x_lin: np.ndarray | None = None) -> Innovation:
        x_prior, P_prior = self._x.copy(), self._P.copy()
        # Sampling distribution for the next linearisation pass.
        x_s, P_s = x_prior.copy(), P_prior.copy()
        # Posterior produced by the latest pass. Kept separate from the sampling
        # covariance so that sample_from="prior" still returns a real posterior.
        x_post, P_post = x_prior.copy(), P_prior.copy()
        last = Innovation(nu=0.0, S=self.R)

        for _ in range(self.iterations):
            z_hat, A, omega = self._linearise(own_xy, x_s, P_s)
            S = float(A @ P_prior @ A) + omega + self.R
            if not np.isfinite(S) or S <= 0.0:
                raise FloatingPointError("non-positive innovation covariance")

            # Predicted measurement at the *prior* mean under this linearisation.
            z_at_prior = z_hat + float(A @ (x_prior - x_s))
            nu = float(wrap_pi(z - z_at_prior))

            K = (P_prior @ A) / S
            x_post = x_prior + K * nu
            P_post = _symmetrise(P_prior - np.outer(K, K) * S)
            last = Innovation(nu=nu, S=S)

            step = float(np.linalg.norm(x_post - x_s))
            x_s = x_post
            P_s = P_post if self.sample_from == "posterior" else P_prior
            if step <= self.tol:
                break

        self._x, self._P = x_post, _symmetrise(P_post)
        return last
