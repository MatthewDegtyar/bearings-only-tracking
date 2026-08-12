"""Filters, as thin adapters over FilterPy.

The estimators in this study are not written here. They are FilterPy's
``ExtendedKalmanFilter`` and ``CubatureKalmanFilter``, which are the reference
Python implementations of the two algorithms being compared. This module supplies
only what a bearings-only problem needs on top of them:

  * the constant-velocity motion model and its process noise,
  * the measurement function and its Jacobian,
  * angle wrapping on the innovation, without which a target near the branch cut
    produces a spurious full-turn residual,
  * the ability to evaluate the innovation *without* applying the update, which
    the validation gate needs in order to decide whether to accept a measurement,
  * an override for the linearisation point, used by the diagnostic filter that
    is handed ground truth in order to isolate whether the linearisation point is
    the mechanism.

Using the library implementations rather than hand-rolled ones is deliberate.
The question this project asks is whether a cubature filter beats an extended
Kalman filter on this geometry, and that question is only meaningful if both are
the standard algorithms rather than something bespoke. Every deviation from
library behaviour is listed in ``DEVIATIONS`` below and tested.

FilterPy: R. Labbe, ``filterpy`` 1.4.5. Its EKF uses the Joseph form covariance
update; its CKF uses the third-degree spherical-radial rule of Arasaratnam and
Haykin (2009).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from filterpy.kalman import CubatureKalmanFilter, ExtendedKalmanFilter
from filterpy.kalman.CubatureKalmanFilter import ckf_transform, spherical_radial_sigmas

from .datagen import wrap_pi

#: Every way this module departs from stock FilterPy behaviour, and why.
DEVIATIONS = {
    "wrapped residual": "bearings are circular; z - z_hat must be wrapped to "
                        "(-pi, pi] or a target near the branch cut produces a "
                        "~2 pi innovation and the track is destroyed",
    "circular mean, gate only": "FilterPy's CKF takes an arithmetic mean of the "
                                "measurement sigma points and ignores z_mean_fn, "
                                "which is wrong for angles spanning the branch "
                                "cut. The gate's own innovation uses a circular "
                                "mean; the library update is left as-is and the "
                                "scenarios are checked not to cross the cut",
    "innovation without update": "the validation gate must see (nu, S) before "
                                 "deciding whether to apply the measurement; "
                                 "FilterPy computes them inside update()",
    "linearisation-point override": "the diagnostic filter evaluates its "
                                    "Jacobian at ground truth, which is not "
                                    "implementable but isolates the mechanism",
}


@dataclass(frozen=True)
class Innovation:
    """Innovation and its predicted variance, for a scalar measurement."""

    nu: float
    """Wrapped innovation [rad]."""
    S: float
    """Innovation covariance."""

    @property
    def nis(self) -> float:
        return self.nu * self.nu / self.S


# --------------------------------------------------------------------------
# Motion model
# --------------------------------------------------------------------------

def cv_transition(dt: float) -> np.ndarray:
    """Constant-velocity transition, state ordered [px, py, vx, vy].

    Exact rather than approximate: the generator is nilpotent, so
    exp(A dt) = I + A dt terminates after two terms. See proofs/vein-1.
    """
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    return F


def cwna_process_noise(q: float, dt: float) -> np.ndarray:
    """Continuous white-noise acceleration process noise of spectral density q.

    Q = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]] per axis. This is the *continuous*
    model, not the discrete (piecewise-constant-acceleration) one, which has a
    different form and different units. See proofs/vein-1.
    """
    Q = np.zeros((4, 4))
    Q[0, 0] = Q[1, 1] = q * dt**3 / 3.0
    Q[2, 2] = Q[3, 3] = q * dt
    Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = q * dt**2 / 2.0
    return Q


# --------------------------------------------------------------------------
# Measurement model
# --------------------------------------------------------------------------

def bearing_of(x: np.ndarray, own_xy: np.ndarray) -> float:
    """Predicted bearing from observer to the state's position [rad]."""
    d = np.asarray(x).ravel()[:2] - np.asarray(own_xy)
    return float(math.atan2(d[1], d[0]))


def bearing_jacobian(x: np.ndarray, own_xy: np.ndarray) -> np.ndarray:
    """d(bearing)/dx, shape (1, 4). Zero in the velocity entries."""
    d = np.asarray(x).ravel()[:2] - np.asarray(own_xy)
    r2 = float(d @ d)
    return np.array([[-d[1] / r2, d[0] / r2, 0.0, 0.0]])


def _angular_residual(a, b):
    """z - z_hat, wrapped, preserving the shape NumPy broadcasting would give.

    The shape matters. FilterPy's CKF holds its state as a column vector after
    ``ckf_transform`` and forms ``x + K y``; returning a flat residual there makes
    the sum broadcast to a matrix instead of a vector. Letting ``np.subtract``
    pick the shape keeps both filters working.
    """
    return wrap_pi(np.subtract(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))


def _circular_mean(sigmas, weights=None):
    """Mean of a set of angles, taken on the unit circle."""
    s = np.asarray(sigmas).ravel()
    return np.array([math.atan2(np.sin(s).mean(), np.cos(s).mean())])


class _Base:
    """Interface shared by both estimators, so the harness cannot tell them apart."""

    def __init__(self, q: float, sigma_bearing: float):
        self.q = float(q)
        self.R = float(sigma_bearing) ** 2

    # --- state access -----------------------------------------------------
    @property
    def state(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def covariance(self) -> np.ndarray:
        raise NotImplementedError

    def set_measurement_noise(self, sigma: float) -> None:
        self.R = float(sigma) ** 2

    # --- the measurement, as one overridable hook -------------------------
    def measurement(self, x, own_xy) -> float:
        """h(x). The single place the measurement function is defined.

        Everything that needs h goes through here: the gate, the innovation, and
        the callable handed to FilterPy's update. A subclass that overrides this
        is therefore honoured everywhere, which an earlier version of this
        adapter got wrong: it hardcoded the bearing inside the update lambda, so
        a linear-measurement subclass was silently ignored during the update and
        the control test it backs proved nothing.
        """
        return bearing_of(x, own_xy)

    def predicted_bearing(self, own_xy) -> float:
        return self.measurement(self.state, own_xy)

    def jacobian(self, own_xy, x_lin=None) -> np.ndarray:
        point = self.state if x_lin is None else np.asarray(x_lin, dtype=float)
        return bearing_jacobian(point, own_xy)


class BearingsOnlyEKF(_Base):
    """FilterPy's ExtendedKalmanFilter, wired for a bearings-only track."""

    def __init__(self, q: float, sigma_bearing: float):
        super().__init__(q, sigma_bearing)
        self._f = ExtendedKalmanFilter(dim_x=4, dim_z=1)

    def initialise(self, x0, P0) -> None:
        self._f.x = np.asarray(x0, dtype=float).reshape(4)
        self._f.P = np.asarray(P0, dtype=float).copy()

    @property
    def state(self):
        return self._f.x.reshape(4)

    @property
    def covariance(self):
        return self._f.P

    def predict(self, dt: float) -> None:
        self._f.F = cv_transition(dt)
        self._f.Q = cwna_process_noise(self.q, dt)
        self._f.predict()

    def innovation(self, z: float, own_xy, x_lin=None) -> Innovation:
        H = self.jacobian(own_xy, x_lin)
        S = float(H @ self._f.P @ H.T) + self.R
        if not np.isfinite(S) or S <= 0.0:
            raise FloatingPointError("non-positive innovation covariance")
        return Innovation(nu=float(wrap_pi(z - self.predicted_bearing(own_xy))), S=S)

    def update(self, z: float, own_xy, x_lin=None) -> Innovation:
        inn = self.innovation(z, own_xy, x_lin)
        self._f.update(
            np.array([z]),
            HJacobian=lambda _x, o=own_xy, xl=x_lin: self.jacobian(o, xl),
            Hx=lambda x, o=own_xy: np.array([self.measurement(x, o)]),
            R=np.array([[self.R]]),
            residual=_angular_residual,
        )
        return inn


class BearingsOnlyCKF(_Base):
    """FilterPy's CubatureKalmanFilter, wired for a bearings-only track.

    FilterPy implements the third-degree spherical-radial rule, which is the
    published cubature Kalman filter. ``degree`` is accepted for compatibility
    with the harness; only 3 is supported here, and requesting 5 raises rather
    than silently giving something else.
    """

    def __init__(self, q: float, sigma_bearing: float, degree: int = 3):
        super().__init__(q, sigma_bearing)
        if degree != 3:
            raise ValueError(
                f"FilterPy implements the third-degree rule only; got degree={degree}. "
                "A degree-5 variant is a departure from the library and lives in "
                "kf2.ckf5 so that the distinction stays visible."
            )
        self.degree = degree
        self._own = np.zeros(2)
        self._f = CubatureKalmanFilter(
            dim_x=4, dim_z=1, dt=1.0,
            hx=lambda x, **kw: np.array([self.measurement(x, self._own)]),
            fx=lambda x, dt, **kw: cv_transition(dt) @ x,
            residual_z=_angular_residual,
            z_mean_fn=_circular_mean,
        )

    def initialise(self, x0, P0) -> None:
        self._f.x = np.asarray(x0, dtype=float).reshape(4)
        self._f.P = np.asarray(P0, dtype=float).copy()

    @property
    def state(self):
        return self._f.x.reshape(4)

    @property
    def covariance(self):
        return self._f.P

    def predict(self, dt: float) -> None:
        self._f.Q = cwna_process_noise(self.q, dt)
        self._f.predict(dt=dt)

    def _measurement_sigmas(self, own_xy):
        """Bearings at the cubature points of the current prior."""
        pts = spherical_radial_sigmas(self.state, self.covariance)
        return pts, np.array([[self.measurement(p, own_xy)] for p in pts])

    def innovation(self, z: float, own_xy, x_lin=None) -> Innovation:
        """(nu, S) without applying the update, for the gate.

        Computed the way the CKF itself computes them: transform the cubature
        points through the measurement function and take their spread. The mean
        is circular, and the spread is taken about that circular mean with
        wrapped residuals.
        """
        _, hs = self._measurement_sigmas(own_xy)
        zp = float(_circular_mean(hs)[0])
        resid = wrap_pi(hs.ravel() - zp)
        S = float((resid @ resid) / len(resid)) + self.R
        if not np.isfinite(S) or S <= 0.0:
            raise FloatingPointError("non-positive innovation covariance")
        return Innovation(nu=float(wrap_pi(z - zp)), S=S)

    def update(self, z: float, own_xy, x_lin=None) -> Innovation:
        inn = self.innovation(z, own_xy, x_lin)
        self._own = np.asarray(own_xy, dtype=float)
        self._f.update(np.array([z]), R=np.array([[self.R]]))
        return inn
