"""SNR-dependent measurement noise.

In a real passive receiver angular accuracy is not a constant -- it is set by
signal-to-noise ratio, which varies with range and with propagation. Holding ``R``
fixed is therefore a modelling assumption, and this module exists to price it.

**Default off.** With ``snr_enabled = False`` the measurement noise is the
constant ``sigma_bearing`` and every existing result reproduces bit-for-bit.

The chain, all of it scenario-controlled:

    SNR(t)   =  SNR_ref * (r_ref / r(t))^2 * fade(t)
    sigma(t) =  k_crlb / sqrt(SNR(t))

The first is free-space spreading for a *passive* receiver: received power falls
as ``1/r^2`` on the one-way path (an active radar would be ``1/r^4``). The second
is the Cramer-Rao form for angle estimation, in which the variance is inversely
proportional to SNR. Together they give ``sigma proportional to range``, which is
the practically important consequence: a passive bearing degrades linearly as the
target opens.

``k_crlb`` absorbs everything about the front end that this project does not
model -- array aperture in wavelengths, element count, snapshot count, and the
``1/cos(theta)`` beam-broadening away from broadside. That is deliberate: the
point here is that ``sigma_theta`` *falls out of SNR* rather than being a free
constant, not that any particular array is simulated. By default it is calibrated
so that ``sigma(SNR_ref) == sigma_bearing``, which makes the SNR model agree with
the constant-R model at the reference range and isolates the effect of variation.
"""

from __future__ import annotations

import numpy as np

from .config import Scenario


def crlb_bearing_sigma(snr_linear: np.ndarray | float, k_crlb: float) -> np.ndarray:
    """Angular standard deviation [rad] from linear SNR.

    ``sigma = k_crlb / sqrt(SNR)``. The Cramer-Rao bound for angle estimation has
    variance inversely proportional to SNR for any of the usual array/estimator
    combinations (conventional beamforming, MUSIC, ML); they differ only in the
    constant, which is what ``k_crlb`` carries. See the module docstring.
    """
    snr = np.asarray(snr_linear, dtype=float)
    if np.any(snr <= 0.0):
        raise ValueError("SNR must be positive")
    return k_crlb / np.sqrt(snr)


def db_to_linear(db: np.ndarray | float) -> np.ndarray:
    return np.power(10.0, np.asarray(db, dtype=float) / 10.0)


def fade_db(sc: Scenario, t: np.ndarray) -> np.ndarray:
    """Propagation fade in dB, in ``[-depth, 0]``.

    A smooth periodic fade rather than a random one, so a scenario stays
    reproducible from its parameters and the fade cannot be confused with the
    measurement noise it modulates. Depth 0 disables it.
    """
    t = np.asarray(t, dtype=float)
    if sc.snr_fade_depth_db <= 0.0 or sc.snr_fade_period <= 0.0:
        return np.zeros_like(t)
    phase = 2.0 * np.pi * t / sc.snr_fade_period
    return -sc.snr_fade_depth_db * 0.5 * (1.0 - np.cos(phase))


def snr_db(sc: Scenario, ranges: np.ndarray, t: np.ndarray) -> np.ndarray:
    """SNR in dB along a range/time profile."""
    ranges = np.asarray(ranges, dtype=float)
    spreading = -20.0 * np.log10(np.maximum(ranges, 1e-6) / sc.snr_ref_range)
    return sc.snr_ref_db + spreading + fade_db(sc, t)


def crlb_constant(sc: Scenario) -> float:
    """``k_crlb``, either taken from the scenario or calibrated.

    Calibrated by default so that ``sigma(SNR_ref) == sigma_bearing``: with the
    model on, at the reference range and no fade, the noise equals the constant-R
    case. That makes the SNR knob vary the noise without also relocating it.
    """
    if sc.crlb_constant is not None:
        return float(sc.crlb_constant)
    return float(sc.sigma_bearing * np.sqrt(db_to_linear(sc.snr_ref_db)))


def bearing_sigma_series(sc: Scenario, ranges: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-step measurement noise sigma [rad].

    Returns a constant series equal to ``sigma_bearing`` when the model is off,
    so callers need no branch and the off path stays bit-identical.
    """
    ranges = np.asarray(ranges, dtype=float)
    if not sc.snr_enabled:
        return np.full(ranges.shape, sc.sigma_bearing, dtype=float)
    return crlb_bearing_sigma(db_to_linear(snr_db(sc, ranges, t)), crlb_constant(sc))


def nominal_ranges(sc: Scenario, own_xy: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Range profile of the *nominal* target -- initial state, no process noise.

    Used for the design-time constant-R assumptions below. A real system choosing
    a fixed R would do it from an expected geometry, not from the realised one, so
    deriving these from truth per run would flatter them.
    """
    t = np.asarray(t, dtype=float)
    px = sc.tgt_x0 + sc.tgt_vx0 * t
    py = sc.tgt_y0 + sc.tgt_vy0 * t
    return np.hypot(px - np.asarray(own_xy)[:, 0], py - np.asarray(own_xy)[:, 1])


def assumed_sigma(sc: Scenario, own_xy: np.ndarray, t: np.ndarray) -> np.ndarray | None:
    """The sigma series the *filter* uses, per ``sc.r_assumption``.

    ``"true"``   -- the actual per-step sigma. Realistic: a receiver measures its
                    own SNR, so this is available at runtime without truth.
    ``"mean"``   -- one constant, the root-mean-square sigma over the nominal
                    geometry. The natural "just pick an R" choice.
    ``"best"``   -- one constant, the smallest sigma over the nominal geometry.
                    The optimistic choice, and the one that should hurt most.

    Returns None when the SNR model is off, meaning "leave R alone".
    """
    if not sc.snr_enabled:
        return None
    if sc.r_assumption == "true":
        return None  # filled in per run from the realised range
    nominal = bearing_sigma_series(sc, nominal_ranges(sc, own_xy, t), t)
    if sc.r_assumption == "mean":
        return np.full(nominal.shape, float(np.sqrt(np.mean(nominal**2))))
    if sc.r_assumption == "best":
        return np.full(nominal.shape, float(nominal.min()))
    raise ValueError(f"unknown r_assumption {sc.r_assumption!r}")
