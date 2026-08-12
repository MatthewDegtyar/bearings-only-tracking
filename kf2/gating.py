"""Validation gating and measurement association.

In Phase 1 (``pd = 1``, no clutter) every scan holds exactly one bearing and this
component is nearly a pass-through. It exists as a real component from the start
because the Phase 3 result depends on it being one: accepted innovations are a
*truncated* sample, since the gate rejects exactly the large-innovation events
that would signal overconfidence. Naive NIS therefore reads artificially healthy
after gating, and an adaptive scheme built on it shrinks R, which tightens the
gate, which improves apparent NIS further -- a self-reinforcing loop toward a
confident, tightly-gated, wrong filter.

:func:`truncated_chi2_moments` is the correction. Both the gate threshold and the
measurement dimension are known, so it is analytic rather than estimated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .filters import BearingsOnlyEKF, Innovation


def gate_threshold(prob: float, dim: int = 1) -> float:
    """Chi-square validation-gate threshold for a given gate probability."""
    if not 0.0 < prob < 1.0:
        raise ValueError("gate probability must be in (0, 1)")
    return float(chi2.ppf(prob, dim))


def truncated_chi2_moments(threshold: float, dim: int = 1) -> tuple[float, float]:
    """Mean and variance of chi2(dim) conditioned on being inside the gate.

    Uses ``x f_k(x) = k f_{k+2}(x)``, so

        E[X | X <= T]   = k F_{k+2}(T) / F_k(T)
        E[X^2 | X <= T] = k(k+2) F_{k+4}(T) / F_k(T)

    Exact, not an approximation. As ``T -> inf`` these tend to ``k`` and ``2k``.
    """
    if threshold <= 0.0:
        raise ValueError("threshold must be > 0")
    k = float(dim)
    f_k = chi2.cdf(threshold, k)
    if f_k <= 0.0:
        raise ValueError("gate accepts nothing")
    mean = k * chi2.cdf(threshold, k + 2) / f_k
    second = k * (k + 2.0) * chi2.cdf(threshold, k + 4) / f_k
    return float(mean), float(second - mean * mean)


@dataclass(frozen=True)
class Association:
    """Outcome of gating one scan against one track."""

    accepted: bool
    z: float | None
    innovation: Innovation | None
    picked_index: int | None
    n_candidates: int
    n_in_gate: int

    @property
    def nis(self) -> float | None:
        return None if self.innovation is None else self.innovation.nis


def associate(
    ekf: BearingsOnlyEKF,
    own_xy: np.ndarray,
    scan: np.ndarray,
    threshold: float,
    x_lin: np.ndarray | None = None,
) -> Association:
    """Nearest-neighbour association under a chi-square validation gate.

    Candidates are ranked by normalised squared innovation, which is the right
    metric rather than raw angular distance: it accounts for the track's own
    uncertainty, so a poorly known track correctly accepts a wider spread.

    Returns ``accepted=False`` when the scan is empty or nothing falls inside the
    gate -- a missed update, not an error. The caller must then predict without
    updating, which is what makes ``pd < 1`` a real perturbation rather than a
    dropped sample.

    ``x_lin`` is forwarded to the estimator's ``innovation``; it exists for the
    oracle diagnostic, which linearises at ground truth. Estimators that have no
    linearisation point ignore it.
    """
    scan = np.asarray(scan, dtype=float).ravel()
    if scan.size == 0:
        return Association(False, None, None, None, 0, 0)

    innovations = [ekf.innovation(float(z), own_xy, x_lin) for z in scan]
    d2 = np.array([inn.nis for inn in innovations])
    inside = d2 <= threshold
    n_in_gate = int(inside.sum())
    if n_in_gate == 0:
        return Association(False, None, None, None, scan.size, 0)

    # Nearest neighbour among the gated candidates.
    masked = np.where(inside, d2, np.inf)
    idx = int(np.argmin(masked))
    return Association(
        accepted=True,
        z=float(scan[idx]),
        innovation=innovations[idx],
        picked_index=idx,
        n_candidates=int(scan.size),
        n_in_gate=n_in_gate,
    )
