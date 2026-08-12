"""One filter run, with everything it produced kept.

Before this existed the same loop was written four times: once in
``montecarlo``, twice in the case exporter and once in the trace exporter. Each
copy kept whatever its caller happened to need and threw the rest away, so
asking a new question meant writing the loop again. Three of those copies drifted
in small ways and one of them silently disagreed about which step counted as
settled.

Everything now goes through :func:`track`. It records the full trajectory and
the derived quantities that are expensive to recompute, and callers project down
to what they want. ``montecarlo.run_trial`` is a thin projection of it, so the
statistics path and the visualisation path cannot diverge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import datagen
from .config import Scenario
from .evaluation import nees_of
from .gating import associate, gate_threshold


@dataclass(frozen=True)
class Track:
    """One estimator's pass over one run of one scenario."""

    scenario: Scenario
    run: int
    estimator: str

    t: np.ndarray
    own: datagen.OwnshipTrack
    truth: np.ndarray
    """True target state per step, shape (n, 4)."""
    est: np.ndarray
    """Filtered state per step, shape (n, 4)."""
    cov: np.ndarray
    """Filtered covariance per step, shape (n, 4, 4)."""

    nees: np.ndarray
    nis: np.ndarray
    accepted: np.ndarray
    """Whether a measurement was used at each step."""
    in_fov: np.ndarray
    """Whether the target was inside the sensor aperture, regardless of detection."""
    bearing: np.ndarray
    """The measurement actually used, NaN where none was."""
    boresight: np.ndarray
    range: np.ndarray

    n_candidates: np.ndarray
    misassociated: int

    @property
    def pos_err(self) -> np.ndarray:
        return np.hypot(*(self.truth[:, :2] - self.est[:, :2]).T)

    @property
    def vel_err(self) -> np.ndarray:
        return np.hypot(*(self.truth[:, 2:] - self.est[:, 2:]).T)

    @property
    def pos_sigma(self) -> np.ndarray:
        return np.sqrt(self.cov[:, 0, 0] + self.cov[:, 1, 1])

    def directional_error(self) -> tuple[np.ndarray, np.ndarray]:
        """Position error resolved along and across the line of sight [m].

        The split a bearings-only tracker lives and dies by: the measurement
        constrains the across direction and says nothing about the along
        direction, so a single total error hides which of the two failed.
        """
        e = self.truth[:, :2] - self.est[:, :2]
        d = self.est[:, :2] - self.own.xy
        los = np.arctan2(d[:, 1], d[:, 0])
        c, s = np.cos(los), np.sin(los)
        return np.abs(e[:, 0] * c + e[:, 1] * s), np.abs(-e[:, 0] * s + e[:, 1] * c)

    def directional_sigma(self) -> tuple[np.ndarray, np.ndarray]:
        """Claimed uncertainty along and across the line of sight [m]."""
        d = self.est[:, :2] - self.own.xy
        los = np.arctan2(d[:, 1], d[:, 0])
        c, s = np.cos(los), np.sin(los)
        xx, xy, yy = self.cov[:, 0, 0], self.cov[:, 0, 1], self.cov[:, 1, 1]
        along = np.sqrt(np.maximum(0.0, c * c * xx + 2 * c * s * xy + s * s * yy))
        cross = np.sqrt(np.maximum(0.0, s * s * xx - 2 * c * s * xy + c * c * yy))
        return along, cross

    @property
    def lost(self) -> bool:
        """Whether the track diverged, ignoring the opening transient.

        The settling window is taken from the scenario rather than chosen here,
        because three of the four old copies of this loop picked their own.
        """
        k0 = self.scenario.settle_index
        return bool(self.pos_err[k0:].max() > self.scenario.divergence_pos_error)

    def settled(self, arr: np.ndarray) -> np.ndarray:
        """``arr`` with the opening transient dropped."""
        return arr[self.scenario.settle_index:]


def track(
    sc: Scenario,
    run: int = 0,
    estimator: str = "ekf",
    own: datagen.OwnshipTrack | None = None,
) -> Track:
    """Run one estimator over one run and keep everything it produced."""
    from .montecarlo import ESTIMATORS, ORACLE_ESTIMATORS

    truth = datagen.target_truth(sc, run)
    if own is None:
        own, detections = datagen.engagement(sc, truth, run)
    else:
        # a caller supplied the track, which is only valid open-loop
        if sc.own_pursuit:
            raise ValueError("a pursuing observer's track depends on the run; "
                             "do not pass one in")
        detections = datagen.generate_detections(sc, truth, own, run)
    threshold = gate_threshold(sc.gate_prob, dim=1)

    try:
        factory = ESTIMATORS[estimator]
    except KeyError:
        raise ValueError(f"unknown estimator {estimator!r}; have {sorted(ESTIMATORS)}") from None
    filt = factory(sc)
    filt.initialise(datagen.initial_estimate(sc, truth[0], run), datagen.initial_covariance(sc))
    oracle = estimator in ORACLE_ESTIMATORS

    n = sc.steps + 1
    est = np.zeros((n, 4))
    cov = np.zeros((n, 4, 4))
    nees = np.full(n, np.nan)
    nis = np.full(n, np.nan)
    accepted = np.zeros(n, dtype=bool)
    bearing = np.full(n, np.nan)
    n_candidates = np.zeros(n, dtype=int)
    misassociated = 0

    # What R the filter assumes each step. None leaves it alone, which is the
    # path taken whenever the SNR model is off, so results stay identical.
    filter_sigma = None
    if sc.snr_enabled:
        from .snr import assumed_sigma

        assumed = assumed_sigma(sc, own.xy, own.t)
        filter_sigma = datagen.measurement_sigma(sc, truth, own) if assumed is None else assumed

    def record(k: int) -> None:
        est[k] = filt.state
        cov[k] = filt.covariance
        nees[k] = nees_of(truth[k] - filt.state, filt.covariance)

    record(0)
    for k in range(1, n):
        filt.predict(sc.dt)
        if filter_sigma is not None:
            filt.set_measurement_noise(filter_sigma[k])
        x_lin = truth[k] if oracle else None
        assoc = associate(filt, own.xy[k], detections.per_step[k], threshold, x_lin)
        n_candidates[k] = assoc.n_candidates
        if assoc.accepted:
            filt.update(assoc.z, own.xy[k], x_lin)
            nis[k] = assoc.nis
            accepted[k] = True
            bearing[k] = assoc.z
            if detections.truth_index[k] is not None and assoc.picked_index != detections.truth_index[k]:
                misassociated += 1
        record(k)

    true_b = datagen.bearing(truth[:, :2], own.xy)
    return Track(
        scenario=sc, run=run, estimator=estimator,
        t=own.t, own=own, truth=truth, est=est, cov=cov,
        nees=nees, nis=nis, accepted=accepted,
        in_fov=datagen.in_field_of_view(sc, true_b, own),
        bearing=bearing,
        boresight=datagen.sensor_boresight(sc, true_b, own),
        range=datagen.target_range(truth, own),
        n_candidates=n_candidates, misassociated=misassociated,
    )
