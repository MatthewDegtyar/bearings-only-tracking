"""Monte Carlo orchestration.

Nothing here reports a single run: single-run trajectory plots hide every effect
this project cares about, so divergence is a rate and consistency is a
distribution.

This module is glue. It owns no models and no statistics -- it wires
:mod:`kf2.datagen`, :mod:`kf2.ekf` and :mod:`kf2.gating` together and hands plain
arrays to :mod:`kf2.evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import datagen
from .config import Scenario
from .ckf5 import BearingsOnlyCKF5, IteratedCKF
from .filters import BearingsOnlyCKF, BearingsOnlyEKF
from .evaluation import (
    GateReport,
    evaluate_consistency,
    evaluate_robustness,
    nees_of,
    survivor_mask,
)
from .gating import associate, gate_threshold, truncated_chi2_moments
from .snr import assumed_sigma


@dataclass(frozen=True)
class RunResult:
    """One Monte Carlo trial. Arrays are indexed by time step 0..steps."""

    run: int
    nees: np.ndarray
    """NaN where the covariance was not positive definite."""
    nis: np.ndarray
    """NaN at k=0 and wherever no measurement was accepted."""
    pos_err: np.ndarray
    vel_err: np.ndarray
    pos_sigma: np.ndarray
    """sqrt(trace of the position block of P) -- what the filter claims."""
    accepted: np.ndarray
    n_candidates: np.ndarray
    misassociated: int
    """Accepted a clutter bearing instead of the target's. Zero in Phase 1."""
    truth: np.ndarray
    """(steps+1, 4). Carried so callers never have to regenerate it."""


@dataclass(frozen=True)
class McResult:
    scenario: Scenario
    ownship: datagen.OwnshipTrack
    nees: np.ndarray  # (n_runs, steps+1)
    nis: np.ndarray  # (n_runs, steps+1)
    pos_err: np.ndarray
    vel_err: np.ndarray
    pos_sigma: np.ndarray
    accepted: np.ndarray
    truth_mean: np.ndarray  # (steps+1, 4) mean truth, for plotting geometry
    misassociation_rate: float
    estimator: str = "ekf"

    @property
    def n_runs(self) -> int:
        return self.nees.shape[0]

    @property
    def t(self) -> np.ndarray:
        return self.ownship.t


#: Estimators are interchangeable by construction: they share an interface, and
#: the noise streams do not depend on which one runs. So a comparison between
#: them is *paired* -- every estimator sees byte-identical measurements.
#:
#: Values are factories taking the Scenario, so estimator options live in the
#: scenario like everything else that can change a number.
ESTIMATORS = {
    # The two being compared, both from FilterPy.
    "ekf": lambda sc: BearingsOnlyEKF(sc.filter_q, sc.sigma_bearing),
    "ckf": lambda sc: BearingsOnlyCKF(sc.filter_q, sc.sigma_bearing),
    # A departure from the library: the same sampling update at degree 5.
    # Named so it cannot be mistaken for the library filter. See kf2/ckf5.py.
    "ckf5": lambda sc: BearingsOnlyCKF5(sc.filter_q, sc.sigma_bearing, 5),
    # Kept only because earlier work measured it; not part of the comparison.
    # Keyword arguments throughout: positional order here was silently wrong once.
    "ickf": lambda sc: IteratedCKF(
        q=sc.filter_q,
        sigma_bearing=sc.sigma_bearing,
        iterations=sc.ckf_iterations,
        tol=sc.ckf_iteration_tol,
        sample_from=sc.ckf_sample_from,
        degree=sc.ckf_degree,
    ),
}

ORACLE_ESTIMATORS = {"oracle"}
ESTIMATORS["oracle"] = ESTIMATORS["ekf"]


def run_trial(
    sc: Scenario,
    run: int,
    own: datagen.OwnshipTrack | None = None,
    estimator: str = "ekf",
) -> RunResult:
    """Execute one trial. Pure in ``(scenario, run, estimator)``.

    A projection of :func:`kf2.run.track` down to the scalars the statistics
    need. The loop itself lives there and is shared with every other consumer,
    so the numbers behind a figure and the numbers behind a table cannot come
    from two subtly different implementations.
    """
    from .run import track as _track

    tr = _track(sc, run, estimator, own)
    return RunResult(
        run=run,
        nees=tr.nees,
        nis=tr.nis,
        pos_err=tr.pos_err,
        vel_err=tr.vel_err,
        pos_sigma=tr.pos_sigma,
        accepted=tr.accepted,
        n_candidates=tr.n_candidates,
        misassociated=tr.misassociated,
        truth=tr.truth,
    )


def run_monte_carlo(sc: Scenario, progress: bool = False, estimator: str = "ekf") -> McResult:
    # Under pursuit the track differs per run, so it cannot be shared.
    own = None if sc.own_pursuit else datagen.ownship_track(sc)
    results = []
    truth_sum = np.zeros((sc.steps + 1, 4))
    for run in range(sc.mc_runs):
        r = run_trial(sc, run, own, estimator)
        results.append(r)
        truth_sum += r.truth
        if progress and (run + 1) % max(1, sc.mc_runs // 10) == 0:
            print(f"  {run + 1}/{sc.mc_runs} runs", flush=True)

    stack = lambda attr: np.vstack([getattr(r, attr) for r in results])  # noqa: E731
    total_updates = sum(int(r.accepted.sum()) for r in results)
    misassoc = sum(r.misassociated for r in results)

    return McResult(
        scenario=sc,
        ownship=own,
        nees=stack("nees"),
        nis=stack("nis"),
        pos_err=stack("pos_err"),
        vel_err=stack("vel_err"),
        pos_sigma=stack("pos_sigma"),
        accepted=stack("accepted"),
        truth_mean=truth_sum / sc.mc_runs,
        misassociation_rate=(misassoc / total_updates) if total_updates else 0.0,
        estimator=estimator,
    )


def nis_reference(sc: Scenario) -> float:
    """Expected NIS of accepted innovations under this scenario's gate.

    The validation gate is always on -- even in Phase 1, where it rejects ~0.3%
    of true measurements at the default 99.7% gate. Those rejections are not
    random: the gate removes exactly the largest innovations, so the accepted
    sample is truncated and reads low. Testing against the naive value of 1
    would flag a correct filter.
    """
    return truncated_chi2_moments(gate_threshold(sc.gate_prob, dim=1), dim=1)[0]


def evaluate(mc: McResult) -> GateReport:
    """Apply the consistency gate to a Monte Carlo result.

    Consistency is computed twice: over all runs (the verdict) and over surviving
    runs only (reported). See :class:`kf2.evaluation.GateReport` for why neither
    is sufficient alone.
    """
    sc = mc.scenario
    kw = dict(
        n_windows=sc.gate_windows,
        alpha=sc.gate_alpha,
        max_ci_width_frac=sc.max_ci_width_frac,
        max_ci_width_frac_nis=sc.max_ci_width_frac_nis,
        nis_target=nis_reference(sc),
    )
    alive = survivor_mask(
        mc.pos_err,
        settle_index=sc.settle_index,
        divergence_pos_error=sc.divergence_pos_error,
    )
    survivors = (
        evaluate_consistency(mc.nees[alive], mc.nis[alive], **kw) if alive.any() else None
    )
    return GateReport(
        consistency=evaluate_consistency(mc.nees, mc.nis, **kw),
        robustness=evaluate_robustness(
            mc.pos_err,
            mc.nees,
            settle_index=sc.settle_index,
            divergence_pos_error=sc.divergence_pos_error,
        ),
        consistency_survivors=survivors,
    )
