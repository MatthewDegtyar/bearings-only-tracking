"""Observability-aware consistency monitoring for passive AOA tracking.

Components, in dependency order:

===================  =========================================================
:mod:`kf2.config`    the Scenario dataclass -- every result-affecting quantity
:mod:`kf2.rng`       named, order-independent noise streams
:mod:`kf2.datagen`   ownship track, target truth, sensor detections
:mod:`kf2.ekf`       bearings-only EKF
:mod:`kf2.gating`    validation gate and nearest-neighbour association
:mod:`kf2.evaluation` NEES/NIS consistency statistics and the gate
:mod:`kf2.montecarlo` orchestration; owns no models and no statistics
===================  =========================================================

`datagen` and `ekf` must not share code -- see the datagen docstring.
"""

from .config import Scenario
from .evaluation import GateReport, Verdict
from .montecarlo import McResult, evaluate, run_monte_carlo, run_trial

__all__ = [
    "Scenario",
    "GateReport",
    "Verdict",
    "McResult",
    "evaluate",
    "run_monte_carlo",
    "run_trial",
]
