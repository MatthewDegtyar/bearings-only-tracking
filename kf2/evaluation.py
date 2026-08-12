"""Consistency evaluation and the gate.

This module takes **plain arrays**, not a Monte Carlo result object. That is
deliberate: it means the statistics can be exercised against a synthetic null --
samples drawn from the distribution a correct filter would produce -- and the
gate's false-alarm rate measured directly, with no EKF in the loop. An earlier
version of this project could only test the gate *through* the filter, and
shipped a gate that failed 58% of seeds on a filter that was correct by
construction.

Two rules the design turns on:

**A criterion that cannot reject is not a pass.** Verdicts are three-valued. Once
tracks diverge, the per-run NEES distribution is heavy-tailed enough that the
sample variance explodes and a confidence interval brackets the target by width
alone. That is ignorance, not evidence, and it reports INCONCLUSIVE.

**Multiple tests need family-wise control.** The gate runs ``2 + 2*n_windows``
tests. Run each at 5% and the probability that *some* test fires on a correct
filter is ``1 - 0.95**14`` = 51%. Every test therefore uses a Sidak-corrected
per-test level so the *family-wise* rate is ``gate_alpha``. The correction is
conservative here, because the window statistics are positively correlated with
each other and with the whole-run statistic; conservative is the safe direction
and :mod:`tests.test_evaluation` measures what it actually delivers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import norm


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCO"


def combine(*verdicts: Verdict) -> Verdict:
    """Worst verdict wins: FAIL beats INCONCLUSIVE beats PASS."""
    if any(v is Verdict.FAIL for v in verdicts):
        return Verdict.FAIL
    if any(v is Verdict.INCONCLUSIVE for v in verdicts):
        return Verdict.INCONCLUSIVE
    return Verdict.PASS


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------


def nees_of(error: np.ndarray, P: np.ndarray) -> float:
    """Normalised estimation error squared, or NaN if P is not usable.

    Shared by the Monte Carlo driver and by the linear-measurement control, so
    the control cannot validate a different NEES computation from the one the
    results use.
    """
    try:
        L = np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        return float("nan")
    y = np.linalg.solve(L, np.asarray(error, dtype=float))
    return float(y @ y)


def n_gate_tests(n_windows: int) -> int:
    """Number of statistical tests the consistency gate performs.

    Two whole-run bias tests (NEES, NIS) plus one per window per metric. Kept as
    a function so the Sidak correction and the tests cannot disagree about it.
    """
    return 2 + 2 * int(n_windows)


def sidak_alpha(alpha: float, n_tests: int) -> float:
    """Per-test level giving family-wise ``alpha`` over ``n_tests``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1")
    return 1.0 - (1.0 - alpha) ** (1.0 / n_tests)


def sidak_z(alpha: float, n_tests: int) -> float:
    """Two-sided normal quantile for the Sidak-corrected per-test level."""
    return float(norm.ppf(1.0 - sidak_alpha(alpha, n_tests) / 2.0))


def mean_ci(samples: np.ndarray, z: float) -> tuple[float, float, float, int]:
    """Mean and CLT interval over independent samples, ignoring NaN.

    Returns ``(mean, lo, hi, n)``. Runs are independent, so a CLT interval over
    per-run statistics is valid. Time steps *within* a run are not independent,
    which is why nothing here ever forms an interval over time steps.
    """
    s = np.asarray(samples, dtype=float).ravel()
    s = s[np.isfinite(s)]
    n = s.size
    if n < 2:
        return (float(s.mean()) if n else float("nan"), float("-inf"), float("inf"), n)
    mean = float(s.mean())
    half = z * float(s.std(ddof=1)) / np.sqrt(n)
    return mean, mean - half, mean + half, n


def window_slices(n_points: int, n_windows: int) -> list[slice]:
    """Contiguous, near-equal windows covering ``range(n_points)``."""
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")
    edges = [i * n_points // n_windows for i in range(n_windows + 1)]
    return [slice(edges[i], edges[i + 1]) for i in range(n_windows)]


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Criterion:
    name: str
    statistic: float
    target: float
    ci_lo: float
    ci_hi: float
    n_samples: int
    verdict: Verdict
    note: str = ""

    @property
    def ci_width(self) -> float:
        return self.ci_hi - self.ci_lo

    @property
    def resolution(self) -> float:
        """Smallest relative bias this test could have rejected.

        A PASS means "no bias larger than this was detected", not "no bias".
        Reporting it keeps a pass from being read as stronger than it is -- which
        matters here because the null is not exactly true: a bearings-only EKF
        carries an intrinsic linearisation bias, so a large enough run count will
        eventually reject any baseline.
        """
        if not np.isfinite(self.ci_width) or self.target == 0:
            return float("inf")
        return 0.5 * self.ci_width / abs(self.target)

    def line(self) -> str:
        ci = f"[{self.ci_lo:.3f}, {self.ci_hi:.3f}]"
        extra = f"  {self.note}" if self.note else ""
        res = f"res {100 * self.resolution:4.1f}%" if np.isfinite(self.resolution) else "res  n/a"
        return (
            f"  [{self.verdict.value}] {self.name:<28} {self.statistic:8.3f}  "
            f"CI {ci:<20} target {self.target:.3f}  {res}{extra}"
        )


def bias_criterion(
    name: str,
    samples: np.ndarray,
    target: float,
    z: float,
    max_ci_width_frac: float,
) -> Criterion:
    """Test whether the mean of per-run statistics is consistent with ``target``."""
    mean, lo, hi, n = mean_ci(samples, z)
    width = hi - lo
    if n < 2 or not np.isfinite(width):
        return Criterion(name, mean, target, lo, hi, n, Verdict.INCONCLUSIVE, "too few samples")
    if width > max_ci_width_frac * target:
        return Criterion(
            name, mean, target, lo, hi, n, Verdict.INCONCLUSIVE,
            f"interval too wide to reject (median {np.nanmedian(samples):.3f})",
        )
    verdict = Verdict.PASS if lo <= target <= hi else Verdict.FAIL
    return Criterion(name, mean, target, lo, hi, n, verdict)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsistencyReport:
    criteria: tuple[Criterion, ...]
    alpha: float
    n_tests: int
    z: float

    @property
    def verdict(self) -> Verdict:
        return combine(*(c.verdict for c in self.criteria))

    @property
    def passed(self) -> bool:
        return self.verdict is Verdict.PASS

    @property
    def failures(self) -> tuple[Criterion, ...]:
        return tuple(c for c in self.criteria if c.verdict is not Verdict.PASS)

    def table(self) -> str:
        head = (
            f"consistency: {self.n_tests} tests, family-wise alpha {self.alpha:.3f}, "
            f"per-test z {self.z:.3f}"
        )
        return "\n".join([head, *(c.line() for c in self.criteria)])


@dataclass(frozen=True)
class RobustnessReport:
    divergence_rate: float
    divergence_rate_final: float
    nees_coverage: float
    max_divergence_rate: float
    verdict: Verdict

    def table(self) -> str:
        return (
            f"  [{self.verdict.value}] robustness                  "
            f"track loss {100 * self.divergence_rate:.2f}% "
            f"(final-step only {100 * self.divergence_rate_final:.2f}%), "
            f"NEES coverage {self.nees_coverage:.4f}"
        )


@dataclass(frozen=True)
class GateReport:
    consistency: ConsistencyReport
    """Over **all** runs, including any that lost the track. This is the
    criterion the gate verdict uses."""
    robustness: RobustnessReport
    consistency_survivors: ConsistencyReport | None = None
    """Over surviving runs only. Reported, never used for the verdict.

    Both are needed because neither alone is honest. Including lost tracks lets a
    handful of enormous outliers dominate the mean, so the statistic drifts
    toward measuring the divergence rate. Excluding them conditions on success
    and is optimistically biased -- the same selection effect the validation gate
    produces on NIS. Quoting both, with the track-loss rate beside them, says
    which regime a number came from.
    """

    @property
    def passed(self) -> bool:
        return self.consistency.passed and self.robustness.verdict is Verdict.PASS

    def table(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        out = [self.consistency.table(), self.robustness.table()]
        if self.consistency_survivors is not None:
            s = self.consistency_survivors.criteria
            nees = next((c for c in s if c.name.startswith("NEES bias")), None)
            nis = next((c for c in s if c.name.startswith("NIS bias")), None)
            if nees is not None and nis is not None:
                out.append(
                    f"         survivors only: NEES {nees.statistic:.3f} "
                    f"(n={nees.n_samples}), NIS {nis.statistic:.3f}  -- reported, "
                    f"not used for the verdict"
                )
        tail = "" if verdict == "PASS" else "\n  A failure here is a bug, not a phenomenon."
        return "\n".join(out) + f"\n\n  GATE: {verdict}{tail}"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def evaluate_consistency(
    nees: np.ndarray,
    nis: np.ndarray,
    *,
    n_windows: int,
    alpha: float,
    max_ci_width_frac: float,
    max_ci_width_frac_nis: float | None = None,
    dim_nees: int = 4,
    dim_nis: int = 1,
    nis_target: float | None = None,
) -> ConsistencyReport:
    """Evaluate NEES/NIS consistency.

    Parameters
    ----------
    nees, nis:
        Shape ``(n_runs, n_points)``. NaN marks a missing sample -- a step where
        no measurement was accepted, or where the covariance was not positive
        definite. NaN is ignored, never treated as zero.
    n_windows:
        Contiguous windows for the time-localised criterion. A whole-run average
        dilutes an excursion confined to part of the run by the ratio of run
        length to excursion length, so a filter that is badly inconsistent over
        its final quarter passes a whole-run test. That is exactly the signature
        of observability degrading over a run.
    nis_target:
        Expected NIS of the *accepted* innovations. Defaults to ``dim_nis``,
        which is only correct with no validation gate. Under gating the accepted
        innovations are a truncated sample -- the gate rejects exactly the large
        innovations that would reveal overconfidence -- so the right reference is
        :func:`kf2.gating.truncated_chi2_moments`. Passing ``dim_nis`` when a
        gate is active makes a correct filter look 3% optimistic at a 99.7% gate
        and 24% optimistic at a 95% gate.

        Note what this does *not* fix: state-space consistency is conditioned on
        the same selection event and remains a separate problem, so the NEES
        target stays ``dim_nees``.
    """
    nees = np.atleast_2d(np.asarray(nees, dtype=float))
    nis = np.atleast_2d(np.asarray(nis, dtype=float))

    n_tests = n_gate_tests(n_windows)
    z = sidak_z(alpha, n_tests)
    nis_goal = float(dim_nis if nis_target is None else nis_target)
    # Per-metric width limits: see Scenario.max_ci_width_frac_nis.
    nis_width = max_ci_width_frac if max_ci_width_frac_nis is None else max_ci_width_frac_nis

    criteria: list[Criterion] = []
    with np.errstate(invalid="ignore"):
        criteria.append(
            bias_criterion("NEES bias (whole run)", _row_means(nees), dim_nees, z, max_ci_width_frac)
        )
        criteria.append(
            bias_criterion("NIS bias (whole run)", _row_means(nis), nis_goal, z, nis_width)
        )
        for i, sl in enumerate(window_slices(nees.shape[1], n_windows)):
            criteria.append(
                bias_criterion(
                    f"NEES window {i} [{sl.start}:{sl.stop}]",
                    _row_means(nees[:, sl]), dim_nees, z, max_ci_width_frac,
                )
            )
        for i, sl in enumerate(window_slices(nis.shape[1], n_windows)):
            criteria.append(
                bias_criterion(
                    f"NIS window {i} [{sl.start}:{sl.stop}]",
                    _row_means(nis[:, sl]), nis_goal, z, nis_width,
                )
            )

    return ConsistencyReport(criteria=tuple(criteria), alpha=alpha, n_tests=n_tests, z=z)


def survivor_mask(
    pos_err: np.ndarray, *, settle_index: int, divergence_pos_error: float
) -> np.ndarray:
    """Boolean mask of runs that never lost the track.

    The single definition of survival in the project. :func:`evaluate_robustness`
    reports its complement as the track-loss rate and the survivors-only
    consistency report is conditioned on it, so the two can never disagree about
    which runs were lost.

    A track is lost if position error exceeds the threshold at *any* step after
    the settling period -- not merely at the final step. A track that loses lock
    at t = 400 and drifts back inside the threshold by t = 600 has still been
    lost. The settling period keeps a wide prior from scoring as divergence on
    its own initial error.
    """
    pos_err = np.atleast_2d(np.asarray(pos_err, dtype=float))
    tail = pos_err[:, settle_index:]
    return ~(np.nanmax(tail, axis=1) > divergence_pos_error)


def evaluate_robustness(
    pos_err: np.ndarray,
    nees: np.ndarray,
    *,
    settle_index: int,
    divergence_pos_error: float,
    max_divergence_rate: float = 0.01,
) -> RobustnessReport:
    """Track loss and sample coverage.

    Track loss is judged over the whole run after a settling period, not at the
    final step: a track that loses lock at t = 400 and drifts back inside the
    threshold by t = 600 has still been lost. The settling period exists so that
    a wide prior is not itself scored as divergence.

    ``nees_coverage`` below 1 means some runs contributed no NEES sample at some
    step, so every NEES figure is conditioned on the filter having survived --
    the same selection effect Phase 3 studies in the association gate. It is
    surfaced rather than absorbed.
    """
    pos_err = np.atleast_2d(np.asarray(pos_err, dtype=float))
    nees = np.atleast_2d(np.asarray(nees, dtype=float))

    ever_lost = ~survivor_mask(
        pos_err, settle_index=settle_index, divergence_pos_error=divergence_pos_error
    )
    lost_at_end = pos_err[:, -1] > divergence_pos_error

    coverage = float(np.isfinite(nees).mean())
    rate = float(np.mean(ever_lost))
    verdict = Verdict.PASS if (rate <= max_divergence_rate and coverage >= 1.0) else Verdict.FAIL
    return RobustnessReport(
        divergence_rate=rate,
        divergence_rate_final=float(np.mean(lost_at_end)),
        nees_coverage=coverage,
        max_divergence_rate=max_divergence_rate,
        verdict=verdict,
    )


def _row_means(a: np.ndarray) -> np.ndarray:
    """Per-run mean, ignoring NaN; NaN for a run with no samples at all."""
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return np.full(a.shape[0], np.nan)
    counts = np.isfinite(a).sum(axis=1)
    out = np.full(a.shape[0], np.nan)
    ok = counts > 0
    if ok.any():
        out[ok] = np.nanmean(a[ok], axis=1)
    return out
