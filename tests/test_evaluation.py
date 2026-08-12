"""Tests for the consistency gate.

The important tests here drive :mod:`kf2.evaluation` with **synthetic data** --
NEES/NIS samples drawn from the distribution a correct filter produces -- rather
than through an EKF. That makes the gate's false-alarm rate directly measurable
over thousands of trials, which is the check that was missing when this project
shipped a gate that failed 58% of seeds on a filter correct by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from kf2.evaluation import (
    Verdict,
    bias_criterion,
    combine,
    evaluate_consistency,
    evaluate_robustness,
    mean_ci,
    n_gate_tests,
    sidak_alpha,
    sidak_z,
    window_slices,
)

ALPHA = 0.05
N_WINDOWS = 6
MAX_CI = 0.5


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_sidak_reduces_to_alpha_for_one_test():
    assert sidak_alpha(0.05, 1) == pytest.approx(0.05)
    assert sidak_z(0.05, 1) == pytest.approx(1.959963984540054, rel=1e-12)


def test_sidak_controls_the_family():
    """(1 - per_test)^n_tests == 1 - alpha, by construction."""
    for n in (1, 2, 6, 14, 100):
        per = sidak_alpha(0.05, n)
        assert (1.0 - per) ** n == pytest.approx(0.95, rel=1e-12)
        assert sidak_z(0.05, n) >= 1.959963984540054


def test_more_tests_means_wider_intervals():
    assert sidak_z(0.05, 14) > sidak_z(0.05, 6) > sidak_z(0.05, 1)
    # The correction the old gate was missing, at the size it actually ran.
    assert sidak_z(0.05, 14) == pytest.approx(2.905, abs=0.01)


def test_n_gate_tests_matches_what_evaluate_runs():
    rng = np.random.default_rng(0)
    nees = rng.chisquare(4, size=(40, 60))
    nis = rng.chisquare(1, size=(40, 60))
    report = evaluate_consistency(
        nees, nis, n_windows=N_WINDOWS, alpha=ALPHA, max_ci_width_frac=MAX_CI
    )
    assert len(report.criteria) == n_gate_tests(N_WINDOWS)
    assert report.n_tests == len(report.criteria)


def test_mean_ci_ignores_nan_and_reports_count():
    x = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    mean, lo, hi, n = mean_ci(x, z=1.96)
    assert n == 4
    assert mean == pytest.approx(2.5)
    assert lo < mean < hi


def test_window_slices_cover_exactly_once():
    for n, w in ((600, 6), (601, 6), (10, 3), (7, 7)):
        sls = window_slices(n, w)
        assert len(sls) == w
        covered = np.concatenate([np.arange(n)[s] for s in sls])
        assert np.array_equal(covered, np.arange(n))


def test_combine_worst_wins():
    assert combine(Verdict.PASS, Verdict.PASS) is Verdict.PASS
    assert combine(Verdict.PASS, Verdict.INCONCLUSIVE) is Verdict.INCONCLUSIVE
    assert combine(Verdict.FAIL, Verdict.INCONCLUSIVE) is Verdict.FAIL


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def test_wide_interval_is_inconclusive_not_a_pass():
    """The bug that let a filter with mean NEES of 7.5 million report PASS.

    Heavy-tailed samples make the CLT interval so wide it brackets the target by
    width alone. That is ignorance, not evidence.
    """
    rng = np.random.default_rng(1)
    samples = np.concatenate([rng.chisquare(4, 200), [1e7, 5e6, 2e7]])
    c = bias_criterion("heavy tail", samples, 4.0, z=1.96, max_ci_width_frac=MAX_CI)
    assert c.verdict is Verdict.INCONCLUSIVE
    assert c.ci_lo <= 4.0 <= c.ci_hi, "it does bracket the target -- that is the trap"
    assert "median" in c.note


def test_clean_samples_pass_and_biased_samples_fail():
    rng = np.random.default_rng(2)
    good = rng.chisquare(4, 4000)
    assert bias_criterion("good", good, 4.0, 1.96, MAX_CI).verdict is Verdict.PASS
    assert bias_criterion("biased", good * 1.5, 4.0, 1.96, MAX_CI).verdict is Verdict.FAIL


# ---------------------------------------------------------------------------
# The check that was missing: measured false-alarm rate under the null
# ---------------------------------------------------------------------------


def _null_samples(rng, n_runs, n_steps, run_scale_sd=0.0):
    """NEES/NIS a *correct* filter would produce.

    ``run_scale_sd`` adds a per-run multiplicative effect shared across all time
    steps, reproducing the correlation between windows within a run that real
    NEES has. Positive correlation makes the Sidak correction conservative; the
    test checks the family-wise rate under both.
    """
    scale = 1.0
    if run_scale_sd:
        scale = np.exp(run_scale_sd * rng.standard_normal((n_runs, 1)))
        scale = scale / scale.mean()
    nees = rng.chisquare(4, size=(n_runs, n_steps)) * scale
    nis = rng.chisquare(1, size=(n_runs, n_steps)) * scale
    return nees, nis


@pytest.mark.parametrize("run_scale_sd", [0.0, 0.25])
def test_family_wise_false_alarm_rate_is_controlled(run_scale_sd):
    """Measured FPR on a correct filter must not exceed the nominal alpha.

    600 independent trials gives a standard error of ~0.9% at a 5% rate, so the
    0.5 + 5% ceiling below is ~4 sigma. The previous gate ran 14 tests at 5%
    each with no correction and sat at 58%.
    """
    rng = np.random.default_rng(20260730)
    trials, failures = 600, 0
    for _ in range(trials):
        nees, nis = _null_samples(rng, 200, 120, run_scale_sd)
        report = evaluate_consistency(
            nees, nis, n_windows=N_WINDOWS, alpha=ALPHA, max_ci_width_frac=MAX_CI
        )
        failures += not report.passed
    fpr = failures / trials
    assert fpr <= ALPHA + 0.035, f"family-wise false-alarm rate {fpr:.3f} exceeds nominal {ALPHA}"


def test_uncorrected_gate_would_have_failed_this_test():
    """Pins the magnitude of the bug, so the correction cannot be quietly removed.

    Running the same 14 tests at an uncorrected 5% each puts the family-wise rate
    far above nominal. This is the arithmetic that produced 58% in practice.
    """
    assert 1.0 - (1.0 - ALPHA) ** n_gate_tests(N_WINDOWS) > 0.45


# ---------------------------------------------------------------------------
# Power: the reason the windowed criterion exists
# ---------------------------------------------------------------------------


def _late_excursion_report(base_nees, base_nis, f):
    nees = base_nees.copy()
    nees[:, 500:] *= f
    report = evaluate_consistency(
        nees, base_nis, n_windows=N_WINDOWS, alpha=ALPHA, max_ci_width_frac=MAX_CI
    )
    return {c.name: c for c in report.criteria}, report


def test_windowed_criterion_catches_what_the_whole_run_test_misses():
    """An excursion confined to one window, sized to slip past the whole-run test.

    The gain is quantitative, not categorical: an excursion of relative size
    ``d`` in one of ``W`` windows moves the whole-run mean by ``d/W`` but the
    window mean by ``d``, while the window interval is only ``sqrt(W)`` wider for
    having ``W`` times fewer samples. Net sensitivity advantage ``sqrt(W)``.
    So there is a band of excursion sizes the windowed criterion catches and the
    whole-run criterion does not -- this test sits in that band.
    """
    rng = np.random.default_rng(7)
    base_nees = rng.chisquare(4, size=(600, 600))
    base_nis = rng.chisquare(1, size=(600, 600))

    by_name, report = _late_excursion_report(base_nees, base_nis, 1.02)
    assert by_name["NEES bias (whole run)"].verdict is Verdict.PASS
    assert by_name["NEES window 5 [500:600]"].verdict is Verdict.FAIL
    assert by_name["NEES window 0 [0:100]"].verdict is Verdict.PASS, "clean windows not blamed"
    assert not report.passed


def test_windowed_sensitivity_advantage_is_about_sqrt_n_windows():
    """Measure both detection thresholds instead of asserting a story about them."""
    rng = np.random.default_rng(11)
    base_nees = rng.chisquare(4, size=(600, 600))
    base_nis = rng.chisquare(1, size=(600, 600))

    grid = np.arange(1.0, 1.08, 0.0025)

    def threshold(name):
        for f in grid:
            by_name, _ = _late_excursion_report(base_nees, base_nis, f)
            if by_name[name].verdict is Verdict.FAIL:
                return f - 1.0
        return float("inf")

    whole = threshold("NEES bias (whole run)")
    window = threshold("NEES window 5 [500:600]")
    assert 0.0 < window < whole, "the windowed criterion must be the more sensitive one"
    ratio = whole / window
    assert 1.5 <= ratio <= 3.5, f"expected ~sqrt({N_WINDOWS}) = 2.45, measured {ratio:.2f}"


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_track_loss_counts_mid_run_excursions():
    """A track lost at t=50 that drifts back by the end is still a lost track."""
    pos_err = np.zeros((3, 100))
    pos_err[0, 50] = 5000.0  # lost, then recovers
    pos_err[1, -1] = 5000.0  # lost at the end
    nees = np.ones((3, 100))

    r = evaluate_robustness(
        pos_err, nees, settle_index=10, divergence_pos_error=2000.0, max_divergence_rate=0.01
    )
    assert r.divergence_rate == pytest.approx(2 / 3)
    assert r.divergence_rate_final == pytest.approx(1 / 3)
    assert r.verdict is Verdict.FAIL


def test_settling_period_excludes_initial_transient():
    pos_err = np.zeros((2, 100))
    pos_err[:, 0:5] = 9000.0  # large initial error from a wide prior
    r = evaluate_robustness(
        pos_err, np.ones((2, 100)), settle_index=10, divergence_pos_error=2000.0
    )
    assert r.divergence_rate == 0.0


def test_missing_nees_samples_are_surfaced_not_absorbed():
    nees = np.ones((4, 50))
    nees[0, 10] = np.nan
    r = evaluate_robustness(
        np.zeros((4, 50)), nees, settle_index=0, divergence_pos_error=1.0
    )
    assert r.nees_coverage < 1.0
    assert r.verdict is Verdict.FAIL, "conditioning on survival must block a pass"


def test_chi2_reference_values_are_sane():
    """Guards the scipy dependency behaving as assumed."""
    assert chi2.ppf(0.997, 1) == pytest.approx(8.8074, abs=1e-3)
    assert chi2.ppf(0.95, 4) == pytest.approx(9.4877, abs=1e-3)


# ---------------------------------------------------------------------------
# Divergence vs overconfidence (survivor conditioning)
# ---------------------------------------------------------------------------


def test_survivor_mask_is_the_single_definition_of_track_loss():
    """The mask and the reported track-loss rate must be complements, so the
    survivors-only statistic and the loss rate can never disagree about which
    runs were lost."""
    from kf2.evaluation import survivor_mask

    pos_err = np.zeros((5, 100))
    pos_err[0, 50] = 5000.0   # lost mid-run, recovers
    pos_err[1, -1] = 5000.0   # lost at the end
    pos_err[2, 0:5] = 9000.0  # wide prior only -- inside the settling period

    alive = survivor_mask(pos_err, settle_index=10, divergence_pos_error=2000.0)
    assert list(alive) == [False, False, True, True, True]

    r = evaluate_robustness(
        pos_err, np.ones((5, 100)), settle_index=10, divergence_pos_error=2000.0
    )
    assert r.divergence_rate == pytest.approx(1.0 - alive.mean())


def test_lost_tracks_dominate_the_all_runs_mean():
    """Why both statistics are reported.

    A handful of diverged runs with enormous NEES drag the all-runs mean far
    from the surviving population's. Quoting only the first measures divergence
    rate; quoting only the second conditions on success.
    """
    rng = np.random.default_rng(3)
    nees = rng.chisquare(4, size=(100, 50))
    nis = rng.chisquare(1, size=(100, 50))
    pos_err = np.zeros((100, 50))
    nees[:4] *= 300.0          # four diverged runs
    pos_err[:4, 30] = 9e3

    all_runs = evaluate_consistency(
        nees, nis, n_windows=2, alpha=ALPHA, max_ci_width_frac=MAX_CI
    )
    from kf2.evaluation import survivor_mask

    alive = survivor_mask(pos_err, settle_index=5, divergence_pos_error=2000.0)
    surv = evaluate_consistency(
        nees[alive], nis[alive], n_windows=2, alpha=ALPHA, max_ci_width_frac=MAX_CI
    )
    a = next(c for c in all_runs.criteria if c.name.startswith("NEES bias"))
    s = next(c for c in surv.criteria if c.name.startswith("NEES bias"))

    assert alive.sum() == 96
    assert a.statistic > 5 * s.statistic, "the outliers must dominate the all-runs mean"
    assert s.statistic == pytest.approx(4.0, abs=0.4), "survivors are consistent"
