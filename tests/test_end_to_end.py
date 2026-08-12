"""End-to-end tests: the components wired together.

The gate's statistics are verified against a synthetic null in
:mod:`tests.test_evaluation`. What is tested here is that the *pipeline* --
datagen -> ekf -> gating -> evaluation -- produces a consistent filter on the
baseline, and that it does so for reasons that survive changing the seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from kf2 import Scenario, evaluate, run_monte_carlo, run_trial
from kf2.montecarlo import nis_reference
from kf2.config import replace
from kf2.evaluation import Verdict

# Small enough to run in seconds, large enough that the gate has power.
QUICK = Scenario(name="quick", mc_runs=120, steps=300, gate_windows=4)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_a_trial_is_a_pure_function_of_scenario_and_run():
    a = run_trial(QUICK, 3)
    b = run_trial(QUICK, 3)
    assert np.array_equal(np.nan_to_num(a.nees), np.nan_to_num(b.nees))
    assert np.array_equal(a.pos_err, b.pos_err)


def test_different_runs_and_seeds_give_different_data():
    a = run_trial(QUICK, 3)
    assert not np.array_equal(a.pos_err, run_trial(QUICK, 4).pos_err)
    assert not np.array_equal(a.pos_err, run_trial(replace(QUICK, seed=1), 3).pos_err)


def test_run_order_does_not_affect_results():
    """Guards the property that makes a parallel sweep safe."""
    forward = [run_trial(QUICK, r).pos_err for r in range(4)]
    backward = list(reversed([run_trial(QUICK, r).pos_err for r in reversed(range(4))]))
    assert all(np.array_equal(a, b) for a, b in zip(forward, backward))


# ---------------------------------------------------------------------------
# Phase 1 gate
# ---------------------------------------------------------------------------


def test_baseline_passes_the_gate():
    mc = run_monte_carlo(QUICK)
    report = evaluate(mc)
    assert report.passed, "\n" + report.table()
    assert report.robustness.nees_coverage == 1.0
    assert report.robustness.divergence_rate == 0.0
    assert mc.misassociation_rate == 0.0, "phase 1 has nothing to misassociate"


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_the_gate_is_not_seed_dependent(seed):
    """The check that was missing before.

    A gate that passes on one seed and fails on the next is measuring the seed,
    not the filter. This is a smoke test over a handful of seeds; the precise
    false-alarm rate is measured in test_evaluation against a synthetic null.
    """
    report = evaluate(run_monte_carlo(replace(QUICK, seed=seed * 7919)))
    assert report.passed, f"\nseed {seed * 7919}\n" + report.table()


def test_every_step_yields_a_nees_sample_on_the_baseline():
    mc = run_monte_carlo(replace(QUICK, mc_runs=40))
    assert np.isfinite(mc.nees).all()
    assert np.isnan(mc.nis[:, 0]).all(), "k=0 has no innovation"
    # NIS is present exactly where the gate accepted, and NaN elsewhere -- never
    # zero, which would drag the mean down as if it were a real sample.
    assert np.isfinite(mc.nis[mc.accepted]).all()
    assert np.isnan(mc.nis[~mc.accepted]).all()


def test_the_validation_gate_is_active_even_in_phase_1():
    """Phase 1 has no clutter and pd = 1, but the gate still rejects the tail.

    At a 99.7% gate it discards 0.3% of true measurements, and those are always
    the largest innovations -- so the accepted NIS sample is truncated. This is
    the Phase 3 mechanism already present at Phase 1 scale, which is why
    ``gate_prob`` defaults to effectively open: a tight gate is a perturbation,
    and Phase 1 must not inherit one by accident.
    """
    sc = replace(QUICK, mc_runs=60, gate_prob=0.997)
    mc = run_monte_carlo(sc)
    accepted = mc.accepted[:, 1:].mean()
    assert accepted == pytest.approx(0.997, abs=0.003)
    assert accepted < 1.0, "a finite gate must reject something"


def test_nis_matches_the_gate_aware_reference_not_the_naive_one():
    """Measured NIS must match the truncated-chi-square mean, not 1.0.

    This is what made the baseline fail against a naive reference: the filter
    was correct and the reference was wrong.
    """
    sc = replace(QUICK, mc_runs=150, gate_prob=0.997)
    mc = run_monte_carlo(sc)
    measured = np.nanmean(mc.nis)
    assert measured == pytest.approx(nis_reference(sc), abs=0.01)
    assert measured < 1.0, "gating biases NIS low"
    assert nis_reference(sc) == pytest.approx(0.9709, abs=1e-3)


def test_opening_the_gate_restores_the_naive_reference():
    """With the default near-open gate the truncation all but vanishes.

    Not exactly: at gate_prob = 0.9999 the reference is 0.9984, still 0.16% below
    the naive 1.0. Small enough to be far under the harness's resolution, but the
    gate is never truly off, and the reference is always computed rather than
    assumed.
    """
    sc = replace(QUICK, mc_runs=150)
    assert nis_reference(sc) == pytest.approx(0.9984, abs=1e-3)
    assert nis_reference(sc) < 1.0
    mc = run_monte_carlo(sc)
    assert np.nanmean(mc.nis) == pytest.approx(nis_reference(sc), abs=0.02)


# ---------------------------------------------------------------------------
# The gate must react to a genuinely broken filter
# ---------------------------------------------------------------------------


def test_an_overconfident_filter_is_caught():
    """Starve the filter's process noise: it under-models the target's motion,
    so error grows while the covariance does not."""
    sc = replace(QUICK, q=1e-2, q_filter=1e-6)
    report = evaluate(run_monte_carlo(sc))
    assert not report.passed
    nees = next(c for c in report.consistency.criteria if c.name.startswith("NEES bias"))
    assert nees.verdict is not Verdict.PASS
    assert nees.statistic > 4.0, "under-modelled process noise reads as overconfidence"


def test_a_conservative_filter_is_also_caught():
    """Consistency is two-sided: an over-inflated covariance is wrong too, even
    though it is 'safe'. RMSE would not distinguish this from a good filter."""
    sc = replace(QUICK, q=1e-6, q_filter=1e-2)
    report = evaluate(run_monte_carlo(sc))
    assert not report.passed
    nees = next(c for c in report.consistency.criteria if c.name.startswith("NEES bias"))
    assert nees.statistic < 4.0


def test_a_wide_prior_diverges_and_the_gate_says_so_honestly():
    """The heavy-tail case: the bias criterion must report INCONCLUSIVE rather
    than passing on an interval too wide to reject anything."""
    sc = replace(QUICK, p0_pos=3000.0, mc_runs=150, steps=600)
    report = evaluate(run_monte_carlo(sc))
    assert not report.passed
    assert report.robustness.divergence_rate > 0.05
    # Track loss judged over the run, not at the final step only.
    assert report.robustness.divergence_rate >= report.robustness.divergence_rate_final
    nees = next(c for c in report.consistency.criteria if c.name.startswith("NEES bias"))
    assert nees.verdict is Verdict.INCONCLUSIVE
    assert "median" in nees.note


# ---------------------------------------------------------------------------
# Perturbations that Phase 3 will study; here only that they wire through
# ---------------------------------------------------------------------------


def test_missed_detections_reduce_the_accepted_update_count():
    mc = run_monte_carlo(replace(QUICK, pd=0.6, mc_runs=30))
    rate = mc.accepted[:, 1:].mean()
    assert rate == pytest.approx(0.6, abs=0.05)
    # NIS is absent, not zero, where no measurement was accepted.
    assert np.isnan(mc.nis[~mc.accepted]).all()


def test_clutter_produces_misassociations():
    mc = run_monte_carlo(replace(QUICK, clutter_rate=4.0, mc_runs=30))
    assert mc.misassociation_rate > 0.0
    assert mc.accepted[:, 1:].mean() > 0.9, "gate should still accept something most scans"
