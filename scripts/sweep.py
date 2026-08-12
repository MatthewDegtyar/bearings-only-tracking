#!/usr/bin/env python3
"""The headline sweep.

Sweeps initial position uncertainty -- the knob controlling how hard the
measurement nonlinearity is exercised -- and reports, per estimator:

    NEES (all)       the reference signal over every run, including lost tracks.
                     This is what the gate verdict uses.
    NEES (survivors) the same over runs that never lost the track.
    NIS              the runtime proxy (needs no truth).
    track loss       fraction of runs that lost the track after settling.

Both NEES columns are reported because neither alone is honest: including lost
tracks lets a few enormous outliers dominate the mean, so the statistic drifts
toward measuring the divergence rate; excluding them conditions on success and is
optimistically biased. The gap between the columns is the size of that problem.

Estimators, all on byte-identical measurement streams:

    ekf     first-order linearisation at the running estimate
    ckf     cubature (sigma-point) update -- the fix
    ickf    iterated cubature update
    oracle  Jacobian at ground truth. NOT IMPLEMENTABLE -- an achievability
            bound. If it removes an error the linearisation *point* is the
            mechanism; if it does not, linearisation itself is.

    python3 scripts/sweep.py [--runs N] [--estimators ekf,ckf,oracle]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kf2 import Scenario, evaluate, run_monte_carlo  # noqa: E402
from kf2.config import replace  # noqa: E402
from kf2.montecarlo import ESTIMATORS  # noqa: E402

P0_GRID = (300.0, 450.0, 600.0, 800.0, 1000.0)
DEFAULT_ESTIMATORS = ("ekf", "ckf", "ickf", "oracle")


def pick(report, prefix):
    return next(c for c in report.criteria if c.name.startswith(prefix))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=400)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--estimators", type=str, default=",".join(DEFAULT_ESTIMATORS))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results"))
    a = ap.parse_args()

    ests = [e.strip() for e in a.estimators.split(",") if e.strip()]
    unknown = [e for e in ests if e not in ESTIMATORS]
    if unknown:
        sys.exit(f"unknown estimator(s) {unknown}; have {sorted(ESTIMATORS)}")

    base = Scenario(name="sweep", mc_runs=a.runs, steps=a.steps)
    rows = []
    # Per-step mean NEES/NIS curves, emitted from the same run that produces the
    # tables so a figure can never disagree with a number.
    curves: dict[str, np.ndarray] = {}

    print(f"{a.runs} runs x {a.steps} steps per point; estimators: {', '.join(ests)}\n")
    header = (
        f"{'p0_pos':>7} {'est':>7} | {'NEES all':>9} {'res':>6} {'v':>5} "
        f"| {'NEES surv':>10} {'n':>5} | {'NIS':>6} {'res':>5} {'v':>5} | {'loss':>6}"
    )
    print(header)
    print("-" * len(header))

    for p0 in P0_GRID:
        sc = replace(base, p0_pos=p0)
        for est in ests:
            t0 = time.perf_counter()
            mc = run_monte_carlo(sc, estimator=est)
            rep = evaluate(mc)
            n_all = pick(rep.consistency, "NEES bias")
            s_all = pick(rep.consistency, "NIS bias")
            n_sur = pick(rep.consistency_survivors, "NEES bias") if rep.consistency_survivors else None

            rows.append(
                dict(
                    p0_pos=p0, estimator=est,
                    nees=n_all.statistic, nees_lo=n_all.ci_lo, nees_hi=n_all.ci_hi,
                    nees_res=n_all.resolution, nees_verdict=n_all.verdict.value,
                    nees_survivors=(n_sur.statistic if n_sur else None),
                    nees_survivors_lo=(n_sur.ci_lo if n_sur else None),
                    nees_survivors_hi=(n_sur.ci_hi if n_sur else None),
                    n_survivors=(n_sur.n_samples if n_sur else 0),
                    n_runs=n_all.n_samples,
                    nis=s_all.statistic, nis_lo=s_all.ci_lo, nis_hi=s_all.ci_hi,
                    nis_res=s_all.resolution, nis_verdict=s_all.verdict.value,
                    nis_target=s_all.target,
                    track_loss=rep.robustness.divergence_rate,
                    seconds=time.perf_counter() - t0,
                )
            )
            key = f"{int(p0)}_{est}"
            curves[f"nees_{key}"] = np.nanmean(mc.nees, axis=0)
            curves[f"nis_{key}"] = np.nanmean(mc.nis, axis=0)
            curves["t"] = mc.t

            surv = f"{n_sur.statistic:10.3f} {n_sur.n_samples:5d}" if n_sur else f"{'--':>10} {0:5d}"
            print(
                f"{p0:7.0f} {est:>7} | {n_all.statistic:9.3f} {100 * n_all.resolution:5.1f}% "
                f"{n_all.verdict.value:>5} | {surv} "
                f"| {s_all.statistic:6.3f} {100 * s_all.resolution:4.1f}% {s_all.verdict.value:>5} "
                f"| {100 * rep.robustness.divergence_rate:5.1f}%",
                flush=True,
            )

    # --- departures, stated as numbers rather than left to the reader ---
    by = {(r["p0_pos"], r["estimator"]): r for r in rows}
    print("\nDeparture from each metric's own target (all runs):")
    cols = [e for e in ests if e != "oracle"]
    head = f"{'p0_pos':>7} | " + " ".join(f"{e + ' NEES':>11}" for e in cols) + f" | {'EKF NIS':>8}"
    if "oracle" in ests:
        head += f" | {'oracle':>8}"
    print(head)
    print("-" * len(head))
    for p0 in P0_GRID:
        cells = []
        for e in cols:
            r = by.get((p0, e))
            cells.append(f"{100 * (r['nees'] - 4.0) / 4.0:10.1f}%" if r else f"{'--':>11}")
        line = f"{p0:7.0f} | " + " ".join(cells)
        e = by.get((p0, "ekf"))
        line += f" | {100 * (e['nis'] - e['nis_target']) / e['nis_target']:7.1f}%" if e else ""
        if "oracle" in ests:
            o = by.get((p0, "oracle"))
            line += f" | {100 * (o['nees'] - 4.0) / 4.0:7.1f}%" if o else ""
        print(line)

    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "sweep.json"
    path.write_text(
        json.dumps({"scenario": base.to_dict(), "estimators": ests, "rows": rows}, indent=2) + "\n"
    )
    curve_path = a.out / "sweep_curves.npz"
    np.savez_compressed(curve_path, **curves)
    print(f"\nwrote {path} and {curve_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
