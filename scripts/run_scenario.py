#!/usr/bin/env python3
"""Run a scenario and apply the Phase 1 consistency gate.

    python3 scripts/run_scenario.py scenarios/baseline.json [--runs N] [--seed S]
                                    [--amp DEG] [--out results]

Exits non-zero if the gate does not pass, so it drops straight into CI.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kf2 import Scenario, evaluate, run_monte_carlo  # noqa: E402
from kf2.config import replace  # noqa: E402
from kf2.montecarlo import nis_reference  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", type=pathlib.Path)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--amp", type=float, default=None, help="ownship manoeuvre amplitude [deg]")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    sc = Scenario.load(a.scenario)
    overrides = {}
    if a.runs is not None:
        overrides["mc_runs"] = a.runs
    if a.seed is not None:
        overrides["seed"] = a.seed
    if a.amp is not None:
        overrides["own_manoeuvre_amp_deg"] = a.amp
    if overrides:
        sc = replace(sc, **overrides)

    print(sc.summary())
    print(f"  NIS reference under a {sc.gate_prob:g} gate: {nis_reference(sc):.4f}")
    print()

    t0 = time.perf_counter()
    mc = run_monte_carlo(sc, progress=not a.quiet)
    elapsed = time.perf_counter() - t0

    report = evaluate(mc)
    print()
    print(report.table())
    print(f"\n  ({sc.mc_runs} runs in {elapsed:.1f}s)")

    a.out.mkdir(parents=True, exist_ok=True)
    npz = a.out / f"{sc.name}.npz"
    np.savez_compressed(
        npz,
        nees=mc.nees,
        nis=mc.nis,
        pos_err=mc.pos_err,
        vel_err=mc.vel_err,
        pos_sigma=mc.pos_sigma,
        accepted=mc.accepted,
        truth_mean=mc.truth_mean,
        own_x=mc.ownship.x,
        own_y=mc.ownship.y,
        own_psi=mc.ownship.psi,
        t=mc.t,
    )
    sc.save(a.out / f"{sc.name}.scenario.json")
    print(f"wrote {npz} and {a.out / (sc.name + '.scenario.json')}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
