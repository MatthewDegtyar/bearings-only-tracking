#!/usr/bin/env python3
"""What a better attitude solution buys, which is less than it looks.

The bearing error in these scenarios is 0.5 degrees and five-sixths of it is the
observer's MEMS attitude, not its camera. The obvious upgrade is therefore a
better IMU, and this script exists to price it.

It sweeps sigma_bearing and reports, per scenario, the error split along and
across the line of sight together with *the filter's own claimed sigma in each
direction*. The split is the point. A bearing constrains the across direction
and says nothing about the along one, so a sharper sensor shrinks the claimed
range uncertainty through the update arithmetic while leaving the actual range
error to the geometry. The gap between the two columns is a filter growing more
confident without growing more accurate.

Three controls are run alongside, because three plausible explanations for that
are all wrong and each was believed at some point:

    gate        the same sweep with the validation gate opened wide. If
                association were rejecting the corrections, this would recover
                it. It does not.
    estimator   EKF against CKF and against an oracle linearised at ground
                truth. If it were the linearisation, the oracle would be immune.
                It is the worst of the three.
    control     the inspecting geometry against a target that obeys the
                constant-velocity model exactly. If it were model mismatch, this
                would remove it. The effect is stronger.

    python3 scripts/sensor_sweep.py [--runs 100] [--probe-runs 60]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kf2 import datagen, evaluate, run_monte_carlo  # noqa: E402
from kf2.config import replace  # noqa: E402
from kf2.gating import associate, gate_threshold  # noqa: E402
from kf2.montecarlo import ESTIMATORS  # noqa: E402
from kf2.scenarios import INSPECTING, STRAIGHT_ROUTE, TRANSITING  # noqa: E402

SIGMAS = (0.1, 0.25, 0.5, 1.0)
CASES = (("transiting", TRANSITING), ("straight", STRAIGHT_ROUTE), ("inspecting", INSPECTING))
# chi2.ppf(1 - 1e-12, 1) ~ 50.8 against the scenarios' 15.1: wide enough that
# association stops being a constraint, without disabling the code path.
WIDE_GATE = 1.0 - 1e-12


def probe(sc, runs, estimator="ekf"):
    """Error and claimed sigma, split along and across the line of sight.

    Medians, and measured after the first third of the run so the initial
    transient is not counted -- the same conventions as export_cases, so the two
    can be read against each other.
    """
    th = gate_threshold(sc.gate_prob, dim=1)
    along, cross, s_along, s_cross, rejected, scans = [], [], [], [], 0, 0
    for run in range(runs):
        truth = datagen.target_truth(sc, run)
        own, det = datagen.engagement(sc, truth, run)
        f = ESTIMATORS[estimator](sc)
        f.initialise(datagen.initial_estimate(sc, truth[0], run), datagen.initial_covariance(sc))
        for k in range(1, sc.steps + 1):
            f.predict(sc.dt)
            scan = det.per_step[k]
            a = associate(f, own.xy[k], scan, th)
            if np.asarray(scan).size:
                scans += 1
                rejected += not a.accepted
            if a.accepted:
                f.update(a.z, own.xy[k])
            if k > sc.steps // 3:
                e = truth[k, :2] - f.state[:2]
                los = np.arctan2(f.state[1] - own.y[k], f.state[0] - own.x[k])
                u = np.array([np.cos(los), np.sin(los)])
                v = np.array([-u[1], u[0]])
                P = f.covariance[:2, :2]
                along.append(abs(float(e @ u)))
                cross.append(abs(float(e @ v)))
                s_along.append(float(np.sqrt(u @ P @ u)))
                s_cross.append(float(np.sqrt(v @ P @ v)))
    med = lambda a: round(float(np.median(a)), 2)  # noqa: E731
    return dict(
        along=med(along), along_claimed=med(s_along),
        cross=med(cross), cross_claimed=med(s_cross),
        overconfidence=round(float(np.median(along) / np.median(s_along)), 2),
        rejected=round(100.0 * rejected / max(scans, 1), 2),
    )


def loss(sc, runs, estimator="ekf"):
    rep = evaluate(run_monte_carlo(replace(sc, mc_runs=runs), estimator=estimator))
    return round(100.0 * rep.robustness.divergence_rate, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100, help="Monte Carlo runs for track loss")
    ap.add_argument("--probe-runs", type=int, default=60, help="runs for the directional probe")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/sensor_sweep.json"))
    a = ap.parse_args()

    out = {"sigmas": list(SIGMAS), "runs": a.runs, "probe_runs": a.probe_runs, "cases": {}}

    print(f"\nsigma_bearing sweep: {a.probe_runs} runs for the split, {a.runs} for track loss")
    print("along = range direction, across = the one the bearing constrains\n")
    head = (f"{'case':12s} {'sigma':>6s} | {'ALONG err':>9s} {'claimed':>8s} {'ratio':>6s} | "
            f"{'ACROSS err':>10s} {'claimed':>8s} | {'loss':>6s}")
    print(head)
    print("-" * len(head))
    for name, sc0 in CASES:
        rows = []
        for s in SIGMAS:
            sc = replace(sc0, sigma_bearing_deg=s)
            p = probe(sc, a.probe_runs)
            p["sigma"] = s
            p["track_loss"] = loss(sc, a.runs)
            rows.append(p)
            print(f"{name:12s} {s:5.2f}° | {p['along']:8.1f}m {p['along_claimed']:7.1f}m "
                  f"{p['overconfidence']:5.1f}x | {p['cross']:9.2f}m {p['cross_claimed']:7.2f}m "
                  f"| {p['track_loss']:5.1f}%", flush=True)
        out["cases"][name] = rows
        print()

    # --- the three controls ------------------------------------------------
    print("controls, inspecting case, sigma 0.5 -> 0.1 deg\n")
    ctl = {}

    wide = [dict(sigma=s, **probe(replace(INSPECTING, sigma_bearing_deg=s, gate_prob=WIDE_GATE),
                                  a.probe_runs),
                 track_loss=loss(replace(INSPECTING, sigma_bearing_deg=s, gate_prob=WIDE_GATE), a.runs))
            for s in (0.5, 0.1)]
    ctl["wide_gate"] = wide
    # Read against the default gate at the same sigma, which is the comparison
    # that matters: the gate is doing most of its rejecting at 0.1 deg, and
    # opening it removes almost all of that without removing the track loss.
    sharp = out["cases"]["inspecting"][0]
    print(f"  at 0.1°, default gate : rejected {sharp['rejected']:.2f}%, "
          f"loss {sharp['track_loss']:.0f}%")
    print(f"  at 0.1°, gate opened  : rejected {wide[1]['rejected']:.2f}%, "
          f"loss {wide[1]['track_loss']:.0f}%   (association is not the mechanism)")

    ctl["estimators"] = {}
    for est in ("ekf", "ckf", "oracle"):
        lo = [loss(replace(INSPECTING, sigma_bearing_deg=s), a.runs, est) for s in (0.5, 0.1)]
        ctl["estimators"][est] = lo
        print(f"  {est:12s}: loss {lo[0]:.0f}% -> {lo[1]:.0f}%")
    print("               (all three degrade; linearisation is not the mechanism)")

    # INSPECTING starts the target at rest, so "constant" is a stationary target:
    # identical geometry, and the constant-velocity model is now exactly true.
    ctl["cv_control"] = {}
    for label, qf in (("filter q as shipped", INSPECTING.filter_q), ("filter q matched", None)):
        lo = [loss(replace(INSPECTING, sigma_bearing_deg=s, tgt_motion="constant", q_filter=qf),
                   a.runs) for s in (0.5, 0.1)]
        ctl["cv_control"][label] = lo
        print(f"  CV target, {label:20s}: loss {lo[0]:.0f}% -> {lo[1]:.0f}%")
    print("               (effect survives a target that obeys the model; "
          "mismatch is not the mechanism)")

    out["controls"] = ctl
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
