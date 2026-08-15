#!/usr/bin/env python3
"""Export a set of comparable cases as one bundle, for the comparison viewer.

Each case carries two things that answer different questions:

  * one run, both estimators, for the geometry. What the engagement looks like.
  * a Monte Carlo summary with its resolution, for the claim. What is true.

The second is not optional. A single run cannot settle anything here -- across
surviving runs the cubature filter removes anywhere from +127% to -125% of the
EKF's error depending which one you draw -- so every case states its ensemble
figures and, next to them, the smallest effect that ensemble could have
detected. Where the resolution exceeds the effect, the case is marked
inconclusive rather than quietly displayed as a result.

    python3 scripts/export_cases.py [--runs 200] [--stride 3]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import math
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kf2 import datagen, evaluate, run_monte_carlo  # noqa: E402
from kf2.gating import associate, gate_threshold  # noqa: E402
from kf2.montecarlo import ESTIMATORS  # noqa: E402
from kf2.config import replace  # noqa: E402
from kf2.scenarios import INSPECTING, PURSUING, STRAIGHT_ROUTE, TRANSITING  # noqa: E402

from export_sim import trace  # noqa: E402


def cases():
    """The comparison set.

    One argument in three cases. Range is recoverable from bearings only when
    two things hold at once: the observer accelerates, so its own known motion
    supplies a length scale, and the target holds a course, so that scale is not
    absorbed into the target's own unpredictability. Each case removes one.
    """
    return [
        dict(key="inspecting", label="Intruder inspecting",
             blurb="A hand-flown intruder that keeps changing heading absorbs the parallax the "
                   "patrol route generates, so the sentry holds cross-range to a few metres and "
                   "never recovers range.",
             sc=INSPECTING),
        dict(key="transiting", label="Intruder transiting",
             blurb="The same patrol and geometry with the intruder crossing on a steady course, "
                   "which makes the filter's constant-velocity assumption true and is worth "
                   "roughly five times in range accuracy.",
             sc=TRANSITING),
        dict(key="straight", label="Patrol flies straight",
             blurb="A steady intruder again, but with no weave in the patrol route there is no "
                   "parallax and range degrades anyway.",
             sc=STRAIGHT_ROUTE),
        dict(key="pursuing", label="Patrol turns to pursue",
             blurb="Turning toward the latest bearing keeps the target in frame and closes the "
                   "range from 291 m to 97 m, but flying at something accelerates along the "
                   "line of sight rather than across it, so the sentry collects twice the "
                   "measurements and learns less about distance.",
             sc=PURSUING),
    ]


def summarise(sc, runs, stride):
    """Ensemble figures for one case, each with the resolution beside it.

    Also returns the per-step mean NEES for each estimator, taken from the same
    Monte Carlo, so the curve and the number can never disagree.
    """
    out, curves = {}, {}
    for est in ("ekf", "ckf", "oracle"):
        mc = run_monte_carlo(replace(sc, mc_runs=runs), estimator=est)
        curves[est] = [round(float(v), 3) for v in np.nanmean(mc.nees, axis=0)[::stride]]
        rep = evaluate(mc)
        n = next(c for c in rep.consistency.criteria if c.name.startswith("NEES bias"))
        s_ = next(c for c in rep.consistency.criteria if c.name.startswith("NIS bias"))
        ns = None
        if rep.consistency_survivors:
            ns = next((c for c in rep.consistency_survivors.criteria
                       if c.name.startswith("NEES bias")), None)
        stat = ns.statistic if ns and np.isfinite(ns.statistic) else n.statistic
        res = ns.resolution if ns else n.resolution
        out[est] = dict(
            nees=round(float(stat), 4),
            nees_dep=round(100.0 * (stat - 4.0) / 4.0, 2),
            nees_res=round(100.0 * float(res), 2),
            nis=round(float(s_.statistic), 5),
            nis_dep=round(100.0 * (s_.statistic - s_.target) / s_.target, 3),
            nis_res=round(100.0 * float(s_.resolution), 3),
            track_loss=round(100.0 * rep.robustness.divergence_rate, 2),
        )
    e = out["ekf"]
    # Two separate questions. "real" asks whether NEES departed by more than
    # this run count can resolve; "hidden" asks whether NIS stayed below its own
    # resolution, meaning the runtime check could not see it. The finding is
    # both at once, and an earlier version required NIS to move, which marked
    # exactly the successful cases as failures.
    e["real"] = bool(abs(e["nees_dep"]) > e["nees_res"])
    e["hidden"] = bool(abs(e["nis_dep"]) < e["nis_res"])
    e["conclusive"] = e["real"]
    if not e["real"]:
        e["decoupling"], e["decoupling_bound"] = None, None
    elif e["hidden"]:
        e["decoupling"], e["decoupling_bound"] = None, round(abs(e["nees_dep"]) / e["nis_res"], 1)
    else:
        e["decoupling"] = round(e["nees_dep"] / e["nis_dep"], 1)
        e["decoupling_bound"] = None
    ck = out["ckf"]
    e["removed"] = (round(100.0 * (1.0 - ck["nees_dep"] / e["nees_dep"]))
                    if abs(e["nees_dep"]) > 1e-9 else None)
    return out, curves


def directional_error(sc, runs):
    """Estimate error split along and across the line of sight [m].

    The right metric for a bearings-only tracker, and the one NEES cannot give.
    A bearing constrains the across direction and says nothing about the along
    direction, so a single total error hides the whole story: these scenarios
    point at the intruder to within metres while being hundreds of metres wrong
    about how far away it is.

    Measured after the first third of the run so the initial transient is not
    counted, and reported as medians because the along-sight distribution has a
    long tail wherever range is weakly observable.

    ``along_start`` and ``along_end`` are the same statistic over the first and
    last 15% of the run instead, which is what answers "did the filter learn the
    range or not". A case can end worse than it started; two of these do, and a
    figure quoted only after the transient hides that.

    ``in_frame`` is the fraction of scans holding the target at all. It belongs
    beside the error because the geometry that ranges well and the geometry that
    keeps the target in frame are not the same one, and the README's claim to
    that effect has to come from somewhere.
    """
    th = gate_threshold(sc.gate_prob, dim=1)
    n0, n1 = int(0.15 * sc.steps), int(0.85 * sc.steps)
    out = {}
    seen = []
    for name in ("ekf", "ckf"):
        along, cross, early, late = [], [], [], []
        for run in range(runs):
            truth = datagen.target_truth(sc, run)
            own, det = datagen.engagement(sc, truth, run)
            if name == "ekf":
                seen.append(np.mean([np.asarray(s).size > 0 for s in det.per_step[1:]]))
            f = ESTIMATORS[name](sc)
            f.initialise(datagen.initial_estimate(sc, truth[0], run), datagen.initial_covariance(sc))
            for k in range(1, sc.steps + 1):
                f.predict(sc.dt)
                a = associate(f, own.xy[k], det.per_step[k], th)
                if a.accepted:
                    f.update(a.z, own.xy[k])
                e = truth[k, :2] - f.state[:2]
                los = np.arctan2(f.state[1] - own.y[k], f.state[0] - own.x[k])
                u = np.array([np.cos(los), np.sin(los)])
                if k > sc.steps // 3:
                    along.append(abs(float(e @ u)))
                    cross.append(abs(float(e @ np.array([-u[1], u[0]]))))
                if k <= n0:
                    early.append(abs(float(e @ u)))
                elif k >= n1:
                    late.append(abs(float(e @ u)))
        a0, a1 = float(np.median(early)), float(np.median(late))
        out[name] = dict(
            along=round(float(np.median(along)), 1),
            cross=round(float(np.median(cross)), 1),
            along_p90=round(float(np.percentile(along, 90)), 1),
            cross_p90=round(float(np.percentile(cross, 90)), 1),
            along_start=round(a0, 1), along_end=round(a1, 1),
            range_improved_pct=round(100.0 * (1.0 - a1 / a0), 1),
        )
    out["in_frame"] = round(100.0 * float(np.mean(seen)), 1)
    return out


def geometry(sc, run):
    """Cheap descriptors the sidebar can show without opening the case."""
    truth = datagen.target_truth(sc, run)
    own, _ = datagen.engagement(sc, truth, run)
    r = datagen.target_range(truth, own)
    b = datagen.bearing(truth[:, :2], own.xy)
    return dict(
        range_min=round(float(r.min())), range_max=round(float(r.max())),
        bearing_sweep=round(float(np.degrees(np.ptp(datagen.wrap_pi(b - b[0])))), 1),
        own_manoeuvre=sc.own_manoeuvre_amp_deg, p0=sc.p0_pos,
        duration=sc.duration, sigma_bearing_deg=sc.sigma_bearing_deg,
        fov=sc.sensor_fov_deg, tgt_manoeuvre=sc.tgt_manoeuvre_amp_deg,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--run", type=int, default=0, help="which run to draw")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--curves", type=pathlib.Path, default=pathlib.Path("results/sweep_curves.npz"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/cases.json"))
    a = ap.parse_args()

    bundle = []
    for c in cases():
        t0 = time.perf_counter()
        sc = c["sc"]
        traces = {e: trace(sc, a.run, e) for e in ("ekf", "ckf")}
        if a.stride > 1:
            for tr in traces.values():
                for k in ("t", "range", "pos_err", "detected", "in_fov", "bearing", "nees", "boresight"):
                    tr[k] = tr[k][:: a.stride]
                for k in ("own", "truth", "est", "cov"):
                    tr[k] = {kk: vv[:: a.stride] for kk, vv in tr[k].items()}

        stats, curves = summarise(sc, a.runs, a.stride)
        ens = dict(p0=sc.p0_pos, runs=a.runs, nees=curves)
        direction = directional_error(sc, min(a.runs, 120))
        bundle.append(dict(
            key=c["key"], label=c["label"], blurb=c["blurb"],
            scenario=dict(name=sc.name, dt=sc.dt * a.stride,
                          duration=sc.duration, own_speed=sc.own_speed,
                          tgt_speed=sc.tgt_speed, sensor_fov_deg=sc.sensor_fov_deg,
                          sigma_bearing_deg=sc.sigma_bearing_deg, p0_pos=sc.p0_pos,
                          pd=sc.pd, pd_half_range=sc.pd_half_range,
                          manoeuvre_deg=sc.tgt_manoeuvre_amp_deg, q=sc.q),
            geometry=geometry(sc, a.run), stats=stats, ensemble=ens,
            direction=direction, traces=traces,
        ))
        e = stats["ekf"]
        d = (f"D {e['decoupling']:6.1f}" if e["decoupling"] is not None
             else (f"D >{e['decoupling_bound']:5.1f}" if e["decoupling_bound"] is not None else "D    n/a"))
        verdict = ("error real, hidden from NIS" if e["real"] and e["hidden"]
                   else "error real, NIS saw it" if e["real"]
                   else "NOT RESOLVED")
        print(f"  {c['label']:26s} NEES {e['nees_dep']:+7.1f}% (res {e['nees_res']:5.1f}%)  "
              f"NIS {e['nis_dep']:+6.2f}% (res {e['nis_res']:5.2f}%)  {d}  "
              f"{verdict:28s} [{time.perf_counter() - t0:.0f}s]"
              f"  along {direction['ekf']['along']:.0f} m, across {direction['ekf']['cross']:.1f} m",
              flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(runs=a.runs, cases=bundle), separators=(",", ":")))
    print(f"\nwrote {a.out} ({a.out.stat().st_size / 1024:.0f} KB, {len(bundle)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
