#!/usr/bin/env python3
"""Export one run of a scenario as JSON, for visualisation.

``run_trial`` keeps only the scalars the statistics need, so the estimate and
its covariance are not retained. This walks the same loop and records them.

It deliberately reuses ``datagen``, ``ESTIMATORS`` and ``associate`` rather than
reimplementing any of it: a visualisation that quietly disagreed with the
simulator would be worse than no visualisation. The only thing added here is
recording.

    python3 scripts/export_sim.py [--scenario drone] [--run 0] [--out results/sim.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kf2 import datagen  # noqa: E402
from kf2.evaluation import nees_of  # noqa: E402
from kf2.gating import associate, gate_threshold  # noqa: E402
from kf2.montecarlo import ESTIMATORS, ORACLE_ESTIMATORS  # noqa: E402
from kf2.scenarios import get  # noqa: E402


def trace(sc, run: int, estimator: str) -> dict:
    truth = datagen.target_truth(sc, run)
    own, detections = datagen.engagement(sc, truth, run)
    threshold = gate_threshold(sc.gate_prob, dim=1)

    filt = ESTIMATORS[estimator](sc)
    filt.initialise(datagen.initial_estimate(sc, truth[0], run), datagen.initial_covariance(sc))
    oracle = estimator in ORACLE_ESTIMATORS

    n = sc.steps + 1
    est = np.zeros((n, 4))
    cov = np.zeros((n, 2, 2))
    nees = np.full(n, np.nan)
    accepted = np.zeros(n, dtype=bool)
    measured = np.full(n, np.nan)

    def record(k):
        est[k] = filt.state
        cov[k] = filt.covariance[:2, :2]
        nees[k] = nees_of(truth[k] - filt.state, filt.covariance)

    record(0)
    for k in range(1, n):
        filt.predict(sc.dt)
        x_lin = truth[k] if oracle else None
        a = associate(filt, own.xy[k], detections.per_step[k], threshold, x_lin)
        if a.accepted:
            filt.update(a.z, own.xy[k], x_lin)
            accepted[k] = True
            measured[k] = a.z
        record(k)

    true_b = datagen.bearing(truth[:, :2], own.xy)
    vis = datagen.in_field_of_view(sc, true_b, own)
    boresight = datagen.sensor_boresight(sc, true_b, own)
    rng = datagen.target_range(truth, own)
    err = np.hypot(*(truth[:, :2] - est[:, :2]).T)

    def r3(a):
        return [round(float(v), 3) for v in np.asarray(a).ravel()]

    return dict(
        estimator=estimator,
        t=r3(own.t),
        own=dict(x=r3(own.x), y=r3(own.y), psi=r3(own.psi)),
        # Where the sensor is actually pointing, which is the platform heading
        # only when the sensor is bolted to the airframe.
        boresight=[round(float(v), 5) for v in boresight],
        truth=dict(x=r3(truth[:, 0]), y=r3(truth[:, 1])),
        est=dict(x=r3(est[:, 0]), y=r3(est[:, 1])),
        # Upper triangle of the position covariance, for the 1-sigma ellipse.
        cov=dict(xx=r3(cov[:, 0, 0]), xy=r3(cov[:, 0, 1]), yy=r3(cov[:, 1, 1])),
        detected=[bool(v) for v in accepted],
        in_fov=[bool(v) for v in vis],
        bearing=[None if not np.isfinite(v) else round(float(v), 6) for v in measured],
        range=r3(rng),
        pos_err=r3(err),
        nees=[None if not np.isfinite(v) else round(float(v), 4) for v in nees],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="drone")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--estimators", default="ekf")
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth step")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/sim.json"))
    ap.add_argument("--curves", type=pathlib.Path, default=pathlib.Path("results/sweep_curves.npz"),
                    help="ensemble NEES curves from scripts/sweep.py")
    ap.add_argument("--curves-p0", type=int, default=600,
                    help="which initial-uncertainty point to take the ensemble from")
    a = ap.parse_args()

    sc = get(a.scenario)
    traces = {e: trace(sc, a.run, e) for e in a.estimators.split(",")}

    if a.stride > 1:
        for tr in traces.values():
            for key in ("t", "range", "pos_err", "detected", "in_fov", "bearing", "nees"):
                tr[key] = tr[key][:: a.stride]
            for key in ("own", "truth", "est", "cov"):
                tr[key] = {k: v[:: a.stride] for k, v in tr[key].items()}

    # The ensemble. A single run cannot show the correction: across surviving
    # runs the cubature filter removes anywhere from +127% to -125% of the
    # EKF's error depending on which one you pick, and the median run shows it
    # doing worse. The 43% figure is a mean over 400 runs, so the only honest
    # way to show it is to show the mean.
    ensemble = None
    if a.curves.exists():
        import numpy as _np
        cur = _np.load(a.curves)
        want = {e: f"nees_{a.curves_p0}_{e}" for e in ("ekf", "ckf", "oracle")}
        have = {e: k for e, k in want.items() if k in cur.files}
        if have:
            step = max(1, len(cur["t"]) // 400)
            ensemble = dict(
                p0=a.curves_p0,
                t=[round(float(v), 2) for v in cur["t"][::step]],
                nees={e: [round(float(v), 4) for v in cur[k][::step]] for e, k in have.items()},
            )
            print(f"  ensemble: p0={a.curves_p0} m, {', '.join(have)} "
                  f"({len(ensemble['t'])} points)")
        else:
            print(f"  no ensemble curves for p0={a.curves_p0}; have "
                  f"{sorted({k.split('_')[1] for k in cur.files if k.startswith('nees_')})}")

    payload = dict(
        ensemble=ensemble,
        scenario=dict(
            name=sc.name, dt=sc.dt * a.stride, steps=len(traces[list(traces)[0]]["t"]),
            duration=sc.duration, own_speed=sc.own_speed, tgt_speed=sc.tgt_speed,
            sensor_fov_deg=sc.sensor_fov_deg, sigma_bearing_deg=sc.sigma_bearing_deg,
            p0_pos=sc.p0_pos, pd=sc.pd, pd_half_range=sc.pd_half_range,
            manoeuvre_deg=sc.tgt_manoeuvre_amp_deg, q=sc.q,
        ),
        traces=traces,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, separators=(",", ":")))
    kb = a.out.stat().st_size / 1024
    print(f"wrote {a.out} ({kb:.0f} KB, {payload['scenario']['steps']} frames, "
          f"{', '.join(traces)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
