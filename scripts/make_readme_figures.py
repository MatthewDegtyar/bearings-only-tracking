#!/usr/bin/env python3
"""README figures, regenerated from committed results with no manual steps.

    python3 scripts/make_readme_figures.py

Reads ``results/cases.json`` and ``results/sweep.json``; writes PNGs into
``docs/img/``. Palette and typography follow ``scripts/plot.py`` so the README
and the report look like the same project.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, Polygon  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"

# Same categorical slots as scripts/plot.py, assigned by entity and never by
# rank, so a figure that drops a series does not repaint the survivors.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CRITICAL = "#d03b3b"

plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def fov_wedge(ax, x, y, psi, half_deg, reach, colour, alpha=0.05):
    """The observable cone, drawn where the camera is actually pointed."""
    a = np.radians(np.linspace(psi - half_deg, psi + half_deg, 24))
    pts = np.column_stack([x + reach * np.cos(a), y + reach * np.sin(a)])
    ax.add_patch(Polygon(np.vstack([[x, y], pts]), closed=True,
                         facecolor=colour, edgecolor="none", alpha=alpha, zorder=0))


def frame(ax, tracks, pad=0.14):
    """Limit the view to where the action is, so the cone is clipped rather than
    left to drive the axes. Equal aspect is kept: this is a plan view."""
    xs = np.concatenate([np.asarray(t["x"], float) for t in tracks])
    ys = np.concatenate([np.asarray(t["y"], float) for t in tracks])
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * (1 + pad)
    ax.set_xlim(cx - half * 1.12, cx + half * 1.12)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")


def engagement(case, path):
    """One run: what the observer flew, where the target was, what the filter thought."""
    tr = case["traces"]["ekf"]
    own, truth, est, cov = tr["own"], tr["truth"], tr["est"], tr["cov"]
    fov_half = case["scenario"]["sensor_fov_deg"] / 2.0

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    frame(ax, [own, truth, est])

    # the cone, sampled sparsely so the plot stays readable. Clipped by the frame.
    for k in range(0, len(own["x"]), 50):
        fov_wedge(ax, own["x"][k], own["y"][k], np.degrees(own["psi"][k]),
                  fov_half, 1400, S3, alpha=0.07)

    # 2-sigma ellipses: 2.45 sigma is the 95% contour in 2-D, 2 sigma is 86%
    for k in range(8, len(est["x"]), 14):
        P = np.array([[cov["xx"][k], cov["xy"][k]], [cov["xy"][k], cov["yy"][k]]])
        w, v = np.linalg.eigh(P)
        w = np.maximum(w, 1e-9)
        ax.add_patch(Ellipse((est["x"][k], est["y"][k]),
                             2 * 2.45 * np.sqrt(w[1]), 2 * 2.45 * np.sqrt(w[0]),
                             angle=np.degrees(np.arctan2(v[1, 1], v[0, 1])),
                             facecolor=S2, edgecolor="none", alpha=0.13, zorder=1))

    ax.plot(own["x"], own["y"], color=S3, lw=2, zorder=3, label="observer (known)")
    ax.plot(truth["x"], truth["y"], color=INK, lw=2, zorder=4, label="target truth")
    ax.plot(est["x"], est["y"], color=S2, lw=2, ls=(0, (4, 2)), zorder=5,
            label="EKF estimate")

    for xs, ys, c in ((own, None, S3), (truth, None, INK), (est, None, S2)):
        ax.plot(xs["x"][0], xs["y"][0], "o", ms=5, color=c, zorder=6)

    ax.annotate("target starts here", (truth["x"][0], truth["y"][0]),
                textcoords="offset points", xytext=(10, -4), color=MUTED, fontsize=8)
    ax.annotate("observer starts here", (own["x"][0], own["y"][0]),
                textcoords="offset points", xytext=(10, -12), color=MUTED, fontsize=8)
    ax.annotate("2σ covariance: long along the line of sight,\nthin across it",
                (est["x"][106], est["y"][106]),
                textcoords="offset points", xytext=(24, 34), color=S2, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=S2, lw=0.8, alpha=0.7))
    ax.annotate("field of view", (own["x"][100], own["y"][100]),
                textcoords="offset points", xytext=(-24, 92), color=S3, fontsize=8)

    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title("One engagement: the filter never receives a distance",
                 color=INK, fontsize=11.5, loc="left", pad=26)
    d = case["direction"]["ekf"]
    ax.text(0, 1.012,
            f"{case['label']}   ·   bearing only, 0.5° noise   ·   "
            f"along-range error {d['along']:.0f} m, cross-range {d['cross']:.1f} m",
            transform=ax.transAxes, color=MUTED, fontsize=8.5, va="bottom")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK_2)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def scenarios_grid(cases, path):
    """Four geometries, the same sensor, four different outcomes."""
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    for i, (case, ax) in enumerate(zip(cases, axes.ravel()), start=1):
        tr = case["traces"]["ekf"]
        own, truth, est = tr["own"], tr["truth"], tr["est"]
        fov_half = case["scenario"]["sensor_fov_deg"] / 2.0
        frame(ax, [own, truth, est])
        for k in range(0, len(own["x"]), 50):
            fov_wedge(ax, own["x"][k], own["y"][k], np.degrees(own["psi"][k]),
                      fov_half, 1400, S3, alpha=0.065)
        ax.plot(own["x"], own["y"], color=S3, lw=1.9, zorder=3)
        ax.plot(truth["x"], truth["y"], color=INK, lw=1.9, zorder=4)
        ax.plot(est["x"], est["y"], color=S2, lw=1.5, ls=(0, (3, 2)), zorder=5)
        ax.plot(own["x"][0], own["y"][0], "o", ms=4, color=S3, zorder=6)
        ax.plot(truth["x"][0], truth["y"][0], "o", ms=4, color=INK, zorder=6)

        d, st = case["direction"]["ekf"], case["stats"]["ekf"]
        ax.set_title(f"{i}. {case['label']}", color=INK, fontsize=10,
                     loc="left", pad=20)
        ax.text(0, 1.02,
                f"range err {d['along']:.0f} m   ·   track loss {st['track_loss']:.0f}%",
                transform=ax.transAxes, color=MUTED, fontsize=8.2, va="bottom")
        ax.tick_params(labelsize=7.5)
        tidy(ax)

    fig.text(0.5, 0.975,
             "observer (green)   ·   target truth (black)   ·   EKF estimate (orange dashed)",
             ha="center", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def ekf_vs_ckf(sweep, path):
    """The comparison the project exists to make, across prior width."""
    rows = sweep["rows"]
    p0s = sorted({r["p0_pos"] for r in rows})
    series = [("ekf", "EKF", S1, "-"), ("ckf", "CKF (FilterPy, degree 3)", S2, "-"),
              ("ckf5", "CKF degree 5", S3, (0, (4, 2)))]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(0, color=AXIS, lw=1.2, zorder=1)
    nudge = {"ekf": 0, "ckf": -11, "ckf5": 7}   # keep the end labels off each other
    for key, label, colour, ls in series:
        dep, res = [], []
        for p in p0s:
            r = next(x for x in rows if x["p0_pos"] == p and x["estimator"] == key)
            dep.append(100 * (r["nees"] / 4.0 - 1))
            res.append(100 * r["nees_res"] / 4.0)
        ax.plot(p0s, dep, color=colour, lw=2, ls=ls, marker="o", ms=5,
                label=label, zorder=4)
        ax.fill_between(p0s, np.array(dep) - np.array(res), np.array(dep) + np.array(res),
                        color=colour, alpha=0.10, lw=0, zorder=2)
        ax.annotate(f"{dep[-1]:.0f}%", (p0s[-1], dep[-1]),
                    textcoords="offset points", xytext=(9, nudge[key] - 3),
                    color=colour, fontsize=9, fontweight="bold")

    ax.set_xlabel("initial position uncertainty, 1σ (m)")
    ax.set_ylabel("NEES departure from its target (%)")
    ax.set_xticks(p0s)
    ax.set_xlim(p0s[0] - 30, p0s[-1] + 95)
    ax.set_title("Overconfidence against prior width, 400 runs per point",
                 color=INK, fontsize=11.5, loc="left", pad=26)
    ax.text(0, 1.012,
            "0% is a perfectly calibrated filter. Shaded band is the smallest "
            "effect each point could resolve.",
            transform=ax.transAxes, color=MUTED, fontsize=8.5, va="bottom")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK_2)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> int:
    res = ROOT / "results"
    if not (res / "cases.json").exists() or not (res / "sweep.json").exists():
        print("missing results; run scripts/export_cases.py and scripts/sweep.py",
              file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    cases = json.loads((res / "cases.json").read_text())["cases"]
    sweep = json.loads((res / "sweep.json").read_text())

    engagement(cases[1], OUT / "engagement.png")
    scenarios_grid(cases, OUT / "scenarios.png")
    ekf_vs_ckf(sweep, OUT / "ekf-vs-ckf.png")
    for p in sorted(OUT.glob("*.png")):
        print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
