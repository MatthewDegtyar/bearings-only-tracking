#!/usr/bin/env python3
"""Figures, regenerated from committed results with no manual steps.

    python3 scripts/plot.py [--results results]

Reads ``results/sweep.json``, ``results/sweep_curves.npz`` and
``results/gate_sweep.json``; writes PNGs plus a ``.txt`` caption for each into
``results/figures/``. A figure whose inputs are missing is skipped with the
command that would produce them, rather than failing the whole run.

Matplotlib only -- no seaborn, no style packages.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --- palette -------------------------------------------------------------
# Categorical slots 1-3, validated all-pairs on the light surface. Text always
# wears an ink token, never a series colour; identity is carried by the mark.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#e1e0d9"
CRITICAL = "#d03b3b"

plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 1.8,
        "savefig.facecolor": SURFACE,
    }
)

ESTIMATOR_STYLE = {
    "ekf": (S1, "-", "EKF (straight-line approximation)"),
    "ckf": (S2, "-", "CKF (samples the curve; the fix)"),
}


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, path: pathlib.Path, caption: str) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    path.with_suffix(".txt").write_text(caption.strip() + "\n")
    print(f"wrote {path} (+ caption)")


def pct(value, target):
    return 100.0 * (value - target) / target


# ---------------------------------------------------------------------------
# 1. The headline: NEES departs, NIS does not
# ---------------------------------------------------------------------------


def figure_headline(sweep, outdir):
    rows = [r for r in sweep["rows"] if r["estimator"] == "ekf"]
    rows.sort(key=lambda r: r["p0_pos"])
    p0 = np.array([r["p0_pos"] for r in rows])
    n = rows[0]["n_runs"]

    nees = np.array([pct(r["nees_survivors"], 4.0) for r in rows])
    nees_lo = np.array([pct(r["nees_survivors_lo"], 4.0) for r in rows])
    nees_hi = np.array([pct(r["nees_survivors_hi"], 4.0) for r in rows])
    nis = np.array([pct(r["nis"], r["nis_target"]) for r in rows])
    nis_lo = np.array([pct(r["nis_lo"], r["nis_target"]) for r in rows])
    nis_hi = np.array([pct(r["nis_hi"], r["nis_target"]) for r in rows])
    nis_res = np.array([100.0 * r["nis_res"] for r in rows])

    # Two panels sharing an x-axis rather than one. Both quantities are a
    # departure from their *own* target so they could share a y-axis honestly --
    # but at a scale that shows +82% the NIS curve and its resolution band are
    # both invisible, and "flat" versus "flat within its own noise floor" are
    # different claims. Only the second is what the data supports, so the lower
    # panel gives NIS a scale on which its resolution exists. This is a split
    # scale, not a dual axis: the units are identical and the zero lines align.
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8.0, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15], "hspace": 0.12},
    )

    ax.axhline(0.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.fill_between(p0, nees_lo, nees_hi, color=S1, alpha=0.15, lw=0, zorder=2)
    ax.plot(p0, nees, color=S1, marker="o", ms=5, mec=SURFACE, mew=1.2, zorder=4,
            label="NEES (needs the true position; simulation only)")
    ax.plot(p0, nis, color=S2, marker="s", ms=5, mec=SURFACE, mew=1.2, zorder=5,
            label="NIS (needs nothing; works in the field)")
    ax.annotate(f"+{nees[-1]:.0f}%", xy=(p0[-1], nees[-1]), xytext=(-4, 10),
                textcoords="offset points", ha="right", color=S1, fontsize=10,
                fontweight="600")
    ax.set_title("The filter gets it badly wrong. The check you can run in the field cannot tell.")
    ax.set_ylabel("distance from expected value [%]")
    ax.legend(loc="upper left", fontsize=8.5)
    tidy(ax)

    ax2.fill_between(p0, -nis_res, nis_res, color=BAND, lw=0, zorder=0)
    ax2.axhline(0.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax2.fill_between(p0, nis_lo, nis_hi, color=S2, alpha=0.20, lw=0, zorder=2)
    ax2.plot(p0, nis, color=S2, marker="s", ms=4.5, mec=SURFACE, mew=1.0, zorder=3)
    ax2.set_ylim(-2.4, 2.4)
    ax2.annotate(f"+{nis[-1]:.1f}%", xy=(p0[-1], nis[-1]), xytext=(-4, 7),
                 textcoords="offset points", ha="right", color=S2, fontsize=9,
                 fontweight="600")
    ax2.annotate("NIS on its own scale. Grey band is the smallest effect this test could detect.",
                 xy=(0.015, 0.10), xycoords="axes fraction", color=MUTED, fontsize=7.5)
    ax2.set_xlabel("how uncertain the filter is at the start [m, one standard deviation per axis]")
    ax2.set_ylabel("same units [%]")
    tidy(ax2)

    save(
        fig,
        outdir / "1_headline_decoupling.png",
        f"""
Figure 1. Departure of each check from its expected value as the filter's starting
uncertainty is increased, for the ordinary EKF, over {n} runs at each
point.
""",
    )


# ---------------------------------------------------------------------------
# 2. The fix
# ---------------------------------------------------------------------------


def figure_fix(sweep, outdir):
    by_est: dict[str, list] = {}
    for r in sweep["rows"]:
        by_est.setdefault(r["estimator"], []).append(r)
    for rows in by_est.values():
        rows.sort(key=lambda r: r["p0_pos"])

    # Two panels because a filter can fail in two different currencies. The NEES
    # panel compares surviving tracks, which is the like-for-like consistency
    # comparison; on its own it would hide a filter that scores well by losing
    # the hard tracks. Reporting only the survivor panel would be exactly the
    # selection effect this project complains about elsewhere.
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.8, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.13},
    )
    n = sweep["rows"][0]["n_runs"]

    if "oracle" in by_est:
        o = by_est["oracle"]
        ax.plot(
            [r["p0_pos"] for r in o],
            [pct(r["nees_survivors"], 4.0) for r in o],
            color=MUTED, ls=(0, (5, 3)), lw=1.6, zorder=2,
            label="best possible (filter given the true position; not buildable)",
        )
        ax2.plot([r["p0_pos"] for r in o], [100 * r["track_loss"] for r in o],
                 color=MUTED, ls=(0, (5, 3)), lw=1.6, zorder=2)

    for est, (colour, ls, label) in ESTIMATOR_STYLE.items():
        if est not in by_est:
            continue
        rows = by_est[est]
        x = np.array([r["p0_pos"] for r in rows])
        y = np.array([pct(r["nees_survivors"], 4.0) for r in rows])
        ax.plot(x, y, color=colour, ls=ls, marker="o", ms=5, mec=SURFACE, mew=1.2,
                zorder=4, label=label)
        ax.annotate(f"{y[-1]:.0f}%", xy=(x[-1], y[-1]), xytext=(6, -2),
                    textcoords="offset points", color=colour, fontsize=8.5, fontweight="600")
        loss = np.array([100 * r["track_loss"] for r in rows])
        ax2.plot(x, loss, color=colour, ls=ls, marker="o", ms=4.5, mec=SURFACE, mew=1.0,
                 zorder=4)
        ax2.annotate(f"{loss[-1]:.0f}%", xy=(x[-1], loss[-1]), xytext=(6, -2),
                     textcoords="offset points", color=colour, fontsize=8.5,
                     fontweight="600")

    if "ekf" in by_est and "ckf" in by_est:
        # Sits in the clear band between the oracle floor and zero, so it cannot
        # collide with either the curves above it or the axis below.
        x0 = by_est["ekf"][0]["p0_pos"]
        ax.annotate("at 300 m the EKF is already honest, so there is nothing to recover",
                    xy=(x0 + 8, 0.55), color=MUTED, fontsize=7.5, va="center")
    ax.axhline(0.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_yticks([0, 5, 10, 20, 50, 100])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # Title states the dependence rather than an average: recovery runs from
    # nothing at the benign end to most of it at the wide end.
    ax.set_title("Sampling the curve helps in proportion to the size of the error.")
    ax.set_ylabel("how far NEES sits from expected [%]\n(surviving tracks; log-like scale)")

    ax.legend(loc="upper left", fontsize=8)
    ax.margins(x=0.12)
    tidy(ax)

    ax2.set_ylabel("tracks lost [%]")
    ax2.set_xlabel("how uncertain the filter is at the start [m, one standard deviation per axis]")
    ax2.annotate("track loss is a separate failure, and the correction does not address it",
                 xy=(0.015, 0.78), xycoords="axes fraction", color=MUTED, fontsize=7.5)
    tidy(ax2)

    ckf = by_est.get("ckf", [])
    ekf = by_est.get("ekf", [])
    rec = ""
    if ckf and ekf:
        d_e, d_c = pct(ekf[-1]["nees_survivors"], 4.0), pct(ckf[-1]["nees_survivors"], 4.0)
        rec = f"At the widest point cubature recovers {100 * (1 - d_c / d_e):.0f}% of the EKF's error."

    save(
        fig,
        outdir / "2_the_fix.png",
        f"""
Figure 2. The same departure for the EKF and the cubature filter, run on identical
measurement sequences, {n} runs at each point, with the fraction of tracks lost
shown in the lower panel.
""",
    )


# ---------------------------------------------------------------------------
# 3. NEES against time
# ---------------------------------------------------------------------------


def figure_time(curves, sweep, outdir, p0=600):
    t = curves["t"]
    available = [e for e in ("ekf", "ckf", "oracle") if f"nees_{p0}_{e}" in curves]
    if not available:
        print(f"skip figure 3: no curves for p0={p0}")
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.axhline(4.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    for est in available:
        y = curves[f"nees_{p0}_{est}"]
        if est == "oracle":
            ax.plot(t, y, color=MUTED, ls=(0, (5, 3)), lw=1.5, zorder=2,
                    label="best possible (not buildable)")
        else:
            colour, ls, label = ESTIMATOR_STYLE[est]
            ax.plot(t, y, color=colour, lw=1.4, zorder=3, label=label)

    ax.annotate("expected value", xy=(t[len(t) // 12], 4.0), xytext=(0, -14),
                textcoords="offset points", color=INK_2, fontsize=8)
    n = next(r["n_runs"] for r in sweep["rows"] if r["p0_pos"] == p0)
    ax.set_title(f"When it goes wrong, and whether it comes back. Starting uncertainty {p0} m.")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("NEES, averaged over runs (4 is expected)")
    ax.legend(loc="upper right", fontsize=8.5)
    tidy(ax)
    fig.tight_layout()

    ekf_y = curves.get(f"nees_{p0}_ekf")
    detail = ""
    if ekf_y is not None:
        pk = int(np.argmax(ekf_y))
        detail = (
            f"The EKF peaks at {ekf_y.max():.2f} near t = {t[pk]:.0f} s and decays to "
            f"{ekf_y[-1]:.2f} by the end of the run."
        )
    save(
        fig,
        outdir / "3_nees_vs_time.png",
        f"""
Figure 3. Mean NEES over the course of a run at {p0} m of starting uncertainty,
averaged over {n} runs, with the expected value of 4 marked for reference.
""",
    )


# ---------------------------------------------------------------------------
# 4. The gate effect
# ---------------------------------------------------------------------------


def figure_gate(gate, outdir):
    rows = sorted(gate["rows"], key=lambda r: r["gate_prob"])
    g = np.array([r["gate_prob"] for r in rows])
    nis = np.array([r["nis"] for r in rows])
    ref = np.array([r["nis_truncated_ref"] for r in rows])
    nees = np.array([r["nees"] for r in rows])
    n = rows[0]["n_runs"]

    # Plot against the gate's *rejection* rate on a log axis. Linear in gate
    # probability crams 0.99 through 0.9999 into the last few percent of the
    # width, which is exactly the operational range.
    x = 1.0 - g
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.3))

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        # Ticks carry the gate probability because that is the knob an engineer
        # sets; the *spacing* is log in rejection rate so the operational range
        # near 1.0 is not crushed into the last few percent of the width.
        ax.set_xlabel("how much the tracker keeps (spacing is log in what it rejects)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in g], fontsize=8)
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.invert_xaxis()  # tighter gates to the right, matching "more gating"

    ax1.axhline(1.0, color=CRITICAL, lw=1.4, ls=(0, (4, 3)), zorder=2,
                label="value if you ignore the rejection (wrong)")
    ax1.plot(x, ref, color=MUTED, ls=(0, (5, 3)), lw=1.8, zorder=3,
             label="value accounting for the rejection (expected)")
    ax1.plot(x, nis, color=S1, marker="o", ms=5, mec=SURFACE, mew=1.2, zorder=4,
             label="NIS actually measured")
    ax1.set_title("Rejecting outliers drags NIS low. The right value can be computed exactly.")
    ax1.set_ylabel("average NIS of the measurements kept")
    ax1.legend(loc="lower left", fontsize=8)
    tidy(ax1)

    ax2.axhline(4.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax2.plot(x, nees, color=S2, marker="o", ms=5, mec=SURFACE, mew=1.2, zorder=3)
    ax2.annotate(f"{nees[0]:.2f}", xy=(x[0], nees[0]), xytext=(6, 6),
                 textcoords="offset points", color=S2, fontsize=9, fontweight="600")
    ax2.annotate("expected value", xy=(x[1], 4.0), xytext=(6, 9),
                 textcoords="offset points", color=INK_2, fontsize=8)
    ax2.set_title("...but the correction does nothing for the real error")
    ax2.set_ylabel("NEES (4 is expected)")
    tidy(ax2)

    fig.suptitle(
        "Correcting for rejection fixes the measurement check, not the real error",
        x=0.006, ha="left", fontsize=10.5, color=INK, fontweight="600",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(
        fig,
        outdir / "4_gate_effect.png",
        f"""
Figure 4. Effect of rejecting surprising measurements, for the ordinary EKF at 300 m
of starting uncertainty, over {n} runs at each point, with NIS on the left and NEES
on the right.
""",
    )


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, default=pathlib.Path("results"))
    a = ap.parse_args()

    outdir = a.results / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    def load_json(name, hint):
        p = a.results / name
        if not p.exists():
            print(f"skip: {p} missing. Run: {hint}")
            return None
        return json.loads(p.read_text())

    sweep = load_json("sweep.json", "python3 scripts/sweep.py")
    gate = load_json("gate_sweep.json", "python3 scripts/gate_sweep.py")
    curves_path = a.results / "sweep_curves.npz"
    curves = np.load(curves_path) if curves_path.exists() else None
    if curves is None:
        print(f"skip: {curves_path} missing. Run: python3 scripts/sweep.py")

    if sweep:
        figure_headline(sweep, outdir)
        figure_fix(sweep, outdir)
        if curves is not None:
            figure_time(curves, sweep, outdir)
    if gate:
        figure_gate(gate, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
