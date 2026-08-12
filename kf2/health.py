"""Is this scenario worth running?

Every scenario in this project's history has been broken at least once, and in
each case the breakage was visible in a handful of cheap numbers that nobody
computed until after a full sweep had been run and interpreted:

  * a 40 degree target manoeuvre that lost 100 per cent of tracks
  * a 60 degree fixed aperture that detected the target on 0 per cent of scans
  * a 180 degree aperture that was neither realistic nor effective
  * a sentry engagement that lost 69.5 per cent of tracks, whose survivor
    statistics were then quoted as a result
  * two "different" cases that were byte-identical because an override was a
    no-op against the default

None of those needed a Monte Carlo to catch. They needed someone to look at
detection rate, range span, bearing sweep and track loss before believing
anything. This module does that, so a scenario can be rejected in seconds
rather than after an hour of interpretation.

    from kf2.health import check
    print(check(MY_SCENARIO))
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import datagen
from .config import Scenario, replace


@dataclass(frozen=True)
class Finding:
    ok: bool
    name: str
    value: str
    note: str

    def __str__(self) -> str:
        return f"  [{'ok' if self.ok else '!!'}] {self.name:22s} {self.value:>18}   {self.note}"


@dataclass(frozen=True)
class Health:
    scenario: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(f.ok for f in self.findings)

    def __str__(self) -> str:
        head = f"{self.scenario}: {'usable' if self.ok else 'NOT USABLE'}"
        return "\n".join([head, *(str(f) for f in self.findings)])


def check(
    sc: Scenario,
    runs: int = 12,
    *,
    min_detect: float = 0.25,
    max_gap_frac: float = 0.35,
    min_sweep_deg: float = 15.0,
    max_track_loss: float = 0.25,
    range_bounds: tuple[float, float] = (50.0, 20000.0),
) -> Health:
    """Cheap viability checks, before any statistics are believed.

    The defaults are deliberately loose. They are not a claim that a scenario is
    good, only that it is not degenerate in one of the ways this project has
    already been caught by.
    """
    det, gaps, sweeps, rmins, rmaxs = [], [], [], [], []
    for run in range(runs):
        truth = datagen.target_truth(sc, run)
        own, d = datagen.engagement(sc, truth, run)
        got = np.array([i is not None for i in d.truth_index[1:]])
        det.append(got.mean())
        worst = cur = 0
        for x in got:
            cur = 0 if x else cur + 1
            worst = max(worst, cur)
        gaps.append(worst * sc.dt)
        b = datagen.bearing(truth[:, :2], own.xy)
        sweeps.append(np.degrees(np.ptp(datagen.wrap_pi(b - b[0]))))
        r = datagen.target_range(truth, own)
        rmins.append(r.min())
        rmaxs.append(r.max())

    detect, gap = float(np.mean(det)), float(np.max(gaps))
    sweep = float(np.mean(sweeps))
    rmin, rmax = float(np.min(rmins)), float(np.max(rmaxs))

    out = [
        Finding(detect >= min_detect, "detection rate", f"{100 * detect:.1f}%",
                f"target must be seen on at least {100 * min_detect:.0f}% of scans"),
        Finding(gap <= max_gap_frac * sc.duration, "worst blackout",
                f"{gap:.1f}s of {sc.duration:.0f}s",
                f"no gap longer than {100 * max_gap_frac:.0f}% of the run"),
        Finding(sweep >= min_sweep_deg, "bearing sweep", f"{sweep:.0f}°",
                "without this, range is not observable at all"),
        Finding(range_bounds[0] <= rmin and rmax <= range_bounds[1], "range span",
                f"{rmin:.0f}-{rmax:.0f} m",
                f"expected inside {range_bounds[0]:.0f}-{range_bounds[1]:.0f} m"),
    ]

    # Track loss needs the filter, so it costs more; it is also the check that
    # has caught the most, so it is not optional.
    from .run import track as _track

    lost = sum(_track(replace(sc, mc_runs=runs), run, "ekf").lost for run in range(runs))
    loss = lost / runs
    out.append(Finding(loss <= max_track_loss, "track loss", f"{100 * loss:.0f}% of {runs}",
                       "survivor statistics stop meaning anything above this"))
    return Health(scenario=sc.name, findings=out)


def compare(scenarios: dict[str, Scenario], **kw) -> str:
    """Health for a set of scenarios, plus a check that they actually differ.

    Two cases in the comparison set were once byte-identical, because the
    override that was meant to distinguish them matched the default. Nothing
    in the output revealed it; the numbers simply agreed exactly.
    """
    lines, seen = [], {}
    for name, sc in scenarios.items():
        lines.append(str(check(sc, **kw)))
        key = tuple(sorted((f.name, getattr(sc, f.name)) for f in sc.__dataclass_fields__.values()
                           if f.name != "name"))
        if key in seen:
            lines.append(f"  [!!] {name} is identical to {seen[key]} in every field but the name")
        seen[key] = name
        lines.append("")
    return "\n".join(lines)
