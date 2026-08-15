# Bearings-only tracking on a low-cost drone

A test harness for camera-only tracking on small UAS, where the sensor gives
direction and never distance. It is scoped to the regime a cheap airframe works
in — a few hundred metres, a consumer camera, an attitude solution that is the
real error source — and built to be run: a scenario is a dataclass, and 400
simulated flights come back in 14 seconds carrying the smallest effect they
could have detected.

**[Run the scenarios in your browser](https://matthewdegtyar.github.io/bearings-only-tracking/)**
— four geometries, both filters with their uncertainty ellipses, the ensemble
comparison underneath. No install, nothing to build.

![One engagement, plan view](docs/img/engagement.png)

The observer is never told how far away the target is. Range has to come out of
the geometry, which is why the ellipses are long along the line of sight and
thin across it.

## What it models

A sentry drone on a patrol route watching an intruder it can only see. A run is
30 s at 20 Hz with the observer at 8 m/s and the target between 97 and 291 m
away — the envelope a consumer 4K camera can hold, since a 0.5 m airframe is 16
pixels across at 97 m and 5 pixels at 291 m.

Bearing error is 0.5°, five-sixths of it the observer's MEMS attitude solution
rather than its optics — centroiding a three-pixel blob is good to 0.006°. That
is the low-cost part of a low-cost drone, and it pays to know which part it is
before trying to fix it.

## Writing a scenario

Everything that can change a number lives on one frozen `Scenario` dataclass, so
a variant is `replace(TRANSITING, p0_pos=120.0)` and nothing that affects a
result is buried in a module.

```python
from kf2 import evaluate, run_monte_carlo
from kf2.config import replace
from kf2.health import check
from kf2.scenarios import TRANSITING

sc = replace(TRANSITING, sigma_bearing_deg=0.25, own_manoeuvre_amp_deg=60.0)
print(check(sc))                                    # viable? 0.5 s
report = evaluate(run_monte_carlo(sc))              # 400 runs, 14 s
```

| step | cost |
|---|---|
| screen a scenario for viability | 0.5 s |
| one estimator, 400 runs | 14 s |
| three estimators, four scenarios, with traces | 5.5 min |
| full test suite | 4.5 min |

`check()` screens on four cheap numbers — detection rate, range span, bearing
sweep, track loss — because every scenario written for this project was broken
at least once in a way one of them would have caught.

Four things keep the answers honest: the simulator and the filters share no
code, estimators see byte-identical measurements, every result carries the
smallest effect it could have detected, and the consistency test is itself
tested against a filter correct by construction.

## Three things it found

### Ranging geometry and detection geometry are not the same geometry

![The four scenarios](docs/img/scenarios.png)

The one that looks cleanest is not the one that tracks best. Case 3 flies dead
straight and improves range error 30% over its starting guess; case 2 differs
only in that its observer weaves, and improves 71%. Case 4 pursues, which is
what an operator would do: it holds the target in frame for the whole run
against roughly half elsewhere, and still ends 63% *worse* on range than it
started, because flying at something accelerates along the line of sight — the
one direction that teaches nothing about distance. Case 1, the hand-flown
intruder, loses the track in 40% of flights. (Range over 120 runs, track loss
over 400, from `results/cases.json`.)

### A better sensor buys confidence, not accuracy

The obvious upgrade for a 0.5° attitude error is a better IMU. Sharpening the
bearing 5× to 0.1°, everything else fixed (`scripts/sensor_sweep.py`):

| | range error | filter's claimed range σ | cross-range error |
|---|---|---|---|
| transiting, observer weaves | 9.3 → 7.9 m | 19.1 → 15.1 m | 1.00 → 0.49 m |
| patrol flies straight | 28.4 → 31.1 m | 27.7 → 22.0 m | 1.07 → 0.54 m |
| intruder inspecting | 51.8 → 88.8 m | 42.7 → 36.7 m | 2.07 → 3.66 m |

Cross-range halves, as it should. Range does not, and gets worse in the two
geometries that cannot recover it — while the claimed range uncertainty shrinks
in all three. Overconfidence goes 1.2× to 2.4× for inspecting and holds at 0.5×
only for transiting, where the weave makes range observable; track loss follows,
37% to 58%. The same script rules out three explanations: opening the gate does
not recover it (rejections 11.9% → 1.8%, loss only 58% → 53%), the cubature
filter and a ground-truth-linearised oracle degrade identically, and the effect
is *stronger* against a control target that obeys the motion model exactly. The
bearing carries no range information, so sharpening it sharpens the covariance
and leaves the error where it was.

### The cubature filter earns its place only when you initialise badly

![NEES departure against prior width](docs/img/ekf-vs-ckf.png)

Both are fine when the starting guess is good and separate as it worsens: at
1000 m of prior uncertainty the EKF understates its own error by 122%, the
cubature filter by 15%. Degree 5 tracks the library's degree 3 closely, so the
extra points do not pay for themselves. This sweep is the one part still at the
old kilometre scale — read it as a statement about prior width, not this
airframe.

## The estimators are off the shelf

Deliberately — a comparison is only worth reading if it measures the algorithms
rather than my implementation of them. Both come from
[FilterPy](https://github.com/rlabbe/filterpy), `kf2/filters.py` is a thin
adapter, and every departure from stock behaviour is listed in
`kf2.filters.DEVIATIONS` with its reason and a test. The one modified estimator,
`kf2/ckf5.py`, is registered separately so nothing can compare a library filter
against a modified one while calling both "the CKF".

## Running it

```sh
pip install numpy scipy filterpy weasyprint pytest

python3 -m pytest -q                                   # 4.5 min
python3 scripts/export_cases.py                        # the four scenarios
python3 scripts/sensor_sweep.py                        # what a better IMU buys
python3 scripts/sweep.py                               # initial-uncertainty sweep
python3 scripts/make_readme_figures.py                 # the figures above
python3 scripts/make_sim_report.py                     # rebuild the report
python3 scripts/make_viz.py --data results/cases.json \
    --out viz/compare.html --template scripts/viz_compare.html --serve --pages
```

`--pages` writes `docs/index.html`, the GitHub Pages entry point: one
self-contained file with the run data inlined. Every number in this README and
in [`report/simulator-report.pdf`](report/simulator-report.pdf) is read back
from `results/`, so neither can drift from the data.

## Layout

`kf2/config.py` holds the `Scenario` dataclass — everything that can change a
number. `kf2/scenarios.py` names the geometries and argues for each one.
`kf2/datagen.py` builds truth and measurements and shares no code with
`kf2/filters.py`, the FilterPy adapters. `kf2/evaluation.py` turns runs into
NEES, NIS and resolution; `kf2/health.py`, `kf2/gating.py` and `kf2/run.py` are
the screening, association and filter loop.
