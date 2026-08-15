"""The drone-observation additions: target manoeuvre, sensor aperture, range-dependent detection.

Three properties matter and none of them is obvious from reading the code:

1. Every new knob is off by default, so results recorded before these existed
   still reproduce bit-for-bit. This is asserted rather than assumed.
2. The manoeuvre is a *turn*, not an acceleration: it changes heading and leaves
   speed alone. A bug here would quietly feed energy into the target and the
   filter would be blamed for it.
3. Common random numbers survive the new knobs. If moving the aperture shifted
   the measurement stream, estimator comparisons under different apertures would
   be comparing different data, and every difference would be suspect.
"""

from __future__ import annotations

import numpy as np
import pytest

from kf2 import datagen
from kf2.config import Scenario, replace
from kf2.rng import Stream, stream_rng
from kf2.scenarios import INSPECTING, PURSUING, SCENARIOS, get


# --- 1. defaults are off ----------------------------------------------------


def test_new_knobs_default_to_the_previous_behaviour():
    sc = Scenario()
    assert sc.tgt_manoeuvre_amp_deg == 0.0
    assert sc.sensor_fov_deg == 360.0
    assert sc.pd_half_range is None


def test_manoeuvre_off_reproduces_the_old_truth_bit_for_bit():
    sc = Scenario()
    rng = stream_rng(sc.seed, 3, Stream.PROCESS)
    x0 = np.array([sc.tgt_x0, sc.tgt_y0, sc.tgt_vx0, sc.tgt_vy0])
    old = datagen._integrate_em(sc, datagen._velocity_increments(sc, rng), x0)
    assert np.array_equal(datagen.target_truth(sc, 3), old)


def test_manoeuvre_increments_are_exactly_zero_when_off():
    assert not datagen._manoeuvre_increments(Scenario()).any()


def test_amplitude_alone_does_nothing_without_selecting_the_sinusoid():
    """tgt_motion gates the manoeuvre. Setting only the amplitude used to leave
    three tests asserting nothing at all, so the gate is now tested directly."""
    amp_only = replace(Scenario(), tgt_manoeuvre_amp_deg=30.0)
    assert not datagen._manoeuvre_increments(amp_only).any()
    selected = replace(amp_only, tgt_motion="sinusoid")
    assert datagen._manoeuvre_increments(selected).any()


def test_open_aperture_and_flat_detection_cost_nothing():
    sc = Scenario()
    own = datagen.ownship_track(sc)
    truth = datagen.target_truth(sc, 0)
    b = datagen.bearing(truth[:, :2], own.xy)
    assert datagen.in_field_of_view(sc, b, own).all()
    assert np.array_equal(datagen.detection_probability(sc, truth, own), np.full(sc.steps + 1, sc.pd))


# --- 2. the manoeuvre is a turn ---------------------------------------------


def test_manoeuvre_holds_speed():
    """A heading change must not change speed. Tested with the process noise
    off, since that is the only thing that should alter speed."""
    sc = replace(Scenario(), q=0.0, tgt_motion="sinusoid", tgt_manoeuvre_amp_deg=40.0)
    tr = datagen.target_truth(sc, 0)
    speed = np.hypot(tr[:, 2], tr[:, 3])
    assert np.allclose(speed, sc.tgt_speed, rtol=0, atol=1e-9)


def test_manoeuvre_actually_bends_the_path():
    sc = replace(Scenario(), q=0.0)
    straight = datagen.target_truth(sc, 0)
    bent = datagen.target_truth(replace(sc, tgt_motion="sinusoid", tgt_manoeuvre_amp_deg=30.0), 0)
    assert np.abs(bent[:, :2] - straight[:, :2]).max() > 100.0


def test_manoeuvring_truth_still_matches_the_independent_integrator():
    """The vectorised integrator is only trusted because an independent substep
    loop agrees with it. Injecting the manoeuvre as increments keeps that check
    live; this asserts it did not quietly stop applying."""
    sc = replace(Scenario(), tgt_motion="sinusoid", tgt_manoeuvre_amp_deg=25.0)
    dv = datagen._velocity_increments(sc, stream_rng(sc.seed, 0, Stream.PROCESS))
    dv = dv + datagen._manoeuvre_increments(sc)
    x0 = np.array([sc.tgt_x0, sc.tgt_y0, sc.tgt_vx0, sc.tgt_vy0])
    fast = datagen._integrate_em(sc, dv, x0)
    slow = datagen._integrate_em_reference(sc, dv, x0)
    assert np.abs(fast - slow).max() < 1e-6


def test_target_heading_law_is_centred_on_the_initial_heading():
    sc = replace(Scenario(), tgt_motion="sinusoid", tgt_manoeuvre_amp_deg=20.0)
    psi = datagen.target_heading(sc, np.linspace(0.0, sc.duration, 101))
    assert psi[0] == pytest.approx(sc.tgt_psi0)
    assert np.ptp(psi) == pytest.approx(2 * np.radians(20.0), rel=1e-3)


# --- 3. the sensor ----------------------------------------------------------


def test_narrower_aperture_never_sees_more():
    sc = Scenario()
    own = datagen.ownship_track(sc)
    b = datagen.bearing(datagen.target_truth(sc, 0)[:, :2], own.xy)
    seen = [datagen.in_field_of_view(replace(sc, sensor_fov_deg=f), b, own).sum()
            for f in (360.0, 180.0, 90.0, 45.0)]
    assert seen == sorted(seen, reverse=True)
    assert seen[-1] < seen[0]


def test_detection_probability_halves_at_the_stated_range():
    sc = replace(Scenario(), pd=1.0, pd_half_range=5000.0)
    own = datagen.ownship_track(sc)
    truth = datagen.target_truth(sc, 0)
    r = datagen.target_range(truth, own)
    pd = datagen.detection_probability(sc, truth, own)
    k = int(np.argmin(np.abs(r - 5000.0)))
    assert pd[k] == pytest.approx(0.5, abs=0.01)
    # Monotone in range: closer is never worse.
    assert np.all(np.diff(pd[np.argsort(r)]) <= 1e-12)


def test_target_outside_the_aperture_is_never_detected():
    sc = replace(Scenario(), sensor_fov_deg=30.0, pd=1.0)
    own = datagen.ownship_track(sc)
    truth = datagen.target_truth(sc, 0)
    vis = datagen.in_field_of_view(sc, datagen.bearing(truth[:, :2], own.xy), own)
    det = datagen.generate_detections(sc, truth, own, 0)
    got = np.array([i is not None for i in det.truth_index])
    assert not (got & ~vis).any(), "a detection was produced with the target out of frame"
    assert got[1:].sum() > 0, "aperture rejected everything; the test proves nothing"


# --- 4. common random numbers ----------------------------------------------


@pytest.mark.parametrize("changed", [
    {"sensor_fov_deg": 120.0},
    {"pd_half_range": 6000.0},
    {"sensor_fov_deg": 90.0, "pd_half_range": 6000.0},
])
def test_new_knobs_do_not_desynchronise_the_measurement_stream(changed):
    """Where both scenarios detect the target, the bearing must be identical.
    Otherwise a comparison across apertures is a comparison across datasets."""
    base = replace(Scenario(), pd=0.9)
    own = datagen.ownship_track(base)
    alt = replace(base, **changed)
    d1 = datagen.generate_detections(base, datagen.target_truth(base, 0), own, 0)
    d2 = datagen.generate_detections(alt, datagen.target_truth(alt, 0), own, 0)
    shared = [k for k in range(1, base.steps + 1)
              if d1.truth_index[k] is not None and d2.truth_index[k] is not None]
    assert len(shared) > 50, "too few shared detections for the test to mean anything"
    for k in shared:
        b1 = d1.per_step[k][d1.truth_index[k]]
        b2 = d2.per_step[k][d2.truth_index[k]]
        assert b1 == pytest.approx(b2, abs=0.0)


# --- 5. the scenario itself -------------------------------------------------


def test_scenarios_are_registered_and_validate():
    assert get("inspecting") is INSPECTING
    assert set(SCENARIOS) >= {"inspecting", "transiting", "straight-route", "pursuing"}
    with pytest.raises(ValueError):
        get("nonexistent")


def test_patrol_geometry_is_not_degenerate():
    """The scenario is only useful if the target is usually visible, the range
    stays at drone scale, and the bearing sweeps enough for range to be
    observable at all. Each of those was got wrong at least once."""
    sc = INSPECTING
    own = datagen.ownship_track(sc)
    seen, spreads = [], []
    for run in range(8):
        truth = datagen.target_truth(sc, run)
        r = datagen.target_range(truth, own)
        # Drone scale means what a consumer camera can hold: a 0.5 m airframe is
        # three pixels at 520 m, so a geometry that runs past 320 m is measuring
        # a sensor nobody has. The floor catches the opposite failure, an
        # intruder that wanders into the observer's lap.
        assert 40.0 < r.min() and r.max() < 320.0, f"range {r.min():.0f}-{r.max():.0f} m"
        b = datagen.bearing(truth[:, :2], own.xy)
        spreads.append(np.degrees(np.ptp(datagen.wrap_pi(b - b[0]))))
        det = datagen.generate_detections(sc, truth, own, run)
        seen.append(np.mean([i is not None for i in det.truth_index[1:]]))
    assert np.mean(seen) > 0.4, f"target detected on only {100 * np.mean(seen):.0f}% of steps"
    assert np.mean(spreads) > 20.0, f"bearing sweeps only {np.mean(spreads):.0f} deg; range unobservable"


def test_sporadic_target_hovers_and_changes_legs():
    """The inspecting intruder must actually behave like one: it should spend a
    meaningful fraction of the run stationary and change velocity repeatedly,
    rather than being a smooth curve with a different name."""
    sc = INSPECTING
    tr = datagen.target_truth(sc, 0)
    speed = np.hypot(tr[:, 2], tr[:, 3])
    hover = np.mean(speed < 1.5)
    legs = int(np.sum(np.abs(np.diff(speed)) > 1.0))
    assert 0.1 < hover < 0.7, f"hovering {100 * hover:.0f}% of the run"
    assert legs >= 4, f"only {legs} velocity changes in {sc.duration:.0f} s"
    assert speed.max() <= sc.tgt_speed_max * 1.3


def test_sporadic_target_stays_near_what_it_is_inspecting():
    sc = INSPECTING
    for run in range(4):
        tr = datagen.target_truth(sc, run)
        d = np.hypot(tr[:, 0] - sc.tgt_poi_x, tr[:, 1] - sc.tgt_poi_y)
        assert d.max() < 3.0 * sc.tgt_poi_radius, f"wandered {d.max():.0f} m away"


def test_sporadic_motion_has_its_own_random_stream():
    """Changing the sensor must not perturb how the intruder flies."""
    a = datagen.target_truth(INSPECTING, 0)
    b = datagen.target_truth(replace(INSPECTING, sigma_bearing_deg=0.9), 0)
    assert np.array_equal(a, b)


# --- 6. pursuit -------------------------------------------------------------


def test_pursuit_is_off_by_default_and_reproduces_the_open_loop_path():
    """Turning pursuit on must not disturb any earlier scenario."""
    sc = Scenario()
    assert sc.own_pursuit is False
    truth = datagen.target_truth(sc, 0)
    own, det = datagen.engagement(sc, truth, 0)
    own2 = datagen.ownship_track(sc)
    det2 = datagen.generate_detections(sc, truth, own2, 0)
    assert np.array_equal(own.x, own2.x) and np.array_equal(own.y, own2.y)
    assert np.array_equal(own.psi, own2.psi)
    for a, b in zip(det.per_step, det2.per_step):
        assert np.array_equal(a, b)


def test_pursuing_observer_steers_toward_the_target():
    """The whole point: the bearing off the nose must shrink once it has seen it."""
    truth = datagen.target_truth(PURSUING, 0)
    own, det = datagen.engagement(PURSUING, truth, 0)
    off = np.abs(datagen.wrap_pi(datagen.bearing(truth[:, :2], own.xy) - own.psi))
    early, late = np.degrees(off[1:20]).mean(), np.degrees(off[-20:]).mean()
    assert late < early, f"pursuit should close the aspect: {early:.1f} -> {late:.1f} deg"
    assert np.hypot(*(truth[-1, :2] - own.xy[-1])) < np.hypot(*(truth[0, :2] - own.xy[0]))


def test_pursuit_track_does_not_depend_on_the_estimator():
    """The observer steers on the measured bearing, never on a filter estimate.

    If this ever fails, the EKF and CKF would be flying different paths and
    seeing different measurements, and every comparison in the study would be
    confounded.
    """
    from kf2.run import track as _track

    a = _track(PURSUING, 3, "ekf")
    b = _track(PURSUING, 3, "ckf")
    assert np.array_equal(a.own.x, b.own.x)
    assert np.array_equal(a.own.y, b.own.y)
    assert np.array_equal(a.own.psi, b.own.psi)
    assert np.array_equal(a.truth, b.truth)
    seen = [k for k in range(len(a.bearing)) if np.isfinite(a.bearing[k])]
    assert len(seen) > 50
    for k in seen:
        assert a.bearing[k] == b.bearing[k], "estimators must see identical measurements"


def test_pursuit_costs_range_accuracy_and_buys_bearing_accuracy():
    """Radial acceleration does not make range observable (see proofs/vein-5).

    Flying at the target doubles the measurements and thirds the range, and is
    still worse at ranging than weaving. That is the scenario's reason to exist,
    so it is pinned rather than left as a remark.
    """
    from kf2.run import track as _track

    def measure(sc, n=25):
        A, C = [], []
        for run in range(n):
            t = _track(sc, run, "ekf")
            a, c = t.directional_error()
            A += list(t.settled(a))
            C += list(t.settled(c))
        return float(np.median(A)), float(np.median(C))

    weave = replace(PURSUING, own_pursuit=False, own_manoeuvre_amp_deg=40.0)
    r_pursue, c_pursue = measure(PURSUING)
    r_weave, c_weave = measure(weave)

    assert r_pursue > r_weave * 1.2, (
        f"pursuit should be worse at range: {r_pursue:.0f} m vs {r_weave:.0f} m"
    )
    assert c_pursue < c_weave, (
        f"pursuit should be better at bearing: {c_pursue:.1f} m vs {c_weave:.1f} m"
    )
