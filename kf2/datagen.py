"""Data generation: ownship trajectory, target truth, and sensor detections.

**This module is the "no inverse crime" boundary.** Nothing here may share code
with :mod:`kf2.ekf`. The truth model integrates the continuous-time system with a
fine substep and injects noise per substep; the filter propagates a coarse
exact-discrete model with a closed-form Q. Handing the filter the same dynamics
and process noise that produced the data assumes away model mismatch and yields
consistency results that look excellent and mean nothing.

The mismatch is deliberate, one-sided, and quantified. For ``n`` substeps of
size ``h = dt/n`` the Euler-Maruyama step covariance is

    Var(p)    = q h^3 (n-1)n(2n-1)/6     ->  q dt^3/3  as n -> inf
    Cov(p, v) = q h^2 n(n-1)/2           ->  q dt^2/2
    Var(v)    = q h n = q dt             (exact for every n)

so at 50 substeps the truth is 2.98% quieter in position variance and 2.0% in
cross-covariance than the filter assumes. :func:`em_step_covariance` returns this
in closed form; the tests check the generator against it rather than against the
filter's Q, which leaves room to detect an actual bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

from .config import DEG, Scenario
from .rng import Stream, stream_rng

TWO_PI = 2.0 * np.pi


def wrap_pi(angle):
    """Wrap an angle (or array) to (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % TWO_PI - np.pi


def bearing(target_xy, own_xy):
    """Bearing from ownship to target, measured counter-clockwise from +x.

    Accepts (2,) or (..., 2) arrays.
    """
    d = np.asarray(target_xy) - np.asarray(own_xy)
    return np.arctan2(d[..., 1], d[..., 0])


# ---------------------------------------------------------------------------
# Ownship
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnshipTrack:
    """Deterministic: identical across every Monte Carlo run of a scenario.

    The ownship trajectory is a known input, not an estimated quantity, so it is
    generated once and shared by all runs.
    """

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    psi: np.ndarray

    @property
    def xy(self) -> np.ndarray:
        return np.column_stack((self.x, self.y))


def ownship_heading(sc: Scenario, t) -> np.ndarray:
    """psi(t) = psi0 + amp * sin(2 pi periods t / T).

    A smooth single-parameter manoeuvre law. ``amp = 0`` is a straight line,
    which makes range unobservable -- this is the Phase 2 sweep knob.
    """
    t = np.asarray(t, dtype=float)
    amp = sc.own_manoeuvre_amp_deg * DEG
    T = sc.duration
    if T <= 0:
        return np.full_like(t, sc.own_psi0_deg * DEG)
    return sc.own_psi0_deg * DEG + amp * np.sin(TWO_PI * sc.own_manoeuvre_periods * t / T)


def ownship_track(sc: Scenario) -> OwnshipTrack:
    """Integrate the ownship heading law onto a position track.

    Integration is on a fine grid (``truth_substeps`` per dt) with the
    trapezoidal rule, then sampled at the coarse times. The heading law is
    smooth, so the residual quadrature error is orders of magnitude below
    anything that matters here.
    """
    sub = sc.truth_substeps
    n_fine = sc.steps * sub
    t_fine = np.arange(n_fine + 1) * (sc.dt / sub)
    psi_fine = ownship_heading(sc, t_fine)
    vx_fine = sc.own_speed * np.cos(psi_fine)
    vy_fine = sc.own_speed * np.sin(psi_fine)

    h = sc.dt / sub
    # Cumulative trapezoid, written out so the module has no scipy dependency.
    x_fine = np.concatenate(([0.0], np.cumsum(0.5 * h * (vx_fine[1:] + vx_fine[:-1]))))
    y_fine = np.concatenate(([0.0], np.cumsum(0.5 * h * (vy_fine[1:] + vy_fine[:-1]))))

    idx = np.arange(sc.steps + 1) * sub
    return OwnshipTrack(
        t=t_fine[idx],
        x=sc.own_x0 + x_fine[idx],
        y=sc.own_y0 + y_fine[idx],
        vx=vx_fine[idx],
        vy=vy_fine[idx],
        psi=psi_fine[idx],
    )


# ---------------------------------------------------------------------------
# Target truth
# ---------------------------------------------------------------------------


def em_step_covariance(q: float, dt: float, substeps: int) -> np.ndarray:
    """Exact covariance of one coarse step of the Euler-Maruyama scheme.

    State ordering [px, py, vx, vy]. See the module docstring for the derivation.
    """
    n = float(substeps)
    h = dt / n
    var_p = q * h**3 * (n - 1.0) * n * (2.0 * n - 1.0) / 6.0
    cov_pv = q * h**2 * n * (n - 1.0) / 2.0
    var_v = q * h * n
    Q = np.zeros((4, 4))
    Q[0, 0] = Q[1, 1] = var_p
    Q[2, 2] = Q[3, 3] = var_v
    Q[0, 2] = Q[2, 0] = cov_pv
    Q[1, 3] = Q[3, 1] = cov_pv
    return Q


def target_heading(sc: Scenario, t) -> np.ndarray:
    """psi_t(t) = psi_t0 + amp * sin(2 pi periods t / T).

    The same law the ownship follows, with the target's own amplitude. With
    ``tgt_manoeuvre_amp_deg = 0`` this is the constant initial heading, so the
    target flies the straight line the filter's motion model assumes.
    """
    t = np.asarray(t, dtype=float)
    amp = sc.tgt_manoeuvre_amp_deg * DEG
    T = sc.duration
    if amp == 0.0 or T <= 0:
        return np.full_like(t, sc.tgt_psi0)
    return sc.tgt_psi0 + amp * np.sin(TWO_PI * sc.tgt_manoeuvre_periods * t / T)


def _sporadic_velocity(sc: Scenario, run: int) -> np.ndarray:
    """Velocity profile for a human-flown inspection, at substep resolution.

    Piecewise constant: hold a heading and speed for an exponentially
    distributed while, then pick new ones. A fraction of legs are hovers. When
    the intruder has wandered outside ``tgt_poi_radius`` of the thing it is
    inspecting, the next leg is biased back toward it, which is what keeps the
    track in the sentry's neighbourhood instead of flying off.

    Drawn from its own stream, so changing the sensor or the estimator cannot
    perturb the intruder's behaviour and scenarios stay paired.
    """
    rng = stream_rng(sc.seed, run, Stream.TARGET_MANOEUVRE)
    sub = sc.truth_substeps
    n = sc.steps * sub + 1
    h = sc.dt / sub
    v = np.zeros((n, 2))
    p = np.array([sc.tgt_x0, sc.tgt_y0], dtype=float)
    poi = np.array([sc.tgt_poi_x, sc.tgt_poi_y], dtype=float)

    k = 0
    while k < n:
        leg = max(1, int(rng.exponential(sc.tgt_segment_mean_s) / h))
        if rng.random() < sc.tgt_hover_prob:
            vel = np.zeros(2)
        else:
            offset = p - poi
            if np.hypot(*offset) > sc.tgt_poi_radius:
                # Head back, give or take 60 degrees of operator discretion.
                base = math.atan2(-offset[1], -offset[0])
                psi = base + rng.uniform(-np.pi / 3, np.pi / 3)
            else:
                psi = rng.uniform(-np.pi, np.pi)
            speed = sc.tgt_speed_max * rng.uniform(0.25, 1.0)
            vel = speed * np.array([math.cos(psi), math.sin(psi)])
        end = min(n, k + leg)
        v[k:end] = vel
        p = p + vel * h * (end - k)
        k = end
    return v


def _manoeuvre_increments(sc: Scenario) -> np.ndarray:
    """Deterministic per-substep velocity increments from the target manoeuvre.

    Shape matches the stochastic increments, (steps, substeps, 2), so the two
    simply add and the existing integrator handles the manoeuvre without
    knowing about it. That matters: the vectorised integrator is checked
    against an independent substep loop, and adding the manoeuvre here keeps
    that check covering the manoeuvring case too.

    Returns exact zeros when the manoeuvre is off, so every earlier result
    reproduces bit-for-bit.
    """
    if sc.tgt_motion != "sinusoid" or sc.tgt_manoeuvre_amp_deg == 0.0:
        return np.zeros((sc.steps, sc.truth_substeps, 2))
    sub = sc.truth_substeps
    h = sc.dt / sub
    t = np.arange(sc.steps * sub + 1) * h
    psi = target_heading(sc, t)
    v = sc.tgt_speed * np.stack((np.cos(psi), np.sin(psi)), axis=-1)
    return np.diff(v, axis=0).reshape(sc.steps, sub, 2)


def _velocity_increments(sc: Scenario, rng: np.random.Generator) -> np.ndarray:
    """Per-substep velocity increments, shape (steps, substeps, 2)."""
    h = sc.dt / sc.truth_substeps
    sigma = np.sqrt(sc.q * h)
    return sigma * rng.standard_normal((sc.steps, sc.truth_substeps, 2))


def _integrate_em(sc: Scenario, dv: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Vectorised Euler-Maruyama integration.

    Algebraically identical to the explicit substep loop in
    :func:`_integrate_em_reference`, which the tests assert. Over one coarse step
    with substeps i = 0..n-1,

        p_k = p_{k-1} + dt * v_{k-1} + h * sum_i (n-1-i) dv_i
        v_k = v_{k-1} + sum_i dv_i

    so the substep loop collapses into two weighted sums, and the remaining
    step-to-step recursion is a pair of cumulative sums.
    """
    n = sc.truth_substeps
    h = sc.dt / n
    weights = (n - 1 - np.arange(n)).astype(float)  # (n,)

    a = h * np.einsum("i,kij->kj", weights, dv)  # position kick, (steps, 2)
    b = dv.sum(axis=1)  # velocity kick, (steps, 2)

    p0, v0 = x0[:2], x0[2:]
    v = np.vstack((v0, v0 + np.cumsum(b, axis=0)))  # (steps+1, 2)
    p = np.vstack((p0, p0 + sc.dt * np.cumsum(v[:-1], axis=0) + np.cumsum(a, axis=0)))
    return np.hstack((p, v))


def _integrate_em_reference(sc: Scenario, dv: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Explicit substep loop. Exists only so the vectorised form has something
    independent to be checked against."""
    h = sc.dt / sc.truth_substeps
    out = np.zeros((sc.steps + 1, 4))
    x = np.array(x0, dtype=float)
    out[0] = x
    for k in range(sc.steps):
        for i in range(sc.truth_substeps):
            x[0] += x[2] * h
            x[1] += x[3] * h
            x[2] += dv[k, i, 0]
            x[3] += dv[k, i, 1]
        out[k + 1] = x
    return out


def target_truth(sc: Scenario, run: int) -> np.ndarray:
    """Target truth sampled at the filter rate. Shape (steps + 1, 4)."""
    rng = stream_rng(sc.seed, run, Stream.PROCESS)
    x0 = np.array([sc.tgt_x0, sc.tgt_y0, sc.tgt_vx0, sc.tgt_vy0], dtype=float)
    if sc.tgt_motion == "sporadic":
        v = _sporadic_velocity(sc, run)
        x0 = np.array([sc.tgt_x0, sc.tgt_y0, v[0, 0], v[0, 1]], dtype=float)
        dv = _velocity_increments(sc, rng) + np.diff(v, axis=0).reshape(sc.steps, sc.truth_substeps, 2)
    else:
        dv = _velocity_increments(sc, rng) + _manoeuvre_increments(sc)
    return _integrate_em(sc, dv, x0)


def initial_covariance(sc: Scenario) -> np.ndarray:
    return np.diag([sc.p0_pos**2, sc.p0_pos**2, sc.p0_vel**2, sc.p0_vel**2])


def initial_estimate(sc: Scenario, truth0: np.ndarray, run: int) -> np.ndarray:
    """Draw the initial estimate from the prior it will be handed.

    Seeding the filter at truth with a non-trivial P0 makes NEES start at zero
    and reports a consistency the filter has not earned.
    """
    rng = stream_rng(sc.seed, run, Stream.INIT)
    L = np.linalg.cholesky(initial_covariance(sc))
    return truth0 + L @ rng.standard_normal(4)


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detections:
    """Sensor output: a variable-length set of bearings per scan.

    Even in Phase 1 -- where ``pd = 1`` and there is no clutter, so every scan
    holds exactly one bearing -- the container is a *set*. That is what lets
    :mod:`kf2.gating` be a real component now rather than a rewrite later.

    ``per_step[0]`` is empty by convention: the filter is initialised at k = 0
    and the first update happens at k = 1.
    """

    per_step: list[np.ndarray]
    truth_index: list[int | None]
    """Index of the target-originated bearing within ``per_step[k]``, or None if
    the target was not detected. Diagnostic only -- never visible to the filter."""

    def __len__(self) -> int:
        return len(self.per_step)

    @property
    def detected(self) -> np.ndarray:
        return np.array([i is not None for i in self.truth_index], dtype=bool)


def generate_detections(sc: Scenario, truth: np.ndarray, own: OwnshipTrack, run: int) -> Detections:
    """Generate one scan of bearings per time step.

    Open loop only. Under pursuit the observer's path depends on what it has
    already detected, so a track built in advance is the wrong one and any
    detections computed against it describe a different engagement. Use
    :func:`engagement` instead; this raises rather than returning a plausible
    wrong answer, which it did once.

    Noise, detection and clutter draw from separate streams, and the measurement
    noise is drawn for *every* step regardless of whether the target is detected.
    Both choices keep the streams aligned when pd or clutter_rate changes, so
    scenarios stay paired under common random numbers.
    """
    if sc.own_pursuit:
        raise ValueError(
            "this scenario pursues, so the observer track depends on the detections; "
            "call datagen.engagement(sc, truth, run) instead"
        )

    rng_meas = stream_rng(sc.seed, run, Stream.MEASUREMENT)
    rng_det = stream_rng(sc.seed, run, Stream.DETECTION)
    rng_clutter = stream_rng(sc.seed, run, Stream.CLUTTER)

    n = sc.steps + 1
    true_b = bearing(truth[:, :2], own.xy)
    # Per-step noise scale. With the SNR model off this is a constant array equal
    # to sigma_bearing, so the product below is bit-identical to the scalar form.
    # Scaling a fixed sequence of standard normals also keeps the stream aligned
    # when the SNR knob moves, so scenarios stay paired.
    sigma_t = measurement_sigma(sc, truth, own)
    noise = sigma_t * rng_meas.standard_normal(n)
    # One uniform draw per step whatever the settings, so changing the field of
    # view or the range fall-off cannot desynchronise the stream and scenarios
    # stay paired under common random numbers.
    boresight = sensor_boresight(sc, true_b, own)
    u = rng_det.random(n)
    detected = (u < detection_probability(sc, truth, own)) & in_field_of_view(sc, true_b, own)
    n_clutter = rng_clutter.poisson(sc.clutter_rate, size=n) if sc.clutter_rate > 0 else np.zeros(n, int)
    fov = sc.clutter_fov_deg * DEG

    per_step: list[np.ndarray] = [np.empty(0)]
    truth_index: list[int | None] = [None]

    for k in range(1, n):
        # Clutter is uniform across the field of view, centred on the ownship
        # heading -- it is not placed relative to the target, which would
        # quietly make association easier than it is.
        m = int(n_clutter[k])
        false_b = wrap_pi(boresight[k] + fov * (rng_clutter.random(m) - 0.5)) if m else np.empty(0)

        # Drawn for every scan that has clutter, whether or not the target was
        # detected. Drawing it only on detection would desynchronise the clutter
        # stream the first time a detection is missed, so scenarios differing in
        # pd would stop being paired.
        pos = int(rng_clutter.integers(0, m + 1)) if m else 0
        if detected[k]:
            z = wrap_pi(true_b[k] + noise[k])
            scan = np.insert(false_b, pos, z)
            per_step.append(scan)
            truth_index.append(pos)
        else:
            per_step.append(false_b)
            truth_index.append(None)

    return Detections(per_step=per_step, truth_index=truth_index)


def sensor_boresight(sc: Scenario, true_bearing: np.ndarray, own: OwnshipTrack) -> np.ndarray:
    """Direction the sensor is pointing at each step [rad].

    Without ``sensor_slew_deg_s`` the sensor looks along the platform heading.
    With it, the sensor is gimballed and turns toward the contact at up to the
    stated rate, starting from the heading. It lags a fast-crossing target and
    catches up when the geometry allows, which is what puts the target outside a
    narrow aperture occasionally rather than never or always.
    """
    if sc.sensor_slew_deg_s is None:
        return own.psi
    max_step = sc.sensor_slew_deg_s * DEG * sc.dt
    b = np.asarray(true_bearing, dtype=float)
    out = np.empty_like(b)
    psi = float(own.psi[0])
    for k in range(len(b)):
        err = wrap_pi(b[k] - psi)
        psi = wrap_pi(psi + np.clip(err, -max_step, max_step))
        out[k] = psi
    return out


def in_field_of_view(sc: Scenario, true_bearing: np.ndarray, own: OwnshipTrack) -> np.ndarray:
    """Whether the target lies inside the sensor's aperture at each step.

    The sensor looks along the ownship heading with a total width of
    ``sensor_fov_deg``. At the default of 360 degrees this is True everywhere
    and costs nothing.
    """
    if sc.sensor_fov_deg >= 360.0:
        return np.ones(len(own.t), dtype=bool)
    off_axis = np.abs(wrap_pi(np.asarray(true_bearing) - sensor_boresight(sc, true_bearing, own)))
    return off_axis <= 0.5 * sc.sensor_fov_deg * DEG


def detection_probability(sc: Scenario, truth: np.ndarray, own: OwnshipTrack) -> np.ndarray:
    """Per-step probability of detecting the target, before the field of view.

    Constant ``pd`` unless ``pd_half_range`` is set, in which case it follows

        pd(r) = pd / (1 + (r / pd_half_range) ** pd_falloff_exponent)

    which equals pd at zero range, half of it at ``pd_half_range``, and falls
    smoothly -- a sharp cutoff would put a discontinuity in the data the filter
    has no way to model.
    """
    n = len(own.t)
    if sc.pd_half_range is None:
        return np.full(n, sc.pd)
    r = target_range(truth, own)
    return sc.pd / (1.0 + (r / sc.pd_half_range) ** sc.pd_falloff_exponent)


def engagement(sc: Scenario, truth: np.ndarray, run: int) -> tuple[OwnshipTrack, "Detections"]:
    """Ownship track and detections together.

    Without pursuit these are independent and this is just the two existing
    calls in order. With pursuit they are coupled: where the observer flies
    depends on what it has seen, and what it sees depends on where it flew. The
    loop below steps them forward together.

    The observer steers on the measured bearing only. It never touches a filter
    estimate, so the flight path is the same whichever estimator is running and
    the two still see identical measurements. That property is asserted in the
    tests, because losing it would quietly invalidate every EKF/CKF comparison.
    """
    if not sc.own_pursuit:
        own = ownship_track(sc)
        return own, generate_detections(sc, truth, own, run)

    rng_meas = stream_rng(sc.seed, run, Stream.MEASUREMENT)
    rng_det = stream_rng(sc.seed, run, Stream.DETECTION)
    rng_clutter = stream_rng(sc.seed, run, Stream.CLUTTER)

    n = sc.steps + 1
    sub = sc.truth_substeps
    h = sc.dt / sub
    max_turn = sc.own_pursuit_turn_rate_deg_s * DEG * sc.dt
    hold_steps = int(round(sc.own_pursuit_hold_s / sc.dt))

    x = np.zeros(n); y = np.zeros(n); psi = np.zeros(n)
    vx = np.zeros(n); vy = np.zeros(n)
    x[0], y[0] = sc.own_x0, sc.own_y0
    psi[0] = sc.own_psi0_deg * DEG
    vx[0], vy[0] = sc.own_speed * math.cos(psi[0]), sc.own_speed * math.sin(psi[0])

    # Noise is drawn for every step up front, exactly as in the open-loop path,
    # so that turning pursuit on does not resequence the streams.
    noise_all = rng_meas.standard_normal(n)
    u_det = rng_det.random(n)
    n_clutter = (rng_clutter.poisson(sc.clutter_rate, size=n) if sc.clutter_rate > 0
                 else np.zeros(n, int))
    fov = sc.clutter_fov_deg * DEG

    per_step: list[np.ndarray] = [np.empty(0)]
    truth_index: list[int | None] = [None]
    in_fov = np.zeros(n, dtype=bool)
    boresight = np.zeros(n)
    last_bearing = None
    last_seen = -10**9

    for k in range(1, n):
        # fly the previous heading for one step
        psi[k] = psi[k - 1]
        if last_bearing is not None and (k - 1 - last_seen) <= hold_steps:
            err = wrap_pi(last_bearing - psi[k - 1])
            psi[k] = wrap_pi(psi[k - 1] + np.clip(err, -max_turn, max_turn))
        vx[k] = sc.own_speed * math.cos(psi[k])
        vy[k] = sc.own_speed * math.sin(psi[k])
        x[k] = x[k - 1] + vx[k] * sc.dt
        y[k] = y[k - 1] + vy[k] * sc.dt

        # now measure from where we ended up
        pos = np.array([x[k], y[k]])
        true_b = bearing(truth[k, :2], pos)
        boresight[k] = psi[k]
        off = abs(wrap_pi(true_b - psi[k]))
        seen_fov = sc.sensor_fov_deg >= 360.0 or off <= 0.5 * sc.sensor_fov_deg * DEG
        in_fov[k] = seen_fov

        r = float(np.hypot(*(truth[k, :2] - pos)))
        pd = sc.pd if sc.pd_half_range is None else sc.pd / (1.0 + (r / sc.pd_half_range) ** sc.pd_falloff_exponent)
        detected = bool(u_det[k] < pd) and seen_fov

        m = int(n_clutter[k])
        false_b = wrap_pi(psi[k] + fov * (rng_clutter.random(m) - 0.5)) if m else np.empty(0)
        pos_idx = int(rng_clutter.integers(0, m + 1)) if m else 0
        if detected:
            z = wrap_pi(true_b + sc.sigma_bearing * noise_all[k])
            per_step.append(np.insert(false_b, pos_idx, z))
            truth_index.append(pos_idx)
            last_bearing, last_seen = z, k
        else:
            per_step.append(false_b)
            truth_index.append(None)

    own = OwnshipTrack(t=np.arange(n) * sc.dt, x=x, y=y, vx=vx, vy=vy, psi=psi)
    return own, Detections(per_step=per_step, truth_index=truth_index)


def target_range(truth: np.ndarray, own: OwnshipTrack) -> np.ndarray:
    """Ownship-to-target range per step [m]."""
    d = np.asarray(truth)[:, :2] - own.xy
    return np.hypot(d[:, 0], d[:, 1])


def measurement_sigma(sc: Scenario, truth: np.ndarray, own: OwnshipTrack) -> np.ndarray:
    """Per-step bearing noise sigma [rad] actually applied to the measurements.

    Constant ``sigma_bearing`` unless the SNR model is enabled; see :mod:`kf2.snr`.
    """
    from .snr import bearing_sigma_series

    return bearing_sigma_series(sc, target_range(truth, own), own.t)
