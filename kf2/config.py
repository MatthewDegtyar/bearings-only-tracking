"""Scenario configuration.

Every quantity that can change a result lives here. Nothing that affects a
number is allowed to be a literal buried in a module: a run must be reproducible
from a scenario file and a seed alone.

The dataclass is frozen. Experiments derive variants with
``dataclasses.replace(scenario, p0_pos=1000.0)`` rather than mutating a shared
object, so a scenario can never be altered halfway through a sweep.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, replace  # noqa: F401  (replace re-exported)
from dataclasses import asdict, fields

DEG = math.pi / 180.0


@dataclass(frozen=True)
class Scenario:
    name: str = "unnamed"
    seed: int = 20260730

    # --- timing -----------------------------------------------------------
    dt: float = 1.0
    """Filter and measurement interval [s]."""
    steps: int = 600
    """Number of measurement updates. The run spans steps + 1 time points."""
    truth_substeps: int = 50
    """Integration substeps per dt in the truth model. See datagen."""
    mc_runs: int = 500

    # --- ownship ----------------------------------------------------------
    # Heading follows psi(t) = psi0 + amp * sin(2*pi*periods*t/T).
    # `amp` is the single observability knob: amp = 0 gives a straight-line
    # ownship and an unobservable range direction.
    own_speed: float = 12.0
    own_psi0_deg: float = 0.0
    own_manoeuvre_amp_deg: float = 35.0
    own_manoeuvre_periods: float = 1.0
    own_x0: float = 0.0
    own_y0: float = 0.0

    # --- observer pursuit (defaults are 'off') ---------------------------
    own_pursuit: bool = False
    """Steer the observer toward the target once it has been seen.

    The observer turns toward the most recent *measured bearing*, not toward any
    filter estimate. That is both what a real airframe can do, since a bearing
    needs no range, and what keeps the ownship track independent of which
    estimator is running: swapping EKF for CKF must not change the flight path,
    or the two would no longer see the same measurements.

    Off by default, so every earlier scenario reproduces bit-for-bit."""
    own_pursuit_turn_rate_deg_s: float = 12.0
    """Turn rate limit while pursuing [deg/s]."""
    own_pursuit_hold_s: float = 3.0
    """How long a bearing stays actionable after the last detection [s]. Beyond
    this the observer holds its heading rather than chasing a stale bearing."""

    # --- target (truth) ---------------------------------------------------
    tgt_x0: float = 6000.0
    tgt_y0: float = 4000.0
    tgt_vx0: float = -3.0
    tgt_vy0: float = 1.5

    # --- target manoeuvre (Phase 4; defaults are 'off') -------------------
    # Target heading follows psi_t(t) = psi_t0 + amp * sin(2*pi*periods*t/T),
    # with psi_t0 and speed taken from (tgt_vx0, tgt_vy0). amp = 0 leaves the
    # target on the constant-velocity path the filter assumes, which is the
    # behaviour every earlier result was produced with.
    tgt_motion: str = "constant"
    """How the target flies: "constant", "sinusoid" or "sporadic".

    "sporadic" is a piecewise-constant velocity process: the operator holds a
    heading and speed for a while, stops to look at something, then moves off
    on a new one. Segment durations are exponential and each segment draws a
    fresh heading and speed, with a chance of hovering outright.

    This is the honest model for a human flying an inspection, and it is the
    hard case for a tracker, because the constant-velocity assumption the filter
    makes is violated at every segment boundary rather than smoothly."""

    tgt_poi_x: float = 0.0
    tgt_poi_y: float = 0.0
    tgt_poi_radius: float = 120.0
    """The intruder is inspecting something, so it stays near it. Beyond this
    radius the next segment is biased back toward the point of interest."""
    tgt_segment_mean_s: float = 6.0
    """Mean duration of one constant-velocity leg [s]."""
    tgt_speed_max: float = 12.0
    """Fastest leg [m/s]. Consumer airframes cruise well below their limit."""
    tgt_hover_prob: float = 0.35
    """Chance that a leg is a hover. Inspection is mostly stopping and looking."""

    tgt_manoeuvre_amp_deg: float = 0.0
    """Peak heading excursion of the target [deg]. 0 disables the manoeuvre.

    This is the one knob that breaks the filter's *motion* model rather than its
    measurement model. A manoeuvring target violates the constant-velocity
    assumption, so it is a state-space error, and the attenuation argument of the
    report predicts the innovation check will barely see it. That prediction is
    the reason the knob exists."""
    tgt_manoeuvre_periods: float = 1.0
    """Heading oscillations over the run."""

    # --- sensor field of view (defaults are 'off') ------------------------
    sensor_fov_deg: float = 360.0
    """Angular width the sensor can see, centred on the ownship heading [deg].

    360 means no restriction, which is what every earlier result assumed. A real
    camera sees perhaps 60 degrees, so the observer must point at the target to
    detect it -- and the manoeuvre that makes range observable is the same
    manoeuvre that swings the target out of frame. That conflict is the point."""

    sensor_slew_deg_s: float | None = None
    """Maximum rate at which the sensor can be pointed [deg/s].

    None bolts the sensor to the platform heading, which is what every earlier
    result assumed and is the right model for a fixed forward-looking seeker.
    Set it and the sensor is gimballed: the boresight turns toward the contact
    at up to this rate, which is what an operator or an autopilot actually does
    and what makes a narrow aperture usable at all. With the boresight fixed to
    heading, a straight-flying observer with a 60 degree aperture detects the
    target on zero scans, because the boresight never moves and the bearing
    drifts off it and never returns."""

    pd_half_range: float | None = None
    """Range at which detection probability falls to half of ``pd`` [m].

    None keeps pd constant with range, the earlier behaviour. Set it and
    detection follows pd / (1 + (r/pd_half_range)^pd_falloff_exponent), a smooth
    fall-off with no discontinuity for the filter to trip over."""
    pd_falloff_exponent: float = 4.0
    """Sharpness of the detection fall-off. Larger is more cliff-like."""

    # --- models -----------------------------------------------------------
    q: float = 1.0e-3
    """Continuous white-noise acceleration PSD used by the truth model."""
    q_filter: float | None = None
    """Filter's q. None means 'matched to truth'."""
    sigma_bearing_deg: float = 0.5

    # --- initialisation ---------------------------------------------------
    p0_pos: float = 300.0
    p0_vel: float = 1.5

    # --- detection and clutter (Phase 3; defaults are 'off') --------------
    pd: float = 1.0
    """Probability of detecting the target on a scan."""
    clutter_rate: float = 0.0
    """Expected number of false bearings per scan (Poisson)."""
    clutter_fov_deg: float = 360.0
    gate_prob: float = 0.9999
    """Validation-gate probability; the chi-square threshold follows from it.

    Defaults to *effectively open*. Every perturbation knob defaults off so each
    experiment moves exactly one, and a tight gate is a perturbation: it truncates
    the accepted innovations and inflates NEES even with no clutter and pd = 1.
    At 0.997 the gate rejects 0.3% of true measurements -- always the largest
    innovations -- which measurably biases the baseline. Phase 3 turns this down
    deliberately; Phase 1 must not inherit it by accident. It stays finite rather
    than infinite so the gating code path is always exercised.
    """

    # --- SNR-dependent measurement noise (defaults are 'off') -------------
    snr_enabled: bool = False
    """Make sigma_bearing follow SNR. Off, R is the constant sigma_bearing and
    every result reproduces bit-for-bit."""
    snr_ref_db: float = 20.0
    """SNR at the reference range, before fading."""
    snr_ref_range: float = 5000.0
    """Range at which snr_ref_db applies [m]."""
    snr_fade_depth_db: float = 0.0
    """Peak fade depth [dB]. 0 disables fading."""
    snr_fade_period: float = 120.0
    """Fade period [s]."""
    crlb_constant: float | None = None
    """k_crlb in sigma = k / sqrt(SNR). None calibrates it so that
    sigma(snr_ref_db) == sigma_bearing. See kf2.snr."""
    r_assumption: str = "true"
    """What R the filter assumes: "true" (track the measured SNR), "mean"
    (one constant, RMS over the nominal geometry) or "best" (one constant,
    best-case SNR). Only meaningful when snr_enabled."""

    # --- interceptor (Phase 5; only used by scenarios that fly one) --------
    # The interceptor never sees the target. It is vectored on whatever the
    # sentry's filter believes, so the filter's error arrives as a miss distance
    # rather than as a covariance statistic.
    int_x0: float = 0.0
    int_y0: float = 0.0
    int_psi0_deg: float = 0.0
    """Initial heading. Deliberately not aimed at the target: the interceptor
    flies its launch heading until it commits, then has to turn."""
    int_speed: float = 0.0
    """Interceptor speed [m/s]. 0 means no interceptor in this scenario."""
    int_turn_rate_deg_s: float = 6.0
    """Turn rate limit [deg/s]. A hard turn late costs more than a gentle one
    early, which is what makes an early, wrong solution expensive."""
    int_commit_time: float = 0.0
    """Time before guidance engages [s]. Until then the interceptor flies its
    launch heading, which is how a late commit on a better track is modelled."""
    int_single_shot: bool = False
    """Freeze the aim point at commit instead of updating it.

    With continuous guidance the interceptor keeps correcting until closest
    approach, by which time the range has closed and the estimate is good, so
    the miss says almost nothing about the track that produced it. A fire and
    forget weapon takes one solution and flies it, which is what makes the
    estimate error at the moment of commitment the thing that decides the
    outcome. That is the case worth simulating."""

    # --- estimator options ------------------------------------------------
    ckf_degree: int = 5
    """Polynomial degree of exactness of the cubature rule.

    The bearing's curvature enters the innovation *variance* at fourth order. A
    degree-3 rule does not integrate that term, so what it captures depends on
    where its points sit -- which made the fix frame-dependent, with measured
    recovery swinging 26-68% under rotations of the same physical problem. A
    degree-5 rule integrates it by design and the recovery collapses to a single
    value at every rotation. Degree 3 is retained only to reproduce that finding."""
    ckf_iterations: int = 3
    """Passes for the iterated cubature update. 1 reduces exactly to plain CKF."""
    ckf_iteration_tol: float = 1e-3
    """State-delta convergence tolerance [m-ish], with the iteration count as cap."""
    ckf_sample_from: str = "posterior"
    """Which covariance later iterations sample from: "posterior" (textbook IPLF)
    or "prior" (move the sampling point only)."""

    # --- evaluation -------------------------------------------------------
    gate_windows: int = 6
    """Contiguous windows for the time-localised consistency criterion."""
    gate_alpha: float = 0.05
    """FAMILY-WISE false-alarm rate across every statistical test in the gate."""
    max_ci_width_frac: float = 0.5
    """Widest CI for a NEES criterion, as a fraction of the target: resolution
    must beat 25%. This catches the genuinely uninformative case -- an interval
    so wide it brackets the target whatever the data says."""
    max_ci_width_frac_nis: float = 0.5
    """The same limit for NIS. Separable, but equal by default -- see below.

    A review objected that at 0.5 the gate certified a 15% NIS error as PASS
    (oracle, p0 = 1000 m, resolution 16.4%). Tightening this was tried and is the
    wrong fix, twice over: at 0.25 every windowed NEES criterion goes
    inconclusive (window resolution ~13%), and at 0.05 every windowed NIS one
    does (~3.7% at a 75-step window). Windowed criteria inherently have less
    power, so any absolute resolution floor destroys them first.

    The real point is that "PASS at 16.4% resolution" is not a false statement --
    it says no effect above 16.4% was found, which is true. The defect was in
    *quoting* that as evidence of consistency without its resolution. So the fix
    is reporting discipline (every criterion carries its resolution, and no PASS
    is cited without it), not a threshold. This knob exists for a study with the
    run count to afford a tighter NIS limit."""
    divergence_pos_error: float = 2000.0
    divergence_settle_frac: float = 0.1

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.truth_substeps < 1:
            raise ValueError("truth_substeps must be >= 1")
        if self.mc_runs < 1:
            raise ValueError("mc_runs must be >= 1")
        if self.gate_windows < 1:
            raise ValueError("gate_windows must be >= 1")
        if not 0.0 < self.gate_alpha < 1.0:
            raise ValueError("gate_alpha must be in (0, 1)")
        if not 0.0 < self.pd <= 1.0:
            raise ValueError("pd must be in (0, 1]")
        if self.clutter_rate < 0.0:
            raise ValueError("clutter_rate must be >= 0")
        if self.own_pursuit_turn_rate_deg_s <= 0.0:
            raise ValueError("own_pursuit_turn_rate_deg_s must be > 0")
        if self.own_pursuit_hold_s < 0.0:
            raise ValueError("own_pursuit_hold_s must be >= 0")
        if self.tgt_motion not in ("constant", "sinusoid", "sporadic"):
            raise ValueError("tgt_motion must be 'constant', 'sinusoid' or 'sporadic'")
        if self.tgt_segment_mean_s <= 0.0:
            raise ValueError("tgt_segment_mean_s must be > 0")
        if self.tgt_speed_max < 0.0:
            raise ValueError("tgt_speed_max must be >= 0")
        if not 0.0 <= self.tgt_hover_prob <= 1.0:
            raise ValueError("tgt_hover_prob must be in [0, 1]")
        if self.tgt_poi_radius <= 0.0:
            raise ValueError("tgt_poi_radius must be > 0")
        if self.tgt_manoeuvre_amp_deg < 0.0:
            raise ValueError("tgt_manoeuvre_amp_deg must be >= 0")
        if not 0.0 < self.sensor_fov_deg <= 360.0:
            raise ValueError("sensor_fov_deg must be in (0, 360]")
        if self.sensor_slew_deg_s is not None and self.sensor_slew_deg_s <= 0.0:
            raise ValueError("sensor_slew_deg_s must be > 0")
        if self.pd_half_range is not None and self.pd_half_range <= 0.0:
            raise ValueError("pd_half_range must be > 0")
        if self.pd_falloff_exponent <= 0.0:
            raise ValueError("pd_falloff_exponent must be > 0")
        if not 0.0 < self.gate_prob < 1.0:
            raise ValueError("gate_prob must be in (0, 1)")
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.q < 0.0 or (self.q_filter is not None and self.q_filter < 0.0):
            raise ValueError("q must be >= 0")
        if not 0.0 <= self.divergence_settle_frac < 1.0:
            raise ValueError("divergence_settle_frac must be in [0, 1)")
        if not 0.0 < self.max_ci_width_frac_nis < 1.0:
            raise ValueError("max_ci_width_frac_nis must be in (0, 1)")
        if self.int_speed < 0.0:
            raise ValueError("int_speed must be >= 0")
        if self.int_turn_rate_deg_s <= 0.0:
            raise ValueError("int_turn_rate_deg_s must be > 0")
        if self.int_commit_time < 0.0:
            raise ValueError("int_commit_time must be >= 0")
        if self.ckf_degree not in (3, 5):
            raise ValueError("ckf_degree must be 3 or 5")
        if self.ckf_iterations < 1:
            raise ValueError("ckf_iterations must be >= 1")
        if self.ckf_sample_from not in ("posterior", "prior"):
            raise ValueError("ckf_sample_from must be 'posterior' or 'prior'")
        if self.r_assumption not in ("true", "mean", "best"):
            raise ValueError("r_assumption must be 'true', 'mean' or 'best'")
        if self.snr_ref_range <= 0.0:
            raise ValueError("snr_ref_range must be > 0")
        if self.snr_fade_depth_db < 0.0:
            raise ValueError("snr_fade_depth_db must be >= 0")
        if self.crlb_constant is not None and self.crlb_constant <= 0.0:
            raise ValueError("crlb_constant must be > 0")

    # --- derived ----------------------------------------------------------
    @property
    def sigma_bearing(self) -> float:
        """Bearing noise standard deviation [rad]."""
        return self.sigma_bearing_deg * DEG

    @property
    def filter_q(self) -> float:
        return self.q if self.q_filter is None else self.q_filter

    @property
    def tgt_speed(self) -> float:
        """Target speed from its initial velocity [m/s]."""
        return math.hypot(self.tgt_vx0, self.tgt_vy0)

    @property
    def tgt_psi0(self) -> float:
        """Target initial heading [rad]."""
        return math.atan2(self.tgt_vy0, self.tgt_vx0)

    @property
    def duration(self) -> float:
        return self.dt * self.steps

    @property
    def settle_index(self) -> int:
        """First time index counted when judging track loss."""
        return int(self.divergence_settle_frac * self.steps)

    # --- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            # A silently ignored typo in a scenario file is a reproducibility
            # bug: it surfaces later as an unexplained change in results.
            raise ValueError(f"unknown scenario keys: {sorted(unknown)}")
        return cls(**d)

    def save(self, path: str | pathlib.Path) -> None:
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Scenario":
        return cls.from_dict(json.loads(pathlib.Path(path).read_text()))

    def summary(self) -> str:
        return (
            f"{self.name}: seed {self.seed}, {self.mc_runs} runs x {self.steps} steps "
            f"@ {self.dt}s ({self.duration:.0f}s)\n"
            f"  ownship {self.own_speed} m/s, manoeuvre {self.own_manoeuvre_amp_deg} deg "
            f"x {self.own_manoeuvre_periods}\n"
            f"  target ({self.tgt_x0:.0f}, {self.tgt_y0:.0f}) m, "
            f"({self.tgt_vx0}, {self.tgt_vy0}) m/s\n"
            f"  q {self.q:.2e} (filter {self.filter_q:.2e}), "
            f"sigma_bearing {self.sigma_bearing_deg} deg\n"
            f"  P0 pos {self.p0_pos} m, vel {self.p0_vel} m/s; "
            f"pd {self.pd}, clutter {self.clutter_rate}"
        )
