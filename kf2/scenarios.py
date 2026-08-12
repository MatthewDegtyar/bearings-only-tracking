"""Named scenarios.

A scenario is the complete description of an experiment: geometry, kinematics,
sensor and noise. Keeping them here rather than in the scripts means a result
can be reproduced from a name and a seed, and that two scripts cannot silently
disagree about what a case means.

Everything here is close-range work with a civilian airframe. The sensor is a
consumer 4K camera, which fixes the scale: at 70 degrees across 3840 pixels one
pixel is 0.018 degrees, so a 0.5 m quadcopter is three pixels wide at about
520 m and one pixel at 1.5 km. Reliable tracking therefore lives inside a few
hundred metres, and every geometry below sits in 150 to 700 m.

The bearing error is 0.5 degrees, and it is worth being clear why, because it
looks coarse for a 4K sensor. It is not an optics figure. Centroiding a
three-pixel blob is good to about 0.006 degrees; the gimbal readout contributes
0.1; the airframe's MEMS attitude solution contributes 0.5. Combined in
quadrature that is 0.51 degrees, five-sixths of it attitude. The camera is not
the limit and a better camera would not help.
"""

from __future__ import annotations

import math

from .config import DEG, Scenario


def _velocity(speed: float, heading_deg: float) -> tuple[float, float]:
    psi = heading_deg * DEG
    return speed * math.cos(psi), speed * math.sin(psi)


# ---------------------------------------------------------------------------
# A sentry drone on a fixed patrol route, watching an intruder
# ---------------------------------------------------------------------------

_COMMON = dict(
    # 60 s at 10 Hz.
    dt=0.1, steps=600, truth_substeps=10, mc_runs=400,
    # Consumer airframe on a fixed route, camera bolted forward. The route
    # weaves because a straight one gives no parallax and therefore no range.
    own_speed=10.0, own_psi0_deg=0.0,
    own_manoeuvre_amp_deg=40.0, own_manoeuvre_periods=2.0,
    sensor_fov_deg=84.0, clutter_fov_deg=84.0, sensor_slew_deg_s=None,
    sigma_bearing_deg=0.5,
    # Residual airframe wobble only. Where a scenario's target manoeuvres, the
    # manoeuvre is modelled explicitly rather than hidden in the process noise.
    q=0.004,
    p0_pos=150.0, p0_vel=8.0,
    # Range is barely observable in some of these cases, so a threshold much
    # tighter than this would be measuring the physics rather than divergence.
    divergence_pos_error=400.0,
)

# The intruder is being flown by a person looking at something: it holds a
# heading for a few seconds, stops to look, then moves off again. This is the
# hard case, and the honest one.
INSPECTING = Scenario(
    name="inspecting",
    tgt_x0=550.0, tgt_y0=220.0, tgt_vx0=0.0, tgt_vy0=0.0,
    tgt_motion="sporadic",
    tgt_poi_x=550.0, tgt_poi_y=220.0, tgt_poi_radius=110.0,
    tgt_segment_mean_s=6.0, tgt_speed_max=12.0, tgt_hover_prob=0.35,
    # Sized for a target that changes velocity by up to 12 m/s every few
    # seconds. Matching the truth's q here would be wrong: the manoeuvre is
    # real motion the filter cannot see coming, not sensor noise.
    q_filter=4.0,
    **_COMMON,
)

# The same patrol against a target that holds a course. Everything else is
# identical, so the difference isolates what the target's predictability is
# worth: range error falls from about 155 m to about 24 m.
#
# The start and course were chosen so the intruder stays in the forward cone as
# long as the inspecting one does. An earlier version put it crossing close
# ahead, where it swept past the sentry and ended up behind: bearing ran from 22
# degrees to 140 and detection fell to 26 per cent against the inspecting case's
# 52. That made the comparison worthless, since the transiting case was then
# winning on range while being handed half the measurements. Starting it further
# out and sending it across rather than past holds detection at 53 per cent,
# matching the inspecting case, so what differs between them is observability
# alone.
TRANSITING = Scenario(
    name="transiting",
    tgt_x0=700.0, tgt_y0=200.0,
    tgt_vx0=_velocity(5.0, 90.0)[0], tgt_vy0=_velocity(5.0, 90.0)[1],
    tgt_motion="constant", q_filter=0.05,
    **_COMMON,
)

# A steady target again, but the patrol flies straight. Without own-ship
# acceleration there is no parallax, and range degrades even though the target
# is behaving. Range needs both ingredients.
STRAIGHT_ROUTE = Scenario(
    name="straight-route",
    **{**_COMMON, "own_manoeuvre_amp_deg": 0.0},
    tgt_x0=700.0, tgt_y0=200.0,
    tgt_vx0=_velocity(5.0, 90.0)[0], tgt_vy0=_velocity(5.0, 90.0)[1],
    tgt_motion="constant", q_filter=0.05,
)

# The observer turns toward the intruder as soon as it sees it, and keeps
# turning toward the latest measured bearing. The intruder holds a course with a
# sinusoidal weave on top, so it is predictable in the mean but never exactly
# constant-velocity.
#
# This is the scenario that contradicts the obvious tactic. Flying at the target
# is what an operator would do, and it buys a great deal: the target stays in the
# forward cone so detection goes from about half the scans to all of them, the
# range closes from 728 m to 235 m, and angular accuracy improves from 0.28 to
# 0.12 degrees. It also makes range estimation distinctly worse, 120 m against
# 75 m for the weaving route.
#
# The reason is not subtle once stated. Range is observable only through observer
# acceleration that has a component across the line of sight; acceleration
# straight down the bearing contributes nothing, which is Fogel and Gavish's
# necessary-but-not-sufficient condition. A pursuing observer accelerates almost
# entirely along the line of sight by construction, so it collects twice the
# measurements and learns less from them about distance. The bearing sweep tells
# the same story: 30 degrees pursuing against 54 weaving.
PURSUING = Scenario(
    name="pursuing",
    **{**_COMMON, "own_manoeuvre_amp_deg": 0.0},
    tgt_x0=700.0, tgt_y0=200.0,
    tgt_vx0=_velocity(5.0, 90.0)[0], tgt_vy0=_velocity(5.0, 90.0)[1],
    tgt_motion="sinusoid",
    tgt_manoeuvre_amp_deg=25.0,
    tgt_manoeuvre_periods=3.0,
    q_filter=0.05,
    own_pursuit=True,
    own_pursuit_turn_rate_deg_s=12.0,
    own_pursuit_hold_s=3.0,
)

SCENARIOS = {sc.name: sc for sc in (INSPECTING, TRANSITING, STRAIGHT_ROUTE, PURSUING)}


def get(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise ValueError(f"unknown scenario {name!r}; have {sorted(SCENARIOS)}") from None
