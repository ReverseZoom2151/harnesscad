"""First-order analytic performance scorers for standard parametric objects.

Motivation
----------
This module is a correct, deterministic reference implementation of the
FIRST-ORDER analytic performance functions discussed in the "Design-to-
Performance" section (section 7) of:

    Makatura et al., "How Can Large Language Models Help Humans in Design
    and Manufacturing?"

In that paper GPT-4 is asked to author closed-form analytic performance
functions for standard objects (chairs, cabinets, quadcopters). The generated
functions capture the right first-order physical intuition but contain
arithmetic mistakes. The value here is therefore a small, self-contained,
stdlib-only library of the same first-order formulas, implemented correctly and
deterministically so they can serve as a trustworthy reference.

Scope and non-duplication
--------------------------
This module is intentionally standalone. It does NOT touch the geometry kernel
and is unrelated to:

  * ``quality/estimate.py``      -- mass/stock/cost/BOM from kernel geometry
  * ``bench/engdesign_*``        -- VLM evaluation harness
  * ``verifiers/dfm.py``         -- design-for-manufacture rule checks

Everything below is a closed-form engineering estimate. All models are
deliberately first-order (single dominant failure/energy mode, idealized
geometry) and should be treated as quick screening tools, not FEA.

Units
-----
SI throughout unless a name says otherwise (``_mm``, ``_kg``, ``_mAh``,
``_kmh``, ``_min``, ``_km``). Stresses are in pascals (Pa), lengths in metres,
areas in square metres, volumes in cubic metres, forces in newtons (N).
"""

from __future__ import annotations

from dataclasses import dataclass

# Standard gravitational acceleration (m/s^2). Exposed so callers/tests can
# pin it explicitly and stay deterministic.
STANDARD_GRAVITY = 9.81


# ---------------------------------------------------------------------------
# 1) Chair mechanical failure  (paper Fig. 46, ``will_chair_break``)
# ---------------------------------------------------------------------------


def chair_leg_compressive_stress(weight_kg, leg_cross_sectional_area_m2, g=STANDARD_GRAVITY):
    """Compressive stress in a single chair leg under a seated occupant.

    First-order model (paper): the occupant weight is shared equally by the
    four legs, so each leg carries a force ``F = (weight_kg / 4) * g`` newtons.
    The axial compressive stress is that force divided by the leg's
    cross-sectional area::

        sigma = (weight_kg / 4 * g) / area          [Pa]

    Parameters
    ----------
    weight_kg : float
        Occupant mass in kilograms (> 0 is expected physically; 0 is allowed
        and yields 0 stress).
    leg_cross_sectional_area_m2 : float
        Cross-sectional area of one leg in square metres (must be > 0).
    g : float
        Gravitational acceleration in m/s^2.

    Returns
    -------
    float
        Compressive stress in pascals (Pa).
    """
    if leg_cross_sectional_area_m2 <= 0.0:
        raise ValueError("leg_cross_sectional_area_m2 must be positive")
    if weight_kg < 0.0:
        raise ValueError("weight_kg must be non-negative")
    force_per_leg_n = (weight_kg / 4.0) * g
    return force_per_leg_n / leg_cross_sectional_area_m2


@dataclass(frozen=True)
class ChairFailureResult:
    """Per-mode and overall first-order chair failure verdict.

    Attributes
    ----------
    leg_compressive_stress_pa : float
        Axial stress in one leg (Pa).
    seat_bending_stress_pa : float
        Peak bending stress in the seat panel (Pa).
    back_stress_pa : float
        Bearing/compressive stress in the backrest (Pa).
    leg_fails : bool
        True if leg compressive stress exceeds the leg yield stress.
    seat_fails : bool
        True if seat bending stress exceeds the seat bending strength.
    back_fails : bool
        True if back stress exceeds the back strength.
    will_break : bool
        Logical OR of the three per-mode failures.
    """

    leg_compressive_stress_pa: float
    seat_bending_stress_pa: float
    back_stress_pa: float
    leg_fails: bool
    seat_fails: bool
    back_fails: bool
    will_break: bool


def seat_bending_stress(weight_kg, seat_thickness_m, seat_length_m, seat_width_m,
                        g=STANDARD_GRAVITY):
    """Peak bending stress in the seat panel, modeled as a simply-supported beam.

    First-order model: treat the seat as a simply-supported beam of span
    ``L = seat_length_m`` carrying a central point load ``W = weight_kg * g``.
    For a central point load on a simply-supported beam the maximum bending
    moment (at mid-span) is::

        M = W * L / 4

    The rectangular cross-section (width ``b = seat_width_m``, thickness
    ``t = seat_thickness_m``) has second moment of area and extreme-fibre
    distance::

        I = b * t**3 / 12
        c = t / 2

    so the peak bending stress is::

        sigma = M * c / I

    Returns
    -------
    float
        Peak bending stress in pascals (Pa).
    """
    if seat_thickness_m <= 0.0 or seat_length_m <= 0.0 or seat_width_m <= 0.0:
        raise ValueError("seat dimensions must be positive")
    load_n = weight_kg * g
    moment = load_n * seat_length_m / 4.0
    second_moment = seat_width_m * seat_thickness_m ** 3 / 12.0
    c = seat_thickness_m / 2.0
    return moment * c / second_moment


def back_stress(weight_kg, back_height_m, back_width_m, load_fraction=1.0 / 3.0,
                g=STANDARD_GRAVITY):
    """Bearing stress in the backrest.

    First-order model (paper): a fraction of the occupant weight rests on the
    backrest -- the paper uses one third. That force is spread over the back's
    face area::

        sigma = (load_fraction * weight_kg * g) / (back_height_m * back_width_m)

    Returns
    -------
    float
        Back stress in pascals (Pa).
    """
    if back_height_m <= 0.0 or back_width_m <= 0.0:
        raise ValueError("back dimensions must be positive")
    force_n = load_fraction * weight_kg * g
    return force_n / (back_height_m * back_width_m)


def will_chair_break(weight_kg, leg_area_m2, leg_yield_pa, seat_thickness_m,
                     seat_length_m, seat_width_m, seat_bending_strength_pa,
                     back_height_m, back_width_m, back_strength_pa,
                     g=STANDARD_GRAVITY):
    """Multi-mode first-order chair failure check (paper ``will_chair_break``).

    Evaluates three independent first-order failure modes and returns their
    individual verdicts plus an overall ``will_break`` that is the logical OR
    of all modes:

    1. Leg compressive failure -- ``chair_leg_compressive_stress`` vs
       ``leg_yield_pa`` (load shared across 4 legs).
    2. Seat bending failure -- ``seat_bending_stress`` (simply-supported beam
       with central load) vs ``seat_bending_strength_pa``.
    3. Backrest failure -- ``back_stress`` (one third of the weight over the
       back face) vs ``back_strength_pa``.

    A mode "fails" when its computed stress strictly exceeds the corresponding
    strength (equal-to-threshold is treated as safe).

    Returns
    -------
    ChairFailureResult
        Frozen dataclass with per-mode stresses, per-mode booleans, and the
        overall ``will_break`` flag.
    """
    leg_sigma = chair_leg_compressive_stress(weight_kg, leg_area_m2, g=g)
    seat_sigma = seat_bending_stress(weight_kg, seat_thickness_m, seat_length_m,
                                     seat_width_m, g=g)
    back_sigma = back_stress(weight_kg, back_height_m, back_width_m, g=g)

    leg_fails = leg_sigma > leg_yield_pa
    seat_fails = seat_sigma > seat_bending_strength_pa
    back_fails = back_sigma > back_strength_pa

    return ChairFailureResult(
        leg_compressive_stress_pa=leg_sigma,
        seat_bending_stress_pa=seat_sigma,
        back_stress_pa=back_sigma,
        leg_fails=leg_fails,
        seat_fails=seat_fails,
        back_fails=back_fails,
        will_break=leg_fails or seat_fails or back_fails,
    )


def can_support(weight_kg, leg_area_m2, leg_yield_pa, seat_thickness_m,
                seat_length_m, seat_width_m, seat_bending_strength_pa,
                back_height_m, back_width_m, back_strength_pa,
                g=STANDARD_GRAVITY):
    """Yes/no query for the paper: can the chair support this occupant?

    Simply the inverse of :func:`will_chair_break`.

    Returns
    -------
    bool
        True if no failure mode is triggered.
    """
    return not will_chair_break(
        weight_kg, leg_area_m2, leg_yield_pa, seat_thickness_m, seat_length_m,
        seat_width_m, seat_bending_strength_pa, back_height_m, back_width_m,
        back_strength_pa, g=g,
    ).will_break


# ---------------------------------------------------------------------------
# 2) Cabinet metrics  (paper sections 7.1.1 and 9.1.3)
# ---------------------------------------------------------------------------


def cabinet_storage_capacity(height, width, depth, thickness, num_shelves=1):
    """Usable interior storage volume of a cabinet.

    First-order model: the interior cavity is the exterior box shrunk by the
    wall thickness on every side::

        interior = (width - 2t) * (depth - 2t) * (height - 2t)

    Each shelf is a full-footprint panel of thickness ``t`` that occupies part
    of that cavity, so the usable volume is::

        capacity = interior - num_shelves * (width - 2t) * (depth - 2t) * t

    All arguments share the same length unit; the result is in that unit cubed.

    Raises
    ------
    ValueError
        If any interior dimension is not strictly positive, if thickness or
        ``num_shelves`` is negative, or if the shelves would occupy more than
        the interior height.
    """
    if thickness < 0.0:
        raise ValueError("thickness must be non-negative")
    if num_shelves < 0:
        raise ValueError("num_shelves must be non-negative")
    inner_w = width - 2.0 * thickness
    inner_d = depth - 2.0 * thickness
    inner_h = height - 2.0 * thickness
    if inner_w <= 0.0 or inner_d <= 0.0 or inner_h <= 0.0:
        raise ValueError("interior dimensions must be positive")
    interior_volume = inner_w * inner_d * inner_h
    shelf_volume = num_shelves * inner_w * inner_d * thickness
    if shelf_volume > interior_volume:
        raise ValueError("shelves occupy more than the interior volume")
    return interior_volume - shelf_volume


def cabinet_material_cost(height, width, depth, thickness, cost_per_volume,
                          num_shelves=1):
    """Material cost of a cabinet's solid parts (walls + shelves).

    First-order model: the material volume is the solid part of the box -- the
    exterior box minus the interior air cavity -- plus the shelf panels::

        wall_volume  = H*W*D - (W - 2t)*(D - 2t)*(H - 2t)
        shelf_volume = num_shelves * (W - 2t)*(D - 2t) * t
        cost = (wall_volume + shelf_volume) * cost_per_volume

    Note the interior cavity here is computed BEFORE subtracting any shelves;
    the shelves are then added explicitly. Cost is in whatever currency unit
    ``cost_per_volume`` implies (currency per length-unit-cubed).

    Raises
    ------
    ValueError
        On non-positive interior dimensions or negative inputs.
    """
    if thickness < 0.0:
        raise ValueError("thickness must be non-negative")
    if num_shelves < 0:
        raise ValueError("num_shelves must be non-negative")
    if cost_per_volume < 0.0:
        raise ValueError("cost_per_volume must be non-negative")
    inner_w = width - 2.0 * thickness
    inner_d = depth - 2.0 * thickness
    inner_h = height - 2.0 * thickness
    if inner_w <= 0.0 or inner_d <= 0.0 or inner_h <= 0.0:
        raise ValueError("interior dimensions must be positive")
    exterior_volume = height * width * depth
    interior_volume = inner_w * inner_d * inner_h
    wall_volume = exterior_volume - interior_volume
    shelf_volume = num_shelves * inner_w * inner_d * thickness
    return (wall_volume + shelf_volume) * cost_per_volume


def shelf_sag_load_capacity(span_L, depth_b, thickness_t, modulus_E,
                            delta_allow):
    """Allowable total load on a shelf before it sags past a deflection limit.

    "Sagulator"-style first-order model (the paper references the woodworking
    Sagulator). The shelf is modeled as a simply-supported beam carrying a
    uniformly distributed load ``w`` (force per unit length). The mid-span
    deflection of such a beam is::

        delta = 5 * w * L**4 / (384 * E * I),   I = b * t**3 / 12

    Setting ``delta = delta_allow`` and solving for the allowable distributed
    load::

        w_allow = 384 * E * I * delta_allow / (5 * L**4)

    The returned value is the allowable TOTAL load ``w_allow * L`` (a force),
    i.e. the total uniformly-distributed weight the shelf can carry before the
    mid-span deflection reaches ``delta_allow``.

    Assumptions: simply-supported ends, linear-elastic material, uniform load,
    small deflection, shelf bends about its thickness axis (``I = b*t^3/12``).

    Parameters
    ----------
    span_L : float
        Unsupported span between supports (length units, > 0).
    depth_b : float
        Shelf depth / beam width (> 0).
    thickness_t : float
        Shelf thickness (> 0).
    modulus_E : float
        Material Young's modulus (force / length^2, > 0).
    delta_allow : float
        Maximum allowable mid-span deflection (length units, > 0).

    Returns
    -------
    float
        Allowable total distributed load (force units).
    """
    if span_L <= 0.0 or depth_b <= 0.0 or thickness_t <= 0.0:
        raise ValueError("shelf geometry must be positive")
    if modulus_E <= 0.0:
        raise ValueError("modulus_E must be positive")
    if delta_allow <= 0.0:
        raise ValueError("delta_allow must be positive")
    second_moment = depth_b * thickness_t ** 3 / 12.0
    w_allow = 384.0 * modulus_E * second_moment * delta_allow / (5.0 * span_L ** 4)
    return w_allow * span_L


def cabinet_wheelchair_accessibility_score(height, depth,
                                           ideal_height=0.9,
                                           ideal_depth=0.6):
    """Bounded 0..10 wheelchair-accessibility heuristic for a cabinet.

    Paper intent (section 9.1.3): assign a "higher accessibility score to
    shorter and deeper cabinets". A seated wheelchair user reaches lower
    surfaces and benefits from deeper (more reachable-from-front) storage.

    Chosen first-order heuristic, monotonic by construction:

      * A height term that DECREASES with height, reaching the top of its
        range at/below ``ideal_height`` metres (a comfortable seated reach
        height, default 0.90 m) and falling linearly to zero at twice that.
      * A depth term that INCREASES with depth, saturating at ``ideal_depth``
        metres (default 0.60 m).

    The two terms are averaged and scaled to [0, 10], then clamped. The result
    is strictly non-increasing in ``height`` and non-decreasing in ``depth``.

    Returns
    -------
    float
        Accessibility score in the closed interval [0, 10].
    """
    if height <= 0.0 or depth <= 0.0:
        raise ValueError("height and depth must be positive")
    if ideal_height <= 0.0 or ideal_depth <= 0.0:
        raise ValueError("ideal_height and ideal_depth must be positive")

    # Height term: 1.0 at height <= ideal_height, linearly to 0.0 at 2*ideal.
    height_term = 1.0 - (height - ideal_height) / ideal_height
    if height_term > 1.0:
        height_term = 1.0
    if height_term < 0.0:
        height_term = 0.0

    # Depth term: 0.0 at depth 0, linearly to 1.0 at ideal_depth, saturating.
    depth_term = depth / ideal_depth
    if depth_term > 1.0:
        depth_term = 1.0
    if depth_term < 0.0:
        depth_term = 0.0

    score = 10.0 * 0.5 * (height_term + depth_term)
    if score < 0.0:
        score = 0.0
    if score > 10.0:
        score = 10.0
    return score


# ---------------------------------------------------------------------------
# 3) Quadcopter  (paper section 7.1.2) -- first-order approximations
# ---------------------------------------------------------------------------


def copter_hover_time_min(battery_capacity_mAh, battery_voltage_V,
                          total_current_draw_A, usable_fraction=0.8):
    """First-order quadcopter hover endurance, in minutes.

    Energy-based first-order approximation: only a fraction of the nominal
    battery charge is usably deliverable (to protect the cells). Hover time is
    usable charge divided by the total current the craft draws while hovering::

        hover_time_min = (capacity_Ah * usable_fraction) / current_A * 60

    where ``capacity_Ah = battery_capacity_mAh / 1000``. This ignores voltage
    sag, motor efficiency curves, and payload dynamics -- it is a first-order
    endurance bound only. ``battery_voltage_V`` is accepted for interface
    completeness (pack energy context) but does not enter the charge balance.

    Returns
    -------
    float
        Hover time in minutes.
    """
    if battery_capacity_mAh <= 0.0:
        raise ValueError("battery_capacity_mAh must be positive")
    if battery_voltage_V <= 0.0:
        raise ValueError("battery_voltage_V must be positive")
    if total_current_draw_A <= 0.0:
        raise ValueError("total_current_draw_A must be positive")
    if not 0.0 < usable_fraction <= 1.0:
        raise ValueError("usable_fraction must be in (0, 1]")
    capacity_Ah = battery_capacity_mAh / 1000.0
    hours = (capacity_Ah * usable_fraction) / total_current_draw_A
    return hours * 60.0


def copter_max_range_km(hover_time_min, cruise_speed_kmh):
    """First-order maximum quadcopter range, in kilometres.

    Optimistic first-order bound: assume the craft can cruise at
    ``cruise_speed_kmh`` for its entire hover endurance::

        range_km = (hover_time_min / 60) * cruise_speed_kmh

    This is optimistic because forward flight and hovering do not draw the same
    power; it treats endurance as if it were all available for cruise. Use as
    an upper bound only.

    Returns
    -------
    float
        Range in kilometres.
    """
    if hover_time_min < 0.0:
        raise ValueError("hover_time_min must be non-negative")
    if cruise_speed_kmh < 0.0:
        raise ValueError("cruise_speed_kmh must be non-negative")
    return (hover_time_min / 60.0) * cruise_speed_kmh
