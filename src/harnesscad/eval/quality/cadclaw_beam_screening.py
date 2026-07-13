"""First-order static screening for belt-driven gantry frames.

A small, correct, stdlib-only library of the closed-form structural
checks a machine-frame assembly is screened against before a full FEA:

  * **Section properties** of a solid rectangular bar — area, second
    moments ``Iy`` / ``Iz``, elastic section moduli ``Sy`` / ``Sz``, and
    the Roark closed-form torsion constant ``J``. Depends only on width
    and height, so it is the deterministic bridge from a bounding-box
    signature to a bending calculation.
  * **Simply-supported beam deflection** under a central point load plus
    the beam's own distributed self-weight — the two load terms a gantry
    X-beam actually sees. Returns each term separately so the dominant
    one is visible.
  * **Motor torque budget** for a belt-driven axis: sums the
    acceleration, rolling-friction and (optionally) gravity forces,
    converts to required pulley torque through the belt efficiency, and
    compares against the derated holding torque with a safety factor.
  * **Belt tension** against published breaking / working loads.

Everything is a first-order screening estimate (single dominant mode,
idealized supports, linear-elastic material, small deflection) — a fast
gate, not a substitute for FEA. It is intentionally distinct from the
chair/shelf first-order scorers elsewhere in ``quality`` (which model
panel bending and shelf sag) and from any CAD-kernel geometry: this
module is pure arithmetic on SI inputs.

Units are SI unless a name says otherwise (``_mm``, ``_kg`` ...): metres,
newtons, newton-metres, pascals, kg, kg/m^3, m/s^2. ``STANDARD_GRAVITY``
is exposed so callers can pin it and stay deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

STANDARD_GRAVITY = 9.80665  # m/s^2


# ---------------------------------------------------------------------------
# Section properties
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SectionProperties:
    """Cross-section properties of a prismatic beam about its local axes.

    Local convention: ``width`` runs along local y, ``height`` along
    local z. For a flat bar standing on edge (height >> width) local z is
    the strong bending axis, so ``Iz`` / ``Sz`` are the strong-axis
    values.
    """
    area: float
    Iy: float
    Iz: float
    Sy: float
    Sz: float
    J: float

    @property
    def strong_axis_modulus(self) -> float:
        return max(self.Sy, self.Sz)


def rectangular_section(width: float, height: float) -> SectionProperties:
    """Section properties of a solid rectangle ``width`` x ``height``.

    ``J`` uses the Roark closed-form torsion constant for a solid
    rectangle, accurate at any aspect ratio; for a thin bar it converges
    to (1/3)*long*short^3.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    b, h = float(width), float(height)
    area = b * h
    Iz = b * h ** 3 / 12.0
    Iy = h * b ** 3 / 12.0
    Sz = b * h ** 2 / 6.0
    Sy = h * b ** 2 / 6.0
    long_s, short_s = (h, b) if h >= b else (b, h)
    r = short_s / long_s
    J = long_s * short_s ** 3 * (1.0 / 3.0 - 0.21 * r * (1.0 - r ** 4 / 12.0))
    return SectionProperties(area=area, Iy=Iy, Iz=Iz, Sy=Sy, Sz=Sz, J=J)


# ---------------------------------------------------------------------------
# Beam deflection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeflectionResult:
    point_load_mm: float
    self_weight_mm: float
    total_mm: float
    limit_mm: float
    passed: bool


def simply_supported_deflection(span_m: float, point_load_kg: float,
                                I_m4: float, beam_kg_per_m: float,
                                E_Pa: float = 68.9e9,
                                limit_mm: float = 0.5,
                                gravity: float = STANDARD_GRAVITY,
                                ) -> DeflectionResult:
    """Mid-span deflection of a simply-supported beam.

    Superposes a central point load ``P*L^3/(48 E I)`` and the beam's
    distributed self-weight ``5 w L^4/(384 E I)``. ``E_Pa`` defaults to
    aluminium (~68.9 GPa).
    """
    if span_m <= 0:
        raise ValueError("span must be positive")
    if I_m4 <= 0 or E_Pa <= 0:
        raise ValueError("I and E must be positive")
    P = point_load_kg * gravity
    w = beam_kg_per_m * gravity  # N/m
    d_point = (P * span_m ** 3) / (48.0 * E_Pa * I_m4)
    d_weight = (5.0 * w * span_m ** 4) / (384.0 * E_Pa * I_m4)
    total_mm = (d_point + d_weight) * 1000.0
    return DeflectionResult(
        point_load_mm=d_point * 1000.0,
        self_weight_mm=d_weight * 1000.0,
        total_mm=total_mm,
        limit_mm=limit_mm,
        passed=total_mm <= limit_mm,
    )


# ---------------------------------------------------------------------------
# Motor torque budget
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MotorBudgetResult:
    force_accel_N: float
    force_friction_N: float
    force_gravity_N: float
    force_total_N: float
    torque_required_Nm: float
    torque_available_Nm: float
    safety_factor: float
    passed: bool


def motor_torque_budget(mass_kg: float, n_motors: int,
                        pulley_radius_m: float, motor_torque_Nm: float,
                        accel_m_s2: float = 0.5,
                        gravity_axis: bool = False,
                        friction_coeff: float = 0.01,
                        belt_efficiency: float = 0.95,
                        torque_derating: float = 0.7,
                        min_safety: float = 1.5,
                        gravity: float = STANDARD_GRAVITY,
                        ) -> MotorBudgetResult:
    """Torque budget for a belt-driven axis (per motor).

    Sums the inertial, rolling-friction and (optional) gravity force on
    the moving mass, splits it across ``n_motors``, converts to required
    pulley torque through ``belt_efficiency``, and compares against the
    derated holding torque.
    """
    if n_motors <= 0:
        raise ValueError("n_motors must be >= 1")
    if pulley_radius_m <= 0 or belt_efficiency <= 0:
        raise ValueError("pulley_radius and belt_efficiency must be positive")
    F_accel = mass_kg * accel_m_s2
    F_friction = mass_kg * gravity * friction_coeff
    F_gravity = mass_kg * gravity if gravity_axis else 0.0
    F_total = F_accel + F_friction + F_gravity
    F_per_motor = F_total / n_motors
    T_required = F_per_motor * pulley_radius_m / belt_efficiency
    T_available = motor_torque_Nm * torque_derating
    safety = T_available / T_required if T_required > 0 else float("inf")
    return MotorBudgetResult(
        force_accel_N=F_accel,
        force_friction_N=F_friction,
        force_gravity_N=F_gravity,
        force_total_N=F_total,
        torque_required_Nm=T_required,
        torque_available_Nm=T_available,
        safety_factor=safety,
        passed=safety >= min_safety,
    )


# ---------------------------------------------------------------------------
# Belt tension
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BeltTensionResult:
    tension_per_belt_N: float
    breaking_N: float
    working_N: float
    safety_to_break: float
    safety_to_working: float
    passed: bool


def belt_tension(force_N: float, n_belts: int = 1,
                 breaking_N: float = 900.0, working_N: float = 450.0,
                 min_safety: float = 2.0) -> BeltTensionResult:
    """Belt tension safety against breaking and working limits."""
    if n_belts <= 0:
        raise ValueError("n_belts must be >= 1")
    per_belt = force_N / n_belts
    s_break = breaking_N / per_belt if per_belt > 0 else float("inf")
    s_work = working_N / per_belt if per_belt > 0 else float("inf")
    return BeltTensionResult(
        tension_per_belt_N=per_belt,
        breaking_N=breaking_N,
        working_N=working_N,
        safety_to_break=s_break,
        safety_to_working=s_work,
        passed=s_work >= min_safety,
    )
