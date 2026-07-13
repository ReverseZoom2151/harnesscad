"""SizingCalc — an engineering-sizing assistant.

Given a *requirement* (a load/torque/pressure/ratio), recall the governing
mechanical-design formula and return a concrete driving dimension. Each formula
is a small, correctly-implemented, cited closed-form solution — the kind an
engineer reaches for at the layout stage to rough-size a feature long before any
detailed geometry exists.

Everything here is stdlib-only and deterministic: the same requirement always
produces the same number. Results are plain dicts so they drop straight into the
:class:`skeleton.layout.Skeleton` parameter table.

Formula registry (all SI-consistent; the canonical units are N, mm, MPa = N/mm^2,
N*mm for torque, so stresses and dimensions come out in mm / MPa directly):

  - shaft_diameter_torsion   solid round shaft in pure torsion
  - plate_thickness_bending  simply-supported rectangular strip under pressure
  - bolt_count_shear         number of bolts to carry a transverse shear load
  - gear_teeth_from_ratio    pinion/gear tooth counts from ratio + centre distance
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Tuple


@dataclass(frozen=True)
class SizingFormula:
    """One governing formula.

    ``compute(inputs, safety_factor) -> value`` returns the sized dimension in the
    formula's ``dimension`` unit. ``inputs`` are the required requirement keys;
    ``citation`` names the textbook relation so a reviewer can audit it.
    """

    name: str
    dimension: str
    inputs: Tuple[str, ...]
    compute: Callable[[Mapping[str, float], float], float]
    citation: str = ""

    def describe(self) -> dict:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "inputs": list(self.inputs),
            "citation": self.citation,
        }


# ---------------------------------------------------------------------------
# The formulas. Each is the standard closed form; the docstring cites it.
# The safety factor scales the *demand* (load/torque/pressure) up, which is the
# conservative convention: a bigger effective load -> a bigger dimension.
# ---------------------------------------------------------------------------
def _shaft_diameter_torsion(inp: Mapping[str, float], sf: float) -> float:
    """Solid circular shaft in pure torsion.

    Elastic torsion formula tau = T*r / J, with J = pi*d^4/32 and r = d/2, gives
    the surface shear stress tau = 16*T / (pi*d^3). Solving for the diameter at
    the allowable shear stress:

        d = ( 16 * T / (pi * tau_allow) ) ** (1/3)

    Ref: Shigley, *Mechanical Engineering Design*, torsion of round bars
    (eq. tau = 16 T / (pi d^3)); Hibbeler, *Mechanics of Materials*, ch. Torsion.

    Inputs: torque [N*mm], allowable_shear [MPa]. Returns diameter [mm].
    """
    torque = float(inp["torque"]) * sf
    tau = float(inp["allowable_shear"])
    return (16.0 * torque / (math.pi * tau)) ** (1.0 / 3.0)


def _plate_thickness_bending(inp: Mapping[str, float], sf: float) -> float:
    """Thickness of a simply-supported rectangular strip under uniform pressure.

    Model a unit-width (or width ``b``) strip of span L carrying a uniform load
    w = p*b per unit length as a simply-supported beam. Maximum bending moment is
    M = w*L^2/8. For a rectangular section the modulus is Z = b*t^2/6 and the
    bending stress sigma = M/Z. Setting sigma = sigma_allow and solving for t:

        t = sqrt( 6 * M / (b * sigma_allow) )
          = sqrt( 6 * (p*b*L^2/8) / (b * sigma_allow) )
          = L * sqrt( 0.75 * p / sigma_allow )        (independent of width b)

    Ref: Gere & Goodno, *Mechanics of Materials*, simply-supported beam under
    uniform load M_max = wL^2/8; rectangular section modulus Z = b t^2 / 6.

    Inputs: pressure [MPa], span [mm], allowable_stress [MPa]. Returns t [mm].
    """
    p = float(inp["pressure"]) * sf
    span = float(inp["span"])
    sigma = float(inp["allowable_stress"])
    b = float(inp.get("width", 1.0)) if hasattr(inp, "get") else 1.0
    moment = p * b * span * span / 8.0
    return math.sqrt(6.0 * moment / (b * sigma))


def _bolt_count_shear(inp: Mapping[str, float], sf: float) -> float:
    """Number of identical bolts to carry a transverse shear load in single shear.

    Each bolt of nominal diameter d has shear area A = pi*d^2/4 and capacity
    P = A * tau_allow. The required count is the load divided by per-bolt
    capacity, rounded up:

        n = ceil( F / ( (pi*d^2/4) * tau_allow ) )

    Ref: Shigley, bolted-joint shear; AISC bolt shear-strength basis
    (V = A_b * F_v). Returns an integer count (as a float for a uniform API).

    Inputs: load [N], bolt_diameter [mm], allowable_shear [MPa].
    """
    load = float(inp["load"]) * sf
    d = float(inp["bolt_diameter"])
    tau = float(inp["allowable_shear"])
    per_bolt = (math.pi * d * d / 4.0) * tau
    return float(math.ceil(load / per_bolt))


def _gear_teeth_from_ratio(inp: Mapping[str, float], sf: float) -> float:
    """Pinion tooth count from a gear ratio and centre distance at a given module.

    For a standard external spur-gear pair the centre distance is
    C = m*(N1 + N2)/2 with N2 = i*N1 (i = gear ratio). Substituting:

        C = m*N1*(1 + i)/2   ->   N1 = 2*C / ( m*(1 + i) )

    The pinion count is rounded to the nearest whole tooth. (The mating gear is
    N2 = round(i*N1); see :meth:`SizingCalc.gear_pair` for both.) Safety factor
    does not apply to a kinematic count, so ``sf`` is ignored here.

    Ref: Shigley, spur-gear geometry, C = m(N_p + N_g)/2, m = module.

    Inputs: ratio [-], center_distance [mm], module [mm]. Returns N1 [teeth].
    """
    ratio = float(inp["ratio"])
    c = float(inp["center_distance"])
    m = float(inp["module"])
    n1 = 2.0 * c / (m * (1.0 + ratio))
    return float(round(n1))


def default_formulas() -> Dict[str, SizingFormula]:
    """The built-in formula registry (name -> SizingFormula)."""
    formulas = [
        SizingFormula(
            name="shaft_diameter_torsion",
            dimension="diameter_mm",
            inputs=("torque", "allowable_shear"),
            compute=_shaft_diameter_torsion,
            citation="tau = 16 T / (pi d^3)  ->  d = (16 T / (pi tau))^(1/3) "
                     "[Shigley, Mechanical Engineering Design]",
        ),
        SizingFormula(
            name="plate_thickness_bending",
            dimension="thickness_mm",
            inputs=("pressure", "span", "allowable_stress"),
            compute=_plate_thickness_bending,
            citation="M = wL^2/8, Z = b t^2/6, sigma = M/Z  ->  "
                     "t = L*sqrt(0.75 p / sigma) [Gere & Goodno]",
        ),
        SizingFormula(
            name="bolt_count_shear",
            dimension="count",
            inputs=("load", "bolt_diameter", "allowable_shear"),
            compute=_bolt_count_shear,
            citation="n = ceil( F / ((pi d^2/4) tau) )  [AISC/Shigley bolt shear]",
        ),
        SizingFormula(
            name="gear_teeth_from_ratio",
            dimension="teeth",
            inputs=("ratio", "center_distance", "module"),
            compute=_gear_teeth_from_ratio,
            citation="C = m(N1+N2)/2, N2 = i N1  ->  N1 = 2C/(m(1+i)) [Shigley]",
        ),
    ]
    return {f.name: f for f in formulas}


class SizingCalc:
    """Engineering-sizing assistant over a registry of governing formulas."""

    def __init__(self, formulas: Dict[str, SizingFormula] = None) -> None:
        self.formulas: Dict[str, SizingFormula] = (
            dict(formulas) if formulas is not None else default_formulas())

    # --- introspection ----------------------------------------------------
    def names(self) -> List[str]:
        return list(self.formulas)

    def formula(self, name: str) -> SizingFormula:
        return self.formulas[name]

    def register(self, formula: SizingFormula) -> None:
        self.formulas[formula.name] = formula

    # --- the one call -----------------------------------------------------
    def size(self, requirement: Mapping[str, float]) -> dict:
        """Size a dimension from a requirement.

        ``requirement`` names a formula via ``"formula"`` (or the alias
        ``"kind"``) and supplies that formula's inputs plus an optional
        ``"safety_factor"`` (default 1.0). Returns::

            {dimension, formula, inputs, value, safety_factor, citation}

        Raises KeyError for an unknown formula and for any missing input, so a
        bad requirement fails loudly rather than silently mis-sizing.
        """
        name = requirement.get("formula") or requirement.get("kind")
        if name is None:
            raise KeyError("requirement must name a 'formula' (or 'kind')")
        if name not in self.formulas:
            raise KeyError(
                f"unknown sizing formula '{name}'; known: {sorted(self.formulas)}")
        f = self.formulas[name]
        sf = float(requirement.get("safety_factor", 1.0))
        missing = [k for k in f.inputs if k not in requirement]
        if missing:
            raise KeyError(
                f"formula '{name}' requires inputs {list(f.inputs)}; missing {missing}")
        inputs = {k: float(requirement[k]) for k in f.inputs}
        value = f.compute(requirement, sf)
        return {
            "dimension": f.dimension,
            "formula": f.name,
            "inputs": inputs,
            "value": value,
            "safety_factor": sf,
            "citation": f.citation,
        }

    # --- convenience ------------------------------------------------------
    def gear_pair(self, ratio: float, center_distance: float, module: float) -> dict:
        """Both tooth counts for an external spur pair (pinion N1 + gear N2)."""
        res = self.size({
            "formula": "gear_teeth_from_ratio",
            "ratio": ratio, "center_distance": center_distance, "module": module,
        })
        n1 = res["value"]
        res["pinion_teeth"] = n1
        res["gear_teeth"] = float(round(ratio * n1))
        return res
