"""sizing — the engineering-sizing front-of-pipeline for HarnessCAD.

Recalls governing mechanical formulas and turns *requirements* (torque, pressure,
load, ratio) into *numeric driving dimensions* (shaft diameter, plate thickness,
bolt count, gear teeth). These numbers feed the top-down :mod:`skeleton` master
layout's parameter table BEFORE any near-final geometry is emitted — the
layout-first / rough-sizing / constraint-reasoning wedge that carries the value.
"""

from __future__ import annotations

from harnesscad.domain.sizing.calc import SizingCalc, SizingFormula, default_formulas

__all__ = ["SizingCalc", "SizingFormula", "default_formulas"]
