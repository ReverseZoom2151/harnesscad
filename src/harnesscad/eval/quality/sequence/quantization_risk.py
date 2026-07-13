"""Forecast scale-relative CAD invalidity introduced by parameter quantization."""

from __future__ import annotations


def quantization_risks(*, step, extrusion=None, radii=(), clearances=(), arc_points=()):
    if step <= 0:
        raise ValueError("step must be positive")
    quantize = lambda value: round(float(value)/step)*step
    issues = []
    if extrusion is not None and quantize(extrusion) == 0:
        issues.append("zero-extrusion")
    rounded_radii = [quantize(value) for value in radii]
    if len(rounded_radii) != len(set(rounded_radii)):
        issues.append("coincident-radii")
    if any(abs(quantize(value)) == 0 for value in clearances):
        issues.append("collapsed-clearance")
    for a, b, c in arc_points:
        qa, qb, qc = ([quantize(value) for value in point] for point in (a, b, c))
        cross = (qb[0]-qa[0])*(qc[1]-qa[1])-(qb[1]-qa[1])*(qc[0]-qa[0])
        if cross == 0:
            issues.append("collinear-arc")
    return tuple(dict.fromkeys(issues))
