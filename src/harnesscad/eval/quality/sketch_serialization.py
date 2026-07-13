"""Point-based, redundant sketch serialization with consistency validation."""

from __future__ import annotations

import math


def serialize_line(entity_id, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    direction = (dx / length, dy / length) if length else (0.0, 0.0)
    return {"id": entity_id, "type": "line", "start": tuple(start), "end": tuple(end),
            "length": length, "direction": direction,
            "angle": math.atan2(dy, dx) if length else 0.0}


def serialize_circle(entity_id, center, radius):
    if radius <= 0:
        raise ValueError("radius must be positive")
    return {"id": entity_id, "type": "circle", "center": tuple(center),
            "radius": float(radius), "diameter": float(radius) * 2}


def serialize_arc(entity_id, start, mid, end):
    return {"id": entity_id, "type": "arc", "start": tuple(start),
            "mid": tuple(mid), "end": tuple(end)}


def validate_redundancy(item, tolerance=1e-6):
    issues = []
    if item.get("type") == "line":
        expected = serialize_line(item.get("id", ""), item["start"], item["end"])
        for key in ("length", "angle"):
            if key in item and abs(float(item[key]) - expected[key]) > tolerance:
                issues.append(f"inconsistent-{key}")
        if "direction" in item and math.dist(tuple(item["direction"]),
                                              expected["direction"]) > tolerance:
            issues.append("inconsistent-direction")
    if item.get("type") == "circle" and "diameter" in item:
        if abs(float(item["diameter"]) - 2 * float(item["radius"])) > tolerance:
            issues.append("inconsistent-diameter")
    return tuple(issues)


def serialize_sketch(primitives, constraints=()):
    ids = [item["id"] for item in primitives]
    if len(ids) != len(set(ids)):
        raise ValueError("primitive ids must be unique")
    known = set(ids)
    for constraint in constraints:
        if any(ref not in known for ref in constraint.get("primitives", ())):
            raise ValueError("constraint references unknown primitive")
    return {"schema": "harnesscad.sketch.v1",
            "primitives": tuple(sorted(primitives, key=lambda item: item["id"])),
            "constraints": tuple(sorted(constraints, key=lambda item: item["id"]))}
