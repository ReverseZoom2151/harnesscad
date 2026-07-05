"""Seeded sketch-boolean recipes with an injected planar geometry adapter."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class SketchPrimitive:
    kind: str
    parameters: tuple[float, ...]
    operation: str


def sketch_recipe(seed, *, minimum=3, maximum=8):
    rng = random.Random(seed)
    count = rng.randint(minimum, maximum)
    items = []
    for index in range(count):
        if rng.random() < .5:
            params = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(.05, .5))
            kind = "circle"
        else:
            params = (rng.uniform(-1, 1), rng.uniform(-1, 1),
                      rng.uniform(.1, 1), rng.uniform(.1, 1),
                      rng.uniform(0, 180))
            kind = "rotated_rectangle"
        items.append(SketchPrimitive(kind, params, "union" if index == 0
                                     or rng.random() < .6 else "cut"))
    return tuple(items)


def realize_recipe(recipe, adapter):
    try:
        result = adapter.empty()
        for primitive in recipe:
            shape = adapter.primitive(primitive.kind, primitive.parameters)
            result = adapter.boolean(result, shape, primitive.operation)
        loops = adapter.boundary_loops(result)
        if not loops:
            return {"accepted": False, "reason": "empty-result", "loops": ()}
        if adapter.intersects(loops):
            return {"accepted": False, "reason": "intersecting-loops", "loops": loops}
        if any(adapter.length(edge) <= 0 for loop in loops for edge in loop):
            return {"accepted": False, "reason": "zero-length-edge", "loops": loops}
        return {"accepted": True, "reason": "", "loops": loops}
    except Exception as exc:
        return {"accepted": False, "reason": f"adapter-error:{type(exc).__name__}",
                "loops": ()}
