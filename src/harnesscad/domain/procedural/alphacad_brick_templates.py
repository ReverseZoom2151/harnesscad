"""Parametric deterministic voxel-object templates (AlphaCAD).

Source: ``AlphaCAD-main`` (``summit-demo/cad_templates.py``), a BrickGPT demo.
The BrickGPT paper itself (already implemented in the harness as
``geometry/brick_structure.py`` and friends) generates brick structures with a
*trained* language model. AlphaCAD adds a purely deterministic fallback: a
library of hand-written *parametric* generators that emit a 1-unit-tall cuboid
brick layout for each object category directly from ``(width, depth, height)``
plus a seed, with no model in the loop.

This module reimplements that idea, stdlib only and fully deterministic. Each
generator selects a *style* from its dimensions, then fills a voxel grid with
bricks tagged by ``part_type`` (leg / surface / wall / seat / ...). Randomness
is confined to a local ``random.Random(seed)`` -- never the global RNG -- so a
given ``(dims, seed)`` always yields byte-identical output.

A brick is a dict ``{'id', 'x', 'y', 'z', 'layer', 'part_type', 'description'}``
and a model is a dict ``{'width','depth','height','total_bricks','bricks',
'object_type','description','style','features'}``. This is the same shape the
scoring / composition / consensus modules in this package consume.
"""

from __future__ import annotations

import random
from typing import Callable

Brick = dict
Model = dict


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _brick(bid: int, x: int, y: int, z: int, part_type: str, description: str) -> Brick:
    return {
        "id": bid,
        "x": x,
        "y": y,
        "z": z,
        "layer": z + 1,
        "part_type": part_type,
        "description": description,
    }


def _finish(bricks: list[Brick], width: int, depth: int, height: int,
            object_type: str, description: str, style: str | None = None,
            features: dict | None = None) -> Model:
    model: Model = {
        "width": width,
        "depth": depth,
        "height": height,
        "total_bricks": len(bricks),
        "bricks": bricks,
        "object_type": object_type,
        "description": description,
    }
    if style is not None:
        model["style"] = style
    if features is not None:
        model["features"] = features
    return model


def generate_table(width: int = 6, depth: int = 4, height: int = 3, seed: int = 42) -> Model:
    """A table: four (or three) legs plus a patterned top surface."""
    rng = random.Random(seed)
    width = _clamp(width, 4, 12)
    depth = _clamp(depth, 3, 8)
    height = _clamp(height, 2, 4)

    aspect = width / depth
    if height <= 2:
        style = "coffee"
    elif aspect > 2.0:
        style = "console"
    elif width >= 8 or depth >= 6:
        style = "dining"
    else:
        style = "desk"

    if style == "coffee":
        leg_thickness = 2 if width >= 6 and depth >= 4 else 1
        has_center = rng.random() < 0.4
    elif style == "console":
        leg_thickness = 1
        has_center = rng.random() < 0.3
    elif style == "dining":
        leg_thickness = 2 if width >= 8 else 1
        has_center = rng.random() < 0.2
    else:
        leg_thickness = 1
        has_center = rng.random() < 0.1

    if style == "console" and rng.random() < 0.3:
        legs = [(1, 0), (width - 2, 0), (width // 2, depth - 1)]
    else:
        inset = leg_thickness - 1
        legs = [
            (inset, inset), (width - 1 - inset, inset),
            (inset, depth - 1 - inset), (width - 1 - inset, depth - 1 - inset),
        ]

    bricks: list[Brick] = []
    bid = 0
    for h in range(height - 1):
        for lx, ly in legs:
            for dx in range(leg_thickness):
                for dy in range(leg_thickness):
                    x = min(lx + dx, width - 1)
                    y = min(ly + dy, depth - 1)
                    bricks.append(_brick(bid, x, y, h, "leg", f"{style} table leg"))
                    bid += 1

    if has_center and width >= 6 and depth >= 4:
        cx, cy = width // 2, depth // 2
        for h in range(height - 1):
            bricks.append(_brick(bid, cx, cy, h, "leg", "center support"))
            bid += 1

    surface_style = rng.choice(["solid", "border", "cross_pattern"])
    for x in range(width):
        for y in range(depth):
            place = True
            if surface_style == "border" and width >= 6 and depth >= 5:
                place = x <= 1 or x >= width - 2 or y <= 1 or y >= depth - 2
            elif surface_style == "cross_pattern" and width >= 6 and depth >= 4:
                place = (x == width // 2 or y == depth // 2 or x == 0
                         or x == width - 1 or y == 0 or y == depth - 1)
            if place:
                bricks.append(_brick(bid, x, y, height - 1, "surface", f"{style} table surface"))
                bid += 1

    return _finish(
        bricks, width, depth, height, "table",
        f"{style} table {width}x{depth}x{height} with {surface_style} top",
        style,
        {"leg_thickness": leg_thickness, "center_support": has_center,
         "surface_pattern": surface_style},
    )


def generate_chair(width: int = 3, depth: int = 3, height: int = 4, seed: int = 42) -> Model:
    """A chair: legs, seat, backrest, optional armrests."""
    rng = random.Random(seed)
    width = _clamp(width, 2, 6)
    depth = _clamp(depth, 2, 6)
    height = _clamp(height, 3, 6)

    if width >= 4:
        style = "armchair"
    elif height >= 5:
        style = "highback"
    elif depth <= 2:
        style = "stool"
    else:
        style = "dining"

    seat_height = max(1, height // 2)
    back_height = height - 1
    if style == "armchair":
        has_arms = rng.random() < 0.7
        seat_pattern = rng.choice(["solid", "center_void"])
    elif style == "highback":
        has_arms = rng.random() < 0.3
        seat_pattern = "solid"
        back_height = height
    elif style == "stool":
        has_arms = False
        seat_pattern = "solid"
        back_height = seat_height + 1
    else:
        has_arms = rng.random() < 0.2
        seat_pattern = rng.choice(["solid", "border"])

    bricks: list[Brick] = []
    bid = 0
    legs = [(0, 0), (width - 1, 0), (0, depth - 1), (width - 1, depth - 1)]
    for h in range(seat_height):
        for lx, ly in legs:
            bricks.append(_brick(bid, lx, ly, h, "leg", f"{style} chair leg"))
            bid += 1

    for x in range(width):
        for y in range(depth):
            place = True
            if seat_pattern == "center_void" and width >= 4 and depth >= 3:
                place = x == 0 or x == width - 1 or y == 0 or y == depth - 1
            elif seat_pattern == "border" and width >= 3 and depth >= 3:
                place = (x == 0 or x == width - 1 or y == 0 or y == depth - 1
                         or (x == 1 and y == 1) or (x == width - 2 and y == depth - 2))
            if place:
                bricks.append(_brick(bid, x, y, seat_height, "seat", f"{style} seat"))
                bid += 1

    if style == "highback":
        back_cols = range(width)
    elif style == "stool":
        back_cols = [width // 2] if width > 2 else [0]
    else:
        back_cols = range(max(1, width - 1)) if width >= 4 else range(width)

    back_y = depth - 1
    for h in range(seat_height + 1, back_height + 1):
        for x in back_cols:
            bricks.append(_brick(bid, x, back_y, h, "back", f"{style} chair back"))
            bid += 1

    if has_arms and width >= 3:
        arm_h = seat_height + 1
        for y in range(depth):
            bricks.append(_brick(bid, 0, y, arm_h, "armrest", "left armrest"))
            bid += 1
        for y in range(depth):
            bricks.append(_brick(bid, width - 1, y, arm_h, "armrest", "right armrest"))
            bid += 1

    return _finish(
        bricks, width, depth, height, "chair",
        f"{style} chair {width}x{depth}x{height}" + (" with armrests" if has_arms else ""),
        style,
        {"armrests": has_arms, "seat_pattern": seat_pattern,
         "back_style": "full" if style == "highback" else "standard"},
    )


def generate_tower(width: int = 3, depth: int = 3, height: int = 8, seed: int = 42) -> Model:
    """A tower: solid two-layer base with hollow perimeter walls above."""
    width = _clamp(width, 2, 8)
    depth = _clamp(depth, 2, 8)
    height = _clamp(height, 4, 20)
    bricks: list[Brick] = []
    bid = 0
    for h in range(2):
        for x in range(width):
            for y in range(depth):
                bricks.append(_brick(bid, x, y, h, "base", "tower base"))
                bid += 1
    for h in range(2, height):
        for x in range(width):
            for y in range(depth):
                is_perimeter = x == 0 or x == width - 1 or y == 0 or y == depth - 1
                if is_perimeter or width <= 3 or depth <= 3:
                    bricks.append(_brick(bid, x, y, h, "wall", "tower wall"))
                    bid += 1
    return _finish(bricks, width, depth, height, "tower",
                   f"tower {width}x{depth}x{height} with hollow center")


def generate_car(width: int = 6, depth: int = 3, height: int = 2, seed: int = 42) -> Model:
    """A car: full lower body plus a centred upper cabin."""
    width = _clamp(width, 4, 10)
    depth = _clamp(depth, 2, 4)
    height = _clamp(height, 2, 3)
    bricks: list[Brick] = []
    bid = 0
    for x in range(width):
        for y in range(depth):
            bricks.append(_brick(bid, x, y, 0, "body", "car body"))
            bid += 1
    if height > 1:
        cabin_start = width // 4
        cabin_end = width - width // 4
        for x in range(cabin_start, cabin_end):
            for y in range(depth):
                bricks.append(_brick(bid, x, y, 1, "cabin", "car cabin"))
                bid += 1
    return _finish(bricks, width, depth, height, "car", f"car {width}x{depth} with cabin")


def generate_bookshelf(width: int = 4, depth: int = 2, height: int = 6, seed: int = 42) -> Model:
    """A bookshelf: two side walls with evenly spaced horizontal shelves."""
    width = max(3, width)
    depth = max(1, depth)
    height = max(3, height)
    shelf_spacing = max(2, height // 3)
    bricks: list[Brick] = []
    bid = 0
    for h in range(height):
        for y in range(depth):
            bricks.append(_brick(bid, 0, y, h, "wall", "left wall"))
            bid += 1
            bricks.append(_brick(bid, width - 1, y, h, "wall", "right wall"))
            bid += 1
    for level in range(0, height, shelf_spacing):
        for x in range(width):
            for y in range(depth):
                bricks.append(_brick(bid, x, y, level, "shelf", "shelf surface"))
                bid += 1
    return _finish(bricks, width, depth, height, "bookshelf",
                   f"bookshelf {width}x{depth}x{height} with {height // shelf_spacing} shelves")


def generate_basket(width: int = 4, depth: int = 4, height: int = 3, seed: int = 42) -> Model:
    """A basket: full base with hollow perimeter walls above (open top)."""
    width = _clamp(width, 3, 8)
    depth = _clamp(depth, 3, 8)
    height = _clamp(height, 2, 6)
    bricks: list[Brick] = []
    bid = 0
    for x in range(width):
        for y in range(depth):
            bricks.append(_brick(bid, x, y, 0, "base", "basket base"))
            bid += 1
    for h in range(1, height):
        for x in range(width):
            for y in range(depth):
                if x == 0 or x == width - 1 or y == 0 or y == depth - 1:
                    bricks.append(_brick(bid, x, y, h, "wall", "basket wall"))
                    bid += 1
    return _finish(bricks, width, depth, height, "basket",
                   f"woven basket {width}x{depth}x{height} with hollow center")


def generate_bottle(width: int = 2, depth: int = 2, height: int = 6, seed: int = 42) -> Model:
    """A bottle: a wider base with a narrower centred neck above."""
    width = _clamp(width, 2, 3)
    depth = _clamp(depth, 2, 3)
    height = _clamp(height, 4, 12)
    base_height = height // 3
    bricks: list[Brick] = []
    bid = 0
    for h in range(base_height):
        for x in range(width):
            for y in range(depth):
                bricks.append(_brick(bid, x, y, h, "base", "bottle base"))
                bid += 1
    neck_w = max(1, width - 1)
    neck_d = max(1, depth - 1)
    ox = (width - neck_w) // 2
    oy = (depth - neck_d) // 2
    for h in range(base_height, height):
        for x in range(ox, ox + neck_w):
            for y in range(oy, oy + neck_d):
                bricks.append(_brick(bid, x, y, h, "wall", "bottle neck"))
                bid += 1
    return _finish(bricks, width, depth, height, "bottle",
                   f"bottle {width}x{depth}x{height} with narrow neck")


def generate_bus(width: int = 8, depth: int = 3, height: int = 3, seed: int = 42) -> Model:
    """A bus: a full rectangular body with a roof top layer."""
    width = _clamp(width, 6, 12)
    depth = _clamp(depth, 3, 4)
    height = _clamp(height, 2, 4)
    bricks: list[Brick] = []
    bid = 0
    for h in range(height):
        for x in range(width):
            for y in range(depth):
                part = "body" if h < height - 1 else "roof"
                bricks.append(_brick(bid, x, y, h, part, f"bus {part}"))
                bid += 1
    return _finish(bricks, width, depth, height, "bus",
                   f"bus {width}x{depth}x{height} - public transport vehicle")


# Category -> generator, covering the 21 BrickGPT categories by adapting the
# eight base generators (bed/bench -> table, mug/bowl -> basket, ...).
CATEGORY_TEMPLATES: dict[str, Callable[[int, int, int, int], Model]] = {
    "basket": generate_basket,
    "bed": lambda w, d, h, s: generate_table(max(6, w), max(3, d), max(1, h // 3) + 1, s),
    "bench": lambda w, d, h, s: generate_table(max(4, w), d, max(2, h // 2), s),
    "birdhouse": lambda w, d, h, s: generate_tower(_clamp(w, 3, 4), _clamp(d, 3, 4), max(4, h), s),
    "bookshelf": generate_bookshelf,
    "bottle": generate_bottle,
    "bowl": lambda w, d, h, s: generate_basket(w, d, _clamp(h, 2, 3), s),
    "bus": generate_bus,
    "camera": lambda w, d, h, s: generate_car(_clamp(w, 3, 5), _clamp(d, 2, 3), _clamp(h, 2, 3), s),
    "car": generate_car,
    "chair": generate_chair,
    "guitar": lambda w, d, h, s: generate_car(max(6, w), _clamp(d, 2, 3), max(2, h // 2), s),
    "jar": generate_bottle,
    "mug": lambda w, d, h, s: generate_basket(_clamp(w, 3, 4), _clamp(d, 3, 4), _clamp(h, 3, 4), s),
    "piano": lambda w, d, h, s: generate_table(max(6, w), max(4, d), max(3, h), s),
    "pot": lambda w, d, h, s: generate_basket(w, d, h, s),
    "sofa": lambda w, d, h, s: generate_chair(max(4, w), d, h, s),
    "table": generate_table,
    "tower": generate_tower,
    "train": lambda w, d, h, s: generate_bus(max(8, w), max(3, d), h, s),
    "vessel": lambda w, d, h, s: generate_bottle(w, d, h, s),
}


def get_template(category: str, width: int, depth: int, height: int, seed: int = 42) -> Model:
    """Return a voxel model for ``category`` (falls back to a tower)."""
    func = CATEGORY_TEMPLATES.get(category.lower())
    if func is None:
        return generate_tower(width, depth, height, seed)
    return func(width, depth, height, seed)
