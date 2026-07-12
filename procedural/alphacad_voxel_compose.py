"""Side-by-side voxel-model composition and dimensional mutation (AlphaCAD).

Source: ``AlphaCAD-main`` (``summit-demo/vote_server.py``: ``compose_cads`` and
``apply_mutation``). Two small, deterministic scene/edit utilities used by the
demo:

* ``compose`` lays several voxel objects next to each other along X with a fixed
  gap, re-offsetting brick ids and x-coordinates so the result is a single
  well-formed model with globally unique ids -- a minimal 1-D bin placement for
  building multi-object scenes.
* ``mutate_dims`` applies a named parametric edit ("taller", "wider", ...) that
  scales one dimension by +/-30% and clamps it to the BrickGPT 20x20x20 grid --
  a deterministic parametric design-variation operator.

Both are pure, stdlib only, deterministic. Models are the dicts produced by
``procedural.alphacad_brick_templates``.
"""

from __future__ import annotations

# Grid limits (BrickGPT constraints): height up to 20, footprint up to 12.
_MAX_H = 20
_MAX_WD = 12
_MIN_H = 2
_MIN_WD = 2
_MIN_D = 1


def compose(models: list[dict], spacing: int = 2) -> dict:
    """Place ``models`` side-by-side along +X with ``spacing`` empty columns.

    Brick ids are renumbered contiguously and x-coordinates shifted so nothing
    overlaps. Returns a single model dict spanning the whole arrangement.
    """
    bricks: list[dict] = []
    id_offset = 0
    x_offset = 0
    max_depth = 0
    max_height = 0
    for model in models:
        w = model["width"]
        max_depth = max(max_depth, model["depth"])
        max_height = max(max_height, model["height"])
        for b in model.get("bricks", []):
            bricks.append({
                "id": b["id"] + id_offset,
                "x": b["x"] + x_offset,
                "y": b["y"],
                "z": b["z"],
                "layer": b.get("layer", b["z"] + 1),
                "part_type": b.get("part_type", "default"),
                "description": b.get("description", "brick"),
            })
        id_offset += model["total_bricks"]
        x_offset += w + spacing
    total_width = max(0, x_offset - spacing) if models else 0
    return {
        "width": max(1, total_width),
        "depth": max(1, max_depth),
        "height": max(1, max_height),
        "total_bricks": len(bricks),
        "bricks": bricks,
        "object_type": "+".join(m.get("object_type", "obj") for m in models),
        "description": " + ".join(m.get("description", "object") for m in models),
    }


def _scale_up(value: int, hi: int) -> int:
    return min(hi, value + max(1, int(round(value * 0.3))))


def _scale_down(value: int, lo: int) -> int:
    return max(lo, value - max(1, int(round(value * 0.3))))


def mutate_dims(width: int, depth: int, height: int, mutation: str | None) -> tuple[int, int, int]:
    """Apply a named +/-30% dimensional mutation, clamped to the grid.

    Recognised names: ``taller``/``+height``, ``shorter``/``-height``,
    ``wider``/``+width``, ``narrower``/``-width``, ``deeper``/``+depth``,
    ``shallower``/``-depth``. Anything else (or ``None``) is a no-op.
    """
    if not mutation:
        return width, depth, height
    m = mutation.lower()
    if m in ("taller", "+height"):
        height = _scale_up(height, _MAX_H)
    elif m in ("shorter", "-height"):
        height = _scale_down(height, _MIN_H)
    elif m in ("wider", "+width"):
        width = _scale_up(width, _MAX_WD)
    elif m in ("narrower", "-width"):
        width = _scale_down(width, _MIN_WD)
    elif m in ("deeper", "+depth"):
        depth = _scale_up(depth, _MAX_WD)
    elif m in ("shallower", "-depth"):
        depth = _scale_down(depth, _MIN_D)
    return width, depth, height
