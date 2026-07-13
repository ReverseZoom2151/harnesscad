"""Parametric augmentation (datagen expansion helper).

The generators (datagen/generators.py) draw one part per sample; augmentation
multiplies the yield of each *verified* part by emitting geometric variants that
teach the model invariances — a bracket mirrored across a plane, transposed
90 degrees, or perturbed in its dimensions is still the same design intent, and a
model that has seen all of them learns that the label is invariant to those
transforms. This is the cheap way to broaden trajectory coverage without paying
the generation + solver-in-the-loop cost again.

:func:`augment` takes one verified :class:`~datagen.pipeline.Sample` and a seed
and returns a deterministic list of Sample-like variants. Every variant keeps the
op stream **structurally identical** (same op tags, sketch ids, constraint counts
and ordering) and only transforms numeric coordinates/dimensions, so each variant
is still a valid, buildable op stream on the same backend — mirrors and transposes
never touch a width/radius/distance (which the backend requires > 0), and
perturbations scale by a strictly positive factor.

Determinism: a single ``random.Random(seed)`` drives the perturbation factors;
the transform set and order are fixed. Same (sample, seed) -> byte-identical
variants. Stdlib only, absolute imports, no wall clock.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from harnesscad.data.datagen.pipeline import Sample

# Op-dict fields that are X / Y coordinates (safe to negate or swap).
_COORD_X = ("x", "cx", "x1", "x2")
_COORD_Y = ("y", "cy", "y1", "y2")
# Op-dict fields that are magnitudes (scaled by perturbation; never negated).
_SCALE_FIELDS = (
    "x", "y", "cx", "cy", "x1", "y1", "x2", "y2",
    "w", "h", "r", "distance", "value", "diameter", "depth", "radius", "spacing",
)


def _neg(op: Dict[str, Any], fields) -> Dict[str, Any]:
    out = dict(op)
    for f in fields:
        if isinstance(out.get(f), (int, float)) and not isinstance(out.get(f), bool):
            out[f] = -out[f]
    return out


def _mirror_x(op: Dict[str, Any]) -> Dict[str, Any]:
    return _neg(op, _COORD_X)


def _mirror_y(op: Dict[str, Any]) -> Dict[str, Any]:
    return _neg(op, _COORD_Y)


def _rotate_90(op: Dict[str, Any]) -> Dict[str, Any]:
    """Transpose the X/Y axes: swap coordinate pairs and w<->h. Keeps all
    magnitudes positive, so the op stream stays buildable."""
    out = dict(op)
    for xf, yf in (("x", "y"), ("cx", "cy"), ("x1", "y1"), ("x2", "y2")):
        if xf in out or yf in out:
            xv, yv = out.get(xf), out.get(yf)
            if xf in out:
                out[xf] = yv if yv is not None else out[xf]
            if yf in out:
                out[yf] = xv if xv is not None else out[yf]
    if "w" in out and "h" in out:
        out["w"], out["h"] = out["h"], out["w"]
    return out


def _perturb(factor: float):
    def _apply(op: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(op)
        for f in _SCALE_FIELDS:
            v = out.get(f)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[f] = round(v * factor, 4)
        return out
    return _apply


def _map_ops(ops: List[Dict[str, Any]], fn) -> List[Dict[str, Any]]:
    return [fn(op) for op in ops]


def _transform_params(params: Dict[str, Any], kind: str, factor: float) -> Dict[str, Any]:
    """Keep the recorded params consistent with the transformed ops (so the
    verifiers-as-labor decomposition still reads correctly)."""
    p = dict(params)
    holes = p.get("holes")
    if kind == "rotate_90":
        if "w" in p and "h" in p:
            p["w"], p["h"] = p["h"], p["w"]
        if isinstance(holes, list):
            p["holes"] = [{**h, "cx": h.get("cy"), "cy": h.get("cx")} for h in holes]
    elif kind == "mirror_x":
        if isinstance(holes, list):
            p["holes"] = [{**h, "cx": -h["cx"]} if "cx" in h else dict(h) for h in holes]
    elif kind == "mirror_y":
        if isinstance(holes, list):
            p["holes"] = [{**h, "cy": -h["cy"]} if "cy" in h else dict(h) for h in holes]
    elif kind == "perturb":
        for f in ("w", "h", "thickness", "hole_r"):
            if isinstance(p.get(f), (int, float)) and not isinstance(p.get(f), bool):
                p[f] = round(p[f] * factor, 4)
        if isinstance(holes, list):
            p["holes"] = [
                {k: (round(v * factor, 4) if isinstance(v, (int, float))
                     and not isinstance(v, bool) else v)
                 for k, v in h.items()}
                for h in holes
            ]
    p["augmentation"] = kind
    return p


def augment(sample: Sample, seed: int) -> List[Sample]:
    """Return deterministic buildable variants of ``sample``.

    Emits, in order: ``mirror_x``, ``mirror_y``, ``rotate_90``, and two
    dimension ``perturb`` variants (factors drawn from a seeded RNG). Each variant
    is a :class:`~datagen.pipeline.Sample` whose ``ops`` are the transformed op
    stream; its ``digest`` is left empty (the variant is unverified until re-run
    through a session) and its ``summary`` records the augmentation lineage.
    """
    rng = random.Random(seed)
    f1 = round(rng.uniform(0.8, 1.25), 4)
    f2 = round(rng.uniform(0.8, 1.25), 4)

    plan = [
        ("mirror_x", _mirror_x, 1.0),
        ("mirror_y", _mirror_y, 1.0),
        ("rotate_90", _rotate_90, 1.0),
        ("perturb", _perturb(f1), f1),
        ("perturb", _perturb(f2), f2),
    ]

    variants: List[Sample] = []
    for i, (kind, fn, factor) in enumerate(plan):
        new_ops = _map_ops(sample.ops, fn)
        new_params = _transform_params(sample.params, kind, factor)
        tag = f"{kind}:{factor}" if kind == "perturb" else kind
        variants.append(Sample(
            brief=f"[aug:{tag}] {sample.brief}",
            generator=sample.generator,
            params=new_params,
            ops=new_ops,
            digest="",  # unverified variant; re-run through a session to verify
            summary={
                "augmentation": tag,
                "augmentation_index": i,
                "source_digest": sample.digest,
                "source_generator": sample.generator,
            },
        ))
    return variants
