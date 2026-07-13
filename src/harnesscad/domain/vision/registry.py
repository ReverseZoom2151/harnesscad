"""The VISION surface -- pixels in, CISP out (and back again).

``domain/vision`` carried a sketch rasteriser, a Canny edge extractor, a raster
-> vector primitive fitter, a patch tokeniser, ViT-MAE patch masking, a pixel ->
millimetre calibrator, a camera-pose grid, a multi-view grid composer, a
mask-point sampler, an embedding cache, a digest-bound geometry-prompt bundle and
a residual guard. Every one of them was deterministic, tested, and reachable
from nothing.

They are not twelve unrelated utilities -- they are one pipeline, in both
directions:

    entities -> rasterize -> RasterImage            (CAD -> pixels)
    grid -> edges -> vectorize -> primitives -> ops (pixels -> CAD)

:func:`trace` runs the second one end to end, and the ops it returns apply
cleanly to a :class:`~harnesscad.core.loop.HarnessSession`. With a
:func:`calibrate` from a reference object of known width, the ops come out in
MILLIMETRES rather than pixels -- which is the difference between a picture and
a part.

WHAT IS NOT HERE. There is no learned model anywhere in this module and there
never will be: :func:`tokens`, :func:`mask` and :func:`cache` are the *inputs*
a vision backbone would consume, and :func:`prompt` is the bundle you would hand
one. The harness does not ship a backbone, so those routes stop at the boundary
rather than pretending to cross it.

Adapters only: the vision modules are never modified. Deterministic, stdlib-only,
no network.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry
from harnesscad.core.cisp.ops import AddCircle, AddLine, NewSketch, Op

__all__ = [
    "VisionError",
    "rasterize",
    "edges",
    "vectorize",
    "primitives_to_ops",
    "trace",
    "calibrate",
    "measure",
    "tokens",
    "detokenize",
    "mask",
    "sample_points",
    "poses",
    "view_grid",
    "prompt",
    "cache",
    "guard",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

_VIS = "harnesscad.domain.vision."


class VisionError(ValueError):
    """Base class for every vision-surface failure."""


# --------------------------------------------------------------------------- #
# CAD -> pixels
# --------------------------------------------------------------------------- #
def rasterize(entities: Sequence[Mapping[str, Any]], resolution: int = 64,
              coord_range: Tuple[float, float] = (1.0, 64.0)):
    """Sketch entities (line / arc / circle dicts) -> a :class:`RasterImage`."""
    from harnesscad.domain.vision.sketch_raster import rasterize_sketch

    return rasterize_sketch([dict(e) for e in entities], resolution=int(resolution),
                            coord_range=tuple(coord_range))


# --------------------------------------------------------------------------- #
# pixels -> CAD
# --------------------------------------------------------------------------- #
def edges(image: Sequence[Sequence[float]], sigma: float = 1.0,
          low: float = 0.1, high: float = 0.2) -> Tuple[Tuple[int, ...], ...]:
    """Canny edge map of a greyscale image: blur -> Sobel -> NMS -> hysteresis."""
    from harnesscad.domain.vision.edge_extract import extract_edges

    return extract_edges([list(row) for row in image], sigma=float(sigma),
                         low=float(low), high=float(high))


def vectorize(grid: Sequence[Sequence[int]], connectivity: int = 8,
              min_size: int = 2) -> List[Any]:
    """A binary raster -> fitted LINE / CIRCLE / ARC primitives (with residuals)."""
    from harnesscad.domain.vision.vectorize import vectorize as _vectorize

    return list(_vectorize(tuple(tuple(int(v) for v in row) for row in grid),
                           connectivity=int(connectivity),
                           min_size=int(min_size)))


def primitives_to_ops(primitives: Sequence[Any], calibration: Optional[Any] = None,
                      plane: str = "XY", start_sketch: int = 1) -> List[Op]:
    """Fitted primitives -> CISP ops on ONE sketch.

    With a ``calibration`` (from :func:`calibrate`) the coordinates come out in
    the calibrated unit; without one they stay in PIXELS, and the caller is
    responsible for knowing that. An :class:`ArcFit` becomes a chord
    :class:`AddLine`: CISP has no arc op, and inventing one would be a lie about
    what the harness can represent.
    """
    from harnesscad.domain.vision.pixel_calibration import pixels_to_metric

    def conv(v: float) -> float:
        if calibration is None:
            return float(v)
        return float(pixels_to_metric(calibration, float(v)))

    ops: List[Op] = [NewSketch(plane=plane)]
    sid = "sk%d" % int(start_sketch)
    for p in primitives:
        kind = type(p).__name__
        if kind == "LineFit":
            ops.append(AddLine(sketch=sid,
                               x1=conv(p.start[0]), y1=conv(p.start[1]),
                               x2=conv(p.end[0]), y2=conv(p.end[1])))
        elif kind == "CircleFit":
            ops.append(AddCircle(sketch=sid, cx=conv(p.center[0]),
                                 cy=conv(p.center[1]), r=conv(p.radius)))
        elif kind == "ArcFit":
            # CISP has no arc: the chord is the honest approximation, and it is
            # named as such rather than silently promoted to a spline.
            ops.append(AddLine(sketch=sid,
                               x1=conv(p.start[0]), y1=conv(p.start[1]),
                               x2=conv(p.end[0]), y2=conv(p.end[1])))
        else:
            raise VisionError("unknown primitive kind %r" % kind)
    return ops


def trace(image: Sequence[Sequence[float]], calibration: Optional[Any] = None,
          sigma: float = 1.0, low: float = 0.1, high: float = 0.2,
          connectivity: int = 8, min_size: int = 2,
          start_sketch: int = 1) -> Dict[str, Any]:
    """The whole inverse leg: a greyscale image -> edges -> primitives -> CISP ops."""
    edge_map = edges(image, sigma=sigma, low=low, high=high)
    prims = vectorize(edge_map, connectivity=connectivity, min_size=min_size)
    ops = primitives_to_ops(prims, calibration=calibration,
                            start_sketch=start_sketch)
    return {
        "edge_pixels": sum(sum(1 for v in row if v) for row in edge_map),
        "primitives": [{"kind": type(p).__name__, "residual": p.residual,
                        "size": p.size} for p in prims],
        "ops": ops,
    }


# --------------------------------------------------------------------------- #
# Calibration: pixels are not millimetres
# --------------------------------------------------------------------------- #
def calibrate(reference_pixel_width: float, known_width: float, unit: str = "mm"):
    """A reference object of KNOWN width -> the px -> metric calibration."""
    from harnesscad.domain.vision.pixel_calibration import calibrate_from_reference

    return calibrate_from_reference(float(reference_pixel_width),
                                    float(known_width), unit=unit)


def measure(calibration: Any, object_points: Sequence[Tuple[float, float]]):
    """The metric size of an object, from its pixel contour and a calibration."""
    from harnesscad.domain.vision.pixel_calibration import measure_object_size

    return measure_object_size(calibration,
                               [tuple(float(v) for v in p) for p in object_points])


# --------------------------------------------------------------------------- #
# The model boundary: what a vision backbone would consume. No backbone here.
# --------------------------------------------------------------------------- #
def tokens(grid: Sequence[Sequence[int]], patch_size: int = 8) -> dict:
    """Patchify a raster into the token grid a ViT would read."""
    from harnesscad.domain.vision.patch_tokenizer import (
        patch_grid, patch_occupancy, tokenize,
    )

    g = tuple(tuple(int(v) for v in row) for row in grid)
    pg = patch_grid(len(g), int(patch_size))
    return {
        "per_side": pg.per_side,
        "num_patches": pg.num_patches,
        "token_dim": pg.token_dim,
        "tokens": tokenize(g, int(patch_size)),
        "occupancy": patch_occupancy(g, int(patch_size)),
    }


def detokenize(token_seq: Sequence[Any], patch_size: int, per_side: int):
    """Tokens -> the raster they came from (the round trip must be exact)."""
    from harnesscad.domain.vision.patch_tokenizer import detokenize as _detok

    return _detok(tuple(token_seq), int(patch_size), int(per_side))


def mask(grid: Sequence[Sequence[int]], patch: int = 8, ratio: float = 0.75,
         seed: int = 0, fill: int = 0):
    """ViT-MAE-style patch masking. Deterministic in ``seed`` -- no global RNG."""
    from harnesscad.domain.vision.patch_mask import apply_mask

    return apply_mask(tuple(tuple(int(v) for v in row) for row in grid),
                      int(patch), float(ratio), seed=int(seed), fill=int(fill))


def sample_points(binary_mask: Sequence[Sequence[int]], count: int = 8,
                  seed: int = 0):
    """Seeded point prompts sampled from a binary foreground mask."""
    from harnesscad.domain.vision.mask_sampling import sample_mask

    return sample_mask(tuple(tuple(int(v) for v in row) for row in binary_mask),
                       int(count), seed=int(seed))


def poses() -> List[dict]:
    """The camera-pose ID grid: a stable integer id per (azimuth, elevation)."""
    from harnesscad.domain.vision.pose_grid import all_poses, view_direction

    return [{"id": pid, "azimuth": pose.azimuth, "elevation": pose.elevation,
             "direction": view_direction(pid)}
            for pid, pose in all_poses()]


def view_grid(views: Sequence[Any]):
    """Compose orthogonal views into the 2x2 grid a multi-view adapter expects."""
    from harnesscad.domain.vision.multiview_grid import compose_grid

    return compose_grid(list(views))


def prompt(mesh_digest: str, views: Sequence[Mapping[str, Any]]):
    """A DIGEST-BOUND multi-view geometry prompt bundle.

    The digest binds the prompt to the mesh it was rendered from: a bundle whose
    views drifted from the model is caught, not silently believed.
    """
    from harnesscad.domain.vision.geometry_prompt import GeometryPrompt, PromptView

    bundle = GeometryPrompt(mesh_digest=str(mesh_digest),
                            views=tuple(PromptView(**dict(v)) for v in views))
    bundle.validate()
    return bundle


def cache():
    """A content-addressed cache for frozen-backbone embeddings.

    The encoder is INJECTED (``cache().get_or_compute(data, encoder, ...)``): the
    cache key covers the checkpoint and the preprocessing, so two different
    backbones never collide on one key.
    """
    from harnesscad.domain.vision.embedding_cache import EmbeddingCache

    return EmbeddingCache()


def guard(base: Sequence[float], residual: Sequence[float]):
    """Reject a residual correction that is not finite, shaped or bounded.

    A model's residual output is untrusted input like any other.
    """
    from harnesscad.domain.vision.residual_guard import guard_residual

    return guard_residual(list(base), list(residual))


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index() -> Dict[str, Any]:
    return {e.dotted: e for e in capability_registry.find(package="vision")}


def _available(dotted: str) -> bool:
    return dotted in _index()


_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("forward", "rasterize", _VIS + "sketch_raster",
     "sketch entities -> a rasterised image"),
    ("inverse", "edges", _VIS + "edge_extract",
     "greyscale image -> a Canny edge map"),
    ("inverse", "vectorize", _VIS + "vectorize",
     "binary raster -> fitted line / circle / arc primitives"),
    ("inverse", "trace", _VIS + "vectorize",
     "the whole inverse leg: image -> edges -> primitives -> CISP ops"),
    ("metric", "calibrate", _VIS + "pixel_calibration",
     "a reference of known width -> px->mm calibration; metric measurement"),
    ("model", "tokens", _VIS + "patch_tokenizer",
     "patchify a raster into the token grid a ViT reads (no ViT here)"),
    ("model", "mask", _VIS + "patch_mask",
     "ViT-MAE patch masking, deterministic in the seed"),
    ("model", "sample_points", _VIS + "mask_sampling",
     "seeded point prompts from a binary foreground mask"),
    ("model", "cache", _VIS + "embedding_cache",
     "content-addressed cache for an INJECTED frozen backbone"),
    ("model", "prompt", _VIS + "geometry_prompt",
     "digest-bound multi-view geometry prompt bundle"),
    ("model", "guard", _VIS + "residual_guard",
     "reject a model residual that is not finite / shaped / bounded"),
    ("view", "poses", _VIS + "pose_grid",
     "the camera-pose ID grid"),
    ("view", "view_grid", _VIS + "multiview_grid",
     "compose orthogonal views into a 2x2 multi-view grid"),
    ("eval", "instance_matching", _VIS + "instance_matching",
     "mask IoU / one-to-many matching / mask NMS (used by the bench suite)"),
)


def routed_modules() -> Tuple[str, ...]:
    return tuple(sorted({m for _g, _n, m, _d in _ROUTES if _available(m)}))


def discover() -> List[dict]:
    return [{"group": g, "route": n, "module": m, "doc": d,
             "present": _available(m)}
            for (g, n, m, d) in _ROUTES]


def unadapted() -> List[Tuple[str, str]]:
    routed = set(routed_modules())
    return [(d, "no route yet") for d in sorted(_index())
            if d not in routed and not d.endswith(".registry")]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every vision route")
    parser.add_argument("--trace", default=None, metavar="JSON",
                        help="a greyscale image (JSON 2-D array, or @file) to trace to ops")
    parser.add_argument("--calibrate", default=None, metavar="PX,MM",
                        help="reference pixel width and known width, for --trace")
    parser.add_argument("--apply", action="store_true",
                        help="apply the traced ops to a stub-backed HarnessSession")
    parser.add_argument("--poses", action="store_true",
                        help="list the camera-pose ID grid")
    parser.add_argument("--unadapted", action="store_true",
                        help="list vision modules with no route")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def _load(text: str) -> Any:
    if text.startswith("@"):
        with open(text[1:], "r", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(text)


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "unadapted", False):
        for dotted, reason in unadapted():
            print("%s\n    %s" % (dotted, reason))
        return 0

    if getattr(args, "poses", False):
        print(json.dumps(poses(), indent=2, sort_keys=True, default=repr))
        return 0

    if getattr(args, "trace", None):
        image = _load(args.trace)
        cal = None
        if getattr(args, "calibrate", None):
            px, mm = args.calibrate.split(",")
            cal = calibrate(float(px), float(mm))
        result = trace(image, calibration=cal)
        if getattr(args, "apply", False):
            from harnesscad.core.loop import HarnessSession
            from harnesscad.io.backends.stub import StubBackend

            session = HarnessSession(StubBackend())
            applied = session.apply_ops(result["ops"])
            print("primitives: %d" % len(result["primitives"]))
            print("ok:         %s" % applied.ok)
            print("applied:    %d" % applied.applied)
            print("summary:    %s" % json.dumps(session.summary(), sort_keys=True))
            return 0 if applied.ok else 1
        print(json.dumps({
            "edge_pixels": result["edge_pixels"],
            "primitives": result["primitives"],
            "ops": [op.to_dict() for op in result["ops"]],
        }, indent=2, sort_keys=True))
        return 0

    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["route"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-8s %-*s  %s" % (mark, r["group"], width, r["route"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad vision",
        description="vision surface: rasterise, trace an image back to CISP ops, "
                    "calibrate pixels to millimetres")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
