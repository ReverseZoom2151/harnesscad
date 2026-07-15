"""The DRAWING-UNDERSTANDING surface -- a drawing's pixels/vectors in, structure out.

``domain/drawings`` carries two kinds of module. One kind is the CAD-drawing
*production* path (dimensions, linetypes, projections, viewports, entity render),
which the geometry services and ``io.drawing`` already reach. The other kind is a
cluster of deterministic front-ends for the *learned* drawing-understanding
papers -- the classical, stdlib-only halves of models whose trained weights live
outside this repo. That second cluster was reachable from nothing. This module is
its dispatcher: one entry surface that imports and routes every member.

    lines(...) / circles(...)   -> Hough primitive detection from edge points
    point_cloud(...)            -> SymPoint's primitive-as-point representation
    decode_instances(...)       -> SymPoint winner-takes-all query decoding
    semantic_map(...)           -> SymPoint mask-weighted per-point class scores
    instance_queue(...) / cutmix(...) -> SymPoint point-level CutMix augmentation
    primitive_graph(...)        -> CADTransformer endpoint-KNN primitive graph
    text_graph(...)             -> text-enhanced type-aware primitive graph
    encode_raster(...) / tokens(...) -> RECAD raster-sketch codec
    render_eval(...)            -> PICASSO image-based metrics (ImgMSE, Chamfer)
    self_supervision_dataset(...) / render_loss(...) -> PICASSO render self-supervision
    hand_drawn_render(...)      -> Vitruvion hand-drawn stroke noise

Adapters only: the drawing modules are never modified. Deterministic, stdlib-only,
no network, no trained weights (those stop at the boundary rather than pretending
to cross it).
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "DrawingsError",
    "lines",
    "circles",
    "point_cloud",
    "decode_instances",
    "semantic_map",
    "instance_queue",
    "harvest_instances",
    "cutmix",
    "primitive_graph",
    "text_graph",
    "encode_raster",
    "decode_raster",
    "tokens",
    "detokenize",
    "render_eval",
    "self_supervision_dataset",
    "render_loss",
    "hand_drawn_render",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

_DRAW = "harnesscad.domain.drawings."


class DrawingsError(ValueError):
    """Base class for every drawing-understanding surface failure."""


# --------------------------------------------------------------------------- #
# Hough primitive detection (classical, deterministic)
# --------------------------------------------------------------------------- #
def lines(points: Sequence[Sequence[float]], *, theta_steps: int = 180,
          rho_res: float = 1.0, threshold: int = 2,
          max_lines: int = 0) -> List[dict]:
    """Detect polar lines from 2D edge points by Hough voting."""
    from harnesscad.domain.drawings.hough_primitives import hough_lines

    found = hough_lines(points, theta_steps=theta_steps, rho_res=rho_res,
                        threshold=threshold, max_lines=max_lines)
    return [{"rho": h.rho, "theta_rad": h.theta_rad, "votes": h.votes}
            for h in found]


def circles(points: Sequence[Sequence[float]], radii: Sequence[float], *,
            center_res: float = 1.0, threshold: int = 3, angle_steps: int = 60,
            max_circles: int = 0) -> List[dict]:
    """Detect circles of the supplied radii from edge points by Hough voting."""
    from harnesscad.domain.drawings.hough_primitives import hough_circles

    found = hough_circles(points, radii, center_res=center_res,
                          threshold=threshold, angle_steps=angle_steps,
                          max_circles=max_circles)
    return [{"cx": c.cx, "cy": c.cy, "radius": c.radius, "votes": c.votes}
            for c in found]


# --------------------------------------------------------------------------- #
# SymPoint: primitives as points, query decoding, CutMix
# --------------------------------------------------------------------------- #
def point_cloud(primitives: Sequence[Tuple[str, Sequence[float]]]) -> Dict[str, List[object]]:
    """SymPoint parallel-array point set from ``(kind, geom)`` primitive specs."""
    from harnesscad.domain.drawings.primitive_points import Primitive, to_point_set

    prims = [Primitive(kind, geom) for (kind, geom) in primitives]
    return to_point_set(prims)


def semantic_map(class_logits: Sequence[Sequence[float]],
                 mask_logits: Sequence[Sequence[float]]) -> List[List[float]]:
    """SymPoint per-point class scores: mask-weighted mixture of query classes."""
    from harnesscad.domain.drawings.query_grouping import semantic_inference

    return semantic_inference(class_logits, mask_logits)


def decode_instances(class_logits: Sequence[Sequence[float]],
                     mask_logits: Sequence[Sequence[float]],
                     object_score: float = 0.1,
                     overlap_threshold: float = 0.8,
                     mask_threshold: float = 0.5) -> List[Dict[str, object]]:
    """SymPoint winner-takes-all decoding of queries into disjoint instances."""
    from harnesscad.domain.drawings.query_grouping import instance_inference

    return instance_inference(class_logits, mask_logits,
                              object_score=object_score,
                              overlap_threshold=overlap_threshold,
                              mask_threshold=mask_threshold)


def instance_queue(capacity: int):
    """A bounded FIFO of harvested thing instances (SymPoint ``instance_queues``)."""
    from harnesscad.domain.drawings.instance_cutmix import InstanceQueue

    return InstanceQueue(int(capacity))


def harvest_instances(sample: Mapping[str, object], stuff_start: int = 30) -> List[dict]:
    """Split a point-cloud sample into its harvestable thing instances."""
    from harnesscad.domain.drawings.instance_cutmix import extract_instances

    return extract_instances(dict(sample), stuff_start=stuff_start)


def cutmix(sample: Mapping[str, object], queue, dx: float, dy: float) -> dict:
    """Paste every queued instance into ``sample``, shifted by ``(dx, dy)``."""
    from harnesscad.domain.drawings.instance_cutmix import cutmix as _cutmix

    return _cutmix(dict(sample), queue, float(dx), float(dy))


# --------------------------------------------------------------------------- #
# Primitive graphs (CADTransformer + text-enhanced)
# --------------------------------------------------------------------------- #
def primitive_graph(primitives: Sequence[Tuple[str, Sequence[float], int, int]],
                    minx: float, miny: float, width: float, height: float,
                    max_degree: int = 4) -> Dict[str, object]:
    """CADTransformer ``svg2graph`` graph from ``(kind, geom, sem, inst)`` specs."""
    from harnesscad.domain.drawings.primitive_graph import (
        Primitive, build_primitive_graph,
    )

    prims = [Primitive(kind, geom, semantic_id=sem, instance_id=inst)
             for (kind, geom, sem, inst) in primitives]
    return build_primitive_graph(prims, minx, miny, width, height,
                                 max_degree=max_degree)


def text_graph(geom_centers: Sequence[Sequence[float]],
               text_items: Sequence[Tuple[float, float, str]],
               min_count: int = 1, k: int = 16, diag: float = 1.0) -> List[list]:
    """Text-enhanced type-aware primitive graph: nodes + per-node edge features."""
    from harnesscad.domain.drawings.text_symbol_graph import (
        build_nodes, type_aware_edge_features,
    )

    nodes = build_nodes(geom_centers, text_items, min_count=min_count)
    return type_aware_edge_features(nodes, k=k, diag=diag)


# --------------------------------------------------------------------------- #
# RECAD raster-sketch codec
# --------------------------------------------------------------------------- #
def encode_raster(grid: Sequence[Sequence[int]], factor: int = 8,
                  levels: int = 5) -> List[List[int]]:
    """Block-quantise a binary sketch canvas into a coarse occupancy grid."""
    from harnesscad.domain.drawings.raster_codec import encode_blocks

    return encode_blocks([list(r) for r in grid], factor=factor, levels=levels)


def decode_raster(coarse: Sequence[Sequence[int]], factor: int = 8,
                  out_height: Optional[int] = None, out_width: Optional[int] = None,
                  levels: int = 5, threshold: float = 0.5) -> List[List[int]]:
    """Expand a coarse occupancy grid back to a binary sketch canvas."""
    from harnesscad.domain.drawings.raster_codec import decode_blocks

    return decode_blocks([list(r) for r in coarse], factor=factor,
                         out_height=out_height, out_width=out_width,
                         levels=levels, threshold=threshold)


def tokens(grid: Sequence[Sequence[int]]) -> dict:
    """Losslessly run-length encode a binary sketch canvas."""
    from harnesscad.domain.drawings.raster_codec import encode_tokens

    stream = encode_tokens([list(r) for r in grid])
    return {"height": stream.height, "width": stream.width,
            "runs": list(stream.runs)}


def detokenize(stream: Mapping[str, object]) -> List[List[int]]:
    """Decode a run-length token stream back to its exact binary canvas."""
    from harnesscad.domain.drawings.raster_codec import TokenStream, decode_tokens

    ts = TokenStream(height=int(stream["height"]), width=int(stream["width"]),
                     runs=[int(v) for v in stream["runs"]])  # type: ignore[union-attr]
    return decode_tokens(ts)


# --------------------------------------------------------------------------- #
# PICASSO: image metrics + rendering self-supervision
# --------------------------------------------------------------------------- #
def render_eval(pred: Sequence[Sequence[float]], target: Sequence[Sequence[float]],
                threshold: float = 0.5) -> Dict[str, float]:
    """PICASSO image-based metrics: ImgMSE, Chamfer, pixel accuracy, foreground IoU."""
    from harnesscad.domain.drawings.raster_metrics import render_eval as _render_eval

    return _render_eval([list(r) for r in pred], [list(r) for r in target],
                        threshold)


def self_supervision_dataset(seed: int = 0, count: int = 4, width: int = 64,
                             height: int = 64, n_primitives: int = 4,
                             stroke_width: float = 1.5, levels: int = 5) -> list:
    """A deterministic dataset of PICASSO label-free render-compare training pairs."""
    from harnesscad.domain.drawings.render_self_supervision import (
        make_self_supervision_dataset,
    )

    return make_self_supervision_dataset(random.Random(seed), int(count),
                                         width=width, height=height,
                                         n_primitives=n_primitives,
                                         stroke_width=stroke_width, levels=levels)


def render_loss(pair, candidate) -> float:
    """The label-free rendering-consistency loss of a candidate against a pair."""
    return pair.loss(candidate)


# --------------------------------------------------------------------------- #
# Vitruvion hand-drawn stroke noise
# --------------------------------------------------------------------------- #
def hand_drawn_render(entities: Sequence[object], seed: int = 0,
                      **kwargs) -> List[List[Tuple[float, float]]]:
    """One wobbly hand-drawn polyline per sketch entity (Vitruvion RenderNoise)."""
    from harnesscad.domain.drawings.hand_drawn_noise import HandDrawnNoise

    return HandDrawnNoise(entities, seed=seed, **kwargs).render()


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index() -> Dict[str, Any]:
    return {e.dotted: e
            for e in capability_registry.find(package="drawings")}


def _available(dotted: str) -> bool:
    return dotted in _index()


_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("primitive", "lines", _DRAW + "hough_primitives",
     "Hough line/circle detection from 2D edge points"),
    ("sympoint", "point_cloud", _DRAW + "primitive_points",
     "primitive-as-point representation for symbol spotting"),
    ("sympoint", "point_cloud", _DRAW + "point_features",
     "the per-point feature vector built on the point set"),
    ("sympoint", "decode_instances", _DRAW + "query_grouping",
     "winner-takes-all query decoding into disjoint instances"),
    ("sympoint", "cutmix", _DRAW + "instance_cutmix",
     "point-level CutMix and point-set augmentation"),
    ("graph", "primitive_graph", _DRAW + "primitive_graph",
     "CADTransformer endpoint-KNN primitive graph"),
    ("graph", "text_graph", _DRAW + "text_symbol_graph",
     "text-enhanced primitive graph with type-aware edge features"),
    ("raster", "encode_raster", _DRAW + "raster_codec",
     "RECAD raster-sketch block quantisation + run-length token codec"),
    ("raster", "render_eval", _DRAW + "raster_metrics",
     "PICASSO image-based metrics (ImgMSE, Chamfer, IoU)"),
    ("render", "self_supervision_dataset", _DRAW + "render_self_supervision",
     "PICASSO label-free render-compare training scheme"),
    ("render", "self_supervision_dataset", _DRAW + "rasterizer",
     "the explicit sketch rasteriser the render loss compares against"),
    ("render", "render_loss", _DRAW + "render_loss",
     "the multiscale image loss driving render self-supervision"),
    ("noise", "hand_drawn_render", _DRAW + "hand_drawn_noise",
     "Vitruvion Matern-GP hand-drawn stroke noise"),
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
                        help="list every drawing-understanding route")
    parser.add_argument("--unadapted", action="store_true",
                        help="list drawing modules with no route")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "unadapted", False):
        for dotted, reason in unadapted():
            print("%s\n    %s" % (dotted, reason))
        return 0

    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["route"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-10s %-*s  %s" % (mark, r["group"], width, r["route"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad drawings",
        description="drawing-understanding surface: Hough detection, SymPoint, "
                    "primitive graphs, raster codec, render self-supervision, "
                    "hand-drawn noise")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
