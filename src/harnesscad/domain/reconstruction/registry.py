"""Reconstruction route registry -- the discoverable inverse leg of the harness.

:mod:`harnesscad.domain.reconstruction.ingest_pipeline` wires ONE reconstruction
path: family-tagged CAD tokens -> CISP ops. The reconstruction tree carries a
hundred more modules -- point-cloud fitters, silhouette carvers, B-rep graph
encoders, sketch canonicalisers, SVG/orthographic recovery, scene graphs,
CAD -> program translators, reconstruction metrics -- each correct in isolation
and reachable from nothing.

This module makes them *dispatchable* the way the bench and verifier registries
do for their fleets:

*   **Discovery** goes through :mod:`harnesscad.registry` (the static AST index).
    A route only exists if every module it adapts is actually in the tree; a
    module that leaves the tree takes its route with it.
*   **Routes are keyed by (input kinds -> output kind).** A caller can ask "what
    can turn a point cloud into CAD?" -- :func:`routes_for` answers with real,
    runnable routes rather than a directory listing.
*   **Adapters live here.** The reconstruction modules are never modified. Every
    adapter imports its modules *inside* the call, so an optional-dependency
    failure is a raised error at call time, not an import-time crash -- while the
    static index still sees the import edge.

RIVALS ARE SELECTED BY NAME, NEVER BLENDED
------------------------------------------
Three rivalries run through this tree and each is a *choice*, not an average:

``token_family``
    deepcad / skexgen / hnc / vitruvion quantise with mutually incompatible
    rules (256 round-half-even levels, 64 truncating levels, a floor quantiser
    plus a 25-frame rotation codebook, floor-quantise with bin-centre
    reconstruction). See :mod:`.ingest_pipeline`, which owns and enforces this.

``brep_graph_encoding``
    cadparser (categorical face/edge/coedge one-hots on a topological B-rep),
    uvnet (sampled UV-grid geometry per face, U-grid per edge) and graphbrep
    (a permutation-canonicalised surface-adjacency matrix) are three different
    encodings of "the B-rep as a graph". They take different inputs and produce
    different graphs; there is no merged graph.

``canonical_sketch_ordering``
    gencad / skexgen / deepcad-profile-assembly each define a *different*
    canonical order for the loops and curves of a sketch. Two of them applied to
    the same sketch give different token streams; using one paper's order with
    another paper's decoder is a real bug.

:func:`rivals` lists them; :func:`select` is how you pick one; passing a member
of one family where another is expected raises :class:`RivalMismatch`.

Stdlib-only, absolute imports, deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "KINDS",
    "Route",
    "RouteError",
    "UnknownRoute",
    "RivalMismatch",
    "RIVAL_FAMILIES",
    "routes",
    "route",
    "routes_for",
    "inputs",
    "outputs",
    "run",
    "rivals",
    "select",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

RECONSTRUCTION_PACKAGE = "reconstruction"
#: ``io/ingest`` is the *front* of the same leg -- files, scans and drawings come
#: in through it, so its modules are routed here too (as ``ingest.*``).
INGEST_PACKAGE = "ingest"
_PKG = "harnesscad.domain.reconstruction."
_IO = "harnesscad.io.ingest."

#: The data kinds routes consume and produce. A route names its kinds so a
#: caller can ask "point_cloud -> ?" and get an answer that actually runs.
KINDS: Tuple[str, ...] = (
    # geometry in
    "point_cloud", "point_clouds", "silhouettes", "contour2d", "segments",
    "voxels", "mesh", "heatmaps", "drawing_svg", "file",
    # cad structures
    "tokens", "commands", "cisp_ops", "brep_topology", "brep_parametric",
    "brep_faces", "brep_entities", "brep_primitives", "adjacency_matrix",
    "brep_graph", "complex_text", "probabilistic_complex", "chain_complex",
    "edges3d", "surfaces", "sketch_loops", "sketch_curves", "extrudes",
    "node_tree", "csg_model", "knowledge_graph", "sketch_graph",
    "command_workflow", "scene_primitives", "scene_graph", "program_steps",
    # derived
    "primitives", "solid", "topology", "wireframe", "features", "labels",
    "descriptor", "graph_key", "actions", "diagnostics", "score",
    "program_text", "text", "params", "plan", "condition", "codes",
    "answer", "regions", "episode", "selection",
)


class RouteError(ValueError):
    """Base class for a reconstruction dispatch failure."""


class UnknownRoute(RouteError):
    """A route name that is not in the registry."""


class RivalMismatch(RouteError):
    """A rival member was requested from the wrong family, or does not exist."""


# --------------------------------------------------------------------------- #
# Route type
# --------------------------------------------------------------------------- #
Adapter = Callable[..., Any]


@dataclass(frozen=True)
class Route:
    """One adapted reconstruction capability: ``inputs -> output``."""

    name: str
    inputs: Tuple[str, ...]
    output: str
    dotted: Tuple[str, ...]          # the reconstruction modules this adapts
    adapter: Adapter
    doc: str = ""
    family: str = ""                 # the rival family this route belongs to
    summary: str = ""                # docstring line of the first module (from the index)
    tags: Tuple[str, ...] = ()

    def call(self, **payload: Any) -> Any:
        """Run the underlying modules. Raises on bad input -- it never guesses."""
        return self.adapter(**payload)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            "output": self.output,
            "modules": list(self.dotted),
            "family": self.family,
            "doc": self.doc,
            "summary": self.summary,
        }


# --------------------------------------------------------------------------- #
# Adapters.
#
# Each one imports its modules LOCALLY and normalises their real public API into
# a keyword-argument call. The reconstruction modules are never modified.
# --------------------------------------------------------------------------- #

# --- tokens ----------------------------------------------------------------
def _a_tokens_to_ops(*, tokens, family, backend="stub", arc_policy="chord"):
    """Family-tagged CAD tokens -> an applied, editable CISP op stream."""
    from harnesscad.domain.reconstruction import ingest_pipeline as ip

    seq = tokens if isinstance(tokens, ip.TokenSequence) else ip.TokenSequence.from_dict(tokens)
    return ip.ingest_tokens(seq, family=family, backend=backend, arc_policy=arc_policy)


def _a_tokens_text2cad(*, cad_vec):
    """Text2CAD's exact CAD-sequence vector codec: vectors -> a CAD model dict."""
    from harnesscad.domain.reconstruction.tokens import text2cad_vector_codec as m

    return m.decode_model([tuple(t) for t in cad_vec])


def _a_tokens_text2cad_encode(*, model, padding=False):
    from harnesscad.domain.reconstruction.tokens import text2cad_vector_codec as m

    vectors = m.encode_model(model, padding=padding)
    return {"cad_vec": vectors.cad_vec, "flag_vec": vectors.flag_vec,
            "index_vec": vectors.index_vec}


def _a_tokens_text2cad_serialize(*, faces, extrusion):
    """Text2CAD's flat token stream (Khan et al., Table 3)."""
    from harnesscad.domain.reconstruction.tokens import text2cad_tokens as m

    model = m.CadModel(faces=faces, extrusion=extrusion)
    return {"tokens": m.serialize_model(model), "vocabulary": m.vocabulary_size()}


def _a_tokens_image2cadseq(*, ops, nc=None):
    """Sim-Gallery DSL: gallery ops -> the vectorised feature matrix + its parse."""
    from harnesscad.domain.reconstruction.tokens import image2cadseq as m

    matrix = m.build_feature_matrix(list(ops)) if nc is None \
        else m.build_feature_matrix(list(ops), nc=nc)
    return {"matrix": matrix, "op_types": m.op_type_sequence(matrix),
            "parsed": m.parse_feature_matrix(matrix),
            "dsl": m.program_to_dsl(list(ops))}


def _a_tokens_pht_cad(*, primitives):
    """PHT-CAD Efficient Hybrid Parametrization: primitives -> tokens + efficiency."""
    from harnesscad.domain.reconstruction.tokens import pht_cad as m

    report = m.efficiency(list(primitives))
    return {"tokens": m.to_tokens(list(primitives)),
            "fields": report.ehp_fields,
            "baseline_fields": report.baseline_fields,
            "compression_ratio": report.compression_ratio}


def _a_tokens_pointer_cad(*, values=(), angles=(), q=8):
    """Pointer-CAD vocabulary: values/angles -> token ids and back."""
    from harnesscad.domain.reconstruction.tokens import pointer_cad as m

    v = [m.quantize_nv(float(x), q) for x in values]
    a = [m.quantize_ag(float(x), q) for x in angles]
    report = m.quantization_report(q)
    return {
        "value_tokens": v,
        "angle_tokens": a,
        "values_roundtrip": [m.dequantize_nv(t, q) for t in v],
        "angles_roundtrip": [m.dequantize_ag(t, q) for t in a],
        "vocab_size": m.vocab_size(q),
        "max_abs_error": report.max_abs_error,
    }


def _a_tokens_cad2program(*, program, resolution=None):
    """cad2program fixed-slot command template: a shape program -> commands."""
    from harnesscad.domain.reconstruction.tokens import cad2program as m

    kw = {} if resolution is None else {"resolution": resolution}
    return {"commands": m.encode_program(program, **kw),
            "quantization_error": m.quantization_error(program, **kw)}


def _a_tokens_deepcad_arc(*, start, end, sweep_angle, flag):
    """DeepCAD's arc macro: (end point, sweep, ccw flag) <-> a full arc."""
    from harnesscad.domain.reconstruction.tokens import deepcad_arc_macro as m

    arc = m.arc_from_macro(start, end, sweep_angle, flag)
    return {"arc": arc, "mid": arc.mid_point(), "bbox": m.arc_bbox(arc),
            "macro": m.arc_to_macro(arc.start_point, arc.mid_point(),
                                    arc.end_point, arc.center)}


def _a_tokens_hnc_codebooks(*, vectors, codes):
    """HNC-CAD nearest-codebook assignment (the neural code, not a quantiser)."""
    from harnesscad.domain.reconstruction.tokens import hnc_codebooks as m

    book = m.Codebook(tuple(tuple(c) for c in codes))
    assignments = book.assign_batch([tuple(v) for v in vectors])
    return {"assignments": list(assignments),
            "utilization": m.utilization(list(assignments), book.size()),
            "perplexity": m.codebook_perplexity(list(assignments), book.size())}


def _a_tokens_hnc_spl(*, loops, extrusions):
    """HNC-CAD Solid-Profile-Loop hierarchy: loops -> the S-P-L tree."""
    from harnesscad.domain.reconstruction.tokens import hnc_spl_tree as m

    loop_objs = tuple(m.Loop(tuple(m.Curve(tuple(tuple(p) for p in c)) for c in loop))
                      for loop in loops)
    profile = m.profile_from_loops(loop_objs)
    solid = m.solid_from_profiles((profile,), tuple(tuple(e) for e in extrusions))
    return {"loops": loop_objs, "profile": profile, "solid": solid,
            "tokens": [m.loop_token_indices(loop) for loop in loop_objs]}


def _a_tokens_skexgen_codes(*, codes):
    """SkexGen disentangled codebook layout: topology | geometry | extrude."""
    from harnesscad.domain.reconstruction.tokens import skexgen_code_layout as m

    valid = m.filter_valid(codes)
    return {"valid": list(valid),
            "split": [m.split_code(c) for c in valid],
            "usage": m.codebook_usage(list(valid))}


def _a_tokens_sketch2cad_scene(*, objects, pose_id=0):
    """Sketch2CAD scene-descriptor codec: scene objects <-> a token stream."""
    from harnesscad.domain.reconstruction.tokens import sketch2cad_scene as m

    codec = m.SceneDescriptorCodec()
    encoded = codec.encode_scene(pose_id, list(objects))
    return {"tokens": encoded, "decoded": codec.decode_scene(encoded),
            "vocab_size": codec.config.vocab_size()
            if hasattr(codec, "config") else None}


def _a_tokens_vitruvion_constraints(*, edges, gather_idxs):
    """Vitruvion's constraint hypergraph as a pointer-token stream."""
    from harnesscad.domain.reconstruction.tokens import vitruvion_constraints as m

    val = m.tokenize_constraints(edges, list(gather_idxs))
    return {"tokens": val,
            "constraints": m.constraints_from_tokens(val, list(gather_idxs))}


def _a_tokens_gencad_quantize(*, values, kind="unit", n=None):
    """GenCAD/DeepCAD's exact normalise+quantise pipeline (a RIVAL quantiser)."""
    from harnesscad.domain.reconstruction.tokens import gencad_quantize as m

    fns = {
        "unit": (m.quantize_unit, m.dequantize_unit),
        "angle": (m.quantize_angle, m.dequantize_angle),
        "size": (m.quantize_sketch_size, m.dequantize_sketch_size),
    }
    if kind not in fns:
        raise RouteError("gencad quantiser kind must be one of %s"
                         % ", ".join(sorted(fns)))
    fwd, inv = fns[kind]
    kw = {} if n is None else {"n": n}
    levels = [fwd(float(v), **kw) for v in values]
    return {"levels": levels, "roundtrip": [inv(l, **kw) for l in levels],
            "step": m.quantization_step(**kw),
            "max_error": m.max_quantization_error(**kw)}


# --- fitting: geometry out of clouds, silhouettes, images, drawings --------
def _a_fit_bbox_primitive(*, points, shape="cube", sample=None, seed=0):
    """point cloud -> a fitted architectural primitive (bounding-box fit).

    Normalisation and furthest-point sampling come from cadrille's point adapter;
    the axis-aligned bound comes from the PS-CAD candidate selector; the solid is
    lifted by the Sketch2CAD architectural primitive builder. The fit is an
    honest AABB fit -- it recovers position and size exactly for a box-shaped
    cloud and is an approximation for anything else, which is why the report
    carries ``method`` and ``residual``.
    """
    from harnesscad.domain.reconstruction.fitting import pointcloud_adapter as pa
    from harnesscad.domain.reconstruction.fitting import primitive_shapes as ps
    from harnesscad.domain.reconstruction.sequences import candidate_selection as cs

    pts = [tuple(float(c) for c in p) for p in points]
    if not pts:
        raise RouteError("cannot fit a primitive to an empty point cloud")
    if sample is not None:
        pts, _indices = pa.furthest_point_sampling(pts, k=int(sample), seed=int(seed))
    lo, hi = cs.axis_aligned_bbox(pts)
    size = tuple(hi[i] - lo[i] for i in range(3))
    position = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2])
    mesh = ps.build_shape(shape, position=position, rotation=(0.0, 0.0), size=size)
    fit_lo, fit_hi = mesh.bounding_box()
    residual = max(abs(fit_lo[i] - lo[i]) for i in range(3))
    residual = max(residual, max(abs(fit_hi[i] - hi[i]) for i in range(3)))
    return {
        "primitives": [{"shape": shape, "position": position,
                        "rotation": (0.0, 0.0), "size": size}],
        "mesh": mesh,
        "bbox": (lo, hi),
        "n_points": len(pts),
        "method": "axis-aligned bounding-box fit",
        "residual": residual,
    }


def _a_fit_normalize_cloud(*, points, k=None, seed=0):
    """cadrille's point-cloud input adapter: unit cube + furthest-point sampling."""
    from harnesscad.domain.reconstruction.fitting import pointcloud_adapter as m

    pts = [tuple(float(c) for c in p) for p in points]
    if k is None:
        return {"points": m.normalize_unit_cube(pts)}
    return {"points": m.prepare_point_input(pts, k=int(k), seed=int(seed))}


def _a_fit_visual_hull(*, grid, silhouettes):
    """Silhouette masks -> the carved visual hull (GaussianCAD's initialisation)."""
    from harnesscad.domain.reconstruction.fitting import visual_hull as m

    centers = m.carve_visual_hull(grid, list(silhouettes))
    return {"voxels": centers, "occupancy": m.hull_occupancy(grid, list(silhouettes)),
            "bbox": m.hull_bounding_box(centers) if centers else None}


def _a_fit_extrude_contour(*, contour, depth):
    """A 2D metric contour + depth -> a prism solid (cvcad solid regeneration)."""
    from harnesscad.domain.reconstruction.fitting import solid_regeneration as m

    solid = m.extrude_contour([tuple(float(c) for c in p) for p in contour],
                              float(depth))
    return {"solid": solid, "volume": solid.volume,
            "surface_area": solid.surface_area, "bbox": solid.bounding_box}


def _a_fit_wireframe(*, segments, endpoints=None, epsilon=1e-6):
    """Dense line proposals -> a bound, validated Structured Visual Geometry wireframe."""
    from harnesscad.domain.reconstruction.fitting import wireframe_schema as ws

    raw = [(tuple(a), tuple(b)) for a, b in segments]
    wf = ws.build_wireframe(raw, eps=float(epsilon))
    out = {"wireframe": wf, "validity": ws.validity(wf),
           "bbox": ws.bounding_box(wf), "total_length": ws.total_length(wf)}
    if endpoints is not None:
        from harnesscad.domain.reconstruction.fitting import wireframe_binding as wb

        pts = [tuple(p) for p in endpoints]
        bound = wb.bind_and_select(raw, pts, float(epsilon))
        out["bound"] = bound
    return out


def _a_fit_loi_align(*, proposals, node_endpoints, threshold, scale=1.0):
    """Joint-Decoupled Line-of-Interest alignment: drop the false-positive lines."""
    from harnesscad.domain.reconstruction.fitting import loi_align as m

    kept = m.filter_false_positives([tuple(map(tuple, s)) for s in proposals],
                                    [tuple(map(tuple, e)) for e in node_endpoints],
                                    float(threshold), float(scale))
    return {"segments": kept, "n_dropped": len(proposals) - len(kept)}


def _a_fit_residual_regions(*, full, previous, threshold, radius, side="missing"):
    """PS-CAD residual guidance: where does the current solid differ from the target?"""
    from harnesscad.domain.reconstruction.fitting import residual_regions as m

    guidance = m.compute_pref(full, previous, threshold=float(threshold))
    region = m.highest_residual_region(guidance, radius=float(radius), side=side)
    return {"guidance": guidance, "region": region,
            "missing_ratio": guidance.missing_ratio()}


def _a_fit_param_decode(*, face_heatmap, context_normal, context_depth, height,
                        width, camera, threshold=0.5):
    """Sketch2CAD op maps -> the stitching face (position + normal) of an operation."""
    from harnesscad.domain.reconstruction.fitting import param_decode as m

    face = m.decode_stitching_face(face_heatmap, context_normal, context_depth,
                                   int(height), int(width), camera,
                                   threshold=float(threshold))
    return {"face": face, "point": face.point, "normal": face.normal,
            "area_px": face.area_px()}


def _a_fit_progressive_tuning(*, kind, raw_params):
    """PHT-CAD progressive hierarchical tuning: coarse -> fine parameter refinement."""
    from harnesscad.domain.reconstruction.fitting import progressive_tuning as m

    prediction = m.run_pipeline(kind, [float(v) for v in raw_params])
    return {"prediction": prediction, "level": prediction.level,
            "params": prediction.params, "schedule": m.schedule()}


def _a_fit_condition(*, text=None, points=None, voxel=None):
    """DreamCAD's multi-modal condition schema: text / points / voxels -> one feature."""
    from harnesscad.domain.reconstruction.fitting import condition_schema as m

    schema = m.ConditionSchema()
    out: Dict[str, Any] = {"feature_dim": schema.feature_dim}
    if text is not None:
        out["text"] = schema.encode_text(text)
    if points is not None:
        out["points"] = schema.encode_points([tuple(p) for p in points])
    if voxel is not None:
        out["voxel"] = schema.encode_voxel_feature(voxel)
    if len(out) == 1:
        raise RouteError("condition route needs at least one modality "
                         "(text=, points= or voxel=)")
    return out


def _a_fit_pointcloud_candidates(*, cloud, provider, compiler, sampler,
                                 count=10, sample_count=1024, seed=0):
    """Compile-and-distance selection of generated candidates against a target cloud."""
    from harnesscad.domain.reconstruction.fitting import pointcloud_candidates as m

    return m.select_pointcloud_candidate(
        [tuple(p) for p in cloud], provider, compiler, sampler,
        count=int(count), sample_count=int(sample_count), seed=int(seed))


# --- ortho: the SVG / orthographic pipeline --------------------------------
def _a_ortho_reconstruct(*, svg, scale=1.0, tolerance=0.005, stitcher=None):
    """Orthographic SVG views -> 3D edges, face loops and a manifold gate."""
    from harnesscad.domain.reconstruction.ortho import pipeline as m

    result = m.reconstruct(svg, scale=float(scale), tolerance=float(tolerance),
                           stitcher=stitcher)
    return {
        "edges": result.edges,
        "loops": result.loops,
        "faces": result.faces,
        "manifold": result.manifold,
        "reports": [(r.name, r.output_count) for r in result.reports],
        "diagnostics": result.diagnostics,
        "stitch": result.stitch.status,
    }


def _a_ortho_merge(*, nodes, bbox_tolerance=0.08, shape_tolerance=0.2):
    """Two-signal (bbox + shape) clustering of duplicate recovered geometry."""
    from harnesscad.domain.reconstruction.ortho import brep_merge as m

    return {"clusters": m.cluster(list(nodes), bbox_tolerance=bbox_tolerance,
                                  shape_tolerance=shape_tolerance)}


def _a_ortho_stitch_geometry(*, edge_samples, surface_distance, points=None):
    """Kernel-free geometry reconciliation: vertex averaging + edge consistency."""
    from harnesscad.domain.reconstruction.ortho import geometry_stitch as m

    out = {"consistency": m.consistency(edge_samples, surface_distance)}
    if points is not None:
        out["average"] = m.average_vertices([tuple(p) for p in points])
    return out


def _a_ortho_point_labels(*, points, radius):
    """Deterministic suppression of scan-point B-rep labels (boundary before junction)."""
    from harnesscad.domain.reconstruction.ortho import point_labels as m

    return {"kept": m.suppress(list(points), float(radius))}


def _a_ortho_primitive_relations(*, a, b, tolerance=0.01):
    """Infer the relation (parallel / perpendicular / skew) between two primitives."""
    from harnesscad.domain.reconstruction.ortho import primitive_relations as m

    relation = m.infer(a, b, tolerance=float(tolerance))
    return {"relation": relation, "projected": m.project(a, b, relation)}


def _a_ortho_primitive_stitch(*, value, step, residual, max_iterations=20,
                              tolerance=1e-6):
    """Fixed-point primitive stitching: iterate a step until the residual settles."""
    from harnesscad.domain.reconstruction.ortho import primitive_stitch as m

    return {"value": m.stitch(value, step, residual,
                              max_iterations=int(max_iterations),
                              tolerance=float(tolerance))}


def _a_ortho_primitive_intersections(*, primitives, adjacency, pair_intersection,
                                     triple_intersection):
    """Assemble a topology from pairwise / triple primitive intersections."""
    from harnesscad.domain.reconstruction.ortho import primitive_intersections as m

    return {"assembled": m.assemble(primitives, adjacency, pair_intersection,
                                    triple_intersection)}


# --- brep: the three RIVAL graph encodings + the rest ----------------------
def _a_brep_graph_cadparser(*, faces, edges):
    """RIVAL 1/3 -- CADParser: face/edge/coedge nodes with CATEGORICAL features."""
    from harnesscad.domain.reconstruction.brep import cadparser_graph as m

    brep = m.BRep(
        faces=tuple(f if isinstance(f, m.FaceDef) else m.FaceDef(
            id=str(f["id"]), surface_type=f.get("surface_type", "plane"),
            area=float(f.get("area", 0.0)),
            loops=tuple(tuple((str(e), bool(o)) for e, o in loop)
                        for loop in f.get("loops", ()))) for f in faces),
        edges=tuple(e if isinstance(e, m.EdgeDef) else m.EdgeDef(
            id=str(e["id"]), curve_type=e.get("curve_type", "line"),
            length=float(e.get("length", 0.0))) for e in edges),
    )
    graph = m.build_graph(brep)
    return {"encoding": "cadparser", "graph": graph, "n_nodes": graph.n_nodes,
            "adjacency": m.adjacency_matrix(graph),
            "node_features": m.node_features(graph)}


def _a_brep_graph_uvnet(*, faces, edges, curv_num_u=10, surf_num_u=10, surf_num_v=10):
    """RIVAL 2/3 -- UV-Net: face nodes carrying SAMPLED UV-grids, edge U-grids."""
    from harnesscad.domain.reconstruction.brep import uvnet_face_adjacency as m

    face_entries = tuple(
        f if isinstance(f, m.FaceEntry) else m.FaceEntry(
            surface=f["surface"], trim_loops=f.get("trim_loops"),
            name=str(f.get("name", ""))) for f in faces)
    edge_entries = tuple(
        e if isinstance(e, m.EdgeEntry) else m.EdgeEntry(
            curve=e["curve"], faces=tuple(e.get("faces", (0, 0))),
            name=str(e.get("name", ""))) for e in edges)
    graph = m.build_face_adjacency(face_entries, edge_entries,
                                   curv_num_u=int(curv_num_u),
                                   surf_num_u=int(surf_num_u),
                                   surf_num_v=int(surf_num_v))
    return {"encoding": "uvnet", "graph": graph, "n_nodes": graph.num_nodes,
            "adjacency": m.adjacency_matrix(graph),
            "connected": m.is_connected(graph),
            "summary": m.graph_summary(graph)}


def _a_brep_graph_graphbrep(*, matrix, permutation_guard=None):
    """RIVAL 3/3 -- GraphBrep: a permutation-CANONICALISED surface-adjacency matrix."""
    from harnesscad.domain.reconstruction.brep import graphbrep_canonical as m

    mat = tuple(tuple(int(v) for v in row) for row in matrix)
    kw = {} if permutation_guard is None else {"permutation_guard": int(permutation_guard)}
    return {"encoding": "graphbrep", "matrix": m.canonical_matrix(mat, **kw),
            "key": m.canonical_key(mat, **kw),
            "wl_signature": m.wl_signature(mat),
            "labelling": m.canonical_labelling(mat, **kw)}


def _a_brep_entity_features(*, entities):
    """JoinABLe B-rep entity features: one vector per face/edge, for joint prediction."""
    from harnesscad.domain.reconstruction.brep import entity_features as m

    return {"names": m.entity_feature_names(),
            "features": [m.entity_feature_vector(e) for e in entities]}


def _a_brep_structured(*, root, width=None, seed=0):
    """Fixed-width face-edge-vertex hierarchy encoding (padded, validated)."""
    from harnesscad.domain.reconstruction.brep import structured_brep as m

    m.validate_tree(root)
    out: Dict[str, Any] = {"valid": True, "unique": m.unique_children(root.children)}
    if width is not None:
        out["padded"] = m.pad_children(root.children, int(width), seed=int(seed))
    return out


def _a_brep_topology_predict(*, edges, surfaces, tolerance, tau=0.5):
    """CMT topology predictor: edges + surface boxes -> a face-edge adjacency."""
    from harnesscad.domain.reconstruction.brep import topology_predictor as m

    e = tuple(tuple(tuple(float(c) for c in p) for p in edge) for edge in edges)
    s = tuple(surfaces)
    adjacency = m.predict(e, s, float(tolerance), tau=float(tau))
    return {"adjacency": adjacency,
            "scores": m.topology_scores(e, s, float(tolerance)),
            "surface_edges": m.surface_edges(adjacency)}


def _a_brep_complex_nms(*, complex, valid_th=0.5, dist_th=0.05, incidence_th=0.5):
    """ComplexGen NMS: a PROBABILISTIC B-rep complex -> a definite one."""
    from harnesscad.domain.reconstruction.brep import chain_complex_nms as m

    suppressed = m.nms(complex, valid_th=float(valid_th), dist_th=float(dist_th))
    extraction = m.extract(suppressed, valid_th=float(valid_th),
                           incidence_th=float(incidence_th))
    return {"extraction": extraction, "complex": extraction.complex,
            "repair": m.repair_extraction(extraction.complex)}


def _a_brep_complex_io(*, text=None, complex_file=None, precision=6):
    """ComplexGen ``.complex`` chain-complex file: parse and serialise."""
    from harnesscad.domain.reconstruction.brep import chain_complex_io as m

    if text is not None:
        cf = m.parse_complex(text)
        return {"file": cf, "chain_complex": cf.to_chain_complex()}
    if complex_file is not None:
        return {"text": m.serialize_complex(complex_file, precision=int(precision))}
    raise RouteError("chain-complex IO needs text= (parse) or complex_file= (write)")


# --- sketch: the RIVAL canonical orderings + plane recovery ----------------
def _a_sketch_order_gencad(*, loops):
    """RIVAL 1/3 -- GenCAD/DeepCAD canonical profile order (``cadlib/sketch.py``)."""
    from harnesscad.domain.reconstruction.sketch import gencad_canonical_order as m

    return {"ordering": "gencad",
            "loops": m.canonicalize_profile([list(loop) for loop in loops])}


def _a_sketch_order_skexgen(*, sketch):
    """RIVAL 2/3 -- SkexGen canonical order (``sort_faces``/``sort_loops``/``sort_curves``)."""
    from harnesscad.domain.reconstruction.sketch import skexgen_canonical_order as m

    return {"ordering": "skexgen", "sketch": m.canonicalize_sketch(sketch)}


def _a_sketch_order_deepcad_profile(*, commands):
    """RIVAL 3/3 -- DeepCAD loop/profile assembly straight off the command stream."""
    from harnesscad.domain.reconstruction.sketch import profile_assembly as m

    loops = m.split_loops(list(commands))
    return {"ordering": "deepcad_profile_assembly",
            "loops": loops, "profile": m.canonical_profile(loops),
            "profiles": m.split_profiles(list(commands))}


def _a_sketch_reference_plane(*, target_normal, target_point, extrudes, tolerance=None):
    """OpenECAD reference-plane finding: which existing face hosts this sketch?"""
    from harnesscad.domain.reconstruction.sketch import reference_plane as m

    kw = {} if tolerance is None else {"tol": float(tolerance)}
    result = m.find_reference_plane(tuple(target_normal), tuple(target_point),
                                    list(extrudes), **kw)
    return {"found": result.found, "result": result,
            "call_kwargs": result.as_call_kwargs()}


def _a_sketch_pointer_plane(*, face_normal, direction, rotation_deg=0.0, scale=1.0):
    """Pointer-CAD sketch-plane frame: a face normal + a direction symbol -> a frame."""
    from harnesscad.domain.reconstruction.sketch import pointer_sketch_plane as m

    frame = m.build_frame(tuple(face_normal), direction,
                          rotation_deg=float(rotation_deg), scale=float(scale))
    return {"frame": frame, "orthonormal": m.is_orthonormal(frame)}


# --- recognize -------------------------------------------------------------
def _a_recognize_machining(*, faces):
    """Rule-based machining-feature recognition (holes, pockets, slots, ...)."""
    from harnesscad.domain.reconstruction.recognize import machining_features as m

    return {"features": m.detect_features(list(faces)),
            "counts": m.feature_counts(list(faces))}


def _a_recognize_point_features(*, points, k=8):
    """Per-point geometric features (covariance descriptors) for part segmentation."""
    from harnesscad.domain.reconstruction.recognize import point_features as m

    return {"features": m.point_features([tuple(p) for p in points], k=int(k))}


def _a_recognize_prototypes(*, support_features, support_labels, query_features,
                            n_prototypes=1, seed=0):
    """RIVAL 1/2 -- few-shot part segmentation by NEAREST MULTI-PROTOTYPE."""
    from harnesscad.domain.reconstruction.recognize import prototypes as m

    return {"method": "prototypes",
            "labels": m.segment(support_features, support_labels, query_features,
                                n_prototypes=int(n_prototypes), seed=int(seed))}


def _a_recognize_label_propagation(*, features, labels, n_classes, k=8, sigma=1.0,
                                   alpha=0.99, method="closed_form", epochs=50):
    """RIVAL 2/2 -- few-shot part segmentation by TRANSDUCTIVE LABEL PROPAGATION."""
    from harnesscad.domain.reconstruction.recognize import label_propagation as m

    return {"method": "label_propagation",
            "labels": m.predict(features, labels, int(n_classes), k=int(k),
                                sigma=float(sigma), alpha=float(alpha),
                                method=method, epochs=int(epochs))}


def _a_recognize_label_transfer(*, query_points, source_points, source_labels):
    """Nearest-neighbour label transfer between two clouds (segmentation eval)."""
    from harnesscad.domain.reconstruction.recognize import label_transfer as m

    return {"labels": m.transfer_labels([tuple(p) for p in query_points],
                                        [tuple(p) for p in source_points],
                                        list(source_labels))}


def _a_recognize_partseg_metrics(*, pred, gold, labels=None):
    """Part-segmentation IoU / mIoU / accuracy."""
    from harnesscad.domain.reconstruction.recognize import partseg_metrics as m

    return {"mean_iou": m.mean_iou(list(pred), list(gold), labels),
            "per_class_iou": m.per_class_iou(list(pred), list(gold), labels),
            "accuracy": m.accuracy(list(pred), list(gold))}


def _a_recognize_partseg_episode(*, dataset, ways, shots, queries, seed=0):
    """Deterministic C-way K-shot episode construction for few-shot segmentation."""
    from harnesscad.domain.reconstruction.recognize import partseg_episodes as m

    episode = m.build_episode(dataset, ways, int(shots), int(queries), seed=int(seed))
    return {"episode": episode, "support": m.flatten_support(episode)}


def _a_recognize_instance_offsets(*, centers, instances, background_id=-1):
    """Instance centroid-offset targets + the grouping they induce."""
    from harnesscad.domain.reconstruction.recognize import instance_offsets as m

    pts = [tuple(p) for p in centers]
    inst = list(instances)
    offsets = m.offset_targets(pts, inst, background_id=background_id)
    shifted = m.shift_to_centroid(pts, offsets, inst, background_id=background_id)
    return {"offsets": offsets, "shifted": shifted,
            "groups": m.group_by_shifted_center(shifted, inst,
                                                background_id=background_id)}


def _a_recognize_consistency(*, points, labels, k=10):
    """Segmentation consistency: do nearby points share a part label?"""
    from harnesscad.domain.reconstruction.recognize import segmentation_consistency as m

    return {"consistency": m.segmentation_consistency(
        [tuple(p) for p in points], list(labels), k=int(k))}


def _a_recognize_shape_descriptor(*, points, d2_bins=16, shell_bins=16,
                                  samples=1024, seed=0):
    """Rotation-invariant shape descriptor (D2 + radial shells + PCA) for retrieval."""
    from harnesscad.domain.reconstruction.recognize import shape_descriptors as m

    return {"descriptor": m.descriptor_vector([tuple(p) for p in points],
                                              d2_bins=int(d2_bins),
                                              shell_bins=int(shell_bins),
                                              samples=int(samples), seed=int(seed))}


def _a_recognize_graph_descriptor(*, graph, wl_iterations=3, wl_dims=32):
    """Structural graph descriptor (degree histogram + WL colours) for CAD retrieval."""
    from harnesscad.domain.reconstruction.recognize import graph_descriptors as m

    return {"descriptor": m.descriptor_vector(graph, wl_iterations=int(wl_iterations),
                                              wl_dims=int(wl_dims))}


def _a_recognize_model_distance(*, cloud_a, cloud_b, metric="chamfer", resolution=16):
    """CAD-model distance for non-categorical clustering (chamfer or voxel Jaccard)."""
    from harnesscad.domain.reconstruction.recognize import model_distances as m

    a = [tuple(p) for p in cloud_a]
    b = [tuple(p) for p in cloud_b]
    if metric == "chamfer":
        return {"metric": "chamfer", "distance": m.chamfer_distance(a, b)}
    if metric == "voxel":
        return {"metric": "voxel",
                "distance": m.voxel_jaccard_distance(a, b, resolution=int(resolution))}
    raise RouteError("model distance metric must be 'chamfer' or 'voxel'")


def _a_recognize_augment(*, points, seed, subsample_to=None):
    """Seeded point-cloud augmentation: the positive pair for contrastive retrieval."""
    from harnesscad.domain.reconstruction.recognize import pointcloud_augment as m

    return {"pair": m.positive_pair([tuple(p) for p in points], int(seed),
                                    subsample_to=subsample_to)}


def _a_recognize_dedup(*, models, bits=6):
    """mmABC dataset curation: hash-and-drop duplicate models."""
    from harnesscad.domain.reconstruction.recognize import dedup as m

    prepared = tuple((str(name), tuple(tuple(p) for p in pts)) for name, pts in models)
    result = m.deduplicate(prepared, bits=int(bits))
    return {"kept": result.kept, "dropped": result.dropped, "groups": result.groups}


# --- scene -----------------------------------------------------------------
def _a_scene_build(*, primitives, config=None):
    """CAD primitives -> a scene graph (directional / adjacency / support relations)."""
    from harnesscad.domain.reconstruction.scene import construction as m

    return {"graph": m.build_scene_graph(list(primitives), config)}


def _a_scene_enrich(*, graph, labels, vocabulary=None, affordances=None):
    """Semantic enrichment of a scene graph: group / name / affordance per node."""
    from harnesscad.domain.reconstruction.scene import enrichment as m

    enriched = m.enrich_graph(graph, dict(labels), vocabulary=vocabulary,
                              affordances=affordances)
    return {"graph": enriched, "coverage": m.coverage(enriched)}


def _a_scene_validate(*, graph, require_inverses=True, report_isolated=True):
    """Scene-graph consistency: dangling edges, self loops, inverse consistency, cycles."""
    from harnesscad.domain.reconstruction.scene import validity as m

    report = m.check_scene_graph(graph, require_inverses=bool(require_inverses),
                                 report_isolated=bool(report_isolated))
    return {"ok": report.ok(), "issues": report.issues, "codes": report.codes()}


def _a_scene_query(*, graph, obj_type=None, source=None, target=None, relation=None):
    """Query / reason over a scene graph (types, relations, paths, components)."""
    from harnesscad.domain.reconstruction.scene import query as m

    out: Dict[str, Any] = {"counts": m.count_by_type(graph)}
    if obj_type is not None:
        out["objects"] = m.objects_of_type(graph, obj_type)
    if source is not None and target is not None:
        out["relation"] = m.relation_between(graph, source, target)
        out["path"] = m.shortest_path(graph, source, target, relation=relation)
        out["path_exists"] = m.path_exists(graph, source, target, relation=relation)
    return out


def _a_scene_functional(*, graph, unit_groups, pipe_groups, labels=None, relation=None):
    """Functional-relation extraction over a scene graph (Algorithm 1)."""
    from harnesscad.domain.reconstruction.scene import functional_relations as m

    units = m.find_functional_units(graph, unit_groups, relation=relation)
    return {"units": units,
            "functional_graph": m.extract_functional_relations(
                graph, units, pipe_groups, labels=labels, relation=relation)}


def _a_scene_answer(*, model, question):
    """Grounded CAD question answering (abstains rather than guessing)."""
    from harnesscad.domain.reconstruction.scene import answer_engine as m

    answer = m.answer_question(model, question)
    return {"answer": answer, "abstained": answer.abstained,
            "parts": answer.part_ids}


# --- sequences -------------------------------------------------------------
def _a_seq_grammar(*, tokens):
    """CADParser command-workflow grammar: is this token sequence even legal?"""
    from harnesscad.domain.reconstruction.sequences import command_grammar as m

    return {"valid": m.is_valid(list(tokens)), "final_state": m.run(list(tokens))}


def _a_seq_valid_actions(*, faces, boolean_ops=None):
    """RLCAD Algorithm 1: the legal extrude/revolve actions for a face set."""
    from harnesscad.domain.reconstruction.sequences import valid_actions as m

    kw = {} if boolean_ops is None else {"boolean_ops": tuple(boolean_ops)}
    actions = m.generate_valid_actions(list(faces), **kw)
    aset = m.ValidActionSet(tuple(faces),
                            tuple(boolean_ops) if boolean_ops else (m.NEWBODY,))
    return {"actions": actions, "action_space_size": aset.action_space_size()}


def _a_seq_action_plan(*, graph):
    """Graph-CAD stage 2: a decomposition graph -> an ordered CAD action sequence."""
    from harnesscad.domain.reconstruction.sequences import action_plan as m

    actions = m.plan_actions(graph)
    return {"actions": actions, "plan": m.render_plan(actions),
            "valid": m.validate_plan(actions),
            "histogram": m.action_histogram(actions)}


def _a_seq_brep_linkage(*, state, steps):
    """Pointer-CAD B-rep <-> command linkage: replay a step list against a B-rep state."""
    from harnesscad.domain.reconstruction.sequences import brep_linkage as m

    return {"results": m.replay(state, list(steps))}


def _a_seq_candidate_select(*, candidates, strategy="bbox_iou", target_cloud=None,
                            reference_box=None, reference_bool=None, seed=0):
    """PS-CAD single-step selection: pick among candidate modelling steps."""
    from harnesscad.domain.reconstruction.sequences import candidate_selection as m

    return {"selection": m.select_candidate(
        list(candidates), strategy=strategy, target_cloud=target_cloud,
        reference_box=reference_box, reference_bool=reference_bool, seed=int(seed))}


def _a_seq_cascade(*, n, steps, seed=0):
    """CMT cascade stages + the masked-autoregressive reveal schedule."""
    from harnesscad.domain.reconstruction.sequences import cascade_schedule as m

    stages = m.cascade_stages()
    return {"stages": stages, "order_valid": m.validate_stage_order(stages),
            "schedule": m.mar_schedule(int(n), int(steps), seed=int(seed))}


def _a_seq_cell_graph(*, model, probe, res=12, tolerance=None, limit=None):
    """Surface-CSG cell adjacency + the plausible construction sequences it admits."""
    from harnesscad.domain.reconstruction.sequences import cell_graph as m

    graph = m.build_cell_graph(model, probe, res=int(res), tol=tolerance)
    return {"graph": graph, "connected": graph.is_connected(),
            "sequences": m.plausible_sequences(graph, limit=limit)}


def _a_seq_construction(*, graph, order="interleaved"):
    """SketchGraphs construction-sequence extraction (interleaved or constraints-last)."""
    from harnesscad.domain.reconstruction.sequences import construction_sequence as m

    if order == "interleaved":
        seq = m.interleaved_sequence(graph)
    elif order == "constraints_last":
        seq = m.constraints_last_sequence(graph)
    else:
        raise RouteError("construction order must be 'interleaved' or 'constraints_last'")
    return {"sequence": seq, "tokens": seq.tokens(), "valid": seq.is_valid(),
            "replayed": m.replay(seq)}


def _a_seq_gym(*, target, actions, max_steps=None):
    """RLCAD gym environment: the deterministic core of the training loop."""
    from harnesscad.domain.reconstruction.sequences import gym_env as m

    env = m.RevolveGymEnv(target, tuple(actions)) if max_steps is None \
        else m.RevolveGymEnv(target, tuple(actions), max_steps=int(max_steps))
    env.reset()
    return {"env": env, "action_keys": env.action_keys(),
            "valid_action_keys": env.valid_action_keys(),
            "reward": m.composite_reward(env.state(), target)}


def _a_seq_part_sequence(*, grid=None, sequence=None, h=None, w=None,
                         codebook_size=None, prompt=()):
    """TAR3D next-part prediction: a triplane index grid <-> its token sequence."""
    from harnesscad.domain.reconstruction.sequences import part_sequence as m

    if grid is not None:
        seq = m.build_sequence(grid)
    elif sequence is not None:
        seq = list(sequence)
    else:
        raise RouteError("part-sequence route needs grid= or sequence=")
    out: Dict[str, Any] = {"sequence": seq,
                           "targets": m.next_part_targets(seq, tuple(prompt))}
    if None not in (h, w, codebook_size):
        out["valid"] = m.is_valid_sequence(seq, int(h), int(w), int(codebook_size))
        out["detokenized"] = m.detokenize(seq, int(h), int(w), int(codebook_size))
    return out


def _a_seq_stats(*, workflow=None, corpus=None):
    """CADParser workflow statistics + the last-step data augmentation."""
    from harnesscad.domain.reconstruction.sequences import sequence_stats as m

    out: Dict[str, Any] = {}
    if workflow is not None:
        wf = list(workflow)
        out["length"] = m.sequence_length(wf)
        out["steps"] = m.split_steps(wf)
        out["augmented"] = m.augment(wf)
    if corpus is not None:
        c = [list(w) for w in corpus]
        out["operation_ratio"] = m.operation_ratio(c)
        out["length_distribution"] = m.length_distribution(c)
    if not out:
        raise RouteError("sequence-stats route needs workflow= or corpus=")
    return out


def _a_seq_shared_attributes(*, models, keys=None):
    """Img2CAD's sheaf-inspired shared attribute space (the conditional factorisation)."""
    from harnesscad.domain.reconstruction.sequences import shared_attributes as m

    prior = m.SharedAttributePrior()
    for model in models:
        prior.add_model(model)
    prior.fit()
    return {"prior": prior, "keys": prior.keys(), "count": prior.count(),
            "coverage": prior.coverage()}


# --- translate: CAD -> program / text --------------------------------------
def _a_translate_cadquery(*, commands):
    """DeepCAD command sequence -> a CadQuery PROGRAM (and its source)."""
    from harnesscad.domain.reconstruction.translate import cadquery_translate as m

    program = m.translate_to_program(list(commands))
    return {"language": "cadquery", "program": program,
            "source": m.translate_to_code(list(commands))}


def _a_translate_brep_describe(*, brep):
    """B-rep primitives -> natural-language descriptions (FutureCAD / BRepGround)."""
    from harnesscad.domain.reconstruction.translate import brep_describe as m

    prims = list(brep)
    return {"descriptions": m.describe_all(prims),
            "detailed": [m.describe_detailed(p) for p in prims],
            "round_trips": [m.round_trips(p, prims) for p in prims]}


def _a_translate_grounding(*, steps):
    """Is every feature in the program grounded in a resolvable B-rep query?"""
    from harnesscad.domain.reconstruction.translate import grounding_consistency as m

    report = m.check_program(list(steps))
    return {"ok": report.ok, "steps": report.steps, "failures": report.failures()}


# --- evaluate: pred/gold scoring routes ------------------------------------
def _a_eval_dimension_accuracy(*, gold, pred, tau_v, tau_e):
    from harnesscad.domain.reconstruction.evaluate import dimension_accuracy as m

    result = m.dimension_accuracy(list(gold), list(pred), float(tau_v), float(tau_e))
    return {"accuracy": result.accuracy, "correct": result.correct,
            "total": result.total}


def _a_eval_edge_face_prf(*, pred_edges=None, gold_edges=None, pred_faces=None,
                          gold_faces=None, tolerance=1e-6):
    from harnesscad.domain.reconstruction.evaluate import edge_face_prf as m

    out: Dict[str, Any] = {}
    if pred_edges is not None and gold_edges is not None:
        out["edge"] = m.edge_prf(list(pred_edges), list(gold_edges), float(tolerance))
    if pred_faces is not None and gold_faces is not None:
        out["face"] = m.face_prf(list(pred_faces), list(gold_faces))
    if not out:
        raise RouteError("edge/face PRF needs pred_edges+gold_edges or pred_faces+gold_faces")
    return out


def _a_eval_expressivity(*, observed, supported, policy="reject"):
    from harnesscad.domain.reconstruction.evaluate import expressivity as m

    report = m.expressivity_report(observed, supported, policy=policy)
    return {"reconstructable": report.reconstructable(),
            "unsupported": report.unsupported, "policy": report.policy}


def _a_eval_failure_audit(*, watertight=True, missing_faces=0, self_intersections=0,
                          consistency_error=0.0, overmerged=0):
    from harnesscad.domain.reconstruction.evaluate import failure_audit as m

    return {"failure": m.classify(
        watertight=bool(watertight), missing_faces=int(missing_faces),
        self_intersections=int(self_intersections),
        consistency_error=float(consistency_error), overmerged=int(overmerged))}


def _a_eval_op_map(*, face_pred, face_truth, curve_pred, curve_truth, user_stroke,
                   curve_head="regression"):
    from harnesscad.domain.reconstruction.evaluate import op_map_metrics as m

    return m.operation_report(face_pred, face_truth, curve_pred, curve_truth,
                              user_stroke, curve_head=curve_head)


def _a_eval_pmse(*, targets, predicted, param_targets, param_pred,
                 lambda_ce=1.0, lambda_p_mse=1.0):
    from harnesscad.domain.reconstruction.evaluate import pmse_loss as m

    loss = m.combined_loss(list(targets), list(predicted), list(param_targets),
                           list(param_pred), lambda_ce=float(lambda_ce),
                           lambda_p_mse=float(lambda_p_mse))
    return {"total": loss.total, "ce": loss.ce, "p_mse": loss.p_mse}


def _a_eval_pointer(*, predictions, valid_sets):
    from harnesscad.domain.reconstruction.evaluate import pointer_metrics as m

    acc = m.pointer_accuracy(list(predictions), list(valid_sets))
    return {"accuracy": acc.accuracy(), "hits": acc.hits, "total": acc.total}


def _a_eval_primitive_match(*, pred, gold, iou_threshold=0.5):
    from harnesscad.domain.reconstruction.evaluate import primitive_match_metrics as m

    return m.evaluate(list(pred), list(gold), iou_threshold=float(iou_threshold))


def _a_eval_sequence_efficiency(*, max_faces, max_edges_per_face, total_edge_count):
    from harnesscad.domain.reconstruction.evaluate import sequence_efficiency as m

    report = m.compare(int(max_faces), int(max_edges_per_face), int(total_edge_count))
    return {"tree_length": report.tree_length, "graph_length": report.graph_length,
            "sequence_reduction": report.sequence_reduction,
            "attention_reduction": report.attention_reduction,
            "redundancy_ratio": report.redundancy_ratio}


def _a_eval_topology_validity(*, adjacency, edges=None, require_edges=2):
    from harnesscad.domain.reconstruction.evaluate import topology_validity as m

    adj = tuple(tuple(bool(v) for v in row) for row in adjacency)
    e = None if edges is None else tuple(
        tuple(tuple(float(c) for c in p) for p in edge) for edge in edges)
    return {"valid": m.is_valid(adj, e, require_edges=int(require_edges)),
            "adjacency_diagnostics": m.check_adjacency(adj, require_edges=int(require_edges))}


def _a_eval_ver_score(*, results):
    from harnesscad.domain.reconstruction.evaluate import ver_score as m

    return {"ver_score": m.ver_score(list(results))}


# --- io.ingest: the front of the ingest leg (files, scans, drawings in) ----
def _a_ingest_import_brep(*, path):
    """Load a reference solid (STEP / IGES / STL) through OCCT -- honest when absent."""
    from harnesscad.io.ingest import import_brep as m

    part = m.import_solid(path)
    return {"part": part, "ok": part.ok(), "format": part.fmt,
            "metrics": part.metrics, "bbox": part.bbox,
            "available": part.available, "note": part.note}


def _a_ingest_metadata(*, path):
    """STEP AP242 metadata (name / material / BOM / PMI / assembly tree) via XCAF."""
    from harnesscad.io.ingest import metadata as m

    meta = m.extract_metadata(path)
    return {"metadata": meta, "ok": meta.ok(), "available": meta.available,
            "note": meta.note}


def _a_ingest_fidelity(*, imported, rebuilt=None, rel_tol=None):
    """Round-trip fidelity: does the rebuilt model still measure like the import?"""
    from harnesscad.io.ingest import fidelity as m

    kw = {} if rel_tol is None else {"rel_tol": float(rel_tol)}
    report = m.roundtrip_fidelity(imported, rebuilt, **kw)
    return {"report": report, "ok": report.ok(), "deltas": report.deltas,
            "lost_metadata": report.lost_metadata}


def _a_ingest_reconcile(*, evidence, required_sources=None, relative_tolerance=0.01):
    """Cross-source reconciliation: model vs drawing vs reference -> discrepancies."""
    from harnesscad.io.ingest import reconcile as m

    kw: Dict[str, Any] = {"relative_tolerance": float(relative_tolerance)}
    if required_sources is not None:
        kw["required_sources"] = tuple(required_sources)
    report = m.reconcile(list(evidence), **kw)
    return {"report": report, "ok": report.ok(),
            "discrepancies": report.discrepancies, "by_kind": report.by_kind()}


def _a_ingest_cross_section(*, triangles, origin, normal, tolerance=1e-6):
    """Analytic plane/triangle cross-section -> stitched 2D contours (mesh -> contour)."""
    from harnesscad.io.ingest import cross_section as m

    return {"contours": m.cross_section(list(triangles), tuple(origin),
                                        tuple(normal), tolerance=float(tolerance))}


def _a_ingest_brep_hierarchy(*, hierarchy, manifold=False):
    """Kernel-neutral B-rep hierarchy: strict integrity checks + face neighbours."""
    from harnesscad.io.ingest import brep_hierarchy as m  # noqa: F401 - type owner

    issues = hierarchy.validate(manifold=bool(manifold))
    return {"ok": not issues, "issues": issues,
            "face_neighbors": hierarchy.face_neighbors()}


def _a_ingest_brep_annotations(*, entities, tags=(), previous=None):
    """Persistent entity ids across a rebuild + external tag assignment."""
    from harnesscad.io.ingest import brep_annotations as m

    out: Dict[str, Any] = {}
    if previous is not None:
        out["persisted"] = m.persist_entity_ids(dict(previous), list(entities.values()))
    result = m.assign_external_tags(dict(entities), list(tags))
    out.update({"assignments": result.assignments, "annotations": result.annotations,
                "issues": result.issues})
    return out


def _a_ingest_brep_sequence(*, source_digest, instruction, steps):
    """History-free B-rep edit sequence: a stable, digestible serialisation."""
    from harnesscad.io.ingest import brep_sequence as m

    seq = m.BrepEditSequence.build(str(source_digest), str(instruction), list(steps))
    return {"sequence": seq, "digest": seq.digest(), "payload": seq.to_dict()}


def _a_ingest_brep_tokens(*, items, orientation_semantic=True, padding=1):
    """Canonical encoding of a cyclic directed B-rep loop (rotation-invariant)."""
    from harnesscad.io.ingest import brep_tokens as m

    return {"canonical": m.canonical_cycle(list(items),
                                           orientation_semantic=bool(orientation_semantic)),
            "tokens": m.loop_tokens(list(items),
                                    orientation_semantic=bool(orientation_semantic),
                                    padding=int(padding))}


def _a_ingest_scan_labels(*, labels, complex):
    """Chain-complex validation of pointwise scan-to-B-rep labels."""
    from harnesscad.io.ingest import scan_brep_labels as m

    return {"issues": m.validate(list(labels), complex)}


def _a_ingest_sketch_validity(*, entities, constraints=()):
    """CadVLM whole-sketch validity: primitive and constraint token checks."""
    from harnesscad.io.ingest import sketch_validity as m

    report = m.check_sketch(list(entities), tuple(constraints))
    return {"valid": report.valid, "issues": report.all_issues()}


def _a_ingest_constraint_validity(*, constraints, primitive_types):
    """DAVINCI valid-subreference set: which constraint pairs are even legal?"""
    from harnesscad.io.ingest import constraint_validity as m

    result = m.filter_constraints(list(constraints), list(primitive_types))
    return {"valid": result.valid, "invalid": result.invalid,
            "all_valid": result.all_valid()}


def _a_ingest_constraint_preserving(*, sketch, seed=None, quarter_turns=None):
    """Constraint-Preserving Transformations: augment a sketch without breaking it."""
    from harnesscad.io.ingest import constraint_preserving as m

    if quarter_turns is not None:
        return {"sketch": m.rotate_sketch(sketch, int(quarter_turns))}
    if seed is None:
        raise RouteError("constraint-preserving augmentation needs seed= or quarter_turns=")
    permuted, perm = m.random_permutation(sketch, int(seed))
    return {"sketch": permuted, "permutation": perm,
            "preserved": m.constraints_preserved(sketch, permuted, perm)}


def _a_ingest_entity_sequence(*, entities=None, sequence=None):
    """CadVLM entity-level embedding sequence (the inductive bias), and its parse."""
    from harnesscad.io.ingest import entity_sequence as m

    if entities is not None:
        return {"sequence": m.build_sequence(list(entities)),
                "flat": m.flat_sequence(list(entities)),
                "segments": m.entity_segments(list(entities))}
    if sequence is not None:
        parsed = m.parse_sequence(sequence)
        return {"parsed": parsed, "entity_count": parsed.entity_count()}
    raise RouteError("entity-sequence route needs entities= or sequence=")


def _a_ingest_sketch_frame(*, origin, angle, points=(), bins=None):
    """A sketch frame: local <-> world, with optional coordinate quantisation."""
    from harnesscad.io.ingest import sketch_frame as m

    frame = m.SketchFrame(tuple(origin), float(angle))
    out: Dict[str, Any] = {
        "frame": frame,
        "world": [frame.local_to_world(tuple(p)) for p in points],
        "local": [frame.world_to_local(tuple(p)) for p in points],
    }
    if bins is not None:
        out["quantized"] = m.quantize([c for p in points for c in p], int(bins))
    return out


def _a_ingest_spatial_order(*, patches):
    """Stable Morton (Z-order) ordering for quadtree surface patches."""
    from harnesscad.io.ingest import spatial_order as m

    return {"order": m.patch_order(list(patches))}


def _a_ingest_orientation(*, candidates, experts, expert_weights=None, seed=0,
                          coarse_count=16, fine_per_mode=4):
    """Orientation hypotheses fused by a product of experts, then sampled."""
    from harnesscad.io.ingest import orientation as m

    distribution = m.product_of_experts(list(candidates), dict(experts),
                                        expert_weights=expert_weights)
    samples = m.coarse_to_fine_samples(distribution, seed=int(seed),
                                       coarse_count=int(coarse_count),
                                       fine_per_mode=int(fine_per_mode))
    return {"distribution": distribution, "best": distribution.best(),
            "confidence": distribution.confidence(),
            "ambiguity": distribution.ambiguity(), "samples": samples}


def _a_ingest_assembly_transform(*, points, extent=3):
    """The one invertible condition-derived transform shared by an assembly pair."""
    from harnesscad.io.ingest import assembly_transform as m

    transform = m.fit_condition_transform([tuple(p) for p in points], extent=extent)
    applied = [transform.apply(tuple(p)) for p in points]
    return {"transform": transform, "applied": applied,
            "inverted": [transform.invert(p) for p in applied]}


def _a_ingest_contact_faces(*, left_id, left, right_id, right, sampler, projector,
                            tolerance=0.1, min_support=1):
    """Bidirectional sampled contact evidence between two injected parametric faces."""
    from harnesscad.io.ingest import contact_faces as m

    evidence = m.contact_evidence(left_id, left, right_id, right, sampler=sampler,
                                  projector=projector, tolerance=float(tolerance),
                                  min_support=int(min_support))
    return {"evidence": evidence, "contact": evidence.contact,
            "min_distance": evidence.min_distance}


def _a_ingest_bezier(*, control_points, t=None, degree=None, s=None):
    """Analytic Bezier evaluation (curve or triangular patch)."""
    from harnesscad.io.ingest import bezier as m

    if degree is not None:
        if s is None or t is None:
            raise RouteError("a Bezier triangle needs s= and t=")
        return {"point": m.bezier_triangle(control_points, int(degree), float(s),
                                           float(t))}
    if t is None:
        raise RouteError("a Bezier curve needs t=")
    return {"point": m.bezier_curve(control_points, float(t))}


def _a_ingest_fourier_features(*, point, frequencies=(1.0, 2.0, 4.0),
                               include_coordinates=True):
    """Declared Fourier coordinate features (the positional encoding, made explicit)."""
    from harnesscad.io.ingest import fourier_features as m

    return {"features": m.fourier_features(tuple(point), tuple(frequencies),
                                           include_coordinates=bool(include_coordinates))}


def _a_ingest_step_reserialize(*, step, precision=6):
    """DFS-based STEP reserialisation with CoT-style structural annotations."""
    from harnesscad.io.ingest import step_reserialize as m

    return {"text": m.reserialize(step, precision=int(precision)),
            "order": m.dfs_order(step), "branches": m.branch_stats(step),
            "annotated": m.annotate(step)}


# --------------------------------------------------------------------------- #
# The route table: (name, inputs, output, modules, adapter, doc, family)
#
# It binds adapters to MODULES; Route objects are materialised from the
# capability index (see _build_routes), so what exists in the tree is what
# exists here.
# --------------------------------------------------------------------------- #
_ROUTE_TABLE: Tuple[Tuple[str, Tuple[str, ...], str, Tuple[str, ...], Adapter, str, str], ...] = (
    # --- tokens -> CAD ------------------------------------------------------
    ("tokens.to_cisp", ("tokens",), "cisp_ops", ("ingest_pipeline",),
     _a_tokens_to_ops,
     "family-tagged CAD tokens -> an applied, editable CISP op stream", "token_family"),
    ("tokens.text2cad.decode", ("tokens",), "commands",
     ("tokens.text2cad_vector_codec",), _a_tokens_text2cad,
     "Text2CAD vector codec -> a CAD model", ""),
    ("tokens.text2cad.encode", ("commands",), "tokens",
     ("tokens.text2cad_vector_codec",), _a_tokens_text2cad_encode,
     "a CAD model -> Text2CAD's exact cad/flag/index vectors", ""),
    ("tokens.text2cad.serialize", ("sketch_loops",), "tokens",
     ("tokens.text2cad_tokens",), _a_tokens_text2cad_serialize,
     "faces + extrusion -> the flat Text2CAD token stream", ""),
    ("tokens.image2cadseq", ("commands",), "tokens",
     ("tokens.image2cadseq",), _a_tokens_image2cadseq,
     "Sim-Gallery DSL ops <-> the vectorised feature matrix", ""),
    ("tokens.pht_cad", ("sketch_curves",), "tokens", ("tokens.pht_cad",),
     _a_tokens_pht_cad,
     "PHT-CAD Efficient Hybrid Parametrization: primitives -> tokens", ""),
    ("tokens.pointer_cad", ("params",), "tokens", ("tokens.pointer_cad",),
     _a_tokens_pointer_cad,
     "Pointer-CAD vocabulary: values/angles <-> token ids", ""),
    ("tokens.cad2program", ("commands",), "tokens", ("tokens.cad2program",),
     _a_tokens_cad2program,
     "cad2program fixed-slot command template + quantisation error", ""),
    ("tokens.deepcad_arc", ("sketch_curves",), "sketch_curves",
     ("tokens.deepcad_arc_macro",), _a_tokens_deepcad_arc,
     "DeepCAD arc macro: (end, sweep, ccw) <-> a full arc", ""),
    ("tokens.hnc_codebook", ("features",), "codes", ("tokens.hnc_codebooks",),
     _a_tokens_hnc_codebooks,
     "HNC-CAD nearest-codebook neural-code assignment", ""),
    ("tokens.hnc_spl_tree", ("sketch_loops",), "tokens", ("tokens.hnc_spl_tree",),
     _a_tokens_hnc_spl,
     "HNC-CAD Solid-Profile-Loop hierarchy from loops", ""),
    ("tokens.skexgen_codes", ("codes",), "codes", ("tokens.skexgen_code_layout",),
     _a_tokens_skexgen_codes,
     "SkexGen disentangled codebook layout: topology | geometry | extrude", ""),
    ("tokens.sketch2cad_scene", ("scene_primitives",), "tokens",
     ("tokens.sketch2cad_scene",), _a_tokens_sketch2cad_scene,
     "Sketch2CAD scene-descriptor codec (scene objects <-> tokens)", ""),
    ("tokens.vitruvion_constraints", ("sketch_graph",), "tokens",
     ("tokens.vitruvion_constraints",), _a_tokens_vitruvion_constraints,
     "Vitruvion's constraint hypergraph as a pointer-token stream", ""),
    ("tokens.gencad_quantize", ("params",), "tokens", ("tokens.gencad_quantize",),
     _a_tokens_gencad_quantize,
     "GenCAD/DeepCAD exact quantiser (a RIVAL of the deepcad ingest quantiser)", ""),

    # --- point clouds / images / drawings -> geometry -----------------------
    ("fit.bbox_primitive", ("point_cloud",), "primitives",
     ("fitting.pointcloud_adapter", "fitting.primitive_shapes",
      "sequences.candidate_selection"), _a_fit_bbox_primitive,
     "point cloud -> a fitted primitive (axis-aligned bounding-box fit)", ""),
    ("fit.normalize_cloud", ("point_cloud",), "point_cloud",
     ("fitting.pointcloud_adapter",), _a_fit_normalize_cloud,
     "unit-cube normalisation + furthest-point sampling (cadrille)", ""),
    ("fit.visual_hull", ("silhouettes",), "voxels", ("fitting.visual_hull",),
     _a_fit_visual_hull,
     "silhouette masks -> a carved visual hull (voxel centres)", ""),
    ("fit.extrude_contour", ("contour2d",), "solid",
     ("fitting.solid_regeneration",), _a_fit_extrude_contour,
     "2D contour + depth -> a prism solid with volume and area", ""),
    ("fit.wireframe", ("segments",), "wireframe",
     ("fitting.wireframe_schema", "fitting.wireframe_binding"), _a_fit_wireframe,
     "line proposals -> a bound, validated Structured Visual Geometry wireframe", ""),
    ("fit.loi_align", ("segments",), "segments", ("fitting.loi_align",),
     _a_fit_loi_align,
     "Joint-Decoupled Line-of-Interest alignment: drop false-positive lines", ""),
    ("fit.residual_regions", ("voxels",), "regions",
     ("fitting.residual_regions",), _a_fit_residual_regions,
     "PS-CAD residual guidance: where does the current solid miss the target?", ""),
    ("fit.param_decode", ("heatmaps",), "params", ("fitting.param_decode",),
     _a_fit_param_decode,
     "Sketch2CAD op maps -> the stitching face of an operation", ""),
    ("fit.progressive_tuning", ("params",), "params",
     ("fitting.progressive_tuning",), _a_fit_progressive_tuning,
     "PHT-CAD progressive hierarchical parameter refinement", ""),
    ("fit.condition", ("condition",), "features", ("fitting.condition_schema",),
     _a_fit_condition,
     "DreamCAD multi-modal condition encoding (text / points / voxels)", ""),
    ("fit.pointcloud_candidates", ("point_cloud",), "selection",
     ("fitting.pointcloud_candidates",), _a_fit_pointcloud_candidates,
     "compile-and-distance selection of candidates against a target cloud", ""),

    # --- drawings -----------------------------------------------------------
    ("ortho.reconstruct", ("drawing_svg",), "topology", ("ortho.pipeline",),
     _a_ortho_reconstruct,
     "orthographic SVG views -> 3D edges, face loops and a manifold gate", ""),
    ("ortho.merge", ("primitives",), "primitives", ("ortho.brep_merge",),
     _a_ortho_merge,
     "two-signal (bbox + shape) clustering of duplicate recovered geometry", ""),
    ("ortho.stitch_geometry", ("edges3d",), "diagnostics",
     ("ortho.geometry_stitch",), _a_ortho_stitch_geometry,
     "kernel-free geometry reconciliation: vertex averaging + edge consistency", ""),
    ("ortho.point_labels", ("labels",), "labels", ("ortho.point_labels",),
     _a_ortho_point_labels,
     "deterministic suppression of scan-point B-rep labels", ""),
    ("ortho.primitive_relations", ("primitives",), "diagnostics",
     ("ortho.primitive_relations",), _a_ortho_primitive_relations,
     "infer + project the relation between two recovered primitives", ""),
    ("ortho.primitive_stitch", ("params",), "params", ("ortho.primitive_stitch",),
     _a_ortho_primitive_stitch,
     "fixed-point primitive stitching (iterate a step to a residual)", ""),
    ("ortho.primitive_intersections", ("primitives",), "topology",
     ("ortho.primitive_intersections",), _a_ortho_primitive_intersections,
     "assemble a topology from pairwise / triple primitive intersections", ""),

    # --- B-rep: THE THREE RIVAL GRAPH ENCODINGS -----------------------------
    ("brep.graph.cadparser", ("brep_topology",), "brep_graph",
     ("brep.cadparser_graph",), _a_brep_graph_cadparser,
     "RIVAL: face/edge/coedge graph with CATEGORICAL features (CADParser)",
     "brep_graph_encoding"),
    ("brep.graph.uvnet", ("brep_parametric",), "brep_graph",
     ("brep.uvnet_face_adjacency",), _a_brep_graph_uvnet,
     "RIVAL: face-adjacency graph with SAMPLED UV-grid features (UV-Net)",
     "brep_graph_encoding"),
    ("brep.graph.graphbrep", ("adjacency_matrix",), "graph_key",
     ("brep.graphbrep_canonical",), _a_brep_graph_graphbrep,
     "RIVAL: permutation-CANONICALISED surface-adjacency matrix (GraphBrep)",
     "brep_graph_encoding"),
    ("brep.entity_features", ("brep_entities",), "features",
     ("brep.entity_features",), _a_brep_entity_features,
     "JoinABLe entity feature vectors for joint prediction", ""),
    ("brep.structured", ("node_tree",), "tokens", ("brep.structured_brep",),
     _a_brep_structured,
     "fixed-width face-edge-vertex hierarchy encoding (padded, validated)", ""),
    ("brep.topology_predict", ("edges3d",), "adjacency_matrix",
     ("brep.topology_predictor",), _a_brep_topology_predict,
     "CMT topology predictor: edges + surface boxes -> a face-edge adjacency", ""),
    ("brep.complex_nms", ("probabilistic_complex",), "chain_complex",
     ("brep.chain_complex_nms",), _a_brep_complex_nms,
     "ComplexGen NMS: a probabilistic B-rep complex -> a definite one", ""),
    ("brep.complex_io", ("complex_text",), "chain_complex",
     ("brep.chain_complex_io",), _a_brep_complex_io,
     "ComplexGen .complex chain-complex file: parse / serialise", ""),

    # --- sketch: THE THREE RIVAL CANONICAL ORDERINGS ------------------------
    ("sketch.order.gencad", ("sketch_loops",), "sketch_loops",
     ("sketch.gencad_canonical_order",), _a_sketch_order_gencad,
     "RIVAL: GenCAD/DeepCAD canonical loop+profile order",
     "canonical_sketch_ordering"),
    ("sketch.order.skexgen", ("sketch_loops",), "sketch_loops",
     ("sketch.skexgen_canonical_order",), _a_sketch_order_skexgen,
     "RIVAL: SkexGen canonical face/loop/curve order",
     "canonical_sketch_ordering"),
    ("sketch.order.deepcad_profile", ("commands",), "sketch_loops",
     ("sketch.profile_assembly",), _a_sketch_order_deepcad_profile,
     "RIVAL: DeepCAD loop/profile assembly off the command stream",
     "canonical_sketch_ordering"),
    ("sketch.reference_plane", ("extrudes",), "params",
     ("sketch.reference_plane",), _a_sketch_reference_plane,
     "OpenECAD reference-plane finding: which existing face hosts this sketch?", ""),
    ("sketch.pointer_plane", ("params",), "params",
     ("sketch.pointer_sketch_plane",), _a_sketch_pointer_plane,
     "Pointer-CAD sketch-plane frame from a face normal + direction symbol", ""),

    # --- recognize ----------------------------------------------------------
    ("recognize.machining_features", ("brep_faces",), "features",
     ("recognize.machining_features",), _a_recognize_machining,
     "rule-based machining-feature recognition (holes, pockets, slots, ...)", ""),
    ("recognize.point_features", ("point_cloud",), "features",
     ("recognize.point_features",), _a_recognize_point_features,
     "per-point covariance descriptors for CAD part segmentation", ""),
    ("recognize.segment.prototypes", ("features",), "labels",
     ("recognize.prototypes",), _a_recognize_prototypes,
     "RIVAL: few-shot part segmentation by nearest multi-prototype",
     "few_shot_part_segmentation"),
    ("recognize.segment.label_propagation", ("features",), "labels",
     ("recognize.label_propagation",), _a_recognize_label_propagation,
     "RIVAL: few-shot part segmentation by transductive label propagation",
     "few_shot_part_segmentation"),
    ("recognize.label_transfer", ("point_cloud",), "labels",
     ("recognize.label_transfer",), _a_recognize_label_transfer,
     "nearest-neighbour label transfer between two clouds", ""),
    ("recognize.partseg_metrics", ("labels",), "score",
     ("recognize.partseg_metrics",), _a_recognize_partseg_metrics,
     "part-segmentation IoU / mIoU / accuracy", ""),
    ("recognize.partseg_episode", ("labels",), "episode",
     ("recognize.partseg_episodes",), _a_recognize_partseg_episode,
     "deterministic C-way K-shot episode construction", ""),
    ("recognize.instance_offsets", ("point_cloud",), "labels",
     ("recognize.instance_offsets",), _a_recognize_instance_offsets,
     "instance centroid-offset targets and the grouping they induce", ""),
    ("recognize.segmentation_consistency", ("labels",), "score",
     ("recognize.segmentation_consistency",), _a_recognize_consistency,
     "segmentation consistency: do nearby points share a part label?", ""),
    ("recognize.shape_descriptor", ("point_cloud",), "descriptor",
     ("recognize.shape_descriptors",), _a_recognize_shape_descriptor,
     "rotation-invariant shape descriptor (D2 + shells + PCA) for retrieval", ""),
    ("recognize.graph_descriptor", ("brep_graph",), "descriptor",
     ("recognize.graph_descriptors",), _a_recognize_graph_descriptor,
     "structural graph descriptor (degree histogram + WL) for CAD retrieval", ""),
    ("recognize.model_distance", ("point_clouds",), "score",
     ("recognize.model_distances",), _a_recognize_model_distance,
     "CAD-model distance for clustering (chamfer or voxel Jaccard)", ""),
    ("recognize.augment", ("point_cloud",), "point_clouds",
     ("recognize.pointcloud_augment",), _a_recognize_augment,
     "seeded augmentation: the positive pair for contrastive retrieval", ""),
    ("recognize.dedup", ("point_clouds",), "selection", ("recognize.dedup",),
     _a_recognize_dedup,
     "mmABC dataset curation: hash-and-drop duplicate models", ""),

    # --- scene --------------------------------------------------------------
    ("scene.build", ("scene_primitives",), "scene_graph", ("scene.construction",),
     _a_scene_build,
     "CAD primitives -> a scene graph (directional / adjacency / support)", ""),
    ("scene.enrich", ("scene_graph",), "scene_graph", ("scene.enrichment",),
     _a_scene_enrich,
     "semantic enrichment: group / name / affordance per scene node", ""),
    ("scene.validate", ("scene_graph",), "diagnostics", ("scene.validity",),
     _a_scene_validate,
     "scene-graph consistency: dangling edges, inverses, cycles", ""),
    ("scene.query", ("scene_graph",), "answer", ("scene.query",), _a_scene_query,
     "query / reason over a scene graph (types, relations, paths)", ""),
    ("scene.functional_relations", ("scene_graph",), "topology",
     ("scene.functional_relations",), _a_scene_functional,
     "functional-relation extraction over a scene graph", ""),
    ("scene.answer", ("scene_graph",), "answer", ("scene.answer_engine",),
     _a_scene_answer,
     "grounded CAD question answering (abstains rather than guessing)", ""),

    # --- sequences ----------------------------------------------------------
    ("sequence.grammar", ("tokens",), "diagnostics",
     ("sequences.command_grammar",), _a_seq_grammar,
     "CADParser command-workflow grammar: is this token sequence legal?", ""),
    ("sequence.valid_actions", ("brep_faces",), "actions",
     ("sequences.valid_actions",), _a_seq_valid_actions,
     "RLCAD Algorithm 1: the legal extrude/revolve actions for a face set", ""),
    ("sequence.action_plan", ("knowledge_graph",), "plan",
     ("sequences.action_plan",), _a_seq_action_plan,
     "Graph-CAD: a decomposition graph -> an ordered CAD action sequence", ""),
    ("sequence.brep_linkage", ("command_workflow",), "brep_topology",
     ("sequences.brep_linkage",), _a_seq_brep_linkage,
     "Pointer-CAD B-rep <-> command linkage: replay steps against a B-rep", ""),
    ("sequence.candidate_select", ("point_clouds",), "selection",
     ("sequences.candidate_selection",), _a_seq_candidate_select,
     "PS-CAD single-step selection among candidate modelling steps", ""),
    ("sequence.cascade_schedule", ("params",), "plan",
     ("sequences.cascade_schedule",), _a_seq_cascade,
     "CMT cascade stages + masked-autoregressive reveal schedule", ""),
    ("sequence.cell_graph", ("csg_model",), "plan", ("sequences.cell_graph",),
     _a_seq_cell_graph,
     "surface-CSG cell adjacency + the plausible construction sequences", ""),
    ("sequence.construction", ("sketch_graph",), "commands",
     ("sequences.construction_sequence",), _a_seq_construction,
     "SketchGraphs construction-sequence extraction (interleaved / last)", ""),
    ("sequence.gym", ("voxels",), "actions", ("sequences.gym_env",), _a_seq_gym,
     "RLCAD gym environment: the deterministic core of the training loop", ""),
    ("sequence.part_sequence", ("tokens",), "tokens",
     ("sequences.part_sequence",), _a_seq_part_sequence,
     "TAR3D next-part prediction: triplane index grid <-> token sequence", ""),
    ("sequence.stats", ("command_workflow",), "score",
     ("sequences.sequence_stats",), _a_seq_stats,
     "CADParser workflow statistics + last-step data augmentation", ""),
    ("sequence.shared_attributes", ("commands",), "params",
     ("sequences.shared_attributes",), _a_seq_shared_attributes,
     "Img2CAD sheaf-inspired shared attribute space (conditional factorisation)", ""),

    # --- translate: CAD -> program / text -----------------------------------
    ("translate.cadquery", ("commands",), "program_text",
     ("translate.cadquery_translate",), _a_translate_cadquery,
     "DeepCAD command sequence -> a CadQuery program and its source", ""),
    ("translate.brep_describe", ("brep_primitives",), "text",
     ("translate.brep_describe",), _a_translate_brep_describe,
     "B-rep primitives -> natural-language descriptions", ""),
    ("translate.grounding_consistency", ("program_steps",), "diagnostics",
     ("translate.grounding_consistency",), _a_translate_grounding,
     "is every program feature grounded in a resolvable B-rep query?", ""),

    # --- evaluate -----------------------------------------------------------
    ("evaluate.dimension_accuracy", ("params",), "score",
     ("evaluate.dimension_accuracy",), _a_eval_dimension_accuracy,
     "PHT-CAD Dimension Accuracy (type + value + element match)", ""),
    ("evaluate.edge_face_prf", ("edges3d",), "score", ("evaluate.edge_face_prf",),
     _a_eval_edge_face_prf,
     "coordinate-tolerant reconstruction precision / recall / F1", ""),
    ("evaluate.expressivity", ("features",), "score", ("evaluate.expressivity",),
     _a_eval_expressivity,
     "explicit supported-feature coverage and the OOD approximation policy", ""),
    ("evaluate.failure_audit", ("diagnostics",), "score",
     ("evaluate.failure_audit",), _a_eval_failure_audit,
     "stable failure taxonomy for reconstructed B-reps", ""),
    ("evaluate.op_map", ("heatmaps",), "score", ("evaluate.op_map_metrics",),
     _a_eval_op_map,
     "Sketch2CAD op-map metrics (face heatmap + curve class IoU)", ""),
    ("evaluate.pmse", ("params",), "score", ("evaluate.pmse_loss",), _a_eval_pmse,
     "PHT-CAD Parametric MSE loss (CE + P-MSE)", ""),
    ("evaluate.pointer", ("tokens",), "score", ("evaluate.pointer_metrics",),
     _a_eval_pointer,
     "Pointer-CAD pointer accuracy and topological soundness", ""),
    ("evaluate.primitive_match", ("primitives",), "score",
     ("evaluate.primitive_match_metrics",), _a_eval_primitive_match,
     "cad2program reconstruction / retrieval / parameter accuracy (Hungarian IoU)", ""),
    ("evaluate.sequence_efficiency", ("params",), "score",
     ("evaluate.sequence_efficiency",), _a_eval_sequence_efficiency,
     "GraphBrep efficiency / compactness (tree vs graph sequence length)", ""),
    ("evaluate.topology_validity", ("adjacency_matrix",), "score",
     ("evaluate.topology_validity",), _a_eval_topology_validity,
     "CMT topological validity of a generated B-rep", ""),
    ("evaluate.ver_score", ("text",), "score", ("evaluate.ver_score",),
     _a_eval_ver_score,
     "GeoCAD Ver-score: vertex-based text-to-CAD consistency", ""),

    # --- io.ingest: files / scans / drawings in ------------------------------
    ("ingest.import_brep", ("file",), "brep_topology", (_IO + "import_brep",),
     _a_ingest_import_brep,
     "load a reference solid (STEP / IGES / STL) via OCCT", ""),
    ("ingest.metadata", ("file",), "params", (_IO + "metadata",),
     _a_ingest_metadata,
     "STEP AP242 metadata (name / material / BOM / PMI / assembly tree)", ""),
    ("ingest.fidelity", ("brep_topology",), "score", (_IO + "fidelity",),
     _a_ingest_fidelity,
     "round-trip fidelity of an imported / rebuilt model", ""),
    ("ingest.reconcile", ("diagnostics",), "diagnostics", (_IO + "reconcile",),
     _a_ingest_reconcile,
     "cross-source reconciliation: model vs drawing vs reference", ""),
    ("ingest.cross_section", ("mesh",), "contour2d", (_IO + "cross_section",),
     _a_ingest_cross_section,
     "analytic plane/triangle cross-section -> stitched 2D contours", ""),
    ("ingest.brep_hierarchy", ("brep_topology",), "diagnostics",
     (_IO + "brep_hierarchy",), _a_ingest_brep_hierarchy,
     "kernel-neutral B-rep hierarchy: strict integrity checks + neighbours", ""),
    ("ingest.brep_annotations", ("brep_entities",), "labels",
     (_IO + "brep_annotations",), _a_ingest_brep_annotations,
     "persistent entity ids across a rebuild + external tag assignment", ""),
    ("ingest.brep_sequence", ("command_workflow",), "tokens",
     (_IO + "brep_sequence",), _a_ingest_brep_sequence,
     "history-free B-rep edit sequence: a stable, digestible serialisation", ""),
    ("ingest.brep_tokens", ("sketch_loops",), "tokens", (_IO + "brep_tokens",),
     _a_ingest_brep_tokens,
     "canonical encoding of a cyclic directed B-rep loop", ""),
    ("ingest.scan_labels", ("labels",), "diagnostics", (_IO + "scan_brep_labels",),
     _a_ingest_scan_labels,
     "chain-complex validation of pointwise scan-to-B-rep labels", ""),
    ("ingest.sketch_validity", ("sketch_graph",), "diagnostics",
     (_IO + "sketch_validity",), _a_ingest_sketch_validity,
     "CadVLM whole-sketch primitive/constraint validity", ""),
    ("ingest.constraint_validity", ("sketch_graph",), "diagnostics",
     (_IO + "constraint_validity",), _a_ingest_constraint_validity,
     "DAVINCI valid-subreference set: which constraint pairs are legal?", ""),
    ("ingest.constraint_preserving", ("sketch_graph",), "sketch_graph",
     (_IO + "constraint_preserving",), _a_ingest_constraint_preserving,
     "Constraint-Preserving Transformations: augment without breaking", ""),
    ("ingest.entity_sequence", ("sketch_curves",), "tokens",
     (_IO + "entity_sequence",), _a_ingest_entity_sequence,
     "CadVLM entity-level embedding sequence (and its parse)", ""),
    ("ingest.sketch_frame", ("params",), "params", (_IO + "sketch_frame",),
     _a_ingest_sketch_frame,
     "a sketch frame: local <-> world, with optional quantisation", ""),
    ("ingest.spatial_order", ("surfaces",), "tokens", (_IO + "spatial_order",),
     _a_ingest_spatial_order,
     "stable Morton (Z-order) ordering for quadtree surface patches", ""),
    ("ingest.orientation", ("params",), "params", (_IO + "orientation",),
     _a_ingest_orientation,
     "orientation hypotheses fused by a product of experts, then sampled", ""),
    ("ingest.assembly_transform", ("point_cloud",), "params",
     (_IO + "assembly_transform",), _a_ingest_assembly_transform,
     "the one invertible condition-derived transform shared by an assembly pair", ""),
    ("ingest.contact_faces", ("surfaces",), "diagnostics", (_IO + "contact_faces",),
     _a_ingest_contact_faces,
     "bidirectional sampled contact evidence between two parametric faces", ""),
    ("ingest.bezier", ("sketch_curves",), "sketch_curves", (_IO + "bezier",),
     _a_ingest_bezier,
     "analytic Bezier evaluation (curve or triangular patch)", ""),
    ("ingest.fourier_features", ("point_cloud",), "features",
     (_IO + "fourier_features",), _a_ingest_fourier_features,
     "declared Fourier coordinate features (the positional encoding, explicit)", ""),
    ("ingest.step_reserialize", ("file",), "text", (_IO + "step_reserialize",),
     _a_ingest_step_reserialize,
     "DFS-based STEP reserialisation with structural annotations", ""),
)


# --------------------------------------------------------------------------- #
# Rival families -- selected by name, never blended.
# --------------------------------------------------------------------------- #
RIVAL_FAMILIES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("token_family",
     "The four CAD quantiser families. Different level counts, different rounding "
     "rules, a rotation codebook instead of angles: decoding one family's tokens "
     "with another's dequantiser silently rescales every coordinate. "
     "Enforced in ingest_pipeline (FamilyMismatch).",
     ("deepcad", "skexgen", "hnc", "vitruvion")),
    ("brep_graph_encoding",
     "Three encodings of 'the B-rep as a graph'. cadparser uses categorical "
     "face/edge/coedge features on a topological B-rep; uvnet uses sampled UV-grid "
     "geometry per face; graphbrep canonicalises a surface-adjacency matrix under "
     "node permutation. They take different inputs and produce different graphs.",
     ("brep.graph.cadparser", "brep.graph.uvnet", "brep.graph.graphbrep")),
    ("canonical_sketch_ordering",
     "Three canonical orders for the same sketch. Applying one paper's order and "
     "then another paper's decoder is a real bug: the loops come out permuted.",
     ("sketch.order.gencad", "sketch.order.skexgen", "sketch.order.deepcad_profile")),
    ("few_shot_part_segmentation",
     "Two different algorithms for the same few-shot task (nearest multi-prototype "
     "vs transductive label propagation). They disagree by design; pick one.",
     ("recognize.segment.prototypes", "recognize.segment.label_propagation")),
)


def rivals() -> Dict[str, Tuple[str, ...]]:
    """family -> the mutually-exclusive members in it (never merged)."""
    return {name: members for name, _doc, members in RIVAL_FAMILIES}


def rival_doc(family: str) -> str:
    for name, doc, _members in RIVAL_FAMILIES:
        if name == family:
            return doc
    raise RivalMismatch("no such rival family: %r (known: %s)"
                        % (family, ", ".join(sorted(rivals()))))


def select(family: str, member: str) -> str:
    """Select ONE member of a rival family. Refuses a member of another family.

    This is the only sanctioned way to pick between rivals: it never falls back,
    never merges, and tells you which family the member you named actually
    belongs to.
    """
    table = rivals()
    if family not in table:
        raise RivalMismatch("no such rival family: %r (known: %s)"
                            % (family, ", ".join(sorted(table))))
    if member in table[family]:
        return member
    owner = [f for f, members in table.items() if member in members]
    if owner:
        raise RivalMismatch(
            "%r belongs to the %r family, not %r; rival families are mutually "
            "incompatible and are never blended" % (member, owner[0], family))
    raise RivalMismatch(
        "unknown member %r for family %r; the selectable members are %s"
        % (member, family, ", ".join(table[family])))


# --------------------------------------------------------------------------- #
# Discovery: materialise Route objects from the capability index.
# --------------------------------------------------------------------------- #
_ROUTES: Optional[Dict[str, Route]] = None
_UNADAPTED: Tuple[str, ...] = ()


def _build_routes() -> Dict[str, Route]:
    """Join the route table onto the AST capability index (package='reconstruction').

    The index is the source of truth for *what exists*: a route whose modules are
    not indexed is not offered, however many adapters point at it.
    """
    global _UNADAPTED
    entries = {e.dotted: e
               for e in (capability_registry.find(package=RECONSTRUCTION_PACKAGE)
                         + capability_registry.find(package=INGEST_PACKAGE))}
    adapted: set = set()
    out: Dict[str, Route] = {}
    for name, ins, output, mods, adapter, doc, family in _ROUTE_TABLE:
        dotted = tuple(m if m.startswith("harnesscad.") else _PKG + m for m in mods)
        if any(d not in entries for d in dotted):
            continue                        # module left the tree -> route leaves too
        for kind in ins + (output,):
            if kind not in KINDS:
                raise ValueError("route %r uses unknown kind %r" % (name, kind))
        if family and family not in {f for f, _d, _m in RIVAL_FAMILIES}:
            raise ValueError("route %r names unknown rival family %r" % (name, family))
        if name in out:
            raise ValueError("duplicate route name %r" % name)
        adapted.update(dotted)
        first = entries[dotted[0]]
        out[name] = Route(name=name, inputs=tuple(ins), output=output,
                          dotted=dotted, adapter=adapter, doc=doc, family=family,
                          summary=first.summary, tags=tuple(first.tags))
    _UNADAPTED = tuple(sorted(d for d in entries if d not in adapted))
    return out


def _all() -> Dict[str, Route]:
    global _ROUTES
    if _ROUTES is None:
        _ROUTES = _build_routes()
    return _ROUTES


def routes(input: Optional[str] = None, output: Optional[str] = None,
           family: Optional[str] = None) -> Tuple[Route, ...]:
    """Every discovered route, optionally filtered by input kind / output kind / family."""
    out = [r for r in _all().values()
           if (input is None or input in r.inputs)
           and (output is None or r.output == output)
           and (family is None or r.family == family)]
    return tuple(sorted(out, key=lambda r: r.name))


def route(name: str) -> Route:
    try:
        return _all()[name]
    except KeyError:
        raise UnknownRoute("no such reconstruction route: %r (known: %s)"
                           % (name, ", ".join(sorted(_all())))) from None


def routes_for(input: str, output: Optional[str] = None) -> Tuple[Route, ...]:
    """"What can turn a <input> into a <output>?" -- real, runnable answers."""
    if input not in KINDS:
        raise RouteError("unknown input kind %r (known: %s)"
                         % (input, ", ".join(KINDS)))
    if output is not None and output not in KINDS:
        raise RouteError("unknown output kind %r" % (output,))
    return routes(input=input, output=output)


def inputs() -> Tuple[str, ...]:
    return tuple(sorted({k for r in _all().values() for k in r.inputs}))


def outputs() -> Tuple[str, ...]:
    return tuple(sorted({r.output for r in _all().values()}))


def run(name: str, **payload: Any) -> Any:
    """Run one named route. Raises on a bad name or bad payload -- never guesses."""
    return route(name).call(**payload)


def unadapted() -> Tuple[str, ...]:
    """Reconstruction modules the index knows but no route binds (discovery, not silence)."""
    _all()
    return _UNADAPTED


# --------------------------------------------------------------------------- #
# CLI (wired into core.cli as `harnesscad reconstruct`)
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every discovered reconstruction route")
    parser.add_argument("--from", dest="from_kind", default=None,
                        help="filter by input kind (e.g. point_cloud)")
    parser.add_argument("--to", dest="to_kind", default=None,
                        help="filter by output kind (e.g. primitives)")
    parser.add_argument("--kinds", action="store_true",
                        help="list the input/output kinds routes are keyed by")
    parser.add_argument("--rivals", action="store_true",
                        help="list the rival families that are selected, never blended")
    parser.add_argument("--unadapted", action="store_true",
                        help="list reconstruction modules with no route yet")
    parser.add_argument("--show", default=None, help="show one route by name")
    parser.add_argument("--json", action="store_true", help="emit JSON")


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "rivals", False):
        for family, doc, members in RIVAL_FAMILIES:
            print(f"{family}: (selected by name, NEVER blended)")
            print(f"    {doc}")
            for member in members:
                print(f"    - {member}")
        return 0

    if getattr(args, "kinds", False):
        print("inputs:  " + ", ".join(inputs()))
        print("outputs: " + ", ".join(outputs()))
        return 0

    if getattr(args, "unadapted", False):
        for dotted in unadapted():
            print(dotted)
        print(f"-- {len(unadapted())} reconstruction modules without a route")
        return 0

    if getattr(args, "show", None):
        try:
            r = route(args.show)
        except UnknownRoute as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(r.to_dict(), sort_keys=True, indent=2))
        return 0

    selected = routes(input=getattr(args, "from_kind", None),
                      output=getattr(args, "to_kind", None))
    if getattr(args, "json", False):
        print(json.dumps([r.to_dict() for r in selected], sort_keys=True, indent=2))
        return 0
    for r in selected:
        rival = f"  [RIVAL: {r.family}]" if r.family else ""
        print(f"{r.name:<38} {'+'.join(r.inputs):<18} -> {r.output}{rival}")
        print(f"    {r.doc}")
        print(f"    {', '.join(r.dotted)}")
    print(f"-- {len(selected)} routes / {len(routes())} discovered "
          f"/ {len(unadapted())} reconstruction modules unrouted")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad reconstruct",
        description="reconstruction route registry (input kind -> output kind)")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
