"""Metric registry and suite runner for the bench layer.

The ``eval/bench`` tree carries ~200 metric modules mined from the text-to-CAD
literature. Each one is correct and tested in isolation, but they were never
runnable *together*: their public APIs differ wildly (some take point clouds,
some take meshes, some take token streams; some return a float, some a dataclass,
some a dict), and several of them are **rivals** -- different papers' answers to
the same question, which by design give different numbers on the same input.

This module makes them runnable as a benchmark layer:

*   **Discovery** goes through :mod:`harnesscad.registry` (the static AST index).
    The metric list is derived from ``registry.find(package="bench")``, not from a
    hardcoded inventory: a metric module that disappears from the tree disappears
    from here, and one whose docstring/tags change is re-described automatically.
*   **Adapters** normalise each module's real public API into a uniform
    ``score(pred, gold) -> float | dict``. The metric modules themselves are never
    modified; every adaptation lives here. An adapter declares the *input kinds*
    it needs (``points``, ``mesh``, ``voxels``, ``commands``, ``sketch``, ...), so
    the runner can skip a metric whose inputs a sample does not carry rather than
    fabricate them.
*   **Suites** are named, explicit protocols. This is the load-bearing rule:

        RIVAL METRICS ARE NEVER AVERAGED TOGETHER.

    ``chamfer_unit_sphere`` (centroid + unit-sphere normalisation, squared CD),
    ``chamfer_unit_cube`` (bounding-box normalisation into [-0.5, 0.5]^3, mean CD
    x1000) and ``chamfer_bbox_judged`` (Text-to-CadQuery: centroid + max-extent
    scale, *unhalved* squared CD x1000) give numbers that differ by orders of
    magnitude on the *same* two point clouds -- because the papers defined
    different protocols, not because one is wrong. ``betti_graded`` (fuzzy
    log-ratio score) and ``betti_exact`` (hard equality of the Betti vector)
    disagree by design too. A suite selects ONE member of each rival family;
    :data:`RIVAL_FAMILIES` is enforced at definition time, so a suite that blends
    rivals cannot even be constructed. Every number in a report is stamped with
    the metric name and the module that produced it.
*   **The runner** is deterministic (metrics sorted by name, samples in the given
    order, no clock, no randomness) and fault-tolerant: a metric that raises is
    recorded as an ``error`` entry and the suite carries on.

Typical use::

    from harnesscad.eval.bench import registry as bench

    bench.suites()                       # -> ("cadrille", "deepcad", ...)
    bench.metrics(kind="geometry")       # -> (Metric, ...)
    report = bench.run_suite("deepcad", samples)
    report.to_dict()["aggregates"]

A *sample* is ``{"id": str, "pred": {...}, "gold": {...}}`` where ``pred`` and
``gold`` are payload dicts keyed by input kind (see :data:`INPUT_KINDS`).

Stdlib-only, absolute imports, deterministic.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "KINDS",
    "INPUT_KINDS",
    "Metric",
    "Suite",
    "MetricResult",
    "Report",
    "RIVAL_FAMILIES",
    "metrics",
    "metric",
    "kinds",
    "suites",
    "suite",
    "rivals",
    "unadapted",
    "UNADAPTED_REASONS",
    "reasons",
    "run_metric",
    "run_suite",
    "add_arguments",
    "run",
    "main",
]

BENCH_PACKAGE = "bench"

#: The metric families this layer knows about.
KINDS: Tuple[str, ...] = (
    "geometry", "sequence", "sketch", "vision", "retrieval", "generative",
)

#: The payload keys an adapter may require of a sample's ``pred``/``gold`` dicts.
INPUT_KINDS: Tuple[str, ...] = (
    "points",        # list[(x, y, z)]                 -- sampled surface points
    "points2d",      # list[(x, y)]                    -- sampled sketch points
    "voxels",        # list[(i, j, k)]                 -- sparse occupancy indices
    "mesh",          # {"vertices": [...], "faces": [[i, j, k], ...]}
    "commands",      # list[{"type": str, "params": [float, ...]}]
    "deepcad_rows",  # list[{"type": str, "x": int, "y": int, ...}] (16 slots)
    "op_tokens",     # list[str]                       -- flat op/API token stream
    "tokens",        # list[(int, int)]                -- (px, py) token pairs
    "params",        # {name: value}
    "sketch",        # {"primitives": [...], "constraints": [...]}
    "entities",      # list[(type, *coords)]
    "raster",        # list[list[int]]                 -- binary ink canvas
    "mask",          # list[list[float]]               -- soft silhouette mask
    "depth",         # list[float]
    "latents",       # list[list[float]]
    "ranking",       # list[float]                     -- graded relevance, ranked
    "curvatures",    # list[(grad3, hess3x3)]          -- SDF gradient + Hessian samples
    "slot_rows",     # list[[cmd, a0..a15]]            -- DeepCAD 17-slot int rows
    "op_matrix",     # list[[op_type, params...]]      -- int op matrix
    "code",          # str                             -- CadQuery/Python source
    "cad_sequence",  # {"curves": [...], "extrusion": {...}}
    "sketch_map",    # {sketch_id: [primitive_tuple, ...]}
    "mask_pixels",   # list[(i, j)]                    -- occupied pixel ids
    # -- kinds added with the second adapter wave -------------------------------
    "adjacency",     # {node_id: [neighbour_id, ...]}  -- element adjacency graph
    "labels",        # list[int]                       -- per-element semantic label
    "face_labels",   # {face_id: label}                -- stable-id face labelling
    "cluster_labels",# list[int]                       -- per-latent cluster id
    "instances",     # list[[element_id, ...]]         -- instance masks (id sets)
    "symbol_instances",  # {"lengths": [float], "instances":
                     #   [{"class_id": int, "indices": [int], "score": float}]}
    "bbox",          # [min_x, max_x, min_y, max_y, min_z, max_z]
    "scad",          # str                             -- OpenSCAD source
    "deepcad_commands",  # list[{"type": str, **params}] -- DeepCAD command rows
    "text2cad_model",    # list[{"sketch": [[curve, ...]], "extrusion": {...}}]
    "primitive_tokens",  # list[[t1..t8]]              -- DAVINCI 8-token blocks
    "pose",          # {"R": [[3x3]], "t": [3]}        -- one rigid 6D pose
    "poses",         # list[{"R": [[3x3]], "t": [3]}]  -- a camera trajectory
    "part_names",    # list[str]                       -- retrieved part filenames
    "scored_candidates",  # {"scores": [float], "labels": [0|1]}
    "similarity",    # list[list[float]]               -- pairwise similarity matrix
    "design",        # {"curves": [{"kind": str, "points": [(x, y), ...]}, ...]}
)


# ---------------------------------------------------------------------------
# Metric / Suite / Report types
# ---------------------------------------------------------------------------

Adapter = Callable[[dict, dict], object]


@dataclass(frozen=True)
class Metric:
    """One adapted metric: a uniform ``score(pred, gold)`` over a bench module."""

    name: str                      # unique, stable, e.g. "geometry.chamfer_unit_sphere"
    kind: str                      # one of KINDS
    dotted: str                    # the bench module this adapts
    inputs: Tuple[str, ...]        # payload keys required in BOTH pred and gold
    adapter: Adapter
    summary: str = ""              # the module's docstring first line (from the index)
    tags: Tuple[str, ...] = ()

    def score(self, pred: dict, gold: dict):
        """Run the underlying module. Returns a float or a dict of named numbers."""
        return self.adapter(pred, gold)

    def applicable(self, sample: dict) -> bool:
        pred = sample.get("pred") or {}
        gold = sample.get("gold") or {}
        return all(k in pred and k in gold for k in self.inputs)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "dotted": self.dotted,
            "inputs": list(self.inputs),
            "summary": self.summary,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class Suite:
    """A named evaluation protocol: an explicit, rival-free selection of metrics."""

    name: str
    description: str
    metric_names: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "metrics": list(self.metric_names),
        }


@dataclass(frozen=True)
class MetricResult:
    """One (metric, sample) outcome. ``status`` is ok | error | skipped."""

    metric: str
    kind: str
    dotted: str
    sample_id: str
    status: str
    value: object = None
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "kind": self.kind,
            "dotted": self.dotted,
            "sample_id": self.sample_id,
            "status": self.status,
            "value": self.value,
            "error": self.error,
        }


@dataclass
class Report:
    """The outcome of a suite run: every number stamped with the metric that made it."""

    suite: str
    n_samples: int
    metric_names: Tuple[str, ...]
    results: List[MetricResult] = field(default_factory=list)

    # -- views ---------------------------------------------------------------
    def ok(self) -> List[MetricResult]:
        return [r for r in self.results if r.status == "ok"]

    def errors(self) -> List[MetricResult]:
        return [r for r in self.results if r.status == "error"]

    def skipped(self) -> List[MetricResult]:
        return [r for r in self.results if r.status == "skipped"]

    def by_metric(self, name: str) -> List[MetricResult]:
        return [r for r in self.results if r.metric == name]

    def value(self, metric_name: str, sample_id: str):
        """The value one named metric produced for one sample (None if absent)."""
        for r in self.results:
            if r.metric == metric_name and r.sample_id == sample_id:
                return r.value
        return None

    def aggregates(self) -> Dict[str, Dict[str, float]]:
        """Per-metric means of the numeric fields it produced.

        Aggregation is *within* a metric only -- numbers from different metrics are
        never pooled, because rival metrics measure the same thing on different
        scales (see the module docstring).
        """
        buckets: Dict[str, Dict[str, List[float]]] = {}
        for r in self.ok():
            fields = buckets.setdefault(r.metric, {})
            for key, num in _numeric_fields(r.value):
                fields.setdefault(key, []).append(num)
        out: Dict[str, Dict[str, float]] = {}
        for metric_name in sorted(buckets):
            fields = buckets[metric_name]
            out[metric_name] = {
                key: sum(vals) / len(vals)
                for key, vals in sorted(fields.items()) if vals
            }
        return out

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "n_samples": self.n_samples,
            "metrics": list(self.metric_names),
            "n_ok": len(self.ok()),
            "n_error": len(self.errors()),
            "n_skipped": len(self.skipped()),
            "aggregates": self.aggregates(),
            "results": [r.to_dict() for r in self.results],
        }


def _numeric_fields(value) -> List[Tuple[str, float]]:
    """Flatten a metric's return value into ``(field, number)`` pairs."""
    if isinstance(value, bool):
        return [("value", 1.0 if value else 0.0)]
    if isinstance(value, (int, float)):
        v = float(value)
        return [] if math.isnan(v) else [("value", v)]
    if isinstance(value, dict):
        out: List[Tuple[str, float]] = []
        for key in sorted(value):
            item = value[key]
            if isinstance(item, bool):
                out.append((key, 1.0 if item else 0.0))
            elif isinstance(item, (int, float)) and not math.isnan(float(item)):
                out.append((key, float(item)))
        return out
    return []


# ---------------------------------------------------------------------------
# Adapters.
#
# Each adapter imports its metric module LOCALLY and normalises that module's real
# public API into score(pred, gold). The metric modules are never modified. Local
# imports keep the registry cheap to import and turn an optional-dependency blow-up
# into a recorded error entry rather than an import-time crash -- while the static
# AST index still sees the import edge, so these modules stop being orphans.
# ---------------------------------------------------------------------------

def _mesh(payload: dict) -> Tuple[list, list]:
    m = payload["mesh"]
    return list(m["vertices"]), [tuple(f) for f in m["faces"]]


# -- geometry: the chamfer rivals (DIFFERENT numbers on the same input, by design)

def _chamfer_unit_sphere(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import chamfer_unit_sphere as m
    protocol = m.GeometryProtocol()
    a = m.normalize(pred["points"], protocol)
    b = m.normalize(gold["points"], protocol)
    return float(m.squared_chamfer(a, b))


def _chamfer_unit_cube(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import chamfer_unit_cube as m
    a = m.normalize_to_unit_cube(pred["points"])
    b = m.normalize_to_unit_cube(gold["points"])
    return float(m.chamfer_distance(a, b))


def _chamfer_raw(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import chamfer as m
    return float(m.symmetric_chamfer(pred["points"], gold["points"]))


def _chamfer_bbox_judged(pred: dict, gold: dict):
    from harnesscad.eval.bench.protocols import chamfer_bbox_judged as m
    sample = m.evaluate_sample("sample", pred["points"], gold["points"])
    return {"cd": float(sample.chamfer), "f1": float(sample.f1),
            "iou": float(sample.iou)}


def _chamfer_scaled_step(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import step_file_metrics as m
    return float(m.scaled_chamfer_distance(pred["points"], gold["points"]))


def _chamfer_orientation_aligned(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import orientation_align as m
    return float(m.aligned_chamfer(pred["points"], gold["points"]))


def _accuracy_completeness(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import accuracy_completeness as m
    p, g = pred["points"], gold["points"]
    precision, recall, fscore = m.precision_recall_fscore(p, g, tau=0.05)
    return {
        "accuracy": float(m.accuracy(p, g)),
        "completeness": float(m.completeness(p, g)),
        "chamfer_l1": float(m.chamfer_l1(p, g)),
        "precision": float(precision),
        "recall": float(recall),
        "fscore": float(fscore),
    }


def _edge_chamfer_recon(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import edge_chamfer_recon as m
    report = m.evaluate_reconstruction(gold["points"], pred["points"])
    return {"cd": float(report.cd), "hd": float(report.hd),
            "invalidity_ratio": float(report.ir)}


def _hausdorff_iogt(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import hausdorff_iogt as m
    return {
        "pcd": float(m.point_cloud_distance(pred["points"], gold["points"])),
        "hausdorff": float(m.hausdorff_distance(pred["points"], gold["points"])),
        "iogt": float(m.iogt(pred["points"], gold["points"])),
    }


def _complex_chamfer(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import complex_matching as m
    return float(m.chamfer_distance(pred["points"], gold["points"]))


def _factorization_symmetry(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import factorization_fidelity as m
    return {
        "chamfer": float(m.chamfer_distance(pred["points"], gold["points"])),
        "symmetry_chamfer_x": float(m.symmetry_chamfer(pred["points"], "x")),
    }


def _contact_heatmap(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import contact_heatmap as m
    heat = m.contact_heatmap(pred["points"], bin_size=0.5)
    values = [float(v) for v in heat.values()]
    return {"occupied_bins": float(len(values)),
            "mean_contact": sum(values) / len(values) if values else 0.0,
            "max_contact": max(values) if values else 0.0}


# -- geometry: voxels

def _voxel_iou_from_points(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import voxel_iou as m
    return float(m.voxel_iou(m.voxelize_points(pred["points"]),
                             m.voxelize_points(gold["points"])))


def _solid_voxel_iou(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import voxel_iou as m
    return float(m.voxel_iou([tuple(v) for v in pred["voxels"]],
                             [tuple(v) for v in gold["voxels"]]))


# -- geometry: the betti rivals (graded vs exact -- they disagree by design)

def _betti_graded(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import betti_graded as m
    pv, pf = _mesh(pred)
    gv, gf = _mesh(gold)
    result = m.topo_match(m.MeshSurface(pv, pf), m.MeshSurface(gv, gf))
    out = {"score": float(result.score)}
    out.update({("axis_" + k): float(v) for k, v in result.per_axis_scores.items()})
    return out


def _betti_exact(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import betti_exact as m
    p = [tuple(v) for v in pred["voxels"]]
    g = [tuple(v) for v in gold["voxels"]]
    return {
        "betti_match": float(m.betti_match(p, g)),
        "genus_match": float(m.genus_match(p, g)),
        "component_match": float(m.component_match(p, g)),
        "cavity_match": float(m.cavity_match(p, g)),
    }


# -- geometry: meshes

def _topology_euler(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import topology_euler as m
    pv, pf = _mesh(pred)
    gv, gf = _mesh(gold)
    chi_pred = m.euler_characteristic(pv, pf)
    chi_gt = m.euler_characteristic(gv, gf)
    correct = m.topology_correctness(chi_gt, chi_pred)
    return {"chi_pred": float(chi_pred), "chi_gt": float(chi_gt),
            "topology_correct": float(correct if correct is not None else 0)}


def _mesh_discrepancy(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import mesh_discrepancy as m
    pv, pf = _mesh(pred)
    gv, gf = _mesh(gold)
    cmp_ = m.compare(m.Mesh.of(pv, pf), m.Mesh.of(gv, gf))
    return {
        "pred_watertight": float(cmp_.pred_watertight),
        "gt_watertight": float(cmp_.gt_watertight),
        "euler_match": float(cmp_.euler_match) if cmp_.euler_match is not None else 0.0,
        "sphericity_discrepancy": (float(cmp_.sphericity_discrepancy)
                                   if cmp_.sphericity_discrepancy is not None else 0.0),
    }


def _mesh_quality(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import mesh_quality as m
    pv, pf = _mesh(pred)
    report = m.mesh_quality_report(pv, pf)
    return {k: float(v) for k, v in report.items() if isinstance(v, (int, float))}


def _mesh_topology(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import mesh_topology as m
    pv, pf = _mesh(pred)
    return {"dangling_edge_length": float(m.dangling_edge_length(pv, pf)),
            "self_intersection_ratio": float(
                m.self_intersection_ratio(len(pf), ()))}


def _curvature_developability(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import curvature_developability as m
    # A curvature sample is the module's ``(grad, hess)`` pair: a 3-vector SDF
    # gradient and its 3x3 Hessian. Pass both through untouched.
    to_samples = lambda payload: [
        ([float(v) for v in grad], [[float(v) for v in row] for row in hess])
        for grad, hess in payload["curvatures"]]
    samples = to_samples(pred)
    reference = to_samples(gold)
    return {
        "developability_ratio": float(m.developability_ratio(samples)),
        "mean_abs_gaussian_curvature": float(m.mean_abs_gaussian_curvature(samples)),
        "max_abs_gaussian_curvature": float(m.max_abs_gaussian_curvature(samples)),
        "reference_mean_abs_gaussian_curvature":
            float(m.mean_abs_gaussian_curvature(reference)),
    }


# -- sequence

def _command_f1(pred: dict, gold: dict):
    from harnesscad.eval.bench.sequence import command_f1 as m
    result = m.command_metrics(pred["commands"], gold["commands"])
    return {"macro_f1": float(result["macro_f1"]),
            **{f"{fam}_f1": float(result[fam]["f1"])
               for fam in ("line", "arc", "circle", "extrude")}}


def _reconstruction_accuracy(pred: dict, gold: dict):
    from harnesscad.eval.bench.sequence import reconstruction_accuracy as m
    result = m.reconstruction_accuracy(pred["deepcad_rows"], gold["deepcad_rows"])
    return {k: float(v) for k, v in result.items()}


def _autoencoder_accuracy(pred: dict, gold: dict):
    from harnesscad.eval.bench.sequence import autoencoder_accuracy as m
    result = m.evaluate_model(pred["slot_rows"], gold["slot_rows"])
    return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}


def _sequence_edit_distance(pred: dict, gold: dict):
    from harnesscad.eval.bench.sequence import sequence_edit_distance as m
    result = m.sequence_edit_distance(pred["op_tokens"], gold["op_tokens"])
    return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}


def _multilevel_sequence(pred: dict, gold: dict):
    from harnesscad.eval.bench.sequence import multilevel_sequence_eval as m
    preds, gts = [list(pred["op_matrix"])], [list(gold["op_matrix"])]
    return {
        "accuracy_seq_op_types": float(m.accuracy_seq_op_types(preds, gts)),
        "edit_distance_seq_op_types": float(m.edit_distance_seq_op_types(preds, gts)),
        "accuracy_op_types": float(m.accuracy_op_types(preds, gts)),
    }


def _token_accuracy(pred: dict, gold: dict):
    from harnesscad.eval.bench.sequence import token_accuracy as m
    result = m.token_accuracy(pred["tokens"], gold["tokens"])
    return {"correct": float(result.correct), "total": float(result.total),
            "accuracy": float(result.accuracy)}


def _parameter_accuracy(pred: dict, gold: dict):
    from harnesscad.eval.bench.sequence import parameter_accuracy as m
    result = m.parameter_accuracy(pred["params"], gold["params"])
    return {"matched": float(result["matched"]), "total": float(result["total"]),
            "accuracy": float(result["accuracy"])}


def _sequence_length_stats(pred: dict, gold: dict):
    from harnesscad.eval.bench.sequence import sequence_length_stats as m
    return {"pred_length": float(m.effective_length(pred["op_tokens"], eos_token=None)),
            "gold_length": float(m.effective_length(gold["op_tokens"], eos_token=None))}


def _sequence_invalidity(pred: dict, gold: dict):
    from harnesscad.eval.bench.sequence import invalidity_ratio as m
    return float(m.invalidity_ratio([pred["cad_sequence"]]))


def _pass_at_k(pred: dict, gold: dict):
    from harnesscad.eval.bench.sequence import pass_at_k as m
    from harnesscad.eval.bench.sequence import sequence_edit_distance as sed
    exact = sed.sequence_edit_distance(pred["op_tokens"], gold["op_tokens"])["distance"] == 0
    return float(m.estimate_pass_at_k(1, 1 if exact else 0, 1))


def _code_ast_metrics(pred: dict, gold: dict):
    from harnesscad.eval.bench.sequence import code_ast_metrics as m
    fn = m.function_accuracy(gold["code"], pred["code"])
    par = m.parameter_accuracy(gold["code"], pred["code"])
    out = {"parsing_rate": float(m.parsing_rate([pred["code"]]))}
    out.update({("function_" + k): float(v) for k, v in fn.items()
                if isinstance(v, (int, float, bool))})
    out.update({("parameter_" + k): float(v) for k, v in par.items()
                if isinstance(v, (int, float, bool))})
    return out


# -- sketch

def _sketch_f1(pred: dict, gold: dict):
    from harnesscad.eval.bench.sketch import sketch_f1 as m
    p, g = pred["sketch"], gold["sketch"]
    result = m.sketch_f1(p.get("primitives", ()), g.get("primitives", ()),
                         p.get("constraints", ()), g.get("constraints", ()))
    return {"primitive_f1": float(result["primitive"]["f1"]),
            "primitive_precision": float(result["primitive"]["precision"]),
            "primitive_recall": float(result["primitive"]["recall"]),
            "constraint_f1": float(result["constraint"]["f1"])}


def _entity_sketch_f1(pred: dict, gold: dict):
    from harnesscad.eval.bench.sketch import entity_sketch_f1 as m
    result = m.cadvlm_metrics([tuple(tuple(e) for e in pred["entities"])],
                              [tuple(tuple(e) for e in gold["entities"])])
    return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}


def _sketch_chamfer_2d(pred: dict, gold: dict):
    from harnesscad.eval.bench.sketch import sketch_chamfer_2d as m
    return float(m.chamfer_2d(pred["points2d"], gold["points2d"]))


def _raster_vectorization(pred: dict, gold: dict):
    from harnesscad.eval.bench.sketch import raster_vectorization as m
    prf = m.raster_precision_recall_f1(pred["raster"], gold["raster"])
    return {"raster_iou": float(m.raster_iou(pred["raster"], gold["raster"])),
            "precision": float(prf.precision), "recall": float(prf.recall),
            "f1": float(prf.f1)}


def _image_symmetry(pred: dict, gold: dict):
    from harnesscad.eval.bench.sketch import image_symmetry as m
    return {"pred_symmetry": float(m.symmetry_score(pred["raster"])),
            "gold_symmetry": float(m.symmetry_score(gold["raster"]))}


def _sketch_sequence_metrics(pred: dict, gold: dict):
    from harnesscad.eval.bench.sketch import sketch_sequence_metrics as m
    to_map = lambda payload: {
        key: tuple(tuple(item) for item in items)
        for key, items in payload["sketch_map"].items()}
    result = m.metrics(to_map(pred), to_map(gold))
    return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}


def _loop_curve_score(pred: dict, gold: dict):
    from harnesscad.eval.bench.sketch import loop_curve_score as m
    result = m.evaluate([pred["commands"]], [gold["commands"]], executable=True)
    if isinstance(result, dict):
        return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
    return float(result)


# -- vision

def _silhouette_iou(pred: dict, gold: dict):
    from harnesscad.eval.bench.vision import silhouette_iou as m
    return {"soft_iou": float(m.soft_iou(pred["mask"], gold["mask"])),
            "iou_loss": float(m.iou_loss(pred["mask"], gold["mask"]))}


def _depth_metrics(pred: dict, gold: dict):
    from harnesscad.eval.bench.vision import depth_metrics as m
    result = m.evaluate_depth(pred["depth"], gold["depth"])
    if not isinstance(result, dict):
        result = getattr(result, "__dict__", {})
    return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}


def _vision_mask_iou(pred: dict, gold: dict):
    from harnesscad.eval.bench.vision import vision_metrics as m
    return float(m.mask_iou([tuple(p) for p in gold["mask_pixels"]],
                            [tuple(p) for p in pred["mask_pixels"]]))


def _multiview_consistency(pred: dict, gold: dict):
    from harnesscad.eval.bench.vision import multiview_consistency_anova as m
    return float(m.consistency_score(pred["depth"]))


# -- retrieval

def _ranked_retrieval(pred: dict, gold: dict):
    from harnesscad.eval.bench.retrieval import ranked_retrieval_metrics as m
    gains = list(pred["ranking"])
    return {"ndcg_at_5": float(m.ndcg_at_k(gains, 5)),
            "reciprocal_rank": float(m.reciprocal_rank(gains)),
            "success_at_5": float(m.success_at_k(gains, 5))}


def _tiered_retrieval(pred: dict, gold: dict):
    from harnesscad.eval.bench.retrieval import tiered_retrieval_metrics as m
    relevances = list(pred["ranking"])
    n_relevant = sum(1 for g in gold["ranking"] if g > 0) or 1
    return {"nearest_neighbour": float(m.nearest_neighbour(relevances)),
            "first_tier": float(m.first_tier(relevances, n_relevant)),
            "second_tier": float(m.second_tier(relevances, n_relevant)),
            "anmrr": float(m.anmrr(relevances, n_relevant))}


def _representation_quality(pred: dict, gold: dict):
    from harnesscad.eval.bench.retrieval import representation_quality as m
    pairs = list(zip(pred["latents"], gold["latents"]))
    return {"alignment": float(m.alignment(pairs)),
            "uniformity": float(m.uniformity(pred["latents"]))}


def _latent_cluster_quality(pred: dict, gold: dict):
    from harnesscad.eval.bench.retrieval import latent_cluster_quality as m
    points = list(pred["latents"])
    labels = [i % 2 for i in range(len(points))]
    return {"silhouette": float(m.silhouette_coefficient(points, labels)),
            "sse": float(m.sse(points, labels))}


def _embedding_cosine(pred: dict, gold: dict):
    from harnesscad.eval.bench.retrieval import embedding_postprocess as m
    sims = [float(m.cosine_similarity(p, g))
            for p, g in zip(pred["latents"], gold["latents"])]
    return sum(sims) / len(sims) if sims else 0.0


# -- generative

def _fid(pred: dict, gold: dict):
    from harnesscad.eval.bench.generative import fid as m
    return float(m.fid_score(gold["latents"], pred["latents"]))


def _one_nna(pred: dict, gold: dict):
    from harnesscad.eval.bench.generative import one_nna as m
    return float(m.one_nna(list(pred["latents"]), list(gold["latents"]),
                           distance=math.dist))


def _diversity_feature_space(pred: dict, gold: dict):
    from harnesscad.eval.bench.generative import diversity_feature_space as m
    return {"pairwise_diversity": float(m.pairwise_diversity(pred["latents"]))}


# ---------------------------------------------------------------------------
# Second adapter wave. Same rules: the modules are untouched, every adaptation
# lives here, and an adapter declares only inputs a sample can honestly carry.
# ---------------------------------------------------------------------------

def _numbers(payload: dict) -> Dict[str, float]:
    """Every scalar in a metric's dict/dataclass return, as floats."""
    if not isinstance(payload, dict):
        payload = getattr(payload, "__dict__", {})
    return {k: float(v) for k, v in payload.items()
            if isinstance(v, (int, float, bool))}


def _adjacency(payload: dict) -> Dict[int, set]:
    return {int(node): {int(n) for n in neighbours}
            for node, neighbours in payload["adjacency"].items()}


# -- geometry (wave 2)

def _design(payload: dict):
    from harnesscad.domain.editing import sketch_edit_schema as schema
    curves = tuple(
        schema.Curve(kind=str(c["kind"]),
                     points=tuple((float(p[0]), float(p[1])) for p in c["points"]))
        for c in payload["design"]["curves"])
    return schema.Design(curves=curves)


def _chamfer_refinement_2d(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import refinement_convergence as m
    return float(m.chamfer_symmetric(_design(pred), _design(gold)))


def _boundary_fscore(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import boundary_fscore as m
    adjacency = _adjacency(pred)
    precision, recall, f1 = m.boundary_prf(adjacency, list(pred["labels"]),
                                           list(gold["labels"]))
    return {"boundary_precision": float(precision),
            "boundary_recall": float(recall),
            "boundary_f1": float(f1)}


def _dimension_accuracy(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import dimension_accuracy as m
    expected = m.measure_bbox_dimensions(*[float(v) for v in gold["bbox"]])
    result = m.measure_and_score(tuple(float(v) for v in pred["bbox"]), expected)
    out = {"average_accuracy": float(result.average_accuracy),
           "all_within_tolerance": float(result.all_within_tolerance)}
    out.update({("accuracy_" + name): float(cmp_.accuracy)
                for name, cmp_ in result.comparisons.items()})
    return out


def _program_shape_match(pred: dict, gold: dict):
    from harnesscad.eval.bench.geometry import program_shape_match as m
    report = m.score(pred["scad"], gold["scad"])
    return {"compiles": float(report.compiles),
            "voxel_iou": float(report.voxel_iou),
            "volume_ratio": float(report.volume_ratio),
            "bbox_iou": float(report.bbox_iou),
            "centroid_offset": float(report.centroid_offset)}


# -- sequence (wave 2)

def _deepcad_commands(payload: dict) -> list:
    from harnesscad.domain.reconstruction.tokens import deepcad_commands as dc
    out = []
    for row in payload["deepcad_commands"]:
        params = {k: v for k, v in row.items() if k != "type"}
        out.append(dc.command(row["type"], **params))
    return out


def _sequence_f1(pred: dict, gold: dict):
    from harnesscad.eval.bench.sequence import sequence_f1 as m
    evaluation = m.evaluate_sequence(_deepcad_commands(pred),
                                     _deepcad_commands(gold))
    return {k: float(v) for k, v in m.aggregate_f1([evaluation]).items()}


def _code_validity(pred: dict, gold: dict):
    from harnesscad.eval.bench.sequence import code_validity as m
    result = m.validate_cad_code(pred["code"])
    rate = m.valid_syntax_rate([result])
    return {"valid": float(bool(result["valid"])),
            "valid_syntax_rate": float(rate if rate is not None else 0.0)}


def _primitive_f1_null_class(pred: dict, gold: dict):
    from harnesscad.eval.bench.protocols import primitive_f1_null_class as m
    report = m.evaluate_model(list(gold["text2cad_model"]),
                              list(pred["text2cad_model"]))
    out = {"type_accuracy": float(report.accuracy),
           "macro_f1": float(report.macro["f1"]),
           "micro_f1": float(report.micro["f1"]),
           "extrusion_f1": float(report.extrusion.f1)}
    out.update({(name + "_f1"): float(score.f1)
                for name, score in report.curves.items()})
    return out


# -- sketch (wave 2)

def _set_prediction_f1(pred: dict, gold: dict):
    from harnesscad.eval.bench.sketch import set_prediction_f1 as m
    result = m.evaluate([list(t) for t in pred["primitive_tokens"]],
                        [list(t) for t in gold["primitive_tokens"]])
    out: Dict[str, float] = {}
    for key, item in result.items():
        if isinstance(item, (int, float, bool)):
            out[key] = float(item)
        elif isinstance(item, dict):
            out.update({f"{key}_{k}": float(v) for k, v in item.items()
                        if isinstance(v, (int, float, bool))})
    return out


# -- vision (wave 2)

def _face_segmentation(pred: dict, gold: dict):
    from harnesscad.eval.bench.vision import face_segmentation as m
    result = m.face_segmentation_metrics(dict(gold["face_labels"]),
                                         dict(pred["face_labels"]))
    if not result.get("available"):
        raise ValueError(f"face ids differ: {result.get('error')}")
    return {"accuracy": float(result["accuracy"]),
            "macro_iou": float(result["macro_iou"])}


def _pointwise_semantic(pred: dict, gold: dict):
    from harnesscad.eval.bench.vision import pointwise_semantic_eval as m
    p = [int(v) for v in pred["labels"]]
    g = [int(v) for v in gold["labels"]]
    n_classes = max(p + g) + 1
    result = m.point_wise_eval(p, g, num_classes=n_classes,
                               ignore_label=n_classes)
    return {k: float(v) for k, v in result.items()
            if isinstance(v, (int, float))}


def _instance_segmentation(pred: dict, gold: dict):
    from harnesscad.eval.bench.vision import instance_segmentation as m
    result = m.instance_metrics([set(inst) for inst in pred["instances"]],
                                [set(inst) for inst in gold["instances"]])
    return {k: float(v) for k, v in result.items()}


def _weighted_instances(payload: dict) -> list:
    lengths = [float(v) for v in payload["symbol_instances"]["lengths"]]
    return [(int(inst["class_id"]),
             {int(i): lengths[int(i)] for i in inst["indices"]})
            for inst in payload["symbol_instances"]["instances"]]


def _length_weighted_panoptic(pred: dict, gold: dict):
    from harnesscad.eval.bench.vision import length_weighted_panoptic as m
    p, g = _weighted_instances(pred), _weighted_instances(gold)
    out = {k: float(v) for k, v in m.panoptic_quality(p, g).items()}
    micro = m.per_class_f1(p, g).get("micro", {})
    out.update({("micro_" + k): float(v) for k, v in micro.items()
                if isinstance(v, (int, float))})
    return out


def _point_weighted_panoptic(pred: dict, gold: dict):
    from harnesscad.eval.bench.vision import point_weighted_panoptic as m
    lengths = [float(v) for v in pred["symbol_instances"]["lengths"]]
    predictions = [(int(i["class_id"]), float(i.get("score", 1.0)),
                    [int(x) for x in i["indices"]])
                   for i in pred["symbol_instances"]["instances"]]
    truth = [(int(i["class_id"]), [int(x) for x in i["indices"]])
             for i in gold["symbol_instances"]["instances"]]
    classes = [c for c, _, _ in predictions] + [c for c, _ in truth]
    n_classes = max(classes) + 1 if classes else 1
    report = m.evaluate(predictions, truth, lengths, num_classes=n_classes,
                        thing_classes=tuple(range(n_classes)), stuff_classes=())
    overall = report.get("all", report)
    return {k: float(v) for k, v in overall.items()
            if isinstance(v, (int, float))}


def _object_pose_add(pred: dict, gold: dict):
    from harnesscad.eval.bench.vision import object_pose_add as m
    r_p, t_p = pred["pose"]["R"], pred["pose"]["t"]
    r_g, t_g = gold["pose"]["R"], gold["pose"]["t"]
    model_points = [tuple(float(c) for c in p) for p in gold["points"]]
    return {
        "add": float(m.add(r_p, t_p, r_g, t_g, model_points)),
        "add_s": float(m.add_s(r_p, t_p, r_g, t_g, model_points)),
        "rotation_error_deg": float(m.rotation_angle_error_deg(r_p, r_g)),
        "translation_error": float(m.translation_error(t_p, t_g)),
        "accuracy_5cm_5deg": float(m.pose_accuracy_5cm_5deg(r_p, t_p, r_g, t_g)),
    }


def _camera_pose_trajectory(pred: dict, gold: dict):
    from harnesscad.eval.bench.vision import camera_pose_trajectory as m
    to_poses = lambda payload: [(p["R"], p["t"]) for p in payload["poses"]]
    return {k: float(v)
            for k, v in m.evaluate_trajectory(to_poses(pred),
                                              to_poses(gold)).items()
            if isinstance(v, (int, float))}


# -- retrieval (wave 2)

def _clustering_external(pred: dict, gold: dict):
    from harnesscad.eval.bench.retrieval import clustering_external_indices as m
    p = [int(v) for v in pred["cluster_labels"]]
    g = [int(v) for v in gold["cluster_labels"]]
    return {"nmi": float(m.normalized_mutual_information(p, g)),
            "adjusted_rand_index": float(m.adjusted_rand_index(p, g)),
            "rand_index": float(m.rand_index(p, g)),
            "clustering_accuracy": float(m.clustering_accuracy(p, g)),
            "purity": float(m.purity(p, g))}


def _clustering_internal(pred: dict, gold: dict):
    from harnesscad.eval.bench.retrieval import clustering_internal_indices as m
    points = [list(v) for v in pred["latents"]]
    labels = [int(v) for v in pred["cluster_labels"]]
    return {"davies_bouldin": float(m.davies_bouldin_index(points, labels)),
            "calinski_harabasz": float(m.calinski_harabasz_index(points, labels)),
            "dunn": float(m.dunn_index(points, labels))}


def _graded_retrieval(pred: dict, gold: dict):
    from harnesscad.eval.bench.retrieval import graded_retrieval_eval as m
    # Paired protocol: pred latent i is the query, gold latent i is its ONE
    # relevant database entry (the sample's own pairing -- no relevance labels
    # are invented).
    queries = [list(v) for v in pred["latents"]]
    database = [list(v) for v in gold["latents"]]
    relevant = [[i] for i in range(len(queries))]
    report = m.evaluate_retrieval(queries, database, relevant, ks=(1, 5))
    out = {f"recall_at_{k}": float(v) for k, v in report.recall.items()}
    out.update({f"ndcg_at_{k}": float(v) for k, v in report.ndcg.items()})
    return out


def _gallery_retrieval(pred: dict, gold: dict):
    from harnesscad.eval.bench.retrieval import gallery_retrieval_eval as m
    report = m.evaluate_retrieval([list(v) for v in pred["latents"]],
                                  [int(c) for c in pred["cluster_labels"]],
                                  [list(v) for v in gold["latents"]],
                                  [int(c) for c in gold["cluster_labels"]])
    return {"nn_accuracy": float(report.nn_accuracy),
            "nn_f1": float(report.nn_f1),
            "ndcg": float(report.ndcg),
            "macro_map": float(report.macro_map),
            "micro_map": float(report.micro_map)}


def _image_retrieval_accuracy(pred: dict, gold: dict):
    from harnesscad.eval.bench.retrieval import image_retrieval_accuracy as m
    image = [list(v) for v in pred["latents"]]
    cad = [list(v) for v in gold["latents"]]
    batch = min(len(image), len(cad))
    if batch < 2:
        raise ValueError("batch retrieval needs at least two paired latents")
    mean, std = m.retrieval_accuracy(image, cad, batch_size=batch,
                                     repeats=4, seed=0)
    return {"retrieval_accuracy": float(mean), "retrieval_std": float(std),
            "random_guess": float(m.random_guess_accuracy(batch))}


def _latent_alignment(pred: dict, gold: dict):
    from harnesscad.eval.bench.retrieval import latent_alignment as m
    quality = m.alignment_quality([list(v) for v in pred["latents"]],
                                  [list(v) for v in gold["latents"]])
    return {"mean_paired_cosine": float(quality.mean_paired_cosine),
            "mean_cross_cosine": float(quality.mean_cross_cosine),
            "top1_accuracy": float(quality.top1_accuracy),
            "mean_reciprocal_rank": float(quality.mean_reciprocal_rank)}


def _part_retrieval(pred: dict, gold: dict):
    from harnesscad.eval.bench.retrieval import part_retrieval_eval as m
    p = [str(n) for n in pred["part_names"]]
    g = [str(n) for n in gold["part_names"]]
    out = {k: float(v) for k, v in m.relevance(p, g).items()}
    out["exact_match"] = float(m.exact_match(p, g))
    return out


def _joint_prediction_ranking(pred: dict, gold: dict):
    from harnesscad.eval.bench.retrieval import joint_prediction_ranking as m
    scores = [float(s) for s in pred["scored_candidates"]["scores"]]
    labels = [int(v) for v in gold["scored_candidates"]["labels"]]
    rank = m.rank_of_first_hit(scores, labels)
    return {"hit_at_1": float(m.hit_at_top_k(scores, labels, 1)),
            "hit_at_5": float(m.hit_at_top_k(scores, labels, 5)),
            "reciprocal_rank": float(1.0 / rank) if rank else 0.0}


# -- generative (wave 2)

def _diversity_similarity_matrix(pred: dict, gold: dict):
    from harnesscad.eval.bench.generative import diversity_similarity_matrix as m
    report = m.conceptset_report([[float(v) for v in row]
                                  for row in pred["similarity"]])
    return {k: float(v) for k, v in report.items()
            if isinstance(v, (int, float, bool))}


# ---------------------------------------------------------------------------
# The adapter table: metric name -> (module dotted, kind, inputs, adapter).
#
# It binds adapters to MODULES; the metric objects themselves are materialised
# from the capability index (see _build_metrics), so what exists in the tree is
# what exists here.
# ---------------------------------------------------------------------------

_P = "harnesscad.eval.bench."

_ADAPTER_TABLE: Tuple[Tuple[str, str, str, Tuple[str, ...], Adapter], ...] = (
    # name, module dotted, kind, inputs, adapter
    ("geometry.chamfer_unit_sphere", _P + "geometry.chamfer_unit_sphere", "geometry",
     ("points",), _chamfer_unit_sphere),
    ("geometry.chamfer_unit_cube", _P + "geometry.chamfer_unit_cube", "geometry",
     ("points",), _chamfer_unit_cube),
    ("geometry.chamfer_raw", _P + "geometry.chamfer", "geometry",
     ("points",), _chamfer_raw),
    ("geometry.chamfer_bbox_judged", _P + "protocols.chamfer_bbox_judged", "geometry",
     ("points",), _chamfer_bbox_judged),
    ("geometry.chamfer_scaled_step", _P + "geometry.step_file_metrics", "geometry",
     ("points",), _chamfer_scaled_step),
    ("geometry.chamfer_orientation_aligned", _P + "geometry.orientation_align", "geometry",
     ("points",), _chamfer_orientation_aligned),
    ("geometry.chamfer_complex", _P + "geometry.complex_matching", "geometry",
     ("points",), _complex_chamfer),
    ("geometry.accuracy_completeness", _P + "geometry.accuracy_completeness", "geometry",
     ("points",), _accuracy_completeness),
    ("geometry.edge_chamfer_recon", _P + "geometry.edge_chamfer_recon", "geometry",
     ("points",), _edge_chamfer_recon),
    ("geometry.hausdorff_iogt", _P + "geometry.hausdorff_iogt", "geometry",
     ("points",), _hausdorff_iogt),
    ("geometry.factorization_fidelity", _P + "geometry.factorization_fidelity", "geometry",
     ("points",), _factorization_symmetry),
    ("geometry.contact_heatmap", _P + "geometry.contact_heatmap", "geometry",
     ("points",), _contact_heatmap),
    ("geometry.voxel_iou_points", _P + "geometry.voxel_iou", "geometry",
     ("points",), _voxel_iou_from_points),
    ("geometry.voxel_iou_grid", _P + "geometry.voxel_iou", "geometry",
     ("voxels",), _solid_voxel_iou),
    ("geometry.betti_graded", _P + "geometry.betti_graded", "geometry",
     ("mesh",), _betti_graded),
    ("geometry.betti_exact", _P + "geometry.betti_exact", "geometry",
     ("voxels",), _betti_exact),
    ("geometry.topology_euler", _P + "geometry.topology_euler", "geometry",
     ("mesh",), _topology_euler),
    ("geometry.mesh_discrepancy", _P + "geometry.mesh_discrepancy", "geometry",
     ("mesh",), _mesh_discrepancy),
    ("geometry.mesh_quality", _P + "geometry.mesh_quality", "geometry",
     ("mesh",), _mesh_quality),
    ("geometry.mesh_topology", _P + "geometry.mesh_topology", "geometry",
     ("mesh",), _mesh_topology),
    ("geometry.curvature_developability", _P + "geometry.curvature_developability", "geometry",
     ("curvatures",), _curvature_developability),

    ("sequence.command_f1", _P + "sequence.command_f1", "sequence",
     ("commands",), _command_f1),
    ("sequence.reconstruction_accuracy", _P + "sequence.reconstruction_accuracy", "sequence",
     ("deepcad_rows",), _reconstruction_accuracy),
    ("sequence.autoencoder_accuracy", _P + "sequence.autoencoder_accuracy", "sequence",
     ("slot_rows",), _autoencoder_accuracy),
    ("sequence.edit_distance", _P + "sequence.sequence_edit_distance", "sequence",
     ("op_tokens",), _sequence_edit_distance),
    ("sequence.multilevel", _P + "sequence.multilevel_sequence_eval", "sequence",
     ("op_matrix",), _multilevel_sequence),
    ("sequence.token_accuracy", _P + "sequence.token_accuracy", "sequence",
     ("tokens",), _token_accuracy),
    ("sequence.parameter_accuracy", _P + "sequence.parameter_accuracy", "sequence",
     ("params",), _parameter_accuracy),
    ("sequence.length_stats", _P + "sequence.sequence_length_stats", "sequence",
     ("op_tokens",), _sequence_length_stats),
    ("sequence.invalidity_ratio", _P + "sequence.invalidity_ratio", "sequence",
     ("cad_sequence",), _sequence_invalidity),
    ("sequence.pass_at_k", _P + "sequence.pass_at_k", "sequence",
     ("op_tokens",), _pass_at_k),
    ("sequence.code_ast_metrics", _P + "sequence.code_ast_metrics", "sequence",
     ("code",), _code_ast_metrics),

    ("sketch.sketch_f1", _P + "sketch.sketch_f1", "sketch",
     ("sketch",), _sketch_f1),
    ("sketch.entity_sketch_f1", _P + "sketch.entity_sketch_f1", "sketch",
     ("entities",), _entity_sketch_f1),
    ("sketch.chamfer_2d", _P + "sketch.sketch_chamfer_2d", "sketch",
     ("points2d",), _sketch_chamfer_2d),
    ("sketch.raster_vectorization", _P + "sketch.raster_vectorization", "sketch",
     ("raster",), _raster_vectorization),
    ("sketch.image_symmetry", _P + "sketch.image_symmetry", "sketch",
     ("raster",), _image_symmetry),
    ("sketch.sequence_metrics", _P + "sketch.sketch_sequence_metrics", "sketch",
     ("sketch_map",), _sketch_sequence_metrics),
    ("sketch.loop_curve_score", _P + "sketch.loop_curve_score", "sketch",
     ("commands",), _loop_curve_score),

    ("vision.silhouette_iou", _P + "vision.silhouette_iou", "vision",
     ("mask",), _silhouette_iou),
    ("vision.depth_metrics", _P + "vision.depth_metrics", "vision",
     ("depth",), _depth_metrics),
    ("vision.mask_iou", _P + "vision.vision_metrics", "vision",
     ("mask_pixels",), _vision_mask_iou),
    ("vision.multiview_consistency", _P + "vision.multiview_consistency_anova", "vision",
     ("depth",), _multiview_consistency),

    ("retrieval.ranked", _P + "retrieval.ranked_retrieval_metrics", "retrieval",
     ("ranking",), _ranked_retrieval),
    ("retrieval.tiered", _P + "retrieval.tiered_retrieval_metrics", "retrieval",
     ("ranking",), _tiered_retrieval),
    ("retrieval.representation_quality", _P + "retrieval.representation_quality", "retrieval",
     ("latents",), _representation_quality),
    ("retrieval.latent_cluster_quality", _P + "retrieval.latent_cluster_quality", "retrieval",
     ("latents",), _latent_cluster_quality),
    ("retrieval.embedding_cosine", _P + "retrieval.embedding_postprocess", "retrieval",
     ("latents",), _embedding_cosine),

    ("generative.fid", _P + "generative.fid", "generative",
     ("latents",), _fid),
    ("generative.one_nna", _P + "generative.one_nna", "generative",
     ("latents",), _one_nna),
    ("generative.diversity", _P + "generative.diversity_feature_space", "generative",
     ("latents",), _diversity_feature_space),

    # -- second adapter wave ---------------------------------------------------
    ("geometry.chamfer_refinement_2d", _P + "geometry.refinement_convergence",
     "geometry", ("design",), _chamfer_refinement_2d),
    ("geometry.boundary_fscore", _P + "geometry.boundary_fscore", "geometry",
     ("adjacency", "labels"), _boundary_fscore),
    ("geometry.dimension_accuracy", _P + "geometry.dimension_accuracy", "geometry",
     ("bbox",), _dimension_accuracy),
    ("geometry.program_shape_match", _P + "geometry.program_shape_match", "geometry",
     ("scad",), _program_shape_match),

    ("sequence.sequence_f1", _P + "sequence.sequence_f1", "sequence",
     ("deepcad_commands",), _sequence_f1),
    ("sequence.code_validity", _P + "sequence.code_validity", "sequence",
     ("code",), _code_validity),
    ("sequence.primitive_f1_null_class", _P + "protocols.primitive_f1_null_class",
     "sequence", ("text2cad_model",), _primitive_f1_null_class),

    ("sketch.set_prediction_f1", _P + "sketch.set_prediction_f1", "sketch",
     ("primitive_tokens",), _set_prediction_f1),

    ("vision.face_segmentation", _P + "vision.face_segmentation", "vision",
     ("face_labels",), _face_segmentation),
    ("vision.pointwise_semantic", _P + "vision.pointwise_semantic_eval", "vision",
     ("labels",), _pointwise_semantic),
    ("vision.instance_segmentation", _P + "vision.instance_segmentation", "vision",
     ("instances",), _instance_segmentation),
    ("vision.length_weighted_panoptic", _P + "vision.length_weighted_panoptic",
     "vision", ("symbol_instances",), _length_weighted_panoptic),
    ("vision.point_weighted_panoptic", _P + "vision.point_weighted_panoptic",
     "vision", ("symbol_instances",), _point_weighted_panoptic),
    ("vision.object_pose_add", _P + "vision.object_pose_add", "vision",
     ("pose", "points"), _object_pose_add),
    ("vision.camera_pose_trajectory", _P + "vision.camera_pose_trajectory", "vision",
     ("poses",), _camera_pose_trajectory),

    ("retrieval.clustering_external", _P + "retrieval.clustering_external_indices",
     "retrieval", ("cluster_labels",), _clustering_external),
    ("retrieval.clustering_internal", _P + "retrieval.clustering_internal_indices",
     "retrieval", ("latents", "cluster_labels"), _clustering_internal),
    ("retrieval.graded_retrieval", _P + "retrieval.graded_retrieval_eval",
     "retrieval", ("latents",), _graded_retrieval),
    ("retrieval.gallery_retrieval", _P + "retrieval.gallery_retrieval_eval",
     "retrieval", ("latents", "cluster_labels"), _gallery_retrieval),
    ("retrieval.image_retrieval_accuracy", _P + "retrieval.image_retrieval_accuracy",
     "retrieval", ("latents",), _image_retrieval_accuracy),
    ("retrieval.latent_alignment", _P + "retrieval.latent_alignment", "retrieval",
     ("latents",), _latent_alignment),
    ("retrieval.part_retrieval", _P + "retrieval.part_retrieval_eval", "retrieval",
     ("part_names",), _part_retrieval),
    ("retrieval.joint_prediction_ranking", _P + "retrieval.joint_prediction_ranking",
     "retrieval", ("scored_candidates",), _joint_prediction_ranking),

    ("generative.diversity_similarity_matrix",
     _P + "generative.diversity_similarity_matrix", "generative",
     ("similarity",), _diversity_similarity_matrix),
)


#: Bench modules deliberately left unadapted, with the reason. A module listed
#: here needs an input a per-sample ``pred``/``gold`` pair cannot honestly supply
#: (a corpus, an execution trace, a human rating, an injected CAD kernel), or its
#: semantics cannot be read off its public API. Faking such an input would produce
#: a number that means nothing, so these stay in :func:`unadapted`.
UNADAPTED_REASONS: Tuple[Tuple[str, str], ...] = (
    (_P + "geometry.compositional_metrics",
     "aggregates externally-computed CD/IV/PR values; takes no pred/gold input"),
    (_P + "geometry.interface_match",
     "needs STEP mating sub-volumes plus an injected IoU-at-pose kernel"),
    (_P + "geometry.solid_iou",
     "needs an injected solid-modelling adapter (inertia frames, symmetry search)"),
    (_P + "geometry.primitive_fidelity",
     "no docstring; the meaning of its five positional arguments is not recoverable"),
    (_P + "geometry.edit_boundary_coherence",
     "needs an edit/keep voxel-mask partition and a source shape, not a pred/gold pair"),
    (_P + "geometry.design_distance_curve",
     "needs mrCAD Design objects (stroke geometry), which the sample schema lacks"),
    (_P + "sequence.error_taxonomy",
     "classifies a FreeCAD execution stderr string; requires running the code"),
    (_P + "sequence.controllability",
     "needs a FlexCAD mask target plus the pre-edit model; a pred/gold pair has neither"),
    (_P + "retrieval.nt_xent_loss",
     "a training objective over dropout views, not a prediction-vs-ground-truth metric"),
    (_P + "retrieval.graph_nt_xent_loss",
     "a training objective over augmented graph views; needs an embed_fn, not a sample"),
    (_P + "retrieval.metric_learning_losses",
     "VICReg/MMCL training losses over paired augmentations, not an evaluation metric"),
    (_P + "retrieval.clustering_algorithms",
     "clustering inference algorithms (k-means/spectral), not an evaluation metric"),
    (_P + "retrieval.openset_recognition",
     "needs a set of KNOWN and a set of UNKNOWN query scores; a sample carries neither"),
    (_P + "retrieval.fewshot_scaling",
     "a dataset-level scaling protocol over train/test splits and repeats"),
    (_P + "retrieval.grounding_metrics",
     "needs the model's SELECTED id set alongside the ranking; the schema has no such field"),
    (_P + "generative.brep_set_metrics",
     "set-level COV/MMD/JSD over a generated corpus vs a training corpus"),
    (_P + "generative.sequence_set_ratios",
     "set-level unique/novel ratios over a generated corpus vs a training corpus"),
    (_P + "generative.prompt_similarity",
     "cross-product similarity between generation SETTINGS; needs an embed function"),
    (_P + "generative.render_distribution",
     "no docstring; needs an injected feature-distance function over a corpus"),
    (_P + "sequence.loss_masks",
     "training-time loss masking, not an evaluation metric"),
    (_P + "sequence.tokenizer_frontier",
     "no docstring; a tokenizer-vocabulary sweep, not a pred/gold metric"),
)


# ---------------------------------------------------------------------------
# Discovery: materialise Metric objects from the capability index.
# ---------------------------------------------------------------------------

_METRICS: Optional[Dict[str, Metric]] = None
_UNADAPTED: Tuple[str, ...] = ()


def _build_metrics() -> Dict[str, Metric]:
    """Join the adapter table onto the AST capability index (package='bench').

    The index is the source of truth for *what exists*: a module that is not in
    ``registry.find(package="bench")`` yields no metric, however many adapters
    point at it. The adapter table only says *how* to call what exists.
    """
    global _UNADAPTED
    entries = {e.dotted: e for e in capability_registry.find(package=BENCH_PACKAGE)}
    adapted_modules = set()
    out: Dict[str, Metric] = {}
    for name, dotted, kind, inputs, adapter in _ADAPTER_TABLE:
        entry = entries.get(dotted)
        if entry is None:      # the module left the tree -> the metric leaves too
            continue
        if kind not in KINDS:
            raise ValueError(f"metric {name!r} has unknown kind {kind!r}")
        for key in inputs:
            if key not in INPUT_KINDS:
                raise ValueError(f"metric {name!r} needs unknown input kind {key!r}")
        if name in out:
            raise ValueError(f"duplicate metric name {name!r}")
        adapted_modules.add(dotted)
        out[name] = Metric(
            name=name, kind=kind, dotted=dotted, inputs=tuple(inputs),
            adapter=adapter, summary=entry.summary, tags=tuple(entry.tags),
        )
    _UNADAPTED = tuple(sorted(d for d in entries if d not in adapted_modules))
    return out


def _all() -> Dict[str, Metric]:
    global _METRICS
    if _METRICS is None:
        _METRICS = _build_metrics()
    return _METRICS


def metrics(kind: Optional[str] = None, tag: Optional[str] = None) -> Tuple[Metric, ...]:
    """Every discovered metric, optionally filtered by kind and/or capability tag."""
    out = [m for m in _all().values()
           if (kind is None or m.kind == kind) and (tag is None or tag in m.tags)]
    return tuple(sorted(out, key=lambda m: m.name))


def metric(name: str) -> Metric:
    try:
        return _all()[name]
    except KeyError:
        raise KeyError(f"no such metric: {name!r}") from None


def kinds() -> Tuple[str, ...]:
    return tuple(sorted({m.kind for m in _all().values()}))


def unadapted() -> Tuple[str, ...]:
    """Bench modules in the index that no adapter binds yet (discovery, not silence)."""
    _all()
    return _UNADAPTED


# ---------------------------------------------------------------------------
# Rival families -- the reason suites exist.
# ---------------------------------------------------------------------------

#: Metrics inside one family answer the SAME question under DIFFERENT protocols.
#: Their numbers are not comparable and must never be pooled into one average.
#: A suite may select at most one member of each family (enforced below).
RIVAL_FAMILIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("chamfer_distance_3d", (
        "geometry.chamfer_unit_sphere",       # centroid + unit-sphere, squared CD
        "geometry.chamfer_unit_cube",         # bbox -> [-0.5,0.5]^3, mean CD x1000
        "geometry.chamfer_bbox_judged",       # centroid + max-extent, unhalved sq CD x1000
        "geometry.chamfer_raw",               # no normalisation, mean CD
        "geometry.chamfer_scaled_step",       # centroid-aligned, GT-scale-normalised
        "geometry.chamfer_orientation_aligned",  # min over proper axis rotations
        "geometry.chamfer_complex",           # directed/complex-matching CD
        "geometry.accuracy_completeness",     # Acc/Comp L1 chamfer
        "geometry.edge_chamfer_recon",        # unit-box normalised CD + HD
        "geometry.hausdorff_iogt",            # symmetric point-cloud distance + IoGT
        "geometry.factorization_fidelity",    # raw CD + symmetry CD
    )),
    ("betti_topology", (
        "geometry.betti_graded",              # fuzzy log-ratio score in [0, 1]
        "geometry.betti_exact",               # hard equality of the Betti vector
    )),
    ("volumetric_iou", (
        "geometry.voxel_iou_points",          # voxelise sampled points, Jaccard
        "geometry.voxel_iou_grid",            # supplied occupancy grids, Jaccard
        "geometry.chamfer_bbox_judged",       # also reports its own IoU protocol
        "geometry.program_shape_match",       # CSG lattice voxel IoU + bbox IoU
    )),
    ("sequence_accuracy", (
        "sequence.reconstruction_accuracy",   # DeepCAD ACC_cmd / ACC_param (eta=3)
        "sequence.autoencoder_accuracy",      # slot-tolerance command/param accuracy
        "sequence.multilevel",                # op-type accuracy + edit distance
    )),
    ("sketch_f1", (
        "sketch.sketch_f1",                   # primitive+constraint F1, tolerant match
        "sketch.entity_sketch_f1",            # CadVLM entity/sketch/CAD-F1
    )),
    ("raster_overlap", (
        "sketch.raster_vectorization",        # binary ink IoU/PRF
        "vision.mask_iou",                    # detection-style mask IoU
    )),
    # -- families introduced with the second adapter wave ---------------------
    ("chamfer_distance_2d", (
        "sketch.chamfer_2d",                  # plain symmetric 2D chamfer
        "geometry.chamfer_refinement_2d",     # mrCAD refinement chamfer
        "sketch.set_prediction_f1",           # DAVINCI CD: chamfer over SAMPLED
                                              # points of Hungarian-matched primitives
    )),
    ("primitive_f1", (
        "sequence.command_f1",                # per-command-family F1, positional
        "sequence.sequence_f1",               # CAD-SIGNet: Hungarian loop matching
        "sequence.primitive_f1_null_class",   # Text2CAD: bbox matching + Null class
    )),
    ("labelwise_agreement", (
        "vision.face_segmentation",           # per-face accuracy + macro IoU
        "vision.pointwise_semantic",          # SymPoint mIoU/fwIoU/mAcc/pACC
    )),
    ("panoptic_quality", (
        "vision.instance_segmentation",       # mask-IoU greedy PQ/RQ/SQ
        "vision.length_weighted_panoptic",    # CADTransformer log-length weighted PQ
        "vision.point_weighted_panoptic",     # SymPoint point-weighted PQ
    )),
    ("validity_rate", (
        "sequence.invalidity_ratio",          # structural CAD-sequence invalidity
        "sequence.code_validity",             # static AST safety/output contract
    )),
    ("latent_retrieval_accuracy", (
        "retrieval.graded_retrieval",         # GC-CAD graded Recall@k / NDCG@k
        "retrieval.gallery_retrieval",        # query/gallery 1-NN accuracy + mAP
        "retrieval.image_retrieval_accuracy", # GenCAD batch retrieval accuracy R_B
        "retrieval.latent_alignment",         # GenCAD-3D full-library Top-1 / MRR
    )),
    ("set_diversity", (
        "generative.diversity",               # mean pairwise DISTANCE in feature space
        "generative.diversity_similarity_matrix",  # 1 - mean pairwise SIMILARITY
    )),
)


def reasons() -> Dict[str, str]:
    """dotted -> why that bench module is deliberately left unadapted."""
    return {dotted: reason for dotted, reason in UNADAPTED_REASONS}


def rivals() -> Dict[str, Tuple[str, ...]]:
    """family -> the mutually-exclusive metrics in it."""
    return {name: members for name, members in RIVAL_FAMILIES}


def _rival_conflicts(metric_names: Sequence[str]) -> List[Tuple[str, Tuple[str, ...]]]:
    conflicts = []
    chosen = set(metric_names)
    for family, members in RIVAL_FAMILIES:
        hit = tuple(sorted(chosen.intersection(members)))
        if len(hit) > 1:
            conflicts.append((family, hit))
    return conflicts


class RivalBlendError(ValueError):
    """A suite tried to select two rival metrics -- their numbers are not comparable."""


# ---------------------------------------------------------------------------
# Suites -- named protocols, each rival-free by construction.
# ---------------------------------------------------------------------------

_SUITE_DEFS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("deepcad",
     "DeepCAD reconstruction protocol: unit-SPHERE normalised squared Chamfer, "
     "command/parameter accuracy over the 16 DeepCAD slots, invalidity ratio.",
     ("geometry.chamfer_unit_sphere", "sequence.reconstruction_accuracy",
      "sequence.command_f1", "sequence.invalidity_ratio",
      "sequence.token_accuracy")),

    ("cadrille",
     "cadrille / CAD-Recode protocol: unit-CUBE normalised mean Chamfer x1000, "
     "voxel IoU, sequence edit distance. Deliberately NOT the DeepCAD Chamfer.",
     ("geometry.chamfer_unit_cube", "geometry.voxel_iou_points",
      "sequence.edit_distance", "sequence.length_stats")),

    ("text_to_cadquery",
     "Text-to-CadQuery protocol: judged-candidate CD/F1/volumetric-IoU under the "
     "repo's bounding-box normalisation, plus code-level sequence metrics.",
     ("geometry.chamfer_bbox_judged", "sequence.edit_distance",
      "sequence.pass_at_k", "sequence.code_ast_metrics")),

    ("ps_cad",
     "PS-CAD / surface-reconstruction protocol: accuracy, completeness, "
     "threshold F-score, Hausdorff and edge-Chamfer on unit-box clouds.",
     ("geometry.accuracy_completeness", "geometry.mesh_quality",
      "geometry.topology_euler")),

    ("topology_graded",
     "Graded topology protocol: fuzzy log-ratio Betti match on meshes (rivals "
     "topology_exact -- never run both into one average).",
     ("geometry.betti_graded", "geometry.topology_euler",
      "geometry.mesh_discrepancy", "geometry.mesh_topology")),

    ("topology_exact",
     "Exact topology protocol: hard Betti-vector equality on voxel solids "
     "(rivals topology_graded).",
     ("geometry.betti_exact", "geometry.voxel_iou_grid")),

    ("sketch",
     "Sketch protocol: primitive/constraint F1, 2D sketch Chamfer, raster "
     "vectorisation IoU/PRF.",
     ("sketch.sketch_f1", "sketch.chamfer_2d", "sketch.raster_vectorization",
      "sketch.image_symmetry", "sketch.sequence_metrics")),

    ("cadvlm",
     "CadVLM protocol: entity/sketch/CAD-F1 over entity tuples (rivals the "
     "sketch suite's primitive F1).",
     ("sketch.entity_sketch_f1", "sketch.chamfer_2d")),

    ("vision",
     "Vision protocol: silhouette IoU, depth metrics, multiview consistency.",
     ("vision.silhouette_iou", "vision.depth_metrics",
      "vision.multiview_consistency", "vision.mask_iou")),

    ("retrieval",
     "Retrieval protocol: ranked (nDCG/MRR) and tiered (NN/FT/ST/ANMRR) metrics, "
     "plus embedding-space quality.",
     ("retrieval.ranked", "retrieval.tiered", "retrieval.representation_quality",
      "retrieval.latent_cluster_quality", "retrieval.embedding_cosine")),

    ("generative",
     "Generative protocol: FID over CAD latents, 1-NNA, feature-space diversity.",
     ("generative.fid", "generative.one_nna", "generative.diversity")),

    ("text2cad",
     "Text2CAD protocol (Khan et al., NeurIPS 2024, Sec. 5.1): the reference "
     "generate_report -- bbox-matched primitive F1 with a Null class, extrusion "
     "F1 -- plus the invalidity ratio. Rivals the DeepCAD command F1.",
     ("sequence.primitive_f1_null_class", "sequence.invalidity_ratio")),

    ("sympoint",
     "SymPoint symbol-spotting protocol (ECCV 2024): the point-weighted panoptic "
     "half and the point-wise semantic half of the same benchmark.",
     ("vision.point_weighted_panoptic", "vision.pointwise_semantic")),

    ("cluster3d",
     "Cluster3D protocol (Sec. 4.2-4.3): external agreement indices (NMI/ARI/ACC) "
     "against a reference labelling plus internal validity indices on the "
     "embedding itself.",
     ("retrieval.clustering_external", "retrieval.clustering_internal")),

    ("geometry_smoke",
     "Every geometry metric that only needs a point cloud, EXCEPT the rivals: one "
     "Chamfer protocol (unit sphere) and one IoU protocol are selected.",
     ("geometry.chamfer_unit_sphere", "geometry.voxel_iou_points",
      "geometry.contact_heatmap")),
)


def _build_suites() -> Dict[str, Suite]:
    known = set(_all())
    out: Dict[str, Suite] = {}
    for name, description, metric_names in _SUITE_DEFS:
        missing = [n for n in metric_names if n not in known]
        if missing:
            raise ValueError(f"suite {name!r} names unknown metrics: {missing}")
        conflicts = _rival_conflicts(metric_names)
        if conflicts:
            raise RivalBlendError(
                f"suite {name!r} selects rival metrics that are not comparable: "
                + "; ".join(f"{fam}: {', '.join(hit)}" for fam, hit in conflicts))
        out[name] = Suite(name=name, description=description,
                          metric_names=tuple(metric_names))
    return out


_SUITES: Optional[Dict[str, Suite]] = None


def _suite_map() -> Dict[str, Suite]:
    global _SUITES
    if _SUITES is None:
        _SUITES = _build_suites()
    return _SUITES


def suites() -> Tuple[str, ...]:
    return tuple(sorted(_suite_map()))


def suite(name: str) -> Suite:
    try:
        return _suite_map()[name]
    except KeyError:
        raise KeyError(
            f"no such suite: {name!r} (known: {', '.join(suites())})") from None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _sample_id(sample: dict, position: int) -> str:
    return str(sample.get("id", f"sample-{position}"))


def run_metric(m: Metric, sample: dict, position: int = 0) -> MetricResult:
    """Score one sample with one metric. Never raises: failures become entries."""
    sid = _sample_id(sample, position)
    base = dict(metric=m.name, kind=m.kind, dotted=m.dotted, sample_id=sid)
    if not m.applicable(sample):
        missing = ", ".join(k for k in m.inputs
                            if k not in (sample.get("pred") or {})
                            or k not in (sample.get("gold") or {}))
        return MetricResult(status="skipped", error=f"sample lacks: {missing}", **base)
    try:
        value = m.score(sample.get("pred") or {}, sample.get("gold") or {})
    except Exception as exc:  # noqa: BLE001 - a metric must never abort the suite
        return MetricResult(status="error",
                            error=f"{type(exc).__name__}: {exc}", **base)
    if isinstance(value, dict):
        value = {k: value[k] for k in sorted(value)}
    return MetricResult(status="ok", value=value, **base)


def run_suite(name: str, samples: Sequence[dict],
              extra_metrics: Sequence[Metric] = ()) -> Report:
    """Run a named suite over ``samples``. Deterministic; errors are recorded.

    Ordering is fixed: samples in the order given, metrics sorted by name. A metric
    that raises yields an ``error`` entry and the run continues. A metric whose
    inputs the sample does not carry yields a ``skipped`` entry -- never a guess.
    """
    selected = [metric(n) for n in suite(name).metric_names] + list(extra_metrics)
    conflicts = _rival_conflicts([m.name for m in selected])
    if conflicts:
        raise RivalBlendError(
            f"run of suite {name!r} would blend rival metrics: "
            + "; ".join(f"{fam}: {', '.join(hit)}" for fam, hit in conflicts))
    selected.sort(key=lambda m: m.name)

    results: List[MetricResult] = []
    for position, sample in enumerate(samples):
        for m in selected:
            results.append(run_metric(m, sample, position))
    return Report(suite=name, n_samples=len(samples),
                  metric_names=tuple(m.name for m in selected), results=results)


# ---------------------------------------------------------------------------
# CLI (wired into core.cli as `harnesscad bench`)
# ---------------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every discovered metric")
    parser.add_argument("--suites", action="store_true",
                        help="list the named suites and the metrics each selects")
    parser.add_argument("--rivals", action="store_true",
                        help="list the rival families that must never be averaged")
    parser.add_argument("--unadapted", action="store_true",
                        help="list bench modules with no adapter yet")
    parser.add_argument("--kind", default=None, choices=list(KINDS),
                        help="filter --list by metric kind")
    parser.add_argument("--suite", default=None, help="run this named suite")
    parser.add_argument("--input", default=None,
                        help="path to a JSON file of samples for --suite")
    parser.add_argument("--json", action="store_true",
                        help="print the report as JSON")


def _load_samples(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        payload = payload.get("samples", [])
    if not isinstance(payload, list):
        raise ValueError("input must be a JSON array of samples, "
                         "or an object with a 'samples' array")
    return payload


def run(args: argparse.Namespace) -> int:
    if getattr(args, "suites", False):
        for name in suites():
            s = suite(name)
            print(f"{s.name}")
            print(f"    {s.description}")
            for n in s.metric_names:
                m = metric(n)
                print(f"    - {n:<38} {m.dotted}")
        print(f"-- {len(suites())} suites")
        return 0

    if getattr(args, "rivals", False):
        for family, members in RIVAL_FAMILIES:
            print(f"{family}: (never averaged together)")
            for n in members:
                print(f"    - {n}")
        return 0

    if getattr(args, "unadapted", False):
        why = reasons()
        for dotted in unadapted():
            reason = why.get(dotted, "")
            print(f"{dotted}" + (f"\n    reason: {reason}" if reason else ""))
        print(f"-- {len(unadapted())} bench modules without an adapter "
              f"({len(why)} with a stated reason)")
        return 0

    if getattr(args, "list", False) or not getattr(args, "suite", None):
        selected = metrics(kind=getattr(args, "kind", None))
        for m in selected:
            print(f"{m.name:<40} {m.kind:<11} [{'+'.join(m.inputs)}]")
            print(f"    {m.dotted}")
            if m.summary:
                print(f"    {m.summary}")
        print(f"-- {len(selected)} metrics / {len(metrics())} discovered "
              f"/ {len(unadapted())} bench modules unadapted")
        return 0

    if not getattr(args, "input", None):
        print("error: --suite requires --input <samples.json>", file=sys.stderr)
        return 2
    try:
        samples = _load_samples(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: could not load samples from {args.input!r}: {exc}",
              file=sys.stderr)
        return 2
    try:
        report = run_suite(args.suite, samples)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), sort_keys=True, indent=2))
        return 0 if not report.errors() else 1

    print(f"suite:    {report.suite}")
    print(f"samples:  {report.n_samples}")
    print(f"metrics:  {len(report.metric_names)}")
    print(f"ok/err/skip: {len(report.ok())}/{len(report.errors())}/"
          f"{len(report.skipped())}")
    print("aggregates (per metric -- rivals are never pooled):")
    aggregates = report.aggregates()
    if not aggregates:
        print("  (none)")
    for metric_name in sorted(aggregates):
        print(f"  {metric_name}  [{metric(metric_name).dotted}]")
        for key, val in aggregates[metric_name].items():
            print(f"    {key:<28} {val!r}")
    if report.errors():
        print("errors:")
        for r in report.errors():
            print(f"  {r.metric} @{r.sample_id}: {r.error}")
    return 0 if not report.errors() else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad bench",
        description="metric registry and suite runner for the bench layer")
    add_arguments(parser)
    return run(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
