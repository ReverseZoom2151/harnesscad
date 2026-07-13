"""Text2CAD's exact evaluation protocol (``CADSequence.generate_report``).

Reference implementation: ``CadSeqProc/cad_sequence.py`` (``generate_report``,
``extrusionAccuracyReport``), ``SketchSequence.loop_match``,
``LoopSequence.match_primitives`` / ``loop_distance``, ``Curve.curve_distance`` and
``Evaluation/eval_seq.py`` of the released Text2CAD code (Khan et al., NeurIPS 2024).

Why this is not ``bench.text2cad_sequence_f1``
----------------------------------------------
That module implements the protocol as *described* (Hungarian loop matching, then
per-primitive F1) with a loop-to-loop cost equal to the primitive-multiset
disagreement. The released code differs in every load-bearing detail:

1. **Sketches are aligned positionally** (padded with ``None`` to the longer list),
   not matched.
2. **Loops are matched by geometry**: the cost between two loops is the (scaled) L2
   distance between their 2x2 bounding boxes, ``||bbox_gt * scale - bbox_pred * scale||``,
   and unmatchable slots (``None`` padding) cost a fixed ``multiplier`` (2 for loops).
3. **Curves inside a matched loop pair are matched again by Hungarian**, with the same
   scaled-bbox L2 cost and ``multiplier = 1`` for ``None`` slots.
4. The matched curve pairs are turned into two *label* streams over
   ``{0: line, 1: arc, 2: circle, 3: Null}`` -- an unmatched curve is a prediction of
   (or a ground truth of) the **Null class**. Precision / recall / F1 per primitive
   then fall straight out of the 4x4 confusion matrix, together with macro/micro
   averages and the overall type accuracy. This Null-class treatment is what makes
   the reported precision penalise hallucinated curves.
5. **Extrusion** is scored separately: ``recall = n_min/n_gt``, ``precision =
   n_min/n_pred``, ``F1 = 2PR/(P+R)`` over the *number* of extrusions, plus an L1
   parameter report over ``(total distance, ox, oy, oz, theta, phi, gamma, sketch
   scale)`` and a boolean-operation hit count. Before comparison the parameters are
   divided by ``NORM_FACTOR = 0.75`` -- the extrusion parameters live in ``0..0.75``
   after normalisation and are rescaled to ``0..1`` for reporting.

Curves are plain dicts as produced by :mod:`reconstruction.t2c3_cad_vec_codec`::

    {"type": "line",   "start": (x, y), "end": (x, y)}
    {"type": "arc",    "start": (x, y), "mid": (x, y), "end": (x, y)}
    {"type": "circle", "center": (x, y), "radius": r}

Deterministic and stdlib-only; the Hungarian solver is reused from
``bench.text2cad_sequence_f1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from harnesscad.eval.bench.text2cad_sequence_f1 import hungarian_assignment

CURVE_TYPES: tuple[str, ...] = ("line", "arc", "circle")
NULL_LABEL = 3                      # the 4th class: "no curve"
N_LABELS = len(CURVE_TYPES) + 1
NORM_FACTOR = 0.75                  # extrusion params live in 0..0.75

LOOP_MULTIPLIER = 2                 # cost of matching a loop with None
CURVE_MULTIPLIER = 1                # cost of matching a curve with None


class EvalProtocolError(ValueError):
    """Raised for malformed evaluation inputs."""


# --- geometry ---------------------------------------------------------------
def curve_bbox(curve: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis-aligned bbox ``((min_x, min_y), (max_x, max_y))`` of one curve."""
    kind = curve["type"]
    if kind == "line":
        pts = [curve["start"], curve["end"]]
    elif kind == "arc":
        pts = [curve["start"], curve["mid"], curve["end"]]
    elif kind == "circle":
        cx, cy = curve["center"]
        r = curve["radius"]
        pts = [(cx - r, cy - r), (cx + r, cy + r)]
    else:
        raise EvalProtocolError(f"unknown curve type {kind!r}")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return ((min(xs), min(ys)), (max(xs), max(ys)))


def loop_bbox(loop: list[dict]) -> tuple[tuple[float, float], tuple[float, float]]:
    if not loop:
        raise EvalProtocolError("empty loop")
    boxes = [curve_bbox(c) for c in loop]
    return (
        (min(b[0][0] for b in boxes), min(b[0][1] for b in boxes)),
        (max(b[1][0] for b in boxes), max(b[1][1] for b in boxes)),
    )


def bbox_distance(box_a, box_b, scale: float) -> float:
    """L2 distance between two scaled 2x2 bounding boxes (flattened)."""
    total = 0.0
    for i in range(2):
        for j in range(2):
            total += (box_a[i][j] * scale - box_b[i][j] * scale) ** 2
    return math.sqrt(total)


def loop_distance(gt_loop: list[dict], pred_loop: list[dict], scale: float) -> float:
    return bbox_distance(loop_bbox(gt_loop), loop_bbox(pred_loop), scale)


def curve_distance(gt_curve: dict, pred_curve: dict, scale: float) -> float:
    return bbox_distance(curve_bbox(gt_curve), curve_bbox(pred_curve), scale)


# --- matching ---------------------------------------------------------------
def _pad(items: list, n: int) -> list:
    return list(items) + [None] * (n - len(items))


def _assign(gt_items, pred_items, cost_fn, scale: float, multiplier: float):
    """Hungarian match with ``None`` padding; returns list of ``(gt, pred)`` pairs."""
    gt_items = list(gt_items) if gt_items else [None]
    pred_items = list(pred_items) if pred_items else [None]
    n_gt, n_pred = len(gt_items), len(pred_items)
    n_max = max(n_gt, n_pred)
    gt_items = _pad(gt_items, n_max)
    pred_items = _pad(pred_items, n_max)

    cost = [[float(multiplier)] * n_max for _ in range(n_max)]
    for i in range(n_gt):
        for j in range(n_pred):
            if gt_items[i] is not None and pred_items[j] is not None:
                cost[i][j] = cost_fn(gt_items[i], pred_items[j], scale)

    assign = hungarian_assignment(cost)
    return [(gt_items[i], pred_items[assign[i]]) for i in range(n_max)]


def match_loops(gt_loops, pred_loops, scale: float, multiplier: float = LOOP_MULTIPLIER):
    """Match ground-truth loops to predicted loops within one sketch."""
    return _assign(gt_loops, pred_loops, loop_distance, scale, multiplier)


def match_primitives(gt_loop, pred_loop, scale: float,
                     multiplier: float = CURVE_MULTIPLIER):
    """Match curves inside a matched loop pair (either side may be ``None``)."""
    return _assign(gt_loop, pred_loop, curve_distance, scale, multiplier)


def sketch_loops(sketch) -> list[list[dict]]:
    """Flatten a ``[face][loop][curve]`` sketch into its list of loops."""
    return [loop for face in sketch for loop in face]


def match_model_curves(gt_model: list[dict], pred_model: list[dict],
                       scale: float = 1.0) -> list[tuple]:
    """Positional sketch alignment -> loop matching -> curve matching.

    Returns the flat list of matched ``(gt_curve, pred_curve)`` pairs, where either
    element may be ``None`` (an unmatched curve).
    """
    gt_sketches = [p["sketch"] for p in gt_model]
    pred_sketches = [p["sketch"] for p in pred_model]
    n_max = max(len(gt_sketches), len(pred_sketches))
    gt_sketches = _pad(gt_sketches, n_max)
    pred_sketches = _pad(pred_sketches, n_max)

    pairs: list[tuple] = []
    for gt_sk, pred_sk in zip(gt_sketches, pred_sketches):
        gt_lp = sketch_loops(gt_sk) if gt_sk is not None else None
        pred_lp = sketch_loops(pred_sk) if pred_sk is not None else None
        for gt_loop, pred_loop in match_loops(gt_lp, pred_lp, scale):
            pairs += match_primitives(gt_loop, pred_loop, scale)
    return pairs


# --- labels / confusion matrix ---------------------------------------------
def _label(curve) -> int:
    if curve is None:
        return NULL_LABEL
    kind = curve["type"].lower()
    if kind not in CURVE_TYPES:
        raise EvalProtocolError(f"unknown curve type {kind!r}")
    return CURVE_TYPES.index(kind)


def label_streams(matched_pairs: list[tuple]) -> tuple[list[int], list[int]]:
    """``(y_true, y_pred)`` over ``{line, arc, circle, Null}`` from matched pairs."""
    y_true = [_label(gt) for gt, _ in matched_pairs]
    y_pred = [_label(pred) for _, pred in matched_pairs]
    return y_true, y_pred


def confusion_matrix(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    cm = [[0] * N_LABELS for _ in range(N_LABELS)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def _prf(tp: int, total_pred: int, total_gt: int) -> tuple[float, float, float]:
    precision = tp / total_pred if total_pred else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


@dataclass(frozen=True)
class CurveScore:
    curve_type: str
    correct: int
    total_pred: int
    total_gt: int
    precision: float
    recall: float
    f1: float


def curve_scores(cm: list[list[int]]) -> dict[str, CurveScore]:
    """Per-primitive precision/recall/F1 straight from the 4x4 confusion matrix."""
    scores = {}
    for i, name in enumerate(CURVE_TYPES):
        tp = cm[i][i]
        total_pred = sum(cm[r][i] for r in range(N_LABELS))
        total_gt = sum(cm[i])
        p, r, f = _prf(tp, total_pred, total_gt)
        scores[name] = CurveScore(name, tp, total_pred, total_gt, p, r, f)
    return scores


def macro_average(cm: list[list[int]], y_true: list[int], y_pred: list[int]) -> dict:
    """Macro P/R/F1 over the labels present in ``y_true`` or ``y_pred`` (sklearn rule)."""
    present = sorted(set(y_true) | set(y_pred))
    if not present:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    ps, rs, fs = [], [], []
    for i in present:
        tp = cm[i][i]
        p, r, f = _prf(tp, sum(cm[x][i] for x in range(N_LABELS)), sum(cm[i]))
        ps.append(p)
        rs.append(r)
        fs.append(f)
    n = len(present)
    return {"precision": sum(ps) / n, "recall": sum(rs) / n, "f1": sum(fs) / n}


def micro_average(y_true: list[int], y_pred: list[int]) -> dict:
    """Micro P/R/F1; with a full label set these all equal the type accuracy."""
    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / len(y_true)
    return {"precision": acc, "recall": acc, "f1": acc}


def type_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


# --- extrusion --------------------------------------------------------------
def _rescale(extrusion: dict, factor: float) -> dict:
    """Divide the length-like parameters by ``NORM_FACTOR`` (angles are untouched)."""
    return {
        "extent_one": extrusion["extent_one"] * factor,
        "extent_two": extrusion["extent_two"] * factor,
        "origin": tuple(v * factor for v in extrusion["origin"]),
        "euler": tuple(extrusion["euler"]),
        "sketch_size": extrusion["sketch_size"] * factor,
        "boolean": extrusion["boolean"],
    }


@dataclass(frozen=True)
class ExtrusionScore:
    num_gt: int
    num_pred: int
    num_matched: int
    precision: float
    recall: float
    f1: float
    boolean_correct: int
    parameter_l1: dict[str, float]


def extrusion_report(gt_model: list[dict], pred_model: list[dict],
                     scaling_factor: float = NORM_FACTOR) -> ExtrusionScore:
    """Extrusion count P/R/F1 plus the L1 parameter report of the reference code."""
    if scaling_factor == 0:
        raise EvalProtocolError("scaling_factor must be non-zero")
    factor = 1.0 / scaling_factor
    gt = [_rescale(p["extrusion"], factor) for p in gt_model]
    pred = [_rescale(p["extrusion"], factor) for p in pred_model]

    n_gt, n_pred = len(gt), len(pred)
    n_min = min(n_gt, n_pred)

    report = {k: 0.0 for k in ("dist", "o_x", "o_y", "o_z", "theta", "phi", "gamma", "s")}
    boolean_correct = 0
    for i in range(n_min):
        g, p = gt[i], pred[i]
        report["dist"] += abs(
            (abs(g["extent_one"]) + abs(g["extent_two"]))
            - (abs(p["extent_one"]) + abs(p["extent_two"]))
        )
        for k, idx in (("o_x", 0), ("o_y", 1), ("o_z", 2)):
            report[k] += abs(g["origin"][idx] - p["origin"][idx])
        for k, idx in (("theta", 0), ("phi", 1), ("gamma", 2)):
            report[k] += abs(g["euler"][idx] - p["euler"][idx])
        report["s"] += abs(g["sketch_size"] - p["sketch_size"])
        boolean_correct += int(g["boolean"] == p["boolean"])

    recall = n_min / n_gt if n_gt else 0.0
    precision = n_min / n_pred if n_pred else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return ExtrusionScore(
        num_gt=n_gt, num_pred=n_pred, num_matched=n_min,
        precision=precision, recall=recall, f1=f1,
        boolean_correct=boolean_correct, parameter_l1=report,
    )


# --- top level --------------------------------------------------------------
@dataclass(frozen=True)
class ModelReport:
    curves: dict[str, CurveScore]
    confusion: list[list[int]]
    macro: dict
    micro: dict
    accuracy: float
    extrusion: ExtrusionScore


def evaluate_model(gt_model: list[dict], pred_model: list[dict],
                   scale: float = 1.0) -> ModelReport:
    """Full per-sample report: the deterministic core of ``generate_report``."""
    pairs = match_model_curves(gt_model, pred_model, scale)
    y_true, y_pred = label_streams(pairs)
    cm = confusion_matrix(y_true, y_pred)
    return ModelReport(
        curves=curve_scores(cm),
        confusion=cm,
        macro=macro_average(cm, y_true, y_pred),
        micro=micro_average(y_true, y_pred),
        accuracy=type_accuracy(y_true, y_pred),
        extrusion=extrusion_report(gt_model, pred_model),
    )


def aggregate_reports(reports: list[ModelReport]) -> dict:
    """Mean of the per-sample scores, as ``Evaluation/eval_seq.py`` aggregates them.

    Per-primitive means are taken **only over samples that contain that primitive in
    the ground truth** (``report_df[report_df['<type>_total_gt'] > 0].mean()``), and
    are reported in percent.
    """
    out: dict = {}
    for name in CURVE_TYPES:
        rows = [r.curves[name] for r in reports if r.curves[name].total_gt > 0]
        if rows:
            out[name] = {
                "precision": sum(r.precision for r in rows) / len(rows) * 100,
                "recall": sum(r.recall for r in rows) / len(rows) * 100,
                "f1": sum(r.f1 for r in rows) / len(rows) * 100,
            }
        else:
            out[name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if reports:
        out["extrusion"] = {
            "precision": sum(r.extrusion.precision for r in reports) / len(reports) * 100,
            "recall": sum(r.extrusion.recall for r in reports) / len(reports) * 100,
            "f1": sum(r.extrusion.f1 for r in reports) / len(reports) * 100,
        }
    else:
        out["extrusion"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return out
