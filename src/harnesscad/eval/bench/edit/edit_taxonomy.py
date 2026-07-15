"""neuralCAD-Edit edit-operation taxonomy and benchmark task structure (eval module).

Deterministic re-encoding of the *benchmark* defined by neuralCAD-Edit (Perrett et
al., "neuralCAD-Edit: An Expert Benchmark for Multimodal-Instructed 3D CAD Model
Editing"): 192 editing requests / 384 edits, each a multimodal instruction that
transforms a starting B-rep into an edited B-rep, scored against an expert ground
truth. The trained VLM editors and DINO/CLIP feature extractors are external; the
*task structure, the edit-operation taxonomy and the scoring rubric* are a fixed
specification, built here as an eval module.

TASK STRUCTURE (from the repo's DB schema and ``benchmark_evals/edit.py``)
--------------------------------------------------------------------------
A benchmark item is a *request*: ``brep_start`` (the model to edit), a natural
language ``instruction`` (optionally with reference images -> multimodal), a
``difficulty`` in ``{easy, medium, hard}``, and an expert ``brep_end`` ground
truth. An editor produces an edited B-rep; it is scored two ways:

* **Geometry / feature similarity vs ground truth**
  (``utils/evals_feature_geometric.py``): DINOv2 cosine, CLIP-visual cosine,
  Chamfer similarity ``1 / (chamfer_distance + 1e-8)``, and voxel IoU
  (``voxel_divisor = 100``). A *start* similarity (vs ``brep_start``) is also taken
  so an edit that changed nothing can be told from one that reached the target.
* **VLM expert rating** (``config/edit_192_external.json``): two 1-7 scales,
  ``instruction-understanding`` and ``quality``, with fixed anchor descriptions.

EDIT-OPERATION TAXONOMY
-----------------------
The instructions map onto a small, fixed family of CAD edit operations. This
module enumerates them and provides a deterministic keyword classifier so an
instruction can be tagged without a model.

Results are reported per difficulty tier. This is DISTINCT from
``bench.data.difficulty_tiers`` (Text2CAD generation L1-L4 geometry tiers): here
difficulty labels an *edit* task, and the metrics compare an edit against an
expert ground truth.

Stdlib only, deterministic. No model runs; features/scores are injected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = [
    "DIFFICULTIES",
    "EditOperation",
    "EDIT_OPERATION_KEYWORDS",
    "RATING_SCALES",
    "classify_instruction",
    "EditTask",
    "EditBenchmark",
    "chamfer_similarity",
    "voxel_iou_similarity",
    "aggregate_rating",
    "edit_effectiveness",
    "score_edit",
]

DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")


class EditOperation:
    """The neuralCAD-Edit edit-operation families (parametric CAD edit taxonomy)."""

    ADD = "add"                # create/add a new feature or body
    REMOVE = "remove"          # delete/remove a feature or body
    MODIFY = "modify"          # change a dimension / parameter value
    TRANSFORM = "transform"    # move / rotate / reposition
    PATTERN = "pattern"        # mirror / array / linear or circular pattern
    BOOLEAN = "boolean"        # cut / union / intersect between bodies
    FILLET = "fillet"          # round an edge
    CHAMFER = "chamfer"        # chamfer / bevel an edge
    SHELL = "shell"            # hollow / shell / wall-thickness
    SKETCH = "sketch"          # edit the underlying sketch / profile
    MATERIAL = "material"      # appearance / material / colour
    CONSTRAINT = "constraint"  # add/change a constraint or relation

    ALL = (
        ADD, REMOVE, MODIFY, TRANSFORM, PATTERN, BOOLEAN,
        FILLET, CHAMFER, SHELL, SKETCH, MATERIAL, CONSTRAINT,
    )


# Ordered so earlier families win on ties (checked in this order).
EDIT_OPERATION_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (EditOperation.FILLET, ("fillet", "round the", "rounded", "round edge")),
    (EditOperation.CHAMFER, ("chamfer", "bevel")),
    (EditOperation.SHELL, ("shell", "hollow", "wall thickness", "wall-thickness")),
    (EditOperation.PATTERN, ("mirror", "pattern", "array", "duplicate", "repeat")),
    (EditOperation.BOOLEAN, ("cut", "subtract", "union", "unite", "intersect", "boolean", "combine")),
    (EditOperation.TRANSFORM, ("move", "translate", "rotate", "reposition", "shift", "offset")),
    (EditOperation.SKETCH, ("sketch", "profile", "contour", "cross-section", "cross section")),
    (EditOperation.MATERIAL, ("material", "colour", "color", "appearance", "texture")),
    (EditOperation.CONSTRAINT, ("constraint", "constrain", "parallel", "perpendicular", "coincident", "tangent")),
    (EditOperation.REMOVE, ("remove", "delete", "get rid of", "eliminate")),
    (EditOperation.ADD, ("add", "create", "insert", "new hole", "attach")),
    (EditOperation.MODIFY, ("increase", "decrease", "resize", "change", "make it", "scale", "widen", "thicken", "lengthen", "shorten", "larger", "smaller", "adjust", "set the")),
)


# The 1-7 rating anchors from config/edit_192_external.json (rating_obj prompt).
RATING_SCALES: dict = {
    "instruction_understanding": {
        1: "Makes things worse or goes in completely the wrong direction",
        2: "Does nothing",
        3: "Rough understanding, many parts incorrect or incomplete",
        4: "Mostly does what is asked, noticeable errors or omissions",
        5: "Does what is asked, small errors or omissions",
        6: "Perfect - follows the request precisely",
        7: "Above and beyond - perfect with helpful expert extras",
    },
    "quality": {
        1: "No model or edit produced",
        2: "Very poor - erroneous / impossible geometry",
        3: "Poor - simplistic, blocky",
        4: "Average - acceptable first pass",
        5: "Good attempt with room for improvement",
        6: "High quality, as from an experienced designer",
        7: "Perfect - extremely polished design",
    },
}


def classify_instruction(text: str) -> str:
    """Tag an edit instruction with its most likely operation family.

    Deterministic keyword match in the fixed :data:`EDIT_OPERATION_KEYWORDS` order;
    returns :data:`EditOperation.MODIFY` when nothing matches (the benchmark's most
    common dimensional-change default).
    """
    low = text.lower()
    for family, keywords in EDIT_OPERATION_KEYWORDS:
        for kw in keywords:
            # word-boundary match so e.g. "remove" is not read as "move".
            if re.search(r"\b" + re.escape(kw) + r"\b", low):
                return family
    return EditOperation.MODIFY


@dataclass(frozen=True)
class EditTask:
    """One benchmark request: a starting model + a multimodal edit instruction."""

    task_id: str
    instruction: str
    difficulty: str
    brep_start: str
    brep_end_gt: str
    multimodal: bool = False
    operation: str = ""

    def __post_init__(self):
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}, got {self.difficulty!r}")
        if not self.operation:
            object.__setattr__(self, "operation", classify_instruction(self.instruction))


@dataclass(frozen=True)
class EditBenchmark:
    """A set of :class:`EditTask` with composition summaries."""

    tasks: tuple[EditTask, ...]

    def by_difficulty(self) -> dict:
        out: dict = {d: [] for d in DIFFICULTIES}
        for t in self.tasks:
            out[t.difficulty].append(t)
        return out

    def difficulty_counts(self) -> dict:
        return {d: len(ts) for d, ts in self.by_difficulty().items()}

    def operation_counts(self) -> dict:
        counts: dict = {op: 0 for op in EditOperation.ALL}
        for t in self.tasks:
            counts[t.operation] = counts.get(t.operation, 0) + 1
        return counts


def chamfer_similarity(chamfer_distance: float, eps: float = 1e-8) -> float:
    """neuralCAD-Edit chamfer similarity: ``1 / (chamfer_distance + eps)`` (higher = closer)."""
    if chamfer_distance < 0.0:
        raise ValueError("chamfer distance must be non-negative")
    return 1.0 / (chamfer_distance + eps)


def voxel_iou_similarity(intersection: int, union: int) -> float:
    """Voxel IoU of edited vs ground-truth occupancy (``|A & B| / |A | B|``)."""
    if union < 0 or intersection < 0 or intersection > union:
        raise ValueError("require 0 <= intersection <= union")
    return 0.0 if union == 0 else intersection / union


def aggregate_rating(ratings: Sequence[Mapping[str, float]]) -> dict:
    """Mean of the two 1-7 rater scales over a list of per-rater dicts."""
    if not ratings:
        return {"instruction_understanding": 0.0, "quality": 0.0}
    keys = ("instruction_understanding", "quality")
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in ratings if k in r]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out


def edit_effectiveness(gt_similarity: float, start_similarity: float) -> float:
    """How much closer to the target than to the unedited start (>0 means progress).

    neuralCAD-Edit records both a ground-truth similarity and a start similarity so
    a no-op edit (high start similarity, low gt similarity) is penalised.
    """
    return gt_similarity - start_similarity


def score_edit(
    gt_similarity: float,
    start_similarity: float,
    ratings: Sequence[Mapping[str, float]] = (),
) -> dict:
    """Bundle the geometry-vs-gt and rating signals for one edit into a report dict."""
    r = aggregate_rating(ratings)
    return {
        "gt_similarity": gt_similarity,
        "start_similarity": start_similarity,
        "effectiveness": edit_effectiveness(gt_similarity, start_similarity),
        "rating_instruction_understanding": r["instruction_understanding"],
        "rating_quality": r["quality"],
    }
