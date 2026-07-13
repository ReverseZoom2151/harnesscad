"""Progressive Hierarchical Tuning (PHT) for PHT-CAD (Niu et al. 2025, Sec. 4.3).

PHT is PHT-CAD's central contribution: a coarse-to-fine, three-stage curriculum
that progressively grows the model's capability for parametric primitive
analysis. This module models the *deterministic* machinery of that curriculum --
stage definitions, the data subset and hyper-parameters each stage consumes, the
replay/sampling rule that guards against forgetting, and the coarse-to-fine
parameter refinement that hands predictions from one stage to the next. The
learned VLM itself is external; here we build the schedule and the refinement
operator around it.

Three stages (paper Fig. 5):

  Stage 1 -- Primitive Perception Tuning
      Recognise/classify individual primitives and emit their (coarse) params.
      Data subset: ``single_primitive``. lr 1e-4, 2 epochs, tokens 4096.
      Coarse-to-fine role: establishes primitive *type & existence*.

  Stage 2 -- Structural Perception Tuning
      Count all primitives, output their params via EHP, infer inter-primitive
      constraints. Data subset: ``sketch_structural``. lr (inherited), 1 epoch,
      tokens 8192. Coarse-to-fine role: *coarse parameters* + structure.

  Stage 3 -- Annotation-geometry Alignment
      Predict primitives + constraints + dimensional annotations while retaining
      non-dimensioned generalisation. Data subset: ``dimensional_annotated``.
      lr 2e-5, replay 50% of stage-2 data. Coarse-to-fine role: *fine params*.

The ablation (paper Tab. 4) shows removing Stage 1 costs ~12% accuracy and
removing Stage 2 ~15%; :func:`ablation_accuracy` reproduces that monotone
ordering deterministically from a base accuracy and the reported deltas so a
caller can reason about stage importance without the trained model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Ordered names of the three PHT stages.
STAGE_NAMES = ("primitive_perception", "structural_perception",
               "annotation_geometry_alignment")

# Which coarse-to-fine refinement level each stage is responsible for.
STAGE_LEVEL = {
    "primitive_perception": "type",
    "structural_perception": "coarse_params",
    "annotation_geometry_alignment": "fine_params",
}


@dataclass(frozen=True)
class StageSpec:
    """Immutable description of one PHT stage."""

    index: int
    name: str
    data_subset: str
    learning_rate: float
    epochs: int
    max_tokens: int
    replay_fraction: float   # fraction of the PREVIOUS stage's data mixed in
    refines: str             # coarse-to-fine level this stage delivers


# The canonical three-stage schedule (paper Sec. 5.1 implementation details).
_SCHEDULE: tuple[StageSpec, ...] = (
    StageSpec(1, "primitive_perception", "single_primitive",
              1e-4, 2, 4096, 0.0, "type"),
    StageSpec(2, "structural_perception", "sketch_structural",
              1e-4, 1, 8192, 0.0, "coarse_params"),
    StageSpec(3, "annotation_geometry_alignment", "dimensional_annotated",
              2e-5, 1, 8192, 0.5, "fine_params"),
)


def schedule() -> tuple[StageSpec, ...]:
    """Return the ordered three-stage PHT schedule."""
    return _SCHEDULE


def stage(index: int) -> StageSpec:
    """Return the :class:`StageSpec` for a 1-based ``index`` (1, 2, or 3)."""
    if index < 1 or index > len(_SCHEDULE):
        raise ValueError(f"stage index out of range 1..{len(_SCHEDULE)}: {index}")
    return _SCHEDULE[index - 1]


def build_mixture(index: int, stage_sizes: dict[str, int]) -> dict[str, int]:
    """Sample counts to draw for stage ``index`` given per-subset sizes.

    Includes this stage's own subset in full plus ``replay_fraction`` of the
    *previous* stage's subset (rounded down) to counter catastrophic forgetting
    -- exactly the "50% data of the second stage is sampled" rule of Stage 3.
    """
    spec = stage(index)
    mix: dict[str, int] = {spec.data_subset: stage_sizes.get(spec.data_subset, 0)}
    if spec.replay_fraction > 0 and index > 1:
        prev = stage(index - 1)
        replay = int(stage_sizes.get(prev.data_subset, 0) * spec.replay_fraction)
        if replay > 0:
            mix[prev.data_subset] = replay
    return mix


# ---------------------------------------------------------------------------
# Coarse-to-fine progressive parameter refinement.
# ---------------------------------------------------------------------------

# Ordered refinement levels; each stage sharpens the tolerance below.
REFINEMENT_LEVELS = ("type", "coarse_params", "fine_params")

# Quantisation grid (in EHP normalised units, range [0,1000)) applied at each
# level. Coarse stages snap parameters to a wide grid; later stages relax to a
# fine grid, so predictions are refined monotonically from coarse to fine.
_LEVEL_GRID = {"type": None, "coarse_params": 50.0, "fine_params": 1.0}


@dataclass
class Prediction:
    """A mutable coarse-to-fine primitive prediction carried across stages."""

    kind: str
    exists: bool = True
    params: list[float] = field(default_factory=list)
    level: str = "type"


def _snap(value: float, grid: float) -> float:
    return round(value / grid) * grid


def refine(pred: Prediction, target_level: str,
           raw_params: list[float] | None = None) -> Prediction:
    """Advance ``pred`` to ``target_level`` in the coarse-to-fine hierarchy.

    ``raw_params`` supplies the (fine) ground-truth-ish parameter estimate the
    stage produces; it is quantised to the stage's grid so early stages commit
    only coarse values and later stages sharpen them. Refinement may only move
    forward through :data:`REFINEMENT_LEVELS`.
    """
    if target_level not in REFINEMENT_LEVELS:
        raise ValueError(f"unknown refinement level: {target_level!r}")
    cur = REFINEMENT_LEVELS.index(pred.level)
    nxt = REFINEMENT_LEVELS.index(target_level)
    if nxt < cur:
        raise ValueError(
            f"cannot refine backward from {pred.level!r} to {target_level!r}")

    if target_level == "type":
        return Prediction(pred.kind, pred.exists, list(pred.params), "type")

    source = raw_params if raw_params is not None else pred.params
    grid = _LEVEL_GRID[target_level]
    snapped = [_snap(float(v), grid) for v in source] if grid else list(source)
    return Prediction(pred.kind, pred.exists, snapped, target_level)


def run_pipeline(kind: str, raw_params: list[float]) -> Prediction:
    """Drive a prediction through all three coarse-to-fine stages in order."""
    pred = Prediction(kind, level="type")
    pred = refine(pred, "type")
    pred = refine(pred, "coarse_params", raw_params)
    pred = refine(pred, "fine_params", raw_params)
    return pred


# ---------------------------------------------------------------------------
# Ablation reconstruction (paper Tab. 4): stage importance ordering.
# ---------------------------------------------------------------------------

# Accuracy drop (absolute) attributed to omitting a stage, from the paper text:
# "~12% without Stage 1", "~15% without Stage 2".
_ABLATION_DROP = {"primitive_perception": 0.12, "structural_perception": 0.15}


def ablation_accuracy(base_accuracy: float, omitted: str | None) -> float:
    """Accuracy when ``omitted`` stage is skipped (``None`` = full pipeline)."""
    if omitted is None:
        return base_accuracy
    if omitted not in _ABLATION_DROP:
        raise ValueError(f"stage not ablatable: {omitted!r}")
    return base_accuracy - _ABLATION_DROP[omitted]
