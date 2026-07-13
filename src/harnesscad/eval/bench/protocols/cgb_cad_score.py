"""CAD Score composition: validity gate, axis weights, editing renormalization.

CADGenBench composes one candidate's score from three orthogonal axes, gated by
validity:

    cad_score = 0                                                if not valid
              = 0.4*shape + 0.4*interface + 0.2*topology         (generation)
              = 0.6*shape_renorm + 0.3*interface + 0.1*topology  (editing)

Three definitions here that the harness does not already encode:

**1. Weight renormalization over present axes.** A sample without authored
mating sub-volumes simply has no interface axis. Its weight is redistributed
over the axes that *are* present rather than diluting the mean with a zero, so
a sample cannot be punished for a metric its fixture never defined.

**2. Validity as a hard gate, not a term.** An invalid solid scores exactly
zero, so it can never outrank a worse-but-valid one. (The harness's existing
validity work - ``verifiers.geometry.BRepValidityCheck``,
``bench.text2cad2_invalidity_ratio`` - reports validity; none of it *gates* a
composite score.)

**3. Editing no-op renormalization.** For an editing sample the unedited input
is already a near-GT solid, so all three global-similarity axes score it high:
the raw composition would reward doing nothing. The shape axis is therefore
renormalized against the no-op baseline ``b_shape = shape(input, GT)``:

    s_renorm = max(0, (s_raw - b_shape) / (1 - b_shape))

which maps the no-op to 0 and a perfect candidate to 1. Topology and interface
stay **raw** (most edits do not move them, and a candidate that breaks them
should still be penalized), and the editing weights are shape-dominant, which
caps a no-op at ``0.3 + 0.1 = 0.4``.

``b_shape`` is a per-sample constant (it depends only on the input, the GT and
the metric code), so it is precomputed at authoring time and committed. Two
guards travel with it: an authoring **headroom floor** (an edit whose
``1 - b_shape`` is below :data:`EDIT_HEADROOM_FLOOR` cannot be resolved by the
renormalized axis and the fixture is rejected) and a **version stamp** so a
stale baseline hard-errors instead of silently scoring against old math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

STATUS_VALID = "valid"
STATUS_INVALID = "invalid"
STATUS_MISSING = "missing"

GENERATION_AXIS_WEIGHTS: Dict[str, float] = {
    "shape": 0.4,
    "interface": 0.4,
    "topology": 0.2,
}

EDITING_AXIS_WEIGHTS: Dict[str, float] = {
    "shape": 0.6,
    "interface": 0.3,
    "topology": 0.1,
}

# Authoring gate: an editing fixture must leave the shape metric at least this
# much headroom (1 - b_shape). Below it the no-op already scores essentially
# perfectly, so the renormalized axis carries noise rather than signal.
EDIT_HEADROOM_FLOOR = 2e-3

AXES = ("shape", "interface", "topology")


class StaleBaselineError(RuntimeError):
    """A committed no-op baseline was produced by a different metric version."""


class EditHeadroomError(ValueError):
    """An editing fixture leaves too little shape headroom to be scorable."""


@dataclass(frozen=True)
class AxisScores:
    """The three axis values for one candidate; ``None`` means "not applicable"."""

    shape: Optional[float] = None
    interface: Optional[float] = None
    topology: Optional[float] = None

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "shape": self.shape,
            "interface": self.interface,
            "topology": self.topology,
        }


def shape_similarity(
    surface_distance_f1: Optional[float], volume_iou: Optional[float]
) -> Optional[float]:
    """The shape axis: the plain **mean** of its two sub-metrics.

    Surface-distance F1 places the surfaces; volume IoU places the material. A
    candidate has to satisfy both. A sub-metric that could not be computed drops
    out and the other stands alone; if neither is available the axis is absent
    (``None``), not zero.
    """
    parts = [v for v in (surface_distance_f1, volume_iou) if v is not None]
    if not parts:
        return None
    return sum(float(v) for v in parts) / len(parts)


def renormalize_shape(raw_shape: float, baseline_shape: float) -> float:
    """Map the raw shape score onto the no-op-anchored ``[0, 1]`` scale."""
    headroom = 1.0 - baseline_shape
    if headroom <= 0.0:
        return 1.0 if raw_shape > baseline_shape else 0.0
    return max(0.0, min(1.0, (raw_shape - baseline_shape) / headroom))


def check_edit_headroom(baseline_shape: float, *, floor: float = EDIT_HEADROOM_FLOOR) -> float:
    """Return the headroom ``1 - b_shape``, raising when it is below *floor*."""
    headroom = 1.0 - float(baseline_shape)
    if headroom < floor:
        raise EditHeadroomError(
            f"editing fixture leaves headroom {headroom:.6f} < {floor}: the no-op "
            "already scores essentially perfectly, so the renormalized shape axis "
            "cannot resolve the edit"
        )
    return headroom


def check_baseline_fresh(baseline: dict, version: str, *, fixture: str = "") -> None:
    """Raise :class:`StaleBaselineError` when the baseline's version stamp moved.

    The committed ``b_shape`` is only valid for the shape/alignment code that
    produced it; scoring against a stale number silently corrupts every editing
    sample, so a mismatch fails loud.
    """
    stamped = baseline.get("metric_version")
    if stamped != version:
        raise StaleBaselineError(
            f"no-op baseline for fixture {fixture!r} was computed with metric "
            f"version {stamped!r} but the grader runs {version!r}; regenerate it"
        )


def build_edit_baseline(
    *,
    baseline_shape: float,
    version: str,
    surface_distance_f1: Optional[float] = None,
    volume_iou: Optional[float] = None,
    alignment_rmse: Optional[float] = None,
) -> dict:
    """Assemble a committed no-op baseline record (headroom gate applied)."""
    headroom = check_edit_headroom(baseline_shape)
    return {
        "shape_similarity_score": float(baseline_shape),
        "shape_surface_distance_f1": surface_distance_f1,
        "shape_volume_iou": volume_iou,
        "alignment_rmse": (
            round(float(alignment_rmse), 4) if alignment_rmse is not None else None
        ),
        "headroom": round(headroom, 6),
        "metric_version": version,
    }


def weighted_axis_mean(
    axes: AxisScores, weights: Optional[Dict[str, float]] = None
) -> Optional[float]:
    """Weighted mean over the axes that are present, weights renormalized.

    ``weights=None`` means equal weighting (a plain arithmetic mean). Returns
    ``None`` when no axis applies at all.
    """
    values = axes.as_dict()
    num = 0.0
    den = 0.0
    for axis in AXES:
        value = values[axis]
        if value is None:
            continue
        w = 1.0 if weights is None else float(weights.get(axis, 0.0))
        num += w * float(value)
        den += w
    if den == 0.0:
        return None
    return num / den


def cad_score(
    axes: AxisScores,
    *,
    is_valid: bool,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """One candidate's CAD Score in ``[0, 1]``; ``0`` when the validity gate fails."""
    if not is_valid:
        return 0.0
    score = weighted_axis_mean(axes, weights if weights is not None else GENERATION_AXIS_WEIGHTS)
    return 0.0 if score is None else score


def generation_score(
    axes: AxisScores, *, is_valid: bool = True
) -> float:
    """CAD Score under the generation composition (0.4 / 0.4 / 0.2)."""
    return cad_score(axes, is_valid=is_valid, weights=GENERATION_AXIS_WEIGHTS)


def editing_score(
    axes: AxisScores,
    *,
    baseline_shape: float,
    is_valid: bool = True,
) -> float:
    """CAD Score under the editing composition (renormalized shape, 0.6/0.3/0.1)."""
    if not is_valid:
        return 0.0
    renormalized = (
        None
        if axes.shape is None
        else renormalize_shape(float(axes.shape), float(baseline_shape))
    )
    renormed_axes = AxisScores(
        shape=renormalized, interface=axes.interface, topology=axes.topology
    )
    return cad_score(renormed_axes, is_valid=True, weights=EDITING_AXIS_WEIGHTS)


def noop_ceiling(weights: Optional[Dict[str, float]] = None) -> float:
    """The best a no-op edit can score: everything but the (renormalized) shape axis.

    With :data:`EDITING_AXIS_WEIGHTS` this is ``0.3 + 0.1 = 0.4``, which is the
    whole point of the reweighting: any real shape improvement clears it.
    """
    w = EDITING_AXIS_WEIGHTS if weights is None else weights
    return float(w.get("interface", 0.0)) + float(w.get("topology", 0.0))
