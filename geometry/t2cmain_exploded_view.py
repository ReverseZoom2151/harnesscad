"""Deterministic exploded-view layout for an occurrence-structured assembly.

Ported from ``packages/cadjs/src/lib/viewer/explodedView.js`` of the ``text-to-cad``
(CAD Skills) repository.  Given the axis-aligned bounds of every part of an
assembly plus each part's occurrence id (see
``programs.t2cmain_cad_ref_selectors``), it produces the translation each part
must receive so the assembly reads as an exploded drawing.  The harness had no
exploded-view / assembly-presentation solver.

Two strategies, both purely geometric:

**Axis explosion.**  Parts are grouped by occurrence prefix, groups are sorted
along the chosen axis, and near-coplanar groups are optionally merged into a
single *layer* (tolerance derived from the model span and the thickest group, so
it scales with the model rather than being an absolute epsilon).  Layers are then
pushed apart one at a time by *stacking*: each layer is translated only far
enough that its leading face clears the trailing face of the layer before it,
plus a gap.  The gap is ``max(minimum_gap, min(thickness_prev, thickness_cur) *
0.22 * spacing)`` -- so thin shims do not get pushed as far as thick housings,
and the explosion never self-intersects regardless of part sizes.  Distances are
therefore never negative and layer order is preserved.

**Radial explosion.**  Groups fly outward along the direction from the model
centroid to the group centroid.  A group centred exactly on the model centroid
has no such direction, so it is assigned a *golden-angle spiral* direction
(``pi * (3 - sqrt(5))`` around a uniformly-spaced z), which distributes the
degenerate cases evenly over the sphere instead of piling them on one axis.

Both strategies honour ``keep_base_grounded``: the lowest layer stays put (axis
mode), or the whole explosion is lifted so nothing sinks below the original
model floor (radial mode).  ``translation_at_progress`` and
``ease_progress`` (cubic ease-out) let the result be animated deterministically.

Stdlib-only, no scene graph, no randomness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from programs.t2cmain_cad_ref_selectors import (
    common_occurrence_prefix,
    occurrence_segments,
)

Vector3 = Tuple[float, float, float]
Bounds = Tuple[Vector3, Vector3]

EPSILON = 1e-6
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
MAX_DEPTH = 4
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class Part:
    """One explodable part: an occurrence id and its axis-aligned bounds."""

    part_id: str
    bounds: Bounds


@dataclass(frozen=True)
class ExplodedViewSettings:
    """Normalised layout settings."""

    axis: str = "z"
    direction: str = "positive"
    spacing: float = 1.45
    depth: int = 1
    keep_base_grounded: bool = True
    merge_coplanar: bool = True


@dataclass(frozen=True)
class PartState:
    """The computed translation for one part."""

    part_id: str
    group_key: str
    layer_index: int
    direction: Vector3
    distance: float
    translation: Vector3


def normalize_settings(**overrides) -> ExplodedViewSettings:
    """Clamp user settings into the supported ranges."""
    axis = str(overrides.get("axis", "z")).strip().lower().lstrip("-+")
    if axis not in AXIS_INDEX and axis != "radial":
        axis = "z"
    raw_axis = str(overrides.get("axis", "")).strip()
    direction = str(
        overrides.get("direction") or ("negative" if raw_axis.startswith("-") else "positive")
    ).strip().lower()
    if direction not in ("positive", "negative"):
        direction = "positive"
    try:
        spacing = float(overrides.get("spacing", 1.45))
    except (TypeError, ValueError):
        spacing = 1.45
    spacing = min(max(spacing, 0.25), 4.0)
    try:
        depth = int(overrides.get("depth", 1))
    except (TypeError, ValueError):
        depth = 1
    depth = min(max(depth, 1), MAX_DEPTH)
    return ExplodedViewSettings(
        axis=axis,
        direction=direction,
        spacing=spacing,
        depth=depth,
        keep_base_grounded=bool(overrides.get("keep_base_grounded", True)),
        merge_coplanar=bool(overrides.get("merge_coplanar", True)),
    )


def merge_bounds(bounds_list: Sequence[Optional[Bounds]]) -> Optional[Bounds]:
    entries = [entry for entry in bounds_list if entry is not None]
    if not entries:
        return None
    return (
        (
            min(entry[0][0] for entry in entries),
            min(entry[0][1] for entry in entries),
            min(entry[0][2] for entry in entries),
        ),
        (
            max(entry[1][0] for entry in entries),
            max(entry[1][1] for entry in entries),
            max(entry[1][2] for entry in entries),
        ),
    )


def bounds_center(bounds: Optional[Bounds]) -> Vector3:
    if bounds is None:
        return (0.0, 0.0, 0.0)
    return tuple(
        (bounds[0][axis] + bounds[1][axis]) / 2.0 for axis in range(3)
    )  # type: ignore[return-value]


def bounds_size(bounds: Optional[Bounds], axis: int) -> float:
    if bounds is None:
        return 0.0
    return max(bounds[1][axis] - bounds[0][axis], 0.0)


def bounds_radius(bounds: Optional[Bounds]) -> float:
    if bounds is None:
        return 0.0
    return (
        math.sqrt(sum(bounds_size(bounds, axis) ** 2 for axis in range(3))) / 2.0
    )


def shift_bounds(bounds: Bounds, translation: Vector3, amount: float = 1.0) -> Bounds:
    return (
        tuple(bounds[0][axis] + translation[axis] * amount for axis in range(3)),
        tuple(bounds[1][axis] + translation[axis] * amount for axis in range(3)),
    )  # type: ignore[return-value]


def group_key(part_id: str, depth: int = 1, common_prefix: Sequence[str] = ()) -> str:
    """Occurrence prefix that identifies a part's exploded-view group.

    The key keeps ``len(common_prefix) + depth`` segments (never more than the
    part itself has).  Parts with no occurrence id group by their raw id.
    """
    text = str(part_id or "").strip()
    segments = occurrence_segments(text)
    if not segments:
        return text
    safe_depth = min(max(int(depth), 1), MAX_DEPTH)
    prefix_length = min(len(common_prefix), max(len(segments) - 1, 0))
    group_length = min(len(segments), prefix_length + safe_depth)
    return ".".join(segments[:group_length]) or text


@dataclass
class _Group:
    key: str
    order: int
    parts: List[Part]
    bounds: Optional[Bounds]


def _build_groups(parts: Sequence[Part], settings: ExplodedViewSettings) -> List[_Group]:
    prefix = common_occurrence_prefix([part.part_id for part in parts])
    groups: Dict[str, _Group] = {}
    for index, part in enumerate(parts):
        key = group_key(part.part_id, settings.depth, prefix) or f"part:{index}"
        group = groups.get(key)
        if group is None:
            group = _Group(key=key, order=index, parts=[], bounds=None)
            groups[key] = group
        group.parts.append(part)
    ordered = sorted(groups.values(), key=lambda group: group.order)
    for group in ordered:
        group.bounds = merge_bounds([part.bounds for part in group.parts])
    return [group for group in ordered if group.bounds is not None]


def _fallback_direction(index: int, count: int) -> Vector3:
    """Golden-angle spiral point on the unit sphere -- an even spread of the
    directions for groups that sit exactly on the model centroid."""
    total = max(1, count)
    z = 1.0 - ((index + 0.5) / total) * 2.0 if total > 1 else 0.0
    radius = math.sqrt(max(0.0, 1.0 - z * z))
    angle = index * GOLDEN_ANGLE
    vector = (math.cos(angle) * radius, math.sin(angle) * radius, z)
    length = math.sqrt(sum(component ** 2 for component in vector))
    if length <= EPSILON:
        return (1.0, 0.0, 0.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _radial_states(
    groups: List[_Group], bounds: Bounds, settings: ExplodedViewSettings
) -> List[PartState]:
    model_center = bounds_center(bounds)
    model_radius = max(bounds_radius(bounds), EPSILON)
    spacing = settings.spacing
    base_distance = max(model_radius * 0.32 * spacing, model_radius * 0.12)
    sign = -1.0 if settings.direction == "negative" else 1.0

    states: List[PartState] = []
    for index, group in enumerate(groups):
        center = bounds_center(group.bounds)
        offset = tuple(center[axis] - model_center[axis] for axis in range(3))
        length = math.sqrt(sum(component ** 2 for component in offset))
        if length <= EPSILON:
            direction = _fallback_direction(index, len(groups))
        else:
            direction = tuple(component / length for component in offset)
        direction = tuple(component * sign for component in direction)

        group_radius = bounds_radius(group.bounds)
        travel = base_distance + min(
            group_radius * 0.18 * spacing, model_radius * 0.2 * spacing
        )
        translation = tuple(component * travel for component in direction)
        for part in group.parts:
            states.append(
                PartState(
                    part_id=part.part_id,
                    group_key=group.key,
                    layer_index=index,
                    direction=direction,  # type: ignore[arg-type]
                    distance=travel,
                    translation=translation,  # type: ignore[arg-type]
                )
            )

    if settings.keep_base_grounded and states:
        floor = bounds[0][2]
        by_id = {part.part_id: part for group in groups for part in group.parts}
        exploded = merge_bounds(
            [
                shift_bounds(by_id[state.part_id].bounds, state.translation)
                for state in states
            ]
        )
        if exploded is not None and exploded[0][2] < floor - EPSILON:
            model_max_size = max(
                max(bounds_size(bounds, axis) for axis in range(3)), EPSILON
            )
            lift = min(floor - exploded[0][2], model_max_size * settings.spacing)
            lifted: List[PartState] = []
            for state in states:
                translation = (
                    state.translation[0],
                    state.translation[1],
                    state.translation[2] + lift,
                )
                distance = math.sqrt(sum(c ** 2 for c in translation))
                direction = (
                    tuple(c / distance for c in translation)
                    if distance > EPSILON
                    else state.direction
                )
                lifted.append(
                    PartState(
                        part_id=state.part_id,
                        group_key=state.group_key,
                        layer_index=state.layer_index,
                        direction=direction,  # type: ignore[arg-type]
                        distance=distance,
                        translation=translation,
                    )
                )
            states = lifted
    return states


def solve_exploded_view(
    parts: Sequence[Part], bounds: Optional[Bounds] = None, **options
) -> Tuple[PartState, ...]:
    """Compute the exploded translation of every part.

    Fewer than two explodable parts, or fewer than two distinct groups, yields an
    empty result -- there is nothing to explode.
    """
    settings = normalize_settings(**options)
    explodable = [
        part
        for part in parts
        if str(part.part_id or "").strip() and part.part_id != "__model__"
    ]
    if len(explodable) < 2:
        return ()
    model_bounds = bounds or merge_bounds([part.bounds for part in explodable])
    if model_bounds is None:
        return ()

    groups = _build_groups(explodable, settings)
    if len(groups) < 2:
        return ()

    if settings.axis == "radial":
        return tuple(_radial_states(groups, model_bounds, settings))

    axis_index = AXIS_INDEX[settings.axis]
    model_radius = max(bounds_radius(model_bounds), EPSILON)
    model_span = max(bounds_size(model_bounds, axis_index), EPSILON)

    ordered = sorted(
        groups,
        key=lambda group: (bounds_center(group.bounds)[axis_index], group.order),
    )
    thicknesses = [bounds_size(group.bounds, axis_index) for group in ordered]
    max_thickness = max(thicknesses + [model_span * 0.08, EPSILON])
    positive = sorted(value for value in thicknesses if value > EPSILON)
    median_thickness = (
        positive[len(positive) // 2] if positive else max_thickness
    )
    layer_tolerance = max(model_span * 0.025, max_thickness * 0.18, EPSILON)
    spacing = settings.spacing
    minimum_gap = (
        max(model_span * 0.075, median_thickness * 0.35, model_radius * 0.035, EPSILON)
        * spacing
    )
    sign = -1.0 if settings.direction == "negative" else 1.0
    axis_vector = tuple(
        sign if axis == axis_index else 0.0 for axis in range(3)
    )  # type: ignore[assignment]

    layers: List[Dict] = []
    for group in ordered:
        center = bounds_center(group.bounds)[axis_index]
        previous = layers[-1] if layers else None
        if (
            settings.merge_coplanar
            and previous is not None
            and abs(center - previous["center"]) <= layer_tolerance
        ):
            previous["groups"].append(group)
            previous["bounds"] = merge_bounds([previous["bounds"], group.bounds])
            count = len(previous["groups"])
            previous["center"] = (
                previous["center"] * (count - 1) + center
            ) / count
        else:
            layers.append({"center": center, "bounds": group.bounds, "groups": [group]})

    def layer_gap(previous: Optional[Dict], current: Optional[Dict]) -> float:
        previous_thickness = (
            bounds_size(previous["bounds"], axis_index)
            if previous is not None
            else median_thickness
        )
        current_thickness = (
            bounds_size(current["bounds"], axis_index)
            if current is not None
            else median_thickness
        )
        return max(
            minimum_gap, min(previous_thickness, current_thickness) * 0.22 * spacing
        )

    states: List[PartState] = []
    previous_max: Optional[float] = None
    previous_layer: Optional[Dict] = None
    for layer_index, layer in enumerate(layers):
        layer_min = layer["bounds"][0][axis_index]
        layer_max = layer["bounds"][1][axis_index]
        if settings.keep_base_grounded and layer_index == 0:
            axis_distance = 0.0
            previous_max = layer_max
        else:
            target_min = (
                layer_min
                if previous_max is None
                else previous_max + layer_gap(previous_layer, layer)
            )
            axis_distance = max(0.0, target_min - layer_min)
            previous_max = layer_max + axis_distance
        previous_layer = layer

        translation = tuple(
            component * axis_distance for component in axis_vector
        )  # type: ignore[assignment]
        length = math.sqrt(sum(component ** 2 for component in translation))
        direction = (
            tuple(component / length for component in translation)
            if length > EPSILON
            else axis_vector
        )
        for group in layer["groups"]:
            for part in group.parts:
                states.append(
                    PartState(
                        part_id=part.part_id,
                        group_key=group.key,
                        layer_index=layer_index,
                        direction=direction,  # type: ignore[arg-type]
                        distance=length,
                        translation=translation,  # type: ignore[arg-type]
                    )
                )
    return tuple(states)


def ease_progress(value: float) -> float:
    """Cubic ease-out on ``[0, 1]``."""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        amount = 0.0
    amount = min(max(amount, 0.0), 1.0)
    return 1.0 - (1.0 - amount) ** 3


def translation_at_progress(state: PartState, progress: float = 1.0) -> Vector3:
    """Linear interpolation of a part's translation at animation ``progress``."""
    try:
        amount = float(progress)
    except (TypeError, ValueError):
        amount = 0.0
    amount = min(max(amount, 0.0), 1.0)
    return tuple(component * amount for component in state.translation)  # type: ignore[return-value]


def exploded_bounds(
    parts: Sequence[Part],
    states: Sequence[PartState],
    progress: float = 1.0,
    fallback: Optional[Bounds] = None,
) -> Optional[Bounds]:
    """Bounds of the whole assembly at a given explosion progress."""
    by_id = {part.part_id: part for part in parts}
    shifted = [
        shift_bounds(
            by_id[state.part_id].bounds, translation_at_progress(state, progress)
        )
        for state in states
        if state.part_id in by_id
    ]
    return merge_bounds(shifted) or fallback
