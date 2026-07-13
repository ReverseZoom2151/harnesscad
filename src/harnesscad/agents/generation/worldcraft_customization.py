"""Object-customization parameter schema for placed scene assets.

Paper: *WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM
Agents* (Liu, Tang, Tai), Sec. 3.2 (ForgeIt).

WorldCraft lets a user *customize individual objects* -- their geometry, scale,
material, texture, colour and pose -- through natural-language edits that ForgeIt
turns into precise parameter changes, refined over multi-turn conversation. The
LLM/critic loop and procedural mesh generation are external; what is
deterministic and locally buildable is the **typed customization schema**: the
set of attributes that describe a legal customization, their validation, and the
pure function that applies a customization to an
:class:`reconstruction.worldcraft_layout_spec.ObjectPlacement`.

This module is DISTINCT from the scene-graph modules (which annotate derived
relations) and from the layout solver (which searches poses): it defines *what a
single object's editable attributes are* and applies edits deterministically.

Provides:

* :class:`Color` -- an RGBA colour with 0..1 channel validation and hex parsing;
* :class:`MaterialSpec` -- PBR-style material parameters (base colour, metallic,
  roughness, optional texture name/emission) with range validation;
* :class:`CustomizationSchema` -- the closed set of customizable attribute names
  and their allowed value types / ranges, with a :meth:`validate` method;
* :class:`ObjectCustomization` -- a concrete, validated bundle of attribute
  overrides (scale factor, material, colour, yaw, label, tags);
* :func:`apply_customization` -- returns a NEW customized ``ObjectPlacement``
  (the input is never mutated), composing scale/yaw with the existing pose and
  writing material/colour into the placement attributes.
* :func:`merge_customizations` -- fold a sequence of edits (multi-turn refinement)
  into a single effective customization, later edits overriding earlier ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.reconstruction.worldcraft_layout_spec import ObjectPlacement, Pose


# --------------------------------------------------------------------------- #
# Colour                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Color:
    """RGBA colour with channels in ``[0, 1]``."""

    r: float
    g: float
    b: float
    a: float = 1.0

    def __post_init__(self) -> None:
        for name, v in (("r", self.r), ("g", self.g), ("b", self.b), ("a", self.a)):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"colour channel {name}={v} outside [0, 1]")

    @staticmethod
    def from_hex(text: str) -> "Color":
        """Parse ``#rrggbb`` or ``#rrggbbaa`` (case-insensitive, ``#`` optional)."""
        s = text.lstrip("#")
        if len(s) not in (6, 8):
            raise ValueError(f"expected 6 or 8 hex digits, got {text!r}")
        vals = [int(s[i:i + 2], 16) / 255.0 for i in range(0, len(s), 2)]
        if len(vals) == 3:
            vals.append(1.0)
        return Color(*vals)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)


# --------------------------------------------------------------------------- #
# Material                                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MaterialSpec:
    """PBR-style material parameters attached to a customized object."""

    base_color: Color = field(default_factory=lambda: Color(0.8, 0.8, 0.8, 1.0))
    metallic: float = 0.0
    roughness: float = 0.5
    texture: Optional[str] = None
    emission: float = 0.0

    def __post_init__(self) -> None:
        for name, v in (("metallic", self.metallic), ("roughness", self.roughness)):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name}={v} outside [0, 1]")
        if self.emission < 0.0:
            raise ValueError("emission must be non-negative")

    def to_dict(self) -> Dict[str, object]:
        return {
            "base_color": list(self.base_color.to_tuple()),
            "metallic": self.metallic,
            "roughness": self.roughness,
            "texture": self.texture,
            "emission": self.emission,
        }


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #
# Closed set of customizable attribute names ForgeIt exposes for an object.
_SCALE = "scale"
_MATERIAL = "material"
_COLOR = "color"
_YAW = "yaw"
_LABEL = "label"
_TAGS = "tags"

_ATTRIBUTE_NAMES = (_SCALE, _MATERIAL, _COLOR, _YAW, _LABEL, _TAGS)


class CustomizationSchema:
    """Validates that a raw attribute dict is a legal object customization.

    ``min_scale`` / ``max_scale`` bound the uniform scale factor. Unknown
    attribute names are rejected so that a mistyped edit fails loudly instead of
    being silently dropped.
    """

    def __init__(self, min_scale: float = 0.05, max_scale: float = 20.0) -> None:
        if not (0.0 < min_scale <= max_scale):
            raise ValueError("require 0 < min_scale <= max_scale")
        self.min_scale = min_scale
        self.max_scale = max_scale

    @property
    def attribute_names(self) -> Tuple[str, ...]:
        return _ATTRIBUTE_NAMES

    def validate(self, attributes: Dict[str, object]) -> List[str]:
        """Return a list of error strings (empty when the customization is legal)."""
        errors: List[str] = []
        for key in attributes:
            if key not in _ATTRIBUTE_NAMES:
                errors.append(f"unknown attribute {key!r}")
        if _SCALE in attributes:
            sc = attributes[_SCALE]
            if not isinstance(sc, (int, float)) or isinstance(sc, bool):
                errors.append("scale must be a number")
            elif not (self.min_scale <= float(sc) <= self.max_scale):
                errors.append(f"scale {sc} outside [{self.min_scale}, {self.max_scale}]")
        if _YAW in attributes:
            yaw = attributes[_YAW]
            if not isinstance(yaw, (int, float)) or isinstance(yaw, bool):
                errors.append("yaw must be a number (radians)")
        if _MATERIAL in attributes and not isinstance(attributes[_MATERIAL], MaterialSpec):
            errors.append("material must be a MaterialSpec")
        if _COLOR in attributes and not isinstance(attributes[_COLOR], Color):
            errors.append("color must be a Color")
        if _LABEL in attributes and not isinstance(attributes[_LABEL], str):
            errors.append("label must be a string")
        if _TAGS in attributes:
            tags = attributes[_TAGS]
            if not isinstance(tags, (list, tuple)) or not all(isinstance(t, str) for t in tags):
                errors.append("tags must be a sequence of strings")
        return errors

    def is_valid(self, attributes: Dict[str, object]) -> bool:
        return not self.validate(attributes)


# --------------------------------------------------------------------------- #
# Concrete customization                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObjectCustomization:
    """A validated bundle of attribute overrides for one placed object."""

    scale: Optional[float] = None
    material: Optional[MaterialSpec] = None
    color: Optional[Color] = None
    yaw: Optional[float] = None
    label: Optional[str] = None
    tags: Tuple[str, ...] = ()

    def as_attribute_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        if self.scale is not None:
            out[_SCALE] = self.scale
        if self.material is not None:
            out[_MATERIAL] = self.material
        if self.color is not None:
            out[_COLOR] = self.color
        if self.yaw is not None:
            out[_YAW] = self.yaw
        if self.label is not None:
            out[_LABEL] = self.label
        if self.tags:
            out[_TAGS] = list(self.tags)
        return out

    def validated(self, schema: Optional[CustomizationSchema] = None) -> "ObjectCustomization":
        """Raise ``ValueError`` if this customization violates ``schema``."""
        schema = schema or CustomizationSchema()
        errors = schema.validate(self.as_attribute_dict())
        if errors:
            raise ValueError("invalid customization: " + "; ".join(errors))
        return self


def merge_customizations(edits: Sequence[ObjectCustomization]) -> ObjectCustomization:
    """Fold multi-turn edits into one; later non-null fields override earlier.

    ``tags`` accumulate (order-stable, de-duplicated) across all edits, matching
    the additive nature of tagging in an iterative refinement conversation.
    """
    scale = material = color = yaw = label = None
    tags: List[str] = []
    for e in edits:
        if e.scale is not None:
            scale = e.scale
        if e.material is not None:
            material = e.material
        if e.color is not None:
            color = e.color
        if e.yaw is not None:
            yaw = e.yaw
        if e.label is not None:
            label = e.label
        for t in e.tags:
            if t not in tags:
                tags.append(t)
    return ObjectCustomization(
        scale=scale, material=material, color=color, yaw=yaw, label=label, tags=tuple(tags)
    )


def apply_customization(
    placement: ObjectPlacement,
    customization: ObjectCustomization,
    *,
    schema: Optional[CustomizationSchema] = None,
) -> ObjectPlacement:
    """Return a NEW placement with the customization applied (input untouched).

    Scale multiplies the pose's existing per-axis scale uniformly; yaw sets the
    absolute vertical rotation. Material, colour, label and tags are written into
    the placement's ``attributes`` under stable keys.
    """
    customization.validated(schema)

    pose: Pose = placement.pose
    if customization.scale is not None:
        sx, sy, sz = pose.scale
        f = customization.scale
        pose = pose.with_scale(sx * f, sy * f, sz * f)
    if customization.yaw is not None:
        ox, oy, _ = pose.orientation
        pose = replace(pose, orientation=(ox, oy, customization.yaw))

    attrs: Dict[str, object] = dict(placement.attributes)
    if customization.material is not None:
        attrs["material"] = customization.material
    if customization.color is not None:
        attrs["color"] = customization.color
    if customization.label is not None:
        attrs["label"] = customization.label
    if customization.tags:
        existing = list(attrs.get("tags", []))  # type: ignore[arg-type]
        for t in customization.tags:
            if t not in existing:
                existing.append(t)
        attrs["tags"] = existing

    return ObjectPlacement(
        object_id=placement.object_id,
        category=placement.category,
        half_extent=placement.half_extent,
        pose=pose,
        parent_id=placement.parent_id,
        attributes=attrs,
    )
