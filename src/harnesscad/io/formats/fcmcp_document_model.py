"""fcmcp_document_model -- FreeCAD document-object-model wire codec.

Transferred from the ``get_document_context`` serialiser of ``freecad_mcp`` (an
MCP server that exposes a live FreeCAD session to an LLM agent). The live RPC
bridge, the GUI workbench and the two arbitrary-``exec`` tools it registers
(``send_command`` / ``run_script``) are out of scope -- the harness already owns
a real MCP server (:mod:`surfaces.mcp`). What is *not* already in the harness is
the FreeCAD-native **document-object-model encoding**: the exact JSON shape the
agent receives for a FreeCAD ``ActiveDocument`` and the placement / view maths
that shape encodes.

That shape is FreeCAD-specific and distinct from anything in the harness:

  * ``surfaces.mcp`` encodes generic CISP ops + a ``cad://model/tree`` state, not
    a FreeCAD object tree (``TypeId`` strings, the ``Name`` vs ``Label``
    distinction, ``ViewObject.Visibility``, ``Placement`` as axis-angle);
  * ``backends.ocp_occt_api_catalog`` catalogues OCCT *API symbols*, not a
    document instance.

This module models that wire format host-free:

  * :class:`Rotation` -- FreeCAD stores a placement rotation as a unit **axis +
    angle** (``rot.Axis`` / ``rot.Angle``); this carries the deterministic
    axis-angle <-> quaternion maths (Coin3D camera orientations are quaternions),
    axis normalisation, and identity detection.
  * :class:`Placement`, :class:`ShapeInfo`, :class:`FreeCADObject`,
    :class:`DocumentInfo`, :class:`ViewState`, :class:`DocumentContext` -- the
    object tree.
  * :func:`encode_document_context` / :func:`decode_document_context` -- reproduce
    and invert the exact ``get_document_context`` JSON, round-trip stable.
  * :func:`parse_type_id` -- split a FreeCAD ``Namespace::Class`` TypeId.
  * :func:`validate_context` -- deterministic structural checks (object_count
    agreement, well-formed TypeIds, unit-ish axes).

Stdlib only, deterministic: same document in, same JSON out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "Rotation",
    "Placement",
    "ShapeInfo",
    "FreeCADObject",
    "DocumentInfo",
    "ViewState",
    "DocumentContext",
    "parse_type_id",
    "encode_document_context",
    "decode_document_context",
    "validate_context",
    "axis_angle_to_quaternion",
    "quaternion_to_axis_angle",
    "normalize_axis",
]

Vec3 = Tuple[float, float, float]

# FreeCAD's identity rotation reports axis (0, 0, 1) with angle 0.
_DEFAULT_AXIS: Vec3 = (0.0, 0.0, 1.0)
_EPS = 1e-12


# ===========================================================================
# Axis-angle / quaternion maths (deterministic; the placement + camera core)
# ===========================================================================
def normalize_axis(axis: Vec3) -> Vec3:
    """Return ``axis`` scaled to unit length.

    A (near-)zero vector -- which FreeCAD produces for an identity rotation --
    normalises to the default ``(0, 0, 1)`` axis, matching ``rot.Axis``.
    """
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    n = math.sqrt(x * x + y * y + z * z)
    if n <= _EPS:
        return _DEFAULT_AXIS
    return (x / n, y / n, z / n)


def axis_angle_to_quaternion(axis: Vec3, angle: float) -> Tuple[float, float, float, float]:
    """Axis (unit) + angle (radians) -> quaternion ``(x, y, z, w)``.

    Uses FreeCAD/Coin3D's ``(x, y, z, w)`` component order, with
    ``w = cos(angle/2)`` and ``(x, y, z) = axis * sin(angle/2)``.
    """
    ux, uy, uz = normalize_axis(axis)
    half = float(angle) / 2.0
    s = math.sin(half)
    return (ux * s, uy * s, uz * s, math.cos(half))


def quaternion_to_axis_angle(q: Tuple[float, float, float, float]) -> Tuple[Vec3, float]:
    """Quaternion ``(x, y, z, w)`` -> (unit axis, angle in ``[0, pi]``).

    The quaternion is normalised first. A pure-scalar quaternion (no rotation)
    yields the default axis with angle 0, matching FreeCAD's identity report.
    """
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= _EPS:
        return (_DEFAULT_AXIS, 0.0)
    x, y, z, w = x / n, y / n, z / n, w / n
    # Clamp w into range before acos to stay deterministic against FP drift.
    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s <= _EPS:
        return (_DEFAULT_AXIS, 0.0)
    return ((x / s, y / s, z / s), angle)


# ===========================================================================
# Value objects
# ===========================================================================
@dataclass(frozen=True)
class Rotation:
    """A FreeCAD placement rotation, stored as FreeCAD stores it: axis + angle.

    ``rotation`` serialises to the ``[axis.x, axis.y, axis.z, angle]`` list the
    ``freecad_mcp`` bridge emits.
    """

    axis: Vec3 = _DEFAULT_AXIS
    angle: float = 0.0  # radians

    @property
    def is_identity(self) -> bool:
        return abs(self.angle) <= _EPS

    def unit(self) -> "Rotation":
        """Return an equivalent rotation with a unit axis."""
        return Rotation(normalize_axis(self.axis), float(self.angle))

    def to_quaternion(self) -> Tuple[float, float, float, float]:
        return axis_angle_to_quaternion(self.axis, self.angle)

    @classmethod
    def from_quaternion(cls, q: Tuple[float, float, float, float]) -> "Rotation":
        axis, angle = quaternion_to_axis_angle(q)
        return cls(axis, angle)

    def to_list(self) -> List[float]:
        ax, ay, az = self.axis
        return [float(ax), float(ay), float(az), float(self.angle)]

    @classmethod
    def from_list(cls, data: List[float]) -> "Rotation":
        if len(data) != 4:
            raise ValueError("rotation list must have 4 elements [ax, ay, az, angle]")
        return cls((float(data[0]), float(data[1]), float(data[2])), float(data[3]))


@dataclass(frozen=True)
class Placement:
    """A FreeCAD ``Placement``: a base position plus an axis-angle rotation."""

    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Rotation = field(default_factory=Rotation)

    def to_dict(self) -> Dict[str, Any]:
        px, py, pz = self.position
        return {
            "position": [float(px), float(py), float(pz)],
            "rotation": self.rotation.to_list(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Placement":
        pos = data.get("position") or [0.0, 0.0, 0.0]
        if len(pos) != 3:
            raise ValueError("placement position must have 3 elements")
        rot = data.get("rotation")
        rotation = Rotation.from_list(rot) if rot is not None else Rotation()
        return cls((float(pos[0]), float(pos[1]), float(pos[2])), rotation)


@dataclass(frozen=True)
class ShapeInfo:
    """The scalar shape summary FreeCAD reports for an object with a ``Shape``."""

    type: str  # ShapeType: Solid / Shell / Compound / ...
    volume: Optional[float] = None
    area: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "volume": None if self.volume is None else float(self.volume),
            "area": None if self.area is None else float(self.area),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShapeInfo":
        vol = data.get("volume")
        area = data.get("area")
        return cls(
            type=str(data.get("type", "")),
            volume=None if vol is None else float(vol),
            area=None if area is None else float(area),
        )


@dataclass(frozen=True)
class FreeCADObject:
    """One node of a FreeCAD document tree, as the agent sees it.

    ``name`` is the immutable internal id (``obj.Name``); ``label`` is the
    user-facing, mutable name (``obj.Label``); ``type_id`` is the FreeCAD
    ``TypeId`` (``Namespace::Class``). ``placement`` / ``shape`` are present only
    for objects that carry them.
    """

    name: str
    label: str
    type_id: str
    visibility: Optional[bool] = None
    placement: Optional[Placement] = None
    shape: Optional[ShapeInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "type": self.type_id,
            "visibility": self.visibility,
        }
        if self.placement is not None:
            out["placement"] = self.placement.to_dict()
        if self.shape is not None:
            out["shape"] = self.shape.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FreeCADObject":
        placement = data.get("placement")
        shape = data.get("shape")
        return cls(
            name=str(data.get("name", "")),
            label=str(data.get("label", "")),
            type_id=str(data.get("type", "")),
            visibility=data.get("visibility"),
            placement=Placement.from_dict(placement) if placement is not None else None,
            shape=ShapeInfo.from_dict(shape) if shape is not None else None,
        )


@dataclass(frozen=True)
class DocumentInfo:
    """Top-level document metadata (``doc.Name`` / ``doc.FileName`` / count)."""

    name: str
    filename: Optional[str] = None
    object_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "filename": self.filename,
            "object_count": int(self.object_count),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentInfo":
        return cls(
            name=str(data.get("name", "")),
            filename=data.get("filename"),
            object_count=int(data.get("object_count", 0)),
        )


@dataclass(frozen=True)
class ViewState:
    """The Coin3D camera state (``getCameraNode()``): type + position + quaternion.

    ``camera_orientation`` is a quaternion ``[x, y, z, w]`` (an ``SbRotation``);
    :meth:`orientation_axis_angle` decodes it to axis-angle for interpretation.
    """

    camera_type: str
    camera_position: Vec3
    camera_orientation: Tuple[float, float, float, float]

    def orientation_axis_angle(self) -> Tuple[Vec3, float]:
        return quaternion_to_axis_angle(self.camera_orientation)

    def to_dict(self) -> Dict[str, Any]:
        px, py, pz = self.camera_position
        ox, oy, oz, ow = self.camera_orientation
        return {
            "camera_type": self.camera_type,
            "camera_position": [float(px), float(py), float(pz)],
            "camera_orientation": [float(ox), float(oy), float(oz), float(ow)],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ViewState":
        pos = data.get("camera_position") or [0.0, 0.0, 0.0]
        ori = data.get("camera_orientation") or [0.0, 0.0, 0.0, 1.0]
        if len(pos) != 3:
            raise ValueError("camera_position must have 3 elements")
        if len(ori) != 4:
            raise ValueError("camera_orientation must have 4 elements")
        return cls(
            camera_type=str(data.get("camera_type", "")),
            camera_position=(float(pos[0]), float(pos[1]), float(pos[2])),
            camera_orientation=(float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])),
        )


@dataclass(frozen=True)
class DocumentContext:
    """The whole ``get_document_context`` payload: document + objects + view."""

    document: Optional[DocumentInfo]
    objects: List[FreeCADObject] = field(default_factory=list)
    view: Optional[ViewState] = None

    def object_by_name(self, name: str) -> Optional[FreeCADObject]:
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None


# ===========================================================================
# TypeId parsing
# ===========================================================================
def parse_type_id(type_id: str) -> Tuple[str, str]:
    """Split a FreeCAD ``TypeId`` into ``(namespace, class)``.

    FreeCAD TypeIds are ``Namespace::Class`` (``Part::Box`` ->
    ``("Part", "Box")``, ``Sketcher::SketchObject`` ->
    ``("Sketcher", "SketchObject")``). A TypeId with no ``::`` is returned as
    ``("", type_id)``. Only the first separator splits (namespaces never nest in
    FreeCAD's registration, but a trailing class may contain none).
    """
    if "::" not in type_id:
        return ("", type_id)
    ns, _, cls = type_id.partition("::")
    return (ns, cls)


# ===========================================================================
# Encode / decode -- the exact freecad_mcp wire shape
# ===========================================================================
def encode_document_context(ctx: DocumentContext) -> Dict[str, Any]:
    """Serialise a :class:`DocumentContext` to the ``get_document_context`` JSON.

    Reproduces the bridge's exact shape: ``{document, objects, view}`` where an
    empty document is ``{"document": None, "objects": [], "view": None}``.
    """
    return {
        "document": None if ctx.document is None else ctx.document.to_dict(),
        "objects": [obj.to_dict() for obj in ctx.objects],
        "view": None if ctx.view is None else ctx.view.to_dict(),
    }


def decode_document_context(data: Dict[str, Any]) -> DocumentContext:
    """Inverse of :func:`encode_document_context`; round-trip stable."""
    doc = data.get("document")
    view = data.get("view")
    objects = data.get("objects") or []
    return DocumentContext(
        document=DocumentInfo.from_dict(doc) if doc is not None else None,
        objects=[FreeCADObject.from_dict(o) for o in objects],
        view=ViewState.from_dict(view) if view is not None else None,
    )


# ===========================================================================
# Validation
# ===========================================================================
def validate_context(ctx: DocumentContext) -> List[str]:
    """Return a deterministic list of structural issues (empty == clean).

    Checks the invariants a well-formed FreeCAD context must satisfy:

      * ``document.object_count`` agrees with ``len(objects)``;
      * every ``TypeId`` is well-formed (a ``Namespace::Class`` with non-empty
        parts) -- FreeCAD never registers a bare or empty TypeId;
      * object ``name`` ids are unique (FreeCAD ``Name`` is a document-unique id);
      * every placement rotation axis is non-degenerate (normalisable);
      * shape ``volume`` / ``area`` are non-negative when present.
    """
    issues: List[str] = []
    if ctx.document is not None and ctx.document.object_count != len(ctx.objects):
        issues.append(
            "object_count %d != len(objects) %d"
            % (ctx.document.object_count, len(ctx.objects))
        )
    seen: Dict[str, int] = {}
    for i, obj in enumerate(ctx.objects):
        seen[obj.name] = seen.get(obj.name, 0) + 1
        ns, cls = parse_type_id(obj.type_id)
        if not cls or "::" not in obj.type_id or not ns:
            issues.append("object[%d] %r has malformed TypeId %r" % (i, obj.name, obj.type_id))
        if obj.placement is not None:
            ax = obj.placement.rotation.axis
            if math.sqrt(sum(c * c for c in ax)) <= _EPS and not obj.placement.rotation.is_identity:
                issues.append("object[%d] %r has a degenerate rotation axis" % (i, obj.name))
        if obj.shape is not None:
            if obj.shape.volume is not None and obj.shape.volume < 0.0:
                issues.append("object[%d] %r has negative volume" % (i, obj.name))
            if obj.shape.area is not None and obj.shape.area < 0.0:
                issues.append("object[%d] %r has negative area" % (i, obj.name))
    for name, count in sorted(seen.items()):
        if count > 1:
            issues.append("duplicate object name %r (x%d)" % (name, count))
    return issues
