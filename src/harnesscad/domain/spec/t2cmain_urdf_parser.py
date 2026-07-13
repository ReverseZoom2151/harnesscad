"""Strict URDF (Unified Robot Description Format) parser and tree validator.

Ported from ``packages/cadjs/src/lib/urdf/parseUrdf.js`` in the ``text-to-cad``
(CAD Skills) repository, which parses a robot description into the link/joint
records consumed by its viewer.  The harness previously had no robot-description
reader at all, so this whole capability is new.

What makes the source parser worth transferring is that it is *strict*: rather
than silently tolerating a malformed robot, it rejects every structural defect
that would make downstream kinematics meaningless.  This module reproduces those
rules exactly:

* joint types are restricted to ``fixed`` / ``revolute`` / ``continuous`` /
  ``prismatic``; anything else is an error;
* ``revolute`` and ``prismatic`` joints must declare a ``<limit>`` with finite
  ``lower``/``upper``; ``continuous`` joints get the +/-180 degree display range;
* revolute limits are radians in the file and are converted to *degrees*;
  prismatic limits stay in native linear units (the convention the kinematics
  solver in ``geometry.t2cmain_urdf_kinematics`` expects);
* both endpoints of a joint must name declared links;
* duplicate link names, duplicate joint names, a link with two parents, a cycle,
  and a forest (zero or several roots) are all rejected -- a URDF must be a
  single rooted tree;
* ``<mimic>`` must reference an existing joint and carry finite multiplier and
  offset;
* geometry primitives (``box``/``cylinder``/``sphere``) must carry positive
  dimensions, and ``<material><color rgba>`` components must lie in ``[0, 1]``
  (they are canonicalised to a ``#rrggbb`` string, and named top-level materials
  are resolved for links that reference a material only by name).

Parsing uses ``xml.etree.ElementTree`` and is namespace-tolerant.  The result is
a :class:`geometry.t2cmain_urdf_kinematics.RobotModel` plus link visual records,
so a parsed robot can be posed immediately.  Deterministic: document order is
preserved and no clock, network, or randomness is involved.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.kinematics.t2cmain_urdf_kinematics import (
    IDENTITY_TRANSFORM,
    Joint,
    JointMimic,
    RobotModel,
    Transform,
    multiply_transforms,
    rotation_transform_from_rpy,
    translation_transform,
)

SUPPORTED_JOINT_TYPES = ("fixed", "continuous", "revolute", "prismatic")


class UrdfParseError(ValueError):
    """Raised when a URDF document is missing, malformed, or structurally invalid."""


@dataclass(frozen=True)
class Primitive:
    """A URDF geometric primitive (``box``, ``cylinder`` or ``sphere``)."""

    type: str
    size: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None
    length: Optional[float] = None


@dataclass(frozen=True)
class Visual:
    """One ``<visual>`` of a link: a placement plus a primitive or a mesh ref."""

    origin_transform: Transform = IDENTITY_TRANSFORM
    primitive: Optional[Primitive] = None
    mesh_filename: str = ""
    mesh_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    color: str = ""


@dataclass(frozen=True)
class Link:
    """A URDF link and its visuals."""

    name: str
    visuals: Tuple[Visual, ...] = ()


@dataclass(frozen=True)
class UrdfDocument:
    """A fully validated robot description."""

    name: str
    model: RobotModel
    links: Tuple[Link, ...] = ()
    materials: Dict[str, str] = field(default_factory=dict)

    @property
    def root_link(self) -> str:
        return self.model.root_link

    @property
    def joints(self) -> Tuple[Joint, ...]:
        return self.model.joints


def _local_name(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _children(element: Optional[ElementTree.Element], tag: str) -> List[ElementTree.Element]:
    if element is None:
        return []
    return [child for child in list(element) if _local_name(child.tag) == tag]


def _first(element: Optional[ElementTree.Element], tag: str) -> Optional[ElementTree.Element]:
    found = _children(element, tag)
    return found[0] if found else None


def _number_list(
    value: Optional[str], count: int, fallback: Sequence[float], context: str
) -> List[float]:
    if value is None or not value.strip():
        return list(fallback)
    return _required_number_list(value, count, context)


def _required_number_list(value: Optional[str], count: int, context: str) -> List[float]:
    if value is None or not value.strip():
        raise UrdfParseError(f"{context} must declare {count} numeric values")
    parts = value.split()
    if len(parts) != count:
        raise UrdfParseError(f"{context} must declare {count} numeric values")
    numbers: List[float] = []
    for part in parts:
        try:
            number = float(part)
        except ValueError as error:
            raise UrdfParseError(f"{context} must declare {count} numeric values") from error
        if math.isnan(number) or math.isinf(number):
            raise UrdfParseError(f"{context} must declare {count} numeric values")
        numbers.append(number)
    return numbers


def _positive_attribute(element: ElementTree.Element, name: str, context: str) -> float:
    raw = element.get(name)
    try:
        value = float(raw) if raw is not None else float("nan")
    except ValueError as error:
        raise UrdfParseError(f"{context} must declare a positive {name}") from error
    if math.isnan(value) or math.isinf(value) or value <= 0.0:
        raise UrdfParseError(f"{context} must declare a positive {name}")
    return value


def _finite_attribute(element: ElementTree.Element, name: str, context: str) -> float:
    raw = element.get(name)
    try:
        value = float(raw) if raw is not None else float("nan")
    except ValueError as error:
        raise UrdfParseError(f"{context} has invalid limits") from error
    if math.isnan(value) or math.isinf(value):
        raise UrdfParseError(f"{context} has invalid limits")
    return value


def parse_origin_transform(origin: Optional[ElementTree.Element]) -> Transform:
    """``translate(xyz) @ rotate_rpy(rpy)`` from a URDF ``<origin>`` element."""
    if origin is None:
        return IDENTITY_TRANSFORM
    x, y, z = _number_list(origin.get("xyz"), 3, (0.0, 0.0, 0.0), "URDF origin xyz")
    roll, pitch, yaw = _number_list(origin.get("rpy"), 3, (0.0, 0.0, 0.0), "URDF origin rpy")
    return multiply_transforms(
        translation_transform(x, y, z), rotation_transform_from_rpy(roll, pitch, yaw)
    )


def parse_rgba_color(rgba_text: Optional[str], context: str) -> str:
    """Canonicalise a URDF ``rgba`` attribute to ``#rrggbb`` (alpha dropped)."""
    values = _number_list(rgba_text, 4, (0.0, 0.0, 0.0, 1.0), context)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise UrdfParseError(f"{context} must use rgba values between 0 and 1")
    return "#" + "".join(f"{round(value * 255):02x}" for value in values[:3])


def _material_color(material: ElementTree.Element, context: str) -> str:
    color = _first(material, "color")
    if color is None:
        return ""
    rgba = (color.get("rgba") or "").strip()
    if not rgba:
        return ""
    return parse_rgba_color(rgba, context)


def _parse_primitive(geometry: ElementTree.Element, context: str) -> Optional[Primitive]:
    box = _first(geometry, "box")
    if box is not None:
        size = _required_number_list(box.get("size"), 3, f"{context} box size")
        if any(value <= 0.0 for value in size):
            raise UrdfParseError(f"{context} box size values must be positive")
        return Primitive(type="box", size=(size[0], size[1], size[2]))

    cylinder = _first(geometry, "cylinder")
    if cylinder is not None:
        return Primitive(
            type="cylinder",
            radius=_positive_attribute(cylinder, "radius", f"{context} cylinder"),
            length=_positive_attribute(cylinder, "length", f"{context} cylinder"),
        )

    sphere = _first(geometry, "sphere")
    if sphere is not None:
        return Primitive(
            type="sphere",
            radius=_positive_attribute(sphere, "radius", f"{context} sphere"),
        )
    return None


def _parse_visual(
    visual: ElementTree.Element,
    named_materials: Dict[str, str],
    link_name: str,
    index: int,
) -> Optional[Visual]:
    context = f"URDF link {link_name} visual {index}"
    geometry = _first(visual, "geometry")
    if geometry is None:
        return None

    primitive = _parse_primitive(geometry, context)
    mesh_filename = ""
    mesh_scale = (1.0, 1.0, 1.0)
    mesh = _first(geometry, "mesh")
    if primitive is None:
        if mesh is None:
            return None
        mesh_filename = (mesh.get("filename") or "").strip()
        if not mesh_filename:
            raise UrdfParseError(f"{context} mesh must declare a filename")
        scale = _number_list(mesh.get("scale"), 3, (1.0, 1.0, 1.0), f"{context} mesh scale")
        mesh_scale = (scale[0], scale[1], scale[2])

    color = ""
    material = _first(visual, "material")
    if material is not None:
        color = _material_color(material, f"{context} material")
        if not color:
            name = (material.get("name") or "").strip()
            color = named_materials.get(name, "") if name else ""

    return Visual(
        origin_transform=parse_origin_transform(_first(visual, "origin")),
        primitive=primitive,
        mesh_filename=mesh_filename,
        mesh_scale=mesh_scale,
        color=color,
    )


def _parse_mimic(joint_element: ElementTree.Element, joint_name: str) -> Optional[JointMimic]:
    mimic = _first(joint_element, "mimic")
    if mimic is None:
        return None
    master = (mimic.get("joint") or "").strip()
    if not master:
        raise UrdfParseError(f"URDF mimic joint {joint_name} must reference another joint")
    try:
        multiplier = float((mimic.get("multiplier") or "1").strip() or "1")
        offset = float((mimic.get("offset") or "0").strip() or "0")
    except ValueError as error:
        raise UrdfParseError(
            f"URDF mimic joint {joint_name} has invalid multiplier or offset"
        ) from error
    if any(math.isnan(v) or math.isinf(v) for v in (multiplier, offset)):
        raise UrdfParseError(f"URDF mimic joint {joint_name} has invalid multiplier or offset")
    return JointMimic(joint=master, multiplier=multiplier, offset=offset)


def _parse_joint(joint_element: ElementTree.Element, link_names: set) -> Joint:
    name = (joint_element.get("name") or "").strip()
    if not name:
        raise UrdfParseError("URDF joint name is required")
    joint_type = (joint_element.get("type") or "").strip().lower()
    if joint_type not in SUPPORTED_JOINT_TYPES:
        raise UrdfParseError(
            f"Unsupported URDF joint type: {joint_type or '(missing)'}"
        )

    parent = _first(joint_element, "parent")
    child = _first(joint_element, "child")
    parent_link = (parent.get("link") or "").strip() if parent is not None else ""
    child_link = (child.get("link") or "").strip() if child is not None else ""
    if not parent_link or not child_link:
        raise UrdfParseError(f"URDF joint {name} must declare parent and child links")
    if parent_link not in link_names or child_link not in link_names:
        raise UrdfParseError(f"URDF joint {name} references missing links")

    if joint_type == "fixed":
        axis = (1.0, 0.0, 0.0)
    else:
        axis_element = _first(joint_element, "axis")
        raw_axis = axis_element.get("xyz") if axis_element is not None else None
        values = _number_list(raw_axis, 3, (1.0, 0.0, 0.0), f"URDF joint {name} axis")
        axis = (values[0], values[1], values[2])

    min_value_deg = 0.0
    max_value_deg = 0.0
    if joint_type == "continuous":
        min_value_deg, max_value_deg = -180.0, 180.0
    elif joint_type in ("revolute", "prismatic"):
        limit = _first(joint_element, "limit")
        if limit is None:
            raise UrdfParseError(f"URDF {joint_type} joint {name} requires <limit>")
        context = f"URDF {joint_type} joint {name}"
        lower = _finite_attribute(limit, "lower", context)
        upper = _finite_attribute(limit, "upper", context)
        if joint_type == "revolute":
            min_value_deg, max_value_deg = math.degrees(lower), math.degrees(upper)
        else:
            min_value_deg, max_value_deg = lower, upper

    return Joint(
        name=name,
        type=joint_type,
        parent_link=parent_link,
        child_link=child_link,
        origin_transform=parse_origin_transform(_first(joint_element, "origin")),
        axis=axis,
        default_value_deg=0.0,
        min_value_deg=min_value_deg,
        max_value_deg=max_value_deg,
        mimic=_parse_mimic(joint_element, name),
    )


def validate_tree(link_names: Sequence[str], joints: Sequence[Joint]) -> str:
    """Validate that links and joints form a single rooted, acyclic tree.

    Returns the root link name.  Raises :class:`UrdfParseError` for duplicate
    joint names, a link with several parents, a forest, or a cycle.
    """
    names = set(link_names)
    seen_joints: set = set()
    children: set = set()
    joints_by_parent: Dict[str, List[str]] = {}
    for joint in joints:
        if joint.name in seen_joints:
            raise UrdfParseError(f"Duplicate URDF joint name: {joint.name}")
        seen_joints.add(joint.name)
        if joint.child_link in children:
            raise UrdfParseError(f"URDF link {joint.child_link} has multiple parents")
        children.add(joint.child_link)
        joints_by_parent.setdefault(joint.parent_link, []).append(joint.child_link)

    roots = [name for name in link_names if name not in children]
    if len(roots) != 1:
        raise UrdfParseError(
            "URDF must form a single rooted tree; found roots " + repr(sorted(roots))
        )
    root = roots[0]

    visited: set = set()
    stack = [root]
    while stack:
        link_name = stack.pop()
        if link_name in visited:
            raise UrdfParseError(f"URDF contains a cycle at link {link_name}")
        visited.add(link_name)
        stack.extend(joints_by_parent.get(link_name, []))
    unreachable = names - visited
    if unreachable:
        raise UrdfParseError(
            "URDF contains unreachable links " + repr(sorted(unreachable))
        )
    return root


def parse_urdf(text: str) -> UrdfDocument:
    """Parse and validate a URDF document string."""
    try:
        root_element = ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        raise UrdfParseError(f"URDF is not well-formed XML: {error}") from error
    if _local_name(root_element.tag) != "robot":
        raise UrdfParseError("URDF root element must be <robot>")

    named_materials: Dict[str, str] = {}
    for material in _children(root_element, "material"):
        name = (material.get("name") or "").strip()
        if not name:
            continue
        color = _material_color(material, f"URDF material {name}")
        if color:
            named_materials[name] = color

    links: List[Link] = []
    link_names: List[str] = []
    for link_element in _children(root_element, "link"):
        name = (link_element.get("name") or "").strip()
        if not name:
            raise UrdfParseError("URDF link name is required")
        if name in link_names:
            raise UrdfParseError(f"Duplicate URDF link name: {name}")
        link_names.append(name)
        visuals = []
        for index, visual_element in enumerate(_children(link_element, "visual")):
            visual = _parse_visual(visual_element, named_materials, name, index)
            if visual is not None:
                visuals.append(visual)
        links.append(Link(name=name, visuals=tuple(visuals)))

    if not links:
        raise UrdfParseError("URDF must declare at least one link")

    name_set = set(link_names)
    joints = tuple(
        _parse_joint(joint_element, name_set)
        for joint_element in _children(root_element, "joint")
    )

    joint_names = {joint.name for joint in joints}
    for joint in joints:
        if joint.mimic is not None and joint.mimic.joint not in joint_names:
            raise UrdfParseError(
                f"URDF mimic joint {joint.name} references missing joint {joint.mimic.joint}"
            )

    root_link = validate_tree(link_names, joints)
    model = RobotModel(
        root_link=root_link,
        joints=joints,
        link_names=tuple(link_names),
        root_world_transform=IDENTITY_TRANSFORM,
    )
    return UrdfDocument(
        name=(root_element.get("name") or "").strip(),
        model=model,
        links=tuple(links),
        materials=named_materials,
    )
