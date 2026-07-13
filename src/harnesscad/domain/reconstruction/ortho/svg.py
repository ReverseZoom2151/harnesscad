"""Safe, stdlib-only parsing of simple vector orthographic drawings."""

from __future__ import annotations

import math
import re
from xml.etree import ElementTree as ET

from .model import Diagnostic, Edge2D, OrthographicInput, View2D

_COMMAND = re.compile(r"[MLHVZmlhvz]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_VIEWS = {"front", "bottom", "left"}


def _number(node, name: str, default: float = 0.0) -> float:
    return float(node.attrib.get(name, default))


def _hidden(node) -> bool:
    style = (node.attrib.get("style", "") + " " + node.attrib.get("class", "")).lower()
    return "dash" in style or "hidden" in style


def _path_points(data: str) -> list[tuple[float, float]]:
    tokens = _COMMAND.findall(data)
    out: list[tuple[float, float]] = []
    i, cmd, current, first = 0, "", (0.0, 0.0), None
    while i < len(tokens):
        token = tokens[i]
        if token.isalpha():
            cmd, i = token, i + 1
            if cmd.lower() == "z" and first is not None:
                out.append(first)
            continue
        relative = cmd.islower()
        if cmd.lower() in {"m", "l"}:
            if i + 1 >= len(tokens):
                raise ValueError("path coordinate is incomplete")
            point = (float(tokens[i]), float(tokens[i + 1]))
            i += 2
        elif cmd.lower() == "h":
            point, i = (float(tokens[i]), current[1]), i + 1
        elif cmd.lower() == "v":
            point, i = (current[0], float(tokens[i])), i + 1
        else:
            raise ValueError(f"unsupported path command {cmd!r}")
        if relative:
            point = (current[0] + point[0], current[1] + point[1])
        current = point
        first = first or point
        out.append(point)
        if cmd.lower() == "m":
            cmd = "l" if cmd == "m" else "L"
    return out


def parse_svg(text: str, *, scale: float = 1.0, tolerance: float = 0.005):
    """Parse groups named ``front``, ``bottom`` and ``left``.

    Only ``line``, ``circle`` and paths composed of M/L/H/V/Z are accepted.
    Scripts, external references, transforms and general curves are rejected.
    """
    diagnostics: list[Diagnostic] = []
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        return None, (Diagnostic("unsafe-svg", "DOCTYPE/entities are forbidden"),)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        return None, (Diagnostic("invalid-svg", str(exc)),)
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1].lower()
        if tag in {"script", "use", "image", "foreignobject"} or "transform" in node.attrib:
            diagnostics.append(Diagnostic("unsafe-svg", f"unsupported SVG element/attribute: {tag}"))
    if diagnostics:
        return None, tuple(diagnostics)

    found: dict[str, list[Edge2D]] = {name: [] for name in _VIEWS}
    for group in root.iter():
        if group.tag.rsplit("}", 1)[-1].lower() != "g":
            continue
        name = (group.attrib.get("data-view") or group.attrib.get("id") or "").lower()
        if name not in _VIEWS:
            continue
        seq = 0
        for node in group.iter():
            tag = node.tag.rsplit("}", 1)[-1].lower()
            source = node.attrib.get("id", f"{name}-{seq}")
            if tag == "line":
                points = ((_number(node, "x1"), _number(node, "y1")),
                          (_number(node, "x2"), _number(node, "y2")))
                found[name].append(Edge2D(name, "line", points, _hidden(node), source))
                seq += 1
            elif tag == "circle":
                cx, cy, r = _number(node, "cx"), _number(node, "cy"), _number(node, "r")
                if r <= 0:
                    diagnostics.append(Diagnostic("invalid-circle", "circle radius must be positive",
                                                  context={"source_id": source}))
                    continue
                points = ((cx + r, cy), (cx, cy + r), (cx - r, cy))
                found[name].append(Edge2D(name, "circle", points, _hidden(node), source))
                seq += 1
            elif tag == "path":
                try:
                    points = _path_points(node.attrib.get("d", ""))
                except ValueError as exc:
                    diagnostics.append(Diagnostic("unsupported-path", str(exc),
                                                  context={"source_id": source}))
                    continue
                for a, b in zip(points, points[1:]):
                    found[name].append(Edge2D(name, "line", (a, b), _hidden(node), source))
                    seq += 1
    views = {name: View2D(name, tuple(found[name])) for name in sorted(_VIEWS)}
    drawing = OrthographicInput(views, scale, tolerance)
    diagnostics.extend(validate_input(drawing))
    return drawing, tuple(diagnostics)


def validate_input(drawing: OrthographicInput) -> tuple[Diagnostic, ...]:
    out: list[Diagnostic] = []
    if not math.isfinite(drawing.scale) or drawing.scale <= 0:
        out.append(Diagnostic("invalid-scale", "scale must be finite and positive"))
    if not math.isfinite(drawing.tolerance) or drawing.tolerance <= 0:
        out.append(Diagnostic("invalid-tolerance", "tolerance must be finite and positive"))
    for name in sorted(_VIEWS):
        view = drawing.views.get(name)
        if view is None:
            out.append(Diagnostic("missing-view", f"required view {name!r} is absent"))
        elif not view.edges:
            out.append(Diagnostic("empty-view", f"required view {name!r} has no edges"))
    unknown = sorted(set(drawing.views) - _VIEWS)
    if unknown:
        out.append(Diagnostic("unknown-view", f"unsupported views: {', '.join(unknown)}",
                              severity="warning"))
    return tuple(out)
