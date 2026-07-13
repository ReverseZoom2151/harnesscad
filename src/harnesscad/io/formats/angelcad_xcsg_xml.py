"""XCSG: the XML interchange format of an AngelCAD CSG tree (read + write).

AngelCAD's script compiler (``as_csg``) does not evaluate booleans itself -- it
serialises the typed CSG tree to an XML file with the ``.xcsg`` extension and
hands that to the ``xcsg`` boolean engine.  ``shape::write_xcsg`` builds::

    <xcsg version="1.0" secant_tolerance="0.01">
      <metadata name="model"/>
      <difference3d>
        <cuboid dx="10" dy="10" dz="10" center="true">
          <tmatrix>
            <trow c0="1" c1="0" c2="0" c3="5"/>
            ... 4 rows ...
          </tmatrix>
        </cuboid>
        <sphere r="6"/>
      </difference3d>
    </xcsg>

Two properties of the format matter and are reproduced faithfully here:

* a shape carries its *own* accumulated 4x4 transform as a ``<tmatrix>`` child,
  written only when it differs from the identity (``solid::transform(xml_node&)``).
  A chain of AngelCAD ``transform`` nodes therefore *collapses* into one matrix
  on the shape -- serialisation is where transform composition happens;
* a boolean's operands are simply nested elements, no wrapper tag
  (``xcsg_vector`` with an empty tag), and structured parameters become child
  elements: ``<vertices><vertex x= y= z=/></vertices>``,
  ``<faces><face><fv index=/></face></faces>``, ``<spline_path><cpoint .../>``.

This is a CSG interchange format the harness did not have: it has STL and GLB
(``formats.t2cdean_*``, meshes) and OpenSCAD source (``programs.solidpy_scad_emit``,
untyped text), but no *tree* format that survives a round trip with types intact.

:func:`dumps` / :func:`loads` are exact inverses on canonical documents
(``dumps(loads(x)) == x``), and :func:`loads` type-checks nothing -- run
``programs.angelcad_typed_csg.check`` on the result for that.

Pure stdlib (``xml.etree``), deterministic byte output.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.programs.angelcad_typed_csg import OPS, Node, TMatrix, identity

__all__ = [
    "XcsgError",
    "dumps",
    "loads",
    "write_xcsg",
    "read_xcsg",
    "flatten_transforms",
]

_VERSION = "1.0"


class XcsgError(Exception):
    """Malformed .xcsg document."""


def _num(v: float) -> str:
    """AngelCAD writes 12 significant digits; keep integers integral."""
    s = "%.12g" % float(v)
    return s


def _bool(v: bool) -> str:
    return "true" if v else "false"


def _parse_num(s: str, what: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        raise XcsgError("expected a number for %s, got %r" % (what, s))


def _parse_bool(s: str, what: str) -> bool:
    if s == "true":
        return True
    if s == "false":
        return False
    raise XcsgError("expected true/false for %s, got %r" % (what, s))


# --------------------------------------------------------------------------
# transform flattening
# --------------------------------------------------------------------------


def flatten_transforms(node: Node, acc: Optional[TMatrix] = None) -> Node:
    """Push ``transform`` wrappers down onto the shapes they wrap.

    Returns an equivalent tree in which every ``transform`` node directly wraps a
    non-transform node, and nested transforms have been multiplied together
    (outer * inner) -- exactly what ``solid::transform`` does when it writes
    ``m_transform = matrix * m_transform``.
    """
    if node.op == "transform":
        if not node.children:
            raise XcsgError("transform node has no child")
        m = node.params.get("matrix")
        if not isinstance(m, TMatrix):
            raise XcsgError("transform node has no matrix")
        combined = acc * m if acc is not None else m
        return flatten_transforms(node.children[0], combined)

    inner = Node(
        node.op,
        node.params,
        tuple(flatten_transforms(c) for c in node.children),
    )
    if acc is None or acc.is_identity():
        return inner
    return Node("transform", {"matrix": acc}, (inner,))


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------


def _write_tmatrix(parent: ET.Element, m: TMatrix) -> None:
    xm = ET.SubElement(parent, "tmatrix")
    for row in m.rows:
        ET.SubElement(
            xm, "trow", {("c%d" % i): _num(row[i]) for i in range(4)}
        )


def _write_shape(parent: ET.Element, node: Node, matrix: Optional[TMatrix]) -> ET.Element:
    if node.op == "transform":
        m = node.params.get("matrix")
        if not isinstance(m, TMatrix):
            raise XcsgError("transform node has no matrix")
        if not node.children:
            raise XcsgError("transform node has no child")
        combined = matrix * m if matrix is not None else m
        return _write_shape(parent, node.children[0], combined)

    spec = OPS.get(node.op)
    if spec is None:
        raise XcsgError("unknown operator %r" % node.op)

    elem = ET.SubElement(parent, node.op)

    # scalar attributes first, in the operator's declared order
    for name in sorted(spec.params):
        kind = spec.params[name]
        if name not in node.params:
            continue
        value = node.params[name]
        if kind in ("num", "num+", "angle+"):
            elem.set(name, _num(value))
        elif kind == "bool":
            elem.set(name, _bool(bool(value)))

    if matrix is not None and not matrix.is_identity():
        _write_tmatrix(elem, matrix)

    # structured parameters
    if node.op in ("polygon", "polyhedron"):
        dim = 2 if node.op == "polygon" else 3
        xv = ET.SubElement(elem, "vertices")
        for p in node.params.get("points", ()):
            attrs = {"x": _num(p[0]), "y": _num(p[1])}
            if dim == 3:
                attrs["z"] = _num(p[2])
            ET.SubElement(xv, "vertex", attrs)
        faces = node.params.get("faces") or ()
        if faces:
            xf = ET.SubElement(elem, "faces")
            for face in faces:
                xface = ET.SubElement(xf, "face")
                for iv in face:
                    ET.SubElement(xface, "fv", {"index": str(int(iv))})
    elif node.op == "sweep":
        xp = ET.SubElement(elem, "spline_path")
        for p in node.params.get("path", ()):
            ET.SubElement(
                xp, "cpoint", {"x": _num(p[0]), "y": _num(p[1]), "z": _num(p[2])}
            )

    for child in node.children:
        _write_shape(elem, child, None)
    return elem


def _indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for i, child in enumerate(elem):
            _indent(child, level + 1)
            last = i == len(elem) - 1
            if not child.tail or not child.tail.strip():
                child.tail = pad if last else pad + "  "
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def dumps(
    root: Node,
    secant_tolerance: Optional[float] = None,
    model_name: Optional[str] = None,
) -> str:
    """Serialise a typed CSG tree to an .xcsg document."""
    xroot = ET.Element("xcsg", {"version": _VERSION})
    if secant_tolerance is not None and secant_tolerance > 0.0:
        xroot.set("secant_tolerance", _num(secant_tolerance))
    if model_name:
        ET.SubElement(xroot, "metadata", {"name": model_name})
    _write_shape(xroot, root, None)
    _indent(xroot)
    return ET.tostring(xroot, encoding="unicode") + "\n"


def write_xcsg(path: str, root: Node, **kwargs: Any) -> str:
    text = dumps(root, **kwargs)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------


def _read_tmatrix(elem: ET.Element) -> TMatrix:
    rows: List[List[float]] = []
    for xrow in elem.findall("trow"):
        rows.append([_parse_num(xrow.get("c%d" % i), "trow/c%d" % i) for i in range(4)])
    if len(rows) != 4:
        raise XcsgError("tmatrix needs 4 trow elements, got %d" % len(rows))
    return TMatrix(rows)


def _read_shape(elem: ET.Element) -> Node:
    spec = OPS.get(elem.tag)
    if spec is None:
        raise XcsgError("unknown xcsg element %r" % elem.tag)

    params: Dict[str, Any] = {}
    for name, kind in spec.params.items():
        raw = elem.get(name)
        if raw is None:
            continue
        if kind in ("num", "num+", "angle+"):
            params[name] = _parse_num(raw, "%s/%s" % (elem.tag, name))
        elif kind == "bool":
            params[name] = _parse_bool(raw, "%s/%s" % (elem.tag, name))

    xverts = elem.find("vertices")
    if xverts is not None:
        dim = 2 if elem.tag == "polygon" else 3
        pts: List[Tuple[float, ...]] = []
        for xv in xverts.findall("vertex"):
            coords = [_parse_num(xv.get("x"), "vertex/x"), _parse_num(xv.get("y"), "vertex/y")]
            if dim == 3:
                coords.append(_parse_num(xv.get("z"), "vertex/z"))
            pts.append(tuple(coords))
        params["points"] = pts

    xfaces = elem.find("faces")
    if xfaces is not None:
        faces: List[Tuple[int, ...]] = []
        for xf in xfaces.findall("face"):
            faces.append(
                tuple(int(_parse_num(xfv.get("index"), "fv/index")) for xfv in xf.findall("fv"))
            )
        params["faces"] = faces
    elif elem.tag == "polyhedron":
        params["faces"] = []

    xpath = elem.find("spline_path")
    if xpath is not None:
        params["path"] = [
            (
                _parse_num(cp.get("x"), "cpoint/x"),
                _parse_num(cp.get("y"), "cpoint/y"),
                _parse_num(cp.get("z"), "cpoint/z"),
            )
            for cp in xpath.findall("cpoint")
        ]

    children = [
        _read_shape(child)
        for child in elem
        if child.tag in OPS
    ]

    node = Node(elem.tag, params, children)

    xm = elem.find("tmatrix")
    if xm is not None:
        node = Node("transform", {"matrix": _read_tmatrix(xm)}, (node,))
    return node


def loads(text: str) -> Node:
    """Parse an .xcsg document into a typed CSG tree."""
    try:
        xroot = ET.fromstring(text)
    except ET.ParseError as exc:
        raise XcsgError("invalid XML: %s" % exc)
    if xroot.tag != "xcsg":
        raise XcsgError("root element must be <xcsg>, got <%s>" % xroot.tag)
    shapes = [child for child in xroot if child.tag in OPS]
    if len(shapes) != 1:
        raise XcsgError("xcsg must contain exactly one root shape, got %d" % len(shapes))
    return _read_shape(shapes[0])


def read_xcsg(path: str) -> Node:
    with open(path, "r", encoding="utf-8") as fh:
        return loads(fh.read())
