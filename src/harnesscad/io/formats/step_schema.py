"""A simplified STEP schema subset for common CAD entities.

STEP-LLM (Shi et al., DATE 2026) teaches an LLM the *grammar* of STEP files: the
entity types and their attribute layouts. Full ISO 10303 schemas (e.g.
AUTOMOTIVE_DESIGN) define thousands of entities; the paper's training corpus is
restricted to comparatively small B-rep models. This module captures the small,
practically important subset of geometric and topological entities together with
enough attribute metadata to (a) construct well-formed instances programmatically
and (b) validate that a parsed instance carries the right arity and attribute
*kinds*.

Each entity is described by an :class:`EntityDef` listing its attributes as
``(name, kind)`` pairs, where ``kind`` is one of:

  * ``"str"``    - a label / text attribute
  * ``"real"``   - a real number
  * ``"int"``    - an integer
  * ``"enum"``   - an enumeration literal (``.T.`` etc.)
  * ``"ref"``    - a single cross-reference to another entity
  * ``"reflist"``- a list of cross-references
  * ``"reallist"``- a list of reals (e.g. a coordinate tuple)
  * ``"bool"``   - a part-21 ``.T.``/``.F.`` logical

The subset mirrors the entities that dominate ABC-derived STEP files: points,
directions, vectors, placements, the elementary curves/surfaces, and the B-rep
topology chain (vertex -> edge -> loop -> face -> shell -> solid).

This module is pure/deterministic and depends only on
:mod:`formats.stepllm_parser` for the value model.
"""

from __future__ import annotations

from dataclasses import dataclass

from harnesscad.io.formats.step import Enum, Entity, Real, Ref


@dataclass(frozen=True)
class Attr:
    name: str
    kind: str


@dataclass(frozen=True)
class EntityDef:
    name: str
    attrs: tuple

    @property
    def arity(self) -> int:
        return len(self.attrs)


def _d(name: str, *attrs) -> EntityDef:
    return EntityDef(name, tuple(Attr(n, k) for n, k in attrs))


# Attribute layouts follow ISO 10303-42 (geometry) / -41 (topology).
_DEFS = (
    _d("CARTESIAN_POINT", ("name", "str"), ("coordinates", "reallist")),
    _d("DIRECTION", ("name", "str"), ("direction_ratios", "reallist")),
    _d("VECTOR", ("name", "str"), ("orientation", "ref"), ("magnitude", "real")),
    _d("AXIS2_PLACEMENT_3D",
       ("name", "str"), ("location", "ref"), ("axis", "ref"),
       ("ref_direction", "ref")),
    _d("AXIS2_PLACEMENT_2D",
       ("name", "str"), ("location", "ref"), ("ref_direction", "ref")),
    _d("LINE", ("name", "str"), ("pnt", "ref"), ("dir", "ref")),
    _d("CIRCLE", ("name", "str"), ("position", "ref"), ("radius", "real")),
    _d("ELLIPSE",
       ("name", "str"), ("position", "ref"), ("semi_axis_1", "real"),
       ("semi_axis_2", "real")),
    _d("PLANE", ("name", "str"), ("position", "ref")),
    _d("CYLINDRICAL_SURFACE",
       ("name", "str"), ("position", "ref"), ("radius", "real")),
    _d("CONICAL_SURFACE",
       ("name", "str"), ("position", "ref"), ("radius", "real"),
       ("semi_angle", "real")),
    _d("SPHERICAL_SURFACE",
       ("name", "str"), ("position", "ref"), ("radius", "real")),
    _d("TOROIDAL_SURFACE",
       ("name", "str"), ("position", "ref"), ("major_radius", "real"),
       ("minor_radius", "real")),
    _d("VERTEX_POINT", ("name", "str"), ("vertex_geometry", "ref")),
    _d("EDGE_CURVE",
       ("name", "str"), ("edge_start", "ref"), ("edge_end", "ref"),
       ("edge_geometry", "ref"), ("same_sense", "bool")),
    _d("ORIENTED_EDGE",
       ("name", "str"), ("edge_start", "ref"), ("edge_end", "ref"),
       ("edge_element", "ref"), ("orientation", "bool")),
    _d("EDGE_LOOP", ("name", "str"), ("edge_list", "reflist")),
    _d("FACE_BOUND",
       ("name", "str"), ("bound", "ref"), ("orientation", "bool")),
    _d("FACE_OUTER_BOUND",
       ("name", "str"), ("bound", "ref"), ("orientation", "bool")),
    _d("ADVANCED_FACE",
       ("name", "str"), ("bounds", "reflist"), ("face_geometry", "ref"),
       ("same_sense", "bool")),
    _d("CLOSED_SHELL", ("name", "str"), ("cfs_faces", "reflist")),
    _d("OPEN_SHELL", ("name", "str"), ("cfs_faces", "reflist")),
    _d("MANIFOLD_SOLID_BREP", ("name", "str"), ("outer", "ref")),
)

DEFS: dict = {d.name: d for d in _DEFS}


def known(name: str) -> bool:
    return name in DEFS


def entity_def(name: str) -> EntityDef:
    try:
        return DEFS[name]
    except KeyError as exc:
        raise KeyError(f"unknown entity type {name!r}") from exc


def _kind_ok(kind: str, value) -> bool:
    if kind == "str":
        return isinstance(value, str)
    if kind == "real":
        return isinstance(value, Real) or isinstance(value, int)
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "enum":
        return isinstance(value, Enum)
    if kind == "bool":
        return isinstance(value, Enum) and value.name in ("T", "F", "U")
    if kind == "ref":
        return isinstance(value, Ref)
    if kind == "reflist":
        return isinstance(value, (list, tuple)) and all(
            isinstance(v, Ref) for v in value)
    if kind == "reallist":
        return isinstance(value, (list, tuple)) and all(
            isinstance(v, Real) or isinstance(v, int) for v in value)
    raise ValueError(f"unknown attribute kind {kind!r}")


def check_attributes(entity: Entity) -> list:
    """Return a list of human-readable problems for one instance (empty = ok).

    Unknown entity types are *not* an error here (a file may legitimately use
    entities outside this subset); they simply yield no attribute checks.
    """

    if entity.keyword is None or not known(entity.keyword):
        return []
    d = entity_def(entity.keyword)
    problems: list = []
    if len(entity.params) != d.arity:
        problems.append(
            f"#{entity.id} {entity.keyword}: expected {d.arity} attributes, "
            f"got {len(entity.params)}")
        return problems
    for attr, value in zip(d.attrs, entity.params):
        if not _kind_ok(attr.kind, value):
            problems.append(
                f"#{entity.id} {entity.keyword}.{attr.name}: expected "
                f"{attr.kind}, got {type(value).__name__}")
    return problems


# --- convenience constructors ------------------------------------------------

def _real(x) -> Real:
    if isinstance(x, Real):
        return x
    f = float(x)
    if f == int(f):
        return Real(f"{int(f)}.")
    return Real(repr(f))


def make(name: str, ent_id: int, *params) -> Entity:
    """Build an :class:`Entity` for ``name``, validating arity eagerly."""

    d = entity_def(name)
    if len(params) != d.arity:
        raise ValueError(
            f"{name} takes {d.arity} attributes, got {len(params)}")
    return Entity(ent_id, name, list(params))


def cartesian_point(ent_id: int, x, y, z, name: str = "") -> Entity:
    return make("CARTESIAN_POINT", ent_id, name,
                [_real(x), _real(y), _real(z)])


def direction(ent_id: int, dx, dy, dz, name: str = "") -> Entity:
    return make("DIRECTION", ent_id, name, [_real(dx), _real(dy), _real(dz)])


def axis2_placement_3d(ent_id: int, location, axis, ref_direction,
                       name: str = "") -> Entity:
    return make("AXIS2_PLACEMENT_3D", ent_id, name,
                Ref(location), Ref(axis), Ref(ref_direction))


def plane(ent_id: int, position, name: str = "") -> Entity:
    return make("PLANE", ent_id, name, Ref(position))


def circle(ent_id: int, position, radius, name: str = "") -> Entity:
    return make("CIRCLE", ent_id, name, Ref(position), _real(radius))


def line(ent_id: int, pnt, direction_ref, name: str = "") -> Entity:
    return make("LINE", ent_id, name, Ref(pnt), Ref(direction_ref))
