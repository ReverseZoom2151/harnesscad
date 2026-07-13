"""Backend-agnostic code-CAD operation vocabulary and program validator.

CodeToCAD's defining contribution is not its Blender/build123d back ends but its
*interface*: a fixed vocabulary of CAD operations (draw / extrude / revolve / loft /
sweep / boolean / fillet / chamfer / mirror / pattern / transform / joint) whose
signatures are declared once and implemented by any provider.  Upstream this lives
as a set of ``raise NotImplementedError`` stubs, so nothing checks that a generated
program actually *type-checks* against the vocabulary.

This module turns the vocabulary into machine-readable data and adds the checker a
text-to-CAD pipeline needs before it ever touches a kernel:

* :data:`OPERATIONS` -- name -> :class:`OperationSpec` (category, parameters,
  produced entity kind).  Parameters carry a semantic ``kind``
  (``length`` / ``angle`` / ``scalar`` / ``bool`` / ``plane`` / ``cardinal`` /
  ``entity`` / ``entity_list`` / ``string``) and a required flag.
* :func:`validate_call` -- one call: unknown operation, missing required argument,
  unknown argument, and value-level checks (lengths and angles must parse as unit
  expressions, cardinals must resolve, planes must be XY/XZ/YZ).
* :func:`validate_program` -- a whole call sequence: builds a symbol table of
  produced entities, rejects duplicate names, forward references (use before
  definition), references to undefined entities and entity-kind mismatches
  (e.g. extruding a solid, or filleting a sketch).
* :func:`entity_kinds` / :func:`describe_operation` -- introspection for prompting
  and for grammar-constrained decoding.

Pure stdlib, deterministic: errors come back in a stable, sorted-by-call order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harnesscad.domain.geometry.topology.codetocad_cardinal_landmark import CardinalError, resolve_cardinal
from harnesscad.domain.geometry.transforms.codetocad_transform_stack import PLANES
from harnesscad.domain.numeric.codetocad_length_expression import (
    LENGTH,
    PERCENT,
    SCALAR,
    ExpressionError,
    parse_angle,
    parse_quantity,
)

__all__ = [
    "ParamSpec",
    "OperationSpec",
    "Call",
    "OPERATIONS",
    "CATEGORIES",
    "ENTITY_KINDS",
    "operation_names",
    "describe_operation",
    "entity_kinds",
    "validate_call",
    "validate_program",
]

# Entity kinds a call may produce or consume.
SKETCH = "sketch"
SOLID = "solid"
LANDMARK = "landmark"
ENTITY_KINDS = (SKETCH, SOLID, LANDMARK)


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str
    required: bool = True
    accepts: tuple = ()  # for entity/entity_list: allowed entity kinds
    default: object = None


@dataclass(frozen=True)
class OperationSpec:
    name: str
    category: str
    params: tuple
    produces: str | None = None
    doc: str = ""

    def param(self, name: str) -> ParamSpec | None:
        for spec in self.params:
            if spec.name == name:
                return spec
        return None

    @property
    def required(self) -> tuple:
        return tuple(p.name for p in self.params if p.required)


def _p(name, kind, required=True, accepts=(), default=None):
    return ParamSpec(name, kind, required, tuple(accepts), default)


_OPERATION_LIST = (
    # -- 2D drawing ---------------------------------------------------------
    OperationSpec(
        "line", "draw",
        (_p("start", "point"), _p("end", "point"), _p("plane", "plane", False, default="xy")),
        SKETCH, "A straight segment between two points.",
    ),
    OperationSpec(
        "arc", "draw",
        (_p("start", "point"), _p("end", "point"), _p("radius", "length"),
         _p("plane", "plane", False, default="xy")),
        SKETCH, "A circular arc through two points with a radius.",
    ),
    OperationSpec(
        "circle", "draw",
        (_p("center", "point"), _p("radius", "length"),
         _p("plane", "plane", False, default="xy")),
        SKETCH, "A closed circle.",
    ),
    OperationSpec(
        "rectangle", "draw",
        (_p("center", "point"), _p("width", "length"), _p("height", "length"),
         _p("plane", "plane", False, default="xy")),
        SKETCH, "An axis-aligned rectangle.",
    ),
    OperationSpec(
        "polygon", "draw",
        (_p("center", "point"), _p("radius", "length"), _p("sides", "scalar"),
         _p("plane", "plane", False, default="xy")),
        SKETCH, "A regular n-gon.",
    ),
    OperationSpec(
        "spline", "draw",
        (_p("points", "point_list"), _p("plane", "plane", False, default="xy")),
        SKETCH, "An interpolating spline through control points.",
    ),
    # -- sketch -> solid ----------------------------------------------------
    OperationSpec(
        "extrude", "solid",
        (_p("profile", "entity", accepts=(SKETCH,)), _p("height", "length"),
         _p("draft_angle", "angle", False, default=0),
         _p("subtract", "entity_list", False, accepts=(SKETCH,))),
        SOLID, "Extrude a profile into a solid.",
    ),
    OperationSpec(
        "revolve", "solid",
        (_p("profile", "entity", accepts=(SKETCH,)), _p("axis", "entity", accepts=(SKETCH,)),
         _p("angle", "angle")),
        SOLID, "Revolve a profile around an axis.",
    ),
    OperationSpec(
        "loft", "solid",
        (_p("profile", "entity", accepts=(SKETCH,)), _p("to", "entity", accepts=(SKETCH,)),
         _p("merge", "bool", False, default=True)),
        SOLID, "Loft between two profiles.",
    ),
    OperationSpec(
        "sweep", "solid",
        (_p("profile", "entity", accepts=(SKETCH,)), _p("path", "entity", accepts=(SKETCH,))),
        SOLID, "Sweep a profile along a path.",
    ),
    # -- booleans -----------------------------------------------------------
    OperationSpec(
        "union", "boolean",
        (_p("this", "entity", accepts=(SOLID,)), _p("that", "entity", accepts=(SOLID,)),
         _p("delete_that", "bool", False, default=True)),
        SOLID, "Merge two solids.",
    ),
    OperationSpec(
        "subtract", "boolean",
        (_p("this", "entity", accepts=(SOLID,)), _p("that", "entity", accepts=(SOLID,)),
         _p("delete_that", "bool", False, default=True)),
        SOLID, "Remove that from this.",
    ),
    OperationSpec(
        "intersection", "boolean",
        (_p("this", "entity", accepts=(SOLID,)), _p("that", "entity", accepts=(SOLID,))),
        SOLID, "Keep only the overlap of two solids.",
    ),
    OperationSpec(
        "concat", "boolean",
        (_p("this", "entity", accepts=(SOLID,)), _p("that", "entity", accepts=(SOLID,))),
        SOLID, "Compound two solids without a boolean merge.",
    ),
    # -- features -----------------------------------------------------------
    OperationSpec(
        "fillet", "feature",
        (_p("solid", "entity", accepts=(SOLID,)), _p("radius", "length"),
         _p("at", "cardinal", False)),
        SOLID, "Round edges of a solid.",
    ),
    OperationSpec(
        "chamfer", "feature",
        (_p("solid", "entity", accepts=(SOLID,)), _p("distance", "length"),
         _p("at", "cardinal", False)),
        SOLID, "Bevel edges of a solid.",
    ),
    OperationSpec(
        "hollow", "feature",
        (_p("solid", "entity", accepts=(SOLID,)), _p("thickness", "length"),
         _p("open_at", "cardinal", False)),
        SOLID, "Shell a solid to a wall thickness.",
    ),
    OperationSpec(
        "hole", "feature",
        (_p("solid", "entity", accepts=(SOLID,)), _p("at", "cardinal"),
         _p("radius", "length"), _p("depth", "length", False)),
        SOLID, "Cut a cylindrical hole at a landmark.",
    ),
    OperationSpec(
        "mirror", "feature",
        (_p("solid", "entity", accepts=(SOLID,)), _p("plane", "plane"),
         _p("merge", "bool", False, default=True)),
        SOLID, "Mirror a solid across a plane.",
    ),
    OperationSpec(
        "linear_pattern", "feature",
        (_p("solid", "entity", accepts=(SOLID,)), _p("count", "scalar"),
         _p("spacing", "length"), _p("axis", "string", False, default="x")),
        SOLID, "Repeat a solid along an axis.",
    ),
    OperationSpec(
        "circular_pattern", "feature",
        (_p("solid", "entity", accepts=(SOLID,)), _p("count", "scalar"),
         _p("angle", "angle", False, default="360deg"), _p("axis", "string", False, default="z")),
        SOLID, "Repeat a solid around an axis.",
    ),
    # -- transforms ---------------------------------------------------------
    OperationSpec(
        "translate", "transform",
        (_p("target", "entity", accepts=(SOLID, SKETCH)), _p("x", "length", False, default=0),
         _p("y", "length", False, default=0), _p("z", "length", False, default=0)),
        None, "Move an entity in place.",
    ),
    OperationSpec(
        "rotate", "transform",
        (_p("target", "entity", accepts=(SOLID, SKETCH)), _p("x", "angle", False, default=0),
         _p("y", "angle", False, default=0), _p("z", "angle", False, default=0)),
        None, "Rotate an entity in place (X, then Y, then Z).",
    ),
    OperationSpec(
        "scale", "transform",
        (_p("target", "entity", accepts=(SOLID, SKETCH)), _p("x", "scalar", False, default=1),
         _p("y", "scalar", False, default=1), _p("z", "scalar", False, default=1)),
        None, "Scale an entity in place.",
    ),
    # -- landmarks ----------------------------------------------------------
    OperationSpec(
        "landmark", "landmark",
        (_p("of", "entity", accepts=(SOLID, SKETCH)), _p("at", "cardinal"),
         _p("offset_x", "length", False, default=0), _p("offset_y", "length", False, default=0),
         _p("offset_z", "length", False, default=0)),
        LANDMARK, "Name an anchor point on an entity's bounding box.",
    ),
    # -- joints / constraints ----------------------------------------------
    OperationSpec(
        "fix", "joint",
        (_p("this", "entity", accepts=(LANDMARK,)), _p("at", "entity", accepts=(LANDMARK,)),
         _p("offset_x", "length", False, default=0), _p("offset_y", "length", False, default=0),
         _p("offset_z", "length", False, default=0)),
        None, "Coincide two landmarks.",
    ),
    OperationSpec(
        "tangent", "joint",
        (_p("this", "entity", accepts=(SKETCH,)), _p("at", "entity", accepts=(SKETCH,))),
        None, "Make two edges tangent.",
    ),
    OperationSpec(
        "parallel", "joint",
        (_p("this", "entity", accepts=(SKETCH,)), _p("at", "entity", accepts=(SKETCH,))),
        None, "Make two edges parallel.",
    ),
    OperationSpec(
        "perpendicular", "joint",
        (_p("this", "entity", accepts=(SKETCH,)), _p("at", "entity", accepts=(SKETCH,))),
        None, "Make two edges perpendicular.",
    ),
    OperationSpec(
        "revolute", "joint",
        (_p("this", "entity", accepts=(LANDMARK,)), _p("at", "entity", accepts=(LANDMARK,)),
         _p("axis", "string", False, default="z"),
         _p("limit_min", "angle", False), _p("limit_max", "angle", False)),
        None, "Hinge joint about one axis.",
    ),
    OperationSpec(
        "prismatic", "joint",
        (_p("this", "entity", accepts=(LANDMARK,)), _p("at", "entity", accepts=(LANDMARK,)),
         _p("axis", "string", False, default="z"),
         _p("limit_min", "length", False), _p("limit_max", "length", False)),
        None, "Sliding joint along one axis.",
    ),
    OperationSpec(
        "ball", "joint",
        (_p("this", "entity", accepts=(LANDMARK,)), _p("at", "entity", accepts=(LANDMARK,))),
        None, "Gimbal joint free about all three axes.",
    ),
    OperationSpec(
        "rigid", "joint",
        (_p("this", "entity", accepts=(LANDMARK,)), _p("at", "entity", accepts=(LANDMARK,))),
        None, "Fixed joint with no freedom of movement.",
    ),
)

OPERATIONS: dict[str, OperationSpec] = {spec.name: spec for spec in _OPERATION_LIST}
CATEGORIES: tuple = tuple(
    sorted({spec.category for spec in _OPERATION_LIST})
)


@dataclass(frozen=True)
class Call:
    """One operation invocation; ``result`` names the entity it produces."""

    operation: str
    args: dict = field(default_factory=dict)
    result: str | None = None


def operation_names(category: str | None = None) -> list[str]:
    names = [
        name
        for name, spec in OPERATIONS.items()
        if category is None or spec.category == category
    ]
    return sorted(names)


def describe_operation(name: str) -> str:
    spec = OPERATIONS.get(name)
    if spec is None:
        raise KeyError("unknown operation: " + str(name))
    parts = []
    for param in spec.params:
        text = param.name + ": " + param.kind
        if not param.required:
            text = "[" + text + "]"
        parts.append(text)
    produces = spec.produces or "none"
    return "{0}({1}) -> {2}".format(spec.name, ", ".join(parts), produces)


def entity_kinds(calls) -> dict:
    """Symbol table: result name -> entity kind, for the calls that produce one."""
    table: dict[str, str] = {}
    for call in calls:
        spec = OPERATIONS.get(call.operation)
        if spec is None or spec.produces is None or not call.result:
            continue
        table[call.result] = spec.produces
    return table


# ---------------------------------------------------------------------------
# value-level checks
# ---------------------------------------------------------------------------


def _check_value(param: ParamSpec, value, errors: list, prefix: str) -> None:
    kind = param.kind
    if kind in ("length", "angle"):
        try:
            if kind == "angle":
                parse_angle(value)
            else:
                quantity = parse_quantity(value)
                if quantity.kind not in (LENGTH, SCALAR, PERCENT):
                    raise ExpressionError("expected a length, got " + quantity.kind)
        except ExpressionError as error:
            errors.append(
                "{0}: argument '{1}' is not a valid {2}: {3}".format(
                    prefix, param.name, kind, error
                )
            )
    elif kind == "scalar":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(
                "{0}: argument '{1}' must be a number".format(prefix, param.name)
            )
    elif kind == "bool":
        if not isinstance(value, bool):
            errors.append(
                "{0}: argument '{1}' must be a bool".format(prefix, param.name)
            )
    elif kind == "plane":
        if str(value).lower() not in PLANES:
            errors.append(
                "{0}: argument '{1}' must be one of {2}".format(
                    prefix, param.name, ", ".join(PLANES)
                )
            )
    elif kind == "cardinal":
        try:
            resolve_cardinal(value)
        except CardinalError as error:
            errors.append(
                "{0}: argument '{1}' is not a cardinal direction: {2}".format(
                    prefix, param.name, error
                )
            )
    elif kind == "point":
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            errors.append(
                "{0}: argument '{1}' must be a 3-element point".format(
                    prefix, param.name
                )
            )
    elif kind == "point_list":
        if not isinstance(value, (list, tuple)) or not value:
            errors.append(
                "{0}: argument '{1}' must be a non-empty list of points".format(
                    prefix, param.name
                )
            )
        else:
            for point in value:
                if not isinstance(point, (list, tuple)) or len(point) != 3:
                    errors.append(
                        "{0}: argument '{1}' contains a non-point".format(
                            prefix, param.name
                        )
                    )
                    break
    elif kind == "entity":
        if not isinstance(value, str) or not value:
            errors.append(
                "{0}: argument '{1}' must be an entity name".format(prefix, param.name)
            )
    elif kind == "entity_list":
        if not isinstance(value, (list, tuple)):
            errors.append(
                "{0}: argument '{1}' must be a list of entity names".format(
                    prefix, param.name
                )
            )
    elif kind == "string":
        if not isinstance(value, str):
            errors.append(
                "{0}: argument '{1}' must be a string".format(prefix, param.name)
            )


def validate_call(call: Call, index: int = 0) -> list:
    """Structural + value-level checks for one call (no symbol resolution)."""
    prefix = "call {0} ({1})".format(index, call.operation)
    spec = OPERATIONS.get(call.operation)
    if spec is None:
        return ["call {0}: unknown operation '{1}'".format(index, call.operation)]

    errors: list[str] = []
    if not isinstance(call.args, dict):
        return [prefix + ": args must be a dict"]

    for name in spec.required:
        if name not in call.args:
            errors.append(
                "{0}: missing required argument '{1}'".format(prefix, name)
            )
    for name in sorted(call.args):
        param = spec.param(name)
        if param is None:
            errors.append("{0}: unknown argument '{1}'".format(prefix, name))
            continue
        _check_value(param, call.args[name], errors, prefix)

    if spec.produces is not None and not call.result:
        errors.append(
            "{0}: operation produces a {1} but no result name was given".format(
                prefix, spec.produces
            )
        )
    if spec.produces is None and call.result:
        errors.append(
            "{0}: operation produces nothing but a result name was given".format(prefix)
        )
    return errors


def validate_program(calls) -> list:
    """Validate a call sequence; returns a list of error strings (empty == valid).

    Checks every :func:`validate_call` rule plus, across calls:
    duplicate result names, use of an undefined entity, forward references, and
    entity-kind mismatches against each parameter's ``accepts`` set.
    """
    errors: list[str] = []
    table: dict[str, str] = {}

    for index, call in enumerate(calls):
        call_errors = validate_call(call, index)
        errors.extend(call_errors)
        spec = OPERATIONS.get(call.operation)
        if spec is None:
            continue
        prefix = "call {0} ({1})".format(index, call.operation)

        for name in sorted(call.args):
            param = spec.param(name)
            if param is None or param.kind not in ("entity", "entity_list"):
                continue
            value = call.args[name]
            references = value if param.kind == "entity_list" else [value]
            if not isinstance(references, (list, tuple)):
                continue
            for reference in references:
                if not isinstance(reference, str):
                    continue
                if reference not in table:
                    errors.append(
                        "{0}: argument '{1}' references undefined entity "
                        "'{2}' (forward reference or typo)".format(
                            prefix, name, reference
                        )
                    )
                    continue
                actual = table[reference]
                if param.accepts and actual not in param.accepts:
                    errors.append(
                        "{0}: argument '{1}' expects {2} but '{3}' is a {4}".format(
                            prefix,
                            name,
                            "/".join(param.accepts),
                            reference,
                            actual,
                        )
                    )

        if spec.produces is not None and call.result:
            if call.result in table:
                errors.append(
                    "{0}: duplicate result name '{1}'".format(prefix, call.result)
                )
            else:
                table[call.result] = spec.produces

    return errors
