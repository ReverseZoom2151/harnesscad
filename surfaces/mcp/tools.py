"""MCP-style tool catalog derived from the CISP op registry (sec.5 & sec.9).

Each CISP op (``cisp.ops._REGISTRY``) becomes a :class:`ToolDefinition` with:

  - a **5-component description** (what / when-to-use / when-NOT / side-effects /
    output) — descriptions are load-bearing; the model routes off them;
  - **typed params** derived from the op's dataclass fields (enum-heavy, flat);
  - an **output spec**;
  - **annotations** (auto-assigned read-only / destructive -> approval tier).

Plus the non-op tools the environment exposes: ``measure``, ``query``,
``verify`` (alias ``run_check``), ``export``, ``reset``, ``render``.

``ToolCatalog.to_mcp()`` emits the JSON tool schema a FastMCP server would
register; ``resources()`` exposes model-state observations; ``prompts()``
exposes op templates. Tool *results* carry a ``reward`` field
(:class:`ToolResult`), and tools raise **typed errors**
(:class:`UnknownToolError` / :class:`ToolValidationError` /
:class:`ToolExecutionError`) for the agent to observe and repair.

Stdlib only; no MCP SDK. Everything is plain dataclasses / JSON.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cisp.ops import CONSTRAINT_DOF, _REGISTRY, parse_op
from surfaces.mcp.annotations import Annotations, annotate
from verifiers.verify import Severity


# ===========================================================================
# Typed errors (sec.9: "tools raise typed errors for the agent to observe")
# ===========================================================================
class MCPError(Exception):
    """Base for every tool-layer error. Carries a machine code + structured data."""

    code = "mcp-error"

    def __init__(self, message: str, **data: Any) -> None:
        super().__init__(message)
        self.message = message
        self.data = data

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "data": self.data}


class UnknownToolError(MCPError):
    """Raised when a tool name is not in the catalog."""

    code = "unknown-tool"


class ToolValidationError(MCPError):
    """Raised when arguments miss a required param or violate an enum/type."""

    code = "invalid-params"


class ToolExecutionError(MCPError):
    """Raised when a tool runs but the kernel/verifier rejects the result.

    Carries the verifier ``diagnostics`` (e.g. radius-too-large, over-constrained,
    empty-solid) and the ``reward`` so the agent can observe and repair.
    """

    code = "execution-failed"


# ===========================================================================
# Reward helpers (single source of truth; reused by the Gym env)
# ===========================================================================
def reward_from_apply(result) -> float:
    """Reward for an applyOps result: pass -> positive, fail -> negative.

    A clean pass is 1.0; each WARNING (e.g. an under-constrained sketch) shaves
    0.1 (never below 0.0 while still ``ok``); a rejected/verify-failed batch is
    -1.0. This is the verifier-as-reward from sec.5/sec.6.
    """
    if not getattr(result, "ok", False):
        return -1.0
    warns = sum(1 for d in result.diagnostics if d.severity is Severity.WARNING)
    return max(0.0, 1.0 - 0.1 * warns)


def reward_from_verify(ok: bool, diagnostics: List) -> float:
    """Reward for a read-only verify: +1.0 clean, penalised by warnings, -1.0 fail."""
    if not ok:
        return -1.0
    warns = sum(1 for d in diagnostics if getattr(d, "severity", None) is Severity.WARNING)
    return max(0.0, 1.0 - 0.1 * warns)


# ===========================================================================
# 5-component description + typed params
# ===========================================================================
@dataclass(frozen=True)
class ToolDescription:
    """The 5-component tool description (sec.9). Every component is load-bearing."""

    what: str
    when: str
    when_not: str
    side_effects: str
    output: str

    def is_complete(self) -> bool:
        return all(bool(c and c.strip()) for c in (
            self.what, self.when, self.when_not, self.side_effects, self.output))

    def to_dict(self) -> dict:
        return {
            "what": self.what,
            "when": self.when,
            "whenNot": self.when_not,
            "sideEffects": self.side_effects,
            "output": self.output,
        }

    def text(self) -> str:
        return (
            f"{self.what}\n"
            f"When to use: {self.when}\n"
            f"When NOT to use: {self.when_not}\n"
            f"Side effects: {self.side_effects}\n"
            f"Output: {self.output}"
        )


@dataclass(frozen=True)
class ParamSpec:
    """One typed parameter of a tool (a flat, enum-heavy JSON-schema property)."""

    name: str
    type: str  # JSON-schema type: string|number|integer|boolean|array
    required: bool = False
    default: Any = None
    description: str = ""
    enum: Optional[List[Any]] = None
    nullable: bool = False

    def to_schema(self) -> dict:
        typ: Any = [self.type, "null"] if self.nullable else self.type
        schema: Dict[str, Any] = {"type": typ}
        if self.description:
            schema["description"] = self.description
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        if self.default is not None and not self.required:
            schema["default"] = self.default
        if self.type == "array":
            schema["items"] = {"type": "integer"}
        return schema


@dataclass(frozen=True)
class ToolDefinition:
    """A single MCP-style tool: name + 5-part description + typed IO + annotations."""

    name: str
    description: ToolDescription
    params: List[ParamSpec]
    output: Dict[str, Any]
    annotations: Annotations
    op_tag: Optional[str] = None  # set when derived from a CISP op

    @property
    def tier(self) -> int:
        return self.annotations.tier

    def input_schema(self) -> dict:
        props = {p.name: p.to_schema() for p in self.params}
        required = [p.name for p in self.params if p.required]
        return {"type": "object", "properties": props, "required": required}

    def to_mcp(self) -> dict:
        """The JSON tool schema a FastMCP server would register."""
        return {
            "name": self.name,
            "description": self.description.text(),
            "descriptionComponents": self.description.to_dict(),
            "inputSchema": self.input_schema(),
            "outputSchema": self.output,
            "annotations": self.annotations.to_dict(),
            "op": self.op_tag,
        }

    def validate_args(self, arguments: Dict[str, Any]) -> None:
        """Raise :class:`ToolValidationError` on missing/unknown/enum/type errors."""
        if not isinstance(arguments, dict):
            raise ToolValidationError(f"{self.name}: arguments must be an object")
        by_name = {p.name: p for p in self.params}
        for p in self.params:
            if p.required and p.name not in arguments:
                raise ToolValidationError(
                    f"{self.name}: missing required param '{p.name}'",
                    tool=self.name, param=p.name)
        for key, val in arguments.items():
            spec = by_name.get(key)
            if spec is None:
                raise ToolValidationError(
                    f"{self.name}: unknown param '{key}'", tool=self.name, param=key)
            if val is None:
                if not (spec.nullable or not spec.required):
                    raise ToolValidationError(
                        f"{self.name}: param '{key}' may not be null",
                        tool=self.name, param=key)
                continue
            if spec.enum is not None and val not in spec.enum:
                raise ToolValidationError(
                    f"{self.name}: param '{key}'={val!r} not in {spec.enum}",
                    tool=self.name, param=key)
            if spec.type == "number" and not isinstance(val, (int, float)):
                raise ToolValidationError(
                    f"{self.name}: param '{key}' must be a number", tool=self.name, param=key)
            if spec.type == "integer" and not isinstance(val, int):
                raise ToolValidationError(
                    f"{self.name}: param '{key}' must be an integer", tool=self.name, param=key)
            if spec.type == "string" and not isinstance(val, str):
                raise ToolValidationError(
                    f"{self.name}: param '{key}' must be a string", tool=self.name, param=key)
            if spec.type == "array" and not isinstance(val, (list, tuple)):
                raise ToolValidationError(
                    f"{self.name}: param '{key}' must be an array", tool=self.name, param=key)


# ===========================================================================
# Tool result (carries the reward field — sec.5)
# ===========================================================================
@dataclass
class ToolResult:
    """The value a tool returns to the agent; carries the reward signal (sec.5)."""

    tool: str
    ok: bool
    content: Any
    reward: float = 0.0
    diagnostics: List[dict] = field(default_factory=list)
    is_error: bool = False

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "content": self.content,
            "reward": self.reward,
            "diagnostics": self.diagnostics,
            "isError": self.is_error,
        }


# ===========================================================================
# Description / param metadata for the CISP ops
# ===========================================================================
def _map_type(annotation: str):
    """Map a dataclass annotation string to a (json_type, nullable) pair."""
    a = annotation.strip()
    nullable = False
    if a.startswith("Optional[") and a.endswith("]"):
        nullable = True
        a = a[len("Optional["):-1].strip()
    table = {
        "str": "string", "float": "number", "int": "integer",
        "bool": "boolean", "tuple": "array", "list": "array",
    }
    return table.get(a, "string"), nullable


# enum overrides keyed by (op_tag, field)
_ENUMS = {
    ("new_sketch", "plane"): ["XY", "XZ", "YZ"],
    ("constrain", "kind"): sorted(CONSTRAINT_DOF.keys()),
    ("boolean", "kind"): ["union", "cut", "intersect"],
}

# per-field docs (fallback is generated)
_PARAM_DOCS = {
    ("new_sketch", "plane"): "Datum plane the sketch lives on.",
    ("constrain", "kind"): "Constraint type; distance/radius need a numeric value.",
    ("constrain", "a"): "Primary sketch entity id the constraint acts on.",
    ("constrain", "b"): "Second entity id (for binary constraints); omit otherwise.",
    ("constrain", "value"): "Numeric value for dimensional constraints (distance/radius).",
    ("extrude", "sketch"): "Sketch id whose closed profile is extruded.",
    ("extrude", "distance"): "Signed extrusion distance; must be non-zero.",
    ("fillet", "edges"): "Edge ids to round.",
    ("fillet", "radius"): "Fillet radius; must be > 0 and < adjacent edge length.",
    ("boolean", "kind"): "Boolean operation: union | cut | intersect.",
    ("boolean", "target"): "Solid the operation is applied to.",
    ("boolean", "tool"): "Second solid used as the boolean tool body.",
    ("add_circle", "r"): "Circle radius; must be > 0.",
}

# 5-component descriptions per op.
_OP_DESCRIPTIONS = {
    "new_sketch": ToolDescription(
        what="Create a new empty sketch on a datum plane; a sketch is the 2D "
             "substrate every profile is drawn on.",
        when="At the start of a feature, before adding points/lines/circles/"
             "rectangles that a later extrude or revolve will consume.",
        when_not="Do not create a sketch to add geometry to an existing one "
                 "(reuse its id); do not sketch when you only need to modify an "
                 "existing solid (use fillet/boolean).",
        side_effects="Mutates the model: appends a sketch node to the feature "
                     "tree and returns a new sketch id (e.g. 'sk1'). Reversible via rollback.",
        output="ApplyOps result (ok, applied, digest, diagnostics); the new "
               "sketch id shows up in the feature-tree resource."),
    "add_point": ToolDescription(
        what="Add a point entity to an existing sketch.",
        when="To place a reference/construction point a constraint or profile "
             "will later anchor to.",
        when_not="Not for closed profiles (use lines/rectangles/circles); not "
                 "on a non-existent sketch id.",
        side_effects="Mutates the sketch: adds an entity and raises the sketch DOF.",
        output="ApplyOps result; the new entity id appears in sketch state."),
    "add_line": ToolDescription(
        what="Add a straight line segment between two points in a sketch.",
        when="To build up an open/closed polyline profile edge by edge.",
        when_not="Not to fully define a rectangle (use add_rectangle); not on a "
                 "missing sketch.",
        side_effects="Mutates the sketch: adds an entity and raises the sketch DOF.",
        output="ApplyOps result with the new entity id."),
    "add_circle": ToolDescription(
        what="Add a circle of a given radius to a sketch.",
        when="For round profiles: bosses, holes, cylinders (paired with extrude).",
        when_not="Not with radius <= 0 (raises a typed bad-value error); not on a "
                 "missing sketch.",
        side_effects="Mutates the sketch: adds an entity and raises the sketch DOF.",
        output="ApplyOps result with the new entity id."),
    "add_rectangle": ToolDescription(
        what="Add an axis-aligned rectangle (origin + width + height) to a sketch.",
        when="The fast path for a rectangular profile / plate before extruding.",
        when_not="Not with non-positive width or height (typed bad-value error); "
                 "not on a missing sketch.",
        side_effects="Mutates the sketch: adds an entity and raises the sketch DOF.",
        output="ApplyOps result with the new entity id."),
    "constrain": ToolDescription(
        what="Apply a geometric or dimensional constraint (coincident/horizontal/"
             "vertical/parallel/perpendicular/distance/radius/equal) to sketch "
             "entities, removing degrees of freedom.",
        when="After drawing sketch entities, to drive the sketch toward zero DOF "
             "(fully constrained) before extruding.",
        when_not="Not for 3D features; dimensional kinds (distance/radius) require "
                 "a numeric value; do not over-constrain.",
        side_effects="Mutates sketch DOF; over-constraining raises an "
                     "over-constrained ERROR diagnostic that rolls the op back.",
        output="ApplyOps result; the sketch_dof resource reflects the reduced DOF."),
    "extrude": ToolDescription(
        what="Extrude a sketch profile by a signed distance to create or extend a solid.",
        when="Once a sketch has a closed profile and is adequately constrained, "
             "to turn 2D into 3D.",
        when_not="Not on an empty sketch (no profile) or with distance 0 (typed "
                 "errors); not to modify an existing solid's edges (use fillet).",
        side_effects="Mutates the model: adds a solid feature and sets solid_present true.",
        output="ApplyOps result with the new feature id and an updated digest."),
    "fillet": ToolDescription(
        what="Round one or more edges of an existing solid with a constant radius.",
        when="After a solid exists, to break sharp edges for manufacturability / "
             "stress relief.",
        when_not="Not before a solid exists (no-solid error); radius must be > 0 "
                 "and smaller than adjacent edge lengths.",
        side_effects="Mutates the solid topology; a too-large / non-positive radius "
                     "raises a typed error.",
        output="ApplyOps result; face/edge counts in the summary change."),
    "boolean": ToolDescription(
        what="Combine two solids by union, cut, or intersect.",
        when="To merge bodies, subtract a tool body (holes/pockets), or keep the "
             "common volume.",
        when_not="Not with fewer than two solids (no-solid error); not for edge "
                 "treatment (use fillet).",
        side_effects="Mutates the model: replaces bodies with the combined result; "
                     "a cut that nulls the body raises a typed error.",
        output="ApplyOps result with the combined feature and digest."),
    "revolve": ToolDescription(
        what="Revolve a sketch profile about an axis by an angle to create a solid "
             "of revolution (shafts, bosses, turned parts).",
        when="Once a closed profile exists and the shape is rotationally symmetric "
             "about an axis; the turned analogue of extrude.",
        when_not="Not on an empty sketch; not for prismatic shapes (use extrude); "
                 "angle 0 is a typed bad-value error.",
        side_effects="Mutates the model: adds a solid feature and sets solid_present true.",
        output="ApplyOps result with the new feature id and an updated digest."),
    "chamfer": ToolDescription(
        what="Bevel one or more edges of a solid with a flat angled face (distinct "
             "from fillet's rounded edge).",
        when="After a solid exists, to break sharp edges with a straight bevel for "
             "assembly lead-ins or deburring.",
        when_not="Not before a solid exists (no-solid error); distance must be > 0 "
                 "and smaller than adjacent edges.",
        side_effects="Mutates the solid topology; a too-large / non-positive distance "
                     "raises a typed error.",
        output="ApplyOps result; face/edge counts in the summary change."),
    "hole": ToolDescription(
        what="Drill a semantic hole (simple/counterbore/countersink/threaded) at a "
             "point on a face, through or to a blind depth.",
        when="To add a manufacturable, DFM-legible hole feature rather than a raw "
             "circle+boolean-cut.",
        when_not="Not before a solid exists; a blind hole needs a depth; diameter "
                 "must be > 0.",
        side_effects="Mutates the solid: subtracts the hole and records it as a "
                     "recognizable feature for DFM/BOM.",
        output="ApplyOps result with the hole feature id and updated digest."),
    "shell": ToolDescription(
        what="Hollow a solid to a constant wall thickness, optionally removing "
             "faces to open the shell.",
        when="For thin-walled parts: enclosures, housings, cups.",
        when_not="Not before a solid exists; thickness must be non-zero and less "
                 "than the smallest wall.",
        side_effects="Mutates the solid into a shelled body; an over-thick shell "
                     "that nulls the body raises a typed error.",
        output="ApplyOps result; volume and face counts change."),
    "draft": ToolDescription(
        what="Apply a draft angle to faces about a neutral plane (moulding/casting "
             "release).",
        when="To add manufacturing draft to vertical walls of a cast or moulded part.",
        when_not="Not before a solid exists; angle must be positive. May report "
                 "not-yet-supported on the current kernel build.",
        side_effects="Mutates face angles; validates references before applying.",
        output="ApplyOps result, or a typed not-yet-supported diagnostic."),
    "loft": ToolDescription(
        what="Loft a solid through a sequence of profile sketches (ruled or smooth).",
        when="For transitional shapes blending between differing cross-sections.",
        when_not="Needs two or more valid profiles. May report not-yet-supported "
                 "for unreliable coplanar-profile cases.",
        side_effects="Mutates the model: adds a lofted solid feature.",
        output="ApplyOps result, or a typed not-yet-supported diagnostic."),
    "sweep": ToolDescription(
        what="Sweep a profile sketch along a path to create a solid (pipes, rails, "
             "extruded channels).",
        when="For constant-section geometry that follows a curved path.",
        when_not="Needs a valid profile and path. May report not-yet-supported on "
                 "the current kernel build.",
        side_effects="Mutates the model: adds a swept solid feature.",
        output="ApplyOps result, or a typed not-yet-supported diagnostic."),
    "linear_pattern": ToolDescription(
        what="Replicate a feature in a linear array along a direction with a count "
             "and spacing.",
        when="To repeat a hole/boss/cut evenly along a line without re-modeling it.",
        when_not="Needs an existing feature to pattern; count must be >= 2.",
        side_effects="Mutates the model: unions the replicated instances.",
        output="ApplyOps result with the patterned feature and updated digest."),
    "circular_pattern": ToolDescription(
        what="Replicate a feature around an axis with a count over a total angle "
             "(bolt circles, radial ribs).",
        when="To repeat a feature evenly around a center, e.g. a bolt-hole circle.",
        when_not="Needs an existing feature and an axis; count must be >= 2.",
        side_effects="Mutates the model: unions the rotated instances.",
        output="ApplyOps result with the patterned feature and updated digest."),
    "mirror": ToolDescription(
        what="Mirror a feature or the whole body across a plane to create a "
             "symmetric copy.",
        when="For symmetric parts, to model one side and mirror it.",
        when_not="Needs an existing body/feature and a mirror plane.",
        side_effects="Mutates the model: unions the mirrored geometry.",
        output="ApplyOps result with the mirrored feature and updated digest."),
}


def _fallback_description(tag: str) -> "ToolDescription":
    """Synthesize a description for any registered op lacking a curated entry.

    Keeps the catalog registry-derived: a newly registered CISP op is covered
    automatically (with a generic 5-part description) instead of KeyError-ing.
    """
    label = tag.replace("_", " ")
    return ToolDescription(
        what=f"Apply the '{label}' CISP operation to the model.",
        when=f"When the design step calls for a {label} operation and its "
             "references/parameters are valid.",
        when_not="Not when its typed references do not exist yet, or its numeric "
                 "parameters are out of range (raises a typed diagnostic).",
        side_effects="Mutates the model per the op's semantics; reversible via "
                     "rollback; invalid input is block-and-corrected.",
        output="ApplyOps result (ok, applied, digest, diagnostics) plus the reward.")

# Output spec shared by every mutating op tool.
_OP_OUTPUT = {
    "type": "object",
    "description": "CISP applyOps result plus the reward signal.",
    "properties": {
        "ok": {"type": "boolean"},
        "applied": {"type": "integer"},
        "digest": {"type": "string"},
        "diagnostics": {"type": "array"},
        "rejected": {"type": ["object", "null"]},
        "reward": {"type": "number"},
    },
}


# ===========================================================================
# Auxiliary (non-op) tools
# ===========================================================================
def _aux_tools() -> List[ToolDefinition]:
    query_views = ["summary", "sketch_dof", "validity"]
    tools: List[ToolDefinition] = []

    tools.append(ToolDefinition(
        name="measure",
        description=ToolDescription(
            what="Read a geometric measurement / property of the current model "
                 "(dimensions, counts, mass properties) without changing it.",
            when="To check a dimension or property against the contract before "
                 "deciding the next op.",
            when_not="Not to change geometry; not a substitute for verify (which "
                     "runs the full plural checker).",
            side_effects="Read-only: never mutates the model (approval tier 1).",
            output="Structured measurement data (JSON)."),
        params=[ParamSpec("what", "string", required=False, default="summary",
                          description="Which measurement/projection to read.",
                          enum=query_views)],
        output={"type": "object", "description": "Structured measurement data."},
        annotations=annotate("measure")))

    tools.append(ToolDefinition(
        name="query",
        description=ToolDescription(
            what="Project a read-only view of model state ('summary', "
                 "'sketch_dof', 'validity').",
            when="To observe the feature tree, DOF, or validity between ops.",
            when_not="Not to mutate the model; use the op tools for changes.",
            side_effects="Read-only (approval tier 1).",
            output="The requested projection as JSON."),
        params=[ParamSpec("what", "string", required=False, default="summary",
                          description="Projection name.", enum=query_views)],
        output={"type": "object", "description": "The requested projection."},
        annotations=annotate("query")))

    verify_desc = ToolDescription(
        what="Run the plural geometry verifier (constraint solver + B-rep "
             "validity + DFM checks) and return diagnostics.",
        when="After a change, to check the model against invariants and the "
             "contract before checkpointing or exporting.",
        when_not="Not for a single measurement (use measure); does not mutate.",
        side_effects="Read-only: computes diagnostics, never mutates. Carries "
                     "the reward signal (pass = positive).",
        output="{ ok, diagnostics[] }; ok drives the reward.")
    verify_out = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}, "diagnostics": {"type": "array"},
                       "reward": {"type": "number"}},
    }
    tools.append(ToolDefinition(
        name="verify", description=verify_desc, params=[], output=verify_out,
        annotations=annotate("verify")))
    # run_check is the sec.5 alias for verify.
    tools.append(ToolDefinition(
        name="run_check", description=verify_desc, params=[], output=verify_out,
        annotations=annotate("run_check")))

    tools.append(ToolDefinition(
        name="export",
        description=ToolDescription(
            what="Serialize the current model to an external format (STEP/STL/JSON).",
            when="Once the model passes verification, to hand off the finished part.",
            when_not="Not on an unverified/degenerate model; this is an "
                     "irreversible hand-off.",
            side_effects="Destructive / irreversible external side effect (writes "
                         "an artifact); requires approval (tier 3).",
            output="{ fmt, content } — the serialized model."),
        params=[ParamSpec("fmt", "string", required=False, default="step",
                          description="Export format.", enum=["step", "stl", "json"])],
        output={"type": "object", "properties": {"fmt": {"type": "string"},
                                                  "content": {"type": "string"}}},
        annotations=annotate("export")))

    tools.append(ToolDefinition(
        name="reset",
        description=ToolDescription(
            what="Discard all model state and return the environment to an empty model.",
            when="To start a new episode / task from a clean slate.",
            when_not="Not mid-task unless abandoning the current model — it is "
                     "irreversible.",
            side_effects="Destructive: clears the feature tree and op history "
                         "(tier 3). Idempotent.",
            output="The initial observation of the empty model."),
        params=[],
        output={"type": "object", "description": "Initial observation."},
        annotations=annotate("reset")))

    tools.append(ToolDefinition(
        name="render",
        description=ToolDescription(
            what="Render the current solid to multi-view images (isometric + "
                 "orthographic) for a vision observer.",
            when="To visually inspect the shape as part of the hybrid observation.",
            when_not="Not a geometric check (use verify/measure); returns None "
                     "per view when no kernel/solid is present.",
            side_effects="Read-only: never mutates; may be a headless no-op "
                         "(returns a note) (tier 1).",
            output="{ images: {view: bytes|None}, note, fmt }."),
        params=[
            ParamSpec("views", "array", required=False, default=None,
                      description="View names to render (default iso/front/top/right)."),
            ParamSpec("fmt", "string", required=False, default="svg",
                      description="Image format.", enum=["svg", "png"]),
        ],
        output={"type": "object", "description": "Per-view images + note."},
        annotations=annotate("render")))

    return tools


# ===========================================================================
# The catalog
# ===========================================================================
class ToolCatalog:
    """The MCP-style tool catalog: one tool per CISP op + the auxiliary tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self._build_op_tools()
        for t in _aux_tools():
            self._tools[t.name] = t

    # --- construction -----------------------------------------------------
    def _build_op_tools(self) -> None:
        for tag, cls in sorted(_REGISTRY.items()):
            params: List[ParamSpec] = []
            for f in dataclasses.fields(cls):
                jtype, nullable = _map_type(str(f.type))
                required = f.default == ""  # "" sentinel means "must fill in"
                enum = _ENUMS.get((tag, f.name))
                doc = _PARAM_DOCS.get((tag, f.name), f"{f.name} for the {tag} op.")
                default = None if f.default == "" else f.default
                if isinstance(default, tuple):
                    default = list(default)
                params.append(ParamSpec(
                    name=f.name, type=jtype, required=required, default=default,
                    description=doc, enum=enum, nullable=nullable))
            desc = _OP_DESCRIPTIONS.get(tag) or _fallback_description(tag)
            self._tools[tag] = ToolDefinition(
                name=tag, description=desc, params=params, output=dict(_OP_OUTPUT),
                annotations=annotate(tag), op_tag=tag)

    # --- access -----------------------------------------------------------
    def names(self) -> List[str]:
        return sorted(self._tools)

    def op_tools(self) -> List[ToolDefinition]:
        return [t for t in self._tools.values() if t.op_tag is not None]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> ToolDefinition:
        return self.get(name)

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(f"unknown tool '{name}'", tool=name,
                                   known=self.names())

    # --- MCP surfaces -----------------------------------------------------
    def to_mcp(self) -> List[dict]:
        """The full JSON tool list a FastMCP server would register."""
        return [self._tools[n].to_mcp() for n in self.names()]

    def resources(self) -> List[dict]:
        """MCP resources = observations of model state (sec.5)."""
        return [
            {"uri": "cad://model/tree", "name": "feature_tree",
             "description": "The feature tree: sketches, entities and features "
                            "(the intent-bearing, editable model state).",
             "mimeType": "application/json"},
            {"uri": "cad://model/validity", "name": "validity",
             "description": "Plural-verifier diagnostics: constraint DOF, B-rep "
                            "validity, DFM.",
             "mimeType": "application/json"},
            {"uri": "cad://model/measurements", "name": "measurements",
             "description": "Geometric measurements / summary (counts, "
                            "solid presence, sketch DOF).",
             "mimeType": "application/json"},
        ]

    def read_resource(self, uri: str, session) -> dict:
        """Materialise a resource from a live HarnessSession (never mutates)."""
        backend = session.backend
        if uri == "cad://model/tree":
            return {"summary": backend.query("summary"),
                    "ops": [op.to_dict() for op in session.opdag.ops()]}
        if uri == "cad://model/measurements":
            return {"summary": backend.query("summary"),
                    "sketch_dof": backend.query("sketch_dof")}
        if uri == "cad://model/validity":
            diags: List[dict] = []
            for v in session.verifiers:
                diags += [d.to_dict()
                          for d in v.check(backend, session.opdag).diagnostics]
            ok = not any(d["severity"] == "error" for d in diags)
            return {"ok": ok, "diagnostics": diags}
        raise UnknownToolError(f"unknown resource '{uri}'", uri=uri)

    def prompts(self) -> List[dict]:
        """MCP prompts = op templates (sec.5)."""
        return [
            {
                "name": "rectangular_plate",
                "description": "A fully-constrained rectangular plate: sketch a "
                               "rectangle, dimension it, extrude to thickness.",
                "arguments": [
                    {"name": "width", "description": "plate width", "required": True},
                    {"name": "height", "description": "plate height", "required": True},
                    {"name": "thickness", "description": "extrude distance", "required": True},
                ],
                "template": [
                    {"op": "new_sketch", "plane": "XY"},
                    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0,
                     "w": "{width}", "h": "{height}"},
                    {"op": "constrain", "kind": "distance", "a": "e1", "value": "{width}"},
                    {"op": "constrain", "kind": "distance", "a": "e1", "value": "{height}"},
                    {"op": "extrude", "sketch": "sk1", "distance": "{thickness}"},
                ],
            },
            {
                "name": "cylinder",
                "description": "A cylinder: sketch a circle then extrude it.",
                "arguments": [
                    {"name": "radius", "description": "circle radius", "required": True},
                    {"name": "height", "description": "extrude distance", "required": True},
                ],
                "template": [
                    {"op": "new_sketch", "plane": "XY"},
                    {"op": "add_circle", "sketch": "sk1", "cx": 0.0, "cy": 0.0, "r": "{radius}"},
                    {"op": "extrude", "sketch": "sk1", "distance": "{height}"},
                ],
            },
            {
                "name": "filleted_block",
                "description": "A block with rounded edges: plate then fillet.",
                "arguments": [
                    {"name": "radius", "description": "fillet radius", "required": True},
                ],
                "template": [
                    {"op": "new_sketch", "plane": "XY"},
                    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0},
                    {"op": "extrude", "sketch": "sk1", "distance": 5.0},
                    {"op": "fillet", "edges": [], "radius": "{radius}"},
                ],
            },
        ]

    # --- execution (tool-result carries reward; raises typed errors) ------
    def call(self, name: str, arguments: Optional[Dict[str, Any]] = None,
             *, session=None) -> ToolResult:
        """Invoke a tool against a live HarnessSession.

        Op tools apply through the session and return a :class:`ToolResult`
        carrying the verifier-derived ``reward``; a rejected op raises
        :class:`ToolExecutionError` (with diagnostics + reward) for the agent to
        observe. Read-only tools return their projection with reward 0 (verify
        returns pass/fail reward). Unknown tools / bad params raise typed errors.
        """
        arguments = dict(arguments or {})
        tool = self.get(name)              # -> UnknownToolError
        tool.validate_args(arguments)      # -> ToolValidationError

        if tool.op_tag is not None:
            if session is None:
                raise ToolExecutionError(
                    f"{name}: op tool requires a session", tool=name)
            op = parse_op({"op": tool.op_tag, **arguments})
            result = session.apply_ops([op])
            reward = reward_from_apply(result)
            diags = [d.to_dict() for d in result.diagnostics]
            if not result.ok:
                raise ToolExecutionError(
                    f"{name}: op rejected by kernel/verifier", tool=name,
                    diagnostics=diags, rejected=result.rejected, reward=reward)
            content = result.to_dict()
            content["reward"] = reward
            return ToolResult(name, True, content, reward=reward, diagnostics=diags)

        return self._call_aux(tool, arguments, session)

    def _call_aux(self, tool: ToolDefinition, args: Dict[str, Any],
                  session) -> ToolResult:
        name = tool.name
        if session is None:
            raise ToolExecutionError(f"{name}: requires a session", tool=name)
        backend = session.backend

        if name in ("query", "measure"):
            what = args.get("what", "summary")
            if what == "validity":
                content = self.read_resource("cad://model/validity", session)
            else:
                content = {"what": what, "result": backend.query(what)}
            return ToolResult(name, True, content, reward=0.0)

        if name in ("verify", "run_check"):
            v = self.read_resource("cad://model/validity", session)
            warns = sum(1 for d in v["diagnostics"] if d["severity"] == "warning")
            reward = -1.0 if not v["ok"] else max(0.0, 1.0 - 0.1 * warns)
            return ToolResult(name, v["ok"], v, reward=reward,
                              diagnostics=v["diagnostics"], is_error=not v["ok"])

        if name == "export":
            fmt = args.get("fmt", "step")
            content = {"fmt": fmt, "content": backend.export(fmt)}
            return ToolResult(name, True, content, reward=0.0)

        if name == "render":
            content = _render_via_module(backend, args.get("views"), args.get("fmt", "svg"))
            return ToolResult(name, True, content, reward=0.0)

        if name == "reset":
            session.opdag.truncate(0)
            backend.reset()
            session.opdag.checkpoint("start")
            content = {"summary": backend.query("summary")}
            return ToolResult(name, True, content, reward=0.0)

        raise ToolExecutionError(f"{name}: no executor", tool=name)


# --- render hook (lazy import of render.py; graceful when absent) ----------
def _render_via_module(backend, views, fmt) -> dict:
    try:
        import surfaces.render as _render_mod  # lazy; optional cadquery/OCP under the hood
    except Exception as exc:  # noqa: BLE001
        return {"images": {}, "note": f"render module unavailable ({exc})", "fmt": fmt}
    kwargs: Dict[str, Any] = {"fmt": fmt}
    if views:
        kwargs["views"] = views
    result = _render_mod.render(backend, **kwargs)
    return {
        "images": {k: (v is not None) for k, v in result.images.items()},
        "note": result.note,
        "fmt": result.fmt,
        "any_rendered": result.any_rendered,
    }
