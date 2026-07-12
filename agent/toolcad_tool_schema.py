"""TOOLCAD tool/action schema — typed CAD-tool calls for tool-using agents.

TOOLCAD ("Exploring Tool-Using Large Language Models in Text-to-CAD Generation
with Reinforcement Learning") deploys the LLM as a *tool-using agent*: instead
of emitting CAD code or raw tokens, the agent selects and invokes typed CAD
tools (the paper's ``TOOLLIBRARY``, App. A.2 / Fig. 11-13):

    freecad-set_coord_system, freecad-create_complex_sketch,
    freecad-create_simple_sketch, freecad-boolean_operation,
    freecad-multiple_fuse, freecad-extrude_face

Each tool has a *signature* (typed, ordered arguments with required/optional and
allowed literal sets) and, when executed, returns an ``InterfaceResult`` — the
paper's structured message labelled ``success`` or ``fail`` (Fig. 12/13). This
is the executable tool interface the agent's ``<tool_call>`` names and fills.

This is deliberately DISTINCT from ``surfaces.mcp.tools.ToolCatalog`` (which
wraps the repository's CISP ops for the harness spine): this module models the
*agent-facing tool contract* — the fixed primitive-based tool library, argument
typing/validation, and the success/fail InterfaceResult the paper's step-wise
reward and reflective ReAct loop consume. No FreeCAD/OCCT dependency: execution
is a deterministic in-memory geometry-object bookkeeper so the schema, argument
validation, and success/fail semantics are testable stdlib-only.

Stdlib only, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple


# --- Argument specification ------------------------------------------------

@dataclass(frozen=True)
class ArgSpec:
    """One typed argument of a CAD tool signature."""

    name: str
    type: str  # "str" | "int" | "float" | "number" | "bool" | "list" | "literal"
    required: bool = True
    choices: Tuple[str, ...] = ()  # for type == "literal"

    _PY = {
        "str": (str,),
        "int": (int,),
        "float": (float, int),
        "number": (int, float),
        "bool": (bool,),
        "list": (list, tuple),
    }

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("arg name required")
        if self.type not in self._PY and self.type != "literal":
            raise ValueError(f"unknown arg type: {self.type}")
        if self.type == "literal" and not self.choices:
            raise ValueError("literal arg requires choices")

    def check(self, value: Any) -> Optional[str]:
        """Return an error string if ``value`` is invalid for this arg, else None."""
        if self.type == "literal":
            if value not in self.choices:
                return (
                    f"argument '{self.name}' must be one of "
                    f"{list(self.choices)}, got {value!r}"
                )
            return None
        # bool is a subclass of int: reject bools where a number is wanted.
        if self.type in ("int", "float", "number") and isinstance(value, bool):
            return f"argument '{self.name}' expects {self.type}, got bool"
        if not isinstance(value, self._PY[self.type]):
            return (
                f"argument '{self.name}' expects {self.type}, "
                f"got {type(value).__name__}"
            )
        return None


# --- Tool signature --------------------------------------------------------

@dataclass(frozen=True)
class ToolSignature:
    """Typed signature of a single CAD tool in the TOOLCAD tool library."""

    name: str
    summary: str
    args: Tuple[ArgSpec, ...] = ()
    produces: bool = True  # whether a successful call creates a new geometry object

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.summary.strip():
            raise ValueError("tool signature needs name and summary")
        seen: set[str] = set()
        for a in self.args:
            if a.name in seen:
                raise ValueError(f"duplicate arg '{a.name}' in tool {self.name}")
            seen.add(a.name)

    @property
    def required_args(self) -> Tuple[str, ...]:
        return tuple(a.name for a in self.args if a.required)

    def validate(self, arguments: Mapping[str, Any]) -> Tuple[str, ...]:
        """Return a tuple of error messages (empty == valid)."""
        errors: list[str] = []
        by_name = {a.name: a for a in self.args}
        for name in self.required_args:
            if name not in arguments:
                errors.append(f"missing required argument '{name}'")
        for key, value in arguments.items():
            spec = by_name.get(key)
            if spec is None:
                errors.append(f"unknown argument '{key}' for tool {self.name}")
                continue
            err = spec.check(value)
            if err:
                errors.append(err)
        return tuple(errors)


# --- Structured interface result (Fig. 12/13) ------------------------------

@dataclass(frozen=True)
class InterfaceResult:
    """The paper's structured tool response, labelled success or fail."""

    success: bool
    description: str
    produced_object: Optional[str] = None  # name of the new geometry entity

    @property
    def label(self) -> str:
        return "success" if self.success else "fail"


# --- Typed tool call -------------------------------------------------------

@dataclass(frozen=True)
class ToolCall:
    """A parsed, typed tool invocation: ``name`` + ``arguments``."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool call needs a name")


# --- Tool library + executable environment ---------------------------------

class ToolLibrary:
    """Registry of typed CAD tools (the paper's fixed TOOLLIBRARY)."""

    def __init__(self, signatures: Sequence[ToolSignature] = ()) -> None:
        self._sigs: dict[str, ToolSignature] = {}
        for s in signatures:
            self.register(s)

    def register(self, sig: ToolSignature) -> None:
        if sig.name in self._sigs:
            raise ValueError(f"duplicate tool: {sig.name}")
        self._sigs[sig.name] = sig

    def __contains__(self, name: str) -> bool:
        return name in self._sigs

    def names(self) -> Tuple[str, ...]:
        return tuple(self._sigs)

    def get(self, name: str) -> ToolSignature:
        return self._sigs[name]

    def validate_call(self, call: ToolCall) -> Tuple[str, ...]:
        """Validate a call against the library; unknown tool is an error."""
        sig = self._sigs.get(call.name)
        if sig is None:
            return (f"unknown tool '{call.name}'",)
        return sig.validate(call.arguments)


class ToolExecutionState:
    """Deterministic in-memory geometry-object bookkeeper.

    Mirrors the paper's "geometric object list of actual CAD entities" (Sec.
    3.3) used to alleviate tool-execution hallucinations. Executes typed tool
    calls, enforcing referential preconditions (e.g. a boolean_operation's
    operands must already exist) and returning success/fail InterfaceResults.
    """

    def __init__(self, library: ToolLibrary) -> None:
        self._library = library
        self._objects: dict[str, str] = {}  # name -> creating tool
        self._counter = 0

    @property
    def objects(self) -> Tuple[str, ...]:
        return tuple(self._objects)

    def _fresh_name(self, sig: ToolSignature, arguments: Mapping[str, Any]) -> str:
        for key in ("name", "sketch_name"):
            val = arguments.get(key)
            if isinstance(val, str) and val.strip():
                return val
        self._counter += 1
        return f"{sig.name}_{self._counter}"

    def execute(self, call: ToolCall) -> InterfaceResult:
        errors = self._library.validate_call(call)
        if errors:
            return InterfaceResult(False, "; ".join(errors))
        sig = self._library.get(call.name)

        # Referential precondition: any argument named *_object_name that refers
        # to an operand must already exist in the geometry-object list.
        for key, value in call.arguments.items():
            if key.endswith("_object_name") and isinstance(value, str):
                if value not in self._objects:
                    return InterfaceResult(
                        False,
                        f"operand '{value}' does not exist in the current model",
                    )

        if not sig.produces:
            return InterfaceResult(True, f"{sig.name} succeeded")

        obj = self._fresh_name(sig, call.arguments)
        if obj in self._objects:
            return InterfaceResult(
                False, f"object '{obj}' already exists; choose a unique name"
            )
        self._objects[obj] = sig.name
        return InterfaceResult(
            True, f"{sig.name} created '{obj}'", produced_object=obj
        )


# --- Default TOOLCAD library (App. A.2) ------------------------------------

def default_toolcad_library() -> ToolLibrary:
    """Return the paper's fixed primitive-based CAD tool library."""
    return ToolLibrary((
        ToolSignature(
            "set_coord_system",
            "Create/position a coordinate system for the next part.",
            (
                ArgSpec("origin", "list", required=True),
                ArgSpec("euler_angles", "list", required=False),
                ArgSpec("name", "str", required=False),
            ),
        ),
        ToolSignature(
            "create_simple_sketch",
            "Draw a single-loop 2D sketch profile.",
            (
                ArgSpec("profile", "str", required=True),
                ArgSpec("sketch_name", "str", required=False),
            ),
        ),
        ToolSignature(
            "create_complex_sketch",
            "Batch-draw a composite sketch of lines, circles, arcs, splines.",
            (
                ArgSpec("elements", "list", required=True),
                ArgSpec("sketch_name", "str", required=False),
            ),
        ),
        ToolSignature(
            "extrude_face",
            "Extrude a sketch-derived face into a 3D solid.",
            (
                ArgSpec("sketch_name", "str", required=True),
                ArgSpec("distance", "number", required=True),
                ArgSpec("name", "str", required=False),
            ),
        ),
        ToolSignature(
            "boolean_operation",
            "cut/fuse/common between two existing solids.",
            (
                ArgSpec("base_object_name", "str", required=True),
                ArgSpec("tool_object_name", "str", required=True),
                ArgSpec("operation", "literal", required=True,
                        choices=("cut", "fuse", "common")),
                ArgSpec("name", "str", required=False),
            ),
        ),
        ToolSignature(
            "multiple_fuse",
            "Fuse a list of existing solids into one.",
            (
                ArgSpec("object_names", "list", required=True),
                ArgSpec("name", "str", required=False),
            ),
        ),
    ))
