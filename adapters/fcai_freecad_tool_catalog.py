"""FreeCAD workbench operation catalogue: a deterministic capability model.

Mined from ``freecad-ai`` (an AI assistant workbench for FreeCAD). That project
exposes FreeCAD's Python scripting surface to an LLM as ~50 structured tools,
each a ``ToolDefinition`` with a name, a category, and a typed parameter schema
(``freecad_ai/tools/freecad_tools.py`` + ``registry.py``). The LLM agent loop,
the Qt panels and the live ``import FreeCAD`` calls are all external; what is
*not* external is the catalogue itself -- the enumeration of which FreeCAD
operations exist, which workbench each belongs to, and what typed parameters
each takes. That is a deterministic capability model, and it is the piece worth
reimplementing standalone.

How this differs from what the harness already has:

* ``backends/ocp_occt_api_catalog`` catalogues the *OCCT kernel* API -- the
  low-level ``BRepPrimAPI_MakeBox`` / ``gp_Pnt`` C++ classes and their methods.
  That is the geometry kernel underneath FreeCAD.
* ``generation/query2cad_macro`` models the FreeCAD *Part* workbench only:
  primitive solids plus boolean fuse/cut/common, serialised to a ``.FCMacro``.
* THIS module models the FreeCAD *document-object / feature* layer that sits on
  top of the kernel: the parametric PartDesign feature tree (Body, Sketch, Pad,
  Pocket, Revolve, Loft, Sweep, Fillet, Chamfer, patterns, mirror, shell, datum
  geometry), the Sketcher, Draft, Spreadsheet/expression, Assembly and
  inspection operations -- 53 operations across 11 workbench groupings -- with
  each operation's typed parameter schema, required/optional flags and enum
  domains.

An LLM that drives FreeCAD hallucinates operation names ("pad" vs "pad_sketch"),
omits required parameters, and passes out-of-domain enum values. The cheapest
deterministic guard is to check a proposed operation call against this catalogue
before it is ever handed to FreeCAD: :func:`FreeCadToolCatalog.check_call`
validates the operation name, required parameters, unknown parameters and enum
domains, and returns a structured verdict with difflib near-miss suggestions so
a generator can repair the call.

Everything here is stdlib-only and deterministic: same catalogue in, same
verdict out. No FreeCAD, no LLM, no network.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "ParamSpec",
    "Operation",
    "CallCheck",
    "FreeCadToolCatalog",
    "default_catalog",
    "WORKBENCHES",
    "PARAM_TYPES",
]

# The 11 workbench groupings the catalogue partitions operations into. These
# mirror FreeCAD's own workbench structure (PartDesign, Part, Sketcher, Draft,
# Spreadsheet, Assembly) plus functional groupings for document / inspection /
# GUI operations and the freecad-ai composite "skill" macros.
WORKBENCHES = (
    "PartDesign", "Part", "Sketcher", "Draft", "Spreadsheet", "Assembly",
    "Inspection", "Document", "Mesh", "Gui", "Skill", "Composite",
)

# Parameter primitive types used across the schema (JSON-schema-ish).
PARAM_TYPES = ("string", "number", "integer", "boolean", "array", "object")


_CATALOG = {
    'add_assembly_joint': {
        'workbench': 'Assembly', 'category': 'modeling',
        'params': (
            ('assembly_name', 'string', True, None),
            ('part1_name', 'string', True, None),
            ('face1', 'string', True, None),
            ('part2_name', 'string', True, None),
            ('face2', 'string', True, None),
            ('joint_type', 'string', False, None),
            ('label', 'string', False, None),
        ),
    },
    'add_part_to_assembly': {
        'workbench': 'Assembly', 'category': 'modeling',
        'params': (
            ('assembly_name', 'string', True, None),
            ('part_name', 'string', True, None),
            ('position', 'array', False, None),
        ),
    },
    'boolean_operation': {
        'workbench': 'Part', 'category': 'modeling',
        'params': (
            ('operation', 'string', True, ('fuse', 'cut', 'common')),
            ('object1', 'string', True, None),
            ('object2', 'string', True, None),
            ('label', 'string', False, None),
        ),
    },
    'capture_viewport': {
        'workbench': 'Gui', 'category': 'view',
        'params': (
            ('filepath', 'string', True, None),
            ('width', 'integer', False, None),
            ('height', 'integer', False, None),
            ('background', 'string', False, ('Current', 'White', 'Black', 'Transparent')),
        ),
    },
    'chamfer_edges': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('object_name', 'string', True, None),
            ('edges', 'array', False, None),
            ('size', 'number', False, None),
            ('label', 'string', False, None),
        ),
    },
    'create_assembly': {
        'workbench': 'Assembly', 'category': 'modeling',
        'params': (
            ('label', 'string', False, None),
            ('part_names', 'array', False, None),
            ('ground_first', 'boolean', False, None),
        ),
    },
    'create_body': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('label', 'string', False, None),
        ),
    },
    'create_datum_line': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('point1', 'array', False, None),
            ('point2', 'array', False, None),
            ('support', 'string', False, None),
            ('edge', 'string', False, None),
            ('axis', 'string', False, ('X', 'Y', 'Z')),
            ('body_name', 'string', False, None),
            ('label', 'string', False, None),
        ),
    },
    'create_datum_plane': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('plane', 'string', False, ('XY', 'XZ', 'YZ')),
            ('support', 'string', False, None),
            ('face', 'string', False, None),
            ('offset', 'number', False, None),
            ('body_name', 'string', False, None),
            ('label', 'string', False, None),
        ),
    },
    'create_enclosure_lid': {
        'workbench': 'Composite', 'category': 'modeling',
        'params': (
            ('length', 'number', True, None),
            ('width', 'number', True, None),
            ('wall_thickness', 'number', True, None),
            ('clearance', 'number', False, None),
            ('lip_height', 'number', False, None),
            ('label', 'string', False, None),
        ),
    },
    'create_inner_ridge': {
        'workbench': 'Composite', 'category': 'modeling',
        'params': (
            ('body_name', 'string', True, None),
            ('length', 'number', True, None),
            ('width', 'number', True, None),
            ('wall_thickness', 'number', True, None),
            ('ridge_width', 'number', False, None),
            ('ridge_height', 'number', False, None),
            ('z_position', 'number', True, None),
            ('label', 'string', False, None),
        ),
    },
    'create_primitive': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('shape_type', 'string', True, ('box', 'cylinder', 'sphere', 'cone', 'torus')),
            ('label', 'string', False, None),
            ('body_name', 'string', False, None),
            ('operation', 'string', False, ('additive', 'subtractive')),
            ('length', 'number', False, None),
            ('width', 'number', False, None),
            ('height', 'number', False, None),
            ('radius', 'number', False, None),
            ('radius2', 'number', False, None),
            ('x', 'number', False, None),
            ('y', 'number', False, None),
            ('z', 'number', False, None),
        ),
    },
    'create_sketch': {
        'workbench': 'Sketcher', 'category': 'modeling',
        'params': (
            ('plane', 'string', False, ('XY', 'XZ', 'YZ')),
            ('body_name', 'string', False, None),
            ('geometries', 'array', False, None),
            ('constraints', 'array', False, None),
            ('label', 'string', False, None),
            ('offset', 'number', False, None),
            ('support', 'string', False, None),
            ('face', 'string', False, None),
        ),
    },
    'create_snap_tabs': {
        'workbench': 'Composite', 'category': 'modeling',
        'params': (
            ('body_name', 'string', True, None),
            ('length', 'number', True, None),
            ('width', 'number', True, None),
            ('wall_thickness', 'number', True, None),
            ('clearance', 'number', False, None),
            ('lip_height', 'number', False, None),
            ('tab_width', 'number', False, None),
            ('tab_height', 'number', False, None),
            ('protrusion', 'number', False, None),
            ('label', 'string', False, None),
        ),
    },
    'create_spreadsheet': {
        'workbench': 'Spreadsheet', 'category': 'modeling',
        'params': (
            ('variables', 'object', True, None),
            ('label', 'string', False, None),
        ),
    },
    'create_variable_set': {
        'workbench': 'Spreadsheet', 'category': 'modeling',
        'params': (
            ('variables', 'object', True, None),
            ('label', 'string', False, None),
        ),
    },
    'create_wedge': {
        'workbench': 'Part', 'category': 'modeling',
        'params': (
            ('length', 'number', False, None),
            ('width', 'number', False, None),
            ('height', 'number', False, None),
            ('top_length', 'number', False, None),
            ('top_width', 'number', False, None),
            ('label', 'string', False, None),
            ('body_name', 'string', False, None),
            ('operation', 'string', False, ('additive', 'subtractive')),
            ('x', 'number', False, None),
            ('y', 'number', False, None),
            ('z', 'number', False, None),
        ),
    },
    'describe_model': {
        'workbench': 'Inspection', 'category': 'query',
        'params': (
            ('object_name', 'string', True, None),
        ),
    },
    'duplicate_object': {
        'workbench': 'Draft', 'category': 'modeling',
        'params': (
            ('object_name', 'string', True, None),
            ('translate_x', 'number', False, None),
            ('translate_y', 'number', False, None),
            ('translate_z', 'number', False, None),
            ('rotate_axis_x', 'number', False, None),
            ('rotate_axis_y', 'number', False, None),
            ('rotate_axis_z', 'number', False, None),
            ('rotate_angle', 'number', False, None),
            ('label', 'string', False, None),
        ),
    },
    'edit_sketch': {
        'workbench': 'Sketcher', 'category': 'modeling',
        'params': (
            ('sketch_name', 'string', True, None),
            ('clear_all', 'boolean', False, None),
            ('add_geometries', 'array', True, None),
            ('remove_geometries', 'array', False, None),
            ('add_constraints', 'array', True, None),
            ('remove_constraints', 'array', False, None),
            ('label', 'string', False, None),
        ),
    },
    'execute_code': {
        'workbench': 'Document', 'category': 'general',
        'params': (
            ('code', 'string', True, None),
        ),
    },
    'export_model': {
        'workbench': 'Mesh', 'category': 'file',
        'params': (
            ('format', 'string', True, ('stl', 'step', 'iges')),
            ('filename', 'string', True, None),
            ('objects', 'array', False, None),
        ),
    },
    'fillet_edges': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('object_name', 'string', True, None),
            ('edges', 'array', False, None),
            ('radius', 'number', False, None),
            ('label', 'string', False, None),
        ),
    },
    'get_document_state': {
        'workbench': 'Document', 'category': 'query',
        'params': (
        ),
    },
    'linear_pattern': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('feature_name', 'string', True, None),
            ('direction', 'string', False, None),
            ('length', 'number', True, None),
            ('occurrences', 'integer', True, None),
            ('label', 'string', False, None),
        ),
    },
    'list_documents': {
        'workbench': 'Document', 'category': 'query',
        'params': (
        ),
    },
    'list_edges': {
        'workbench': 'Inspection', 'category': 'query',
        'params': (
            ('object_name', 'string', True, None),
            ('filter', 'string', False, None),
        ),
    },
    'list_faces': {
        'workbench': 'Inspection', 'category': 'query',
        'params': (
            ('object_name', 'string', True, None),
            ('filter', 'string', False, None),
        ),
    },
    'loft_sketches': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('section_names', 'array', True, None),
            ('closed', 'boolean', False, None),
            ('ruled', 'boolean', False, None),
            ('subtractive', 'boolean', False, None),
            ('body_name', 'string', False, None),
            ('label', 'string', False, None),
        ),
    },
    'measure': {
        'workbench': 'Inspection', 'category': 'query',
        'params': (
            ('measure_type', 'string', True, ('volume', 'area', 'bbox', 'distance', 'edges')),
            ('target', 'string', True, None),
            ('target2', 'string', False, None),
        ),
    },
    'mirror_feature': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('feature_name', 'string', True, None),
            ('plane', 'string', False, None),
            ('label', 'string', False, None),
        ),
    },
    'modify_property': {
        'workbench': 'Document', 'category': 'modeling',
        'params': (
            ('object_name', 'string', True, None),
            ('property_name', 'string', True, None),
            ('value', 'string', True, None),
        ),
    },
    'multi_transform': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('feature_names', 'array', True, None),
            ('transformations', 'array', True, None),
            ('label', 'string', False, None),
        ),
    },
    'pad_sketch': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('sketch_name', 'string', True, None),
            ('length', 'string', False, None),
            ('symmetric', 'boolean', False, None),
            ('label', 'string', False, None),
            ('body_name', 'string', False, None),
        ),
    },
    'pocket_sketch': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('sketch_name', 'string', True, None),
            ('length', 'number', False, None),
            ('through_all', 'boolean', False, None),
            ('label', 'string', False, None),
            ('body_name', 'string', False, None),
        ),
    },
    'polar_pattern': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('feature_name', 'string', True, None),
            ('axis', 'string', False, None),
            ('angle', 'number', False, None),
            ('occurrences', 'integer', True, None),
            ('label', 'string', False, None),
        ),
    },
    'redo': {
        'workbench': 'Document', 'category': 'general',
        'params': (
            ('steps', 'integer', False, None),
        ),
    },
    'report_skill_params': {
        'workbench': 'Skill', 'category': 'query',
        'params': (
            ('params', 'object', True, None),
        ),
    },
    'revolve_sketch': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('sketch_name', 'string', True, None),
            ('axis', 'string', False, None),
            ('angle', 'number', False, None),
            ('subtractive', 'boolean', False, None),
            ('body_name', 'string', False, None),
            ('label', 'string', False, None),
        ),
    },
    'run_macro': {
        'workbench': 'Document', 'category': 'general',
        'params': (
            ('macro', 'string', True, None),
        ),
    },
    'scale_object': {
        'workbench': 'Part', 'category': 'modeling',
        'params': (
            ('object_name', 'string', True, None),
            ('scale_x', 'number', False, None),
            ('scale_y', 'number', False, None),
            ('scale_z', 'number', False, None),
            ('uniform', 'number', False, None),
            ('copy', 'boolean', False, None),
            ('label', 'string', False, None),
        ),
    },
    'section_object': {
        'workbench': 'Part', 'category': 'modeling',
        'params': (
            ('object_name', 'string', True, None),
            ('tool_object', 'string', False, None),
            ('plane', 'string', False, ('XY', 'XZ', 'YZ')),
            ('offset', 'number', False, None),
            ('label', 'string', False, None),
        ),
    },
    'select_geometry': {
        'workbench': 'Gui', 'category': 'interactive',
        'params': (
            ('prompt', 'string', False, None),
            ('select_type', 'string', False, ('any', 'edge', 'face', 'vertex')),
            ('max_count', 'integer', False, None),
        ),
    },
    'set_expression': {
        'workbench': 'Spreadsheet', 'category': 'modeling',
        'params': (
            ('object_name', 'string', True, None),
            ('property_name', 'string', True, None),
            ('expression', 'string', True, None),
        ),
    },
    'set_view': {
        'workbench': 'Gui', 'category': 'view',
        'params': (
            ('orientation', 'string', True, ('isometric', 'front', 'back', 'top', 'bottom', 'left', 'right')),
            ('fit_all', 'boolean', False, None),
            ('projection', 'string', False, ('Orthographic', 'Perspective')),
        ),
    },
    'shell_object': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('object_name', 'string', True, None),
            ('faces', 'array', False, None),
            ('thickness', 'number', False, None),
            ('join', 'string', False, ('Arc', 'Intersection')),
            ('reversed', 'boolean', False, None),
            ('label', 'string', False, None),
        ),
    },
    'sweep_sketch': {
        'workbench': 'PartDesign', 'category': 'modeling',
        'params': (
            ('profile_name', 'string', True, None),
            ('spine_name', 'string', True, None),
            ('subtractive', 'boolean', False, None),
            ('body_name', 'string', False, None),
            ('label', 'string', False, None),
        ),
    },
    'switch_document': {
        'workbench': 'Document', 'category': 'query',
        'params': (
            ('document_name', 'string', True, None),
        ),
    },
    'transform_object': {
        'workbench': 'Part', 'category': 'modeling',
        'params': (
            ('object_name', 'string', True, None),
            ('translate_x', 'number', False, None),
            ('translate_y', 'number', False, None),
            ('translate_z', 'number', False, None),
            ('rotate_axis_x', 'number', False, None),
            ('rotate_axis_y', 'number', False, None),
            ('rotate_axis_z', 'number', False, None),
            ('rotate_angle', 'number', False, None),
            ('relative', 'boolean', False, None),
        ),
    },
    'undo': {
        'workbench': 'Document', 'category': 'general',
        'params': (
            ('steps', 'integer', False, None),
            ('until', 'string', False, None),
        ),
    },
    'undo_history': {
        'workbench': 'Document', 'category': 'query',
        'params': (
        ),
    },
    'use_skill': {
        'workbench': 'Skill', 'category': 'query',
        'params': (
            ('name', 'string', True, None),
            ('args', 'string', False, None),
        ),
    },
    'zoom_object': {
        'workbench': 'Gui', 'category': 'view',
        'params': (
            ('object_name', 'string', True, None),
        ),
    },
}

@dataclass(frozen=True)
class ParamSpec:
    """One typed parameter of an operation."""
    name: str
    type: str
    required: bool
    enum: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class Operation:
    """One FreeCAD operation exposed as a callable tool."""
    name: str
    workbench: str
    category: str
    params: Tuple[ParamSpec, ...]

    def required_params(self) -> Tuple[str, ...]:
        return tuple(p.name for p in self.params if p.required)

    def param(self, name: str) -> Optional[ParamSpec]:
        for p in self.params:
            if p.name == name:
                return p
        return None


@dataclass
class CallCheck:
    """Structured verdict for a proposed operation call."""
    ok: bool
    operation: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestion: Optional[str] = None  # near-miss operation name, if unknown

    def __bool__(self) -> bool:
        return self.ok


def _build_operations() -> Dict[str, Operation]:
    ops: Dict[str, Operation] = {}
    for name, info in _CATALOG.items():
        params = tuple(
            ParamSpec(pn, pt, req, en) for (pn, pt, req, en) in info["params"]
        )
        ops[name] = Operation(
            name=name,
            workbench=info["workbench"],
            category=info["category"],
            params=params,
        )
    return ops


class FreeCadToolCatalog:
    """Queryable catalogue of FreeCAD operations with call validation."""

    def __init__(self, operations: Optional[Dict[str, Operation]] = None):
        self._ops: Dict[str, Operation] = (
            dict(operations) if operations is not None else _build_operations()
        )

    # -- queries ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._ops)

    def __contains__(self, name: str) -> bool:
        return name in self._ops

    def names(self) -> List[str]:
        return sorted(self._ops)

    def get(self, name: str) -> Optional[Operation]:
        return self._ops.get(name)

    def by_workbench(self, workbench: str) -> List[Operation]:
        return sorted(
            (o for o in self._ops.values() if o.workbench == workbench),
            key=lambda o: o.name,
        )

    def by_category(self, category: str) -> List[Operation]:
        return sorted(
            (o for o in self._ops.values() if o.category == category),
            key=lambda o: o.name,
        )

    def workbench_histogram(self) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for o in self._ops.values():
            hist[o.workbench] = hist.get(o.workbench, 0) + 1
        return hist

    def suggest(self, name: str, n: int = 1) -> List[str]:
        """Return the closest known operation name(s) to ``name``."""
        return difflib.get_close_matches(name, list(self._ops), n=n, cutoff=0.5)

    # -- validation ---------------------------------------------------------
    def check_call(self, name: str, params: Dict[str, object]) -> CallCheck:
        """Validate a proposed operation call against the catalogue.

        Checks, in order: operation exists (with near-miss suggestion),
        required parameters present, unknown parameters flagged (warning),
        parameter values inside their enum domain when one is declared.
        """
        op = self._ops.get(name)
        if op is None:
            near = self.suggest(name, n=1)
            return CallCheck(
                ok=False,
                operation=name,
                errors=["unknown operation '%s'" % name],
                suggestion=near[0] if near else None,
            )

        errors: List[str] = []
        warnings: List[str] = []
        known = {p.name: p for p in op.params}

        for req in op.required_params():
            if req not in params:
                errors.append("missing required parameter '%s'" % req)

        for given in params:
            if given not in known:
                near = difflib.get_close_matches(
                    given, list(known), n=1, cutoff=0.5)
                hint = (" (did you mean '%s'?)" % near[0]) if near else ""
                warnings.append("unknown parameter '%s'%s" % (given, hint))
                continue
            spec = known[given]
            if spec.enum is not None:
                val = params[given]
                if isinstance(val, str) and val not in spec.enum:
                    errors.append(
                        "parameter '%s'='%s' not in %s"
                        % (given, val, list(spec.enum))
                    )

        return CallCheck(
            ok=not errors,
            operation=name,
            errors=errors,
            warnings=warnings,
        )

    def to_json_schema(self, name: str) -> Dict[str, object]:
        """Emit a JSON-schema object for one operation's parameters."""
        op = self._ops.get(name)
        if op is None:
            raise KeyError(name)
        properties: Dict[str, object] = {}
        required: List[str] = []
        for p in op.params:
            prop: Dict[str, object] = {"type": p.type}
            if p.enum is not None:
                prop["enum"] = list(p.enum)
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        schema: Dict[str, object] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema


_DEFAULT: Optional[FreeCadToolCatalog] = None


def default_catalog() -> FreeCadToolCatalog:
    """Return the shared default catalogue instance (built once)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = FreeCadToolCatalog()
    return _DEFAULT
