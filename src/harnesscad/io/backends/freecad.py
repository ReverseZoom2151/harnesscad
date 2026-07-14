"""FreeCADBackend — a real parametric B-rep GeometryBackend over FreeCAD.

FreeCAD is the closest thing in the open-source world to the harness's own model:
an op stream that builds a *feature tree* over a real B-rep kernel (OCCT). This
backend drives it headless and answers every query from the kernel's **analytic**
mass properties — circles are real ``Part.Circle`` arcs, never polygons, so a
boolean cut removes exactly the volume it should, to machine precision.

How it works
------------
FreeCAD ships its own Python (a different build from the host's — 3.11 vs 3.12
here), so it cannot be imported in-process. Instead:

1. The CISP op stream is held by a composed :class:`~harnesscad.io.backends.frep.FRepBackend`
   (via :class:`~harnesscad.io.backends.external.ExternalToolBackend`), which owns
   id allocation, sketch DOF, block-and-correct, ``SetParam`` replay and the op
   log — so the harness drives this backend *identically* to frep/stub/cadquery.
2. That model's kernel-neutral CSG tree (``FRepBackend.root()``) is serialised to
   JSON and lowered onto ``Part`` shapes by :mod:`freecad_driver`, run under
   ``freecadcmd``.
3. The driver writes ``result.json`` (exact B-rep volume / area / topology /
   feature tree) and the requested exports. Results travel through a FILE, never
   stdout: ``freecadcmd`` redirects ``print`` into FreeCAD's own console.

Determinism: the driver + its spec are written to a directory named by the
SHA-256 of the program text, so the same model never re-runs the tool and no wall
clock or PID ever enters a path.

Absence: if FreeCAD is not installed the constructor raises
:class:`~harnesscad.io.backends.base.BackendUnavailable`, so the CISP server falls
back and the suite skips. Override discovery with ``HARNESSCAD_FREECAD``.

Three modules that the mining campaign built for exactly this, and that nothing
used until now, are load-bearing here:

* :mod:`harnesscad.io.adapters.freecad_catalog` — the 53-operation FreeCAD tool
  catalogue. Every CISP op is mapped to the FreeCAD operation that realises it
  and validated against the catalogue's parameter spec *before* it can mutate
  state, so an op this backend cannot honour is refused with a typed diagnostic
  naming the real FreeCAD operation.
* :mod:`harnesscad.domain.programs.expressions.freecad_expressions` — FreeCAD's
  parametric expression engine. ``SetParam`` accepts an expression *string*
  (``"2 * 12 + 4mm"``), evaluated with FreeCAD's own grammar and units.
* :mod:`harnesscad.io.formats.freecad_document` — the document-object wire codec.
  ``query('document')`` returns the real FreeCAD feature tree (TypeIds,
  Placements, per-object shape info) decoded through it.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from harnesscad.core.cisp.ops import Op, SetParam
from harnesscad.domain.programs.expressions import freecad_expressions
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.adapters.freecad_catalog import default_catalog
from harnesscad.io.backends.base import ApplyResult, BackendUnavailable
from harnesscad.io.backends.external import (
    DEFAULT_TIMEOUT, ExternalToolBackend, blend_radius, cache_dir,
    find_executable, program_digest,
)
from harnesscad.io.backends.freecad_driver import DRIVER_SOURCE
from harnesscad.io.formats import freecad_document as fcdoc

#: Where FreeCAD's headless binary hides, per platform. ``find_executable``
#: prefers ``HARNESSCAD_FREECAD``, then PATH, then these globs (newest wins).
EXECUTABLE_NAMES = ("freecadcmd", "FreeCADCmd", "freecad", "FreeCAD")
EXECUTABLE_PATTERNS = (
    os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs",
                 "FreeCAD*", "bin", "freecadcmd.exe"),
    os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs",
                 "FreeCAD*", "bin", "FreeCADCmd.exe"),
    r"C:\Program Files\FreeCAD*\bin\FreeCADCmd.exe",
    r"C:\Program Files\FreeCAD*\bin\freecadcmd.exe",
    "/usr/bin/freecadcmd",
    "/usr/local/bin/freecadcmd",
    "/Applications/FreeCAD.app/Contents/Resources/bin/FreeCADCmd",
    "/Applications/FreeCAD.app/Contents/MacOS/FreeCADCmd",
)

#: Every CISP op mapped to the FreeCAD operation (in the 53-op catalogue) that
#: realises it. This is the backend's contract with FreeCAD, made checkable.
OP_TO_FREECAD: Dict[str, str] = {
    "new_sketch": "create_sketch",
    "add_point": "edit_sketch",
    "add_line": "edit_sketch",
    "add_circle": "edit_sketch",
    "add_rectangle": "edit_sketch",
    "constrain": "edit_sketch",
    "extrude": "pad_sketch",
    "revolve": "revolve_sketch",
    "boolean": "boolean_operation",
    "fillet": "fillet_edges",
    "chamfer": "chamfer_edges",
    "hole": "pocket_sketch",
    "shell": "shell_object",
    "linear_pattern": "linear_pattern",
    "circular_pattern": "polar_pattern",
    "mirror": "mirror_feature",
    "add_instance": "add_part_to_assembly",
    "mate": "add_assembly_joint",
    "set_param": "modify_property",
    # Honestly unmapped: the F-rep model the op state is held in cannot express
    # these, so they are refused rather than faked (see UNSUPPORTED).
    "draft": "",
    "loft": "loft_sketches",
    "sweep": "sweep_sketch",
}

#: FreeCAD TypeIds for the feature tree, per F-rep body kind.
TYPE_IDS: Dict[str, str] = {
    "extrude": "PartDesign::Pad",
    "revolve": "PartDesign::Revolution",
    "boolean": "Part::Boolean",
    "fillet": "PartDesign::Fillet",
    "chamfer": "PartDesign::Chamfer",
    "hole": "PartDesign::Pocket",
    "shell": "PartDesign::Thickness",
    "mirror": "PartDesign::Mirrored",
    "linear_pattern": "PartDesign::LinearPattern",
    "circular_pattern": "PartDesign::PolarPattern",
}


def _err(code: str, msg: str, where: Optional[str] = None) -> ApplyResult:
    return ApplyResult(False, [], [Diagnostic(Severity.ERROR, code, msg, where)])


class FreeCADBackend(ExternalToolBackend):
    """A GeometryBackend that lowers the CISP op stream onto FreeCAD's B-rep kernel."""

    TOOL = "freecad"

    #: Ops FreeCAD *could* do but this backend's op-state model cannot express.
    #: Refused BEFORE they reach the op log, with the real FreeCAD operation named
    #: so the message is actionable rather than a bare "unsupported".
    UNSUPPORTED: Dict[str, str] = {
        "draft": "no draft-angle feature is wired (FreeCAD has no single op for it)",
        "loft": "needs two positioned profiles; the op model holds coplanar "
                "sketches only (FreeCAD op: loft_sketches)",
        "sweep": "needs a 3D path; the op model holds coplanar sketches only "
                 "(FreeCAD op: sweep_sketch)",
    }

    #: Real B-rep exports -- STEP and BREP are the kernel's own, not a mesh.
    FORMATS: Tuple[str, ...] = ("step", "brep", "iges", "stl", "stl-ascii",
                                "stl-binary", "glb")

    #: The exports the driver always produces (so one FreeCAD run answers all).
    DRIVER_EXPORTS: Tuple[str, ...] = ("stl", "step", "brep", "iges")

    def __init__(self, executable: Optional[str] = None,
                 timeout: int = DEFAULT_TIMEOUT) -> None:
        super().__init__(executable=executable, timeout=timeout)
        self.catalog = default_catalog()
        self._result_cache: Optional[Tuple[str, dict]] = None

    # -- discovery ---------------------------------------------------------
    @classmethod
    def locate(cls) -> str:
        path, searched = find_executable("HARNESSCAD_FREECAD", EXECUTABLE_NAMES,
                                         EXECUTABLE_PATTERNS)
        if path is None:
            raise BackendUnavailable(
                "freecad",
                "FreeCAD not found. Install it (winget install FreeCAD.FreeCAD, "
                "apt install freecad, brew install --cask freecad) or point "
                "HARNESSCAD_FREECAD at freecadcmd.",
                searched)
        return path

    # -- op stream ---------------------------------------------------------
    def reset(self) -> None:
        super().reset()
        self._result_cache = None

    def apply(self, op: Op) -> ApplyResult:
        """Validate against FreeCAD's real tool catalogue, then delegate.

        Two gates before anything can mutate: the op must map to an operation the
        53-op FreeCAD catalogue actually has, and (for ``SetParam``) an expression
        value must parse under FreeCAD's own expression grammar.
        """
        tag = getattr(type(op), "OP", "")
        bad = self._catalog_check(tag)
        if bad is not None:
            return bad
        if isinstance(op, SetParam):
            op, bad = self._resolve_expression(op)
            if bad is not None:
                return bad
        result = super().apply(op)
        if result.ok:
            self._result_cache = None
        return result

    def _catalog_check(self, tag: str) -> Optional[ApplyResult]:
        """Refuse an op that maps to no FreeCAD operation, with a suggestion."""
        if not tag or tag in self.UNSUPPORTED:
            return None  # UNSUPPORTED is reported by the base with its own reason
        name = OP_TO_FREECAD.get(tag)
        if not name:
            return _err("unsupported-op",
                        "the freecad backend has no FreeCAD operation for '%s'" % tag)
        if name not in self.catalog:
            suggestion = self.catalog.suggest(name)
            return _err("unsupported-op",
                        "op '%s' maps to FreeCAD operation '%s', which is not in "
                        "the tool catalogue%s" % (
                            tag, name,
                            "; did you mean '%s'?" % suggestion[0] if suggestion else ""))
        return None

    def _resolve_expression(self, op: SetParam) -> Tuple[SetParam, Optional[ApplyResult]]:
        """Evaluate a string ``SetParam`` value as a FreeCAD parametric expression.

        ``SetParam(target='f1', param='distance', value='2 * 4 + 4mm')`` is a
        legal FreeCAD expression; the engine understands its grammar, functions
        and units. A non-string value passes through untouched.
        """
        if not isinstance(op.value, str):
            return op, None
        try:
            expr = freecad_expressions.parse(op.value)
            env = self._expression_env()
            missing = [k for k in expr.reference_keys() if k not in env]
            if missing:
                return op, _err("bad-ref",
                                "expression '%s' references unknown %s"
                                % (op.value, ", ".join(sorted(missing))))
            value = expr.evaluate(env)
        except freecad_expressions.ExpressionError as exc:
            return op, _err("bad-value",
                            "invalid FreeCAD expression '%s': %s" % (op.value, exc))
        return SetParam(target=op.target, param=op.param, value=value), None

    def _expression_env(self) -> Dict[str, float]:
        """Numeric params of the current op log, keyed as FreeCAD references.

        ``SetParam`` targets an op by its INDEX in the op log, so the same index
        is what an expression refers to: op ``i``'s numeric params are addressable
        as ``op<i>.<param>`` (``op2.distance``) -- FreeCAD's own ``Pad.Length``
        reference shape, over the harness's op stream. So an edit can be written
        relative to the model it is editing: ``SetParam(2, 'distance',
        '2 * op2.distance')`` doubles the pad.
        """
        env: Dict[str, float] = {}
        for i, op in enumerate(self._frep._oplog):
            for key, val in vars(op).items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    env["op%d.%s" % (i, key)] = float(val)
        return env

    # -- the tool ----------------------------------------------------------
    def spec(self) -> dict:
        """The JSON the driver consumes: the CSG tree + the final edge blend."""
        root = self.root()
        r = c = 0.0
        if root is not None:
            r, c = blend_radius(root)
        return {
            "root": None if root is None else root.spec(),
            "round": r,
            "chamfer": c,
            "exports": list(self.DRIVER_EXPORTS),
        }

    def program(self) -> str:
        """The driver text + its spec, as ONE string, so the content hash covers both."""
        payload = json.dumps(self.spec(), sort_keys=True, separators=(",", ":"))
        return "# spec: %s\n%s" % (payload, DRIVER_SOURCE)

    def _run(self, source: str, workdir: str, out_path: str) -> None:
        """Write driver + spec into the content-addressed workdir and run FreeCAD."""
        driver = os.path.join(workdir, "driver.py")
        with open(driver, "w", encoding="utf-8") as fh:
            fh.write(DRIVER_SOURCE)
        with open(os.path.join(workdir, "spec.json"), "w", encoding="utf-8") as fh:
            json.dump(self.spec(), fh, sort_keys=True)
        result_path = os.path.join(workdir, "result.json")
        if os.path.isfile(result_path):
            os.remove(result_path)
        proc = subprocess.run([self.executable, driver], cwd=workdir,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=self.timeout)
        if not os.path.isfile(result_path):
            raise RuntimeError(
                "FreeCAD produced no result (exit %d): %s"
                % (proc.returncode,
                   (proc.stderr or b"").decode("utf-8", "replace")[-500:]))
        with open(result_path, encoding="utf-8") as fh:
            result = json.load(fh)
        if not result.get("ok"):
            raise RuntimeError("FreeCAD failed to build the model: %s"
                               % result.get("error", "unknown"))

    def _artifacts(self) -> dict:
        """Run FreeCAD (once, cached on the model digest) and return its result.json."""
        key = self.state_digest()
        if self._result_cache is not None and self._result_cache[0] == key:
            return self._result_cache[1]
        if self.root() is None:
            result = {"ok": True, "solid_present": False, "workdir": None}
            self._result_cache = (key, result)
            return result
        source = self.program()
        workdir = cache_dir(self.TOOL, program_digest(source))
        result_path = os.path.join(workdir, "result.json")
        if not os.path.isfile(result_path):
            self._run(source, workdir, os.path.join(workdir, "model.stl"))
        with open(result_path, encoding="utf-8") as fh:
            result = json.load(fh)
        result["workdir"] = workdir
        self._result_cache = (key, result)
        return result

    # -- queries (answered from EXACT B-rep, not from a mesh) --------------
    def query(self, q: str) -> dict:
        if q == "document":
            return self._document()
        if q == "catalog":
            return self._catalog_view()
        if q in ("validity", "measure", "metrics"):
            result = self._artifacts()
            if not result.get("solid_present"):
                if q == "validity":
                    return {"manifold": False, "watertight": False,
                            "is_valid": False, "solid_present": False}
                if q == "measure":
                    return {"volume": 0.0, "bbox": [0.0, 0.0, 0.0]}
                return {}
            if q == "validity":
                valid = bool(result.get("is_valid"))
                closed = bool(result.get("is_closed"))
                return {
                    "manifold": valid,      # a valid OCCT solid is 2-manifold
                    "watertight": valid and closed,
                    "is_valid": valid,
                    "solid_present": True,
                    "faces": result.get("faces"),
                    "edges": result.get("edges"),
                    "solids": result.get("solids"),
                }
            if q == "measure":
                return {"volume": float(result["volume"]),
                        "bbox": [float(v) for v in result["bbox"]]}
            return self._metrics()
        return super().query(q)

    def _metrics(self, density: float = 1.0) -> dict:
        result = self._artifacts()
        if not result.get("solid_present"):
            return {}
        volume = float(result["volume"])
        return {
            "volume": volume,
            "mass": volume * density,
            "surface_area": float(result["surface_area"]),
            "bbox": [float(v) for v in result["bbox"]],
            "center_of_mass": [float(v) for v in result["center_of_mass"]],
            "faces": int(result["faces"]),
            "edges": int(result["edges"]),
            "vertices": int(result["vertices"]),
            "solids": int(result["solids"]),
        }

    def _validity(self) -> dict:
        return self.query("validity")

    def _document(self) -> dict:
        """The real FreeCAD feature tree, through the freecad_document wire codec.

        Decoding the driver's payload into :class:`DocumentContext` and re-encoding
        it is not ceremony: it VALIDATES the shape of what FreeCAD returned (a bad
        TypeId or a malformed Placement raises here), and it is the same wire
        format the rest of the harness already speaks.
        """
        result = self._artifacts()
        raw = result.get("document")
        if not raw:
            ctx = fcdoc.DocumentContext(document=None, objects=[], view=None)
            return fcdoc.encode_document_context(ctx)
        ctx = fcdoc.decode_document_context(raw)
        payload = fcdoc.encode_document_context(ctx)
        payload["issues"] = fcdoc.validate_context(ctx)
        # The harness's own feature stream, tagged with the FreeCAD TypeId that
        # realises each op -- the parametric tree the op stream *is*.
        payload["features"] = [
            {"id": f["id"], "type": f["type"],
             "type_id": TYPE_IDS.get(f["type"], "Part::Feature"),
             "freecad_op": OP_TO_FREECAD.get(f["type"], "")}
            for f in self.features
        ]
        return payload

    def _catalog_view(self) -> dict:
        """How the CISP op set lands on FreeCAD's 53-operation tool catalogue."""
        supported = {}
        for tag, name in sorted(OP_TO_FREECAD.items()):
            if tag in self.UNSUPPORTED or not name:
                continue
            supported[tag] = name
        return {
            "operations": len(self.catalog),
            "workbenches": self.catalog.workbench_histogram(),
            "op_to_freecad": supported,
            "unsupported": dict(self.UNSUPPORTED),
        }

    # -- export ------------------------------------------------------------
    def export(self, fmt: str):
        """STEP / BREP / IGES come straight from the kernel; STL via the base."""
        f = str(fmt).lower()
        if f not in self.FORMATS:
            raise ValueError("the freecad backend cannot export '%s' (supported: %s)"
                             % (fmt, ", ".join(self.FORMATS)))
        if f not in ("step", "brep", "iges"):
            return super().export(f)
        result = self._artifacts()
        if not result.get("solid_present"):
            raise ValueError("nothing to export: no solid present")
        path = os.path.join(result["workdir"], "model." + f)
        if not os.path.isfile(path):
            raise RuntimeError("FreeCAD did not write the %s export" % f)
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def regenerate(self) -> List[Diagnostic]:
        """Rebuild through FreeCAD and report a B-rep the kernel judged invalid."""
        if self.root() is None:
            return []
        try:
            result = self._artifacts()
        except BackendUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - the tool failed on this model
            return [Diagnostic(Severity.ERROR, "kernel-error",
                               "freecad failed to build the model: %s" % exc)]
        if not result.get("solid_present"):
            return [Diagnostic(Severity.ERROR, "empty-solid",
                               "freecad produced no geometry (empty solid)")]
        if not result.get("is_valid"):
            return [Diagnostic(Severity.ERROR, "invalid-brep",
                               "freecad produced an invalid B-rep solid")]
        return []
