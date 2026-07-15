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

from harnesscad.core.cisp.ops import Chamfer, Constrain, Fillet, Op, SetParam, Shell
from harnesscad.domain.geometry.topology import selector_dsl, topological_naming
from harnesscad.domain.programs.expressions import freecad_expressions
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.adapters.freecad_catalog import default_catalog
from harnesscad.io.backends import frep as frep_mod
from harnesscad.io.backends.base import ApplyResult, BackendUnavailable
from harnesscad.io.backends.external import (
    DEFAULT_TIMEOUT, ExternalToolBackend, cache_dir, find_executable,
    program_digest,
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


#: Linear tessellation deflection for the STL export, in mm.
#:
#: ``Shape.exportStl(path)`` hard-codes ``BRepMesh_IncrementalMesh(shape, 0.01)``
#: (src/Mod/Part/App/TopoShape.cpp) and exposes no way to change it, so a backend
#: that used it would ship a mesh whose quality it never declared. We tessellate
#: through ``MeshPart.meshFromShape`` instead, which takes the deflection
#: explicitly. The value matches FreeCAD's own default, so the STL is unchanged
#: in quality -- it is now *declared* rather than inherited, and tunable.
STL_LINEAR_DEFLECTION = 0.01

#: Angular deflection, by FreeCAD's own ``defaultAngularDeflection`` law
#: (TopoShape.cpp): ``min(0.1, linear * 5 + 0.005)``.
STL_ANGULAR_DEFLECTION = 0.055

#: The STEP schema headless FreeCAD writes, and the only one it CAN write.
#:
#: ``Shape.exportStep`` issues no ``Interface_Static::SetCVal("write.step.schema")``
#: call (TopoShape.cpp); the only caller of ``Part::Interface::writeStepScheme`` in
#: the whole tree is ``src/Mod/Import/Gui/AppImportGuiPy.cpp``, and ``ImportGui``
#: raises ``Cannot load Gui module in console application`` under ``freecadcmd``.
#: ``Part.Interface`` is not exposed to Python at all (verified on 1.1.1). So the
#: schema is OCCT's compiled-in default: AP214 (AUTOMOTIVE_DESIGN, ISO 10303-214).
#: We do not assume it -- the driver reads FILE_SCHEMA back out of the file it
#: wrote and reports it, and :meth:`export_info` surfaces it.
STEP_SCHEMA = "AP214"

#: FreeCAD's internal length unit is the millimetre; the STEP writer emits
#: ``SI_UNIT(.MILLI.,.METRE.)``. Verified by reading the written file.
STEP_UNIT = "MM"

#: A ``Shell`` with an EMPTY ``faces`` list is a CLOSED hollow -- a sealed
#: internal void, NOT "remove some default face". The driver builds it with
#: ``makeOffsetShape(-t)`` + ``cut`` rather than ``makeThickness`` (which is
#: defined in terms of the faces it removes and so cannot express a sealed void).
CLOSED_SHELL_IS_SEALED = True


def _err(code: str, msg: str, where: Optional[str] = None) -> ApplyResult:
    return ApplyResult(False, [], [Diagnostic(Severity.ERROR, code, msg, where)])


def _selector_source() -> str:
    """The selector DSL module's source, to prepend to the driver.

    The driver runs under FreeCAD's own interpreter and may not import
    ``harnesscad``, but edge/face selection MUST use the same grammar and
    semantics as the CadQuery backend or the differential oracle is comparing two
    different parts. Inlining the one module both backends already share is what
    guarantees that. It is stdlib-only, so it runs unchanged over there.
    """
    with open(selector_dsl.__file__, encoding="utf-8") as fh:
        return fh.read()


def shell_face_selector(name: str) -> str:
    """A ``frep.SHELL_FACES`` name as a CadQuery direction-extremum selector.

    ``frep.SHELL_FACES`` maps each name to ``(axis, sign)`` -- axis 0/1/2 = X/Y/Z,
    sign +1 = the max face, -1 = the min face. That is precisely what ``>X`` /
    ``<X`` mean in the selector DSL, so ``top`` -> ``>Z`` and ``bottom`` -> ``<Z``.
    A string that is already a selector (``<Z``) passes through untouched.
    """
    key = str(name).strip().lower()
    pair = frep_mod.SHELL_FACES.get(key)
    if pair is None:
        return str(name).strip()
    axis, sign = pair
    return "%s%s" % (">" if sign > 0 else "<", "XYZ"[axis])


def _profile_spec(prof) -> dict:
    """An ``frep._Profile`` in the same JSON shape ``Node.spec()`` gives it."""
    return {"rects": [list(r) for r in prof.rects],
            "circles": [list(c) for c in prof.circles],
            "polys": [[list(p) for p in q] for q in prof.polys]}


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
        "hull": "no convex-hull feature is wired (the F-rep op model builds no "
                "hull node and FreeCAD has no single hull operation), so a hull is "
                "refused rather than faked",
        "minkowski": "no Minkowski / offset-solid feature is wired here; a ball "
                     "dilation is built in the frep SDF kernel and in OpenSCAD's "
                     "minkowski(), and refused here",
    }

    #: OCCT offsets with either join, exposed as the ``join`` argument of
    #: ``makeThickness`` / ``makeOffsetShape``. The driver now passes it.
    SHELL_JOINS: Tuple[str, ...] = ("arc", "intersection")

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
        bad = self._selector_check(op)
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

    def _selector_check(self, op: Op) -> Optional[ApplyResult]:
        """Refuse a malformed edge/face selector BEFORE it reaches the kernel.

        ``Fillet.edges`` / ``Chamfer.edges`` / ``Shell.faces`` are CadQuery
        selector strings; they are parsed with the very grammar the driver will
        use, so a typo is a typed ``bad-value`` here rather than an opaque
        FreeCAD traceback later. An empty tuple is legal (it means "all").
        """
        if isinstance(op, (Fillet, Chamfer)):
            sels, what = op.edges, "edge"
        elif isinstance(op, Shell):
            # Shell.faces is the harness's NAMED vocabulary (frep.SHELL_FACES);
            # frep validates it and turns a bad name away first. Each name maps
            # onto a selector, so only a non-name is parsed as one.
            sels, what = op.faces, "face"
        else:
            return None
        for sel in sels or ():
            text = str(sel).strip()
            if not text or text.lower() in frep_mod.SHELL_FACES:
                continue
            try:
                selector_dsl.parse(text)
            except selector_dsl.SelectorError as exc:
                return _err("bad-value",
                            "%s %s selector %r is malformed: %s"
                            % (getattr(type(op), "OP", "op"), what, text, exc))
        return None

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

    # -- lowering the op stream ---------------------------------------------
    def _blends(self) -> List[dict]:
        """The fillet/chamfer ops, IN ORDER, each with its own edge selector.

        The F-rep model does not keep these as tree nodes: ``frep.blend_tree``
        rewrites the whole tree, stamping one radius onto every leaf, because an
        SDF has no edges to select. A B-rep kernel DOES.
        ``Shape.makeFillet(radius, edges)`` (https://wiki.freecad.org/Part_scripting)
        takes an explicit edge list, so this backend must honour ``Fillet.edges``
        rather than blend everything -- filleting the wrong edges is a silent
        correctness bug (a 10mm box filleted r=2 on its four VERTICAL edges has
        volume 965.6637061435918, exactly the analytic value; the same box
        filleted on all twelve has 907.7049926967563 -- a different part).

        Derived from the op log, not accumulated, so ``SetParam``'s replay is
        automatically correct.
        """
        blends: List[dict] = []
        for op in self._frep._oplog:
            if isinstance(op, Fillet):
                blends.append({"kind": "fillet", "value": float(op.radius),
                               "selectors": [str(s) for s in op.edges]})
            elif isinstance(op, Chamfer):
                blends.append({"kind": "chamfer", "value": float(op.distance),
                               "value2": (None if op.distance2 is None
                                          else float(op.distance2)),
                               "selectors": [str(s) for s in op.edges]})
        return blends

    def _root_spec(self) -> Optional[dict]:
        """``root.spec()`` with each shell node's face names turned into selectors.

        The F-rep shell node names the faces it opens in the harness's own
        vocabulary (``frep.SHELL_FACES``: ``top`` / ``bottom`` / ``+z`` / ``zmin``
        ...); FreeCAD has no such names, only real B-rep faces. Each name is an
        (axis, sign) pair, which is exactly a ``>``/``<`` direction-extremum
        selector -- so ``bottom`` becomes ``<Z`` and the driver picks the real
        face by geometry.
        """
        root = self.root()
        if root is None:
            return None
        spec = root.spec()

        def walk(node: dict) -> None:
            for key in ("a", "b", "child"):
                child = node.get(key)
                if isinstance(child, dict):
                    walk(child)
            if node.get("t") == "shell":
                node["faces"] = [shell_face_selector(f)
                                 for f in (node.get("faces") or ())]

        walk(spec)
        return spec

    def _sketch_specs(self) -> List[dict]:
        """Every CISP sketch, with its constraints, for FreeCAD's REAL solver.

        FreeCAD has a genuine geometric constraint solver (planegcs) behind
        ``Sketcher::SketchObject`` (https://wiki.freecad.org/Sketcher_scripting):
        ``addGeometry`` + ``addConstraint`` + ``solve()``, with ``.DoF`` and
        ``.FullyConstrained`` reporting the verdict. Every other backend counts
        DOF from a table (``ops.CONSTRAINT_DOF``); this one SOLVES. So a
        ``constrain`` op here actually moves the geometry, and the solid is built
        from the solved profile.

        ``original_profile`` is how the driver knows which tree node the solved
        profile replaces: the F-rep node carries the profile, not the sketch id.
        A profile shared by two constrained sketches would be ambiguous, so such
        a sketch is reported but not substituted.
        """
        by_sketch: Dict[str, List[dict]] = {}
        for op in self._frep._oplog:
            if not isinstance(op, Constrain):
                continue
            ent = self.entities.get(op.a)
            if ent is None:
                continue
            by_sketch.setdefault(ent["sketch"], []).append(
                {"kind": op.kind, "a": op.a, "b": op.b,
                 "value": None if op.value is None else float(op.value)})

        originals: Dict[str, str] = {}
        for sid, sk in self.sketches.items():
            if by_sketch.get(sid):
                prof = frep_mod._profile_of(sk, self.entities)
                originals[sid] = json.dumps(_profile_spec(prof), sort_keys=True)
        shared = {k for k in originals
                  if list(originals.values()).count(originals[k]) > 1}

        specs: List[dict] = []
        for sid in sorted(self.sketches, key=lambda s: (len(s), s)):
            sk = self.sketches[sid]
            entry = {
                "id": sid,
                "plane": sk["plane"],
                "entities": [{"id": eid, "type": self.entities[eid]["type"],
                              "params": self.entities[eid]["params"]}
                             for eid in sk["entities"]],
                "constraints": by_sketch.get(sid, []),
            }
            if sid in originals and sid not in shared:
                entry["original_profile"] = _profile_spec(
                    frep_mod._profile_of(sk, self.entities))
            elif sid in shared:
                entry["constraints"] = []  # reported, but may not move the model
                entry["ambiguous_profile"] = True
            specs.append(entry)
        return specs

    # -- the tool ----------------------------------------------------------
    def spec(self) -> dict:
        """The JSON the driver consumes: the CSG tree, the ordered edge blends
        with their selectors, the sketches with their constraints, and the
        declared export settings."""
        return {
            "root": self._root_spec(),
            # NO op-log blends. Each Fillet/Chamfer is a "blend" NODE in the tree,
            # applied at its own position in the feature history (so a pattern can
            # replicate the pad WITHOUT its fillet). Sending them here as well
            # applied every blend a SECOND time, at the root -- which re-selected
            # "|Z" on a shape whose vertical edges were already arcs and rounded the
            # rest, turning a 995.6 part into the 971.3 all-twelve-edges part.
            "sketches": self._sketch_specs(),
            "exports": list(self.DRIVER_EXPORTS),
            "stl_linear_deflection": STL_LINEAR_DEFLECTION,
            "stl_angular_deflection": STL_ANGULAR_DEFLECTION,
        }

    def driver_source(self) -> str:
        """The full driver text: the shared selector DSL, then the driver."""
        return _selector_source() + "\n\n" + DRIVER_SOURCE

    def program(self) -> str:
        """The driver text + its spec, as ONE string, so the content hash covers both."""
        payload = json.dumps(self.spec(), sort_keys=True, separators=(",", ":"))
        return "# spec: %s\n%s" % (payload, self.driver_source())

    def _run(self, source: str, workdir: str, out_path: str) -> None:
        """Write driver + spec into the content-addressed workdir and run FreeCAD."""
        driver = os.path.join(workdir, "driver.py")
        with open(driver, "w", encoding="utf-8") as fh:
            fh.write(self.driver_source())
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
        if q == "constraints":
            return self._constraints()
        if q == "sketch_dof":
            return self._sketch_dof()
        if q == "topology":
            return self._topology()
        if q == "export":
            return self.export_info()
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

    # -- constraints, answered by FreeCAD's REAL solver ---------------------
    def _sketch_reports(self) -> Dict[str, dict]:
        """FreeCAD's per-sketch solver verdict, keyed by CISP sketch id."""
        try:
            result = self._artifacts()
        except Exception:  # noqa: BLE001 - a broken model still has sketches
            return {}
        return {r["id"]: r for r in (result.get("sketches") or [])}

    def _sketch_dof(self) -> dict:
        """Remaining DOF per sketch -- from planegcs, not from a lookup table.

        Every other backend answers this from ``ops.CONSTRAINT_DOF``, a fixed
        "this kind removes N DOF" table that cannot see redundancy, conflict or
        an under-determined system. FreeCAD's ``SketchObject.DoF`` is what its
        solver actually has left (https://wiki.freecad.org/Sketcher_scripting), so
        this backend is the only one that can honour a ``constrain`` op for real.
        The shape of the answer is the F-rep model's -- ``{sketch_id: dof}`` -- so
        the harness drives every backend through one surface; only the NUMBER is
        better. ``query('constraints')`` carries the solver's full verdict
        (conflicts, redundancy, fully-constrained).
        """
        table = self._frep.query("sketch_dof")
        reports = self._sketch_reports()
        if not reports:
            return table
        out: Dict[str, Any] = {}
        for sid, dof in table.items():
            rep = reports.get(sid)
            if rep is None or rep.get("dof") is None:
                out[sid] = dof
            else:
                out[sid] = int(rep["dof"])
        return out

    def _constraints(self) -> dict:
        """The solver's full verdict: solved, DOF, conflicts, redundancy."""
        reports = self._sketch_reports()
        sketches = []
        for sid in sorted(reports, key=lambda s: (len(s), s)):
            r = reports[sid]
            sketches.append({
                "id": sid,
                "plane": r.get("plane"),
                "constraints": r.get("constraints", 0),
                "solved": bool(r.get("solved")),
                "status": r.get("status"),
                "dof": r.get("dof"),
                "fully_constrained": bool(r.get("fully_constrained")),
                "conflicting": r.get("conflicting") or [],
                "redundant": r.get("redundant") or [],
                "malformed": r.get("malformed") or [],
                "errors": r.get("errors") or [],
            })
        return {"solver": "freecad-planegcs", "sketches": sketches}

    # -- topological naming -------------------------------------------------
    def face_records(self) -> List[topological_naming.FaceRecord]:
        """The current solid's faces as :class:`FaceRecord`s.

        The classic topological-naming problem: FreeCAD names sub-shapes by INDEX
        (``Edge7``, ``Face3``), and an index is not an identity -- an upstream
        edit reorders, splits or merges faces, so a stored reference silently
        moves. The repo's :mod:`~harnesscad.domain.geometry.topology.topological_naming`
        answers it by geometry: quantised (surface kind, normal, centroid, area)
        hashed into a stable fingerprint that survives a rebuild.

        Wired here so a face reference taken before a ``SetParam`` can be migrated
        onto the rebuilt solid with :func:`topological_naming.resolve_reference`.
        """
        result = self._artifacts()
        out = []
        for rec in result.get("face_records") or []:
            normal = rec.get("normal")
            out.append(topological_naming.FaceRecord(
                id=rec["id"],
                surface=rec.get("surface", "planar"),
                normal=None if normal is None else tuple(float(c) for c in normal),
                centroid=tuple(float(c) for c in rec["centroid"]),
                area=float(rec["area"]),
            ))
        return out

    def _topology(self) -> dict:
        """Faces + edges of the current solid, each with a rebuild-stable id.

        ``fingerprint`` is what a reference should be STORED as; ``id`` is the
        FreeCAD index name, which is what it must never be stored as.
        """
        result = self._artifacts()
        if not result.get("solid_present"):
            return {"faces": [], "edges": [], "solid_present": False}
        faces = [
            {"id": rec.id, "surface": rec.surface, "normal": rec.normal,
             "centroid": rec.centroid, "area": rec.area,
             "fingerprint": topological_naming.fingerprint(rec)}
            for rec in self.face_records()
        ]
        return {
            "solid_present": True,
            "faces": faces,
            "edges": list(result.get("edge_records") or []),
        }

    def match_faces(self, old_faces) -> topological_naming.MatchReport:
        """Match a face set captured BEFORE a rebuild onto the solid as it is now.

        Classifies every face as matched / split / merged / deleted / created --
        the failure modes an index-based name cannot even represent.
        """
        return topological_naming.match_topology(list(old_faces),
                                                 self.face_records())

    def resolve_face(self, old_faces, old_id: str):
        """Migrate ONE stored face reference across a rebuild.

        ``old_faces`` is the face set as it was when the reference was taken;
        ``old_id`` is the face referenced then. Returns a
        :class:`topological_naming.ReferenceResolution` -- the surviving face id,
        the fragments if the face was split, or an explicit reason if the
        reference went stale. This is what a fillet/shell reference SHOULD be
        stored as; a bare ``Face3`` cannot survive an upstream edit.
        """
        new_faces = self.face_records()
        report = topological_naming.match_topology(list(old_faces), new_faces)
        return topological_naming.resolve_reference(old_id, report,
                                                    new_faces=new_faces)

    # -- export metadata ----------------------------------------------------
    def export_info(self) -> dict:
        """What the exports ACTUALLY declare -- read back out of the files.

        A wrong STEP schema or an unset STL deflection is a silent quality bug, so
        neither is assumed: the driver reads FILE_SCHEMA and the unit clause out
        of the STEP it just wrote, and reports the deflection it tessellated with.
        """
        result = self._artifacts()
        info = dict(result.get("export") or {})
        info["formats"] = list(self.FORMATS)
        info["step_schema_declared"] = STEP_SCHEMA
        info["step_unit_declared"] = STEP_UNIT
        info["stl_linear_deflection_declared"] = STL_LINEAR_DEFLECTION
        return info

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
