"""environment_freecad — a live FreeCAD GUI as a first-class Environment.

This is the thing the whole package exists for, and it is honest about what it
is. It implements :class:`harnesscad.core.environment.Environment`, and it
declares — in machine-readable form, not in a docstring —

    content_digest      = False   (a running Qt app has no content hash)
    nonmutating_reject  = False   (a refused dialog has already opened panels)
    synchronous_read    = False   (reads are out-of-band and may lag a recompute)
    deterministic_replay= False   (wall-clock settling, focus, window state)
    resolve_before_act  = True    (and THAT is what a kernel backend cannot do)

Forcing this behind :class:`GeometryBackend` would have made it fabricate a
digest. It does not have one, so it does not have one.

How the geometry gets out
-------------------------
**The agent never saves.** There is no path in this package that invokes File>Save.
The harness owns all file I/O: FreeCAD is launched with a harness-owned macro
(:data:`MACRO_SOURCE`) which installs a ``QTimer`` polling a control directory the
harness chose. To export, the harness drops a command file; the macro recomputes,
measures the body's exact B-rep, writes the STEP/STL/BREP to the scratch path, and
writes back a result file. The agent has no filename, no Save button (it is on the
deny list), and no way to touch a user file.

The differential oracle
-----------------------
The same op stream is driven through this GUI and through the *scripted*
``FreeCADBackend``. Both must produce the same part. The scripted backend already
matches ANALYTIC to 4.5e-16 on all 20 CISP ops, so it is ground truth, and the GUI
is measured against it across the GUI boundary. Nobody else in the CUA field can
run that test, because nobody else knows what is supposed to be on the screen.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Op
from harnesscad.core.environment import (
    CapabilityError, Capabilities, Observation, StepResult, coerce_ops,
)
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.cua import bindings_freecad as B
from harnesscad.io.cua import frames, guardrails, uia

#: Where the FreeCAD *GUI* binary lives (the scripted backend wants freecadcmd;
#: this one wants the windowed executable).
EXECUTABLE_NAMES = ("freecad.exe", "FreeCAD.exe", "freecad", "FreeCAD")
EXECUTABLE_PATTERNS = (
    os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs",
                 "FreeCAD*", "bin", "freecad.exe"),
    r"C:\Program Files\FreeCAD*\bin\freecad.exe",
    "/usr/bin/freecad",
    "/Applications/FreeCAD.app/Contents/MacOS/FreeCAD",
)


def find_gui_executable() -> Optional[str]:
    import glob

    override = os.environ.get("HARNESSCAD_FREECAD_GUI")
    if override and os.path.isfile(override):
        return override
    for pattern in EXECUTABLE_PATTERNS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    for name in EXECUTABLE_NAMES:
        hit = shutil.which(name)
        if hit:
            return hit
    return None


def available() -> Tuple[bool, str]:
    """(can this environment run here, why not). Never raises, never hangs."""
    if not uia.available():
        return False, "uiautomation/Windows not available"
    if find_gui_executable() is None:
        return False, "the FreeCAD GUI executable was not found"
    return True, ""


#: The harness's export channel. Runs INSIDE FreeCAD, owned by the harness, and
#: reachable only by the harness (the agent never learns the control directory).
#: It never calls Save: it measures the B-rep and writes exports to a scratch path.
MACRO_SOURCE = r'''
import os, json, traceback
import FreeCAD

try:
    from PySide6 import QtCore
except Exception:
    from PySide import QtCore

CTRL = os.environ.get("HARNESSCAD_CUA_CTRL", "")


def _shape():
    doc = FreeCAD.ActiveDocument
    if doc is None:
        return None
    doc.recompute()
    objs = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"]
    if not objs:
        objs = [o for o in doc.Objects
                if getattr(o, "Shape", None) is not None
                and not o.Shape.isNull()]
    shape = None
    for o in objs:
        s = getattr(o, "Shape", None)
        if s is None or s.isNull():
            continue
        shape = s if shape is None else shape.fuse(s)
    if shape is None:
        return None
    # A Body's Shape is a COMPOUND, and a compound has no CenterOfMass and no
    # meaningful volume. Fuse its solids into one and remove the seam faces the
    # fuse leaves behind, so face/edge counts match the scripted backend's solid
    # rather than counting the internal boundary twice.
    if shape.ShapeType == "Compound":
        solids = shape.Solids
        if not solids:
            return None
        merged = solids[0]
        for s in solids[1:]:
            merged = merged.fuse(s)
        shape = merged
    if len(shape.Solids) > 1:
        merged = shape.Solids[0]
        for s in shape.Solids[1:]:
            merged = merged.fuse(s)
        shape = merged
    try:
        shape = shape.removeSplitter()
    except Exception:
        pass
    return shape


def _measure(shape):
    bb = shape.BoundBox
    com = shape.CenterOfMass
    return {
        "volume": float(shape.Volume),
        "surface_area": float(shape.Area),
        "bbox": [float(bb.XLength), float(bb.YLength), float(bb.ZLength)],
        "bbox_min": [float(bb.XMin), float(bb.YMin), float(bb.ZMin)],
        "center_of_mass": [float(com.x), float(com.y), float(com.z)],
        "faces": len(shape.Faces), "edges": len(shape.Edges),
        "vertices": len(shape.Vertexes), "solids": len(shape.Solids),
        "is_valid": bool(shape.isValid()),
        "is_closed": bool(shape.isClosed()),
    }


def _write(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)


def _tick():
    if not CTRL:
        return
    cmd_path = os.path.join(CTRL, "cmd.json")
    if not os.path.isfile(cmd_path):
        return
    try:
        with open(cmd_path) as fh:
            cmd = json.load(fh)
    except Exception:
        return
    try:
        os.remove(cmd_path)
    except Exception:
        pass
    out = {"id": cmd.get("id"), "ok": False}
    try:
        doc = FreeCAD.ActiveDocument
        out["document"] = None if doc is None else doc.Name
        out["objects"] = [] if doc is None else [
            {"name": o.Name, "type_id": o.TypeId} for o in doc.Objects]
        shape = _shape()
        if shape is None:
            out["error"] = "no solid in the document"
            out["solid_present"] = False
        else:
            out.update(_measure(shape))
            out["solid_present"] = True
            out["ok"] = True
            exports = cmd.get("exports") or {}
            written = {}
            for fmt, path in exports.items():
                f = str(fmt).lower()
                if f == "step":
                    shape.exportStep(path)
                elif f == "brep":
                    shape.exportBrep(path)
                elif f == "stl":
                    shape.exportStl(path)
                else:
                    continue
                written[f] = path
            out["exports"] = written
    except Exception as exc:
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        out["trace"] = traceback.format_exc()
    _write(os.path.join(CTRL, "result_%s.json" % out.get("id")), out)


_TIMER = QtCore.QTimer()
_TIMER.timeout.connect(_tick)
_TIMER.start(200)
FreeCAD.__harnesscad_timer = _TIMER
if CTRL:
    _write(os.path.join(CTRL, "ready.json"), {"ready": True, "pid": os.getpid()})
'''


class FreeCADGuiEnvironment:
    """A live FreeCAD GUI, driven by its accessibility tree."""

    #: What it can honestly do. Read this before relying on anything.
    CAPABILITIES = Capabilities(
        name="freecad-gui",
        content_digest=False,
        nonmutating_reject=False,
        synchronous_read=False,
        deterministic_replay=False,
        export=True,
        export_formats=("step", "brep", "stl"),
        supported_ops=("new_sketch", "add_rectangle", "extrude"),
        unsupported_ops={k: v for k, v in B.REQUIRES_VIEWPORT.items()
                         if k not in ("new_sketch", "add_rectangle", "extrude")},
        resolve_before_act=True,
        notes=(
            "no content digest: a running Qt application has no content hash of "
            "its document, and this environment will not fabricate one",
            "reject is MUTATING: by the time a dialog refuses a value it has "
            "already opened a task panel and moved focus",
            "reads are asynchronous: the a11y tree and the export channel may lag "
            "the application's own recompute",
            "the agent NEVER saves; the harness owns all file I/O and exports to a "
            "scratch path through its own macro channel",
            "only the parameter-dialog subset is drivable coordinate-free; every "
            "op needing an edge/face/tree PICK is refused, not faked",
        ),
    )

    def __init__(self, executable: Optional[str] = None,
                 scratch: Optional[guardrails.Scratch] = None,
                 launch_timeout: float = 90.0,
                 settle: float = 2.0) -> None:
        ok, why = available()
        if not ok:
            raise uia.UiaUnavailable("the FreeCAD GUI environment cannot run: " + why)
        self.executable = executable or find_gui_executable()
        self.scratch = scratch or guardrails.Scratch()
        self.launch_timeout = float(launch_timeout)
        self.settle = float(settle)
        self.proc: Optional[subprocess.Popen] = None
        self.driver: Optional[uia.UiaDriver] = None
        self.guards = guardrails.Guardrails(
            expected_window_classes=(B.WINDOW_CLASS,))
        self._cmd_id = 0
        self._steps = 0
        self._pending: List[Op] = []
        self._built: List[Op] = []
        self._outcomes: List[Dict[str, Any]] = []
        self._launched = False

    # -- Environment -------------------------------------------------------
    def capabilities(self) -> Capabilities:
        return self.CAPABILITIES

    def reset(self) -> Observation:
        """Fresh app, fresh document, fresh body. Nothing is reused across resets."""
        self.close()
        self._launch()
        self._steps = 0
        self._pending = []
        self._built = []
        self._outcomes = []
        self._invoke(B.NEW_DOCUMENT)
        self._invoke(B.NEW_BODY)
        return self.observe()

    def step(self, action) -> StepResult:
        """Buffer ops until a RECIPE matches, then drive the GUI for it.

        An op the GUI cannot do coordinate-free is REFUSED with the reason (it
        needs a viewport pick), not approximated. Half a part is worse than no
        part, because it looks like a part.
        """
        ops = coerce_ops(action)
        self._steps += 1
        caps = self.capabilities()
        diags: List[Diagnostic] = []
        for op in ops:
            tag = getattr(type(op), "OP", "")
            if not caps.supports(tag):
                diags.append(Diagnostic(Severity.ERROR, "unsupported-op",
                                        "freecad-gui cannot drive '%s': %s"
                                        % (tag, caps.why_not(tag))))
        if diags:
            return StepResult(ok=False, verified=False, observation=self.observe(),
                              reward=-1.0, diagnostics=diags,
                              info={"step": self._steps})

        self._pending.extend(ops)
        matches, _ = B.match_recipes(self._pending)
        executed = 0
        verified = True
        for recipe, matched in matches:
            bad = B.check_guards(recipe, matched)
            if bad:
                diags.append(Diagnostic(
                    Severity.ERROR, "unsupported-op",
                    "freecad-gui recipe '%s' does not apply: %s"
                    % (recipe.id, "; ".join(bad))))
                verified = False
                break
            try:
                outcome = self._run_recipe(recipe, matched)
            except (guardrails.GuardrailViolation, uia.UiaError) as exc:
                diags.append(Diagnostic(Severity.ERROR, "gui-error", str(exc)))
                verified = False
                break
            self._outcomes.append(outcome)
            if not outcome["ok"]:
                diags.append(Diagnostic(Severity.ERROR, "gui-error",
                                        outcome.get("error", "unverified action")))
                verified = False
                break
            self._built.extend(matched)
            executed += len(matched)
        # Ops consumed by an executed recipe leave the buffer; the rest wait for
        # the ops that complete their pattern.
        self._pending = self._pending[executed:]
        ok = not diags
        return StepResult(
            ok=ok, verified=ok and verified,
            observation=self.observe(),
            reward=1.0 if ok else -1.0,
            diagnostics=diags,
            info={"step": self._steps, "executed_ops": executed,
                  "pending_ops": len(self._pending),
                  "outcomes": self._outcomes[-3:]},
        )

    def observe(self) -> Observation:
        """Hybrid: the a11y tree's structured facts + the ONE rect that needs vision.

        ``digest`` is None. It is always None. That is the honest answer.
        """
        state: Dict[str, Any] = {"ops_built": [op.to_dict() for op in self._built],
                                 "ops_pending": len(self._pending)}
        images: Dict[str, Any] = {}
        if self.driver is not None:
            title = self.driver.title()
            state.update(guardrails.dirty_tripwire(title))
            try:
                elems = self.driver.tree()
                state["elements"] = len(elems)
                state["top_windows"] = self.driver.top_windows()
            except uia.UiaError:
                state["elements"] = 0
            try:
                vp = self.driver.viewport_frame()
                images["viewport"] = vp.to_dict()
            except uia.UiaError:
                images["viewport"] = None
        return Observation(kind="hybrid", state=state, digest=None, images=images,
                           step=self._steps,
                           notes=("no content digest: this environment does not "
                                  "have one and will not invent one",))

    def export(self, fmt: str):
        """Export through the HARNESS's macro channel — never the app's Save."""
        f = str(fmt).lower()
        if f not in self.CAPABILITIES.export_formats:
            raise ValueError("freecad-gui cannot export %r (supported: %s)"
                             % (fmt, ", ".join(self.CAPABILITIES.export_formats)))
        path = self.scratch.export_path("model." + f)
        result = self._command({f: path})
        if not result.get("ok"):
            raise uia.UiaError("the GUI export failed: %s"
                               % result.get("error", "unknown"))
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=15)
            except Exception:  # noqa: BLE001
                try:
                    self.proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        self.proc = None
        self.driver = None
        self._launched = False

    # -- capability-gated --------------------------------------------------
    def state_digest(self) -> str:
        raise CapabilityError(
            self.CAPABILITIES.name, "content_digest",
            "a running CAD application has no content hash of its document. "
            "Hashing a screenshot, or the op stream we THINK we applied, would be "
            "exactly the silent lie this harness exists to eradicate.")

    def query(self, q: str) -> dict:
        if q in ("measure", "metrics", "validity", "document"):
            return self.measure(q)
        raise CapabilityError(self.CAPABILITIES.name, "synchronous_read",
                              "reads go through the asynchronous macro channel; "
                              "use measure()")

    def measure(self, q: str = "measure") -> dict:
        """The exact B-rep measurement, read out of the running application.

        This is out-of-band and asynchronous — hence ``synchronous_read=False`` —
        but it is the REAL kernel's answer, not a guess from pixels, so the
        verifier fleet can be run against it unchanged.
        """
        result = self._command({})
        if not result.get("ok"):
            error = result.get("error", "")
            if "no solid" in error:
                # An empty document HAS no measurement. That is a fact, not a
                # failure, and it is reported as one.
                return {"solid_present": False, "error": error}
            # Anything else is a failed read, and a failed read must never come
            # back as a dict that merely happens to be missing 'volume' -- a caller
            # would compare a KeyError-shaped hole against a reference and never
            # know the measurement did not happen.
            raise uia.UiaError("the GUI measurement failed: %s" % error)
        if q == "validity":
            return {"manifold": result["is_valid"],
                    "watertight": result["is_valid"] and result["is_closed"],
                    "is_valid": result["is_valid"], "solid_present": True,
                    "faces": result["faces"], "edges": result["edges"],
                    "solids": result["solids"]}
        if q == "document":
            return {"document": result.get("document"),
                    "objects": result.get("objects", [])}
        if q == "measure":
            return {"volume": result["volume"], "bbox": result["bbox"]}
        return {k: result[k] for k in
                ("volume", "surface_area", "bbox", "center_of_mass", "faces",
                 "edges", "vertices", "solids") if k in result}

    # -- the GUI -----------------------------------------------------------
    def _launch(self) -> None:
        frames.assert_frames_agree()
        ctrl = self.scratch.path("ctrl")
        os.makedirs(ctrl, exist_ok=True)
        macro = self.scratch.path("harness_channel.FCMacro")
        with open(macro, "w", encoding="utf-8") as fh:
            fh.write(MACRO_SOURCE)
        env = dict(os.environ)
        env["HARNESSCAD_CUA_CTRL"] = ctrl
        self._ctrl = ctrl
        self.proc = subprocess.Popen([self.executable, macro], env=env)
        deadline = time.time() + self.launch_timeout
        ready = os.path.join(ctrl, "ready.json")
        while time.time() < deadline:
            if os.path.isfile(ready):
                break
            if self.proc.poll() is not None:
                raise uia.UiaError("FreeCAD exited during launch (code %s)"
                                   % self.proc.returncode)
            time.sleep(0.5)
        else:
            raise uia.UiaError("FreeCAD's harness channel never came up in %.0fs"
                               % self.launch_timeout)
        self.driver = uia.UiaDriver(pid=self.proc.pid, class_name=B.WINDOW_CLASS,
                                    timeout=self.launch_timeout)
        self.driver.window()
        self.guards.window_rect = self.driver.window_rect()
        self.driver.focus_window()
        self._launched = True

    def _invoke(self, control: B.Control, settle: Optional[float] = None) -> uia.Outcome:
        element = self.driver.wait_for(timeout=20.0, enabled_only=True,
                                       **control.query())
        if element is None:
            raise uia.UiaError("control %r never appeared (or never enabled)"
                               % (control,))
        self.guards.window_rect = self.driver.window_rect()
        outcome = guardrails.guarded_invoke(
            self.driver, element, self.guards,
            settle=self.settle if settle is None else settle)
        if not outcome.ok:
            raise uia.UiaError("invoking %r was not verified: %s"
                               % (control, outcome.error))
        return outcome

    def _run_recipe(self, recipe: B.Recipe, ops: Sequence[Op]) -> Dict[str, Any]:
        """Open the dialog, WRITE-AND-READ-BACK every field, confirm, verify."""
        t0 = time.time()
        before = self.driver.snapshot()
        for button in recipe.buttons:
            self._invoke(button)
        values = B.bind_values(recipe, ops)
        locale = self.driver.locale()
        writes: List[dict] = []
        for fb in recipe.fields:
            element = self.driver.wait_for(timeout=15.0, aid_suffix=fb.aid_suffix,
                                           control_type=fb.control_type)
            if element is None:
                raise uia.UiaError("dialog field %r never appeared" % fb.aid_suffix)
            self.guards.enforce(element, self.driver.top_windows())
            outcome, report = self.driver.set_quantity(element, values[fb.aid_suffix],
                                                       locale=locale)
            if not outcome.ok:
                # THE 375mm BUG DIES HERE. A field we cannot prove we set is a
                # field we did not set, and the op does not proceed.
                self._invoke(B.CANCEL_BUTTON, settle=1.0)
                return {"ok": False, "recipe": recipe.id,
                        "error": "field %s: %s" % (fb.aid_suffix, outcome.error),
                        "evidence": outcome.evidence,
                        "elapsed": time.time() - t0}
            writes.append(report.to_dict())
        # SECOND read-back, over every field, immediately before OK. The first one
        # proves the value landed; this one proves nothing later moved it.
        for fb in recipe.fields:
            element = self.driver.find(aid_suffix=fb.aid_suffix,
                                       control_type=fb.control_type)
            check = self.driver.verify_quantity(element, values[fb.aid_suffix],
                                                locale=locale)
            if not check.ok:
                self._invoke(B.CANCEL_BUTTON, settle=1.0)
                return {"ok": False, "recipe": recipe.id,
                        "error": "pre-confirm check failed on %s: %s"
                                 % (fb.aid_suffix, check.error),
                        "evidence": check.evidence, "elapsed": time.time() - t0}
        confirm = self._invoke(recipe.confirm)
        after = self.driver.snapshot()
        return {"ok": True, "recipe": recipe.id, "writes": writes,
                "confirm": confirm.to_dict(),
                "dirty": after.dirty,
                "diff": before.diff(after),
                "elapsed": time.time() - t0}

    # -- the harness's own channel (never the app's Save) -------------------
    def _command(self, exports: Dict[str, str], timeout: float = 60.0) -> dict:
        if not self._launched:
            raise uia.UiaError("the environment is not running")
        self._cmd_id += 1
        cid = self._cmd_id
        result_path = os.path.join(self._ctrl, "result_%d.json" % cid)
        tmp = os.path.join(self._ctrl, "cmd.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"id": cid, "exports": exports}, fh)
        os.replace(tmp, os.path.join(self._ctrl, "cmd.json"))
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.isfile(result_path):
                with open(result_path, encoding="utf-8") as fh:
                    return json.load(fh)
            time.sleep(0.2)
        raise uia.UiaError("the FreeCAD harness channel did not answer in %.0fs"
                           % timeout)

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "FreeCADGuiEnvironment":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
        self.scratch.cleanup()
