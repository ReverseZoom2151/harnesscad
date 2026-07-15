"""environment_fusion - a live Autodesk Fusion 360 GUI as an Environment, with
Fusion's Python API (``adsk.core`` / ``adsk.fusion``) as the ORACLE.

Fusion sits between FreeCAD and the COM tools. Like FreeCAD it has a first-class
Python API, but that API lives INSIDE Fusion (a script/add-in), so - exactly as
FreeCAD's macro channel does - the harness reaches it out-of-process through a
control channel the agent never touches. Like SolidWorks/Inventor it is a
commercial tool people are paid to use, and its ribbon is the actuator.

The a11y caveat, stated honestly
--------------------------------
Fusion draws much of its interface in a Qt/OpenGL surface, so its accessibility
tree is PARTIAL: the top ribbon and command dialogs expose UIAutomation nodes, but
the canvas and some Qt-drawn panels do not. This environment declares that in its
capability notes; where the ribbon is not reachable through the tree the honest
answer is a refusal, never a vision guess.

Units: Fusion's API is in CENTIMETRES (its internal unit), like Inventor and unlike
SolidWorks's SI. The oracle converts cm -> mm (x10 length, x1000 volume, x100
area) at its boundary.

Capabilities vs FreeCAD's GUI: ``synchronous_read = True`` - Fusion's
``PhysicalProperties`` is a synchronous structured read (the oracle), which
FreeCAD's out-of-band macro channel is not. Still ``content_digest = False`` (a
Fusion design has no content hash; its version id is a document handle) and
``deterministic_replay = False`` (cloud regen, timeline, window state).

Credentials and safety: Fusion is an Autodesk-account application, but this module
NEVER handles the account, its password, or any token - it only talks to an
already-signed-in local install through the in-process API. The agent NEVER saves:
Save/Export/Delete are on the deny-list, a fresh scratch design is created per
reset, and exports go to a harness-chosen scratch path through the API's own
ExportManager.

Reachability: the oracle needs the ``adsk`` module, which imports ONLY inside a
running Fusion; the actuator needs ``uiautomation``. Both are almost certainly
absent here, so :func:`available` reports what is missing and the environment SKIPS
with :class:`FusionUnavailable`; a live run awaits Fusion.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

from harnesscad.core.cisp.ops import Op
from harnesscad.core.environment import (
    CapabilityError, Capabilities, Observation, StepResult, coerce_ops,
)
from harnesscad.io.backends.base import BackendUnavailable
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.cua import bindings_fusion as B
from harnesscad.io.cua import guardrails


class FusionUnavailable(BackendUnavailable):
    """The Fusion environment cannot run here. Carries the precise reason."""

    def __init__(self, message: str, searched=()) -> None:
        super().__init__("Fusion 360", message, searched)


# ---------------------------------------------------------------------------
# the ORACLE - Fusion's own Python API, reached out-of-band by the harness
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MassProperties:
    """The oracle's structured read of the design's geometry, in mm units."""

    volume_mm3: float
    surface_area_mm2: float
    centroid_mm: Tuple[float, float, float]
    mass: float

    def to_dict(self) -> dict:
        return {"volume": self.volume_mm3, "surface_area": self.surface_area_mm2,
                "center_of_mass": list(self.centroid_mm), "mass": self.mass}


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box from the oracle, in mm units."""

    low_mm: Tuple[float, float, float]
    high_mm: Tuple[float, float, float]

    @property
    def size_mm(self) -> Tuple[float, float, float]:
        return tuple(self.high_mm[i] - self.low_mm[i] for i in range(3))  # type: ignore[return-value]

    def to_dict(self) -> dict:
        return {"bbox": list(self.size_mm), "bbox_min": list(self.low_mm),
                "bbox_max": list(self.high_mm)}


def api_available() -> Tuple[bool, str]:
    """(is Fusion's ``adsk`` API importable here, why not). Never raises.

    ``adsk`` imports ONLY inside a running Fusion process; outside it, this is the
    honest 'not running inside Fusion' skip.
    """
    try:
        import adsk.core  # noqa: F401
        import adsk.fusion  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, ("the Fusion 'adsk' API is not importable (it exists only "
                       "inside a running Fusion process): %s" % exc)
    return True, ""


class FusionOracle:
    """Fusion's Python API as the oracle. Reads only; the agent never sees it.

    CRITICAL: lengths are converted from Fusion's centimetre unit to millimetres
    at this boundary.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = float(timeout)
        self._app = None       # adsk.core.Application
        self._design = None    # adsk.fusion.Design
        self._doc = None       # adsk.core.Document (scratch)

    def available(self) -> Tuple[bool, str]:
        return api_available()

    def _connect(self):
        if self._app is not None:
            return self._app
        ok, why = self.available()
        if not ok:
            raise FusionUnavailable(why)
        import adsk.core
        app = adsk.core.Application.get()
        if app is None:
            raise FusionUnavailable("adsk.core.Application.get() returned None; "
                                    "Fusion is not running")
        self._app = app
        return app

    def new_scratch_design(self):
        """Create a fresh harness-owned design document. No user file is opened."""
        app = self._connect()
        import adsk.core
        import adsk.fusion
        try:
            self._doc = app.documents.add(
                adsk.core.DocumentTypes.FusionDesignDocumentType)
            self._design = adsk.fusion.Design.cast(app.activeProduct)
        except Exception as exc:  # noqa: BLE001
            raise FusionUnavailable(
                "Fusion would not create a scratch design: %s" % exc) from exc
        return self._doc

    def _root(self):
        if self._design is None:
            raise FusionUnavailable("no scratch design; call new_scratch_design()")
        return self._design.rootComponent

    def mass_properties(self) -> MassProperties:
        """rootComponent.physicalProperties -> volume/area/centroid, cm -> mm."""
        props = self._root().physicalProperties
        volume = float(props.volume) * B.CM3_TO_MM3
        area = float(props.area) * B.CM2_TO_MM2
        com = props.centerOfMass   # a Point3D in cm
        centroid = (float(com.x) * B.CM_TO_MM, float(com.y) * B.CM_TO_MM,
                    float(com.z) * B.CM_TO_MM)
        mass = float(props.mass)
        return MassProperties(volume, area, centroid, mass)

    def bounding_box(self) -> BoundingBox:
        """rootComponent.boundingBox -> AABB, cm -> mm."""
        bb = self._root().boundingBox
        lo, hi = bb.minPoint, bb.maxPoint
        low = (float(lo.x) * B.CM_TO_MM, float(lo.y) * B.CM_TO_MM,
               float(lo.z) * B.CM_TO_MM)
        high = (float(hi.x) * B.CM_TO_MM, float(hi.y) * B.CM_TO_MM,
                float(hi.z) * B.CM_TO_MM)
        return BoundingBox(low, high)

    def body_count(self) -> int:
        """rootComponent.bRepBodies.count - the structural oracle."""
        try:
            return int(self._root().bRepBodies.count)
        except Exception:  # noqa: BLE001
            return 0

    def export(self, path: str, fmt: str) -> str:
        """HARNESS-owned export to a scratch path through ExportManager.

        Never a ribbon Save/Export the agent could reach. Returns the path.
        """
        if self._design is None:
            raise FusionUnavailable("no scratch design; call new_scratch_design()")
        mgr = self._design.exportManager
        f = str(fmt).lower()
        try:
            if f == "step":
                options = mgr.createSTEPExportOptions(path)
            elif f == "stl":
                options = mgr.createSTLExportOptions(self._root(), path)
            else:
                raise FusionUnavailable("Fusion oracle cannot export %r" % fmt)
            mgr.execute(options)
        except FusionUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise FusionUnavailable("Fusion export to %r failed: %s"
                                    % (path, exc)) from exc
        return path

    def close(self) -> None:
        try:
            if self._doc is not None:
                self._doc.close(False)  # saveChanges=False: the harness never saves
        except Exception:  # noqa: BLE001
            pass
        self._doc = None
        self._design = None
        self._app = None


# ---------------------------------------------------------------------------
# the ACTUATOR - the partial-a11y ribbon
# ---------------------------------------------------------------------------
class RibbonActuator(Protocol):
    """The GUI-driving surface: Fusion's ribbon + command dialogs. Coordinate-free
    and resolve-before-act. Fusion's a11y tree is PARTIAL; an unreachable control
    is a refusal, never a vision guess."""

    def available(self) -> Tuple[bool, str]: ...
    def focus(self) -> bool: ...
    def click_control(self, query: Dict[str, str]) -> bool: ...
    def fill_field(self, automation_id: str, value: float, unit: str = "mm") -> str: ...
    def read_field(self, automation_id: str) -> str: ...
    def top_windows(self) -> List[str]: ...
    def title(self) -> str: ...
    def close(self) -> None: ...


class UiaRibbonActuator:
    """A ``uiautomation``-backed ribbon actuator for Fusion. Lazy and gated.

    Because Fusion's a11y tree is partial, a control that does not resolve raises,
    and the environment refuses the op rather than guessing at a pixel.
    """

    def __init__(self, window_class: str = B.WINDOW_CLASS) -> None:
        self.window_class = window_class
        self._driver = None

    def available(self) -> Tuple[bool, str]:
        try:
            from harnesscad.io.cua import uia
        except Exception as exc:  # noqa: BLE001
            return False, "the UIA driver is not importable: %s" % exc
        if not uia.available():
            return False, ("uiautomation/Windows is not available, so the Fusion "
                           "ribbon cannot be driven")
        return True, ""

    def _ensure(self):
        if self._driver is not None:
            return self._driver
        ok, why = self.available()
        if not ok:
            raise FusionUnavailable(why)
        from harnesscad.io.cua import uia
        self._driver = uia.UiaDriver(class_name=self.window_class)
        self._driver.window()
        return self._driver

    def focus(self) -> bool:
        return bool(self._ensure().focus_window())

    def click_control(self, query: Dict[str, str]) -> bool:
        driver = self._ensure()
        element = driver.wait_for(timeout=15.0, enabled_only=True, **query)
        if element is None:
            return False
        return bool(driver.invoke(element).ok)

    def fill_field(self, automation_id: str, value: float, unit: str = "mm") -> str:
        driver = self._ensure()
        element = driver.wait_for(timeout=15.0, aid_suffix=automation_id)
        if element is None:
            raise FusionUnavailable("field %r never appeared" % automation_id)
        driver.set_quantity(element, float(value), locale=driver.locale())
        return driver.read_value(element)

    def read_field(self, automation_id: str) -> str:
        driver = self._ensure()
        element = driver.find(aid_suffix=automation_id)
        return str(driver.read_value(element))

    def top_windows(self) -> List[str]:
        try:
            return list(self._ensure().top_windows())
        except Exception:  # noqa: BLE001
            return []

    def title(self) -> str:
        try:
            return str(self._ensure().title())
        except Exception:  # noqa: BLE001
            return ""

    def close(self) -> None:
        self._driver = None


# ---------------------------------------------------------------------------
# reachability
# ---------------------------------------------------------------------------
def available(oracle: Optional[FusionOracle] = None,
              actuator: Optional[RibbonActuator] = None) -> Tuple[bool, str]:
    """(can this environment run here, why not). Never raises, never hangs."""
    missing: List[str] = []
    orc = oracle if oracle is not None else FusionOracle()
    ok, why = orc.available()
    if not ok:
        missing.append("oracle: " + why)
    act = actuator if actuator is not None else UiaRibbonActuator()
    ok, why = act.available()
    if not ok:
        missing.append("actuator: " + why)
    if missing:
        return False, "; ".join(missing)
    return True, ""


def _numeric_matches(intended: float, read_back: str, rel_tol: float = 1e-4) -> bool:
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(read_back).replace(",", "."))
    if not m:
        return False
    try:
        got = float(m.group(0))
    except ValueError:
        return False
    if intended == 0.0:
        return abs(got) < 1e-9
    return abs(got - intended) <= max(rel_tol * abs(intended), 1e-6)


# ---------------------------------------------------------------------------
# the Environment
# ---------------------------------------------------------------------------
class FusionGuiEnvironment:
    """A live Fusion 360 GUI as an Environment; the adsk Python API is the oracle."""

    CAPABILITIES = Capabilities(
        name="fusion-gui",
        content_digest=False,
        nonmutating_reject=False,
        # TRUE: PhysicalProperties is a synchronous structured read via the API.
        synchronous_read=True,
        deterministic_replay=False,
        export=True,
        export_formats=("step", "stl"),
        supported_ops=tuple(B.RECIPES.keys()),
        unsupported_ops=dict(B.REQUIRES_PICK),
        resolve_before_act=True,
        notes=(
            "the ribbon is the ACTUATOR; Fusion's adsk Python API is the ORACLE, "
            "reached out-of-band by the harness so the agent never touches it",
            "synchronous_read is TRUE (unlike FreeCAD's GUI): PhysicalProperties is "
            "a synchronous structured read through the API",
            "Fusion's a11y tree is PARTIAL (Qt/OpenGL canvas): unreachable controls "
            "are refused, never guessed at with vision",
            "no content digest: a Fusion design has no content hash; its version id "
            "is a document handle, and this environment will not fake one",
            "reject is MUTATING: a refused command dialog has already opened a panel "
            "and begun a preview",
            "the agent NEVER saves, exports or deletes; the harness owns all file "
            "I/O and exports to a scratch path through the API's ExportManager",
            "no Autodesk account, password or token is ever handled",
            "only the sketch-a-profile-then-feature (+Combine-all-bodies) subset is "
            "drivable coordinate-free; every pick-dependent op is refused, not faked",
            "CRITICAL: Fusion's API is in CENTIMETRES; geometry is converted "
            "cm -> mm (x10 length, x1000 volume) at the oracle boundary",
        ),
    )

    def __init__(self, oracle: Optional[FusionOracle] = None,
                 actuator: Optional[RibbonActuator] = None,
                 scratch: Optional[guardrails.Scratch] = None) -> None:
        ok, why = available(oracle, actuator)
        if not ok:
            raise FusionUnavailable("the Fusion environment cannot run: " + why)
        self.oracle = oracle or FusionOracle()
        self.actuator: RibbonActuator = actuator or UiaRibbonActuator()
        self.scratch = scratch or guardrails.Scratch()
        self.guards = guardrails.Guardrails(
            expected_window_classes=(B.WINDOW_CLASS,))
        self._steps = 0
        self._built: List[Op] = []
        self._outcomes: List[Dict[str, Any]] = []

    # -- Environment -------------------------------------------------------
    def capabilities(self) -> Capabilities:
        return self.CAPABILITIES

    def reset(self) -> Observation:
        self.close()
        self.oracle.new_scratch_design()
        self.actuator.focus()
        self._steps = 0
        self._built = []
        self._outcomes = []
        return self.observe()

    def step(self, action) -> StepResult:
        ops = coerce_ops(action)
        self._steps += 1
        caps = self.capabilities()
        diags: List[Diagnostic] = []
        for op in ops:
            tag = getattr(type(op), "OP", "")
            if not caps.supports(tag):
                diags.append(Diagnostic(Severity.ERROR, "unsupported-op",
                                        "fusion-gui cannot drive '%s': %s"
                                        % (tag, caps.why_not(tag))))
        if diags:
            return StepResult(ok=False, verified=False, observation=self.observe(),
                              reward=-1.0, diagnostics=diags,
                              info={"step": self._steps})

        before = self._safe_volume()
        executed = 0
        for op in ops:
            recipe = B.RECIPES[getattr(type(op), "OP", "")]
            try:
                outcome = self._run_recipe(recipe, op)
            except (guardrails.GuardrailViolation, FusionUnavailable) as exc:
                diags.append(Diagnostic(Severity.ERROR, "gui-error", str(exc)))
                break
            self._outcomes.append(outcome)
            if not outcome["ok"]:
                diags.append(Diagnostic(Severity.ERROR, "gui-error",
                                        outcome.get("error", "unverified action")))
                break
            self._built.append(op)
            executed += 1

        ok = not diags
        verified = ok and self._verify_via_oracle(before, executed)
        return StepResult(
            ok=ok, verified=verified, observation=self.observe(),
            reward=1.0 if ok else -1.0, diagnostics=diags,
            info={"step": self._steps, "executed_ops": executed,
                  "outcomes": self._outcomes[-3:]})

    def observe(self) -> Observation:
        state: Dict[str, Any] = {"ops_built": [op.to_dict() for op in self._built]}
        try:
            state["body_count"] = self.oracle.body_count()
        except FusionUnavailable as exc:
            state["oracle_error"] = str(exc)
        try:
            state.update(guardrails.dirty_tripwire(self.actuator.title()))
            state["top_windows"] = self.actuator.top_windows()
        except Exception:  # noqa: BLE001
            pass
        return Observation(kind="structured", state=state, digest=None,
                           step=self._steps,
                           notes=("no content digest: a Fusion design has none and "
                                  "this environment will not invent one",))

    def export(self, fmt: str):
        f = str(fmt).lower()
        if f not in self.CAPABILITIES.export_formats:
            raise ValueError("fusion-gui cannot export %r (supported: %s)"
                             % (fmt, ", ".join(self.CAPABILITIES.export_formats)))
        path = self.scratch.export_path("model." + f)
        self.oracle.export(path, f)
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def close(self) -> None:
        try:
            self.oracle.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.actuator.close()
        except Exception:  # noqa: BLE001
            pass

    # -- capability-gated --------------------------------------------------
    def state_digest(self) -> str:
        raise CapabilityError(
            self.CAPABILITIES.name, "content_digest",
            "a Fusion design has no content hash of its geometry; its version id is "
            "a document handle, not a content digest. Returning it as one would be "
            "a silent lie.")

    def query(self, q: str) -> dict:
        if q in ("measure", "metrics"):
            out = self.oracle.mass_properties().to_dict()
            out.update(self.oracle.bounding_box().to_dict())
            return out
        if q == "measure_volume":
            return {"volume": self.oracle.mass_properties().volume_mm3}
        if q == "bbox":
            return self.oracle.bounding_box().to_dict()
        if q == "validity":
            return {"body_count": self.oracle.body_count()}
        raise CapabilityError(self.CAPABILITIES.name, "synchronous_read",
                              "unknown query %r; try measure/bbox/validity" % q)

    def measure(self, q: str = "measure") -> dict:
        return self.query(q if q != "measure" else "measure")

    # -- the ribbon recipe runner -----------------------------------------
    def _run_recipe(self, recipe: B.GuiRecipe, op: Op) -> Dict[str, Any]:
        t0 = time.time()
        if not self.actuator.click_control(recipe.control.query()):
            return {"ok": False, "op": recipe.op,
                    "error": "command %r not reachable (Fusion a11y is partial)"
                             % recipe.control.command,
                    "elapsed": time.time() - t0}
        writes: List[dict] = []
        for fb in recipe.fields:
            value = B.value_for(op, fb.source) * getattr(fb, "scale", 1.0)
            read_back = self.actuator.fill_field(fb.automation_id, value, fb.unit)
            landed = _numeric_matches(value, read_back)
            writes.append({"field": fb.automation_id, "intended": value,
                           "unit": fb.unit, "read_back": read_back,
                           "verified": landed})
            if not landed:
                self.actuator.click_control(B.CANCEL.query())
                return {"ok": False, "op": recipe.op,
                        "error": "field %s: read back %r, intended %g - not verified"
                                 % (fb.automation_id, read_back, value),
                        "writes": writes, "elapsed": time.time() - t0}
        self.actuator.click_control(recipe.confirm.query())
        return {"ok": True, "op": recipe.op, "writes": writes,
                "elapsed": time.time() - t0}

    def _safe_volume(self) -> float:
        try:
            return self.oracle.mass_properties().volume_mm3
        except Exception:  # noqa: BLE001
            return 0.0

    def _verify_via_oracle(self, before_volume: float, executed: int) -> bool:
        if executed == 0:
            return True
        try:
            after = self.oracle.mass_properties().volume_mm3
        except FusionUnavailable:
            return False
        if after != before_volume:
            return True
        last = self._built[-1] if self._built else None
        return getattr(type(last), "OP", "") in ("new_sketch", "add_rectangle",
                                                  "add_circle")

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "FusionGuiEnvironment":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
        self.scratch.cleanup()
