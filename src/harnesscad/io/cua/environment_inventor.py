"""environment_inventor - a live Autodesk Inventor GUI as an Environment, with the
COM API (``Inventor.Application``) as a SEPARATE, agent-untouchable ORACLE.

This is the Inventor sibling of :mod:`environment_solidworks`: drive the WPF ribbon
through the accessibility tree (the ACTUATOR), read the part back through Inventor's
COM automation object (the ORACLE). The agent only ever touches the ribbon; the
COM read is a synchronous structured read it never actuates, so the environment
declares ``synchronous_read = True`` honestly.

The one thing to get right that SolidWorks does not have
--------------------------------------------------------
Inventor's API is in its internal DATABASE units, which are CENTIMETRES, not the SI
metres SolidWorks reports. So ``ComponentDefinition.MassProperties.Volume`` is
cm^3 and ``RangeBox`` points are cm; the oracle converts cm -> mm (x10) for length
and cm^3 -> mm^3 (x1000) for volume at its boundary. Get that wrong and the
differential compare is silently off by 1000, which is exactly the class of error
this project exists to make impossible.

Capabilities differ from FreeCAD's GUI the same way SolidWorks's do:
``synchronous_read = True`` (the COM oracle), ``resolve_before_act = True``
(a11y-grounded ribbon), ``content_digest = False`` and ``deterministic_replay =
False`` (a running document has no content hash; rebuild order and window state
are not reproducible).

Credentials and safety: no licence, password or secret is ever handled - the
module only attaches to a local COM install. The agent NEVER saves: the deny-list
covers Save/SaveAs/Delete/Exit, a fresh scratch part is created per reset, and
exports go to a harness-chosen scratch path through the oracle's own SaveAs.

Reachability: needs ``win32com`` + an installed Inventor (oracle) and
``uiautomation`` (actuator). Almost certainly absent here, so :func:`available`
reports what is missing and the environment SKIPS with
:class:`InventorUnavailable`; a live run awaits the installed application.
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
from harnesscad.io.cua import bindings_inventor as B
from harnesscad.io.cua import guardrails


class InventorUnavailable(BackendUnavailable):
    """The Inventor environment cannot run here. Carries the precise reason."""

    def __init__(self, message: str, searched=()) -> None:
        super().__init__("Inventor", message, searched)


# ---------------------------------------------------------------------------
# the ORACLE - the COM automation object the agent never touches
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MassProperties:
    """The oracle's structured read of the part's geometry, in mm units."""

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


def com_available(progid: str = B.COM_PROGID) -> Tuple[bool, str]:
    """(is a usable COM stack importable AND the app installed, why not).

    Never raises and never LAUNCHES the app: it checks that ``win32com`` imports
    and that ``progid`` is registered in HKCR (the honest 'is Inventor installed'
    signal), so a machine with pywin32 but no Inventor skips cleanly rather than
    reporting available and then failing on the first COM Dispatch.
    """
    try:
        import win32com.client  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, ("win32com (pywin32) is not importable, so the Inventor COM "
                       "oracle cannot attach: %s" % exc)
    try:
        import winreg
        winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, progid + r"\CLSID"))
    except Exception:  # noqa: BLE001
        return False, ("the Inventor COM class %r is not registered on this machine "
                       "(Inventor does not appear to be installed)" % progid)
    return True, ""


class InventorOracle:
    """The COM oracle (``Inventor.Application``). Reads only; the agent never sees it.

    CRITICAL: every length it returns is converted from Inventor's centimetre
    database units to millimetres at this boundary.
    """

    def __init__(self, progid: str = B.COM_PROGID, timeout: float = 30.0) -> None:
        self.progid = progid
        self.timeout = float(timeout)
        self._app = None     # Inventor.Application
        self._doc = None     # PartDocument (the scratch part)

    def available(self) -> Tuple[bool, str]:
        return com_available(self.progid)

    def _dispatch(self):
        if self._app is not None:
            return self._app
        ok, why = self.available()
        if not ok:
            raise InventorUnavailable(why)
        import win32com.client
        try:
            self._app = win32com.client.Dispatch(self.progid)
        except Exception as exc:  # noqa: BLE001
            raise InventorUnavailable(
                "Inventor COM object %r would not dispatch (is Inventor installed "
                "and licensed on this machine?): %s" % (self.progid, exc)) from exc
        try:
            self._app.Visible = True
        except Exception:  # noqa: BLE001
            pass
        return self._app

    def new_scratch_part(self):
        """Create a fresh harness-owned part document. No user file is opened."""
        app = self._dispatch()
        try:
            template = app.FileManager.GetTemplateFile(B.PART_DOCUMENT_OBJECT)
        except Exception:  # noqa: BLE001
            template = ""
        try:
            self._doc = app.Documents.Add(B.PART_DOCUMENT_OBJECT, template, True)
        except Exception as exc:  # noqa: BLE001
            raise InventorUnavailable(
                "Inventor would not create a scratch part through COM: %s" % exc
            ) from exc
        return self._doc

    def _require_def(self):
        if self._doc is None:
            raise InventorUnavailable("no scratch part; call new_scratch_part()")
        return self._doc.ComponentDefinition

    def mass_properties(self) -> MassProperties:
        """ComponentDefinition.MassProperties -> volume/area/centroid, cm -> mm."""
        mp = self._require_def().MassProperties
        volume = float(mp.Volume) * B.CM3_TO_MM3
        area = float(mp.Area) * B.CM2_TO_MM2
        com = mp.CenterOfMass   # a Point in cm
        centroid = (float(com.X) * B.CM_TO_MM, float(com.Y) * B.CM_TO_MM,
                    float(com.Z) * B.CM_TO_MM)
        mass = float(mp.Mass)
        return MassProperties(volume, area, centroid, mass)

    def bounding_box(self) -> BoundingBox:
        """ComponentDefinition.RangeBox -> AABB, cm -> mm."""
        rb = self._require_def().RangeBox
        lo, hi = rb.MinPoint, rb.MaxPoint
        low = (float(lo.X) * B.CM_TO_MM, float(lo.Y) * B.CM_TO_MM,
               float(lo.Z) * B.CM_TO_MM)
        high = (float(hi.X) * B.CM_TO_MM, float(hi.Y) * B.CM_TO_MM,
                float(hi.Z) * B.CM_TO_MM)
        return BoundingBox(low, high)

    def body_count(self) -> int:
        """SurfaceBodies.Count - the structural oracle for verification."""
        try:
            return int(self._require_def().SurfaceBodies.Count)
        except Exception:  # noqa: BLE001
            return 0

    def export(self, path: str) -> str:
        """HARNESS-owned SaveAs to a scratch path. Never a ribbon Save.

        Native SaveAs is used here; a STEP/STL translator add-in produces those
        formats in a live deployment. Returns the path written.
        """
        if self._doc is None:
            raise InventorUnavailable("no scratch part; call new_scratch_part()")
        try:
            self._doc.SaveAs(path, False)
        except Exception as exc:  # noqa: BLE001
            raise InventorUnavailable("Inventor SaveAs to %r failed: %s"
                                      % (path, exc)) from exc
        return path

    def close(self) -> None:
        try:
            if self._doc is not None:
                self._doc.Close(True)  # skipSave=True: the harness never saves
        except Exception:  # noqa: BLE001
            pass
        self._doc = None
        self._app = None


# ---------------------------------------------------------------------------
# the ACTUATOR - the ribbon via the accessibility tree
# ---------------------------------------------------------------------------
class RibbonActuator(Protocol):
    """The GUI-driving surface: Inventor's ribbon + command dialogs. Coordinate-
    free and resolve-before-act, so the guardrails can REFUSE before dispatch."""

    def available(self) -> Tuple[bool, str]: ...
    def focus(self) -> bool: ...
    def click_control(self, query: Dict[str, str]) -> bool: ...
    def fill_field(self, automation_id: str, value: float, unit: str = "mm") -> str: ...
    def read_field(self, automation_id: str) -> str: ...
    def top_windows(self) -> List[str]: ...
    def title(self) -> str: ...
    def close(self) -> None: ...


class UiaRibbonActuator:
    """A ``uiautomation``-backed ribbon actuator for Inventor. Lazy and gated."""

    def __init__(self, window_class: str = B.WINDOW_CLASS) -> None:
        self.window_class = window_class
        self._driver = None

    def available(self) -> Tuple[bool, str]:
        try:
            from harnesscad.io.cua import uia
        except Exception as exc:  # noqa: BLE001
            return False, "the UIA driver is not importable: %s" % exc
        if not uia.available():
            return False, ("uiautomation/Windows is not available, so the Inventor "
                           "ribbon cannot be driven")
        return True, ""

    def _ensure(self):
        if self._driver is not None:
            return self._driver
        ok, why = self.available()
        if not ok:
            raise InventorUnavailable(why)
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
            raise InventorUnavailable("field %r never appeared" % automation_id)
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
def available(oracle: Optional[InventorOracle] = None,
              actuator: Optional[RibbonActuator] = None) -> Tuple[bool, str]:
    """(can this environment run here, why not). Never raises, never hangs."""
    missing: List[str] = []
    orc = oracle if oracle is not None else InventorOracle()
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
class InventorGuiEnvironment:
    """A live Inventor GUI as an Environment; the COM API is the oracle."""

    CAPABILITIES = Capabilities(
        name="inventor-gui",
        content_digest=False,
        nonmutating_reject=False,
        # TRUE: MassProperties is a synchronous structured read through COM.
        synchronous_read=True,
        deterministic_replay=False,
        export=True,
        export_formats=("step", "stl"),
        supported_ops=tuple(B.RECIPES.keys()),
        unsupported_ops=dict(B.REQUIRES_PICK),
        resolve_before_act=True,
        notes=(
            "the WPF ribbon is the ACTUATOR (driven through the accessibility tree, "
            "coordinate-free); the COM object Inventor.Application is the ORACLE",
            "synchronous_read is TRUE (unlike FreeCAD's GUI): MassProperties is a "
            "synchronous structured read through COM",
            "no content digest: a running Inventor document has none and this "
            "environment will not fabricate one",
            "reject is MUTATING: a refused command dialog has already opened a panel "
            "and begun a preview",
            "the agent NEVER saves, deletes or exits; the harness owns all file I/O "
            "and exports to a scratch path through the oracle's own SaveAs",
            "no licence, password or secret is ever handled",
            "only the sketch-a-profile-then-feature (+Combine-all-bodies) subset is "
            "drivable coordinate-free; every pick-dependent op is refused, not faked",
            "CRITICAL: Inventor's API is in CENTIMETRES; geometry is converted "
            "cm -> mm (x10 length, x1000 volume) at the oracle boundary",
        ),
    )

    def __init__(self, oracle: Optional[InventorOracle] = None,
                 actuator: Optional[RibbonActuator] = None,
                 scratch: Optional[guardrails.Scratch] = None) -> None:
        ok, why = available(oracle, actuator)
        if not ok:
            raise InventorUnavailable("the Inventor environment cannot run: " + why)
        self.oracle = oracle or InventorOracle()
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
        self.oracle.new_scratch_part()
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
                                        "inventor-gui cannot drive '%s': %s"
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
            except (guardrails.GuardrailViolation, InventorUnavailable) as exc:
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
        except InventorUnavailable as exc:
            state["oracle_error"] = str(exc)
        try:
            state.update(guardrails.dirty_tripwire(self.actuator.title()))
            state["top_windows"] = self.actuator.top_windows()
        except Exception:  # noqa: BLE001
            pass
        return Observation(kind="structured", state=state, digest=None,
                           step=self._steps,
                           notes=("no content digest: a running Inventor document "
                                  "has none and this environment will not invent "
                                  "one",))

    def export(self, fmt: str):
        f = str(fmt).lower()
        if f not in self.CAPABILITIES.export_formats:
            raise ValueError("inventor-gui cannot export %r (supported: %s)"
                             % (fmt, ", ".join(self.CAPABILITIES.export_formats)))
        path = self.scratch.export_path("model." + f)
        self.oracle.export(path)
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
            "a running Inventor document has no content hash of its geometry. "
            "Returning a rebuild/version id as one would be a silent lie.")

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
                    "error": "command %r not reachable" % recipe.control.command,
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
        except InventorUnavailable:
            return False
        if after != before_volume:
            return True
        last = self._built[-1] if self._built else None
        return getattr(type(last), "OP", "") in ("new_sketch", "add_rectangle",
                                                  "add_circle")

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "InventorGuiEnvironment":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
        self.scratch.cleanup()
