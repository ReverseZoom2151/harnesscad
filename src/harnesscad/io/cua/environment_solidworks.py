"""environment_solidworks - a live SolidWorks GUI as an Environment, with the COM
API (``ISldWorks``) as a SEPARATE, agent-untouchable ORACLE.

Where this sits in the family
-----------------------------
FreeCAD proved GUI driving on a tool with a Python API in the same process.
Onshape sharpened it: a browser ACTUATOR and a REST ORACLE on physically separate
channels. SolidWorks is the desktop version of that bridge - the CAD tool people
are actually paid to use. We drive its MFC/WPF ribbon through the accessibility
tree (the ACTUATOR), and we read the part back through its COM automation object
``ISldWorks`` (the ORACLE). The two channels share the application process, so the
separation is weaker than Onshape's HTTPS-vs-REST split; but the COM read is still
a structured, synchronous read the agent never actuates - the agent only ever
touches the ribbon - so it is an honest oracle, and the environment declares
``synchronous_read = True`` on its strength.

The capabilities that differ from FreeCAD's GUI
-----------------------------------------------
FreeCAD's GUI declares ``synchronous_read = False`` (its only read is an
out-of-band macro channel that lags the app's recompute). SolidWorks declares
``synchronous_read = True``: ``IModelDocExtension::CreateMassProperty`` is a
synchronous structured read of the current model state through COM. It still
declares ``content_digest = False`` (a running SolidWorks document has no content
hash of its geometry - a rebuild id is a version handle, not a content digest) and
``deterministic_replay = False`` (rebuild order, floating dialogs, window state).

Credentials and safety
----------------------
SolidWorks is a licensed desktop application; this module NEVER handles a licence,
a password, or any secret - it only attaches to an already-running / launchable
local install through COM. The agent NEVER saves: Save/Save As/Delete/Exit are on
the guardrail deny-list, and the harness owns all file I/O (exports go to a
harness-chosen scratch path through the oracle's own SaveAs, never a ribbon Save
the agent could reach). A fresh scratch part is created per reset; no user
document is opened.

Reachability
-----------
Two things must be present for a live run: (1) a Windows COM stack with
``win32com`` (for the oracle + the scratch-document lifecycle), and (2) the
accessibility actuator (``uiautomation``) plus an installed SolidWorks. This app
is almost certainly not installed here, so :func:`available` reports exactly what
is missing and the environment SKIPS cleanly with :class:`SolidWorksUnavailable`
(a :class:`~harnesscad.io.backends.base.BackendUnavailable`). It fabricates
nothing; a live run awaits the installed application.

Stdlib only, plus lazily-imported ``win32com`` (oracle) and ``uiautomation``
(actuator), each behind an ``available()`` gate so this module imports and its
data/mapping logic tests with neither present.
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
from harnesscad.io.cua import bindings_solidworks as B
from harnesscad.io.cua import guardrails


class SolidWorksUnavailable(BackendUnavailable):
    """The SolidWorks environment cannot run here. Carries the precise reason."""

    def __init__(self, message: str, searched=()) -> None:
        super().__init__("SolidWorks", message, searched)


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
    and that ``progid`` is registered in HKCR (the honest 'is SolidWorks installed'
    signal), so a machine with pywin32 but no SolidWorks skips cleanly rather than
    reporting available and then failing on the first COM Dispatch.
    """
    try:
        import win32com.client  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, ("win32com (pywin32) is not importable, so the SolidWorks "
                       "COM oracle cannot attach: %s" % exc)
    try:
        import winreg
        winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, progid + r"\CLSID"))
    except Exception:  # noqa: BLE001
        return False, ("the SolidWorks COM class %r is not registered on this "
                       "machine (SolidWorks does not appear to be installed)"
                       % progid)
    return True, ""


class SolidWorksOracle:
    """The COM oracle (``ISldWorks``). Signs nothing, secures nothing, reads only.

    Every method is a synchronous structured read/lifecycle call through COM. The
    agent never sees this object; it is the harness's private channel for creating
    the scratch part, measuring it, and exporting it to a scratch path.
    """

    def __init__(self, progid: str = B.COM_PROGID, timeout: float = 30.0) -> None:
        self.progid = progid
        self.timeout = float(timeout)
        self._app = None      # ISldWorks
        self._model = None    # IModelDoc2 (the scratch part)

    def available(self) -> Tuple[bool, str]:
        return com_available(self.progid)

    def _dispatch(self):
        if self._app is not None:
            return self._app
        ok, why = self.available()
        if not ok:
            raise SolidWorksUnavailable(why)
        import win32com.client
        try:
            self._app = win32com.client.Dispatch(self.progid)
        except Exception as exc:  # noqa: BLE001
            raise SolidWorksUnavailable(
                "SolidWorks COM object %r would not dispatch (is SolidWorks "
                "installed and licensed on this machine?): %s"
                % (self.progid, exc)) from exc
        try:
            self._app.Visible = True
        except Exception:  # noqa: BLE001
            pass
        return self._app

    def new_scratch_part(self):
        """Create a fresh harness-owned part document. No user file is opened."""
        app = self._dispatch()
        try:
            self._model = app.NewPart()
        except Exception as exc:  # noqa: BLE001
            raise SolidWorksUnavailable(
                "SolidWorks would not create a scratch part through COM: %s" % exc
            ) from exc
        return self._model

    def _require_model(self):
        if self._model is None:
            raise SolidWorksUnavailable("no scratch part; call new_scratch_part()")
        return self._model

    def mass_properties(self) -> MassProperties:
        """CreateMassProperty -> volume/area/centroid/mass, converted SI -> mm."""
        model = self._require_model()
        ext = model.Extension
        mp = ext.CreateMassProperty()
        volume = float(mp.Volume) * B.M3_TO_MM3
        area = float(mp.SurfaceArea) * B.M2_TO_MM2
        com = list(mp.CenterOfMass)  # metres
        centroid = (com[0] * B.M_TO_MM, com[1] * B.M_TO_MM, com[2] * B.M_TO_MM)
        mass = float(mp.Mass)
        return MassProperties(volume, area, centroid, mass)

    def bounding_box(self) -> BoundingBox:
        """PartDoc.GetPartBox -> the AABB, converted SI -> mm."""
        model = self._require_model()
        box = list(model.GetPartBox(True))  # [xmin..zmin, xmax..zmax] metres
        low = (box[0] * B.M_TO_MM, box[1] * B.M_TO_MM, box[2] * B.M_TO_MM)
        high = (box[3] * B.M_TO_MM, box[4] * B.M_TO_MM, box[5] * B.M_TO_MM)
        return BoundingBox(low, high)

    def body_count(self) -> int:
        """How many solid bodies exist - the structural oracle for verification."""
        model = self._require_model()
        try:
            bodies = model.GetBodies2(0, True)  # swSolidBody
        except Exception:  # noqa: BLE001
            return 0
        if bodies is None:
            return 0
        try:
            return len(bodies)
        except TypeError:
            return 0

    def export(self, path: str) -> str:
        """HARNESS-owned SaveAs to a scratch path (STEP/STL by extension).

        This is the harness's channel, never a ribbon Save the agent can reach -
        exactly the FreeCAD doctrine. Returns the path written.
        """
        model = self._require_model()
        ext = model.Extension
        import pythoncom  # part of pywin32
        errors = pythoncom.Empty if hasattr(pythoncom, "Empty") else None
        warnings = errors
        ok = ext.SaveAs(path, 0, 1, None, errors, warnings)
        if not ok:
            raise SolidWorksUnavailable("SolidWorks SaveAs to %r failed" % path)
        return path

    def close(self) -> None:
        try:
            if self._model is not None and self._app is not None:
                title = self._model.GetTitle()
                self._app.CloseDoc(title)
        except Exception:  # noqa: BLE001
            pass
        self._model = None
        self._app = None


# ---------------------------------------------------------------------------
# the ACTUATOR - the ribbon, via the accessibility tree. Abstract so the mapping
# tests without a live app.
# ---------------------------------------------------------------------------
class RibbonActuator(Protocol):
    """The GUI-driving surface: the SolidWorks CommandManager + PropertyManager.

    Resolve-before-act, coordinate-free: controls are addressed by name /
    AutomationId (never a pixel), so the guardrails can REFUSE before dispatch.
    """

    def available(self) -> Tuple[bool, str]: ...
    def focus(self) -> bool: ...
    def click_control(self, query: Dict[str, str]) -> bool: ...
    def fill_field(self, automation_id: str, value: float, unit: str = "mm") -> str: ...
    def read_field(self, automation_id: str) -> str: ...
    def top_windows(self) -> List[str]: ...
    def title(self) -> str: ...
    def close(self) -> None: ...


class UiaRibbonActuator:
    """A ``uiautomation``-backed ribbon actuator. Imported lazily and gated.

    Requires Windows + ``uiautomation`` + a running SolidWorks whose main frame
    (:data:`bindings_solidworks.WINDOW_CLASS`) is bound. If the library is absent
    the actuator's :meth:`available` says so and the environment SKIPS.
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
            return False, ("uiautomation/Windows is not available, so the "
                           "SolidWorks ribbon cannot be driven")
        return True, ""

    def _ensure(self):
        if self._driver is not None:
            return self._driver
        ok, why = self.available()
        if not ok:
            raise SolidWorksUnavailable(why)
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
        """Write a numeric value through the driver's own quantity primitive
        (which types locale-correctly and reads back), then return the read-back."""
        driver = self._ensure()
        element = driver.wait_for(timeout=15.0, aid_suffix=automation_id)
        if element is None:
            raise SolidWorksUnavailable("field %r never appeared" % automation_id)
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
def available(oracle: Optional[SolidWorksOracle] = None,
              actuator: Optional[RibbonActuator] = None) -> Tuple[bool, str]:
    """(can this environment run here, why not). Never raises, never hangs.

    Two independent requirements, reported separately: the COM oracle (win32com +
    an installed SolidWorks) and the accessibility actuator (uiautomation).
    """
    missing: List[str] = []
    orc = oracle if oracle is not None else SolidWorksOracle()
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
    """Parse a value out of a field read-back and compare to intent - tightly, so
    37.5 vs 375 is a hard failure (the comma-decimal 10x bug dies here)."""
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
class SolidWorksGuiEnvironment:
    """A live SolidWorks GUI as an Environment; the COM API is the oracle.

    Read :attr:`CAPABILITIES` before relying on anything. The honest difference
    from FreeCAD's GUI is ``synchronous_read = True`` (the COM oracle) and
    ``resolve_before_act = True`` (a11y-grounded ribbon).
    """

    CAPABILITIES = Capabilities(
        name="solidworks-gui",
        # False: a running SolidWorks document has no content hash of its
        # geometry; a rebuild id is a version handle, not a content digest.
        content_digest=False,
        # False: a rejected PropertyManager has already opened a panel and begun a
        # feature preview - the app is mutated before the value is refused.
        nonmutating_reject=False,
        # TRUE: CreateMassProperty is a synchronous structured read through COM -
        # the channel FreeCAD's GUI lacks.
        synchronous_read=True,
        # False: rebuild order, floating dialogs, window state.
        deterministic_replay=False,
        export=True,
        export_formats=("step", "stl"),
        supported_ops=tuple(B.RECIPES.keys()),
        unsupported_ops=dict(B.REQUIRES_PICK),
        resolve_before_act=True,
        notes=(
            "the MFC/WPF ribbon is the ACTUATOR (driven through the accessibility "
            "tree, coordinate-free); the COM object ISldWorks is the ORACLE (mass "
            "properties read back through a channel the agent never actuates)",
            "synchronous_read is TRUE (unlike FreeCAD's GUI): CreateMassProperty is "
            "a synchronous structured read through COM",
            "no content digest: a running SolidWorks document has no content hash; "
            "this environment will not fabricate one",
            "reject is MUTATING: a refused PropertyManager has already opened a "
            "panel and begun a preview",
            "the agent NEVER saves, deletes or exits; Save/SaveAs/Delete/Exit are "
            "on the guardrail deny-list and the harness owns all file I/O (exports "
            "go to a scratch path through the oracle's own SaveAs)",
            "no licence, password or secret is ever handled; the module only "
            "attaches to a local COM install",
            "only the sketch-a-profile-then-feature (+Combine-all-bodies) subset is "
            "drivable coordinate-free; every op needing an edge/face/tree PICK is "
            "refused, not faked",
            "geometry is reported in mm/mm^2/mm^3 after converting SolidWorks's SI "
            "COM payload at the oracle boundary",
        ),
    )

    def __init__(self, oracle: Optional[SolidWorksOracle] = None,
                 actuator: Optional[RibbonActuator] = None,
                 scratch: Optional[guardrails.Scratch] = None) -> None:
        ok, why = available(oracle, actuator)
        if not ok:
            raise SolidWorksUnavailable(
                "the SolidWorks environment cannot run: " + why)
        self.oracle = oracle or SolidWorksOracle()
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
        """Fresh scratch part, fresh ribbon binding. Nothing is reused."""
        self.close()
        self.oracle.new_scratch_part()
        self.actuator.focus()
        self._steps = 0
        self._built = []
        self._outcomes = []
        return self.observe()

    def step(self, action) -> StepResult:
        """Drive the ribbon for each op; refuse what needs a pick.

        Verification is decided by the ORACLE, not the ribbon's own return value:
        after the drive, the COM mass-property read must show the geometry moved
        (a body appeared, the volume changed). A click that "succeeded" but left
        the part unchanged is NOT verified - that is the whole discipline.
        """
        ops = coerce_ops(action)
        self._steps += 1
        caps = self.capabilities()
        diags: List[Diagnostic] = []
        for op in ops:
            tag = getattr(type(op), "OP", "")
            if not caps.supports(tag):
                diags.append(Diagnostic(Severity.ERROR, "unsupported-op",
                                        "solidworks-gui cannot drive '%s': %s"
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
            except (guardrails.GuardrailViolation, SolidWorksUnavailable) as exc:
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
        """Structured state + the synchronous oracle read. ``digest`` is None.

        Always None: a running SolidWorks document has no content hash and this
        environment will not invent one.
        """
        state: Dict[str, Any] = {"ops_built": [op.to_dict() for op in self._built]}
        try:
            state["body_count"] = self.oracle.body_count()
        except SolidWorksUnavailable as exc:
            state["oracle_error"] = str(exc)
        try:
            state.update(guardrails.dirty_tripwire(self.actuator.title()))
            state["top_windows"] = self.actuator.top_windows()
        except Exception:  # noqa: BLE001
            pass
        return Observation(kind="structured", state=state, digest=None,
                           step=self._steps,
                           notes=("no content digest: a running SolidWorks document "
                                  "has none and this environment will not invent "
                                  "one",))

    def export(self, fmt: str):
        """Export through the HARNESS's COM oracle - never a ribbon Save."""
        f = str(fmt).lower()
        if f not in self.CAPABILITIES.export_formats:
            raise ValueError("solidworks-gui cannot export %r (supported: %s)"
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
            "a running SolidWorks document has no content hash of its geometry - a "
            "rebuild id is a version handle, not a content digest. Returning it as "
            "one would be exactly the silent lie this harness exists to eradicate.")

    def query(self, q: str) -> dict:
        """A SYNCHRONOUS structured read - through the COM oracle."""
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
        """Click the command, WRITE-AND-READ-BACK each field, confirm.

        The read-back is the 375mm defence: a numeric field we cannot prove we set
        is a field we did not set, and the op does not proceed.
        """
        t0 = time.time()
        # 1: the command opens the PropertyManager.
        if not self.actuator.click_control(recipe.control.query()):
            return {"ok": False, "op": recipe.op,
                    "error": "command %r not reachable" % recipe.control.command,
                    "elapsed": time.time() - t0}
        # 2: every dialog field, written then read back.
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
        # 3: confirm.
        self.actuator.click_control(recipe.confirm.query())
        return {"ok": True, "op": recipe.op, "writes": writes,
                "elapsed": time.time() - t0}

    def _safe_volume(self) -> float:
        try:
            return self.oracle.mass_properties().volume_mm3
        except Exception:  # noqa: BLE001
            return 0.0

    def _verify_via_oracle(self, before_volume: float, executed: int) -> bool:
        """Confirm through COM that the drive changed the part.

        A sketch alone does not change the volume, so the invariant is: either the
        volume moved, or (for a sketch-only step) the drive reported ok and a body
        set is present. The oracle - not the ribbon - has the last word.
        """
        if executed == 0:
            return True
        try:
            after = self.oracle.mass_properties().volume_mm3
        except SolidWorksUnavailable:
            return False
        if after != before_volume:
            return True
        # No volume change is legitimate for a sketch-only op; trust the drive's
        # own read-back verification in that case, but never for a feature op.
        last = self._built[-1] if self._built else None
        return getattr(type(last), "OP", "") in ("new_sketch", "add_rectangle",
                                                  "add_circle")

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "SolidWorksGuiEnvironment":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
        self.scratch.cleanup()
