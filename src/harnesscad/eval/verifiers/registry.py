"""Verifier registry + dispatcher — the fleet behind the block-and-correct loop.

The repo ships ~30 verifier modules under ``harnesscad.eval.verifiers``. Only
three of them were ever wired into :class:`~harnesscad.core.loop.HarnessSession`
(sketch DOF, solid presence, B-rep validity); the rest were correct, tested and
unreachable. This module makes the whole fleet dispatchable behind one uniform
protocol.

Design
------
*   **Discovery, not a hardcoded list.** Verifiers are found through the static
    capability registry (:mod:`harnesscad.registry`) by package/tag, then
    imported lazily. A module that fails to import (optional dep) is skipped,
    never fatal.
*   **Two adapter kinds.**
    - :class:`NativeVerifier` wraps any class in the package that already speaks
      the harness protocol -- ``name`` attribute + ``check(backend, opdag) ->
      VerifyReport`` -- and is constructible with no arguments.
    - :class:`FunctionVerifier` wraps the function-style modules (tolerance
      stacks, plausibility, kernel preflight, ...) whose public API is a set of
      pure functions. The adapters live *here*; the verifier modules themselves
      are never modified.
*   **Nothing crashes the loop.** A verifier that raises is caught and reported
    as a WARNING diagnostic (``verifier-error``); the run continues.
*   **Deterministic.** Verifiers are sorted by (tier, name); diagnostics come
    back in that order. No wall clock, no randomness.

The diagnostic type is the harness's own
:class:`harnesscad.eval.verifiers.verify.Diagnostic` -- nothing new is invented.

Typical use::

    from harnesscad.eval.verifiers.registry import model_state, run_all
    diags = run_all(model_state(backend, opdag))
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from harnesscad import registry as capability_registry
from harnesscad.eval.verifiers import soundness as _soundness
from harnesscad.eval.verifiers.verify import Diagnostic, Severity, VerifyReport

__all__ = [
    "TIERS",
    "CORE",
    "LINT",
    "PHYSICS",
    "DOMAIN",
    "ModelState",
    "model_state",
    "Verifier",
    "NativeVerifier",
    "FunctionVerifier",
    "discover",
    "run_all",
    "run_report",
    "conformance",
]

# --- tiers -----------------------------------------------------------------
# CORE    -- the checks the loop has always run (kept out of the fleet so the
#            session never double-reports them).
# LINT    -- cheap, state-only checks (plan preflight, DFM, standards, ...).
# PHYSICS -- simulation / stability / plausibility.
# DOMAIN  -- checks that need domain data the backend may not expose (bricks,
#            rims, aero, tolerance chains). They INFO-skip when data is absent.
CORE = "core"
LINT = "lint"
PHYSICS = "physics"
DOMAIN = "domain"
TIERS: Tuple[str, ...] = (CORE, LINT, PHYSICS, DOMAIN)

_TIER_ORDER = {t: i for i, t in enumerate(TIERS)}

# Module name -> tier. A module not listed here lands in LINT.
_MODULE_TIER: Dict[str, str] = {
    "assembly": LINT,
    "access": LINT,
    "compliance": LINT,
    "completeness": LINT,
    "dfm": LINT,
    "functional": LINT,
    "geometry": CORE,
    "interference": LINT,
    "precheck": LINT,
    "standards": LINT,
    "simulation": PHYSICS,
}

# Verifier modules whose classes must NOT be auto-discovered into the fleet:
# ``verify`` holds the CORE checks the session already runs directly, and this
# module is the dispatcher itself.
_EXCLUDED_MODULES = frozenset({"verify", "registry"})


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ModelState:
    """Read-only view of the model a verifier is asked about.

    Wraps the backend + OpDAG the loop already owns, and memoises the backend
    queries so running 20 verifiers does not re-query 20 times. Every accessor
    is total: a backend that does not answer a query yields ``{}`` / ``[]``.
    """

    def __init__(self, backend: Any, opdag: Any) -> None:
        self.backend = backend
        self.opdag = opdag
        self._q: Dict[str, dict] = {}

    # -- backend projections ------------------------------------------------
    def query(self, what: str) -> dict:
        if what not in self._q:
            try:
                res = self.backend.query(what)
            except Exception:  # noqa: BLE001 - a hostile backend must not crash us
                res = {}
            self._q[what] = res if isinstance(res, dict) else {}
        return self._q[what]

    @property
    def summary(self) -> dict:
        return self.query("summary")

    @property
    def sketch_dof(self) -> dict:
        return self.query("sketch_dof")

    @property
    def assembly(self) -> dict:
        return self.query("assembly")

    @property
    def parts(self) -> List[dict]:
        parts = self.assembly.get("parts") or []
        return [p for p in parts if isinstance(p, dict)]

    def part_bboxes(self) -> List[Tuple[str, Tuple[float, float, float, float, float, float]]]:
        """(part id, AABB) for every placed part that carries one. Ordered."""
        out = []
        for i, p in enumerate(self.parts):
            bb = p.get("bbox")
            if bb is None or len(bb) < 6:
                continue
            out.append((str(p.get("id") or f"part{i}"), tuple(float(v) for v in bb[:6])))
        return out

    # -- op-stream projections ---------------------------------------------
    def ops(self) -> List[Any]:
        try:
            return list(self.opdag.ops())
        except Exception:  # noqa: BLE001
            return []

    def ops_of(self, *types: type) -> List[Any]:
        return [o for o in self.ops() if isinstance(o, types)]

    def envelope(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Approximate model AABB derived from the OP STREAM (kernel-free).

        The stub backend carries no geometry, so the envelope is reconstructed
        from the sketch primitives and the extrude distances that consumed them.
        It is an advisory footprint, not a measured B-rep bound -- which is
        exactly the resolution the LINT/PHYSICS tiers need (is the fillet larger
        than the part? is the shell thicker than the wall?).
        """
        from harnesscad.core.cisp.ops import AddCircle, AddLine, AddPoint, AddRectangle, Extrude, Revolve

        planar: Dict[str, List[float]] = {}   # sketch id -> [minx, miny, maxx, maxy]
        order: List[str] = []                 # sketch ids in creation order

        def _grow(sid: str, x0: float, y0: float, x1: float, y1: float) -> None:
            box = planar.get(sid)
            if box is None:
                planar[sid] = [x0, y0, x1, y1]
            else:
                box[0] = min(box[0], x0)
                box[1] = min(box[1], y0)
                box[2] = max(box[2], x1)
                box[3] = max(box[3], y1)

        n_sketch = 0
        for op in self.ops():
            if type(op).__name__ == "NewSketch":
                n_sketch += 1
                order.append(f"sk{n_sketch}")
                continue
            if isinstance(op, AddRectangle):
                _grow(op.sketch, min(op.x, op.x + op.w), min(op.y, op.y + op.h),
                      max(op.x, op.x + op.w), max(op.y, op.y + op.h))
            elif isinstance(op, AddCircle):
                _grow(op.sketch, op.cx - op.r, op.cy - op.r, op.cx + op.r, op.cy + op.r)
            elif isinstance(op, AddLine):
                _grow(op.sketch, min(op.x1, op.x2), min(op.y1, op.y2),
                      max(op.x1, op.x2), max(op.y1, op.y2))
            elif isinstance(op, AddPoint):
                _grow(op.sketch, op.x, op.y, op.x, op.y)

        lo = [0.0, 0.0, 0.0]
        hi = [0.0, 0.0, 0.0]
        seen = False
        for op in self.ops():
            box = None
            depth = 0.0
            if isinstance(op, Extrude):
                box = planar.get(op.sketch)
                depth = abs(float(op.distance))
            elif isinstance(op, Revolve):
                box = planar.get(op.sketch)
                if box is not None:
                    depth = 2.0 * max(abs(box[0]), abs(box[2]))
            if box is None:
                continue
            corners = ((box[0], box[1], 0.0), (box[2], box[3], depth))
            if not seen:
                lo = [min(corners[0][i], corners[1][i]) for i in range(3)]
                hi = [max(corners[0][i], corners[1][i]) for i in range(3)]
                seen = True
            else:
                lo = [min(lo[i], corners[0][i], corners[1][i]) for i in range(3)]
                hi = [max(hi[i], corners[0][i], corners[1][i]) for i in range(3)]
        if not seen:
            return None
        return (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])


def model_state(backend: Any, opdag: Any) -> ModelState:
    """Build the state a fleet run reads (never mutates the backend)."""
    return ModelState(backend, opdag)


# ---------------------------------------------------------------------------
# Protocol + adapters
# ---------------------------------------------------------------------------

class Verifier(Protocol):
    """What the dispatcher needs from anything it runs.

    ``tier`` is the COST/SCOPE tier (core / lint / physics / domain: when to run
    it). ``soundness`` is the TRUST tier (proven / measured / heuristic: whether
    its word may be given to a model as an instruction). They are orthogonal and
    must not be confused -- the pressure experiment lost 8.3 points precisely
    because the fleet had the first and not the second.
    """

    name: str
    tier: str

    def applies_to(self, state: ModelState) -> bool: ...

    def check(self, state: ModelState) -> List[Diagnostic]: ...


class NativeVerifier:
    """Adapts a class that already speaks ``check(backend, opdag) -> VerifyReport``."""

    def __init__(self, inner: Any, tier: str, dotted: str = "") -> None:
        self.inner = inner
        self.tier = tier
        self.dotted = dotted
        self.name = str(getattr(inner, "name", type(inner).__name__))

    @property
    def soundness(self):
        """Declared trust tier (quarantined as HEURISTIC when undeclared)."""
        return _soundness.soundness_or_untrusted(self.name)

    def applies_to(self, state: ModelState) -> bool:  # noqa: ARG002 - always eligible
        return True

    def check(self, state: ModelState) -> List[Diagnostic]:
        report = self.inner.check(state.backend, state.opdag)
        diags = getattr(report, "diagnostics", None)
        if diags is None and isinstance(report, list):
            diags = report
        return list(diags or [])


class FunctionVerifier:
    """Adapts a function-style verifier module (an ``applies`` + a ``run``)."""

    def __init__(self, name: str, tier: str, applies, run, dotted: str = "") -> None:
        self.name = name
        self.tier = tier
        self.dotted = dotted
        self._applies = applies
        self._run = run

    @property
    def soundness(self):
        """Declared trust tier (quarantined as HEURISTIC when undeclared)."""
        return _soundness.soundness_or_untrusted(self.name)

    def applies_to(self, state: ModelState) -> bool:
        return bool(self._applies(state))

    def check(self, state: ModelState) -> List[Diagnostic]:
        return list(self._run(state) or [])


# --- diagnostic helpers -----------------------------------------------------

def _err(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.ERROR, code, msg, where)


def _warn(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.WARNING, code, msg, where)


def _info(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.INFO, code, msg, where)


def _num(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Adapters for the function-style verifier modules.
#
# Each adapter imports its module INSIDE the closure: an optional-dependency
# failure then degrades to a caught `verifier-error` diagnostic instead of
# breaking the import of this dispatcher.
# ---------------------------------------------------------------------------

def _preflight_message(fail, thickness: Optional[float], min_extent: float) -> str:
    """Message text for a kernel-preflight finding, phrased by soundness tier.

    The two PROVEN findings (a shell that leaves no cavity; a feature on a body
    of zero volume) are the ones allowed to reach a model, so they are the ones
    that must not read as ORDERS. They lead with the observation, attach the
    arithmetic as evidence, and carry the kernel's imperative last, marked as a
    suggestion (see verifiers.soundness.observe). A capable model can reason
    from evidence; it can only obey an order -- which is exactly how the 14b
    obeyed a false one and destroyed a correct washer.

    The HEURISTIC findings (RADIUS_TOO_LARGE above all) keep their original text
    verbatim. They never reach a model, and the humans who do read them are
    already used to that wording.
    """
    code = str(fail.code)
    if code == "THICKNESS_TOO_LARGE" and thickness is not None:
        return _soundness.observe(
            fail.message,
            f"the shell offsets every wall inward by the thickness, so two walls "
            f"on opposite sides of an extent of {_num(min_extent)} mm meet when "
            f"2 x thickness >= that extent; here 2 x {_num(thickness)} = "
            f"{_num(2.0 * thickness)} >= {_num(min_extent)}, leaving a cavity of "
            f"zero volume (the operation cannot produce a hollow body)",
            fail.suggestion)
    if code == "ZERO_VOLUME":
        return _soundness.observe(
            fail.message,
            "at least one extent of the body is zero, so the feature has no "
            "material to act on",
            fail.suggestion)
    return f"{fail.message} ({fail.suggestion})"


def _adapter_kernel_preflight() -> FunctionVerifier:
    """`kernel_preflight` -- would this feature blow up the kernel?

    Runs the preflight predicates against the op-stream envelope: a fillet whose
    radius exceeds the part, a shell thicker than the wall, a degenerate body.
    These are the failures OCCT reports as an opaque exception; catching them
    here turns them into a fixable diagnostic.

    Soundness is per-code, not per-verifier (see verifiers.soundness):
    ``preflight-THICKNESS_TOO_LARGE`` and ``preflight-ZERO_VOLUME`` are PROVEN
    theorems and are fed back to the model -- the shell one is the harness's
    only structural advantage over a blind loop. ``preflight-RADIUS_TOO_LARGE``
    is HEURISTIC (it measures a fillet against half the smallest extent of the
    whole body, but a fillet acts on an EDGE, which need not span that extent),
    so it is logged for humans and never instructs the model.
    """

    def applies(state: ModelState) -> bool:
        return state.envelope() is not None

    def run(state: ModelState) -> List[Diagnostic]:
        from harnesscad.core.cisp.ops import Chamfer, Fillet, Shell
        from harnesscad.eval.verifiers.kernel_preflight import (
            BoundingBox, ShapeInfo, check_nonzero_volume,
            preflight_fillet, preflight_shell,
        )

        bb = state.envelope()
        if bb is None:
            return []
        box = BoundingBox(bb[0], bb[1], bb[2], bb[3], bb[4], bb[5])
        shape = ShapeInfo(id="model", bbox=box, volume=box.volume, manifold=True)
        min_extent = box.min_extent()
        diags: List[Diagnostic] = []

        fail = check_nonzero_volume(shape)
        if fail is not None:
            diags.append(_warn("preflight-" + fail.code,
                               _preflight_message(fail, None, min_extent),
                               "model"))

        for i, op in enumerate(state.ops()):
            thickness: Optional[float] = None
            if isinstance(op, Fillet):
                fail = preflight_fillet(shape, float(op.radius))
            elif isinstance(op, Chamfer):
                fail = preflight_fillet(shape, float(op.distance))
            elif isinstance(op, Shell):
                thickness = float(op.thickness)
                fail = preflight_shell(shape, thickness)
            else:
                continue
            if fail is not None:
                diags.append(_warn("preflight-" + fail.code,
                                   _preflight_message(fail, thickness, min_extent),
                                   f"op[{i}]:{type(op).__name__.lower()}"))
        return diags

    return FunctionVerifier("kernel-preflight", LINT, applies, run,
                            "harnesscad.eval.verifiers.kernel_preflight")


def _adapter_shell_envelope() -> FunctionVerifier:
    """`shell-envelope` -- a Shell must not GROW the part.

    A CAD shell removes material: ``bbox_after(shell) <= bbox_before(shell)``.
    Nothing in the fleet asserted that, and the F-rep backend's two-sided Curv
    shell (``|f| - t/2``) dilated every shelled part by ``t/2`` per side -- a
    60x40x20 box shelled at 3 mm measured 63x43x23 with zero diagnostics. The
    backend is fixed (it now hollows inward); this is the backstop that catches
    any backend which regresses, by comparing the MEASURED bbox against the
    op-stream envelope (which is shell-free by construction).

    It only speaks when the comparison is sound: a Shell must be present, and
    every op in the plan must be one the envelope already bounds (sketch
    primitives, extrude, boolean, hole, fillet, chamfer, shell). Ops that
    legitimately push geometry outside the envelope -- mirror, pattern, sweep,
    loft, draft, instances -- make the check abstain rather than lie.
    """

    _BOUNDED = (
        "NewSketch", "AddPoint", "AddLine", "AddCircle", "AddRectangle",
        "Constrain", "Extrude", "Boolean", "Hole", "Fillet", "Chamfer",
        "Shell", "SetParam",
    )

    def _shell_ops(state: ModelState) -> List[Any]:
        return [o for o in state.ops() if type(o).__name__ == "Shell"]

    def applies(state: ModelState) -> bool:
        if not _shell_ops(state):
            return False
        if state.envelope() is None:
            return False
        return all(type(o).__name__ in _BOUNDED for o in state.ops())

    def run(state: ModelState) -> List[Diagnostic]:
        env = state.envelope()
        measured = state.query("measure").get("bbox")
        if env is None or not measured or len(measured) < 3:
            return []
        before = (env[3] - env[0], env[4] - env[1], env[5] - env[2])
        after = tuple(float(v) for v in measured[:3])
        diags: List[Diagnostic] = []
        for axis, b, a in zip(("X", "Y", "Z"), before, after):
            # Mesh extraction is a sampled approximation: allow 0.5% slack.
            if b <= 0.0 or a <= b * 1.005 + 1e-9:
                continue
            diags.append(_err(
                "shell-grew-part",
                f"the shell GREW the part along {axis}: {_num(b)} mm before, "
                f"{_num(a)} mm after. A shell hollows a solid inward and can "
                "only remove material; the outside dimensions are now wrong.",
                "shell"))
        return diags

    return FunctionVerifier("shell-envelope", LINT, applies, run,
                            "harnesscad.eval.verifiers.registry")


def _adapter_plausibility() -> FunctionVerifier:
    """`plausibility` -- degenerate extents, absurd sizes, extreme aspect, fill ratio.

    Fill-ratio / surface-area findings are only trustworthy against a MEASURED
    volume (``backend.query('measure')['volume']``). When the backend has no
    kernel we fall back to the op-stream envelope, and then only the
    size/aspect *warnings* are reported -- the fill-ratio issues would be
    vacuous (the envelope's volume is the bounding box's by construction).
    """

    def applies(state: ModelState) -> bool:
        return state.envelope() is not None

    def run(state: ModelState) -> List[Diagnostic]:
        from harnesscad.eval.verifiers.plausibility import AABB, check_physical_plausibility

        bb = state.envelope()
        if bb is None:
            return []
        box = AABB(bb[0], bb[3], bb[1], bb[4], bb[2], bb[5])
        diags: List[Diagnostic] = []
        ext = box.extents()
        for axis, e in zip("xyz", ext):
            if e <= 0.0:
                diags.append(_warn(
                    "degenerate-extent",
                    f"model envelope has zero {axis}-extent (flat/degenerate body)",
                    "model"))

        measure = state.query("measure")
        measured = measure.get("volume") is not None
        volume = float(measure["volume"]) if measured else float(box.volume)
        area = float(measure.get("surface_area") or
                     2.0 * (ext[0] * ext[1] + ext[1] * ext[2] + ext[0] * ext[2]))
        res = check_physical_plausibility(volume, area, box)
        for w in res.get("warnings", []):
            diags.append(_warn("implausible-shape", str(w), "model"))
        if measured:
            for issue in res.get("issues", []):
                diags.append(_warn("implausible-solid", str(issue), "model"))
        return diags

    return FunctionVerifier("plausibility", PHYSICS, applies, run,
                            "harnesscad.eval.verifiers.plausibility")


def _adapter_clearance_shift() -> FunctionVerifier:
    """`clearance_shift` -- for every clashing pair, the minimal separating move."""

    def applies(state: ModelState) -> bool:
        return len(state.part_bboxes()) >= 2

    def run(state: ModelState) -> List[Diagnostic]:
        from harnesscad.eval.verifiers.clearance_shift import boxes_overlap, suggest_clearance_shift

        boxes = state.part_bboxes()
        diags: List[Diagnostic] = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                (ida, a), (idb, b) = boxes[i], boxes[j]
                if not boxes_overlap(a, b):
                    continue
                fix = suggest_clearance_shift(a, b)
                diags.append(_warn(
                    "clearance-shift",
                    f"parts '{ida}' and '{idb}' overlap; shift '{idb}' by "
                    f"{_num(fix.shift_mm)} along {fix.axis} to clear",
                    f"{ida}|{idb}"))
        return diags

    return FunctionVerifier("clearance-shift", LINT, applies, run,
                            "harnesscad.eval.verifiers.clearance_shift")


def _adapter_standability() -> FunctionVerifier:
    """`standability` -- does the assembly stand up, or does it tip over?"""

    def applies(state: ModelState) -> bool:
        return bool(state.part_bboxes())

    def run(state: ModelState) -> List[Diagnostic]:
        from harnesscad.eval.verifiers.standability import evaluate_standability

        boxes = [b for _, b in state.part_bboxes()]
        z_floor = min(b[2] for b in boxes)
        contacts = []
        for b in boxes:
            if abs(b[2] - z_floor) > 1e-9:
                continue
            contacts += [(b[0], b[1], b[2]), (b[3], b[1], b[2]),
                         (b[3], b[4], b[2]), (b[0], b[4], b[2])]
        if len(contacts) < 3:
            return [_info("standability-skipped",
                          "fewer than 3 ground contacts; standability not evaluated", "model")]
        cx = sum((b[0] + b[3]) / 2.0 for b in boxes) / len(boxes)
        cy = sum((b[1] + b[4]) / 2.0 for b in boxes) / len(boxes)
        cz = sum((b[2] + b[5]) / 2.0 for b in boxes) / len(boxes)
        report = evaluate_standability((cx, cy, cz), contacts)
        if not report.supported:
            return [_warn("tips-over",
                          "centre of mass falls outside the support polygon "
                          "(the model tips over)", "model")]
        if not report.robust:
            return [_warn("marginally-stable",
                          "centre of mass is inside the support polygon but the "
                          "stability margin is marginal", "model")]
        return []

    return FunctionVerifier("standability", PHYSICS, applies, run,
                            "harnesscad.eval.verifiers.standability")


def _adapter_tolerance_stack() -> FunctionVerifier:
    """`tolerance_stack` -- worst-case / RSS stack-up of a declared dimension chain.

    Needs tolerances the op vocabulary does not carry, so it reads them from the
    backend's optional ``tolerance_chain`` projection and INFO-skips otherwise.
    """

    def applies(state: ModelState) -> bool:
        return bool(state.query("tolerance_chain").get("dimensions"))

    def run(state: ModelState) -> List[Diagnostic]:
        from harnesscad.eval.verifiers.tolerance_stack import Dimension, ToleranceChain

        spec = state.query("tolerance_chain")
        chain = ToleranceChain(str(spec.get("name", "chain")))
        for d in spec.get("dimensions", []):
            chain.add(Dimension(
                name=str(d["name"]), nominal=float(d["nominal"]),
                plus=float(d.get("plus", 0.0)), minus=float(d.get("minus", 0.0)),
                direction=float(d.get("direction", 1.0))))
        res = chain.analyze(float(spec["target"]), float(spec["tolerance"]))
        if res.worst_case_passed:
            return []
        return [_warn("tolerance-stack",
                      f"chain '{res.chain_name}' fails worst-case: "
                      f"[{_num(res.worst_case_min)}, {_num(res.worst_case_max)}] "
                      f"vs target {_num(res.target)} +/- {_num(res.tolerance)}",
                      res.chain_name)]

    return FunctionVerifier("tolerance-stack", DOMAIN, applies, run,
                            "harnesscad.eval.verifiers.tolerance_stack")


def _adapter_dimension_qa() -> FunctionVerifier:
    """`dimension_qa` -- measured vs. requested dimensions (needs a target set)."""

    def applies(state: ModelState) -> bool:
        q = state.query("dimension_qa")
        return bool(q.get("nominal")) and bool(q.get("measured"))

    def run(state: ModelState) -> List[Diagnostic]:
        from harnesscad.eval.verifiers.dimension_qa import compare_dimensions

        q = state.query("dimension_qa")
        report = compare_dimensions(dict(q["nominal"]), dict(q["measured"]),
                                    float(q.get("tolerance", 0.1)))
        return [_warn("dimension-out-of-tolerance",
                      f"{c.name}: measured {_num(c.measured)} vs nominal "
                      f"{_num(c.nominal)} (deviation {_num(c.deviation)})", c.name)
                for c in report.failures()]

    return FunctionVerifier("dimension-qa", DOMAIN, applies, run,
                            "harnesscad.eval.verifiers.dimension_qa")


def _adapter_edit_consistency() -> FunctionVerifier:
    """`edit_consistency` -- do the sketch constraints still hold after an edit?"""

    def applies(state: ModelState) -> bool:
        q = state.query("sketch_geometry")
        return bool(q.get("primitives")) and bool(q.get("constraints"))

    def run(state: ModelState) -> List[Diagnostic]:
        from harnesscad.eval.verifiers.edit_consistency import check_constraints

        q = state.query("sketch_geometry")
        report = check_constraints(q["primitives"], q["constraints"])
        return [_warn("constraint-violated",
                      f"constraint '{c.ctype}' on {list(c.refs)} is violated "
                      f"(residual {_num(c.residual)})", c.ctype)
                for c in report.violated()]

    return FunctionVerifier("edit-consistency", DOMAIN, applies, run,
                            "harnesscad.eval.verifiers.edit_consistency")


def _adapter_validity_gate() -> FunctionVerifier:
    """`validity_gate` -- watertight / manifold gate on a tessellated candidate."""

    def applies(state: ModelState) -> bool:
        return bool(state.query("mesh").get("faces"))

    def run(state: ModelState) -> List[Diagnostic]:
        from harnesscad.eval.verifiers.validity_gate import validate_candidate

        mesh = state.query("mesh")
        res = validate_candidate(mesh)
        diags: List[Diagnostic] = []
        if not res.is_valid:
            for reason in res.reasons:
                diags.append(_err("invalid-mesh", str(reason), "mesh"))
        for flag in res.flags:
            diags.append(_warn("mesh-advisory", str(flag), "mesh"))
        return diags

    return FunctionVerifier("validity-gate", DOMAIN, applies, run,
                            "harnesscad.eval.verifiers.validity_gate")


def _adapter_brick_validity() -> FunctionVerifier:
    """`brick_validity` + `brick_buildability` -- voxel/brick assemblies."""

    def applies(state: ModelState) -> bool:
        return bool(state.query("bricks").get("bricks"))

    def run(state: ModelState) -> List[Diagnostic]:
        from harnesscad.eval.verifiers.brick_buildability import is_buildable
        from harnesscad.eval.verifiers.brick_validity import first_unstable_index

        spec = state.query("bricks")
        structure = spec["bricks"]
        diags: List[Diagnostic] = []
        if not is_buildable(structure):
            diags.append(_err("not-buildable",
                              "no support-respecting assembly order exists for this "
                              "brick structure", "bricks"))
        idx = first_unstable_index(structure)
        if idx is not None:
            diags.append(_warn("unstable-brick",
                               f"brick {idx} makes the structure unstable", f"brick[{idx}]"))
        return diags

    return FunctionVerifier("brick-validity", DOMAIN, applies, run,
                            "harnesscad.eval.verifiers.brick_validity")


def _adapter_rim_feasibility() -> FunctionVerifier:
    """`rim_feasibility` -- wheel-rim manufacturability (contours, spokes, symmetry)."""

    def applies(state: ModelState) -> bool:
        return bool(state.query("rim").get("contours"))

    def run(state: ModelState) -> List[Diagnostic]:
        from harnesscad.eval.verifiers.rim_feasibility import validate_design

        spec = state.query("rim")
        report = validate_design(spec["contours"], spec.get("spokes") or [],
                                 dict(spec.get("spec") or {}))
        if report.feasible:
            return []
        return [_warn("rim-infeasible", f"rim design is infeasible: {r}", "rim")
                for r in (report.reasons or ["unspecified"])]

    return FunctionVerifier("rim-feasibility", DOMAIN, applies, run,
                            "harnesscad.eval.verifiers.rim_feasibility")


def _adapter_modal_frequency() -> FunctionVerifier:
    """`modal_frequency` -- first elastic mode vs. the stiffness floor."""

    def applies(state: ModelState) -> bool:
        q = state.query("modal")
        return "mass" in q and "frequency" in q

    def run(state: ModelState) -> List[Diagnostic]:
        from harnesscad.eval.verifiers.modal_frequency import evaluate_wheel

        q = state.query("modal")
        ev = evaluate_wheel(float(q["mass"]), float(q["frequency"]),
                            int(q.get("mode_index", 7)),
                            q.get("stiffness_floor"))
        if ev.meets_stiffness_floor is False:
            return [_warn("below-stiffness-floor",
                          f"{ev.mode_label} stiffness {_num(ev.stiffness)} is below the "
                          f"floor {_num(float(ev.stiffness_floor or 0.0))}", "modal")]
        return []

    return FunctionVerifier("modal-frequency", PHYSICS, applies, run,
                            "harnesscad.eval.verifiers.modal_frequency")


def _adapter_drag_proxy() -> FunctionVerifier:
    """`drag_proxy` -- frontal-area drag surrogate for aero-constrained bodies."""

    def applies(state: ModelState) -> bool:
        q = state.query("aero")
        return bool(q.get("points")) and "cd_max" in q

    def run(state: ModelState) -> List[Diagnostic]:
        from harnesscad.eval.verifiers.drag_proxy import LinearDragModel, car_dimensions_from_points

        q = state.query("aero")
        dims = car_dimensions_from_points([tuple(p) for p in q["points"]])
        model = LinearDragModel(float(q.get("slope", 1.0)), float(q.get("intercept", 0.0)))
        cd = model.cd(dims.frontal_area())
        if cd > float(q["cd_max"]):
            return [_warn("drag-too-high",
                          f"drag proxy Cd {_num(cd)} exceeds the budget "
                          f"{_num(float(q['cd_max']))}", "aero")]
        return []

    return FunctionVerifier("drag-proxy", PHYSICS, applies, run,
                            "harnesscad.eval.verifiers.drag_proxy")


# The adapter table: dotted module -> builder. Explicit *because each adapter is
# bespoke code* -- the discovery of the protocol-native verifiers below is not.
_ADAPTERS = (
    _adapter_clearance_shift,
    _adapter_kernel_preflight,
    _adapter_shell_envelope,
    _adapter_plausibility,
    _adapter_standability,
    _adapter_tolerance_stack,
    _adapter_dimension_qa,
    _adapter_edit_consistency,
    _adapter_validity_gate,
    _adapter_brick_validity,
    _adapter_rim_feasibility,
    _adapter_modal_frequency,
    _adapter_drag_proxy,
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _zero_arg_constructible(cls: type) -> bool:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    for i, (pname, p) in enumerate(sig.parameters.items()):
        if i == 0:  # self
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.default is inspect.Parameter.empty:
            return False
    return True


def _speaks_harness_protocol(cls: type) -> bool:
    if not isinstance(getattr(cls, "name", None), str):
        return False
    check = getattr(cls, "check", None)
    if not callable(check):
        return False
    try:
        params = list(inspect.signature(check).parameters)
    except (TypeError, ValueError):
        return False
    return params[:3] == ["self", "backend", "opdag"]


def _discover_native() -> List[NativeVerifier]:
    """Every protocol-native check class in the verifiers package.

    Found via the static capability registry (no hardcoded module list), then
    imported lazily. Import failures (optional deps) are skipped silently.
    """
    out: List[NativeVerifier] = []
    for entry in capability_registry.find(package="verifiers"):
        if entry.name in _EXCLUDED_MODULES:
            continue
        try:
            mod = capability_registry.load(entry.dotted)
        except Exception:  # noqa: BLE001 - optional deps / broken module
            continue
        tier = _MODULE_TIER.get(entry.name, LINT)
        for sym in entry.symbols:  # AST order == sorted, deterministic
            obj = getattr(mod, sym, None)
            if not inspect.isclass(obj) or obj.__module__ != entry.dotted:
                continue
            if not _speaks_harness_protocol(obj) or not _zero_arg_constructible(obj):
                continue
            try:
                inst = obj()
            except Exception:  # noqa: BLE001
                continue
            out.append(NativeVerifier(inst, tier, entry.dotted))
    return out


_CACHE: Optional[List[Verifier]] = None


def discover(refresh: bool = False) -> List[Verifier]:
    """The whole fleet: protocol-native checks + the function-module adapters.

    Deterministically ordered by (tier, name). Cached -- discovery imports ~30
    modules, so the loop pays that once per process.
    """
    global _CACHE
    if _CACHE is not None and not refresh:
        return list(_CACHE)
    fleet: List[Verifier] = list(_discover_native())
    for build in _ADAPTERS:
        try:
            fleet.append(build())
        except Exception:  # noqa: BLE001 - a broken adapter must not kill the fleet
            continue
    fleet.sort(key=lambda v: (_TIER_ORDER.get(v.tier, len(TIERS)), v.name))
    _CACHE = fleet
    return list(fleet)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def run_all(state: ModelState,
            tiers: Optional[Sequence[str]] = None,
            skip: Iterable[str] = (),
            only: Optional[Iterable[str]] = None,
            verifiers: Optional[Sequence[Verifier]] = None,
            ) -> List[Diagnostic]:
    """Run the fleet and return the aggregated diagnostics.

    - ``tiers``    -- restrict to these tiers (default: everything but CORE,
                      which the session already runs itself).
    - ``skip``     -- verifier names to leave out.
    - ``only``     -- if given, the ONLY verifier names to run.
    - ``verifiers``-- override the fleet (used by tests to inject one).

    Never raises. A verifier that blows up becomes a WARNING diagnostic and the
    run continues, so the fleet can never take the loop down.

    Every diagnostic is STAMPED with the soundness tier of the verifier that
    produced it (``Diagnostic.soundness``). The dispatcher is the only place
    that knows the provenance of a diagnostic, so it is the only place that can
    honestly attribute one; downstream, ``soundness.model_facing`` uses the
    stamp to decide what may be spoken to a model.
    """
    wanted = tuple(tiers) if tiers is not None else tuple(t for t in TIERS if t != CORE)
    skipset = set(skip)
    onlyset = set(only) if only is not None else None
    fleet = list(verifiers) if verifiers is not None else discover()

    diags: List[Diagnostic] = []
    for v in fleet:
        name = getattr(v, "name", type(v).__name__)
        if getattr(v, "tier", LINT) not in wanted:
            continue
        if name in skipset:
            continue
        if onlyset is not None and name not in onlyset:
            continue
        try:
            if not v.applies_to(state):
                continue
            produced = v.check(state)
        except Exception as exc:  # noqa: BLE001 - THE point of this dispatcher
            crash = _warn("verifier-error",
                          f"verifier '{name}' raised {type(exc).__name__}: {exc}",
                          name)
            # A crashed verifier tells the model nothing it can act on.
            crash.soundness = _soundness.HEURISTIC
            diags.append(crash)
            continue
        for d in produced or []:
            if isinstance(d, Diagnostic):
                diags.append(_soundness.stamp(d, name))
    return diags


def run_report(state: ModelState, **kwargs) -> VerifyReport:
    """`run_all` wrapped in the harness's VerifyReport (ok == no ERROR)."""
    return VerifyReport(run_all(state, **kwargs))


def conformance(backend: Any, opdag: Any, title: str = "HarnessCAD Conformance Certificate"):
    """A signed conformance certificate covering the whole discovered fleet."""
    from harnesscad.eval.verifiers.conformance_report import ConformanceReport

    state = model_state(backend, opdag)
    natives = [v.inner for v in discover() if isinstance(v, NativeVerifier)]
    report = ConformanceReport.from_verifiers(backend, opdag, natives, title=title)
    adapted = run_all(state, verifiers=[v for v in discover()
                                        if isinstance(v, FunctionVerifier)],
                      tiers=TIERS)
    counts: Dict[str, int] = {}
    for d in adapted:
        counts[d.severity.value] = counts.get(d.severity.value, 0) + 1
    report.measurements["fleet"] = {
        "verifiers": len(discover()),
        "adapted_diagnostics": dict(sorted(counts.items())),
    }
    return report
