"""Pre-execution plan-feasibility precheck — a *symbolic* op-plan linter.

Every other verifier reads *built geometry* through a backend. This one runs
*before* the backend does: it walks the CISP op plan (``opdag.ops()`` or a raw
op list) as pure symbols and rejects plans that cannot possibly succeed, so the
loop never pays for a kernel build that was doomed from the start.

It is the cheap gate the blueprint's block-and-correct loop wants up front
(sec.21 "recycling"; sec.18 sequencing): catch the structurally impossible
before geometry, and hand back a clear reason per issue.

What it flags (each an ERROR ``infeasible-plan`` with a specific reason):

  * negative / zero dimensions — extrude distance 0, non-positive radius,
    diameter, thickness, chamfer setback, blind-hole depth, pattern spacing.
  * an :class:`ops.Extrude` (or Revolve / Loft / Sweep) of a sketch that has no
    profile entities — an *empty* sketch cannot become a solid.
  * a dangling reference — an op naming a sketch that was never created.
  * a solid-consuming op before any solid exists — Fillet / Chamfer / Shell /
    Draft / Hole-on-a-face / Mirror / Pattern with no prior solid, or a Boolean
    with fewer than two solids.
  * a hole that does not fit the material *around* it: its disc spans the whole
    IN-PLANE extent of the stock at its position, so no wall survives on either
    side. (The hole diameter is an in-plane quantity and is *never* compared
    against the extrude depth — those are orthogonal; see the note below.)
  * a shell whose wall thickness meets or exceeds the available stock (nothing
    left) or falls below ``rules.min_wall``.
  * a pattern whose count is below ``rules.min_pattern_count`` (default 2).
  * mutually exclusive / duplicate mates — two mates coupling the same pair.

And two WARNINGs about ambiguous sketch consumption:

  * ``ambiguous-sketch`` — a sketch holding more than one closed profile is
    extruded: which profile becomes the solid is unspecified.
  * ``sketch-reused`` — one sketch is consumed by more than one solid-producing
    op: almost certainly an unintended double extrude.

.. note:: **Deleted rule (unsound).** An earlier revision flagged a hole whose
   *diameter* met or exceeded the base extrude *distance* ("plate/stock wall").
   Those are orthogonal quantities: the diameter lies in the sketch plane, the
   extrude distance along Z. An 80 mm disc 8 mm thick with a 30 mm bore is a
   washer — it builds correctly and the rule called it infeasible. The rule is
   gone; a hole is now only rejected against the in-plane material around it,
   and when the in-plane extent is not knowable the rule does not fire at all.

Purely symbolic: it inspects ops only (no backend query, no geometry, no
mutation) so it can run pre-execution and is fully deterministic. ``check`` is
the :class:`verify.Verifier` entrypoint; ``check_ops`` is the backend-free core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from harnesscad.core.cisp.ops import (
    Op, NewSketch, AddPoint, AddLine, AddCircle, AddRectangle,
    Constrain, Extrude, Fillet, Boolean,
    Revolve, Chamfer, Hole, Shell, Draft,
    Loft, Sweep, LinearPattern, CircularPattern, Mirror,
    AddInstance, Mate, SetParam,
    CONSTRAINT_DOF,
)
from harnesscad.eval.verifiers.verify import Diagnostic, Severity, VerifyReport


# --------------------------------------------------------------------------- #
# Rules (configurable feasibility limits)
# --------------------------------------------------------------------------- #
@dataclass
class PrecheckRules:
    """Configurable limits for the plan-feasibility precheck (millimetres).

    ``min_wall``        — thinnest producible wall; a shell thinner than this is
        infeasible.
    ``wall_thickness``  — explicit stock thickness available to a *shell*. When
        ``None`` the precheck infers it from the base extrude distance. It is
        NOT used for holes: a hole's diameter is an in-plane dimension and the
        stock thickness is along the extrude axis.
    ``min_pattern_count`` — smallest meaningful pattern count (a pattern of one
        is a no-op).
    """

    min_wall: float = 0.5
    wall_thickness: Optional[float] = None
    min_pattern_count: int = 2

    def to_dict(self) -> dict:
        return {
            "min_wall": self.min_wall,
            "wall_thickness": self.wall_thickness,
            "min_pattern_count": self.min_pattern_count,
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PrecheckRules":
        d = d or {}
        defaults = cls()
        wt = d.get("wall_thickness", defaults.wall_thickness)
        return cls(
            min_wall=float(d.get("min_wall", defaults.min_wall)),
            wall_thickness=(None if wt is None else float(wt)),
            min_pattern_count=int(d.get("min_pattern_count", defaults.min_pattern_count)),
        )


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
class PrecheckCheck:
    """A :class:`verify.Verifier` (``name='precheck'``) plan linter.

    ``check(backend, opdag)`` reads ``opdag.ops()`` and returns a
    :class:`verify.VerifyReport`; every infeasibility is an ERROR
    ``infeasible-plan`` (so the report is ``ok == False``) carrying a specific
    reason. ``check_ops`` runs the same analysis on a raw op list, backend-free.
    """

    name = "precheck"

    def __init__(self, rules: Optional[PrecheckRules] = None) -> None:
        self.rules = rules or PrecheckRules()

    def check(self, backend, opdag) -> VerifyReport:
        ops = _extract_ops(opdag)
        if ops is None:
            return VerifyReport([_info(
                "precheck-skipped",
                "plan precheck skipped: no op plan available (opdag exposes no "
                "ops() and no op list was provided).")])
        return self.check_ops(ops)

    def check_ops(self, ops: List[Op]) -> VerifyReport:
        """Symbolically lint a raw op plan and return a :class:`VerifyReport`."""
        return VerifyReport(_PlanState(self.rules).run(list(ops or [])))


# --------------------------------------------------------------------------- #
# Symbolic plan walker
# --------------------------------------------------------------------------- #
class _PlanState:
    """Mutable symbolic state accumulated while walking the op plan.

    Tracks only what feasibility needs: which sketches exist, whether they carry
    a profile and how many *closed* profiles that is, the in-plane bounds of
    every sketch (for the hole rule), how many solids have been produced (for
    booleans), the stock thickness along the extrude axis (for shells), which
    sketches have already been consumed by a solid-producing op, and which mate
    pairs have already been coupled.
    """

    def __init__(self, rules: PrecheckRules) -> None:
        self.rules = rules
        self.sketch_entities: dict = {}   # sid -> entity count
        self.sketch_profiles: dict = {}   # sid -> closed-profile count
        self.sketch_bounds: dict = {}     # sid -> [minx, miny, maxx, maxy]
        self.sketch_consumers: dict = {}  # sid -> [where, ...] (solid-producing)
        self._sk_n = 0
        self.n_solids = 0
        self.n_solidify = 0                # solid-producing ops seen
        self.wall: Optional[float] = None  # stock thickness along the extrude axis
        # In-plane bounds of the stock: the union of the bounds of every sketch
        # that has actually been turned into a solid. ``None`` until one has.
        self.stock_planar: Optional[List[float]] = None
        self.mate_pairs: set = set()
        self.diags: List[Diagnostic] = []

    @property
    def have_solid(self) -> bool:
        return self.n_solids > 0

    def run(self, ops: List[Op]) -> List[Diagnostic]:
        for i, op in enumerate(ops):
            self._visit(i, op)
        return self.diags

    # -- dispatch ----------------------------------------------------------- #
    def _visit(self, i: int, op: Op) -> None:
        where = f"op[{i}]"
        if isinstance(op, NewSketch):
            self._sk_n += 1
            sid = f"sk{self._sk_n}"
            self.sketch_entities[sid] = 0
            self.sketch_profiles[sid] = 0
        elif isinstance(op, AddPoint):
            self._add_entity(op.sketch, where)
            self._grow(op.sketch, op.x, op.y, op.x, op.y)
        elif isinstance(op, AddLine):
            self._add_entity(op.sketch, where)
            self._grow(op.sketch, min(op.x1, op.x2), min(op.y1, op.y2),
                       max(op.x1, op.x2), max(op.y1, op.y2))
        elif isinstance(op, AddCircle):
            if op.r <= 0:
                self._bad(where, f"circle radius must be > 0 (got {op.r:g}).")
            self._add_entity(op.sketch, where, closed_profile=True)
            self._grow(op.sketch, op.cx - abs(op.r), op.cy - abs(op.r),
                       op.cx + abs(op.r), op.cy + abs(op.r))
        elif isinstance(op, AddRectangle):
            if op.w <= 0 or op.h <= 0:
                self._bad(where, f"rectangle w/h must be > 0 (got {op.w:g}x{op.h:g}).")
            self._add_entity(op.sketch, where, closed_profile=True)
            self._grow(op.sketch, min(op.x, op.x + op.w), min(op.y, op.y + op.h),
                       max(op.x, op.x + op.w), max(op.y, op.y + op.h))
        elif isinstance(op, Constrain):
            if op.kind not in CONSTRAINT_DOF:
                self._bad(where, f"unknown constraint kind '{op.kind}'.")
        elif isinstance(op, Extrude):
            self._solidify(op.sketch, "extrude", where,
                           zero_dim=(op.distance == 0),
                           zero_msg="extrude distance must be non-zero.",
                           set_wall=abs(op.distance))
        elif isinstance(op, Revolve):
            self._solidify(op.sketch, "revolve", where,
                           zero_dim=(op.angle == 0),
                           zero_msg="revolve angle must be non-zero.")
        elif isinstance(op, Loft):
            self._loft(op, where)
        elif isinstance(op, Sweep):
            self._sweep(op, where)
        elif isinstance(op, Fillet):
            self._require_solid("fillet", where)
            if op.radius <= 0:
                self._bad(where, f"fillet radius must be > 0 (got {op.radius:g}).")
        elif isinstance(op, Chamfer):
            self._require_solid("chamfer", where)
            if op.distance <= 0:
                self._bad(where, f"chamfer distance must be > 0 (got {op.distance:g}).")
        elif isinstance(op, Boolean):
            if op.kind not in ("union", "cut", "intersect"):
                self._bad(where, f"unknown boolean kind '{op.kind}'.")
            if self.n_solids < 2:
                self._bad(where,
                          f"boolean '{op.kind}' needs two solids but only "
                          f"{self.n_solids} exist so far.")
        elif isinstance(op, Hole):
            self._hole(op, where)
        elif isinstance(op, Shell):
            self._shell(op, where)
        elif isinstance(op, Draft):
            self._require_solid("draft", where)
            if not op.neutral_plane:
                self._bad(where, "draft requires a neutral_plane.")
        elif isinstance(op, Mirror):
            self._require_solid("mirror", where)
        elif isinstance(op, (LinearPattern, CircularPattern)):
            self._pattern(op, where)
        elif isinstance(op, Mate):
            self._mate(op, where)
        # AddPoint/AddInstance/SetParam need no feasibility gate here.

    # -- helpers ------------------------------------------------------------ #
    def _add_entity(self, sketch: str, where: str,
                    closed_profile: bool = False) -> None:
        if sketch not in self.sketch_entities:
            self._bad(where, f"references unknown sketch '{sketch}'.")
            return
        self.sketch_entities[sketch] += 1
        if closed_profile:
            self.sketch_profiles[sketch] = self.sketch_profiles.get(sketch, 0) + 1

    def _grow(self, sketch: str, x0: float, y0: float, x1: float, y1: float) -> None:
        """Accumulate the in-plane AABB of a sketch (unknown sketches ignored)."""
        if sketch not in self.sketch_entities:
            return
        box = self.sketch_bounds.get(sketch)
        if box is None:
            self.sketch_bounds[sketch] = [float(x0), float(y0), float(x1), float(y1)]
        else:
            box[0] = min(box[0], float(x0))
            box[1] = min(box[1], float(y0))
            box[2] = max(box[2], float(x1))
            box[3] = max(box[3], float(y1))

    def _sketch_ok(self, sketch: str, feature: str, where: str) -> bool:
        if sketch not in self.sketch_entities:
            self._bad(where, f"{feature} references unknown sketch '{sketch}'.")
            return False
        if self.sketch_entities[sketch] <= 0:
            self._bad(where, f"{feature} of empty sketch '{sketch}' (no profile "
                             "entities) cannot produce a solid.")
            return False
        return True

    def _consume(self, sketch: str, feature: str, where: str) -> None:
        """Book a sketch as consumed by a solid-producing op, and warn on the
        two ambiguities that produce silently-wrong solids:

        * more than one closed profile in the consumed sketch — which loop is
          the profile is unspecified (the backends union them);
        * the same sketch consumed twice — the second solid is a duplicate at a
          different depth, almost never intended.
        """
        prior = self.sketch_consumers.setdefault(sketch, [])
        if prior:
            self._warn(where, "sketch-reused",
                       f"sketch '{sketch}' is consumed by more than one "
                       f"solid-producing op ({', '.join(prior)} and {where}); "
                       "the same profile becomes two solids at different depths; "
                       "almost certainly unintended.")
        prior.append(where)
        n_profiles = self.sketch_profiles.get(sketch, 0)
        if n_profiles > 1:
            self._warn(where, "ambiguous-sketch",
                       f"{feature} of sketch '{sketch}', which holds "
                       f"{n_profiles} closed profiles; which profile is the "
                       "solid is ambiguous. Put one profile per sketch.")

    def _stock_grow(self, sketch: str) -> None:
        """Fold a solidified sketch's in-plane AABB into the stock footprint."""
        box = self.sketch_bounds.get(sketch)
        if box is None:
            return
        if self.stock_planar is None:
            self.stock_planar = list(box)
        else:
            s = self.stock_planar
            s[0] = min(s[0], box[0])
            s[1] = min(s[1], box[1])
            s[2] = max(s[2], box[2])
            s[3] = max(s[3], box[3])

    def _solidify(self, sketch: str, feature: str, where: str,
                  zero_dim: bool, zero_msg: str,
                  set_wall: Optional[float] = None) -> None:
        ok = self._sketch_ok(sketch, feature, where)
        if zero_dim:
            self._bad(where, zero_msg)
        if ok and not zero_dim:
            self._consume(sketch, feature, where)
            self.n_solids += 1
            self.n_solidify += 1
            self._stock_grow(sketch)
            if set_wall is not None and set_wall > 0:
                self.wall = set_wall

    def _loft(self, op: Loft, where: str) -> None:
        if len(op.sketches) < 2:
            self._bad(where, "loft requires at least two profile sketches.")
            return
        if all(self._sketch_ok(s, "loft", where) for s in op.sketches):
            for s in op.sketches:
                self._consume(s, "loft", where)
                self._stock_grow(s)
            self.n_solids += 1
            self.n_solidify += 1

    def _sweep(self, op: Sweep, where: str) -> None:
        ok = self._sketch_ok(op.sketch, "sweep", where)
        ok = self._sketch_ok(op.path, "sweep path", where) and ok
        if ok:
            self._consume(op.sketch, "sweep", where)
            self.n_solids += 1
            self.n_solidify += 1

    def _require_solid(self, feature: str, where: str) -> bool:
        if not self.have_solid:
            self._bad(where, f"{feature} requires an existing solid, but none "
                             "has been created yet.")
            return False
        return True

    def _current_wall(self) -> Optional[float]:
        if self.rules.wall_thickness is not None:
            return self.rules.wall_thickness
        return self.wall

    def _hole(self, op: Hole, where: str) -> None:
        if op.diameter <= 0:
            self._bad(where, f"hole diameter must be > 0 (got {op.diameter:g}).")
        if not op.through and (op.depth is None or op.depth <= 0):
            self._bad(where, "blind hole requires depth > 0.")
        if op.kind not in ("simple", "counterbore", "countersink"):
            self._bad(where, f"unknown hole kind '{op.kind}'.")
        ref = str(op.face_or_sketch)
        # A face-based hole needs a prior solid; a sketch-datum hole does not.
        if not ref.startswith("sk") and not self.have_solid:
            self._bad(where, "hole on a face requires an existing solid, but "
                             "none has been created yet.")
        self._hole_in_plane(op, where)

    def _hole_in_plane(self, op: Hole, where: str) -> None:
        """Does the hole fit the material AROUND it, in the sketch plane?

        A hole's diameter is an in-plane dimension, so the only material it can
        consume is the in-plane extent of the stock — never the extrude depth
        (that is what the hole is drilled *along*). Two soundly-decidable cases:

        * the disc spans the stock from edge to edge along an in-plane axis:
          nothing is left on either side, the plate is severed — ERROR;
        * the disc crosses the stock boundary while its centre is inside: the
          "hole" is an open notch, remaining wall <= 0 — WARNING.

        When the stock's in-plane extent is not knowable (no solidified sketch
        with bounds), NOTHING fires: a false positive here is far more costly
        than a missed check.
        """
        box = self.stock_planar
        if box is None or op.diameter <= 0 or self.n_solidify != 1:
            # n_solidify != 1: several bodies, so the hole's (x, y) cannot be
            # placed in a single unambiguous stock frame. Stay silent.
            return
        r = op.diameter / 2.0
        cx, cy = float(op.x), float(op.y)
        minx, miny, maxx, maxy = box
        spans_x = (cx - r <= minx) and (cx + r >= maxx)
        spans_y = (cy - r <= miny) and (cy + r >= maxy)
        if spans_x or spans_y:
            axis, extent = ("X", maxx - minx) if spans_x else ("Y", maxy - miny)
            self._bad(where,
                      f"hole diameter {op.diameter:g} mm spans the whole "
                      f"in-plane {axis} extent ({extent:g} mm) of the stock at "
                      f"({cx:g}, {cy:g}); no wall remains on either side. "
                      "(The extrude depth is irrelevant here: a hole is bounded "
                      "by the material around it, not by the plate thickness.)")
            return
        inside = minx <= cx <= maxx and miny <= cy <= maxy
        breakout = min(cx - r - minx, cy - r - miny, maxx - (cx + r), maxy - (cy + r))
        if inside and breakout < 0.0:
            self._warn(where, "thin-wall",
                       f"hole diameter {op.diameter:g} mm at ({cx:g}, {cy:g}) "
                       f"crosses the stock boundary by {-breakout:g} mm; it "
                       "leaves no wall there and cuts an open notch, not a hole.")

    def _shell(self, op: Shell, where: str) -> None:
        self._require_solid("shell", where)
        if op.thickness <= 0:
            self._bad(where, f"shell thickness must be > 0 (got {op.thickness:g}).")
            return
        if op.thickness < self.rules.min_wall:
            self._bad(where,
                      f"shell wall {op.thickness:g} mm is below the minimum "
                      f"manufacturable wall {self.rules.min_wall:g} mm.")
        wall = self._current_wall()
        if wall is not None and wall > 0 and op.thickness >= wall:
            self._bad(where,
                      f"shell thickness {op.thickness:g} mm >= available stock "
                      f"{wall:g} mm; the wall consumes the whole solid.")

    def _pattern(self, op, where: str) -> None:
        self._require_solid("pattern", where)
        if op.count < self.rules.min_pattern_count:
            self._bad(where,
                      f"pattern count {op.count} is below the minimum "
                      f"{self.rules.min_pattern_count}; a pattern must replicate "
                      "at least twice.")
        if isinstance(op, LinearPattern) and op.spacing <= 0:
            self._bad(where, f"linear pattern spacing must be > 0 (got {op.spacing:g}).")

    def _mate(self, op: Mate, where: str) -> None:
        if op.a and op.b:
            pair = frozenset((op.a, op.b))
            if pair in self.mate_pairs:
                self._bad(where,
                          f"mutually exclusive mates: '{op.a}' and '{op.b}' are "
                          "already coupled; a second mate over-constrains the "
                          "same pair.")
            else:
                self.mate_pairs.add(pair)

    def _bad(self, where: str, reason: str) -> None:
        self.diags.append(Diagnostic(
            Severity.ERROR, "infeasible-plan", reason, where))

    def _warn(self, where: str, code: str, reason: str) -> None:
        self.diags.append(Diagnostic(Severity.WARNING, code, reason, where))


# --------------------------------------------------------------------------- #
# Op extraction
# --------------------------------------------------------------------------- #
def _extract_ops(opdag) -> Optional[List[Op]]:
    """Pull the op list from an OpDAG, a raw list, or anything ops()-shaped."""
    if opdag is None:
        return None
    if isinstance(opdag, (list, tuple)):
        return list(opdag)
    ops_attr = getattr(opdag, "ops", None)
    if callable(ops_attr):
        try:
            return list(ops_attr())
        except Exception:  # noqa: BLE001 - degrade rather than crash
            return None
    return None


# --------------------------------------------------------------------------- #
# Wiring helper
# --------------------------------------------------------------------------- #
def with_precheck(verifiers, rules: Optional[PrecheckRules] = None) -> List:
    """Return a new verifier list with a :class:`PrecheckCheck` appended
    (mirrors :func:`interference.with_interference`)."""
    return list(verifiers) + [PrecheckCheck(rules)]


def _info(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.INFO, code, msg, where)
