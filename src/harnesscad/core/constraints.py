"""2D sketch constraint solving — a real DOF model for the harness.

Two complementary pieces live here:

1. :class:`ConstraintGraph` — a dependency-free, genuine degrees-of-freedom
   analysis over the CISP abstract sketch model (the ``PRIMITIVE_DOF`` /
   ``CONSTRAINT_DOF`` conventions from :mod:`cisp.ops`). Unlike a naive additive
   heuristic (``dof -= weight`` and "over-constrained iff dof < 0"), it performs
   a rank-style analysis: entities are pooled into connected components via
   union-find, each constraint can only remove DOF that is actually *available*
   in the component it couples, and any constraint that removes less than its
   nominal weight is flagged **redundant**. That lets it classify a sketch as
   over-constrained because of a redundant/​conflicting constraint even when the
   naive net DOF is still >= 0. This is the model the CadQuery backend uses so
   ``query('sketch_dof')`` reflects a real analysis while staying consistent with
   the harness's abstract DOF conventions. On top of the classification it offers
   a *graceful relaxation* path (:meth:`ConstraintGraph.relax` /
   :meth:`ConstraintGraph.conflicts` / ``analyze(relax=True)``): instead of hard-
   failing an over-constrained sketch it returns a best-effort
   :class:`RelaxationResult` — the conflicting-constraint set plus a minimal
   drop suggestion that restores consistency — and, for under-constrained
   sketches, the free DOF and which entities are still unpinned. This reuses the
   rank/redundancy analysis and is a documented heuristic, not a physical solver.

2. :class:`SolveSpaceSketch` — a thin wrapper around the SolveSpace solver
   (``python-solvespace``, an OPTIONAL extra: ``pip install
   harnesscad[constraints]``). It builds *real* 2D geometry (points / lines /
   circles) and geometric + dimensional constraints, actually SOLVES the system,
   and reports the residual DOF and an over/under/well-constrained status from
   the solver's own result flag. SolveSpace is imported LAZILY, so this module
   imports cleanly whether or not the extra is installed; use
   :func:`solvespace_available` to probe.

The two share the same :class:`SketchStatus` vocabulary. The abstract graph is
always available (stdlib only); the SolveSpace wrapper is the real-geometry
upgrade for callers that build concrete entities.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from harnesscad.core.cisp.ops import CONSTRAINT_DOF, PRIMITIVE_DOF


class SketchStatus(str, Enum):
    """Classification of a sketch's constraint state."""

    EMPTY = "empty"
    UNDER = "under-constrained"
    WELL = "well-constrained"
    OVER = "over-constrained"


# ---------------------------------------------------------------------------
# Abstract DOF analysis (stdlib only)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConstraintRef:
    """A stable, human-readable reference to one constraint in a graph.

    ``index`` is the position returned by :meth:`ConstraintGraph.add_constraint`,
    so a caller can drop a suggested constraint by rebuilding the sketch without
    that index. ``kind`` / ``entities`` / ``value`` are carried for logging and
    for a repair loop that wants to explain *what* it is proposing to drop.
    """

    index: int
    kind: str
    entities: Tuple[str, ...]
    value: Optional[float] = None


@dataclass(frozen=True)
class RelaxationResult:
    """Best-effort resolution of an over-/under-constrained sketch.

    This is a graph-rank / least-squares *heuristic*, not a physical solver: the
    :class:`ConstraintGraph` rank analysis (union-find DOF pooling plus
    redundant-constraint detection) is reused to decide, without moving any real
    geometry, whether a consistent placement exists and — when it does not —
    which constraints are implicated and a minimal set to drop to restore
    consistency. ``residual`` is the number of unsatisfiable (redundant) DOF,
    i.e. a rank-deficiency count standing in for a true least-squares residual.

    Fields
    ------
    feasible
        ``True`` when the sketch already admits a consistent placement (status is
        EMPTY, WELL or UNDER). ``False`` for OVER-constrained sketches; applying
        ``dropped_suggestions`` is expected to make it feasible.
    residual
        Heuristic residual: the redundant DOF count (``0.0`` when feasible).
    status
        The sketch's :class:`SketchStatus`.
    conflicting_constraints
        The redundant / over-removing constraints implicated in the conflict
        (empty when feasible). Ranked worst-first (least effective binding).
    dropped_suggestions
        A minimal set of constraints whose removal restores consistency (empty
        when feasible). A subset of ``conflicting_constraints``.
    free_dof
        Degrees of freedom still unbound after the constraints that actually
        bind (>= 0). Non-zero for an UNDER-constrained sketch.
    unpinned_entities
        Entity ids whose connected component still has free DOF — the entities a
        repair loop would need to pin to reach a well-constrained sketch.
    """

    feasible: bool
    residual: float
    status: SketchStatus
    conflicting_constraints: List[ConstraintRef] = field(default_factory=list)
    dropped_suggestions: List[ConstraintRef] = field(default_factory=list)
    free_dof: int = 0
    unpinned_entities: List[str] = field(default_factory=list)

    @property
    def conflict_indices(self) -> List[int]:
        """Indices of the conflicting constraints (for programmatic dropping)."""
        return [r.index for r in self.conflicting_constraints]

    @property
    def dropped_indices(self) -> List[int]:
        """Indices of the suggested-to-drop constraints."""
        return [r.index for r in self.dropped_suggestions]


@dataclass(frozen=True)
class _CRec:
    kind: str
    entities: Tuple[str, ...]
    weight: int
    value: Optional[float] = None


@dataclass(frozen=True)
class DofAnalysis:
    """Result of :meth:`ConstraintGraph.analyze`.

    ``residual_dof`` is the signed net DOF (``entity_dof - constraint_dof``): it
    matches the harness's historical additive number and can go negative when a
    sketch is over-determined. ``effective_removed`` is the DOF the constraints
    *actually* remove given availability (rank), and ``redundant_dof`` is the
    excess (``constraint_dof - effective_removed``) — a genuine, non-additive
    signal of redundancy.
    """

    entity_dof: int
    constraint_dof: int
    residual_dof: int
    effective_removed: int
    redundant_dof: int
    redundant_constraints: List[int]
    status: SketchStatus
    relaxation: Optional[RelaxationResult] = None

    @property
    def well_constrained(self) -> bool:
        return self.status is SketchStatus.WELL

    @property
    def over_constrained(self) -> bool:
        return self.status is SketchStatus.OVER

    @property
    def under_constrained(self) -> bool:
        return self.status is SketchStatus.UNDER

    @property
    def free_dof(self) -> int:
        """DOF still free after the constraints that actually bind (>= 0)."""
        return self.entity_dof - self.effective_removed


class ConstraintGraph:
    """A genuine per-entity / per-constraint DOF model for one sketch.

    Entities contribute ``PRIMITIVE_DOF[kind]`` degrees of freedom; constraints
    remove up to ``CONSTRAINT_DOF[kind]``, but only DOF that is available in the
    connected component they couple. Redundant constraints are detected as those
    that remove less than their nominal weight.
    """

    def __init__(self) -> None:
        self._kinds: Dict[str, str] = {}          # eid -> entity kind
        self._entity_dof: Dict[str, int] = {}     # eid -> dof
        self._constraints: List[_CRec] = []

    # -- construction -------------------------------------------------------
    def add_entity(self, eid: str, kind: str) -> str:
        if kind not in PRIMITIVE_DOF:
            raise ValueError(f"unknown entity kind '{kind}'")
        if eid in self._kinds:
            raise ValueError(f"duplicate entity '{eid}'")
        self._kinds[eid] = kind
        self._entity_dof[eid] = PRIMITIVE_DOF[kind]
        return eid

    def add_constraint(
        self,
        kind: str,
        a: str,
        b: Optional[str] = None,
        value: Optional[float] = None,
    ) -> int:
        if kind not in CONSTRAINT_DOF:
            raise ValueError(f"unknown constraint kind '{kind}'")
        ents = tuple(e for e in (a, b) if e is not None)
        for e in ents:
            if e not in self._kinds:
                raise KeyError(f"unknown entity '{e}'")
        self._constraints.append(_CRec(kind, ents, CONSTRAINT_DOF[kind], value))
        return len(self._constraints) - 1

    # -- totals -------------------------------------------------------------
    def total_entity_dof(self) -> int:
        return sum(self._entity_dof.values())

    def total_constraint_dof(self) -> int:
        return sum(c.weight for c in self._constraints)

    def residual_dof(self) -> int:
        """Signed net DOF (entity - constraint); may be negative if over-determined."""
        return self.total_entity_dof() - self.total_constraint_dof()

    @property
    def entity_count(self) -> int:
        return len(self._kinds)

    @property
    def constraint_count(self) -> int:
        return len(self._constraints)

    # -- analysis -----------------------------------------------------------
    def _rank_subset(self, active: Optional[Set[int]] = None) -> Dict[str, object]:
        """Union-find rank analysis over a subset of constraints.

        ``active`` restricts which constraint indices participate (``None`` = all).
        Returns the effective removed DOF, the redundant (over-removing) indices,
        the per-constraint effective removal, and — per entity — the free DOF left
        in its connected component (used to report unpinned entities). This is the
        single rank kernel shared by :meth:`analyze`, :meth:`relax` and the drop
        search, so all of them agree on the same non-additive DOF model.
        """
        parent: Dict[str, str] = {e: e for e in self._kinds}
        cap: Dict[str, int] = dict(self._entity_dof)   # removable capacity per root
        removed: Dict[str, int] = {e: 0 for e in self._kinds}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> str:
            rx, ry = find(x), find(y)
            if rx == ry:
                return rx
            parent[ry] = rx
            cap[rx] += cap[ry]
            removed[rx] += removed[ry]
            return rx

        redundant: List[int] = []
        eff_per: Dict[int, int] = {}
        effective_removed = 0
        for i, c in enumerate(self._constraints):
            if active is not None and i not in active:
                continue
            if not c.entities:
                eff_per[i] = 0
                if c.weight > 0:
                    redundant.append(i)
                continue
            root = find(c.entities[0])
            for e in c.entities[1:]:
                root = union(root, e)
            root = find(root)
            available = cap[root] - removed[root]
            eff = min(c.weight, max(available, 0))
            removed[root] += eff
            effective_removed += eff
            eff_per[i] = eff
            if eff < c.weight:
                redundant.append(i)

        # entities whose connected component still has slack are "unpinned".
        unpinned = sorted(
            e for e in self._kinds
            if cap[find(e)] - removed[find(e)] > 0
        )
        return {
            "effective_removed": effective_removed,
            "redundant": redundant,
            "eff_per": eff_per,
            "unpinned": unpinned,
        }

    def analyze(self, relax: bool = False) -> DofAnalysis:
        """Rank-style DOF analysis of the sketch.

        With ``relax=True`` the returned :class:`DofAnalysis` also carries a
        :attr:`DofAnalysis.relaxation` (see :meth:`relax`) so a caller can get the
        classification *and* the best-effort conflict set in one call. The
        ``relax=False`` default is byte-for-byte the historical result, so
        :meth:`residual_dof`, :meth:`status` and any digest over the core fields
        are unchanged.
        """
        rank = self._rank_subset(None)
        effective_removed = int(rank["effective_removed"])
        redundant = list(rank["redundant"])  # type: ignore[arg-type]

        entity_dof = self.total_entity_dof()
        constraint_dof = self.total_constraint_dof()
        residual = entity_dof - constraint_dof
        redundant_dof = constraint_dof - effective_removed
        free = entity_dof - effective_removed

        if not self._kinds:
            status = SketchStatus.EMPTY
        elif redundant_dof > 0:
            status = SketchStatus.OVER
        elif free > 0:
            status = SketchStatus.UNDER
        else:
            status = SketchStatus.WELL

        result = DofAnalysis(
            entity_dof=entity_dof,
            constraint_dof=constraint_dof,
            residual_dof=residual,
            effective_removed=effective_removed,
            redundant_dof=redundant_dof,
            redundant_constraints=redundant,
            status=status,
        )
        if relax:
            result = replace(result, relaxation=self.relax())
        return result

    def status(self) -> SketchStatus:
        return self.analyze().status

    # -- relaxation / best-effort resolution --------------------------------
    def _ref(self, i: int) -> ConstraintRef:
        c = self._constraints[i]
        return ConstraintRef(index=i, kind=c.kind, entities=c.entities, value=c.value)

    def _drop_set(self) -> List[int]:
        """Greedy minimal set of constraint indices to drop to remove conflict.

        Iteratively drops the least-effective (most wasteful, then most-recently
        added) redundant constraint and re-ranks, until no redundancy remains.
        This is the rank-heuristic analogue of removing rows from an
        over-determined least-squares system until it is consistent; it is not
        guaranteed globally minimal but is minimal for the common cases (a single
        excess dimension, a duplicated/​conflicting constraint).
        """
        active: Set[int] = set(range(len(self._constraints)))
        dropped: List[int] = []
        while True:
            rank = self._rank_subset(active)
            redundant = [i for i in rank["redundant"] if i in active]  # type: ignore[operator]
            if not redundant:
                break
            eff_per = rank["eff_per"]  # type: ignore[assignment]
            # worst first: smallest effective binding, ties broken by latest index.
            choice = min(redundant, key=lambda i: (eff_per[i], -i))
            active.discard(choice)
            dropped.append(choice)
            if len(dropped) >= len(self._constraints):
                break  # safety valve; should never trip
        return dropped

    def relax(self) -> RelaxationResult:
        """Best-effort resolution of an over-/under-constrained sketch.

        Analysis-only (no geometry is moved). For an OVER-constrained sketch it
        returns the conflicting constraints and a minimal drop suggestion that
        restores consistency; for an UNDER-constrained sketch it reports the free
        DOF and the still-unpinned entities. Documented as a graph-rank /
        least-squares *heuristic* — see :class:`RelaxationResult`.
        """
        rank = self._rank_subset(None)
        analysis = self.analyze()
        conflicting = [self._ref(i) for i in analysis.redundant_constraints]

        if analysis.status is SketchStatus.OVER:
            feasible = False
            dropped = [self._ref(i) for i in self._drop_set()]
        else:
            feasible = True
            dropped = []

        return RelaxationResult(
            feasible=feasible,
            residual=float(analysis.redundant_dof),
            status=analysis.status,
            conflicting_constraints=conflicting,
            dropped_suggestions=dropped,
            free_dof=analysis.free_dof,
            unpinned_entities=list(rank["unpinned"]),  # type: ignore[arg-type]
        )

    def conflicts(self) -> List[ConstraintRef]:
        """The conflicting (redundant / over-removing) constraints, worst-first.

        Convenience wrapper over :meth:`relax` for a repair loop that only wants
        the conflict set; empty when the sketch is consistent.
        """
        return self.relax().conflicting_constraints


# ---------------------------------------------------------------------------
# SolveSpace-backed real 2D solver (optional extra)
# ---------------------------------------------------------------------------
def _import_solvespace():
    """Lazy import of python-solvespace (the SolveSpace solver)."""
    import python_solvespace  # noqa: WPS433 (deliberately local / lazy)
    return python_solvespace


def solvespace_available() -> bool:
    """True when the ``constraints`` extra (python-solvespace) is importable."""
    try:
        _import_solvespace()
        return True
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class SolveResult:
    """Outcome of a SolveSpace solve."""

    solved: bool
    residual_dof: int              # sketch-relative DOF (base scaffolding removed)
    status: SketchStatus
    result_flag: int
    failures: Tuple[int, ...] = field(default_factory=tuple)


class SolveSpaceSketch:
    """Build real 2D geometry + constraints and solve with SolveSpace.

    A 2D work plane plus one shared 3D normal form fixed "scaffolding"; the DOF
    reported by :meth:`solve` is *sketch-relative* — the constant scaffolding
    freedom is subtracted so an unconstrained pair of points reads 4, a single
    circle reads 3, and so on (matching the harness's DOF intuition).

    Requires ``pip install harnesscad[constraints]``. Constructing without the
    extra raises ``ImportError``; probe with :func:`solvespace_available` first.
    """

    def __init__(self) -> None:
        ps = _import_solvespace()
        self._ps = ps
        self._sys = ps.SolverSystem()
        self._wp = self._sys.create_2d_base()
        self._normal = self._sys.add_normal_3d(0.0, 0.0, 0.0, 1.0)
        self._sys.solve()
        self._base_dof = self._sys.dof()   # constant scaffolding freedom
        self._n_constraints = 0

    # -- entities -----------------------------------------------------------
    def add_point(self, x: float, y: float):
        return self._sys.add_point_2d(x, y, self._wp)

    def add_line(self, p1, p2):
        return self._sys.add_line_2d(p1, p2, self._wp)

    def add_circle(self, center, radius: float):
        dist = self._sys.add_distance(radius, self._wp)
        return self._sys.add_circle(self._normal, center, dist, self._wp)

    # -- constraints --------------------------------------------------------
    def constrain(self, kind: str, a, b=None, value: Optional[float] = None) -> None:
        sys, wp = self._sys, self._wp
        if kind == "coincident":
            sys.coincident(a, b, wp)
        elif kind == "horizontal":
            sys.horizontal(a, wp)
        elif kind == "vertical":
            sys.vertical(a, wp)
        elif kind == "parallel":
            sys.parallel(a, b, wp)
        elif kind == "perpendicular":
            sys.perpendicular(a, b, wp)
        elif kind == "distance":
            if value is None:
                raise ValueError("'distance' constraint requires a value")
            sys.distance(a, b, value, wp)
        elif kind == "radius":
            if value is None:
                raise ValueError("'radius' constraint requires a value")
            sys.diameter(a, 2.0 * value)          # SolveSpace works in diameters
        elif kind == "equal":
            sys.equal(a, b, wp)
        else:
            raise ValueError(f"unknown constraint kind '{kind}'")
        self._n_constraints += 1

    # -- solve --------------------------------------------------------------
    def solve(self) -> SolveResult:
        ps = self._ps
        flag = self._sys.solve()
        okay = flag == ps.ResultFlag.OKAY
        try:
            failures = tuple(int(h) for h in self._sys.failures())
        except Exception:  # noqa: BLE001
            failures = ()

        if okay:
            residual = self._sys.dof() - self._base_dof   # dof() valid only after solve
            status = SketchStatus.WELL if residual <= 0 else SketchStatus.UNDER
        else:
            residual = -1
            if flag == ps.ResultFlag.TOO_MANY_UNKNOWNS:
                status = SketchStatus.UNDER
            else:  # INCONSISTENT / DIDNT_CONVERGE -> over-/conflicting-constrained
                status = SketchStatus.OVER

        return SolveResult(
            solved=okay,
            residual_dof=residual,
            status=status,
            result_flag=int(flag),
            failures=failures,
        )
