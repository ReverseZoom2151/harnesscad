"""Assembly / insertion-sequence planner — the *ordering* half of assembly QA.

Where :mod:`verifiers.assembly` answers "is this set of mates solvable?" (DOF
accounting) and :mod:`verifiers.interference` answers "do the placed parts
clash?", this module answers the *process* question the blueprint's assembly
tier still leaves open:

    In what ORDER can the parts be inserted, and along which direction does each
    part travel into its seat, so that it never collides with the parts already
    placed?

That is the classic assembly-sequence / disassembly problem. We solve it on the
placed geometry (world-space axis-aligned bounding boxes, exactly the records
:func:`backends...query('assembly')` already exposes for the interference gate)
by a *removal* search: a part can be taken out of the current sub-assembly if it
has a collision-free straight-line escape along one of a small candidate set of
axes (+-X / +-Y / +-Z, or the mate axis). The reverse of a full disassembly is a
valid assembly order, and each removal direction negated is the part's insertion
vector.

Collision along a candidate axis is tested two ways, cheapest-sound-first
(mirroring :mod:`verifiers.interference`):

  * **bbox corridor** (always available, pure python): sweep the moving part's
    AABB to infinity along the escape direction; the part is blocked iff that
    semi-infinite corridor interpenetrates an already-present part's AABB (face
    contact at the seat is excluded by a small epsilon). This is the sound,
    deterministic fallback the tests exercise.
  * **swept OCCT common** (only when CadQuery/OCCT is importable *and* both parts
    carry a real shape): the moving solid is translated in steps along the escape
    direction and boolean-``common``-ed with each obstacle; a non-empty common at
    any step is a real clash. Refines a bbox "blocked" verdict to catch the
    puzzle-piece case where AABBs overlap but the solids clear. Any kernel
    failure degrades to the bbox verdict — never a crash.

The search is a deterministic depth-first disassembly with memoised dead-ends, so
it finds a complete sequence whenever one exists and otherwise reports the parts
that have *no* collision-free escape at all (ERROR-worthy — they cannot be
inserted without disturbing their neighbours).

Standalone, stdlib + math only; OCCT is touched solely inside CadQuery-guarded
paths. Fully deterministic for a fixed input.

Two entry points:
  * :func:`plan_assembly_sequence` — run the planner on a backend, a raw
    ``query('assembly')`` dict, or an :class:`verifiers.assembly.AssemblyModel`
    (pass ``bboxes=`` for the model, whose parts carry no geometry of their own).
  * :class:`SequenceCheck` — a :class:`verifiers.verify.Verifier`
    (``name='assembly-sequence'``) wrapping the planner: INFO-skip on a
    stub/single-part model, ERROR when no full sequence exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.eval.verifiers.assembly import AssemblyModel, Mate
from harnesscad.eval.verifiers.verify import Diagnostic, Severity, VerifyReport

Vec3 = Tuple[float, float, float]
BBox = Tuple[float, float, float, float, float, float]  # xmin,ymin,zmin,xmax,ymax,zmax


# --------------------------------------------------------------------------- #
# Candidate escape axes (deterministic order)
# --------------------------------------------------------------------------- #
#: The six principal insertion/removal directions, tried in this fixed order.
AXES: List[Tuple[str, Vec3]] = [
    ("+X", (1.0, 0.0, 0.0)),
    ("-X", (-1.0, 0.0, 0.0)),
    ("+Y", (0.0, 1.0, 0.0)),
    ("-Y", (0.0, -1.0, 0.0)),
    ("+Z", (0.0, 0.0, 1.0)),
    ("-Z", (0.0, 0.0, -1.0)),
]

_AXIS_NAME = {v: n for n, v in AXES}


def _axis_name(vec: Vec3) -> str:
    return _AXIS_NAME.get(vec, f"({vec[0]:g},{vec[1]:g},{vec[2]:g})")


# --------------------------------------------------------------------------- #
# Internal part placement record
# --------------------------------------------------------------------------- #
@dataclass
class _Placement:
    """One placed part as the planner needs it: an id, an optional world AABB,
    and an optional OCCT shape for the exact swept test."""

    id: str
    bbox: Optional[BBox] = None
    shape: object = None

    def center(self) -> Optional[Vec3]:
        if self.bbox is None:
            return None
        b = self.bbox
        return ((b[0] + b[3]) / 2.0, (b[1] + b[4]) / 2.0, (b[2] + b[5]) / 2.0)


# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #
@dataclass
class AssemblySequence:
    """A planned insertion sequence for a placed assembly.

    * ``insertion_order``    — part ids in the order they should be inserted.
    * ``insertion_vectors``  — per-part unit direction the part travels *into*
      its seat (negated escape direction), ``{id: (x, y, z)}``.
    * ``disassembly_order``  — the reverse: a safe order to take parts *off*.
    * ``critical_chain``     — the longest chain of mate-coupled parts that must
      be built in order (the sequence's schedule bottleneck).
    * ``blocked_parts``      — parts with NO collision-free escape/insertion axis
      against the rest of the assembly (ERROR-worthy).
    * ``ok``                 — a complete collision-free sequence was found.
    * ``trivial``            — fewer than two placed parts (nothing to sequence).
    """

    insertion_order: List[str] = field(default_factory=list)
    insertion_vectors: Dict[str, Vec3] = field(default_factory=dict)
    disassembly_order: List[str] = field(default_factory=list)
    critical_chain: List[str] = field(default_factory=list)
    blocked_parts: List[str] = field(default_factory=list)
    ok: bool = False
    trivial: bool = False
    n_parts: int = 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "trivial": self.trivial,
            "n_parts": self.n_parts,
            "insertion_order": list(self.insertion_order),
            "insertion_vectors": {k: list(v)
                                  for k, v in self.insertion_vectors.items()},
            "disassembly_order": list(self.disassembly_order),
            "critical_chain": list(self.critical_chain),
            "blocked_parts": list(self.blocked_parts),
        }

    def render(self) -> str:
        """A compact multi-line human summary of the plan."""
        if self.trivial:
            return (f"assembly sequence: trivial "
                    f"({self.n_parts} part(s), nothing to sequence)")
        lines: List[str] = []
        if self.ok:
            steps = []
            for pid in self.insertion_order:
                vec = self.insertion_vectors.get(pid)
                steps.append(f"{pid}[{_axis_name(vec)}]" if vec else pid)
            lines.append("insert: " + " -> ".join(steps))
            lines.append("remove: " + " -> ".join(self.disassembly_order))
            if self.critical_chain:
                lines.append("critical chain: "
                             + " -> ".join(self.critical_chain))
        else:
            lines.append(f"NO valid full insertion sequence "
                         f"for {self.n_parts} parts")
            if self.blocked_parts:
                lines.append("blocked (no collision-free axis): "
                             + ", ".join(self.blocked_parts))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public planner
# --------------------------------------------------------------------------- #
def plan_assembly_sequence(model_or_backend,
                           bboxes: Optional[Dict[str, Sequence[float]]] = None,
                           *,
                           eps: float = 1e-6) -> AssemblySequence:
    """Plan a collision-free insertion sequence for a placed assembly.

    ``model_or_backend`` may be:
      * a backend exposing ``query('assembly')`` (parts carry ``bbox`` / ``shape``),
      * the raw ``{parts, mates, transforms}`` dict that query returns, or
      * a :class:`verifiers.assembly.AssemblyModel` — in which case ``bboxes``
        (``{part_id: [xmin,ymin,zmin,xmax,ymax,zmax]}``) supplies the geometry,
        since a model carries only ids + mates.

    ``bboxes`` also overrides/augments per-part boxes for the other two forms.
    Returns an :class:`AssemblySequence`. ``eps`` is the face-contact tolerance
    that keeps seated neighbours from counting as corridor obstacles.
    """
    placements, mates = _resolve(model_or_backend, bboxes)
    seq = AssemblySequence(n_parts=len(placements))

    if len(placements) < 2:
        seq.trivial = True
        seq.ok = True
        seq.insertion_order = [p.id for p in placements]
        seq.disassembly_order = list(reversed(seq.insertion_order))
        return seq

    order_removal = _search_disassembly(placements, eps)
    if order_removal is not None:
        # order_removal: list of (placement_index, escape_vec) in removal order.
        by_index = {p_i: vec for p_i, vec in order_removal}
        removal_ids = [placements[p_i].id for p_i, _ in order_removal]
        seq.disassembly_order = removal_ids
        seq.insertion_order = list(reversed(removal_ids))
        # Insertion vector = negated escape (into the seat, not out of it).
        for p_i, vec in order_removal:
            pid = placements[p_i].id
            seq.insertion_vectors[pid] = (-vec[0], -vec[1], -vec[2])
        seq.ok = True
        seq.critical_chain = _critical_chain(seq.insertion_order, mates)
    else:
        seq.ok = False
        seq.blocked_parts = _fully_blocked(placements, eps)

    return seq


# --------------------------------------------------------------------------- #
# Input resolution
# --------------------------------------------------------------------------- #
def _resolve(source, bboxes: Optional[Dict[str, Sequence[float]]]
             ) -> Tuple[List[_Placement], List[Mate]]:
    """Normalise any accepted input to (placements, mates)."""
    box_override = _norm_boxes(bboxes)

    # AssemblyModel -> ids + mates; geometry comes from bboxes override.
    if isinstance(source, AssemblyModel):
        placements = [_Placement(id=pid, bbox=box_override.get(pid))
                      for pid in source.parts]
        return placements, list(source.mates)

    # Backend -> read the assembly query; else assume a raw dict.
    raw: Optional[dict]
    if hasattr(source, "query") and callable(getattr(source, "query")):
        raw = _query(source, "assembly")
    elif isinstance(source, dict):
        raw = source
    else:
        raw = None
    raw = raw or {}

    placements = []
    for i, p in enumerate(raw.get("parts", []) or []):
        if isinstance(p, dict):
            pid = str(p.get("id", p.get("name", f"part{i}")))
            bb = _as_bbox(p.get("bbox"))
            shape = p.get("shape")
        else:
            pid, bb, shape = str(p), None, None
        if pid in box_override:
            bb = box_override[pid]
        placements.append(_Placement(id=pid, bbox=bb, shape=shape))

    mates = [Mate.from_dict(m) for m in raw.get("mates", []) or []
             if isinstance(m, dict)]
    return placements, mates


def _norm_boxes(bboxes: Optional[Dict[str, Sequence[float]]]
                ) -> Dict[str, BBox]:
    out: Dict[str, BBox] = {}
    if not bboxes:
        return out
    for k, v in bboxes.items():
        bb = _as_bbox(v)
        if bb is not None:
            out[str(k)] = bb
    return out


def _as_bbox(v) -> Optional[BBox]:
    if v is None:
        return None
    seq = list(v)
    if len(seq) < 6:
        return None
    return (float(seq[0]), float(seq[1]), float(seq[2]),
            float(seq[3]), float(seq[4]), float(seq[5]))


# --------------------------------------------------------------------------- #
# Disassembly search (deterministic DFS with memoisation)
# --------------------------------------------------------------------------- #
def _search_disassembly(placements: List[_Placement], eps: float
                        ) -> Optional[List[Tuple[int, Vec3]]]:
    """Return a full removal order [(index, escape_vec), ...] or None.

    Depth-first: at each state try to remove parts in ascending id order, each
    along the first clear candidate axis; recurse on the smaller sub-assembly.
    Dead-end states are memoised so the search stays polynomial in practice and
    is fully deterministic.
    """
    n = len(placements)
    all_present = frozenset(range(n))
    # Removal order preference: ascending part id, stable.
    order = sorted(range(n), key=lambda i: (placements[i].id, i))
    failed: set = set()
    path: List[Tuple[int, Vec3]] = []

    def dfs(present: frozenset) -> bool:
        if not present:
            return True
        if present in failed:
            return False
        others_cache = list(present)
        for idx in order:
            if idx not in present:
                continue
            obstacles = [placements[k] for k in others_cache if k != idx]
            vec = _free_axis(placements[idx], obstacles, eps)
            if vec is None:
                continue
            path.append((idx, vec))
            if dfs(present - {idx}):
                return True
            path.pop()
        failed.add(present)
        return False

    if dfs(all_present):
        return list(path)
    return None


def _free_axis(part: _Placement, obstacles: List[_Placement],
               eps: float) -> Optional[Vec3]:
    """The first candidate escape axis clear of every obstacle, or None.

    A part with no bbox cannot be tested against corridors, so it is treated as
    freely removable along +X (there is nothing to collide with meaningfully).
    """
    if part.bbox is None:
        return AXES[0][1]
    # Prefer the mate/separation axis (snapped to a principal direction) first,
    # then the six principal axes in fixed order.
    candidates: List[Vec3] = []
    mate_axis = _separation_axis(part, obstacles)
    if mate_axis is not None:
        candidates.append(mate_axis)
    for _, vec in AXES:
        if vec not in candidates:
            candidates.append(vec)

    for vec in candidates:
        if _axis_clear(part, obstacles, vec, eps):
            return vec
    return None


def _axis_clear(part: _Placement, obstacles: List[_Placement],
                vec: Vec3, eps: float) -> bool:
    """True iff ``part`` can escape along ``vec`` without hitting any obstacle."""
    for obst in obstacles:
        if obst.bbox is None:
            continue
        if _corridor_blocked(part.bbox, obst.bbox, vec, eps):
            # Optionally refine an AABB block with the exact swept OCCT test.
            if not _swept_clear_occt(part, obst, vec, eps):
                return False
    return True


def _corridor_blocked(pb: BBox, ob: BBox, vec: Vec3, eps: float) -> bool:
    """True iff the semi-infinite sweep of AABB ``pb`` along ``vec`` overlaps
    the AABB ``ob`` (interior overlap; seat-face contact excluded by ``eps``)."""
    ax = 0 if vec[0] else (1 if vec[1] else 2)
    sign = vec[ax]
    # The two lateral dimensions must interpenetrate for any collision.
    for d in range(3):
        if d == ax:
            continue
        overlap = min(pb[d + 3], ob[d + 3]) - max(pb[d], ob[d])
        if overlap <= eps:
            return False
    # Along the travel axis, the corridor covers everything ahead of the part.
    if sign > 0:
        # sweep covers [pb_min_ax, +inf): blocked if obstacle reaches into it.
        return ob[ax + 3] > pb[ax] + eps
    # sweep covers (-inf, pb_max_ax]: blocked if obstacle reaches into it.
    return ob[ax] < pb[ax + 3] - eps


def _separation_axis(part: _Placement,
                     obstacles: List[_Placement]) -> Optional[Vec3]:
    """A principal axis pointing away from the obstacles' collective centre,
    used as the first-tried "mate axis" candidate. None if undecidable."""
    pc = part.center()
    if pc is None or not obstacles:
        return None
    cs = [o.center() for o in obstacles if o.center() is not None]
    if not cs:
        return None
    ox = sum(c[0] for c in cs) / len(cs)
    oy = sum(c[1] for c in cs) / len(cs)
    oz = sum(c[2] for c in cs) / len(cs)
    diff = (pc[0] - ox, pc[1] - oy, pc[2] - oz)
    # Snap to the dominant principal component.
    ad = [abs(diff[0]), abs(diff[1]), abs(diff[2])]
    m = max(ad)
    if m <= 1e-12:
        return None
    ax = ad.index(m)
    sign = 1.0 if diff[ax] >= 0 else -1.0
    vec = [0.0, 0.0, 0.0]
    vec[ax] = sign
    return (vec[0], vec[1], vec[2])


def _fully_blocked(placements: List[_Placement], eps: float) -> List[str]:
    """Parts that have NO collision-free escape against the *full* assembly."""
    blocked: List[str] = []
    for i, p in enumerate(placements):
        others = [placements[k] for k in range(len(placements)) if k != i]
        if _free_axis(p, others, eps) is None:
            blocked.append(p.id)
    return sorted(blocked)


# --------------------------------------------------------------------------- #
# Critical dependency chain
# --------------------------------------------------------------------------- #
def _critical_chain(insertion_order: List[str],
                    mates: List[Mate]) -> List[str]:
    """Longest chain of mate-coupled parts respecting the insertion order.

    Builds the mate adjacency, then does a DP over the insertion order: the
    longest chain ending at a part extends the longest chain of any earlier
    mated neighbour. Deterministic (id tie-breaks)."""
    adj: Dict[str, set] = {pid: set() for pid in insertion_order}
    for m in mates:
        a, b = m.a, m.b
        if b is None:
            continue
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    pos = {pid: i for i, pid in enumerate(insertion_order)}
    best_len: Dict[str, int] = {}
    prev: Dict[str, Optional[str]] = {}
    for pid in insertion_order:
        choice_len, choice_prev = 0, None
        for nbr in sorted(adj[pid]):
            if pos.get(nbr, 1 << 30) < pos[pid] and best_len.get(nbr, 0) > choice_len:
                choice_len = best_len[nbr]
                choice_prev = nbr
        best_len[pid] = choice_len + 1
        prev[pid] = choice_prev
    # Reconstruct the longest chain; require at least one mate edge (len >= 2).
    if not best_len:
        return []
    end = max(insertion_order, key=lambda p: (best_len[p], -pos[p]))
    if best_len[end] < 2:
        return []
    chain: List[str] = []
    cur: Optional[str] = end
    while cur is not None:
        chain.append(cur)
        cur = prev[cur]
    chain.reverse()
    return chain


# --------------------------------------------------------------------------- #
# Optional exact swept collision via OCCT (guarded; degrades to bbox verdict)
# --------------------------------------------------------------------------- #
def _swept_clear_occt(part: _Placement, obst: _Placement,
                      vec: Vec3, eps: float) -> bool:
    """Refine an AABB "blocked" verdict with an exact swept boolean-common.

    Only attempted when CadQuery/OCCT is importable *and* both parts carry a
    shape; samples the moving part's solid along the escape direction and
    returns False (still blocked) if any sample overlaps the obstacle. Any
    failure (no kernel, no shapes) returns False so the sound bbox verdict
    stands — this can only *clear* an AABB false positive, never invent one.
    """
    if part.shape is None or obst.shape is None:
        return False
    if not _cadquery_available():
        return False
    try:
        from OCP.gp import gp_Trsf, gp_Vec  # noqa: WPS433
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform  # noqa: WPS433

        wa = getattr(part.shape, "wrapped", part.shape)
        wb = getattr(obst.shape, "wrapped", obst.shape)
        # Corridor length: enough to sweep the part fully past the obstacle.
        if part.bbox is None or obst.bbox is None:
            return False
        ax = 0 if vec[0] else (1 if vec[1] else 2)
        span = (max(part.bbox[ax + 3], obst.bbox[ax + 3])
                - min(part.bbox[ax], obst.bbox[ax]))
        length = span + max(1.0, span)
        steps = 8
        for k in range(steps + 1):
            t = length * k / steps
            trsf = gp_Trsf()
            trsf.SetTranslation(gp_Vec(vec[0] * t, vec[1] * t, vec[2] * t))
            moved = BRepBuilderAPI_Transform(wa, trsf, True).Shape()
            vol = _common_volume(moved, wb)
            if vol is not None and vol > eps:
                return False  # still overlapping somewhere along the sweep
        return True  # swept clear -> the AABB block was a false positive
    except Exception:  # noqa: BLE001 - any kernel failure keeps the bbox verdict
        return False


def _common_volume(wa, wb) -> Optional[float]:
    try:
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp

        common = BRepAlgoAPI_Common(wa, wb)
        common.Build()
        if not common.IsDone():
            return None
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(common.Shape(), props)
        return abs(float(props.Mass()))
    except Exception:  # noqa: BLE001
        return None


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401, WPS433
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
class SequenceCheck:
    """A :class:`verifiers.verify.Verifier` wrapping the sequence planner.

    ``name = 'assembly-sequence'``. ``check(backend, opdag)`` reads
    ``query('assembly')`` and emits:

      * INFO  ``assembly-sequence-skipped``  — backend has no ``'assembly'``
        query (stub) or it is empty.
      * INFO  ``assembly-sequence-trivial``  — fewer than two placed parts.
      * ERROR ``no-assembly-sequence``       — no collision-free full insertion
        order exists (with a per-part ``blocked-insertion`` ERROR for each part
        that has no collision-free axis at all).
      * INFO  ``assembly-sequence``          — the found order (never fatal).

    Only the no-sequence / blocked cases are ERRORs, so a well-sequenced
    assembly leaves ``report.ok`` True.
    """

    name = "assembly-sequence"

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = float(eps)

    def check(self, backend, opdag) -> VerifyReport:
        raw = _query(backend, "assembly")
        if not raw:
            return VerifyReport([_info(
                "assembly-sequence-skipped",
                "assembly-sequence check skipped: backend has no 'assembly' "
                "query (only an assembly-aware backend exposes placed parts).")])
        return self.check_model(raw)

    def check_model(self, model_or_backend,
                    bboxes: Optional[Dict[str, Sequence[float]]] = None
                    ) -> VerifyReport:
        """Run the planner on a model/dict/backend and turn it into a report."""
        seq = plan_assembly_sequence(model_or_backend, bboxes, eps=self.eps)
        return VerifyReport(sequence_diagnostics(seq))


def sequence_diagnostics(seq: AssemblySequence) -> List[Diagnostic]:
    """Turn an :class:`AssemblySequence` into verifier diagnostics."""
    if seq.trivial or seq.n_parts < 2:
        return [_info(
            "assembly-sequence-trivial",
            f"assembly-sequence check skipped: {seq.n_parts} placed part(s) — "
            "at least two are needed to sequence an insertion.")]
    if seq.ok:
        steps = " -> ".join(
            f"{pid}[{_axis_name(seq.insertion_vectors.get(pid, (0, 0, 0)))}]"
            for pid in seq.insertion_order)
        diags = [_info(
            "assembly-sequence",
            f"valid insertion order for {seq.n_parts} parts: {steps}.")]
        if seq.critical_chain:
            diags.append(_info(
                "assembly-sequence-critical",
                "critical dependency chain: "
                + " -> ".join(seq.critical_chain) + "."))
        return diags

    diags = [_err(
        "no-assembly-sequence",
        f"no collision-free insertion order exists for {seq.n_parts} parts: at "
        "least one part cannot be placed without disturbing already-placed "
        "parts (interlocked / captured geometry).")]
    for pid in seq.blocked_parts:
        diags.append(_err(
            "blocked-insertion",
            f"part '{pid}' has no collision-free insertion axis "
            "(+-X / +-Y / +-Z or the mate axis) against the rest of the "
            "assembly.", pid))
    return diags


def with_sequence(verifiers, eps: float = 1e-6) -> List:
    """Return a new verifier list with a :class:`SequenceCheck` appended
    (mirrors :func:`verifiers.assembly.with_assembly`)."""
    return list(verifiers) + [SequenceCheck(eps=eps)]


# --------------------------------------------------------------------------- #
# Graceful-degradation helpers (mirror verifiers/assembly.py)
# --------------------------------------------------------------------------- #
def _query(backend, q: str) -> Optional[dict]:
    try:
        result = backend.query(q)
    except Exception:  # noqa: BLE001 - unsupported query must degrade, not crash
        return None
    return result or None


def _err(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.ERROR, code, msg, where)


def _warn(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.WARNING, code, msg, where)


def _info(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.INFO, code, msg, where)
