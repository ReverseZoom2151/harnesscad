"""Face identity across parametric rebuilds -- fingerprint + graph-relative hybrid.

OpenCAD's ``TOPOLOGY.md`` states the topological-naming problem head-on: a
constraint that targets ``box-0001:face:0`` silently points at the wrong geometry
once an upstream fillet or boolean rebuilds the shape, because faces get
reindexed, split, merged or deleted.  It surveys four prior-art families (OCC
TNaming, Build123d-style geometric hashing, Fusion-style graph-relative identity,
STEP occurrence paths) and lists the concrete failure modes any solution must
handle: boolean fan-out, fillet face splitting, feature reorder, symmetric
geometry, imported geometry, backend portability.

This module implements the *deterministic* half of that hybrid so the harness can
answer "is this the same face I constrained earlier?" without a kernel:

* :func:`fingerprint` -- a stable digest of quantised geometric attributes
  (surface kind, normal, plane offset, area, centroid) -- Build123d-style hashing;
* :class:`Provenance` -- which operation produced the face and from which parent
  face + local index -- Fusion-style graph-relative identity, which disambiguates
  what pure hashing cannot (symmetric bodies);
* :func:`match_topology` -- greedy, cost-ranked, order-independent matching of an
  old face set to a rebuilt one, classifying every face as **matched**, **split**
  (one old -> N coplanar new fragments), **merged** (N old -> one new),
  **deleted** or **created**, and flagging **ambiguous** matches whose two best
  candidates are within a margin (the symmetry failure mode);
* :func:`resolve_reference` -- migrate a stored subshape reference across a
  rebuild, returning the surviving ID, the fragment alternatives after a split,
  and an explicit reason when the reference is stale.

Deterministic: quantised attributes, sorted cost ordering, ID tie-breaks; no
clock, no randomness.

Public API
----------
``FaceRecord``, ``Provenance``, ``MatchReport``, ``ReferenceResolution``
``fingerprint``, ``match_topology``, ``resolve_reference``
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Provenance",
    "FaceRecord",
    "FacePair",
    "MatchReport",
    "ReferenceResolution",
    "fingerprint",
    "match_cost",
    "match_topology",
    "resolve_reference",
]

Vec3 = Tuple[float, float, float]


def _quantise(value: float, quantum: float) -> float:
    if quantum <= 0.0:
        return float(value)
    return round(value / quantum) * quantum + 0.0  # +0.0 normalises -0.0


@dataclass(frozen=True)
class Provenance:
    """Graph-relative identity: who made this face, and from what."""

    operation: str = ""            # e.g. "fillet_edges"
    node_id: str = ""              # feature-tree node that ran the operation
    parent_face_ids: Tuple[str, ...] = ()   # old faces this face derives from
    local_index: int = 0           # index within the operation's output

    def key(self) -> str:
        return "%s|%s|%s|%d" % (
            self.operation,
            self.node_id,
            ",".join(self.parent_face_ids),
            self.local_index,
        )


@dataclass(frozen=True)
class FaceRecord:
    """A face as seen by the identity system."""

    id: str
    surface: str = "planar"          # planar | cylindrical | spherical | blend | ...
    normal: Optional[Vec3] = None
    centroid: Vec3 = (0.0, 0.0, 0.0)
    area: float = 0.0
    provenance: Optional[Provenance] = None

    def plane_offset(self) -> Optional[float]:
        """Signed distance of the face plane from the origin (planar faces only)."""
        if self.normal is None:
            return None
        n = _unit(self.normal)
        return n[0] * self.centroid[0] + n[1] * self.centroid[1] + n[2] * self.centroid[2]


def _unit(v: Vec3) -> Vec3:
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )


def fingerprint(
    face: FaceRecord,
    *,
    position_quantum: float = 1e-3,
    area_quantum: float = 1e-3,
    normal_quantum: float = 1e-2,
) -> str:
    """Stable digest of a face's quantised geometry (kernel-agnostic hash)."""
    normal = _unit(face.normal) if face.normal is not None else None
    parts = [
        face.surface,
        "n=%s"
        % (
            "none"
            if normal is None
            else ",".join("%.6f" % _quantise(c, normal_quantum) for c in normal)
        ),
        "c=%s" % ",".join("%.6f" % _quantise(c, position_quantum) for c in face.centroid),
        "a=%.6f" % _quantise(face.area, area_quantum),
    ]
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def match_cost(
    old: FaceRecord,
    new: FaceRecord,
    *,
    scale: float = 1.0,
    normal_tolerance: float = 0.2,
) -> Optional[float]:
    """Dissimilarity of two faces; ``None`` when they cannot be the same face."""
    if old.surface != new.surface:
        return None
    if (old.normal is None) != (new.normal is None):
        return None
    normal_term = 0.0
    if old.normal is not None and new.normal is not None:
        cos = max(-1.0, min(1.0, _dot(_unit(old.normal), _unit(new.normal))))
        angle = math.acos(cos)
        if angle > normal_tolerance:
            return None
        normal_term = angle / max(normal_tolerance, 1e-9)

    position_term = _dist(old.centroid, new.centroid) / max(scale, 1e-9)
    denom = max(old.area, new.area, 1e-9)
    area_term = abs(old.area - new.area) / denom
    provenance_bonus = 0.0
    if (
        new.provenance is not None
        and old.id in new.provenance.parent_face_ids
    ):
        provenance_bonus = -0.5  # graph-relative evidence outranks geometry
    return 1.0 * position_term + 1.0 * area_term + 0.5 * normal_term + provenance_bonus


@dataclass(frozen=True)
class FacePair:
    old_id: str
    new_id: str
    cost: float
    ambiguous: bool = False


@dataclass
class MatchReport:
    matched: List[FacePair] = field(default_factory=list)
    splits: Dict[str, List[str]] = field(default_factory=dict)   # old -> new fragments
    merges: Dict[str, List[str]] = field(default_factory=dict)   # new -> old sources
    deleted: List[str] = field(default_factory=list)
    created: List[str] = field(default_factory=list)
    ambiguous: List[str] = field(default_factory=list)           # old ids

    def mapping(self) -> Dict[str, str]:
        return {pair.old_id: pair.new_id for pair in self.matched}


def _bbox_scale(faces: Sequence[FaceRecord]) -> float:
    if not faces:
        return 1.0
    xs = [f.centroid[0] for f in faces]
    ys = [f.centroid[1] for f in faces]
    zs = [f.centroid[2] for f in faces]
    span = max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
    )
    return max(span, 1.0)


def _coplanar(a: FaceRecord, b: FaceRecord, *, position_tolerance: float,
              normal_tolerance: float) -> bool:
    if a.normal is None or b.normal is None:
        return False
    cos = max(-1.0, min(1.0, _dot(_unit(a.normal), _unit(b.normal))))
    if math.acos(cos) > normal_tolerance:
        return False
    oa, ob = a.plane_offset(), b.plane_offset()
    if oa is None or ob is None:
        return False
    return abs(oa - ob) <= position_tolerance


def match_topology(
    old_faces: Sequence[FaceRecord],
    new_faces: Sequence[FaceRecord],
    *,
    cost_threshold: float = 1.0,
    ambiguity_margin: float = 1e-3,
    position_tolerance: float = 1e-6,
    normal_tolerance: float = 0.2,
) -> MatchReport:
    """Match an old face set to a rebuilt one and classify every face."""
    scale = _bbox_scale(list(old_faces) + list(new_faces))

    # Exact fingerprint pass first (fast, unambiguous when unique on both sides).
    report = MatchReport()
    old_by_id = {f.id: f for f in old_faces}
    new_by_id = {f.id: f for f in new_faces}

    candidates: List[Tuple[float, str, str]] = []
    best_two: Dict[str, List[float]] = {}
    for old in old_faces:
        costs: List[float] = []
        for new in new_faces:
            cost = match_cost(
                old, new, scale=scale, normal_tolerance=normal_tolerance
            )
            if cost is None or cost > cost_threshold:
                continue
            candidates.append((cost, old.id, new.id))
            costs.append(cost)
        best_two[old.id] = sorted(costs)[:2]

    candidates.sort(key=lambda t: (round(t[0], 12), t[1], t[2]))

    used_old: Dict[str, str] = {}
    used_new: Dict[str, str] = {}
    for cost, old_id, new_id in candidates:
        if old_id in used_old or new_id in used_new:
            continue
        pair_costs = best_two.get(old_id, [])
        ambiguous = (
            len(pair_costs) >= 2
            and abs(pair_costs[1] - pair_costs[0]) <= ambiguity_margin
        )
        used_old[old_id] = new_id
        used_new[new_id] = old_id
        report.matched.append(
            FacePair(old_id=old_id, new_id=new_id, cost=cost, ambiguous=ambiguous)
        )
        if ambiguous:
            report.ambiguous.append(old_id)

    report.matched.sort(key=lambda p: p.old_id)
    report.ambiguous.sort()

    unmatched_new = [f for f in new_faces if f.id not in used_new]
    unmatched_old = [f for f in old_faces if f.id not in used_old]

    # Split detection: a leftover new face that is coplanar with (or provenance-linked
    # to) an already-matched old face is a fragment of that face.
    for new in sorted(unmatched_new, key=lambda f: f.id):
        owner: Optional[str] = None
        if new.provenance is not None:
            for parent in new.provenance.parent_face_ids:
                if parent in old_by_id:
                    owner = parent
                    break
        if owner is None:
            for old_id, mapped_new in sorted(used_old.items()):
                old = old_by_id[old_id]
                if _coplanar(
                    old,
                    new,
                    position_tolerance=position_tolerance,
                    normal_tolerance=normal_tolerance,
                ):
                    owner = old_id
                    break
        if owner is None:
            report.created.append(new.id)
            continue
        fragments = report.splits.setdefault(owner, [])
        if owner in used_old and used_old[owner] not in fragments:
            fragments.append(used_old[owner])
        fragments.append(new.id)
        used_new[new.id] = owner

    # Merge detection: an unmatched old face whose geometry is subsumed by a
    # matched new face (same surface + coplanar) merged into it.
    for old in sorted(unmatched_old, key=lambda f: f.id):
        owner_new: Optional[str] = None
        for new_id, mapped_old in sorted(used_new.items()):
            new = new_by_id.get(new_id)
            if new is None:
                continue
            if _coplanar(
                old,
                new,
                position_tolerance=position_tolerance,
                normal_tolerance=normal_tolerance,
            ):
                owner_new = new_id
                break
        if owner_new is None:
            report.deleted.append(old.id)
            continue
        sources = report.merges.setdefault(owner_new, [])
        mapped = used_new.get(owner_new)
        if mapped and mapped not in sources:
            sources.append(mapped)
        sources.append(old.id)

    report.created.sort()
    report.deleted.sort()
    for fragments in report.splits.values():
        fragments.sort()
    for sources in report.merges.values():
        sources.sort()
    return report


@dataclass(frozen=True)
class ReferenceResolution:
    old_id: str
    new_id: Optional[str]
    status: str  # "matched" | "split" | "merged" | "deleted" | "ambiguous" | "unknown"
    alternatives: Tuple[str, ...] = ()
    reason: str = ""

    @property
    def is_stale(self) -> bool:
        return self.new_id is None


def resolve_reference(
    old_id: str,
    report: MatchReport,
    *,
    new_faces: Optional[Sequence[FaceRecord]] = None,
) -> ReferenceResolution:
    """Migrate a stored face reference across a rebuild using *report*.

    Splits resolve to the largest fragment (deterministic tie-break by ID) and
    expose the remaining fragments as ``alternatives`` so a caller -- or an agent --
    can pick the other one explicitly rather than silently pointing at the wrong
    geometry.
    """
    areas = {f.id: f.area for f in (new_faces or ())}

    if old_id in report.splits:
        fragments = list(report.splits[old_id])
        ranked = sorted(fragments, key=lambda fid: (-areas.get(fid, 0.0), fid))
        chosen = ranked[0]
        return ReferenceResolution(
            old_id=old_id,
            new_id=chosen,
            status="split",
            alternatives=tuple(f for f in ranked if f != chosen),
            reason="Face was split into %d fragments; largest chosen." % len(fragments),
        )

    for new_id, sources in report.merges.items():
        if old_id in sources:
            return ReferenceResolution(
                old_id=old_id,
                new_id=new_id,
                status="merged",
                reason="Face merged with %s." % ", ".join(s for s in sources if s != old_id),
            )

    mapping = report.mapping()
    if old_id in mapping:
        status = "ambiguous" if old_id in report.ambiguous else "matched"
        reason = (
            "Multiple candidates within the ambiguity margin (symmetric geometry)."
            if status == "ambiguous"
            else "Face survived the rebuild."
        )
        return ReferenceResolution(
            old_id=old_id, new_id=mapping[old_id], status=status, reason=reason
        )

    if old_id in report.deleted:
        return ReferenceResolution(
            old_id=old_id,
            new_id=None,
            status="deleted",
            reason="Face no longer exists after the rebuild.",
        )

    return ReferenceResolution(
        old_id=old_id,
        new_id=None,
        status="unknown",
        reason="Reference is not part of the matched topology.",
    )
