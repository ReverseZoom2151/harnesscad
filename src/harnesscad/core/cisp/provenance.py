"""CISP provenance citations -- traceSDD's orphan-REQ check, applied to geometry.

The idea, borrowed and made geometric
--------------------------------------
traceSDD (Panda, arXiv 2606.30689) makes hallucination detection a *set
difference*: every line of generated code cites a REQ-XXX, and two O(1) checks
fall out. A **cited-but-nonexistent REQ** (a line that names a requirement not in
the spec) is a hallucinated citation; an **orphan REQ** (a requirement no line
claims) is unimplemented. Its killer finding is that every injected hallucination
still *passed all functional tests* -- so traceability catches what measurement
cannot.

CAD has the exact same two failure modes, and here the "spec" is measured
geometry, so the citation is checkable against the artifact rather than against
prose:

  * an **orphan op** is an op whose application changed *nothing measurable* --
    the CAD analogue of a cited-but-nonexistent REQ, a hallucinated build step
    that a green measured-gate would never notice (the part is still watertight,
    still the right volume, and one op did no work). This is the field-liveness
    census (:mod:`harnesscad.eval.selftest.field_liveness`, every (op, field)
    must move the geometry) lifted from the field to the whole op.
  * an **orphan feature** is measured geometry that *no op claims* -- the reverse
    set difference, a face/edge/quantity present in the artifact that no step in
    the program produced. It is the direct attack on the MGC's many-to-one
    residual: a part can hit the right volume for the wrong reason, but it cannot
    cite an op that is not there.

The design keeps traceSDD's cheapness: attribution is a set of feature keys per
op, the two checks are set differences, and nothing here depends on a kernel.

Backend agnosticism
-------------------
This module never builds geometry and never imports a backend. It is handed a
``measure_state`` callable that turns an op prefix into a *measurement* -- a plain
mapping of named, comparable quantities (volume, surface_area, n_faces, n_edges,
a bbox tuple, genus, ...). :func:`build_provenance` drives the replay by calling
that callable on growing prefixes and diffing successive measurements, so the same
code runs over frep, OCCT, Manifold, a stub, or a synthetic fixture with no engine
at all. What a measurement *contains* is the caller's choice; the only contract is
that two states that differ geometrically produce mappings that differ in at least
one key -- exactly the property ``field_liveness.signature`` is built to give.

    measure_state : Callable[[Sequence[Op]], Optional[Mapping[str, Any]]]
        Apply the op prefix (0..k ops) and return a measurement of the resulting
        state, or ``None`` / ``{}`` when no state can be measured (no engine). It
        is called len(ops)+1 times: once with the empty prefix for the baseline,
        then once per op prefix.

What this proves, and what it does not
--------------------------------------
An orphan op provably did no measurable work; a non-orphan op provably moved some
measured quantity. Neither proves the op did the *right* work -- attribution is a
floor, not a ceiling, the same honest caveat field-liveness carries. It catches
the dead step and the unclaimed feature, which measurement alone does not.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Op, canonical_json

__all__ = [
    "EPS",
    "Measurement",
    "MeasureState",
    "OpDelta",
    "Provenance",
    "build_provenance",
    "orphan_ops",
    "unattributed_features",
    "attributed_features",
]

#: Below this a scalar move is noise, not a change. Matches the rounding floor
#: field_liveness uses so an op that only jitters a coordinate reads as an orphan.
EPS: float = 1e-9

#: A measurement is a mapping of feature key -> comparable value. Numeric values
#: (and tuples/lists of numerics, e.g. a bbox) are diffed by magnitude; anything
#: else (bool, str, None, genus flags) is diffed by equality.
Measurement = Mapping[str, Any]

#: Apply an op prefix and measure the resulting state (see the module docstring).
MeasureState = Callable[[Sequence[Op]], Optional[Measurement]]


# ---------------------------------------------------------------------------
# Diffing two measurements.
# ---------------------------------------------------------------------------

def _numeric_delta(before: Any, after: Any) -> Optional[float]:
    """Magnitude of a numeric change, or ``None`` when the values are not numeric.

    A ``None`` return is the caller's signal to fall back to equality. Booleans
    are NOT numbers here (``True`` is not "one more than nothing") -- they compare
    by equality, like genus or watertightness flags.
    """
    if isinstance(before, bool) or isinstance(after, bool):
        return None
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return abs(float(after) - float(before))
    if isinstance(before, (list, tuple)) and isinstance(after, (list, tuple)):
        total = 0.0
        for i in range(max(len(before), len(after))):
            b = before[i] if i < len(before) else 0.0
            a = after[i] if i < len(after) else 0.0
            if isinstance(b, bool) or isinstance(a, bool):
                return None
            if not isinstance(b, (int, float)) or not isinstance(a, (int, float)):
                return None
            total += abs(float(a) - float(b))
        return total
    return None


def _feature_delta(before: Optional[Measurement],
                   after: Optional[Measurement],
                   eps: float = EPS) -> Tuple[Tuple[str, ...], float]:
    """The feature keys an op moved, and a scalar magnitude of the whole move.

    A key counts as moved when its numeric magnitude exceeds ``eps``, or -- for a
    non-numeric value -- when it is simply not equal. A key present on one side
    and absent on the other has appeared or vanished, which is a move.
    """
    b_map: Mapping[str, Any] = before or {}
    a_map: Mapping[str, Any] = after or {}
    changed: List[str] = []
    magnitude = 0.0
    for k in sorted(set(b_map) | set(a_map)):
        present = (k in b_map) and (k in a_map)
        b = b_map.get(k)
        a = a_map.get(k)
        if not present:
            changed.append(k)
            magnitude += 1.0
            continue
        d = _numeric_delta(b, a)
        if d is None:
            if b != a:
                changed.append(k)
                magnitude += 1.0
        elif d > eps:
            changed.append(k)
            magnitude += d
    return tuple(changed), magnitude


def _op_id(index: int, op: Op) -> str:
    """A stable, content-addressed citation for an op at a given position.

    ``index`` disambiguates two identical ops in one stream; the hash pins the
    op's content so a citation survives serialisation (see ops.canonical_json).
    """
    digest = hashlib.sha1(canonical_json(op).encode("utf-8")).hexdigest()[:8]
    return "%03d:%s#%s" % (index, op.OP, digest)


# ---------------------------------------------------------------------------
# The provenance structure.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpDelta:
    """The measured geometry change one op caused, and its citation.

    ``changed`` is the set of feature keys this op moved -- the geometry it
    *claims*. An op with an empty ``changed`` and no ``error`` is an ORPHAN: it
    ran and moved nothing measurable.
    """

    index: int
    op_id: str
    op_tag: str
    before: Measurement
    after: Measurement
    changed: Tuple[str, ...]
    magnitude: float
    error: str = ""

    @property
    def is_orphan(self) -> bool:
        """Ran, but moved nothing measurable. A measurement error is NOT an orphan
        (we could not observe the op, so we do not get to call it dead)."""
        return not self.changed and not self.error

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "op_id": self.op_id,
            "op": self.op_tag,
            "changed": list(self.changed),
            "magnitude": self.magnitude,
            "error": self.error,
            "orphan": self.is_orphan,
        }


@dataclass
class Provenance:
    """Every op's geometry delta, indexed for the two orphan set differences."""

    deltas: List[OpDelta] = field(default_factory=list)
    #: len(ops)+1 measurements: the baseline (empty prefix) then one per op.
    measurements: List[Measurement] = field(default_factory=list)
    #: Set when the replay could not measure any state (no engine). The provenance
    #: is then empty and every downstream check degrades to a clean PASS.
    skipped: str = ""

    @property
    def baseline(self) -> Measurement:
        return self.measurements[0] if self.measurements else {}

    def by_index(self) -> Dict[int, OpDelta]:
        """op index -> its delta. The forward map traceSDD keys citations on."""
        return {d.index: d for d in self.deltas}

    def by_op_id(self) -> Dict[str, OpDelta]:
        return {d.op_id: d for d in self.deltas}

    def attributed_features(self) -> set:
        """The union of every feature key some op claims to have moved.

        This is the CITED set. Any measured feature outside it is an orphan
        feature (see :func:`unattributed_features`).
        """
        out: set = set()
        for d in self.deltas:
            out.update(d.changed)
        return out

    def to_dict(self) -> dict:
        return {
            "structure": "cisp_provenance",
            "skipped": self.skipped,
            "n_ops": len(self.deltas),
            "attributed_features": sorted(self.attributed_features()),
            "deltas": [d.to_dict() for d in self.deltas],
        }


# ---------------------------------------------------------------------------
# Building it: replay the prefix, measure after each op, diff.
# ---------------------------------------------------------------------------

def build_provenance(ops: Sequence[Op],
                     measure_state: MeasureState,
                     eps: float = EPS) -> Provenance:
    """Replay the op prefix, measuring after each op, and record each op's delta.

    ``measure_state`` is called with the empty prefix (the baseline) and then with
    each 1..k prefix; successive measurements are diffed to attribute geometry to
    the op that appeared between them. An op whose delta is null is a candidate
    orphan op. If the baseline measurement is ``None``/empty AND every subsequent
    measurement is too, the replay had no state to measure and the provenance is
    returned ``skipped`` (a clean degrade for the no-engine case).
    """
    prov = Provenance()
    try:
        baseline = measure_state(list(ops[:0]))
    except Exception as exc:  # noqa: BLE001 - a measurement crash is an observation
        prov.skipped = "baseline measure_state raised: %s" % exc
        return prov
    prov.measurements.append(baseline or {})

    prev: Measurement = baseline or {}
    saw_any_measurement = bool(baseline)
    for i, op in enumerate(ops):
        prefix = list(ops[: i + 1])
        error = ""
        try:
            after = measure_state(prefix)
        except Exception as exc:  # noqa: BLE001
            after = None
            error = "%s: %s" % (type(exc).__name__, exc)
        after_map: Measurement = after or {}
        if after:
            saw_any_measurement = True
        changed, magnitude = _feature_delta(prev, after_map, eps)
        prov.deltas.append(OpDelta(
            index=i,
            op_id=_op_id(i, op),
            op_tag=op.OP,
            before=prev,
            after=after_map,
            changed=changed,
            magnitude=magnitude,
            error=error,
        ))
        prov.measurements.append(after_map)
        prev = after_map

    if not saw_any_measurement:
        prov.skipped = ("measure_state returned no state for any prefix "
                        "(no engine / empty measurement)")
    return prov


# ---------------------------------------------------------------------------
# The two set differences.
# ---------------------------------------------------------------------------

def orphan_ops(prov: Provenance) -> List[OpDelta]:
    """Ops that changed nothing measurable -- the cited-but-nonexistent REQs.

    An orphan op is a build step the program includes and the geometry does not
    reflect: a hallucinated step. (An op whose measurement errored is excluded --
    it was not observed, so it is not provably dead.)
    """
    return [d for d in prov.deltas if d.is_orphan]


def unattributed_features(prov: Provenance,
                          measured_features: Sequence[str]) -> List[str]:
    """Measured geometry no op claims -- the reverse set difference (orphan features).

    ``measured_features`` is the set of feature identifiers actually present in the
    finished artifact, in the SAME namespace as the delta ``changed`` keys (e.g.
    ``"volume"``, ``"n_faces"``, ``"genus"``, or per-entity ids the caller emits).
    The orphan features are exactly ``set(measured_features) - attributed`` -- an
    O(1)-per-element set difference, language-agnostic, the traceSDD orphan-REQ
    check run in reverse. A non-empty result means the artifact carries geometry
    that the op stream cannot account for.
    """
    attributed = prov.attributed_features()
    return sorted(set(measured_features) - attributed)


def attributed_features(prov: Provenance) -> List[str]:
    """The features some op claims (the CITED set). Convenience over the property."""
    return sorted(prov.attributed_features())
