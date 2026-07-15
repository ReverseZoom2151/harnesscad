"""Measured Geometric Contract (MGC): the Specify-phase artifact of PDD.

This module is the machine-checkable spine of Parts-Driven Development (see
``audit/pdd_synthesis.md``). Where software spec-driven development stops at
spec-anchored because its validation is fallible prose-testing, CAD can reach a
further rung -- parts-as-measured-source -- because a part's acceptance criteria
are *measurable quantities of the produced artifact*. The MGC formalizes the
parsed part brief (``part_brief_parser.py`` / ``design_brief.py``) as a set of
predicates, each a measured quantity with a tolerance, that a differential
oracle can check against a re-measured file.

Two disciplines from the PDD spec are load-bearing here:

* **The anti-guess rule** (``[NEEDS CLARIFICATION]`` markers). Any measurable the
  brief does *not* state is emitted as an ``unbound=True`` predicate -- a
  clarification marker -- and its magnitude is *never* inferred. This is the
  coordinate-space-guessed-from-magnitude anti-pattern the harness deleted a bug
  to enforce.
* **Refuse-with-taint.** A ``MEASURED`` predicate whose measured value is absent
  or ``None`` resolves to ``MISSING`` (a taint), never a silent pass. ``volume =
  None`` must never read as satisfying the contract.

This module is pure Python (dataclasses, hashlib) -- it holds only the contract
data and the comparison logic. It imports no geometry kernel; the measurement it
checks against is supplied by the gate as a plain mapping.

Absolute imports under ``harnesscad.``, deterministic, stdlib-only.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple, Union

__all__ = [
    "PredicateKind",
    "Predicate",
    "MeasuredGeometricContract",
    "PredicateResult",
    "ContractReport",
    "EXACT_KEYS",
    "check",
    "compile_contract",
]

# Numeric target types accepted by a predicate.
TargetValue = Union[float, int, tuple, None]

# Keys whose target is an exact quantity (integer count or discrete tuple); for
# these the tolerance is ignored and equality must be exact. Every other
# MEASURED key is a float compared within its absolute tolerance.
EXACT_KEYS: Tuple[str, ...] = (
    "genus",
    "hole_count",
    "mobility_dof",
)


class PredicateKind(enum.Enum):
    """Whether a predicate is gate-checkable or merely advisory.

    ``MEASURED`` predicates name a geometry quantity that can be measured off the
    produced artifact and gated (volume, genus, min wall, ...). ``ADVISORY``
    predicates capture aesthetic or ergonomic intent ("looks premium",
    "comfortable grip") that has no gate; they are surfaced but never block
    satisfaction. This mirrors the harness's PROVEN/MEASURED vs HEURISTIC split.
    """

    MEASURED = "measured"
    ADVISORY = "advisory"


@dataclass(frozen=True)
class Predicate:
    """A single contract clause: one measurable outcome with a tolerance.

    ``key`` is a stable measurable name (e.g. ``"volume_mm3"``, ``"genus"``,
    ``"min_wall_mm"``, ``"mass_g"``, ``"hole_count"``, ``"center_of_mass_mm"``,
    ``"mobility_dof"``, ``"interference_mm3"``). ``target`` is the contracted
    value (a float, an integer count, a tuple such as a bounding box or a centre
    of mass, or ``None`` when the predicate is unbound). ``tolerance`` is an
    absolute epsilon applied to float comparisons and ignored for the exact
    integer/tuple keys in :data:`EXACT_KEYS`.

    ``kind`` records whether the clause is gate-checkable. ``unbound`` is ``True``
    when the brief did not specify this measurable -- a ``[NEEDS CLARIFICATION]``
    marker whose magnitude must never be guessed. ``hidden`` reserves a clause for
    the TDAD-style hidden-predicate split (held out during generation, evaluated
    after) so the model cannot game the visible contract.
    """

    key: str
    target: TargetValue
    tolerance: float = 0.0
    kind: PredicateKind = PredicateKind.MEASURED
    unbound: bool = False
    hidden: bool = False
    note: str = ""

    def is_exact(self) -> bool:
        """True when this predicate compares by equality rather than epsilon."""
        return self.key in EXACT_KEYS

    def canonical(self) -> str:
        """Stable, order-independent string form for content addressing."""
        return "|".join(
            (
                self.key,
                _canonical_target(self.target),
                repr(float(self.tolerance)),
                self.kind.value,
                "1" if self.unbound else "0",
                "1" if self.hidden else "0",
            )
        )


@dataclass
class MeasuredGeometricContract:
    """The full Measured Geometric Contract for a single part.

    ``predicates`` is the ordered set of contract clauses. ``intent`` is the
    free-text WHAT of the part (never the HOW -- ops live in the CISP plan). The
    contract is content-addressable via :meth:`digest`, so two contracts with the
    same clauses hash identically regardless of authoring order.
    """

    part_id: str
    predicates: Tuple[Predicate, ...] = ()
    intent: str = ""

    def measured(self) -> Tuple[Predicate, ...]:
        """Gate-checkable, bound predicates (kind == MEASURED and not unbound)."""
        return tuple(
            p
            for p in self.predicates
            if p.kind is PredicateKind.MEASURED and not p.unbound
        )

    def unbound(self) -> Tuple[Predicate, ...]:
        """The ``[NEEDS CLARIFICATION]`` markers -- measurables the brief omitted."""
        return tuple(p for p in self.predicates if p.unbound)

    def advisory(self) -> Tuple[Predicate, ...]:
        """Aesthetic/ergonomic predicates with no gate."""
        return tuple(p for p in self.predicates if p.kind is PredicateKind.ADVISORY)

    def visible(self) -> Tuple[Predicate, ...]:
        """Predicates exposed during generation (not held out)."""
        return tuple(p for p in self.predicates if not p.hidden)

    def hidden(self) -> Tuple[Predicate, ...]:
        """Predicates held out for the anti-gaming hidden-split evaluation."""
        return tuple(p for p in self.predicates if p.hidden)

    def digest(self) -> str:
        """A stable content hash of the sorted predicate tuple.

        Deterministic: predicates are sorted by their canonical form before
        hashing, so authoring order does not affect the digest. Uses SHA-256 over
        a canonical repr -- no wall-clock or random input.
        """
        parts = sorted(p.canonical() for p in self.predicates)
        payload = "\n".join([f"part_id={self.part_id}", *parts])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class PredicateResult:
    """The outcome of checking one predicate against a measurement.

    ``status`` is one of ``"PASS"``, ``"FAIL"``, ``"UNBOUND"``, ``"ADVISORY"`` or
    ``"MISSING"``. ``delta`` is the signed difference (measured - target) for a
    comparable float predicate, or ``None`` when no numeric delta applies (exact,
    tuple, advisory, unbound or missing cases).
    """

    predicate: Predicate
    measured_value: Any
    status: str
    delta: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "key": self.predicate.key,
            "target": self.predicate.target,
            "tolerance": self.predicate.tolerance,
            "kind": self.predicate.kind.value,
            "unbound": self.predicate.unbound,
            "hidden": self.predicate.hidden,
            "measured_value": self.measured_value,
            "status": self.status,
            "delta": self.delta,
            "note": self.predicate.note,
        }


@dataclass
class ContractReport:
    """The result of checking a whole contract against a measurement.

    ``satisfied`` is ``True`` only when every ``MEASURED`` bound predicate PASSes.
    UNBOUND, ADVISORY and MISSING statuses do not count as satisfying -- and a
    MEASURED predicate that measured MISSING (value absent or ``None``) blocks
    satisfaction, mirroring the harness's refuse-with-taint discipline where
    ``volume = None`` must never read as a pass.
    """

    results: Tuple[PredicateResult, ...] = ()

    @property
    def satisfied(self) -> bool:
        gating = [
            r
            for r in self.results
            if r.predicate.kind is PredicateKind.MEASURED and not r.predicate.unbound
        ]
        if not gating:
            return False
        return all(r.status == "PASS" for r in gating)

    def failures(self) -> Tuple[PredicateResult, ...]:
        """Bound MEASURED predicates that measured FAIL."""
        return tuple(r for r in self.results if r.status == "FAIL")

    def clarifications(self) -> Tuple[PredicateResult, ...]:
        """Unbound predicates -- the outstanding ``[NEEDS CLARIFICATION]`` items."""
        return tuple(r for r in self.results if r.status == "UNBOUND")

    def missing(self) -> Tuple[PredicateResult, ...]:
        """Bound MEASURED predicates with no (or ``None``) measured value -- taint."""
        return tuple(r for r in self.results if r.status == "MISSING")

    def as_dict(self) -> dict:
        return {
            "satisfied": self.satisfied,
            "results": [r.as_dict() for r in self.results],
        }


def check(
    contract: MeasuredGeometricContract, measurement: Mapping[str, Any]
) -> ContractReport:
    """Check a contract against a measurement mapping -- the PDD validate phase.

    For each predicate the value at ``measurement[predicate.key]`` is looked up
    and the outcome is *proven by measurement*, not asserted:

    * ADVISORY predicates always resolve to ``ADVISORY`` (no gate).
    * unbound predicates always resolve to ``UNBOUND`` (a clarification marker).
    * a MEASURED predicate whose key is absent, or whose value is ``None``,
      resolves to ``MISSING`` -- the taint case; it never reads as a pass.
    * otherwise the value is compared to the target: within the absolute
      tolerance for floats, and by exact equality for the integer/tuple keys in
      :data:`EXACT_KEYS`.
    """
    results = []
    for pred in contract.predicates:
        results.append(_check_one(pred, measurement))
    return ContractReport(results=tuple(results))


def _check_one(pred: Predicate, measurement: Mapping[str, Any]) -> PredicateResult:
    if pred.kind is PredicateKind.ADVISORY:
        value = measurement.get(pred.key) if _has(measurement, pred.key) else None
        return PredicateResult(pred, value, "ADVISORY", None)

    if pred.unbound:
        return PredicateResult(pred, None, "UNBOUND", None)

    if not _has(measurement, pred.key):
        return PredicateResult(pred, None, "MISSING", None)

    value = measurement.get(pred.key)
    if value is None:
        # Refuse-with-taint: a None measurement is never a pass.
        return PredicateResult(pred, None, "MISSING", None)

    if pred.is_exact() or _is_sequence(pred.target) or not _is_number(value):
        passed = _exact_equal(value, pred.target)
        return PredicateResult(pred, value, "PASS" if passed else "FAIL", None)

    # Float comparison within an absolute tolerance.
    if pred.target is None:
        return PredicateResult(pred, value, "MISSING", None)
    delta = float(value) - float(pred.target)
    passed = abs(delta) <= float(pred.tolerance)
    return PredicateResult(pred, value, "PASS" if passed else "FAIL", delta)


def compile_contract(brief: Any) -> MeasuredGeometricContract:
    """Compile a parsed part brief into a Measured Geometric Contract.

    Accepts a :class:`~harnesscad.domain.spec.part_brief_parser.PartSpec`, a
    :class:`~harnesscad.domain.spec.design_brief.CADBrief`, or a plain mapping,
    and is defensive about partial input. Every measurable the brief *states* is
    derived into a bound predicate (dimensions -> volume + bounding box; holes ->
    hole count + genus; wall -> min wall; material + volume -> mass; assembly ->
    mobility DOF; interference-free -> zero overlap). Every measurable the brief
    does *not* state is emitted as an ``unbound=True`` predicate -- a
    ``[NEEDS CLARIFICATION]`` marker whose magnitude is never guessed.
    """
    fields = _brief_fields(brief)
    part_id = _coerce_str(fields.get("part_id") or fields.get("model") or "part")
    intent = _coerce_str(fields.get("intent") or fields.get("model") or "")

    width = _coerce_float(fields.get("width"))
    depth = _coerce_float(fields.get("depth"))
    height = _coerce_float(fields.get("height"))
    kind = _coerce_str(fields.get("kind")).lower()
    holes = _coerce_int(fields.get("holes"))
    hole_diameter = _coerce_float(fields.get("hole_diameter"))
    wall = _coerce_float(fields.get("wall"))
    density = _coerce_float(fields.get("density_g_mm3"))
    mobility = fields.get("mobility_dof")
    is_assembly = _truthy(fields.get("is_assembly")) or mobility is not None

    predicates = []

    # --- Bounding box + volume, derived only from stated dimensions. ---
    have_dims = None not in (width, depth, height)
    if have_dims:
        bbox = (float(width), float(depth), float(height))
        predicates.append(
            Predicate(
                key="bbox_mm",
                target=bbox,
                tolerance=0.0,
                kind=PredicateKind.MEASURED,
                note="overall dimensions stated in brief",
            )
        )
        volume = _nominal_volume(kind, width, depth, height, wall)
        predicates.append(
            Predicate(
                key="volume_mm3",
                target=volume,
                tolerance=max(1e-3, volume * 1e-4),
                kind=PredicateKind.MEASURED,
                note="nominal solid volume from stated dimensions",
            )
        )
    else:
        predicates.append(_unbound("bbox_mm", "no overall dimensions in brief"))
        predicates.append(_unbound("volume_mm3", "volume needs stated dimensions"))

    # --- Holes -> hole count + genus (each through-hole adds one to the genus). ---
    if fields.get("holes") is not None:
        predicates.append(
            Predicate(
                key="hole_count",
                target=int(holes),
                kind=PredicateKind.MEASURED,
                note="hole count stated in brief",
            )
        )
        predicates.append(
            Predicate(
                key="genus",
                target=int(holes),
                kind=PredicateKind.MEASURED,
                note="one through-hole per stated hole",
            )
        )
        if hole_diameter and hole_diameter > 0.0:
            predicates.append(
                Predicate(
                    key="hole_diameter_mm",
                    target=float(hole_diameter),
                    tolerance=1e-2,
                    kind=PredicateKind.MEASURED,
                    note="hole diameter stated in brief",
                )
            )
    else:
        predicates.append(_unbound("hole_count", "brief does not state a hole count"))
        predicates.append(_unbound("genus", "topology needs a stated hole count"))

    # --- Wall -> minimum wall thickness. ---
    if wall and wall > 0.0:
        predicates.append(
            Predicate(
                key="min_wall_mm",
                target=float(wall),
                tolerance=1e-2,
                kind=PredicateKind.MEASURED,
                note="wall thickness stated in brief",
            )
        )
    else:
        predicates.append(
            _unbound("min_wall_mm", "brief does not state a minimum wall thickness")
        )

    # --- Material density + volume -> mass. ---
    if density and density > 0.0 and have_dims:
        volume = _nominal_volume(kind, width, depth, height, wall)
        mass = volume * float(density)
        predicates.append(
            Predicate(
                key="mass_g",
                target=mass,
                tolerance=max(1e-3, mass * 1e-3),
                kind=PredicateKind.MEASURED,
                note="mass from stated density and nominal volume",
            )
        )
    else:
        predicates.append(
            _unbound("mass_g", "mass needs a stated material density and dimensions")
        )

    # Centre of mass is never guessed from magnitude -- it stays unbound unless
    # the brief states it explicitly.
    com = fields.get("center_of_mass_mm")
    if _is_sequence(com):
        predicates.append(
            Predicate(
                key="center_of_mass_mm",
                target=tuple(float(v) for v in com),
                tolerance=1e-2,
                kind=PredicateKind.MEASURED,
                note="centre of mass stated in brief",
            )
        )
    else:
        predicates.append(
            _unbound("center_of_mass_mm", "brief does not state a centre of mass")
        )

    # --- Assembly: mobility DOF + interference-free. ---
    if is_assembly:
        if mobility is not None:
            predicates.append(
                Predicate(
                    key="mobility_dof",
                    target=int(_coerce_int(mobility) or 0),
                    kind=PredicateKind.MEASURED,
                    note="assembly mobility stated in brief",
                )
            )
        else:
            predicates.append(
                _unbound("mobility_dof", "assembly does not state a mobility DOF")
            )
        predicates.append(
            Predicate(
                key="interference_mm3",
                target=0.0,
                tolerance=1e-6,
                kind=PredicateKind.MEASURED,
                note="parts must not interfere (zero overlap)",
            )
        )

    # --- Advisory intent carried through but never gated. ---
    aesthetic = fields.get("aesthetic")
    if _coerce_str(aesthetic):
        predicates.append(
            Predicate(
                key="aesthetic",
                target=None,
                kind=PredicateKind.ADVISORY,
                note=_coerce_str(aesthetic),
            )
        )

    return MeasuredGeometricContract(
        part_id=part_id,
        predicates=tuple(predicates),
        intent=intent,
    )


# --------------------------------------------------------------------------- #
# Helpers (pure, defensive; no kernel imports).
# --------------------------------------------------------------------------- #


def _unbound(key: str, note: str) -> Predicate:
    """A ``[NEEDS CLARIFICATION]`` marker: measurable present, magnitude unknown."""
    return Predicate(
        key=key,
        target=None,
        tolerance=0.0,
        kind=PredicateKind.MEASURED,
        unbound=True,
        note=note,
    )


def _nominal_volume(
    kind: str,
    width: float,
    depth: float,
    height: float,
    wall: Optional[float],
) -> float:
    """Nominal solid volume from stated dimensions (hollow box if walled)."""
    outer = float(width) * float(depth) * float(height)
    if kind in ("box", "enclosure") and wall and wall > 0.0:
        inner_w = max(float(width) - 2.0 * float(wall), 0.0)
        inner_d = max(float(depth) - 2.0 * float(wall), 0.0)
        inner_h = max(float(height) - float(wall), 0.0)
        return outer - (inner_w * inner_d * inner_h)
    return outer


def _brief_fields(brief: Any) -> dict:
    """Flatten a brief (dataclass, CADBrief, mapping, or object) into a dict."""
    if brief is None:
        return {}
    if isinstance(brief, Mapping):
        fields = dict(brief)
    elif hasattr(brief, "__dict__"):
        fields = {k: v for k, v in vars(brief).items() if not k.startswith("_")}
    else:
        fields = {}

    # Best-effort extraction from a CADBrief's free-text dimension field.
    dims_text = fields.get("overall_dimensions")
    if dims_text and None in (fields.get("width"), fields.get("depth"), fields.get("height")):
        parsed = _parse_dims(str(dims_text))
        if parsed:
            fields.setdefault("width", parsed[0])
            fields.setdefault("depth", parsed[1])
            fields.setdefault("height", parsed[2])
    return fields


def _parse_dims(text: str) -> Optional[Tuple[float, float, float]]:
    """Extract a ``WxDxH`` triple from free text, if present."""
    import re

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return tuple(float(v) for v in match.groups())  # type: ignore[return-value]


def _has(measurement: Mapping[str, Any], key: str) -> bool:
    try:
        return key in measurement
    except TypeError:
        return False


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (tuple, list))


def _exact_equal(value: Any, target: Any) -> bool:
    if _is_sequence(value) and _is_sequence(target):
        if len(value) != len(target):
            return False
        return all(_scalar_equal(a, b) for a, b in zip(value, target))
    return _scalar_equal(value, target)


def _scalar_equal(a: Any, b: Any) -> bool:
    if _is_number(a) and _is_number(b):
        return float(a) == float(b)
    return a == b


def _canonical_target(target: TargetValue) -> str:
    if target is None:
        return "None"
    if _is_sequence(target):
        return "(" + ",".join(repr(float(v)) if _is_number(v) else repr(v) for v in target) + ")"
    if _is_number(target):
        return repr(float(target))
    return repr(target)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "assembly")
    return bool(value)
