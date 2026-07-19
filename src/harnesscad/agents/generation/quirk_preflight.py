"""Kernel-quirk preflight over a CISP op stream: warn before OCCT bites.

Two OCCT quirk catalogs are wired here:

* :mod:`harnesscad.agents.generation.occt_quirks` -- the client-side catalog,
  whose operation families are
  exactly CISP op families: ``boolean``, ``revolve``, ``loft``, ``face-query``.
  It also carries two callable predicates -- ``ring_holes_feasible`` (Roshera's
  saddle-boolean refusal formula) and ``overlap_is_near_tangent`` (OpenCAD's
  BBOX_NEAR_TANGENT preflight).
* :mod:`harnesscad.agents.generation.occt_quirks_oce` -- the kernel-side
  catalog, whose families are kernel-internal
  (``wire-heal``, ``pcurve``, ``seam``, ``step-import``, ...). Only ``loft``
  and ``fillet`` name operations a caller emits, so only those are reachable
  from an op stream; the rest describe repairs the kernel does to itself.

This module is the adapter between the two and the op stream. It ADVISES -- it
never refuses. That is the deliberate split: a quirk says "OCCT is known to
misbehave here", which is a reason to warn a model, not a reason to refuse a
caller's op. Refusal on an op stream belongs to
:mod:`harnesscad.core.cisp.op_gate`, which judges admissibility against a
catalog the caller supplied.

``ring_holes_feasible``'s own reason strings say "REFUSED:" (the source
refuses); they are surfaced here as warnings, because this preflight has no
mandate the caller did not give it.

Pure stdlib, deterministic, no kernel and no model.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from harnesscad.agents.generation import occt_quirks, occt_quirks_oce

__all__ = ["QuirkWarning", "quirk_warnings"]


#: CISP op tag -> the CLIENT catalog's operation family.
_CLIENT_FAMILY: Dict[str, str] = {
    "boolean": "boolean",
    "revolve": "revolve",
    "loft": "loft",
}

#: CISP op tag -> the OCE (kernel-side) catalog's operation family. Only the
#: families an op stream can actually name; the kernel-internal ones cannot be
#: triggered from here and would be noise.
_OCE_FAMILY: Dict[str, str] = {
    "loft": "loft",
    "fillet": "fillet",
}

#: Quirks DECIDED below by a closed-form predicate against the op's real
#: numbers, rather than listed for every op of their family. Emitting these from
#: the family sweep as well would both cry wolf (warning that a 90-degree
#: revolve might be a 0-degree revolve) and double-report the ops that do trip
#: them. The predicates own these ids; the sweep skips them.
_PREDICATE_OWNED = frozenset({
    "saddle-boolean-adjacent-holes",   # _ring_warnings
    "revolve-zero-degrees",            # the angle check in quirk_warnings
})


class QuirkWarning(tuple):
    """One advisory finding: ``(op_index, op_name, quirk_id, message)``.

    A tuple subclass so it stays trivially comparable/serialisable while still
    reading by name at a call site.
    """

    __slots__ = ()

    def __new__(cls, op_index: int, op_name: str, quirk_id: str, message: str):
        return super().__new__(cls, (op_index, op_name, quirk_id, message))

    @property
    def op_index(self) -> int:
        return self[0]

    @property
    def op_name(self) -> str:
        return self[1]

    @property
    def quirk_id(self) -> str:
        return self[2]

    @property
    def message(self) -> str:
        return self[3]


def _as_dict(op: Any) -> Dict[str, Any]:
    """An op's dict form, whether it is a CISP Op or already a dict."""
    to_dict = getattr(op, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    if isinstance(op, Mapping):
        return dict(op)
    return {}


def _num(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _ring_warnings(ops: Sequence[Dict[str, Any]]) -> List[QuirkWarning]:
    """Saddle-boolean check for a hole replicated around a circular pattern.

    The Roshera formula needs (count, ring_radius, hole_radius); an op stream
    spells that as a `hole` at (x, y) off the axis plus a `circular_pattern`
    with a count. The ring radius is the hole's distance from the origin, which
    is where `circular_pattern`'s default axis stands -- so a pattern about a
    MOVED axis is not judged here rather than judged wrongly.
    """
    out: List[QuirkWarning] = []
    holes = [(i, d) for i, d in enumerate(ops) if d.get("op") == "hole"]
    if not holes:
        return out
    for index, d in enumerate(ops):
        if d.get("op") != "circular_pattern":
            continue
        count = int(_num(d.get("count"), 0))
        axis = d.get("axis") or ()
        # Only the default origin axis is interpretable as "the ring centre".
        if len(axis) == 6 and (_num(axis[0]) or _num(axis[1])):
            continue
        for hole_index, hole in holes:
            ring_r = math.hypot(_num(hole.get("x")), _num(hole.get("y")))
            hole_r = _num(hole.get("diameter")) / 2.0
            if ring_r <= 0 or hole_r <= 0:
                continue
            verdict = occt_quirks.ring_holes_feasible(count, ring_r, hole_r)
            if not verdict.ok:
                out.append(QuirkWarning(
                    index, "circular_pattern",
                    verdict.quirk_id or "saddle-boolean-adjacent-holes",
                    f"hole at op[{hole_index}] patterned {count}x: "
                    + str(verdict.reason).replace("REFUSED:", "").strip()))
    return out


def quirk_warnings(
    ops: Sequence[Union[Any, Mapping[str, Any]]],
) -> List[QuirkWarning]:
    """Advisory kernel-quirk findings for a CISP op stream, in op order.

    Deterministic and cheap: catalog lookups plus two closed-form predicates,
    no kernel call. Returns [] for a stream that names no quirk-bearing op --
    which is the common case, so the check costs a dict build per op.
    """
    dicts = [_as_dict(op) for op in ops]
    out: List[QuirkWarning] = []

    for index, d in enumerate(dicts):
        tag = str(d.get("op", ""))

        # A revolve whose angle reduces to 0 mod 360 is OCCT's zero-degree
        # trap: the kernel does NOT read it as a full revolution.
        if tag == "revolve":
            angle = _num(d.get("angle"), 360.0)
            if angle % 360.0 == 0 and angle != 360.0:
                out.append(QuirkWarning(
                    index, tag, "revolve-zero-degrees",
                    f"angle={angle:g} reduces to 0 mod 360; OCCT does not read "
                    f"that as a full revolution. Normalise to "
                    f"{occt_quirks.normalize_revolve_angle(angle):g}."))

        for catalog, family_map in ((occt_quirks, _CLIENT_FAMILY),
                                    (occt_quirks_oce, _OCE_FAMILY)):
            family = family_map.get(tag)
            if family is None:
                continue
            for quirk in catalog.quirks_for_operation(family):
                if quirk.id in _PREDICATE_OWNED:
                    continue
                out.append(QuirkWarning(
                    index, tag, quirk.id,
                    f"{quirk.quirk} Trigger: {quirk.trigger} "
                    f"Workaround: {quirk.workaround}"))

    out.extend(_ring_warnings(dicts))
    return out
