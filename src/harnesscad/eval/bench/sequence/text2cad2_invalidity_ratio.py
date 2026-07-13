"""Text2CAD Invalidity Ratio (IR) metric (Khan et al., Table 1 / Sec. 15).

The Text2CAD paper reports the **Invalidity Ratio** alongside F1 and Chamfer
distance: "the proportion of invalid CAD sequences". Sec. 15 (Failure Cases)
gives the *exact* deterministic conditions that make a generated sequence
invalid ("Invalidity ... occurs in approximately 1% of the test samples"):

  * a **line** whose start and end points are identical;
  * an **arc** with the same start and end points (degenerate);
  * an **extrusion** with **zero depth on both sides** (``d+ == d- == 0``).

A sequence is invalid if it contains *any* such element. The Invalidity Ratio
is ``#invalid / #total``. This module implements only the deterministic validity
predicate and the ratio; it does not learn or generate anything.

The predicate operates on lightweight dicts so it stays decoupled from any
particular decoder, e.g.::

    {"curves": [{"type": "line", "start": (0, 0), "end": (0, 0)}],
     "extrusion": {"d_plus": 0.0, "d_minus": 0.0}}

Existing metric modules (``bench.cadrille_metrics``, ``bench.diffusioncad_
generation_metrics``) compute different quantities and none encode these exact
degeneracy rules.
"""

from __future__ import annotations

from dataclasses import dataclass

EPS = 1e-9


def _same_point(a, b, eps: float = EPS) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def line_is_invalid(curve: dict, eps: float = EPS) -> bool:
    """A line is invalid when start and end coincide."""
    return _same_point(curve["start"], curve["end"], eps)


def arc_is_invalid(curve: dict, eps: float = EPS) -> bool:
    """An arc is invalid when start and end coincide (degenerate span)."""
    return _same_point(curve["start"], curve["end"], eps)


def curve_is_invalid(curve: dict, eps: float = EPS) -> bool:
    kind = curve.get("type")
    if kind == "line":
        return line_is_invalid(curve, eps)
    if kind == "arc":
        return arc_is_invalid(curve, eps)
    # circles and other primitives have no stated degeneracy rule here.
    return False


def extrusion_is_invalid(extrusion: dict, eps: float = EPS) -> bool:
    """Invalid when the extrusion depth is zero on *both* sides."""
    d_plus = abs(extrusion.get("d_plus", 0.0))
    d_minus = abs(extrusion.get("d_minus", 0.0))
    return d_plus <= eps and d_minus <= eps


@dataclass(frozen=True)
class ValidityReport:
    invalid_curves: int
    invalid_extrusion: bool

    @property
    def is_invalid(self) -> bool:
        return self.invalid_curves > 0 or self.invalid_extrusion


def inspect_sequence(sequence: dict, eps: float = EPS) -> ValidityReport:
    """Return a per-sequence validity report."""
    curves = sequence.get("curves", [])
    invalid_curves = sum(1 for c in curves if curve_is_invalid(c, eps))
    extr = sequence.get("extrusion")
    invalid_extr = extrusion_is_invalid(extr, eps) if extr is not None else False
    return ValidityReport(invalid_curves=invalid_curves, invalid_extrusion=invalid_extr)


def sequence_is_invalid(sequence: dict, eps: float = EPS) -> bool:
    """True if the sequence contains any degenerate curve or dead extrusion."""
    return inspect_sequence(sequence, eps).is_invalid


def invalidity_ratio(sequences, eps: float = EPS) -> float:
    """Fraction of sequences that are invalid (Table 1 ``IR``).

    Returns 0.0 for an empty collection.
    """
    seqs = list(sequences)
    if not seqs:
        return 0.0
    invalid = sum(1 for s in seqs if sequence_is_invalid(s, eps))
    return invalid / len(seqs)


def invalidity_percentage(sequences, eps: float = EPS) -> float:
    """Invalidity ratio expressed as a percentage (paper reports ``~1%``)."""
    return 100.0 * invalidity_ratio(sequences, eps)
