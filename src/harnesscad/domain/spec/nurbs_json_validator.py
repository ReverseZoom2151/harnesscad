"""Validator for NURBGen's structured NURBS-surface JSON representation (NURBGen).

Mined from *NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS
Modeling*. NURBGen fine-tunes an LLM to emit each B-Rep face as a JSON object of
NURBS parameters -- poles (control points), ``u_knots``/``v_knots``,
``u_mults``/``v_mults``, ``u_degree``/``v_degree``, periodic flags and ``weights``
(paper Fig. 2) -- which is then converted to B-Rep with pythonOCC. A generated JSON
is only convertible if these fields are mutually consistent; NURBGen's usefulness
therefore hinges on a deterministic validity check, which is what this module ports.

:func:`validate_face` returns the list of structural errors in one face object
(empty == valid). :func:`validate_model` validates a whole ``{"face_0": {...}, ...}``
document. The consistency rules follow the OpenCASCADE ``Geom_BSplineSurface``
contract:

*   poles form a rectangular ``n_u x n_v`` grid of ``d``-vectors;
*   ``weights`` (if present) match the pole grid and are strictly positive;
*   knot sequences are strictly increasing and match their multiplicity arrays;
*   for a non-periodic direction, ``sum(mults) == n_poles + degree + 1``; for a
    periodic direction, ``sum(mults) == n_poles``;
*   degrees are ``>= 1``.

Stdlib-only, deterministic.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

__all__ = [
    "validate_face",
    "validate_model",
    "is_valid_face",
]

_REQUIRED = (
    "poles", "u_knots", "v_knots", "u_mults", "v_mults",
    "u_degree", "v_degree",
)


def _grid_shape(poles) -> tuple:
    """Return ``(n_u, n_v, dim)`` or raise ``ValueError`` if not a rectangular grid."""
    if not isinstance(poles, (list, tuple)) or not poles:
        raise ValueError("poles must be a non-empty list")
    n_u = len(poles)
    row_lens = set()
    dims = set()
    for row in poles:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("each pole row must be a non-empty list")
        row_lens.add(len(row))
        for pt in row:
            if not isinstance(pt, (list, tuple)):
                raise ValueError("each pole must be a coordinate list")
            dims.add(len(pt))
    if len(row_lens) != 1:
        raise ValueError("pole grid is not rectangular")
    if len(dims) != 1:
        raise ValueError("poles have inconsistent dimensionality")
    return n_u, row_lens.pop(), dims.pop()


def _check_direction(
    errors: List[str], axis: str, n_poles: int, knots, mults, degree, periodic
) -> None:
    if not isinstance(degree, int) or degree < 1:
        errors.append(f"{axis}_degree must be an integer >= 1")
        return
    if len(knots) != len(mults):
        errors.append(f"{axis}_knots and {axis}_mults length mismatch")
        return
    if len(knots) < 2:
        errors.append(f"{axis}_knots needs at least two knots")
        return
    for a, b in zip(knots, knots[1:]):
        if not b > a:
            errors.append(f"{axis}_knots must be strictly increasing")
            break
    if any((not isinstance(m, int)) or m < 1 for m in mults):
        errors.append(f"{axis}_mults must be positive integers")
        return
    total = sum(mults)
    expected = n_poles if periodic else n_poles + degree + 1
    if total != expected:
        kind = "periodic" if periodic else "non-periodic"
        errors.append(
            f"{axis} ({kind}): sum(mults)={total} but expected {expected} "
            f"for {n_poles} poles at degree {degree}"
        )


def validate_face(face: Dict) -> List[str]:
    """Return the list of structural errors in one NURBS-surface face object."""
    errors: List[str] = []
    for key in _REQUIRED:
        if key not in face:
            errors.append(f"missing required field: {key}")
    if errors:
        return errors

    try:
        n_u, n_v, _dim = _grid_shape(face["poles"])
    except ValueError as exc:
        return [str(exc)]

    weights = face.get("weights")
    if weights is not None:
        if len(weights) != n_u or any(len(r) != n_v for r in weights):
            errors.append("weights grid shape does not match poles grid")
        elif any(w <= 0 for row in weights for w in row):
            errors.append("weights must be strictly positive")

    _check_direction(errors, "u", n_u, face["u_knots"], face["u_mults"],
                     face["u_degree"], bool(face.get("u_periodic", 0)))
    _check_direction(errors, "v", n_v, face["v_knots"], face["v_mults"],
                     face["v_degree"], bool(face.get("v_periodic", 0)))
    return errors


def is_valid_face(face: Dict) -> bool:
    """True iff the face object is structurally valid."""
    return not validate_face(face)


def validate_model(model: Dict) -> Dict[str, List[str]]:
    """Validate a whole ``{"face_i": {...}}`` document.

    Returns a mapping of only the faces that have errors to their error lists; an
    empty mapping means the whole model is valid.
    """
    out: Dict[str, List[str]] = {}
    if not model:
        return {"<model>": ["model has no faces"]}
    for name, face in model.items():
        if not isinstance(face, dict):
            out[name] = ["face is not an object"]
            continue
        errs = validate_face(face)
        if errs:
            out[name] = errs
    return out
