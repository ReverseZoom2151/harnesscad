"""CADBench-Verified metrics (HARNESS_BLUEPRINT.md sec.16).

The blueprint's metric set, ranked editability/validity ABOVE fidelity:

  1. sketch_editability  — fraction of fully-constrained sketches (dof == 0,
                           zero over-constraints), from query('sketch_dof').
  2. program_execution   — did the op stream rebuild without kernel errors, i.e.
                           ApplyOpsResult.ok (the whole batch applied + verified).
  3. brep_validity       — watertight / manifold / valid solid, from
                           query('validity') (with a solid-presence fallback for
                           backends that don't expose a real topology check).
  4. dimension_match     — measured geometry vs the acceptance spec within
                           tolerance, over the summary/validity/measure families.

Plus trajectory_efficiency = optimal_len / actual_len (sec.16: η = L* / L_agent).

Every function reads only backend queries / the ApplyOpsResult — none mutate.
A backend that can't answer a query returns ``{}`` (the StubBackend does this for
`validity`/`measure`); those fields are treated as "not applicable" and skipped
rather than scored as failures, so the same metric code runs on every backend.
"""

from __future__ import annotations

from typing import Tuple

from cisp.protocol import ApplyOpsResult

# Default relative tolerance for dimensional comparisons (1%).
DEFAULT_TOLERANCE = 0.01


# --- individual metrics -----------------------------------------------------
def program_execution(result: ApplyOpsResult) -> bool:
    """Did the feature sequence rebuild without kernel/verify errors?"""
    return bool(result.ok)


def sketch_editability(backend) -> float:
    """Fraction of sketches that are fully constrained (dof == 0).

    Over-constrained (dof < 0) and under-constrained (dof > 0) sketches both
    count against the fraction. A model with no sketches is vacuously 1.0.
    """
    dofs = backend.query("sketch_dof") or {}
    if not dofs:
        return 1.0
    fully = sum(1 for d in dofs.values() if d == 0)
    return fully / len(dofs)


def _gather(backend) -> dict:
    """Merge the measurable state across query families into one flat dict.

    Later families override earlier ones. A synthetic ``is_valid`` is derived
    from solid presence when the backend exposes no real ``validity`` check, so
    validity is always assertable (True when nothing has been built yet).
    """
    merged: dict = {}
    merged.update(backend.query("summary") or {})
    validity = backend.query("validity") or {}
    merged.update(validity)
    if "is_valid" not in merged:
        feature_count = merged.get("feature_count", 0)
        if feature_count > 0:
            merged["is_valid"] = bool(merged.get("solid_present"))
        else:
            merged["is_valid"] = True  # nothing built -> nothing invalid
    merged.update(backend.query("measure") or {})
    return merged


def brep_validity(backend) -> bool:
    """Is the current B-rep valid (watertight/manifold/no self-intersection)?

    Uses query('validity') when the backend provides it; otherwise falls back to
    solid presence (see :func:`_gather`).
    """
    return bool(_gather(backend).get("is_valid", True))


def _within(expected, actual, tol: float) -> bool:
    """Tolerance-aware equality: exact for bools, relative for numbers,
    element-wise for lists/tuples (e.g. a bbox), exact otherwise."""
    if isinstance(expected, bool):
        return isinstance(actual, bool) and expected == actual
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return False
        return all(_within(e, a, tol) for e, a in zip(expected, actual))
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) <= tol * abs(expected) + 1e-9
    return expected == actual


def dimension_match(backend, acceptance: dict) -> Tuple[bool, dict]:
    """Compare measured geometry against the acceptance spec.

    Walks the ``summary`` / ``validity`` / ``measure`` families of the acceptance
    dict, comparing each expected field to the backend's measurement within the
    spec's ``tolerance``. A field the backend cannot measure (absent from the
    gathered state) is *skipped*, not failed.

    Returns ``(ok, details)`` where ``ok`` is True iff no field failed, and
    ``details`` records per-field pass/fail/skip outcomes.
    """
    tol = float(acceptance.get("tolerance", DEFAULT_TOLERANCE))
    measured = _gather(backend)
    checks = []
    for family in ("summary", "validity", "measure"):
        spec = acceptance.get(family) or {}
        for key, expected in spec.items():
            if key not in measured:
                checks.append({"family": family, "key": key, "status": "skipped"})
                continue
            ok = _within(expected, measured[key], tol)
            checks.append({
                "family": family, "key": key,
                "status": "pass" if ok else "fail",
                "expected": expected, "actual": measured[key],
            })
    n_fail = sum(1 for c in checks if c["status"] == "fail")
    n_pass = sum(1 for c in checks if c["status"] == "pass")
    n_skip = sum(1 for c in checks if c["status"] == "skipped")
    details = {"checks": checks, "passed": n_pass, "failed": n_fail, "skipped": n_skip}
    return n_fail == 0, details


def trajectory_efficiency(optimal_len: int, actual_len: int) -> float:
    """η = L* / L_agent, capped at 1.0 (sec.16 trajectory efficiency).

    The reference op stream is the optimal L*; a solver that emits more ops than
    optimal scores below 1.0. Guards against a zero/empty trajectory.
    """
    if actual_len <= 0:
        return 0.0
    return min(1.0, optimal_len / actual_len)
