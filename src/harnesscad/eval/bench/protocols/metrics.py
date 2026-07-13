"""CADBench-Verified metrics (docs/blueprint.md sec.16).

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

from typing import List, Optional, Tuple

from harnesscad.core.cisp.protocol import ApplyOpsResult

# Default relative tolerance for dimensional comparisons (1%).
DEFAULT_TOLERANCE = 0.01

# Op parameters that are auto-generated reference handles (sketch/entity ids,
# boolean targets/tools). They are stable under deterministic replay but would
# spuriously differ between a *generated* op-DAG and the reference, so the
# sequence-level match ignores them and keys only on op tag + geometric params
# (the DeepCAD/CAD-LLM "entity/sketch accuracy" convention).
_REFERENCE_PARAM_KEYS = frozenset({"sketch", "a", "b", "target", "tool"})


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


# --- suite-level program-execution rate -------------------------------------
def _rebuilt_ok(item) -> bool:
    """Extract the "op stream rebuilt without kernel errors" bit from one item.

    Accepts either an :class:`ApplyOpsResult` (has ``.ok``), a ``TaskResult``
    (has ``.program_execution``), or a raw bool.
    """
    if hasattr(item, "program_execution"):
        return bool(item.program_execution)
    if hasattr(item, "ok"):
        return bool(item.ok)
    return bool(item)


def program_execution_rate(results) -> Optional[float]:
    """Fraction of a suite's tasks whose full op stream rebuilt cleanly.

    This is the suite-level aggregate of the per-task :func:`program_execution`
    (ApplyOpsResult.ok) — the DeepCAD/CAD-LLM "program execution / build success
    rate". ``results`` is any iterable of ApplyOpsResults, TaskResults, or bools.
    Returns ``None`` (not applicable) for an empty suite, never divides by zero.
    """
    items = list(results)
    if not items:
        return None
    return sum(1 for it in items if _rebuilt_ok(it)) / len(items)


# --- CAD sequence F1 (entity/op accuracy of the generated op-DAG) -----------
def _as_op_dict(op) -> dict:
    """Normalise an op (a cisp.ops.Op or its dict form) to a plain dict."""
    if hasattr(op, "to_dict"):
        return op.to_dict()
    return dict(op)


def _op_signature(op, ndigits: int = 6):
    """A hashable signature = op tag + geometric params (rounded, ids dropped).

    Reference-handle params (sketch/entity ids) are excluded so the match keys
    on op type and its dimensional/geometric parameters, mirroring DeepCAD's
    entity-accuracy scoring rather than exact string identity.
    """
    d = _as_op_dict(op)
    tag = d.get("op")
    params = []
    for key in sorted(d):
        if key == "op" or key in _REFERENCE_PARAM_KEYS:
            continue
        value = d[key]
        if isinstance(value, float):
            value = round(value, ndigits)
        elif isinstance(value, (list, tuple)):
            value = tuple(
                round(v, ndigits) if isinstance(v, float) else v for v in value)
        params.append((key, value))
    return (tag, tuple(params))


def cad_sequence_f1(built_ops, reference_ops) -> Optional[dict]:
    """Entity/op-level precision, recall and F1 of a generated op-DAG.

    The DeepCAD / CAD-LLM "Entity/Sketch Accuracy, CAD F1" metric: match each
    generated op against the reference by op tag + key params (multiset match),
    then report precision = matched/generated, recall = matched/reference, and
    their harmonic mean F1. Two identical op lists score 1.0; a partial match
    scores below 1.0. Returns ``None`` when no reference ops are available.
    """
    if reference_ops is None or built_ops is None:
        return None
    built = [_op_signature(o) for o in built_ops]
    ref = [_op_signature(o) for o in reference_ops]

    if not built and not ref:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "matched": 0, "n_built": 0, "n_reference": 0}

    # Multiset intersection: each reference op can be matched at most once.
    remaining: dict = {}
    for sig in ref:
        remaining[sig] = remaining.get(sig, 0) + 1
    matched = 0
    for sig in built:
        if remaining.get(sig, 0) > 0:
            remaining[sig] -= 1
            matched += 1

    precision = matched / len(built) if built else 0.0
    recall = matched / len(ref) if ref else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "matched": matched, "n_built": len(built), "n_reference": len(ref),
    }


# --- assembly metrics (mate accuracy + collision) ---------------------------
def _assembly_of(item) -> Optional[dict]:
    """Coerce ``item`` to an assembly-state dict, or ``None`` when there isn't one.

    Accepts a backend (calls ``query('assembly')``), an already-queried dict, or
    anything falsy/unsupported (-> ``None``). The ``assembly`` query family is an
    optional, concurrently-added capability; a backend that doesn't expose it
    returns ``{}`` and every assembly metric degrades to "not applicable".
    """
    if item is None:
        return None
    if isinstance(item, dict):
        return item or None
    query = getattr(item, "query", None)
    if callable(query):
        try:
            asm = query("assembly")
        except Exception:  # noqa: BLE001 - unknown query key -> not applicable
            return None
        return asm or None
    return None


def _has_interference(asm: dict) -> bool:
    """Does an assembly-state dict report any interpenetration?"""
    if asm.get("interferences"):
        return True
    if asm.get("collision") or asm.get("interference"):
        return True
    return int(asm.get("interference_count", 0) or 0) > 0


def _is_multipart(asm: dict) -> bool:
    """A collision/mate check only applies once there are >= 2 parts."""
    if "part_count" in asm:
        return int(asm.get("part_count", 0) or 0) >= 2
    if "parts" in asm:
        return len(asm.get("parts") or []) >= 2
    # No explicit part count: presence of mates implies an assembly.
    return bool(asm.get("mates"))


def assembly_mate_accuracy(built_assembly, reference_assembly) -> Optional[dict]:
    """Mate-type accuracy + residual-DOF error of a built assembly vs reference.

    Reads the ``assembly`` query family (mates + residual degrees of freedom).
    ``mate_type_accuracy`` is the fraction of reference mates whose type matches
    the built mate at the same position; ``residual_dof_error`` is the absolute
    difference in leftover assembly DOF. Returns ``None`` (INFO / skip) when
    either side has no assembly — e.g. a single-part model.
    """
    built = _assembly_of(built_assembly)
    reference = _assembly_of(reference_assembly)
    if not built or not reference:
        return None

    built_mates = list(built.get("mates") or [])
    ref_mates = list(reference.get("mates") or [])

    def _mate_type(m) -> object:
        return m.get("type") if isinstance(m, dict) else m

    if ref_mates:
        correct = sum(
            1 for i, rm in enumerate(ref_mates)
            if i < len(built_mates)
            and _mate_type(built_mates[i]) == _mate_type(rm))
        mate_type_accuracy = correct / len(ref_mates)
    else:
        # No reference mates: vacuously correct iff the build added none either.
        correct = 0
        mate_type_accuracy = 1.0 if not built_mates else 0.0

    built_dof = built.get("residual_dof")
    ref_dof = reference.get("residual_dof")
    if built_dof is None or ref_dof is None:
        residual_dof_error = None
    else:
        residual_dof_error = abs(float(built_dof) - float(ref_dof))

    return {
        "mate_type_accuracy": mate_type_accuracy,
        "residual_dof_error": residual_dof_error,
        "matched_mates": correct,
        "n_built_mates": len(built_mates),
        "n_reference_mates": len(ref_mates),
    }


def collision_rate(results_or_backends) -> Optional[float]:
    """Fraction of multi-part assemblies exhibiting interpenetration.

    Reads each item's ``assembly`` interference signal. Only multi-part
    assemblies count toward the denominator; a suite (or backend) with no
    assembly at all returns ``None`` (single-part -> not applicable), never a
    misleading 0.0.
    """
    assemblies = []
    for item in results_or_backends:
        asm = _assembly_of(item)
        if asm and _is_multipart(asm):
            assemblies.append(asm)
    if not assemblies:
        return None
    return sum(1 for a in assemblies if _has_interference(a)) / len(assemblies)
