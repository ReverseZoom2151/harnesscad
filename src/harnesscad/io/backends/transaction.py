"""Transactional kernel ops: every mutating op commits or leaves state byte-equivalent.

Without this, a backend op that fails partway leaves whatever state it had
managed to reach: a fillet that got halfway through surgery leaves a shell with
a dangling boundary edge, a boolean that hits a coplanar-face degeneracy after
splitting one face leaves the split face orphaned. Nothing downstream can tell
that wreckage apart from intent -- the verifiers happily MEASURE the corrupted
body and report its (real, wrong) volume. The historical escapes are both bad:
reseed the whole model from a previous checkpoint (cost grows with history), or
hope a heal pass untangles it.

This module raises the contract to transactional. Two primitives, applied at the
op entry point:

  1. PRE-FLIGHT (:func:`preflight`) -- cheap, op-specific input checks run
     BEFORE any mutation and before any snapshot, so a guaranteed-fail op never
     pays for a snapshot it will only throw away. Opt-in: a backend exposing
     ``validate_can_apply(op)`` gets gated; one that does not is unaffected.
  2. ROLLBACK (:func:`with_rollback`) -- snapshots the backend's state, runs the
     op body, and on failure restores the snapshot BEFORE the failure reaches
     the caller.

The invariant this buys, and the one the selfcheck proves by digest rather than
by inspection: **after a failed op the state is byte-identical to the pre-call
state**, not merely similar-looking.

What is deliberately NOT rolled back: attributes a backend names in
``_transaction_exclude``. An audit log / recorder must survive the rollback --
op success and failure are observable through the recorder, so restoring it
would erase the very record that the op was attempted.

Attribution: pattern (not code) from Roshera-CAD's geometry engine,
``roshera-backend/geometry-engine/src/operations/lifecycle.rs`` (validate_can_apply
pre-flight + with_rollback snapshot/restore, recorder excluded from the
snapshot, documented half-done-fillet / orphaned-face failure modes). Roshera-CAD
is licensed FSL-1.1 (Functional Source License, non-compete; Apache-2.0 future
license) -- NOT a permissive license, so nothing is copied: this is an
independent Python implementation of the documented facts.

Default-safe by construction: existing backends are untouched and keep working.
Transactionality is opt-in by WRAPPING a backend in :class:`TransactionalBackend`.
Pure stdlib, deterministic, no kernel.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import pickle
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence

from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.backends.base import ApplyResult


class RollbackFailed(RuntimeError):
    """Restoring a snapshot itself failed -- the state is NOT trustworthy.

    Raised only when the restore path breaks, which is the one case where the
    byte-equivalence promise cannot be kept. It carries the original error so
    the cause that triggered the rollback is not lost behind the rollback's own
    failure.
    """

    def __init__(self, message: str, original: Optional[BaseException] = None) -> None:
        self.original = original
        super().__init__(message)


# --------------------------------------------------------------------------- #
# state fingerprinting
# --------------------------------------------------------------------------- #

def _canonical(value: Any) -> Any:
    """A pickle-free canonical form: nested containers -> sorted/ordered tuples.

    The fallback for states holding objects pickle refuses (locks, modules, live
    kernel handles). Falls back to ``repr`` at the leaves, which is why the
    pickle path is preferred when it works.
    """
    if isinstance(value, dict):
        return ("dict", tuple(sorted(
            (repr(k), _canonical(v)) for k, v in value.items())))
    if isinstance(value, (list, tuple)):
        tag = "list" if isinstance(value, list) else "tuple"
        return (tag, tuple(_canonical(v) for v in value))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted(repr(_canonical(v)) for v in value)))
    if isinstance(value, (str, int, float, bool, bytes)) or value is None:
        return ("scalar", repr(value))
    if hasattr(value, "__dict__"):
        return ("obj", type(value).__name__, _canonical(vars(value)))
    return ("repr", repr(value))


def state_fingerprint(backend: object) -> str:
    """A digest of the backend's FULL mutable state (not just its model digest).

    ``GeometryBackend.state_digest`` hashes what a backend considers its model;
    this hashes everything the backend actually holds, so a rollback that
    restored the model but leaked a counter, an id sequence, or a stray cache
    entry is still caught. This is the byte-equivalence probe.

    Deterministic within a process: dicts preserve insertion order, and the
    ``_canonical`` fallback sorts by key.
    """
    state = _snapshot_source(backend)
    try:
        raw = pickle.dumps(state, protocol=4)
    except Exception:  # noqa: BLE001 - unpicklable state -> structural fallback
        raw = repr(_canonical(state)).encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()


def _excluded(backend: object) -> frozenset:
    """Attribute names the backend declares outside the transaction."""
    names = getattr(backend, "_transaction_exclude", ())
    if isinstance(names, str):
        names = (names,)
    return frozenset(str(n) for n in names)


def _snapshot_source(backend: object) -> Dict[str, Any]:
    """The transactional slice of ``backend.__dict__`` (excludes recorder etc.)."""
    skip = _excluded(backend)
    return {k: v for k, v in vars(backend).items() if k not in skip}


def snapshot_state(backend: object) -> Dict[str, Any]:
    """Deep-copy the backend's transactional state.

    Deep, not shallow: a shallow copy shares the nested dicts/lists a half-done
    op mutates in place, so restoring it would restore nothing.
    """
    try:
        return copy.deepcopy(_snapshot_source(backend))
    except Exception as exc:  # noqa: BLE001
        raise RollbackFailed(
            f"cannot snapshot {type(backend).__name__}: "
            f"{type(exc).__name__}: {exc}", exc) from exc


def restore_state(backend: object, snapshot: Dict[str, Any]) -> None:
    """Restore a snapshot taken by :func:`snapshot_state`.

    Attributes CREATED during the failed op are removed (a rollback that left
    them behind would not be byte-equivalent); excluded attributes are left
    exactly as the op left them.
    """
    skip = _excluded(backend)
    live = vars(backend)
    for key in [k for k in live if k not in skip and k not in snapshot]:
        del live[key]
    for key, value in snapshot.items():
        live[key] = copy.deepcopy(value)


# --------------------------------------------------------------------------- #
# the two primitives
# --------------------------------------------------------------------------- #

def preflight(backend: object, op: object) -> Optional[Diagnostic]:
    """Run the backend's cheap pre-flight, if it has one.

    Returns the blocking :class:`Diagnostic`, or ``None`` when the op may
    proceed. A backend without ``validate_can_apply`` always returns ``None`` --
    that is what keeps this default-safe.

    By design this does NOT run a full model validation: pre-flight is cheap, or
    its cost dominates every small op. Deep checks belong after the op.
    """
    check = getattr(backend, "validate_can_apply", None)
    if check is None:
        return None
    verdict = check(op)
    if verdict is None or verdict is True:
        return None
    if isinstance(verdict, Diagnostic):
        return verdict
    return Diagnostic(Severity.ERROR, "preflight", str(verdict), None)


@contextmanager
def with_rollback(backend: object) -> Iterator[Dict[str, Any]]:
    """Snapshot around a mutating op; restore the snapshot if the body raises.

    On a clean exit the snapshot is dropped and the mutation stands. On ANY
    exception the pre-op state is restored and the exception propagates
    unchanged -- the caller sees the real error, against an intact model.
    """
    snapshot = snapshot_state(backend)
    try:
        yield snapshot
    except BaseException as exc:
        try:
            restore_state(backend, snapshot)
        except Exception as restore_exc:  # noqa: BLE001
            raise RollbackFailed(
                f"rollback of {type(backend).__name__} failed after "
                f"{type(exc).__name__}: {restore_exc}", exc) from exc
        raise


# --------------------------------------------------------------------------- #
# the wrapper
# --------------------------------------------------------------------------- #

class TransactionalBackend:
    """Wraps any ``GeometryBackend`` so every ``apply`` is all-or-nothing.

    Delegates the whole :class:`~harnesscad.io.backends.base.GeometryBackend`
    protocol to ``inner`` and adds, around ``apply`` only:

      * pre-flight via ``inner.validate_can_apply`` (when present) -- refused
        before a snapshot is taken;
      * snapshot/restore, so a raising op or (with ``rollback_on_failure``) an
        ``ok=False`` op leaves ``inner`` byte-identical to its pre-call state.

    ``rollback_on_failure=True`` (the default) also rolls back ``ok=False``
    results. The protocol already asks backends to not mutate when they return
    ``ok=False``; this makes that ENFORCED rather than trusted -- a backend that
    half-mutates before noticing is corrected instead of believed.

    A raised exception still propagates: this changes the STATE on failure, not
    the failure itself.

    ``failures`` counts rolled-back ops -- an op that fails is not an op that
    never happened, and the count is what a caller reports.
    """

    #: The wrapper's own bookkeeping is not part of the inner model's state.
    _transaction_exclude = ("inner", "failures", "rollback_on_failure")

    def __init__(self, inner: object, rollback_on_failure: bool = True) -> None:
        self.inner = inner
        self.rollback_on_failure = bool(rollback_on_failure)
        self.failures = 0

    # -- the transactional entry point ------------------------------------- #
    def apply(self, op: object) -> ApplyResult:
        diag = preflight(self.inner, op)
        if diag is not None:
            self.failures += 1
            return ApplyResult(False, [], [diag])

        snapshot = snapshot_state(self.inner)
        try:
            result = self.inner.apply(op)
        except BaseException as exc:
            self._restore(snapshot, exc)
            self.failures += 1
            raise
        if self.rollback_on_failure and not getattr(result, "ok", False):
            self._restore(snapshot, None)
            self.failures += 1
        return result

    def _restore(self, snapshot: Dict[str, Any],
                 cause: Optional[BaseException]) -> None:
        try:
            restore_state(self.inner, snapshot)
        except Exception as exc:  # noqa: BLE001
            raise RollbackFailed(
                f"rollback of {type(self.inner).__name__} failed: {exc}",
                cause) from (cause or exc)

    def state_fingerprint(self) -> str:
        """Byte-equivalence probe over the WRAPPED backend's full state."""
        return state_fingerprint(self.inner)

    # -- plain delegation --------------------------------------------------- #
    def reset(self) -> None:
        self.failures = 0
        self.inner.reset()

    def regenerate(self) -> List[Diagnostic]:
        return self.inner.regenerate()

    def query(self, q: str) -> dict:
        return self.inner.query(q)

    def export(self, fmt: str):
        return self.inner.export(fmt)

    def state_digest(self) -> str:
        return self.inner.state_digest()

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes the wrapper does not define itself, so a
        # backend-specific extra (e.g. `solid_present`) stays reachable.
        return getattr(self.inner, name)


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

class _HalfDoneBackend:
    """A backend whose op corrupts state and THEN fails -- the half-done fillet.

    Deliberately the worst case: it mutates a nested structure in place, adds a
    brand-new attribute, bumps a counter, and only then raises (or returns
    ok=False). Nothing about it is polite.
    """

    _transaction_exclude = ("audit",)

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.faces = {"f1": {"edges": ["e1", "e2"], "tol": 1e-6}}
        self.counter = 0
        self.audit: List[str] = []

    def _corrupt(self, op: str) -> None:
        self.faces["f1"]["edges"].append("orphan")   # in-place nested mutation
        self.faces["f2"] = {"edges": [], "tol": 0.5}  # orphaned split face
        self.counter += 1
        self.scratch = "left behind"                  # brand-new attribute
        self.audit.append(op)

    def apply(self, op: str) -> ApplyResult:
        if op == "raise":
            self._corrupt(op)
            raise RuntimeError("boolean degeneracy")
        if op == "soft-fail":
            self._corrupt(op)
            return ApplyResult(False, [], [])
        self.faces[op] = {"edges": [], "tol": 1e-6}
        self.counter += 1
        self.audit.append(op)
        return ApplyResult(True, [op])

    def validate_can_apply(self, op: str) -> Optional[Diagnostic]:
        if op == "refused":
            return Diagnostic(Severity.ERROR, "preflight",
                              "radius exceeds local feasibility", None)
        return None

    def regenerate(self) -> List[Diagnostic]:
        return []

    def query(self, q: str) -> dict:
        return {"faces": len(self.faces)}

    def export(self, fmt: str):
        return ""

    def state_digest(self) -> str:
        return hashlib.sha256(
            repr(sorted(self.faces)).encode("utf-8")).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transactional kernel ops: pre-flight + snapshot rollback, "
                    "so a failed op leaves byte-equivalent state (pattern from "
                    "Roshera-CAD lifecycle.rs).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="prove byte-equivalence after a raising op, an "
                             "ok=False op, and a refused op, by comparing a "
                             "full-state digest before and after.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    # 0. The unprotected backend really is corrupted -- otherwise the rest of
    #    this selfcheck proves nothing.
    raw = _HalfDoneBackend()
    before_raw = state_fingerprint(raw)
    try:
        raw.apply("raise")
    except RuntimeError:
        pass
    assert state_fingerprint(raw) != before_raw, "control case did not corrupt"
    assert "f2" in raw.faces and "orphan" in raw.faces["f1"]["edges"]
    print("[selfcheck] control: unwrapped backend IS left corrupted "
          "(orphaned face f2, dangling edge)")

    # 1. Raising op under the wrapper -> byte-identical state.
    be = TransactionalBackend(_HalfDoneBackend())
    be.apply("f9")  # some real work first, so the snapshot is not the empty one
    before = be.state_fingerprint()
    before_digest = be.state_digest()
    try:
        be.apply("raise")
        raise AssertionError("exception did not propagate")
    except RuntimeError as exc:
        assert "boolean degeneracy" in str(exc)
    after = be.state_fingerprint()
    assert after == before, "state not byte-equivalent after a raising op"
    assert be.state_digest() == before_digest
    assert "f2" not in be.inner.faces
    assert be.inner.faces["f1"]["edges"] == ["e1", "e2"]
    assert not hasattr(be.inner, "scratch"), "op's new attribute survived"
    assert be.inner.counter == 1
    print(f"[selfcheck] raising op: full-state digest identical "
          f"({before[:16]}...), error still propagated")

    # 2. The recorder is EXCLUDED: the attempt stays on the record.
    assert be.inner.audit == ["f9", "raise"], be.inner.audit
    print("[selfcheck] excluded recorder survives rollback (attempt recorded)")

    # 3. ok=False op -> rolled back too (enforced, not trusted).
    before = be.state_fingerprint()
    res = be.apply("soft-fail")
    assert res.ok is False
    assert be.state_fingerprint() == before, "ok=False left mutations behind"
    print("[selfcheck] ok=False op: block-and-correct ENFORCED by rollback")

    # 4. Pre-flight refuses before any snapshot/mutation.
    before = be.state_fingerprint()
    res = be.apply("refused")
    assert res.ok is False and res.diagnostics[0].code == "preflight"
    assert be.state_fingerprint() == before
    print("[selfcheck] pre-flight refusal is gated before mutation")

    # 5. Opt-out is honoured: rollback_on_failure=False keeps ok=False mutations.
    loose = TransactionalBackend(_HalfDoneBackend(), rollback_on_failure=False)
    before = loose.state_fingerprint()
    loose.apply("soft-fail")
    assert loose.state_fingerprint() != before
    print("[selfcheck] rollback_on_failure=False opt-out honoured")

    # 6. Default-safe: a backend with no validate_can_apply is unaffected, and a
    #    succeeding op commits normally.
    class _Plain(_HalfDoneBackend):
        validate_can_apply = None

    plain = TransactionalBackend(_Plain())
    plain.inner.validate_can_apply = None
    assert preflight(object(), "anything") is None
    ok = plain.apply("f3")
    assert ok.ok and "f3" in plain.inner.faces
    assert plain.failures == 0
    print("[selfcheck] no-preflight backend unaffected; success commits")

    # 7. Failure count is real bookkeeping.
    assert be.failures == 3, be.failures
    print(f"[selfcheck] failures counted: {be.failures}")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
