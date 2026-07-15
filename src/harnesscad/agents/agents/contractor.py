"""Contractor -- recursive parent->child work delegation (hierarchical subcontracts).

The book's "Contractor" model (Ch.19, ``audit/book_agentic_design_patterns.md``)
is not a visible/hidden test split -- that is TDAD, and it already lives in
``domain/spec/contract_split.py``. The Contractor model is *hierarchical
delegation*: a parent holds a machine-verifiable acceptance spec, DECOMPOSES it
into subcontracts, DELEGATES each to a worker, VERIFIES each returned result
against the subcontract it was issued for, and only then ACCEPTS it -- and each
worker may itself be a contractor, so the decomposition recurses.

Mapped onto CAD, the acceptance spec is the Measured Geometric Contract (MGC,
``domain/spec/contract.py``). A :class:`Contractor` takes an assembly's MGC and
decomposes it into per-part contracts (or a complex part into sub-feature
contracts), delegates each to an injected child worker, and checks every returned
result against the sub-part contract it was issued for before accepting it. The
recursion is the point: a per-part subcontract may itself be an assembly, and the
same Contractor decomposes it one level further.

THREE DISCIPLINES, LOAD-BEARING
-------------------------------
* **Verify before accept.** A child result is accepted only after it is checked
  against its own subcontract. A result that fails verification does not become
  part of the assembly.
* **A failed subcontract fails the parent.** There is no silent acceptance: one
  REJECTED (or unresolved) child makes the whole parent contract REJECTED, all
  the way up to the root. A partially-built assembly is not a shipped assembly.
* **Negotiate, never guess.** A subcontract that still carries unbound /
  ``[NEEDS CLARIFICATION]`` predicates is surfaced back to the parent as
  INPUT_REQUIRED -- the magnitude the brief omitted is never invented to let the
  work proceed. This is the same anti-guess rule the MGC enforces, lifted to the
  delegation boundary.

A DELIBERATELY-UNWIRED LIBRARY
------------------------------
Like ``agents.supervisor``, this module is a runnable library, not a wired-in
route: the child worker, the decomposition and the verifier are all INJECTED, so
the Contractor hard-depends on no model and no geometry kernel. The real MGC
check is used lazily (``domain.spec.contract.check``) only when a caller hands in
a genuine MGC together with a measured result; its absence degrades to the
injected verifier and never breaks import. The ``--selfcheck`` CLI exercises a
synthetic multi-level decomposition on injected doubles -- no model, no kernel --
proving the parent->child->verify->accept path and the failing-subcontract-fails-
parent path.

Absolute imports under ``harnesscad.``, stdlib-only at import, deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ACCEPTED",
    "REJECTED",
    "INPUT_REQUIRED",
    "Subcontract",
    "Award",
    "ContractorReport",
    "Contractor",
    "award_contract",
    "main",
]

# --------------------------------------------------------------------------- #
# The three terminal statuses of a subcontract.
# --------------------------------------------------------------------------- #
#: The child result was verified against its subcontract and taken into the
#: assembly. For a composite node: every child was ACCEPTED.
ACCEPTED = "ACCEPTED"
#: Something is provably unacceptable -- a child result failed verification, a
#: child worker raised, or a descendant subcontract was itself REJECTED. A single
#: REJECTED child rejects the whole parent; nothing is silently accepted.
REJECTED = "REJECTED"
#: The subcontract still carries an unbound / ``[NEEDS CLARIFICATION]`` predicate.
#: It is surfaced back to the parent for negotiation -- never guessed past -- and,
#: because the work cannot proceed, it also blocks the parent from ACCEPTED.
INPUT_REQUIRED = "INPUT_REQUIRED"


# The two injected collaborators the Contractor is built against:
#
#   child_worker(subcontract) -> result
#       The leaf executor. ``result`` is anything the verifier can judge -- a
#       HarnessSession, a measured mapping, a mesh, a work record. Injected so
#       the Contractor hard-depends on no model and no kernel, exactly as
#       ``pdd.pipeline`` injects an executor.
#
#   verify(spec, result) -> (ok: bool, reasons: Sequence[str])
#       Checks a returned result against the subcontract it was issued for. The
#       default (:func:`_default_verify`) lazily uses the real MGC check.
#
#   decompose(spec) -> Sequence[Subcontract | spec]
#       Splits a contract into sub-part / sub-feature subcontracts. Returning
#       nothing marks the node a leaf, to be handed to ``child_worker``.
Worker = Callable[["Subcontract"], Any]
Verifier = Callable[[Any, Any], Tuple[bool, Sequence[str]]]
Decomposer = Callable[[Any], Sequence[Any]]


# --------------------------------------------------------------------------- #
# The unit of delegated work, and the record of how it was discharged.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Subcontract:
    """One unit of delegated work: an id, the acceptance spec, and a label.

    ``spec`` is the acceptance contract for this unit -- an MGC, or anything
    duck-typed onto the surface the Contractor reads (``part_id``, ``digest()``,
    ``unbound()``). ``label`` is a free-text role ("housing", "bore") kept only
    for the audit trail.
    """

    id: str
    spec: Any
    label: str = ""


@dataclass
class Award:
    """The record of how one subcontract was discharged -- recursively.

    ``status`` is one of :data:`ACCEPTED`, :data:`REJECTED`, :data:`INPUT_REQUIRED`.
    ``children`` are the awards of the sub-subcontracts a composite node was
    decomposed into (empty for a leaf). ``clarifications`` names the unbound
    predicate keys that forced INPUT_REQUIRED. ``result`` is the accepted work
    product (the child worker's output for a leaf, or the map of child results for
    a composite) and is ``None`` unless the node was ACCEPTED.
    """

    subcontract_id: str
    part_id: str
    digest: Optional[str]
    status: str
    children: Tuple["Award", ...] = ()
    reasons: Tuple[str, ...] = ()
    clarifications: Tuple[str, ...] = ()
    result: Any = None

    @property
    def accepted(self) -> bool:
        return self.status == ACCEPTED

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> Tuple["Award", ...]:
        """Every leaf award beneath (and including) this node, depth-first."""
        if self.is_leaf:
            return (self,)
        out: List[Award] = []
        for child in self.children:
            out.extend(child.leaves())
        return tuple(out)

    def as_dict(self) -> dict:
        return {
            "subcontract_id": self.subcontract_id,
            "part_id": self.part_id,
            "digest": self.digest,
            "status": self.status,
            "reasons": list(self.reasons),
            "clarifications": list(self.clarifications),
            "children": [c.as_dict() for c in self.children],
        }


@dataclass
class ContractorReport:
    """The single structured result of a Contractor run -- the whole award tree."""

    root: Award

    @property
    def status(self) -> str:
        return self.root.status

    @property
    def accepted(self) -> bool:
        """True only when the root subcontract was ACCEPTED (every leaf verified)."""
        return self.root.status == ACCEPTED

    @property
    def clarifications(self) -> Tuple[str, ...]:
        """Every outstanding ``[NEEDS CLARIFICATION]`` key across the whole tree."""
        seen: List[str] = []
        for award in _walk(self.root):
            for key in award.clarifications:
                if key not in seen:
                    seen.append(key)
        return tuple(seen)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "accepted": self.accepted,
            "clarifications": list(self.clarifications),
            "root": self.root.as_dict(),
        }


# --------------------------------------------------------------------------- #
# The Contractor
# --------------------------------------------------------------------------- #


class Contractor:
    """Recursively decomposes a contract, delegates, verifies, and accepts.

    Only the child worker *needs* injecting (it does the leaf work). ``decompose``
    defaults to "everything is a leaf" (no decomposition); ``verify`` defaults to
    :func:`_default_verify`, which lazily uses the real MGC check when it can and
    otherwise reads an explicit ok/measurement off the result. ``max_depth``
    guards against a decomposition that never bottoms out.
    """

    def __init__(
        self,
        child_worker: Worker,
        *,
        decompose: Optional[Decomposer] = None,
        verify: Optional[Verifier] = None,
        max_depth: int = 8,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self._child_worker = child_worker
        self._decompose = decompose
        self._verify = verify or _default_verify
        self._max_depth = int(max_depth)

    def run(self, top: Any, *, subcontract_id: str = "root") -> ContractorReport:
        """Award the top-level contract and return the whole award tree."""
        sub = _as_subcontract(top, subcontract_id)
        return ContractorReport(root=self._award(sub, depth=0))

    # -- the recursive core ------------------------------------------------- #

    def _award(self, sub: Subcontract, depth: int) -> Award:
        spec = sub.spec
        part_id = _spec_part_id(spec) or sub.id
        digest = _spec_digest(spec)

        def award(status: str, **kw: Any) -> Award:
            return Award(sub.id, part_id, digest, status, **kw)

        # Guard: a decomposition that does not bottom out is a defect, not a part.
        if depth > self._max_depth:
            return award(
                REJECTED,
                reasons=("max decomposition depth %d exceeded" % self._max_depth,),
            )

        # 1) Negotiate, never guess. An unbound / [NEEDS CLARIFICATION] predicate
        #    is surfaced back to the parent -- the work does not proceed on a
        #    guessed magnitude.
        clar = _unbound_keys(spec)
        if clar:
            return award(
                INPUT_REQUIRED,
                clarifications=tuple(clar),
                reasons=(
                    "subcontract carries unbound predicate(s): " + ", ".join(clar),
                ),
            )

        # 2) Decompose into child subcontracts (assembly -> parts, part ->
        #    features). A decomposer that raises fails this node -- a contract we
        #    cannot split is not a contract we may silently accept.
        try:
            children = self._children_of(spec)
        except Exception as exc:  # noqa: BLE001 -- a broken split is a real defect
            return award(
                REJECTED,
                reasons=(
                    "decompose raised %s: %s" % (type(exc).__name__, exc),
                ),
            )

        if children:
            child_awards = tuple(self._award(c, depth + 1) for c in children)
            status, reasons = _fold_children(child_awards)
            clarifications = _collect_clarifications(child_awards)
            result = None
            if status == ACCEPTED:
                # Every child verified -> the assembled result is the map of the
                # accepted child products, keyed by subcontract id.
                result = {a.subcontract_id: a.result for a in child_awards}
            return award(
                status,
                children=child_awards,
                reasons=reasons,
                clarifications=clarifications,
                result=result,
            )

        # 3) Leaf: DELEGATE to the injected child worker.
        try:
            work = self._child_worker(sub)
        except Exception as exc:  # noqa: BLE001 -- a raising worker built nothing
            return award(
                REJECTED,
                reasons=(
                    "child worker raised %s: %s" % (type(exc).__name__, exc),
                ),
            )

        # ...then VERIFY the returned result against THIS subcontract before
        # accepting it. Verification is not optional and its failure is not soft.
        try:
            ok, vreasons = self._verify(spec, work)
        except Exception as exc:  # noqa: BLE001 -- a raising verifier cannot certify
            return award(
                REJECTED,
                reasons=(
                    "verifier raised %s: %s" % (type(exc).__name__, exc),
                ),
            )
        if not ok:
            return award(
                REJECTED,
                reasons=tuple(vreasons) or ("child result failed verification",),
            )
        return award(ACCEPTED, result=work)

    def _children_of(self, spec: Any) -> Tuple[Subcontract, ...]:
        if self._decompose is None:
            return ()
        items = self._decompose(spec) or ()
        out: List[Subcontract] = []
        for index, item in enumerate(items):
            out.append(_as_subcontract(item, "child-%d" % index))
        return tuple(out)


# --------------------------------------------------------------------------- #
# Module-level convenience (mirrors run_pdd's free function).
# --------------------------------------------------------------------------- #


def award_contract(
    top: Any,
    child_worker: Worker,
    *,
    decompose: Optional[Decomposer] = None,
    verify: Optional[Verifier] = None,
    max_depth: int = 8,
    subcontract_id: str = "root",
) -> ContractorReport:
    """Run one Contractor over ``top`` and return its :class:`ContractorReport`.

    Args:
        top: the top-level contract (an MGC or duck-typed spec) or a
            :class:`Subcontract`.
        child_worker: ``child_worker(subcontract) -> result`` leaf executor.
            Injected so the Contractor hard-depends on no model and no kernel.
        decompose: ``decompose(spec) -> Sequence[Subcontract | spec]``. When
            omitted, every node is a leaf (no decomposition).
        verify: ``verify(spec, result) -> (ok, reasons)``. When omitted,
            :func:`_default_verify` is used (lazy MGC check, else ok/measurement).
        max_depth: recursion guard for the decomposition.
        subcontract_id: id stamped on the root award.
    """
    contractor = Contractor(
        child_worker, decompose=decompose, verify=verify, max_depth=max_depth
    )
    return contractor.run(top, subcontract_id=subcontract_id)


# --------------------------------------------------------------------------- #
# Folding a composite node's children into one status.
# --------------------------------------------------------------------------- #


def _fold_children(awards: Sequence[Award]) -> Tuple[str, Tuple[str, ...]]:
    """Fold child awards into the parent's status -- one bad child fails the parent.

    REJECTED dominates (something is provably wrong), then INPUT_REQUIRED (a
    child could not proceed without negotiation), and only an all-ACCEPTED set
    of children yields ACCEPTED. Nothing is silently accepted.
    """
    rejected = [a for a in awards if a.status == REJECTED]
    if rejected:
        return REJECTED, tuple(
            "subcontract %s REJECTED: %s"
            % (a.subcontract_id, "; ".join(a.reasons) or "no reason given")
            for a in rejected
        )
    input_required = [a for a in awards if a.status == INPUT_REQUIRED]
    if input_required:
        return INPUT_REQUIRED, tuple(
            "subcontract %s needs input: %s"
            % (a.subcontract_id, ", ".join(a.clarifications) or "unspecified")
            for a in input_required
        )
    return ACCEPTED, ()


def _collect_clarifications(awards: Sequence[Award]) -> Tuple[str, ...]:
    seen: List[str] = []
    for award in awards:
        for key in award.clarifications:
            if key not in seen:
                seen.append(key)
    return tuple(seen)


def _walk(award: Award) -> List[Award]:
    out = [award]
    for child in award.children:
        out.extend(_walk(child))
    return out


# --------------------------------------------------------------------------- #
# The default verifier + lazy-import / duck-typing helpers.
# --------------------------------------------------------------------------- #


def _default_verify(spec: Any, result: Any) -> Tuple[bool, Tuple[str, ...]]:
    """Check a returned result against its subcontract -- the accept decision.

    Layered, most-specific first, so it works with a real MGC and with plain
    injected doubles alike:

    1. A ``None`` result built nothing -> never accepted.
    2. A result whose ``ok`` attribute is ``False`` reports its own failure.
    3. A real MGC spec plus a measured result -> the actual MGC check runs
       (lazily imported); every bound MEASURED predicate must pass.
    4. Otherwise accept only an explicit ``ok is True``; refuse when there is
       nothing to verify against (an unverifiable result is not an accepted one).
    """
    if result is None:
        return False, ("child produced no result",)

    ok_attr = getattr(result, "ok", None)
    if ok_attr is False:
        return False, tuple(_result_reasons(result)) or ("child result reports not-ok",)

    measurement = _measurement_of(result)
    if measurement is not None and _looks_like_contract(spec):
        report = _check_contract(spec, measurement)
        if report is not None:
            if getattr(report, "satisfied", False):
                return True, ()
            reasons: List[str] = []
            fails = [_result_key(r) for r in _call0(report, "failures") or ()]
            missing = [_result_key(r) for r in _call0(report, "missing") or ()]
            if fails:
                reasons.append("failed predicate(s): " + ", ".join(fails))
            if missing:
                reasons.append("missing measurement(s): " + ", ".join(missing))
            if not reasons:
                reasons.append("contract not satisfied")
            return False, tuple(reasons)

    if ok_attr is True:
        return True, ()
    return False, (
        "no verifier could confirm the child result "
        "(no MGC measurement and no ok flag)",
    )


def _lazy(module: str) -> Optional[Any]:
    """Import a sibling module lazily; return None only if it is genuinely absent."""
    try:
        import importlib

        return importlib.import_module(module)
    except ImportError:
        return None


def _check_contract(spec: Any, measurement: Mapping[str, Any]) -> Optional[Any]:
    """Run the real MGC check if the contract module is importable, else None."""
    mod = _lazy("harnesscad.domain.spec.contract")
    if mod is None or not hasattr(mod, "check"):
        return None
    try:
        return mod.check(spec, measurement)
    except Exception:  # noqa: BLE001 -- a raising check does not certify
        return None


def _looks_like_contract(obj: Any) -> bool:
    return (
        hasattr(obj, "predicates")
        and hasattr(obj, "measured")
        and hasattr(obj, "unbound")
    )


def _measurement_of(result: Any) -> Optional[Mapping[str, Any]]:
    """The contract-keyed measurement carried by a result, if any."""
    if isinstance(result, Mapping):
        inner = result.get("measurement")
        if isinstance(inner, Mapping):
            return inner
        return result
    measurement = getattr(result, "measurement", None)
    if isinstance(measurement, Mapping):
        return measurement
    return None


def _result_reasons(result: Any) -> Tuple[str, ...]:
    reasons = getattr(result, "reasons", None)
    if isinstance(reasons, (list, tuple)):
        return tuple(str(r) for r in reasons)
    return ()


def _spec_part_id(spec: Any) -> str:
    value = getattr(spec, "part_id", None)
    return str(value) if value else ""


def _spec_digest(spec: Any) -> Optional[str]:
    fn = getattr(spec, "digest", None)
    if callable(fn):
        try:
            return str(fn())
        except Exception:  # noqa: BLE001
            return None
    if isinstance(fn, str):
        return fn
    return None


def _unbound_keys(spec: Any) -> List[str]:
    """The keys of the spec's unbound / [NEEDS CLARIFICATION] predicates."""
    fn = getattr(spec, "unbound", None)
    preds: Sequence[Any] = ()
    if callable(fn):
        try:
            preds = fn() or ()
        except Exception:  # noqa: BLE001
            preds = ()
    elif isinstance(fn, (list, tuple)):
        preds = fn
    return [str(getattr(p, "key", p)) for p in preds]


def _as_subcontract(obj: Any, default_id: str) -> Subcontract:
    if isinstance(obj, Subcontract):
        return obj
    part_id = _spec_part_id(obj) or default_id
    return Subcontract(id=part_id, spec=obj)


def _result_key(result: Any) -> str:
    pred = getattr(result, "predicate", None)
    if pred is not None:
        return str(getattr(pred, "key", pred))
    return str(getattr(result, "key", result))


def _call0(obj: Any, method: str) -> Any:
    fn = getattr(obj, method, None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# CLI -- including --selfcheck on injected synthetic doubles (no kernel, no model)
# --------------------------------------------------------------------------- #


def _synthetic_accept() -> ContractorReport:
    """The parent->child->verify->accept path on injected doubles.

    A "gearbox" assembly decomposes into two parts; one of them ("housing") is
    itself an assembly decomposed one level further into two features -- a 2-level
    (assembly -> part -> feature) recursion. Every leaf worker returns a
    measurement that satisfies its subcontract, so every subcontract verifies and
    the root is ACCEPTED. Fully self-contained: no sibling module is consulted."""
    gearbox = _fake_tree()

    def child_worker(sub: Subcontract) -> "_FakeResult":
        # A faithful worker: measure exactly what the subcontract asks for.
        return _FakeResult(measurement=_targets(sub.spec))

    return award_contract(
        gearbox, child_worker, decompose=_fake_decompose, verify=_fake_verify
    )


def _synthetic_reject() -> ContractorReport:
    """The failing-subcontract-fails-parent path.

    Identical decomposition, but the worker corrupts the "bore" feature's
    measurement. That leaf fails verification (REJECTED), which rejects its
    parent "housing", which rejects the root "gearbox" -- proving a single failed
    subcontract fails the whole parent, with nothing silently accepted."""
    gearbox = _fake_tree()

    def child_worker(sub: Subcontract) -> "_FakeResult":
        measurement = _targets(sub.spec)
        if sub.id == "bore":
            # Bore the hole to the wrong diameter: a provably-wrong result.
            measurement = dict(measurement)
            measurement["hole_diameter_mm"] = 99.0
        return _FakeResult(measurement=measurement)

    return award_contract(
        gearbox, child_worker, decompose=_fake_decompose, verify=_fake_verify
    )


def _synthetic_input_required() -> ContractorReport:
    """The negotiate-never-guess path.

    The "shaft" part's subcontract carries an unbound predicate (the brief never
    stated its length), so it is surfaced as INPUT_REQUIRED and the work does not
    proceed on a guessed magnitude -- which also blocks the root from ACCEPTED."""
    gearbox = _fake_tree(shaft_unbound=True)

    def child_worker(sub: Subcontract) -> "_FakeResult":
        return _FakeResult(measurement=_targets(sub.spec))

    return award_contract(
        gearbox, child_worker, decompose=_fake_decompose, verify=_fake_verify
    )


def _fake_tree(shaft_unbound: bool = False) -> "_FakeContract":
    """A synthetic assembly: gearbox -> {housing -> {wall, bore}, shaft}."""
    wall = _FakeContract(
        "wall",
        predicates=(_FakePredicate("min_wall_mm", 2.0),),
    )
    bore = _FakeContract(
        "bore",
        predicates=(
            _FakePredicate("hole_count", 1),
            _FakePredicate("hole_diameter_mm", 8.0),
        ),
    )
    housing = _FakeContract("housing", children=(wall, bore))
    shaft_preds = (_FakePredicate("volume_mm3", 500.0),)
    if shaft_unbound:
        shaft_preds = shaft_preds + (
            _FakePredicate("length_mm", None, unbound=True),
        )
    shaft = _FakeContract("shaft", predicates=shaft_preds)
    return _FakeContract("gearbox", children=(housing, shaft))


def _fake_decompose(spec: "_FakeContract") -> Tuple[Subcontract, ...]:
    """Decompose a synthetic contract by its declared children (empty -> leaf)."""
    return tuple(
        Subcontract(id=child.part_id, spec=child, label=child.part_id)
        for child in spec.children
    )


def _fake_verify(spec: "_FakeContract", result: "_FakeResult") -> Tuple[bool, Tuple[str, ...]]:
    """Check a synthetic result against every bound predicate of its subcontract."""
    if result is None:
        return False, ("no result",)
    measurement = result.measurement
    reasons: List[str] = []
    for pred in spec.measured():
        if pred.key not in measurement:
            reasons.append("missing " + pred.key)
        elif measurement[pred.key] != pred.target:
            reasons.append(
                "wrong %s (got %r, want %r)"
                % (pred.key, measurement[pred.key], pred.target)
            )
    if reasons:
        return False, tuple(reasons)
    return True, ()


def _targets(spec: "_FakeContract") -> Dict[str, Any]:
    return {pred.key: pred.target for pred in spec.measured()}


@dataclass(frozen=True)
class _FakePredicate:
    """A stand-in contract predicate, duck-typed onto .key/.target/.kind/.unbound."""

    key: str
    target: Any
    unbound: bool = False

    @property
    def kind(self) -> str:
        return "measured"


@dataclass
class _FakeContract:
    """A stand-in contract/subcontract spec, duck-typed onto what the Contractor reads."""

    part_id: str
    predicates: Tuple[_FakePredicate, ...] = ()
    children: Tuple["_FakeContract", ...] = ()

    def measured(self) -> Tuple[_FakePredicate, ...]:
        return tuple(p for p in self.predicates if not p.unbound)

    def unbound(self) -> Tuple[_FakePredicate, ...]:
        return tuple(p for p in self.predicates if p.unbound)

    def digest(self) -> str:
        return "fake:" + self.part_id


@dataclass
class _FakeResult:
    """A stand-in work product carrying the measurement the verifier judges."""

    measurement: Dict[str, Any] = field(default_factory=dict)


def _print_report(label: str, report: ContractorReport) -> None:
    print(
        "[%s] status=%s accepted=%s"
        % (label, report.status, report.accepted)
    )
    for award in _walk(report.root):
        indent = "    "
        print(
            "%s- %s (%s) %s"
            % (indent, award.subcontract_id, award.part_id, award.status)
        )
        for reason in award.reasons:
            print("%s    reason: %s" % (indent, reason))
        if award.clarifications:
            print(
                "%s    [NEEDS CLARIFICATION]: %s"
                % (indent, ", ".join(award.clarifications))
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` runs a synthetic multi-level decomposition
    on injected doubles (no kernel, no model), demonstrating an ACCEPTED path, a
    REJECTED-because-a-subcontract-failed path, and an INPUT_REQUIRED path."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.agents.agents.contractor",
        description="Contractor: recursive parent->child work delegation "
        "(hierarchical subcontracts) over Measured Geometric Contracts.",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="run the Contractor on injected synthetic doubles (no kernel/model) "
        "and print an ACCEPTED, a REJECTED, and an INPUT_REQUIRED award tree.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the award tree(s) as JSON."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    accepted = _synthetic_accept()
    rejected = _synthetic_reject()
    input_required = _synthetic_input_required()

    if args.json:
        print(
            json.dumps(
                {
                    "accept_path": accepted.as_dict(),
                    "reject_path": rejected.as_dict(),
                    "input_required_path": input_required.as_dict(),
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
    else:
        _print_report("ACCEPT path", accepted)
        _print_report("REJECT path", rejected)
        _print_report("INPUT_REQUIRED path", input_required)

    # Assert the three demonstrated invariants so a regression fails loudly rather
    # than printing a wrong award.
    ok = (
        accepted.status == ACCEPTED
        and accepted.accepted
        and rejected.status == REJECTED
        and not rejected.accepted
        and input_required.status == INPUT_REQUIRED
        and "length_mm" in input_required.clarifications
    )
    if not ok:
        print(
            "SELFCHECK FAILED: expected ACCEPTED/REJECTED/INPUT_REQUIRED, got "
            "%s/%s/%s" % (accepted.status, rejected.status, input_required.status),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
