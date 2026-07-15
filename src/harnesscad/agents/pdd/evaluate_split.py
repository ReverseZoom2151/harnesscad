"""Evaluate a submission against the VISIBLE and the HIDDEN halves of its MGC.

This is the anti-gaming evaluator of Parts-Driven Development (see
``audit/pdd_synthesis.md``, the TDAD / Kitchen-Loop rung). The Measured
Geometric Contract (MGC) is the Specify-phase answer key; a generator that can
read every acceptance criterion can satisfy the LETTER of each one without
producing the intended part -- the MGC's many-to-one residual made exploitable.
The fix both TDAD (Rehan, arXiv 2603.08806) and the Kitchen Loop (Roy, arXiv
2603.25697) converge on is to HOLD PART OF THE CONTRACT BACK: a visible set
shapes the part, a hidden set the generator never saw scores whether that part
GENERALISES to criteria it could not have tuned toward.

WHAT THIS DOES
--------------
Given a brief (or an already-compiled MGC) and a measurement of a submission:

1. compile the brief into an MGC (:func:`harnesscad.domain.spec.contract.compile_contract`);
2. split it deterministically into a ``visible_contract`` and a ``hidden_contract``
   (:func:`harnesscad.domain.spec.contract_split.split_contract`);
3. check the SAME measurement against BOTH halves
   (:func:`harnesscad.domain.spec.contract.check`);
4. report the GENERALISATION GAP -- a submission that satisfies the visible
   contract but FAILS the hidden one gamed the visible contract: it hit the
   criteria it was shown and missed the ones held back.

A hidden contract with no bound MEASURED predicate is NOT evaluable -- there is
nothing to generalise to -- and no gap is ever claimed from it (the empty-gate
case must not read as a failure).

NO KERNEL, NO MODEL
-------------------
Everything below the contract layer is pure Python. This module runs no model
and touches no geometry kernel: it grades a measurement that already exists.
The ``contract`` and ``contract_split`` collaborators are imported LAZILY, so
importing this module never forces them to exist; the ``--selfcheck`` exercises
the whole bridge on a synthetic brief and synthetic measurements with neither.

Absolute imports under ``harnesscad.``, stdlib-only at module import, deterministic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "GAMING_NOTE",
    "SplitEvaluation",
    "evaluate_split",
    "measurement_from_contract",
    "render",
    "main",
]

#: The honest framing, printed on every report. Mirrors ``contract_split.py``.
GAMING_NOTE = (
    "A generator that can read every acceptance criterion can satisfy the "
    "letter of each without producing the intended part. Holding predicates "
    "back forces generalisation: passing the visible contract but FAILING the "
    "hidden one is the model gaming the visible contract. A hidden contract "
    "with no bound MEASURED predicate is not evaluable -- no gap is claimed."
)

# The gating predicate statuses: a bound MEASURED predicate that was actually
# checked resolves to one of these. UNBOUND / ADVISORY are non-gating.
_GATING_STATUSES = ("PASS", "FAIL", "MISSING")


@dataclass
class SplitEvaluation:
    """One submission judged against both halves of its MGC.

    ``visible_satisfied`` / ``hidden_satisfied`` are the contract verdicts on the
    two halves. ``generalization_gap`` is the headline: the visible contract
    passes and the (evaluable) hidden contract fails -- the model satisfied what
    it was shown and missed what was held back. ``hidden_evaluable`` is ``False``
    when the split left no bound MEASURED predicate in the hidden half, in which
    case no gap is ever claimed.
    """

    part_id: str
    contract_digest: str = ""
    hidden_fraction: float = 0.0
    n_visible_gating: int = 0
    n_hidden_gating: int = 0
    visible_satisfied: bool = False
    hidden_satisfied: bool = False
    hidden_evaluable: bool = False
    generalization_gap: bool = False
    visible_failed: List[str] = field(default_factory=list)
    visible_missing: List[str] = field(default_factory=list)
    visible_clarifications: List[str] = field(default_factory=list)
    hidden_failed: List[str] = field(default_factory=list)
    hidden_missing: List[str] = field(default_factory=list)
    hidden_clarifications: List[str] = field(default_factory=list)
    hidden_keys: List[str] = field(default_factory=list)
    note: str = GAMING_NOTE

    def to_dict(self) -> dict:
        return {
            "part_id": self.part_id,
            "contract_digest": self.contract_digest,
            "hidden_fraction": self.hidden_fraction,
            "n_visible_gating": self.n_visible_gating,
            "n_hidden_gating": self.n_hidden_gating,
            "visible_satisfied": self.visible_satisfied,
            "hidden_satisfied": self.hidden_satisfied,
            "hidden_evaluable": self.hidden_evaluable,
            "generalization_gap": self.generalization_gap,
            "visible_failed": list(self.visible_failed),
            "visible_missing": list(self.visible_missing),
            "visible_clarifications": list(self.visible_clarifications),
            "hidden_failed": list(self.hidden_failed),
            "hidden_missing": list(self.hidden_missing),
            "hidden_clarifications": list(self.hidden_clarifications),
            "hidden_keys": list(self.hidden_keys),
            "note": self.note,
        }


def evaluate_split(
    brief: Any,
    measurement: Mapping[str, Any],
    *,
    hidden_fraction: float = 0.4,
    seed_key: str = "part_id",
    compile_fn: Optional[Callable[[Any], Any]] = None,
    split_fn: Optional[Callable[..., Tuple[Any, Any]]] = None,
    check_fn: Optional[Callable[[Any, Mapping[str, Any]], Any]] = None,
) -> SplitEvaluation:
    """Compile, split, and grade a submission against both halves of its MGC.

    Args:
        brief: a part brief (mapping / ``PartSpec`` / ``CADBrief`` -- anything
            ``compile_contract`` accepts) OR an already-compiled MGC (duck-typed
            on ``.predicates`` + ``.digest``), in which case it is split directly.
        measurement: a mapping keyed by contract predicate keys (``volume_mm3``,
            ``bbox_mm``, ``genus``, ...) -- the submission's measured geometry.
        hidden_fraction: the fraction of eligible predicates held back.
        seed_key: the contract attribute that seeds the deterministic split.
        compile_fn / split_fn / check_fn: optional injected collaborators (for
            tests / doubles). When omitted the real ``contract`` /
            ``contract_split`` functions are imported lazily.

    Returns a :class:`SplitEvaluation`. Runs no model and no kernel.
    """
    contract = _resolve_contract(brief, compile_fn)
    split = split_fn or _lazy_split()
    check = check_fn or _lazy_check()

    visible, hidden = split(
        contract, hidden_fraction=hidden_fraction, seed_key=seed_key
    )
    visible_report = check(visible, measurement)
    hidden_report = check(hidden, measurement)

    return _build_evaluation(
        contract, hidden, hidden_fraction, visible_report, hidden_report
    )


def measurement_from_contract(contract: Any) -> dict:
    """Derive a FAITHFUL measurement from a contract's bound MEASURED targets.

    Reads each bound MEASURED predicate's ``target`` straight into a measurement
    keyed by predicate key, so the result satisfies every gating predicate by
    construction. Used by the self-check to build a submission that is correct on
    the whole contract before selectively corrupting the hidden half. Advisory
    and unbound predicates carry no target and are skipped.
    """
    out: dict = {}
    for pred in _bound_measured(contract):
        out[str(getattr(pred, "key", ""))] = getattr(pred, "target", None)
    return out


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #
def render(evaluation: SplitEvaluation) -> str:
    """A split evaluation as text: the two verdicts and the generalisation gap."""
    lines: List[str] = []
    lines.append("MGC HIDDEN-SPLIT EVALUATION -- visible vs held-out contract")
    lines.append("=" * 72)
    lines.append("part:   %s" % (evaluation.part_id or "?"))
    lines.append("digest: %s" % (evaluation.contract_digest or "?"))
    lines.append("hidden_fraction=%.2f  visible_gates=%d  hidden_gates=%d"
                 % (evaluation.hidden_fraction, evaluation.n_visible_gating,
                    evaluation.n_hidden_gating))
    lines.append("")
    lines.append("visible contract: %s"
                 % ("SATISFIED" if evaluation.visible_satisfied else "FAILED"))
    if evaluation.visible_failed:
        lines.append("    visible fails: " + ", ".join(sorted(evaluation.visible_failed)))
    if evaluation.visible_missing:
        lines.append("    visible missing: " + ", ".join(sorted(evaluation.visible_missing)))
    if not evaluation.hidden_evaluable:
        lines.append("hidden contract:  not evaluable (no bound MEASURED predicate held back)")
    else:
        lines.append("hidden contract:  %s"
                     % ("SATISFIED" if evaluation.hidden_satisfied else "FAILED"))
        if evaluation.hidden_failed:
            lines.append("    hidden fails: " + ", ".join(sorted(evaluation.hidden_failed)))
        if evaluation.hidden_missing:
            lines.append("    hidden missing: " + ", ".join(sorted(evaluation.hidden_missing)))
    lines.append("held-out keys: " + (", ".join(evaluation.hidden_keys) or "(none)"))
    lines.append("")
    lines.append("GENERALISATION GAP: %s"
                 % ("YES -- gamed the visible contract" if evaluation.generalization_gap
                    else "no"))
    lines.append("")
    lines.append(GAMING_NOTE)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Internals (pure, defensive; no kernel imports).
# --------------------------------------------------------------------------- #
def _build_evaluation(
    contract: Any,
    hidden_contract: Any,
    hidden_fraction: float,
    visible_report: Any,
    hidden_report: Any,
) -> SplitEvaluation:
    visible_gating = _gating_results(visible_report)
    hidden_gating = _gating_results(hidden_report)
    hidden_evaluable = len(hidden_gating) > 0

    visible_satisfied = bool(_attr(visible_report, "satisfied", False))
    hidden_satisfied = bool(_attr(hidden_report, "satisfied", False))
    # The gap is only meaningful when there is a hidden gate to fail; an empty
    # hidden contract reads as unsatisfied by MGC semantics, which must NOT be
    # mistaken for a submission gaming the visible half.
    gap = bool(visible_satisfied and hidden_evaluable and not hidden_satisfied)

    return SplitEvaluation(
        part_id=str(_attr(contract, "part_id", "") or ""),
        contract_digest=str(_call0(contract, "digest") or ""),
        hidden_fraction=float(hidden_fraction),
        n_visible_gating=len(visible_gating),
        n_hidden_gating=len(hidden_gating),
        visible_satisfied=visible_satisfied,
        hidden_satisfied=hidden_satisfied,
        hidden_evaluable=hidden_evaluable,
        generalization_gap=gap,
        visible_failed=_result_keys(visible_report, "failures"),
        visible_missing=_result_keys(visible_report, "missing"),
        visible_clarifications=_result_keys(visible_report, "clarifications"),
        hidden_failed=_result_keys(hidden_report, "failures"),
        hidden_missing=_result_keys(hidden_report, "missing"),
        hidden_clarifications=_result_keys(hidden_report, "clarifications"),
        hidden_keys=[str(getattr(p, "key", "")) for p in _bound_measured(hidden_contract)],
    )


def _resolve_contract(brief: Any, compile_fn: Optional[Callable[[Any], Any]]) -> Any:
    if _looks_like_contract(brief):
        return brief
    compile_fn = compile_fn or _lazy_compile()
    return compile_fn(brief)


def _looks_like_contract(obj: Any) -> bool:
    return hasattr(obj, "predicates") and hasattr(obj, "digest")


def _bound_measured(contract: Any) -> List[Any]:
    """Bound MEASURED predicates of a contract, duck-typed.

    Prefers the contract's own ``measured()`` accessor; falls back to filtering
    ``predicates`` by kind and the ``unbound`` flag for foreign / double types.
    """
    accessor = getattr(contract, "measured", None)
    if callable(accessor):
        try:
            return list(accessor())
        except Exception:  # noqa: BLE001
            pass
    out: List[Any] = []
    for pred in getattr(contract, "predicates", ()) or ():
        if _is_measured_kind(pred) and not bool(getattr(pred, "unbound", False)):
            out.append(pred)
    return out


def _is_measured_kind(pred: Any) -> bool:
    kind = getattr(pred, "kind", None)
    name = getattr(kind, "name", None)
    if name is None:
        name = str(kind)
    return name.strip().upper() == "MEASURED"


def _gating_results(report: Any) -> List[Any]:
    """Results of a report that gated -- a bound MEASURED predicate got a verdict."""
    out: List[Any] = []
    for r in getattr(report, "results", ()) or ():
        if str(getattr(r, "status", "")).upper() in _GATING_STATUSES:
            out.append(r)
    return out


def _result_keys(report: Any, method: str) -> List[str]:
    """The predicate keys behind a report accessor (``failures``/``missing``/...)."""
    items = _call0(report, method) or ()
    keys: List[str] = []
    for r in items:
        pred = getattr(r, "predicate", None)
        if pred is not None:
            keys.append(str(getattr(pred, "key", pred)))
        else:
            keys.append(str(getattr(r, "key", r)))
    return keys


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _call0(obj: Any, method: str) -> Any:
    fn = getattr(obj, method, None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _lazy_compile() -> Callable[[Any], Any]:
    from harnesscad.domain.spec.contract import compile_contract
    return compile_contract


def _lazy_split() -> Callable[..., Tuple[Any, Any]]:
    from harnesscad.domain.spec.contract_split import split_contract
    return split_contract


def _lazy_check() -> Callable[[Any, Mapping[str, Any]], Any]:
    from harnesscad.domain.spec.contract import check
    return check


# --------------------------------------------------------------------------- #
# Self-check: a synthetic brief and synthetic measurements, no kernel, no model.
# --------------------------------------------------------------------------- #
def _corrupt(value: Any) -> Any:
    """A wrong-but-same-shape value: guaranteed to fail its predicate.

    Flips a bool, bumps an int past exact equality, shifts a float well beyond
    any plausible tolerance, and corrupts each element of a tuple/list target.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + max(1.0, abs(value) * 0.5) + 1.0
    if isinstance(value, (tuple, list)):
        return tuple(_corrupt(v) for v in value)
    return value


def _selfcheck() -> int:
    """Split a synthetic MGC and demonstrate the generalisation gap, kernel-free.

    Compiles a real MGC from a synthetic brief (the ``contract`` module is pure
    Python, so this touches no kernel and runs no model), splits it, then grades
    two synthetic submissions against both halves:

    * a FAITHFUL submission (correct on every predicate) -- passes visible AND
      hidden, so there is NO gap;
    * a GAMED submission (correct on the visible predicates, wrong on exactly the
      held-out ones) -- passes visible and FAILS hidden, so the gap is caught.

    The gamed submission is built by corrupting precisely the keys the split held
    back, so the demonstration holds no matter which predicates the deterministic
    hash selects.
    """
    compile_contract = _lazy_compile()
    split_contract = _lazy_split()

    brief = {
        "part_id": "selfcheck-plate",
        "width": 80.0,
        "depth": 40.0,
        "height": 12.0,
        "kind": "plate",
        "holes": 2,
        "hole_diameter": 6.0,
        "wall": 3.0,
        "material": "aluminum",
        "intent": "synthetic hidden-split self-check plate",
    }
    hidden_fraction = 0.5

    contract = compile_contract(brief)
    _visible, hidden = split_contract(contract, hidden_fraction=hidden_fraction)
    hidden_keys = [str(getattr(p, "key", "")) for p in _bound_measured(hidden)]

    faithful = measurement_from_contract(contract)
    gamed = dict(faithful)
    for key in hidden_keys:
        gamed[key] = _corrupt(faithful[key])

    faithful_eval = evaluate_split(contract, faithful, hidden_fraction=hidden_fraction)
    gamed_eval = evaluate_split(contract, gamed, hidden_fraction=hidden_fraction)

    # Determinism: the same contract splits the same way every time.
    _v2, hidden2 = split_contract(contract, hidden_fraction=hidden_fraction)
    hidden_keys2 = [str(getattr(p, "key", "")) for p in _bound_measured(hidden2)]

    ok = True

    def _require(cond: bool, message: str) -> None:
        nonlocal ok
        if not cond:
            ok = False
            print("selfcheck FAIL: " + message)

    _require(hidden_keys, "the split held back no bound MEASURED predicate")
    _require(hidden_keys == hidden_keys2, "the split is not deterministic")

    _require(faithful_eval.visible_satisfied,
             "faithful submission should satisfy the visible contract")
    _require(faithful_eval.hidden_satisfied,
             "faithful submission should satisfy the hidden contract")
    _require(not faithful_eval.generalization_gap,
             "faithful submission must not show a generalisation gap")

    _require(gamed_eval.visible_satisfied,
             "gamed submission should still satisfy the VISIBLE contract")
    _require(not gamed_eval.hidden_satisfied,
             "gamed submission should FAIL the hidden contract")
    _require(gamed_eval.hidden_evaluable,
             "the hidden contract should be evaluable (has a bound gate)")
    _require(gamed_eval.generalization_gap,
             "gamed submission must show the generalisation gap")

    print(render(faithful_eval))
    print("")
    print(render(gamed_eval))
    print("")
    print("selfcheck: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.agents.pdd.evaluate_split",
        description="Evaluate a submission against the visible and hidden halves "
                    "of its Measured Geometric Contract (TDAD anti-gaming split).",
    )
    parser.add_argument(
        "--selfcheck", action="store_true",
        help="run the synthetic hidden-split fixture (no kernel, no model) and "
             "demonstrate a faithful (no gap) and a gamed (gap) submission.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit the self-check evaluations as JSON.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    if not args.json:
        return _selfcheck()

    # JSON mode: re-run the fixture quietly and emit both evaluations.
    compile_contract = _lazy_compile()
    split_contract = _lazy_split()
    brief = {
        "part_id": "selfcheck-plate", "width": 80.0, "depth": 40.0, "height": 12.0,
        "kind": "plate", "holes": 2, "hole_diameter": 6.0, "wall": 3.0,
        "material": "aluminum", "intent": "synthetic hidden-split self-check plate",
    }
    hidden_fraction = 0.5
    contract = compile_contract(brief)
    _visible, hidden = split_contract(contract, hidden_fraction=hidden_fraction)
    hidden_keys = [str(getattr(p, "key", "")) for p in _bound_measured(hidden)]
    faithful = measurement_from_contract(contract)
    gamed = dict(faithful)
    for key in hidden_keys:
        gamed[key] = _corrupt(faithful[key])
    payload = {
        "faithful": evaluate_split(
            contract, faithful, hidden_fraction=hidden_fraction).to_dict(),
        "gamed": evaluate_split(
            contract, gamed, hidden_fraction=hidden_fraction).to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
