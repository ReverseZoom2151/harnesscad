"""Inter-rule conflict detection over a set of active rules.

A living codebook accumulates rules from many clauses, standards and revisions;
some of them contradict each other on the *same* parameter within the *same*
scope — e.g. one clause says a wall must be ``>= 2 mm`` while another caps it at
``<= 1.5 mm``. :func:`detect_conflicts` finds those mutually-unsatisfiable pairs
by intersecting each rule's feasible numeric interval; an empty intersection is a
conflict, cited back to both offending rules.

Deterministic: rules are grouped and compared in a stable, sorted order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from standards.registry import Rule, _scope_key

_INF = float("inf")
_EPS = 1e-9


@dataclass
class Conflict:
    """A mutually contradictory pair of rules on one parameter + scope."""

    rule_a: Rule
    rule_b: Rule
    parameter: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "rule_a": self.rule_a.id,
            "rule_b": self.rule_b.id,
            "parameter": self.parameter,
            "reason": self.reason,
        }


def detect_conflicts(rules: Sequence[Rule]) -> List[Conflict]:
    """Return every contradictory rule pair in ``rules`` (deterministic order).

    Rules are grouped by ``(parameter, scope)``; within a group each pair is
    tested for an empty feasible intersection. Rules with incomparable units are
    skipped rather than flagged (avoids false positives).
    """
    # Stable ordering so output is deterministic regardless of input order.
    ordered = sorted(enumerate(rules), key=lambda t: (
        t[1].parameter, _scope_key(t[1].scope), t[1].comparator,
        _sort_num(t[1].limit), t[1].id, t[0]))

    groups: dict = {}
    for _orig_idx, rule in ordered:
        key = (rule.parameter, _scope_key(rule.scope))
        groups.setdefault(key, []).append(rule)

    conflicts: List[Conflict] = []
    for (parameter, _skey), group in sorted(groups.items(),
                                            key=lambda kv: kv[0][0]):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                reason = _pair_conflict(group[i], group[j])
                if reason:
                    conflicts.append(Conflict(
                        rule_a=group[i], rule_b=group[j],
                        parameter=parameter, reason=reason))
    return conflicts


# --------------------------------------------------------------------------- #
# Pairwise test
# --------------------------------------------------------------------------- #
def _pair_conflict(a: Rule, b: Rule) -> Optional[str]:
    if not _units_comparable(a, b):
        return None

    # Set-membership ('in') handled separately from numeric intervals.
    if a.comparator == "in" or b.comparator == "in":
        return _membership_conflict(a, b)

    iva = _interval(a)
    ivb = _interval(b)
    if iva is None or ivb is None:
        return None

    lo = max(iva[0], ivb[0])
    hi = min(iva[1], ivb[1])
    if lo > hi + _EPS:
        return (
            f"unsatisfiable: {_describe(a)} and {_describe(b)} cannot both hold "
            f"(feasible range for '{a.parameter}' is empty)."
        )
    return None


def _interval(rule: Rule) -> Optional[Tuple[float, float]]:
    """Feasible closed interval [lo, hi] a single rule permits, or None."""
    c = rule.comparator
    if c not in ("<=", ">=", "==", "near") or rule.limit is None:
        return None
    v = float(rule.limit)
    if c == ">=":
        return (v, _INF)
    if c == "<=":
        return (-_INF, v)
    if c == "==":
        return (v, v)
    # near: a tolerance band so only clearly-incompatible bounds conflict.
    tol = max(1e-6, 0.05 * abs(v))
    return (v - tol, v + tol)


def _membership_conflict(a: Rule, b: Rule) -> Optional[str]:
    """Conflict logic when at least one rule is an 'in' set constraint."""
    def allowed(rule: Rule):
        return {float(x) for x in (rule.values or []) if _isnum(x)}

    if a.comparator == "in" and b.comparator == "in":
        sa, sb = allowed(a), allowed(b)
        if sa and sb and not (sa & sb):
            return (f"disjoint allowed sets: {_describe(a)} and {_describe(b)} "
                    f"share no permitted value for '{a.parameter}'.")
        return None

    # One 'in' set vs one numeric bound: conflict if no member satisfies it.
    set_rule, bound_rule = (a, b) if a.comparator == "in" else (b, a)
    members = allowed(set_rule)
    iv = _interval(bound_rule)
    if not members or iv is None:
        return None
    if not any(iv[0] - _EPS <= m <= iv[1] + _EPS for m in members):
        return (f"no permitted value satisfies the bound: {_describe(set_rule)} "
                f"vs {_describe(bound_rule)} for '{a.parameter}'.")
    return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _units_comparable(a: Rule, b: Rule) -> bool:
    if a.unit and b.unit and a.unit != b.unit:
        return False
    return True


def _describe(rule: Rule) -> str:
    if rule.comparator == "in":
        return f"[{rule.clause}] {rule.parameter} in {rule.values}"
    unit = f" {rule.unit}" if rule.unit else ""
    return f"[{rule.clause}] {rule.parameter} {rule.comparator} {rule.limit}{unit}"


def _isnum(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _sort_num(v) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else _INF
