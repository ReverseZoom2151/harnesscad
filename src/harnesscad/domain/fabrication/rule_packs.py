"""Versioned declarative fabrication rule packs.

THIRD-PARTY DATA
----------------
``VENDORED_PACKS`` at the bottom of this module is **redistributed
third-party data**, not an independent reimplementation: it is the content of
IntentForge's four bracket rule packs
(``src/intentforge/knowledge/packs/data/{assembly,manufacturing,mechanical,
structural}.yaml``), transcribed from YAML into a Python dict literal so it
loads without a YAML dependency. The rule content is unchanged.

IntentForge is licensed under the **Apache License, Version 2.0**. Redistributing
this data obliges us to pass the licence on (s4(a)), to state that we changed the
form (s4(b)) and to keep upstream's attribution notices (s4(c)); upstream ships
no ``NOTICE`` file, so s4(d) is inapplicable. The full entry -- what was taken,
what changed, and where the licence text lives -- is in the repository root
``THIRD-PARTY.md``, with the licence itself at
``THIRD-PARTY-LICENSES/Apache-2.0.txt``.

Everything else in this module -- the dataclasses, the expression interpreter and
the evaluator -- is HarnessCAD's own code.

A pack is a named, versioned bundle of *condition-expression* rules: each
rule carries a boolean ``condition.expression`` over named design metrics
(``"hole_edge_distance >= 1.5 * hole_diameter"``), the metrics it needs
(``required_metrics``), an optional ``when`` gate (metric equality tests
that decide whether the rule applies at all), a ``severity`` and
``confidence``, and declarative reasoning metadata (``tradeoffs``,
``depends_on``, ``can_conflict_with``, ...).  :func:`evaluate` runs a pack
against a plain metrics dict and returns severity-tagged findings plus the
declared dependency interactions between failed findings.

The condition expressions are evaluated by a small recursive-descent
interpreter (:func:`evaluate_expression`) over the metrics dict -- never by
``eval``.  A missing metric or malformed expression yields a typed
``not_evaluable`` finding instead of a crash, recorded as an
``evaluation_error``.

NOT to be confused with :mod:`harnesscad.domain.standards.registry`, whose
``RulePack`` bundles comparator-style *standards clauses* (parameter /
comparator / limit records indexed by standard + version).  This module's
:class:`FabricationRulePack` uses free-form boolean
expressions over derived design metrics, with reasoning metadata and
dependency links between rules.  The public names here are prefixed
(``FabricationRulePack``, ``FabricationRule``) to keep the two apart.

Stdlib only; deterministic (findings follow pack rule order, interactions
are emitted sorted by rule id).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SEVERITIES",
    "PACK_STATUSES",
    "ConditionEvaluationError",
    "RuleCondition",
    "Tradeoff",
    "FabricationRule",
    "FabricationRulePack",
    "RuleFinding",
    "RuleInteraction",
    "EvaluationResult",
    "evaluate_expression",
    "evaluate",
    "evaluate_packs",
    "VENDORED_PACKS",
    "vendored_packs",
    "main",
]

# Rule severity levels and pack lifecycle statuses.
SEVERITIES = ("info", "recommendation", "warning", "error")
PACK_STATUSES = ("active", "deprecated")

# Rule-to-rule reference fields carried inside ``reasoning``.
RULE_REFERENCE_FIELDS = (
    "can_conflict_with",
    "depends_on",
    "duplicates",
    "mitigates",
    "mitigated_by",
    "reinforces",
)


class ConditionEvaluationError(ValueError):
    """Raised when a declarative rule condition cannot be safely evaluated.

    A missing metric, an unsupported operator or a malformed expression raises
    this; :func:`evaluate` converts it into a ``not_evaluable`` finding.
    """


# --------------------------------------------------------------------------- #
# Expression evaluator (recursive descent, no eval, no ast)
# --------------------------------------------------------------------------- #
#
# Grammar (comparisons and boolean operators, plus ``not`` and
# parentheses for completeness):
#
#   expr        := or_expr
#   or_expr     := and_expr ( "or" and_expr )*
#   and_expr    := not_expr ( "and" not_expr )*
#   not_expr    := "not" not_expr | comparison
#   comparison  := arith ( ("=="|"!="|"<="|">="|"<"|">") arith )*   (chained)
#   arith       := term ( ("+"|"-") term )*
#   term        := unary ( ("*"|"/") unary )*
#   unary       := "-" unary | atom
#   atom        := NUMBER | "true" | "false" | NAME | "(" expr ")"
#
# Names resolve against the metrics dict; ``true``/``false`` are literals
# (so YAML-style booleans work).

_COMPARATOR_TOKENS = ("==", "!=", "<=", ">=", "<", ">")


def _tokenize(expression: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    n = len(expression)
    while i < n:
        ch = expression[i]
        if ch.isspace():
            i += 1
            continue
        two = expression[i : i + 2]
        if two in ("==", "!=", "<=", ">="):
            tokens.append(two)
            i += 2
            continue
        if ch in "<>+-*/()":
            tokens.append(ch)
            i += 1
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and expression[i + 1].isdigit()):
            j = i
            seen_dot = False
            while j < n and (expression[j].isdigit() or (expression[j] == "." and not seen_dot)):
                if expression[j] == ".":
                    seen_dot = True
                j += 1
            tokens.append(expression[i:j])
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (expression[j].isalnum() or expression[j] == "_"):
                j += 1
            tokens.append(expression[i:j])
            i = j
            continue
        raise ConditionEvaluationError(f"unexpected character in expression: {ch!r}")
    return tokens


class _Parser:
    """Recursive-descent evaluator over a metrics dict."""

    def __init__(self, tokens: List[str], metrics: Dict[str, Any]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._metrics = metrics

    def _peek(self) -> Optional[str]:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _next(self) -> str:
        token = self._peek()
        if token is None:
            raise ConditionEvaluationError("unexpected end of expression")
        self._pos += 1
        return token

    def parse(self) -> Any:
        value = self._or_expr()
        if self._peek() is not None:
            raise ConditionEvaluationError(f"unexpected token: {self._peek()!r}")
        return value

    def _or_expr(self) -> Any:
        value = self._and_expr()
        while self._peek() == "or":
            self._next()
            right = self._and_expr()
            value = bool(value) or bool(right)
        return value

    def _and_expr(self) -> Any:
        value = self._not_expr()
        while self._peek() == "and":
            self._next()
            right = self._not_expr()
            value = bool(value) and bool(right)
        return value

    def _not_expr(self) -> Any:
        if self._peek() == "not":
            self._next()
            return not bool(self._not_expr())
        return self._comparison()

    def _comparison(self) -> Any:
        left = self._arith()
        result: Optional[bool] = None
        while self._peek() in _COMPARATOR_TOKENS:
            op = self._next()
            right = self._arith()
            try:
                if op == "==":
                    ok = left == right
                elif op == "!=":
                    ok = left != right
                elif op == "<=":
                    ok = left <= right
                elif op == ">=":
                    ok = left >= right
                elif op == "<":
                    ok = left < right
                else:
                    ok = left > right
            except TypeError as exc:
                raise ConditionEvaluationError(f"invalid comparison: {exc}") from exc
            result = ok if result is None else (result and ok)
            if not ok:
                # Chained comparisons short-circuit false like Python's.
                # Consume the rest of the chain for syntax checking.
                while self._peek() in _COMPARATOR_TOKENS:
                    self._next()
                    self._arith()
                return False
            left = right
        return left if result is None else result

    def _arith(self) -> Any:
        value = self._term()
        while self._peek() in ("+", "-"):
            op = self._next()
            right = self._term()
            try:
                value = value + right if op == "+" else value - right
            except TypeError as exc:
                raise ConditionEvaluationError(f"invalid arithmetic: {exc}") from exc
        return value

    def _term(self) -> Any:
        value = self._unary()
        while self._peek() in ("*", "/"):
            op = self._next()
            right = self._unary()
            try:
                if op == "*":
                    value = value * right
                else:
                    if right == 0:
                        raise ConditionEvaluationError("division by zero")
                    value = value / right
            except TypeError as exc:
                raise ConditionEvaluationError(f"invalid arithmetic: {exc}") from exc
        return value

    def _unary(self) -> Any:
        if self._peek() == "-":
            self._next()
            operand = self._unary()
            try:
                return -operand
            except TypeError as exc:
                raise ConditionEvaluationError(f"invalid negation: {exc}") from exc
        return self._atom()

    def _atom(self) -> Any:
        token = self._next()
        if token == "(":
            value = self._or_expr()
            if self._next() != ")":
                raise ConditionEvaluationError("missing closing parenthesis")
            return value
        first = token[0]
        if first.isdigit() or first == ".":
            try:
                return float(token) if "." in token else int(token)
            except ValueError as exc:
                raise ConditionEvaluationError(f"invalid number: {token!r}") from exc
        if first.isalpha() or first == "_":
            # YAML-style boolean literals.
            if token == "true":
                return True
            if token == "false":
                return False
            if token in ("and", "or", "not"):
                raise ConditionEvaluationError(f"misplaced keyword: {token!r}")
            if token not in self._metrics:
                raise ConditionEvaluationError(f"missing metric: {token}")
            return self._metrics[token]
        raise ConditionEvaluationError(f"unexpected token: {token!r}")


def evaluate_expression(expression: str, metrics: Dict[str, Any]) -> bool:
    """Evaluate a restricted rule expression without executing Python code.

    Supports comparisons (including chained), ``and`` / ``or`` / ``not``,
    ``+ - * /`` arithmetic, unary minus, parentheses, numeric literals,
    ``true`` / ``false`` and metric names.  Raises
    :class:`ConditionEvaluationError` for unknown metrics, bad syntax or
    division by zero.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ConditionEvaluationError("condition expression must be a non-empty string")
    tokens = _tokenize(expression)
    if not tokens:
        raise ConditionEvaluationError("condition expression must be a non-empty string")
    return bool(_Parser(tokens, metrics).parse())


# --------------------------------------------------------------------------- #
# Dataclasses for the pack format
# --------------------------------------------------------------------------- #
@dataclass
class RuleCondition:
    """A rule's declarative condition block.

    ``expression`` is the boolean pass condition (the rule *passes* when it
    evaluates true).  ``required_metrics`` lists the metric names the
    expression reads.  ``when`` is an optional gate: a mapping of metric name
    to expected value; if any entry differs from the metrics dict, the rule
    does not apply and produces no finding.
    """

    expression: str
    required_metrics: List[str] = field(default_factory=list)
    when: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RuleCondition":
        expression = d.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("condition.expression is required")
        when = d.get("when") or {}
        if not isinstance(when, dict):
            raise ValueError("condition.when must be a mapping")
        return cls(
            expression=expression,
            required_metrics=list(d.get("required_metrics") or []),
            when=dict(when),
        )

    def applies(self, metrics: Dict[str, Any]) -> bool:
        """True when every ``when`` gate entry matches the metrics dict."""
        for key, expected in self.when.items():
            if metrics.get(key) != expected:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"expression": self.expression}
        if self.required_metrics:
            out["required_metrics"] = list(self.required_metrics)
        if self.when:
            out["when"] = dict(self.when)
        return out


@dataclass
class Tradeoff:
    """One declarative benefit / cost pair from a rule's reasoning block."""

    benefit: str
    cost: str
    affected_parameters: List[str] = field(default_factory=list)
    recommendation: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tradeoff":
        if not isinstance(d.get("benefit"), str) or not d["benefit"].strip():
            raise ValueError("tradeoff needs a benefit")
        if not isinstance(d.get("cost"), str) or not d["cost"].strip():
            raise ValueError("tradeoff needs a cost")
        return cls(
            benefit=d["benefit"],
            cost=d["cost"],
            affected_parameters=list(d.get("affected_parameters") or []),
            recommendation=str(d.get("recommendation") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"benefit": self.benefit, "cost": self.cost}
        if self.affected_parameters:
            out["affected_parameters"] = list(self.affected_parameters)
        if self.recommendation:
            out["recommendation"] = self.recommendation
        return out


@dataclass
class FabricationRule:
    """One condition-expression fabrication rule.

    Versioned (``rule_version``,
    ``status``, ``created_by``, ``last_updated``), scoped (``applies_to``
    model families), with a :class:`RuleCondition` and free-form ``reasoning``
    metadata carrying ``tradeoffs`` / ``depends_on`` / interaction links.
    """

    id: str
    name: str
    category: str
    description: str
    applies_to: List[str]
    condition: RuleCondition
    severity: str
    recommendation: str
    source_reference: str
    confidence: float
    rule_version: str = "1.0"
    created_by: str = ""
    last_updated: str = ""
    status: str = "active"
    reasoning: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unsupported severity: {self.severity!r}")
        if self.status not in PACK_STATUSES:
            raise ValueError(f"unsupported rule status: {self.status!r}")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError("confidence must be between 0 and 1")
        parts = self.rule_version.split(".")
        if not parts or not all(part.isdigit() for part in parts):
            raise ValueError("rule_version must use numeric dot notation, for example '1.0'")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FabricationRule":
        return cls(
            id=str(d["id"]),
            name=str(d["name"]),
            category=str(d["category"]),
            description=str(d["description"]),
            applies_to=list(d.get("applies_to") or []),
            condition=RuleCondition.from_dict(dict(d.get("condition") or {})),
            severity=str(d["severity"]),
            recommendation=str(d.get("recommendation") or ""),
            source_reference=str(d.get("source_reference") or ""),
            confidence=float(d["confidence"]),
            rule_version=str(d.get("rule_version") or "1.0"),
            created_by=str(d.get("created_by") or ""),
            last_updated=str(d.get("last_updated") or ""),
            status=str(d.get("status") or "active"),
            reasoning=dict(d.get("reasoning") or {}),
        )

    # Convenience accessors over the reasoning block. ------------------------
    @property
    def depends_on(self) -> List[str]:
        return list(self.reasoning.get("depends_on") or [])

    @property
    def tradeoffs(self) -> List[Tradeoff]:
        return [Tradeoff.from_dict(dict(t)) for t in (self.reasoning.get("tradeoffs") or [])]

    def rule_references(self, field_name: str) -> List[str]:
        """Referenced rule ids for one of :data:`RULE_REFERENCE_FIELDS`."""
        if field_name not in RULE_REFERENCE_FIELDS:
            raise ValueError(f"unknown rule reference field: {field_name!r}")
        value = self.reasoning.get(field_name) or []
        return [str(item) for item in value]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rule_version": self.rule_version,
            "status": self.status,
            "created_by": self.created_by,
            "last_updated": self.last_updated,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "applies_to": list(self.applies_to),
            "condition": self.condition.to_dict(),
            "severity": self.severity,
            "recommendation": self.recommendation,
            "source_reference": self.source_reference,
            "confidence": self.confidence,
            "reasoning": dict(self.reasoning),
        }


@dataclass
class FabricationRulePack:
    """A versioned, auditable group of condition-expression rules.

    Identity
    (``pack_id`` + numeric-dot ``pack_version``), a ``category`` shared by all
    rules in the pack, the ``supported_model_families`` envelope, a ``status``
    lifecycle flag and free-form provenance ``metadata``.  Distinct from
    ``harnesscad.domain.standards.registry.RulePack`` (comparator clauses).
    """

    pack_id: str
    pack_version: str
    name: str
    description: str
    category: str
    supported_model_families: List[str]
    rules: List[FabricationRule]
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self) -> None:
        if self.status not in PACK_STATUSES:
            raise ValueError(f"unsupported pack status: {self.status!r}")
        parts = self.pack_version.split(".")
        if not parts or not all(part.isdigit() for part in parts):
            raise ValueError("pack_version must use numeric dot notation, for example '1.0'")
        rule_ids = [rule.id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("duplicate rule id inside rule pack")
        mismatched = sorted(rule.id for rule in self.rules if rule.category != self.category)
        if mismatched:
            raise ValueError(
                "rules must match pack category unless split into another pack: "
                + ", ".join(mismatched)
            )
        family_set = set(self.supported_model_families)
        outside = sorted(
            rule.id for rule in self.rules if not set(rule.applies_to).issubset(family_set)
        )
        if outside:
            raise ValueError(
                "rules must not apply to families outside supported_model_families: "
                + ", ".join(outside)
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FabricationRulePack":
        return cls(
            pack_id=str(d["pack_id"]),
            pack_version=str(d["pack_version"]),
            name=str(d["name"]),
            description=str(d.get("description") or ""),
            category=str(d["category"]),
            supported_model_families=list(d.get("supported_model_families") or []),
            rules=[FabricationRule.from_dict(dict(r)) for r in (d.get("rules") or [])],
            status=str(d.get("status") or "active"),
            metadata=dict(d.get("metadata") or {}),
            source=str(d.get("source") or ""),
        )

    def get(self, rule_id: str) -> FabricationRule:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        raise KeyError(rule_id)

    def active_rules(self) -> List[FabricationRule]:
        return [rule for rule in self.rules if rule.status == "active"]

    def rules_for_family(self, family: str) -> List[FabricationRule]:
        return [rule for rule in self.rules if family in rule.applies_to]

    def provenance(self) -> Dict[str, Any]:
        """The pack's provenance block as recorded in finding metadata."""
        return {
            "pack_id": self.pack_id,
            "pack_version": self.pack_version,
            "pack_category": self.category,
            "pack_source": self.source,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "pack_version": self.pack_version,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "supported_model_families": list(self.supported_model_families),
            "status": self.status,
            "metadata": dict(self.metadata),
            "source": self.source,
            "rules": [rule.to_dict() for rule in self.rules],
        }


# --------------------------------------------------------------------------- #
# Findings and evaluation
# --------------------------------------------------------------------------- #
FINDING_STATUSES = ("pass", "fail", "not_evaluable")


@dataclass
class RuleFinding:
    """Result of applying one fabrication rule.

    ``status`` is ``"pass"``, ``"fail"`` or ``"not_evaluable"`` (a missing
    required metric or a broken expression).  ``passed`` is a plain
    boolean, False for not-evaluable rules too; the typed ``status``
    disambiguates.  ``metadata`` carries the expression, versioning fields,
    the snapshot of required metrics, the pack provenance block and, when
    relevant, the ``evaluation_error`` text.
    """

    rule_id: str
    rule_name: str
    category: str
    severity: str
    status: str
    passed: bool
    message: str
    recommendation: str
    confidence: float
    condition: str
    tradeoffs: List[Tradeoff] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "category": self.category,
            "severity": self.severity,
            "status": self.status,
            "passed": self.passed,
            "message": self.message,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "condition": self.condition,
            "tradeoffs": [t.to_dict() for t in self.tradeoffs],
            "metadata": dict(self.metadata),
        }


@dataclass
class RuleInteraction:
    """A declared relationship surfaced between evaluated rules.

    Interaction semantics:
    ``depends_on`` links are emitted when the *source* rule failed and the
    referenced rule has any finding at all; ``reinforces``, ``conflicts`` and
    ``duplicates`` require both rules to have failed; ``mitigates`` fires when
    the target failed.  Interactions annotate; they do not suppress findings.
    """

    interaction_type: str
    rule_ids: List[str]
    description: str
    effect: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interaction_type": self.interaction_type,
            "rule_ids": list(self.rule_ids),
            "description": self.description,
            "effect": self.effect,
            "metadata": dict(self.metadata),
        }


@dataclass
class EvaluationResult:
    """Findings plus rule interactions from evaluating pack(s) on metrics."""

    findings: List[RuleFinding]
    interactions: List[RuleInteraction]

    def failed(self) -> List[RuleFinding]:
        return [f for f in self.findings if f.status == "fail"]

    def not_evaluable(self) -> List[RuleFinding]:
        return [f for f in self.findings if f.status == "not_evaluable"]

    def by_severity(self, severity: str) -> List[RuleFinding]:
        return [f for f in self.findings if f.severity == severity]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "interactions": [i.to_dict() for i in self.interactions],
        }


def _evaluate_rule(
    rule: FabricationRule,
    metrics: Dict[str, Any],
    provenance: Dict[str, Any],
) -> RuleFinding:
    metadata: Dict[str, Any] = {
        "expression": rule.condition.expression,
        "rule_version": rule.rule_version,
        "rule_status": rule.status,
        "source_reference": rule.source_reference,
        "created_by": rule.created_by,
        "last_updated": rule.last_updated,
        "metrics": {key: metrics.get(key) for key in rule.condition.required_metrics},
    }
    metadata.update(provenance)
    missing = [key for key in rule.condition.required_metrics if key not in metrics]
    try:
        if missing:
            raise ConditionEvaluationError("missing metric: " + ", ".join(missing))
        passed = evaluate_expression(rule.condition.expression, metrics)
        status = "pass" if passed else "fail"
        message = f"{rule.name} passed." if passed else rule.description
    except ConditionEvaluationError as exc:
        passed = False
        status = "not_evaluable"
        message = f"Knowledge rule could not be evaluated: {exc}"
        metadata["evaluation_error"] = str(exc)
    return RuleFinding(
        rule_id=rule.id,
        rule_name=rule.name,
        category=rule.category,
        severity=rule.severity,
        status=status,
        passed=passed,
        message=message,
        recommendation=rule.recommendation,
        confidence=rule.confidence,
        condition=rule.condition.expression,
        tradeoffs=rule.tradeoffs,
        metadata=metadata,
    )


def _interactions(
    rules: List[FabricationRule], findings: List[RuleFinding]
) -> List[RuleInteraction]:
    """Emit interaction annotations over the findings.

    Interactions only annotate, never suppress.  Iteration is sorted by rule id so output
    is deterministic regardless of pack ordering.
    """
    all_ids = {f.rule_id for f in findings}
    failed = {f.rule_id for f in findings if f.status == "fail"}
    out: List[RuleInteraction] = []
    seen: set = set()

    def add(kind: str, source: str, target: str, description: str, effect: str, origin: str) -> None:
        key = (kind, source, target)
        if key in seen:
            return
        seen.add(key)
        out.append(
            RuleInteraction(
                interaction_type=kind,
                rule_ids=[source, target],
                description=description,
                effect=effect,
                metadata={"source": origin},
            )
        )

    for rule in sorted(rules, key=lambda item: item.id):
        if rule.id not in all_ids:
            continue
        source_failed = rule.id in failed
        reasoning = rule.reasoning or {}
        mitigation = str(reasoning.get("mitigation") or "")
        for target in rule.rule_references("reinforces"):
            if source_failed and target in failed:
                add(
                    "reinforces", rule.id, target,
                    "The findings reinforce the same engineering concern.",
                    "reinforced engineering concern should receive higher review priority",
                    "reasoning.reinforces",
                )
        for target in rule.rule_references("can_conflict_with"):
            if source_failed and target in failed:
                add(
                    "conflicts", rule.id, target,
                    "The recommendations may compete within the available design envelope.",
                    mitigation or "review both recommendations before changing parameters",
                    "reasoning.can_conflict_with",
                )
        for target in rule.rule_references("depends_on"):
            if source_failed and target in all_ids:
                add(
                    "depends_on", rule.id, target,
                    "This finding depends on another design condition or clearance rule.",
                    "resolve the dependency before treating this recommendation as isolated",
                    "reasoning.depends_on",
                )
        for target in rule.rule_references("duplicates"):
            if source_failed and target in failed:
                add(
                    "duplicates", rule.id, target,
                    "These findings cover overlapping engineering concerns.",
                    "merge duplicate recommendations during prioritization",
                    "reasoning.duplicates",
                )
        for target in rule.rule_references("mitigates"):
            if target in failed:
                add(
                    "mitigates", rule.id, target,
                    "One recommendation may reduce the concern raised by another rule.",
                    mitigation or "apply the mitigation only after validating the updated design",
                    "reasoning.mitigates",
                )
        for target in rule.rule_references("mitigated_by"):
            if source_failed:
                add(
                    "mitigates", rule.id, target,
                    "This finding has an encoded mitigation rule or feature.",
                    mitigation or "apply the mitigation only after validating the updated design",
                    "reasoning.mitigated_by",
                )
    return out


def evaluate(
    pack: FabricationRulePack,
    metrics: Dict[str, Any],
    family: Optional[str] = None,
) -> EvaluationResult:
    """Evaluate one pack against a metrics dict.

    Rules are visited in pack
    order; deprecated rules are skipped; rules whose ``applies_to`` does not
    include the family are skipped (``family`` defaults to
    ``metrics["family"]`` and, when absent entirely, no family filter is
    applied); rules whose ``when`` gate does not match produce no finding;
    everything else yields a pass, fail or typed not-evaluable finding.
    Declared ``depends_on`` (and the other reasoning links) between the
    resulting findings are surfaced as :class:`RuleInteraction` annotations,
    they order review
    priority but never suppress findings.
    """
    if family is None:
        family = metrics.get("family") or metrics.get("object_type")
    findings: List[RuleFinding] = []
    evaluated_rules: List[FabricationRule] = []
    provenance = pack.provenance()
    for rule in pack.rules:
        if rule.status != "active":
            continue
        if family is not None and family not in rule.applies_to:
            continue
        if not rule.condition.applies(metrics):
            continue
        findings.append(_evaluate_rule(rule, metrics, provenance))
        evaluated_rules.append(rule)
    return EvaluationResult(findings, _interactions(evaluated_rules, findings))


def evaluate_packs(
    packs: Sequence[FabricationRulePack],
    metrics: Dict[str, Any],
    family: Optional[str] = None,
) -> EvaluationResult:
    """Evaluate several packs; interactions are computed across all findings
    (packs are flattened into one registry before evaluating)."""
    findings: List[RuleFinding] = []
    rules: List[FabricationRule] = []
    if family is None:
        family = metrics.get("family") or metrics.get("object_type")
    for pack in packs:
        provenance = pack.provenance()
        for rule in pack.rules:
            if rule.status != "active":
                continue
            if family is not None and family not in rule.applies_to:
                continue
            if not rule.condition.applies(metrics):
                continue
            findings.append(_evaluate_rule(rule, metrics, provenance))
            rules.append(rule)
    return EvaluationResult(findings, _interactions(rules, findings))


# --------------------------------------------------------------------------- #
# Vendored rule packs -- THIRD-PARTY DATA, Apache-2.0
# --------------------------------------------------------------------------- #
# Redistributed from IntentForge (Apache License 2.0) -- see the module
# docstring, the root THIRD-PARTY.md entry, and THIRD-PARTY-LICENSES/
# Apache-2.0.txt.  Do NOT drop this notice while these packs remain in the file:
# Apache-2.0 s4 requires it on redistribution, and this repository is public.
# Removing the attribution means removing the data.
#
# Converted 1:1 from the YAML files; every provenance field (pack_id,
# pack_version, metadata.migrated_from, source_reference, created_by,
# last_updated) is preserved.  ``source`` records the original resource path.
VENDORED_PACKS: Dict[str, Dict[str, Any]] = {
    "bracket_assembly": {
        "pack_id": "bracket_assembly",
        "pack_version": "1.0",
        "name": "Bracket Assembly Rules",
        "description": "Assembly and installation heuristics for supported bracket model families.",
        "category": "assembly",
        "supported_model_families": ["wall_mounted_bracket", "l_bracket"],
        "status": "active",
        "metadata": {
            "source_of_truth": True,
            "migrated_from": "intentforge.knowledge.data/bracket_rules.yaml",
        },
        "source": "intentforge/knowledge/packs/data/assembly.yaml",
        "rules": [
            {
                "id": "fastener_accessibility_001",
                "rule_version": "1.0",
                "status": "active",
                "created_by": "intentforge-team",
                "last_updated": "2026-07-10",
                "name": "Fastener Accessibility",
                "category": "assembly",
                "description": "Fastener clearance near the edge may make installation or tool access difficult.",
                "applies_to": ["wall_mounted_bracket", "l_bracket"],
                "condition": {
                    "expression": "fastener_edge_clearance >= 2 * hole_diameter",
                    "required_metrics": ["fastener_edge_clearance", "hole_diameter"],
                    "when": {"mounting_holes_active": True},
                },
                "severity": "recommendation",
                "recommendation": "Increase clearance around fasteners where installation tools need access.",
                "source_reference": "IntentForge Phase 20 assembly accessibility heuristic.",
                "confidence": 0.64,
                "reasoning": {
                    "implications": [
                        "Limited clearance around fasteners may make installation difficult even if holes fit geometrically.",
                    ],
                    "affects": [
                        "fastener_accessibility",
                        "hole_edge_distance",
                        "installation_access",
                    ],
                    "tradeoffs": [
                        {
                            "benefit": "Increasing fastener clearance can improve installation access.",
                            "cost": "Increasing clearance may require a larger bracket or different hole layout.",
                            "affected_parameters": [
                                "fastener_accessibility",
                                "plate_width",
                                "hole_spacing",
                            ],
                            "recommendation": "Increase clearance around fasteners before treating the layout as assembly-ready.",
                        },
                    ],
                    "priority_weight": 0.58,
                    "depends_on": ["hole_edge_margin_001"],
                    "mitigation": "Increase the surrounding clearance before relying on the current hole layout.",
                    "limitations": [
                        "Accessibility depends on the actual fastener and installation tool, which are not modeled.",
                    ],
                },
            },
            {
                "id": "installation_difficulty_001",
                "rule_version": "1.0",
                "status": "active",
                "created_by": "intentforge-team",
                "last_updated": "2026-07-10",
                "name": "Installation Difficulty",
                "category": "assembly",
                "description": "Bracket width is low relative to fastener diameter, which can make handling and installation harder.",
                "applies_to": ["wall_mounted_bracket", "l_bracket"],
                "condition": {
                    "expression": "bracket_width >= 4 * hole_diameter",
                    "required_metrics": ["bracket_width", "hole_diameter"],
                    "when": {"mounting_holes_active": True},
                },
                "severity": "recommendation",
                "recommendation": "Increase bracket width or reduce fastener diameter if installation clearance matters.",
                "source_reference": "IntentForge Phase 20 assembly heuristic.",
                "confidence": 0.58,
                "reasoning": {
                    "implications": [
                        "Narrow bracket width relative to fastener size may make handling and tool access harder.",
                    ],
                    "affects": [
                        "bracket_width",
                        "hole_diameter",
                        "installation_access",
                    ],
                    "tradeoffs": [
                        {
                            "benefit": "Increasing bracket width may improve handling and fastener access.",
                            "cost": "A wider bracket uses more material and may conflict with compact design intent.",
                            "affected_parameters": ["bracket_width", "hole_diameter"],
                            "recommendation": "Increase bracket width when installation clearance matters.",
                        },
                    ],
                    "priority_weight": 0.5,
                    "depends_on": ["fastener_accessibility_001"],
                    "mitigation": "Increase width or specify a smaller fastener only after checking hole strength requirements.",
                    "limitations": [
                        "Installation difficulty is heuristic without the actual installation environment.",
                    ],
                },
            },
        ],
    },
    "bracket_manufacturing": {
        "pack_id": "bracket_manufacturing",
        "pack_version": "1.0",
        "name": "Bracket Manufacturing Rules",
        "description": "Manufacturing-oriented heuristics for supported bracket model families.",
        "category": "manufacturing",
        "supported_model_families": ["wall_mounted_bracket", "l_bracket"],
        "status": "active",
        "metadata": {
            "source_of_truth": True,
            "migrated_from": "intentforge.knowledge.data/bracket_rules.yaml",
        },
        "source": "intentforge/knowledge/packs/data/manufacturing.yaml",
        "rules": [
            {
                "id": "tool_clearance_001",
                "rule_version": "1.0",
                "status": "active",
                "created_by": "intentforge-team",
                "last_updated": "2026-07-10",
                "name": "Tool Clearance",
                "category": "manufacturing",
                "description": "Small features may limit machining or cutting-tool accessibility.",
                "applies_to": ["wall_mounted_bracket", "l_bracket"],
                "condition": {
                    "expression": "tool_clearance >= 2.0",
                    "required_metrics": ["tool_clearance"],
                },
                "severity": "warning",
                "recommendation": "Keep accessible feature widths above 2 mm unless a process-specific tool is specified.",
                "source_reference": "IntentForge Phase 20 initial manufacturing heuristic.",
                "confidence": 0.62,
                "reasoning": {
                    "implications": [
                        "Small feature clearances may limit available manufacturing processes.",
                    ],
                    "affects": [
                        "tool_clearance",
                        "manufacturing_process",
                        "feature_size",
                    ],
                    "tradeoffs": [
                        {
                            "benefit": "Increasing tool clearance can improve manufacturing accessibility.",
                            "cost": "Increasing clearance may require larger features or simplified geometry.",
                            "affected_parameters": ["tool_clearance", "feature_size"],
                            "recommendation": "Increase feature clearance unless a specific manufacturing process is documented.",
                        },
                    ],
                    "priority_weight": 0.55,
                    "depends_on": ["manufacturing_simplicity_001"],
                    "mitigation": "Simplify or enlarge small features before assuming they are manufacturable.",
                    "limitations": [
                        "This clearance check does not replace process-specific tooling review.",
                    ],
                },
            },
            {
                "id": "manufacturing_simplicity_001",
                "rule_version": "1.0",
                "status": "active",
                "created_by": "intentforge-team",
                "last_updated": "2026-07-10",
                "name": "Manufacturing Simplicity",
                "category": "manufacturing",
                "description": "The design includes several optional features that may add manufacturing complexity.",
                "applies_to": ["wall_mounted_bracket", "l_bracket"],
                "condition": {
                    "expression": "active_optional_feature_count <= 4",
                    "required_metrics": ["active_optional_feature_count"],
                },
                "severity": "recommendation",
                "recommendation": "Remove optional features that are not required by design intent.",
                "source_reference": "IntentForge Phase 20 design-for-manufacturing heuristic.",
                "confidence": 0.6,
                "reasoning": {
                    "implications": [
                        "Multiple optional features can make a simple bracket harder to manufacture or inspect.",
                    ],
                    "affects": [
                        "optional_features",
                        "manufacturing_complexity",
                        "review_time",
                    ],
                    "tradeoffs": [
                        {
                            "benefit": "Removing unnecessary optional features can simplify manufacturing.",
                            "cost": "Removing features may also remove intended weight reduction, accessibility, or handling benefits.",
                            "affected_parameters": [
                                "optional_features",
                                "manufacturing_complexity",
                            ],
                            "recommendation": "Preserve optional features only when they are tied to explicit design intent.",
                        },
                    ],
                    "priority_weight": 0.42,
                    "can_conflict_with": [
                        "gusset_recommendation_001",
                        "corner_radius_001",
                    ],
                    "mitigation": "Keep optional features that resolve stronger mechanical or assembly findings.",
                    "limitations": [
                        "Simplicity is process-dependent and does not directly estimate cost.",
                    ],
                },
            },
        ],
    },
    "bracket_mechanical": {
        "pack_id": "bracket_mechanical",
        "pack_version": "1.0",
        "name": "Bracket Mechanical Rules",
        "description": "Mechanical engineering heuristics for supported bracket model families.",
        "category": "mechanical",
        "supported_model_families": ["wall_mounted_bracket", "l_bracket"],
        "status": "active",
        "metadata": {
            "source_of_truth": True,
            "migrated_from": "intentforge.knowledge.data/bracket_rules.yaml",
        },
        "source": "intentforge/knowledge/packs/data/mechanical.yaml",
        "rules": [
            {
                "id": "hole_edge_margin_001",
                "rule_version": "1.0",
                "status": "active",
                "created_by": "intentforge-team",
                "last_updated": "2026-07-10",
                "name": "Hole Edge Margin",
                "category": "mechanical",
                "description": "Hole edge distance is below the recommended margin for local material strength.",
                "applies_to": ["wall_mounted_bracket", "l_bracket"],
                "condition": {
                    "expression": "hole_edge_distance >= 1.5 * hole_diameter",
                    "required_metrics": ["hole_edge_distance", "hole_diameter"],
                    "when": {"mounting_holes_active": True},
                },
                "severity": "warning",
                "recommendation": "Increase hole edge distance or reduce hole diameter to improve local strength.",
                "source_reference": "IntentForge Phase 20 initial mechanical design heuristic.",
                "confidence": 0.9,
                "reasoning": {
                    "implications": [
                        "Insufficient edge margin may reduce local material support around the hole.",
                    ],
                    "affects": [
                        "plate_width",
                        "hole_spacing",
                        "fastener_accessibility",
                    ],
                    "tradeoffs": [
                        {
                            "benefit": "Increasing edge margin may improve local support around fasteners.",
                            "cost": "Increasing edge margin may require a wider or taller plate.",
                            "affected_parameters": ["plate_width", "hole_spacing"],
                            "recommendation": "Increase plate width before reducing hole spacing when both are constrained.",
                        },
                    ],
                    "priority_weight": 0.9,
                    "can_conflict_with": ["hole_spacing_001"],
                    "mitigation": "Increase plate width before reducing hole spacing.",
                    "limitations": [
                        "This edge-distance rule is heuristic and does not replace load-specific analysis.",
                    ],
                },
            },
            {
                "id": "hole_spacing_001",
                "rule_version": "1.0",
                "status": "active",
                "created_by": "intentforge-team",
                "last_updated": "2026-07-10",
                "name": "Hole Spacing",
                "category": "mechanical",
                "description": "Hole spacing is below the recommended spacing between fasteners.",
                "applies_to": ["wall_mounted_bracket", "l_bracket"],
                "condition": {
                    "expression": "hole_spacing >= 3 * hole_diameter",
                    "required_metrics": ["hole_spacing", "hole_diameter"],
                    "when": {"mounting_holes_active": True},
                },
                "severity": "warning",
                "recommendation": "Increase spacing between holes where the design envelope allows it.",
                "source_reference": "IntentForge Phase 20 initial mechanical design heuristic.",
                "confidence": 0.86,
                "reasoning": {
                    "implications": [
                        "Close hole spacing may reduce material between fasteners and complicate load sharing.",
                    ],
                    "affects": [
                        "hole_spacing",
                        "plate_width",
                        "installation_access",
                    ],
                    "tradeoffs": [
                        {
                            "benefit": "Increasing hole spacing may reduce interaction between nearby fastener holes.",
                            "cost": "Increasing spacing may require a larger plate envelope.",
                            "affected_parameters": ["hole_spacing", "plate_width"],
                            "recommendation": "Increase plate width when spacing and edge distance both need improvement.",
                        },
                    ],
                    "priority_weight": 0.82,
                    "can_conflict_with": ["hole_edge_margin_001"],
                    "mitigation": "Increase the available design envelope before moving holes closer to an edge.",
                    "limitations": [
                        "This spacing rule is heuristic and does not model actual fastener load transfer.",
                    ],
                },
            },
            {
                "id": "gusset_recommendation_001",
                "rule_version": "1.0",
                "status": "active",
                "created_by": "intentforge-team",
                "last_updated": "2026-07-10",
                "name": "L-Bracket Gusset Recommendation",
                "category": "mechanical",
                "description": "Tall vertical L-bracket legs with low thickness may benefit from a triangular gusset.",
                "applies_to": ["l_bracket"],
                "condition": {
                    "expression": "vertical_leg_height_to_thickness <= 12 or gusset_enabled == true",
                    "required_metrics": [
                        "vertical_leg_height_to_thickness",
                        "gusset_enabled",
                    ],
                },
                "severity": "recommendation",
                "recommendation": "Consider adding a triangular gusset when the vertical leg is tall relative to thickness.",
                "source_reference": "IntentForge Phase 20 initial L-bracket design heuristic.",
                "confidence": 0.72,
                "reasoning": {
                    "implications": [
                        "A tall thin vertical leg may benefit from additional support near the inside corner.",
                    ],
                    "affects": [
                        "gusset_enabled",
                        "gusset_thickness",
                        "part_weight",
                        "manufacturing_complexity",
                    ],
                    "tradeoffs": [
                        {
                            "benefit": "Adding a gusset may improve bending stiffness near the bracket corner.",
                            "cost": "Adding a gusset increases material, weight, and manufacturing complexity.",
                            "affected_parameters": [
                                "gusset_enabled",
                                "gusset_thickness",
                                "manufacturing_complexity",
                            ],
                            "recommendation": "Add a gusset only when stiffness intent is more important than simplicity.",
                        },
                    ],
                    "priority_weight": 0.76,
                    "can_conflict_with": [
                        "manufacturing_simplicity_001",
                        "cutout_stiffness_tradeoff_001",
                    ],
                    "mitigates": ["thin_section_warning_001"],
                    "mitigation": "Add a triangular gusset after confirming the added material does not conflict with assembly.",
                    "limitations": [
                        "This gusset recommendation is heuristic and is not a stiffness simulation.",
                    ],
                },
            },
            {
                "id": "corner_radius_001",
                "rule_version": "1.0",
                "status": "active",
                "created_by": "intentforge-team",
                "last_updated": "2026-07-10",
                "name": "Corner Radius",
                "category": "mechanical",
                "description": "Requested corner radius is small relative to material thickness.",
                "applies_to": ["wall_mounted_bracket", "l_bracket"],
                "condition": {
                    "expression": "corner_radius >= 0.25 * thickness",
                    "required_metrics": ["corner_radius", "thickness"],
                    "when": {"rounded_corners_active": True},
                },
                "severity": "recommendation",
                "recommendation": "Use a corner radius at least one quarter of thickness when rounded corners are requested.",
                "source_reference": "IntentForge Phase 20 manufacturability and stress concentration heuristic.",
                "confidence": 0.7,
                "reasoning": {
                    "implications": [
                        "Very small requested radii may not provide meaningful edge relief.",
                    ],
                    "affects": [
                        "corner_radius",
                        "outside_edge_fillet_radius",
                        "manufacturing_complexity",
                    ],
                    "tradeoffs": [
                        {
                            "benefit": "Rounded corners may improve handling and reduce sharp edges.",
                            "cost": "Rounded corners may increase machining or modeling complexity depending on process.",
                            "affected_parameters": [
                                "corner_radius",
                                "outside_edge_fillet_radius",
                            ],
                            "recommendation": "Keep corner radius proportional to thickness when rounded corners are requested.",
                        },
                    ],
                    "priority_weight": 0.45,
                    "can_conflict_with": ["manufacturing_simplicity_001"],
                    "mitigation": "Use the smallest documented radius that satisfies the design intent.",
                    "limitations": [
                        "This radius guidance does not quantify stress reduction.",
                    ],
                },
            },
        ],
    },
    "bracket_structural": {
        "pack_id": "bracket_structural",
        "pack_version": "1.0",
        "name": "Bracket Structural Rules",
        "description": "Structural heuristics for supported bracket model families.",
        "category": "structural",
        "supported_model_families": ["wall_mounted_bracket", "l_bracket"],
        "status": "active",
        "metadata": {
            "source_of_truth": True,
            "migrated_from": "intentforge.knowledge.data/bracket_rules.yaml",
        },
        "source": "intentforge/knowledge/packs/data/structural.yaml",
        "rules": [
            {
                "id": "cutout_stiffness_tradeoff_001",
                "rule_version": "1.0",
                "status": "active",
                "created_by": "intentforge-team",
                "last_updated": "2026-07-10",
                "name": "Cutout Stiffness Tradeoff",
                "category": "structural",
                "description": "Center cutout area is large enough to meaningfully reduce plate stiffness.",
                "applies_to": ["wall_mounted_bracket"],
                "condition": {
                    "expression": "cutout_area_ratio <= 0.25",
                    "required_metrics": ["cutout_area_ratio"],
                    "when": {"center_cutout_active": True},
                },
                "severity": "warning",
                "recommendation": "Reduce cutout size or increase thickness if stiffness is important.",
                "source_reference": "IntentForge Phase 20 initial structural heuristic.",
                "confidence": 0.68,
                "reasoning": {
                    "implications": [
                        "A large center cutout may reduce remaining material and stiffness.",
                    ],
                    "affects": [
                        "cutout_width",
                        "cutout_height",
                        "plate_thickness",
                        "stiffness_intent",
                    ],
                    "tradeoffs": [
                        {
                            "benefit": "A cutout can reduce mass and material use.",
                            "cost": "A cutout may reduce stiffness or remaining section width.",
                            "affected_parameters": [
                                "cutout_width",
                                "cutout_height",
                                "plate_thickness",
                            ],
                            "recommendation": "Reduce cutout size or increase thickness if stiffness matters.",
                        },
                    ],
                    "priority_weight": 0.72,
                    "reinforces": ["thin_section_warning_001"],
                    "can_conflict_with": ["gusset_recommendation_001"],
                    "mitigation": "Reduce cutout size before relying on added features to restore stiffness.",
                    "limitations": [
                        "This cutout rule does not quantify stiffness reduction.",
                    ],
                },
            },
            {
                "id": "thin_section_warning_001",
                "rule_version": "1.0",
                "status": "active",
                "created_by": "intentforge-team",
                "last_updated": "2026-07-10",
                "name": "Thin Section Warning",
                "category": "structural",
                "description": "Remaining material section is thin relative to plate thickness.",
                "applies_to": ["wall_mounted_bracket", "l_bracket"],
                "condition": {
                    "expression": "minimum_section_thickness >= 2 * thickness",
                    "required_metrics": ["minimum_section_thickness", "thickness"],
                },
                "severity": "warning",
                "recommendation": "Increase remaining material width or thickness to avoid fragile sections.",
                "source_reference": "IntentForge Phase 20 initial structural heuristic.",
                "confidence": 0.74,
                "reasoning": {
                    "implications": [
                        "Thin remaining material can make local sections more sensitive to loads or manufacturing defects.",
                    ],
                    "affects": [
                        "minimum_section_thickness",
                        "plate_thickness",
                        "cutout_width",
                        "cutout_height",
                    ],
                    "tradeoffs": [
                        {
                            "benefit": "Increasing remaining section width may improve robustness.",
                            "cost": "Increasing section width may require less cutout area or a larger part.",
                            "affected_parameters": [
                                "minimum_section_thickness",
                                "plate_thickness",
                                "cutout_width",
                                "cutout_height",
                            ],
                            "recommendation": "Increase remaining material width or thickness before relying on thin sections.",
                        },
                    ],
                    "priority_weight": 0.8,
                    "reinforces": ["cutout_stiffness_tradeoff_001"],
                    "mitigated_by": ["gusset_recommendation_001"],
                    "mitigation": "Increase remaining material width or add support only after validating geometry.",
                    "limitations": [
                        "This thin-section warning is not a fracture or fatigue analysis.",
                    ],
                },
            },
        ],
    },
}


def vendored_packs() -> List[FabricationRulePack]:
    """The vendored packs as typed objects, sorted by pack_id."""
    return [
        FabricationRulePack.from_dict(VENDORED_PACKS[pack_id])
        for pack_id in sorted(VENDORED_PACKS)
    ]


# --------------------------------------------------------------------------- #
# Selfcheck
# --------------------------------------------------------------------------- #
def _selfcheck() -> Dict[str, Any]:
    packs = vendored_packs()
    assert len(packs) == 4, "expected 4 vendored packs"
    total_rules = sum(len(pack.rules) for pack in packs)
    assert total_rules == 10, f"expected 10 vendored rules, got {total_rules}"
    for pack in packs:
        assert pack.status == "active"
        assert pack.metadata.get("migrated_from"), "provenance metadata missing"
        round_trip = FabricationRulePack.from_dict(pack.to_dict())
        assert round_trip.to_dict() == pack.to_dict(), "pack round-trip mismatch"

    # Expression evaluator basics, including and/or/not and chaining.
    assert evaluate_expression("1 + 2 * 3 == 7", {}) is True
    assert evaluate_expression("-2 < -1 < 0", {}) is True
    assert evaluate_expression("not (1 > 2) and (true or false)", {}) is True
    assert evaluate_expression("a / b >= 2 and a - b != 0", {"a": 8, "b": 4}) is True
    assert evaluate_expression("gusset_enabled == true", {"gusset_enabled": True}) is True
    try:
        evaluate_expression("unknown_metric > 1", {})
    except ConditionEvaluationError:
        pass
    else:
        raise AssertionError("missing metric must raise ConditionEvaluationError")
    try:
        evaluate_expression("a / b > 1", {"a": 1, "b": 0})
    except ConditionEvaluationError:
        pass
    else:
        raise AssertionError("division by zero must raise ConditionEvaluationError")

    # A healthy wall-mounted bracket: every applicable rule passes.
    healthy = {
        "family": "wall_mounted_bracket",
        "mounting_holes_active": True,
        "center_cutout_active": True,
        "rounded_corners_active": True,
        "fastener_edge_clearance": 12.0,
        "hole_edge_distance": 12.0,
        "hole_diameter": 5.0,
        "hole_spacing": 20.0,
        "bracket_width": 100.0,
        "tool_clearance": 5.0,
        "active_optional_feature_count": 3,
        "cutout_area_ratio": 0.1,
        "minimum_section_thickness": 12.0,
        "thickness": 4.0,
        "corner_radius": 2.0,
    }
    result = evaluate_packs(packs, healthy)
    assert result.findings, "healthy metrics must still produce findings"
    assert all(f.status == "pass" for f in result.findings), [
        (f.rule_id, f.status) for f in result.findings
    ]
    assert not result.interactions, "no interactions expected when nothing failed"
    # gusset rule applies only to l_bracket and must have been filtered out.
    assert "gusset_recommendation_001" not in {f.rule_id for f in result.findings}

    # Failing case: tight hole layout fails edge margin, clearance and width
    # rules; depends_on interactions must surface (fastener_accessibility_001
    # depends on hole_edge_margin_001; installation_difficulty_001 depends on
    # fastener_accessibility_001).
    failing = dict(healthy)
    failing.update(
        {
            "fastener_edge_clearance": 4.0,
            "hole_edge_distance": 4.0,
            "bracket_width": 15.0,
        }
    )
    result = evaluate_packs(packs, failing)
    statuses = {f.rule_id: f.status for f in result.findings}
    assert statuses["hole_edge_margin_001"] == "fail"
    assert statuses["fastener_accessibility_001"] == "fail"
    assert statuses["installation_difficulty_001"] == "fail"
    depends = {
        tuple(i.rule_ids)
        for i in result.interactions
        if i.interaction_type == "depends_on"
    }
    assert ("fastener_accessibility_001", "hole_edge_margin_001") in depends
    assert ("installation_difficulty_001", "fastener_accessibility_001") in depends
    failed_finding = next(f for f in result.findings if f.rule_id == "hole_edge_margin_001")
    assert failed_finding.severity == "warning"
    assert failed_finding.tradeoffs and failed_finding.tradeoffs[0].benefit
    assert failed_finding.metadata["pack_id"] == "bracket_mechanical"
    assert failed_finding.metadata["pack_version"] == "1.0"

    # depends_on annotates even when the dependency itself passed
    # (only the target needs to have a finding).
    partial = dict(healthy)
    partial["fastener_edge_clearance"] = 4.0
    result = evaluate_packs(packs, partial)
    statuses = {f.rule_id: f.status for f in result.findings}
    assert statuses["hole_edge_margin_001"] == "pass"
    assert statuses["fastener_accessibility_001"] == "fail"
    depends = {
        tuple(i.rule_ids)
        for i in result.interactions
        if i.interaction_type == "depends_on"
    }
    assert ("fastener_accessibility_001", "hole_edge_margin_001") in depends

    # Not-evaluable: drop a required metric and the rule reports a typed
    # not_evaluable finding rather than crashing.
    broken = dict(healthy)
    del broken["tool_clearance"]
    result = evaluate_packs(packs, broken)
    tool = next(f for f in result.findings if f.rule_id == "tool_clearance_001")
    assert tool.status == "not_evaluable"
    assert tool.passed is False
    assert "evaluation_error" in tool.metadata
    assert tool.message.startswith("Knowledge rule could not be evaluated")

    # when-gating: inactive mounting holes suppress the hole rules entirely.
    gated = dict(healthy)
    gated["mounting_holes_active"] = False
    result = evaluate_packs(packs, gated)
    gated_ids = {f.rule_id for f in result.findings}
    assert "hole_edge_margin_001" not in gated_ids
    assert "fastener_accessibility_001" not in gated_ids
    assert "tool_clearance_001" in gated_ids

    # l_bracket family: gusset rule (or-expression with boolean literal).
    l_metrics = {
        "family": "l_bracket",
        "mounting_holes_active": False,
        "vertical_leg_height_to_thickness": 20.0,
        "gusset_enabled": True,
        "tool_clearance": 4.0,
        "active_optional_feature_count": 1,
        "minimum_section_thickness": 20.0,
        "thickness": 4.0,
    }
    mechanical = next(p for p in packs if p.pack_id == "bracket_mechanical")
    result = evaluate(mechanical, l_metrics)
    gusset = next(f for f in result.findings if f.rule_id == "gusset_recommendation_001")
    assert gusset.status == "pass", "gusset_enabled == true must satisfy the or-branch"
    l_metrics["gusset_enabled"] = False
    result = evaluate(mechanical, l_metrics)
    gusset = next(f for f in result.findings if f.rule_id == "gusset_recommendation_001")
    assert gusset.status == "fail"

    # Structural pack on wall_mounted_bracket: reinforces interaction when
    # both cutout and thin-section rules fail.
    structural_fail = dict(healthy)
    structural_fail.update({"cutout_area_ratio": 0.5, "minimum_section_thickness": 3.0})
    result = evaluate_packs(packs, structural_fail)
    reinforced = {
        tuple(i.rule_ids)
        for i in result.interactions
        if i.interaction_type == "reinforces"
    }
    assert ("cutout_stiffness_tradeoff_001", "thin_section_warning_001") in reinforced

    return {
        "packs": len(packs),
        "rules": total_rules,
        "pack_ids": [pack.pack_id for pack in packs],
        "checks": "pass/fail/not_evaluable/when-gate/depends_on/reinforces all verified",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` runs the vendored packs
    against synthetic metrics covering pass, fail, not-evaluable, when-gated
    and depends_on paths, with assertions."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.fabrication.rule_packs",
        description="Condition-expression fabrication rule packs. The bundled "
        "packs are third-party data redistributed from IntentForge under "
        "Apache-2.0; see THIRD-PARTY.md.",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="evaluate the vendored packs against synthetic metrics and "
        "assert the pass/fail/not-evaluable/depends_on behaviour.",
    )
    parser.add_argument("--json", action="store_true", help="emit the selfcheck summary as JSON.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    summary = _selfcheck()
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "rule_packs selfcheck OK: {packs} packs, {rules} rules ({ids}); {checks}".format(
                packs=summary["packs"],
                rules=summary["rules"],
                ids=", ".join(summary["pack_ids"]),
                checks=summary["checks"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
