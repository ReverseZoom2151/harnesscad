"""Deterministic next-operation ranking for the CISP baseline.

This is intentionally a heuristic, not a pretend learned model.  It converts
the current operation history, backend summary, sketch DOF and diagnostics into
valid *operation kinds* that a planner can parameterize next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class OpSuggestion:
    """A ranked CISP operation kind and the evidence for suggesting it."""

    op: str
    confidence: float
    reason: str
    score: int

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "confidence": self.confidence,
            "reason": self.reason,
            "score": self.score,
        }


class NextOperationRanker:
    """Rank valid next CISP operation kinds with no model dependency."""

    def rank(
        self,
        opdag: Any,
        backend: Any,
        diagnostics: Iterable[Any] = (),
        *,
        top_k: int = 5,
    ) -> list[OpSuggestion]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if top_k == 0:
            return []

        ops = list(opdag.ops())
        summary = _query(backend, "summary")
        dof = _query(backend, "sketch_dof")
        assembly = _query(backend, "assembly")
        tags = [getattr(op, "OP", "") for op in ops]
        sketches = int(summary.get("sketch_count", tags.count("new_sketch")) or 0)
        entities = int(summary.get("entity_count", 0) or 0)
        features = int(summary.get("feature_count", 0) or 0)
        solid = bool(summary.get("solid_present", False))
        instances = len(assembly.get("parts", ())) if isinstance(assembly, Mapping) else 0
        errors = _error_codes(diagnostics)

        candidates: dict[str, tuple[int, str]] = {}

        def offer(tag: str, score: int, reason: str) -> None:
            old = candidates.get(tag)
            if old is None or score > old[0]:
                candidates[tag] = (score, reason)

        offer("new_sketch", 35 if sketches else 100,
              "start the first sketch" if not sketches else "start another profile")

        if sketches:
            primitive_score = 92 if not entities else 48
            for tag, noun in (
                ("add_rectangle", "closed rectangular profile"),
                ("add_circle", "circular profile"),
                ("add_line", "line geometry"),
                ("add_point", "reference point"),
            ):
                offer(tag, primitive_score, f"add {noun} to the active sketch")

        if entities:
            under = any(float(value) > 0 for value in dof.values()) if dof else True
            if under:
                offer("constrain", 95, "reduce remaining sketch degrees of freedom")
            offer("extrude", 88, "turn the current profile into a prismatic solid")
            offer("revolve", 72, "turn the current profile into a revolved solid")

        if sketches >= 2:
            offer("loft", 63, "join multiple sketch profiles")
            offer("sweep", 58, "use available sketches as profile and path")

        if solid:
            for tag, score, reason in (
                ("hole", 82, "add a semantic manufacturable hole"),
                ("fillet", 70, "round solid edges"),
                ("chamfer", 69, "bevel solid edges"),
                ("shell", 55, "make the solid thin-walled"),
                ("draft", 50, "add manufacturing draft"),
                ("mirror", 47, "extend the solid symmetrically"),
                ("add_instance", 45, "place the solid in an assembly"),
            ):
                offer(tag, score, reason)

        if features:
            offer("linear_pattern", 61, "repeat an existing feature linearly")
            offer("circular_pattern", 60, "repeat an existing feature radially")

        if ops:
            offer("set_param", 44, "revise a parameter in the existing operation history")
        if instances >= 2:
            offer("mate", 78, "constrain two placed assembly instances")

        if errors:
            if ops:
                offer("set_param", 110, "repair the latest error by editing an earlier parameter")
            if "bad-ref" in errors:
                offer("new_sketch", 105, "create a missing reference before retrying the failed operation")

        ordered = sorted(candidates.items(), key=lambda item: (-item[1][0], item[0]))
        if not ordered:
            return []
        max_score = max(score for _, (score, _) in ordered)
        return [
            OpSuggestion(tag, round(score / max_score, 3), reason, score)
            for tag, (score, reason) in ordered[:top_k]
        ]


def rank_next_operations(
    opdag: Any,
    backend: Any,
    diagnostics: Iterable[Any] = (),
    *,
    top_k: int = 5,
) -> list[OpSuggestion]:
    """Convenience API using the deterministic baseline ranker."""

    return NextOperationRanker().rank(
        opdag, backend, diagnostics, top_k=top_k
    )


def top_k_accuracy(
    ranked: Sequence[OpSuggestion],
    expected_op: str,
    *,
    k: Optional[int] = None,
) -> float:
    """Return 1.0 when ``expected_op`` occurs in the first ``k``, else 0.0."""

    limit = len(ranked) if k is None else max(0, int(k))
    return float(any(item.op == expected_op for item in ranked[:limit]))


def reciprocal_rank(ranked: Sequence[OpSuggestion], expected_op: str) -> float:
    """Reciprocal rank metric for a single expected next operation."""

    for index, item in enumerate(ranked, 1):
        if item.op == expected_op:
            return 1.0 / index
    return 0.0


def _query(backend: Any, name: str) -> dict:
    try:
        result = backend.query(name)
    except (KeyError, NotImplementedError, TypeError, ValueError):
        return {}
    return dict(result) if isinstance(result, Mapping) else {}


def _error_codes(diagnostics: Iterable[Any]) -> set[str]:
    codes: set[str] = set()
    for diagnostic in diagnostics:
        if isinstance(diagnostic, Mapping):
            severity = diagnostic.get("severity")
            code = diagnostic.get("code")
        else:
            severity = getattr(diagnostic, "severity", None)
            code = getattr(diagnostic, "code", None)
        severity = getattr(severity, "value", severity)
        if str(severity).lower() == "error" and code:
            codes.add(str(code))
    return codes
