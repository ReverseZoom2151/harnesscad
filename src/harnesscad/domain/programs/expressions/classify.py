"""Classify parametric-property expressions into the paper's C1..C5 categories.

The formative study in Gonzalez et al., *Facilitating the Parametric Definition
of Geometric Properties in Programming-Based CAD* (UIST '24, Section 3.1,
Table 2), classifies every parameter expression used to define a primitive's
size or a spatial transformation into one of five categories:

* **C1 Raw number**   — a non-default numeric literal, e.g. ``4.0``.
* **C2 One variable**  — a single variable reference, e.g. ``var1``.
* **C3 Linear combination** — ``sum(alpha_i x_i) + c``, e.g. ``3 + 2*var1 - var2``.
* **C4 Polynomial**    — non-linear polynomial, e.g. ``3 + 2*var1*var2`` or
  ``size_x*i``.
* **C5 Other**         — other programming structures such as conditionals,
  e.g. ``(var1 > 3) ? 1 : 2``.

The paper notes C1 and C2 are *special cases* of C3 but are kept distinct "seeking
a detailed analysis"; this module honours that by reporting the most specific
category. It reproduces the study's aggregation too: Table 3 tallies category
occurrences across four statement kinds (Primitive / Translate / Rotate / Scale)
and reports per-cell counts and percentages of a grand total. :class:`FormativeTally`
is a deterministic re-implementation of that cross-tabulation, so a corpus of
classified expressions can be summarised exactly as in the paper.

Pure stdlib; classification is total (never raises) — unparseable input is C5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

from harnesscad.domain.programs.expressions.linear_form import (
    BinOp,
    Call,
    Expr,
    Neg,
    NonLinearError,
    Num,
    Ternary,
    Var,
    parse_expression,
    to_linear_form,
)


class Category(str, Enum):
    """The five expression categories from Table 2 of the paper."""

    C1_RAW_NUMBER = "C1"
    C2_ONE_VARIABLE = "C2"
    C3_LINEAR_COMBINATION = "C3"
    C4_POLYNOMIAL = "C4"
    C5_OTHER = "C5"

    @property
    def label(self) -> str:
        return {
            "C1": "Raw number",
            "C2": "One variable",
            "C3": "Linear combination",
            "C4": "Polynomial expression",
            "C5": "Other",
        }[self.value]


@dataclass(frozen=True)
class Classification:
    """Result of classifying one expression."""

    category: Category
    reason: str
    expression: str


# statement kinds tracked by Table 3
STATEMENT_KINDS: Tuple[str, ...] = ("primitive", "translate", "rotate", "scale")


def _is_polynomial(expr: Expr) -> bool:
    """True if ``expr`` is a polynomial in its variables (no ternary/compare/div-by-var).

    A polynomial admits +, -, unary minus, constant*anything, variable*variable,
    and division only by constants. Ternaries, comparisons, and general function
    calls are *not* polynomial (they belong to C5).
    """
    if isinstance(expr, Num):
        return True
    if isinstance(expr, Var):
        return True
    if isinstance(expr, Neg):
        return _is_polynomial(expr.operand)
    if isinstance(expr, BinOp):
        if expr.op in ("+", "-", "*"):
            return _is_polynomial(expr.left) and _is_polynomial(expr.right)
        if expr.op == "/":
            # division is polynomial only when the divisor is constant
            return _is_polynomial(expr.left) and _is_constant_expr(expr.right)
        return False
    # Ternary / Call (including comparison-Calls) are not polynomial
    return False


def _is_constant_expr(expr: Expr) -> bool:
    if isinstance(expr, Num):
        return True
    if isinstance(expr, Var):
        return False
    if isinstance(expr, Neg):
        return _is_constant_expr(expr.operand)
    if isinstance(expr, BinOp):
        return _is_constant_expr(expr.left) and _is_constant_expr(expr.right)
    return False


def classify_expression(text: str) -> Classification:
    """Classify one expression string into C1..C5.

    The order of tests reflects specificity: a bare literal is C1, a bare
    identifier is C2, an affine expression is C3, a non-affine polynomial is C4,
    everything else (conditionals, comparisons, transcendental calls, parse
    errors) is C5.
    """
    stripped = text.strip()
    try:
        expr = parse_expression(stripped)
    except SyntaxError:
        return Classification(Category.C5_OTHER, "unparseable expression", stripped)

    # C1: a single (optionally negated) numeric literal.
    if isinstance(expr, Num):
        return Classification(Category.C1_RAW_NUMBER, "numeric literal", stripped)
    if isinstance(expr, Neg) and isinstance(expr.operand, Num):
        return Classification(Category.C1_RAW_NUMBER, "negated numeric literal", stripped)

    # C2: a single variable reference.
    if isinstance(expr, Var):
        return Classification(Category.C2_ONE_VARIABLE, "single variable", stripped)

    # C3: reduces to an affine linear form.
    try:
        form = to_linear_form(expr)
    except NonLinearError:
        form = None
    if form is not None:
        if form.is_constant:
            # A constant-valued arithmetic expression such as "2+3" is still a
            # raw number after evaluation.
            return Classification(
                Category.C1_RAW_NUMBER, "constant-valued arithmetic", stripped
            )
        if len(form.variables) == 1 and form.constant == 0 and form.coefficient(
            form.variables[0]
        ) == 1:
            return Classification(Category.C2_ONE_VARIABLE, "single variable", stripped)
        return Classification(
            Category.C3_LINEAR_COMBINATION, "linear combination of variables", stripped
        )

    # C4: polynomial but non-linear.
    if _is_polynomial(expr):
        return Classification(
            Category.C4_POLYNOMIAL, "non-linear polynomial expression", stripped
        )

    # C5: conditionals, comparisons, function calls, etc.
    kind = type(expr).__name__
    return Classification(Category.C5_OTHER, f"non-polynomial construct ({kind})", stripped)


def classify_vector(components: Iterable[str]) -> Tuple[Classification, ...]:
    """Classify each component of a vector literal, e.g. ``[5, size_y, size_z+3]``.

    The paper notes a single ``cube(size = [5, size_y, size_z+3])`` is counted
    under C1, C2 *and* C3 (one per component). This returns the per-component
    classifications so a caller can tally them independently.
    """
    return tuple(classify_expression(c) for c in components)


@dataclass
class FormativeTally:
    """Cross-tabulation of categories by statement kind (paper's Table 3).

    Rows are categories C1..C5, columns are the four statement kinds. Records
    are added with :meth:`add`; :meth:`counts`, :meth:`row_total`,
    :meth:`column_total`, :meth:`grand_total`, and :meth:`percentage` reproduce
    the count/percent cells the paper reports.
    """

    _grid: Dict[Tuple[Category, str], int] = field(default_factory=dict)

    def add(self, category: Category, statement_kind: str, count: int = 1) -> None:
        kind = statement_kind.lower()
        if kind not in STATEMENT_KINDS:
            raise ValueError(f"unknown statement kind: {statement_kind!r}")
        if count < 0:
            raise ValueError("count must be non-negative")
        key = (category, kind)
        self._grid[key] = self._grid.get(key, 0) + count

    def classify_and_add(self, text: str, statement_kind: str) -> Classification:
        result = classify_expression(text)
        self.add(result.category, statement_kind)
        return result

    def count(self, category: Category, statement_kind: str) -> int:
        return self._grid.get((category, statement_kind.lower()), 0)

    def counts(self) -> Dict[Tuple[Category, str], int]:
        return dict(self._grid)

    def row_total(self, category: Category) -> int:
        return sum(self.count(category, k) for k in STATEMENT_KINDS)

    def column_total(self, statement_kind: str) -> int:
        kind = statement_kind.lower()
        return sum(self._grid.get((c, kind), 0) for c in Category)

    def grand_total(self) -> int:
        return sum(self._grid.values())

    def percentage(self, category: Category, statement_kind: str) -> float:
        total = self.grand_total()
        if total == 0:
            return 0.0
        return 100.0 * self.count(category, statement_kind) / total

    def linear_share(self) -> float:
        """Share (%) of expressions that are C1, C2 or C3 (the paper's 71% claim).

        The paper's core finding: raw numbers + single variables + linear
        combinations account for the large majority of positioning/sizing
        expressions, so a linear-form model captures most of real usage.
        """
        total = self.grand_total()
        if total == 0:
            return 0.0
        linear = sum(
            self.row_total(c)
            for c in (
                Category.C1_RAW_NUMBER,
                Category.C2_ONE_VARIABLE,
                Category.C3_LINEAR_COMBINATION,
            )
        )
        return 100.0 * linear / total
