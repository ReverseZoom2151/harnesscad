"""Deterministic constrained design-space representation, validity gate and sampler.

Motivated by "How Can Large Language Models Help Humans in Design and
Manufacturing" (Makatura et al.), section 5 "Text-to-Design-Space" (Figures
23/26/30/31). A *design space* is more than a parametric design: it is a
parametric design PLUS per-parameter BOUNDS PLUS inter-parameter CONSTRAINTS.
The paper's concrete examples include a car (``width < length``, ``width >
height``, ``wheel_radius < body_height``), a mug with parameter bounds, and a
Lego brick where GPT-4 derived the NON-TRIVIAL constraint that ``brick_length``
and ``brick_width`` must be MULTIPLES OF 3 (a divisibility constraint) together
with lower bounds. It was then asked to produce a set of distinct VALID
parameter settings ("meaningful lego bricks").

This module builds a small, deterministic, standard-library-only
representation of such a constrained design space:

* :class:`ParameterSpec` - a named parameter, either ``"continuous"`` or
  ``"integer"``, with inclusive ``low``/``high`` bounds.
* Composable, evaluable constraint objects (NO ``eval`` of strings):
    - :class:`Inequality` - ``a op b`` where ``a``/``b`` are parameter names or
      numeric constants and ``op`` in ``{"<","<=",">",">="}``.
    - :class:`Divisible` - ``param % modulus == 0`` (the paper's Lego rule).
  Bounds are implicit from each :class:`ParameterSpec`.
* :class:`DesignSpace` - a list of specs + a list of constraints, with:
    - :meth:`DesignSpace.is_valid` - inside bounds AND integral where required
      AND every constraint satisfied.
    - :meth:`DesignSpace.violations` - a list of human-readable strings, one per
      failing bound/constraint (useful and testable).
    - :meth:`DesignSpace.sample_valid` - constraint-respecting rejection
      sampling via ``random.Random(seed)``; deterministic, distinct, capped so
      it never loops forever on an empty/tight space.
    - :meth:`DesignSpace.enumerate_valid` - exact enumeration of valid
      assignments over a discretized grid; exact for small integer spaces such
      as the Lego brick.

All randomness is seeded through ``random.Random(seed)`` and no wall clock is
read, so results are fully deterministic for a fixed seed. Standard library
only. This module is intentionally standalone and does not import the
categorical/range coverage sampler in ``datacon_designspace_sampler``.
"""

import random

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

_KINDS = ("continuous", "integer")
_OPS = ("<", "<=", ">", ">=")


class ParameterSpec(object):
    """A single named design parameter with inclusive bounds.

    Parameters
    ----------
    name : str
        Non-empty parameter name.
    kind : str
        ``"continuous"`` or ``"integer"``.
    low, high : number
        Inclusive lower/upper bounds. Requires ``low <= high``. For integer
        parameters the bounds must be integral.
    """

    __slots__ = ("name", "kind", "low", "high")

    def __init__(self, name, kind, low, high):
        if not isinstance(name, str) or not name:
            raise ValueError("parameter name must be a non-empty string")
        if kind not in _KINDS:
            raise ValueError("kind must be one of %r, got %r" % (_KINDS, kind))
        if isinstance(low, bool) or isinstance(high, bool):
            raise ValueError("bounds must be numbers, not bools")
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise ValueError("bounds for %r must be numbers" % (name,))
        if not (low <= high):
            raise ValueError("parameter %r requires low <= high" % (name,))
        if kind == "integer":
            if int(low) != low or int(high) != high:
                raise ValueError("integer parameter %r needs integral bounds" % (name,))
            low = int(low)
            high = int(high)
        self.name = name
        self.kind = kind
        self.low = low
        self.high = high

    def contains(self, value):
        """Return True if ``value`` is within bounds and (if integer) integral."""
        if isinstance(value, bool):
            return False
        if not isinstance(value, (int, float)):
            return False
        if self.kind == "integer" and int(value) != value:
            return False
        return self.low <= value <= self.high

    def __repr__(self):
        return "ParameterSpec(%r, %r, %r, %r)" % (
            self.name, self.kind, self.low, self.high)


# ---------------------------------------------------------------------------
# Constraints (composable, evaluable objects - no eval of strings)
# ---------------------------------------------------------------------------

def _resolve(operand, assignment):
    """Resolve an operand to a number: a numeric constant or a parameter name.

    Returns ``None`` if ``operand`` is a parameter name absent from
    ``assignment`` (so a constraint over a missing parameter is reported as
    unsatisfiable rather than crashing).
    """
    if isinstance(operand, bool):
        return float(operand)
    if isinstance(operand, (int, float)):
        return operand
    if isinstance(operand, str):
        if operand in assignment:
            return assignment[operand]
        return None
    raise ValueError("operand must be a parameter name or number, got %r" % (operand,))


class Inequality(object):
    """A pairwise inequality ``a op b``.

    ``a`` and ``b`` are each either a parameter name (str) or a numeric
    constant. ``op`` is one of ``"<"``, ``"<="``, ``">"``, ``">="``. Examples:
    ``Inequality("width", "<", "length")`` or ``Inequality("wheel_radius", "<", 5)``.
    """

    __slots__ = ("a", "op", "b")

    def __init__(self, a, op, b):
        if op not in _OPS:
            raise ValueError("op must be one of %r, got %r" % (_OPS, op))
        if not isinstance(a, (str, int, float)) or isinstance(a, bool):
            raise ValueError("left operand must be a parameter name or number")
        if not isinstance(b, (str, int, float)) or isinstance(b, bool):
            raise ValueError("right operand must be a parameter name or number")
        self.a = a
        self.op = op
        self.b = b

    def parameters(self):
        """Return the set of parameter names referenced by this constraint."""
        names = set()
        if isinstance(self.a, str):
            names.add(self.a)
        if isinstance(self.b, str):
            names.add(self.b)
        return names

    def satisfied(self, assignment):
        left = _resolve(self.a, assignment)
        right = _resolve(self.b, assignment)
        if left is None or right is None:
            return False
        if self.op == "<":
            return left < right
        if self.op == "<=":
            return left <= right
        if self.op == ">":
            return left > right
        return left >= right

    def describe(self, assignment):
        """Return a human-readable description of this constraint's status."""
        left = _resolve(self.a, assignment)
        right = _resolve(self.b, assignment)
        return "inequality %r %s %r not satisfied (%r %s %r)" % (
            self.a, self.op, self.b, left, self.op, right)

    def __repr__(self):
        return "Inequality(%r, %r, %r)" % (self.a, self.op, self.b)


class Divisible(object):
    """A divisibility constraint ``param %% modulus == 0`` (integer-valued).

    This is the paper's non-trivial Lego constraint: ``Divisible("brick_length",
    3)`` requires the value of ``brick_length`` to be a multiple of 3.
    """

    __slots__ = ("param", "modulus")

    def __init__(self, param, modulus):
        if not isinstance(param, str) or not param:
            raise ValueError("param must be a non-empty parameter name")
        if isinstance(modulus, bool) or not isinstance(modulus, int):
            raise ValueError("modulus must be a positive integer")
        if modulus <= 0:
            raise ValueError("modulus must be a positive integer")
        self.param = param
        self.modulus = modulus

    def parameters(self):
        return {self.param}

    def satisfied(self, assignment):
        if self.param not in assignment:
            return False
        value = assignment[self.param]
        if isinstance(value, bool):
            return False
        if not isinstance(value, (int, float)):
            return False
        if int(value) != value:
            return False
        return int(value) % self.modulus == 0

    def describe(self, assignment):
        value = assignment.get(self.param, None)
        return "divisibility %r %% %d == 0 not satisfied (value=%r)" % (
            self.param, self.modulus, value)

    def __repr__(self):
        return "Divisible(%r, %d)" % (self.param, self.modulus)


# ---------------------------------------------------------------------------
# Design space
# ---------------------------------------------------------------------------

class DesignSpace(object):
    """A constrained parametric design space.

    Parameters
    ----------
    parameters : iterable of ParameterSpec
        Parameter specs; names must be unique.
    constraints : iterable, optional
        :class:`Inequality` / :class:`Divisible` objects (or any object exposing
        ``satisfied(assignment)`` and ``describe(assignment)``).
    """

    def __init__(self, parameters, constraints=None):
        specs = list(parameters)
        if not specs:
            raise ValueError("design space needs at least one parameter")
        seen = set()
        for spec in specs:
            if not isinstance(spec, ParameterSpec):
                raise ValueError("parameters must be ParameterSpec instances")
            if spec.name in seen:
                raise ValueError("duplicate parameter name %r" % (spec.name,))
            seen.add(spec.name)
        self.parameters = specs
        self._by_name = {spec.name: spec for spec in specs}
        self.constraints = list(constraints) if constraints else []

    def parameter_names(self):
        return [spec.name for spec in self.parameters]

    def violations(self, assignment):
        """Return a list of strings, one per failing bound/constraint.

        Reports, in order: missing parameters, out-of-bounds / non-integral
        values, then each unsatisfied constraint. An empty list means the
        assignment is valid.
        """
        problems = []
        for spec in self.parameters:
            if spec.name not in assignment:
                problems.append("missing parameter %r" % (spec.name,))
                continue
            value = assignment[spec.name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(
                    "parameter %r value %r is not a number" % (spec.name, value))
                continue
            if spec.kind == "integer" and int(value) != value:
                problems.append(
                    "parameter %r value %r is not integral" % (spec.name, value))
                continue
            if not (spec.low <= value <= spec.high):
                problems.append(
                    "parameter %r value %r out of bounds [%r, %r]" % (
                        spec.name, value, spec.low, spec.high))
        for constraint in self.constraints:
            if not constraint.satisfied(assignment):
                problems.append(constraint.describe(assignment))
        return problems

    def is_valid(self, assignment):
        """Return True iff ``assignment`` is within bounds, integral where
        required, and satisfies every constraint."""
        return not self.violations(assignment)

    # -- sampling ----------------------------------------------------------

    def _draw_one(self, rng):
        row = {}
        for spec in self.parameters:
            if spec.kind == "integer":
                row[spec.name] = rng.randint(spec.low, spec.high)
            else:
                row[spec.name] = rng.uniform(spec.low, spec.high)
        return row

    def sample_valid(self, n, seed, max_attempts_per=1000):
        """Return up to ``n`` DISTINCT valid assignments via rejection sampling.

        Uses ``random.Random(seed)``; integer parameters are drawn with
        ``randint`` and continuous parameters with ``uniform``. Each candidate is
        accepted only if :meth:`is_valid` and not already returned (distinct).

        The attempt budget is capped at ``n * max_attempts_per`` draws, so a
        tight or empty space never loops forever: this method then returns FEWER
        than ``n`` assignments (possibly an empty list). Callers should check
        ``len(result)`` to detect an under-filled request. Deterministic for a
        fixed ``seed``.
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        if max_attempts_per < 1:
            raise ValueError("max_attempts_per must be >= 1")
        rng = random.Random(seed)
        results = []
        seen = set()
        if n == 0:
            return results
        budget = n * max_attempts_per
        for _ in range(budget):
            row = self._draw_one(rng)
            if self.is_valid(row):
                key = tuple(row[name] for name in self.parameter_names())
                if key not in seen:
                    seen.add(key)
                    results.append(row)
                    if len(results) >= n:
                        break
        return results

    # -- enumeration -------------------------------------------------------

    def _axis_values(self, spec, step):
        if spec.kind == "integer":
            values = []
            v = spec.low
            while v <= spec.high:
                values.append(v)
                v += 1
            return values
        # continuous: evenly spaced grid using ``step``.
        if step is None or step <= 0:
            raise ValueError(
                "enumerate_valid needs a positive step for continuous parameter %r"
                % (spec.name,))
        values = []
        v = spec.low
        # Guard against floating drift by counting steps.
        count = int(round((spec.high - spec.low) / step)) + 1
        for i in range(max(count, 1)):
            values.append(spec.low + i * step)
        # Ensure the upper bound is representable and nothing exceeds it.
        return [x for x in values if x <= spec.high + 1e-9]

    def enumerate_valid(self, step=None, max_rows=1000000):
        """Enumerate valid assignments over a discretized grid.

        Integer parameters contribute every integer in ``[low, high]``, so for
        an all-integer space (e.g. the Lego brick) this is an EXACT enumeration
        of every valid assignment within bounds. Continuous parameters are
        discretized with ``step`` (a positive float applied to every continuous
        axis); ``step`` may be omitted only when there are no continuous
        parameters.

        Returns a list of valid assignment dicts. Raises ValueError if the raw
        grid would exceed ``max_rows`` rows (before filtering).
        """
        axes = []
        total = 1
        for spec in self.parameters:
            values = self._axis_values(spec, step)
            axes.append((spec.name, values))
            total *= max(len(values), 1)
            if total > max_rows:
                raise ValueError("grid too large (> %d rows)" % (max_rows,))

        rows = [{}]
        for name, values in axes:
            new_rows = []
            for base in rows:
                for v in values:
                    r = dict(base)
                    r[name] = v
                    new_rows.append(r)
            rows = new_rows

        return [row for row in rows if self.is_valid(row)]
