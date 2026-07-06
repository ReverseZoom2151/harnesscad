"""Deterministic component-design-space sampling for diverse wheel-rim generation.

Motivated by "Generative AI and CAD Automation for Diverse and Novel Mechanical
Component Designs Under Data Constraints", whose method must produce DIVERSE
wheel-rim designs across style categories (five-spoke, multispoke, mesh,
minimalist) and rim specs while operating under a limited data / generation
budget. To make a small budget count, we do not sample naively: we draw a
diverse, well-spread set of design specs across a mixed categorical + continuous
design space so that coverage of each design dimension is maximized.

This module provides:

* A ``DesignSpace`` description (a plain dict) with two dimension kinds:
    - ``("categorical", [choices...])`` e.g.
      ``"spoke_style" -> ("categorical", ["five-spoke","multispoke","mesh","minimalist"])``
    - ``("range", low, high)`` continuous, e.g. ``"rim_diameter" -> ("range", 14, 21)``
    - ``("int_range", low, high)`` integer-valued, e.g. ``"spoke_count" -> ("int_range", 5, 12)``
* ``uniform_sample`` - independent uniform draws.
* ``stratified_sample`` - a Latin-Hypercube-style stratified sampler that spreads
  each marginal (one value per stratum for continuous dims, even round-robin for
  categorical dims), maximizing per-dimension coverage under a fixed budget.
* ``grid_sample`` - full factorial / evenly-spaced grid.
* ``coverage_of_samples`` and ``marginal_coverage`` - quick diversity checks.

All randomness is seeded through ``random.Random(seed)`` so results are
deterministic given a seed. Standard library only.
"""

import random

# ---------------------------------------------------------------------------
# Design space description + validation
# ---------------------------------------------------------------------------

_CONTINUOUS_KINDS = ("range", "int_range")


def validate_space(space):
    """Validate a DesignSpace dict, raising ValueError on any malformed spec.

    A space is a dict mapping ``dim name -> spec`` where spec is one of:
      * ``("categorical", [choices...])`` with a non-empty list of choices
      * ``("range", low, high)`` with low < high (numbers)
      * ``("int_range", low, high)`` with integer low < high

    Returns the space unchanged when valid.
    """
    if not isinstance(space, dict) or not space:
        raise ValueError("design space must be a non-empty dict")
    for name, spec in space.items():
        if not isinstance(name, str) or not name:
            raise ValueError("dimension names must be non-empty strings")
        if not isinstance(spec, (tuple, list)) or len(spec) < 2:
            raise ValueError("spec for %r must be a tuple of length >= 2" % (name,))
        kind = spec[0]
        if kind == "categorical":
            if len(spec) != 2:
                raise ValueError("categorical spec for %r must be (\"categorical\", [choices])" % (name,))
            choices = spec[1]
            if not isinstance(choices, (list, tuple)) or len(choices) == 0:
                raise ValueError("categorical dim %r needs a non-empty list of choices" % (name,))
        elif kind in _CONTINUOUS_KINDS:
            if len(spec) != 3:
                raise ValueError("%s spec for %r must be (kind, low, high)" % (kind, name))
            low, high = spec[1], spec[2]
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise ValueError("bounds for %r must be numbers" % (name,))
            if isinstance(low, bool) or isinstance(high, bool):
                raise ValueError("bounds for %r must be numbers, not bools" % (name,))
            if not (low < high):
                raise ValueError("dim %r requires low < high" % (name,))
            if kind == "int_range":
                if int(low) != low or int(high) != high:
                    raise ValueError("int_range bounds for %r must be integers" % (name,))
        else:
            raise ValueError("unknown dim kind %r for %r" % (kind, name))
    return space


def _is_continuous(spec):
    return spec[0] in _CONTINUOUS_KINDS


# ---------------------------------------------------------------------------
# Single-value draws
# ---------------------------------------------------------------------------

def _draw_uniform(spec, rng):
    kind = spec[0]
    if kind == "categorical":
        return spec[1][rng.randrange(len(spec[1]))]
    low, high = spec[1], spec[2]
    if kind == "int_range":
        return rng.randint(int(low), int(high))
    return rng.uniform(low, high)


def _draw_in_stratum(spec, i, n, rng):
    """Draw one value inside stratum ``i`` of ``n`` for a continuous spec."""
    low, high = spec[1], spec[2]
    kind = spec[0]
    if kind == "int_range":
        low_i, high_i = int(low), int(high)
        total = high_i - low_i + 1
        # Partition the integer range [low_i, high_i] into n contiguous blocks.
        start = low_i + (i * total) // n
        end = low_i + ((i + 1) * total) // n - 1
        if end < start:
            end = start
        if end > high_i:
            end = high_i
        return rng.randint(start, end)
    width = (high - low) / n
    a = low + i * width
    b = low + (i + 1) * width
    return rng.uniform(a, b)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def uniform_sample(space, n, seed):
    """Return ``n`` independent uniform samples (list of dicts). Deterministic."""
    validate_space(space)
    if n < 0:
        raise ValueError("n must be >= 0")
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        row = {}
        for name, spec in space.items():
            row[name] = _draw_uniform(spec, rng)
        samples.append(row)
    return samples


def stratified_sample(space, n, seed):
    """Return ``n`` stratified (Latin-Hypercube-style) samples.

    For each continuous dimension, ``[low, high]`` is partitioned into ``n``
    equal strata and one value is drawn per stratum; the stratum order is
    shuffled per-dimension so dimensions are decorrelated. For each categorical
    dimension, choices are assigned round-robin (order shuffled) so every choice
    appears ``floor(n/k)`` or ``ceil(n/k)`` times. Assembled by index.

    Deterministic given ``seed``. Maximizes coverage of each marginal.
    """
    validate_space(space)
    if n < 0:
        raise ValueError("n must be >= 0")
    rng = random.Random(seed)

    # Build a per-dimension column of length n.
    columns = {}
    for name, spec in space.items():
        if _is_continuous(spec):
            order = list(range(n))
            rng.shuffle(order)
            col = [_draw_in_stratum(spec, stratum, n, rng) for stratum in order]
        else:
            choices = list(spec[1])
            k = len(choices)
            # Even round-robin: build a balanced multiset then shuffle.
            col = [choices[i % k] for i in range(n)]
            rng.shuffle(col)
        columns[name] = col

    samples = []
    for idx in range(n):
        samples.append({name: columns[name][idx] for name in space})
    return samples


def grid_sample(space, per_dim, seed=None):
    """Return a full factorial grid of design specs.

    Categorical dims contribute all their choices; each continuous dim
    contributes ``per_dim`` evenly-spaced values (endpoints inclusive). Total
    row count is capped at 100000; exceeding it raises ValueError.
    """
    validate_space(space)
    if per_dim < 1:
        raise ValueError("per_dim must be >= 1")

    axis_values = {}
    total = 1
    for name, spec in space.items():
        if _is_continuous(spec):
            low, high = spec[1], spec[2]
            if per_dim == 1:
                vals = [(low + high) / 2.0]
            else:
                step = (high - low) / (per_dim - 1)
                vals = [low + step * i for i in range(per_dim)]
            if spec[0] == "int_range":
                vals = sorted(set(int(round(v)) for v in vals))
            axis_values[name] = vals
        else:
            axis_values[name] = list(spec[1])
        total *= len(axis_values[name])
        if total > 100000:
            raise ValueError("grid too large (> 100000 rows)")

    names = list(space.keys())

    # Cartesian product without itertools dependency concerns (stdlib ok, but
    # build manually to keep row dicts).
    rows = [{}]
    for name in names:
        new_rows = []
        for base in rows:
            for v in axis_values[name]:
                r = dict(base)
                r[name] = v
                new_rows.append(r)
        rows = new_rows

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(rows)
    return rows


# ---------------------------------------------------------------------------
# Coverage diagnostics
# ---------------------------------------------------------------------------

def _bin_index(spec, value, bins):
    """Return the bin index in [0, bins) for a continuous value, clamped."""
    low, high = spec[1], spec[2]
    if high == low:
        return 0
    frac = (value - low) / (high - low)
    idx = int(frac * bins)
    if idx < 0:
        idx = 0
    if idx >= bins:
        idx = bins - 1
    return idx


def coverage_of_samples(samples, space, bins=4):
    """Fraction of occupied strata-cells across all dimensions.

    Categorical dims contribute ``len(choices)`` cells; continuous dims
    contribute ``bins`` cells. Returns ``occupied distinct cells / total cells``
    in ``[0, 1]``. Raises ValueError if the total cell count exceeds 1e7.
    """
    validate_space(space)
    if bins < 1:
        raise ValueError("bins must be >= 1")

    total_cells = 1
    for name, spec in space.items():
        if _is_continuous(spec):
            total_cells *= bins
        else:
            total_cells *= len(spec[1])
        if total_cells > 1e7:
            raise ValueError("cell space too large (> 1e7)")

    occupied = set()
    for row in samples:
        key = []
        for name, spec in space.items():
            v = row[name]
            if _is_continuous(spec):
                key.append(_bin_index(spec, v, bins))
            else:
                try:
                    key.append(spec[1].index(v) if isinstance(spec[1], list) else list(spec[1]).index(v))
                except ValueError:
                    key.append(-1)
        occupied.add(tuple(key))

    if total_cells == 0:
        return 0.0
    return len(occupied) / float(total_cells)


def marginal_coverage(samples, space):
    """Per-dimension fraction of that dim's strata/choices that are hit.

    Categorical: distinct choices hit / total choices.
    Continuous: use ``bins = min(len(samples), 10)`` equal bins; distinct bins
    hit / bins.
    """
    validate_space(space)
    result = {}
    n = len(samples)
    for name, spec in space.items():
        if _is_continuous(spec):
            bins = min(n, 10)
            if bins < 1:
                result[name] = 0.0
                continue
            hit = set()
            for row in samples:
                hit.add(_bin_index(spec, row[name], bins))
            result[name] = len(hit) / float(bins)
        else:
            choices = list(spec[1])
            hit = set()
            for row in samples:
                if row[name] in choices:
                    hit.add(row[name])
            result[name] = len(hit) / float(len(choices))
    return result
