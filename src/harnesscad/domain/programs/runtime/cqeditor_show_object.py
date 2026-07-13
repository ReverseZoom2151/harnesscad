"""Deterministic model of CQ-editor's ``show_object`` result collection.

CQ-editor executes a user CAD script in a namespace into which it injects a
``show_object(obj, name=None, options={})`` helper.  When no explicit name is
given, it recovers one by reverse-looking-up the object in the caller's locals
by identity, falling back to the object id.  It also injects a ``rand_color``
helper whose randomness is made reproducible by seeding the global RNG with a
fixed constant before every render.

This module reimplements those two deterministic behaviours, GUI-free and
without relying on live stack frames: name inference operates on an explicit
namespace mapping, and colour generation uses a private seeded ``random.Random``
so results are reproducible independent of global RNG state.
"""

from random import Random


# The fixed seed CQ-editor uses before each render so that show_object colours
# are stable from run to run.
DEFAULT_COLOR_SEED = 59798267586177

# Brightness bounds (out of 255) used by CQ-editor's rand_color helper; the
# upper bound is kept well below 255 to avoid washed-out near-white colours.
_LOWER = 10
_UPPER = 100


def infer_object_name(obj, namespace, fallback=None):
    """Recover the variable name bound to *obj* within *namespace*.

    Mirrors CQ-editor's implicit ``show_object`` naming: search the namespace
    for a value identical (by ``==`` / index lookup, matching the original
    ``list(values).index(obj)``) to *obj* and return its key; if none is found,
    return *fallback* if given, else ``str(id(obj))``.

    :param obj: the object whose bound name is wanted.
    :param namespace: a mapping of names to values (e.g. module/frame locals).
    :param fallback: optional value to return when no name matches; when
        ``None`` the object id string is used, matching the original.
    :returns: the inferred name string.
    """
    keys = list(namespace.keys())
    values = list(namespace.values())
    try:
        idx = values.index(obj)
    except ValueError:
        return str(id(obj)) if fallback is None else fallback
    return keys[idx]


def collect_shown_objects(namespace, predicate):
    """Collect the objects in *namespace* that satisfy *predicate*.

    Reimplements CQ-editor's ``find_cq_objects`` fallback (used when a script
    calls no ``show_object``): every value passing *predicate* is returned keyed
    by its name, with private ``_``-prefixed names skipped.  Insertion order of
    *namespace* is preserved for determinism.

    :param namespace: mapping of names to values.
    :param predicate: callable ``value -> bool`` selecting objects to keep.
    :returns: an ordered ``dict`` of name -> value.
    """
    return {
        name: value
        for name, value in namespace.items()
        if not name.startswith("_") and predicate(value)
    }


def rand_color(alpha=0.0, cfloat=False, rng=None):
    """Generate a reproducible bounded-brightness colour for ``show_object``.

    Deterministic reimplementation of CQ-editor's ``_rand_color``.  Each channel
    is drawn uniformly from ``[10, 100]`` (out of 255).  With ``cfloat=True`` an
    ``(r, g, b, alpha)`` tuple of 0..1 floats is returned; otherwise a
    ``{"alpha": ..., "color": (r, g, b)}`` dict of 0..255 ints is returned.

    :param alpha: transparency component to embed in the result.
    :param cfloat: choose the float-tuple output form when true.
    :param rng: a ``random.Random`` instance; defaults to one seeded with
        :data:`DEFAULT_COLOR_SEED`, making calls reproducible by default.
    :returns: a colour dict (ints) or an ``(r, g, b, alpha)`` tuple (floats).
    """
    if rng is None:
        rng = Random(DEFAULT_COLOR_SEED)
    if cfloat:
        return (
            rng.randint(_LOWER, _UPPER) / 255,
            rng.randint(_LOWER, _UPPER) / 255,
            rng.randint(_LOWER, _UPPER) / 255,
            alpha,
        )
    return {
        "alpha": alpha,
        "color": (
            rng.randint(_LOWER, _UPPER),
            rng.randint(_LOWER, _UPPER),
            rng.randint(_LOWER, _UPPER),
        ),
    }


def color_sequence(count, alpha=0.0, cfloat=False, seed=DEFAULT_COLOR_SEED):
    """Return *count* reproducible colours drawn from a single seeded stream.

    Useful for assigning stable distinct colours to a set of shown objects: a
    single ``random.Random(seed)`` drives every colour so the whole sequence is
    reproducible and identical across runs with the same *seed*.

    :param count: number of colours to generate (must be non-negative).
    :param alpha: transparency component for each colour.
    :param cfloat: choose the float-tuple output form when true.
    :param seed: seed for the shared RNG.
    :returns: a list of *count* colours (see :func:`rand_color` for the form).
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    rng = Random(seed)
    return [rand_color(alpha=alpha, cfloat=cfloat, rng=rng) for _ in range(count)]
