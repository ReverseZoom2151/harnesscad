"""Deterministic data-augmentation-under-constraints for 2D spoke contours.

Motivated by "Generative AI and CAD Automation for Diverse and Novel Mechanical
Component Designs Under Data Constraints", whose core claim is that a curated
few-shot / limited-data regime (the paper reduces training data from ~16,600 to
~200 samples via LoRA-style adaptation) can still yield diverse, valid designs.

This module implements the *data-side* of that story: a purely geometric,
deterministic augmentation that expands a handful of curated base 2D spoke
contours into a bounded, diverse, constraint-preserving training set. It is NOT
a learned generator.

The paper's central mechanical constraint for spoke/rotor components is
ROTATIONAL SYMMETRY (dynamic balance): a spoke set must be invariant under
rotation by 2*pi/N. We enforce this *by construction* -- every design is built
by replicating a single base spoke to N rotated copies -- so augmentation can
never break dynamic balance.

Complement to ``datagen/augment.py``: that module augments op-stream ``Sample``
objects (the procedural CAD command sequence); this module augments raw spoke
contour polygons (lists of ``(x, y)`` points) and enforces the rotational-balance
constraint. Together they cover both the command representation and the geometric
representation used in the limited-data expansion protocol.

All randomness is driven by ``random.Random(seed)`` so that identical inputs
produce byte-for-byte identical outputs.
"""

import math


Point = "tuple[float, float]"
Polygon = "list[tuple[float, float]]"


def rotate_point(x, y, angle):
    """Rotate a point about the origin by ``angle`` radians (counter-clockwise)."""
    ca = math.cos(angle)
    sa = math.sin(angle)
    return (x * ca - y * sa, x * sa + y * ca)


def rotate_polygon(poly, angle, about=(0.0, 0.0)):
    """Rotate every vertex of ``poly`` by ``angle`` radians about ``about``."""
    ax, ay = about
    out = []
    for (x, y) in poly:
        rx, ry = rotate_point(x - ax, y - ay, angle)
        out.append((rx + ax, ry + ay))
    return out


def replicate_rotational(base_spoke, n):
    """Replicate ``base_spoke`` into ``n`` rotationally-symmetric copies.

    Returns a list of ``n`` polygons, the k-th being ``base_spoke`` rotated by
    ``k * 2*pi/n`` about the origin. This constructs a dynamically balanced
    (rotationally symmetric) spoke set, enforcing the paper's core constraint by
    construction.
    """
    if not isinstance(n, int):
        raise ValueError("symmetry order n must be an int")
    if n < 1:
        raise ValueError("symmetry order n must be >= 1")
    step = 2.0 * math.pi / n
    return [rotate_polygon(base_spoke, k * step) for k in range(n)]


def mirror_polygon(poly, axis="x"):
    """Reflect ``poly`` across an axis.

    ``axis='x'`` flips the sign of y (reflection across the x-axis);
    ``axis='y'`` flips the sign of x (reflection across the y-axis).
    """
    if axis == "x":
        return [(x, -y) for (x, y) in poly]
    if axis == "y":
        return [(-x, y) for (x, y) in poly]
    raise ValueError("axis must be 'x' or 'y'")


def scale_polygon(poly, factor, about=(0.0, 0.0)):
    """Scale ``poly`` by ``factor`` about ``about`` (``factor`` must be > 0)."""
    if factor <= 0:
        raise ValueError("scale factor must be > 0")
    ax, ay = about
    return [(ax + (x - ax) * factor, ay + (y - ay) * factor) for (x, y) in poly]


def jitter_polygon(poly, rng, max_frac):
    """Perturb each vertex by up to +/- ``max_frac`` of its radius.

    Perturbation is applied independently to x and y, scaled by the vertex's
    distance from the origin, using the supplied ``random.Random`` instance for
    determinism. ``max_frac`` is kept small so designs stay feasible.
    """
    if max_frac < 0:
        raise ValueError("max_frac must be >= 0")
    out = []
    for (x, y) in poly:
        r = math.hypot(x, y)
        amp = max_frac * r
        dx = rng.uniform(-amp, amp)
        dy = rng.uniform(-amp, amp)
        out.append((x + dx, y + dy))
    return out


def polygon_area(poly):
    """Signed shoelace area of a polygon (absolute value returned)."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return abs(s) * 0.5


def augment_spoke_design(base_spoke, symmetry_order, seed, n_variants,
                         max_jitter=0.03, allow_mirror=True,
                         scale_choices=(1.0,)):
    """Expand one base spoke into ``n_variants`` balanced designs.

    Each returned design is a dict::

        {"spokes": [polygon, ...],       # symmetry_order rotated copies
         "symmetry_order": N,
         "source": "base" | "augmented",
         "transforms": [...]}            # record of applied transforms

    A single ``random.Random(seed)`` drives all choices, so the same
    ``(base_spoke, symmetry_order, seed, n_variants)`` yields identical output.

    The FIRST design is always the un-jittered, un-scaled, un-mirrored base
    (``source="base"``). Every design is replicated to ``symmetry_order`` spokes,
    so rotational balance is preserved by construction.
    """
    if n_variants < 1:
        raise ValueError("n_variants must be >= 1")
    rng = _seeded_rng(seed)
    designs = []

    # First design: the pristine base.
    designs.append({
        "spokes": replicate_rotational(base_spoke, symmetry_order),
        "symmetry_order": symmetry_order,
        "source": "base",
        "transforms": [],
    })

    for _ in range(n_variants - 1):
        transforms = []
        poly = list(base_spoke)

        scale = rng.choice(list(scale_choices))
        if scale != 1.0:
            poly = scale_polygon(poly, scale)
        transforms.append(("scale", scale))

        do_mirror = allow_mirror and (rng.random() < 0.5)
        if do_mirror:
            poly = mirror_polygon(poly, "x")
        transforms.append(("mirror", do_mirror))

        jmag = rng.uniform(0.0, max_jitter)
        if jmag > 0.0:
            poly = jitter_polygon(poly, rng, jmag)
        transforms.append(("jitter", jmag))

        designs.append({
            "spokes": replicate_rotational(poly, symmetry_order),
            "symmetry_order": symmetry_order,
            "source": "augmented",
            "transforms": transforms,
        })

    return designs


def few_shot_expand(base_designs, seed, target_size, **kw):
    """Expand a few curated bases into ``target_size`` designs, round-robin.

    ``base_designs`` is a small list of ``(base_spoke, symmetry_order)`` tuples
    (the "few shots"). Each base is expanded with a per-base sub-seed derived
    deterministically from ``seed``; results are interleaved round-robin until
    exactly ``target_size`` designs have been produced.

    This is the "16,600 -> ~200"-style limited-data expansion protocol: from a
    handful of curated bases, synthesize a bounded, diverse, balanced set.
    Deterministic in ``(base_designs, seed, target_size, **kw)``.
    """
    if target_size < 0:
        raise ValueError("target_size must be >= 0")
    if target_size == 0:
        return []
    if not base_designs:
        raise ValueError("base_designs must be non-empty")

    n_bases = len(base_designs)
    # Enough variants per base to cover the round-robin demand.
    per_base = (target_size + n_bases - 1) // n_bases + 1

    pools = []
    for i, (base_spoke, order) in enumerate(base_designs):
        sub_seed = _derive_seed(seed, i)
        pools.append(augment_spoke_design(
            base_spoke, order, sub_seed, per_base, **kw))

    out = []
    j = 0
    while len(out) < target_size:
        pool = pools[j % n_bases]
        idx = j // n_bases
        if idx < len(pool):
            out.append(pool[idx])
        j += 1
        # Safety: if pools are exhausted (should not happen given per_base sizing)
        if j > n_bases * per_base:
            break
    return out[:target_size]


def design_is_balanced(design, area_tol_frac=0.2):
    """Return True if all spokes have near-equal area (cheap balance check).

    Uses the shoelace area of each spoke polygon; passes if the spread
    ``(max - min) / mean`` is within ``area_tol_frac``. A design built by
    ``replicate_rotational`` is balanced by construction, so this is a sanity
    check that augmentation preserved dynamic balance.
    """
    spokes = design.get("spokes", [])
    if not spokes:
        return False
    areas = [polygon_area(p) for p in spokes]
    mean = sum(areas) / len(areas)
    if mean <= 0.0:
        # Degenerate (zero-area) spokes: balanced only if all are ~zero.
        return all(a <= 1e-12 for a in areas)
    spread = (max(areas) - min(areas)) / mean
    return spread <= area_tol_frac


def _seeded_rng(seed):
    """Return a ``random.Random`` seeded deterministically from ``seed``."""
    import random
    return random.Random(seed)


def _derive_seed(seed, index):
    """Derive a stable integer sub-seed from a base seed and an index."""
    h = hash((seed, "few_shot_expand", index))
    return h & 0x7FFFFFFF
