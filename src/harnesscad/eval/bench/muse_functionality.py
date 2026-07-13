"""MUSE functionality scorer (design-intent alignment, Functionality pillar).

Deterministic re-encoding of the MUSE benchmark's Functionality pillar (Dong et
al., "MUSE: Benchmarking Manufacturable, Functional, and Assemblable
Text-to-CAD Generation"). The pillar has two binary sub-criteria (Table 8):

  * Functional -- does the design provide the structures needed for its
    intended primary (must-have, ~70%) and auxiliary (nice-to-have, ~30%)
    functions, with every component parameter inside the Valid Parameter Space
    Omega (Definition 2)?
  * Robust     -- is the design stable and structurally reliable: does the
    projected centre of mass fall inside the ground support polygon, are there
    enough ground contacts, and do load-bearing members meet a minimum
    thickness (force-transfer path, Table 8)?

This is distinct from ``verifiers/functional.py`` and ``quality/fitness.py``:
here the two MUSE sub-criteria are computed from an injected structured design
with explicit required structures, parameter ranges, and support geometry.

No wall clock, no randomness.
"""

from __future__ import annotations

# Default rubric weighting from the paper (Section: Functional Adaptation).
MUST_HAVE_WEIGHT = 0.70
NICE_TO_HAVE_WEIGHT = 0.30


def parameters_within_omega(parameters, ranges):
    """Definition 2 check: every named parameter lies in its valid range.

    parameters : mapping name -> value.
    ranges     : mapping name -> (lo, hi) inclusive.
    Returns (ok, out_of_range) where out_of_range is a tuple of offending names.
    Parameters without a declared range are ignored; a declared range whose
    parameter is missing counts as a violation (unspecified required parameter).
    """
    offending = []
    for name, (lo, hi) in ranges.items():
        if lo > hi:
            raise ValueError("inverted range for %r" % (name,))
        if name not in parameters:
            offending.append(name)
            continue
        v = parameters[name]
        if v < lo or v > hi:
            offending.append(name)
    return (not offending, tuple(offending))


def _coverage(required, present):
    """Fraction of required structures that are present (case-insensitive)."""
    req = [s.strip().lower() for s in required if s and s.strip()]
    if not req:
        return 1.0
    have = {s.strip().lower() for s in present if s and s.strip()}
    return sum(1 for s in req if s in have) / len(req)


def score_functional(design, *, must_have_weight=MUST_HAVE_WEIGHT,
                     nice_weight=NICE_TO_HAVE_WEIGHT, threshold=None):
    """Binary Functional sub-criterion for one injected design.

    design keys:
      structures        : iterable of structure names present in the model.
      must_have         : iterable of required must-have structure names.
      nice_to_have      : iterable of nice-to-have structure names (optional).
      parameters        : mapping name -> value (optional).
      parameter_ranges  : mapping name -> (lo, hi) i.e. Omega (optional).

    The must-have function must be fully covered and all parameters must be in
    Omega; the weighted coverage score must also clear ``threshold`` (defaults
    to the must-have weight, i.e. must-have fully satisfied). Returns
    {"functional": 0/1, "coverage": float, "omega_ok": bool, "reasons": (...)}.
    """
    if must_have_weight < 0 or nice_weight < 0:
        raise ValueError("weights must be non-negative")
    if threshold is None:
        threshold = must_have_weight
    must_cov = _coverage(design.get("must_have", ()), design.get("structures", ()))
    nice_cov = _coverage(design.get("nice_to_have", ()), design.get("structures", ()))
    coverage = must_have_weight * must_cov + nice_weight * nice_cov

    omega_ok, offending = parameters_within_omega(
        design.get("parameters", {}), design.get("parameter_ranges", {}))

    reasons = []
    if must_cov < 1.0:
        reasons.append("must_have_incomplete")
    if not omega_ok:
        reasons.extend("param_out_of_range:%s" % n for n in offending)
    if coverage + 1e-9 < threshold:
        reasons.append("below_coverage_threshold")

    passed = (must_cov >= 1.0) and omega_ok and (coverage + 1e-9 >= threshold)
    return {"functional": 1 if passed else 0, "coverage": coverage,
            "omega_ok": omega_ok, "reasons": tuple(reasons)}


def _point_in_polygon(point, polygon):
    """Ray-casting point-in-polygon (boundary counts as inside)."""
    x, y = point
    n = len(polygon)
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # On-segment (boundary) test.
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) < 1e-9 and min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 \
                and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
            return True
        if (y1 > y) != (y2 > y):
            xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xint:
                inside = not inside
    return inside


def _convex_hull(points):
    """Andrew's monotone chain hull of a set of (x, y) points."""
    pts = sorted(set(points))
    if len(pts) <= 1:
        return list(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def support_polygon(contacts):
    """Convex hull of ground-contact (x, y) points -> support polygon."""
    return _convex_hull([tuple(c) for c in contacts])


def score_robust(design, *, min_contacts=3, min_member_thickness=1.0):
    """Binary Robust sub-criterion for one injected design.

    design keys:
      ground_contacts    : iterable of (x, y) ground contact points.
      center_of_mass     : (x, y) projected centre of mass.
      load_bearing_members : iterable of dicts with "thickness" (mm) and
                             optional "connected" (bool, load path intact).
    A design is robust when it has at least ``min_contacts`` non-collinear
    ground contacts forming a polygon, the centre of mass projects inside that
    support polygon, and every load-bearing member is thick enough and
    connected. Returns {"robust": 0/1, "reasons": (...), "support_area": float}.
    """
    contacts = [tuple(c) for c in design.get("ground_contacts", ())]
    com = design.get("center_of_mass")
    reasons = []

    poly = support_polygon(contacts)
    area = _polygon_area(poly)
    if len(contacts) < min_contacts:
        reasons.append("insufficient_ground_contacts")
    if area <= 1e-9:
        reasons.append("degenerate_support_polygon")
    if com is None:
        reasons.append("unknown_center_of_mass")
    elif area > 1e-9 and not _point_in_polygon(tuple(com), poly):
        reasons.append("com_outside_support")

    for m in design.get("load_bearing_members", ()):
        name = m.get("name", "?")
        if m.get("thickness", 0.0) < min_member_thickness:
            reasons.append("thin_member:%s" % name)
        if not m.get("connected", True):
            reasons.append("broken_load_path:%s" % name)

    return {"robust": 0 if reasons else 1, "reasons": tuple(reasons),
            "support_area": area}


def _polygon_area(polygon):
    n = len(polygon)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def muse_functionality(design, **kwargs):
    """Full Functionality pillar score for one injected design.

    Recognised keyword args are forwarded: must_have_weight, nice_weight,
    threshold (Functional) and min_contacts, min_member_thickness (Robust).
    Returns the two binary sub-criteria, their average (MUSE Table 3 pillar
    score), and the merged reason list.
    """
    func_keys = {"must_have_weight", "nice_weight", "threshold"}
    robust_keys = {"min_contacts", "min_member_thickness"}
    func_kw = {k: v for k, v in kwargs.items() if k in func_keys}
    robust_kw = {k: v for k, v in kwargs.items() if k in robust_keys}
    unknown = set(kwargs) - func_keys - robust_keys
    if unknown:
        raise TypeError("unexpected keyword args: %s" % sorted(unknown))
    f = score_functional(design, **func_kw)
    r = score_robust(design, **robust_kw)
    average = (f["functional"] + r["robust"]) / 2.0
    return {"functional": f["functional"], "robust": r["robust"],
            "average": average,
            "reasons": tuple(f["reasons"]) + tuple(r["reasons"])}
