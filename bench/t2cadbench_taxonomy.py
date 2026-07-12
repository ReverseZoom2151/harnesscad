"""Text2CAD-Bench difficulty-tier taxonomy (L1-L4 classification).

Deterministic re-encoding of the Text2CAD-Bench geometric-complexity hierarchy
(Wang et al., "Text2CAD-Bench: A Benchmark for LLM-based Text-to-Parametric CAD
Generation", Section 3.3, Appendix A/F). The benchmark stratifies 600 examples
into four tiers *by geometric complexity of the required CadQuery operations*,
not by textual detail:

  * L1 (Basic)        -- primitive shapes with basic finishing features
                         (box/cylinder/sphere/polygon + extrude, chamfer,
                         fillet, through-hole). Concise op chains (~8 LOC).
  * L2 (Intermediate) -- compositional complexity: boolean operations
                         (cut/union/intersect/revolve) and standard features
                         that must be correctly sequenced (feature interactions
                         such as filleting boolean-cut edges).
  * L3 (Advanced)     -- sweep, loft, shell, twist-extrude, freeform/parametric
                         curves (Bezier/spline) and complex patterns.
  * L4 (Real-World)   -- application-domain examples (industrial, consumer,
                         medical, architectural, educational). Determined by
                         application-domain metadata, orthogonal to geometry:
                         an L4 example may be geometrically simple or complex.

Tier is the maximum operation tier present (a single sweep makes a program L3).
L4 is assigned when an application domain is supplied, while the underlying
geometric tier is still reported.

This is DISTINCT from ``bench/engdesign_taxonomy`` (VLM task taxonomy, paper 85)
and ``bench/muse_*`` (assemblability funnel, paper 133): here we classify a
CadQuery operation set into the Text2CAD-Bench L1-L4 geometric hierarchy and
validate benchmark-split composition. Generation is external -- operation lists
and metadata are injected.

No wall clock, no randomness.
"""

from __future__ import annotations

# CadQuery operation -> geometric tier (1, 2, or 3). Names are normalised to
# lower case before lookup; a trailing "()" or leading "." is stripped.
OPERATION_TIERS = {
    # --- L1: primitives + basic finishing ---
    "box": 1, "cylinder": 1, "sphere": 1, "cone": 1, "makecone": 1,
    "wedge": 1, "polygon": 1, "circle": 1, "rect": 1, "rectangle": 1,
    "ellipse": 1, "extrude": 1, "hole": 1, "cutthruall": 1, "cbore": 1,
    "cborehole": 1, "cskhole": 1, "chamfer": 1, "fillet": 1,
    "workplane": 1, "faces": 1, "edges": 1, "vertices": 1, "center": 1,
    "moveto": 1, "lineto": 1, "line": 1, "close": 1, "twodpoints": 1,
    # --- L2: boolean / compositional ---
    "cut": 2, "union": 2, "intersect": 2, "combine": 2, "revolve": 2,
    "mirror": 2,
    # --- L3: advanced features / freeform / patterns ---
    "sweep": 3, "loft": 3, "shell": 3, "twistextrude": 3,
    "spline": 3, "bezier": 3, "parametriccurve": 3, "interpplate": 3,
    "polararray": 3, "rarray": 3, "eachpoint": 3,
}

TIER_LABELS = {1: "L1", 2: "L2", 3: "L3", 4: "L4"}
TIER_NAMES = {1: "Basic", 2: "Intermediate", 3: "Advanced", 4: "Real-World"}

# Advanced modeling features the benchmark deliberately adds over prior
# sketch-extrude-only datasets (Section 1, contributions).
ADVANCED_FEATURES = ("chamfer", "fillet", "sweep", "loft", "shell",
                     "polararray", "rarray")

# Dataset composition (Appendix A, Table 5).
LEVEL_COUNTS = {"L1": 200, "L2": 200, "L3": 100, "L4": 100}
TOTAL_EXAMPLES = 600
TOTAL_PROMPTS = 1200  # dual-style: 600 geometric + 600 sequence

# L4 application-domain distribution (Table 5), as target fractions.
L4_DOMAINS = {
    "industrial": 0.40,
    "consumer": 0.25,
    "medical": 0.15,
    "architectural": 0.10,
    "educational": 0.10,
}


def normalize_operation(op):
    """Canonicalise an operation token: lower-case, strip '.', '()', args."""
    s = str(op).strip().lower()
    if s.startswith("."):
        s = s[1:]
    # drop argument list / trailing parens
    if "(" in s:
        s = s[:s.index("(")]
    return s.strip()


def operation_tier(op):
    """Geometric tier (1-3) of a single operation, or None if unknown."""
    return OPERATION_TIERS.get(normalize_operation(op))


def classify_operations(operations):
    """Classify a CadQuery operation list into a geometric tier (L1-L3).

    operations : iterable of operation tokens (e.g. ["box", ".cut", "loft"]).

    Returns a dict:
      tier        : int 1-3 (max operation tier; defaults to 1 if only unknown
                    or empty).
      label       : "L1"/"L2"/"L3".
      name        : human tier name.
      driver      : the highest-tier known operation that set the tier
                    (None if empty / all unknown).
      unknown     : sorted tuple of unrecognised operation tokens.
      advanced_features : sorted tuple of advanced features used.
      op_count    : number of operations (paper's "API Calls" proxy).
    """
    ops = [normalize_operation(o) for o in operations]
    ops = [o for o in ops if o]
    tier = 1
    driver = None
    unknown = []
    for o in ops:
        t = OPERATION_TIERS.get(o)
        if t is None:
            unknown.append(o)
            continue
        if t > tier:
            tier = t
            driver = o
        elif driver is None and t == tier:
            driver = o
    adv = sorted({o for o in ops if o in ADVANCED_FEATURES})
    return {
        "tier": tier,
        "label": TIER_LABELS[tier],
        "name": TIER_NAMES[tier],
        "driver": driver,
        "unknown": tuple(sorted(set(unknown))),
        "advanced_features": tuple(adv),
        "op_count": len(ops),
    }


def classify_example(operations, application_domain=None):
    """Classify a full benchmark example, honouring L4 domain metadata.

    If ``application_domain`` is a non-empty string, the example is L4
    (real-world) regardless of geometry; the geometric tier is still reported
    under ``geometric_tier``/``geometric_label`` for reference.

    Returns the geometric classification dict augmented with:
      label / tier / name : L4 if a domain is supplied, else the geometric tier.
      geometric_tier, geometric_label : the underlying geometry tier.
      application_domain   : the normalised domain (or None).
    """
    geo = classify_operations(operations)
    geo["geometric_tier"] = geo["tier"]
    geo["geometric_label"] = geo["label"]
    domain = None
    if application_domain is not None and str(application_domain).strip():
        domain = str(application_domain).strip().lower()
        geo["tier"] = 4
        geo["label"] = "L4"
        geo["name"] = TIER_NAMES[4]
    geo["application_domain"] = domain
    return geo


def validate_split(counts):
    """Compare a benchmark split's per-tier counts against the paper's design.

    counts : mapping "L1".."L4" -> int (missing tiers treated as 0).

    Returns a dict:
      counts    : normalised counts,
      expected  : LEVEL_COUNTS,
      total     : sum of counts,
      matches   : True iff counts equal LEVEL_COUNTS exactly,
      deltas    : per-tier (count - expected).
    """
    norm = {k: int(counts.get(k, 0)) for k in ("L1", "L2", "L3", "L4")}
    deltas = {k: norm[k] - LEVEL_COUNTS[k] for k in norm}
    return {
        "counts": norm,
        "expected": dict(LEVEL_COUNTS),
        "total": sum(norm.values()),
        "matches": norm == LEVEL_COUNTS,
        "deltas": deltas,
    }


def l4_domain_deviation(domain_counts):
    """Max absolute deviation of an L4 domain mix from the target fractions.

    domain_counts : mapping domain-name -> int (names normalised, unknown
        domains ignored for the target comparison but counted in the total).

    Returns a dict with per-domain observed fraction, target, absolute
    deviation, the total count, and ``max_deviation``. If the total is zero a
    ValueError is raised.
    """
    norm = {}
    for k, v in domain_counts.items():
        norm[str(k).strip().lower()] = norm.get(str(k).strip().lower(), 0) + int(v)
    total = sum(norm.values())
    if total <= 0:
        raise ValueError("empty L4 domain counts")
    per = {}
    max_dev = 0.0
    for dom, target in L4_DOMAINS.items():
        frac = norm.get(dom, 0) / total
        dev = abs(frac - target)
        per[dom] = {"fraction": frac, "target": target, "deviation": dev}
        if dev > max_dev:
            max_dev = dev
    return {"per_domain": per, "total": total, "max_deviation": max_dev}
