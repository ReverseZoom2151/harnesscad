"""MUSE three-stage funnel scorecard (aggregate benchmark protocol).

Deterministic implementation of the MUSE benchmark's funnel-style, three-stage
evaluation protocol (Dong et al., "MUSE: Benchmarking Manufacturable,
Functional, and Assemblable Text-to-CAD Generation", Section 3.3 / Tables 2-3):

  Stage 1 -- Code Validity:  the generated CadQuery script runs in the sandbox
             and exports a STEP file (one boolean, ``sandbox_success``).
  Stage 2 -- Geometric Validity: four binary geometry checks --
             Watertight, Manifold, Self-Intersection-Free, Overlap-Free.
             ``geom_valid`` iff all four pass.
  Stage 3 -- Design-Intent Alignment: six binary sub-criteria grouped into
             three pillars, each pillar the mean of its two sub-criteria:
               Functionality   = mean(functional, robust)
               Manufacturability = mean(well_toleranced, manufacturable)
               Assemblability  = mean(assembly_ready, connectable)
             Final Score = mean of the three pillar averages.

The protocol is strictly sequential (a funnel): a sample that fails an earlier
stage receives 0 for every downstream metric. This module gates and aggregates
per-design records; the three pillar scorers live in ``bench/muse_functionality``,
``bench/muse_manufacturability`` and ``bench/muse_assemblability``. Generation is
external -- design results are injected.

No wall clock, no randomness.
"""

from __future__ import annotations

GEOMETRY_CHECKS = ("watertight", "manifold", "self_intersection_free",
                   "overlap_free")
SUB_CRITERIA = ("functional", "robust", "well_toleranced", "manufacturable",
                "assembly_ready", "connectable")


def _clamp_unit(name, value):
    v = float(value)
    if v < 0.0 or v > 1.0:
        raise ValueError("%s must be in [0, 1], got %r" % (name, value))
    return v


def evaluate_design(record):
    """Apply the funnel to one injected design record.

    record keys:
      sandbox_success : bool -- Stage 1 code validity.
      watertight, manifold, self_intersection_free, overlap_free : bool/0-1.
      functional, robust, well_toleranced, manufacturable,
      assembly_ready, connectable : 0-1 sub-criteria (binary, or human-mean
      fractions).

    Returns a dict with the gated geometry checks, per-check and geom_valid
    flags, the three gated pillar averages, and the gated final score. Any
    stage that is gated out by an earlier failure is reported as 0.
    """
    code_ok = bool(record.get("sandbox_success", False))

    # Stage 2 geometry checks are only credited if code ran.
    geom = {}
    for c in GEOMETRY_CHECKS:
        raw = _clamp_unit(c, record.get(c, 0.0))
        geom[c] = raw if code_ok else 0.0
    geom_valid = 1.0 if (code_ok and all(geom[c] >= 1.0 for c in GEOMETRY_CHECKS)) \
        else 0.0

    # Stage 3 alignment sub-criteria only credited if geometry is valid.
    subs = {}
    for s in SUB_CRITERIA:
        raw = _clamp_unit(s, record.get(s, 0.0))
        subs[s] = raw if geom_valid >= 1.0 else 0.0

    functionality = (subs["functional"] + subs["robust"]) / 2.0
    manufacturability = (subs["well_toleranced"] + subs["manufacturable"]) / 2.0
    assemblability = (subs["assembly_ready"] + subs["connectable"]) / 2.0
    final = (functionality + manufacturability + assemblability) / 3.0

    out = {
        "sandbox_success": 1.0 if code_ok else 0.0,
        "geom_valid": geom_valid,
        "functionality": functionality,
        "manufacturability": manufacturability,
        "assemblability": assemblability,
        "final_score": final,
    }
    out.update(geom)
    out.update(subs)
    return out


def muse_scorecard(records, *, as_percent=True):
    """Aggregate a set of design records into a MUSE Table-2 style scorecard.

    records : iterable of per-design records accepted by ``evaluate_design``.
    as_percent : if True (default) report means in percent (0-100) as in the
        paper's tables; otherwise as fractions (0-1).

    Returns a dict of column means: sandbox_success, the four geometry checks,
    geom_valid, functionality, manufacturability, assemblability, final_score,
    plus n.
    """
    rows = [evaluate_design(r) for r in records]
    n = len(rows)
    if n == 0:
        raise ValueError("no records")
    scale = 100.0 if as_percent else 1.0
    columns = (("sandbox_success",) + GEOMETRY_CHECKS + ("geom_valid",)
               + ("functionality", "manufacturability", "assemblability",
                  "final_score"))
    out = {"n": n}
    for col in columns:
        out[col] = scale * sum(r[col] for r in rows) / n
    out["rows"] = tuple(rows)
    return out


def cascade_dropoff(scorecard):
    """Stage-to-stage retention drops from an aggregated scorecard (RQ1).

    Returns the absolute drop between successive funnel stages
    (code -> geometry -> alignment), matching the paper's "failure cascade"
    analysis. Values are in the scorecard's own units (percent or fraction).
    """
    code = scorecard["sandbox_success"]
    geom = scorecard["geom_valid"]
    align = scorecard["final_score"]
    return {
        "code_to_geometry": code - geom,
        "geometry_to_alignment": geom - align,
        "code_to_alignment": code - align,
    }
