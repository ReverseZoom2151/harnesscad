"""MUSE manufacturability scorer (design-intent alignment, Manufacturability pillar).

Deterministic re-encoding of the MUSE benchmark's Manufacturability pillar
(Dong et al., "MUSE: Benchmarking Manufacturable, Functional, and Assemblable
Text-to-CAD Generation"). The pillar has two binary sub-criteria (Table 8):

  * Manufacturable  -- geometry is compatible with the declared material and
    manufacturing process (engineering knowledge Tables 5 and 6).
  * Well-toleranced -- seams, clearances and wall thicknesses are consistent
    with the process precision range (Table 6).

This is a *design-intent* scorer keyed off the paper's engineering knowledge
tables. It is distinct from ``verifiers/dfm.py`` (rule-firing DFM verifier) and
from ``bench/engdesign_dfm_scoring.py`` (VLM answer scoring): here we turn the
MUSE material/process tables into concrete numeric feasibility checks over an
injected structured design. Generation is external; the design is injected.

No wall clock, no randomness.
"""

from __future__ import annotations

# --- Engineering Knowledge Table 5: Material Selection -----------------------
# process names are normalised: the paper writes "CNC" as shorthand for
# "CNC Milling".
MATERIALS = {
    "Timber": {"processes": ("CNC Milling", "Laser Cutting"), "brittle": False,
               "flexible": False},
    "ABS": {"processes": ("3D Printing",), "brittle": False, "flexible": False},
    "PLA": {"processes": ("3D Printing",), "brittle": False, "flexible": False},
    "TPU": {"processes": ("3D Printing",), "brittle": False, "flexible": True},
    "Acrylic": {"processes": ("CNC Milling", "Laser Cutting"), "brittle": True,
                "flexible": False},
    "Resin": {"processes": ("3D Printing", "Silicone Casting"), "brittle": True,
              "flexible": False},
    "Sheet Metal": {"processes": ("CNC Milling", "Laser Cutting"),
                    "brittle": False, "flexible": False},
    "Aluminum": {"processes": ("CNC Milling",), "brittle": False,
                 "flexible": False},
    "Steel": {"processes": ("CNC Milling",), "brittle": False, "flexible": False},
}

# --- Engineering Knowledge Table 6: Manufacturing Methods ---------------------
# tol_min / tol_max in mm (the paper's precision column, e.g. +/-0.05-0.10 mm).
# min_wall is a conventional DFM minimum wall/feature size for the process.
PROCESSES = {
    "CNC Milling": {
        "tol_min": 0.05, "tol_max": 0.10, "min_wall": 1.0, "max_edge": 2000.0,
        "build_volume": None, "max_sheet_thickness": None, "subtractive": True},
    "Laser Cutting": {
        "tol_min": 0.05, "tol_max": 0.20, "min_wall": 0.5, "max_edge": 2000.0,
        "build_volume": None, "max_sheet_thickness": 5.0, "subtractive": True},
    "3D Printing": {
        "tol_min": 0.10, "tol_max": 0.50, "min_wall": 0.8, "max_edge": None,
        "build_volume": (300.0, 300.0, 300.0), "max_sheet_thickness": None,
        "subtractive": False},
    "Injection Molding": {
        "tol_min": 0.01, "tol_max": 0.05, "min_wall": 0.5, "max_edge": None,
        "build_volume": None, "max_sheet_thickness": None, "subtractive": False},
    "Silicone Casting": {
        "tol_min": 0.10, "tol_max": 0.30, "min_wall": 1.0, "max_edge": None,
        "build_volume": None, "max_sheet_thickness": None, "subtractive": False},
    "Modular Assembly": {
        "tol_min": 0.10, "tol_max": 0.50, "min_wall": None, "max_edge": None,
        "build_volume": None, "max_sheet_thickness": None, "subtractive": False},
}


def _lookup_material(name):
    if name not in MATERIALS:
        raise ValueError("unknown material: %r" % (name,))
    return MATERIALS[name]


def _lookup_process(name):
    if name not in PROCESSES:
        raise ValueError("unknown process: %r" % (name,))
    return PROCESSES[name]


def process_tolerance(process):
    """Return (tol_min, tol_max) in mm for a manufacturing process."""
    p = _lookup_process(process)
    return (p["tol_min"], p["tol_max"])


def material_process_compatible(material, process):
    """True if the material can be produced by the process (Table 5)."""
    return process in _lookup_material(material)["processes"]


def _wall_violations(components, min_wall):
    """Zero/too-thin wall violations shared by both sub-criteria."""
    viol = []
    for c in components:
        wt = c.get("wall_thickness")
        if wt is None:
            continue
        if wt <= 0.0:
            viol.append("zero_thickness:%s" % c.get("name", "?"))
        elif min_wall is not None and wt < min_wall:
            viol.append("thin_wall:%s" % c.get("name", "?"))
    return viol


def score_manufacturable(design):
    """Binary Manufacturable sub-criterion for one injected design.

    design keys:
      material, process : names from the knowledge tables.
      components        : iterable of dicts with optional keys wall_thickness,
                          thickness (stock/sheet thickness), bbox (x,y,z mm),
                          load_bearing (bool), cantilever (bool),
                          internal_dead_cavity (bool).
    Returns {"manufacturable": 0/1, "violations": (...)}.
    """
    material = design["material"]
    process = design["process"]
    mat = _lookup_material(material)
    proc = _lookup_process(process)
    components = list(design.get("components", ()))
    violations = []

    if not material_process_compatible(material, process):
        violations.append("incompatible_material_process")

    violations.extend(_wall_violations(components, proc["min_wall"]))

    for c in components:
        name = c.get("name", "?")
        thickness = c.get("thickness")
        bbox = c.get("bbox")
        if (proc["max_sheet_thickness"] is not None and thickness is not None
                and thickness > proc["max_sheet_thickness"]):
            violations.append("exceeds_sheet_thickness:%s" % name)
        if proc["build_volume"] is not None and bbox is not None:
            if any(d > lim for d, lim in zip(bbox, proc["build_volume"])):
                violations.append("exceeds_build_volume:%s" % name)
        if proc["max_edge"] is not None and bbox is not None:
            if max(bbox) > proc["max_edge"]:
                violations.append("exceeds_max_edge:%s" % name)
        if proc["subtractive"] and c.get("internal_dead_cavity"):
            violations.append("inaccessible_cavity:%s" % name)
        if (mat["brittle"] and c.get("cantilever") and c.get("load_bearing")):
            violations.append("fragile_brittle_cantilever:%s" % name)

    return {"manufacturable": 0 if violations else 1,
            "violations": tuple(violations)}


def score_well_toleranced(design, *, max_gap_factor=10.0):
    """Binary Well-toleranced sub-criterion for one injected design.

    Checks that every declared assembly clearance (mm) lies inside the
    admissible band [tol_min, tol_max * max_gap_factor] for the process, and
    that no wall is below the process minimum. A gap below tol_min means the
    seam has vanished (illegal fusion); a gap above the band is an exaggerated
    clearance that would prevent real fitting.

    design extra key:
      clearances : iterable of numbers (gap widths in mm) or (label, gap) pairs.
    Returns {"well_toleranced": 0/1, "violations": (...)}.
    """
    if max_gap_factor <= 0:
        raise ValueError("max_gap_factor must be positive")
    process = design["process"]
    proc = _lookup_process(process)
    tol_min, tol_max = proc["tol_min"], proc["tol_max"]
    max_gap = tol_max * max_gap_factor
    violations = []

    for entry in design.get("clearances", ()):
        if isinstance(entry, (tuple, list)):
            label, gap = entry
        else:
            label, gap = "?", entry
        if gap < tol_min:
            violations.append("illegal_fusion:%s" % label)
        elif gap > max_gap:
            violations.append("exaggerated_gap:%s" % label)

    violations.extend(_wall_violations(list(design.get("components", ())),
                                       proc["min_wall"]))

    return {"well_toleranced": 0 if violations else 1,
            "violations": tuple(violations)}


def muse_manufacturability(design, *, max_gap_factor=10.0):
    """Full Manufacturability pillar score for one injected design.

    Returns the two binary sub-criteria, their average (the pillar score used
    in MUSE Table 3), and the merged violation list.
    """
    man = score_manufacturable(design)
    tol = score_well_toleranced(design, max_gap_factor=max_gap_factor)
    average = (man["manufacturable"] + tol["well_toleranced"]) / 2.0
    return {
        "manufacturable": man["manufacturable"],
        "well_toleranced": tol["well_toleranced"],
        "average": average,
        "violations": tuple(man["violations"]) + tuple(tol["violations"]),
    }
