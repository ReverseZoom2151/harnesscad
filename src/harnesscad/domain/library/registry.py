"""The CATALOGUE surface -- retrieval-not-modelling, plus the standards knowledge base.

Two knowledge bases sit under ``domain/``: ``library/`` (parametric standard
parts, part families, induced concepts, part-name semantics) and ``standards/``
(a versioned rule registry, thread tables, heat-set insert bores, ACI colours).
Both were reachable from nothing. This module is the one surface that dispatches
into both, because a caller asking "give me a flange" and a caller asking "what
is the tapping drill for an M4 heat-set insert" are asking the *same kind* of
question: look it up, do not model it.

    catalogue().find("mounting")        -> the parts that answer that function
    instantiate("flange", diameter=90)  -> CISP ops, range-validated
    standards().active_rules(...)       -> the rules in force for a material
    thread("M6")                        -> the standard's own numbers

Every part in the catalogue is **execution-verified**: it is built on a fresh
stub-backed :class:`~harnesscad.core.loop.HarnessSession` and admitted only if
it verifies. A part that no longer builds is not in the catalogue -- the
monotonic-trust invariant, not a promise in a docstring.

Adapters only: the library and standards modules are never modified.
Deterministic, stdlib-only, no network.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "CatalogueError",
    "session_factory",
    "catalogue",
    "part_names",
    "find_part",
    "instantiate",
    "family",
    "gears",
    "standards",
    "ingest_rules",
    "rule_conflicts",
    "thread",
    "heatsert",
    "aci",
    "fits",
    "designation",
    "material",
    "servo",
    "gridfinity",
    "defaults",
    "carbon",
    "carbon_intensity",
    "provenance",
    "normalize_names",
    "name_semantics",
    "induce_concepts",
    "retrieve",
    "resolve_relative",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

_LIB = "harnesscad.domain.library."
_STD = "harnesscad.domain.standards."


class CatalogueError(ValueError):
    """Base class for every catalogue-surface failure."""


# --------------------------------------------------------------------------- #
# Parts catalogue
# --------------------------------------------------------------------------- #
def session_factory() -> Callable[[], Any]:
    """A fresh, dependency-free session -- the execution gate every card passes."""
    from harnesscad.core.loop import HarnessSession
    from harnesscad.io.backends.stub import StubBackend

    return lambda: HarnessSession(StubBackend())


_CATALOG: Optional[Any] = None


def catalogue(refresh: bool = False):
    """The default :class:`PartCatalog`, every card execution-verified. Cached."""
    global _CATALOG
    if _CATALOG is not None and not refresh:
        return _CATALOG
    from harnesscad.domain.library.catalog import build_default_catalog

    _CATALOG = build_default_catalog(session_factory())
    return _CATALOG


def part_names() -> List[str]:
    return list(catalogue().names())


def find_part(query: str, k: int = 5) -> List[dict]:
    """Retrieve by FUNCTION ('mounting', 'bearing') or free text. Ranked, ordered."""
    return [{"name": c.name,
             "tags": list(c.function_tags),
             "description": c.description,
             "notes": c.notes,
             "verified": bool(c.verified),
             "params": {k2: v.get("default") for k2, v in c.param_schema.items()}}
            for c in catalogue().find(query, k=k)]


def instantiate(name: str, **params) -> List[Any]:
    """A catalogue part -> CISP ops, with every parameter range-validated first."""
    return catalogue().instantiate(name, **params)


def family(part: str, axes: Dict[str, Sequence[Any]], unit: str = "mm"):
    """A validated parameter SWEEP of one catalogue part -> a family manifest.

    ``axes`` maps a parameter name to the values it takes ({"thickness": [6, 8,
    10]}). Every point in the grid is built on a fresh session and kept only if
    it verifies -- so a family manifest is a list of variants that provably
    regenerate, with provenance for each.
    """
    from harnesscad.domain.library.family import (
        FamilySpec, ParameterAxis, Validation, generate_family,
    )

    cat = catalogue()
    if part not in cat.names():
        raise CatalogueError("no part %r in the catalogue (have: %s)"
                             % (part, ", ".join(cat.names())))
    if not axes:
        raise CatalogueError("a family needs at least one parameter axis")
    spec = FamilySpec(
        family=part,
        axes=tuple(ParameterAxis(name, tuple(axes[name]), unit)
                   for name in sorted(axes)),
    )
    make = session_factory()

    def builder(context: Dict[str, Any]):
        params = {k: v for k, v in context.items()
                  if k not in ("family", "index")}
        return cat.instantiate(part, **params)

    def validator(ops, context: Dict[str, Any]) -> Validation:
        session = make()
        result = session.apply_ops(list(ops))
        ok = bool(result.ok)
        return Validation(
            accepted=ok,
            checks={"applies": ok},
            message="" if ok else "; ".join(d.message for d in result.diagnostics),
        )

    return generate_family(spec, builder, validator)


# --------------------------------------------------------------------------- #
# Gear trains (ISO-preferred modules, involute geometry, meshing)
# --------------------------------------------------------------------------- #
def gears(module: float, teeth: int, mate_teeth: Optional[int] = None,
          helix_angle: float = 0.0, pressure_angle: float = 20.0,
          snap: bool = False) -> dict:
    """Involute-gear geometry from module + tooth count, optional mesh check.

    With ``snap`` the raw module is first snapped to the nearest ISO-preferred
    value (CAD-GPT's series). When ``mate_teeth`` is given, the pair is checked
    for meshing and the gear ratio + centre distance are returned -- gears with a
    module or helix mismatch report WHY they cannot mesh rather than pretend to.
    """
    from harnesscad.domain.library.gear_train import (
        gear_geometry, mesh_pair, snap_module,
    )

    m = snap_module(float(module)) if snap else float(module)
    g = gear_geometry(m, int(teeth), helix_angle=float(helix_angle),
                      pressure_angle=float(pressure_angle))
    out = {
        "module": g.module, "teeth": g.teeth, "helix_angle": g.helix_angle,
        "pitch_diameter": g.pitch_diameter, "outside_diameter": g.outside_diameter,
        "root_diameter": g.root_diameter, "base_diameter": g.base_diameter,
        "snapped_module": m if snap else None,
    }
    if mate_teeth is not None:
        mate = gear_geometry(m, int(mate_teeth), helix_angle=float(helix_angle),
                             pressure_angle=float(pressure_angle))
        mesh = mesh_pair(g, mate)
        out["mesh"] = {"meshes": bool(mesh.meshes), "gear_ratio": mesh.gear_ratio,
                       "center_distance": mesh.center_distance,
                       "reasons": list(mesh.reasons)}
    return out


# --------------------------------------------------------------------------- #
# Standards knowledge base
# --------------------------------------------------------------------------- #
def standards():
    """A fresh :class:`StandardsRegistry` -- register rule packs into it."""
    from harnesscad.domain.standards.registry import StandardsRegistry

    return StandardsRegistry()


def ingest_rules(text: str, standard: str, version: str):
    """Clause text -> typed :class:`Rule` records (deterministic heuristic, no LLM)."""
    from harnesscad.domain.standards.ingest import ingest_heuristic

    return ingest_heuristic(text, standard, version)


def rule_conflicts(rules: Sequence[Any]) -> List[dict]:
    """Which of these active rules contradict each other, and how."""
    from harnesscad.domain.standards.conflict import detect_conflicts

    return [c.to_dict() for c in detect_conflicts(list(rules))]


def thread(name: str) -> dict:
    """Standard screw-thread dimensions. The TABLE, never a guess.

    The database is keyed by the full designation ('M6x1'); a bare diameter
    ('M6') resolves to the first designation for it in sorted order -- stable,
    but say the pitch you mean. An unknown thread raises: no interpolation.
    """
    from harnesscad.domain.standards.thread_database import (
        hex_height, hex_radius, thread_lookup, thread_names,
    )

    try:
        t = thread_lookup(name)
        resolved = name
    except KeyError:
        matches = sorted(thread_names(prefix=name + "x"))
        if not matches:
            raise
        resolved = matches[0]
        t = thread_lookup(resolved)
    return {
        "name": resolved,
        "requested": name,
        "radius": t.radius,
        "pitch": t.pitch,
        "hex_flat2flat": t.hex_flat2flat,
        "hex_radius": hex_radius(t),
        "hex_height": hex_height(t),
        "units": t.units,
    }


def heatsert(designation: str, wall_thickness: Optional[float] = None) -> dict:
    """Heat-set insert bore schedule for a screw designation.

    The bore is a stack of sections; the schedule gives its depth, its volume
    and -- crucially for DFM -- whether it fits the wall it is going into.
    """
    from harnesscad.domain.standards.heatsert_bores import (
        bore_depth, bore_volume, fits_in_wall, heatsert_bore, insert_dims,
    )

    dims = insert_dims(designation)
    sections = heatsert_bore(designation)
    out = {
        "designation": designation,
        "bore_depth": bore_depth(sections),
        "bore_volume": bore_volume(sections),
        "min_boss_diameter": dims.min_boss_diameter,
        "bolt_clearance_diameter": dims.bolt_clearance_diameter,
        "sections": len(sections),
    }
    if wall_thickness is not None:
        out["fits_in_wall"] = bool(fits_in_wall(designation, float(wall_thickness)))
    return out


def aci(value: Any) -> dict:
    """AutoCAD Color Index: name <-> index <-> RGB, and the nearest legal index."""
    from harnesscad.domain.standards.aci_color import (
        aci_to_name, aci_to_rgb, is_special, name_to_aci, nearest_aci, validate_aci,
    )

    if isinstance(value, (list, tuple)) and len(value) == 3:
        index = nearest_aci(tuple(int(c) for c in value))
    elif isinstance(value, int):
        validate_aci(value)
        index = value
    else:
        index = name_to_aci(str(value))
    return {"index": index, "name": aci_to_name(index),
            "rgb": aci_to_rgb(index), "special": bool(is_special(index))}


def _zone_dict(z: Any) -> dict:
    return {"designation": z.designation, "nominal_mm": z.nominal_mm,
            "hole": bool(z.hole), "grade": z.grade,
            "lower_mm": z.lower_mm, "upper_mm": z.upper_mm,
            "width_mm": z.width_mm, "min_size_mm": z.min_size_mm,
            "max_size_mm": z.max_size_mm, "size_range": z.size_range}


def fits(zone: str, size_mm: float, processes: bool = False) -> dict:
    """ISO 286 tolerance zones and fits at a nominal size. The TABLE, not a guess.

    A paired designation ('H7/g6') resolves the whole fit -- both zones, the
    signed worst-case clearances, and whether it is a clearance / transition /
    interference fit. A single zone ('H7', 'g6') resolves that zone alone.
    With ``processes``, the zone width is checked against the processes that can
    reasonably hold it -- a tolerance no process can hold is a drawing, not a part.
    """
    from harnesscad.domain.standards.iso286_fits import (
        fit, processes_that_can_hold, zone_limits,
    )

    name = str(zone)
    size = float(size_mm)
    if "/" in name:
        f = fit(name, size)
        out = {"designation": f.designation, "nominal_mm": f.nominal_mm,
               "kind": f.kind, "min_clearance_mm": f.min_clearance_mm,
               "max_clearance_mm": f.max_clearance_mm,
               "hole": _zone_dict(f.hole), "shaft": _zone_dict(f.shaft)}
        width = f.hole.width_mm
    else:
        out = _zone_dict(zone_limits(size, name))
        width = out["width_mm"]
    if processes:
        out["processes_that_can_hold"] = list(processes_that_can_hold(width))
    return out


def _plain(value: Any) -> Any:
    """NamedTuple -> dict, recursively; anything else untouched."""
    if hasattr(value, "_asdict"):
        return {k: _plain(v) for k, v in value._asdict().items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def designation(name: str) -> dict:
    """A standard part designation ('M5', '608', '2020', 'NEMA17') -> its dimensions.

    One front door over the whole stock catalogue: cap screws, hex bolts and
    nuts, washers, dowel pins, bearings, extrusion profiles, NEMA motor frames
    and clearance holes. The KIND is inferred from the designation; an unknown
    designation raises rather than resolving to something plausible.
    """
    from harnesscad.domain.standards.part_catalog import resolve

    kind, record = resolve(str(name))
    return {"kind": kind, "requested": str(name), "record": _plain(record)}


def material(name: str) -> dict:
    """A material's cited mechanical properties. Missing values stay None, never 0.

    Every number carries its citation (``record['citations']``): a property with
    no source is absent, not invented. An unknown material raises.
    """
    from harnesscad.domain.standards.materials_db import material as material_record

    return dict(material_record(str(name))._asdict())


def servo(name: str) -> dict:
    """A named servo's envelope: body, lugs, mount-hole layout and drive shaft.

    The hole positions and shaft centre are returned already resolved to (x, y)
    coordinates -- the numbers a bracket is cut from, not a spacing to re-derive.
    """
    from harnesscad.domain.standards.servo_database import (
        servo_lookup, servo_mount_hole_positions, servo_shaft_xy,
    )

    k = servo_lookup(str(name))
    out = dict(k._asdict())
    out["body"] = list(k.body)
    out["mount"] = list(k.mount)
    out["hole"] = list(k.hole)
    out["mount_hole_positions"] = [list(p) for p in servo_mount_hole_positions(k)]
    out["shaft_xy"] = list(servo_shaft_xy(k))
    return out


def gridfinity(nx: int, ny: int, nz: int = 1) -> dict:
    """The Gridfinity envelope for an nx * ny * nz bin. The standard's own numbers.

    Base footprint is the grid it occupies; body footprint is the (smaller) bin
    itself -- the difference is the clearance that makes bins droppable. Grid
    centres are relative to the footprint centre.
    """
    from harnesscad.domain.standards.servo_database import (
        GRIDFINITY, gridfinity_base_footprint, gridfinity_body_footprint,
        gridfinity_body_height, gridfinity_grid_centers,
        gridfinity_hole_offset_from_center,
    )

    return {
        "nx": int(nx), "ny": int(ny), "nz": int(nz),
        "base_footprint": list(gridfinity_base_footprint(int(nx), int(ny))),
        "body_footprint": list(gridfinity_body_footprint(int(nx), int(ny))),
        "body_height": gridfinity_body_height(int(nz)),
        "grid_centers": [list(c) for c in gridfinity_grid_centers(int(nx), int(ny))],
        "hole_offset_from_center": gridfinity_hole_offset_from_center(GRIDFINITY),
        "envelope": dict(GRIDFINITY._asdict()),
    }


def defaults(part_type: Optional[str] = None,
             size: Optional[str] = None,
             smallest_local_extent_mm: Optional[float] = None) -> dict:
    """The house CAD defaults: wall, cosmetic fillet, feature order, origin.

    What to reach for when the prompt did not say. ``part_type`` narrows the
    origin convention to one; ``size`` ('M5') adds that screw's normal clearance
    hole; ``smallest_local_extent_mm`` scales the fillet to the feature it rounds.
    """
    from harnesscad.domain.standards.cad_defaults import (
        BOOLEAN_OVERSHOOT_MM, COSMETIC_FILLET_RANGE_MM, ENCLOSURE_WALL_RANGE_MM,
        ORIGIN_CONVENTIONS, clearance_hole_diameter, clearance_hole_radius,
        default_fillet_radius, default_wall_thickness, feature_order,
        origin_convention,
    )

    out: Dict[str, Any] = {
        "wall_thickness": default_wall_thickness(),
        "wall_range": list(ENCLOSURE_WALL_RANGE_MM),
        "fillet_radius": default_fillet_radius(
            None if smallest_local_extent_mm is None
            else float(smallest_local_extent_mm)),
        "fillet_range": list(COSMETIC_FILLET_RANGE_MM),
        "boolean_overshoot": BOOLEAN_OVERSHOOT_MM,
        "feature_order": list(feature_order()),
        "origin_conventions": dict(ORIGIN_CONVENTIONS),
    }
    if part_type is not None:
        out["origin_convention"] = origin_convention(str(part_type))
    if size is not None:
        out["clearance_hole_diameter"] = clearance_hole_diameter(str(size))
        out["clearance_hole_radius"] = clearance_hole_radius(str(size))
    return out


# --------------------------------------------------------------------------- #
# Standards accounting: embodied carbon + cited provenance
# --------------------------------------------------------------------------- #
def carbon(uses: Sequence[Dict[str, Any]], top_n: Optional[int] = None,
           table: Optional[Dict[str, float]] = None) -> dict:
    """Embodied-carbon (CO2e) accounting over a bill of materials.

    ``uses`` is a sequence of ``{"material", "mass_kg"}`` mappings. Unknown
    materials raise rather than being silently tallied as zero. Optionally
    returns the top-``top_n`` worst offenders. Routes through the standards
    accounting front door (:mod:`harnesscad.domain.standards.accounting`).
    """
    from harnesscad.domain.standards import accounting

    out = {"total_co2e_kg": accounting.carbon_total(uses, table=table)}
    if top_n is not None:
        out["top"] = accounting.carbon_top(uses, n=int(top_n), table=table)
    return out


def carbon_intensity(name: str) -> dict:
    """One material's cited CO2e intensity, straight out of ICE v3 (Bath).

    A different question from :func:`carbon`: not "what does this BOM cost the
    atmosphere" but "what is the number, and who says so". Returns the intensity
    with its recycled content, recyclability and source citation. The dataset is
    loaded lazily on first use; an unknown material raises rather than being
    tallied as zero.
    """
    from harnesscad.domain.standards import embodied_carbon_ice as ice

    if not ice.is_available() and not ice.load_ice():
        raise CatalogueError("the ICE v3 dataset is not available (expected at %s)"
                             % ice.default_ice_path())
    record = ice.lookup_material(str(name))
    if record is None:
        raise CatalogueError("no ICE material %r (have: %s)"
                             % (name, ", ".join(ice.manifest_keys())))
    combined = ice.combined_carbon_intensity(str(name))
    return {
        "requested": str(name),
        "key": record.key,
        "label": record.label,
        "category": record.category,
        "co2e_per_kg": record.co2e_per_kg,
        "combined_co2e_per_kg": None if combined is None else combined[0],
        "combined_source": None if combined is None else combined[1],
        "recycled_content_pct": record.recycled_content_pct,
        "recyclability_pct": record.recyclability_pct,
        "source": record.source,
        "source_url": record.source_url,
        "source_version": record.source_version,
    }


def provenance(spec: Dict[str, Any]) -> dict:
    """Roll a design spec's standards data into a review-ready provenance bundle.

    Returns the cited records, a deterministic digest (identical specs -> identical
    digest), the citation gate, and any references the databases could not vouch
    for. Front door: :mod:`harnesscad.domain.standards.accounting`.
    """
    from harnesscad.domain.standards import accounting

    return accounting.provenance(spec)


# --------------------------------------------------------------------------- #
# Part-name semantics / concepts / retrieval
# --------------------------------------------------------------------------- #
def normalize_names(names: Sequence[str]) -> dict:
    """Name hygiene: strip the CAD tool's default names, dedupe, key."""
    from harnesscad.domain.library.name_normalizer import (
        dedupe_names, is_default_name, name_key, normalize_name,
    )

    kept = [n for n in names if not is_default_name(n)]
    return {
        "user_names": kept,
        "default_names": [n for n in names if is_default_name(n)],
        "normalized": [normalize_name(n) for n in kept],
        "keys": sorted({name_key(n) for n in kept}),
        "deduped": list(dedupe_names(kept)),
    }


def name_semantics(assemblies: Sequence[Sequence[str]],
                   document_ids: Optional[Sequence[str]] = None):
    """Training-free PPMI over assembly part names -- which names mean the same thing.

    ``assemblies`` is one list of body names per assembly document. Only USER
    names count: the tool's default names ('Part1', 'Boss-Extrude1') carry no
    semantics and :mod:`name_normalizer` drops them before any counting.

    Returns the fitted model; ``model.pair_score('bolt', 'screw')`` and
    ``model.rank_candidates(...)`` are the useful edges. No trained embedding,
    no network -- co-occurrence statistics over the corpus you hand it.
    """
    from harnesscad.domain.library.partname_ppmi import build_ppmi

    ids = ([str(d) for d in document_ids] if document_ids is not None
           else ["doc%d" % i for i in range(len(assemblies))])
    corpus = {doc_id: {"body_names": [str(n) for n in names]}
              for doc_id, names in zip(ids, assemblies)}
    return build_ppmi(corpus, ids)


def induce_concepts(corpus: Sequence[Any], max_concepts: int = 5):
    """Search-based concept induction over a sketch corpus -> a ConceptLibrary.

    The compression-gain criterion: a concept earns its place only if abstracting
    its occurrences makes the corpus SHORTER.
    """
    from harnesscad.domain.library.concept_induction import induce_library

    return induce_library(list(corpus), max_concepts=int(max_concepts))


def retrieve(summary_distances: Dict[str, float],
             trigger_distances: Dict[str, float],
             links: Optional[Dict[str, Sequence[str]]] = None):
    """Dual-channel (WHAT / WHEN) rank fusion, with graph-aware link expansion.

    Two retrieval channels score the same corpus differently; the fusion is by
    RANK, not by averaging their incomparable distances.
    """
    from harnesscad.domain.library.dual_channel_fusion import (
        dual_channel_retrieve, expand_by_links,
    )

    fused = dual_channel_retrieve(dict(summary_distances), dict(trigger_distances))
    if links:
        return expand_by_links(fused, {k: list(v) for k, v in links.items()})
    return fused


def resolve_relative(current: float, token: str,
                     minimum: Optional[float] = None,
                     maximum: Optional[float] = None):
    """A RELATIVE property edit -> an absolute number, clamped to the bounds.

    ``token`` is the edit as written: ``"+10%"`` / ``"-5%"`` (percent), ``"*1.5"``
    (scale), ``"+5"`` / ``"-3"`` (delta) or a bare ``"50"`` (absolute). A token
    that is not one of those returns ``None`` -- the resolver does not guess what
    "a bit bigger" means.
    """
    from harnesscad.domain.library.relative_value import resolve

    return resolve(current, token, minimum=minimum, maximum=maximum)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index() -> Dict[str, Any]:
    out = {}
    for pkg in ("library", "standards"):
        for e in capability_registry.find(package=pkg):
            out[e.dotted] = e
    return out


def _available(dotted: str) -> bool:
    return dotted in _index()


_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("parts", "catalogue", _LIB + "catalog",
     "the execution-verified parts catalogue (build_default_catalog)"),
    ("parts", "instantiate", _LIB + "parts",
     "a catalogue part -> range-validated CISP ops"),
    ("parts", "family", _LIB + "family",
     "a validated parameter sweep of one part -> a family manifest"),
    ("parts", "gears", _LIB + "gear_train",
     "ISO-preferred gear modules, involute geometry, and mesh/ratio checks"),
    ("names", "normalize_names", _LIB + "name_normalizer",
     "strip the CAD tool's default names, dedupe, key"),
    ("names", "name_semantics", _LIB + "partname_ppmi",
     "training-free PPMI over assembly part names"),
    ("names", "resolve_relative", _LIB + "relative_value",
     "a relative property edit ('+10%', '*1.5', '+5') -> an absolute number"),
    ("concepts", "induce_concepts", _LIB + "concept_induction",
     "search-based concept induction over a sketch corpus (compression gain)"),
    ("concepts", "induce_concepts", _LIB + "concept_library",
     "the induced ConceptLibrary itself (dedup, cycles, flatten, usage)"),
    ("retrieval", "retrieve", _LIB + "dual_channel_fusion",
     "dual-channel WHAT/WHEN rank fusion with link expansion"),
    ("standards", "standards", _STD + "registry",
     "the versioned standards knowledge base (rule packs, diffs, active rules)"),
    ("standards", "ingest_rules", _STD + "ingest",
     "clause text -> typed Rule records"),
    ("standards", "rule_conflicts", _STD + "conflict",
     "which active rules contradict each other"),
    ("standards", "thread", _STD + "thread_database",
     "standard screw-thread / fastener dimensions"),
    ("standards", "heatsert", _STD + "heatsert_bores",
     "heat-set insert bore schedule, depth/volume, wall fit"),
    ("standards", "aci", _STD + "aci_color",
     "AutoCAD Color Index: name <-> index <-> RGB, nearest legal index"),
    ("standards", "fits", _STD + "iso286_fits",
     "ISO 286 tolerance zones and fits, ISO 2768 general tolerances, "
     "process capability"),
    ("standards", "designation", _STD + "part_catalog",
     "stock designations (screws, nuts, washers, dowels, bearings, extrusion, "
     "NEMA) -> dimensions"),
    ("standards", "material", _STD + "materials_db",
     "cited mechanical properties per material (E, nu, rho, yield, ultimate)"),
    ("standards", "servo", _STD + "servo_database",
     "named servo envelopes: body, lugs, mount-hole layout, drive shaft"),
    ("standards", "gridfinity", _STD + "servo_database",
     "the Gridfinity envelope: base/body footprint, height, grid centres"),
    ("standards", "defaults", _STD + "cad_defaults",
     "the house CAD defaults: wall, cosmetic fillet, feature order, origin"),
    ("standards", "carbon", _STD + "accounting",
     "the standards-accounting front door (embodied carbon + cited provenance)"),
    ("standards", "carbon", _STD + "embodied_carbon",
     "embodied-carbon (CO2e) accounting over a bill of materials; worst offenders"),
    ("standards", "carbon_intensity", _STD + "embodied_carbon_ice",
     "one material's cited CO2e intensity from ICE v3 (Bath), with its source"),
    ("standards", "provenance", _STD + "evidence_bundle",
     "the cited-provenance bundle over a design spec (records, digest, gate)"),
)


def routed_modules() -> Tuple[str, ...]:
    return tuple(sorted({m for _g, _n, m, _d in _ROUTES if _available(m)}))


def discover() -> List[dict]:
    return [{"group": g, "route": n, "module": m, "doc": d,
             "present": _available(m)}
            for (g, n, m, d) in _ROUTES]


def unadapted() -> List[Tuple[str, str]]:
    routed = set(routed_modules())
    out = []
    for dotted in sorted(_index()):
        if dotted in routed or dotted.endswith(".registry"):
            continue
        out.append((dotted, "no route yet"))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every catalogue route")
    parser.add_argument("--parts", action="store_true",
                        help="list the execution-verified parts")
    parser.add_argument("--find", default=None,
                        help="retrieve parts by function tag or free query")
    parser.add_argument("--part", default=None,
                        help="instantiate this part (with --param K=V) and print its ops")
    parser.add_argument("--param", action="append", default=[], metavar="K=V",
                        help="a part parameter (repeatable); V is parsed as JSON")
    parser.add_argument("--apply", action="store_true",
                        help="apply the instantiated part's ops to a stub session")
    parser.add_argument("--thread", default=None,
                        help="look up a standard thread (e.g. M6)")
    parser.add_argument("--heatsert", default=None,
                        help="look up a heat-set insert bore schedule (e.g. M4)")
    parser.add_argument("--wall", type=float, default=None,
                        help="wall thickness for the --heatsert fit check")
    parser.add_argument("--aci", default=None,
                        help="an ACI colour name or index")
    parser.add_argument("--unadapted", action="store_true",
                        help="list library/standards modules with no route")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def _params(pairs: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise CatalogueError("--param expects K=V, got %r" % pair)
        key, _, raw = pair.partition("=")
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "unadapted", False):
        for dotted, reason in unadapted():
            print("%s\n    %s" % (dotted, reason))
        return 0

    if getattr(args, "thread", None):
        print(json.dumps(thread(args.thread), indent=2, sort_keys=True, default=repr))
        return 0

    if getattr(args, "heatsert", None):
        print(json.dumps(heatsert(args.heatsert, getattr(args, "wall", None)),
                         indent=2, sort_keys=True, default=repr))
        return 0

    if getattr(args, "aci", None):
        value = args.aci
        try:
            value = int(value)
        except ValueError:
            pass
        print(json.dumps(aci(value), indent=2, sort_keys=True, default=repr))
        return 0

    if getattr(args, "find", None):
        hits = find_part(args.find)
        if getattr(args, "json", False):
            print(json.dumps(hits, indent=2, sort_keys=True))
        else:
            for h in hits:
                print("%-18s %-28s %s" % (h["name"], ",".join(h["tags"]),
                                          h["description"]))
        return 0

    if getattr(args, "part", None):
        ops = instantiate(args.part, **_params(getattr(args, "param", []) or []))
        if getattr(args, "apply", False):
            session = session_factory()()
            result = session.apply_ops(list(ops))
            print("ok:      %s" % result.ok)
            print("applied: %d" % result.applied)
            print("summary: %s" % json.dumps(session.summary(), sort_keys=True))
            return 0 if result.ok else 1
        print(json.dumps([op.to_dict() for op in ops], indent=2, sort_keys=True))
        return 0

    if getattr(args, "parts", False):
        for name in part_names():
            print(name)
        return 0

    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["route"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-9s %-*s  %s" % (mark, r["group"], width, r["route"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad catalog",
        description="parts catalogue + standards knowledge base "
                    "(retrieval, not modelling)")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
