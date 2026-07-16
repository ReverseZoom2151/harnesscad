"""ICE v3 embodied-carbon database integration (kerf), manifest + loader pattern.

Integrates the material lookup layer of kerf's LCA package (kerf-lca,
``materials.py``; kerf is MIT licensed, (c) 2026 Imran Paruk) which fronts the
Inventory of Carbon and Energy (ICE) v3.0 database by Circular Ecology and the
University of Bath:

    https://circularecology.com/embodied-carbon-footprint-database.html

License rationale (manifest + loader): while kerf's *code* is MIT, the
underlying ICE v3.0 *dataset* carries its own terms that do not clearly permit
redistribution of the numeric values. This module therefore vendors **no**
CO2e-per-kg factors and **no** recycled-content/recyclability percentages.
It embeds only a MANIFEST -- facts about the dataset's shape: material keys,
human-readable labels, categories, alias lists and source metadata -- and a
LOADER that reads the actual numbers at runtime from a locally available copy
of kerf's ``ice_v3.json`` (never shipped with this package). The JSON is
located via, in priority order:

1. an explicit ``path`` argument to :func:`load_ice`;
2. the ``HARNESSCAD_ICE_V3`` environment variable;
3. the default in-repo resources path
   ``resources/cad_repos/kerf-main/kerf-main/packages/kerf-lca/src/kerf_lca/data/ice_v3.json``
   resolved relative to this module's location.

Degrade behaviour (documented choice): when the dataset is not loaded,
value-bearing lookups **raise** the typed :class:`IceNotLoadedError` rather
than returning ``None`` -- an absent database is an environment condition the
caller should distinguish from "material not found" (which *does* return
``None``). Availability is queryable via :func:`is_available` without raising.

Name resolution mirrors kerf's ``lookup_material``: case-insensitive match on
key, then label, then alias, then shortest containing label/alias substring.
Alias resolution works from the manifest alone, so :func:`resolve_key` never
needs the dataset.

This module *extends* :mod:`harnesscad.domain.standards.embodied_carbon` (the
Insights accounting port): :func:`combined_carbon_intensity` prefers a loaded
ICE v3 factor and falls back to that module's ``DEFAULT_CO2E`` table.

Stdlib only, deterministic.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from harnesscad.domain.standards import embodied_carbon

__all__ = [
    "ICE_SOURCE_META",
    "ICE_MANIFEST",
    "ManifestEntry",
    "MaterialRecord",
    "IceNotLoadedError",
    "load_ice",
    "unload_ice",
    "is_available",
    "default_ice_path",
    "manifest_keys",
    "manifest_categories",
    "resolve_key",
    "lookup_material",
    "list_materials",
    "combined_carbon_intensity",
    "main",
]

ENV_VAR = "HARNESSCAD_ICE_V3"

#: Source metadata for the ICE v3.0 dataset (mirrors the JSON ``_meta`` block).
ICE_SOURCE_META: Dict[str, str] = {
    "source": "Inventory of Carbon and Energy (ICE) v3.0, University of Bath",
    "publisher": "Circular Ecology / University of Bath",
    "url": "https://circularecology.com/embodied-carbon-footprint-database.html",
    "units": "kg CO2-eq per kg of material (cradle-to-gate unless noted)",
    "version": "3.0",
    "via": "kerf (MIT, (c) 2026 Imran Paruk), packages/kerf-lca",
}


@dataclass(frozen=True)
class ManifestEntry:
    """Shape facts about one ICE material: no numeric CO2e values."""

    key: str
    label: str
    category: str
    aliases: Tuple[str, ...]


@dataclass(frozen=True)
class MaterialRecord:
    """A fully resolved material record, numbers loaded from ice_v3.json."""

    key: str
    label: str
    category: str
    co2e_per_kg: float
    recycled_content_pct: float
    recyclability_pct: float
    source: str
    source_url: str
    source_version: str


class IceNotLoadedError(RuntimeError):
    """Raised by value-bearing lookups when the ICE v3 dataset is not loaded."""


def _e(key: str, label: str, category: str, *aliases: str) -> ManifestEntry:
    return ManifestEntry(key=key, label=label, category=category, aliases=aliases)


#: Manifest of the ICE v3 dataset as shipped with kerf-lca: keys, labels,
#: categories and aliases only. CO2e and recycled-content numbers are loaded
#: at runtime and are intentionally absent here.
ICE_MANIFEST: Dict[str, ManifestEntry] = {
    m.key: m
    for m in (
        # metals
        _e("steel_general", "Steel (general, virgin)", "metals",
           "steel", "mild steel", "carbon steel"),
        _e("steel_recycled", "Steel (recycled / secondary)", "metals",
           "recycled steel", "secondary steel", "scrap steel"),
        _e("steel_stainless", "Stainless steel", "metals",
           "stainless", "stainless steel", "inox"),
        _e("aluminium_primary", "Aluminium (primary / virgin)", "metals",
           "aluminium", "aluminum", "primary aluminium", "virgin aluminium"),
        _e("aluminium_recycled", "Aluminium (recycled / secondary)", "metals",
           "recycled aluminium", "recycled aluminum", "secondary aluminium"),
        _e("copper", "Copper", "metals", "copper", "cu"),
        _e("brass", "Brass", "metals", "brass", "cu-zn"),
        _e("bronze", "Bronze", "metals", "bronze", "cu-sn"),
        _e("zinc", "Zinc", "metals", "zinc", "zn"),
        _e("lead", "Lead", "metals", "lead", "pb"),
        _e("titanium", "Titanium", "metals", "titanium", "ti", "ti-6al-4v"),
        _e("nickel", "Nickel", "metals", "nickel", "ni"),
        _e("cast_iron", "Cast iron", "metals",
           "cast iron", "grey iron", "ductile iron"),
        # concrete / masonry
        _e("concrete_general", "Concrete (general / OPC)", "concrete_masonry",
           "concrete", "opc concrete", "portland cement concrete"),
        _e("concrete_high_strength", "Concrete (high strength, C40+)",
           "concrete_masonry", "high strength concrete", "c40 concrete", "hsc"),
        _e("cement", "Cement (OPC)", "concrete_masonry",
           "cement", "opc", "portland cement"),
        _e("brick_clay", "Brick (clay / fired)", "concrete_masonry",
           "brick", "clay brick", "fired brick"),
        # glass / ceramics
        _e("glass_flat", "Glass (flat / float)", "glass_ceramics",
           "glass", "flat glass", "float glass", "window glass"),
        _e("glass_toughened", "Glass (toughened / tempered)", "glass_ceramics",
           "toughened glass", "tempered glass", "safety glass"),
        _e("ceramic_general", "Ceramic (general)", "glass_ceramics",
           "ceramic", "fired ceramic", "porcelain"),
        # timber
        _e("timber_softwood", "Timber - softwood (kiln dried)", "timber",
           "timber", "softwood", "pine", "spruce", "fir", "wood"),
        _e("timber_hardwood", "Timber - hardwood (kiln dried)", "timber",
           "hardwood", "oak", "beech", "ash"),
        _e("glulam", "Glulam (glued laminated timber)", "timber",
           "glulam", "glued laminated timber", "engineered timber"),
        _e("plywood", "Plywood", "timber", "plywood", "ply"),
        _e("mdf", "MDF (medium density fibreboard)", "timber",
           "mdf", "medium density fibreboard", "fibreboard"),
        # plastics
        _e("pvc", "PVC (polyvinyl chloride)", "plastics",
           "pvc", "polyvinyl chloride", "vinyl"),
        _e("hdpe", "HDPE (high-density polyethylene)", "plastics",
           "hdpe", "high density polyethylene", "polyethylene hd"),
        _e("ldpe", "LDPE (low-density polyethylene)", "plastics",
           "ldpe", "low density polyethylene", "polyethylene ld"),
        _e("polypropylene", "Polypropylene (PP)", "plastics",
           "pp", "polypropylene"),
        _e("polystyrene", "Polystyrene (PS general)", "plastics",
           "ps", "polystyrene", "hips", "general purpose polystyrene"),
        _e("eps", "EPS (expanded polystyrene)", "plastics",
           "eps", "expanded polystyrene", "styrofoam"),
        _e("abs", "ABS (acrylonitrile butadiene styrene)", "plastics",
           "abs", "acrylonitrile butadiene styrene"),
        _e("nylon", "Nylon (PA6 / PA66)", "plastics",
           "nylon", "pa6", "pa66", "polyamide"),
        _e("pet", "PET (polyethylene terephthalate)", "plastics",
           "pet", "polyethylene terephthalate", "polyester"),
        _e("ptfe", "PTFE (polytetrafluoroethylene)", "plastics",
           "ptfe", "teflon", "polytetrafluoroethylene"),
        _e("pc", "Polycarbonate (PC)", "plastics", "pc", "polycarbonate"),
        _e("pmma", "PMMA (acrylic / Perspex)", "plastics",
           "pmma", "acrylic", "perspex", "plexiglass",
           "polymethyl methacrylate"),
        _e("epoxy_resin", "Epoxy resin", "plastics",
           "epoxy", "epoxy resin", "epoxy adhesive"),
        _e("polyurethane_foam", "Polyurethane foam", "plastics",
           "pu foam", "polyurethane foam", "pur"),
        # composites
        _e("carbon_fibre", "Carbon fibre (CFRP / UD prepreg)", "composites",
           "carbon fibre", "carbon fiber", "cfrp", "carbon composite"),
        _e("gfrp", "Glass fibre reinforced polymer (GFRP)", "composites",
           "gfrp", "fibreglass", "fiberglass", "glass fibre composite"),
        # minerals
        _e("gypsum_plasterboard", "Gypsum plasterboard", "minerals",
           "plasterboard", "drywall", "gypsum board", "sheetrock"),
        _e("mineral_wool", "Mineral wool (rock / glass wool insulation)",
           "minerals", "mineral wool", "rockwool", "glass wool", "insulation"),
        _e("sand", "Sand / aggregate", "minerals",
           "sand", "aggregate", "gravel", "crushed stone"),
        # rubber
        _e("rubber_natural", "Natural rubber", "rubber",
           "natural rubber", "rubber", "latex", "nr"),
        _e("rubber_synthetic", "Synthetic rubber (SBR/EPDM)", "rubber",
           "synthetic rubber", "sbr", "epdm", "neoprene"),
        # textiles
        _e("cotton", "Cotton (raw fibre)", "textiles",
           "cotton", "cotton fabric", "cotton fibre"),
        _e("wool_natural", "Wool (natural)", "textiles",
           "wool", "natural wool", "merino"),
        _e("polyester_fibre", "Polyester fibre / fabric", "textiles",
           "polyester", "polyester fabric", "polyester fibre"),
        # paper
        _e("paper_kraft", "Paper (kraft / general)", "paper",
           "paper", "kraft paper", "cardboard"),
        _e("cardboard", "Cardboard / corrugated board", "paper",
           "cardboard", "corrugated cardboard", "corrugated board"),
        # precious metals
        _e("gold", "Gold", "precious_metals", "gold", "au"),
        _e("silver", "Silver", "precious_metals", "silver", "ag"),
        _e("platinum", "Platinum", "precious_metals", "platinum", "pt"),
    )
}


# --- loader -----------------------------------------------------------------

#: Runtime cache: material key -> raw entry dict from ice_v3.json.
_DATA: Optional[Dict[str, dict]] = None


def default_ice_path() -> pathlib.Path:
    """The in-repo default location of kerf's ice_v3.json.

    Resolved relative to this module: ``<repo>/src/harnesscad/domain/standards``
    -> four parents up is the repo root.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    return (
        repo_root
        / "resources"
        / "cad_repos"
        / "kerf-main"
        / "kerf-main"
        / "packages"
        / "kerf-lca"
        / "src"
        / "kerf_lca"
        / "data"
        / "ice_v3.json"
    )


def _candidate_path(path: Optional[os.PathLike] = None) -> pathlib.Path:
    if path is not None:
        return pathlib.Path(path)
    env = os.environ.get(ENV_VAR)
    if env:
        return pathlib.Path(env)
    return default_ice_path()


def load_ice(path: Optional[os.PathLike] = None) -> bool:
    """Try to load ice_v3.json; return True on success, False if unavailable.

    Resolution order: explicit ``path`` argument, the ``HARNESSCAD_ICE_V3``
    environment variable, then :func:`default_ice_path`. A missing or
    unparseable file degrades cleanly: the module stays in the not-loaded
    state (:func:`is_available` returns False) and this function returns
    False without raising.
    """
    global _DATA
    candidate = _candidate_path(path)
    try:
        with open(candidate, encoding="utf-8") as fh:
            raw = json.load(fh)
        materials = raw["materials"]
    except (OSError, ValueError, KeyError, TypeError):
        _DATA = None
        return False
    if not isinstance(materials, dict):
        _DATA = None
        return False
    _DATA = materials
    return True


def unload_ice() -> None:
    """Discard any loaded dataset (returns the module to the degraded state)."""
    global _DATA
    _DATA = None


def is_available() -> bool:
    """True when the ICE v3 dataset has been loaded and lookups carry values."""
    return _DATA is not None


# --- manifest queries (never need the dataset) -------------------------------

def manifest_keys() -> List[str]:
    """Sorted list of all material keys in the manifest."""
    return sorted(ICE_MANIFEST.keys())


def manifest_categories() -> List[str]:
    """Sorted list of distinct material categories in the manifest."""
    return sorted({m.category for m in ICE_MANIFEST.values()})


def resolve_key(name: str) -> Optional[str]:
    """Resolve a material name to its manifest key, or None.

    Mirrors kerf's lookup order (case-insensitive): exact key, exact label,
    exact alias, then substring against labels and aliases where the shortest
    containing candidate wins (ties break by candidate text then key, for a
    deterministic result). Works from the manifest alone; the ICE dataset
    need not be loaded.
    """
    if not name:
        return None
    needle = name.strip().lower()
    if not needle:
        return None

    if needle in ICE_MANIFEST:
        return needle

    for key, entry in ICE_MANIFEST.items():
        if entry.label.lower() == needle:
            return key

    for key, entry in ICE_MANIFEST.items():
        for alias in entry.aliases:
            if alias.lower() == needle:
                return key

    candidates: List[Tuple[int, str, str]] = []
    for key, entry in ICE_MANIFEST.items():
        label = entry.label.lower()
        if needle in label:
            candidates.append((len(label), label, key))
        for alias in entry.aliases:
            alias_l = alias.lower()
            if needle in alias_l:
                candidates.append((len(alias_l), alias_l, key))
    if candidates:
        candidates.sort()
        return candidates[0][2]
    return None


# --- value-bearing lookups (need the dataset) --------------------------------

def _record(key: str) -> Optional[MaterialRecord]:
    assert _DATA is not None
    entry = _DATA.get(key)
    manifest = ICE_MANIFEST.get(key)
    if entry is None or manifest is None:
        return None
    try:
        co2e = float(entry["embodied_carbon_kg_co2_per_kg"])
        recycled = float(entry.get("recycled_content_pct", 0.0))
        recyclable = float(entry.get("recyclability_pct", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    return MaterialRecord(
        key=key,
        label=manifest.label,
        category=manifest.category,
        co2e_per_kg=co2e,
        recycled_content_pct=recycled,
        recyclability_pct=recyclable,
        source=ICE_SOURCE_META["source"],
        source_url=ICE_SOURCE_META["url"],
        source_version=ICE_SOURCE_META["version"],
    )


def lookup_material(name: str) -> Optional[MaterialRecord]:
    """Look a material up by key, label or alias and return its full record.

    Returns None when the name resolves to no manifest key (or the loaded
    dataset lacks the resolved key). Raises :class:`IceNotLoadedError` when
    the ICE dataset is not loaded; call :func:`load_ice` first or check
    :func:`is_available`.
    """
    if _DATA is None:
        raise IceNotLoadedError(
            "ICE v3 dataset not loaded; call load_ice() (see %s env var)"
            % ENV_VAR
        )
    key = resolve_key(name)
    if key is None:
        return None
    return _record(key)


def list_materials() -> List[MaterialRecord]:
    """All loaded materials as records, sorted by key.

    Raises :class:`IceNotLoadedError` when the dataset is not loaded.
    """
    if _DATA is None:
        raise IceNotLoadedError(
            "ICE v3 dataset not loaded; call load_ice() (see %s env var)"
            % ENV_VAR
        )
    records = []
    for key in manifest_keys():
        rec = _record(key)
        if rec is not None:
            records.append(rec)
    return records


# --- combined lookup: ICE preferred, embodied_carbon fallback ----------------

def combined_carbon_intensity(name: str) -> Optional[Tuple[float, str]]:
    """CO2e factor (kg CO2e / kg) for a material name, from the best source.

    Prefers the ICE v3 dataset when loaded and the name resolves there;
    otherwise falls back to :data:`embodied_carbon.DEFAULT_CO2E` (the
    Insights accounting table). Returns ``(factor, source_tag)`` where
    ``source_tag`` is ``"ice_v3"`` or ``"insights_default"``, or None when
    neither source knows the material. Never raises for unknown names or an
    absent dataset.
    """
    if is_available():
        key = resolve_key(name)
        if key is not None:
            rec = _record(key)
            if rec is not None:
                return (rec.co2e_per_kg, "ice_v3")
    try:
        factor = embodied_carbon.carbon_intensity(name)
    except KeyError:
        return None
    return (factor, "insights_default")


# --- selfcheck ---------------------------------------------------------------

def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    # Manifest shape.
    n_materials = len(ICE_MANIFEST)
    n_aliases = sum(len(m.aliases) for m in ICE_MANIFEST.values())
    n_categories = len(manifest_categories())
    print("manifest: %d materials, %d aliases, %d categories"
          % (n_materials, n_aliases, n_categories))
    check(n_materials == 54, "expected 54 manifest materials")
    check(all(m.aliases for m in ICE_MANIFEST.values()),
          "every manifest entry has aliases")

    # Alias resolution from the manifest alone.
    check(resolve_key("aluminum") == "aluminium_primary",
          "alias: aluminum -> aluminium_primary")
    check(resolve_key("Stainless Steel") == "steel_stainless",
          "alias: stainless steel -> steel_stainless (case-insensitive)")
    check(resolve_key("teflon") == "ptfe", "alias: teflon -> ptfe")
    check(resolve_key("steel_general") == "steel_general", "exact key")
    check(resolve_key("Cast iron") == "cast_iron", "exact label")
    check(resolve_key("unobtainium") is None, "unknown name -> None")
    check(resolve_key("") is None, "empty name -> None")

    # Degrade path: loader pointed at a nonexistent file.
    unload_ice()
    ok = load_ice("Z:/nonexistent/definitely/ice_v3.json")
    check(ok is False, "load of nonexistent path returns False")
    check(is_available() is False, "not available after failed load")
    try:
        lookup_material("steel")
    except IceNotLoadedError:
        print("degrade: lookup_material raised IceNotLoadedError as documented")
    else:
        failures.append("lookup_material must raise IceNotLoadedError")

    # Fallback-to-embodied_carbon path (dataset not loaded).
    fb = combined_carbon_intensity("steel")
    check(fb is not None and fb[1] == "insights_default",
          "combined lookup falls back to insights_default")
    check(combined_carbon_intensity("unobtainium") is None,
          "combined lookup: unknown material -> None")
    print("fallback: steel -> %r" % (fb,))

    # Loaded path, only if the resources JSON is present.
    default = default_ice_path()
    if default.is_file():
        check(load_ice() is True, "default-path load succeeds")
        check(is_available() is True, "available after load")
        rec = lookup_material("aluminum")
        check(rec is not None and rec.key == "aluminium_primary",
              "loaded lookup: aluminum resolves to a record")
        if rec is not None:
            check(rec.co2e_per_kg > 0, "loaded record carries a CO2e value")
            check(rec.source_url == ICE_SOURCE_META["url"],
                  "record carries source metadata")
        combined = combined_carbon_intensity("titanium")
        check(combined is not None and combined[1] == "ice_v3",
              "combined lookup prefers ice_v3 when loaded")
        n_loaded = len(list_materials())
        check(n_loaded == n_materials,
              "loaded record count matches manifest")
        print("loaded: %d records from %s" % (n_loaded, default))
        unload_ice()
    else:
        print("loaded-path check skipped: %s absent (clean degrade verified)"
              % default)

    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1
    print("selfcheck OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="embodied_carbon_ice",
        description="ICE v3 embodied-carbon manifest + loader (via kerf).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="exercise manifest, alias resolution, degrade and fallback paths",
    )
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
