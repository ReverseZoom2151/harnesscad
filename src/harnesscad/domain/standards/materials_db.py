"""Cited engineering-materials database (anvilate + cad-cae-copilot merge).

A single stdlib-only materials database for harnesscad, merging two mined
MIT-licensed sources into one lookup with per-record provenance:

* **anvilate** (MIT, (c) 2026 Clay Good) -- citation-tagged seed materials from
  ``resources/cad_repos/anvilate-main/anvilate-main/src/anvilate/standards/
  data/materials.yaml`` (17 materials, each property carrying an explicit
  source + condition citation: ASM handbook values, ASTM/EN specified minima,
  Shigley tables). The YAML was converted offline to the Python dicts below;
  no yaml import at runtime.
* **cad-cae-copilot** (MIT, (c) 2026 armpro24-blip) -- breadth catalogue from
  ``resources/cad_repos/cad-cae-copilot-main/cad-cae-copilot-main/aieng/src/
  aieng/context/materials.py`` (51 materials across aluminum, steels, tool
  steels, stainless, titanium, copper, magnesium, nickel superalloys,
  engineering plastics, composites and other metals; sources stated there as
  "ASM Handbook, MatWeb, typical manufacturer datasheets").

Merge policy
------------
Where both sources define the same material (11 overlaps: Al 6061-T6,
7075-T6, 2024-T3, 6082-T6, steels 1045 / A36 / 4140 / 4340, stainless 304,
Ti-6Al-4V, bearing bronze C93200), the citation-tagged **anvilate values are
preferred** for E / nu / rho / yield / ultimate, and the copilot record only
supplements fields anvilate lacks (thermal expansion coefficient,
description).  Copilot-only records are marked with the copilot source
string.  Three copilot rows are pure duplicates of other copilot rows
(Ti-Grade5 == Ti-6Al-4V, Brass-C360 == Cu-C36000, Bronze-C932 == Cu-C93200)
and are kept as aliases rather than records.  Note the two 4xxx steels differ
by condition: anvilate carries the *annealed* Shigley values (4140 yield
417 MPa), copilot carried Q&T-like values; the annealed, cited values win.

Units (normalised and fixed)
----------------------------
* ``E_mpa``                     Young's modulus, MPa (anvilate GPa * 1000)
* ``nu``                        Poisson ratio, dimensionless
* ``rho_kg_m3``                 density, kg/m^3 (anvilate g/cm^3 * 1000)
* ``yield_mpa`` / ``ultimate_mpa`` / ``endurance_mpa``   strengths, MPa
* ``thermal_expansion_um_m_k``  CTE, um/(m*K) i.e. 1e-6 / K

Contract compatibility
----------------------
:mod:`harnesscad.domain.spec.contract` keeps its own 12-key density table
(``MATERIAL_DENSITY_G_PER_MM3``) plus spelling/synonym aliases.  Every name
that table accepts resolves here too: :data:`CONTRACT_RECORD_FOR` maps each
contract density-class key to a record in this database whose density matches
exactly, :data:`CONTRACT_ALIASES` mirrors the contract synonym table, and
:func:`contract_density` reproduces the contract lookup (g/mm^3) from these
records.  Six small filler records (PLA, PETG, Nylon, TPU, Resin-SLA, Brass)
exist only to back the contract polymer/brass classes with a consistent
density; their density values come from the contract table's own cited
sources (Prusa/Ultimaker/Formlabs datasheets, MatWeb) and they carry no
strength data.  The ``--selfcheck`` asserts the invariant against a read-only
import of contract.py.

Licensing: both upstream sources are MIT licensed; the anvilate dataset
additionally declares its property *values* CC0-1.0 (source standards are not
redistributed).  This module redistributes values with attribution only.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

__all__ = [
    "MaterialRecord",
    "material",
    "list_materials",
    "material_names",
    "normalize_name",
    "youngs_modulus_mpa",
    "poisson_ratio",
    "density_kg_m3",
    "yield_strength_mpa",
    "ultimate_strength_mpa",
    "CONTRACT_RECORD_FOR",
    "CONTRACT_ALIASES",
    "contract_density",
    "main",
]

# --------------------------------------------------------------------------- #
# Record type
# --------------------------------------------------------------------------- #

SRC_ANVILATE = "anvilate materials.yaml (MIT, (c) 2026 Clay Good)"
SRC_COPILOT = (
    "cad-cae-copilot aieng/context/materials.py (MIT, (c) 2026 armpro24-blip; "
    "ASM Handbook / MatWeb / manufacturer datasheets)"
)
SRC_MERGED = (
    "merged: anvilate (cited values, preferred) + cad-cae-copilot "
    "(thermal expansion, description)"
)
SRC_CONTRACT = (
    "harnesscad contract.py MATERIAL_DENSITY_G_PER_MM3 compatibility filler "
    "(Prusa/Ultimaker/Formlabs datasheets, MatWeb nominal densities)"
)


class MaterialRecord(NamedTuple):
    """One material, units per module docstring. Missing values are ``None``."""

    name: str                                   # canonical record name
    category: str                               # e.g. "aluminum", "stainless_steel"
    source: str                                 # provenance of the record
    E_mpa: Optional[float]                      # Young's modulus (MPa)
    nu: Optional[float]                         # Poisson ratio
    rho_kg_m3: float                            # density (kg/m^3)
    yield_mpa: Optional[float]                  # yield strength (MPa)
    ultimate_mpa: Optional[float]               # ultimate tensile strength (MPa)
    endurance_mpa: Optional[float]              # endurance limit (MPa), if cited
    thermal_expansion_um_m_k: Optional[float]   # CTE (1e-6/K)
    description: str                            # short prose description
    citations: Dict[str, str]                   # per-property source strings


# --------------------------------------------------------------------------- #
# Database construction
# --------------------------------------------------------------------------- #

_DB: Dict[str, MaterialRecord] = {}
_ALIASES: Dict[str, str] = {}   # normalized alias -> canonical record name


def normalize_name(name: str) -> str:
    """Lowercase alphanumeric normalization ('Al 6061-T6' -> 'al6061t6')."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _add(record: MaterialRecord, aliases: Sequence[str] = ()) -> None:
    key = normalize_name(record.name)
    if key in _DB or key in _ALIASES:
        raise ValueError("duplicate material key: %s" % record.name)
    _DB[key] = record
    for alias in aliases:
        akey = normalize_name(alias)
        if not akey or akey == key:
            continue
        if akey in _DB or (akey in _ALIASES and _ALIASES[akey] != key):
            raise ValueError("alias collision: %s" % alias)
        _ALIASES[akey] = key


def _anv(
    name: str,
    category: str,
    E_gpa: float,
    nu: float,
    rho_g_cm3: float,
    ys: float,
    us: float,
    cites: Dict[str, str],
    endurance: Optional[float] = None,
    cte: Optional[float] = None,
    description: str = "",
    merged: bool = False,
    aliases: Sequence[str] = (),
) -> None:
    """Anvilate record (GPa / g-cm3 converted to MPa / kg-m3 on insertion)."""
    _add(
        MaterialRecord(
            name=name,
            category=category,
            source=SRC_MERGED if merged else SRC_ANVILATE,
            E_mpa=E_gpa * 1000.0,
            nu=nu,
            rho_kg_m3=rho_g_cm3 * 1000.0,
            yield_mpa=ys,
            ultimate_mpa=us,
            endurance_mpa=endurance,
            thermal_expansion_um_m_k=cte,
            description=description,
            citations=cites,
        ),
        aliases,
    )


def _cop(
    name: str,
    category: str,
    E_mpa: float,
    nu: float,
    rho: float,
    ys: float,
    us: float,
    cte: float,
    description: str,
    aliases: Sequence[str] = (),
) -> None:
    """cad-cae-copilot record (already MPa / kg-m3)."""
    _add(
        MaterialRecord(
            name=name,
            category=category,
            source=SRC_COPILOT,
            E_mpa=E_mpa,
            nu=nu,
            rho_kg_m3=rho,
            yield_mpa=ys,
            ultimate_mpa=us,
            endurance_mpa=None,
            thermal_expansion_um_m_k=cte,
            description=description,
            citations={"all_properties": SRC_COPILOT},
        ),
        aliases,
    )


_SHIGLEY_ENDURANCE = (
    "derived from ultimate strength; 0.5 x ultimate (Shigley screening "
    "estimate), fully reversed bending [anvilate]"
)


def _build() -> None:
    # ----- anvilate records (17), 11 of them merged with copilot extras ----- #

    _anv(
        "AA-6061-T6", "aluminum", 68.9, 0.33, 2.70, 276.0, 310.0,
        {
            "elastic_modulus": "ASM Aerospace Metals - 6061-T6 (T6 temper, room temperature)",
            "poisson_ratio": "ASM Aerospace Metals - 6061-T6 (room temperature)",
            "density": "ASM Aerospace Metals - 6061-T6 (room temperature)",
            "yield_strength": "ASM Aerospace Metals - 6061-T6 (T6 temper, 0.2% offset)",
            "ultimate_strength": "ASM Aerospace Metals - 6061-T6 (T6 temper)",
            "thermal_expansion": SRC_COPILOT,
        },
        cte=23.6, merged=True,
        description="General-purpose aluminum alloy, good machinability, moderate strength",
        aliases=("Al6061-T6", "al 6061", "6061", "6061-T6", "aluminum 6061",
                 "aluminium 6061", "AA6061"),
    )
    _anv(
        "AA-7075-T6", "aluminum", 71.7, 0.33, 2.81, 503.0, 572.0,
        {
            "elastic_modulus": "ASM Aerospace Metals - 7075-T6 (T6 temper, room temperature)",
            "poisson_ratio": "ASM Aerospace Metals - 7075-T6 (room temperature)",
            "density": "ASM Aerospace Metals - 7075-T6 (room temperature)",
            "yield_strength": "ASM Aerospace Metals - 7075-T6 (T6 temper, 0.2% offset)",
            "ultimate_strength": "ASM Aerospace Metals - 7075-T6 (T6 temper)",
            "thermal_expansion": SRC_COPILOT,
        },
        cte=23.2, merged=True,
        description="High-strength aluminum alloy, aerospace grade, excellent strength-to-weight",
        aliases=("Al7075-T6", "al 7075", "7075", "7075-T6", "aluminum 7075"),
    )
    _anv(
        "AA-2024-T3", "aluminum", 73.1, 0.33, 2.78, 345.0, 483.0,
        {
            "elastic_modulus": "ASM Aerospace Metals - 2024-T3 (T3 temper, room temperature)",
            "poisson_ratio": "ASM Aerospace Metals - 2024-T3 (room temperature)",
            "density": "ASM Aerospace Metals - 2024-T3 (room temperature)",
            "yield_strength": "ASM Aerospace Metals - 2024-T3 (T3 temper, 0.2% offset, 50 ksi)",
            "ultimate_strength": "ASM Aerospace Metals - 2024-T3 (T3 temper, 70 ksi)",
            "thermal_expansion": SRC_COPILOT,
        },
        cte=22.8, merged=True,
        description="High-strength Al-Cu alloy, aerospace structural applications, good fatigue resistance",
        aliases=("Al2024-T3", "al 2024", "2024", "2024-T3", "aluminum 2024"),
    )
    _anv(
        "AA-6063-T6", "aluminum", 68.9, 0.33, 2.70, 214.0, 241.0,
        {
            "elastic_modulus": "ASM Aerospace Metals - 6063-T6 (T6 temper, room temperature)",
            "poisson_ratio": "ASM Aerospace Metals - 6063-T6 (room temperature)",
            "density": "ASM Aerospace Metals - 6063-T6 (room temperature)",
            "yield_strength": "ASM Aerospace Metals - 6063-T6 (T6 temper, 0.2% offset)",
            "ultimate_strength": "ASM Aerospace Metals - 6063-T6 (T6 temper)",
        },
        description="Architectural extrusion aluminum alloy",
        aliases=("Al6063-T6", "al 6063", "6063", "6063-T6"),
    )
    _anv(
        "AA-6082-T6", "aluminum", 70.0, 0.33, 2.70, 250.0, 290.0,
        {
            "elastic_modulus": "ASM - 6082-T6 (T6 temper, room temperature)",
            "poisson_ratio": "ASM - 6082-T6 (room temperature)",
            "density": "ASM - 6082-T6 (room temperature)",
            "yield_strength": "EN 755-2 - EN AW-6082-T6 extrusions, thickness <= 5 mm (Rp0.2 minimum)",
            "ultimate_strength": "EN 755-2 - EN AW-6082-T6 extrusions, thickness <= 5 mm (Rm minimum)",
            "thermal_expansion": SRC_COPILOT,
        },
        cte=23.5, merged=True,
        description="Structural aluminum alloy, good weldability, bridge and truss applications",
        aliases=("Al6082-T6", "al 6082", "6082", "6082-T6"),
    )
    _anv(
        "AA-A356-T6", "aluminum", 72.4, 0.33, 2.68, 205.0, 283.0,
        {
            "elastic_modulus": "ASM - A356.0-T6 (T6, permanent mold, room temperature)",
            "poisson_ratio": "ASM - A356.0-T6 (room temperature)",
            "density": "ASM - A356.0-T6 (room temperature)",
            "yield_strength": "ASM - A356.0-T6 (T6, permanent mold, 0.2% offset, 30 ksi)",
            "ultimate_strength": "ASM - A356.0-T6 (T6, permanent mold, 41 ksi)",
        },
        description="Cast aluminum A356.0-T6, permanent-mold values; common for cast brackets, housings, wheels",
        aliases=("A356", "A356.0-T6", "A356-T6", "cast aluminum A356"),
    )
    _anv(
        "ASTM-A36", "carbon_steel", 200.0, 0.30, 7.85, 250.0, 400.0,
        {
            "elastic_modulus": "AISC Steel Construction Manual (room temperature)",
            "poisson_ratio": "AISC Steel Construction Manual (room temperature)",
            "density": "AISC Steel Construction Manual (room temperature)",
            "yield_strength": "ASTM A36 specified minimum (as-rolled)",
            "ultimate_strength": "ASTM A36 specified minimum (as-rolled)",
            "endurance_limit": _SHIGLEY_ENDURANCE,
            "thermal_expansion": SRC_COPILOT,
        },
        endurance=200.0, cte=12.0, merged=True,
        description="Common structural steel, mild carbon steel, used in construction and general fabrication",
        aliases=("Steel-A36", "A36", "A36 steel", "structural steel A36"),
    )
    _anv(
        "ASTM-A992", "carbon_steel", 200.0, 0.30, 7.85, 345.0, 450.0,
        {
            "elastic_modulus": "AISC Steel Construction Manual (room temperature)",
            "poisson_ratio": "AISC Steel Construction Manual (room temperature)",
            "density": "AISC Steel Construction Manual (room temperature)",
            "yield_strength": "ASTM A992 specified minimum (50 ksi)",
            "ultimate_strength": "ASTM A992 specified minimum (65 ksi)",
            "endurance_limit": _SHIGLEY_ENDURANCE,
        },
        endurance=225.0,
        description="ASTM A992 structural steel (wide-flange shapes)",
        aliases=("A992", "A992 steel"),
    )
    _anv(
        "AISI-1018-CD", "carbon_steel", 200.0, 0.29, 7.87, 370.0, 440.0,
        {
            "elastic_modulus": "ASM - AISI 1018 (room temperature)",
            "poisson_ratio": "ASM - AISI 1018 (room temperature)",
            "density": "ASM - AISI 1018 (room temperature)",
            "yield_strength": "Shigley Mechanical Engineering Design, Table A-20 (cold-drawn, 54 kpsi)",
            "ultimate_strength": "Shigley Mechanical Engineering Design, Table A-20 (cold-drawn, 64 kpsi)",
            "endurance_limit": _SHIGLEY_ENDURANCE,
        },
        endurance=220.0,
        description="AISI 1018 low-carbon steel, cold-drawn",
        aliases=("1018", "AISI 1018", "Steel-1018", "1018-CD", "mild steel"),
    )
    _anv(
        "AISI-1045-CD", "carbon_steel", 200.0, 0.29, 7.87, 530.0, 630.0,
        {
            "elastic_modulus": "ASM - AISI 1045 (room temperature)",
            "poisson_ratio": "ASM - AISI 1045 (room temperature)",
            "density": "ASM - AISI 1045 (room temperature)",
            "yield_strength": "Shigley Mechanical Engineering Design, Table A-20 (cold-drawn, 77 kpsi)",
            "ultimate_strength": "Shigley Mechanical Engineering Design, Table A-20 (cold-drawn, 91 kpsi)",
            "endurance_limit": _SHIGLEY_ENDURANCE,
            "thermal_expansion": SRC_COPILOT,
        },
        endurance=315.0, cte=11.5, merged=True,
        description="Medium-carbon steel, good strength and toughness, widely used in mechanical parts",
        aliases=("Steel-1045", "1045", "AISI 1045", "1045-CD"),
    )
    _anv(
        "AISI-4140", "alloy_steel", 205.0, 0.29, 7.85, 417.0, 655.0,
        {
            "elastic_modulus": "ASM - AISI 4140 (annealed, room temperature)",
            "poisson_ratio": "ASM - AISI 4140 (room temperature)",
            "density": "ASM - AISI 4140 (room temperature)",
            "yield_strength": "Shigley Mechanical Engineering Design, Table A-21 (annealed at 815 C, 60.5 kpsi)",
            "ultimate_strength": "Shigley Mechanical Engineering Design, Table A-21 (annealed at 815 C, 95 kpsi)",
            "endurance_limit": _SHIGLEY_ENDURANCE,
            "thermal_expansion": SRC_COPILOT,
        },
        endurance=327.5, cte=12.3, merged=True,
        description="Chromium-molybdenum alloy steel (annealed values), high fatigue strength, shafts and axles",
        aliases=("Steel-4140", "4140", "AISI 4140"),
    )
    _anv(
        "AISI-4340", "alloy_steel", 205.0, 0.29, 7.85, 470.0, 745.0,
        {
            "elastic_modulus": "ASM - AISI 4340 (annealed, room temperature)",
            "poisson_ratio": "ASM - AISI 4340 (room temperature)",
            "density": "ASM - AISI 4340 (room temperature)",
            "yield_strength": "Shigley Mechanical Engineering Design, Table A-21 (annealed at 810 C, 68.5 kpsi)",
            "ultimate_strength": "Shigley Mechanical Engineering Design, Table A-21 (annealed at 810 C, 108 kpsi)",
            "endurance_limit": _SHIGLEY_ENDURANCE,
            "thermal_expansion": SRC_COPILOT,
        },
        endurance=372.5, cte=12.3, merged=True,
        description="Nickel-chromium-molybdenum alloy steel (annealed values), aerospace and defense",
        aliases=("Steel-4340", "4340", "AISI 4340"),
    )
    _anv(
        "SS-304", "stainless_steel", 193.0, 0.29, 8.00, 205.0, 515.0,
        {
            "elastic_modulus": "ASM - AISI 304 annealed (room temperature)",
            "poisson_ratio": "ASM - AISI 304 annealed (room temperature)",
            "density": "ASM - AISI 304 annealed (room temperature)",
            "yield_strength": "ASTM A240 specified minimum (annealed, 30 ksi)",
            "ultimate_strength": "ASTM A240 specified minimum (annealed, 75 ksi)",
            "thermal_expansion": SRC_COPILOT,
        },
        cte=17.3, merged=True,
        description="Austenitic stainless steel, general-purpose corrosion resistance, food-grade",
        aliases=("Steel-AISI-304", "304", "AISI 304", "stainless 304", "SS304",
                 "304 stainless"),
    )
    _anv(
        "SS-316", "stainless_steel", 193.0, 0.30, 8.00, 205.0, 515.0,
        {
            "elastic_modulus": "ASM - AISI 316 annealed (room temperature)",
            "poisson_ratio": "ASM - AISI 316 annealed (room temperature)",
            "density": "ASM - AISI 316 annealed (room temperature)",
            "yield_strength": "ASTM A240 specified minimum (annealed, 30 ksi)",
            "ultimate_strength": "ASTM A240 specified minimum (annealed, 75 ksi)",
        },
        description="Austenitic stainless steel 316 (annealed), marine/chemical corrosion resistance",
        aliases=("316", "AISI 316", "stainless 316", "SS316", "316 stainless"),
    )
    _anv(
        "Ti-6Al-4V", "titanium", 113.8, 0.342, 4.43, 880.0, 950.0,
        {
            "elastic_modulus": "ASM - Ti-6Al-4V annealed (annealed, room temperature)",
            "poisson_ratio": "ASM - Ti-6Al-4V annealed (room temperature)",
            "density": "ASM - Ti-6Al-4V annealed (room temperature)",
            "yield_strength": "ASM - Ti-6Al-4V annealed (annealed, 0.2% offset)",
            "ultimate_strength": "ASM - Ti-6Al-4V annealed (annealed)",
            "thermal_expansion": SRC_COPILOT,
        },
        cte=8.6, merged=True,
        description="Titanium alloy (Grade 5, annealed), very high strength-to-weight ratio, biocompatible",
        aliases=("Ti-Grade5", "Ti 6Al 4V", "Ti6Al4V", "titanium grade 5",
                 "grade 5 titanium"),
    )
    _anv(
        "C93200-SAE660", "copper_alloy", 100.0, 0.34, 8.93, 125.0, 240.0,
        {
            "elastic_modulus": "Copper Development Association - C93200 (as-cast, room temperature)",
            "poisson_ratio": "Copper Development Association - C93200 (room temperature)",
            "density": "Copper Development Association - C93200 (room temperature)",
            "yield_strength": "Copper Development Association - C93200 (as-cast, 0.5% ext. under load, 18 ksi)",
            "ultimate_strength": "Copper Development Association - C93200 (as-cast, 35 ksi)",
            "thermal_expansion": SRC_COPILOT,
        },
        cte=18.0, merged=True,
        description="Bearing bronze C93200 (SAE 660), standard cast plain-bearing/bushing bronze",
        aliases=("Cu-C93200", "Bronze-C932", "C93200", "C932", "SAE 660",
                 "bearing bronze"),
    )
    _anv(
        "ASTM-A536-65-45-12", "cast_iron", 169.0, 0.28, 7.10, 310.0, 448.0,
        {
            "elastic_modulus": "ASM - ductile iron 65-45-12 (as-cast, room temperature)",
            "poisson_ratio": "ASM - ductile iron (room temperature)",
            "density": "ASM - ductile iron (room temperature)",
            "yield_strength": "ASTM A536 specified minimum (grade 65-45-12, 45 ksi)",
            "ultimate_strength": "ASTM A536 specified minimum (grade 65-45-12, 65 ksi)",
        },
        description="Ductile iron 65-45-12 (ASTM A536); endurance deliberately omitted for cast iron",
        aliases=("ductile iron", "A536", "65-45-12", "ductile iron 65-45-12"),
    )

    # ----- cad-cae-copilot only records (37) ----- #

    _cop("Al5052-H32", "aluminum", 70300, 0.33, 2680, 193, 228, 23.8,
         "Non-heat-treatable Al-Mg alloy, excellent corrosion resistance, good formability",
         aliases=("5052", "al 5052", "5052-H32"))
    _cop("Al5083-H116", "aluminum", 71000, 0.33, 2650, 215, 305, 24.2,
         "Marine-grade Al-Mg alloy, superior corrosion resistance in seawater",
         aliases=("5083", "al 5083", "5083-H116"))
    _cop("Tool-Steel-H13", "tool_steel", 210000, 0.30, 7800, 1380, 1620, 10.4,
         "Hot-work tool steel, excellent thermal fatigue resistance, dies and molds",
         aliases=("H13", "AISI H13", "H13 tool steel"))
    _cop("Tool-Steel-D2", "tool_steel", 210000, 0.30, 7870, 1530, 1730, 10.5,
         "High-carbon high-chromium cold-work tool steel, excellent wear resistance",
         aliases=("D2", "AISI D2", "D2 tool steel"))
    _cop("Steel-316L", "stainless_steel", 193000, 0.27, 7990, 170, 485, 16.0,
         "Austenitic stainless steel 316L (low carbon), corrosion resistant",
         aliases=("316L", "SS316L", "stainless 316L"))
    _cop("Steel-17-4PH", "stainless_steel", 196000, 0.27, 7800, 1000, 1100, 10.8,
         "Precipitation-hardening stainless steel, high strength with good corrosion resistance",
         aliases=("17-4PH", "17-4", "SS17-4PH"))
    _cop("Steel-420", "stainless_steel", 200000, 0.27, 7750, 550, 750, 10.3,
         "Martensitic stainless steel, hardenable by heat treatment, cutlery and tooling",
         aliases=("420", "SS420", "stainless 420"))
    _cop("Steel-440C", "stainless_steel", 200000, 0.30, 7750, 450, 760, 10.2,
         "High-carbon martensitic stainless steel, highest hardness and wear resistance",
         aliases=("440C", "SS440C", "stainless 440C"))
    _cop("Steel-321", "stainless_steel", 193000, 0.28, 8020, 205, 515, 16.6,
         "Stabilized austenitic stainless steel, excellent high-temperature oxidation resistance",
         aliases=("321", "SS321", "stainless 321"))
    _cop("Ti-Grade2", "titanium", 105000, 0.37, 4510, 275, 345, 8.6,
         "Commercially pure titanium, excellent corrosion resistance, biocompatible implants",
         aliases=("titanium grade 2", "grade 2 titanium", "CP titanium"))
    _cop("Cu-C11000", "copper_alloy", 115000, 0.33, 8960, 69, 220, 16.5,
         "Electrolytic tough pitch copper, excellent electrical and thermal conductivity",
         aliases=("C11000", "ETP copper"))
    _cop("Cu-C36000", "copper_alloy", 97000, 0.34, 8430, 200, 400, 20.5,
         "Free-cutting brass, excellent machinability, fasteners and fittings",
         aliases=("C36000", "Brass-C360", "C360", "free-cutting brass"))
    _cop("Cu-C95400", "copper_alloy", 110000, 0.33, 7600, 275, 586, 16.2,
         "Aluminum bronze, high strength and wear resistance, marine hardware",
         aliases=("C95400", "aluminum bronze"))
    _cop("Mg-AZ31B", "magnesium", 45000, 0.35, 1770, 150, 260, 26.0,
         "Wrought magnesium alloy, good formability, lightweight structural applications",
         aliases=("AZ31B", "AZ31", "magnesium AZ31B"))
    _cop("Mg-AZ91D", "magnesium", 45000, 0.35, 1810, 150, 230, 26.0,
         "Cast magnesium alloy, excellent castability, automotive and electronics housings",
         aliases=("AZ91D", "AZ91", "magnesium AZ91D"))
    _cop("Inconel-718", "nickel_alloy", 200000, 0.29, 8190, 1100, 1240, 13.0,
         "Nickel superalloy, high temperature strength, jet engine and turbine components",
         aliases=("718", "IN718", "inconel 718"))
    _cop("Inconel-625", "nickel_alloy", 207000, 0.28, 8440, 517, 930, 12.8,
         "Nickel-chromium superalloy, outstanding corrosion and oxidation resistance",
         aliases=("625", "IN625", "inconel 625"))
    _cop("Hastelloy-C276", "nickel_alloy", 205000, 0.31, 8890, 310, 690, 11.2,
         "Nickel-molybdenum-chromium alloy, exceptional corrosion resistance in harsh chemicals",
         aliases=("C276", "hastelloy"))
    _cop("Monel-400", "nickel_alloy", 179000, 0.32, 8800, 240, 550, 13.9,
         "Nickel-copper alloy, excellent corrosion resistance in marine and chemical environments",
         aliases=("monel", "400 monel"))
    _cop("Nylon-PA66", "polymer", 3000, 0.39, 1140, 80, 85, 80.0,
         "Engineering thermoplastic, self-lubricating, moderate strength, light weight",
         aliases=("PA66", "nylon 66"))
    _cop("PETG-CF", "polymer", 5500, 0.38, 1270, 55, 60, 40.0,
         "Carbon-fiber reinforced PETG, good for FDM 3D printing, improved stiffness",
         aliases=("PETG CF", "carbon fiber PETG"))
    _cop("ABS", "polymer", 2300, 0.39, 1040, 45, 40, 90.0,
         "Acrylonitrile butadiene styrene, tough impact-resistant thermoplastic, consumer products",
         aliases=("acrylonitrile butadiene styrene",))
    _cop("PC", "polymer", 2350, 0.37, 1200, 62, 65, 65.0,
         "Polycarbonate, high impact strength, transparent, optical and safety applications",
         aliases=("polycarbonate",))
    _cop("PEEK", "polymer", 3600, 0.40, 1320, 90, 100, 47.0,
         "Polyether ether ketone, high-performance thermoplastic, chemical and wear resistant",
         aliases=("polyether ether ketone",))
    _cop("PTFE", "polymer", 550, 0.46, 2200, 15, 25, 100.0,
         "Polytetrafluoroethylene (Teflon), extremely low friction, chemical inertness",
         aliases=("teflon", "polytetrafluoroethylene"))
    _cop("POM", "polymer", 2800, 0.35, 1410, 65, 70, 85.0,
         "Polyoxymethylene (Acetal), low friction, high stiffness, precision gears and bearings",
         aliases=("acetal", "delrin", "polyoxymethylene"))
    _cop("Nylon-PA6", "polymer", 2800, 0.39, 1130, 75, 80, 80.0,
         "Cast nylon, good wear resistance, lower moisture absorption than PA66",
         aliases=("PA6", "nylon 6"))
    _cop("Nylon-PA12", "polymer", 1700, 0.39, 1020, 45, 50, 100.0,
         "Low-moisture nylon, good chemical resistance, SLS 3D printing powder",
         aliases=("PA12", "nylon 12"))
    _cop("UHMWPE", "polymer", 900, 0.46, 930, 22, 35, 120.0,
         "Ultra-high molecular weight polyethylene, extremely tough, low friction, liners",
         aliases=())
    _cop("PVC", "polymer", 2800, 0.38, 1400, 45, 50, 50.0,
         "Polyvinyl chloride, rigid or flexible, chemical resistant, piping and construction",
         aliases=("polyvinyl chloride",))
    _cop("CFRP-T300", "composite", 150000, 0.30, 1600, 1500, 1800, 1.0,
         "Carbon fiber reinforced polymer (standard modulus), aerospace structures",
         aliases=("T300", "CFRP T300"))
    _cop("CFRP-T700", "composite", 170000, 0.30, 1600, 2100, 2400, 0.5,
         "Carbon fiber reinforced polymer (intermediate modulus), high-performance sports",
         aliases=("T700", "CFRP T700"))
    _cop("GFRP-E-Glass", "composite", 45000, 0.22, 1850, 350, 450, 12.0,
         "Glass fiber reinforced polymer (E-glass), cost-effective composite, boats and tanks",
         aliases=("E-glass", "fiberglass"))
    _cop("GFRP-S-Glass", "composite", 55000, 0.22, 1850, 450, 550, 10.0,
         "Glass fiber reinforced polymer (S-glass), higher strength and modulus than E-glass",
         aliases=("S-glass",))
    _cop("Zinc-ZA-8", "other_metal", 85000, 0.30, 6300, 225, 310, 27.0,
         "Zinc-aluminum alloy, good castability and wear resistance, die-cast components",
         aliases=("ZA-8", "ZA8", "zinc ZA8"))
    _cop("Cobalt-Chrome-MP1", "other_metal", 240000, 0.30, 8400, 600, 900, 14.0,
         "Cobalt-chrome alloy, biocompatible, high wear resistance, dental and implants",
         aliases=("CoCr", "cobalt chrome", "MP1"))
    _cop("Cast-Iron-Grey", "cast_iron", 110000, 0.26, 7200, 180, 260, 10.5,
         "Grey cast iron, good compressive strength, brittle, vibration damping",
         aliases=("grey cast iron", "gray cast iron", "cast iron"))

    # ----- contract.py compatibility fillers (6; density only, cited) ----- #

    def _filler(name: str, rho: float, description: str,
                aliases: Sequence[str] = ()) -> None:
        _add(
            MaterialRecord(
                name=name, category="polymer" if rho < 3000 else "other_metal",
                source=SRC_CONTRACT, E_mpa=None, nu=None, rho_kg_m3=rho,
                yield_mpa=None, ultimate_mpa=None, endurance_mpa=None,
                thermal_expansion_um_m_k=None, description=description,
                citations={"density": SRC_CONTRACT},
            ),
            aliases,
        )

    _filler("PLA", 1240.0, "Polylactic acid FDM filament (nominal datasheet density)",
            aliases=("polylactic acid",))
    _filler("PETG", 1270.0, "PETG FDM filament, unfilled (nominal datasheet density)",
            aliases=("PET",))
    _filler("Nylon", 1010.0, "Generic nylon/polyamide class (contract table nominal, PA12 datasheet)",
            aliases=("polyamide", "PA"))
    _filler("TPU", 1210.0, "Thermoplastic polyurethane flexible filament (nominal datasheet density)")
    _filler("Resin-SLA", 1180.0, "Standard SLA photopolymer resin (Formlabs nominal density)",
            aliases=("resin", "SLA", "SLA resin"))
    _filler("Brass", 8500.0, "Generic brass class (MatWeb nominal density; see Cu-C36000 for C360 alloy data)")


_build()


# --------------------------------------------------------------------------- #
# Contract compatibility layer
# --------------------------------------------------------------------------- #

# Contract density-class key -> canonical record name here. The record's
# density (kg/m^3 / 1e6) matches contract.MATERIAL_DENSITY_G_PER_MM3 exactly.
CONTRACT_RECORD_FOR: Dict[str, str] = {
    "pla": "PLA",
    "abs": "ABS",
    "petg": "PETG",
    "nylon": "Nylon",
    "tpu": "TPU",
    "resin": "Resin-SLA",
    "aluminum": "AA-6061-T6",
    "steel": "AISI-1018-CD",
    "stainless": "SS-304",
    "titanium": "Ti-6Al-4V",
    "brass": "Brass",
    "copper": "Cu-C11000",
}

# Mirror of contract.py's _MATERIAL_ALIASES (normalized alias -> class key).
CONTRACT_ALIASES: Dict[str, str] = {
    "polylacticacid": "pla",
    "acrylonitrilebutadienestyrene": "abs",
    "petg": "petg",
    "pet": "petg",
    "pa": "nylon",
    "pa12": "nylon",
    "polyamide": "nylon",
    "sla": "resin",
    "aluminium": "aluminum",
    "al": "aluminum",
    "al6061": "aluminum",
    "6061": "aluminum",
    "al7075": "aluminum",
    "7075": "aluminum",
    "mildsteel": "steel",
    "carbonsteel": "steel",
    "1018": "steel",
    "stainlesssteel": "stainless",
    "304": "stainless",
    "316": "stainless",
    "ss304": "stainless",
    "ti": "titanium",
    "ti6al4v": "titanium",
    "cu": "copper",
}


def _register_contract_names() -> None:
    """Ensure every name contract.py accepts also resolves via material()."""
    for key, record_name in CONTRACT_RECORD_FOR.items():
        target = normalize_name(record_name)
        if key not in _DB and key not in _ALIASES:
            _ALIASES[key] = target
    for alias, cls in CONTRACT_ALIASES.items():
        target = normalize_name(CONTRACT_RECORD_FOR[cls])
        if alias not in _DB and alias not in _ALIASES:
            _ALIASES[alias] = target


_register_contract_names()


def contract_density(name: str) -> Optional[float]:
    """Density in g/mm^3 for a contract.py-accepted material name.

    Reproduces contract.py's resolution semantics (class-level, so
    "7075" resolves to the generic aluminum class density, exactly as
    ``contract._density_for_material`` does) but reads the value from this
    database's records via :data:`CONTRACT_RECORD_FOR`. Returns ``None``
    for names outside the contract vocabulary.
    """
    key = normalize_name(name)
    if not key:
        return None
    cls = key if key in CONTRACT_RECORD_FOR else CONTRACT_ALIASES.get(key)
    if cls is None:
        return None
    return _DB[normalize_name(CONTRACT_RECORD_FOR[cls])].rho_kg_m3 / 1.0e6


# --------------------------------------------------------------------------- #
# Lookup API
# --------------------------------------------------------------------------- #


def material(name: str) -> MaterialRecord:
    """Look up a material by canonical name or alias, or raise ``KeyError``.

    Names are normalized to lowercase alphanumerics, so "Al 6061", "6061-T6",
    "AA-6061-T6" and "al6061t6" all resolve to the same record.
    """
    key = normalize_name(name)
    if key in _DB:
        return _DB[key]
    canonical = _ALIASES.get(key)
    if canonical is not None:
        return _DB[canonical]
    raise KeyError('material "%s" not found' % name)


def list_materials(category: Optional[str] = None) -> List[str]:
    """Sorted canonical material names, optionally filtered by category."""
    names = [r.name for r in _DB.values()
             if category is None or r.category == category]
    return sorted(names)


def material_names(prefix: Optional[str] = None) -> List[str]:
    """Sorted canonical names, optionally filtered by (normalized) prefix."""
    names = list_materials()
    if prefix is not None:
        p = normalize_name(prefix)
        names = [n for n in names if normalize_name(n).startswith(p)]
    return names


def youngs_modulus_mpa(name: str) -> Optional[float]:
    """Young's modulus in MPa, or ``None`` when the record carries no value."""
    return material(name).E_mpa


def poisson_ratio(name: str) -> Optional[float]:
    """Poisson ratio (dimensionless), or ``None``."""
    return material(name).nu


def density_kg_m3(name: str) -> float:
    """Density in kg/m^3 (always present)."""
    return material(name).rho_kg_m3


def yield_strength_mpa(name: str) -> Optional[float]:
    """Yield strength in MPa, or ``None``."""
    return material(name).yield_mpa


def ultimate_strength_mpa(name: str) -> Optional[float]:
    """Ultimate tensile strength in MPa, or ``None``."""
    return material(name).ultimate_mpa


# --------------------------------------------------------------------------- #
# Selfcheck / CLI
# --------------------------------------------------------------------------- #


def _approx(a: float, b: float, tol: float = 1.0e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _selfcheck() -> List[str]:
    notes: List[str] = []

    # Known anvilate-cited values.
    al = material("al 6061")
    assert al is material("6061-T6") is material("AA-6061-T6") is material("Al6061-T6")
    assert al.E_mpa == 68900.0, al.E_mpa                # 68.9 GPa
    assert al.rho_kg_m3 == 2700.0
    assert al.nu == 0.33 and al.yield_mpa == 276.0 and al.ultimate_mpa == 310.0
    assert al.thermal_expansion_um_m_k == 23.6          # copilot supplement
    assert "ASM Aerospace Metals" in al.citations["yield_strength"]
    assert material("7075").yield_mpa == 503.0
    assert material("7075").rho_kg_m3 == 2810.0
    a36 = material("A36")
    assert a36.E_mpa == 200000.0 and a36.yield_mpa == 250.0
    assert a36.endurance_mpa == 200.0

    # Merge policy: anvilate wins over copilot for overlapping records.
    s4340 = material("Steel-4340")
    assert s4340.yield_mpa == 470.0, "anvilate annealed value must win over copilot 710"
    assert s4340.thermal_expansion_um_m_k == 12.3       # copilot supplement kept
    assert s4340.source == SRC_MERGED

    # Copilot-sourced breadth records are marked as such.
    inc = material("Inconel-718")
    assert inc.E_mpa == 200000.0 and inc.rho_kg_m3 == 8190.0
    assert "cad-cae-copilot" in inc.source

    # Duplicate copilot rows collapsed into aliases.
    assert material("Ti-Grade5") is material("Ti-6Al-4V")
    assert material("Brass-C360") is material("Cu-C36000")
    assert material("Bronze-C932") is material("C93200-SAE660")

    n = len(list_materials())
    assert n == 60, "expected 60 canonical records, found %d" % n
    notes.append("materials: %d canonical records" % n)

    # Property accessors.
    assert youngs_modulus_mpa("PEEK") == 3600.0
    assert density_kg_m3("titanium") == 4430.0
    assert poisson_ratio("SS316") == 0.30
    assert yield_strength_mpa("H13") == 1380.0
    assert ultimate_strength_mpa("2024") == 483.0

    # Contract.py compatibility invariant (read-only import).
    from harnesscad.domain.spec import contract  # noqa: WPS433 (selfcheck only)

    table = contract.MATERIAL_DENSITY_G_PER_MM3
    aliases = getattr(contract, "_MATERIAL_ALIASES", {})
    for key, expected in table.items():
        got = contract_density(key)
        assert got is not None and _approx(got, expected), (key, got, expected)
        material(key)  # must resolve to a record
    for alias, cls in aliases.items():
        expected = table[cls]
        got = contract_density(alias)
        assert got is not None and _approx(got, expected), (alias, got, expected)
        material(alias)  # must resolve to a record
    notes.append(
        "contract compatibility: %d density keys + %d aliases consistent"
        % (len(table), len(aliases))
    )

    # Every record has a positive density and a citation.
    for rec in _DB.values():
        assert rec.rho_kg_m3 > 0.0, rec.name
        assert rec.citations, rec.name
        assert rec.source, rec.name
    return notes


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="materials_db",
        description="Cited engineering-materials database (anvilate + cad-cae-copilot).",
    )
    parser.add_argument(
        "--selfcheck", action="store_true",
        help="assert known cited values and the contract.py density invariant",
    )
    args = parser.parse_args(argv)
    if args.selfcheck:
        for note in _selfcheck():
            print(note)
        print("materials_db selfcheck OK")
        return 0
    for name in list_materials():
        rec = material(name)
        e = "%.0f MPa" % rec.E_mpa if rec.E_mpa is not None else "-"
        ys = "%.0f MPa" % rec.yield_mpa if rec.yield_mpa is not None else "-"
        print("%-22s %-16s E=%-12s rho=%6.0f kg/m^3  yield=%s"
              % (rec.name, rec.category, e, rec.rho_kg_m3, ys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
