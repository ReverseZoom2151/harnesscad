"""The design-ACCOUNTING surface -- what a finished design owes.

``domain/standards`` carries two deterministic accounting surfaces over a
finished design that were unreachable:

  * :mod:`standards.embodied_carbon` -- the embodied-carbon (CO2e) tally over a
    bill of materials: total the CO2e and rank the worst offenders.
  * :mod:`standards.evidence_bundle` -- the cited-provenance roll-up over a
    design spec: every standards record the part leans on, hashed into
    a review-ready bundle, with a gate for anything the databases cannot vouch
    for.

This module is the surface that reads both -- the accounting a design review
runs AFTER the geometry is fixed: how much carbon does this cost, and is every
claim it makes actually cited?

    carbon_total(uses)          -> total embodied CO2e across a bill of materials
    carbon_top(uses, n)         -> the worst-offender materials, ranked
    provenance(spec)            -> the cited-provenance bundle (records + digest)

Unlike :mod:`standards.registry` (the versioned rule codebook) this surface does
not ingest or resolve rules; it aggregates the accounting a specific design owes.
Everything is deterministic and stdlib-only. Nothing here is modified in the
underlying modules -- adapters only.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry
import harnesscad.domain.standards.embodied_carbon as _carbon
import harnesscad.domain.standards.evidence_bundle as _evidence

__all__ = [
    "StandardsAccountingError",
    "carbon_total",
    "carbon_top",
    "carbon_intensity",
    "provenance",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

_STD = "harnesscad.domain.standards."


class StandardsAccountingError(ValueError):
    """Base class for every standards-accounting failure."""


# --------------------------------------------------------------------------- #
# Embodied carbon
# --------------------------------------------------------------------------- #
def _material_uses(uses: Sequence[Mapping[str, Any]]) -> List[_carbon.MaterialUse]:
    return [_carbon.MaterialUse(str(u["material"]), float(u["mass_kg"])) for u in uses]


def carbon_total(uses: Sequence[Mapping[str, Any]],
                 table: Optional[Mapping[str, float]] = None) -> float:
    """Total embodied CO2e (kg) across a bill of materials.

    ``uses`` is a sequence of ``{"material", "mass_kg"}`` mappings. Unknown
    materials raise ``KeyError`` -- an untallied material is never guessed at.
    """
    tbl = table if table is not None else _carbon.DEFAULT_CO2E
    return _carbon.aggregate(_material_uses(uses), tbl)


def carbon_top(uses: Sequence[Mapping[str, Any]], n: int = 10,
               table: Optional[Mapping[str, float]] = None) -> List[Tuple[str, float]]:
    """The top-``n`` materials by total embodied CO2e, descending (ties by name)."""
    tbl = table if table is not None else _carbon.DEFAULT_CO2E
    return _carbon.top_contributors(_material_uses(uses), n=int(n), table=tbl)


def carbon_intensity(material: str,
                     table: Optional[Mapping[str, float]] = None) -> float:
    """The CO2e factor (kg CO2e / kg) for one material. Raises if unknown."""
    tbl = table if table is not None else _carbon.DEFAULT_CO2E
    return _carbon.carbon_intensity(material, tbl)


# --------------------------------------------------------------------------- #
# Cited provenance
# --------------------------------------------------------------------------- #
def provenance(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Roll a design spec's standards data into a review-ready provenance bundle.

    Returns the bundle's records, the deterministic digest (identical specs ->
    identical digest), and any references the databases could not vouch for.
    """
    bundle = _evidence.collect_provenance(spec)
    return {
        "records": [r.as_dict() for r in bundle.records],
        "kinds": bundle.kinds(),
        "digest": bundle.digest(),
        "missing_citations": bundle.missing_citations(),
        "fully_cited": bundle.is_fully_cited(),
    }


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index() -> Dict[str, Any]:
    return {e.dotted: e
            for e in capability_registry.find(package="standards")}


def _available(dotted: str) -> bool:
    return dotted in _index()


_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("carbon", "carbon_total", _STD + "embodied_carbon",
     "embodied-carbon (CO2e) accounting over a bill of materials; worst offenders"),
    ("provenance", "provenance", _STD + "evidence_bundle",
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
    return [(d, "no accounting route") for d in sorted(_index())
            if d not in routed and not d.endswith(".registry")]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every standards-accounting route (default)")
    parser.add_argument("--carbon", default=None, metavar="JSON",
                        help="a bill of materials (JSON list or @file) to tally")
    parser.add_argument("--provenance", default=None, metavar="JSON",
                        help="a design spec (JSON object or @file) to roll up")
    parser.add_argument("--top", type=int, default=10,
                        help="how many worst-offender materials to rank")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def _load(text: str) -> Any:
    if text.startswith("@"):
        with open(text[1:], "r", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(text)


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "carbon", None):
        uses = _load(args.carbon)
        total = carbon_total(uses)
        top = carbon_top(uses, n=getattr(args, "top", 10))
        if getattr(args, "json", False):
            print(json.dumps({"total_kg_co2e": total, "top": top}, indent=2))
        else:
            print("total kg CO2e: %.4f" % total)
            for name, value in top:
                print("  %-14s %.4f" % (name, value))
        return 0

    if getattr(args, "provenance", None):
        bundle = provenance(_load(args.provenance))
        print(json.dumps(bundle, indent=2, sort_keys=True))
        return 0 if bundle["fully_cited"] else 1

    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["route"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-11s %-*s  %s" % (mark, r["group"], width, r["route"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad standards-accounting",
        description="the design-accounting surface: embodied carbon + cited "
                    "provenance over a finished design")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
