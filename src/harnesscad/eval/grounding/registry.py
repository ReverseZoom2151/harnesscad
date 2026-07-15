"""The GROUNDING surface -- how a CAD agent turns "the Pad button" into a click.

``eval/grounding`` carried the CAD-viewport grounding stack: the ACU field map
(the benchmarks and grounding models the field HAS, and the CAD-viewport blind
spot none of them fill), the model-free Set-of-Marks numbered element list, the
self-labelling click corpus and the ScreenSpot-style CADSpot benchmark, and the
verified-trajectory corpus. The two library halves (the field map, the marks
list) were reachable only by their tests. This module is the dispatcher: one
surface over the whole grounding family.

    field_map()                       -> the ACU field map + the CAD blind spot
    benchmarks(domain)                -> the reference benchmark table, filtered
    marks(boxes)                      -> a screenshot -> a numbered click table + id2xy
    marks_from_elements(elements)     -> the same, from an accessibility tree we own
    click_corpus(outdir, ...)         -> generate the self-labelling click corpus
    benchmark(outdir, ...)            -> build the CADSpot grounding benchmark
    trajectory_corpus(outdir, ...)    -> generate the verified-trajectory corpus

Adapters only: the grounding modules are never modified and are imported LAZILY
inside their routes (``corpus`` / ``cadspot`` pull FreeCAD or the CUA extras and
degrade to a clean "unavailable" when those are absent). Deterministic ids;
nothing here needs a vision model in the loop.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "GroundingError",
    "field_map",
    "field_counts",
    "benchmarks",
    "marks",
    "marks_from_elements",
    "click_corpus",
    "benchmark",
    "trajectory_corpus",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

_GND = "harnesscad.eval.grounding."


class GroundingError(ValueError):
    """Base class for every grounding-surface failure."""


# --------------------------------------------------------------------------- #
# The ACU field map (catalogue)
# --------------------------------------------------------------------------- #
def field_map() -> dict:
    """The field's blind spot as data: which benchmarks/grounders cannot transfer
    to an a11y-less 3D CAD viewport, plus the reference counts."""
    from harnesscad.eval.grounding import catalogue

    report = dict(catalogue.cad_gap())
    report["counts"] = catalogue.counts()
    return report


def field_counts() -> dict:
    """Counts of catalogued benchmarks / grounding models / datasets / safety papers."""
    from harnesscad.eval.grounding import catalogue

    return catalogue.counts()


def benchmarks(domain: Optional[str] = None) -> List[dict]:
    """The reference benchmark table, optionally filtered to one domain."""
    from harnesscad.eval.grounding import catalogue

    rows = (catalogue.benchmarks_for(domain) if domain
            else list(catalogue.BENCHMARKS))
    return [{"name": b.name, "year": b.year, "domain": b.domain, "note": b.note,
             "has_accessibility_tree": b.has_accessibility_tree} for b in rows]


# --------------------------------------------------------------------------- #
# The Set-of-Marks numbered element list (som)
# --------------------------------------------------------------------------- #
def marks(boxes: Sequence[dict], *, source: str = "") -> dict:
    """Number a list of ``{"bbox"|"rect", "label", "kind"}`` boxes into a mark list.

    Returns the SetOfMarks dict: the tool description, the numbered elements, and
    the ``id2xy`` lookup a driving loop carries. Ids are a function of the screen
    (stable reading order), not of the caller's input order.
    """
    from harnesscad.eval.grounding.som import SetOfMarks

    return SetOfMarks.from_boxes(list(boxes), source=source).to_dict()


def marks_from_elements(elements: Sequence[Any], *,
                        clickable_only: bool = True) -> dict:
    """The model-free path: number an accessibility tree (uia Element rects) we own."""
    from harnesscad.eval.grounding.som import SetOfMarks

    return SetOfMarks.from_elements(list(elements),
                                    clickable_only=clickable_only).to_dict()


# --------------------------------------------------------------------------- #
# The benchmark / corpus builders (corpus, cadspot, trajectory_corpus)
# --------------------------------------------------------------------------- #
def click_corpus(outdir: str, count: int = 4, seed: int = 0, **kwargs):
    """Generate the self-labelling (screenshot, description, point, verified) corpus."""
    from harnesscad.eval.grounding import corpus

    return corpus.generate(outdir, count=int(count), seed=int(seed), **kwargs)


def benchmark(outdir: str, count: int = 6, seed: int = 0, **kwargs):
    """Build the ScreenSpot-style CADSpot grounding benchmark (split by region)."""
    from harnesscad.eval.grounding import cadspot

    return cadspot.build(outdir, count=int(count), seed=int(seed), **kwargs)


def trajectory_corpus(outdir: str, count: int = 4, seed: int = 0, **kwargs):
    """Generate the verified-trajectory grounding corpus."""
    from harnesscad.eval.grounding import trajectory_corpus as _tc

    return _tc.generate(outdir, count=int(count), seed=int(seed), **kwargs)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index():
    return {e.dotted: e
            for e in capability_registry.find(layer="eval", package="grounding")}


def _available(dotted: str) -> bool:
    return dotted in _index()


# (group, route, module, doc) -- the dispatch table for the grounding family.
_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("field", "field_map", _GND + "catalogue",
     "the ACU field map + the a11y-less CAD-viewport blind spot, as data"),
    ("marks", "marks", _GND + "som",
     "a screenshot -> a numbered click table + id2xy, model-free"),
    ("corpus", "click_corpus", _GND + "corpus",
     "the self-labelling (screenshot, description, point, verified) click corpus"),
    ("benchmark", "benchmark", _GND + "cadspot",
     "the ScreenSpot-style CADSpot grounding benchmark, split by region"),
    ("corpus", "trajectory_corpus", _GND + "trajectory_corpus",
     "the verified-trajectory grounding corpus"),
)


def routed_modules() -> Tuple[str, ...]:
    return tuple(sorted({m for _g, _n, m, _d in _ROUTES if _available(m)}))


def discover() -> List[dict]:
    return [{"group": g, "route": n, "module": m, "doc": d,
             "present": _available(m)}
            for (g, n, m, d) in _ROUTES]


def unadapted() -> List[Tuple[str, str]]:
    routed = set(routed_modules())
    return [(d, "no route yet") for d in sorted(_index())
            if d not in routed and not d.endswith(".registry")]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every grounding route")
    parser.add_argument("--field-map", action="store_true", dest="field_map",
                        help="print the ACU field map + the CAD blind spot")
    parser.add_argument("--benchmarks", default=None, metavar="DOMAIN",
                        help="print the benchmark table (optionally one domain)")
    parser.add_argument("--marks", default=None, metavar="JSON",
                        help="a box list (JSON or @file) -> the numbered mark table")
    parser.add_argument("--unadapted", action="store_true",
                        help="list grounding modules with no route")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def _load(text: str) -> Any:
    if text.startswith("@"):
        with open(text[1:], "r", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(text)


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "unadapted", False):
        for dotted, reason in unadapted():
            print("%s\n    %s" % (dotted, reason))
        return 0

    if getattr(args, "field_map", False):
        print(json.dumps(field_map(), indent=2, sort_keys=True))
        return 0

    if getattr(args, "benchmarks", None) is not None:
        domain = args.benchmarks or None
        rows = benchmarks(domain)
        if getattr(args, "json", False):
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            for b in rows:
                print("%-22s %-6s %-12s %s" % (b["name"], b["year"], b["domain"],
                                               b["note"]))
        return 0

    if getattr(args, "marks", None):
        result = marks(_load(args.marks))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["route"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-10s %-*s  %s" % (mark, r["group"], width, r["route"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad grounding",
        description="grounding surface: the ACU field map, the Set-of-Marks "
                    "element list, and the click / CADSpot / trajectory corpora")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
