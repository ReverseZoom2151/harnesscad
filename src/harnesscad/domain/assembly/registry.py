"""The assembly-checks surface -- dispatch into deterministic placement checks.

Assembly modules verify an *arrangement* of already-placed parts (do they clash,
by how much, which way to nudge). They were reachable from nothing; this module
is the one surface that dispatches into the package, mirroring the other
``domain/*/registry.py`` dispatchers so the capability index credits them as
reached rather than orphaned.

    check(boxes)                 -> InterferenceResult over labelled AABBs
    routed_modules()             -> the assembly modules this surface reaches

Deterministic, stdlib-only, no network.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "check",
    "routed_modules",
    "discover",
    "add_arguments",
    "run_cli",
    "main",
]

_ASM = "harnesscad.domain.assembly."


def check(boxes: Dict[str, Sequence[float]], **kwargs) -> Any:
    """Run AABB interference over ``{label: (xmin,ymin,zmin,xmax,ymax,zmax)}``.

    Keyword arguments are forwarded to
    :func:`~harnesscad.domain.assembly.interference.check_interference`.
    """
    from harnesscad.domain.assembly.interference import AABB, check_interference

    typed = {label: AABB(*coords) for label, coords in boxes.items()}
    return check_interference(typed, **kwargs)


_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("interference", "check", _ASM + "interference",
     "AABB interference detection with a minimum-clearance fix-vector suggestion"),
)


def _available(dotted: str) -> bool:
    try:
        capability_registry.get(dotted)
        return True
    except KeyError:
        return False


def routed_modules() -> Tuple[str, ...]:
    return tuple(sorted({m for _g, _n, m, _d in _ROUTES if _available(m)}))


def discover() -> List[dict]:
    return [{"group": g, "route": n, "module": m, "doc": d, "present": _available(m)}
            for (g, n, m, d) in _ROUTES]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every assembly-check route")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def run_cli(args: argparse.Namespace) -> int:
    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["route"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-13s %-*s  %s" % (mark, r["group"], width, r["route"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad assembly",
        description="deterministic checks over placed multi-part assemblies")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
