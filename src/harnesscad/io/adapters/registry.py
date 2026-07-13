"""The ECOSYSTEM surface -- which system, which backend, which bridge, which kernel.

``io/adapters`` and ``io/backends`` carried a code-CAD ecosystem catalogue, a
language registry, a requirement-driven backend SELECTOR, a system-to-system
interoperability matrix, a FreeCAD workbench operation catalogue, an OCCT kernel
API inventory, a render-request canonicaliser, a render-payload codec, an
in-memory reference adapter and two host DTO boundaries (Rhino, Zoo). Nothing
dispatched into any of it.

This module is that dispatcher. Every route answers a question of the same shape:
*given what I need, what can actually do it, and how do I get there?*

    select(needs)            -> the ranked backends that meet a requirement
    bridge("cadquery", "freecad") -> can these two hand off, and through what
    freecad_ops("Part")      -> the operations a workbench really exposes
    occt()                   -> the OCCT classes/methods that really exist
    render_key(lang, src)    -> the canonical cache key for a render request

THE HOSTS ARE DTO BOUNDARIES, NOT CLIENTS. :func:`zoo_request` builds the HTTP
request for the Zoo text-to-CAD API and :func:`parse_zoo` reads a response --
it does not send anything. Same for Rhino: :func:`rhino_check` validates a
script against declared host capabilities; executing it needs a Rhino that is
not here. Deterministic, stdlib-only, NO NETWORK: a module that cannot be
exercised without a proprietary host stops at its own boundary.

Adapters only: the adapter/backend modules are never modified.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "EcosystemError",
    "systems",
    "system",
    "languages",
    "select",
    "explain",
    "coverage",
    "bridge",
    "handoff_path",
    "freecad_ops",
    "freecad_check",
    "occt",
    "memory_adapter",
    "rhino_check",
    "zoo_formats",
    "zoo_request",
    "parse_zoo",
    "render_key",
    "render_payload",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

_ADP = "harnesscad.io.adapters."
_BCK = "harnesscad.io.backends."


class EcosystemError(ValueError):
    """Base class for every ecosystem-surface failure."""


# --------------------------------------------------------------------------- #
# The catalogue: what exists
# --------------------------------------------------------------------------- #
def systems() -> List[str]:
    """Every code-CAD system in the catalogue."""
    from harnesscad.io.adapters.ecosystem_catalog import system_names

    return list(system_names())


def system(name: str) -> dict:
    """One system's honest spec: paradigm, kernel, host language, formats."""
    from harnesscad.io.adapters.ecosystem_catalog import get, unknown_attributes

    spec = get(name)
    return {
        "name": spec.name,
        "row": spec.as_row(),
        "is_language": bool(spec.is_language),
        "is_library": bool(spec.is_library),
        "kernel_free": bool(spec.kernel_free),
        "formats_out": list(spec.formats_out),
        "formats_in": list(spec.formats_in),
        "paradigms": list(spec.paradigms),
        "kernel": spec.kernel,
        "unknown": list(unknown_attributes(name)),
    }


def languages() -> List[dict]:
    """The code-CAD LANGUAGES (a subset of the systems) and what they can emit."""
    from harnesscad.io.adapters.language_registry import get, language_names

    return [{"name": n, "row": get(n).as_row()} for n in language_names()]


# --------------------------------------------------------------------------- #
# Selection: what can meet my requirement
# --------------------------------------------------------------------------- #
def select(limit: int = 5, **needs) -> List[dict]:
    """Rank the systems that meet a stated requirement. Ranked, never a single pick.

    ``needs`` are :class:`Requirement` fields (paradigm, kernel, host language,
    required export formats, ...). A system that fails a HARD need is excluded,
    not down-weighted -- 'nearly supports STEP' is not a thing.
    """
    from harnesscad.io.adapters.backend_selector import Requirement, rank

    req = Requirement(**needs)
    return [{"name": c.name, "score": c.score, "support": c.support,
             "reasons": list(c.reasons)}
            for c in rank(req, limit=int(limit))]


def explain(name: str, **needs) -> Any:
    """Why did this system score the way it did against this requirement?"""
    from harnesscad.io.adapters.backend_selector import Requirement, explain as _explain

    return _explain(name, Requirement(**needs))


def coverage(supported: Sequence[str]) -> dict:
    """What the harness's SUPPORTED backends cover, and where the gaps are."""
    from harnesscad.io.adapters.backend_selector import (
        coverage_report, gap_recommendations,
    )

    report = coverage_report(list(supported))
    row = report.as_row()
    return {"row": row if isinstance(row, dict) else list(row),
            "missing_paradigms": list(report.missing_paradigms),
            "missing_kernels": list(report.missing_kernels),
            "missing_formats_out": list(report.missing_formats_out),
            "recommendations": list(gap_recommendations(list(supported)))}


# --------------------------------------------------------------------------- #
# Interop: how do two systems hand off
# --------------------------------------------------------------------------- #
def bridge(src: str, dst: str) -> dict:
    """Can ``src`` hand a model to ``dst``, and through which interchange format?"""
    from harnesscad.io.adapters.interop_matrix import (
        can_handoff, interchange_format, shared_formats,
    )

    return {
        "src": src, "dst": dst,
        "can_handoff": bool(can_handoff(src, dst)),
        "interchange": interchange_format(src, dst),
        "shared_formats": list(shared_formats(src, dst)),
    }


def handoff_path(src: str, dst: str,
                 kinds: Optional[Sequence[str]] = None) -> List[dict]:
    """The shortest chain of BRIDGES that gets a model from ``src`` to ``dst``.

    ``[]`` means there is no path -- not "try harder".
    """
    from harnesscad.io.adapters.interop_matrix import handoff_path as _path

    path = _path(src, dst, kinds=tuple(kinds) if kinds else None)
    return [b.as_row() for b in (path or [])]


# --------------------------------------------------------------------------- #
# Host API catalogues: the operations that REALLY exist
# --------------------------------------------------------------------------- #
def freecad_ops(workbench: Optional[str] = None) -> List[str]:
    """The FreeCAD operations a workbench really exposes (not what an LLM guessed)."""
    from harnesscad.io.adapters.freecad_catalog import default_catalog

    cat = default_catalog()
    if workbench:
        return sorted(op.name for op in cat.by_workbench(workbench))
    return sorted(cat.names())


def freecad_check(operation: str, arguments: Mapping[str, Any]) -> dict:
    """Would this FreeCAD call actually type-check against the catalogue?"""
    from harnesscad.io.adapters.freecad_catalog import default_catalog

    check = default_catalog().check_call(operation, dict(arguments))
    return {"ok": bool(check.ok), "operation": check.operation,
            "errors": list(check.errors), "warnings": list(check.warnings),
            "suggestion": check.suggestion}


def occt(headers_root: Optional[str] = None, modules: Optional[Sequence[str]] = None):
    """The OCCT kernel API inventory. Empty without headers -- it does not invent one."""
    from harnesscad.io.backends.occt_catalog import (
        OcctApiCatalog, build_catalog_from_headers,
    )

    if headers_root is None:
        return OcctApiCatalog()
    return build_catalog_from_headers(headers_root,
                                      modules=tuple(modules) if modules else None)


# --------------------------------------------------------------------------- #
# The adapter contract + the host DTO boundaries
# --------------------------------------------------------------------------- #
def memory_adapter():
    """The in-memory reference implementation of the CAD adapter contract.

    Transactional (begin / apply / verify / commit / rollback), idempotent, and
    revision-tracked -- the executable specification every real adapter must
    satisfy.
    """
    from harnesscad.io.adapters.memory import MemoryCADAdapter

    return MemoryCADAdapter()


def rhino_check(source: str, commands: Sequence[str],
                allowed_commands: Sequence[str],
                target: str = "rhinoscript", script_id: str = "s1",
                allowed_analyses: Sequence[str] = ()) -> List[str]:
    """Validate a Rhino/Grasshopper script against DECLARED host capabilities.

    Returns the issues (unsupported target, denied commands, empty source);
    ``[]`` means the script is admissible. NOTHING IS EXECUTED -- the Rhino host
    is optional and is not here, so this surface stops at the DTO boundary.
    """
    from harnesscad.io.adapters.rhino import (
        HostCapabilities, HostScript, validate_script,
    )

    caps = HostCapabilities(commands=frozenset(allowed_commands),
                            analyses=frozenset(allowed_analyses))
    script = HostScript(id=script_id, target=target, source=source,
                        commands=tuple(commands))
    return list(validate_script(script, caps))


def zoo_formats() -> List[str]:
    """The output formats the Zoo API actually offers (all MESH -- no B-rep)."""
    from harnesscad.io.adapters.zoo_api import OutputFormat

    return sorted(f.value for f in OutputFormat)


def zoo_request(prompt: str, output_format: str = "step",
                api_token: str = "") -> dict:
    """BUILD (never send) the Zoo / KittyCAD text-to-CAD submit request.

    An unsupported ``output_format`` raises ``ZooApiError``: the API's mesh-only
    format list is not something to fudge. See :func:`zoo_formats`.
    """
    from harnesscad.io.adapters.zoo_api import build_submit_request

    req = build_submit_request(prompt, output_format, api_token)
    return {"method": req.method, "url": req.url,
            "headers": dict(req.headers), "body": req.body}


def parse_zoo(response: Mapping[str, Any]) -> dict:
    """Read a Zoo operation response (which somebody else fetched)."""
    from harnesscad.io.adapters.zoo_api import parse_operation

    op = parse_operation(dict(response))
    return {"id": op.id, "status": op.status.value,
            "terminal": bool(op.is_terminal),
            "completed": bool(op.is_completed),
            "failed": bool(op.is_failed),
            "error": op.error}


# --------------------------------------------------------------------------- #
# Render service boundary
# --------------------------------------------------------------------------- #
def render_key(language: str, source: str) -> str:
    """The canonical cache key for a render request. Same model -> same key."""
    from harnesscad.io.backends.render_request import request_key

    return request_key(language, source)


def render_payload(artifact: bytes, metadata: Mapping[str, Any]) -> dict:
    """Encode + decode the single-response artifact/metadata payload. Round-trips."""
    from harnesscad.io.backends.render_payload import decode, encode, roundtrip

    blob = encode(artifact, dict(metadata))
    payload = decode(blob)
    return {
        "bytes": len(blob),
        "artifact_type": payload.artifact_type,
        "is_mesh": bool(payload.is_mesh),
        "roundtrips": roundtrip(artifact, dict(metadata)),
    }


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index() -> Dict[str, Any]:
    out = {}
    for pkg in ("adapters", "backends"):
        for e in capability_registry.find(package=pkg):
            out[e.dotted] = e
    return out


def _available(dotted: str) -> bool:
    return dotted in _index()


_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("catalog", "systems", _ADP + "ecosystem_catalog",
     "every code-CAD system: paradigm, kernel, host language, formats"),
    ("catalog", "languages", _ADP + "language_registry",
     "the code-CAD languages and their capability matrix"),
    ("select", "select", _ADP + "backend_selector",
     "rank the backends that meet a stated requirement; coverage + gaps"),
    ("interop", "bridge", _ADP + "interop_matrix",
     "can two systems hand off, through which format, along which path"),
    ("host", "freecad_ops", _ADP + "freecad_catalog",
     "the FreeCAD operations that really exist, and whether a call type-checks"),
    ("host", "rhino_check", _ADP + "rhino",
     "validate a Rhino script against declared host capabilities (never executes)"),
    ("host", "zoo_request", _ADP + "zoo_api",
     "build/parse the Zoo text-to-CAD request and response (never sends)"),
    ("contract", "memory_adapter", _ADP + "memory",
     "the in-memory reference implementation of the CAD adapter contract"),
    ("kernel", "occt", _BCK + "occt_catalog",
     "the OCCT class/method inventory (empty without headers -- never invented)"),
    ("render", "render_key", _BCK + "render_request",
     "canonical render request + cache key"),
    ("render", "render_payload", _BCK + "render_payload",
     "the artifact+metadata payload codec (round-trips)"),
)


def routed_modules() -> Tuple[str, ...]:
    return tuple(sorted({m for _g, _n, m, _d in _ROUTES if _available(m)}))


UNADAPTED_REASONS: Dict[str, str] = {
    _ADP + "base": "the adapter CONTRACT itself (protocols/DTOs); implementations "
                   "import it, a dispatcher does not call it",
    _BCK + "base": "the backend protocol itself; the backends implement it",
    _BCK + "stub": "a GeometryBackend, constructed directly by the loop/CLI",
    _BCK + "cadquery": "a GeometryBackend, constructed directly by the loop/CLI",
    _BCK + "frep": "a GeometryBackend, constructed directly by the loop/CLI",
    _BCK + "frep_ir": "the F-Rep compiler, used by the frep backend",
}


def discover() -> List[dict]:
    return [{"group": g, "route": n, "module": m, "doc": d,
             "present": _available(m)}
            for (g, n, m, d) in _ROUTES]


def unadapted() -> List[Tuple[str, str]]:
    routed = set(routed_modules())
    return [(d, UNADAPTED_REASONS.get(d, "no route yet")) for d in sorted(_index())
            if d not in routed and not d.endswith(".registry")]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every ecosystem route")
    parser.add_argument("--systems", action="store_true",
                        help="list every code-CAD system in the catalogue")
    parser.add_argument("--system", default=None,
                        help="show one system's honest spec")
    parser.add_argument("--select", default=None, metavar="JSON",
                        help='rank backends against a requirement, e.g. \'{"paradigm": "csg"}\'')
    parser.add_argument("--bridge", default=None, metavar="SRC,DST",
                        help="can these two systems hand off?")
    parser.add_argument("--freecad", default=None, metavar="WORKBENCH",
                        help="list a FreeCAD workbench's real operations")
    parser.add_argument("--unadapted", action="store_true",
                        help="list adapter/backend modules with no route, and why")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "unadapted", False):
        for dotted, reason in unadapted():
            print("%s\n    %s" % (dotted, reason))
        return 0

    if getattr(args, "systems", False):
        for s in systems():
            print(s)
        return 0

    if getattr(args, "system", None):
        print(json.dumps(system(args.system), indent=2, sort_keys=True, default=repr))
        return 0

    if getattr(args, "select", None):
        needs = json.loads(args.select)
        print(json.dumps(select(**needs), indent=2, sort_keys=True, default=repr))
        return 0

    if getattr(args, "bridge", None):
        src, _, dst = args.bridge.partition(",")
        print(json.dumps(bridge(src.strip(), dst.strip()), indent=2, sort_keys=True))
        return 0

    if getattr(args, "freecad", None):
        for op in freecad_ops(args.freecad):
            print(op)
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
        prog="harnesscad ecosystem",
        description="ecosystem surface: which system, which backend, which bridge, "
                    "which kernel")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
