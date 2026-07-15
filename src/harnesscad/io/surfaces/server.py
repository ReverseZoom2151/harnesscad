"""CISP server — the LSP-inspired interface that exposes the harness over JSON.

CISP (CAD Interaction / Sketch Protocol) mirrors the request/response shape of
the Language Server Protocol: a small vocabulary of typed methods, each taking a
JSON object and returning a JSON object. The transport is line-delimited JSON
(one request per line, one response per line) so it drops cleanly into an MCP
server, a subprocess, or a stdio pipe.

Methods (see `CISPServer.handle`):
  - initialize()        -> capabilities + the op vocabulary
  - applyOps(ops)       -> parse each op dict, run the session, return the result
  - query(what)         -> read-only projection ('summary' / 'sketch_dof' / ...)
  - verify()            -> run the session's verifiers, return diagnostics
  - export(fmt)         -> serialise the current model

`handle(request)` is pure and unit-testable; `serve_stdio()` wraps it in a
read-line / write-line loop for tooling.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional, TextIO, Tuple

from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.cisp.ops import _REGISTRY, parse_op
from harnesscad.core.loop import HarnessSession
from harnesscad.eval.verifiers.registry import LINT, PHYSICS, DOMAIN
from harnesscad.eval.verifiers.verify import VerifyReport


# The `verify` method runs the whole advisory fleet (everything but CORE, which
# the session's own verifiers already cover).
FLEET_TIERS = (LINT, PHYSICS, DOMAIN)


# --- backend selection -----------------------------------------------------
#: Every backend the harness can drive. `stub` and `frep` always work (no
#: dependency at all); `cadquery` needs the CadQuery/OCCT wheel; `blender`,
#: `openscad` and `freecad` shell out to a real external kernel and are only
#: available when that binary is installed -- when it is not, they raise
#: BackendUnavailable and we fall back to the stub WITH A NOTE rather than
#: crashing. `cadquery` and `freecad` are the two real B-rep kernels (exact
#: volumes, STEP in and out); the rest are meshes or fields.
BACKENDS = ("stub", "cadquery", "build123d", "frep", "blender", "openscad",
            "freecad", "manifold", "rhino3dm", "microcad", "truck")


def _make_backend(name: str) -> Tuple[Any, str, Optional[str]]:
    """Return (backend, resolved_name, note). Falls back to stub with a note."""
    if name == "cadquery":
        try:
            from harnesscad.io.backends import cadquery as cadquery_backend  # type: ignore
            return cadquery_backend.CadQueryBackend(), "cadquery", None
        except Exception as exc:  # pragma: no cover - depends on optional dep
            return (StubBackend(), "stub",
                    f"cadquery backend unavailable ({exc}); fell back to stub")
    if name == "build123d":
        # The second OCCT B-rep front-end: build123d (algebra mode) over OCP.
        try:
            from harnesscad.io.backends import build123d as build123d_backend  # type: ignore
            return build123d_backend.Build123dBackend(), "build123d", None
        except Exception as exc:  # pragma: no cover - depends on optional dep
            return (StubBackend(), "stub",
                    f"build123d backend unavailable ({exc}); fell back to stub")
    if name == "frep":
        # Kernel-free SDF backend: real geometry, zero third-party dependencies.
        from harnesscad.io.backends.frep import FRepBackend
        return FRepBackend(), "frep", None
    if name == "blender":
        # Headless Blender: real mesh booleans (exact solver), bevel, solidify.
        from harnesscad.io.backends.base import BackendUnavailable
        from harnesscad.io.backends.blender import BlenderBackend
        try:
            return BlenderBackend(), "blender", None
        except BackendUnavailable as exc:
            return (StubBackend(), "stub",
                    f"blender backend unavailable ({exc}); fell back to stub")
    if name == "openscad":
        # The OpenSCAD binary: exact CGAL CSG (no grid, no sampling error).
        from harnesscad.io.backends.base import BackendUnavailable
        from harnesscad.io.backends.openscad import OpenScadBackend
        try:
            return OpenScadBackend(), "openscad", None
        except BackendUnavailable as exc:
            return (StubBackend(), "stub",
                    f"openscad backend unavailable ({exc}); fell back to stub")
    if name == "freecad":
        # Headless FreeCAD: real parametric B-rep (OCCT) + a feature tree.
        from harnesscad.io.backends.base import BackendUnavailable
        from harnesscad.io.backends.freecad import FreeCADBackend
        try:
            return FreeCADBackend(), "freecad", None
        except BackendUnavailable as exc:
            return (StubBackend(), "stub",
                    f"freecad backend unavailable ({exc}); fell back to stub")
    if name == "manifold":
        # The Manifold mesh-boolean kernel: an INDEPENDENT algorithm (guaranteed-
        # manifold booleans, in-process). Adding it makes the differential oracle
        # stronger -- a genuinely different kernel, not another OCCT/F-rep wrapper.
        from harnesscad.io.backends.base import BackendUnavailable
        from harnesscad.io.backends.manifold import ManifoldBackend
        try:
            return ManifoldBackend(), "manifold", None
        except BackendUnavailable as exc:
            return (StubBackend(), "stub",
                    f"manifold backend unavailable ({exc}); fell back to stub")
    if name == "rhino3dm":
        # openNURBS via the standalone rhino3dm wheel: an INDEPENDENT geometry +
        # IO voice. It builds only primitives/extrusions and REFUSES every other
        # op with a typed unsupported-op, but the volume/bbox it reports for those
        # is an independent oracle voice, and it converts to/from .3dm.
        from harnesscad.io.backends.base import BackendUnavailable
        from harnesscad.io.backends.rhino3dm import Rhino3dmBackend
        try:
            return Rhino3dmBackend(), "rhino3dm", None
        except BackendUnavailable as exc:
            return (StubBackend(), "stub",
                    f"rhino3dm backend unavailable ({exc}); fell back to stub")
    if name == "microcad":
        # The microcad (µcad) CLI: a NEW declarative CAD *language* (Rust, v0.5.0
        # alpha). Integrates like OpenSCAD -- emit source, shell out, read the STL
        # back. Absent unless `cargo install microcad` produced a runnable binary.
        from harnesscad.io.backends.base import BackendUnavailable
        from harnesscad.io.backends.microcad import MicrocadBackend
        try:
            return MicrocadBackend(), "microcad", None
        except BackendUnavailable as exc:
            return (StubBackend(), "stub",
                    f"microcad backend unavailable ({exc}); fell back to stub")
    if name == "truck":
        # The truck B-rep NURBS kernel (Rust): a genuinely INDEPENDENT B-rep
        # lineage -- NOT OCCT, unlike cadquery/freecad/build123d, which all share
        # it. As the only non-OCCT B-rep voice it is the strongest addition to the
        # differential oracle. Shells out to a compiled Rust driver; absent unless
        # `cargo build --release` produced the binary.
        from harnesscad.io.backends.base import BackendUnavailable
        from harnesscad.io.backends.truck import TruckBackend
        try:
            return TruckBackend(), "truck", None
        except BackendUnavailable as exc:
            return (StubBackend(), "stub",
                    f"truck backend unavailable ({exc}); fell back to stub")
    return StubBackend(), "stub", None


class CISPMethodError(Exception):
    """Raised for malformed / unknown requests; surfaced as an error response."""


class CISPServer:
    """A session-scoped CISP endpoint.

    One server owns one HarnessSession (one model). Construct fresh per model;
    ``initialize`` is idempotent and simply reports capabilities.
    """

    METHODS = ("initialize", "applyOps", "query", "verify", "export")

    def __init__(self, backend: str = "stub", verify_level: str = "core") -> None:
        self.backend, self.backend_name, self.backend_note = _make_backend(backend)
        self.session = HarnessSession(self.backend, verify_level=verify_level)

    # --- CISP methods -----------------------------------------------------
    def initialize(self) -> Dict[str, Any]:
        return {
            "protocol": "cisp",
            "version": "0",
            "backend": self.backend_name,
            "note": self.backend_note,
            "capabilities": {
                "applyOps": True,
                "query": ["summary", "sketch_dof", "validity", "measure", "metrics"],
                "verify": True,
                "export": ["step", "stl", "iges", "json"],
                "transactional": True,   # block-and-correct + rollback
                "deterministic": True,   # replay -> identical digest
            },
            "ops": sorted(_REGISTRY.keys()),
        }

    def applyOps(self, ops: List[dict]) -> Dict[str, Any]:
        if not isinstance(ops, list):
            raise CISPMethodError("applyOps: 'ops' must be a list of op dicts")
        parsed = []
        for i, raw in enumerate(ops):
            if not isinstance(raw, dict):
                raise CISPMethodError(f"applyOps: op[{i}] is not an object")
            try:
                parsed.append(parse_op(raw))
            except (KeyError, TypeError) as exc:
                raise CISPMethodError(f"applyOps: op[{i}] is invalid: {exc}")
        result = self.session.apply_ops(parsed)
        return result.to_dict()

    def query(self, what: str) -> Dict[str, Any]:
        if not isinstance(what, str):
            raise CISPMethodError("query: 'what' must be a string")
        return {"what": what, "result": self.backend.query(what)}

    def verify(self) -> Dict[str, Any]:
        """Core verifiers + the full discovered fleet.

        ``ok`` is still decided by the CORE verifiers alone -- they are what gate
        the transaction. The fleet's findings are surfaced under ``fleet`` (and
        appended to ``diagnostics``) so an agent can act on them without the
        advisory tiers flipping a valid model to not-ok.
        """
        diags = []
        for v in self.session.verifiers:
            diags += v.check(self.backend, self.session.opdag).diagnostics
        report = VerifyReport(diags)
        fleet = self.session.run_fleet(tiers=FLEET_TIERS)
        return {
            "ok": report.ok,
            "diagnostics": [d.to_dict() for d in report.diagnostics + fleet],
            "fleet": [d.to_dict() for d in fleet],
        }

    def export(self, fmt: str = "step") -> Dict[str, Any]:
        if not isinstance(fmt, str):
            raise CISPMethodError("export: 'fmt' must be a string")
        return {"fmt": fmt, "content": self.backend.export(fmt)}

    # --- dispatch ---------------------------------------------------------
    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Pure request -> response. Never raises; errors become {ok, error}."""
        req_id = request.get("id") if isinstance(request, dict) else None
        try:
            if not isinstance(request, dict):
                raise CISPMethodError("request must be a JSON object")
            method = request.get("method")
            if method not in self.METHODS:
                raise CISPMethodError(f"unknown method '{method}'")
            params = request.get("params") or {}
            if not isinstance(params, dict):
                raise CISPMethodError("'params' must be an object")

            if method == "initialize":
                result = self.initialize()
            elif method == "applyOps":
                result = self.applyOps(params.get("ops", []))
            elif method == "query":
                result = self.query(params.get("what", "summary"))
            elif method == "verify":
                result = self.verify()
            elif method == "export":
                result = self.export(params.get("fmt", "step"))
            else:  # pragma: no cover - guarded above
                raise CISPMethodError(f"unhandled method '{method}'")

            return {"id": req_id, "ok": True, "result": result}
        except CISPMethodError as exc:
            return {"id": req_id, "ok": False,
                    "error": {"code": "bad-request", "message": str(exc)}}
        except Exception as exc:  # pragma: no cover - defensive
            return {"id": req_id, "ok": False,
                    "error": {"code": "internal", "message": str(exc)}}

    # --- transport --------------------------------------------------------
    def serve_stdio(self, stdin: TextIO = None, stdout: TextIO = None) -> None:
        """Read one JSON request per line; write one JSON response per line."""
        stdin = stdin if stdin is not None else sys.stdin
        stdout = stdout if stdout is not None else sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                response = {"id": None, "ok": False,
                            "error": {"code": "parse-error", "message": str(exc)}}
            else:
                response = self.handle(request)
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="CISP stdio server")
    parser.add_argument("--backend", default="stub", choices=list(BACKENDS))
    args = parser.parse_args(argv)
    CISPServer(backend=args.backend).serve_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
