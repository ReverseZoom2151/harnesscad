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

from backends.stub import StubBackend
from cisp.ops import _REGISTRY, parse_op
from loop import HarnessSession
from verify import VerifyReport


# --- backend selection -----------------------------------------------------
def _make_backend(name: str) -> Tuple[Any, str, Optional[str]]:
    """Return (backend, resolved_name, note). Falls back to stub with a note."""
    if name == "cadquery":
        try:
            from backends import cadquery_backend  # type: ignore
            return cadquery_backend.CadQueryBackend(), "cadquery", None
        except Exception as exc:  # pragma: no cover - depends on optional dep
            return (StubBackend(), "stub",
                    f"cadquery backend unavailable ({exc}); fell back to stub")
    return StubBackend(), "stub", None


class CISPMethodError(Exception):
    """Raised for malformed / unknown requests; surfaced as an error response."""


class CISPServer:
    """A session-scoped CISP endpoint.

    One server owns one HarnessSession (one model). Construct fresh per model;
    ``initialize`` is idempotent and simply reports capabilities.
    """

    METHODS = ("initialize", "applyOps", "query", "verify", "export")

    def __init__(self, backend: str = "stub") -> None:
        self.backend, self.backend_name, self.backend_note = _make_backend(backend)
        self.session = HarnessSession(self.backend)

    # --- CISP methods -----------------------------------------------------
    def initialize(self) -> Dict[str, Any]:
        return {
            "protocol": "cisp",
            "version": "0",
            "backend": self.backend_name,
            "note": self.backend_note,
            "capabilities": {
                "applyOps": True,
                "query": ["summary", "sketch_dof", "validity"],
                "verify": True,
                "export": ["step", "stl", "json"],
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
        diags = []
        for v in self.session.verifiers:
            diags += v.check(self.backend, self.session.opdag).diagnostics
        report = VerifyReport(diags)
        return {
            "ok": report.ok,
            "diagnostics": [d.to_dict() for d in report.diagnostics],
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
    parser.add_argument("--backend", default="stub", choices=["stub", "cadquery"])
    args = parser.parse_args(argv)
    CISPServer(backend=args.backend).serve_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
