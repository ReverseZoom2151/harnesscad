"""Architectural layering guard: core/ and domain/ must not depend outward.

The package is being groomed so it can be split into separate distributions
(``harness-core`` / ``harnesscad`` / ``harnessbench``).  For that to work the
dependency arrows have to point INWARD toward the core: the ``core`` and
``domain`` layers may be imported BY the eval / surface layers, but must never
import them.  A module-top-level ``import harnesscad.eval.*`` or
``import harnesscad.io.surfaces.*`` inside ``core`` or ``domain`` is a hard
architectural edge -- it executes at import time and would raise ``ImportError``
the moment eval/bench (or the surface layer) is packaged separately, so it must
be caught in CI.

Escape hatch: a *function-local* (lazy) import is tolerated.  It does not run at
import time, so it does not block extraction, and it lets a verb reach an
optional sibling layer that may be absent (the sanctioned example is the
``bench`` verb in ``core/cli.py``, which now guards its lazy import so the CLI
degrades gracefully when eval/bench is not installed).  Only imports at MODULE
scope -- the direct module body plus any module-level ``try/if/with/for/while``
that runs at import -- are treated as violations; imports nested inside a
``def``/``class`` are ignored.

Enforcement: any top-level outward edge fails this test.  ``KNOWN_BASELINE``
below recorded the pre-existing edges as documented debt and was allowed only to
SHRINK; it has now reached zero, so this is a pure hard gate.  Removing an edge
from the code but leaving a baseline entry is harmless (it is reported, not
failed) so that concurrent cleanups never break this test.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

import harnesscad

PKG_ROOT = pathlib.Path(harnesscad.__file__).resolve().parent

#: Import prefixes that the inward layers (core/domain) must not reach at module
#: scope.  ``eval`` is the scoring/benchmark layer; ``io.surfaces`` is the
#: interaction-surface layer -- both sit OUTSIDE the core.
FORBIDDEN_PREFIXES = ("harnesscad.eval", "harnesscad.io.surfaces")

#: The inward direction is fine: eval/bench is expected to import core/domain.
INWARD_PREFIXES = ("harnesscad.domain", "harnesscad.core")

#: Pre-existing top-level outward edges, as ``"<relpath>::<module>"`` (relpath is
#: POSIX, relative to the harnesscad package root).  This was technical debt to be
#: driven to zero as each edge was relocated or made lazy by its owner; the set
#: must only ever shrink.
#:
#: IT IS NOW EMPTY, so this file is a PURE HARD GATE: any top-level import of
#: ``harnesscad.eval.*`` or ``harnesscad.io.surfaces.*`` from ``core`` or
#: ``domain`` fails, with no grandfathering left.  Do not add entries here.  The
#: eleven original edges were cleared as follows:
#:
#:   * the five ``verifiers.verify`` edges (cisp/protocol, contract, environment,
#:     harness, loop) -- Severity/Diagnostic/VerifyReport/Verifier are op-contract
#:     PROTOCOL types and moved to ``harnesscad.core.diagnostics``; verify.py
#:     re-exports them, so the old import path is unchanged;
#:   * ``chain_complex_nms`` -- the chamfer / sampled-curve distances are plain
#:     geometry and moved to
#:     ``harnesscad.domain.geometry.pointcloud.distance_metrics``, re-exported by
#:     ``eval.bench.geometry.complex_matching`` (same pattern as ``voxel_iou``);
#:   * the remaining five (cli, ingest_pipeline, pipeline, pointcloud_candidates,
#:     answer_engine) -- call-time-only, so the import became function-local.
KNOWN_BASELINE = frozenset()


def _matches(module: str, prefixes) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _module_level_imports(tree: ast.Module):
    """Yield ``(lineno, dotted_module)`` for every import that runs at module
    import time -- descending through module-level control flow but stopping at
    function and class boundaries (those are the lazy escape hatch)."""
    found = []

    def walk(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    found.append((node.lineno, node.module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    found.append((node.lineno, alias.name))
            elif isinstance(node, (ast.Try, ast.If, ast.With, ast.For, ast.While)):
                walk(getattr(node, "body", []))
                walk(getattr(node, "orelse", []))
                walk(getattr(node, "finalbody", []))
                for handler in getattr(node, "handlers", []):
                    walk(handler.body)

    walk(tree.body)
    return found


def _iter_py(*relparts):
    base = PKG_ROOT.joinpath(*relparts)
    return sorted(base.rglob("*.py")) if base.exists() else []


def _outward_violations(relparts):
    """All ``"<relpath>::<module>"`` outward edges under the given subtree."""
    out = set()
    for path in _iter_py(*relparts):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(PKG_ROOT).as_posix()
        for _lineno, module in _module_level_imports(tree):
            if _matches(module, FORBIDDEN_PREFIXES):
                out.add("{0}::{1}".format(rel, module))
    return out


class LayeringTest(unittest.TestCase):
    def test_core_and_domain_do_not_import_outward_at_module_top_level(self):
        violations = _outward_violations(("core",)) | _outward_violations(("domain",))
        new_edges = sorted(violations - KNOWN_BASELINE)
        self.assertEqual(
            new_edges,
            [],
            "New top-level outward dependency edge(s) from core/domain into "
            "eval/ or io.surfaces/ -- this breaks clean extraction. Make the "
            "import function-local (lazy) or move the dependency inward. "
            "Offenders (file::module): " + ", ".join(new_edges),
        )

    def test_baseline_is_shrink_only(self):
        # A baselined edge that has since been fixed is fine -- report it so the
        # baseline can be trimmed, but do not fail (keeps concurrent cleanups
        # green).  This test only guards against the baseline being padded with
        # entries that are not actually violations in the tree at all... which
        # would hide a typo; here we simply surface stale (already-fixed) ones.
        current = _outward_violations(("core",)) | _outward_violations(("domain",))
        stale = sorted(KNOWN_BASELINE - current)
        if stale:
            # Informational only.
            print(
                "\n[layering] baseline entries now clean (safe to remove): "
                + ", ".join(stale)
            )

    def test_inward_direction_is_allowed(self):
        # eval/bench importing core/domain is the CORRECT direction and must NOT
        # be flagged -- this guards against someone "fixing" the guard by banning
        # the wrong arrow.  Such imports exist and the forbidden-predicate leaves
        # them alone.
        inward = set()
        for path in _iter_py("eval", "bench"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for _lineno, module in _module_level_imports(tree):
                if _matches(module, INWARD_PREFIXES):
                    inward.add(module)
        self.assertTrue(
            inward,
            "expected harnesscad.eval.bench to import core/domain somewhere "
            "(the sanctioned inward direction)",
        )
        for module in inward:
            self.assertFalse(
                _matches(module, FORBIDDEN_PREFIXES),
                "inward import {0} must not be treated as a violation".format(module),
            )


if __name__ == "__main__":
    unittest.main()
