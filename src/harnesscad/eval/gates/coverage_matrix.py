"""THE COVERAGE-MATRIX CENSUS + DRIFT PAUSE-GATE (the Kitchen Loop's coverage matrix).

The field-liveness oracle asks a per-CELL question: does THIS (op, field) actually
move THIS backend's geometry? This gate asks the WHOLE-MATRIX question one rung up:
for every CISP op, crossed with every (backend, format) the harness can emit, is that
triple

    * IMPLEMENTED -- the op lowers to that backend and reaches that codec; or
    * REFUSED     -- the backend declares a typed ``unsupported-op`` for it (a KNOWN,
                     HONEST gap: the harness will refuse rather than fake it); or
    * UNKNOWN     -- nobody can say, statically, whether it works. THE DANGEROUS
                     EMPTY CELL. An untested claim reads, in every summary that
                     counts filled cells, as coverage -- and it is the Kitchen Loop's
                     Backtest Service Gap in miniature: 38 green unit tests and a
                     feature nobody ever ran.

REFUSED IS FINE. A refusal is an honest, explicit boundary the harness enforces with a
typed diagnostic; it is coverage, not debt. Only UNKNOWN cells are coverage debt,
because an UNKNOWN cell is a claim the harness has never once checked either way.

HOW A STATUS IS DERIVED -- STATICALLY, NO KERNEL, NO MODEL
---------------------------------------------------------
Every status here is read off DECLARED metadata, never off a running kernel:

* the row axis is the CISP op vocabulary -- ``core.cisp.ops._REGISTRY`` (its op tags);
* the column axis is enumerated from the capability registry
  (``harnesscad.registry.find(package="backends")``): each backend module is imported
  (its kernel is NOT -- the tool is located lazily), and its DECLARED
  ``FORMATS`` tuple gives the format half of each column;
* a backend that declares an ``UNSUPPORTED`` op-set (the ``ExternalToolBackend``
  family: manifold, truck, freecad, openscad, blender, microcad) has a KNOWN support
  surface -- an op in that set is REFUSED, any other op is IMPLEMENTED;
* a backend that declares NO ``UNSUPPORTED`` surface (cadquery, build123d, frep, stub,
  onshape, rhino3dm -- they refuse inline, mid-apply, where no static reader can see
  it) yields UNKNOWN for every op: the honest verdict is "never exercised here".

THE CAP, STATED LOUDLY (silent truncation reads as full coverage)
-----------------------------------------------------------------
* IMPLEMENTED here means the op is NOT declared unsupported and the format IS in the
  backend's declared codec set. It is a STATIC claim, not an execution proof: a
  backend can still refuse an op mid-apply that its class never listed. This gate
  surfaces the empty cells; it does not run the kernels that would fill them.
* When only a subset of columns is censused (``sample=``), the columns left out are
  recorded in :attr:`CoverageReport.skipped` and printed -- never dropped in silence.

THE GATE (a DRIFT pause-gate, the harness's orphan-ceiling discipline)
----------------------------------------------------------------------
``check`` reads a small JSON baseline sidecar (``--baseline``). It FAILS the build when
the live matrix is WORSE than the baseline:

    * the fill rate DROPPED below the baseline (drift down), or
    * the UNKNOWN-cell count GREW.

If the baseline is absent it is ESTABLISHED (written) and the gate PASSES -- the first
run records where we stand; every run after that must be at least as good. This is the
Kitchen Loop's "is the system at least as good as before?" oracle, and it is the same
ratchet the liveness floor and orphan ceiling already run.

stdlib-only, deterministic (no wall clock, no randomness).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry
from harnesscad.core.cisp.ops import _REGISTRY as OP_REGISTRY

__all__ = [
    "IMPLEMENTED",
    "REFUSED",
    "UNKNOWN",
    "STATUSES",
    "DRIFT_EPS",
    "BackendSpec",
    "Cell",
    "CoverageReport",
    "DriftReport",
    "op_vocabulary",
    "codec_formats",
    "discover_backends",
    "classify",
    "census",
    "load_baseline",
    "write_baseline",
    "check",
    "claims",
    "format_text",
    "selfcheck",
    "add_arguments",
    "run",
    "main",
]

#: The three cell states. IMPLEMENTED and REFUSED are both "filled" (we can say what
#: happens); UNKNOWN is the empty cell -- the only one that counts as coverage debt.
IMPLEMENTED = "IMPLEMENTED"
REFUSED = "REFUSED"
UNKNOWN = "UNKNOWN"
STATUSES: Tuple[str, ...] = (IMPLEMENTED, REFUSED, UNKNOWN)

#: A fill rate may not drop by more than this below the baseline before the gate
#: fails. Tiny, only to absorb float noise -- a real regression is far larger.
DRIFT_EPS = 1e-9


# ---------------------------------------------------------------------------
# The static column model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BackendSpec:
    """One backend's DECLARED capability surface. Purely static -- no kernel ran.

    ``unsupported`` is ``None`` when the backend declares NO op-support surface at all
    (it refuses inline, invisibly to a static reader), which makes every op UNKNOWN.
    An empty dict is a POSITIVE claim ("this backend refuses nothing"), which is very
    different from ``None`` ("this backend says nothing").
    """

    name: str
    formats: Tuple[str, ...]
    unsupported: Optional[Dict[str, str]] = None
    primitive_shapes: Optional[Tuple[str, ...]] = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "formats": list(self.formats),
            "declares_support_surface": self.unsupported is not None,
            "unsupported": dict(sorted((self.unsupported or {}).items())),
            "primitive_shapes": (list(self.primitive_shapes)
                                 if self.primitive_shapes is not None else None),
            "note": self.note,
        }


@dataclass(frozen=True)
class Cell:
    """One (op, backend, format) triple and the static verdict on it."""

    op: str
    backend: str
    fmt: str
    status: str
    reason: str

    @property
    def key(self) -> str:
        return "%s|%s|%s" % (self.backend, self.fmt, self.op)

    def to_dict(self) -> dict:
        return {"op": self.op, "backend": self.backend, "fmt": self.fmt,
                "status": self.status, "reason": self.reason}


@dataclass
class CoverageReport:
    """The whole op x (backend, format) matrix and its fill statistics."""

    ops: Tuple[str, ...] = ()
    columns: Tuple[Tuple[str, str], ...] = ()
    cells: Tuple[Cell, ...] = ()
    codec_backed: Dict[str, bool] = field(default_factory=dict)
    skipped: Tuple[str, ...] = ()

    def counts(self) -> Dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for c in self.cells:
            out[c.status] = out.get(c.status, 0) + 1
        return out

    @property
    def total_cells(self) -> int:
        return len(self.cells)

    def fill_rate(self) -> float:
        """Fraction of cells that are NOT UNKNOWN (IMPLEMENTED or explicitly REFUSED).

        REFUSED counts as FILLED: an honest, typed gap is coverage, not debt. Only the
        UNKNOWN cell -- the untested claim -- is unfilled.
        """
        total = self.total_cells
        if not total:
            return 1.0
        counts = self.counts()
        return (counts[IMPLEMENTED] + counts[REFUSED]) / float(total)

    def unknown_cells(self) -> List[Cell]:
        return [c for c in self.cells if c.status == UNKNOWN]

    def to_dict(self) -> dict:
        counts = self.counts()
        return {
            "oracle": "coverage_matrix",
            "ops": list(self.ops),
            "columns": ["%s|%s" % (b, f) for (b, f) in self.columns],
            "codec_backed": dict(sorted(self.codec_backed.items())),
            "total_cells": self.total_cells,
            "counts": counts,
            "fill_rate": self.fill_rate(),
            "unknown_count": counts[UNKNOWN],
            "unknown_cells": sorted(c.key for c in self.unknown_cells()),
            "skipped": list(self.skipped),
            "cells": [c.to_dict() for c in self.cells],
        }


# ---------------------------------------------------------------------------
# Enumeration -- rows from the op registry, columns from the capability registry
# ---------------------------------------------------------------------------

def op_vocabulary() -> Tuple[str, ...]:
    """The CISP op vocabulary: the op tags in ``core.cisp.ops._REGISTRY`` (sorted)."""
    return tuple(sorted(OP_REGISTRY.keys()))


def codec_formats() -> Tuple[str, ...]:
    """Format families backed by a real codec module in ``io/formats`` (sorted).

    Enumerated from the capability registry, not guessed: every module in the
    ``formats`` package is a codec, and its module NAME is the format family it
    encodes (``stl``, ``glb``, ``step``, ``svg``, ...). Used only to ANNOTATE each
    column with whether its format reaches a known codec -- an honesty flag, it does
    not by itself decide a cell's status.
    """
    names = set()
    try:
        for e in capability_registry.find(package="formats"):
            names.add(e.name.lower())
    except Exception:  # noqa: BLE001 - a missing index must not crash the gate
        return ()
    return tuple(sorted(names))


_BACKEND_BASECLASSES = frozenset({"GeometryBackend", "ExternalToolBackend"})


def _concrete_backend_class(module: Any) -> Optional[type]:
    """The concrete ``*Backend`` class defined in ``module`` (not an abstract base)."""
    best = None
    for attr in sorted(vars(module)):
        obj = getattr(module, attr)
        if not isinstance(obj, type):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if not obj.__name__.endswith("Backend"):
            continue
        if obj.__name__ in _BACKEND_BASECLASSES:
            continue
        if not hasattr(obj, "FORMATS"):
            continue
        best = obj
        break
    return best


def discover_backends() -> Tuple[List[BackendSpec], List[str]]:
    """Enumerate backend columns from the capability registry. Returns (specs, skipped).

    For every module in the ``backends`` package that names a ``*Backend`` symbol, the
    MODULE is imported (its kernel is located lazily, so no external tool is invoked)
    and the class's DECLARED ``FORMATS`` / ``UNSUPPORTED`` / ``PRIMITIVE_SHAPES`` are
    read off. A module that cannot be imported, or that carries no emitting backend
    class, is recorded in ``skipped`` -- never dropped silently.
    """
    specs: List[BackendSpec] = []
    skipped: List[str] = []
    seen: set = set()

    try:
        entries = capability_registry.find(package="backends")
    except Exception as exc:  # noqa: BLE001
        return specs, ["backend enumeration failed: %s" % exc]

    candidates = sorted(
        (e for e in entries if any(s.endswith("Backend") for s in e.symbols)),
        key=lambda e: e.dotted,
    )
    for e in candidates:
        if not any(s.endswith("Backend") and s not in _BACKEND_BASECLASSES
                   for s in e.symbols):
            skipped.append("%s: only an abstract backend base, no emitter" % e.dotted)
            continue
        try:
            module = importlib.import_module(e.dotted)
        except Exception as exc:  # noqa: BLE001 - an optional dep must not crash us
            skipped.append("%s: import failed (%s)" % (e.dotted, exc))
            continue
        cls = _concrete_backend_class(module)
        if cls is None:
            skipped.append("%s: no concrete *Backend with a declared FORMATS" % e.dotted)
            continue

        formats = tuple(str(f) for f in getattr(cls, "FORMATS", ()) or ())
        if not formats:
            skipped.append("%s: backend declares no export FORMATS" % e.dotted)
            continue

        # ``e.name`` is the module basename, which is exactly the driver name the
        # harness resolves a backend by (probe/server BACKENDS: 'manifold', 'frep',
        # 'cadquery', ...). Use it as the column's backend identity.
        name = e.name
        if name in seen:
            continue
        seen.add(name)

        # UNSUPPORTED distinguishes a DECLARED surface (dict, maybe empty) from NO
        # surface at all (attribute absent -> None -> every op UNKNOWN).
        unsupported: Optional[Dict[str, str]] = None
        if hasattr(cls, "UNSUPPORTED"):
            raw = getattr(cls, "UNSUPPORTED")
            if isinstance(raw, dict):
                unsupported = {str(k): str(v) for k, v in raw.items()}
        prim = getattr(cls, "PRIMITIVE_SHAPES", None)
        primitive_shapes = (tuple(str(s) for s in prim)
                            if isinstance(prim, (tuple, list)) else None)

        specs.append(BackendSpec(
            name=name, formats=formats, unsupported=unsupported,
            primitive_shapes=primitive_shapes,
            note=("declared support surface" if unsupported is not None
                  else "no static support surface (refuses inline)")))

    specs.sort(key=lambda s: s.name)
    return specs, skipped


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(op: str, spec: BackendSpec) -> Tuple[str, str]:
    """The static verdict for one op on one backend. Pure; no kernel touched."""
    if spec.unsupported is None:
        return (UNKNOWN,
                "backend '%s' declares no op-support surface; whether it lowers "
                "'%s' is never exercised statically (an untested claim)"
                % (spec.name, op))
    if op in spec.unsupported:
        return REFUSED, spec.unsupported[op]
    if op == "primitive" and spec.primitive_shapes is not None:
        if not spec.primitive_shapes:
            return REFUSED, ("backend lowers no primitive shapes (PRIMITIVE_SHAPES "
                             "is empty): every 'primitive' shape is refused")
        return (IMPLEMENTED,
                "op lowers; primitive shapes covered: %s (others refused)"
                % ", ".join(spec.primitive_shapes))
    return (IMPLEMENTED,
            "op is not declared unsupported and the backend exports the format")


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------

def census(ops: Optional[Sequence[str]] = None,
           backends: Optional[Sequence[BackendSpec]] = None,
           sample: Optional[int] = None,
           log: Optional[Callable[[str], None]] = None) -> CoverageReport:
    """Build the op x (backend, format) coverage matrix.

    ``ops`` / ``backends`` may be INJECTED (the ``--selfcheck`` fixture passes a
    synthetic op list and synthetic :class:`BackendSpec` tables, so the classification
    can be exercised with no kernel and no model). Left as ``None`` they are discovered
    from the op registry and the capability registry respectively.

    ``sample`` caps the number of COLUMNS censused. Columns left out are recorded in
    the report's ``skipped`` field and passed to ``log`` -- a truncated census must
    never read as a complete one.
    """
    notes: List[str] = []

    def _emit(msg: str) -> None:
        notes.append(msg)
        if log is not None:
            log(msg)

    op_tags = tuple(sorted(str(o) for o in ops)) if ops is not None else op_vocabulary()

    if backends is None:
        specs, skipped = discover_backends()
        for s in skipped:
            _emit("backend column skipped -- %s" % s)
    else:
        specs = list(backends)

    codec = set(codec_formats())

    # Every column is (backend, format), the format drawn from the backend's own
    # declared FORMATS. Deterministic order: by backend name, then format.
    columns: List[Tuple[str, str]] = []
    codec_backed: Dict[str, bool] = {}
    for spec in sorted(specs, key=lambda s: s.name):
        for fmt in spec.formats:
            columns.append((spec.name, fmt))
            base = fmt.split("-", 1)[0].lower()
            codec_backed[fmt] = base in codec if codec else False
    columns.sort()

    total_columns = len(columns)
    if sample is not None and 0 <= sample < total_columns:
        kept = columns[:sample]
        dropped = columns[sample:]
        _emit("SAMPLED: censused %d of %d columns; %d skipped: %s"
              % (len(kept), total_columns, len(dropped),
                 ", ".join("%s/%s" % (b, f) for b, f in dropped)))
        columns = kept

    spec_by_name = {s.name: s for s in specs}
    cells: List[Cell] = []
    for backend, fmt in columns:
        spec = spec_by_name.get(backend)
        if spec is None:
            continue
        for op in op_tags:
            status, reason = classify(op, spec)
            cells.append(Cell(op=op, backend=backend, fmt=fmt,
                              status=status, reason=reason))

    return CoverageReport(
        ops=op_tags,
        columns=tuple(columns),
        cells=tuple(cells),
        codec_backed=codec_backed,
        skipped=tuple(notes),
    )


# ---------------------------------------------------------------------------
# The drift pause-gate
# ---------------------------------------------------------------------------

def _baseline_doc(report: CoverageReport) -> dict:
    counts = report.counts()
    return {
        "_": "COMMITTED COVERAGE-MATRIX BASELINE. Enforced by "
             "harnesscad.eval.gates.coverage_matrix. The fill rate may only RISE "
             "and the UNKNOWN-cell count may only FALL. A drift down (lower fill "
             "rate) or a grown UNKNOWN count fails the build. REFUSED cells are "
             "honest, filled coverage; only UNKNOWN cells are debt.",
        "oracle": "coverage_matrix",
        "fill_rate": report.fill_rate(),
        "unknown_count": counts[UNKNOWN],
        "total_cells": report.total_cells,
        "implemented": counts[IMPLEMENTED],
        "refused": counts[REFUSED],
        "columns": ["%s|%s" % (b, f) for (b, f) in report.columns],
        "unknown_cells": sorted(c.key for c in report.unknown_cells()),
    }


def load_baseline(path: str) -> Optional[dict]:
    """Read the baseline sidecar, or ``None`` when it is absent/unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def write_baseline(report: CoverageReport, path: str) -> dict:
    """Write the baseline sidecar (deterministic: sorted keys, newline-terminated)."""
    doc = _baseline_doc(report)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return doc


@dataclass
class DriftReport:
    """The gate verdict: is the matrix at least as good as the baseline?"""

    ok: bool = True
    established: bool = False
    violations: List[str] = field(default_factory=list)
    fill_rate: float = 1.0
    baseline_fill_rate: Optional[float] = None
    unknown_count: int = 0
    baseline_unknown_count: Optional[int] = None
    new_unknown: List[str] = field(default_factory=list)
    baseline_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "oracle": "coverage_matrix",
            "ok": self.ok,
            "established": self.established,
            "violations": self.violations,
            "fill_rate": self.fill_rate,
            "baseline_fill_rate": self.baseline_fill_rate,
            "unknown_count": self.unknown_count,
            "baseline_unknown_count": self.baseline_unknown_count,
            "new_unknown": self.new_unknown,
            "baseline_path": self.baseline_path,
        }


def check(report: CoverageReport, baseline_path: Optional[str] = None,
          eps: float = DRIFT_EPS) -> DriftReport:
    """Score a census against a baseline sidecar. A DRIFT pause-gate.

    * No ``baseline_path``, or the file is absent: ESTABLISH the baseline (write it
      when a path is given) and PASS. The first run only records where we stand.
    * Otherwise FAIL when the fill rate dropped below the baseline (drift down) or the
      UNKNOWN-cell count grew. Both are "the system got worse than before".
    """
    counts = report.counts()
    out = DriftReport(
        fill_rate=report.fill_rate(),
        unknown_count=counts[UNKNOWN],
        baseline_path=baseline_path,
    )

    base = load_baseline(baseline_path) if baseline_path else None
    if base is None:
        out.established = True
        out.ok = True
        if baseline_path:
            write_baseline(report, baseline_path)
        return out

    out.baseline_fill_rate = float(base.get("fill_rate", 0.0))
    out.baseline_unknown_count = int(base.get("unknown_count", 0))

    if out.fill_rate < out.baseline_fill_rate - eps:
        out.violations.append(
            "FILL RATE DRIFTED DOWN: %.4f -> %.4f. The matrix has PROPORTIONALLY "
            "more untested claims than the committed baseline: a cell that was "
            "IMPLEMENTED or REFUSED is now UNKNOWN, or the axes grew without "
            "classification." % (out.baseline_fill_rate, out.fill_rate))
    if out.unknown_count > out.baseline_unknown_count:
        out.violations.append(
            "UNKNOWN CELLS GREW: %d -> %d. New empty cells are new coverage debt -- "
            "untested (op, backend, format) claims that read as coverage but have "
            "never been exercised either way."
            % (out.baseline_unknown_count, out.unknown_count))

    base_unknown = set(base.get("unknown_cells", []))
    out.new_unknown = sorted(set(c.key for c in report.unknown_cells()) - base_unknown)
    out.ok = not out.violations
    return out


# ---------------------------------------------------------------------------
# What this gate does and does not establish
# ---------------------------------------------------------------------------

def claims() -> Dict[str, Any]:
    """Exactly what a PASS establishes -- and the cap it is believed past at your peril."""
    return {
        "proves": [
            "no censused (op, backend, format) triple regressed from IMPLEMENTED or "
            "REFUSED to UNKNOWN versus the committed baseline",
            "the count of UNKNOWN cells (untested claims) did not grow",
            "every UNKNOWN cell is named, so the coverage debt is a list, not a feeling",
        ],
        "does_not_prove": [
            "that an IMPLEMENTED cell actually builds: IMPLEMENTED is a STATIC claim "
            "(the op is not declared unsupported and the format is in the backend's "
            "codec set), not an execution proof -- a backend may still refuse an op "
            "mid-apply that its class never listed",
            "anything about columns left out by --sample: those are recorded in "
            "'skipped', and a sampled census is not a complete one",
            "that a REFUSED cell's reason is the whole reason -- it is the backend's "
            "own declared explanation, read verbatim",
        ],
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def format_text(report: CoverageReport, drift: Optional[DriftReport] = None) -> str:
    counts = report.counts()
    lines: List[str] = []
    lines.append("COVERAGE-MATRIX CENSUS (op x backend x format)")
    lines.append("=" * 72)
    lines.append("ops: %d   columns (backend x format): %d   cells: %d"
                 % (len(report.ops), len(report.columns), report.total_cells))
    lines.append("IMPLEMENTED %d   REFUSED %d   UNKNOWN %d   fill rate %.4f"
                 % (counts[IMPLEMENTED], counts[REFUSED], counts[UNKNOWN],
                    report.fill_rate()))
    lines.append("")
    lines.append("(REFUSED is honest, filled coverage; only UNKNOWN cells are debt.)")

    unknown = report.unknown_cells()
    if unknown:
        lines.append("")
        lines.append("UNKNOWN cells (untested claims) -- %d:" % len(unknown))
        for c in sorted(unknown, key=lambda x: x.key)[:40]:
            lines.append("    %s" % c.key)
        if len(unknown) > 40:
            lines.append("    ... %d more" % (len(unknown) - 40))

    if report.skipped:
        lines.append("")
        lines.append("SKIPPED (not censused -- stated, never silent):")
        for s in report.skipped:
            lines.append("    %s" % s)

    if drift is not None:
        lines.append("")
        lines.append("-" * 72)
        if drift.established:
            lines.append("BASELINE ESTABLISHED at %s (fill rate %.4f, %d UNKNOWN). "
                         "PASS." % (drift.baseline_path, drift.fill_rate,
                                    drift.unknown_count))
        elif drift.ok:
            lines.append("PASS: the matrix is at least as good as the baseline "
                         "(fill rate %.4f >= %.4f, UNKNOWN %d <= %d)."
                         % (drift.fill_rate, drift.baseline_fill_rate or 0.0,
                            drift.unknown_count,
                            drift.baseline_unknown_count or 0))
        else:
            lines.append("FAIL: the matrix drifted below the baseline.")
            for v in drift.violations:
                lines.append("  %s" % v)
            for k in drift.new_unknown[:20]:
                lines.append("    new UNKNOWN: %s" % k)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Selfcheck -- synthetic ops + synthetic backend/format tables, NO kernel/model
# ---------------------------------------------------------------------------

def selfcheck() -> Tuple[bool, str]:
    """Prove IMPLEMENTED / REFUSED / UNKNOWN classification and a drift-down FAILURE.

    Everything here is injected and synthetic: three fake ops and two fake backends,
    one that declares an UNSUPPORTED surface (so it yields IMPLEMENTED + REFUSED) and
    one that declares no surface (so it yields UNKNOWN). No backend is imported, no
    kernel is located, no model is run.
    """
    ops = ["op_a", "op_b", "op_c"]
    alpha = BackendSpec(
        name="alpha", formats=("stl", "step"),
        unsupported={"op_c": "alpha refuses op_c: no builder for it"})
    beta = BackendSpec(
        name="beta", formats=("glb",), unsupported=None)  # no surface -> UNKNOWN

    good = census(ops=ops, backends=[alpha, beta])
    counts = good.counts()

    # alpha x {stl, step}: op_a, op_b IMPLEMENTED (4), op_c REFUSED (2).
    # beta x {glb}: op_a, op_b, op_c all UNKNOWN (3).
    checks: List[Tuple[str, bool]] = []
    checks.append(("IMPLEMENTED == 4", counts[IMPLEMENTED] == 4))
    checks.append(("REFUSED == 2", counts[REFUSED] == 2))
    checks.append(("UNKNOWN == 3", counts[UNKNOWN] == 3))
    checks.append(("total cells == 9", good.total_cells == 9))
    # fill rate = (4 + 2) / 9
    checks.append(("fill rate == 6/9", abs(good.fill_rate() - 6.0 / 9.0) < 1e-12))

    # classification spot-checks
    st_a, _ = classify("op_a", alpha)
    st_c, _ = classify("op_c", alpha)
    st_b, _ = classify("op_b", beta)
    checks.append(("op_a on alpha IMPLEMENTED", st_a == IMPLEMENTED))
    checks.append(("op_c on alpha REFUSED", st_c == REFUSED))
    checks.append(("op_b on beta UNKNOWN", st_b == UNKNOWN))

    # Drift gate: establish a baseline from the good census, then run a DEGRADED
    # census (alpha now declares no surface, so its 6 filled cells become UNKNOWN)
    # and require the gate to FAIL on both drift-down and grown-unknown.
    fd, baseline_path = tempfile.mkstemp(prefix="coverage_matrix_selfcheck_",
                                         suffix=".json")
    os.close(fd)
    try:
        established = check(good, baseline_path=baseline_path)
        checks.append(("baseline established + PASS",
                       established.established and established.ok))
        pass_again = check(good, baseline_path=baseline_path)
        checks.append(("unchanged census PASSes", pass_again.ok
                       and not pass_again.established))

        alpha_blind = BackendSpec(name="alpha", formats=("stl", "step"),
                                  unsupported=None)  # lost its support surface
        degraded = census(ops=ops, backends=[alpha_blind, beta])
        drift = check(degraded, baseline_path=baseline_path)
        checks.append(("degraded census FAILS", not drift.ok))
        checks.append(("drift names >= 1 violation", len(drift.violations) >= 1))
        checks.append(("drift reports new UNKNOWN cells", len(drift.new_unknown) >= 1))
    finally:
        try:
            os.remove(baseline_path)
        except OSError:
            pass

    passed = all(ok for _label, ok in checks)
    failed = [label for label, ok in checks if not ok]
    message = ("all %d selfcheck assertions held" % len(checks) if passed
               else "FAILED: " + "; ".join(failed))
    return passed, message


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline", default=None,
                        help="path to the JSON baseline sidecar. Absent file: the "
                             "baseline is established and the gate PASSes. Present: "
                             "the gate fails on drift down / grown UNKNOWN count.")
    parser.add_argument("--update", action="store_true",
                        help="(re)write the baseline from the live census. A "
                             "deliberate act: the diff is the review.")
    parser.add_argument("--sample", type=int, default=None,
                        help="cap the number of (backend, format) columns censused. "
                             "Skipped columns are printed, never dropped silently.")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="emit JSON instead of text")
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the synthetic classification + drift-down fixture "
                             "(no kernel, no model) and exit")


def run(args: argparse.Namespace) -> int:
    if getattr(args, "selfcheck", False):
        passed, message = selfcheck()
        if getattr(args, "as_json", False):
            print(json.dumps({"selfcheck": passed, "message": message},
                             indent=2, sort_keys=True))
        else:
            print("coverage_matrix selfcheck: %s -- %s"
                  % ("PASS" if passed else "FAIL", message))
        return 0 if passed else 1

    report = census(sample=getattr(args, "sample", None),
                    log=lambda m: print(m, file=sys.stderr))

    baseline_path = getattr(args, "baseline", None)
    if getattr(args, "update", False):
        if not baseline_path:
            print("error: --update needs --baseline PATH", file=sys.stderr)
            return 2
        doc = write_baseline(report, baseline_path)
        print("wrote %s (fill rate %.4f, %d UNKNOWN cell(s))"
              % (baseline_path, doc["fill_rate"], doc["unknown_count"]))
        return 0

    drift = check(report, baseline_path=baseline_path)
    if getattr(args, "as_json", False):
        print(json.dumps({"census": report.to_dict(), "gate": drift.to_dict(),
                          "claims": claims()}, indent=2, sort_keys=True))
    else:
        print(format_text(report, drift))
    return 0 if drift.ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="coverage_matrix",
        description="Census the op x backend x format coverage matrix and fail the "
                    "build when it drifts below a committed baseline.")
    add_arguments(parser)
    return run(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
