"""CADBench per-program execution-status + error taxonomy (facts + log manifest).

Source: CADBench (Doris et al. 2026, DeCoDELab; arXiv:2605.10873,
https://github.com/anniedoris/CADBench), MIT (c)2026 Annie Doris. The eval
harness writes one ``bench*_logs.json`` per (model, modality, bench): 486 files,
~1.44M rows in total. Each row is one model-generated CadQuery program's
outcome -- ``{file_id, the six geometry metrics, token/line/op counts, status,
details, label}``, where ``status`` is the execution verdict and ``details`` the
exact interpreter/kernel error string.

This complements the harness's CURATED known-bad corpora
(``cadgenbench_broken``, ``adversarial_code``, ``error_taxonomy``) with the
opposite kind of evidence: the REAL failure DISTRIBUTION of models emitting
CadQuery at scale. But it is NOISY model output, so the discipline is split:

  * the RAW logs are MANIFEST-ONLY -- 486 file paths + SHA-256, resolved from
    ``resources/`` at run time, degrade-to-empty when the checkout is absent.
    The 1.44M rows are never vendored.
  * the distinct STATUS CODES and the ERROR-STRING CATEGORIES are FACTS worth
    keeping, so they are extracted ONCE (by scanning all 486 files) and embedded
    here with their observed counts and canonical example strings, cited to the
    source. These are always present -- they are facts, not licensed bulk.

Every category is machine-checkable: it carries a predicate over a row's
``(status, details)`` pair, so a caller with the raw logs (or its own model
outputs) can re-bucket rows deterministically. :data:`STATUS_CODES` predicates
read only ``status``; :data:`ERROR_CATEGORIES` predicates gate on
``status == 0`` (execution error) and then match ``details`` substrings.

Grading protocol reference (CADBench ``Eval/_main.py``): status 0 = execution
error, 1 = success, 2 = timeout. Verified by scanning all 486 committed log
files (see :data:`TOTAL_ROWS` / :data:`STATUS_ROW_COUNTS`).

Stdlib only. Deterministic. ASCII. No geometry kernel. Degrades to empty when
resources/ is absent (the taxonomy stays; the raw files do not).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from harnesscad.eval.corpus.fixtures import Manifest, load_manifest, sha256_of

__all__ = [
    "StatusCode",
    "ErrorCategory",
    "STATUS_CODES",
    "ERROR_CATEGORIES",
    "EXPECTED_LOG_FILES",
    "TOTAL_ROWS",
    "STATUS_ROW_COUNTS",
    "manifest",
    "status_code",
    "classify_error",
    "log_files",
    "main",
]

_SOURCE = "cadbench_exec_logs"

#: One ``bench*_logs.json`` per (model, modality, bench) in ``tested_models/``.
EXPECTED_LOG_FILES = 486

#: Facts extracted by scanning all 486 committed log files (snapshot 2026-07-20).
TOTAL_ROWS = 1437679
#: status -> row count across the whole corpus. Sums to :data:`TOTAL_ROWS`.
STATUS_ROW_COUNTS = {0: 506440, 1: 922186, 2: 9053}


@dataclass(frozen=True)
class StatusCode:
    """One execution-status code and its meaning; predicate reads ``status``."""

    code: int
    name: str
    meaning: str
    observed_rows: int

    def matches(self, status: int) -> bool:
        return status == self.code


@dataclass(frozen=True)
class ErrorCategory:
    """A recognizer for a family of ``status == 0`` failure strings.

    ``signatures`` are lowercase substrings; a row is in this category iff its
    status is 0 (execution error) and any signature occurs in ``details``. The
    ``example`` is a real string observed verbatim in the logs (a fact).
    """

    cid: str
    description: str
    signatures: Tuple[str, ...]
    observed_rows: int
    example: str

    def matches(self, status: int, details: str) -> bool:
        if status != 0:
            return False
        low = (details or "").lower()
        return any(sig in low for sig in self.signatures)


# --------------------------------------------------------------------------- #
# STATUS CODES -- the three-valued execution verdict (Eval/_main.py), with the
# row counts observed across all 486 committed log files.
# --------------------------------------------------------------------------- #

STATUS_CODES: Tuple[StatusCode, ...] = (
    StatusCode(0, "execution_error",
               "the program raised while executing / exporting (syntax, name, "
               "type, CadQuery or OCC-kernel error); details is the exception",
               506440),
    StatusCode(1, "success",
               "the program executed and exported a solid; details == 'Success'",
               922186),
    StatusCode(2, "timeout",
               "execution exceeded the wall-clock budget; "
               "details == 'Code Execution Timeout'",
               9053),
)


# --------------------------------------------------------------------------- #
# ERROR CATEGORIES -- the distinct families of status-0 ``details`` strings,
# extracted by scanning all 486 files. Ordered by observed frequency. Each
# example is a string that appears verbatim in the corpus. These are FACTS.
# Recognizers may overlap (a taxonomy of matchers, not a strict partition); the
# selfcheck proves each predicate fires on its own recorded example.
# --------------------------------------------------------------------------- #

ERROR_CATEGORIES: Tuple[ErrorCategory, ...] = (
    ErrorCategory(
        "name_error",
        "an identifier (often 'cq', 'math', or a mis-referenced variable) was "
        "used before assignment -- NameError / UnboundLocalError",
        ("is not defined", "cannot access local variable", "unboundlocal"),
        215862,
        "Code Execution Error: name 'cq' is not defined"),
    ErrorCategory(
        "cadquery_geom_error",
        "CadQuery rejected the modelling operation: empty selection, no pending "
        "wires, no solid on the stack, bad plane/selector string, failed loft "
        "or fillet -- the program ran but built no valid geometry",
        ("workplane", "no pending wires", "cannot find a solid",
         "there are no suita", "fillets requires", "plane.__init__",
         "expected three flo", "pending", "selector", "chamfer", "spline",
         "nothing to loft", "no entities specified", "supported names are",
         "expected {", "expected end of text", "expected 'not'", "nth element",
         "cannot convert object type", "segment:"),
        116976,
        "Code Execution Error: No pending wires present"),
    ErrorCategory(
        "occ_kernel_error",
        "the underlying OpenCASCADE kernel failed (BRep_API, GC_MakeArcOfCircle, "
        "BRepSweep, BSplCLib, etc.) -- geometrically invalid input reached OCC",
        ("brep_api", "gc_make", "stdfail", "standard_", "brepbuilderapi",
         "topods", "geomapi", "gcpnts", "brepalgoapi", "bnd_", "brepsweep",
         "ncollection", "bsplclib", "brepoffsetapi", "gp_", "geom_",
         "no result", "arc radius", "bspl"),
        89668,
        "Code Execution Error: GC_MakeArcOfCircle::Value() - no result"),
    ErrorCategory(
        "attribute_error",
        "a method or attribute does not exist on the object (typically a "
        "hallucinated CadQuery API) -- AttributeError",
        ("has no attribute",),
        53142,
        "Code Execution Error: 'Workplane' object has no attribute "
        "'threePointArc'"),
    ErrorCategory(
        "syntax_error",
        "the emitted program is not valid Python (unclosed bracket, bad indent, "
        "unterminated string, stray 'return') -- SyntaxError/IndentationError",
        ("was never closed", "invalid syntax", "unexpected indent",
         "unterminated", "invalid decimal", "positional argument follows",
         "cannot assign", "indentationerror", "syntaxerror", "unmatched",
         "'return' outside", "eof in multi-line", "expected an indented"),
        9377,
        "Code Execution Error: '(' was never closed (<string>, line 18)"),
    ErrorCategory(
        "type_error",
        "wrong argument count/type, bad keyword, non-callable, or unsupported "
        "operand -- TypeError from a mis-shaped API call",
        ("object is not", "takes", "positional argument", "unexpected keyword",
         "__init__()", "not callable", "unsupported operand", "must be",
         "nonetype", "object cannot be interpreted", "argument"),
        8266,
        "Code Execution Error: Workplane.__init__() missing 1 required "
        "positional argument"),
    ErrorCategory(
        "worker_error",
        "the sandboxed worker process died abnormally (segfault/OOM in the "
        "kernel) rather than raising a Python exception",
        ("worker error", "abnormal termination"),
        527,
        "Worker error: Abnormal termination"),
    ErrorCategory(
        "index_key_error",
        "indexing past the end of a sequence or a missing dict key -- "
        "IndexError / KeyError",
        ("list index out of range", "index out of range", "tuple index",
         "keyerror"),
        298,
        "Code Execution Error: list index out of range"),
    ErrorCategory(
        "import_error",
        "an import failed (usually a hallucinated module name like 'cquery') -- "
        "ModuleNotFoundError / ImportError",
        ("no module named", "cannot import name"),
        91,
        "Code Execution Error: No module named 'cquery'"),
    ErrorCategory(
        "zero_division",
        "division by zero in a computed dimension -- ZeroDivisionError",
        ("division by zero", "zerodivision"),
        43,
        "Code Execution Error: float division by zero"),
    ErrorCategory(
        "value_error",
        "a value was out of the accepted domain (e.g. math domain, bad literal "
        "conversion) -- ValueError",
        ("could not convert", "invalid literal", "math domain", "valueerror"),
        10,
        "Code Execution Error: math domain error"),
    ErrorCategory(
        "recursion_error",
        "runaway recursion in the generated program -- RecursionError",
        ("recursion",),
        2,
        "Code Execution Error: maximum recursion depth exceeded in "
        "__instancecheck__"),
)


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def status_code(status: int) -> Optional[StatusCode]:
    """The :class:`StatusCode` matching ``status``, or ``None`` if unknown."""
    for sc in STATUS_CODES:
        if sc.matches(status):
            return sc
    return None


def classify_error(status: int, details: str) -> List[str]:
    """Every :data:`ERROR_CATEGORIES` id whose predicate matches the row.

    Empty for a success/timeout row, and possibly empty for an unrecognised
    status-0 string (the taxonomy is a recognizer set, not a total partition).
    """
    return [c.cid for c in ERROR_CATEGORIES if c.matches(status, details)]


def log_files() -> List[Path]:
    """Every AVAILABLE ``bench*_logs.json`` path. Empty when resources/ absent.

    Callers must treat an empty list as "corpus not present", never as "no
    errors" -- the extracted taxonomy above is the always-present substitute.
    """
    m = manifest()
    return [p for p in (m.resolve(e) for e in m.entries) if p is not None]


def _selfcheck() -> int:
    m = manifest()
    assert m.license == "MIT", m.license
    assert len(m.entries) == EXPECTED_LOG_FILES, len(m.entries)

    # Manifest-only: nothing vendored, every entry a resource path + SHA-256.
    assert not m.verify_vendored(), "no vendored files were expected"
    for e in m.entries:
        assert e.vendored is None, "unexpected vendored file: %s" % e.name
        assert e.resource and e.resource.endswith("_logs.json"), e.name
        assert e.role == "exec_log", e.name
        assert len(e.sha256) == 64, "entry %s has no sha256" % e.name

    # STATUS CODES: non-empty, each machine-checkable, counts sum to the total.
    assert len(STATUS_CODES) == 3, len(STATUS_CODES)
    codes = {sc.code for sc in STATUS_CODES}
    assert codes == {0, 1, 2}, codes
    for sc in STATUS_CODES:
        assert sc.matches(sc.code)
        assert not sc.matches(sc.code + 100)
    assert sum(sc.observed_rows for sc in STATUS_CODES) == TOTAL_ROWS
    for sc in STATUS_CODES:
        assert STATUS_ROW_COUNTS[sc.code] == sc.observed_rows, sc.name

    # ERROR CATEGORIES: non-empty; each is machine-checkable and its predicate
    # FIRES on its own recorded example while STAYING SCOPED to status 0.
    assert ERROR_CATEGORIES, "error taxonomy is empty"
    seen = set()
    for c in ERROR_CATEGORIES:
        assert c.cid not in seen, "duplicate category id %s" % c.cid
        seen.add(c.cid)
        assert c.signatures, "category %s has no signatures" % c.cid
        assert c.example, "category %s has no example" % c.cid
        assert c.observed_rows > 0, c.cid
        # predicate is machine-checkable: matches its example under status 0,
        assert c.matches(0, c.example), (
            "category %s does not match its own example" % c.cid)
        # and never fires on a success or timeout row.
        assert not c.matches(1, c.example), c.cid
        assert not c.matches(2, c.example), c.cid
        # classify_error routes the example to this category.
        assert c.cid in classify_error(0, c.example), c.cid

    # A clean success/timeout string yields no error category.
    assert classify_error(1, "Success") == []
    assert classify_error(2, "Code Execution Timeout") == []

    # File resolution census -- honest manifest-only degrade-to-empty.
    avail = m.availability()
    if avail["present"] == 0:
        print("SELFCHECK OK: manifest-only (%d log files, MIT), resources/ "
              "absent -> raw logs degrade to empty as designed; the extracted "
              "taxonomy is intact (3 status codes summing to %d rows, %d "
              "machine-checkable error categories)"
              % (EXPECTED_LOG_FILES, TOTAL_ROWS, len(ERROR_CATEGORIES)))
        return 0

    files = log_files()
    for p in files[:4]:
        entry = next(e for e in m.entries if m.resolve(e) == p)
        assert sha256_of(p) == entry.sha256, "drift: %s" % entry.name
    print("SELFCHECK OK: %d/%d log files resolvable from resources/ (SHA spot-"
          "checked); taxonomy intact (3 status codes, %d error categories)"
          % (avail["present"], avail["total"], len(ERROR_CATEGORIES)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CADBench per-program execution logs (raw files manifest-"
                    "only; status-code + error-category taxonomy embedded as "
                    "facts, MIT).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate the log manifest and prove the embedded "
                             "taxonomy is non-empty and machine-checkable; "
                             "degrades cleanly when resources/ is absent.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0
    try:
        return _selfcheck()
    except AssertionError as exc:
        print("SELFCHECK FAILED: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
