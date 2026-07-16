"""Worker process for :mod:`harnesscad.io.ingest.step_check` -- never imported.

Runs in a CHILD process, reads one STEP file, prints ONE line of JSON to stdout
and exits. It is the process that is allowed to die: a malformed STEP that
segfaults OCCT, or a pathological one that spins forever, takes this process
down instead of the harness.

Contract with the parent (:func:`harnesscad.io.ingest.step_check.check_step`):
  * stdout's LAST line is a JSON object carrying ``status`` and ``roots``;
  * exit 0 on a clean read, 1 on a read that failed;
  * ``status`` distinguishes ``"empty"`` (the reader SUCCEEDED and the file
    contained no shape roots) from ``"malformed"`` (the reader FAILED). They
    are different facts and are never merged.

``--reader auto`` (default) uses OCCT when installed and otherwise falls back to
the harness's kernel-free part-21 parser, so the check is real on a machine with
no kernel. ``--reader part21`` pins the kernel-free parser, which is what makes
a caller's result reproducible across machines.

The two readers answer slightly different questions and are both right about
their own: OCCT counts TRANSFERABLE roots, so a shape entity with no product
structure around it is legitimately 0 roots to it, while the part-21 reader
counts shape-bearing entities present in DATA. A caller comparing the two is
comparing "can this be transferred" against "does this contain a shape".
"""

from __future__ import annotations

import json
import os
import sys
import traceback

#: Part-21 keywords that carry an actual shape -- the kernel-free notion of a
#: transferable root. A file of only points/directions/units has none.
ROOT_KEYWORDS = frozenset({
    "MANIFOLD_SOLID_BREP",
    "BREP_WITH_VOIDS",
    "FACETED_BREP",
    "SHELL_BASED_SURFACE_MODEL",
    "GEOMETRIC_CURVE_SET",
    "ADVANCED_BREP_SHAPE_REPRESENTATION",
    "MANIFOLD_SURFACE_SHAPE_REPRESENTATION",
    "FACETED_BREP_SHAPE_REPRESENTATION",
    "GEOMETRICALLY_BOUNDED_SURFACE_SHAPE_REPRESENTATION",
    "GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION",
})


def _emit(payload: dict, code: int) -> int:
    # The pid is the parent's proof that the read really happened elsewhere.
    payload.setdefault("pid", os.getpid())
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()
    return code


def _check_occt(path: str) -> dict:
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        return {"status": "malformed", "roots": 0, "reader": "occt",
                "note": f"STEP reader returned {status}"}
    roots = int(reader.NbRootsForTransfer())
    if roots == 0:
        # Parsed CLEANLY and holds nothing. Not a failure -- a real, empty file.
        return {"status": "empty", "roots": 0, "reader": "occt",
                "note": "reader succeeded; file declares no transferable root"}
    transferred = int(reader.TransferRoots())
    return {"status": "ok", "roots": roots, "reader": "occt",
            "transferred": transferred}


def _check_pure(path: str) -> dict:
    from harnesscad.io.formats.step import ParseError, Typed, parse

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    try:
        step = parse(text)
    except ParseError as exc:
        return {"status": "malformed", "roots": 0, "reader": "part21",
                "note": f"ParseError: {exc}"}
    except ValueError as exc:
        return {"status": "malformed", "roots": 0, "reader": "part21",
                "note": f"{type(exc).__name__}: {exc}"}

    roots = 0
    for ent_id in step.order:
        entity = step.entities[ent_id]
        keywords = []
        if entity.keyword is not None:
            keywords.append(entity.keyword)
        keywords.extend(p.keyword for p in entity.params if isinstance(p, Typed))
        if any(k.upper() in ROOT_KEYWORDS for k in keywords):
            roots += 1
    if roots == 0:
        return {"status": "empty", "roots": 0, "reader": "part21",
                "entities": len(step.entities),
                "note": "parsed cleanly; no shape-bearing root entity"}
    return {"status": "ok", "roots": roots, "reader": "part21",
            "entities": len(step.entities)}


def run(path: str, reader: str = "auto") -> int:
    try:
        if reader == "part21":
            payload = _check_pure(path)
        else:
            try:
                payload = _check_occt(path)
            except ImportError:
                payload = _check_pure(path)
    except Exception as exc:  # noqa: BLE001 - a crashing read is a report, not a traceback
        return _emit({"status": "error", "roots": 0,
                      "note": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()[-800:]}, 1)
    return _emit(payload, 0 if payload["status"] in ("ok", "empty") else 1)


def _main(argv: list) -> int:
    reader = "auto"
    args = list(argv)
    if len(args) >= 2 and args[0] == "--reader":
        reader = args[1]
        args = args[2:]
    if len(args) != 1 or reader not in ("auto", "part21"):
        sys.stderr.write(
            "usage: _step_check_worker.py [--reader auto|part21] <step-path>\n")
        return 2
    return run(args[0], reader)


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
