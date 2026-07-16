"""Subprocess-isolated STEP checking: a bad file kills a child, never the parent.

A STEP file is untrusted input. Parsing one in-process gives it three ways to
take the harness with it: a malformed entity graph can segfault the kernel, a
pathological one can spin without ever returning, and either way an ingest run
that was scoring a hundred files dies on the first bad one. So the read happens
in a CHILD process under a wall-clock timeout, and the parent only ever sees a
result object.

The second half of this module is a distinction the parent must not collapse:

    a parse that yields NOTHING is not a parse that FAILED.

Conflating them is a silence-is-success bug in the direction that hurts most:
an empty result reported as a failure hides a real (and really empty) file,
while a failure reported as empty lets a broken file score as "read fine, no
geometry". :class:`StepCheckResult` keeps them apart as ``status="empty"``
(reader succeeded, zero roots) versus ``status="malformed"`` (reader failed),
with ``ok`` true only for the former's well-formed sibling.

Attribution: pattern and policy from the MUSE benchmark (muse-main,
``src/judge_system/geometry_metrics.py``): STEP/geometry checking run through
``subprocess`` under ``timeout_seconds`` "so a malformed STEP can't crash the
parent", the last-stdout-line JSON contract, and the bbox pre-filter that skips
the expensive boolean for solid pairs whose bounding boxes cannot overlap
(``EPS = 1e-6``, strict inequality). MUSE is MIT-licensed (Copyright (c) 2026
MUSE Benchmark contributors); this is an independent implementation of that
design rather than a copy, and the status-vs-empty-roots split is the harness's
own addition.

Pure stdlib in the parent; the worker degrades to the harness's kernel-free
part-21 parser when OCCT is absent. The worker command is INJECTABLE so the
isolation and the timeout are testable without a kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

#: Worker module run as ``python -m`` in the child.
WORKER_MODULE = "harnesscad.io.ingest._step_check_worker"

#: Default wall-clock budget for one file. Generous: a big assembly is slow, a
#: hostile file is infinite, and only the second one should hit this.
DEFAULT_TIMEOUT_S = 120.0

#: Every status this module can report. ``ok``/``empty`` mean the reader
#: SUCCEEDED; the rest mean it did not, each for a different reason.
STATUSES = ("ok", "empty", "malformed", "timeout", "crashed", "error", "missing")


@dataclass(frozen=True)
class StepCheckResult:
    """The outcome of one isolated STEP read.

    ``ok`` is deliberately narrow: only a read that succeeded AND found at least
    one shape root. Ask ``parsed`` for "did the reader succeed", which is true
    for ``empty`` too -- that is the distinction this module exists to keep.
    """

    status: str
    path: str
    roots: int = 0
    note: str = ""
    reader: str = ""
    elapsed_s: float = 0.0
    returncode: Optional[int] = None
    timed_out: bool = False
    killed: bool = False
    details: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """The file was read AND contains geometry."""
        return self.status == "ok"

    @property
    def parsed(self) -> bool:
        """The reader SUCCEEDED -- true for an empty file, false for a broken one."""
        return self.status in ("ok", "empty")

    @property
    def empty(self) -> bool:
        """Read fine, contained no shape root. Not an error."""
        return self.status == "empty"

    def to_dict(self) -> dict:
        return {
            "status": self.status, "path": self.path, "roots": self.roots,
            "note": self.note, "reader": self.reader,
            "elapsed_s": round(self.elapsed_s, 6),
            "returncode": self.returncode, "timed_out": self.timed_out,
            "killed": self.killed, "ok": self.ok, "parsed": self.parsed,
            "details": dict(self.details),
        }


def _worker_command(python_executable: Optional[str], reader: str) -> List[str]:
    return [python_executable or sys.executable, "-m", WORKER_MODULE,
            "--reader", reader]


def check_step(
    path: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    worker_cmd: Optional[Sequence[str]] = None,
    python_executable: Optional[str] = None,
    reader: str = "auto",
    env: Optional[dict] = None,
) -> StepCheckResult:
    """Read ``path`` in a child process under ``timeout_s``. Never raises.

    ``reader`` is ``"auto"`` (OCCT when installed, else the kernel-free part-21
    parser) or ``"part21"`` (pin the kernel-free parser, for a result that does
    not depend on what is installed).

    ``worker_cmd`` replaces the default worker (the file path is appended as the
    final argument, and ``reader`` is left to the injected worker). That
    injection point is what makes the timeout and the isolation testable without
    a geometry kernel -- a test can hand in a worker that hangs, or one that
    lies, and watch the parent survive it.

    On timeout the child is KILLED (not merely abandoned) and reaped, so the
    call cannot leak a spinning process; the result carries ``timed_out=True``
    and ``killed=True``.
    """
    if not path or not os.path.exists(path):
        return StepCheckResult(status="missing", path=path,
                               note=f"file not found: {path!r}")
    if not os.path.isfile(path):
        return StepCheckResult(status="missing", path=path,
                               note=f"not a file: {path!r}")

    if reader not in ("auto", "part21"):
        return StepCheckResult(status="error", path=path,
                               note=f"unknown reader {reader!r}")
    cmd = (list(worker_cmd) if worker_cmd
           else _worker_command(python_executable, reader))
    cmd = cmd + [path]

    child_env = dict(os.environ if env is None else env)
    # The child imports harnesscad; inherit the parent's resolution so a
    # source checkout works without an install step.
    child_env.setdefault("PYTHONPATH", os.pathsep.join(
        p for p in sys.path if p and os.path.isdir(p)))

    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=child_env)
    except OSError as exc:
        return StepCheckResult(
            status="error", path=path, elapsed_s=time.monotonic() - start,
            note=f"cannot start worker: {type(exc).__name__}: {exc}")

    killed = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        # The whole point: the hostile file's process dies, this one does not.
        proc.kill()
        killed = True
        try:
            stdout, stderr = proc.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:  # pragma: no cover - unkillable child
            stdout, stderr = "", "worker did not die after kill"
        return StepCheckResult(
            status="timeout", path=path, elapsed_s=time.monotonic() - start,
            returncode=proc.returncode, timed_out=True, killed=True,
            note=f"worker exceeded {timeout_s:g}s and was killed",
            details={"stderr": (stderr or "")[-500:]})

    elapsed = time.monotonic() - start
    line = ""
    for candidate in reversed((stdout or "").strip().splitlines()):
        if candidate.strip():
            line = candidate.strip()
            break
    if not line:
        # No report at all: the child died before it could speak (segfault,
        # OOM-kill). That is "crashed" -- emphatically not "empty".
        return StepCheckResult(
            status="crashed", path=path, elapsed_s=elapsed,
            returncode=proc.returncode, killed=killed,
            note=f"worker produced no report (exit {proc.returncode})",
            details={"stderr": (stderr or "")[-500:]})
    try:
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("worker report is not a JSON object")
    except Exception as exc:  # noqa: BLE001 - a lying worker is an error, not a crash
        return StepCheckResult(
            status="error", path=path, elapsed_s=elapsed,
            returncode=proc.returncode,
            note=f"unreadable worker report: {exc}",
            details={"stdout_tail": line[:300]})

    status = str(payload.get("status", "error"))
    if status not in STATUSES:
        status = "error"
    return StepCheckResult(
        status=status, path=path, roots=int(payload.get("roots", 0) or 0),
        note=str(payload.get("note", "")), reader=str(payload.get("reader", "")),
        elapsed_s=elapsed, returncode=proc.returncode, killed=killed,
        details={k: v for k, v in payload.items()
                 if k not in ("status", "roots", "note", "reader")})


# --------------------------------------------------------------------------- #
# bbox pre-filter (the cheap half of the interpenetration metric)
# --------------------------------------------------------------------------- #

#: Overlap slack. Two solids sharing exactly a face plane touch, they do not
#: interpenetrate, so the comparisons below are STRICT by this margin.
BBOX_EPS = 1e-6

#: An axis-aligned box as ``(xmin, ymin, zmin, xmax, ymax, zmax)``.
Box = Tuple[float, float, float, float, float, float]


def bbox_overlap(a: Box, b: Box, eps: float = BBOX_EPS) -> bool:
    """Do two axis-aligned boxes overlap with real volume (not just touch)?"""
    return (a[0] < b[3] - eps and b[0] < a[3] - eps and
            a[1] < b[4] - eps and b[1] < a[4] - eps and
            a[2] < b[5] - eps and b[2] < a[5] - eps)


def interpenetration_candidates(boxes: Sequence[Box],
                                eps: float = BBOX_EPS) -> List[Tuple[int, int]]:
    """Index pairs whose bounding boxes overlap -- the only pairs worth a boolean.

    The pre-filter is the whole trick: the pairwise intersect that measures real
    interpenetration is the expensive, hang-prone kernel call, and box rejection
    is exact for the negative case (disjoint boxes cannot possibly overlap in
    volume). Deterministic ascending order.
    """
    out: List[Tuple[int, int]] = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if bbox_overlap(boxes[i], boxes[j], eps):
                out.append((i, j))
    return out


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

_GOOD_STEP = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('selfcheck'),'2;1');
FILE_NAME('t.step','2026-07-16',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CLOSED_SHELL('',(#3));
#3=ADVANCED_FACE('',(),#1,.T.);
#4=MANIFOLD_SOLID_BREP('body',#2);
ENDSEC;
END-ISO-10303-21;
"""

#: Well-formed part-21, zero shape roots: the "empty, not broken" case.
_EMPTY_STEP = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('selfcheck'),'2;1');
FILE_NAME('e.step','2026-07-16',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
ENDSEC;
END-ISO-10303-21;
"""

#: Unterminated string + unbalanced parens: the reader must FAIL on this.
_BROKEN_STEP = """ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#1=CARTESIAN_POINT('unterminated,(0.,0.,0.;
#2=MANIFOLD_SOLID_BREP('body',#
ENDSEC;
"""


def _write(tmpdir: str, name: str, text: str) -> str:
    path = os.path.join(tmpdir, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Subprocess-isolated STEP checking with a timeout, and the "
                    "empty-vs-malformed distinction (MUSE geometry_metrics.py "
                    "pattern).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="prove BOTH arms: a well-formed STEP parses, and a "
                             "hanging worker is killed by the timeout instead of "
                             "hanging the parent.")
    parser.add_argument("--check", metavar="STEP", help="check one STEP file")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--reader", choices=("auto", "part21"), default="auto")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.check:
        result = check_step(args.check, timeout_s=args.timeout,
                            reader=args.reader)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.parsed else 1
    if not args.selfcheck:
        parser.print_help()
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        # ARM 1: a well-formed STEP parses, in a child process. The reader is
        # PINNED to part21 so this arm asserts the same thing on every machine
        # rather than on whether OCCT happens to be installed.
        good = _write(tmp, "good.step", _GOOD_STEP)
        r = check_step(good, timeout_s=60.0, reader="part21")
        assert r.status == "ok", r.to_dict()
        assert r.ok and r.parsed and r.roots >= 1, r.to_dict()
        print(f"[selfcheck] well-formed STEP: status={r.status} "
              f"roots={r.roots} reader={r.reader} "
              f"({r.elapsed_s:.2f}s, out-of-process)")

        # ARM 1b: with a real kernel present, the DEFAULT (auto/OCCT) path on a
        # kernel-written STEP. Skipped, loudly, when OCCT is absent -- a skip
        # that announces itself is not a pass.
        #
        # The fixture is exported in a CHILD process, and that is not fastidious-
        # ness: importing cadquery here made THIS selfcheck exit 139 (OCCT
        # segfaults the interpreter on teardown). The parent of an isolation
        # module must not load the kernel it is isolating -- which is the
        # module's whole thesis, demonstrated on itself.
        real = os.path.join(tmp, "real.step")
        export = subprocess.run(
            [sys.executable, "-c",
             "import sys, cadquery as cq\n"
             "cq.exporters.export(cq.Workplane('XY').box(10,10,10), sys.argv[1])\n",
             real],
            capture_output=True, text=True, timeout=180.0)
        # Gate on the ARTEFACT, not the exit code: OCCT here writes a complete,
        # valid STEP and THEN segfaults on interpreter teardown (exit
        # 0xC0000005, empty stderr). Trusting the returncode would discard a
        # perfectly good file -- and is why check_step below reads the worker's
        # report first and treats the returncode as advisory.
        if not os.path.exists(real) or os.path.getsize(real) == 0:
            print("[selfcheck] SKIP kernel arm: no usable OCCT "
                  f"(exit {export.returncode})")
        else:
            r = check_step(real, timeout_s=120.0)
            assert r.status == "ok", r.to_dict()
            assert r.reader == "occt" and r.roots >= 1, r.to_dict()
            print(f"[selfcheck] kernel-written STEP via OCCT: status={r.status} "
                  f"roots={r.roots} worker_exit={r.returncode} "
                  f"({r.elapsed_s:.2f}s, out-of-process)")

        # ARM 2: a hanging worker is KILLED by the timeout. The parent must
        # return in about the timeout, not in the worker's 60s.
        hang = [sys.executable, "-c",
                "import sys, time\n"
                "sys.stderr.write('spinning\\n')\n"
                "time.sleep(60)\n"]
        t0 = time.monotonic()
        r = check_step(good, timeout_s=1.0, worker_cmd=hang)
        elapsed = time.monotonic() - t0
        assert r.status == "timeout", r.to_dict()
        assert r.timed_out and r.killed, r.to_dict()
        assert elapsed < 30.0, f"parent hung for {elapsed:.1f}s"
        assert r.returncode is not None, "killed worker was not reaped"
        print(f"[selfcheck] hanging worker: killed after {elapsed:.2f}s "
              f"(worker wanted 60s), returncode={r.returncode} -- parent alive")

        # ARM 2b: a worker that dies without reporting is 'crashed', NOT 'empty'.
        die = [sys.executable, "-c", "import os; os._exit(139)"]
        r = check_step(good, timeout_s=30.0, worker_cmd=die)
        assert r.status == "crashed", r.to_dict()
        assert not r.parsed and not r.empty
        print(f"[selfcheck] silently dying worker: status={r.status} "
              f"(exit {r.returncode}) -- not confused with 'empty'")

        # ARM 3: THE distinction -- empty is not malformed.
        empty = _write(tmp, "empty.step", _EMPTY_STEP)
        re_ = check_step(empty, timeout_s=60.0, reader="part21")
        assert re_.status == "empty", re_.to_dict()
        assert re_.parsed is True and re_.ok is False and re_.roots == 0
        broken = _write(tmp, "broken.step", _BROKEN_STEP)
        rb = check_step(broken, timeout_s=60.0, reader="part21")
        assert rb.status == "malformed", rb.to_dict()
        assert rb.parsed is False and rb.ok is False
        assert re_.status != rb.status, "empty and malformed collapsed"
        print(f"[selfcheck] empty(parsed={re_.parsed}) vs "
              f"malformed(parsed={rb.parsed}): distinct, neither is 'ok'")

        # ARM 4: a missing file never reaches a subprocess.
        r = check_step(os.path.join(tmp, "nope.step"))
        assert r.status == "missing" and not r.parsed
        print("[selfcheck] missing file reported without spawning a worker")

    # ARM 5: bbox pre-filter -- disjoint rejected, touching rejected, overlap kept.
    boxes = [
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        (0.5, 0.5, 0.5, 1.5, 1.5, 1.5),   # really overlaps box 0
        (5.0, 5.0, 5.0, 6.0, 6.0, 6.0),   # disjoint from everything
        (1.0, 0.0, 0.0, 2.0, 1.0, 1.0),   # face-touches box 0: NOT interpenetration
    ]
    pairs = interpenetration_candidates(boxes)
    assert pairs == [(0, 1), (1, 3)], pairs
    assert not bbox_overlap(boxes[0], boxes[3]), "touching boxes flagged"
    assert not bbox_overlap(boxes[0], boxes[2])
    print(f"[selfcheck] bbox pre-filter: {len(pairs)} of 6 pairs need a "
          f"boolean; face-touching pair rejected")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
