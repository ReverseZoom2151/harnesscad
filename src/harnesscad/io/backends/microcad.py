"""MicrocadBackend — CISP ops emitted as microcad (µcad) source, built by its CLI.

microcad (µcad, https://microcad.xyz, on crates.io, VCS on Codeberg) is a NEW
open-source **declarative programming language** for CAD (Rust, v0.5.0 EARLY
ALPHA). It is a *language*, not a geometry kernel: you write a ``.µcad`` program
that composes primitives with boolean operators and transforms, and the
``microcad`` CLI evaluates it to a mesh which it exports (STL). It therefore
integrates EXACTLY the way OpenSCAD does, and this backend is built to the same
shape as :mod:`harnesscad.io.backends.openscad`:

    CISP ops -> F-rep CSG tree -> microcad source -> `microcad export` -> STL -> mesh

Everything up to "microcad source" is the shared, kernel-neutral machinery that
already exists in this repo -- the op semantics, id allocation, block-and-correct,
the F-rep CSG DAG (:class:`~harnesscad.io.backends.frep.FRepBackend`) and its
lowering helpers -- reused verbatim through
:class:`~harnesscad.io.backends.external.ExternalToolBackend`. This module only
adds the one thing that is microcad's: how to LOWER that tree into microcad's
surface syntax, and how to drive its CLI.

What was actually observed (2026-07, this machine)
--------------------------------------------------
* ``cargo install microcad`` **still does not produce a binary here** (re-attempted
  2026-07 with cargo/rustc 1.96.0 on the scoop ``rustup-gnu`` toolchain). The
  MinGW link error the earlier attempts hit is no longer the blocker; the build now
  fails EARLIER, at a transitive **native C++ dependency**. microcad v0.5.0 pulls a
  crate whose ``build.rs`` drives CMake (``cmake`` crate 0.1.58), and CMake aborts
  before compiling anything::

      CMake Error: CMake was unable to find a build program corresponding to
      "Ninja". CMAKE_MAKE_PROGRAM is not set.
      CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
      thread 'main' panicked at cmake-0.1.58/src/lib.rs:1132: command did not
      execute successfully, got: exit code: 1 -- build script failed, must exit now
      error: failed to compile `microcad v0.5.0`

  i.e. building microcad from crates.io now requires a working C++ toolchain
  (a CXX compiler) AND the Ninja generator on PATH, neither of which is present /
  configured on this MinGW-only Rust toolchain. This is an environment gap in the
  transitive C++ build dependency, not a microcad or a harness bug, and installing
  CMake + Ninja + a C++ compiler is outside what this task changes. So on this
  machine the CLI is still ABSENT, and -- exactly like the OpenSCAD / Blender /
  FreeCAD backends when their tool is missing -- the constructor raises
  :class:`~harnesscad.io.backends.base.BackendUnavailable`, the server falls back to
  the stub WITH A NOTE, and the tests SKIP. Nothing is faked. The backend stays
  import-guarded/stub until a toolchain with CMake + Ninja + a CXX compiler builds
  the CLI.
* microcad's real syntax and CLI, read from the docs / the Lego-brick example
  (docs.microcad.xyz, codeberg.org/microcad/microcad), and encoded below:

  - primitives ``Cube(size)``, ``Cylinder(h =, r =/d =)``, ``Sphere(r =/d =)``
    (3D, ``use std::geo3d``); ``Rect(width, height)``, ``Circle(r =/d =)``
    (2D, ``use std::geo2d``);
  - values carry a **unit** -- ``8mm``, ``1.2mm`` -- never a bare number for a
    length;
  - 2D -> 3D by method call: ``<sketch>.extrude(<height>)``;
  - transforms are chained method calls: ``.translate(x =, y =, z =)``;
  - booleans are **infix operators**: ``a | b`` (union), ``a - b`` (difference),
    ``a & b`` (intersection); a group ``{ a; b; }.union()`` is the same union;
  - the CLI is ``microcad``: ``microcad check <file>`` type-checks and prints the
    model tree; ``microcad export <file> <out.stl>`` evaluates and writes geometry,
    the **export format chosen by the output extension** -- ``.stl`` for 3D,
    ``.svg`` for 2D. STL is the measurable format this backend reads back (there
    is no STEP/OBJ/3MF exporter in the alpha, and no volume/measurement CLI, so --
    like the other external backends -- mass properties come from the exported
    mesh via :mod:`harnesscad.io.formats.stl`).

The op -> microcad mapping (supported vs refused)
-------------------------------------------------
microcad is a CSG *language* with **no B-rep and no topological entities** (no
persistent edges, no persistent faces), and the alpha's standard library does not
yet document a 2D offset, a revolve, or a general rigid-transform primitive. So,
honestly and per op -- implement exactly, or REFUSE with a typed diagnostic; never
accept a field and drop it:

* **Implemented, emitted as microcad source.** ``extrude`` (of rectangles and
  circles), ``boolean`` (union/cut/intersect -> ``|`` / ``-`` / ``&``), and the
  cylindrical cut of ``hole`` for ``kind='simple'`` and ``kind='counterbore'`` (a
  counterbore is a stacked ``Cylinder`` -- both are straight cylinders, which
  microcad has).
* **Refused, typed ``unsupported-op``.** ``fillet`` / ``chamfer`` / ``draft``
  (no topological edges or faces to select; no 3D erosion), ``loft`` / ``sweep``
  (not in the language), ``revolve`` (no ``rotate_extrude`` verified in the
  alpha), ``shell`` (no 2D ``offset`` / erosion verified), ``mirror`` and the
  ``linear_pattern`` / ``circular_pattern`` families (the alpha's mirror /
  rigid-transform syntax is not verified, and emitting a guessed transform would
  build a different part), and ``hole`` with ``kind='countersink'`` (needs a
  truncated cone, which microcad has no verified primitive for). A sketch on any
  datum plane other than ``XY`` is refused for the same reason: the alpha's
  transform syntax for arbitrary planes is not verified, and a guessed
  orientation would silently misplace the solid.
* **Approximated.** Nothing.

Determinism and the cache key
-----------------------------
The emitted source is byte-stable (sorted, fixed float formatting), and BOTH the
segment count AND the microcad version are stamped into the program text and into
:meth:`MicrocadBackend.state_digest`, so they are part of the content-addressed
cache key -- without the version in the key, upgrading microcad would silently
re-serve the STL the previous build produced (the exact stale-across-upgrade bug
Blender's and FreeCAD's keys once had).
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
from typing import Dict, List, Optional, Sequence

from harnesscad.core.cisp.ops import Hole
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.backends.base import ApplyResult, BackendUnavailable
from harnesscad.io.backends.external import (
    ExternalToolBackend, plane_axes, slab,
)
from harnesscad.io.backends.frep import Node

__all__ = ["MicrocadBackend", "MicrocadError", "lower", "render", "fmt_mm"]

#: Where `cargo install microcad` drops the binary, globbed (never a hard-coded
#: version). CARGO_HOME/bin on this machine, plus the conventional ~/.cargo/bin.
MICROCAD_PATTERNS = (
    os.path.join(os.environ.get("CARGO_HOME", ""), "bin", "microcad.exe"),
    os.path.join(os.environ.get("CARGO_HOME", ""), "bin", "microcad"),
    os.path.expanduser(r"~\.cargo\bin\microcad.exe"),
    os.path.expanduser("~/.cargo/bin/microcad"),
    r"C:\Users\*\scoop\persist\rustup-gnu\.cargo\bin\microcad.exe",
)

MICROCAD_ENV = "HARNESSCAD_MICROCAD"

#: microcad values are lengths with a unit. HarnessCAD works in millimetres, so
#: every length this backend emits carries the `mm` unit suffix.
UNIT = "mm"

#: Clearance added beyond a face so a cut passes cleanly through it.
CUT_PAD = 1.0


class MicrocadError(RuntimeError):
    """microcad ran but did not produce geometry (or the emitter cannot express
    the model in the verified subset of the alpha language)."""


# --------------------------------------------------------------------------
# deterministic number formatting -- the reason the source is byte-stable
# --------------------------------------------------------------------------
def _num(x: float) -> str:
    """A float as a stable, minimal decimal string (never scientific, no -0)."""
    v = float(x)
    if v == 0.0:
        v = 0.0                                  # collapse -0.0 to 0.0
    s = ("%.6f" % v).rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def fmt_mm(x: float) -> str:
    """A length as a microcad literal, e.g. ``20mm`` -- units are mandatory."""
    return _num(x) + UNIT


# --------------------------------------------------------------------------
# lowering: F-rep tree -> microcad source expression
# --------------------------------------------------------------------------
def _require_xy(plane: str, what: str) -> None:
    if str(plane).upper() != "XY":
        raise MicrocadError(
            "microcad backend: a %s on the %r plane is not expressible in the "
            "verified subset -- the alpha's transform syntax for arbitrary datum "
            "planes is not confirmed, and a guessed orientation would silently "
            "misplace the solid. Only XY-plane sketches are lowered." % (what, plane))


def _translate(expr: str, dx: float, dy: float, dz: float) -> str:
    """Chain a ``.translate(...)`` only for the axes that actually move.

    An identity translate is omitted entirely, so a rectangle already at the
    origin emits no transform -- the source stays minimal and byte-stable.
    """
    args: List[str] = []
    if dx:
        args.append("x = %s" % fmt_mm(dx))
    if dy:
        args.append("y = %s" % fmt_mm(dy))
    if dz:
        args.append("z = %s" % fmt_mm(dz))
    if not args:
        return expr
    return "%s.translate(%s)" % (expr, ", ".join(args))


def _profile_2d(profile) -> str:
    """The sketch's 2D region as a microcad expression (a union of its entities).

    ``Rect`` and ``Circle`` are taken to be centred on the local origin (the
    common convention in declarative CAD languages), so each entity is placed by
    translating it to its centre. A profile carrying a free polygon is refused:
    the alpha has no verified 2D polygon primitive, and emitting a guessed one
    would build a different region.
    """
    parts: List[str] = []
    for (x, y, w, h) in profile.rects:
        parts.append(_translate("Rect(%s, %s)" % (fmt_mm(w), fmt_mm(h)),
                                 x + w / 2.0, y + h / 2.0, 0.0))
    for (cx, cy, r) in profile.circles:
        parts.append(_translate("Circle(r = %s)" % fmt_mm(r), cx, cy, 0.0))
    if getattr(profile, "polys", None):
        for verts in profile.polys:
            if len(verts) >= 3:
                raise MicrocadError(
                    "microcad backend: a free polygon sketch is not expressible "
                    "in the verified subset (no confirmed 2D polygon primitive in "
                    "the alpha); only rectangles and circles are lowered")
    if not parts:
        raise MicrocadError("microcad backend: empty sketch profile reached the emitter")
    if len(parts) == 1:
        return parts[0]
    return "(%s)" % " | ".join(parts)


def lower(node: Node) -> str:
    """One F-rep node as a microcad source expression."""
    t = node.t
    if t == "extrude":
        _require_xy(node.d["plane"], "sketch")
        lo, hi = slab(float(node.d["w0"]), float(node.d["w1"]))
        body = "%s.extrude(%s)" % (_profile_2d(node.d["profile"]), fmt_mm(hi - lo))
        return _translate(body, 0.0, 0.0, lo)
    if t == "cyl":
        _require_xy(node.d["plane"], "cylinder")
        lo, hi = slab(float(node.d["w0"]), float(node.d["w1"]))
        r = float(node.d["r"])
        body = "Cylinder(h = %s, r = %s)" % (fmt_mm(hi - lo), fmt_mm(r))
        return _translate(body, float(node.d["cu"]), float(node.d["cv"]), lo)
    if t == "bool":
        a = lower(node.d["a"])
        b = lower(node.d["b"])
        op = node.d["op"]
        sym = {"union": "|", "intersect": "&", "cut": "-"}.get(op)
        if sym is None:
            raise MicrocadError("microcad backend: unknown boolean op %r" % op)
        return "(%s %s %s)" % (a, sym, b)
    if t == "cone":
        raise MicrocadError(
            "microcad backend: a truncated cone (countersink) is not expressible "
            "-- the alpha has no verified cone primitive; Cylinder is straight")
    if t in ("revolve", "shell", "mirror", "pattern"):
        raise MicrocadError(
            "microcad backend: %r reached the emitter but is not in the verified "
            "subset and should have been refused at apply()" % t)
    raise MicrocadError("microcad backend: unknown F-rep node kind %r" % t)


def header(segments: int, version: str) -> str:
    """The source header -- and the reason the cache key is honest.

    The microcad version and the segment count are stamped into the program TEXT,
    and the program text is what the content-addressed cache directory is named
    after. A different kernel build, or a different tessellation, is therefore a
    different artefact -- never a stale hit.
    """
    return ("// Generated by harnesscad MicrocadBackend. Do not edit.\n"
            "// microcad-version: %s\n"
            "// segments: %d\n"
            "use std::geo2d::*;\n"
            "use std::geo3d::*;\n"
            "use std::ops::*;\n"
            "use std::math::*;" % (version, int(segments)))


def render(root: Node, segments: int, version: str = "unknown") -> str:
    """The whole model as microcad source (deterministic, byte-stable)."""
    return "%s\n\n%s;\n" % (header(segments, version), lower(root))


# --------------------------------------------------------------------------
# the backend
# --------------------------------------------------------------------------
class MicrocadBackend(ExternalToolBackend):
    """A GeometryBackend backed by the microcad (µcad) CLI -- a CSG language."""

    TOOL = "microcad"
    #: Whole ops microcad cannot honour HONESTLY, refused with a typed diagnostic
    #: rather than approximated. The reasons are microcad's own: it is a CSG
    #: *language* with no topological entities and no verified erosion / revolve /
    #: rigid-transform in the v0.5.0 alpha standard library.
    UNSUPPORTED: Dict[str, str] = {
        "fillet": "microcad is a CSG language with no topological edges, and the "
                  "alpha has no verified 3D erosion, so a constant-radius blend "
                  "cannot be expressed",
        "chamfer": "microcad has no topological edges to select and no verified "
                   "3D erosion with which to set an edge back",
        "draft": "microcad has no face entities, so a draft angle cannot be "
                 "applied to a named neutral plane and face set",
        "loft": "microcad has no loft operator in the verified language subset",
        "sweep": "microcad has no sweep-along-a-path operator in the verified "
                 "language subset",
        "revolve": "microcad has no verified revolve / rotate_extrude operator in "
                   "the v0.5.0 alpha; refusing rather than guessing its syntax",
        "shell": "microcad has no verified 2D offset / erosion in the v0.5.0 "
                 "alpha, so an inward hollow cannot be built without changing the "
                 "part; refused rather than approximated",
        "mirror": "microcad's mirror syntax is not verified in the v0.5.0 alpha; "
                  "emitting a guessed transform would build a different part",
        "linear_pattern": "microcad's rigid-transform / repeat syntax is not "
                          "verified in the v0.5.0 alpha; a guessed transform would "
                          "misplace the instances",
        "circular_pattern": "microcad's rotation / repeat syntax is not verified "
                            "in the v0.5.0 alpha; a guessed transform would "
                            "misplace the instances",
        "thicken": "microcad has no verified 3D offset / erosion in the v0.5.0 "
                   "alpha, so growing or shrinking a solid by a wall thickness "
                   "cannot be built without changing the part; refused rather "
                   "than approximated",
        "hull": "microcad has no verified convex-hull operator in the v0.5.0 "
                "alpha, so a hull is refused rather than guessed",
        "minkowski": "microcad has no verified Minkowski / 3D offset operator in "
                     "the v0.5.0 alpha; a ball dilation is built in the frep SDF "
                     "kernel and in OpenSCAD's minkowski(), and refused here",
        "transform": "microcad's rigid-transform (translate / rotate) syntax is "
                     "not verified in the v0.5.0 alpha; a guessed transform would "
                     "misplace the body, so an in-place move is refused (as mirror "
                     "and the patterns are)",
        "scale": "microcad's scale syntax is not verified in the v0.5.0 alpha; a "
                 "guessed scale would resize the body wrongly, so it is refused "
                 "rather than faked",
        "pattern_transform": "microcad's rigid-transform / repeat syntax is not "
                             "verified in the v0.5.0 alpha; a guessed transform "
                             "would misplace the instances (as linear_pattern and "
                             "circular_pattern are refused)",
    }
    #: box lowers to an extruded Rect, cylinder to Cylinder -- both verified in the
    #: alpha. cone/sphere/torus/wedge have no verified primitive, so are refused.
    PRIMITIVE_SHAPES = ("box", "cylinder")
    #: STL is the only measurable export the alpha CLI writes; everything else in
    #: FORMATS is derived from that mesh by the shared code, plus 'microcad' source.
    FORMATS = ("stl", "stl-ascii", "stl-binary", "glb", "microcad")

    #: ``microcad --version``, memoised per executable path. Part of the cache key.
    _VERSIONS: Dict[str, str] = {}

    # -- discovery ---------------------------------------------------------
    @classmethod
    def locate(cls) -> str:
        from harnesscad.io.backends.external import find_executable

        path, searched = find_executable(
            MICROCAD_ENV, ("microcad",),
            tuple(p for p in MICROCAD_PATTERNS if p))
        if path is None:
            raise BackendUnavailable(
                "microcad",
                "microcad is not installed (or not on PATH). Install it with "
                "`cargo install microcad` (needs a WORKING Rust link toolchain -- "
                "on a broken MinGW self-contained setup the link step fails on "
                "-lkernel32 / -lgcc_eh), or point %s at the microcad binary. "
                "Searched: %s" % (MICROCAD_ENV, ", ".join(searched)),
                searched)
        return path

    # -- version (a cache-key input, not a cosmetic banner) -----------------
    def tool_version(self) -> str:
        cached = type(self)._VERSIONS.get(self.executable)
        if cached is not None:
            return cached
        try:
            proc = subprocess.run([self.executable, "--version"],
                                  capture_output=True, text=True, timeout=30)
            text = (proc.stdout or "") + (proc.stderr or "")
            version = text.strip().splitlines()[0].strip() if text.strip() else "unknown"
        except Exception:                                   # pragma: no cover
            version = "unknown"
        type(self)._VERSIONS[self.executable] = version
        return version

    def state_digest(self) -> str:
        """Content hash of the model AND of everything that decides its geometry.

        The op stream alone is not enough: the same ops on two microcad builds may
        mesh differently, and the segment count changes a curved part. Both are in
        the digest, so a cache entry can never outlive the tool that made it.
        """
        blob = "%s|%s|segments=%d|version=%s" % (
            self.TOOL, self._frep.state_digest(), self.segments, self.tool_version())
        return hashlib.sha256(blob.encode()).hexdigest()

    # -- op admission ------------------------------------------------------
    @staticmethod
    def _refuse(code: str, msg: str) -> ApplyResult:
        return ApplyResult(False, [], [Diagnostic(Severity.ERROR, code, msg, None)])

    def apply(self, op):
        """Refuse what cannot be honoured, BEFORE anything mutates.

        The base class refuses whole ops (:data:`UNSUPPORTED`). This adds the one
        field-level admission the base cannot do: a ``Hole`` whose ``kind`` is
        ``countersink`` (a truncated cone microcad cannot build). A plain hole and
        a counterbore ARE expressible (straight cylinders), so they pass through --
        including their ``cbore_*`` fields, which the F-rep tree carries and this
        backend lowers rather than dropping.
        """
        if isinstance(op, Hole):
            kind = str(getattr(op, "kind", "simple")).lower()
            if kind == "countersink":
                return self._refuse(
                    "unsupported-op",
                    "the microcad backend cannot cut a countersink: it needs a "
                    "truncated cone, and the v0.5.0 alpha has no verified cone "
                    "primitive (Cylinder is straight). A simple or counterbore "
                    "hole is expressible and accepted.")
            if kind not in ("simple", "counterbore"):
                return self._refuse(
                    "bad-value",
                    "unknown hole kind %r (microcad backend accepts 'simple' and "
                    "'counterbore')" % kind)
        return super().apply(op)

    # -- the program -------------------------------------------------------
    def program(self) -> str:
        root = self.root()
        if root is None:
            raise MicrocadError("microcad backend: no solid to render")
        return render(root, self.segments, version=self.tool_version())

    def _run(self, source: str, workdir: str, out_path: str) -> None:
        """Evaluate the source with the CLI and read back the STL it exports.

        The invocation is ``microcad export <input.µcad> <output.stl>`` -- the
        export format is chosen by the output extension (``.stl`` for 3D). A run
        that fails, or leaves no geometry, RAISES and removes any stale artefact,
        so the file-exists cache can never re-serve a corpse or a previous model.
        """
        src_path = os.path.join(workdir, "model.µcad")
        with open(src_path, "w", encoding="utf-8") as fh:
            fh.write(source)
        if os.path.isfile(out_path):
            os.remove(out_path)
        argv = [self.executable, "export", src_path, out_path]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=self.timeout)
        except FileNotFoundError as exc:                    # pragma: no cover
            raise MicrocadError("microcad CLI not runnable: %s" % exc)
        empty = not (os.path.isfile(out_path) and os.path.getsize(out_path) > 0)
        if proc.returncode != 0 or empty:
            if os.path.isfile(out_path):
                os.remove(out_path)
            detail = (proc.stderr or proc.stdout or "").strip()
            if proc.returncode != 0:
                raise MicrocadError(
                    "microcad export failed (exit %d): %s" % (proc.returncode, detail))
            raise MicrocadError(
                "microcad export exited 0 but wrote no geometry to %s" % out_path)

    # -- export ------------------------------------------------------------
    def export(self, fmt: str):
        if str(fmt).lower() == "microcad":
            return self.program()
        return super().export(fmt)
