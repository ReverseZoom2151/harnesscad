"""TruckBackend — CISP ops driven through the ``truck`` B-rep NURBS kernel.

Why this backend exists (the oracle argument)
----------------------------------------------
The differential oracle (:mod:`harnesscad.eval.selftest.differential`) is only as
strong as the INDEPENDENCE of the kernels it cross-checks. Every other B-rep
backend the harness has -- ``cadquery``, ``freecad``, ``build123d`` -- is a
wrapper around the *same* kernel, OpenCASCADE (OCCT). They agree by
construction, so a bug that lives in OCCT is invisible: three "independent"
backends all return the same wrong number.

``truck`` (https://github.com/ricosjp/truck, Apache-2.0) is a from-scratch B-rep
NURBS kernel written in Rust that is emphatically NOT OCCT -- a different
topological data model, different NURBS evaluation, different boolean engine
(``truck-shapeops``). It is therefore a genuinely independent B-rep voice, the
single strongest addition to the oracle: a B-rep result that truck AGREES with
is one two unrelated B-rep lineages both vouch for. This mirrors what the
Manifold backend did for the mesh world -- as a third, independent kernel it
caught a hand-analytic error where it and cadquery both returned the right
number and the analytic was wrong.

The out-of-process shape (harness stays Python)
-----------------------------------------------
truck is Rust; the harness is Python. So, exactly like the FreeCAD, OpenSCAD and
Blender backends, this one is a subprocess driver. A small compiled Rust CLI
(``truck_driver/``) reads a normalised geometry job (JSON, lowered here from the
kernel-neutral F-rep tree) and writes back a tessellated STL plus a sidecar
(``model.json``: truck's own volume, bounding box, B-rep face/edge counts, and
whether a STEP file was written). The Python side shells out to it, reads the
mesh back through the shared :class:`ExternalToolBackend` spine (so volume /
bbox / manifold checks are computed from genuine geometry), and is fully
content-addressed.

What truck CAN do, honestly
---------------------------
* **extrude** -- ``builder::tsweep`` of a planar face. A box therefore has its
  EXACT analytic volume (60x40x20 = 48000, to the bit), and truck agrees with
  cadquery/freecad on it.
* **revolve** -- ``builder::rsweep`` of a planar face about a world axis (partial
  and full angles). A curved solid of revolution is a real NURBS surface, so its
  volume is near-analytic (faceted only by the tessellation tolerance).
* **boolean** union / intersect / cut -- ``truck_shapeops::or`` / ``and`` (cut is
  ``A AND complement(B)`` via ``Solid::not``). A 10-cube with a 4x4 slot cut
  through it is exactly 1000 - 160 = 840.
* **hole** (simple / counterbore) -- lowered to a boolean cut of cylinders.

What truck CANNOT do here, and is REFUSED (never faked)
-------------------------------------------------------
Refused at the op boundary with a typed ``unsupported-op`` (nothing mutates):

* ``fillet`` / ``chamfer`` -- truck has no edge-blend / setback builder;
* ``draft`` -- no face-taper operation;
* ``loft`` / ``sweep`` -- no skinning / sweep-along-path primitive exposed;
* ``shell`` -- truck has no hollow/thick-solid operation;
* ``linear_pattern`` / ``circular_pattern`` / ``mirror`` -- these need a robust
  union of (typically disjoint) bodies, and ``truck-shapeops`` 0.4 is
  experimental and unreliable for disjoint unions, so honouring them would be
  faking a result the kernel cannot stand behind;
* ``hole(kind='countersink')`` -- introduces a cone/frustum revolved to its own
  axis, which is a degenerate ``rsweep`` (the profile touches the axis) on this
  truck version, so it is refused rather than approximated.

A model whose booleans ``truck-shapeops`` genuinely cannot resolve fails at build
time and is surfaced as a ``kernel-error`` diagnostic from :meth:`regenerate` --
again, never a fabricated mesh.

STEP output
-----------
truck-stepio 0.3 can serialise a solid built by modeling (extrude/revolve) but
NOT the output of a boolean (its own README says so). So ``export('step')`` is
offered only for pure-modeling models; a model containing any boolean reports the
STEP as unavailable rather than writing an empty or wrong file.

Absence
-------
The Rust binary is discovered once, in ``locate()``. If it was never built (no
Rust toolchain, or ``cargo build --release`` was not run) the constructor raises
:class:`~harnesscad.io.backends.base.BackendUnavailable`, so the CISP server falls
back to the stub and the test suite SKIPs -- it never hangs and never fails
merely because the binary is absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Dict, List, Optional, Tuple

from harnesscad.core.cisp.ops import Hole
from harnesscad.io.backends.base import ApplyResult, BackendUnavailable
from harnesscad.io.backends.external import (
    ExternalToolBackend, ccw, circle_loop, plane_normal, profile_loops,
    signed_area, slab, to_world,
)
from harnesscad.io.backends.frep import Node

__all__ = ["TruckBackend", "TruckError"]

#: Environment override pointing straight at the compiled driver binary.
TRUCK_ENV = "HARNESSCAD_TRUCK_DRIVER"

#: Where the Rust crate lives, relative to this module.
_DRIVER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "truck_driver")

#: Tessellation tolerance (absolute, model units) handed to truck's
#: ``triangulation(tol)``. Small enough that a curved solid of revolution lands
#: within a fraction of a percent of its analytic volume; part of the digest.
TESS_TOL = 0.01

#: Boolean tolerance handed to ``truck-shapeops``; part of the digest.
BOOL_TOL = 0.05


class TruckError(RuntimeError):
    """The truck driver ran but produced no usable geometry for this model."""


def _binary_name() -> str:
    return "harnesscad_truck_driver" + (".exe" if os.name == "nt" else "")


def _default_binary_path() -> str:
    return os.path.join(_DRIVER_DIR, "target", "release", _binary_name())


# --------------------------------------------------------------------------
# nesting: split a set of CCW loops into (outer, [holes]) faces
# --------------------------------------------------------------------------
def _centroid(loop) -> Tuple[float, float]:
    n = len(loop)
    return (sum(p[0] for p in loop) / n, sum(p[1] for p in loop) / n)


def _point_in_loop(pt, loop) -> bool:
    """Even-odd ray cast: is ``pt`` inside the 2D polygon ``loop``?"""
    x, y = pt
    inside = False
    n = len(loop)
    j = n - 1
    for i in range(n):
        xi, yi = loop[i]
        xj, yj = loop[j]
        if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _faces_from_loops(loops2d) -> List[dict]:
    """Group 2D CCW loops into faces with holes by containment depth.

    A loop enclosed by an odd number of other loops is a HOLE; one enclosed by
    an even number (0, 2, ...) is an OUTER boundary. Each hole is attached to the
    smallest-area outer that contains it. This makes an annulus (two concentric
    circles) one face with one hole -- not two solid discs.
    """
    if not loops2d:
        return []
    cents = [_centroid(loop) for loop in loops2d]
    areas = [abs(signed_area(loop)) for loop in loops2d]
    containers: List[List[int]] = []
    for i, loop in enumerate(loops2d):
        encl = [k for k in range(len(loops2d))
                if k != i and _point_in_loop(cents[i], loops2d[k])]
        containers.append(encl)
    outers = [i for i in range(len(loops2d)) if len(containers[i]) % 2 == 0]
    holes = [i for i in range(len(loops2d)) if len(containers[i]) % 2 == 1]
    faces: Dict[int, dict] = {i: {"outer": loops2d[i], "holes": []} for i in outers}
    for h in holes:
        # smallest-area outer that contains this hole
        parents = [i for i in outers if i in containers[h]]
        if not parents:
            continue
        parent = min(parents, key=lambda i: areas[i])
        faces[parent]["holes"].append(loops2d[h])
    return [faces[i] for i in outers]


def _loop_to_world(plane: str, loop2d, w: float) -> List[List[float]]:
    return [list(to_world(plane, float(u), float(v), w)) for (u, v) in loop2d]


# --------------------------------------------------------------------------
# the backend
# --------------------------------------------------------------------------
class TruckBackend(ExternalToolBackend):
    """A GeometryBackend backed by the truck Rust B-rep NURBS kernel."""

    TOOL = "truck"

    #: Ops truck cannot honour HONESTLY -- refused with a typed diagnostic rather
    #: than approximated (see the module docstring for the per-op reason).
    UNSUPPORTED: Dict[str, str] = {
        "fillet": "truck has no edge-blend builder, so there is no operation to "
                  "round a named edge to a constant radius",
        "chamfer": "truck has no edge-setback builder, so a chamfer on a named "
                   "edge is not expressible",
        "draft": "truck has no face-taper (draft) operation about a neutral plane",
        "loft": "truck exposes no skinning/loft primitive between profiles in this "
                "driver",
        "sweep": "truck exposes no sweep-along-a-path primitive in this driver",
        "shell": "truck has no hollow / thick-solid operation, so a wall of a "
                 "given thickness cannot be carved without faking it",
        "linear_pattern": "a linear pattern is a union of (usually disjoint) "
                           "bodies, and truck-shapeops 0.4 is unreliable for "
                           "disjoint unions, so it is refused rather than faked",
        "circular_pattern": "a circular pattern is a union of (usually disjoint) "
                            "bodies, and truck-shapeops 0.4 is unreliable for "
                            "disjoint unions, so it is refused rather than faked",
        "mirror": "a mirror is a union of a body and its reflection, and "
                  "truck-shapeops 0.4 is unreliable for such unions, so it is "
                  "refused rather than faked",
    }
    FORMATS = ("stl", "stl-ascii", "stl-binary", "glb", "step")

    #: truck-shapeops has no 'intersection' (sharp) shell join and truck has no
    #: shell at all, so the only join declared is the default -- and shell is
    #: refused outright above, so this is never exercised.
    SHELL_JOINS = ("arc",)

    #: Memoised per-binary: (crate-versions-string, binary-sha256).
    _FINGERPRINTS: Dict[str, Tuple[str, str]] = {}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tess_tol = float(kwargs.get("tess_tol", TESS_TOL))
        self.bool_tol = float(kwargs.get("bool_tol", BOOL_TOL))

    # -- discovery ---------------------------------------------------------
    @classmethod
    def locate(cls) -> str:
        searched: List[str] = []
        override = os.environ.get(TRUCK_ENV)
        if override:
            searched.append("%s=%s" % (TRUCK_ENV, override))
            if os.path.isfile(override):
                return override
        default = _default_binary_path()
        searched.append(default)
        if os.path.isfile(default):
            return default
        raise BackendUnavailable(
            "truck",
            "the truck B-rep kernel driver binary is not built. Build it with a "
            "Rust toolchain: `cargo build --release` in %s (produces %s), or "
            "point %s at the binary. Searched: %s"
            % (_DRIVER_DIR, _default_binary_path(), TRUCK_ENV,
               ", ".join(searched)),
            searched)

    # -- fingerprint (crate versions + binary hash) ------------------------
    def _fingerprint(self) -> Tuple[str, str]:
        cached = type(self)._FINGERPRINTS.get(self.executable)
        if cached is not None:
            return cached
        versions = _crate_versions(os.path.join(_DRIVER_DIR, "Cargo.lock"))
        binhash = _file_sha256(self.executable)
        fp = (versions, binhash)
        type(self)._FINGERPRINTS[self.executable] = fp
        return fp

    def tool_version(self) -> str:
        versions, binhash = self._fingerprint()
        return "%s|bin=%s" % (versions, binhash[:16])

    def state_digest(self) -> str:
        """Content hash of the model AND everything that decides its geometry:
        the op stream, the tessellation and boolean tolerances, the truck crate
        versions AND the driver binary's own hash. (Blender and FreeCAD once
        omitted the tool version from their key and re-served stale geometry a
        newer build would no longer produce -- this key does not repeat that.)"""
        versions, binhash = self._fingerprint()
        blob = "%s|%s|tess=%g|bool=%g|crates=%s|bin=%s" % (
            self.TOOL, self._frep.state_digest(), self.tess_tol, self.bool_tol,
            versions, binhash)
        return hashlib.sha256(blob.encode()).hexdigest()

    # -- op admission ------------------------------------------------------
    def apply(self, op):
        """Refuse, BEFORE anything mutates, the one field-level case the base
        UNSUPPORTED table cannot express: a countersink hole, which would need a
        cone revolved to its own axis (a degenerate rsweep on this truck)."""
        if isinstance(op, Hole) and str(getattr(op, "kind", "")).lower() == "countersink":
            from harnesscad.eval.verifiers.verify import Diagnostic, Severity
            return ApplyResult(False, [], [Diagnostic(
                Severity.ERROR, "unsupported-op",
                "the truck backend does not implement a countersink hole: its "
                "conical mouth is a frustum revolved to its own axis, a "
                "degenerate rsweep on truck 0.6 -- refused rather than "
                "approximated", None)])
        return super().apply(op)

    # -- lowering: F-rep tree -> truck driver JSON job ---------------------
    def _lower(self, node: Node) -> dict:
        t = node.t
        if t == "extrude":
            lo, hi = slab(float(node.d["w0"]), float(node.d["w1"]))
            plane = node.d["plane"]
            n = plane_normal(plane)
            vec = [n[i] * (hi - lo) for i in range(3)]
            loops2d = profile_loops(node.d["profile"], self.segments)
            faces2d = _faces_from_loops(loops2d)
            faces = [{
                "outer": _loop_to_world(plane, f["outer"], lo),
                "holes": [_loop_to_world(plane, h, lo) for h in f["holes"]],
            } for f in faces2d]
            if not faces:
                raise TruckError("truck backend: empty extrude profile")
            return {"type": "extrude", "vector": vec, "faces": faces}
        if t == "cyl":
            lo, hi = slab(float(node.d["w0"]), float(node.d["w1"]))
            plane = node.d["plane"]
            n = plane_normal(plane)
            vec = [n[i] * (hi - lo) for i in range(3)]
            loop2d = ccw(circle_loop(
                float(node.d["cu"]), float(node.d["cv"]), float(node.d["r"]),
                self.segments))
            return {"type": "extrude", "vector": vec,
                    "faces": [{"outer": _loop_to_world(plane, loop2d, lo),
                               "holes": []}]}
        if t == "revolve":
            return self._lower_revolve(node)
        if t == "bool":
            return {"type": "boolean", "op": str(node.d["op"]),
                    "a": self._lower(node.d["a"]), "b": self._lower(node.d["b"])}
        if t == "cone":
            raise TruckError(
                "truck backend: a cone/frustum node reached lowering; it is only "
                "produced by a countersink hole, which this backend refuses")
        raise TruckError("truck backend: cannot lower F-rep node kind %r" % t)

    def _lower_revolve(self, node: Node) -> dict:
        """A revolve as a truck job: the planar profile mapped to world 3D and a
        world-space rotation axis. The profile lies in the sketch plane; the axis
        (given in-plane as (au, av, du, dv, ...)) is mapped to a world origin and
        world direction the same way, so the swept solid is placed like OCCT's."""
        plane = node.d["plane"]
        au, av, du, dv, _nu, _nv = (float(x) for x in node.d["axis"])
        angle = float(node.d.get("angle", 360.0))
        origin = list(to_world(plane, au, av, 0.0))
        p1 = to_world(plane, au + du, av + dv, 0.0)
        axis = [p1[i] - origin[i] for i in range(3)]
        loops2d = profile_loops(node.d["profile"], self.segments)
        faces2d = _faces_from_loops(loops2d)
        faces = [{
            "outer": _loop_to_world(plane, f["outer"], 0.0),
            "holes": [_loop_to_world(plane, h, 0.0) for h in f["holes"]],
        } for f in faces2d]
        if not faces:
            raise TruckError("truck backend: empty revolve profile")
        return {"type": "revolve", "faces": faces, "origin": origin,
                "axis": axis, "angle_deg": angle}

    def job(self) -> dict:
        root = self.root()
        if root is None:
            raise TruckError("truck backend: no solid to render")
        return {"tol": self.tess_tol, "bool_tol": self.bool_tol,
                "node": self._lower(root)}

    def program(self) -> str:
        """The driver job as canonical JSON (deterministic: sorted keys)."""
        return json.dumps(self.job(), sort_keys=True, separators=(",", ":"))

    # -- run the tool ------------------------------------------------------
    def _sidecar_path(self, workdir: str) -> str:
        return os.path.join(workdir, "model.json")

    def _read_sidecar(self, workdir: str) -> Optional[dict]:
        path = self._sidecar_path(workdir)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            return None

    def _run(self, source: str, workdir: str, out_path: str) -> None:
        """Drive the Rust binary and CLASSIFY the result.

        A failed run must leave NOTHING behind: the caller's cache is a file-exists
        check, so a stale STL from a crashed run would be re-served forever and
        reported as success. Clear before, clear after.
        """
        job_path = os.path.join(workdir, "job.json")
        with open(job_path, "w", encoding="utf-8") as fh:
            fh.write(source)
        for stale in ("model.stl", "model.json", "model.step"):
            p = os.path.join(workdir, stale)
            if os.path.isfile(p):
                os.remove(p)
        argv = [self.executable, job_path, workdir]
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=self.timeout)
        empty = not (os.path.isfile(out_path) and os.path.getsize(out_path) > 0)
        if proc.returncode != 0 or empty:
            sidecar = self._read_sidecar(workdir)
            reason = ""
            if sidecar is not None:
                reason = str(sidecar.get("reason", ""))
            if not reason:
                reason = (proc.stderr or proc.stdout or "").strip()
            for stale in ("model.stl", "model.json", "model.step"):
                p = os.path.join(workdir, stale)
                if os.path.isfile(p):
                    os.remove(p)
            raise TruckError(
                "truck driver failed (exit %d): %s" % (proc.returncode, reason))

    # -- B-rep metrics from the driver sidecar -----------------------------
    def brep_metrics(self) -> dict:
        """truck's OWN report of the last build: {volume, bbox, n_faces, n_edges,
        step}. Distinct from the mesh-derived metrics the base computes, these are
        the B-rep topology counts and truck's own closed-mesh volume -- the
        independent numbers the differential oracle can weigh against OCCT."""
        if self.root() is None:
            return {}
        # ensure the tool has run (populates the content-addressed workdir)
        self.stl_bytes()
        from harnesscad.io.backends.external import cache_dir, program_digest
        workdir = cache_dir(self.TOOL, program_digest(self.program()))
        sidecar = self._read_sidecar(workdir)
        return sidecar or {}

    def query(self, q: str) -> dict:
        if q == "brep":
            return self.brep_metrics()
        return super().query(q)

    # -- export ------------------------------------------------------------
    def export(self, fmt: str):
        if str(fmt).lower() == "step":
            return self._export_step()
        return super().export(fmt)

    def _export_step(self) -> bytes:
        if self.root() is None:
            raise TruckError("truck backend: no solid to export to STEP")
        self.stl_bytes()  # drives the tool, writing model.step when supported
        from harnesscad.io.backends.external import cache_dir, program_digest
        workdir = cache_dir(self.TOOL, program_digest(self.program()))
        step_path = os.path.join(workdir, "model.step")
        sidecar = self._read_sidecar(workdir)
        if not (sidecar and sidecar.get("step")) or not os.path.isfile(step_path):
            raise ValueError(
                "the truck backend cannot export this model to STEP: truck-stepio "
                "0.3 cannot serialise the output of a boolean operation, so STEP "
                "is available only for pure extrude/revolve models")
        with open(step_path, "rb") as fh:
            return fh.read()


# --------------------------------------------------------------------------
# helpers: crate versions + binary hash (both go into the cache key)
# --------------------------------------------------------------------------
def _crate_versions(lock_path: str) -> str:
    """A stable string of the pinned truck-* crate versions from Cargo.lock."""
    if not os.path.isfile(lock_path):
        return "unknown"
    versions: List[str] = []
    name: Optional[str] = None
    try:
        with open(lock_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("name = "):
                    name = line.split("=", 1)[1].strip().strip('"')
                elif line.startswith("version = ") and name and name.startswith("truck"):
                    ver = line.split("=", 1)[1].strip().strip('"')
                    versions.append("%s-%s" % (name, ver))
                    name = None
                elif line.startswith("[["):
                    name = None
    except Exception:  # noqa: BLE001
        return "unknown"
    return ",".join(sorted(versions)) if versions else "unknown"


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
    except Exception:  # noqa: BLE001
        return "unknown"
    return h.hexdigest()
