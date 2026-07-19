"""Metric booleans: manifold3d only, never OCCT.

THE POLICY
----------
**No verifier may compute a metric with an OCCT boolean.** Metric booleans go
through ``manifold3d``; :mod:`tests.eval.verifiers.test_metric_booleans_no_occt`
enforces this by scanning the source of every module in
:data:`METRIC_BOOLEAN_PATH` for OCCT boolean imports.

WHY (this is a live hazard, not hypothetical)
---------------------------------------------
cadgenbench hit a hang where its scorer wedged indefinitely on a raw OCCT BREP
boolean (``R & part``) over interface-overlay geometry -- two nearly-coincident
solids whose faces touch along large tangent regions, which is exactly what an
overlap/clearance check feeds a boolean. Their fix was structural rather than a
timeout: route every metric-side boolean through ``manifold3d``, a combinatorial
mesh kernel that is sub-millisecond and bounded by construction, and pin the
policy with a test so an OCCT boolean cannot silently return.

The same hazard was live here. ``interference._common_volume`` and
``access._swept_common_volume`` both called ``BRepAlgoAPI_Common`` on exactly
this geometry -- overlapping placed parts, and a tool cylinder swept flush
against a part face. This module is what they call instead.

TESSELLATION IS NOT BANNED, BOOLEANS ARE
----------------------------------------
Getting an OCCT solid into ``manifold3d`` needs OCCT to tessellate it
(``BRepMesh_IncrementalMesh``). That is fine and is what cadgenbench does too:
meshing is bounded and terminates. Only the *boolean* moves to manifold3d. So
this module imports from ``OCP.BRepMesh`` / ``OCP.BRep`` and the policy test
allows that, while banning ``OCP.BRepAlgoAPI`` (and ``build123d``, the other
route to an OCCT boolean) everywhere on the metric path.

SUB-EPSILON OVERLAP IS NUMERICAL NOISE
--------------------------------------
A boolean over tessellated geometry cannot resolve arbitrarily small overlaps:
two parts that merely touch produce a sliver whose "volume" is tessellation
residue, not a clash. :func:`classify_overlap` names that band explicitly
(``noise``) instead of letting a 1e-12 result read as a defect.

ATTRIBUTION
-----------
The policy, the manifold3d mesh-boolean approach, the mesh<->manifold conversion
and the sub-epsilon noise rule are taken from cadgenbench
(``resources/cad_repos/cadgenbench-main/cadgenbench-main/src/cadgenbench/eval/booleans.py``,
``.../eval/interface_match_viz.py`` and the policy test
``.../tests/eval/test_interface_viz_no_occt.py``), Copyright 2026 Hugging Face,
Apache License 2.0 -- a licence this repository's policy admits, so the
conversion logic is adapted directly with this attribution. The OCCT
tessellation + vertex weld and the OCCT-facing entry points are original: our
inputs are B-rep shapes, where cadgenbench's were already meshes.

Deterministic and lazily imported: with no manifold3d (or no OCCT) every entry
point returns ``None`` and the calling verifier degrades to its bounding-box
approximation, exactly as it did when a kernel call failed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "METRIC_BOOLEAN_PATH",
    "OCCT_BOOLEAN_MARKERS",
    "OVERLAP_NOISE_EPSILON",
    "TESSELLATION_DEFLECTION",
    "DEFAULT_BOOLEAN_BUDGET_S",
    "BOOLEAN_WORKER_MODULE",
    "MeshData",
    "BooleanBudgetResult",
    "classify_overlap",
    "common_volume",
    "intersection_volume",
    "intersection_volume_isolated",
    "manifold_available",
    "manifold_to_mesh_arrays",
    "mesh_to_manifold",
    "occt_boolean_offenders",
    "shape_to_mesh",
    "shape_to_manifold",
    "swept_cylinder_common_volume",
]


#: Every module whose metric booleans are governed by this policy. The enforcing
#: test scans exactly these. ADD a module here when it starts computing a metric
#: from a boolean; never remove one to make the test pass.
METRIC_BOOLEAN_PATH: Tuple[str, ...] = (
    "harnesscad.eval.verifiers.metric_booleans",
    "harnesscad.eval.verifiers.interference",
    "harnesscad.eval.verifiers.access",
)

#: Import fragments that mean "an OCCT boolean can reach this module". The
#: boolean builders live in BRepAlgoAPI (Common/Cut/Fuse/Section); build123d and
#: cadquery both expose OCCT booleans through operators, which is precisely how
#: one returns silently.
OCCT_BOOLEAN_MARKERS: Tuple[str, ...] = (
    "BRepAlgoAPI",
    "BOPAlgo",
    "build123d",
)

#: Overlap volumes at or below this (mm^3) are tessellation residue from faces
#: that merely touch, not a real clash. cadgenbench uses 1.0 mm^3 for the same
#: call. Verifiers may pass a stricter epsilon; they may not pass zero and call
#: the result exact.
OVERLAP_NOISE_EPSILON = 1.0

#: Linear deflection (mm) for the OCCT tessellation feeding the boolean. Small
#: enough that a planar part is exact and a curved one is well within the noise
#: epsilon at typical part scales.
TESSELLATION_DEFLECTION = 0.05

#: Angular deflection (radians) for the same tessellation.
TESSELLATION_ANGULAR = 0.5

#: Default wall-clock budget (seconds) for ONE isolated metric boolean.
#:
#: A normal manifold3d intersection is sub-millisecond (measured min/mean/max over
#: overlapping 10mm cubes: 0.07 / 0.21 / 0.96 ms; cadgenbench calls it "sub-
#: millisecond and bounded"). Upstream manifold's own fuzzers watchdog every
#: boolean at 10s. This budget is 20s -- twice the upstream watchdog and ~20000x a
#: normal boolean -- plus it must also cover the child's interpreter start and
#: ``import manifold3d`` (~0.2s here). So it never trips on real geometry and
#: still bounds a wedged one. It applies only when a caller OPTS IN to isolation
#: (passes ``budget_s``); the in-process fast path is unchanged and untimed.
DEFAULT_BOOLEAN_BUDGET_S = 20.0

#: Worker module run as ``python -m`` in the child to compute one boolean.
BOOLEAN_WORKER_MODULE = "harnesscad.eval.verifiers._metric_boolean_worker"

#: Every status an isolated boolean can report. ``ok``/``empty`` mean the boolean
#: RAN; the rest mean it did not, each for a different reason (and each yields
#: ``volume=None`` so the caller's ``classify_overlap`` reads it as ``unknown``).
BOOLEAN_STATUSES = ("ok", "empty", "timeout", "unavailable", "error")


@dataclass(frozen=True)
class MeshData:
    """A triangle soup welded into a shared-vertex mesh.

    ``vertices`` is a list of (x, y, z); ``triangles`` a list of (i, j, k) index
    triples. Plain Python so the module has no numpy import at module scope.
    """

    vertices: List[Tuple[float, float, float]]
    triangles: List[Tuple[int, int, int]]

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)


def manifold_available() -> bool:
    """True when the manifold3d kernel can be imported."""
    try:
        import manifold3d  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False


# --------------------------------------------------------------------------- #
# OCCT shape -> mesh (tessellation is allowed; booleans are not)
# --------------------------------------------------------------------------- #
def _weld_key(x: float, y: float, z: float, tol: float) -> Tuple[int, int, int]:
    """Quantise a point onto a tolerance grid so coincident vertices collapse.

    OCCT tessellates each face independently, so a shared edge yields two sets of
    identical-but-distinct vertices and the raw soup is not manifold. manifold3d
    requires a closed 2-manifold, so the soup must be welded first. Rounding to a
    grid is deterministic (no floating-point-order dependence, unlike a
    nearest-neighbour weld).
    """
    inv = 1.0 / tol
    return (int(round(x * inv)), int(round(y * inv)), int(round(z * inv)))


def shape_to_mesh(shape, deflection: float = TESSELLATION_DEFLECTION,
                  weld_tolerance: float = 1e-7) -> Optional[MeshData]:
    """Tessellate an OCCT/cadquery shape into a welded :class:`MeshData`.

    Returns ``None`` when OCCT is unavailable or the shape cannot be meshed.
    Never raises. This is the only OCCT-touching step on the metric path, and it
    is a mesher, not a boolean.
    """
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
    except Exception:  # noqa: BLE001 - no OCCT -> caller degrades
        return None

    wrapped = getattr(shape, "wrapped", shape)
    if wrapped is None:
        return None

    try:
        BRepMesh_IncrementalMesh(wrapped, deflection, False,
                                 TESSELLATION_ANGULAR, True)
    except Exception:  # noqa: BLE001 - an unmeshable shape degrades
        return None

    vertices: List[Tuple[float, float, float]] = []
    triangles: List[Tuple[int, int, int]] = []
    index: Dict[Tuple[int, int, int], int] = {}

    def _vertex(px: float, py: float, pz: float) -> int:
        key = _weld_key(px, py, pz, weld_tolerance)
        found = index.get(key)
        if found is None:
            found = len(vertices)
            index[key] = found
            vertices.append((px, py, pz))
        return found

    try:
        explorer = TopExp_Explorer(wrapped, TopAbs_FACE)
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            explorer.Next()
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, location)
            if triangulation is None:
                continue
            transform = location.Transformation()
            reversed_face = face.Orientation() == TopAbs_REVERSED

            local: List[int] = []
            for i in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(i).Transformed(transform)
                local.append(_vertex(point.X(), point.Y(), point.Z()))

            for i in range(1, triangulation.NbTriangles() + 1):
                a, b, c = triangulation.Triangle(i).Get()
                tri = (local[a - 1], local[b - 1], local[c - 1])
                if reversed_face:
                    # Keep the outward winding manifold3d expects.
                    tri = (tri[0], tri[2], tri[1])
                if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                    continue  # welded-away degenerate
                triangles.append(tri)
    except Exception:  # noqa: BLE001 - a bad triangulation degrades
        return None

    if not triangles:
        return None
    return MeshData(vertices=vertices, triangles=triangles)


# --------------------------------------------------------------------------- #
# mesh -> manifold  (adapted from cadgenbench eval/booleans.py, Apache-2.0)
# --------------------------------------------------------------------------- #
def mesh_to_manifold(mesh: MeshData):
    """Ingest a :class:`MeshData` into a ``manifold3d.Manifold``, or ``None``.

    Returns ``None`` (never raises) when manifold3d is missing or rejects the
    mesh -- a rejected mesh means the input was not a closed 2-manifold, and a
    metric computed from it would be meaningless, so the caller must degrade
    rather than guess.
    """
    if mesh is None or mesh.n_triangles == 0:
        return None
    try:
        import manifold3d as m3d
        import numpy as np

        md_mesh = m3d.Mesh(
            vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
            tri_verts=np.ascontiguousarray(mesh.triangles, dtype=np.uint32),
        )
        manifold = m3d.Manifold(md_mesh)
        status = manifold.status
        # The status enum's spelling varies across manifold3d releases; compare
        # by name so this stays forward-compatible (cadgenbench does the same).
        if hasattr(status, "name") and status.name != "NoError":
            return None
        if manifold.is_empty():
            return None
        return manifold
    except Exception:  # noqa: BLE001 - no kernel / bad mesh -> caller degrades
        return None


def shape_to_manifold(shape, deflection: float = TESSELLATION_DEFLECTION):
    """Tessellate an OCCT shape and ingest it into manifold3d, or ``None``."""
    return mesh_to_manifold(shape_to_mesh(shape, deflection))


# --------------------------------------------------------------------------- #
# The metric booleans themselves
# --------------------------------------------------------------------------- #
def intersection_volume(a, b) -> Optional[float]:
    """Volume of the manifold3d intersection of two manifolds, or ``None``.

    ``a ^ b`` is manifold3d's intersection operator. Bounded and deterministic:
    this is the call that replaced the OCCT boolean that hung.
    """
    if a is None or b is None:
        return None
    try:
        common = a ^ b
        if common.is_empty():
            return 0.0
        return abs(float(common.volume()))
    except Exception:  # noqa: BLE001 - a kernel refusal degrades
        return None


# --------------------------------------------------------------------------- #
# The time-budgeted boolean (subprocess isolation, step_check.py pattern)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BooleanBudgetResult:
    """The outcome of one budget-bounded, out-of-process metric boolean.

    ``volume`` is the overlap volume ONLY when ``status`` is ``ok`` (or ``0.0``
    for ``empty``); for every refusal -- ``timeout``, ``unavailable``, ``error``
    -- it is ``None``, so passing ``.volume`` straight to :func:`classify_overlap`
    yields ``unknown``. A timed-out boolean is UNKNOWN, never a fabricated number.
    """

    status: str
    volume: Optional[float]
    elapsed_s: float = 0.0
    returncode: Optional[int] = None
    timed_out: bool = False
    killed: bool = False
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "empty")


def manifold_to_mesh_arrays(manifold):
    """A ``manifold3d.Manifold`` as ``(vert_properties, tri_verts)`` numpy arrays.

    The two operands of an isolated boolean cross the process boundary as plain
    vertex/triangle arrays (``Manifold.to_mesh()``), which is exactly why the
    boolean CAN be isolated: a mesh marshals, a live kernel handle does not.
    Returns ``None`` when manifold3d/numpy are missing or the solid has no mesh.
    """
    if manifold is None:
        return None
    try:
        import numpy as np

        mesh = manifold.to_mesh()
        vp = np.array(mesh.vert_properties, dtype=np.float32, order="C")
        tv = np.array(mesh.tri_verts, dtype=np.uint32, order="C")
        if vp.shape[0] == 0 or tv.shape[0] == 0:
            return None
        return vp, tv
    except Exception:  # noqa: BLE001 - unmeshable -> caller degrades to unavailable
        return None


def intersection_volume_isolated(
    a, b,
    budget_s: float = DEFAULT_BOOLEAN_BUDGET_S,
    worker_cmd: Optional[Sequence[str]] = None,
    python_executable: Optional[str] = None,
) -> BooleanBudgetResult:
    """Compute ``a ^ b``'s volume in a CHILD process under ``budget_s`` seconds.

    The honest mechanism, stated plainly: Python cannot interrupt an in-thread
    C-extension call, so a thread+join watchdog could only DETECT a wedged
    ``a ^ b`` -- never kill it. This runs the boolean in a separate PROCESS
    (:data:`BOOLEAN_WORKER_MODULE`) and, on overrun, KILLS and reaps it, which is
    the only real kill for a wedged kernel call. It is the same isolation
    :mod:`harnesscad.io.ingest.step_check` uses for untrusted STEP files.

    A timeout RETURNS a refusal (``status='timeout'``, ``volume=None``); it never
    raises into the caller and never invents a volume. ``worker_cmd`` overrides
    the default worker (the ``.npz`` input path is appended as the final argument),
    which is what makes the kill+reap path testable: a self-check can point the
    real worker at a stall and watch the parent survive it.
    """
    arrays_a = manifold_to_mesh_arrays(a)
    arrays_b = manifold_to_mesh_arrays(b)
    if arrays_a is None or arrays_b is None:
        return BooleanBudgetResult(
            status="unavailable", volume=None,
            note="an operand had no marshallable mesh (no kernel or empty solid)")

    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        return BooleanBudgetResult(status="unavailable", volume=None,
                                   note="numpy absent: %s" % exc)

    fd, npz_path = tempfile.mkstemp(suffix=".npz", prefix="hc_metric_bool_")
    os.close(fd)
    start = time.monotonic()
    try:
        np.savez(npz_path, va=arrays_a[0], ta=arrays_a[1],
                 vb=arrays_b[0], tb=arrays_b[1])

        cmd = (list(worker_cmd) if worker_cmd
               else [python_executable or sys.executable, "-m",
                     BOOLEAN_WORKER_MODULE])
        cmd = cmd + [npz_path]

        child_env = dict(os.environ)
        # The child imports harnesscad; inherit the parent's resolution so a
        # source checkout works without an install step (as step_check does).
        child_env.setdefault("PYTHONPATH", os.pathsep.join(
            p for p in sys.path if p and os.path.isdir(p)))

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=child_env)
        except OSError as exc:
            return BooleanBudgetResult(
                status="error", volume=None, elapsed_s=time.monotonic() - start,
                note="cannot start worker: %s: %s" % (type(exc).__name__, exc))

        try:
            stdout, stderr = proc.communicate(timeout=budget_s)
        except subprocess.TimeoutExpired:
            # The whole point: the wedged boolean's process dies, this one lives.
            proc.kill()
            try:
                _, stderr = proc.communicate(timeout=10.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - unkillable child
                stderr = "worker did not die after kill"
            return BooleanBudgetResult(
                status="timeout", volume=None,
                elapsed_s=time.monotonic() - start, returncode=proc.returncode,
                timed_out=True, killed=True,
                note="boolean exceeded %g s and was killed" % budget_s)

        elapsed = time.monotonic() - start
        line = ""
        for candidate in reversed((stdout or "").strip().splitlines()):
            if candidate.strip():
                line = candidate.strip()
                break
        if not line:
            return BooleanBudgetResult(
                status="error", volume=None, elapsed_s=elapsed,
                returncode=proc.returncode,
                note="worker produced no report (exit %s): %s"
                     % (proc.returncode, (stderr or "")[-200:]))
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("worker report is not a JSON object")
        except Exception as exc:  # noqa: BLE001
            return BooleanBudgetResult(
                status="error", volume=None, elapsed_s=elapsed,
                returncode=proc.returncode,
                note="unreadable worker report: %s" % exc)

        status = str(payload.get("status", "error"))
        if status not in BOOLEAN_STATUSES:
            status = "error"
        volume = payload.get("volume") if status in ("ok", "empty") else None
        if volume is not None:
            volume = abs(float(volume))
        return BooleanBudgetResult(
            status=status, volume=volume, elapsed_s=elapsed,
            returncode=proc.returncode, note=str(payload.get("note", "")))
    finally:
        try:
            os.remove(npz_path)
        except OSError:  # pragma: no cover - best-effort cleanup
            pass


def classify_overlap(volume: Optional[float],
                     epsilon: float = OVERLAP_NOISE_EPSILON) -> str:
    """Classify an overlap volume: ``'unknown'``, ``'none'``, ``'noise'`` or ``'clash'``.

    ``None`` -> ``'unknown'`` (not measurable; say so rather than imply zero).
    Exactly zero -> ``'none'``. Above zero but ``<= epsilon`` -> ``'noise'``:
    tessellation residue from faces that touch, which must not be reported as a
    defect. Above epsilon -> ``'clash'``.
    """
    if volume is None:
        return "unknown"
    if volume <= 0.0:
        return "none"
    if volume <= epsilon:
        return "noise"
    return "clash"


def common_volume(shape_a, shape_b,
                  deflection: float = TESSELLATION_DEFLECTION,
                  budget_s: Optional[float] = None) -> Optional[float]:
    """Overlap volume of two OCCT solids, computed with manifold3d.

    The drop-in replacement for an OCCT ``BRepAlgoAPI_Common`` + ``VolumeProperties``
    pair. Returns ``None`` when the geometry cannot be measured (no kernel,
    unmeshable shape, non-manifold input), which every caller already handles by
    falling back to its bounding-box approximation.

    ``budget_s`` is the one-line, default-OFF opt-in to the watchdog: leave it
    ``None`` (the default) and the boolean runs in-process exactly as before; pass
    a budget (e.g. :data:`DEFAULT_BOOLEAN_BUDGET_S`) and the boolean runs in a
    child process that is KILLED on overrun, returning ``None`` (an UNKNOWN
    overlap) rather than wedging the caller. Tessellation stays in-process because
    it is bounded; only the hang-prone boolean is isolated.
    """
    a = shape_to_manifold(shape_a, deflection)
    if a is None:
        return None
    b = shape_to_manifold(shape_b, deflection)
    if b is None:
        return None
    if budget_s is None:
        return intersection_volume(a, b)
    return intersection_volume_isolated(a, b, budget_s=budget_s).volume


def swept_cylinder_common_volume(pos: Sequence[float], axis: Sequence[float],
                                 radius: float, length: float, part_shape,
                                 deflection: float = TESSELLATION_DEFLECTION,
                                 segments: int = 64,
                                 budget_s: Optional[float] = None) -> Optional[float]:
    """Overlap volume of a swept tool cylinder and a part, computed with manifold3d.

    The cylinder is built natively in manifold3d (``Manifold.cylinder`` along +Z,
    then rotated onto ``axis`` and translated to ``pos``), so no OCCT primitive
    or boolean is involved. ``segments`` fixes the circular tessellation, making
    the result deterministic and slightly conservative (an inscribed polygon
    under-reports a curved overlap rather than inventing one).

    ``budget_s`` is the same default-OFF watchdog opt-in as :func:`common_volume`:
    ``None`` keeps the in-process boolean; a budget isolates it in a killable
    child, returning ``None`` (UNKNOWN) on overrun instead of hanging.
    """
    if radius <= 0.0 or length <= 0.0:
        return None
    part = shape_to_manifold(part_shape, deflection)
    if part is None:
        return None
    tool = _cylinder_manifold(pos, axis, radius, length, segments)
    if tool is None:
        return None
    if budget_s is None:
        return intersection_volume(tool, part)
    return intersection_volume_isolated(tool, part, budget_s=budget_s).volume


def _cylinder_manifold(pos: Sequence[float], axis: Sequence[float],
                       radius: float, length: float, segments: int):
    """A manifold3d cylinder of ``radius``/``length`` based at ``pos`` along ``axis``."""
    try:
        import manifold3d as m3d
        import numpy as np

        direction = np.asarray(axis, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            return None
        direction = direction / norm

        # Rotation taking +Z onto `direction` (Rodrigues; the anti-parallel case
        # has no unique axis, so pick a deterministic one).
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(z, direction)
        c = float(np.dot(z, direction))
        s = float(np.linalg.norm(v))
        if s < 1e-12:
            rotation = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array([[0.0, -v[2], v[1]],
                           [v[2], 0.0, -v[0]],
                           [-v[1], v[0], 0.0]])
            rotation = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))

        affine = np.zeros((3, 4), dtype=np.float32)
        affine[:3, :3] = rotation
        affine[:3, 3] = np.asarray(pos, dtype=np.float64)
        cylinder = m3d.Manifold.cylinder(length, radius, radius, segments, False)
        return cylinder.transform(np.ascontiguousarray(affine, dtype=np.float32))
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# The policy scanner (the enforcing test's engine)
# --------------------------------------------------------------------------- #
def occt_boolean_offenders(
        modules: Sequence[str] = METRIC_BOOLEAN_PATH) -> Dict[str, List[str]]:
    """Scan each module's SOURCE for OCCT boolean markers.

    Returns ``{module_name: [offending lines]}`` -- empty when the policy holds.
    Source-level, not import-level, on purpose: a lazy ``from OCP.BRepAlgoAPI
    import ...`` inside a function body is invisible to an import-time check, yet
    it is exactly how the hang re-enters.

    The scan tokenises the source and looks only at NAME/OP tokens, so comments
    and docstrings that merely *discuss* the banned API (as this module's own do,
    at length) are not offences -- only real code is. A module whose source
    cannot be read or tokenised is reported as an offender rather than silently
    passing: the policy fails closed.
    """
    import importlib
    import io
    import token as token_module
    import tokenize
    from pathlib import Path

    offenders: Dict[str, List[str]] = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
            source = Path(module.__file__).read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - an unreadable module fails closed
            offenders[name] = ["could not read source: %s" % exc]
            continue

        hits: List[str] = []
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        except Exception as exc:  # noqa: BLE001 - untokenisable source fails closed
            offenders[name] = ["could not tokenise source: %s" % exc]
            continue

        lines = source.splitlines()
        for tok in tokens:
            if tok.type != token_module.NAME:
                continue
            for marker in OCCT_BOOLEAN_MARKERS:
                if marker in tok.string:
                    lineno = tok.start[0]
                    text = lines[lineno - 1].strip() if lineno <= len(lines) else tok.string
                    hits.append("%s:%d: %s" % (name, lineno, text))
                    break
        if hits:
            offenders[name] = hits
    return offenders


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` proves the policy holds and that the
    manifold3d metric booleans return the right volumes on known geometry."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.eval.verifiers.metric_booleans",
        description="Metric booleans via manifold3d (never OCCT) + the policy scanner.",
    )
    parser.add_argument(
        "--selfcheck", action="store_true",
        help="run deterministic checks over the policy and the booleans; exit 0 on success.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    failures: List[str] = []
    checks = 0

    def check(label: str, condition: bool) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(label)

    # 1. THE POLICY: no module on the metric path may reach an OCCT boolean.
    offenders = occt_boolean_offenders()
    check("no OCCT booleans on the metric path (%s)"
          % "; ".join("%s -> %s" % (k, v[0]) for k, v in offenders.items()),
          not offenders)

    # 2. The noise band is named, not silently zeroed.
    check("None is unknown, not zero", classify_overlap(None) == "unknown")
    check("zero is none", classify_overlap(0.0) == "none")
    check("sub-epsilon is noise", classify_overlap(1e-9) == "noise")
    check("epsilon itself is still noise",
          classify_overlap(OVERLAP_NOISE_EPSILON) == "noise")
    check("above epsilon is a clash",
          classify_overlap(OVERLAP_NOISE_EPSILON * 1.001) == "clash")
    check("a stricter epsilon promotes noise to a clash",
          classify_overlap(0.5, epsilon=0.1) == "clash")

    # 3. Missing kernels degrade to None, never to a wrong number.
    check("intersection of nothing is None", intersection_volume(None, None) is None)
    check("common_volume of None is None", common_volume(None, None) is None)
    check("mesh_to_manifold(None) is None", mesh_to_manifold(None) is None)

    if not manifold_available():
        if failures:
            print("SELFCHECK FAILED: %s" % ", ".join(failures), file=sys.stderr)
            return 1
        print("PASS: metric_booleans selfcheck (%d checks; manifold3d absent, "
              "kernel checks skipped)" % checks)
        return 0

    # 4. REAL KERNEL PROPERTY: manifold3d intersection volumes on known cubes.
    #    Two 10mm cubes offset 5mm on each axis share a 5mm cube = 125 mm^3.
    import manifold3d as m3d
    cube_a = m3d.Manifold.cube([10.0, 10.0, 10.0], center=False)
    cube_b = m3d.Manifold.cube([10.0, 10.0, 10.0], center=False).translate([5.0, 5.0, 5.0])
    vol = intersection_volume(cube_a, cube_b)
    check("overlapping cubes intersect in 125 mm^3",
          vol is not None and abs(vol - 125.0) < 1e-3)
    check("a real overlap classifies as a clash", classify_overlap(vol) == "clash")

    far = m3d.Manifold.cube([10.0, 10.0, 10.0], center=False).translate([100.0, 0.0, 0.0])
    vol = intersection_volume(cube_a, far)
    check("disjoint cubes intersect in nothing", vol == 0.0)
    check("no overlap classifies as none", classify_overlap(vol) == "none")

    # A barely-touching pair: 0.5mm cube = 0.125 mm^3, under the epsilon.
    graze = m3d.Manifold.cube([10.0, 10.0, 10.0], center=False).translate([9.5, 9.5, 9.5])
    vol = intersection_volume(cube_a, graze)
    check("a grazing overlap is sub-epsilon",
          vol is not None and 0.0 < vol < OVERLAP_NOISE_EPSILON)
    check("a grazing overlap classifies as noise", classify_overlap(vol) == "noise")

    # 5. REAL KERNEL PROPERTY: the OCCT->manifold bridge preserves volume, i.e.
    #    the tessellate+weld really does produce a closed 2-manifold.
    try:
        import cadquery as cq
    except Exception:  # noqa: BLE001
        cq = None

    if cq is not None:
        box = cq.Workplane("XY").box(10, 10, 10).val()
        mesh = shape_to_mesh(box)
        check("an OCCT box tessellates", mesh is not None and mesh.n_triangles >= 12)
        manifold = shape_to_manifold(box)
        check("a welded OCCT box is accepted by manifold3d", manifold is not None)
        if manifold is not None:
            check("the bridge preserves the box volume",
                  abs(float(manifold.volume()) - 1000.0) < 1e-3)

        # Two OCCT boxes overlapping in a known 4x10x10 = 400 mm^3 slab.
        b1 = cq.Workplane("XY").box(10, 10, 10).val()
        b2 = cq.Workplane("XY").box(10, 10, 10).translate((6, 0, 0)).val()
        vol = common_volume(b1, b2)
        check("OCCT solids overlap in 400 mm^3 via manifold3d",
              vol is not None and abs(vol - 400.0) < 1e-2)

        # The default-OFF opt-in: the SAME overlap, but with the boolean isolated
        # in a killable child, must give the SAME 400 mm^3 -- proving the one-line
        # opt-in on common_volume routes through the watchdog without changing the
        # answer on normal geometry.
        vol = common_volume(b1, b2, budget_s=DEFAULT_BOOLEAN_BUDGET_S)
        check("common_volume opt-in (budget_s) gives the same 400 mm^3",
              vol is not None and abs(vol - 400.0) < 1e-2)

        # The swept-cylinder entry point: a r=2 tool driven 10mm down the -Z
        # axis from above the box passes through 5mm of it => 5*pi*4 mm^3.
        vol = swept_cylinder_common_volume((0, 0, 10), (0, 0, -1), 2.0, 10.0, b1)
        import math
        check("a swept tool cylinder intersects the expected volume",
              vol is not None and abs(vol - 5.0 * math.pi * 4.0) < 1.0)

        # A tool pointed away from the part hits nothing.
        vol = swept_cylinder_common_volume((0, 0, 10), (0, 0, 1), 2.0, 10.0, b1)
        check("a tool aimed away from the part intersects nothing", vol == 0.0)

    # 6. THE WATCHDOG -- both arms, out of process.
    #
    #    ARM A (normal boolean completes untouched): the SAME two overlapping
    #    cubes (125 mm^3), but through the killable child worker. It must return
    #    the right volume, status ok, and a reaped exit 0.
    ra = intersection_volume_isolated(cube_a, cube_b, budget_s=DEFAULT_BOOLEAN_BUDGET_S)
    check("isolated boolean returns the right volume (125 mm^3)",
          ra.status == "ok" and ra.volume is not None
          and abs(ra.volume - 125.0) < 1e-3)
    check("isolated boolean's worker was reaped (returncode 0)",
          ra.returncode == 0)
    check("isolated volume still classifies as a clash",
          classify_overlap(ra.volume) == "clash")
    print("[selfcheck] normal isolated boolean: status=%s volume=%.3f "
          "worker_exit=%s (%.2fs, out-of-process)"
          % (ra.status, ra.volume or -1.0, ra.returncode, ra.elapsed_s))

    #    ARM B (a hanging boolean is caught): point the REAL worker at a 60s
    #    stall and give it a 1s budget. It must be KILLED and reaped -- returning
    #    a timeout refusal with volume=None (UNKNOWN), not a hang and not a wrong
    #    number. The parent must come back in about the budget, not in 60s.
    stall_worker = [sys.executable, "-m", BOOLEAN_WORKER_MODULE, "--stall", "60"]
    t0 = time.monotonic()
    rb = intersection_volume_isolated(cube_a, cube_b, budget_s=1.0,
                                      worker_cmd=stall_worker)
    elapsed = time.monotonic() - t0
    check("a hanging boolean times out (not hangs)", rb.status == "timeout")
    check("a timed-out boolean returns UNKNOWN, not a number", rb.volume is None)
    check("a timed-out boolean classifies as unknown",
          classify_overlap(rb.volume) == "unknown")
    check("the timed-out worker was killed", rb.timed_out and rb.killed)
    check("the killed worker was reaped (returncode set)",
          rb.returncode is not None)
    check("the parent returned near the budget, not the worker's 60s",
          elapsed < 30.0)
    print("[selfcheck] hanging isolated boolean: status=%s volume=%s "
          "killed=%s worker_exit=%s -- parent back in %.2fs (worker wanted 60s)"
          % (rb.status, rb.volume, rb.killed, rb.returncode, elapsed))

    if failures:
        print("SELFCHECK FAILED: %s" % ", ".join(failures), file=sys.stderr)
        return 1
    print("PASS: metric_booleans selfcheck (%d checks; policy holds, manifold3d "
          "volumes correct, OCCT->manifold bridge volume-preserving)" % checks)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
