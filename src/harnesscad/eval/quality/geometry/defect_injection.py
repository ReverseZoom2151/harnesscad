"""Injected-defect benchmark for geometry verifiers (Roshera certificate).

**Roshera** is a Rust B-Rep kernel whose thesis is: *every operation returns a
validity certificate -- the kernel cannot lie*. To prove a certificate is worth
more than "the mesh looks closed", Roshera ships a benchmark
(``geometry-engine/tests/injected_defect_benchmark.rs``) that injects **four
classes of silent geometric lie** into sound parts and asks each verifier how
many it catches:

* B-Rep validity check alone -> 0 / 4
* "mesh looks closed" heuristic -> 2 / 4
* Roshera's certificate -> 4 / 4

This module reimplements that *evaluation taxonomy* -- not another manifold
checker (the harness already has one in
:mod:`harnesscad.domain.geometry.mesh.polyhedron`), but the deterministic
**defect injectors** and the **scoring harness** that turns any verifier into a
catch matrix. It answers the meta-question Roshera's benchmark poses: *does this
verifier actually catch each class of lie?*

The four lie classes (Roshera's ``DefectClass``):

* ``flipped_normal``   -- one facet wound backwards (inconsistent orientation);
* ``torn_seam``        -- a shared vertex split so an edge becomes a boundary
                          (a hole the render hides);
* ``non_manifold``     -- a third facet glued onto an interior edge;
* ``degenerate_facet`` -- a facet collapsed to zero area (self-intersection
                          proxy -- a facet that cannot carry a real normal).

A mesh is ``(vertices, faces)`` with ``faces`` as index tuples. A *verifier* is
any ``callable(mesh) -> bool`` returning ``True`` when it judges the mesh SOUND.
:func:`run_benchmark` injects each class into a clean base mesh and records
whether the verifier flipped to "unsound" -- i.e. caught the lie. A built-in
:func:`topology_verifier` (edge-incidence + orientation) is provided so the
harness is runnable and testable on its own; external verifiers plug in the same
way.

Pure stdlib, deterministic (injectors are seedable but default to a fixed order).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

Vertex = Tuple[float, float, float]
Face = Tuple[int, ...]

__all__ = [
    "DEFECT_CLASSES",
    "Mesh",
    "DefectError",
    "inject_flipped_normal",
    "inject_torn_seam",
    "inject_non_manifold",
    "inject_degenerate_facet",
    "INJECTORS",
    "topology_verifier",
    "BenchmarkResult",
    "run_benchmark",
    "unit_tetrahedron",
    "unit_cube_mesh",
]

DEFECT_CLASSES: Tuple[str, ...] = (
    "flipped_normal",
    "torn_seam",
    "non_manifold",
    "degenerate_facet",
)


class DefectError(ValueError):
    """A defect cannot be injected into the given mesh."""


@dataclass(frozen=True)
class Mesh:
    """A triangle/polygon soup: vertices + face index tuples."""

    vertices: Tuple[Vertex, ...]
    faces: Tuple[Face, ...]

    @staticmethod
    def of(vertices: Sequence[Vertex], faces: Sequence[Sequence[int]]) -> "Mesh":
        return Mesh(
            tuple((float(a), float(b), float(c)) for a, b, c in vertices),
            tuple(tuple(int(i) for i in f) for f in faces),
        )


# --------------------------------------------------------------------------- #
# Injectors                                                                    #
# --------------------------------------------------------------------------- #
def inject_flipped_normal(mesh: Mesh, face_index: int = 0) -> Mesh:
    """Reverse the winding of one face -> inconsistent orientation."""
    if not mesh.faces:
        raise DefectError("mesh has no faces to flip")
    faces = list(mesh.faces)
    fi = face_index % len(faces)
    faces[fi] = tuple(reversed(faces[fi]))
    return Mesh(mesh.vertices, tuple(faces))


def inject_torn_seam(mesh: Mesh, face_index: int = 0) -> Mesh:
    """Duplicate one vertex of a face and re-point that face at the copy.

    The original edge loses a face and becomes a boundary edge -- a torn seam
    (a hole) that a render can hide but a topology check must catch.
    """
    if not mesh.faces:
        raise DefectError("mesh has no faces to tear")
    faces = list(mesh.faces)
    fi = face_index % len(faces)
    face = list(faces[fi])
    if not face:
        raise DefectError("empty face")
    verts = list(mesh.vertices)
    original = face[0]
    verts.append(verts[original])  # coincident duplicate
    face[0] = len(verts) - 1
    faces[fi] = tuple(face)
    return Mesh(tuple(verts), tuple(faces))


def inject_non_manifold(mesh: Mesh) -> Mesh:
    """Glue an extra facet onto an existing interior edge (3 faces share it)."""
    if not mesh.faces or len(mesh.faces[0]) < 2:
        raise DefectError("mesh has no usable edge")
    verts = list(mesh.vertices)
    a, b = mesh.faces[0][0], mesh.faces[0][1]
    # A new apex vertex nudged off the edge, forming a third triangle on (a,b).
    ax, ay, az = verts[a]
    bx, by, bz = verts[b]
    apex = ((ax + bx) / 2.0, (ay + by) / 2.0 + 1.0, (az + bz) / 2.0 + 1.0)
    verts.append(apex)
    faces = list(mesh.faces)
    faces.append((a, b, len(verts) - 1))
    return Mesh(tuple(verts), tuple(faces))


def inject_degenerate_facet(mesh: Mesh, face_index: int = 0) -> Mesh:
    """Collapse one face to zero area by repeating a vertex index."""
    if not mesh.faces:
        raise DefectError("mesh has no faces")
    faces = list(mesh.faces)
    fi = face_index % len(faces)
    face = list(faces[fi])
    if len(face) < 2:
        raise DefectError("face too small to degenerate")
    face[-1] = face[0]  # repeat -> zero area
    faces[fi] = tuple(face)
    return Mesh(mesh.vertices, tuple(faces))


INJECTORS: Dict[str, Callable[[Mesh], Mesh]] = {
    "flipped_normal": inject_flipped_normal,
    "torn_seam": inject_torn_seam,
    "non_manifold": inject_non_manifold,
    "degenerate_facet": inject_degenerate_facet,
}


# --------------------------------------------------------------------------- #
# Reference verifier                                                           #
# --------------------------------------------------------------------------- #
def _edge_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


def topology_verifier(mesh: Mesh) -> bool:
    """A built-in edge-incidence + orientation verifier: True if SOUND.

    Catches all four lie classes: degenerate facets (repeated index / zero
    length), boundary edges (torn seam), non-manifold edges (>2 faces), and
    inconsistent orientation (a directed edge traversed the same way twice).
    """
    # Degenerate facets.
    for f in mesh.faces:
        if len(f) < 3 or len(set(f)) < len(f):
            return False

    undirected: Dict[Tuple[int, int], int] = {}
    directed: Dict[Tuple[int, int], int] = {}
    for f in mesh.faces:
        n = len(f)
        for k in range(n):
            i, j = f[k], f[(k + 1) % n]
            undirected[_edge_key(i, j)] = undirected.get(_edge_key(i, j), 0) + 1
            directed[(i, j)] = directed.get((i, j), 0) + 1

    for count in undirected.values():
        if count != 2:  # boundary (1) or non-manifold (>2)
            return False
    for (i, j), count in directed.items():
        # A consistently-oriented closed surface traverses each undirected edge
        # once in each direction; the same directed edge twice means a flip.
        if count != 1:
            return False
    return True


# --------------------------------------------------------------------------- #
# Benchmark                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkResult:
    """The catch matrix for one verifier against the defect taxonomy."""

    caught: Dict[str, bool] = field(default_factory=dict)
    base_sound: bool = True

    @property
    def catch_count(self) -> int:
        return sum(1 for v in self.caught.values() if v)

    @property
    def total(self) -> int:
        return len(self.caught)

    @property
    def catch_rate(self) -> float:
        return self.catch_count / self.total if self.total else 0.0

    def summary(self) -> str:
        line = f"caught {self.catch_count}/{self.total}"
        detail = ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in sorted(self.caught.items()))
        return f"{line} ({detail}); base_sound={self.base_sound}"


def run_benchmark(
    base_mesh: Mesh,
    verifier: Callable[[Mesh], bool] = topology_verifier,
    classes: Sequence[str] = DEFECT_CLASSES,
) -> BenchmarkResult:
    """Inject each defect class into ``base_mesh`` and score ``verifier``.

    A defect is *caught* when the verifier judges the clean mesh SOUND but the
    injected mesh UNSOUND. If the verifier already rejects the clean base mesh
    the result flags ``base_sound=False`` (all catches are meaningless then).
    """
    base_ok = verifier(base_mesh)
    result = BenchmarkResult(base_sound=base_ok)
    for cls in classes:
        if cls not in INJECTORS:
            raise DefectError(f"unknown defect class {cls!r}")
        try:
            defective = INJECTORS[cls](base_mesh)
        except DefectError:
            result.caught[cls] = False
            continue
        # Caught iff base judged sound and defective judged unsound.
        result.caught[cls] = base_ok and (not verifier(defective))
    return result


# --------------------------------------------------------------------------- #
# Sound base meshes for testing / demos                                        #
# --------------------------------------------------------------------------- #
def unit_tetrahedron() -> Mesh:
    """A closed, consistently-oriented (outward) unit tetrahedron."""
    v = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    # Faces oriented so each undirected edge is traversed once each direction.
    f = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    return Mesh.of(v, f)


def unit_cube_mesh() -> Mesh:
    """A closed cube as 12 outward-oriented triangles."""
    v = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ]
    f = [
        (0, 3, 2), (0, 2, 1),  # bottom (z=0), outward -z
        (4, 5, 6), (4, 6, 7),  # top (z=1), outward +z
        (0, 1, 5), (0, 5, 4),  # front (y=0)
        (2, 3, 7), (2, 7, 6),  # back (y=1)
        (1, 2, 6), (1, 6, 5),  # right (x=1)
        (3, 0, 4), (3, 4, 7),  # left (x=0)
    ]
    return Mesh.of(v, f)
