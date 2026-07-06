"""Geometric-solver feedback for CADCodeVerify (paper section 5.1, App. B.2).

Deterministic re-implementation of the "Geometric solver feedback" baseline
from "Generating CAD Code with Vision-Language Models for 3D Designs"
(Alrashedy et al., ICLR 2025).  The paper uses FreeCAD to compute numerical
values across thirteen geometric categories for both the generated design and
the ground truth, then concatenates them (Eq. 6-7):

    FGS(d)  = {(s, GS(s, d)) : s in S}
    Fr      = FGS(d_gen) (+) FGS(d_gt)

and finally *verbalizes* the paired feedback into natural language describing
how the generated object differs from the ground truth (App. B.2).

FreeCAD is external, but the thirteen geometric properties of a triangle mesh
are ordinary stdlib geometry.  This module computes them from an explicit mesh
(vertices + triangular faces) with no third-party dependency, so the
solver-feedback protocol and its verbalization can be exercised deterministically.
"""
from __future__ import annotations

from math import sqrt

# The thirteen geometric categories S referenced in section 5.1.
CATEGORIES = (
    "width",           # x-extent
    "height",          # z-extent
    "depth",           # y-extent
    "num_vertices",
    "num_faces",
    "num_edges",
    "volume",
    "surface_area",
    "bbox_diagonal",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "bbox_volume",
)


def _vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _tri_area(p0, p1, p2):
    c = _cross(_vsub(p1, p0), _vsub(p2, p0))
    return 0.5 * sqrt(_dot(c, c))


def _signed_tet_volume(p0, p1, p2):
    # Signed volume of the tetrahedron (origin, p0, p1, p2); summed over a
    # closed mesh this yields the enclosed volume (divergence theorem).
    return _dot(p0, _cross(p1, p2)) / 6.0


def geometric_properties(vertices, faces):
    """Compute the thirteen geometric properties GS(s, d) of a mesh.

    ``vertices`` is a list of ``(x, y, z)`` and ``faces`` a list of triangle
    index triples.  Returns an ordered dict keyed by :data:`CATEGORIES`.
    """
    verts = [tuple(float(c) for c in v) for v in vertices]
    tris = [tuple(int(i) for i in f) for f in faces]
    if not verts:
        raise ValueError("mesh has no vertices")
    for f in tris:
        if len(f) != 3:
            raise ValueError("faces must be triangles")

    lo = tuple(min(v[i] for v in verts) for i in range(3))
    hi = tuple(max(v[i] for v in verts) for i in range(3))
    ext = tuple(hi[i] - lo[i] for i in range(3))

    area = 0.0
    vol = 0.0
    edges = set()
    for a, b, c in tris:
        p0, p1, p2 = verts[a], verts[b], verts[c]
        area += _tri_area(p0, p1, p2)
        vol += _signed_tet_volume(p0, p1, p2)
        for u, w in ((a, b), (b, c), (c, a)):
            edges.add((u, w) if u < w else (w, u))

    n = len(verts)
    centroid = tuple(sum(v[i] for v in verts) / n for i in range(3))
    diag = sqrt(sum(e * e for e in ext))

    return {
        "width": ext[0],
        "height": ext[2],
        "depth": ext[1],
        "num_vertices": float(n),
        "num_faces": float(len(tris)),
        "num_edges": float(len(edges)),
        "volume": abs(vol),
        "surface_area": area,
        "bbox_diagonal": diag,
        "centroid_x": centroid[0],
        "centroid_y": centroid[1],
        "centroid_z": centroid[2],
        "bbox_volume": ext[0] * ext[1] * ext[2],
    }


def solver_feedback(gen_mesh, gt_mesh):
    """Paired geometric feedback Fr = FGS(gen) (+) FGS(gt), Eq. 6-7.

    Each mesh is a ``(vertices, faces)`` pair.  Returns per-category records
    with generated value, ground-truth value, absolute difference and signed
    relative difference (generated relative to ground truth).
    """
    g = geometric_properties(*gen_mesh)
    t = geometric_properties(*gt_mesh)
    records = []
    for s in CATEGORIES:
        gv, tv = g[s], t[s]
        rel = (gv - tv) / tv if tv else None
        records.append({
            "category": s,
            "generated": gv,
            "ground_truth": tv,
            "abs_diff": abs(gv - tv),
            "rel_diff": rel,
        })
    return records


def verbalize(feedback, *, rel_tolerance=0.05):
    """Verbalize paired solver feedback into natural-language lines (App. B.2).

    Only categories whose generated value deviates from the ground truth by
    more than ``rel_tolerance`` (or any absolute difference for count/zero-
    reference categories) produce a corrective sentence, matching the paper's
    goal of describing *how* the generated object differs.  Deterministic and
    order-stable.
    """
    lines = []
    for rec in feedback:
        gv, tv = rec["generated"], rec["ground_truth"]
        rel = rec["rel_diff"]
        significant = (abs(gv - tv) > 0) if rel is None else (abs(rel) > rel_tolerance)
        if not significant:
            continue
        direction = "larger" if gv > tv else "smaller"
        lines.append(
            "The generated object's {cat} is {gv:.4g}, which is {dir} than the "
            "ground truth's {tv:.4g}.".format(cat=rec["category"].replace("_", " "),
                                              gv=gv, dir=direction, tv=tv)
        )
    if not lines:
        return "The generated object matches the ground truth within tolerance."
    return "\n".join(lines)
