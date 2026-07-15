"""THE INSTRUMENT. Exact measurements on the solid the model actually built.

Everything the measured oracle knows, it knows through this module, and every
function here is EXACT -- an OCCT B-rep query, not a sample of one. That is the
whole difference between this benchmark and the published ones:

* a **point** is inside the solid or outside it, decided by
  ``BRepClass3d_SolidClassifier`` on the B-rep. Not "inside a 64^3 voxel grid",
  not "inside a mesh we tessellated at 0.1 mm". A hole either is where the brief
  said or it is not, and no discretisation gets a vote.
* a **volume** comes from ``BRepGProp``, exact to machine precision. A 12 mm hole
  and an 8 mm hole differ by 1.4% of the part -- a rounding error to a mesh metric,
  a 10^13-sigma event to this one.
* a **section** through the part gives its area, its centroid and its second moment
  of area, exactly (verified: a 20 x 6 bar sections to I = 360.000 mm^4, and
  b*h^3/12 = 20*6^3/12 = 360). That is what makes the structural constraint in
  :mod:`~harnesscad.eval.hardcorpus.constraints` a MEASUREMENT of the model's own
  geometry rather than a guess about it.
* an **exact boolean IoU**: vol(A and B) / vol(A or B), computed by OCCT's boolean
  kernel and not by throwing darts. We use the STRONGEST POSSIBLE form of the
  field's own metric, deliberately -- so that when it still cannot see our
  near-misses, nobody can answer "your IoU was just noisy".

WHY NOT THE F-REP BACKEND, WHICH HAS A SIGNED-DISTANCE FIELD
------------------------------------------------------------
Because it cannot build the corpus. ``frep`` returns ``unsupported-op`` for
``loft``, ``sweep`` and ``draft`` -- and loft and sweep are precisely the L3 ops
Text2CAD-Bench measures a 70% frontier failure rate on. A benchmark that can only
be graded on an engine that cannot build it is not a benchmark. So the grading
engine is ``cadquery`` (OCCT), which is also an EXACT kernel rather than a sampled
one, and the sampling-theorem caveat that ``eval/corpus/grade.resolvable`` has to
carry for ``frep`` does not arise here at all.

MEASURED FINDING, RECORDED HERE BECAUSE IT SHAPED THE CORPUS
-------------------------------------------------------------
``Draft`` IS NOT BUILDABLE. The cadquery backend rejects it (``rejected='draft'``,
code ``unsupported-op``) and so does frep. Draft is in the CISP op set and in
Text2CAD-Bench's L3 definition, and we CANNOT ship a draft brief, because a brief
whose own reference solution does not build is the exact bug that contaminated v1.
It is dropped, and it is dropped loudly. See ``generate.DROPPED_OPS``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Op

__all__ = ["BACKEND", "Built", "build", "Section", "classify", "inside",
           "volume_of", "bbox_of", "boolean_iou", "section_at", "ray_chords",
           "min_wall_estimate", "bore_radius_at", "void_depth_at", "SURFACE_SEED",
           "sample_surface", "chamfer_distance", "mesh_of"]

Vec3 = Tuple[float, float, float]

#: The grading engine. An exact B-rep kernel that can actually build an L3 op
#: stream. See the module docstring for why it is not ``frep``.
BACKEND = "cadquery"

#: OCCT's point-classification tolerance, in mm. Machine-precision-ish: it is a
#: numeric guard on the B-rep intersection, NOT a tolerance on whether the feature
#: is in the right place. Every probe this corpus places is at least an order of
#: magnitude clear of any surface, so no verdict here is decided by this number.
CLASSIFY_TOL = 1e-7

#: Seed for the surface sampling that Chamfer Distance needs. Chamfer is the
#: FIELD'S metric and it is inherently a sample; ours are not. Fixed so that the
#: weak score is a function of the geometry and of nothing else.
SURFACE_SEED = 20260715


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
@dataclass
class Built:
    """A built op stream, and the honest reason when it is not one."""

    ok: bool = False
    engine: Any = None
    shape: Any = None                 # the raw TopoDS_Shape
    reason: str = ""

    def __bool__(self) -> bool:
        return bool(self.ok and self.shape is not None)


def build(ops: Sequence[Op], backend: str = BACKEND) -> Built:
    """Build an op stream from scratch at ``verify_level='core'``.

    CORE, never FULL. The verifier fleet is a system under test elsewhere in this
    repository; a grader that asked it for permission would be scoring the model on
    its ability to please a rule, and when the rule is wrong (``preflight-
    RADIUS_TOO_LARGE`` fired at r = 3.1 and stayed silent at r = 3.0, the true
    degenerate limit) it would be scoring the model on its ability to please a BUG.
    That is how the pressure corpus came to reward obeying the harness.
    """
    from harnesscad.core.loop import HarnessSession
    from harnesscad.eval.selftest.probe import resolve

    engine, skip = resolve(backend)
    if engine is None:
        return Built(reason="engine %r is not available here: %s" % (backend, skip))
    session = HarnessSession(engine, verify_level="core")
    try:
        result = session.apply_ops(list(ops))
    except Exception as exc:                                    # noqa: BLE001
        return Built(engine=engine,
                     reason="the engine raised %s: %s" % (type(exc).__name__, exc))
    if not getattr(result, "ok", False):
        rej = getattr(result, "rejected", None) or {}
        where = rej.get("where") if isinstance(rej, dict) else None
        code = rej.get("code") if isinstance(rej, dict) else None
        msg = rej.get("message") if isinstance(rej, dict) else None
        return Built(engine=engine,
                     reason="the engine refused the plan after %d op(s) at %r "
                            "(%s: %s)" % (getattr(result, "applied", 0), where,
                                          code or "?", msg or ""))
    shape = _shape_of(engine)
    if shape is None:
        return Built(engine=engine, reason="the plan produced no measurable solid")
    return Built(ok=True, engine=engine, shape=shape)


def _shape_of(engine: Any) -> Optional[Any]:
    fn = getattr(engine, "_combined", None)
    if not callable(fn):
        return None
    try:
        wp = fn()
    except Exception:                                           # noqa: BLE001
        return None
    if wp is None:
        return None
    raw = getattr(wp, "wrapped", None)
    return raw if raw is not None else wp


# --------------------------------------------------------------------------- #
# point membership -- the check a shape metric structurally cannot make
# --------------------------------------------------------------------------- #
def classify(shape: Any, p: Vec3) -> str:
    """'in' | 'out' | 'on' -- exactly, from the B-rep.

    This is the ENTIRE reason the near-miss corpus works. A hole bored at x = 40
    instead of x = 20 has the same volume, the same bounding box, the same genus,
    the same watertightness, and an IoU of 0.957. It differs from the correct part
    in exactly one respect that any metric can name: THERE IS MATERIAL AT
    (20, 20, 6) AND THERE SHOULD NOT BE. This function is how we say that.
    """
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON

    cl = BRepClass3d_SolidClassifier(shape)
    cl.Perform(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), CLASSIFY_TOL)
    state = cl.State()
    if state == TopAbs_IN:
        return "in"
    if state == TopAbs_ON:
        return "on"
    return "out"


def inside(shape: Any, p: Vec3) -> bool:
    return classify(shape, p) == "in"


# --------------------------------------------------------------------------- #
# bulk properties -- exact
# --------------------------------------------------------------------------- #
def volume_of(shape: Any) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return float(props.Mass())


def bbox_of(shape: Any) -> Tuple[float, float, float, float, float, float]:
    """(xmin, ymin, zmin, xmax, ymax, zmax), exactly."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box, True)
    return tuple(float(v) for v in box.Get())                   # type: ignore[return-value]


def extents_of(shape: Any) -> Vec3:
    x0, y0, z0, x1, y1, z1 = bbox_of(shape)
    return (x1 - x0, y1 - y0, z1 - z0)


def boolean_iou(a: Any, b: Any) -> Optional[float]:
    """vol(A and B) / vol(A or B), from OCCT's boolean kernel. NOT a sample.

    The field's own metric, in its strongest form. We use the exact one on purpose:
    when it scores 0.973 on a part with the wrong hole diameter, "your IoU was
    noisy" is not available as an answer.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse

    try:
        common = BRepAlgoAPI_Common(a, b).Shape()
        fused = BRepAlgoAPI_Fuse(a, b).Shape()
        union = volume_of(fused)
        if union <= 0.0:
            return None
        return volume_of(common) / union
    except Exception:                                           # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# section properties -- what makes a STRUCTURAL constraint checkable
# --------------------------------------------------------------------------- #
@dataclass
class Section:
    """A planar cut through the model's own geometry, measured exactly.

    ``area``      mm^2, the cut face's area.
    ``centroid``  the cut face's centre of area, in world coordinates.
    ``inertia``   the second moment of area about the centroidal axis PARALLEL to
                  ``axis``, in mm^4 -- the ``I`` in ``sigma = M*c/I``.
    ``c``         the extreme fibre distance from the centroid, in mm.

    Verified against the closed form: a 20 x 6 rectangular bar sections to
    ``I = 360.000 mm^4``, and ``b*h^3/12 = 20*6^3/12 = 360``.
    """

    ok: bool = False
    area: float = 0.0
    centroid: Vec3 = (0.0, 0.0, 0.0)
    inertia: float = 0.0
    c: float = 0.0
    reason: str = ""

    @property
    def section_modulus(self) -> float:
        """Z = I / c, mm^3. The number a bending stress is actually divided by."""
        return self.inertia / self.c if self.c > 0.0 else 0.0

    def to_dict(self) -> dict:
        return {"ok": self.ok, "area": self.area, "centroid": list(self.centroid),
                "inertia": self.inertia, "c": self.c,
                "section_modulus": self.section_modulus, "reason": self.reason}


def section_at(shape: Any, origin: Vec3, normal: Vec3,
               bend_axis: Vec3 = (0.0, 1.0, 0.0)) -> Section:
    """Cut ``shape`` with the plane (origin, normal) and measure the cut face.

    ``bend_axis`` is the neutral axis the bending moment acts about; ``inertia`` is
    reported about the axis THROUGH THE CENTROID and parallel to it, which is the
    only one Euler-Bernoulli will accept.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepGProp import BRepGProp
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pln, gp_Pnt
    from OCP.GProp import GProp_GProps

    try:
        plane = gp_Pln(gp_Pnt(*[float(v) for v in origin]),
                       gp_Dir(*[float(v) for v in normal]))
        face = BRepBuilderAPI_MakeFace(plane, -1e4, 1e4, -1e4, 1e4).Face()
        cut = BRepAlgoAPI_Common(shape, face).Shape()
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(cut, props)
        area = float(props.Mass())
        if area <= 0.0:
            return Section(reason="the cutting plane misses the solid: there is no "
                                  "material at the section the brief names")
        com = props.CentreOfMass()
        centroid = (float(com.X()), float(com.Y()), float(com.Z()))
        axis = gp_Ax1(gp_Pnt(*centroid),
                      gp_Dir(*[float(v) for v in bend_axis]))
        inertia = float(props.MomentOfInertia(axis))
    except Exception as exc:                                    # noqa: BLE001
        return Section(reason="the section could not be measured: %s" % exc)

    # The extreme fibre: how far the furthest material at this section is from the
    # centroid, measured ALONG the bending direction (normal x bend_axis is in the
    # section plane and perpendicular to the neutral axis).
    fibre = _cross(normal, bend_axis)
    n = math.sqrt(sum(v * v for v in fibre))
    if n <= 0.0:
        return Section(reason="the bending axis is parallel to the section normal; "
                              "no bending is defined")
    fibre = tuple(v / n for v in fibre)                          # type: ignore[assignment]
    x0, y0, z0, x1, y1, z1 = bbox_of(shape)
    corners = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    c = max(abs(sum(fibre[i] * (p[i] - centroid[i]) for i in range(3)))
            for p in corners)
    return Section(ok=True, area=area, centroid=centroid, inertia=inertia, c=c)


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


# --------------------------------------------------------------------------- #
# feature interrogation -- for constraints, where there is no reference part
# --------------------------------------------------------------------------- #
def bore_radius_at(shape: Any, axis_xy: Tuple[float, float], z: float,
                   limit: float = 60.0, tol: float = 0.005) -> Optional[float]:
    """The radius of the VOID centred on a vertical axis at height ``z``.

    ``None`` when the axis point itself is inside material -- there is no bore
    there at all, which for a bolt-clearance constraint is the interesting answer.

    This is how a constraint brief is graded WITHOUT a reference part: we do not
    ask "does your hole match ours", we ask "IS THERE A HOLE HERE, AND IS IT WIDE
    ENOUGH FOR AN M8 BOLT". Many parts satisfy that, which is exactly why IoU
    cannot score it and this can.
    """
    p0 = (axis_xy[0], axis_xy[1], z)
    if classify(shape, p0) != "out":
        return None
    lo, hi = 0.0, 0.0
    r = tol
    while r <= limit:
        if classify(shape, (axis_xy[0] + r, axis_xy[1], z)) != "out":
            hi = r
            break
        lo = r
        r *= 1.6
    if hi <= 0.0:
        return None            # void all the way to the limit: no wall, not a bore
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if classify(shape, (axis_xy[0] + mid, axis_xy[1], z)) == "out":
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def void_depth_at(shape: Any, axis_xy: Tuple[float, float], z_top: float,
                  z_bottom: float, tol: float = 0.005) -> float:
    """How far down from ``z_top`` the void on this axis runs. 0.0 if none.

    A counterbore is a hole with a wider void near the top face; a plain hole is
    not. This is what tells them apart on the model's own geometry.
    """
    if classify(shape, (axis_xy[0], axis_xy[1], z_top - tol)) != "out":
        return 0.0
    lo, hi = z_bottom, z_top
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if classify(shape, (axis_xy[0], axis_xy[1], mid)) == "out":
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return z_top - 0.5 * (lo + hi)


def ray_chords(shape: Any, origin: Vec3, direction: Vec3, length: float,
               steps: int = 400) -> List[float]:
    """The lengths of the maximal MATERIAL intervals a ray passes through.

    A chord through a wall is always at least as long as the wall is thick (a ray
    at angle theta through a slab of thickness t has chord t / cos theta >= t). So
    a chord SHORTER than a required minimum wall PROVES a violation. The converse
    does not hold, and :func:`min_wall_estimate` says so out loud.
    """
    n = math.sqrt(sum(float(d) * float(d) for d in direction))
    if n <= 0.0:
        return []
    d = tuple(float(v) / n for v in direction)
    step = length / float(steps)
    chords: List[float] = []
    run = 0.0
    for i in range(steps + 1):
        t = i * step
        p = (origin[0] + d[0] * t, origin[1] + d[1] * t, origin[2] + d[2] * t)
        if classify(shape, p) == "in":
            run += step
        elif run > 0.0:
            chords.append(run)
            run = 0.0
    if run > 0.0:
        chords.append(run)
    return chords


def min_wall_estimate(shape: Any, rays: int = 96, seed: int = SURFACE_SEED,
                      steps: int = 300) -> Optional[float]:
    """The shortest material chord over ``rays`` seeded rays through the part.

    SOUND IN ONE DIRECTION ONLY, AND THAT IS SAID EVERY TIME IT IS REPORTED. A
    chord is never shorter than the wall it crosses, so::

        min_chord < t_required   =>   the part HAS a wall thinner than t_required.

    That implication is exact and it is the one a manufacturing constraint needs:
    it can only ever FAIL a part that really is too thin. The reverse implication
    is NOT available -- a part can hide a thin wall that no ray in a finite bundle
    happened to cross perpendicular to -- so this function can never CERTIFY a part
    as thick enough, and :mod:`~harnesscad.eval.hardcorpus.constraints` records the
    check as ``sound_direction='violation'`` so that nobody reads a pass as a
    guarantee. A checker that overclaimed here would be the VLM judge with extra
    steps.
    """
    x0, y0, z0, x1, y1, z1 = bbox_of(shape)
    span = max(x1 - x0, y1 - y0, z1 - z0)
    if span <= 0.0:
        return None
    diag = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
    rnd = random.Random(seed)
    best: Optional[float] = None
    for _ in range(rays):
        # A random point on the bbox surface, fired at a random point inside it:
        # the bundle is biased towards rays that actually cross material.
        src = (rnd.uniform(x0 - 0.1, x1 + 0.1),
               rnd.uniform(y0 - 0.1, y1 + 0.1),
               rnd.uniform(z0 - 0.1, z1 + 0.1))
        dst = (rnd.uniform(x0, x1), rnd.uniform(y0, y1), rnd.uniform(z0, z1))
        d = tuple(dst[i] - src[i] for i in range(3))
        if all(abs(v) < 1e-9 for v in d):
            continue
        for chord in ray_chords(shape, src, d, diag * 1.2, steps=steps):
            if best is None or chord < best:
                best = chord
    return best


# --------------------------------------------------------------------------- #
# meshes -- ONLY for the weak metrics. Nothing our oracle says depends on one.
# --------------------------------------------------------------------------- #
def mesh_of(engine: Any) -> Optional[Tuple[List[Vec3], List[Tuple[int, int, int]]]]:
    """The tessellation the harness actually exports. Used by Chamfer Distance.

    Routed through ``io/gate.py``'s own geometry reader so that the mesh the weak
    metrics see is byte-for-byte the mesh the gate measures -- otherwise a
    disagreement between them would be a disagreement about tessellation, not about
    the part, and the headline table would be worthless.
    """
    from harnesscad.io import gate

    try:
        return gate._geometry(engine, engine)                    # noqa: SLF001
    except Exception:                                           # noqa: BLE001
        return None


def sample_surface(verts: Sequence[Vec3], faces: Sequence[Sequence[int]],
                   n: int, seed: int = SURFACE_SEED) -> List[Vec3]:
    """``n`` points on the mesh surface, area-weighted. Seeded, so it is a function.

    This is how Text2CAD-Bench computes Chamfer Distance and it is reproduced
    faithfully, because the point of :mod:`~harnesscad.eval.hardcorpus.weak` is to
    run THEIR metric, not a straw man of it.
    """
    tris = [(verts[f[0]], verts[f[1]], verts[f[2]]) for f in faces if len(f) >= 3]
    if not tris:
        return []
    areas = []
    for a, b, c in tris:
        u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        w = _cross(u, v)
        areas.append(0.5 * math.sqrt(sum(t * t for t in w)))
    total = sum(areas)
    if total <= 0.0:
        return []
    cum: List[float] = []
    acc = 0.0
    for a in areas:
        acc += a
        cum.append(acc / total)
    rnd = random.Random(seed)
    out: List[Vec3] = []
    import bisect
    for _ in range(n):
        i = min(bisect.bisect_left(cum, rnd.random()), len(tris) - 1)
        a, b, c = tris[i]
        r1, r2 = rnd.random(), rnd.random()
        if r1 + r2 > 1.0:
            r1, r2 = 1.0 - r1, 1.0 - r2
        out.append((a[0] + r1 * (b[0] - a[0]) + r2 * (c[0] - a[0]),
                    a[1] + r1 * (b[1] - a[1]) + r2 * (c[1] - a[1]),
                    a[2] + r1 * (b[2] - a[2]) + r2 * (c[2] - a[2])))
    return out


def chamfer_distance(pa: Sequence[Vec3], pb: Sequence[Vec3]) -> Optional[float]:
    """Symmetric Chamfer Distance (mean of the two one-sided means), in mm.

    Text2CAD-Bench's headline metric. Reported here so that the discriminative
    table can state, in the field's own units, what a near-miss costs -- which for
    an 8 mm hole where 12 mm was demanded turns out to be a few hundredths of a
    millimetre, against a "GPT-5.2 scores 93.46" scale on which that is noise.
    """
    if not pa or not pb:
        return None
    try:
        import numpy as np
    except ImportError:                                         # pragma: no cover
        return None
    a = np.asarray(pa, dtype=float)
    b = np.asarray(pb, dtype=float)
    # Chunked to keep the pairwise matrix bounded regardless of sample count.
    def one_sided(x, y) -> float:
        best = np.full(len(x), np.inf)
        for i in range(0, len(y), 1024):
            blk = y[i:i + 1024]
            d = np.sqrt(((x[:, None, :] - blk[None, :, :]) ** 2).sum(-1))
            best = np.minimum(best, d.min(axis=1))
        return float(best.mean())

    return 0.5 * (one_sided(a, b) + one_sided(b, a))
