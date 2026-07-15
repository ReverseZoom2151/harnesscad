"""ExternalToolBackend — the shared spine of the out-of-process CAD backends.

The Blender and OpenSCAD backends both work the same way: they accept the CISP
op stream, hand the geometry to a *real* external kernel (Blender's exact mesh
booleans; OpenSCAD's CGAL CSG), and read the resulting mesh back so that
``query()`` / ``export()`` answer from genuine geometry rather than from a
sampled field.

What they share, and what lives here:

* **op semantics.** Id allocation, sketch DOF bookkeeping, block-and-correct on a
  bad reference, ``SetParam`` replay and the op log are *exactly* the frep
  backend's -- so the harness drives all three identically. Rather than
  re-implement them (three copies that would drift), this base COMPOSES an
  :class:`~harnesscad.io.backends.frep.FRepBackend` and uses it as the op-state
  model and as the source of the CSG tree (:meth:`FRepBackend.root`). The F-rep
  tree is a kernel-neutral CSG DAG; each subclass lowers it into its own tool's
  language.
* **mesh readback.** The tool writes an STL; it is parsed with
  :mod:`harnesscad.io.formats.stl`, welded, and every mass property / validity
  query is computed from that mesh. Nothing is estimated.
* **content-addressed caching.** The tool's inputs are written to a directory
  named by the SHA-256 of the program text, so an identical model never re-runs
  the tool and no wall clock or PID ever enters a path (determinism).
* **graceful absence.** The executable is discovered once, in ``__init__``; if it
  is missing the constructor raises :class:`BackendUnavailable`, so the CISP
  server can fall back and the test suite can skip.

Subclasses supply: the tool name, how to find its executable, which ops they
cannot honour, the program text for a given F-rep root, and how to run the tool.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from glob import glob
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Op, Primitive
from harnesscad.domain.geometry.mesh.halfedge import HalfedgeMesh
from harnesscad.domain.geometry.parametric import facets
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.backends import frep
from harnesscad.io.backends.base import ApplyResult, BackendUnavailable
from harnesscad.io.backends.frep import FRepBackend, Node
from harnesscad.io.formats import glb as glb_fmt
from harnesscad.io.formats import stl as stl_fmt

Vec3 = Tuple[float, float, float]
Mesh = Tuple[List[Vec3], List[Tuple[int, int, int]]]

#: Facet count used for every curved primitive (circles, cylinders, revolves).
#: Both external backends use the SAME number, so the two kernels tessellate a
#: circle into the same polygon and their volumes are directly comparable. It is
#: a fixed constant, never a wall-clock- or resolution-dependent choice.
DEFAULT_SEGMENTS = 64

#: Welding tolerance for the STL that comes back from the tool. Binary STL
#: carries float32 vertices, so coincident corners can differ in the last bit;
#: this is far below any feature size and far above float32 noise.
WELD_TOLERANCE = 1e-4

#: How long a single tool invocation may take before we give up (seconds).
DEFAULT_TIMEOUT = 300


def _err(code: str, msg: str, where: Optional[str] = None) -> ApplyResult:
    return ApplyResult(False, [], [Diagnostic(Severity.ERROR, code, msg, where)])


def find_executable(env_var: str, names: Sequence[str],
                    patterns: Sequence[str] = ()) -> Tuple[Optional[str], List[str]]:
    """Locate an external tool. Returns ``(path_or_None, places_searched)``.

    Order: the explicit ``env_var`` override, then ``PATH``, then the glob
    ``patterns`` (installer locations). Glob hits are sorted and the LAST is
    taken, so the newest installed version wins deterministically -- with no
    hard-coded version number anywhere.
    """
    searched: List[str] = []
    override = os.environ.get(env_var)
    if override:
        searched.append("%s=%s" % (env_var, override))
        if os.path.isfile(override):
            return override, searched
    for name in names:
        searched.append("PATH:%s" % name)
        hit = shutil.which(name)
        if hit:
            return hit, searched
    for pattern in patterns:
        searched.append(pattern)
        hits = sorted(p for p in glob(pattern) if os.path.isfile(p))
        if hits:
            return hits[-1], searched
    return None, searched


def cache_dir(tool: str, digest: str) -> str:
    """A content-addressed scratch directory: same program -> same path, always."""
    path = os.path.join(tempfile.gettempdir(), "harnesscad-" + tool, digest[:32])
    os.makedirs(path, exist_ok=True)
    return path


def program_digest(text: str) -> str:
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# geometry helpers shared by both lowerings
# --------------------------------------------------------------------------
def segments_for(radius: float, segments: int) -> int:
    """Facet count for a curve of ``radius``, via OpenSCAD's own $fn/$fa/$fs law.

    ``segments`` is passed as ``$fn``; the formula
    (:func:`facets.get_fragments_from_r`) is the exact port of OpenSCAD's
    ``Calc::get_fragments_from_r``, so the polygon a circle becomes is identical
    in both backends -- and identical to what OpenSCAD itself would build.
    """
    return facets.get_fragments_from_r(abs(float(radius)), fn=float(segments))


def circle_loop(cx: float, cy: float, r: float, segments: int) -> List[Tuple[float, float]]:
    """The CCW polygon OpenSCAD inscribes for a circle, centred at (cx, cy)."""
    return [(cx + px, cy + py)
            for (px, py) in facets.circle_fragment_points(abs(float(r)), fn=float(segments))]


def rect_loop(x: float, y: float, w: float, h: float) -> List[Tuple[float, float]]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def signed_area(loop: Sequence[Tuple[float, float]]) -> float:
    total = 0.0
    n = len(loop)
    for i in range(n):
        x0, y0 = loop[i]
        x1, y1 = loop[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def ccw(loop: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """The same loop, guaranteed counter-clockwise."""
    out = [(float(x), float(y)) for (x, y) in loop]
    if signed_area(out) < 0.0:
        out.reverse()
    return out


def profile_loops(profile, segments: int) -> List[List[Tuple[float, float]]]:
    """Every closed loop of an F-rep :class:`~frep._Profile`, as a CCW polygon.

    The three entity families a sketch can carry (rectangles, circles, polylines)
    collapse to one representation here: a polygon. Circles use OpenSCAD's facet
    law, so the polygon is the one the kernel would have built anyway.
    """
    loops: List[List[Tuple[float, float]]] = []
    for (x, y, w, h) in profile.rects:
        loops.append(ccw(rect_loop(x, y, w, h)))
    for (cx, cy, r) in profile.circles:
        loops.append(ccw(circle_loop(cx, cy, r, segments)))
    for verts in profile.polys:
        if len(verts) >= 3:
            loops.append(ccw(verts))
    return loops


def plane_axes(plane: str) -> Tuple[int, int, int]:
    return frep._plane_axes(plane)


def to_world(plane: str, u: float, v: float, w: float) -> Vec3:
    return frep._to_world(plane, u, v, w)


def plane_normal(plane: str) -> Vec3:
    """The unit normal of a named datum plane (the axis a mirror flips)."""
    pl = str(plane).upper()
    if pl == "XY":
        return (0.0, 0.0, 1.0)
    if pl == "YZ":
        return (1.0, 0.0, 0.0)
    return (0.0, 1.0, 0.0)  # XZ


def slab(w0: float, w1: float) -> Tuple[float, float]:
    return (w0, w1) if w0 <= w1 else (w1, w0)


def blend_radius(node: Node) -> Tuple[float, float]:
    """The (round, chamfer) radius the F-rep tree carries, if any.

    ``Fillet`` / ``Chamfer`` do not appear as nodes in the F-rep tree: they are a
    REWRITE of it (``frep.blend_tree``) that stamps a radius onto every leaf and
    a blend onto every boolean. A mesh kernel cannot honour that rewrite
    pointwise, but it can honour the intent -- so we read the radius back off the
    tree and hand it to the kernel's own edge-blend (Blender's bevel modifier).
    """
    r = c = 0.0
    stack = [node]
    while stack:
        n = stack.pop()
        r = max(r, float(n.d.get("round", 0.0) or 0.0))
        c = max(c, float(n.d.get("cham", 0.0) or 0.0))
        if n.t == "bool":
            if n.d.get("blend") == "smooth":
                r = max(r, float(n.d.get("k", 0.0) or 0.0))
            elif n.d.get("blend") == "chamfer":
                c = max(c, float(n.d.get("k", 0.0) or 0.0))
            stack.append(n.d["a"])
            stack.append(n.d["b"])
        elif n.t in ("shell", "mirror", "pattern"):
            stack.append(n.d["child"])
    return r, c


# --------------------------------------------------------------------------
# the backend
# --------------------------------------------------------------------------
class ExternalToolBackend:
    """A GeometryBackend that lowers the CISP op stream onto an external kernel."""

    #: Subclass hooks -----------------------------------------------------
    TOOL = "external"
    #: Ops this tool cannot honour, mapped to the reason. Checked BEFORE the op
    #: reaches the op log, so an unsupported op never mutates state.
    UNSUPPORTED: Dict[str, str] = {}
    FORMATS: Tuple[str, ...] = ("stl", "stl-ascii", "stl-binary", "glb")

    #: Primitive shapes (``ops.Primitive.shape``) this tool can LOWER, via the
    #: F-rep node the composed model builds for them. box/cylinder/cone reuse the
    #: extrude/cyl/cone nodes; sphere needs a native primitive in the lowering. A
    #: shape not listed here is refused with a typed ``unsupported-op`` at the op
    #: boundary (so a node the lowering cannot express never reaches the kernel).
    PRIMITIVE_SHAPES: Tuple[str, ...] = ()

    #: Shell join kinds THIS tool's kernel can build (``ops.Shell.kind``).
    #:
    #: The composed FRepBackend is the op-state model, and it refuses what ITS OWN
    #: field cannot evaluate -- an SDF's inward offset is the arc join and nothing
    #: else. But this backend does not build the SDF: it lowers the tree onto a real
    #: kernel, which may well have both joins (OpenSCAD's ``offset(r=)`` vs
    #: ``offset(delta=)``; OCCT's ``join`` argument). So each subclass declares what
    #: its kernel can actually do, and the op-state model is widened to match.
    #: Declaring a join here without lowering it would put the original bug back.
    SHELL_JOINS: Tuple[str, ...] = ("arc",)

    def __init__(self, segments: int = DEFAULT_SEGMENTS,
                 executable: Optional[str] = None,
                 timeout: int = DEFAULT_TIMEOUT) -> None:
        self.segments = int(segments)
        self.timeout = int(timeout)
        self.executable = executable or self.locate()
        self._frep = FRepBackend()
        # The op-state model is frep's, but the GEOMETRY is a real kernel's, and two
        # of frep's refusals are about frep's sampling grid rather than about the
        # op: a wall thinner than a grid cell (these kernels sample no grid -- their
        # booleans are exact), and a shell join its field cannot express. Neither
        # limit is this tool's, so neither is imposed on it.
        self._frep.SHELL_MIN_WALL_CELLS = 0.0
        self._frep.SHELL_JOINS = tuple(self.SHELL_JOINS)
        # These kernels have REAL B-rep edges and select them themselves, from the
        # op log, against their own topology (freecad's apply_blends, blender's
        # bevel). frep must therefore not ALSO cut the edges it approximated from a
        # bounding box, or every named fillet would be applied twice.
        self._frep.EDGE_SELECTORS = True
        self._mesh_cache: Optional[Tuple[str, Mesh]] = None
        self._stl_cache: Optional[Tuple[str, bytes]] = None

    # -- discovery ---------------------------------------------------------
    @classmethod
    def locate(cls) -> str:
        """The tool's executable, or raise :class:`BackendUnavailable`."""
        raise NotImplementedError

    @classmethod
    def available(cls) -> bool:
        """Whether this backend can run here. Never raises -- for skipUnless."""
        try:
            cls.locate()
        except BackendUnavailable:
            return False
        return True

    # -- op stream ---------------------------------------------------------
    def reset(self) -> None:
        self._frep.reset()
        self._mesh_cache = None
        self._stl_cache = None

    def apply(self, op: Op) -> ApplyResult:
        """Validate against this tool's op set, then delegate to the F-rep model.

        Block-and-correct is preserved twice over: an op this tool cannot honour
        is refused here (nothing mutates), and an op with a bad reference is
        refused by the F-rep model (which also does not mutate).
        """
        tag = getattr(type(op), "OP", "")
        reason = self.UNSUPPORTED.get(tag)
        if reason is not None:
            return _err("unsupported-op",
                        "the %s backend does not implement %s: %s"
                        % (self.TOOL, tag, reason), None)
        if isinstance(op, Primitive) and \
                str(op.shape).lower() not in self.PRIMITIVE_SHAPES:
            return _err("unsupported-op",
                        "the %s backend cannot build a %r primitive: its lowering "
                        "has no expression for that shape (supported: %s)"
                        % (self.TOOL, op.shape,
                           ", ".join(self.PRIMITIVE_SHAPES) or "none"), None)
        result = self._frep.apply(op)
        if result.ok:
            self._mesh_cache = None
            self._stl_cache = None
        return result

    # -- state (delegated, so the verifiers see exactly the frep model) -----
    @property
    def sketches(self) -> dict:
        return self._frep.sketches

    @property
    def entities(self) -> dict:
        return self._frep.entities

    @property
    def features(self) -> list:
        return self._frep.features

    @property
    def instances(self) -> list:
        return self._frep.instances

    @property
    def mates(self) -> list:
        return self._frep.mates

    @property
    def solid_present(self) -> bool:
        return self._frep.solid_present

    def root(self) -> Optional[Node]:
        return self._frep.root()

    def bounds(self):
        return self._frep.bounds()

    def state_digest(self) -> str:
        """Content hash of the model. Tagged with the tool: the same op stream on
        two different kernels is the same MODEL but not the same GEOMETRY, and a
        digest that could not tell them apart would be lying."""
        blob = "%s|%s" % (self.TOOL, self._frep.state_digest())
        return hashlib.sha256(blob.encode()).hexdigest()

    # -- the tool ----------------------------------------------------------
    def program(self) -> str:
        """The tool's program text for the current model (subclass hook)."""
        raise NotImplementedError

    def _run(self, source: str, workdir: str, out_path: str) -> None:
        """Run the tool so that ``out_path`` becomes a valid STL (subclass hook)."""
        raise NotImplementedError

    def stl_bytes(self) -> bytes:
        """Drive the tool and return the STL it produced. Cached on the digest."""
        key = self.state_digest()
        if self._stl_cache is not None and self._stl_cache[0] == key:
            return self._stl_cache[1]
        if self.root() is None:
            data = stl_fmt.write_binary_stl([], header=b"harnesscad-" + self.TOOL.encode())
            self._stl_cache = (key, data)
            return data
        source = self.program()
        digest = program_digest(source)
        workdir = cache_dir(self.TOOL, digest)
        out_path = os.path.join(workdir, "model.stl")
        if not (os.path.isfile(out_path) and os.path.getsize(out_path) > 0):
            self._run(source, workdir, out_path)
        with open(out_path, "rb") as fh:
            data = fh.read()
        self._stl_cache = (key, data)
        return data

    def triangles(self) -> List[stl_fmt.Triangle]:
        data = self.stl_bytes()
        if not data:
            return []
        return stl_fmt.parse_stl(data)

    def mesh(self) -> Mesh:
        """The tool's mesh, welded into (vertices, triangles)."""
        key = self.state_digest()
        if self._mesh_cache is not None and self._mesh_cache[0] == key:
            return self._mesh_cache[1]
        tris = self.triangles()
        verts: List[Vec3] = []
        faces: List[Tuple[int, int, int]] = []
        for t in tris:
            i = len(verts)
            verts.extend(t.vertices)
            faces.append((i, i + 1, i + 2))
        m = frep.weld(verts, faces, WELD_TOLERANCE)
        self._mesh_cache = (key, m)
        return m

    # -- protocol ----------------------------------------------------------
    def regenerate(self) -> List[Diagnostic]:
        """Rebuild through the tool and report a mesh the kernel got wrong."""
        if self.root() is None:
            return []
        try:
            verts, faces = self.mesh()
        except BackendUnavailable:
            raise
        except Exception as exc:  # the tool failed on this model
            return [Diagnostic(Severity.ERROR, "kernel-error",
                               "%s failed to build the model: %s" % (self.TOOL, exc))]
        if not faces:
            return [Diagnostic(Severity.ERROR, "empty-solid",
                               "%s produced no geometry (empty solid)" % self.TOOL)]
        he = HalfedgeMesh(verts, faces)
        ok, issues = he.is_2manifold()
        if ok:
            return []
        codes = sorted({i.code for i in issues})
        return [Diagnostic(Severity.ERROR, "invalid-mesh",
                           "%s mesh is not a 2-manifold (%s; %d issues)"
                           % (self.TOOL, ", ".join(codes), len(issues)))]

    def query(self, q: str) -> dict:
        """The SAME query surface the frep backend exposes, answered from the
        tool's mesh instead of from a sampled field."""
        if q == "sketch_dof":
            return self._frep.query("sketch_dof")
        if q == "summary":
            return self._frep.query("summary")
        if q == "assembly":
            return self._frep.query("assembly")
        if q == "validity":
            return self._validity()
        if q == "measure":
            m = self._metrics()
            if not m:
                return {"volume": 0.0, "bbox": [0.0, 0.0, 0.0]}
            return {"volume": m["volume"], "bbox": m["bbox"]}
        if q == "metrics":
            return self._metrics()
        if q == "mass_properties":
            return self.mass_properties()
        if q == "mesh":
            verts, faces = self.mesh()
            return {"vertex_count": len(verts), "triangle_count": len(faces)}
        return {}

    def _validity(self) -> dict:
        if self.root() is None:
            return {"manifold": False, "watertight": False,
                    "is_valid": False, "solid_present": False}
        verts, faces = self.mesh()
        if not faces:
            return {"manifold": False, "watertight": False,
                    "is_valid": False, "solid_present": True}
        he = HalfedgeMesh(verts, faces)
        manifold, issues = he.is_2manifold()
        watertight = he.is_closed()
        return {"manifold": bool(manifold), "watertight": bool(watertight),
                "is_valid": bool(manifold and watertight),
                "solid_present": True,
                "genus": he.genus() if watertight else None,
                "euler_characteristic": he.euler_characteristic(),
                "issues": len(issues)}

    def _metrics(self, density: float = 1.0) -> dict:
        """Mass properties read off the kernel's own mesh.

        The volume is the exact signed volume of the returned triangles (the
        divergence theorem on a closed mesh) -- so for OpenSCAD it is the exact
        volume of the CSG result, with no sampling error at all.
        """
        if self.root() is None:
            return {}
        tris = self.triangles()
        if not tris:
            return {}
        verts, faces = self.mesh()
        volume = abs(stl_fmt.signed_volume(tris))
        area = stl_fmt.surface_area(tris)
        lo, hi = stl_fmt.bounding_box(tris)
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        cz = sum(v[2] for v in verts) / len(verts)
        return {
            "volume": float(volume),
            "mass": float(volume * density),
            "surface_area": float(area),
            "bbox": [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]],
            "center_of_mass": [cx, cy, cz],
            "triangle_count": len(faces),
            "vertex_count": len(verts),
        }

    def mass_properties(self, density: float = 1.0, order: int = 3) -> dict:
        """Volume, centroid and inertia tensor, by the same Gauss quadrature the
        frep backend uses -- but over the KERNEL's mesh."""
        verts, faces = self.mesh()
        if not faces:
            return {}
        shim = FRepBackend()
        shim._mesh_cache = ("external", (verts, faces))
        shim.mesh = lambda **_kw: (verts, faces)  # type: ignore[assignment]
        return FRepBackend.mass_properties(shim, density=density, order=order)

    # -- export ------------------------------------------------------------
    def export(self, fmt: str):
        f = str(fmt).lower()
        tag = ("harnesscad-" + self.TOOL)
        if f in ("stl", "stl-binary", "stlb"):
            return self.stl_bytes()
        if f == "stl-ascii":
            return stl_fmt.write_ascii_stl(self.triangles(), name=tag)
        if f == "glb":
            return glb_fmt.stl_to_glb(
                stl_fmt.write_binary_stl(self.triangles(), header=tag.encode()),
                name=tag)
        raise ValueError("%s backend cannot export '%s' (supported: %s)"
                         % (self.TOOL, fmt, ", ".join(self.FORMATS)))

    def write_stl(self, path: str, force: bool = False) -> int:
        """Write the model to ``path`` -- through the output gate.

        A backend's own ``write_stl`` bypasses the format registry, so the gate
        has to stand here too, or this becomes the hole the registry's gate was
        built to close.
        """
        from harnesscad.io import gate

        data = self.stl_bytes()
        report = gate.guard(data, str(path), source=self, force=force)
        with open(path, "wb") as fh:
            fh.write(data)
        if not report.ok:                              # forced through the gate
            gate.write_sidecar(str(path), report)
        return len(self.triangles())
