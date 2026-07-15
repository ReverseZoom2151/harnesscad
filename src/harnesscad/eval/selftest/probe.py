"""The measurement primitive every oracle in this package is built on.

One job: run an op stream on ONE backend and come back with a comparable
:class:`Observation` -- volume, bbox, genus, watertightness, digest -- or a clean
SKIP when that backend's tool is not installed on this machine.

Three things here are load-bearing and are stated once, here, so the four oracles
never re-derive them:

**1. Availability is resolved, not assumed.** ``server._make_backend`` falls back
to the stub *with a note* when a tool is missing. For a differential test that
fallback is poison: two "backends" that are both secretly the stub agree
perfectly and prove nothing. :func:`resolve` therefore treats a fallback as
UNAVAILABLE and skips it.

**2. The stub carries no geometry.** It answers ``query('measure')`` with ``{}``.
It is not a geometric oracle and is excluded from every geometric comparison --
it is still exercised for op acceptance and determinism. It is also INSTANT, which
makes it the right engine for scoring the op-stream (LINT) verifiers.

**COST.** Every measurement here drives a real engine. ``frep`` samples and marches
a grid (~1-3 s per part); ``freecad``, ``blender`` and ``openscad`` each FORK A
PROCESS. That is fine for ``harnesscad selftest``, which is a report a human asks
for, and unacceptable inside a unit-test suite -- one wedged subprocess wedges CI
forever. So the test suite never reaches an external engine: it runs on ``stub``
and ``frep`` (both in-process, both dependency-free) with tiny corpora, and the
multi-engine sweep is opt-in behind ``HARNESSCAD_SELFTEST_FULL=1``.

**3. Tolerance is per-backend and physical, not a magic number.**

  * ``cadquery`` / ``freecad`` are exact B-rep kernels: an analytic volume is
    matched to machine precision.
  * ``openscad`` / ``blender`` return a mesh: circles are polygonised, so the
    volume of a curved part lands ~0.1-1% low. Planar parts are exact.
  * ``frep`` samples a signed distance field on a grid of
    :data:`~harnesscad.io.backends.frep.DEFAULT_RESOLUTION` cells and marches it.
    Its error therefore SCALES WITH THE PART: the cell size is
    ``max(extent) / resolution``. That expected error is not a disagreement.

The tolerance is deliberately tight enough that the bugs we know about stay
visible: a 3 mm shell that dilates a 60 mm box by 3 mm is 4.8x the frep bbox
tolerance at that size, and a shell whose volume is 32% off is 8x the frep volume
tolerance. A backend that hides behind its own tolerance is a backend whose
tolerance is wrong.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Op
from harnesscad.core.loop import HarnessSession
from harnesscad.core.state.opdag import OpDAG

__all__ = [
    "BACKENDS",
    "GEOMETRIC_BACKENDS",
    "Tolerance",
    "TOLERANCES",
    "Observation",
    "BackendFactory",
    "resolve",
    "available",
    "observe",
    "observe_steps",
    "scale_ops",
    "bbox_delta",
    "volume_rel_delta",
]


#: Every backend the harness can drive, in the order a report should list them.
BACKENDS: Tuple[str, ...] = (
    "stub", "frep", "cadquery", "build123d", "freecad", "openscad", "blender",
    "manifold", "rhino3dm", "microcad", "truck")

#: The ones that answer ``query('measure')`` with real geometry. ``stub`` does
#: not: it is a bookkeeping backend and cannot take part in a geometric oracle.
GEOMETRIC_BACKENDS: Tuple[str, ...] = (
    "frep", "cadquery", "build123d", "freecad", "truck", "openscad", "blender",
    "manifold", "rhino3dm", "microcad")

#: Preference order when a consensus has to be named: an exact B-rep kernel first.
#: build123d and cadquery are both OCCT B-rep, so they rank alongside freecad.
#: microcad is a meshed CSG language (like openscad), so it ranks with the meshers.
EXACTNESS_ORDER: Tuple[str, ...] = (
    "cadquery", "build123d", "freecad", "truck", "openscad", "blender", "manifold",
    "microcad", "rhino3dm", "frep")


@dataclass(frozen=True)
class Tolerance:
    """How far a backend is allowed to be from the truth before it is a bug.

    ``volume_rel``    -- relative volume error on a chunky part.
    ``volume_thin``   -- EXTRA relative volume error proportional to
                         ``cell / min_extent``: a sampled backend loses a fraction
                         of a part whose thinnest feature is only a few cells
                         thick, and that fraction grows exactly as that ratio.
    ``bbox_abs``      -- absolute bbox error in model units.
    ``bbox_scaled``   -- extra bbox error proportional to the part's largest
                         extent (the grid term: cell = max_extent / resolution).
    ``cells``         -- how many cells the sampler puts across the largest
                         extent (0 = not a sampled backend).
    ``kind``          -- 'brep' | 'mesh' | 'field' | 'none', for the report.
    """

    volume_rel: float
    bbox_abs: float
    bbox_scaled: float = 0.0
    volume_thin: float = 0.0
    cells: int = 0
    kind: str = "brep"

    def cell(self, max_extent: float) -> float:
        if self.cells <= 0:
            return 0.0
        return float(max_extent) / float(self.cells)

    def bbox_tol(self, extent: float) -> float:
        return self.bbox_abs + self.bbox_scaled * max(float(extent), 0.0)

    def volume_tol(self, max_extent: float = 0.0, min_extent: float = 0.0) -> float:
        """Relative volume tolerance for a part of these extents.

        For an exact kernel this is a constant. For a sampled one it is not: the
        grid is fixed at ``cells`` across the LARGEST extent, so a 100x50x3 plate
        gets cells 2 mm across trying to resolve a 3 mm wall, and loses ~4% of it.
        Charging that as a bug would bury the real ones -- and setting a single
        flat tolerance loose enough to cover it (8%+) would hide a 12% shell bug.
        So the tolerance is derived from the part: ``cell / min_extent``.
        """
        if self.volume_thin <= 0.0 or min_extent <= 0.0:
            return self.volume_rel
        ratio = self.cell(max_extent) / float(min_extent)
        return min(self.volume_rel + self.volume_thin * ratio, 0.25)


#: frep's grid is DEFAULT_RESOLUTION cells across the part's largest extent, so
#: half a cell is extent/96 ~= 1.04% of the extent. The measured bbox error is far
#: smaller than that (marching cubes interpolates), so half a cell is a safe,
#: physically-derived ceiling rather than a fudge factor.
_FREP_CELLS = 48
_FREP_HALF_CELL = 0.5 / _FREP_CELLS

TOLERANCES: Dict[str, Tolerance] = {
    # Exact OCCT B-rep: an analytic volume must match to machine precision.
    "cadquery": Tolerance(1e-9, 1e-6, 0.0, 0.0, 0, "brep"),
    "build123d": Tolerance(1e-9, 1e-6, 0.0, 0.0, 0, "brep"),
    "freecad": Tolerance(1e-9, 1e-6, 0.0, 0.0, 0, "brep"),
    # Meshed CSG: exact on planar parts, polygonisation error on curved ones.
    "openscad": Tolerance(0.01, 1e-3, 0.0, 0.0, 0, "mesh"),
    "blender": Tolerance(0.01, 1e-3, 0.0, 0.0, 0, "mesh"),
    # Manifold: a guaranteed-manifold mesh-boolean kernel. Same regime as the
    # other mesh engines -- exact on planar parts, shared polygonisation error on
    # curved ones (it facets a circle by the SAME $fn law, so the error matches).
    "manifold": Tolerance(0.01, 1e-3, 0.0, 0.0, 0, "mesh"),
    # microcad: a meshed CSG *language* (µcad CLI). Same regime as the other mesh
    # engines -- exact on planar parts, polygonisation error on curved ones.
    "microcad": Tolerance(0.01, 1e-3, 0.0, 0.0, 0, "mesh"),
    # Sampled SDF: the error scales with the grid, and the grid scales with the
    # part. 1% on a chunky part, plus 20% of (cell / thinnest extent) -- which is
    # the measured behaviour: a 100x50x3 plate (cell/min = 0.69) lands 4.4% low, a
    # 120x30x6 five-hole strip (0.42) lands 6.5% low. Still 10x tighter than the
    # smallest real shell bug (11.7%).
    "frep": Tolerance(0.01, 0.05, _FREP_HALF_CELL, 0.20, _FREP_CELLS, "field"),
    # openNURBS container backend: it only builds box/cylinder primitives, and it
    # reports their volume analytically (w*h*d cross-checked against Box.Volume;
    # pi*r^2*d) with the bounding box read off rhino3dm's own GetBoundingBox. Both
    # are exact for these primitives, so it is an exact voice on the ops it
    # supports -- and it REFUSES everything else rather than guess.
    "rhino3dm": Tolerance(1e-9, 1e-6, 0.0, 0.0, 0, "brep"),
    # truck: a from-scratch Rust B-rep NURBS kernel (NOT OCCT). It is exact on
    # planar solids (a box is 48000 to the bit), but volume/bbox are read back
    # from truck's OWN tessellation of its NURBS surfaces, so a curved solid of
    # revolution carries a small, tolerance-bounded facet error -- the mesh
    # regime, not the machine-precision regime of the OCCT B-reps.
    "truck": Tolerance(0.01, 1e-3, 0.0, 0.0, 0, "brep"),
    # No geometry at all.
    "stub": Tolerance(math.inf, math.inf, 0.0, 0.0, 0, "none"),
}


def tolerance(name: str) -> Tolerance:
    return TOLERANCES.get(name, Tolerance(0.05, 0.5, 0.0, 0.0, 0, "unknown"))


#: A test may hand an oracle its own backend maker -- that is how a DELIBERATELY
#: CORRUPTED backend is injected to prove the oracle actually detects one.
#: ``factory(name) -> backend | None``; ``None`` means "use the real one".
BackendFactory = Callable[[str], Optional[Any]]


@dataclass
class Observation:
    """What one backend saw when it ran one op stream. Comparable across backends."""

    backend: str
    available: bool = True
    skip_reason: str = ""
    ok: bool = False
    applied: int = 0
    rejected: Optional[str] = None          # the op tag that was refused, if any
    codes: List[str] = field(default_factory=list)
    error: str = ""                          # an exception escaped the backend
    volume: Optional[float] = None
    bbox: Optional[Tuple[float, float, float]] = None
    genus: Optional[int] = None
    watertight: Optional[bool] = None
    manifold: Optional[bool] = None
    solid_present: Optional[bool] = None
    digest: str = ""

    # -- derived ------------------------------------------------------------
    @property
    def geometric(self) -> bool:
        """True when this observation carries geometry that can be compared.

        ``ok`` is part of it. When a backend REFUSES an op it keeps -- and happily
        reports -- the pre-op solid: OpenSCAD, asked to shell a 60x40x20 box it
        cannot shell, still answers 48000 mm3, the volume of the unshelled stock.
        That number is a measurement of a DIFFERENT part. Comparing it would
        manufacture a disagreement out of an engine that was honest enough to
        decline, and bury the real ones underneath. A refusal is a capability gap
        and is reported as one.
        """
        return (self.available and not self.error and self.ok
                and self.volume is not None and self.bbox is not None)

    @property
    def extent(self) -> float:
        return max(self.bbox) if self.bbox else 0.0

    @property
    def min_extent(self) -> float:
        return min(self.bbox) if self.bbox else 0.0

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if self.bbox is not None:
            d["bbox"] = [float(v) for v in self.bbox]
        return d


# --- backend resolution ----------------------------------------------------

def resolve(name: str, factory: Optional[BackendFactory] = None):
    """Return (backend, skip_reason). ``backend`` is None when unavailable.

    A silent fallback to the stub counts as UNAVAILABLE: an oracle that compared
    two stubs and found them to agree would be lying.
    """
    if factory is not None:
        injected = factory(name)
        if injected is not None:
            return injected, ""
    from harnesscad.io.surfaces.server import _make_backend

    try:
        backend, resolved, note = _make_backend(name)
    except Exception as exc:  # noqa: BLE001 - an unavailable tool must not crash us
        return None, f"{name} unavailable ({exc})"
    if resolved != name:
        return None, note or f"{name} unavailable (fell back to {resolved})"
    return backend, ""


def available(names: Optional[Sequence[str]] = None,
              factory: Optional[BackendFactory] = None) -> List[str]:
    """The subset of ``names`` whose tool is actually installed here. Ordered."""
    wanted = tuple(names) if names is not None else BACKENDS
    return [n for n in wanted if resolve(n, factory)[0] is not None]


# --- measurement -----------------------------------------------------------

def _read(backend: Any, session: HarnessSession, name: str,
          result: Any, obs: Observation) -> Observation:
    measure = _query(backend, "measure")
    validity = _query(backend, "validity")
    obs.ok = bool(getattr(result, "ok", False))
    obs.applied = int(getattr(result, "applied", 0))
    rej = getattr(result, "rejected", None)
    obs.rejected = str(rej.get("op")) if isinstance(rej, dict) else None
    obs.codes = sorted({d.code for d in getattr(result, "diagnostics", [])})
    vol = measure.get("volume")
    bbox = measure.get("bbox")
    if isinstance(vol, (int, float)):
        obs.volume = float(vol)
    if isinstance(bbox, (list, tuple)) and len(bbox) == 3:
        obs.bbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]))
    g = validity.get("genus")
    obs.genus = int(g) if isinstance(g, (int, float)) else None
    obs.watertight = validity.get("watertight")
    obs.manifold = validity.get("manifold")
    obs.solid_present = validity.get("solid_present")
    try:
        obs.digest = str(backend.state_digest())
    except Exception:  # noqa: BLE001
        obs.digest = ""
    return obs


def _query(backend: Any, what: str) -> dict:
    try:
        res = backend.query(what)
    except Exception:  # noqa: BLE001 - a hostile backend must not crash the oracle
        return {}
    return res if isinstance(res, dict) else {}


def observe(name: str, ops: Sequence[Op],
            factory: Optional[BackendFactory] = None,
            verify_level: str = "core") -> Observation:
    """Run ``ops`` on backend ``name`` and measure the result. Never raises."""
    backend, skip = resolve(name, factory)
    if backend is None:
        return Observation(name, available=False, skip_reason=skip)
    obs = Observation(name)
    try:
        session = HarnessSession(backend, verify_level=verify_level)
        result = session.apply_ops(list(ops))
    except Exception as exc:  # noqa: BLE001 - a crash IS a finding; record it
        obs.error = f"{type(exc).__name__}: {exc}"
        return obs
    return _read(backend, session, name, result, obs)


def observe_steps(name: str, ops: Sequence[Op],
                  factory: Optional[BackendFactory] = None
                  ) -> Tuple[List[Observation], List[Op]]:
    """Measure after EVERY op (this is what a metamorphic property needs).

    Returns (observations, applied_ops); ``observations[i]`` is the state after
    ``ops[i]``. An op the backend refuses ends the walk -- the refusal is
    recorded in the last observation and the caller sees a shorter list.
    """
    backend, skip = resolve(name, factory)
    if backend is None:
        return [Observation(name, available=False, skip_reason=skip)], []
    out: List[Observation] = []
    applied: List[Op] = []
    try:
        session = HarnessSession(backend, verify_level="core")
    except Exception as exc:  # noqa: BLE001
        return [Observation(name, error=f"{type(exc).__name__}: {exc}")], []
    for op in ops:
        obs = Observation(name)
        try:
            result = session.apply_ops([op])
        except Exception as exc:  # noqa: BLE001
            obs.error = f"{type(exc).__name__}: {exc}"
            out.append(obs)
            break
        _read(backend, session, name, result, obs)
        out.append(obs)
        if not obs.ok:
            break
        applied.append(op)
    return out, applied


# --- op-stream transforms + comparison helpers -----------------------------

#: Every op field that carries a LENGTH (and so scales linearly with the part).
#: Angles, counts, indices and names deliberately do not appear.
_LENGTH_FIELDS = frozenset({
    "x", "y", "z", "x1", "y1", "x2", "y2", "cx", "cy", "r", "w", "h",
    "distance", "radius", "thickness", "diameter", "depth", "spacing",
})
#: Fields that are a TUPLE of lengths (two points defining an axis).
_LENGTH_TUPLE_FIELDS = frozenset({"axis"})


def scale_ops(ops: Sequence[Op], k: float) -> List[Op]:
    """Uniformly scale an op stream by ``k`` -- the metamorphic transform.

    The harness has no ``scale`` op, so the relation is stated on the INPUT: a
    plan whose every length is multiplied by k must build a part whose volume is
    k^3 times larger and whose bbox is k times larger. Everything else (angles,
    counts, plane names, references) is left exactly as it was.
    """
    out: List[Op] = []
    for op in ops:
        changes: Dict[str, Any] = {}
        for f in dataclasses.fields(op):
            value = getattr(op, f.name)
            if f.name in _LENGTH_FIELDS and isinstance(value, (int, float)):
                changes[f.name] = float(value) * k
            elif f.name in _LENGTH_TUPLE_FIELDS and isinstance(value, tuple):
                changes[f.name] = tuple(float(v) * k for v in value)
        out.append(dataclasses.replace(op, **changes) if changes else op)
    return out


def plan_opdag(ops: Sequence[Op]) -> OpDAG:
    """An OpDAG holding the WHOLE plan, whether or not a backend accepted it.

    The LINT-tier verifiers read the op stream, not the solid: asking them "is
    this plan bad?" means showing them the plan. If we only showed them the ops a
    backend agreed to build, every plan the backend rejected would score as a
    false negative against a fleet that was never given the evidence.
    """
    dag = OpDAG()
    for op in ops:
        dag.append(op)
    return dag


def volume_rel_delta(a: Optional[float], b: Optional[float]) -> float:
    """Relative volume difference; inf when only one side has a volume."""
    if a is None or b is None:
        return math.inf
    scale = max(abs(a), abs(b))
    if scale < 1e-12:
        return 0.0
    return abs(a - b) / scale


def bbox_delta(a: Optional[Sequence[float]],
               b: Optional[Sequence[float]]) -> float:
    """Largest per-axis absolute bbox difference; inf when one side has none."""
    if a is None or b is None:
        return math.inf
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))
