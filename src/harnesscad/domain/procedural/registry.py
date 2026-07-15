"""Procedural generator registry -- named generators that EMIT CISP OPS.

``domain/procedural`` carried shape grammars, Markov grammars, arrays, tilings,
voxel templates, symmetry operators, key-parameter templates and an incremental
rebuild graph. Every one of them was correct, tested, and reachable from nothing
but its own test. This module is the dispatcher: a generator is selected **by
name**, given parameters, and returns a list of
:mod:`harnesscad.core.cisp.ops` operations that apply cleanly to a
:class:`~harnesscad.core.loop.HarnessSession`.

    ops = emit("shape_grammar", seed=0)
    session.apply_ops(ops)          # <- ordinary CISP, ordinary verification

RIVALS ARE SELECTED BY NAME, NEVER BLENDED
------------------------------------------
*   **grammar** -- ``shape_grammar`` is a context-FREE weighted expansion (every
    alternative for a symbol has one fixed weight). ``markov_grammar``
    (ShapeGraMM) conditions the choice on the PARENT RULE, so the same symbol
    expands differently depending on how it was reached. These are different
    formalisms with different derivations; averaging their outputs is meaningless.
*   **pattern** -- ``patterns`` returns bare 3-tuples of offsets (the tiling
    primitive). ``array_patterns`` (AutoCAD-style) returns
    :class:`Placement` records that also carry a ROTATION, and adds a
    fit-to-length solve. A polar array that rotates its items is not the same
    answer as a radial ring that does not.

Both families are exposed under their own names. Nothing here merges them.

Geometry-only routes (a scene of instances, a parameter realisation, a
re-evaluation graph) are kept separate from the op-emitting routes: they return
their own data, not CISP.

Discovery goes through :mod:`harnesscad.registry`. Adapters live here; the
procedural modules are never modified. Deterministic (every route takes an
explicit ``seed``), stdlib-only, no network.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry
from harnesscad.core.cisp.ops import (
    AddCircle, AddLine, AddRectangle, Extrude, NewSketch, Op,
)

__all__ = [
    "ProceduralError",
    "UnknownGenerator",
    "generators",
    "generator_doc",
    "emit",
    "apply_to",
    "scene",
    "expand_scene",
    "realize_parameters",
    "rebuild",
    "freeze",
    "roof",
    "RIVAL_FAMILIES",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

PROCEDURAL_PACKAGE = "procedural"
_PKG = "harnesscad.domain.procedural."


class ProceduralError(ValueError):
    """Base class for every procedural-surface failure."""


class UnknownGenerator(ProceduralError):
    """A generator name that is not registered."""


# --------------------------------------------------------------------------- #
# Op-emission helpers (the ONE place a placement becomes CISP)
# --------------------------------------------------------------------------- #
def _boxes(cells: Sequence[Tuple[float, float, float, float, float, float]],
           start_sketch: int = 1) -> List[Op]:
    """(x, y, w, h, z_unused, depth) cells -> one sketch + rectangle + extrude each.

    The stub backend names sketches ``sk1``, ``sk2``, ... in creation order, so
    an emitted batch is self-consistent as long as the caller says which index
    the session is at (``start_sketch``, 1 for a fresh session).
    """
    ops: List[Op] = []
    idx = start_sketch
    for (x, y, w, h, _z, depth) in cells:
        if w <= 0.0 or h <= 0.0 or depth <= 0.0:
            continue
        ops.append(NewSketch(plane="XY"))
        ops.append(AddRectangle(sketch="sk%d" % idx, x=float(x), y=float(y),
                                w=float(w), h=float(h)))
        ops.append(Extrude(sketch="sk%d" % idx, distance=float(depth)))
        idx += 1
    return ops


def _discs(points: Sequence[Tuple[float, float]], radius: float, depth: float,
           start_sketch: int = 1) -> List[Op]:
    """A ring/lattice of points -> one sketch of circles, extruded once.

    A circle is the honest emission for a placement that carries NO orientation
    (``procedural.patterns`` yields bare offsets): a disc is rotation-invariant,
    so nothing is being silently invented.
    """
    if not points:
        return []
    ops: List[Op] = [NewSketch(plane="XY")]
    sid = "sk%d" % start_sketch
    for (x, y) in points:
        ops.append(AddCircle(sketch=sid, cx=float(x), cy=float(y), r=float(radius)))
    ops.append(Extrude(sketch=sid, distance=float(depth)))
    return ops


def _oriented_squares(placements: Sequence[Any], size: float, depth: float,
                      start_sketch: int = 1) -> List[Op]:
    """Placements that carry a ROTATION -> square profiles actually turned by it.

    This is where ``array_patterns`` earns its separate name. CISP has no rotated
    rectangle, so the square is emitted as four AddLine segments at the
    placement's angle -- the rotation ends up in the geometry instead of being
    quietly dropped (which is exactly what happens if you emit a disc).
    """
    if not placements:
        return []
    ops: List[Op] = [NewSketch(plane="XY")]
    sid = "sk%d" % start_sketch
    half = float(size) / 2.0
    corners = ((-half, -half), (half, -half), (half, half), (-half, half))
    for p in placements:
        ang = math.radians(float(p.rotation))
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        pts = [(float(p.x) + cx * cos_a - cy * sin_a,
                float(p.y) + cx * sin_a + cy * cos_a) for (cx, cy) in corners]
        for i in range(4):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % 4]
            ops.append(AddLine(sketch=sid, x1=x1, y1=y1, x2=x2, y2=y2))
    ops.append(Extrude(sketch=sid, distance=float(depth)))
    return ops


# --------------------------------------------------------------------------- #
# Family: PATTERN (rivals -- patterns vs array_patterns)
# --------------------------------------------------------------------------- #
def _gen_patterns_linear(count: int = 4, spacing: float = 20.0,
                         radius: float = 3.0, depth: float = 5.0,
                         start_sketch: int = 1, **_kw) -> List[Op]:
    from harnesscad.domain.procedural.patterns import linear

    pts = [(p[0], p[1]) for p in linear(int(count), float(spacing))]
    return _discs(pts, radius, depth, start_sketch)


def _gen_patterns_grid(rows: int = 2, cols: int = 3, spacing: float = 20.0,
                       radius: float = 3.0, depth: float = 5.0,
                       start_sketch: int = 1, **_kw) -> List[Op]:
    from harnesscad.domain.procedural.patterns import grid

    pts = [(p[0], p[1]) for p in grid(int(rows), int(cols),
                                      (float(spacing), float(spacing)))]
    return _discs(pts, radius, depth, start_sketch)


def _gen_patterns_radial(count: int = 6, radius: float = 30.0,
                         hole_radius: float = 3.0, depth: float = 5.0,
                         start_sketch: int = 1, **_kw) -> List[Op]:
    from harnesscad.domain.procedural.patterns import radial

    pts = [(p[0], p[1]) for p in radial(int(count), float(radius))]
    return _discs(pts, hole_radius, depth, start_sketch)


def _gen_patterns_pipe(points: Optional[Sequence[Sequence[float]]] = None,
                       radius: float = 3.0, depth: float = 5.0,
                       start_sketch: int = 1, **_kw) -> List[Op]:
    """A polyline -> a disc at every segment midpoint (a swept-pipe stand-in)."""
    from harnesscad.domain.procedural.patterns import pipe

    pts = points or [(0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (30.0, 20.0, 0.0)]
    segs = pipe([tuple(float(v) for v in p) for p in pts])
    mids = [((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0) for a, b in segs]
    return _discs(mids, radius, depth, start_sketch)


def _gen_array_linear(count: int = 4, spacing: float = 20.0, w: float = 10.0,
                      h: float = 10.0, depth: float = 5.0,
                      start_sketch: int = 1, **_kw) -> List[Op]:
    from harnesscad.domain.procedural.array_patterns import linear_array

    places = linear_array((0.0, 0.0), int(count), (float(spacing), 0.0))
    return _boxes([(p.x, p.y, w, h, 0.0, depth) for p in places], start_sketch)


def _gen_array_rectangular(rows: int = 2, cols: int = 3, row_step: float = 25.0,
                           col_step: float = 25.0, w: float = 10.0, h: float = 10.0,
                           depth: float = 5.0, start_sketch: int = 1,
                           **_kw) -> List[Op]:
    from harnesscad.domain.procedural.array_patterns import rectangular_array

    places = rectangular_array((0.0, 0.0), int(rows), int(cols),
                               (0.0, float(row_step)), (float(col_step), 0.0))
    return _boxes([(p.x, p.y, w, h, 0.0, depth) for p in places], start_sketch)


def _gen_array_polar(count: int = 6, radius: float = 30.0, size: float = 6.0,
                     depth: float = 5.0, rotate_items: bool = True,
                     start_sketch: int = 1, **_kw) -> List[Op]:
    """The AutoCAD polar array -- unlike ``patterns.radial`` it also ROTATES items.

    The rotation is emitted (see :func:`_oriented_squares`), so
    ``rotate_items=True`` and ``rotate_items=False`` produce DIFFERENT geometry.
    That difference is the whole reason this is a separate family.
    """
    from harnesscad.domain.procedural.array_patterns import polar_array

    places = polar_array((0.0, 0.0), float(radius), int(count),
                         rotate_items=bool(rotate_items))
    return _oriented_squares(places, size, depth, start_sketch)


def _gen_array_fit_linear(total_length: float = 100.0, pitch: float = 25.0,
                          w: float = 10.0, h: float = 10.0, depth: float = 5.0,
                          start_sketch: int = 1, **_kw) -> List[Op]:
    """Fit as many instances as the run allows -- the capability ``patterns`` lacks."""
    from harnesscad.domain.procedural.array_patterns import fit_linear_array

    places = fit_linear_array((0.0, 0.0), float(total_length), float(pitch))
    return _boxes([(p.x, p.y, w, h, 0.0, depth) for p in places], start_sketch)


# --------------------------------------------------------------------------- #
# Family: GRAMMAR (rivals -- shape_grammar vs markov_grammar)
# --------------------------------------------------------------------------- #
_DEFAULT_TERMINALS = ("base", "shaft", "cap")


def _gen_shape_grammar(seed: int = 0, cell: float = 12.0, depth: float = 6.0,
                       max_depth: int = 6, start_sketch: int = 1,
                       **_kw) -> List[Op]:
    """Context-free weighted expansion. One extruded cell per derived terminal."""
    from harnesscad.domain.procedural.shape_grammar import Production, derive

    productions = (
        Production("tower", ("base", "body"), 1.0),
        Production("body", ("shaft", "body"), 1.0),
        Production("body", ("cap",), 2.0),
    )
    terminals, _trace, _diag = derive("tower", productions, _DEFAULT_TERMINALS,
                                      seed=int(seed), max_depth=int(max_depth))
    cells = []
    for i, (symbol, _state) in enumerate(terminals):
        width = {"base": 2.0, "shaft": 1.0, "cap": 1.5}.get(symbol, 1.0) * float(cell)
        cells.append((0.0, i * float(cell), width, float(cell), 0.0, float(depth)))
    return _boxes(cells, start_sketch)


def _gen_markov_grammar(seed: int = 0, cell: float = 12.0, depth: float = 6.0,
                        max_depth: int = 8, start_sketch: int = 1,
                        **_kw) -> List[Op]:
    """ShapeGraMM: the SAME symbol expands differently depending on its PARENT rule."""
    from harnesscad.domain.procedural.markov_grammar import MarkovGrammar, Rule

    rules = (
        Rule("r0", "tower", ("base", "body")),
        Rule("r1", "body", ("shaft", "body")),
        Rule("r2", "body", ("cap",)),
    )
    # After a shaft-producing rule, a cap is twice as likely: the non-random
    # repetition pattern a context-free grammar cannot express.
    transitions = {"r1": {"r1": 1.0, "r2": 2.0}}
    grammar = MarkovGrammar(rules, _DEFAULT_TERMINALS, transitions)
    terminals, _trace, _diag = grammar.expand("tower", seed=int(seed),
                                              max_depth=int(max_depth))
    cells = []
    for i, (symbol, lvl) in enumerate(terminals):
        width = {"base": 2.0, "shaft": 1.0, "cap": 1.5}.get(symbol, 1.0) * float(cell)
        cells.append((0.0, i * float(cell), width, float(cell), 0.0,
                      float(depth) * (1.0 + 0.1 * lvl)))
    return _boxes(cells, start_sketch)


# --------------------------------------------------------------------------- #
# Family: VOXEL (brick templates; compose builds ON the templates, not against them)
# --------------------------------------------------------------------------- #
def _brick_cells(model: dict, unit: float) -> List[Tuple[float, ...]]:
    return [(b["x"] * unit, b["y"] * unit, unit, unit, 0.0, unit)
            for b in model.get("bricks", [])]


def _gen_brick_template(category: str = "table", width: int = 6, depth: int = 6,
                        height: int = 6, seed: int = 0, unit: float = 8.0,
                        start_sketch: int = 1, **_kw) -> List[Op]:
    from harnesscad.domain.procedural.brick_templates import get_template

    model = get_template(str(category), int(width), int(depth), int(height),
                         seed=int(seed))
    return _boxes(_brick_cells(model, float(unit)), start_sketch)


def _gen_voxel_compose(categories: Sequence[str] = ("table", "chair"),
                       width: int = 6, depth: int = 6, height: int = 6,
                       seed: int = 0, spacing: int = 2, unit: float = 8.0,
                       mutation: str = "", start_sketch: int = 1,
                       **_kw) -> List[Op]:
    """Several templates laid side by side, optionally after a parametric mutation."""
    from harnesscad.domain.procedural.brick_templates import get_template
    from harnesscad.domain.procedural.voxel_compose import compose, mutate_dims

    w, d, h = int(width), int(depth), int(height)
    if mutation:
        w, d, h = mutate_dims(w, d, h, str(mutation))
    models = [get_template(str(c), w, d, h, seed=int(seed)) for c in categories]
    merged = compose(models, spacing=int(spacing))
    return _boxes(_brick_cells(merged, float(unit)), start_sketch)


# --------------------------------------------------------------------------- #
# Family: SYMMETRY
# --------------------------------------------------------------------------- #
def _gen_symmetry(kind: str = "nfold", order: int = 6, motif: Optional[Sequence] = None,
                  radius: float = 3.0, depth: float = 5.0, axis: str = "y",
                  start_sketch: int = 1, **_kw) -> List[Op]:
    """Expand a motif under an n-fold / bilateral / dihedral symmetry group."""
    from harnesscad.domain.procedural.symmetry import bilateral, dihedral, nfold

    pts = [tuple(float(v) for v in p) for p in (motif or [(25.0, 0.0)])]
    if kind == "nfold":
        copies = nfold(pts, int(order))
    elif kind == "bilateral":
        copies = bilateral(pts, axis=str(axis))
    elif kind == "dihedral":
        copies = dihedral(pts, int(order))
    else:
        raise ProceduralError(
            "unknown symmetry kind %r; known: nfold, bilateral, dihedral" % kind)
    out: List[Tuple[float, float]] = []
    for copy in copies:
        for p in copy:
            out.append((round(float(p[0]), 9), round(float(p[1]), 9)))
    return _discs(out, radius, depth, start_sketch)


# --------------------------------------------------------------------------- #
# The generator table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Gen:
    fn: Callable[..., List[Op]]
    module: str
    family: str
    doc: str


_GENERATORS: Dict[str, _Gen] = {
    # -- pattern family (RIVALS) ------------------------------------------- #
    "patterns.linear": _Gen(
        _gen_patterns_linear, _PKG + "patterns", "pattern",
        "evenly spaced offsets along +X (no rotation)"),
    "patterns.grid": _Gen(
        _gen_patterns_grid, _PKG + "patterns", "pattern",
        "rows x cols lattice of offsets"),
    "patterns.radial": _Gen(
        _gen_patterns_radial, _PKG + "patterns", "pattern",
        "a ring of offsets (items NOT rotated)"),
    "patterns.pipe": _Gen(
        _gen_patterns_pipe, _PKG + "patterns", "pattern",
        "a polyline broken into segments"),
    "array.linear": _Gen(
        _gen_array_linear, _PKG + "array_patterns", "pattern",
        "AutoCAD linear array -- Placements carrying a rotation"),
    "array.rectangular": _Gen(
        _gen_array_rectangular, _PKG + "array_patterns", "pattern",
        "AutoCAD rectangular array (row/col step vectors)"),
    "array.polar": _Gen(
        _gen_array_polar, _PKG + "array_patterns", "pattern",
        "AutoCAD polar array -- items ARE rotated, and the rotation is emitted"),
    "array.fit_linear": _Gen(
        _gen_array_fit_linear, _PKG + "array_patterns", "pattern",
        "fit as many instances as a run of given length allows"),
    # -- grammar family (RIVALS) ------------------------------------------- #
    "shape_grammar": _Gen(
        _gen_shape_grammar, _PKG + "shape_grammar", "grammar",
        "context-free weighted shape grammar (fixed per-alternative weights)"),
    "markov_grammar": _Gen(
        _gen_markov_grammar, _PKG + "markov_grammar", "grammar",
        "ShapeGraMM -- rule choice conditioned on the PARENT rule (a Markov chain)"),
    # -- voxel family ------------------------------------------------------ #
    "brick_template": _Gen(
        _gen_brick_template, _PKG + "brick_templates", "voxel",
        "parametric voxel object (table/chair/tower/car/bookshelf/basket/bottle/bus)"),
    "voxel_compose": _Gen(
        _gen_voxel_compose, _PKG + "voxel_compose", "voxel",
        "several voxel templates side by side, with an optional dimension mutation"),
    # -- symmetry ---------------------------------------------------------- #
    "symmetry": _Gen(
        _gen_symmetry, _PKG + "symmetry", "symmetry",
        "expand a motif under an n-fold / bilateral / dihedral symmetry group"),
}


def generators() -> Tuple[str, ...]:
    """Every generator whose module is present in this tree. Deterministic order."""
    return tuple(sorted(n for n, g in _GENERATORS.items() if _available(g.module)))


def generator_doc(name: str) -> str:
    try:
        return _GENERATORS[name].doc
    except KeyError:
        raise UnknownGenerator("unknown generator %r" % name) from None


def emit(name: str, **params) -> List[Op]:
    """Run ONE named generator and return the CISP ops it emits.

    ``params`` are the generator's own knobs (see ``--list``). Every generator
    takes ``start_sketch`` (default 1: a fresh session) and, where it is
    stochastic, an explicit ``seed``. No generator touches the global RNG.
    """
    try:
        gen = _GENERATORS[name]
    except KeyError:
        raise UnknownGenerator(
            "unknown generator %r; known: %s"
            % (name, ", ".join(sorted(_GENERATORS)))) from None
    if not _available(gen.module):
        raise UnknownGenerator("generator %r is not present in this tree" % name)
    return gen.fn(**params)


def apply_to(session: Any, name: str, **params):
    """Emit and apply in one step, continuing from the session's sketch count.

    Returns the session's own ``ApplyOpsResult`` -- ordinary CISP, ordinary
    verification, nothing special.
    """
    if "start_sketch" not in params:
        summary = {}
        try:
            summary = session.backend.query("summary") or {}
        except Exception:  # noqa: BLE001 - a backend without 'summary' starts at 1
            summary = {}
        params["start_sketch"] = int(summary.get("sketch_count", 0)) + 1
    return session.apply_ops(emit(name, **params))


# --------------------------------------------------------------------------- #
# Geometry-only routes (NOT op emitters -- they return their own data)
# --------------------------------------------------------------------------- #
def scene(dims: Tuple[int, int, int] = (4, 4, 1), cell_size: float = 20.0,
          geometries: Sequence[str] = ("cube", "cylinder"),
          camera: Tuple[float, float, float] = (0.0, 0.0, 200.0),
          max_objects: int = 4, seed: int = 0,
          focal_length: float = 1.0,
          thresholds: Sequence[float] = (10.0, 40.0)):
    """A ShapeGraMM massive model, culled + LOD-resolved for one camera.

    Returns ``(instances, stats)`` from
    :func:`~harnesscad.domain.procedural.instantiate.generate_view` -- placed
    instances, not CISP: a million-instance scene is a renderer's problem, not
    a feature tree's.
    """
    from harnesscad.domain.procedural.instantiate import MassiveModel, generate_view
    from harnesscad.domain.procedural.scope_culling import make_aabb_frustum

    model = MassiveModel(dims=tuple(int(d) for d in dims),
                         cell_size=float(cell_size),
                         geometries=tuple(geometries),
                         max_objects=int(max_objects),
                         base_seed=int(seed))
    nx, ny, nz = model.dims
    lo = (0.0, 0.0, 0.0)
    hi = (nx * model.cell_size, ny * model.cell_size, nz * model.cell_size)
    planes = make_aabb_frustum(lo, hi)
    return generate_view(model, planes, tuple(float(c) for c in camera),
                         focal_length=float(focal_length),
                         thresholds=tuple(float(t) for t in thresholds))


def expand_scene(nodes: Sequence[Any], visible: Callable[[Any], bool],
                 children: Callable[[Any], Sequence[Any]],
                 terminal_key: Callable[[Any], Any]):
    """Lazy, cull-first expansion of a node hierarchy into batched terminals.

    Delegates to :func:`~harnesscad.domain.procedural.lazy_scene.expand`: the
    generic half of the massive-model pipeline, usable on any hierarchy (a
    grammar derivation tree, an assembly tree, a scene graph).
    """
    from harnesscad.domain.procedural.lazy_scene import expand

    return expand(list(nodes), visible, children, terminal_key)


def realize_parameters(key_values: Optional[Dict[str, float]] = None,
                       template: str = "two_stage_opamp") -> Dict[str, float]:
    """Key decisions -> the full dependent-parameter vector.

    The design-template half of the procedural tree: a handful of KEY parameters
    are chosen, and the rest are DERIVED. Returns the realised full vector.
    """
    from harnesscad.domain.procedural.key_params import two_stage_opamp_template

    if template != "two_stage_opamp":
        raise ProceduralError(
            "unknown key-parameter template %r; known: two_stage_opamp" % template)
    tmpl = two_stage_opamp_template()
    values = dict(key_values or {k: 1.0 for k in tmpl.key_params})
    return tmpl.realize(values)


def rebuild(inputs: Dict[str, Any], computes: Sequence[Tuple[str, Sequence[str], Callable]],
            edits: Optional[Dict[str, Any]] = None):
    """Build a procedural graph, evaluate it, then RE-evaluate only what an edit dirtied.

    Returns ``(values, recomputed)``: the final node values and HOW MANY nodes
    the edit actually forced to recompute (the point of the module -- a clean
    graph recomputes zero).
    """
    from harnesscad.domain.procedural.incremental_rebuild import ProceduralGraph

    graph = ProceduralGraph()
    for name in sorted(inputs):
        graph.add_input(name, inputs[name])
    for name, deps, fn in computes:
        graph.add_compute(name, list(deps), fn)
    graph.evaluate()
    _clean, recomputed = graph.evaluate()  # a clean graph recomputes nothing
    for name in sorted(edits or {}):
        graph.set_input(name, (edits or {})[name])
    if edits:
        _values, recomputed = graph.evaluate()
    values = {name: graph.value(name) for name, _deps, _fn in computes}
    return values, recomputed


def freeze(solution: Dict[str, Any], intangible_regions: Sequence[Any] = (),
           desirable_features: Sequence[Any] = (),
           candidate: Optional[Dict[str, Any]] = None):
    """Freeze the marked regions of a solution as hard constraints, then project.

    Returns ``(constraints, violations, projected)``: what is frozen, what a
    candidate violates, and the candidate projected back onto the frozen set.
    """
    from harnesscad.domain.procedural.constraint_freeze import (
        build_constraints, check_preserved, project_onto_constraints,
    )

    constraints = build_constraints(solution, list(intangible_regions),
                                    list(desirable_features))
    if candidate is None:
        return constraints, [], dict(solution)
    violations = check_preserved(candidate, constraints)
    projected = project_onto_constraints(candidate, constraints)
    return constraints, violations, projected


def roof(footprint: Sequence[Sequence[float]], wall_height: float,
         pitch_deg: float,
         generated_points: Optional[Sequence[Sequence[float]]] = None,
         generated_openings: Optional[Sequence[Sequence[float]]] = None,
         required_openings: Optional[Sequence[Sequence[float]]] = None,
         opening_tol: float = 1e-6):
    """A gable roof over a FIXED footprint, plus ShellMaker structural metrics.

    The ridge runs along the footprint's longer axis and the eaves sit exactly on
    its bounding box -- the immutable footprint is respected, never grown. When
    ``generated_points`` / openings are supplied, the footprint-violation and
    opening-preservation metrics are returned. Geometry-only: returns roof data
    and scores, not CISP.
    """
    from harnesscad.domain.procedural.exterior_completion import (
        footprint_violation, generate_gable_roof, opening_preservation,
    )

    fp = [tuple(float(v) for v in p) for p in footprint]
    r = generate_gable_roof(fp, float(wall_height), float(pitch_deg))
    out: Dict[str, Any] = {
        "ridge": [list(r.ridge[0]), list(r.ridge[1])],
        "eaves": [list(e) for e in r.eaves],
        "height": r.height,
    }
    if generated_points is not None:
        out["footprint_violation"] = footprint_violation(
            [tuple(float(v) for v in p) for p in generated_points], fp)
    if required_openings is not None:
        out["opening_preservation"] = opening_preservation(
            [tuple(float(v) for v in p) for p in (generated_openings or [])],
            [tuple(float(v) for v in p) for p in required_openings],
            float(opening_tol))
    return out


# --------------------------------------------------------------------------- #
# Rivals
# --------------------------------------------------------------------------- #
RIVAL_FAMILIES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("grammar",
     "Context-free weighted expansion (shape_grammar) vs parent-rule-conditioned "
     "Markov selection (markov_grammar, ShapeGraMM). Different formalisms, "
     "different derivations. Select one by name.",
     ("shape_grammar", "markov_grammar")),
    ("pattern",
     "Bare offset tuples (patterns.*) vs AutoCAD Placements that also carry a "
     "ROTATION and can fit-to-length (array.*). A rotated polar array is not a "
     "radial ring. Select one by name.",
     ("patterns.linear", "patterns.grid", "patterns.radial", "patterns.pipe",
      "array.linear", "array.rectangular", "array.polar", "array.fit_linear")),
)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index() -> Dict[str, Any]:
    return {e.dotted: e
            for e in capability_registry.find(package=PROCEDURAL_PACKAGE)}


def _available(dotted: str) -> bool:
    return dotted in _index()


def routed_modules() -> Tuple[str, ...]:
    direct = {g.module for g in _GENERATORS.values()}
    direct.update({
        _PKG + "instantiate",
        _PKG + "scope_culling",
        _PKG + "lod",
        _PKG + "lazy_scene",
        _PKG + "key_params",
        _PKG + "incremental_rebuild",
        _PKG + "constraint_freeze",
        _PKG + "exterior_completion",
    })
    return tuple(sorted(d for d in direct if _available(d)))


def discover() -> List[dict]:
    """Every registered route: op-emitting generators + the geometry-only routes."""
    rows: List[dict] = []
    for name in sorted(_GENERATORS):
        g = _GENERATORS[name]
        rows.append({"route": "emit", "name": name, "family": g.family,
                     "module": g.module, "doc": g.doc,
                     "present": _available(g.module)})
    for name, module, doc in (
        ("scene", _PKG + "instantiate",
         "massive-model instances for one camera (culled + LOD)"),
        ("expand_scene", _PKG + "lazy_scene",
         "lazy cull-first expansion of any node hierarchy into batched terminals"),
        ("realize_parameters", _PKG + "key_params",
         "key decisions -> the full dependent-parameter vector"),
        ("rebuild", _PKG + "incremental_rebuild",
         "re-evaluate only the nodes a parameter edit dirtied"),
        ("freeze", _PKG + "constraint_freeze",
         "freeze marked regions as constraints and project a candidate onto them"),
        ("roof", _PKG + "exterior_completion",
         "parametric gable roof over a fixed footprint + ShellMaker structural metrics"),
    ):
        rows.append({"route": "geometry", "name": name, "family": "",
                     "module": module, "doc": doc, "present": _available(module)})
    return rows


UNADAPTED_REASONS: Dict[str, str] = {}


def unadapted() -> List[Tuple[str, str]]:
    routed = set(routed_modules())
    out = []
    for dotted in sorted(_index()):
        if dotted in routed or dotted.endswith(".registry"):
            continue
        out.append((dotted, UNADAPTED_REASONS.get(dotted, "no route yet")))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every registered procedural route")
    parser.add_argument("--rivals", action="store_true",
                        help="list the rival families (selected by name, never blended)")
    parser.add_argument("--unadapted", action="store_true",
                        help="list procedural modules with no route")
    parser.add_argument("--gen", default=None,
                        help="the generator to run (see --list)")
    parser.add_argument("--param", action="append", default=[], metavar="K=V",
                        help="a generator parameter (repeatable); V is parsed as JSON")
    parser.add_argument("--apply", action="store_true",
                        help="apply the emitted ops to a stub-backed HarnessSession")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def _params(pairs: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ProceduralError("--param expects K=V, got %r" % pair)
        key, _, raw = pair.partition("=")
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "rivals", False):
        for family, doc, members in RIVAL_FAMILIES:
            print("%s: (selected by name, NEVER blended)" % family)
            print("    %s" % doc)
            for m in members:
                print("    - %s" % m)
        return 0

    if getattr(args, "unadapted", False):
        for dotted, reason in unadapted():
            print("%s\n    %s" % (dotted, reason))
        return 0

    name = getattr(args, "gen", None)
    if name:
        params = _params(getattr(args, "param", []) or [])
        ops = emit(name, **params)
        if getattr(args, "apply", False):
            from harnesscad.core.loop import HarnessSession
            from harnesscad.io.backends.stub import StubBackend

            session = HarnessSession(StubBackend())
            result = session.apply_ops(ops)
            print("ok:      %s" % result.ok)
            print("applied: %d" % result.applied)
            print("digest:  %s" % session.digest())
            print("summary: %s" % json.dumps(session.summary(), sort_keys=True))
            return 0 if result.ok else 1
        payload = [op.to_dict() for op in ops]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["name"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-8s %-*s  %-8s %s" % (mark, r["route"], width, r["name"],
                                         r["family"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad procedural",
        description="named procedural generators that emit CISP ops")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
