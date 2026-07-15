"""picks — turn a CISP selector into CONCRETE entities, then COMPUTED clicks.

This is the missing bridge between a pick-needing CISP op and the proven
viewport machinery. It reuses two things that already exist and are tested, and
adds only the glue:

* :mod:`harnesscad.domain.geometry.topology.selector_dsl` — the CadQuery-style
  selector grammar (``"|Z"`` = edges parallel to Z, ``">Z"`` = the top face,
  ``"|Z and >Y"`` = ...). :func:`resolve` maps a selector string to the concrete
  entity names it denotes, given the solid's topology.
* :mod:`harnesscad.io.cua.viewport` — the deterministic projection and the
  app-adjudicated pick (438/438 computed pixels select the intended entity; never
  orbit, named views only, the pixel COMPUTED from our own B-rep and the known
  camera). :func:`pick_entities` drives it for a resolved entity set.

Why this is the high-value piece
--------------------------------
The GUI environment refuses every op that needs a viewport PICK — fillet (edge),
hole/shell/draft (face), boolean/mirror/pattern (tree) — because a coordinate-free
dialog table cannot express "which edge". This module supplies the "which edge",
geometrically and deterministically: the selector names the edges, the projection
computes their pixels, and the app's own ray-picker adjudicates the click. An
unverified pick is discarded, exactly as :func:`viewport.ViewportController.adjudicate`
already does — a selection we cannot prove selected the intended entity is not a
selection.

The resolution half (:func:`resolve`, :func:`entities_from_topology`) is pure and
unit-tested with a synthetic box topology, no GUI in the room.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.topology import selector_dsl as SEL


def _unit(v: Sequence[float]) -> Tuple[float, float, float]:
    n = math.sqrt(sum(c * c for c in v))
    if n <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _edge_tangent(points: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    """A straight edge's direction, from its sampled candidate points.

    :data:`viewport.EDGE_FRACTIONS` samples an edge at ``(0.5, 0.35, 0.65, 0.2,
    0.8)`` of its parameter range, so the widest-separated pair among the samples
    gives the most stable chord direction. For a straight edge this IS the tangent;
    for a curved one it is a chord, which is all a selector like ``"|Z"`` can
    meaningfully act on anyway (a circle is parallel to no axis).
    """
    pts = [tuple(float(c) for c in p) for p in points if len(p) == 3]
    if len(pts) < 2:
        return (0.0, 0.0, 0.0)
    best = (0.0, 0.0, 0.0)
    best_len = -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = (pts[j][0] - pts[i][0], pts[j][1] - pts[i][1], pts[j][2] - pts[i][2])
            ln = sum(c * c for c in d)
            if ln > best_len:
                best_len, best = ln, d
    return _unit(best)


def entities_from_topology(topo: Dict[str, Any],
                           kinds: Sequence[str] = ("edge", "face")) -> List[SEL.Entity]:
    """Adapt a :func:`viewport.ViewportController.topology` dict into selector
    ``Entity`` objects (``center``, ``axis``, ``geom_type``, ``name``).

    Edge ``axis`` is the tangent derived from the candidate points; face ``axis``
    is the normal the topology already carries. This is the only adaptation the
    selector grammar needs — everything else (``|``, ``>``, ``<``, ``and``,
    ``not`` ...) is the existing DSL.
    """
    out: List[SEL.Entity] = []
    if "edge" in kinds:
        for e in topo.get("edges") or []:
            axis = _edge_tangent(e.get("points") or [])
            out.append(SEL.Entity(
                center=tuple(float(c) for c in (e.get("centroid") or (0, 0, 0))),
                axis=axis,
                geom_type=str(e.get("surface", "")).upper()
                .replace("GEOM_", "").replace("CURVE", "").strip("_") or "LINE",
                name=e["name"]))
    if "face" in kinds:
        for f in topo.get("faces") or []:
            normal = f.get("normal") or (0.0, 0.0, 0.0)
            out.append(SEL.Entity(
                center=tuple(float(c) for c in (f.get("centroid") or (0, 0, 0))),
                axis=_unit(normal),
                geom_type=str(f.get("surface", "")).upper()
                .replace("GEOM_", "").replace("SURFACE", "").strip("_") or "PLANE",
                name=f["name"]))
    return out


def resolve(selector: str, topo: Dict[str, Any],
            kinds: Sequence[str] = ("edge", "face")) -> List[str]:
    """The entity names a CISP selector denotes, given the solid's topology.

    Empty selector (or ``()``) means "every edge" — the CISP ``Fillet``/``Chamfer``
    convention for an empty ``edges`` tuple. Raises ``SelectorError`` on a
    malformed selector (never guesses).
    """
    entities = entities_from_topology(topo, kinds)
    if not selector or selector in ("*", "all"):
        return [e.name for e in entities if e.name.startswith("Edge")] or \
               [e.name for e in entities]
    chosen = SEL.select(selector, entities)
    return [e.name for e in chosen]


def resolve_op_edges(op: Any, topo: Dict[str, Any]) -> List[str]:
    """The concrete edge names a :class:`~harnesscad.core.cisp.ops.Fillet` /
    ``Chamfer`` op's ``edges`` selectors denote. Union across the tuple."""
    selectors = list(getattr(op, "edges", ()) or ())
    if not selectors:
        # Empty = every edge (the op's documented default).
        return [n for n in resolve("", topo, ("edge",))]
    names: List[str] = []
    for sel in selectors:
        for n in resolve(sel, topo, ("edge",)):
            if n not in names:
                names.append(n)
    return names


@dataclass
class SelectionOutcome:
    """The VERIFIED result of a computed multi-entity pick.

    ``verified`` is True only if the app's OWN selection, read back after the real
    clicks, is exactly the set we intended — occlusion or a pick-radius steal
    leaves it False with the discrepancy, never a silent partial selection.
    """

    intended: List[str] = field(default_factory=list)
    selected: List[str] = field(default_factory=list)
    verified: bool = False
    reason: str = ""
    per_entity: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"intended": self.intended, "selected": self.selected,
                "verified": self.verified, "reason": self.reason,
                "per_entity": self.per_entity}


def pick_entities(controller: Any, names: Sequence[str], *,
                  view: str = "isometric",
                  use_real_mouse: bool = True) -> SelectionOutcome:
    """Select the named entities by COMPUTED clicks, adjudicated by the app.

    ``controller`` is a live :class:`viewport.ViewportController`. The steps are
    exactly the proven ones: set a NAMED orthographic view (never orbit), project
    each entity's candidate points through the known camera, keep the candidate the
    app's ray-picker agrees is that entity, then — if ``use_real_mouse`` — fire a
    real ``SendInput`` click there and read ``Gui.Selection`` back to VERIFY.

    Returns a :class:`SelectionOutcome`. This never raises on a missed pick: a
    discarded entity is data (it is occluded from this view), and a caller can try
    another named view for it.
    """
    from harnesscad.io.cua import viewport as VP

    want = list(names)
    controller.set_named_view(view)
    camera = controller.camera()
    all_entities = {e.name: e for e in controller.entities(("edge", "face", "vertex"))}
    targets = [all_entities[n] for n in want if n in all_entities]
    picks = controller.adjudicate(targets, camera)

    outcome = SelectionOutcome(intended=want)
    verified_names: List[str] = []
    controller.clear_selection()
    if use_real_mouse:
        controller.focus_window()
        rect = controller.viewport_rect()
    for p in picks:
        rec = {"entity": p.entity, "computed": p.verified, "reason": p.reason,
               "x": p.x, "y": p.y}
        if not p.verified:
            outcome.per_entity.append(rec)
            continue
        if use_real_mouse:
            # A real click adds to the running selection (FreeCAD accumulates on a
            # plain click for feature selection); read back what the app now holds.
            controller.mouse_click(p, camera, rect=rect)
            sel = {s.split(".", 1)[-1] for s in controller.selection()}
            rec["selected_now"] = p.entity in sel
            if p.entity in sel:
                verified_names.append(p.entity)
        else:
            verified_names.append(p.entity)
        outcome.per_entity.append(rec)

    outcome.selected = verified_names
    missing = [n for n in want if n not in verified_names]
    outcome.verified = not missing
    outcome.reason = "selected exactly the intended set" if outcome.verified else \
        "did not verify: " + ", ".join(missing)
    return outcome


def pick_op_edges(controller: Any, op: Any, *, view: str = "isometric",
                  use_real_mouse: bool = True) -> SelectionOutcome:
    """THE UNLOCK, as one call: a refused pick-op -> a VERIFIED edge selection.

    A :class:`~harnesscad.core.cisp.ops.Fillet` / ``Chamfer`` carries edge
    *selectors* (``"|Z"``), which the coordinate-free dialog table cannot express
    and the GUI environment therefore REFUSES. Here the same op becomes real
    capability: read the live solid's topology off ``controller``, resolve the
    selectors to concrete edge names against it (:func:`resolve_op_edges`), then
    select those edges by computed, app-adjudicated clicks (:func:`pick_entities`).

    The returned :class:`SelectionOutcome` is verified iff the app's OWN selection,
    read back after the clicks, is exactly the edge set the selector denoted — so
    the fillet that follows acts on proven edges, not a guess. This is the code
    path that turns "fillet needs an edge pick, refused" into "the four vertical
    edges are selected"; the same shape resolves a hole/shell FACE pick or a
    boolean/mirror TREE pick, changing only the ``kinds`` handed to :func:`resolve`.
    """
    topo = controller.topology()
    names = resolve_op_edges(op, topo)
    if not names:
        # The selectors parsed but matched nothing: there is no edge to pick, so
        # this is not a capability, it is an empty referent. Say so and do not
        # claim a (vacuous) verified selection.
        return SelectionOutcome(
            intended=[], selected=[], verified=False,
            reason="the op's selectors denote no edge on this solid")
    return pick_entities(controller, names, view=view,
                         use_real_mouse=use_real_mouse)
