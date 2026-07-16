"""OpenCAD's fluent chained authoring API, lowered onto the CISP typed op stream.

Source: OpenCAD-Examples (resources/cad_repos/OpenCAD-Examples-main). Every
example script in that repo authors a part through a tiny fluent surface --
``Sketch(name=...).rect(w, h).circle(r, center=(x, y), subtract=True)`` chained
into ``Part(name=...).extrude(sketch, depth, name=...).fillet(edges="top",
radius=..., name=...)`` -- and the runtime lowers that chain into a serialized
feature tree. What is ported here is exactly that authoring surface: ``Sketch``
with ``rect`` / ``circle`` (each returning ``self`` and recording a typed
entry) and ``Part`` with ``extrude`` / ``cylinder`` / ``cut`` / ``fillet`` /
``offset``, every method accepting ``name=`` so features stay named the way
OpenCAD names them.

Gap it fills: the harness speaks CISP ops (:mod:`harnesscad.core.cisp.ops`) --
a flat, typed, JSON-round-trippable op stream -- but had no chained,
human-writable authoring front end. Model-generated or example-derived code in
the OpenCAD style could not be executed against the harness. This module is
that front end: each fluent call LOWERS to the existing typed ops (``NewSketch``,
``AddRectangle``, ``AddCircle``, ``Extrude``, ``Hole``, ``Primitive``,
``Boolean``, ``Fillet``), inventing no new vocabulary.

What it complements:
  * :mod:`harnesscad.core.cisp.ops` -- the builder EMITS those frozen
    dataclasses verbatim (``Part.ops()`` / ``Part.ops_dicts()``); nothing here
    re-declares an op.
  * :mod:`harnesscad.core.state.feature_tree` -- the harness already ports
    OpenCAD's feature-tree rebuild/invalidation service; this builder does NOT
    rebuild it. Instead ``Part.features()`` returns the ordered
    ``(feature_name, op_indices)`` list a caller can hydrate feature-tree nodes
    from, so a fluent stream can feed the existing parametric DAG.

Lowering rules (deterministic, documented):
  * ``Sketch.rect(w, h)`` records a rectangle with its corner at the origin --
    the OpenCAD examples place hole centres in (0..w, 0..h), i.e. corner-origin
    coordinates -- lowering to ``AddRectangle(sketch, 0, 0, w, h)``.
  * ``Sketch.circle(r, center, subtract=False)`` lowers to ``AddCircle``; with
    ``subtract=True`` it lowers to a ``Hole(face_or_sketch=<sketch id>, x, y,
    diameter=2*r, through=True)`` emitted AFTER the extrude, matching the
    harness convention (see eval/corpus/analytic.py plate_with_holes).
  * ``Part.extrude(sketch, depth)`` emits ``NewSketch`` + the additive entity
    ops + ``Extrude(sketch, distance=depth)``, then the subtract-circle Holes.
    Sketch ids are ``sk1``, ``sk2``, ... and body/feature ids ``f1``, ``f2``,
    ... in emission order -- the same conventions the analytic corpus uses.
  * ``Part.cylinder(r, h)`` lowers to ``Primitive(shape="cylinder", r=r, h=h)``
    (the closest Primitive signature; axis along +Z per the op's contract).
  * ``Part.cut(other)`` appends the other part's ops into this part's stream
    (renumbering that part's sketch/body ids so both bodies live in one
    stream), then emits ``Boolean(kind="cut", target=<this body>, tool=<other
    body>)`` -- ``kind="cut"`` because that is the exact Boolean field value
    ops.py documents ("union | cut | intersect"); OpenCAD's word for it is
    "difference".
  * ``Part.fillet(edges=..., radius=...)`` maps the friendly edge word onto the
    CadQuery selector grammar the ``Fillet`` op documents: ``"top"`` ->
    ``(">Z",)``, ``"bottom"`` -> ``("<Z",)``, ``"vertical"`` -> ``("|Z",)``,
    anything else (including ``"all"`` / ``""``) -> ``()`` = every edge.
  * ``Part.offset(distance)``: CISP has NO 2D face-offset / reinforcement op
    (``Thicken`` is a solid wall offset with different semantics), so rather
    than inventing an op or silently mislowering, ``offset`` records a NAMED
    NO-OP FEATURE NOTE -- a features() entry with an empty op-index list and
    the distance recorded in ``Part.notes`` -- and emits nothing. This is
    stated here so callers know the OpenCAD ``offset`` chain link survives as
    feature metadata only.

Stdlib only, deterministic, no clock, no randomness.

Selfcheck: rebuilds OpenCAD's mounting-bracket example (80x30 rect, four
3 mm-radius corner holes, one 5 mm-radius centre hole, extrude 4, fillet top
0.75) and asserts the op count and a JSON round-trip through
``ops.parse_op``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple

from harnesscad.core.cisp.ops import (
    AddCircle,
    AddRectangle,
    Boolean,
    Extrude,
    Fillet,
    Hole,
    NewSketch,
    Op,
    Primitive,
    parse_op,
)

__all__ = ["Sketch", "Part", "EDGE_SELECTORS", "main"]

# Friendly edge word -> CadQuery selector tuple, exactly as the Fillet op's
# docstring documents the grammar. An empty tuple means "every edge".
EDGE_SELECTORS: Dict[str, Tuple[str, ...]] = {
    "top": (">Z",),
    "bottom": ("<Z",),
    "vertical": ("|Z",),
}


class Sketch:
    """A recorded 2D profile in the OpenCAD fluent style.

    Methods return ``self`` so calls chain; nothing is lowered until a
    :class:`Part` consumes the sketch (``Part.extrude``). Entries are recorded
    as plain tuples so a sketch is inspectable and replayable.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        # ("rect", w, h, name) | ("circle", r, cx, cy, subtract, name)
        self.entries: List[tuple] = []

    def rect(self, width: float, height: float, name: str = "") -> "Sketch":
        """Record a rectangle with its corner at the origin (OpenCAD placement)."""
        if width <= 0 or height <= 0:
            raise ValueError("rect needs positive width and height")
        self.entries.append(("rect", float(width), float(height), name))
        return self

    def circle(
        self,
        radius: float,
        center: Tuple[float, float] = (0.0, 0.0),
        subtract: bool = False,
        name: str = "",
    ) -> "Sketch":
        """Record a circle; ``subtract=True`` marks it as a through cutout."""
        if radius <= 0:
            raise ValueError("circle needs a positive radius")
        cx, cy = float(center[0]), float(center[1])
        self.entries.append(("circle", float(radius), cx, cy, bool(subtract), name))
        return self


class Part:
    """A recorded part: fluent feature calls lowering to a CISP op stream."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._ops: List[Op] = []
        # Ordered (feature_name, op_indices) for feature-tree hydration.
        self._features: List[Tuple[str, Tuple[int, ...]]] = []
        # Feature notes with no CISP lowering (see .offset).
        self.notes: Dict[str, dict] = {}
        self._sketch_counter = 0
        self._body_counter = 0
        self._last_body: str = ""

    # -- id allocation -------------------------------------------------------
    def _next_sketch_id(self) -> str:
        self._sketch_counter += 1
        return "sk%d" % self._sketch_counter

    def _next_body_id(self) -> str:
        self._body_counter += 1
        return "f%d" % self._body_counter

    def _record_feature(self, name: str, indices: Tuple[int, ...]) -> None:
        self._features.append((name, indices))

    # -- fluent features -----------------------------------------------------
    def extrude(self, sketch: Sketch, depth: float, name: str = "") -> "Part":
        """Lower a sketch to NewSketch + entity ops + Extrude, then Hole ops.

        Additive entries (rects, plain circles) become the profile; every
        ``subtract=True`` circle becomes a through ``Hole`` AFTER the extrude,
        with ``diameter = 2 * radius``.
        """
        if depth <= 0:
            raise ValueError("extrude needs a positive depth")
        sk = self._next_sketch_id()
        body = self._next_body_id()
        start = len(self._ops)
        self._ops.append(NewSketch(plane="XY"))
        holes: List[Tuple[float, float, float, str]] = []  # (cx, cy, r, name)
        for entry in sketch.entries:
            if entry[0] == "rect":
                _, w, h, _ename = entry
                self._ops.append(AddRectangle(sketch=sk, x=0.0, y=0.0, w=w, h=h))
            elif entry[0] == "circle":
                _, r, cx, cy, subtract, ename = entry
                if subtract:
                    holes.append((cx, cy, r, ename))
                else:
                    self._ops.append(AddCircle(sketch=sk, cx=cx, cy=cy, r=r))
            else:  # pragma: no cover - Sketch only records the two kinds
                raise ValueError("unknown sketch entry kind %r" % (entry[0],))
        self._ops.append(Extrude(sketch=sk, distance=float(depth)))
        feature_name = name or sketch.name or ("%s extrude" % (self.name or body))
        self._record_feature(feature_name, tuple(range(start, len(self._ops))))
        for k, (cx, cy, r, ename) in enumerate(holes, start=1):
            idx = len(self._ops)
            self._ops.append(
                Hole(
                    face_or_sketch=sk,
                    x=cx,
                    y=cy,
                    diameter=2.0 * r,
                    depth=None,
                    through=True,
                    kind="simple",
                )
            )
            hole_name = ename or ("%s hole %d" % (feature_name, k))
            self._record_feature(hole_name, (idx,))
        self._last_body = body
        return self

    def cylinder(self, radius: float, height: float, name: str = "") -> "Part":
        """Lower to ``Primitive(shape="cylinder", r=radius, h=height)``."""
        if radius <= 0 or height <= 0:
            raise ValueError("cylinder needs positive radius and height")
        body = self._next_body_id()
        idx = len(self._ops)
        self._ops.append(
            Primitive(shape="cylinder", r=float(radius), h=float(height))
        )
        self._record_feature(name or ("%s cylinder" % (self.name or body)), (idx,))
        self._last_body = body
        return self

    def cut(self, other_part: "Part", name: str = "") -> "Part":
        """Boolean difference: this part minus ``other_part``.

        The other part's ops are appended into this stream (its sketch and
        body ids renumbered into this part's sequence so both bodies exist in
        one op log), then ``Boolean(kind="cut", target=<this body>, tool=
        <other body>)`` is emitted -- the exact field values ops.py documents.
        """
        if not other_part._ops:
            raise ValueError("cut needs a non-empty tool part")
        target = self._last_body
        if not target:
            raise ValueError("cut needs this part to have a body first")
        # Renumber the tool part's sketch ids into this stream.
        sketch_map: Dict[str, str] = {}
        start = len(self._ops)
        for op in other_part._ops:
            d = op.to_dict()
            for field_name in ("sketch", "face_or_sketch"):
                old = d.get(field_name)
                if old:
                    if old not in sketch_map:
                        sketch_map[old] = self._next_sketch_id()
                    d[field_name] = sketch_map[old]
            self._ops.append(parse_op(d))
        # The tool part's features carry over, indices shifted.
        offset = start
        for fname, indices in other_part._features:
            self._record_feature(fname, tuple(i + offset for i in indices))
        tool = self._next_body_id()
        idx = len(self._ops)
        self._ops.append(Boolean(kind="cut", target=target, tool=tool))
        self._record_feature(name or ("%s cut" % (self.name or target)), (idx,))
        return self

    def fillet(self, edges: str = "", radius: float = 1.0, name: str = "") -> "Part":
        """Round edges; the friendly word maps to the Fillet selector grammar."""
        if radius <= 0:
            raise ValueError("fillet needs a positive radius")
        selector = EDGE_SELECTORS.get(str(edges).strip().lower(), ())
        idx = len(self._ops)
        self._ops.append(Fillet(edges=selector, radius=float(radius)))
        self._record_feature(name or ("%s fillet" % (self.name or "part")), (idx,))
        return self

    def offset(self, distance: float, name: str = "") -> "Part":
        """OpenCAD's profile offset / reinforcement: recorded, NOT lowered.

        No CISP op expresses a 2D face-offset reinforcement (``Thicken`` is a
        solid wall offset with different semantics), so this records a named
        no-op feature note -- an empty op-index feature plus a ``notes`` entry
        carrying the distance -- rather than inventing an op. See the module
        docstring.
        """
        note_name = name or ("%s offset" % (self.name or "part"))
        self._record_feature(note_name, ())
        self.notes[note_name] = {"kind": "offset", "distance": float(distance)}
        return self

    # -- exports ---------------------------------------------------------------
    def ops(self) -> List[Op]:
        """The lowered op stream, in emission order."""
        return list(self._ops)

    def ops_dicts(self) -> List[dict]:
        """The op stream as JSON-ready dicts (``Op.to_dict`` verbatim)."""
        return [op.to_dict() for op in self._ops]

    def features(self) -> List[Tuple[str, Tuple[int, ...]]]:
        """Ordered (feature_name, op_indices) pairs.

        Each pair names a feature and the op-stream indices it owns, so a
        caller can hydrate :mod:`harnesscad.core.state.feature_tree` nodes
        (one node per feature, parented in list order) without this module
        re-implementing that service.
        """
        return list(self._features)


# ----------------------------------------------------------------------------- #
# selfcheck
# ----------------------------------------------------------------------------- #
def _selfcheck() -> int:
    # OpenCAD's hardware_mounting_bracket.py, verbatim dimensions.
    profile = (
        Sketch(name="Bracket Profile")
        .rect(80, 30)
        .circle(3, center=(8, 8), subtract=True)
        .circle(3, center=(72, 8), subtract=True)
        .circle(3, center=(8, 22), subtract=True)
        .circle(3, center=(72, 22), subtract=True)
        .circle(5, center=(40, 15), subtract=True)
    )
    part = Part(name="Mounting Bracket").extrude(
        profile, depth=4, name="Bracket Body"
    ).fillet(edges="top", radius=0.75, name="Bracket Edge Relief")

    ops = part.ops()
    print("op stream (%d ops):" % len(ops))
    for i, op in enumerate(ops):
        print("  [%d] %s" % (i, json.dumps(op.to_dict(), sort_keys=True)))
    print("features:")
    for fname, indices in part.features():
        print("  %-28s -> ops %s" % (fname, list(indices)))

    # NewSketch + AddRectangle + Extrude + 5 Holes + Fillet = 9 ops.
    assert len(ops) == 9, "expected 9 ops, got %d" % len(ops)
    assert isinstance(ops[0], NewSketch)
    assert isinstance(ops[1], AddRectangle) and ops[1].w == 80 and ops[1].h == 30
    assert isinstance(ops[2], Extrude) and ops[2].distance == 4
    hole_ops = [op for op in ops if isinstance(op, Hole)]
    assert len(hole_ops) == 5
    assert sorted(h.diameter for h in hole_ops) == [6.0, 6.0, 6.0, 6.0, 10.0]
    assert all(h.through for h in hole_ops)
    assert isinstance(ops[-1], Fillet)
    assert ops[-1].edges == (">Z",) and ops[-1].radius == 0.75

    # JSON round-trip: to_dict -> json -> parse_op reproduces every op.
    payload = json.dumps(part.ops_dicts(), sort_keys=True)
    rebuilt = [parse_op(d) for d in json.loads(payload)]
    assert rebuilt == ops, "JSON round-trip did not reproduce the op stream"

    # Feature list: 1 extrude + 5 holes + 1 fillet = 7 named features.
    feats = part.features()
    assert len(feats) == 7, "expected 7 features, got %d" % len(feats)
    assert feats[0][0] == "Bracket Body"
    assert feats[-1][0] == "Bracket Edge Relief"

    # Cylinder + cut (the grommet path) also lowers cleanly.
    outer = Part(name="Outer Grommet").cylinder(14, 10, name="Outer Cylinder")
    inner = Part(name="Inner Clearance").cylinder(8, 10, name="Inner Cylinder")
    outer.cut(inner, name="Cable Passage")
    g_ops = outer.ops()
    assert len(g_ops) == 3
    assert isinstance(g_ops[0], Primitive) and g_ops[0].r == 14 and g_ops[0].h == 10
    assert isinstance(g_ops[1], Primitive) and g_ops[1].r == 8
    assert isinstance(g_ops[2], Boolean) and g_ops[2].kind == "cut"
    assert g_ops[2].target == "f1" and g_ops[2].tool == "f2"

    # Offset is a recorded no-op note, not an invented op.
    carrier = Part(name="PCB Carrier").extrude(
        Sketch("p").rect(90, 60), depth=3, name="Carrier Plate"
    ).offset(0.4, name="Carrier Reinforcement")
    assert len(carrier.ops()) == 3  # offset emitted nothing
    assert carrier.notes["Carrier Reinforcement"]["distance"] == 0.4
    assert ("Carrier Reinforcement", ()) in carrier.features()

    print("SELFCHECK OK: 9-op bracket, JSON round-trip, grommet cut, offset note")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="OpenCAD fluent builder lowered onto CISP ops."
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="rebuild the OpenCAD mounting-bracket example, print the op "
        "stream and features, and assert the lowering invariants.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0
    try:
        return _selfcheck()
    except AssertionError as exc:
        print("SELFCHECK FAILED: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
