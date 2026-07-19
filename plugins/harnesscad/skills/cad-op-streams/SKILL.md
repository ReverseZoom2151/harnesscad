---
name: cad-op-streams
description: Author or fix a CISP op stream -- the JSON array of ops that harnesscad's apply, export, render, pdd and build verbs all consume. Use when writing an ops file by hand, when the harness reports bad-value/unknown-op/malformed-selector on an op, when converting a shape description into ops, or when the user says "write the ops", "make an ops JSON", "why is this op rejected", or hands you a .json file of {"op": ...} objects.
---

# Authoring CISP op streams

Every geometry-producing verb in this harness takes the same input: a JSON
array of op objects. `apply` runs it, `export` and `render` build from it,
`pdd` measures the part it produces, and `build` generates one and then feeds
it back here. Get the op stream right and the rest of the harness is
mechanical; get it wrong and every downstream verb reports the same failure.

## The shape

A stream is a bare JSON array. Each element is an object with an `"op"` key
naming the op and one key per parameter:

```json
[
  {"op": "new_sketch", "plane": "XY"},
  {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0},
  {"op": "constrain", "kind": "distance", "a": "e1", "value": 20.0},
  {"op": "extrude", "sketch": "sk1", "distance": 5.0}
]
```

Omitted parameters take the op's default. That is a convenience, not a
license to omit: a default that happens to be geometrically valid will build
a part nobody asked for and the gate will pass it, because the gate measures
the part you built, not the part you meant. State every dimension the brief
states.

For the full op list with every field and default, read
`references/op-vocabulary.md` in this skill directory. Do that before writing
an op you have not written before -- an op name or field that does not exist
is rejected at parse time, and guessing costs a round trip.

## Naming and referencing

Ids are positional and implicit; you do not declare them.

- Sketches are `sk1`, `sk2`, ... in the order `new_sketch` appears.
- Features are `f1`, `f2`, ... in the order solid-producing ops appear
  (`extrude`, `revolve`, `loft`, `sweep`, `primitive`, `boolean`, ...).
- Sketch entities are `e1`, `e2`, ... in the order they are added.

So `{"op": "boolean", "kind": "cut", "target": "f1", "tool": "f2"}` cuts the
second solid out of the first. Count the ops above the boolean to work out
which number a feature has; an off-by-one here builds a real part with the
wrong topology and no error, which is the most expensive mistake in this file
format.

Where an op wants a body rather than a specific feature, the backends also
accept `""`, `"last"`, `"solid"` and `"body"` as "the current result". Prefer
those for finishing ops that apply to the whole part -- they cannot go stale
when you insert an op above them, which a numbered `f3` can.

## The usual sequence

Sketch-and-extrude is the spine, and most parts are that spine repeated:

1. `new_sketch` on a plane (`XY`, `XZ`, `YZ`).
2. One or more `add_*` ops naming that sketch.
3. `constrain` ops pinning the sketch's degrees of freedom.
4. `extrude` (or `revolve` / `loft` / `sweep`) to make a solid.
5. `boolean` to combine it with an earlier solid.
6. Finishing ops -- `fillet`, `chamfer`, `shell`, `draft`, `hole`.

For a bare block or cylinder, skip the sketch entirely and use
`{"op": "primitive", "shape": "box", "dx": ..., "dy": ..., "dz": ...}`. A
sketch-and-extrude that reproduces a primitive is four ops that can each be
wrong instead of one.

## Constraints

`constrain` takes `kind`, an entity `a`, an optional `b`, and an optional
`value`. Under-constrained sketches are not an error -- the harness reports
them as `[info] under-constrained: sketch sk1 is under-constrained (dof=4)`
and builds the geometry the ops specified. The reason it is only an info is
that the built part is well-defined; the reason to fix it anyway is that an
under-constrained sketch is not reproducible under edit, so the next
parameter change can move geometry you did not intend to move. Constrain when
the part is meant to be parametric; leave it when you are producing a
one-shot artifact and say which you did.

## Check it before you build

```bash
harnesscad apply <ops.json> --backend frep --verify core
```

`frep` is the right backend for a syntax pass because it meshes anything and
needs no OCCT. The output names the first op it refused:

```
ok:       False
applied:  3
rejected:  {"op": "fillet", "edges": ["e1"], "radius": 8.0}
  [error] bad-value: fillet edge selector is malformed: unexpected token 'e1'
```

`applied: 3` is the useful number -- ops 1 to 3 were accepted, so the fault is
in op 4 and nothing after it has been evaluated yet. Fix that one op and
rerun rather than rewriting the stream.

## Avoid

- Do not invent op names or fields. `references/op-vocabulary.md` is the whole
  vocabulary; anything else fails at parse.
- Do not use an edge selector where the op wants a list -- `fillet` and
  `chamfer` take `edges` as a list, and `"edges": []` means "all edges", which
  is a different request from "edge e1".
- Do not silently drop a dimension from the brief because no op field looks
  like it. Say which dimension the op vocabulary cannot express; an
  unexpressed requirement is the gate's blind spot, not its problem.
- Do not tune numbers until the measurement matches while the shape is wrong.
  If volume is off, find the op that is wrong; scaling a dimension to hit a
  target volume produces a part that passes and is not the part.
