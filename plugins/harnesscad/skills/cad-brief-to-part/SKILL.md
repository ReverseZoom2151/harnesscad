---
name: cad-brief-to-part
description: Turn a natural-language part description into built, verified CAD geometry with harnesscad build, then export or render it. Use when the user describes a physical part in prose ("model a 40 mm bracket with two M5 holes", "make me a plate", "design an enclosure lid"), asks for a STEP/STL file or a picture of a part, or wants an end-to-end run from brief to artifact.
---

# From a brief to a verified part

`build` is the end-to-end verb: it plans an op stream from a brief with a
language model, applies it, reads the gate's diagnostics, and iterates. It is
the only verb here that generates ops; everything else consumes them.

```bash
harnesscad build "<brief>" --backend cadquery --out part.step --trace run.jsonl
```

`build` needs a model client (`--model` names it). If no model is available,
the path is to write the op stream yourself with the `cad-op-streams` skill
and run `apply`/`export` directly -- the rest of the harness needs no model
at all.

## Write the brief with the measurable in it

The gate measures the artifact. A brief that states no quantity gives it
nothing to measure, and a part that satisfies no stated quantity cannot be
certified. Before running, make sure the brief carries the dimensions,
counts and positions the user actually specified.

Where the user did not specify something that matters, ask. Do not pick a
plausible millimetre and proceed -- an invented dimension is indistinguishable
from a specified one in the output, and this harness is built on the premise
that the difference matters. `[NEEDS CLARIFICATION]` is a first-class result
here, not a failure to be helpful.

## Choosing a strategy

`--strategy` decides how the loop uses failure, and the two options differ in
kind rather than degree:

- `refine` (default) feeds the soundness-gated diagnostics back into the next
  plan and replans. Up to `--max-iters` (default 5) rounds. It converges fast
  when the diagnostics are specific.
- `best-of-n` draws `--best-of` independent plans (default 4), applies each in
  a fresh session, and lets the deterministic verifier pick the winner. The
  flag's own help states why that matters: there is no feedback channel, and
  therefore no poisoning surface.

Prefer `best-of-n` when the result will be trusted without a human reading
every diagnostic, because its selection is made by the verifier rather than
by a model reasoning about the verifier. Prefer `refine` when you are in the
loop and the failures are mechanical.

## Choosing a backend

`--backend` accepts `stub`, `cadquery`, `build123d`, `frep`, `blender`,
`openscad`, `freecad`, `manifold`, `rhino3dm`, `microcad`, `truck`.
`build` defaults to `cadquery`.

- `cadquery` / `build123d` / `freecad` / `truck` -- real B-rep, exports STEP.
- `frep` -- meshes anything, needs no OCCT. The safe default for a quick
  structural check and the default for `render`.
- `stub` -- no geometry kernel. Useful for exercising the plumbing; a pass on
  `stub` is not a claim about geometry.

Two independent backends agreeing is a much stronger result than one backend
succeeding, because it is what lets the differential oracle run at all. If
only one is installed, say so when reporting.

## Getting artifacts out

Once ops exist, artifacts do not need `build` again:

```bash
harnesscad export part.step --ops ops.json --backend cadquery
harnesscad render part.png --ops ops.json --view iso --width 1200 --height 900
```

`export` picks the format from the output extension. `render` defaults to the
`frep` backend and the `iso` view; other views are `front`, `back`, `left`,
`side`, `top`, `bottom`, `hero`. `--ssaa` (1..4, default 2) is the quality
knob and `--projection` takes `orthographic` (default) or `perspective`.

Both refuse to write when the output gate rejects the part. That refusal is
the feature.

## After it builds

`build` exiting 0 means the gate did not refuse the artifact. It does not
mean the part matches the brief -- volume, bounding box and genus do not pin
down a shape. Render it and look at it, and where the brief carried hard
numbers, check them with `pdd` against a measured contract (see the `cad-pdd`
skill).

## Avoid

- Do not pass `--force` to get an artifact out of a refused build. It writes
  a `<name>.INVALID.json` sidecar naming every failed measurement, and an
  artifact with that sidecar beside it is not a delivered part.
- Do not raise `--max-iters` or `--best-of` to escape a brief that is
  underspecified. More samples will not invent the missing dimension; they
  will produce more confident wrong parts.
- Do not report a `stub` or `frep` result as though it were a kernel result.
  Name the backend in the report.
- Do not describe a built part as matching the brief on the strength of the
  exit code alone. Say what was measured.
