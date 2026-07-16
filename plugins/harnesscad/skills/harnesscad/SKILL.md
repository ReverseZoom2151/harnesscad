---
name: harnesscad
description: Drive the HarnessCAD verifier-first text-to-CAD harness. Use when generating, verifying, measuring, exporting, rendering, or benchmarking parametric CAD parts from op streams or natural-language briefs with soundness guarantees.
---

# HarnessCAD

Verifier-first text-to-CAD harness: op streams are verified before the kernel runs them, geometry is cross-checked across independent backends, and results are measured, not trusted.

Plugin version 0.1.0; generated from the live CLI parser. The single entry point is the `harnesscad` executable (`python -m harnesscad.core.cli` from a source tree).

## Use this skill when

Use this skill when the user asks to build CAD geometry from a natural-language brief, apply or verify a CISP op stream, export or render a model, ingest an existing model into editable ops, or run the harness's benchmarks and reliability gates.

## Workflow

1. Express the design as a CISP op stream or a natural-language brief.
2. Run the matching verb below; every verb verifies before the kernel runs and exits nonzero when the result is not certified.
3. Read the emitted diagnostics; they are named, actionable errors, not stack traces.
4. Repair the smallest responsible op or parameter and rerun.

## Verbs

Available verbs: agent, apply, assembly, bench, build, capabilities, catalog, core, cua, dataset, demo, drawings, ecosystem, edit, export, fabricate, formats, gallery, generate, govern, grounding, ingest, judge, numeric, pdd, pressure, procedural, program, reconstruct, reliability, render, report, search, selftest, spec, ui, vision.

- `agent`: agent surface: envelopes, gates, approval-gated edits, tool metrics
- `apply`: run a JSON array of ops
- `assembly`: assembly checks over placed parts (AABB interference + fix vectors)
- `bench`: metric registry + suite runner (--list/--suites/--suite <name>)
- `build`: build a part from a natural-language brief via the LLM planner
- `capabilities`: discover/dispatch capability modules (--list/--search/--show/--stats)
- `catalog`: parts catalogue + standards knowledge base (--parts/--find/--part/--thread/--heatsert/--aci)
- `core`: core guards: op-decoding constraints, routing, feature tree, context
- `cua`: computer-use surface: the five live-GUI CAD environments and the deterministic action primitives they compose from
- `dataset`: data-engine pipeline (--list/--presets/--rivals, or run a named pipeline)
- `demo`: run the built-in constrained-plate sample
- `drawings`: drawing-understanding surface: Hough primitives, symbol points, primitive graphs, raster codec/metrics, render self-supervision
- `ecosystem`: which system, which backend, which bridge, which kernel
- `edit`: edit surface: apply a parametric edit, diff it, or run an edit loop
- `export`: run ops (or the demo) and write the model to <out>
- `fabricate`: manufacturing surface: workflows, feasibility, readiness, flat-pack, bricks, export planning
- `formats`: list the I/O capability matrix (read/write/round-trip)
- `gallery`: rendered parts gallery: 16 parts, each exercising a different capability (--list / --build [--out DIR] [--only NAME])
- `generate`: generation strategies driving a session (deterministic stub planner)
- `govern`: governance surface: security gates, research evidence, audit closure
- `grounding`: CAD-viewport grounding stack: catalogue, set-of-marks, corpora
- `ingest`: decode an existing model (CAD tokens or a mesh) into editable CISP ops
- `judge`: deterministic CAD graders: cad-score, betti, best-of-n, compiler-review
- `numeric`: numeric building blocks: diffusion, flow/ODE, noise schedules, multiscale, distillation, state-space
- `pdd`: Parts-Driven Development: brief -> MGC -> CISP -> artifact -> a single measured PASS/FAIL/UNCERTIFIED verdict
- `pressure`: pressure test: does a typed diagnostic beat a blind retry? (runs local ollama models through both loops and scores the geometry)
- `procedural`: named procedural generators that emit CISP ops (--list/--rivals/--gen)
- `program`: code-CAD program surface: parse/validate/emit/review, by --lang
- `reconstruct`: reconstruction route registry (input kind -> output kind)
- `reliability`: repair-loop surface: brep/code/compiler repair, fallback, infeasibility taxonomy, mcts search
- `render`: run ops (or the demo) and rasterise the model to a shaded-solid PNG
- `report`: quality analysis surface (--list/--rivals/--unadapted, or analyse a model)
- `search`: design-space search over a session (--list/--rivals/--strategy)
- `selftest`: SELF-evaluation: the harness evaluating the harness. Six engines differentially tested against each other, an analytic golden corpus, precision/recall PER VERIFIER, and metamorphic laws. Points INWARD: nothing here scores a model.
- `spec`: spec surface: a brief -> a checked spec -> constraints; EXPRESS/Part-21 validation; structured spec formats
- `ui`: interaction surface: command grammar, prediction, overlays, views
- `vision`: vision surface: trace an image back to CISP ops, calibrate px->mm

Each verb has a generated command file under `commands/` with full usage and argument documentation.
