<h1 align="center">
  HarnessCAD
</h1>

<p align="center"><strong>Verifier-First Agentic Text-to-CAD Harness</strong></p>

An in-progress, test-driven research implementation of a CAD harness that treats
geometry as a *typed, verifiable op stream* rather than as generated code. Every
op is checked before it reaches the kernel, and the loop blocks and corrects
rather than retrying blindly. The harness, not the model, is the product.

HarnessCAD is building a deterministic CAD substrate around a typed op protocol,
a block-and-correct apply loop, and a domain layer mined from a complete reading
of 186 text-to-CAD papers and 70 CAD repositories. It is not an end-to-end
product yet: 76% of its modules are not yet called by the harness, the default
geometry backend is a stub, and this repository contains no trained weights,
published benchmark results, or claimed accuracy numbers.

## What is implemented

- CISP: a typed op protocol (`new_sketch`, `add_rectangle`, `constrain`, `extrude`, `boolean`, ...) with a registry, canonical JSON, and a content digest per model state.
- HarnessSession: a block-and-correct apply loop where ops are validated, rejected with typed diagnostics, and never silently dropped; under-constrained sketches report their remaining DOF.
- AgentHarness: a ReAct loop over the op set, with an event-sourced op-DAG so any state is reconstructable and any edit replayable.
- Kernel-free geometry: SDF authoring (primitives, smooth combinators, domain repetition, TPMS, sphere tracing), an f-rep opcode graph with interval arithmetic and forward-mode autodiff, dual contouring with QEF, marching cubes, surface nets.
- Mesh substrate: half-edge topology, adaptive exact-sign predicates (`orient3d`/`insphere` with a rational fallback), 3D BVH, triangle-triangle intersection, winding-number inside tests.
- Formats: STL, GLB, AMF, SVG, DXF, OBJ, STEP (ISO 10303-21), plus an EXPRESS (ISO 10303-11) schema-language parser validated against 662 of 664 real ISO schemas and an inheritance-aware Part-21 validator.
- CAD program analysis: ASTs, validators and emitters for CadQuery, OpenSCAD, and a typed CSG language whose 2D/3D dimension checker catches `circle(3) + cube(2)` with zero geometry.
- Robot description: URDF forward kinematics with mimic-chain resolution, a strict URDF parser, and SRDF semantics cross-validated against the URDF.
- Evaluation: ~200 benchmark modules in which rival metrics are kept deliberately distinct. `chamfer_unit_sphere` and `chamfer_bbox_judged` give different numbers on the same meshes, and the filenames say so.
- A capability registry: a static AST index over all 1,161 modules (nothing is imported to be indexed, so kernel-dependent modules index safely) with tag/text search and lazy loading.
- Protocol surfaces: MCP (other agents consume our tools), ACP (an editor drives the harness), A2A (a peer delegates a task to us).
- 14,431 tests. Stdlib-only, deterministic: no wall clock, seeded randomness.

## What still requires execution

**886 of the 1,161 modules (76%) are imported by nothing but their own test.**
They are correct and tested, but the harness does not yet call them. Two mining
campaigns produced a library of verified implementations, not a fully connected
system. The capability registry makes them discoverable and loadable today
(`harnesscad capabilities --tag sdf`), but wiring the verifiers into the verify
loop and the geometry and format modules into the backends is unfinished work,
not a solved problem. Treat any claim of end-to-end coverage with suspicion until
that orphan count drops.

The default geometry backend is a stub; the OCCT backend and the LLM planner are
optional extras and are not exercised by the test suite. Two open correctness
decisions are recorded rather than silently resolved: `programs/ast/cadquery.py`
is scoped to one paper's CadQuery subset and rejects ~100 real fluent methods,
and the MUSE scorecard ANDs `watertight` and `manifold` as independent checks
when the source defines the first to already imply the second. See
[`docs/corpus/repo-ideas.md`](docs/corpus/repo-ideas.md).

## Install

```bash
git clone <repo> && cd harnesscad
pip install -e .                 # stdlib-only core, no required dependencies
pip install -e ".[cadquery]"     # OCCT geometry backend
pip install -e ".[llm]"          # LLM planner (LiteLLM + Instructor)
pip install -e ".[constraints]"  # SolveSpace sketch solver
```

Python >= 3.10. **The core spine has no runtime dependencies.** Provider keys are
read from the environment; the repository never stores them.

## Core workflow

```text
brief ──▶ planner ──▶ CISP ops ──▶ HarnessSession ──▶ backend ──▶ solid
             ▲                          │                           │
             └───── diagnostics ◀───────┘                           │
                    (block & correct)                               │
                                               verifiers ◀──────────┘
```

```bash
harnesscad demo                                      # constrained-plate sample
harnesscad apply examples/ops_plate.json
harnesscad build "a 20x10x5 plate with a 3mm hole"   # requires [llm]
harnesscad capabilities --stats
```

```python
from harnesscad.core.loop import HarnessSession
from harnesscad.core.cisp.ops import parse_op
from harnesscad.io.backends.stub import StubBackend

session = HarnessSession(StubBackend())
result = session.apply_ops([parse_op(o) for o in ops])
result.ok, result.digest, result.diagnostics
```

The source is laid out as `core/` (op spine, loop, pipeline, CLI), `domain/`
(geometry, numerics, reconstruction, CAD programs), `io/` (formats, ingestion,
backends, surfaces), `eval/` (benchmarks, quality, verifiers), `agents/`,
`data/`, and `governance/`, under `src/harnesscad/`. `tests/` mirrors it exactly.
Modules are named for what they do, not where they were mined from, except where
provenance *is* the meaning: `reconstruction/tokens/` holds `deepcad_quantize`,
`skexgen_quantize`, `hnc_rotation_codebook` and `vitruvion_primitives` side by
side because they are mutually incompatible quantisers, and that disagreement is
the finding.

## Documentation

- [`docs/blueprint.md`](docs/blueprint.md): architecture and design rationale
- [`docs/corpus/paper-ideas.md`](docs/corpus/paper-ideas.md): all 186 papers, what was built from each, and what was not
- [`docs/corpus/repo-ideas.md`](docs/corpus/repo-ideas.md): all 70 repositories, likewise
- [`docs/corpus/audit.md`](docs/corpus/audit.md): corpus audit
- [`docs/corpus/coverage.md`](docs/corpus/coverage.md): coverage summary
- [`audit/`](audit/): mining protocols and machine-readable progress state

Both idea logs record what was *skipped* and why: learned, GPU-bound and
kernel-dependent work is marked external rather than faked.

## Contributing

Modules are stdlib-only, deterministic, and use absolute imports; every module has
a `unittest.TestCase` at the mirrored path under `tests/`. A monolithic
`unittest discover` segfaults at OCCT teardown, so count per module
(`python -m unittest tests.domain.geometry.sdf.test_primitives`).
`tests/test_suite_collectable.py` fails loudly if a test file is added that the
canonical runner would not collect. Seven such files once sat in this suite
holding 26 assertions that had never executed.

## License and citation

Released under the [MIT License](LICENSE). HarnessCAD does not reproduce a single
paper: its domain layer is mined from 186 papers and 70 repositories, and each
module's originating work is attributed in
[`docs/corpus/paper-ideas.md`](docs/corpus/paper-ideas.md) and
[`docs/corpus/repo-ideas.md`](docs/corpus/repo-ideas.md). Cite the originating
work for the specific capability you use, not this repository.
