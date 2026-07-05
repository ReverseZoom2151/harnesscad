<div align="center">

# HarnessCAD

**A native agentic harness for engineering/mechanical text-to-CAD — the harness, not the model, is the product.**

![Tests](https://img.shields.io/badge/tests-159%20passing-brightgreen?style=flat-square)
![Phase](https://img.shields.io/badge/phase-1%20minimal%20harness-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
![Core: stdlib](https://img.shields.io/badge/core-stdlib--only-informational?style=flat-square)

<p>
  <a href="https://www.python.org"><img src="assets/logos/python.svg" height="44" alt="Python" title="Python 3.10+"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/CadQuery/cadquery"><img src="assets/logos/cadquery.svg" height="40" alt="CadQuery" title="CadQuery"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://dev.opencascade.org"><img src="assets/logos/occt.png" height="30" alt="OpenCASCADE" title="OpenCASCADE (OCCT)"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/BerriAI/litellm"><img src="assets/logos/litellm.svg" height="34" alt="LiteLLM" title="LiteLLM"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/567-labs/instructor"><img src="assets/logos/instructor.png" height="40" alt="Instructor" title="Instructor"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/KmolYuan/solvespace"><img src="assets/logos/solvespace.svg" height="42" alt="SolveSpace" title="python-solvespace (SolveSpace)"></a>
</p>

</div>

HarnessCAD turns a natural-language design brief into a precise, *verified* sequence of
parametric CAD operations. It is not a model and it is not a plugin: it is the
**harness** around a frontier model — the loop, the typed op language, the plural
geometry verifier, the event-sourced op history, and the kernel seam — that makes
text-to-CAD reliable enough to trust. The core spine is pure Python standard library
(no required dependencies); a real OpenCASCADE geometry kernel and a provider-agnostic
LLM layer are opt-in extras.

## About

The thesis is simple and load-bearing: **the harness, not the model, is the product.**
Frontier models can already emit CAD code; what they cannot do on their own is know
whether the geometry they emitted is *right*. Structured output guarantees a tool call
*parses* — it never guarantees the solid is manifold, the sketch is fully constrained,
or the boolean did not null the body. The real safety net is always **external
execution plus geometry checks**, never model self-confidence.

CAD is the rare, valuable setting where that safety net can be made rigorous: it is a
**verifiable-reward domain**. Geometry compiles or it does not; constraints solve or
they do not; dimensions, mass, and interference either match the spec or they do not.
The deterministic verifier is simultaneously the reward, the eval, and the ceiling —
so HarnessCAD is built **verifier-first**, around a verifier that is *plural* (several
independent checks whose diagnostics feed back into the loop) rather than a single
end-gate.

The winning loop is already proven in coding agents. Aider's `edit -> compile -> run
tests -> commit` maps one-to-one onto CAD:

| Coding agent (solved) | HarnessCAD |
|---|---|
| Edit a source file | **Emit a typed CISP op** (sketch, constrain, extrude, fillet, boolean) |
| Compile | **Kernel regeneration** of the op stream |
| Compile error | Regen failure (empty profile, failed boolean, over-constrained sketch) |
| Run tests | **Geometry checks** (sketch DOF, manifold / watertight, solid presence) |
| Observe the traceback | Read diagnostics + measurements |
| Git commit on success | **Checkpoint the op-DAG** (deterministic replay + rollback) |

HarnessCAD is **frontier-model-native** (bring any model through the LiteLLM seam;
train nothing) and **kernel-agnostic**: everything above the `GeometryBackend` seam is
pure logic, so the dependency-free stub, the CadQuery/OCCT kernel, and a future
Rust-native kernel are interchangeable behind one interface.

## How it works

Each op the agent emits goes through the same transactional cycle. An op the backend
rejects (bad reference, non-positive radius, kernel exception) never mutates
state — **block-and-correct**. An op that applies but fails a verifier is **rolled
back** to the last good state. Only an accepted *and* verified op is checkpointed:

```
brief ──▶ planner ──▶ [op, op, op] ──▶ HarnessSession.apply_ops
                ▲                              │
                │                     ┌────────┴─────────┐
                │                     ▼                  ▼   (per op)
                │              backend.apply        block-and-correct
                │                     │             (reject, no mutate)
                │                     ▼
                │              backend.regenerate
                │                     ▼
                │              verify (plural)  ──ERROR──▶ rollback last op
                │                     │ ok
                │                     ▼
                │              checkpoint op-DAG
                │                     │
                └──── diagnostics ◀───┘  (re-plan until verified or max_iters)
```

The op-DAG is append-only and content-hashed: each node chains its parent's hash with
the canonical JSON of its op, so an identical op sequence always produces an identical
`digest`. That single invariant gives checkpoint, rollback, and deterministic replay
for free.

## Architecture

Every layer below is kernel-agnostic and LLM-agnostic until the seam that names
otherwise. The from-scratch core is the middle band.

```
┌──────────────────────────────────────────────────────────────────┐
│  LLM seam        llm/  — Message · ToolSpec · CompletionResult    │  provider-neutral
│                  LiteLLMClient (~100 providers)  ·  Instructor     │
├──────────────────────────────────────────────────────────────────┤
│  Build pipeline  pipeline.build — brief -> planner -> session ->  │  one-call end-to-end
│                  verified geometry -> STEP  ·  cli.py build        │
├──────────────────────────────────────────────────────────────────┤
│  Agent           agent/  — Planner (NL brief -> validated ops)    │
│                  runner.run (plan -> apply -> observe -> replan)   │
├──────────────────────────────────────────────────────────────────┤
│  Grounding       memory/  — MemoryStore (working/episodic/        │  skills grow only when
│                  semantic/procedural) · Voyager-style SkillLibrary │  their geometry verifies
├──────────────────────────────────────────────────────────────────┤
│  Harness loop    loop.HarnessSession                              │  ← the from-scratch core
│                  applyOps -> regen -> verify -> checkpoint         │
│                  block-and-correct · transactional rollback        │
├──────────────────────────────────────────────────────────────────┤
│  Plural verifier verify.py — SketchConstraintCheck (DOF) ·        │  diagnostics feed
│                  SolidPresenceCheck · BRepValidityCheck (topology) │  back into the loop
│                  contract.ContractCheck · constraints.py (solver)  │
├──────────────────────────────────────────────────────────────────┤
│  Ops-DAG         state/opdag.py  — append-only, content-hashed    │  "git for CAD"
│                  checkpoint · rollback · deterministic replay      │
├──────────────────────────────────────────────────────────────────┤
│  GeometryBackend backends/base.py  (swappable kernel seam)        │
│    StubBackend (stdlib) · CadQueryBackend (OCCT) · future Rust    │
├──────────────────────────────────────────────────────────────────┤
│  CISP surface    cisp/ (typed ops + protocol) · server.py (stdio) │  LSP-inspired
│                  cli.py · trace.py (event stream)                 │  JSON methods
└──────────────────────────────────────────────────────────────────┘
```

`CISP` is the CAD Interaction / Sketch Protocol — an LSP-inspired vocabulary
(`initialize` / `applyOps` / `query` / `verify` / `export`) over line-delimited JSON,
so the same harness drops into an MCP server, a subprocess, or a stdio pipe unchanged.

Four newer layers wrap that spine. **`pipeline.build`** (and `cli.py build`) is the
single end-to-end entry point: brief -> planner -> `HarnessSession` -> verified
geometry -> STEP. The **plural verifier** now runs three independent checks in the
default set — `SketchConstraintCheck` (DOF), `SolidPresenceCheck`, and the real
`BRepValidityCheck` (OCCT topology / manifold / watertight) — while `constraints.py`
adds a genuine DOF model (`ConstraintGraph` rank analysis, plus an optional
SolveSpace-backed `SolveSpaceSketch` real 2D solver) and `contract.py` adds an
opt-in **Contract** acceptance spec (required dims + tolerances, volume/mass,
feature counts, manifold/validity, named predicates) verified by `ContractCheck`.
The **`memory/`** grounding layer holds the four memory types and a Voyager-style
`SkillLibrary` that admits a skill only when its expanded ops verify. And
**`bench/`** is CADBench-Verified — a SWE-bench-style eval that runs tasks through
the same spine and scores editability, program execution, B-rep validity, and
dimension match per difficulty.

## Quickstart

The core spine has **no dependencies** — clone and run. Python 3.10+.

```sh
git clone <repo> && cd harnesscad
python cli.py demo                             # built-in constrained-plate -> extrude sample
python -m unittest discover -s tests -t . -v   # the full suite (159 tests)
```

### Drive a session directly

Build a plate and extrude it on the dependency-free stub backend. The stub is not
geometry — it models the op *semantics* (DOF tracking, references, digests,
deterministic replay) so the whole harness spine runs with nothing installed.

```python
from backends.stub import StubBackend
from loop import HarnessSession
from cisp.ops import NewSketch, AddRectangle, Constrain, Extrude

session = HarnessSession(StubBackend())
result = session.apply_ops([
    NewSketch(plane="XY"),
    AddRectangle(sketch="sk1", x=0.0, y=0.0, w=20.0, h=10.0),
    Constrain(kind="distance", a="e1", value=20.0),
    Constrain(kind="distance", a="e1", value=10.0),
    Extrude(sketch="sk1", distance=5.0),
])

print("ok:", result.ok)            # -> ok: True
print("applied:", result.applied)  # -> applied: 5
print("digest:", result.digest)    # deterministic content hash of the model
print("summary:", session.summary())
# -> {'sketch_count': 1, 'entity_count': 1, 'feature_count': 1, 'solid_present': True}
```

Ids are assigned deterministically: sketches are `sk1, sk2, ...`, sketch entities are
`e1, e2, ...`, features are `f1, f2, ...`. `apply_ops` returns an `ApplyOpsResult`
(`ok`, `applied`, `digest`, `diagnostics`, `rejected`).

### Close the agent loop

Give the planner any `LLM` (a mock here; a live model in practice) and let the runner
plan, apply, observe diagnostics, and re-plan until the model verifies:

```python
from llm.base import LLM, CompletionResult
from agent.planner import Planner
from agent.runner import run
from loop import HarnessSession
from backends.stub import StubBackend

class MockLLM(LLM):
    def complete(self, messages, tools=None, response_schema=None, **o):
        return CompletionResult(text='''[
          {"op": "new_sketch", "plane": "XY"},
          {"op": "add_circle", "sketch": "sk1", "cx": 0, "cy": 0, "r": 8},
          {"op": "extrude", "sketch": "sk1", "distance": 4}
        ]''')
    def stream(self, *a, **k):
        yield ""

session = HarnessSession(StubBackend())
result = run(session, Planner(MockLLM()), "a round boss 16mm across, 4mm tall")
print("ok:", result.ok, "applied:", result.applied)   # -> ok: True applied: 3
```

To use a real model, swap in the LiteLLM backend (`pip install -e .[llm]`):

```python
from llm.litellm_backend import LiteLLMClient
planner = Planner(LiteLLMClient(model="gpt-4o-mini", temperature=0.0))
```

### Build from a brief

`pipeline.build` is the single end-to-end entry point — brief -> LLM planner ->
`HarnessSession` -> verified geometry -> STEP. Drive it from the CLI:

```sh
export ANTHROPIC_API_KEY=...                    # or OPENAI_API_KEY
python cli.py build "an M6 clearance plate, 40x20x5mm" --out part.step
python cli.py build "a round boss 16mm across, 4mm tall" --backend stub
```

`build` needs a provider key in the environment (`ANTHROPIC_API_KEY` or
`OPENAI_API_KEY`); with neither set it exits with a clear, actionable error. The
`--backend cadquery` default **falls back to the stub** when CadQuery is not
installed (reported in a `note:` line), so the command always runs.

In Python, inject any `LLM` (a mock here so the snippet runs with nothing installed;
a live model in practice) and get a plain result dict back:

```python
from pipeline import build
from llm.base import LLM, CompletionResult

class MockLLM(LLM):
    def complete(self, messages, tools=None, response_schema=None, **o):
        return CompletionResult(text='''[
          {"op": "new_sketch", "plane": "XY"},
          {"op": "add_circle", "sketch": "sk1", "cx": 0, "cy": 0, "r": 8},
          {"op": "extrude", "sketch": "sk1", "distance": 4}
        ]''')
    def stream(self, *a, **k):
        yield ""

result = build("a round boss 16mm across, 4mm tall", llm=MockLLM(), backend="stub")
print("ok:", result["ok"], "applied:", result["applied"], "backend:", result["backend"])
# -> ok: True applied: 3 backend: stub
```

Omit `llm=` and `build` constructs a lazy `LiteLLMClient` (built only on the first
model call) using the environment key. The result dict carries `ok`, `applied`,
`digest`, `diagnostics`, `summary`, the resolved `backend`, and the exported `step`
text when `ok`.

### The CLI

```sh
python cli.py demo                              # constrained-plate sample (stub)
python cli.py apply examples/ops_plate.json     # run a JSON array of ops
python cli.py apply examples/ops_plate.json --backend cadquery   # real OCCT solid
python cli.py build "<brief>" --out part.step   # brief -> verified geometry (needs API key)
```

`cli.py apply` and `cli.py build` exit non-zero when the resulting model is not `ok`,
so they compose in scripts and CI (`python cli.py apply plan.json && next-step`).

### The CISP server

The harness also speaks CISP over stdio — one JSON request per line, one response per
line — for MCP / subprocess integration:

```python
from server import CISPServer

server = CISPServer(backend="stub")   # or "cadquery"
server.handle({"id": 1, "method": "initialize"})
server.handle({"id": 2, "method": "applyOps", "params": {"ops": [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_circle", "sketch": "sk1", "cx": 0, "cy": 0, "r": 5},
    {"op": "extrude", "sketch": "sk1", "distance": 3},
]}})
# python server.py --backend stub        # or run it as a stdio loop
```

## The CISP op set (v0)

These are the *mutating* ops the agent emits; `measure` and `export` are queries handled
by the backend, not the op log. Sketch and constraint ops come first by design — the
wedge is sketch/constraint/layout assist, not one-shot solids. Each op is a frozen,
hashable dataclass with a stable tag, which is what makes the op stream deterministic.

| Op tag | Parameters | What it does |
|--------|------------|--------------|
| `new_sketch` | `plane` (`"XY"` / `"YZ"` / `"XZ"`) | Start a sketch on a datum plane |
| `add_point` | `sketch, x, y` | Add a point (2 DOF) |
| `add_line` | `sketch, x1, y1, x2, y2` | Add a line segment (4 DOF) |
| `add_circle` | `sketch, cx, cy, r` | Add a circle (3 DOF); `r > 0` |
| `add_rectangle` | `sketch, x, y, w, h` | Add a rectangle profile (4 DOF); `w, h > 0` |
| `constrain` | `kind, a, b?, value?` | Apply a geometric/dimensional constraint, reducing sketch DOF |
| `extrude` | `sketch, distance` | Extrude a closed profile into a solid; `distance != 0` |
| `fillet` | `edges, radius` | Round edges of the current solid; `radius > 0` |
| `boolean` | `kind` (`union` / `cut` / `intersect`), `target, tool` | Combine two solids |

Constraint kinds and the DOF each removes: `coincident` (2), `horizontal`, `vertical`,
`parallel`, `perpendicular`, `distance`, `radius`, `equal` (1 each). Dimensional
constraints (`distance`, `radius`) require a numeric `value`. A sketch that reaches
0 remaining DOF is fully constrained; a negative DOF is over-constrained (an ERROR that
gets rolled back), a positive DOF is under-constrained (a warning).

## Dependencies / tech stack

The **core spine is standard-library-only** — there is nothing to install to run the
stub backend, the loop, the verifier, the op-DAG, the CLI, or the CISP server. Real
geometry and live models are **opt-in extras**, imported lazily so the package loads
even when they are absent. Install what you need:

```sh
pip install -e .                        # core only (stdlib)
pip install -e .[cadquery]              # + real OCCT geometry backend
pip install -e .[llm]                   # + provider-agnostic model access
pip install -e .[constraints]           # + real 2D constraint solver (SolveSpace)
pip install -e .[cadquery,llm,constraints]   # everything
```

| | Dependency | Extra | How it's resolved / notes |
|:---:|------------|-------|---------------------------|
| <img src="assets/logos/python.svg" height="40" alt="Python"> | [Python](https://www.python.org) 3.10+ | core | The whole spine is stdlib-only — zero required runtime dependencies |
| <img src="assets/logos/cadquery.svg" height="34" alt="CadQuery"> | [CadQuery](https://github.com/CadQuery/cadquery) | `cadquery` | The real-geometry `GeometryBackend`; imported lazily, so the module loads without it |
| <img src="assets/logos/occt.png" height="30" alt="OpenCASCADE"> | [OpenCASCADE](https://dev.opencascade.org) (OCCT) | `cadquery` | The B-rep kernel under CadQuery (via `cadquery-ocp`); powers real solids, validity checks, and STEP/STL export |
| <img src="assets/logos/litellm.svg" height="32" alt="LiteLLM"> | [LiteLLM](https://github.com/BerriAI/litellm) | `llm` | One call shape across ~100 providers behind the vendor-neutral `LLM` seam; lazy import |
| <img src="assets/logos/instructor.png" height="36" alt="Instructor"> | [Instructor](https://github.com/567-labs/instructor) | `llm` | Optional structured-output coaxing; the harness falls back to plain JSON + `parse_op` when absent |
| <img src="assets/logos/solvespace.svg" height="40" alt="SolveSpace"> | [python-solvespace](https://github.com/KmolYuan/solvespace) | `constraints` | Real 2D sketch constraint solver (SolveSpace) behind `constraints.SolveSpaceSketch`; imported lazily. The stdlib `ConstraintGraph` rank-based DOF analysis needs nothing installed |

The kernel is deliberately behind a seam (`backends/base.py`): the same op stream runs
on the stub, on CadQuery/OCCT, or on a future Rust-native kernel (Fornjot / Truck /
Cadmium) with no change above the backend.

## Project structure

```
harnesscad/
├── cli.py                  # CLI: `demo`, `apply <ops.json>`, `build "<brief>"` (--backend stub|cadquery)
├── pipeline.py             # build() — brief -> planner -> session -> verified geometry -> STEP
├── server.py               # CISPServer: initialize/applyOps/query/verify/export over stdio
├── loop.py                 # HarnessSession — the applyOps->regen->verify->checkpoint spine
├── verify.py               # plural verifier: SketchConstraintCheck, SolidPresenceCheck, BRepValidity
├── checks_geometry.py      # BRepValidityCheck — real OCCT topology check (manifold/watertight)
├── constraints.py          # 2D DOF: ConstraintGraph (rank analysis) + SolveSpaceSketch (real solver)
├── contract.py             # Contract acceptance spec + ContractCheck verifier (dims/mass/topology)
├── trace.py                # observability: typed event stream (Null/InMemory/Jsonl tracers)
├── cisp/
│   ├── ops.py              #   the v0 CISP op set (frozen dataclasses) + parse/canonical JSON
│   └── protocol.py         #   ApplyOpsResult — the shape the agent sees back
├── state/
│   └── opdag.py            #   ops-DAG: append-only, content-hashed history ("git for CAD")
├── backends/
│   ├── base.py             #   GeometryBackend protocol (the swappable kernel seam)
│   ├── stub.py             #   StubBackend — dependency-free op semantics
│   └── cadquery_backend.py #   CadQueryBackend — real OCCT B-rep solids
├── llm/
│   ├── base.py             #   the provider seam: Message, ToolSpec, CompletionResult, LLM
│   ├── litellm_backend.py  #   LiteLLMClient — ~100 providers behind the seam
│   └── structured.py       #   response -> validated ops (with re-promptable error strings)
├── agent/
│   ├── system_prompt.py    #   role + op vocabulary (generated from cisp.ops, never drifts)
│   ├── planner.py          #   Planner — NL brief -> validated CISP ops
│   └── runner.py           #   plan -> apply -> observe -> replan correction loop
├── memory/
│   ├── store.py            #   MemoryStore — working/episodic/semantic/procedural memory
│   └── skills.py           #   SkillLibrary — Voyager-style, execution-verified skill templates
├── bench/
│   ├── task.py             #   CADBench-Verified Task schema (spec + reference ops + acceptance)
│   ├── runner.py           #   run_task / run_suite over the HarnessSession spine
│   └── metrics.py          #   editability, program-execution, B-rep validity, dimension match
├── examples/
│   ├── ops_plate.json      #   a runnable op array (constrained plate -> extrude)
│   └── bench_tasks/        #   easy/medium/hard CADBench-Verified task files
├── tests/                  # 159 unittest tests across every module
├── HARNESS_BLUEPRINT.md    # the founding design doc / north star
└── pyproject.toml          # stdlib core; [cadquery], [llm], [constraints] optional extras
```

Research and reference material lives under a gitignored `resources/` directory and is
never committed — it is not part of the product.

## Roadmap

The staged plan from [HARNESS_BLUEPRINT.md](HARNESS_BLUEPRINT.md). Phase 0/1 and the
first slices of the kernel and LLM layers are in place; the rest is sequenced behind
them.

**Done**

- Phase 0 — the deterministic verifier and the result/diagnostic schema (reward + eval + ceiling).
- Phase 1 — the minimal harness: typed ops, kernel regen, plural verification, checkpoint/rollback, an event-sourced op-DAG, and the single-agent plan/apply/observe/replan loop.
- The `GeometryBackend` seam with a dependency-free stub **and** a real CadQuery/OCCT backend (real B-rep solids, validity checks, STEP/STL export).
- The vendor-neutral LLM layer (LiteLLM backend + structured-output funnel) and the CISP stdio server + CLI.
- The **end-to-end build pipeline** (`pipeline.build`) and its `cli.py build` front door: brief -> planner -> session -> verified geometry -> STEP.
- The **Contract** layer — a machine-verifiable acceptance spec (required dims + tolerances, volume/mass, feature counts, manifold/validity, named predicates) verified by `ContractCheck`.
- A real **2D constraint solver**: the stdlib `ConstraintGraph` rank-based DOF analysis plus the optional SolveSpace-backed `SolveSpaceSketch`, replacing the nominal DOF placeholder. B-rep validity is now in the default verifier set.
- Phase 2 grounding (first slice): `MemoryStore` (working/episodic/semantic/procedural) and a Voyager-style, execution-verified **skill library**.
- **CADBench-Verified**: a SWE-bench-style, programmatically-checked eval harness (sketch editability, program execution, B-rep validity, dimension match) over the harness spine, with easy/medium/hard task files.

**In progress / planned**

- Phase 2 (remainder) — hybrid **RAG grounding** over standards and API docs, and a richer context manager.
- Phase 3 — reliability: Best-of-N + verifier, a Reflexion loop, `before_tool_callback` guardrails, and the full error-recovery ladder.
- A **VLM-judge** verifier and assembly-mate checks to broaden the plural verifier beyond geometry/topology.
- Phase 5 — scale: **multi-agent orchestration** (Designer / Verifier / DFM Critic / Red Team), variant exploration with Elo ranking, and the canvas UI.
- A **data flywheel** — trajectory logging turned into a curated dataset for training/fine-tuning.
- A **Rust-native kernel** (Fornjot / Truck / Cadmium) dropped in behind the existing `GeometryBackend` seam.

## Design doc

The full thesis, layered architecture, verification strategy, and open decisions are in
[HARNESS_BLUEPRINT.md](HARNESS_BLUEPRINT.md) — the north star this codebase is built
toward.

## License

MIT.
</content>
</invoke>
