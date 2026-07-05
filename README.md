<div align="center">

# HarnessCAD

**A native agentic harness for engineering/mechanical text-to-CAD — the harness, not the model, is the product.**

![Tests](https://img.shields.io/badge/tests-3497%20passing-brightgreen?style=flat-square)
![Phase](https://img.shields.io/badge/phases%200--5-implemented-blue?style=flat-square)
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
│  Cost routing    routing.RoutingLLM — classify -> cheapest capable │  wraps the LLM seam
│                  model -> fallback chain · running cost/usage tally │
├──────────────────────────────────────────────────────────────────┤
│  LLM seam        llm/  — Message · ToolSpec · CompletionResult    │  provider-neutral
│                  LiteLLMClient (~100 providers)  ·  Instructor     │
│  Constrained     grammar.py — op JSON Schema + GBNF/EBNF grammar   │  derived from the
│  decoding        + GrammarConstraint post-hoc validator            │  op registry
├──────────────────────────────────────────────────────────────────┤
│  Multi-agent     agents/ — Designer·Modeler·Verifier·DFMCritic·   │  supervisor loop
│                  RedTeam·Reviewer + Supervisor + AsyncOverseer      │  + halt authority
│                  a2a/ — AgentCard · A2AMessage · Task lifecycle     │  inter-agent bus
├──────────────────────────────────────────────────────────────────┤
│  Reliability     reliability/ — strategies/ best_of_n · Reflexion  │  spend compute to
│                  · MCTS search · repair.py (OCCT heal + suggest)    │  raise success ·
│                  guardrails.GuardrailGate (before-tool) ·           │  block / repair bad
│                  ErrorRecovery ladder · loopdetect.LoopDetector     │  ops before they land
├──────────────────────────────────────────────────────────────────┤
│  Front-of-       spec/ (formalize + interview -> RequirementSet) · │  brief -> spec ->
│  pipeline        skeleton/ (master-sketch layout) · sizing/        │  sized skeleton,
│                  (first-principles engineering calc)               │  before any op
├──────────────────────────────────────────────────────────────────┤
│  Ingestion /     ingest/ (STEP/BREP import · decompile -> ops ·    │  bring existing
│  library         metadata) · library/ (parametric parts + catalog) │  geometry back in
│                  · quality/suggest_cots (nearest COTS match)       │
├──────────────────────────────────────────────────────────────────┤
│  Build pipeline  pipeline.build — brief -> planner -> session ->  │  one-call end-to-end
│                  verified geometry -> STEP  ·  cli.py build        │
├──────────────────────────────────────────────────────────────────┤
│  Agent           agent/  — Planner (NL brief -> validated ops)    │
│                  runner.run (plan -> apply -> observe -> replan)   │
├──────────────────────────────────────────────────────────────────┤
│  Grounding       context/ — ContextManager (token budget) +        │  skills grow only when
│                  StagingArea · rag/ — hybrid BM25+vector retriever  │  their geometry verifies
│                  memory/ — MemoryStore + Voyager-style SkillLibrary │
├──────────────────────────────────────────────────────────────────┤
│  Harness loop    loop.HarnessSession                              │  ← the from-scratch core
│                  applyOps -> regen -> verify -> checkpoint         │
│                  block-and-correct · transactional rollback        │
├──────────────────────────────────────────────────────────────────┤
│  Plural verifier verifiers/ — verify (DOF · solid · BRep validity) │  diagnostics feed
│                  geometry · dfm · vision · assembly/mate · inter-   │  back into the loop
│                  ference/clash · standards · compliance · require-  │
│                  ments · reference  ·  constraints.py · contract.py │
├──────────────────────────────────────────────────────────────────┤
│  Quality         quality/ — estimate (mass/cost/BOM) · fitness     │  measure & narrate
│                  (multi-objective) · kinematics (motion/DOF) ·      │  the verified part
│                  anomaly · diff · describe · featuregraph · cots    │
├──────────────────────────────────────────────────────────────────┤
│  Ops-DAG         state/opdag.py  — append-only, content-hashed    │  "git for CAD"
│                  checkpoint · rollback · deterministic replay      │
├──────────────────────────────────────────────────────────────────┤
│  GeometryBackend backends/base.py  (swappable kernel seam)        │
│    StubBackend (stdlib) · CadQueryBackend (OCCT) · future Rust    │
├──────────────────────────────────────────────────────────────────┤
│  Surfaces        cisp/ (typed ops + protocol) · surfaces/ —        │  LSP-inspired
│                  server.py (stdio) · mcp/ (ToolCatalog + CADGymEnv) │  JSON methods
│                  · ui/ (SSE + three-tier approval) · render.py     │
├──────────────────────────────────────────────────────────────────┤
│  Observability   observe.py (spans · KPI metrics · failure         │  cross-cutting
│                  taxonomy · replay) · trace.py (event stream)       │
├──────────────────────────────────────────────────────────────────┤
│  Data engine     dataengine/ (Trajectory + GRPO/DPO/STaR export)  │  offline flywheel
│                  datagen/ (synthetic generators + solver-in-loop)  │
└──────────────────────────────────────────────────────────────────┘
```

`CISP` is the CAD Interaction / Sketch Protocol — an LSP-inspired vocabulary
(`initialize` / `applyOps` / `query` / `verify` / `export`) over line-delimited JSON,
so the same harness drops into an MCP server, a subprocess, or a stdio pipe unchanged.

**The plural verifier.** The `verifiers/` package is the plural verifier. The default
set runs three independent checks — `verify.SketchConstraintCheck` (DOF),
`verify.SolidPresenceCheck`, and the real `geometry.BRepValidityCheck` (OCCT topology /
manifold / watertight) — where `constraints.py` supplies a genuine DOF model
(`ConstraintGraph` rank analysis plus an optional SolveSpace-backed real 2D solver) and
`contract.ContractCheck` adds an opt-in acceptance spec (required dims + tolerances,
volume/mass, feature counts, manifold/validity, named predicates). A stack of opt-in
checks broadens it far beyond a single solid: `dfm.DFMCheck` is a Design-for-
Manufacturing critic (aspect-ratio, thin/small-part) that only ever emits WARNING/INFO;
`vision.VLMJudgeCheck` is the VLM-as-judge for the subjective slice (design intent,
cleanliness), advisory-only with a 0..1 score; `assembly.AssemblyCheck` solves mates
and residual degrees of freedom; `interference.InterferenceCheck` is the clash / swept-
volume detector; `standards.py` snaps dimensions to preferred (Renard) series;
`compliance.ComplianceCheck` enforces house/standard rules; `requirements.RequirementsCheck`
holds the model to a formalised `RequirementSet`; and `reference.ReferenceMatchCheck`
scores the solid against a reference part. The advisory critics are additive by design
— they cannot flip a passing report to failing.

**Quality read-out.** Once a part verifies, `quality/` measures and narrates it:
`estimate.py` computes mass / cost / a bill of materials from a material table,
`fitness.Objective` scores it against a weighted multi-objective spec (with Pareto
dominance), `kinematics.py` validates motion and joint DOF, `anomaly.py` flags
outliers, `diff.py` is a semantic op-level diff, `describe.py` narrates the part in
prose, `featuregraph.py` lifts the op stream into a feature graph, and `suggest_cots.py`
matches features to the nearest catalog part.

**Grounding.** `context/` manages the finite token window explicitly — `ContextManager`
budgets `C >= system + memory + tools + history + reserved`, guards overflow pre-flight,
and assembles a prefix-cache-friendly prompt, while `StagingArea` is the file-based
per-task "anti-RAG". `rag/` is a dependency-free hybrid retriever — structure-aware
chunking feeds a BM25 lexical index and an embedding-free hashed-vector index fused by
reciprocal-rank fusion. `memory/` holds the four memory types and a Voyager-style
`SkillLibrary` that admits a skill only when its expanded ops verify.

**Reliability.** The `reliability/` package spends compute to raise success.
`strategies/best_of_n` draws N seeded candidate plans through fresh sessions and lets
the deterministic verifier pick the winner; `strategies/reflexion` runs a
Read-Act-Reflect-Write loop that writes failure insights to semantic memory and
retries; `strategies/mcts` adds a UCB Monte-Carlo tree search over op expansions scored
by the verifier. Before any op applies, `guardrails.GuardrailGate` (the
`before_tool_callback` hard gate) rejects obviously invalid ops without mutating kernel
state, `loopdetect.LoopDetector` catches an agent retrying the identical op, and
`guardrails.ErrorRecovery` enumerates the detect -> handle -> recover ladder. When a
solid is broken rather than rejected, `repair.py` runs an OCCT shape-healing pass and
emits ranked repair suggestions.

**Multi-agent + surfaces.** `agents/` wraps the single-agent baseline with six role
personas (Designer, Modeler, Verifier, DFMCritic, RedTeam, Reviewer), a `Supervisor`
that chains them and feeds diagnostics back each round, and an `AsyncOverseer` with
halt authority; `a2a/` is the inter-agent wire format (`AgentCard`, `A2AMessage`, and a
guarded `Task` lifecycle with SSE-style events). The `surfaces/` package holds the
outward faces: `surfaces/server.py` speaks CISP over stdio; `surfaces/mcp/` exposes the
environment as an MCP-style server (`ToolCatalog` — one tool per op plus
`measure`/`query`/`verify`/`export`/`render` — with behavioural `annotations`) and a
`CADGymEnv` Gym environment (`reset`/`step` -> obs, verifier-derived reward, done,
info); `surfaces/ui/` is the outward SSE event contract (`UIEvent`/`EventStream`) plus a
three-tier approval gate (AUTO / NOTIFY / REQUIRE) with dry-run previews; and
`surfaces/render.py` renders the current solid to multi-view SVG/PNG bytes as the
observation half of a render -> judge loop. On top of these seams sit the agent-protocol
adapters — `surfaces/mcp/` (an MCP server), `surfaces/a2a_server/` (an A2A server), and
`surfaces/acp/` (an ACP agent) — described under
[Protocol integrations](#protocol-integrations).

**Cross-cutting.** `routing.RoutingLLM` is a drop-in `LLM` that classifies each request
and routes it to the cheapest capable model with a fallback chain and a running cost
tally. `grammar.py` derives an op JSON Schema and a GBNF/EBNF grammar from the op
registry (so they cannot drift) plus a stdlib post-hoc `GrammarConstraint` validator.
`observe.py` computes the blueprint's KPIs from a run trajectory with confidence
intervals, classifies failures into a taxonomy, and replays runs. And the offline
**data engine** folds each run into a canonical `Trajectory` and exports GRPO / DPO /
STaR training rows (`dataengine/`) and audits that corpus (`distribution_audit`,
`active_learning`, `consensus`, `intent`), while `datagen/` bootstraps cold-start data
with seeded synthetic generators and `augment` transforms, kept honest by
solver-in-the-loop verification.

**Front of the pipeline.** Before a single op is emitted, `spec/` turns a loose brief
into a machine-checkable `RequirementSet` — `formalize.py` extracts requirements and
`interview.py` asks the missing questions — which `verifiers/requirements` later holds
the model to. `skeleton/` lays out a master-sketch (envelopes + datums) and `sizing/`
runs first-principles engineering calculations (shaft-in-torsion, plate-in-bending,
bolt-count-in-shear) so the plan starts from sound dimensions rather than a guess.

**Ingestion + library.** `ingest/` brings existing geometry back into the loop —
`import_brep.py` reads STEP/BREP, `decompile.py` lifts an imported solid back into a
CISP op stream, and `metadata.py` extracts part metadata — while
`verifiers/reference` scores a build against that reference. `library/` is a parametric
parts library (`parts.py` op-templates with model cards, `catalog.py`) that
`quality/suggest_cots` searches for the nearest commercial-off-the-shelf match.

Finally, **`bench/`** is CADBench-Verified — a SWE-bench-style eval that runs tasks
through the same spine and scores editability, program execution, B-rep validity, and
dimension match per difficulty.

## Quickstart

The core spine has **no dependencies** — clone and run. Python 3.10+.

```sh
git clone <repo> && cd harnesscad
python cli.py demo                             # built-in constrained-plate -> extrude sample
python -m unittest discover -s tests -t . -v   # the full suite (1575 tests)
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
from surfaces.server import CISPServer

server = CISPServer(backend="stub")   # or "cadquery"
server.handle({"id": 1, "method": "initialize"})
server.handle({"id": 2, "method": "applyOps", "params": {"ops": [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_circle", "sketch": "sk1", "cx": 0, "cy": 0, "r": 5},
    {"op": "extrude", "sketch": "sk1", "distance": 3},
]}})
# python -m surfaces.server --backend stub    # or run it as a stdio loop (serve_stdio)
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

The mechanical-feature ops broaden the wedge from prismatic parts to real machined and
assembled geometry:

| Op tag | Parameters | What it does |
|--------|------------|--------------|
| `revolve` | `sketch, axis, angle` | Revolve a profile about an axis into a solid |
| `chamfer` | `edges, distance` | Bevel edges of the current solid |
| `hole` | `face_or_sketch, x, y, diameter, through, kind` | Drill a hole (`simple`/`counterbore`/`countersink`/tapped) |
| `shell` | `faces, thickness` | Hollow the solid, removing the named faces |
| `draft` | `faces, angle, neutral_plane` | Apply draft angle to faces about a neutral plane |
| `loft` | `sketches, ruled` | Loft a solid through a series of profiles |
| `sweep` | `sketch, path` | Sweep a profile along a path curve |
| `linear_pattern` | `feature, direction, count, spacing` | Repeat a feature along a direction |
| `circular_pattern` | `feature, axis, count, angle` | Repeat a feature about an axis |
| `mirror` | `feature_or_body, plane` | Mirror a feature or body across a datum plane |
| `add_instance` | `part, x, y, z, rx, ry, rz` | Place a library/part instance into an assembly |
| `mate` | `kind, a, b` | Mate two instances (`rigid` / `revolute` / `slider` / ...) |
| `set_param` | `target, param, value` | Re-parameterise an earlier op by index (parametric edit) |

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

The **agent protocols** (MCP, A2A, ACP) are supported the same way: the adapters under
`surfaces/` are stdlib-only and need nothing installed, while the official SDKs
(`mcp`, `a2a-sdk`, `agent-client-protocol`) are optional extras for richer interop. See
[Protocol integrations](#protocol-integrations).

## Project structure

```
harnesscad/
│  # ── core spine (stays at repo root) ──────────────────────────────
├── loop.py                 # HarnessSession — the applyOps->regen->verify->checkpoint spine
├── harness.py              # AgentHarness — the ReAct loop that ties agent + tools together
├── pipeline.py             # build() — brief -> planner -> session -> verified geometry -> STEP
├── cli.py                  # CLI: `demo`, `apply <ops.json>`, `build "<brief>"` (--backend stub|cadquery)
├── contract.py             # Contract acceptance spec + ContractCheck verifier (dims/mass/topology)
├── constraints.py          # 2D DOF: ConstraintGraph (rank analysis) + SolveSpaceSketch (real solver)
├── grammar.py              # op JSON Schema + GBNF/EBNF grammar + GrammarConstraint (from op registry)
├── routing.py              # RoutingLLM — classify -> cheapest capable model -> fallback + cost tally
├── observe.py              # observability: spans, KPI metrics + CIs, failure taxonomy, run replay
├── trace.py                # typed event stream (Null/InMemory/Jsonl tracers)
│  # ── typed op protocol + kernel seam ───────────────────────────────
├── cisp/
│   ├── ops.py              #   the CISP op set (frozen dataclasses) — sketch/feature/assembly + parse/JSON
│   └── protocol.py         #   ApplyOpsResult — the shape the agent sees back
├── state/
│   └── opdag.py            #   ops-DAG: append-only, content-hashed history ("git for CAD") + bisect()
├── backends/
│   ├── base.py             #   GeometryBackend protocol (the swappable kernel seam)
│   ├── stub.py             #   StubBackend — dependency-free op semantics
│   └── cadquery_backend.py #   CadQueryBackend — real OCCT B-rep solids + STEP/STL/IGES export
│  # ── verifiers/  (the plural verifier) ─────────────────────────────
├── verifiers/
│   ├── verify.py           #   Severity/Diagnostic/VerifyReport/Verifier + default_verifiers()
│   ├── geometry.py         #   BRepValidityCheck — real OCCT topology check (manifold/watertight)
│   ├── dfm.py              #   DFMCheck — opt-in Design-for-Manufacturing critic (WARNING/INFO only)
│   ├── vision.py           #   VLMJudgeCheck — VLM-as-judge for the subjective slice (advisory 0..1)
│   ├── assembly.py         #   AssemblyCheck — mate + residual-DOF solver
│   ├── interference.py     #   InterferenceCheck — clash / swept-volume (sweep-and-prune)
│   ├── standards.py        #   nearest_standard — snap dims to preferred (Renard) series
│   ├── compliance.py       #   ComplianceCheck — enforce house/standard rule sets
│   ├── requirements.py     #   RequirementsCheck — hold the model to a formalised RequirementSet
│   ├── reference.py        #   ReferenceMatchCheck — score the solid against a reference part
│   ├── simulation.py       #   analytic stress/buckling checks + external-FEA seam
│   ├── access.py           #   tool-access and service-clearance envelopes
│   ├── precheck.py         #   reject infeasible op plans before geometry execution
│   ├── completeness.py     #   configurable model-completeness gate
│   ├── functional.py       #   functional-behaviour acceptance oracle
│   └── report.py           #   traceable conformance-certificate exporter
│  # ── reliability/  (spend compute to raise success) ────────────────
├── reliability/
│   ├── guardrails.py       #   GuardrailGate (before-tool hard gate) + ErrorRecovery ladder
│   ├── loopdetect.py       #   LoopDetector — pre-apply sliding-window oscillation detector
│   ├── executor.py         #   ToolExecutor — sandbox / retry / timeout / approval layer
│   ├── repair.py           #   repair_solid — OCCT shape-healing + ranked repair suggestions
│   ├── fallback.py         #   nearest-known-good retrieval fallback on failure/timeout
│   └── strategies/
│       ├── best_of_n.py    #     draw N seeded plans, verifier picks the winner
│       ├── reflexion.py    #     ReflexionLoop — Read-Act-Reflect-Write, learns within a run
│       └── mcts.py         #     UCB Monte-Carlo tree search over op expansions
│  # ── quality/  (measure & narrate the verified part) ───────────────
├── quality/
│   ├── estimate.py         #   mass / cost / bill-of-materials from a material table
│   ├── fitness.py          #   Objective — weighted multi-objective score + Pareto dominance
│   ├── kinematics.py       #   motion / joint-DOF validator
│   ├── anomaly.py          #   outlier / anomaly scoring over feature vectors
│   ├── diff.py             #   semantic op-level diff between two models
│   ├── describe.py         #   narrate the part in prose from its facts
│   ├── featuregraph.py     #   lift the op stream into a feature graph
│   ├── suggest_cots.py     #   match features to the nearest commercial-off-the-shelf part
│   ├── assemblyseq.py      #   collision-aware assembly/disassembly sequence planning
│   ├── drawing.py          #   dimensioned 2D engineering drawing generation
│   ├── pareto.py           #   non-dominated sorting and trade-off matrices
│   ├── traceability.py     #   requirement-to-feature/operation traceability matrix
│   ├── batch_edit.py       #   reviewable semantic multi-feature edits
│   └── ask.py              #   grounded natural-language queries over model facts
│  # ── surfaces/  (the outward faces) ────────────────────────────────
├── surfaces/
│   ├── server.py           #   CISPServer: initialize/applyOps/query/verify/export over stdio (serve_stdio)
│   ├── render.py           #   multi-view render of the current solid to SVG/PNG bytes
│   ├── mcp/                #   MCP: agent-consumes-tools face over the ToolCatalog
│   │   ├── server.py       #     MCPServer — tools/call+list, resources, prompts (MCP 2025-11-25)
│   │   ├── jsonrpc.py      #     stdlib JSON-RPC 2.0 framing for the MCP channel
│   │   ├── stdio.py        #     serve_stdio — the MCP stdio transport loop
│   │   ├── __main__.py     #     `python -m surfaces.mcp` — launch the MCP stdio server
│   │   ├── tools.py        #     ToolCatalog — one tool per op + to_mcp() schema, verifier reward
│   │   ├── annotations.py  #     MCP behavioural hints (readOnly/destructive) -> approval tier
│   │   └── gym.py          #     CADGymEnv — reset/step(obs, reward, done, info) RL environment
│   ├── a2a_server/         #   A2A: agent<->agent peer face (HTTP + JSON-RPC)
│   │   ├── app.py          #     serve() — HTTP transport; message/send+stream, tasks/get+cancel
│   │   ├── card.py         #     AgentCard at /.well-known/agent-card.json (text-to-cad skill)
│   │   ├── handler.py      #     JSON-RPC method dispatch -> AgentHarness; STEP as file Part
│   │   ├── wire.py         #     A2A wire types (Message/Part/Artifact/Task) — stdlib
│   │   └── __main__.py     #     `python -m surfaces.a2a_server --port 9100`
│   ├── acp/                #   ACP: editor-drives-the-harness face (Zed Agent Client Protocol v1)
│   │   ├── agent.py        #     ACPAgent — session/prompt -> AgentHarness.run, SSE -> session/update
│   │   ├── bridge.py       #     approval gate -> session/request_permission; STEP -> fs/write_text_file
│   │   ├── jsonrpc.py      #     newline-delimited JSON-RPC 2.0 stdio Connection
│   │   └── __main__.py     #     `python -m surfaces.acp` — the ACP agent Zed spawns
│   └── ui/
│       ├── events.py       #     UIEvent / EventStream — typed SSE wire protocol (+ parsers)
│       └── approval.py     #     ApprovalGate — three-tier AUTO/NOTIFY/REQUIRE + dry-run preview
│  # ── LLM seam + agent loop ─────────────────────────────────────────
├── llm/
│   ├── base.py             #   the provider seam: Message, ToolSpec, CompletionResult, LLM
│   ├── litellm_backend.py  #   LiteLLMClient — ~100 providers behind the seam
│   └── structured.py       #   response -> validated ops (with re-promptable error strings)
├── agent/
│   ├── system_prompt.py    #   role + op vocabulary (generated from cisp.ops, never drifts)
│   ├── planner.py          #   Planner — NL brief -> validated CISP ops
│   └── runner.py           #   plan -> apply -> observe -> replan correction loop
├── agents/
│   ├── roles.py            #   Designer/Modeler/Verifier/DFMCritic/RedTeam/Reviewer personas
│   ├── supervisor.py       #   Supervisor — chains the roles, feeds diagnostics back each round
│   └── overseer.py         #   AsyncOverseer — event-stream monitor with halt authority
├── a2a/
│   ├── messages.py         #   AgentCard, A2AMessage, Part — inter-agent wire vocabulary
│   └── task.py             #   Task lifecycle state machine + TaskStore (SSE-style events)
│  # ── grounding ─────────────────────────────────────────────────────
├── context/
│   ├── manager.py          #   ContextManager — token-window budget + overflow-guarded assembly
│   └── staging.py          #   StagingArea — file-based per-task task-context/ ("anti-RAG")
├── rag/
│   ├── chunk.py            #   structure-aware Markdown chunking (fenced code kept atomic)
│   ├── index.py            #   BM25Index + embedding-free HashedVectorIndex
│   └── retriever.py        #   HybridRetriever — RRF/weighted fusion + build_from_docs()
├── memory/
│   ├── store.py            #   MemoryStore — working/episodic/semantic/procedural memory
│   └── skills.py           #   SkillLibrary — Voyager-style, execution-verified skill templates
│  # ── front-of-pipeline: brief -> spec -> sized skeleton ────────────
├── spec/
│   ├── formalize.py        #   brief -> RequirementSet (typed, machine-checkable)
│   └── interview.py        #   RequirementsInterview — ask the missing questions
├── skeleton/
│   └── layout.py           #   Skeleton — master-sketch layout (envelopes + datums)
├── sizing/
│   └── calc.py             #   first-principles engineering calc (shaft/plate/bolt sizing)
│  # ── ingestion + parts library ─────────────────────────────────────
├── ingest/
│   ├── import_brep.py      #   read STEP / BREP into an ImportedPart
│   ├── decompile.py        #   lift an imported solid back into a CISP op stream
│   ├── metadata.py         #   extract part metadata (units, bounds, provenance)
│   └── fidelity.py         #   STEP -> op-DAG -> STEP round-trip fidelity metrics
├── standards/
│   ├── registry.py         #   versioned, machine-readable engineering rule packs
│   ├── ingest.py           #   cited clause text -> typed rules
│   └── conflict.py         #   incompatible-rule detection
├── library/
│   ├── parts.py            #   parametric part op-templates + model cards
│   └── catalog.py          #   PartCatalog — searchable default catalog
│  # ── exploration, data engine, measurement ────────────────────────
├── exploration/
│   ├── elo.py              #   rating-conserving Elo + Leaderboard
│   └── tournament.py       #   Co-Scientist generate -> debate -> evolve; cluster + rank variants
├── dataengine/
│   ├── trajectory.py       #   Trajectory — canonical training record (steps + dense rewards)
│   ├── export.py           #   to_grpo / to_dpo / to_star + flywheel_metrics
│   ├── distribution_audit.py #  coverage / distribution audit of the corpus
│   ├── active_learning.py  #   pick the highest-information next tasks to label
│   ├── consensus.py        #   multi-sample consensus / agreement scoring
│   ├── intent.py           #   infer design intent from a trajectory
│   └── edit_pairs.py       #   human edit deltas -> preference-training records
├── datagen/
│   ├── generators.py       #   seeded synthetic (brief, ops, params) generators
│   ├── pipeline.py         #   solver-in-the-loop: keep only parts that verifiably build
│   └── augment.py          #   parameter / structural augmentation of verified samples
├── bench/
│   ├── task.py             #   CADBench-Verified Task schema (spec + reference ops + acceptance)
│   ├── runner.py           #   run_task / run_suite over the HarnessSession spine
│   └── metrics.py          #   editability, program-execution, B-rep validity, dimension match
├── examples/
│   ├── ops_plate.json      #   a runnable op array (constrained plate -> extrude)
│   └── bench_tasks/        #   easy/medium/hard CADBench-Verified task files
├── tests/                  # 1575 unittest tests across every module
├── HARNESS_BLUEPRINT.md    # the founding design doc / north star
└── pyproject.toml          # stdlib core; [cadquery], [llm], [constraints] optional extras
```

Research and reference material lives under a gitignored `resources/` directory and is
never committed — it is not part of the product.

The corpus-to-code accounting is documented in
[CAD_CORPUS_AUDIT.md](CAD_CORPUS_AUDIT.md), with a machine-checkable 67-item
atomic register in `audit/cad_idea_register.json`.

### Module map

The modules grouped by layer package, for navigation:

- **Core spine** — `loop.py`, `harness.py`, `pipeline.py`, `cli.py`, `state/opdag.py`, `cisp/`, `backends/`
- **External adapter contract** — `adapters/` (transactional capability discovery, idempotent apply/verify/commit/rollback, deterministic in-memory host)
- **`verifiers/`** (plural verifier) — core geometry/assembly checks plus opt-in DFM, vision, simulation, access, precheck, completeness, functional and conformance reporting (+ root `constraints.py`, `contract.py`)
- **`reliability/`** — guardrails, loop detection, execution, repair, retrieval fallback and search strategies
- **`quality/`** — estimation/fitness, kinematics/anomaly/diff, narration/feature graphs/COTS, assembly sequencing, drawings, Pareto analysis, traceability, batch edit, grounded Q&A, next-op ranking, simulation jobs and revision deltas
- **`surfaces/`** — server/render, MCP/UI, keyboard commands and deterministic graph/history/debug views, plus the agent-protocol adapters `surfaces/mcp/` (MCP server), `surfaces/a2a_server/` (A2A server) and `surfaces/acp/` (ACP agent)
- **Agent + LLM + decoding** — `agent/`, `agents/`, `a2a/`, `llm/`, `routing.py`, `grammar.py`
- **Grounding** — `context/`, `rag/`, `memory/`
- **Front-of-pipeline** — `spec/` (formalize + interview), `skeleton/`, `sizing/`
- **Ingestion + library** — `ingest/` (import, decompile, metadata, round-trip fidelity), `library/` (parts, catalog)
- **Design-space exploration** — `exploration/` (Co-Scientist + Elo tournament)
- **Observability** — `observe.py`, `trace.py`
- **Data engine** — `dataengine/` (export, distribution/bias audit, active learning, consensus, intent, edit-pairs and consented session capture), `datagen/` (generators, pipeline, augment)
- **Measurement** — `bench/`
- **Security** — `security/` (ingest policy, redaction/audit provenance, prompt/tool trust gate)
- **Research governance** — `research/` (evidence-linked claims, reproducibility gates, reviewer ensemble and rollback)

## Protocol integrations

The harness now speaks the four mainstream agent protocols, each a **thin adapter
over an existing seam** — no new harness logic, just a new outward face. They sit on
three complementary axes, so an integration is really a question of *direction*:

- **MCP — downward.** The agent *consumes tools*: `surfaces/mcp/` is a real
  [Model Context Protocol](https://modelcontextprotocol.io) 2025-11-25 stdio server
  that exposes the CISP ops as MCP **tools** (`tools/list`, `tools/call` with
  `isError` self-correction), the model state as MCP **resources**
  (`resources/list` + `read`), and op-templates as MCP **prompts** (`prompts/list` +
  `get`) — all over the existing `ToolCatalog`.
- **A2A — sideways.** The harness is an *agent peer*: `surfaces/a2a_server/` serves a
  [Google Agent2Agent](https://a2a-protocol.org) `AgentCard` at
  `/.well-known/agent-card.json` with a text-to-cad skill and handles `message/send`,
  `message/stream` (SSE), `tasks/get`, and `tasks/cancel` over HTTP + JSON-RPC,
  returning the verified STEP as a file `Part` in an `Artifact`. (IBM's REST *Agent
  Communication Protocol* merged into A2A under the Linux Foundation in Aug 2025, so
  this one adapter covers both — there is no separate ACP-by-IBM adapter.)
- **ACP — inward.** An *editor drives the harness*: `surfaces/acp/` is a
  [Zed Agent Client Protocol](https://agentclientprotocol.com) v1 agent, so Zed (or any
  ACP editor) runs text-to-CAD in-editor. `session/prompt` maps to `AgentHarness.run`,
  our SSE events become `session/update`, the three-tier approval gate becomes
  `session/request_permission` (allow / reject, once / always), and the STEP is written
  back via `fs/write_text_file`. ACP reuses MCP's `ContentBlock` types.

| Protocol | Role (axis) | Module | Entry point |
|----------|-------------|--------|-------------|
| MCP  | agent consumes tools (down)     | `surfaces/mcp/`        | `python -m surfaces.mcp` |
| A2A  | agent ↔ agent peer (sideways)   | `surfaces/a2a_server/` | `python -m surfaces.a2a_server` |
| ACP  | editor drives the harness (in)  | `surfaces/acp/`        | `python -m surfaces.acp` |

Every adapter is **stdlib-first** — the JSON-RPC framing, stdio/HTTP transports, and
wire types are hand-rolled on the standard library, so each runs with nothing installed.
The official SDKs ([`mcp`](https://pypi.org/project/mcp/),
[`a2a-sdk`](https://pypi.org/project/a2a-sdk/),
[`agent-client-protocol`](https://pypi.org/project/agent-client-protocol/)) are optional
extras for richer interop, not a requirement. These are functional adapters with passing
tests, not battle-tested production servers.

```sh
python -m surfaces.mcp                          # MCP stdio server (CISP ops as MCP tools)
python -m surfaces.a2a_server --port 9100       # A2A HTTP+JSON-RPC server + AgentCard
python -m surfaces.acp                          # ACP v1 agent for Zed / any ACP editor
```

Each honours `--backend stub|cadquery` exactly like the CISP server, falling back to the
stub (with a note on stderr) when CadQuery is absent, so all three always run.

## Roadmap

The staged plan from [HARNESS_BLUEPRINT.md](HARNESS_BLUEPRINT.md). Phases 0-5 are now
substantially implemented and tested against the same spine; what remains under
**Planned / future** is deliberately the parts that need a real external backend,
real training runs, or a shipped UI — not new harness logic.

**Done**

- **Phase 0 — foundations.** The deterministic verifier and the result/diagnostic schema (reward + eval + ceiling); the machine-verifiable **Contract** acceptance spec (required dims + tolerances, volume/mass, feature counts, manifold/validity, named predicates) via `ContractCheck`.
- **Phase 1 — the minimal harness.** Typed ops, kernel regen, plural verification, checkpoint/rollback, an event-sourced op-DAG, and the single-agent plan/apply/observe/replan loop. Plus the `GeometryBackend` seam (dependency-free stub **and** real CadQuery/OCCT backend), the vendor-neutral LLM layer, the CISP stdio server + CLI, and the end-to-end `pipeline.build`. A real **2D constraint solver** (stdlib `ConstraintGraph` rank DOF analysis + optional SolveSpace) with B-rep validity in the default verifier set.
- **Phase 2 — grounding.** The `context/` manager (token-budget assembly + overflow guard) and file-based `StagingArea`; the dependency-free hybrid **RAG** layer (`rag/` — BM25 + hashed-vector, RRF fusion); the four-type `MemoryStore` and a Voyager-style, execution-verified **skill library**.
- **Phase 3 — reliability.** `reliability/strategies/` best-of-N + a Reflexion loop; `reliability/guardrails.GuardrailGate` (`before_tool_callback`), the `ErrorRecovery` ladder, and `reliability/loopdetect.LoopDetector`.
- **Phase 4 — measurement.** **CADBench-Verified** (SWE-bench-style, programmatically-checked: editability, program execution, B-rep validity, dimension match, easy/medium/hard tasks) and the `observe.py` observability layer (spans, KPI metrics with confidence intervals, failure taxonomy, run replay). The plural verifier now also spans an opt-in **DFM critic** (`verifiers/dfm`) and a **VLM-judge** (`verifiers/vision`).
- **Phase 5 — scale.** The multi-agent `Supervisor` + role personas (Designer / Modeler / Verifier / DFMCritic / RedTeam / Reviewer) and the `AsyncOverseer` with halt authority; the `a2a/` inter-agent message bus + task lifecycle; the `surfaces/mcp/` tool server + `CADGymEnv` Gym environment; the `surfaces/ui/` SSE event contract + three-tier approval; and grammar-constrained decoding artefacts (`grammar.py`). The data-engine exporters (`dataengine/` — GRPO / DPO / STaR) and synthetic `datagen/` (solver-in-the-loop) are in place; `exploration/` adds Co-Scientist generate-debate-evolve variant search with Elo-tournament ranking + clustering; and `routing.RoutingLLM` adds cost-aware model routing. `harness.AgentHarness` ties the ReAct loop together and `reliability.executor.ToolExecutor` adds the sandbox / retry / timeout / approval layer.
- **Phase 6 — mechanical depth.** The op vocabulary now spans real machined and assembled geometry (`revolve`, `chamfer`, `hole`, `shell`, `draft`, `loft`, `sweep`, `linear_pattern`, `circular_pattern`, `mirror`, `add_instance`, `mate`, `set_param`), with `query('metrics')` mass properties and STL / IGES export alongside STEP, and `state/opdag.bisect()` for fault localisation. The plural verifier grew an **assembly / mate + residual-DOF** solver, an **interference / clash** detector, a **kinematics** motion validator, plus **standards** (preferred-series), **compliance**, **requirements**, and **reference-match** checks. `quality/` adds **mass / cost / BOM** estimation, a multi-objective **fitness** score, semantic diff, part narration, feature graphs, and nearest-COTS suggestion; `reliability/repair` adds OCCT shape-healing; and MCTS joins best-of-N / Reflexion. The front of the pipeline now runs **spec** (formalize + interview) -> **skeleton** (master-sketch) -> **sizing** (engineering calc), and `ingest/` (STEP import + decompile + metadata) with a parametric parts **library** closes the loop on existing geometry.
- **Phase 7 — corpus-derived engineering depth.** Analytic stress/buckling verification with an external-FEA seam; tool-access, feasibility, completeness and functional checks; traceable conformance reports; assembly sequencing; dimensioned 2D drawings; versioned standards ingestion; embodied-carbon/energy objectives; op-DAG branching and three-way merge; Pareto trade-offs, traceability, batch semantic edits and grounded model Q&A; round-trip ingest fidelity; nearest-known-good recovery; and human-edit preference capture.
- **Phase 8 — feasible corpus gaps.** Deterministic next-operation ranking; cached simulation-job orchestration; graph/history/diagnostic view models; transactional external-adapter contracts; cross-source reconciliation; approval-gated multi-turn edits; keyboard commands; safe attachment conditioning; local data and tool trust policies; consented session capture; provenance-bias auditing; time-to-feasibility percentiles; revision cost/carbon deltas; and evidence-gated research governance.

**Planned / future**

- A **Rust-native kernel** (Fornjot / Truck / Cadmium) dropped in behind the existing `GeometryBackend` seam.
- A **real constrained-decoding backend** (XGrammar / Outlines) wired to decode time — today `grammar.py` produces the schema/GBNF and validates post-hoc, but does not constrain the sampler.
- **Training runs.** The GRPO / DPO / STaR exporters exist; actually fine-tuning a model on the flywheel data is future.
- The **canvas UI** implementation on top of the SSE + approval contract (the contract is done; the front end is not).
- A **real embedder** behind the RAG / memory seams (today's vectors are embedding-free hashed n-grams).
- A **live MCP transport** (FastMCP) and remote A2A (HTTP + SSE / webhooks) — the schemas and value objects are ready; the wire transport is not.
- The **data flywheel at scale** — turning logged trajectories into a large curated training corpus.
- Trained **next-operation, B-rep diffusion, assembly and T-spline models**; the repository provides deterministic baselines and interfaces but does not claim model results.
- Production **federated learning/clean rooms**, native proprietary-host connectors, full FEA meshing/solving and CAM/toolpath generation.

## Design doc

The full thesis, layered architecture, verification strategy, and open decisions are in
[HARNESS_BLUEPRINT.md](HARNESS_BLUEPRINT.md) — the north star this codebase is built
toward.

## License

MIT.
</content>
</invoke>
