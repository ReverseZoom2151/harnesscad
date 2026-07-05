<div align="center">

# HarnessCAD

**A native agentic harness for engineering/mechanical text-to-CAD.**

*The harness — not the model — is the product: verifier-first, frontier-model-native, kernel-agnostic.*

[![tests](https://img.shields.io/badge/tests-92%20passing-brightgreen)](tests/)
[![status](https://img.shields.io/badge/status-phase--0-blue)](HARNESS_BLUEPRINT.md)
[![python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](#license)
[![core deps](https://img.shields.io/badge/core%20runtime-stdlib%20only-success)](pyproject.toml)
[![LiteLLM](https://img.shields.io/badge/LLM-LiteLLM-6f42c1?logo=litellm&logoColor=white)](https://github.com/BerriAI/litellm)
[![CadQuery](https://img.shields.io/badge/kernel-CadQuery-orange)](https://github.com/CadQuery/cadquery)
[![OpenCASCADE](https://img.shields.io/badge/geometry-OpenCASCADE-red)](https://dev.opencascade.org/)

</div>

---

## What it is

HarnessCAD is a from-scratch **agentic harness** for turning a natural-language
engineering brief into precise, editable, *verified* parametric CAD. Its founding
bet, argued in full in [`HARNESS_BLUEPRINT.md`](HARNESS_BLUEPRINT.md), is that in
2026 the frontier model is no longer the bottleneck — **the loop around it is**.
So the product is the harness:

- **Verifier-first.** CAD is a rare *verifiable-reward* domain: geometry compiles
  or it doesn't, constraints solve or they don't, a solid is manifold or it isn't.
  A deterministic verifier is simultaneously the **reward, the eval, and the
  ceiling**. HarnessCAD is built around that verifier, not around model confidence
  — *structured output ≠ correct output*, so nothing is trusted until geometry
  checks pass.
- **Frontier-model-native.** No fine-tuned house model required. The agent seam
  takes any chat LLM (via [LiteLLM](https://github.com/BerriAI/litellm), ~100
  providers) and the harness does the correction, verification and checkpointing
  around it.
- **Kernel-agnostic.** Every layer above the geometry kernel — ops, the ops-DAG,
  verifiers, the loop — is pure Python with **zero required runtime dependencies**.
  A backend turns an op stream into geometry behind one `GeometryBackend` protocol:
  a dependency-free **stub** ships today, a **CadQuery/OCCT** backend produces real
  B-rep solids, and a future Rust kernel (Fornjot / Truck / Cadmium) can be dropped
  in behind the same interface.

## Why now

Both of the reference works this design distils — *The Hitchhiker's Guide to
Agentic AI* and *Agentic Design Patterns* — independently name our exact situation
("a product where the agent harness is a core differentiator") as the explicit
trigger to **build custom** rather than adopt a framework. Frontier models in 2026
can already emit competent CAD operations; what's missing is the disciplined loop
that plans, executes against a real kernel, verifies geometry, and rolls back
cleanly on failure. That loop is the constraint, and it is what HarnessCAD builds.

## How it works

The control loop is small and already proven in coding agents (Aider's
`edit → compile → run tests → commit`). HarnessCAD maps that one-to-one onto CAD:

| Coding agent (solved)      | HarnessCAD                                        |
| -------------------------- | ------------------------------------------------- |
| Edit a file                | **Emit a CISP op** (sketch / constrain / extrude) |
| Compile                    | **Kernel regenerate** the feature tree            |
| Run tests                  | **Verify geometry** (DOF, manifold, watertight)   |
| Git commit on green        | **Checkpoint** the ops-DAG                         |
| Read the traceback, retry  | **Read diagnostics, re-plan** (block-and-correct)  |

Concretely, one turn of the loop is:

```
Contract ─▶ plan ─▶ emit CISP op ─▶ regen ─▶ verify ─▶ checkpoint
                       ▲                          │
                       └──── diagnostics ◀────────┘   (block-and-correct)
```

An op the backend rejects (bad reference, non-positive radius) **never mutates
state**; the batch stops and returns diagnostics. An op that applies but fails the
plural verifier (e.g. an over-constrained sketch) is **rolled back** to the last
good state. Every accepted-and-verified op is checkpointed, giving deterministic
replay and rollback to any point.

## Architecture

```
  natural-language brief
          │
  ┌───────▼────────────────────────────────────────────────────┐
  │  LLM seam            llm/  — provider-neutral (LiteLLM, mock) │
  ├───────▼────────────────────────────────────────────────────┤
  │  Agent / planner     agent/  — NL → validated CISP ops,      │
  │                      plan → apply → observe → re-plan        │
  ├───────▼────────────────────────────────────────────────────┤
  │  Harness loop        loop.py — applyOps → regen → verify →   │
  │                      checkpoint, block-and-correct, rollback │
  ├───────▼────────────────────────────────────────────────────┤
  │  Verifier (plural)   verify.py — sketch DOF · solid presence │
  │                      checks_geometry.py — real B-rep validity │
  ├───────▼────────────────────────────────────────────────────┤
  │  Ops-DAG (state)     state/opdag.py — content-hashed,        │
  │                      append-only "git for CAD"               │
  ├───────▼────────────────────────────────────────────────────┤
  │  GeometryBackend     backends/  — StubBackend (no deps)      │
  │                      · CadQueryBackend (OCCT B-rep)          │
  │                      · future Rust kernel (same protocol)    │
  └───────▲────────────────────────────────────────────────────┘
          │
  CISP server (server.py) + CLI (cli.py) + trace (trace.py)
```

The **CISP** surface (a compact, LSP-inspired protocol —
`initialize / applyOps / query / verify / export`) is the stable JSON boundary the
agent targets, so the geometry kernel underneath is swappable without touching the
agent.

## Quickstart

No dependencies are required for the core spine (Python 3.10+, stdlib only).

```bash
git clone https://github.com/ReverseZoom2151/harnesscad
cd harnesscad
```

### Build a constrained plate, then extrude it

```python
from backends.stub import StubBackend
from cisp.ops import NewSketch, AddRectangle, Constrain, Extrude
from loop import HarnessSession

session = HarnessSession(StubBackend())

result = session.apply_ops([
    NewSketch(plane="XY"),
    AddRectangle(sketch="sk1", x=0, y=0, w=20, h=10),
    Constrain(kind="distance", a="e1", value=20),   # pin the 4 DOF a
    Constrain(kind="distance", a="e1", value=10),   # rectangle contributes
    Constrain(kind="distance", a="e1", value=20),
    Constrain(kind="distance", a="e1", value=10),
    Extrude(sketch="sk1", distance=5),
])

print("ok:      ", result.ok)        # True
print("applied: ", result.applied)   # 7
print("digest:  ", result.digest)    # stable across identical replays
print("summary: ", session.summary())
# {'sketch_count': 1, 'entity_count': 1, 'feature_count': 1, 'solid_present': True}
```

`apply_ops` returns an `ApplyOpsResult` — `ok`, `applied`, the deterministic
`digest`, a list of `diagnostics`, and (on failure) the `rejected` op.

### Block-and-correct in action

Reference a sketch that has no profile and the offending op is rejected without
corrupting state:

```python
from backends.stub import StubBackend
from cisp.ops import NewSketch, Extrude
from loop import HarnessSession

session = HarnessSession(StubBackend())
result = session.apply_ops([NewSketch(plane="XY"), Extrude(sketch="sk1", distance=5)])

print(result.ok)        # False
print(result.rejected)  # {'op': 'extrude', 'sketch': 'sk1', 'distance': 5}
for d in result.diagnostics:
    print(d.severity.value, d.code, d.message)
    # error empty-sketch sketch 'sk1' has no profile
```

### From the command line

The CLI drives a session end-to-end. `demo` runs the built-in constrained-plate
sample; `apply` runs any JSON array of ops. Use `--backend cadquery` for real OCCT
geometry (falls back to the stub with a note if CadQuery isn't installed).

```bash
python cli.py demo
python cli.py apply examples/ops_plate.json
python cli.py demo --backend cadquery
```

### Run the tests

```bash
python -m unittest discover -s tests -t . -v
```

## The CISP op set (v0)

The typed, agent-facing operations the model emits. Every op is a frozen dataclass
with a stable `op` tag, so an op stream is deterministic, hashable and diffable —
the substrate for the ops-DAG. Sketch + constraint ops come first *by design*: the
wedge is sketch/constraint/layout assist, not one-shot solids.

| Op tag          | Class          | Purpose                                            |
| --------------- | -------------- | -------------------------------------------------- |
| `new_sketch`    | `NewSketch`    | Start a sketch on a plane (`XY` / `YZ` / `XZ`)      |
| `add_point`     | `AddPoint`     | Add a point primitive to a sketch                  |
| `add_line`      | `AddLine`      | Add a line segment                                 |
| `add_circle`    | `AddCircle`    | Add a circle (radius must be > 0)                  |
| `add_rectangle` | `AddRectangle` | Add a rectangle (w, h must be > 0)                 |
| `constrain`     | `Constrain`    | Apply a geometric/dimensional constraint           |
| `extrude`       | `Extrude`      | Extrude a sketch profile into a solid              |
| `fillet`        | `Fillet`       | Fillet edges of the current solid                  |
| `boolean`       | `Boolean`      | `union` / `cut` / `intersect` two solids           |

Constraint kinds: `coincident`, `horizontal`, `vertical`, `parallel`,
`perpendicular`, `distance`, `radius`, `equal`. Dimensional kinds (`distance`,
`radius`) require a numeric `value`. A sketch that ends over-constrained is an
**error** (rolled back); under-constrained is a **warning**.

## Layout

```
harnesscad/
├── cisp/                    # CISP typed op set + result protocol
│   ├── ops.py               #   the v0 ops (frozen dataclasses, canonical JSON)
│   └── protocol.py          #   ApplyOpsResult { ok, applied, digest, diagnostics }
├── state/
│   └── opdag.py             # content-hashed append-only ops-DAG ("git for CAD")
├── backends/
│   ├── base.py              # GeometryBackend protocol + ApplyResult
│   ├── stub.py              # dependency-free backend (op semantics, no geometry)
│   └── cadquery_backend.py  # real B-rep solids via CadQuery / OCCT
├── verify.py                # plural verifier: sketch DOF, solid presence
├── checks_geometry.py       # real OCCT B-rep validity check (manifold/watertight)
├── loop.py                  # HarnessSession: applyOps → regen → verify → checkpoint
├── agent/                   # NL → ops planner + plan/apply/observe/re-plan runner
├── llm/                     # provider-neutral LLM seam (LiteLLM backend, structured)
├── server.py                # CISP stdio server (initialize/applyOps/query/verify/export)
├── cli.py                   # command-line driver (demo / apply)
├── trace.py                 # structured observability events off the loop spine
├── examples/                # sample op streams (ops_plate.json)
├── tests/                   # unittest suite (77 tests)
├── HARNESS_BLUEPRINT.md     # the founding design doc
└── pyproject.toml
```

> Research and reference material (PDFs, corpus notes) lives under `resources/`,
> which is **gitignored** and not part of the product.

## Roadmap

HarnessCAD follows the staged plan in the blueprint. Phase 0 is the spine that
exists today; later phases are in progress or planned.

- **Phase 0 — foundations** ✅ — the deterministic verifier and Contract-shaped
  result schema (reward + eval + ceiling), CISP typed ops, event-sourced ops-DAG.
- **Phase 1 — minimal harness** ✅ — the Aider loop: typed ops → kernel regen →
  geometry checks → checkpoint, single agent, block-and-correct, rollback.
- **Real geometry backend** — CadQuery/OCCT backend (present) producing real B-rep
  solids with true manifold/watertight validity; **planned**: a real 2D constraint
  solver (planegcs / SolveSpace) to replace the nominal DOF bookkeeping.
- **LLM planner** — provider-neutral seam + NL→ops planner and correction runner
  (present); **planned**: grammar-constrained decoding and Best-of-N + verifier.
- **Phase 2 — grounding** *(planned)* — context manager, episodic/semantic/
  procedural memory, an execution-verified skill library, structure-aware RAG.
- **Phase 3 — reliability** *(planned)* — Reflexion loop, `before_tool_callback`
  guardrails, the full error-recovery ladder.
- **Phase 4 — measurement** *(planned)* — **CADBench-Verified**: a SWE-bench-style,
  contamination-controlled eval (spec → agent builds part → programmatic checker),
  plus full trajectory logging and replay tooling.
- **Phase 5 — scale** *(planned)* — multi-agent (Verifier / DFM / Red-Team),
  variant exploration, a canvas UI over typed SSE events.
- **Future kernel** *(planned)* — a Rust-native geometry backend behind the same
  `GeometryBackend` protocol.

## Design doc

The authoritative thesis, architecture, decisions and sequencing live in
**[`HARNESS_BLUEPRINT.md`](HARNESS_BLUEPRINT.md)** — the north star this codebase
is built to.

## License

MIT.
