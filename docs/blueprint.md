# harnesscad: Agentic Harness Blueprint

> Founding design doc. Synthesized from a full read of *The Hitchhiker's Guide to
> Agentic AI* (603 pp) and *Agentic Design Patterns* (Gulli, 482 pp), mapped to
> the text-to-CAD domain. This is the north star: **build a native agentic
> harness for text-to-CAD; the harness, not the model, is the product.**

---

## 0. Thesis

1. **The harness is the differentiator.** Both books independently name our exact
   situation ("building a product where the agent harness is a core
   differentiator," "you need maximum performance/control, no framework lock-in,
   non-standard orchestration") as the explicit trigger to **build custom**, not
   adopt LangGraph/CrewAI/AutoGen. We steal their *patterns*, not their libraries.
2. **CAD is a verifiable-reward domain.** Geometry compiles or it doesn't;
   constraints solve or they don't; dimensions/mass/interference either match the
   spec or not. This is the rare, valuable setting where the **deterministic
   verifier is simultaneously the reward, the eval, and the ceiling.** It unlocks
   Best-of-N+verifier, RLVR-style methods, and objective benchmarking. It is our
   single biggest structural advantage.
3. **The winning loop is small and already proven in coding agents (Aider):**
   `Contract → Plan → emit typed CAD op → kernel regen → verify geometry →
   observe → repair → checkpoint`. That is Aider's `edit → run tests → commit`
   with **test = geometry verification, compile = kernel regen, commit = feature-
   tree snapshot.**
4. **Structured output ≠ correct output.** Constrained decoding guarantees a tool
   call *parses*; it never guarantees the geometry is *right*. The real safety net
   is always **external execution + geometry checks**, never model self-confidence.

---

## 1. The reference analogy (drives the whole architecture)

| Coding agent (solved) | CAD agent (what we build) |
|---|---|
| Read source files | Read model tree / feature history / sketches / parameters |
| **Edit file** (action) | **CAD operation**: sketch, extrude, revolve, fillet, boolean, pattern, set-param |
| **Compile** | **Kernel regeneration** of the feature tree |
| Compile error | Regen failure (self-intersection, failed boolean, over/under-constrained sketch) |
| **Run tests** | **Geometry checks**: manifold/watertight, interference, mass/volume/CoG, dims/tolerance, DFM |
| Observe traceback | Read regen errors + geometric measurements |
| **Git commit on success** | **Checkpoint / version the feature tree** |
| Shell tool | CAD-kernel API / headless script execution |
| MCP custom tools | Custom CAD ops, measurement queries, meshing, FEA hooks |
| Sandbox | Isolated kernel workspace so a bad op can't corrupt state |
| Repo mental model | Assembly/part structure, feature dependencies, parametric relations |

---

## 2. Layered architecture

```
┌─────────────────────────────────────────────────────────────┐
│  UI  (canvas/artifact: 3D model as live editable artifact)   │  typed SSE events + 3-tier approval
├─────────────────────────────────────────────────────────────┤
│  Agents  (single agent first; Designer/Verifier/DFM later)   │  A2A message format (internal)
├─────────────────────────────────────────────────────────────┤
│  Harness runtime  (ReAct loop · context mgr · event-sourced  │  ← the from-scratch core
│                    state + checkpointer · loop detector)     │
├─────────────────────────────────────────────────────────────┤
│  Patterns  (plan · tool-use · reflection · best-of-N)        │
├─────────────────────────────────────────────────────────────┤
│  Verification + Contract   (the differentiator)              │  geometry checks, not strings
├─────────────────────────────────────────────────────────────┤
│  Memory + RAG  (skill library · episodic/semantic/procedural)│
├─────────────────────────────────────────────────────────────┤
│  Tool layer  (MCP server: tools=ops, resources=model state)  │  typed, sandboxed, before-tool gate
├─────────────────────────────────────────────────────────────┤
│  CAD environment / kernel  (Gym: reset/step/state/render)    │  code-CAD substrate (PALM)
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Core components (the from-scratch skeleton, ~ port of the book's `AgentHarness`)

- **`Message`**: role (system/user/assistant/tool), content, tool_calls, tool_call_id, metadata. Atomic state unit.
- **`ToolDefinition`** (Pydantic): name (verb-noun), description (what / when-to-use / when-NOT / side effects), typed params, output spec, `requires_approval`. Self-serializes to the model's tool schema. **Descriptions are load-bearing: the model routes almost entirely off name+description (10–20% accuracy swings).**
- **`ContextManager`**: owns the window. Budget `C ≥ S(system)+M(memory/RAG)+T(tools)+H(history)+R(reserved)`. Exact-tokenizer counting (NOT the 4-char rule; CAD emits code/JSON, off 20–40%). Pre-flight check on **every** call (guard silent truncation). Pin system prompt + first user message; evict middle with trailing tool results. Compression: summarize old turns (cheap model), importance-weighted retention.
- **`ToolExecutor`**: sandbox, human-approval gate, retry + exponential backoff, timeout, output truncation (mesh/log dumps must not blow the window), and a **`before_tool_callback` hard validation gate** on every kernel op.
- **`LoopDetector`**: hash (op, sorted-args) over a sliding window; break on repeats. CAD agents oscillate (re-extruding, retrying a failing boolean).
- **`AgentHarness`** (orchestrator) runs the ReAct loop: pre-flight → LLM call (temp 0, tools) → store assistant msg → terminal check (`finish=="stop"` or no tool calls) → **parallel** tool dispatch → loop-detect → append results → repeat to `MAX_ITERATIONS`. Per-run `run_id`, structured logging.
- **Event-sourced `State`/`Session`**: **never mutate state directly; append events.** Yields persistence, timestamps, concurrency safety, and, for CAD, a **replayable operation history = free undo/rollback + audit trail.** Pluggable backend (in-mem → SQLite → cloud). Scope keys: session / `user:` / `app:` / `temp:`. A **checkpointer** persists after each geometry-mutating op → resume-after-failure + human-in-loop pause + time-travel replay.

---

## 4. The control loop

Default: **ReAct + Evaluator-Optimizer.** Plan-and-Execute for well-specified/standard parts; hybrid (plan high-level, ReAct within each step) for assemblies.

```
contract   = formalize(user_brief)                 # machine-verifiable acceptance spec
plan       = decompose(contract)                    # ordered feature DAG (deps: sketch before extrude)
checkpoint = snapshot(empty_model)
for step in plan:                                   # or dynamic, re-plan on failure
    while attempts < MAX_ATTEMPTS:
        op       = llm.emit_typed_op(context, tools)   # Pydantic, grammar-constrained, temp 0
        if not before_tool_gate(op): feedback; continue # block invalid geometry, correct
        result   = kernel.apply(op)                      # regen
        checks   = verify(model, contract, step)         # geometry invariants + predicates
        context.observe(result, checks)
        if checks.pass: checkpoint = snapshot(model); break
        reflection = critic.critique(model, contract, checks)  # separate persona
        context.add(reflection)                          # Reflexion: learn within the run
    else:
        rollback(checkpoint); replan_or_escalate()
return model, trajectory
```

Reliability escalation (spend inference compute, tiered to difficulty): **Best-of-N + verifier** (parallel, highest ROI: `P(success)=1−(1−p)^N`) → **ToT / GoT** (GoT merges sub-solutions; good for decomposable assemblies) → **LATS/MCTS** (10–50×, hardest geometry). Keep tasks in the **20–80% solve band** or there's no signal.

---

## 5. The CAD environment (the "world")

- **Gym interface:** `reset() / step(action)->(obs,reward,done) / state() / render() / close()`. Template directly off the book's `FileEditEnv`: swap "pytest passes" for "geometry verifier passes."
- **Substrate = code-CAD (PALM).** The agent writes and the harness *executes* parametric kernel code: **never let the model do geometry math in its head** (the single biggest reliability lever). Deterministic, inspectable, diffable, version-controllable. Candidate kernels (open decision §18): CadQuery / build123d / OpenSCAD / FreeCAD API / OCCT.
- **CAD-Computer Interface (ACI):** expose a **compact, LLM-optimized op set** (`create_sketch`, `extrude`, `fillet`, `measure`, `run_check`, `export`), **not** the raw scripting API. SWE-agent's key lesson: shrink the action space.
- **Hybrid observation:** geometry/B-rep summary (text/JSON: feature list, validity flags, measurements) **+ a rendered viewport image** so a vision model can see the shape. Never leak the ground-truth answer into the observation.
- **Expose the environment as an MCP server** (FastMCP): **tools = action space**, **resources = observations** (model tree, material/standard-parts libraries), **prompts = op templates**, tool-result carries a reward field, a `reset` tool. One interface serves both serving-time agents *and* future RL training, and makes trajectory logging trivial. Annotate `export`/`delete` destructive, `render`/`measure` read-only.

---

## 6. Verification + the Contract (the differentiator)

- **Formalize the brief into a machine-verifiable acceptance Contract** (the book's "Contractor" model, the highest-leverage idea): required dims + tolerances, mass/volume targets, mating interfaces, material, DFM rules, "manifold & watertight," "no self-intersections." The agent iterates with self-validation until **all** contract checks pass, then returns.
- **Verify geometry invariants, never strings.** (The broken `agent_output == expected` is the cautionary tale.) Use bounding box, volume, surface area, mass properties, face/edge/vertex counts, topological signature, and **mesh distance (Hausdorff/Chamfer)** vs. a reference.
- **Multi-family checks:** geometric validity + spec predicates (`verification_fn`) + export integrity (valid STEP/STL). An **imperfect verifier caps effective N and gets gamed** (reward hacking is *certain*, not hypothetical), so diversify checker families and include adversarial cases.
- **LLM-as-judge only for the subjective slice** (design intent, cleanliness): rubric + low temp + JSON output + **swap-augmentation** (judge A-vs-B and B-vs-A, defeat position bias) + validate the judge (κ>0.6, >80% human agreement) or use **G-Eval** (probability-weighted score). Hard geometry is judged by the kernel, not the LLM.

---

## 7. Context management

- Budget + pre-flight token count (exact tokenizer) on every call.
- **Lost-in-the-middle:** put the spec + active constraints + current task at the **head or tail**, never buried mid-history.
- **Prefix caching:** keep a stable system-prompt + tool-definitions prefix leading every request → 60–80% lower TTFT, big cost win.
- Represent the model as a **feature-tree summary, not a full B-rep dump**; compact tool outputs.
- **Context Staging Area (no black-box RAG):** a per-task `task-context/` (`01_BRIEF.md` intent+constraints, `02_MODEL/` current tree as text, `03_DOCS/` specs/standards/DFM) driven by a `context.toml` manifest: **explicit, transparent control over what the model sees each turn.**

---

## 8. Memory

- **Working** (event-sourced state/scratchpad) · **Episodic** (past successful models keyed by NL description, retrieved by similarity to seed generation) · **Semantic** (materials, standards, user preferences) · **Procedural** (generation rules / system prompt, rewritten by a reflection node after failures).
- **Voyager skill library:** a growing, **execution-verified** library of reusable parametric CAD routines (`make_flange(bolt_circle, n_holes)`), embedding-routed, hierarchically composable. **Improves the harness monotonically with zero training.**
- **Read-Act-Reflect-Write / Reflexion:** on failed verification, retrieve prior failures, synthesize an insight ("this boolean fails when faces are coplanar → add offset"), store to semantic memory, recall next attempt.

---

## 9. Tools / MCP

- **Typed (Pydantic) op schema; LLM emits JSON, harness executes against the kernel** ("parse, don't validate" at the boundary). Use **grammar-constrained decoding** (XGrammar/Outlines) so op emission is *syntactically* guaranteed: a CFG for the CAD command language.
- 5-component tool descriptions; enum-heavy, flat schemas; temp 0 for op emission.
- Tools **raise typed errors** (radius-too-large, non-manifold result) for the agent to observe and repair; measurement tools return **clean structured data** for the critic to reason on.

---

## 10. Guardrails & error recovery

- **`before_tool_callback` hard gate** on every kernel op: positive extrude depth, fillet radius < adjacent edge length, boolean won't null the body, dims within manufacturable limits, hole stays in material → **block-and-correct**, don't corrupt the model.
- **Error-recovery ladder:** *detect* (regen fail, non-manifold, boolean fail, over/under-constrained, timeout, empty) → *handle* (log; retry with adjusted params, never the same invalid op unchanged; fallback to a simpler modeling strategy; graceful degradation = deliver valid partial + report failed feature) → *recover* (roll back the feature tree via the event log; reflect/diagnose; replan; escalate).
- **Checkpoint-on-success / rollback-on-failure = transactional CAD.**
- **Least privilege** per tool/agent; **sandboxed** kernel execution; treat all tool/retrieved output as untrusted (indirect-injection defense).

---

## 11. Reasoning

- **CoT** to decompose a brief into an ordered feature tree before emitting code.
- **PALM (generate-and-execute code):** the reliability workhorse for every numeric/geometric computation.
- **ReAct** as the core operational loop; self-correct on kernel errors.
- **ToT / self-consistency**, budget-gated, for the genuinely hard geometry minority (Scaling Inference Law: more thinking on the hard 10%, single-pass on the easy 90%).
- **Classify-then-route** front door for cost (below).

---

## 12. Multi-agent (only when earned: start single-agent)

When justified, supervisor topology mirroring a software/eng team:
**Designer** (spec→plan) → **Modeler** (emit ops) → **Verifier** (geometry/constraint checks) → **DFM Critic** (wall thickness, draft, tool-access, min radii) → **Reviewer** (two-phase critique→reflection, self-prioritizes findings), iterate via a LoopAgent-style escalate-to-stop. Add a **Red Team** agent (hunts non-manufacturable geometry/interference), a **safety monitor**, and an **async overseer** (watches the event stream for loops/stagnation, authority to halt).
- **Design-space exploration:** Co-Scientist **generate → debate → evolve** with **Elo-tournament ranking** of variants (clean when there's no single scalar objective); cluster to avoid redundant search.
- **A2A** message vocabulary (Agent Cards, task lifecycle submitted→working→completed, `contextId`) as the internal format (even in-process) so long geometry/meshing/FEA solves use **async tasks + SSE streaming/webhooks** and remote compute is a drop-in later. **A2A between agents, MCP for each agent's tools** (trust-boundary separation).

---

## 13. Cost control (first-class subsystem)

Classify-then-route (cheap model for param edits / boilerplate / unit conversion; expensive reasoning model for spatial planning / constraint solving) · sequential model fallback (OpenRouter-style) · context pruning/summarization of the op history · cheap Flash-tier pre/post screens · prefix caching · adaptive tool selection (cheap analytic checks before expensive mesh/FEA) · optional W4A16 quantization + n-gram/EAGLE speculative decoding if self-serving.

---

## 14. UI contract (define now, even before a UI)

- **Typed SSE event protocol:** `status | thinking | token | tool_call | tool_result | approval_required | action_rejected | done`.
- **Canvas/artifact paradigm:** the 3D model / CAD code is the persistent, editable artifact with version history (perfect fit; auto-elevate to canvas).
- **Three-tier approval:** Tier-1 auto (read/measure/render), Tier-2 notify (modify), Tier-3 require (export/delete/irreversible), with a **risk indicator** and a **dry-run preview** of predicted geometry changes. MCP annotations auto-assign tiers. Batch related approvals; beware alert fatigue.
- Tool-use cards with **before/after geometry diffs**, a token-budget indicator, always-visible Stop, and undo/rollback.

---

## 15. Observability

Trace / log / metrics triad (spans per LLM call, tool op, state transition, with tokens/cost/latency) · **full trajectory logging** (op + regen result + measurement) · **replay tooling** (CAD failures are semantic, not syntactic; you must replay) · failure taxonomy (regen / reasoning / hallucination / loop / context-overflow / refusal) each with distinct remediation. Targets: **TSR > 85%, tool-call accuracy > 90%, recovery > 60%, escalation < 15%.**

---

## 16. Evaluation: "CADBench-Verified" (a.k.a. "CASP for CAD")

SWE-bench-style: **spec → agent builds part → programmatic checker verifies.** Private +
contamination-controlled; curated for **solvability & unambiguity**; prioritize the
**30–70% solve band.** Two tiers: **test-files** (single-part unit tests) + **evalsets**
(assembly/integration), CI + containerized. Report **task-success-rate + trajectory
efficiency** (`η = L*/L_agent`) per-difficulty with CIs; expert human baselines.

**Concrete metric set (from the corpus AlphaCAD spec; adopt directly). Editability and
validity rank ABOVE fidelity:**
1. **Sketch editability:** % fully-constrained sketches, zero over-constraints.
2. **Program execution:** % feature sequences that rebuild without kernel errors.
3. **B-rep validity:** watertight / manifold / no self-intersection; topology accuracy.
4. **Assembly mates:** mate-type accuracy + **residual DOF** + **collision rate**.

Framing = **"CASP for CAD"**: public leaderboard + a **shippable eval kit** (converters +
validators) so partners test locally without sharing raw geometry. The corpus is emphatic
that **no unified CAD+AI benchmark exists** ("CAD has no GLUE/ImageNet"), so publishing
CADBench early is a **credibility + positioning moat**, not just internal QA. Seed tasks
from SketchGraphs / Fusion 360 Gallery / AutoMate / ABC (see §21).

---

## 17. Training later (optional, gated)

Log every trajectory as `(S_t, A_t=[reasoning, tool_call], R_t, S_{t+1})` with the verifier's scalar reward + **sub-goal labels**. This one format supports **GRPO** (group-normalize N verified trajectories; the agentic default, no critic), **DPO** (chosen=best trace / rejected=worst), and **STaR/RFT** (SFT on verified successes). Use **dense/hierarchical credit** (reward per verifiable sub-goal; on failure, **trajectory-slice** to the first divergence step: 3–5× signal density). **Gate on pass@k:** if the frontier model *never* one-shots a CAD task type, fix tools/context/prompt first. "RL cannot introduce capabilities absent from the base model."

---

## 18. Decisions

- **Wedge (DECIDED): engineering / mechanical CAD.** Mechanical is the cleanest
  verifier and best-tooled loop. BIM/AEC is a *separate, later* native harness
  (architecture is a genuinely different problem: different kernel semantics,
  representations, and acceptance criteria).
- **Build posture (DECIDED): from first principles**, harvesting the best parts
  from the downloaded reference repos to build new infrastructure.
- **STILL OPEN: how deep does "first principles" go? The one scoping decision
  that determines months vs. years.** The mechanical-CAD stack has layers:
  1. **B-rep geometry kernel** (solid-modeling math): OCCT (open) / Parasolid /
     ACIS (commercial). Building this from scratch is a multi-year, decades-of-
     prior-art effort, almost certainly NOT the wedge.
  2. **2D constraint solver** (sketch geometric-constraint solving): planegcs
     (FreeCAD), SolveSpace's solver. Building new is hard-but-tractable and is
     arguably defensible IP.
  3. **Feature / parametric layer** (feature tree, parametric regeneration): build
     new. High leverage; agent-friendliness lives here.
  4. **Agent-CAD interface / DSL / representation (ACI)**: build new. The novel
     differentiator, a compact, LLM-friendly, verifiable CAD command language.
  5. **The harness** (this doc): build new.

  **Recommendation:** "first principles" = layers 3–5 (and possibly 2), built new
  **on top of a proven B-rep kernel (OCCT)**. Do NOT rebuild the geometry math.
  This keeps us fast while still building something that doesn't exist (no agentic
  harness + no agent-native CAD representation exists at scale). Only rebuild the
  kernel if the moat thesis is explicitly "own the kernel," which is a separate,
  years-long bet.

- **Kernel (RESOLVE): OCCT now, behind a `GeometryBackend` trait/interface** so a
  Rust-native kernel can be swapped later. Live Rust candidates surfaced in the
  corpus: **Fornjot** (explicitly mechanical, code-first; closest fit), **Truck**
  (NURBS B-rep), **Cadmium** (append-only event log → unlimited undo + git-style
  branching + physics constraint solving: literally our checkpoint + constraint
  layers, already built). Constraint solver: harvest **planegcs / SolveSpace / D-Cubed-
  style** rather than build from scratch initially.

- **Representation (DECIDED): feature-tree-as-truth.** The corpus is unanimous:
  **feature tree > B-rep > mesh/STL.** The feature tree (sketch ops + constraints +
  features) encodes *intent* and is the editable artifact; B-rep is the derived end
  product; mesh/STL is a dead end for engineering. Keep the tree, not the solid, as
  the canonical state.

- **Agent-facing IR / DSL: THE key open decision.** Three candidates:
  1. **Reuse CadQuery as the op language** (proven in your SpatialHero; fast start).
  2. **A custom KCL-like DSL** on OCCT (Zoo/Onshape-FeatureScript precedent; part of
     the moat, but slower).
  3. **The typed-ops "CISP" API you already spec'd** (Part 2, pp.118–128): ~20
     JSON-schema ops (`create_profile`, `extrude`, `boolean_cut`, `constraint_eq`,
     `export_step`…) + an **ops-DAG = git-for-CAD** (idempotent, diffable, bisect,
     **deterministic replay ≥99% hash-match**) + an **LSP-inspired CISP protocol**
     (`initialize / applyOps / query / verify / export`). This IS the ACI the
     blueprint calls for, already designed by you.
     → **Lean: start by emitting CadQuery (Path 1) behind the CISP typed-ops surface
     (Path 3), so the agent targets a stable JSON op API while CadQuery/OCCT executes
     underneath. CISP becomes the DSL boundary, CadQuery the first backend.**

- **Sequencing, REVISED per competitor diligence (Kinth memo):** senior mechanical
  engineers say end-to-end **text→finished solid is the wrong first target**; the
  defensible wedge is **constraint / sizing / layout / master-sketch assistance for the
  rough first 80%**, with **editable parametric output + STEP interop (sit alongside
  SolidWorks)**. → **Phase 1 = sketch + constraint assist, not one-shot part gen.**

- **Buy-vs-build the model (de-risk):** validate the harness against **Spectral Labs
  SGS-1** (an existing STEP/B-rep foundation model, API-only) or a frontier model
  before training our own. Harness-over-frontier-model remains the default.

---

## 19. Staged build roadmap

- **Phase 0 (foundations):** pick substrate (§18); build the **deterministic verifier** and the **Contract schema** first (they are reward + eval + ceiling).
- **Phase 1 (minimal harness):** the Aider loop of ReAct + typed ops + kernel regen + geometry checks + checkpoint. Single agent. Event-sourced state.
- **Phase 2 (grounding):** context manager + memory (episodic/semantic/procedural) + **skill library** + hybrid RAG (BM25+dense, structure-aware chunking of standards/API docs).
- **Phase 3 (reliability):** Best-of-N + verifier, Reflexion loop, `before_tool_callback` guardrails, error-recovery ladder.
- **Phase 4 (measurement):** CADBench-Verified + full observability/replay.
- **Phase 5 (scale):** multi-agent (Verifier/DFM/Red-Team), variant exploration (Co-Scientist + Elo), the canvas UI.
- **Phase 6 (optional) learn:** fine-tune on logged traces once pass@k justifies it.

---

## 20. Principles (the through-line across both books)

**Verifier-first.** · **Simplicity scales; earn complexity** (workflows before agents, single before multi). · **Prompts are code** (versioned, tested, modular). · **Tools are actuators** (typed, sandboxed, defensively handled). · **State is first-class** (event-sourced, checkpointed). · **Context is finite and precious** (budget it explicitly). · **Structured output ≠ correct output** (external geometry verification is the safety net). · **Errors are inevitable** (graceful recovery is a feature). · **Observability is not optional.**

---

## 21. Grounding in prior work + the mechanical corpus (harvest)

This section folds in a full read of the user's own material (AlphaCAD & SpatialHero
decks; the AI-CAD research papers; the Gaudi/Kinth/Scale-AI docs; and the `docs/skills/`
agentic-pipeline designs) and the two vision decks. **Headline: much of this is already
designed; we integrate and rebuild, we don't start cold.**

### Prior IP to reuse (yours)
- **SpatialHero (MIT Media Lab, Aug 2025)** already built the verification loop:
  CadQuery codegen → render **isometric + orthographic (under/top/front/back)** →
  **VLM-as-judge (GPT-4V) reward 0…1** → PPO. This *is* the multi-view-observation +
  verifier loop. Reuse the render→judge→reward harness wholesale. (The one change since:
  it fine-tuned a 6.7B model; in 2026 we run harness-over-frontier-model instead.)
- **AlphaCAD deck (Augmentation Lab, summer 2025)**: physics-aware rollback (=checkpoint/
  rollback), stability scoring + violations (=verifier), multi-variant + consensus/CLIP
  voting (=Best-of-N + verifier), explainability overlays (=observability), provider
  switch (=model routing). Harness patterns to port (drop the LEGO/brick backend).
- **The corpus "AlphaCAD" spec** (Research Paper Part 2, pp.73–87): an AlphaFold-inspired
  editable-CAD architecture: program/relational/geometric spaces, Evoformer-CAD, Program
  Head, Geometry-Diffusion Head, Assembly Head, **solvers/validators-in-the-loop with a
  recycling loop** (diagnostics fed back K passes). Full pseudocode exists. This is our
  op→regen→verify→checkpoint loop expressed as a model architecture. Reuse it as the
  reference design for the verifier+recycling core.
- **`docs/skills/` = a ready multi-agent pipeline** (Sibyl 17-stage / ResearchClaw
  8-phase-23-stage-3-gate / NanoResearch two-loop; ~55 documented skills with role/tier/
  I/O/tools). Reuse directly as the harness **orchestrator + skill library + multi-agent
  debate/critique/synthesis + gated verifier panel + workspace schema** (canonical JSON +
  narrative logs; 3 model tiers heavy/standard/light; "never-halt, route-around-errors").
  The `standalone/` copies are paste-ready skill prompts: the fastest path to our skill
  library. NOTE: written for *paper production*; re-point the experiment/execution half
  from "train ML on GPUs" to "generate + solver-verify CAD geometry."

### The verifier is PLURAL, with recycling
Not one check but three independent `verify(ruleset)` stages whose diagnostics feed back
into regeneration (not just an end gate):
1. **Constraint solver** (sketch): DOF, over/under-constraint, residuals.
2. **B-rep validator** (topology): watertight / manifold / self-intersection.
3. **Assembly solver**: mate type, residual DOF, collision/interference.
Plus higher-level design-lints (slivers, near-coincident edges) and **tests-as-specs**
(`part.spec.yaml`, "min wall ≥ 2mm") run on save / before "merge." "Solver diagnostics as
signal, not post-hoc filter" is the central idea across every source.

### Data engine (first-class layer, the #1 risk, unanimous)
**There is no "GitHub for CAD"**: no open corpus, and public CAD lacks the *workflow/
process* data (the manufacturing/airflow tweaks). Plan for this from day one:
- **Bootstrap:** synthetic parametric generation + **solver-in-the-loop for ground truth**
  (precedent: a topology-opt team trained on **122k MIT designs, RL + custom verifiers,
  ~20h**). **Verifiers-as-cheap-labor**: decompose expert labeling into binary non-expert
  checks ("is this wall human-scale?"); CadQuery human-in-the-loop scripting as the "Scale
  AI playbook."
- **Seed datasets:** **SketchGraphs** (~15M Onshape parametric sketches w/ constraints),
  **Fusion 360 Gallery** (~8.6k design programs + assemblies/joints), **AutoMate** (~452k
  parts, ~255k assemblies, **1.29M mates**), **ABC** (~1M B-rep). Baselines: DeepCAD,
  SolidGen, BrepGen, VQ-CAD, CAD-Coder.
- **The harness IS the data flywheel:** every session logs `(prompt → plan → ops →
  geometry → tests)`: "the most valuable dataset in CAD: human design intent mapped to
  geometry edits." Track **"human corrections per plan"** as the flagship flywheel metric
  (it should fall over time). Named risk: **synthetic-vs-real distribution gap**. CADBench
  must cover real-part distributions, not just validity pass-rate.

### Competitor map (this space is now contested; differentiate deliberately)
- **Kinth (kinth.ai)**, the closest analog: NL → editable parametric, STEP I/O, concurrent
  design/sim/error-check agents, constraint-solver "10× DOF", Model Cards, top-down
  skeleton CAD; student-first GTM; ~$1.2M raise, 8 LOIs.
- **Zoo (KittyCAD):** ~$25M A; own **KCL** CAD language + GPU kernel (~4 yrs to build).
- **Spectral Labs (SGS-1):** native generative **B-rep foundation model** (STEP-trained),
  API-only, "needs a Cursor-like IDE for distribution" → potential model to plug into our
  harness rather than train first.
- **AdamCAD** (YC W25), **Cosmon** (CAE agent automation), **Autodesk "Bernini"** (internal).
- **Our differentiation:** harness-native + verifier-first + the plural-verifier recycling
  loop + editability/interop-first, on a published CADBench others don't have. Note the
  **Autodesk API ToS (Jul 2025) restricts to internal business use** → reinforces the
  native, kernel-owned (non-plugin) choice.

### GTM / scoping signal (from diligence)
Interop-first (STEP, sit *alongside* SolidWorks, don't rip-and-replace); pick a **vertical
ICP** (configurable-product / engineering-service firms) over "new Autodesk"; student/
university-first adoption; on-prem/locked-down data story for enterprise; ~$0.15/generation
as the unit-economics metric. Market: mechanical/industrial ≈ **$33B software** slice of a
$3.5tn design market; 3D-software TAM ≈ $416B.
</content>
</invoke>
