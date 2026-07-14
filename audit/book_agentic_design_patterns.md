# HarnessCAD audited against *Agentic Design Patterns* (Gulli, 2025)

Source read in full: `resources/extracted_text/Agentic_Design_Patterns.txt` (17,696 lines,
482 pages).

## Chapter count, stated

The book's own table of contents (lines 7-63) enumerates:

* **1 front-matter conceptual chapter** — "What makes an AI system an Agent?" (agent levels 0-3)
* **21 numbered pattern chapters** — the book says so explicitly twice: *"This book extracts 21
  key design patterns"* (line 383) and *"Across 21 dedicated chapters"* (line 396).
* **7 appendices, A through G** — Advanced Prompting; GUI-to-real-world interaction; Agentic
  Frameworks; AgentSpace; AI Agents on the CLI; Under the Hood (reasoning engines); Coding Agents.
  **Appendix F is listed in the TOC (line 59) but is absent from the extracted text** — the file
  jumps from Appendix E (line 15089) to Appendix G (line 15266), and Appendix G is duplicated
  (lines 15266 and 15483). Its row below is marked accordingly.
* Conclusion, Glossary, Index of Terms, FAQ.

**Total rows in the matrix below: 74.** Every numbered chapter gets a row; every sub-pattern the
book gives its own name and definition to (GraphRAG, Agentic RAG, ToT, PAL, RLVR, ReAct, CoD, GoD,
MASS, Scaling Inference Law, the Contractor model, LLM-as-a-Judge, agent trajectories, human-on-the-
loop, the 20 Appendix-A techniques, and so on) gets its own row, because those are the rows where
the evidence differs. No chapter is skipped.

---

## Coverage matrix

| # | Pattern / section (the book's own name) | Verdict | Where in HarnessCAD (file:line) | Gap | Fix (concrete) | Priority |
|---|---|---|---|---|---|---|
| 0 | Agent Levels 0-3 (Intro, "What makes an AI system an Agent?") | PRESENT | `src/harnesscad/core/harness.py:91` `AgentHarness`; `src/harnesscad/core/pipeline.py:126` `build` | We are a Level-2 strategic problem-solver (tools + context engineering + self-improvement). Level 3 (multi-agent) exists but is orphaned. | See row 7. | Low |
| 1 | Prompt Chaining (Pipeline pattern) | PARTIAL | `src/harnesscad/core/pipeline.py:126` `build`: brief -> Planner -> HarnessSession -> STEP | The chain is a single LLM hop wrapped in a retry loop. The book's own worked example (line 950-967) chains *code generation*: outline -> draft -> static-analysis -> refine -> tests. We do outline+draft in one prompt. No decomposition of the brief before the op emission. | Add a `decompose` stage in `pipeline.build`: brief -> feature list (LLM) -> per-feature op batch, each verified before the next. `agents/generation/design_plan.py` already emits the plan structure and is unwired. | Medium |
| 1a | Structured Output (Ch.1, "The Role of Structured Output", line 803) | PRESENT | `src/harnesscad/core/grammar.py:169` `op_json_schema`; `:255` `op_grammar` (GBNF); `:315` `GrammarConstraint`; `src/harnesscad/agents/llm/structured.py` `ParsedOps` | None. This is stronger than the book: the schema is *derived from the op registry* so it cannot drift. | — | — |
| 1b | Context Engineering (Ch.1, line 1073; Appendix A, line 13536) | PRESENT | `src/harnesscad/agents/context/manager.py:168` `ContextManager`; `:79` `BudgetReport` (hard budget C >= S+M+T+H+R); `src/harnesscad/agents/context/staging.py`, `progressive_tiers.py` | Pre-flight is optional in `harness.py:284` `_preflight` and silently no-ops if absent. | Make `ContextManager` a required collaborator of `AgentHarness`, not an optional one. | Low |
| 2 | Routing | PRESENT | `src/harnesscad/core/routing.py:305` `RoutingLLM`; `:129` `HeuristicClassifier`; `:233` `CostTable`; `:300` `AllRoutesFailed` sequential fallback | Rule-based routing only. The book names four mechanisms (line 1233-1264): LLM-based, embedding-based, rule-based, ML-model. We ship one. | Add an LLM-based classifier route behind the existing `Classifier` protocol (`routing.py:52`) — the seam is already there. | Low |
| 3 | Parallelization | **ABSENT** | Nothing. `grep -rl "concurrent.futures\|ThreadPoolExecutor\|asyncio\|multiprocessing" src/harnesscad` returns **zero files**. | Total. The book's use case 5 is literally *"Validation and Verification: performing multiple independent checks concurrently"* (line 1878-1888) — which is exactly what the 23-verifier fleet is, and we run it strictly serially (`eval/verifiers/registry.py` `run_all`). Best-of-N (`eval/reliability/strategies/best_of_n.py`) is inherently parallel and runs sequentially. The 6-engine differential oracle (`eval/selftest/differential.py:310`) runs 6 kernels one after another. | Wrap `registry.run_all` and `differential.run` in a `ThreadPoolExecutor`. Verifiers are pure readers of `ModelState` — this is safe today and is the single cheapest wall-clock win in the repo. | **High** |
| 4 | Reflection (Generator-Critic / Producer-Reviewer) | PARTIAL | `src/harnesscad/eval/reliability/strategies/reflexion.py:139` `ReflexionLoop` (READ-ACT-REFLECT-WRITE); `src/harnesscad/agents/agents/roles.py:187` `Verifier`, `:211` `DFMOutcome` critic | **`reflexion.py` is an ORPHAN** (`registry.orphans()`). The live loop (`core/loop.py`, `agents/agent/runner.py`) has no reflect step at all: it feeds diagnostics straight back to the producer. The book (line 2427-2449) is explicit that the *separated* Critic is what buys objectivity, and our only shipped critic is the verifier fleet — which is not an LLM critic and, as measured, is not calibrated. | Wire `ReflexionLoop` into `AgentHarness` as an escalation tier after N failed iterations. It is written, tested and unreachable. | **High** |
| 5 | Tool Use (Function Calling) | PRESENT | `src/harnesscad/io/surfaces/mcp/tools.py` `ToolCatalog`/`ToolDefinition`; `src/harnesscad/eval/reliability/executor.py:101` `ToolExecutor`; `src/harnesscad/agents/agent/tool_schema.py` | None material. Tool descriptions are generated from the op registry. | — | — |
| 6 | Planning | PARTIAL | `src/harnesscad/agents/agent/planner.py:66` `Planner`; `agents/agent/plan_envelope.py`; `agents/generation/design_plan.py` | The Planner emits *ops*, not a *plan*. There is no plan artifact that survives an iteration, no re-planning against a plan, and no plan approval step (Ch.6, line 3933: DeepResearch presents the plan to the user before executing). `design_plan.py` and `plan_envelope.py` exist and neither is on the live path. | Emit a `PlanEnvelope` first (ops + rationale + expected contract deltas), check it against `core/contract.py:104` `Contract` *before* touching the kernel, then execute. | Medium |
| 7 | Multi-Agent Collaboration | PARTIAL | `src/harnesscad/agents/agents/supervisor.py:68` `Supervisor` (Designer -> Modeler -> Verifier -> DFMCritic -> RedTeam -> Reviewer); `agents/agents/roles.py:97`; `agents/agents/blackboard.py:87` `DesignBlackboard`; `agents/agents/overseer.py:40` `AsyncOverseer` | **`supervisor.py` is an ORPHAN.** The entire multi-agent tier is built, tested and unreachable from `core/cli.py` and `core/harness.py`. Nothing in the shipped path is multi-agent. | Add `harnesscad agent --supervisor` to the CLI and make `AgentHarness` able to take a `Supervisor` in place of a `Planner` (same `plan_parsed` seam). | Medium |
| 7a | Hierarchical / Supervisor / Network / Custom topologies (Ch.7, line 4368-4419) | PARTIAL | `supervisor.py:68` is the Supervisor topology; `blackboard.py:87` is the Network topology | Only two of six named topologies; both orphaned. No topology *search* (see MASS, row 17i). | Same as row 7. | Low |
| 7b | Agent-as-a-Tool (Ch.7, line 4800) | ABSENT | — | The book's `AgentTool` wrapper (one agent invoked as a tool by another) has no analogue. Our `ToolCatalog` holds only CISP ops. | Register `Supervisor` roles as `ToolDefinition`s in `io/surfaces/mcp/tools.py`. Low value until row 7 lands. | Low |
| 8 | Memory Management (short-term / long-term) | PRESENT | `src/harnesscad/agents/memory/store.py:114` `MemoryStore` (working/episodic/semantic/procedural); `agents/memory/skills.py:80` `SkillLibrary` (Voyager, execution-verified); `agents/memory/error_notebook.py:131` `ErrorNotebook`; `agents/memory/decay.py`; short-term = `core/state/opdag.py:54` `OpDAG` | The memory store is not consulted by `agents/agent/planner.py` on the live path — the planner builds messages from brief + state + diagnostics only (`planner.py:90-107`). Memory exists; the agent does not remember. | In `Planner.build_messages`, prepend `MemoryStore.recall(brief)` results. One call site. | **High** |
| 8a | Semantic / Episodic / Procedural memory (Ch.8, line 5610-5631) | PRESENT | `memory/store.py:82` `Episode`; `memory/skills.py:40` `Skill` (procedural); `memory/store.py:114` semantic tier | Same wiring gap as row 8. | — | — |
| 9 | Learning and Adaptation | PARTIAL | `src/harnesscad/agents/exploration/evolution.py`, `evolution_strategy.py` (AlphaEvolve-shaped); `agents/memory/skills.py:80` (monotonic skill growth = SICA-shaped); `data/dataengine/` (95 modules, training-data synthesis) | No online learning on the live path. The skill library grows only if someone calls it. No RLVR/DPO training loop despite CAD being a *perfect* verifiable-reward domain (the book calls this out at line 10261: RLVR trains on "problems with known correct answers (like math or code)"). `agents/agent/tool_reward.py:70` `ToolUseReward` computes the reward and nothing consumes it. | Ship a `harnesscad train` that runs the verifier fleet as the reward signal over the brief corpus and emits DPO pairs. The reward function, the corpus and the verifier are all already written. | Medium |
| 9a | Self-Improving Coding Agent (SICA) (Ch.9, line 5997) | ABSENT | — | We do not modify our own source. Justified: an agent that rewrites `eval/verifiers/` while the verifiers are the trust root is a soundness catastrophe, and the pressure test already shows what one bad rule costs. **N/A would be dishonest — the pattern is applicable to the tool layer** (a self-improving *op-macro* library is exactly `skills.py`), we just haven't closed it. | Let `SkillLibrary` propose new skills from successful trajectories, admit on verification. `skills.py:250` `build_default_library` is 90% of it. | Low |
| 9b | AlphaEvolve / OpenEvolve (Ch.9, line 6109) | PARTIAL | `agents/exploration/evolution.py`; `agents/exploration/elo.py`; `agents/exploration/tournament.py` | Present as modules, absent as a shipped command. | Expose `harnesscad explore --evolve`. | Low |
| 10 | Model Context Protocol (MCP) | PRESENT | `src/harnesscad/io/surfaces/mcp/jsonrpc.py` (JSON-RPC 2.0 over stdio); `mcp/tools.py` `ToolCatalog`; `mcp/annotations.py:39` (readOnlyHint/destructiveHint); `mcp/gym.py:118`; tested in `tests/core/cisp/test_mcp.py` | No remote transport (SSE/HTTP), stdio only. The book flags the real risk (line 6306-6330): *"there is a risk that developers simply wrap pre-existing, legacy APIs without modification"* — our tools ARE the CISP ops, which is the agent-friendly design the book asks for. | Add the HTTP/SSE transport when a remote consumer exists. Not urgent. | Low |
| 11 | Goal Setting and Monitoring | PRESENT | `src/harnesscad/core/contract.py:104` `Contract` (dims+tolerances, mass, feature counts, manifold/watertight, named predicates); `:186` `ContractCheck`; `:378` `contract_from_brief_schema`; monitoring: `core/harness.py:211` per-iteration contract check | The `Contract` is optional (`harness.py:132` `contract: Optional[Contract] = None`), and `core/pipeline.py:126` `build` — the entry point everything actually uses — never constructs one. The harness ships with its goal-checking turned off by default. | Have `pipeline.build` call `contract_from_brief_schema` and ask the LLM for a contract *before* planning ops. This is the single highest-leverage unwired thing in the repo. | **High** |
| 12 | Exception Handling and Recovery | PRESENT | `src/harnesscad/eval/reliability/guardrails.py:49` `GuardrailGate` (pre-apply), `:198` `ErrorRecovery` (staged ladder); `core/loop.py:120` transactional rollback; `core/state/opdag.py:54` checkpoint/rollback; `eval/reliability/repair_loop.py`; `eval/reliability/fallback.py:84` `RetrievalFallback` | **`fallback.py` is an ORPHAN.** The book's graceful-degradation tier (line 7462, 9773) is written and unreachable: on total failure we return nothing, not the closest known-good precedent. | Wire `RetrievalFallback` as the terminal branch of `AgentHarness.run` when `stop_reason in {"max_iterations","loop"}`. | Medium |
| 12a | State Rollback (Ch.12, line 7500) | PRESENT | `core/loop.py:184` `_rollback_last`; `core/state/opdag.py:54` (content-hashed, branch/merge/bisect) | None. Stronger than the book — we have git-for-CAD semantics, the book has a paragraph. | — | — |
| 13 | Human-in-the-Loop | PARTIAL | `src/harnesscad/io/surfaces/ui/approval.py:35` `ApprovalTier`, `:269` `ApprovalGate`, `:176` `DryRunPreview`; `eval/reliability/executor.py:113` `approval` hook, `:156` gate | The approval gate only fires on Tier-3 ops (export/delete). There is **no human checkpoint on the plan** (Ch.6/13: approve the plan before execution), and no escalation path when the loop fails — `AgentHarness.run` just returns `ok=False`. The book (line 7755) names *escalation policies* as a first-class HITL aspect; ours escalate to nobody. | Add a `hitl` callback to `AgentHarness` invoked on `stop_reason != "converged"`, and on plan emission when a `Contract` predicate is unsatisfiable. | Medium |
| 13a | Human-on-the-Loop (Ch.13, line 7831) | PRESENT | `agents/agents/overseer.py:40` `AsyncOverseer` (watches the trace stream, authority to halt); `governance/security/policy.py:26` `DataPolicy` (policy set by human, enforced by machine) | Overseer is instantiable but not attached by `AgentHarness` by default. | `AgentHarness(tracer=overseer)` in `pipeline.build`. One line. | Low |
| 14 | Knowledge Retrieval (RAG) | PRESENT | `src/harnesscad/agents/rag/retriever.py:54` `HybridRetriever` (BM25 + dense fusion — the book's own recommendation, line 8142); `agents/rag/index.py`, `chunk.py`, `rerank.py`, `exemplar_select.py` | Same wiring gap as memory: the Planner does not retrieve. `api_knowledge.py` and `exemplar_select.py` exist to put worked CISP examples in the prompt and nothing calls them. | Same fix as row 8 — one call in `Planner.build_messages`. | **High** |
| 14a | GraphRAG (Ch.14, line 8201) | ABSENT | — | Not N/A: a CAD assembly *is* a knowledge graph (`eval/quality/graph/feature_graph.py`, `attributed_brep_graph.py`, `intent_graph.py` all exist), and cross-part queries ("which fastener mates this flange") are exactly the multi-hop case the book says vector RAG fails at. | Back `HybridRetriever` with `eval/quality/graph/intent_graph.py` for assembly-scoped queries. | Low |
| 14b | Agentic RAG (Ch.14, line 8223) | ABSENT | — | No reasoning layer over retrieval. No source validation, no conflict reconciliation, no gap detection. Retrieval is passive. | Only worth doing after row 14 (basic RAG on the live path). | Low |
| 15 | Inter-Agent Communication (A2A) | PRESENT | `src/harnesscad/agents/a2a/messages.py:226` `AgentCard`, `:316` `A2AMessage`, `:178` `AgentSkill`, `:138` `Artifact`; `agents/a2a/task.py:43` `TaskState` (submitted/working/input_required/completed/failed), `:157` `Task`, `:319` `TaskStore`; `io/surfaces/a2a_server/` | In-process only; no HTTP/SSE/webhook transport. That is a deliberate and stated design choice (`a2a/messages.py:3`). No second agent to talk to anyway (row 7). | Ship the transport when the supervisor is wired. | Low |
| 16 | Resource-Aware Optimization | PRESENT | `src/harnesscad/core/routing.py:305` `RoutingLLM` (classify-then-route), `:233` `CostTable`, `:203` `usage_from_result`, `:300` sequential fallback chain; `eval/bench/harness/agent_cost.py`; `eval/bench/harness/resource_tradeoff.py` | Cost is tallied but no budget is *enforced* — there is no `max_spend` that aborts a run. The book's own framing (line 9243) is "achieve goals within specified resource budgets". | Add `budget: Optional[float]` to `RoutingLLM`; raise `BudgetExhausted` and let `AgentHarness` degrade to `RetrievalFallback`. | Low |
| 16a | Critique Agent (Ch.16, line 9397) | PARTIAL | `agents/agents/roles.py:211` `DFMOutcome` critic; `eval/verifiers/vlm_judge.py:190` `VLMJudgeCheck` | **`vlm_judge.py` is an ORPHAN.** The only critic on the live path is the unmeasured verifier fleet. | See the eval indictment. | **High** |
| 17 | Reasoning Techniques (chapter) | PARTIAL | See 17a-17k. | The agent has no explicit reasoning stage. It emits ops at temp 0 from a single prompt. | See rows below. | — |
| 17a | Chain-of-Thought (CoT) | ABSENT | `data/dataengine/schemas/cot_record.py`, `cot_trace.py` exist as *data schemas*; `eval/reliability/cot_grammar_gate.py` gates CoT output | We record CoT traces for training data and never *use* CoT in the loop. `agents/agent/system_prompt.py:81` `build_system_prompt` asks for "a JSON array of op objects, nothing else" — we explicitly forbid the model from thinking. | Allow a `reasoning` field before the ops array in the schema (`core/grammar.py:169`) and strip it before validation. Cheap; measurable via the pressure harness. | Medium |
| 17b | Tree-of-Thought (ToT) | ABSENT | — | Not N/A: `eval/reliability/strategies/mcts.py` is a tree search over op plans and is an ORPHAN. ToT is the same shape. | Wire `mcts.py` as an escalation tier alongside Best-of-N. | Low |
| 17c | Self-Correction / Self-Refinement | PRESENT | `core/loop.py:84` block-and-correct; `agents/agent/runner.py:27` replan-on-diagnostics | This is the harness's entire thesis and **it lost the experiment** (`assets/pressure/report.md:3`). See the indictment. | See gaps 1-3. | **High** |
| 17d | Program-Aided Language Models (PALM) | PRESENT | The whole architecture: the LLM emits typed ops, a deterministic kernel executes them (`core/cisp/ops.py`, `io/backends/`). This is PALM with a geometry kernel instead of a Python interpreter. | None. | — | — |
| 17e | Reinforcement Learning with Verifiable Rewards (RLVR) | ABSENT | `agents/agent/tool_reward.py:70` `ToolUseReward`; `eval/quality/reward/execution_reward.py`, `composite_reward.py` | We are sitting on the ideal RLVR domain — a deterministic verifier that scores any candidate — and have never trained anything. Reward functions written, never consumed. | See row 9. | Medium |
| 17f | ReAct (Reason and Act) | PARTIAL | `core/harness.py:1` declares itself "the ReAct orchestrator"; the loop at `:148-241` is Act -> Observe -> Act | It is **Act-Observe, not Reason-Act-Observe.** There is no Thought step (row 17a). Calling it ReAct in the docstring is overclaiming. | Add the reasoning field (row 17a), or rename the class. | Medium |
| 17g | Chain of Debates (CoD) | ABSENT | `agents/exploration/tournament.py` (Elo tournament over designs) is the closest thing and is orphaned | Not N/A — the six independent backends are natural debaters, and `eval/selftest/differential.py:236` `compare` already resolves their disagreement by clustering. That is a *geometric* CoD and it is better than the book's LLM version. What is absent is debate over *design intent*. | Low priority; the differential oracle covers the case that matters. | Low |
| 17h | Graph of Debates (GoD) | N/A | — | **Justified:** GoD's value is resolving contested claims with no ground truth. In our domain the ground truth is computable (six kernels + closed-form golden parts, `eval/selftest/golden.py`). Debate is the wrong tool when you own an oracle. | — | — |
| 17i | MASS (Multi-Agent System Search) | ABSENT | — | Not N/A: the book's finding (line 10416) is that MAS effectiveness depends on prompt quality *and* topology, and both should be searched. We have `agents/exploration/prompt_evolution.py` (prompt search) and `agents/agents/supervisor.py` (one hard-coded topology), and neither is wired. | Only relevant after row 7. | Low |
| 17j | Scaling Inference Law | PARTIAL | `eval/reliability/strategies/best_of_n.py` (ORPHAN); `eval/bench/harness/candidate_scaling.py` | The book's central claim (line 10479) — a small model with a bigger thinking budget beats a big model without one — is *exactly the claim the pressure test should have been measuring*, and Best-of-N is the cheapest way to buy it. It is orphaned. The pressure test spent its budget on typed feedback (which lost) instead of on samples. | Add a `--best-of N` arm to `eval/pressure/loops.py`. This is the experiment that will actually move the number. | **High** |
| 17k | Deep Research | N/A | — | **Justified:** Deep Research is iterative *web* search + synthesis. HarnessCAD's knowledge source is a geometry kernel, not a corpus; there is no open-world retrieval to deepen. The analogous capability (iteratively refining against an oracle) is `eval/selftest/properties.py:134`. | — | — |
| 18 | Guardrails / Safety Patterns | PRESENT | `src/harnesscad/eval/reliability/guardrails.py:49` `GuardrailGate` (before_tool_callback hard gate); `governance/security/tool_gate.py:70` `ToolTrustGate`, `:61` `prompt_risks` (jailbreak/instruction-smuggling patterns); `governance/security/policy.py:101` `SecureIngestGate` (path traversal, redaction, allowlist); `io/gate.py:157` `GateReport` (the output gate) | Input guardrails are pattern-based, not model-based. The book recommends a cheap LLM pre-screen (line 10785, 11416). Given our domain (CAD briefs, not open chat) this is defensible. | Optional. | Low |
| 18a | Input Validation / Sanitization | PRESENT | `governance/security/tool_gate.py:61` `prompt_risks`; `governance/security/policy.py:82` `redact_metadata` | — | — | — |
| 18b | Output Filtering / Post-processing | PRESENT | `src/harnesscad/io/gate.py:157` `GateReport`, `:178` `InvalidArtifact` — *"Every artifact leaving the harness is either verified valid, or refused with a reason. There is no third outcome."* | None. **This is better than anything in the book.** The book's output filter is an LLM checking for toxicity; ours is a measured geometric soundness proof. | — | — |
| 18c | Behavioral Constraints (prompt-level) | PRESENT | `agents/agent/system_prompt.py:81` `build_system_prompt` (op vocabulary generated from the registry; design rules; output contract) | — | — | — |
| 18d | Tool Use Restrictions | PRESENT | `governance/security/tool_gate.py:24` `ToolPolicy` (explicit allowlist + `TrustTier`); `io/surfaces/mcp/annotations.py:39` (destructive/readOnly hints) | — | — | — |
| 18e | Checkpoint and Rollback (Ch.18, "Engineering Reliable Agents", line 11536) | PRESENT | `core/state/opdag.py:54` `OpDAG`; `core/loop.py:177` `checkpoint`, `:180` `rollback` | — | — | — |
| 18f | Observability through Structured Logging (line 11554) | PRESENT | `core/trace.py:55` `Tracer`, `:99` `JsonlTracer`; `core/observe.py:417` `Metrics`, `:538` `FailureTaxonomy`, `:737` `Replayer`, `:279` `wilson_interval` | None. Confidence intervals on agent metrics is beyond what the book asks for. | — | — |
| 18g | Principle of Least Privilege (line 11565) | PRESENT | `governance/security/tool_gate.py:16` `TrustTier`; `io/surfaces/ui/approval.py:35` `ApprovalTier` | — | — | — |
| 18h | Modularity / Separation of Concerns (line 11545) | PARTIAL | The 7-layer split (`core`/`domain`/`io`/`eval`/`agents`/`data`/`governance`) | 1,219 modules, **323 orphans** (`registry.orphans()`, `src/harnesscad/registry.py:529`). 26% of the product is unreachable. Modularity without wiring is not modularity, it is a warehouse. | Triage the orphan list. See "Incoherences". | **High** |
| 19 | Evaluation and Monitoring | PARTIAL | ~200 benchmark modules under `eval/bench/`; `eval/selftest/` (4 oracles); `core/observe.py:417` `Metrics`; `eval/pressure/` (the controlled experiment) | **This is the chapter we violated. See the indictment below.** | — | **High** |
| 19a | Agent Response Assessment | PRESENT | `eval/pressure/metrics.py` `grade`; `eval/bench/protocols/success_rate.py` | The book's warning about naive exact-match (line 11722-11735) does not apply: we grade geometrically. | — | — |
| 19b | Latency Monitoring | PRESENT | `core/observe.py:108` `Span`, `:163` `SpanCollector`; `eval/bench/harness/latency_speedup.py` | — | — | — |
| 19c | Token-Usage Tracking | PRESENT | `core/routing.py:174` `Usage`, `:185` `count_tokens`, `:233` `CostTable` | — | — | — |
| 19d | LLM-as-a-Judge | PARTIAL | `eval/verifiers/vlm_judge.py:190` `VLMJudgeCheck` (advisory-only, INFO/WARNING, never ERROR — correct by design) | **ORPHAN.** Not on the live path, not in the fleet, never measured. | Either wire it as an advisory INFO tier or delete it. A judge nobody calls is dead weight that implies coverage we do not have. | Medium |
| 19e | Judge calibration / judge-vs-human agreement | PARTIAL | `eval/bench/judges/judge_calibration.py:6` `calibrate_threshold` (**ORPHAN**); `eval/bench/judges/judge_human_agreement.py` (MUSE 3-level agreement); `eval/bench/judges/perceived_actual_gap.py` | We built the calibration machinery and **never pointed it at our own judges.** The book's own caveat (line 7311-7317) says an LLM "may incorrectly assess its performance as successful" — we generalised that warning to LLM judges and never applied it to *rule* judges, which was the fatal category error. | Run `judge_calibration` against the verifier fleet, not just against LLM judges. `eval/selftest/fleet_audit.py:309` `audit` is the right tool and already exists. | **High** |
| 19f | Automated Metrics vs Human vs LLM-Judge trade-off table (line 12058) | PRESENT | The trade-off is resolved in our favour: `eval/selftest/differential.py:236` (6-engine oracle) and `eval/selftest/golden.py` (closed-form ground truth) give us *automated + objective + complete*, the row the book marks "potential limitation in capturing complete capabilities". | — | — | — |
| 19g | Agent Trajectories (exact/in-order/any-order match, precision, recall) | PARTIAL | `core/harness.py:74` `trajectory`; `core/observe.py:348` `RunTrajectory`, `:396` `group_runs`, `:804` `replay`; `agents/agent/tool_trajectory.py`; `eval/bench/harness/tool_trajectory.py`, `correction_trajectory.py` | Trajectories are recorded and replayable. There is **no ground-truth trajectory corpus** to compare against — the book (line 12097-12111) names six comparison methods and we implement none of them on our own runs. We measure outcomes, never process. | Build a golden-trajectory set for the 28 pressure briefs and score in-order match. This is how you find out *which* verifier is derailing the loop, which is precisely what the pressure test could not tell us. | Medium |
| 19h | Test files / Evalset files (ADK) | PARTIAL | `eval/pressure/briefs.py` (28 briefs); `eval/selftest/golden.py` | Briefs carry expected outcomes but not expected *tool trajectories* or intermediate responses. `assets/pressure/report.md:92` records the exact cost: *"the brief carries `bbox=None`, so the corpus is blind to it too. Fleet and corpus share the blind spot."* | Add expected bbox/volume/genus to every brief. The report names this as an unfixed hole. | **High** |
| 19i | The "Contractor" model (formal contract, negotiation, quality-focused iterative execution, hierarchical subcontracts) | PARTIAL | `core/contract.py:104` `Contract` (pillar 1, the formalized contract); `core/harness.py:211` (pillar 3, iterative execution against it) | Pillar 2 (**negotiation** — the agent flags an ambiguous or unsatisfiable contract *before* execution) is absent. Pillar 4 (**hierarchical subcontracts**) is absent. And the contract itself is optional and unused by `pipeline.build` (row 11). | Implement contract negotiation: `ContractCheck` can already prove a contract unsatisfiable against a plan; surface that as a pre-execution question to the human. | Medium |
| 20 | Prioritization | PARTIAL | `agents/agents/roles.py:72` `prioritize(findings)`; `eval/quality/reward/pareto.py:49` `Objective` (multi-objective Pareto front); `eval/verifiers/registry.py:63` fleet TIERS (core/lint/physics/domain = a cost ordering) | The *loop* does not prioritize. When 9 diagnostics come back, all 9 go to the model. There is no urgency/importance/dependency ranking of which one to fix first, and no dynamic re-prioritization. The book's whole point (line 12363) is that an agent facing many conflicting demands without a ranking "may experience reduced efficiency... or failures". | Rank the model-facing diagnostics by (soundness tier, severity, dependency order) and feed back the top-k, not all. `roles.py:72` `prioritize` already exists — call it from `Planner.build_messages`. | Medium |
| 21 | Exploration and Discovery | PARTIAL | `agents/exploration/` (19 modules): `tournament.py`, `elo.py` (Elo-ranked hypothesis tournament — the Co-Scientist pattern verbatim), `evolution.py`, `designspace_sampler.py`, `latin_hypercube.py`, `variation.py`, `greedy_refine.py`, `variant_consensus.py:32` `consensus_base` | The whole package is off the shipped path. `harnesscad` has no `explore` command (`core/cli.py:621` subparser list). Built, tested, unreachable. | Add `harnesscad explore`. The Co-Scientist generate->debate->evolve loop is fully implemented in `tournament.py`. | Medium |
| A | Appendix A: Advanced Prompting (chapter) | PARTIAL | See A1-A20. | — | — | — |
| A1 | Zero-Shot Prompting | PRESENT | `agents/agent/system_prompt.py:81` — the shipped planner is zero-shot | This is a *finding*, not a feature: we run the weakest prompting technique in the book on the hardest task in the book. | See A3. | — |
| A2 | One-Shot Prompting | ABSENT | — | — | Subsumed by A3. | — |
| A3 | Few-Shot / Many-Shot Prompting | ABSENT | `agents/rag/exemplar_select.py` and `agents/context/exemplar_prompt.py` exist to do exactly this and are **unwired** | The book (line 13418-13449) is unambiguous that few-shot is "particularly effective for tasks where the desired output requires adhering to a specific format" — i.e. our exact problem (emit a valid typed op array). We ship zero examples. The pressure test's weak models (1.5b, 3b) failed at *format*, which is the failure few-shot fixes. | Put 3-5 verified brief->ops exemplars in the system prompt via `exemplar_select.py`. **This is the cheapest available win in the entire audit** and it is a one-line call. | **High** |
| A4 | System Prompting | PRESENT | `agents/agent/system_prompt.py:81` | — | — | — |
| A5 | Role Prompting | PRESENT | `agents/agents/roles.py` personas; `agents/agent/system_prompt.py` | — | — | — |
| A6 | Delimiters | PRESENT | `agents/agent/planner.py:90-107` (`DESIGN BRIEF:` / `CURRENT MODEL STATE:` / `PRIOR ATTEMPT FAILED —`) | — | — | — |
| A7 | Structured Output + Pydantic facade | PRESENT | `core/grammar.py:169` `op_json_schema`; `agents/llm/structured.py` `ParsedOps`; `agents/llm/generation_contract.py` | We use dataclasses + a hand-rolled validator instead of Pydantic. Functionally equivalent, stdlib-only by policy. | — | — |
| A8 | Chain of Thought (CoT) | ABSENT | See row 17a. | — | — | Medium |
| A9 | Self-Consistency | PARTIAL | `agents/exploration/variant_consensus.py:32` `consensus_base` (majority vote over candidate models) | **ORPHAN**, and it votes over *voxels*, not over reasoning paths. There is no majority-vote-over-N-plans on the live path, despite the verifier making it nearly free (the book: draw N, majority-vote). | Combine with Best-of-N (row 17j): draw N plans, keep those that verify, take the geometric consensus. | Medium |
| A10 | Step-Back Prompting | ABSENT | — | Not N/A: "what are the principles of a good sheet-metal bracket" before "make this bracket" is a real, applicable move for a CAD planner. | Cheap experiment for the pressure harness. | Low |
| A11 | Tree of Thoughts | ABSENT | See row 17b. | — | — | Low |
| A12 | Tool Use / Function Calling | PRESENT | See row 5. | — | — | — |
| A13 | ReAct | PARTIAL | See row 17f. | — | — | — |
| A14 | Automatic Prompt Engineering (APE) / DSPy-style programmatic optimization | PARTIAL | `agents/exploration/prompt_evolution.py`; `agents/exploration/image_prompt_sweep.py`; `agents/exploration/technique_trials.py` | **ORPHANS.** The book (line 14010-14042) says APE needs a goldset + an objective function. **We have both**: 28 briefs and a deterministic verifier. We have never optimized our own system prompt. | Run `prompt_evolution.py` against `eval/pressure/briefs.py` with the grader as the objective. This is a weekend and it is measurable. | Medium |
| A15 | Iterative Prompting / Refinement | PRESENT | `core/loop.py`, `agents/agent/runner.py:27` | — | — | — |
| A16 | Negative Examples | ABSENT | `agents/memory/error_notebook.py:131` `ErrorNotebook` is exactly this (past mistakes as corrective few-shot exemplars) and is not on the live path | — | Wire alongside A3. | Low |
| A17 | Analogies | N/A | — | **Justified:** the book presents this as a creative-framing device for open-ended generation. Our output space is a typed op grammar with a formal contract; analogy adds ambiguity to a channel whose whole design goal is to remove it. | — | — |
| A18 | Factored Cognition / Decomposition | PARTIAL | `agents/generation/design_plan.py`; `agents/exploration/decomp_state.py`, `decomp_reward.py` | ORPHANS. See row 1. | — | Medium |
| A19 | Retrieval Augmented Generation (as a prompting technique) | PARTIAL | See row 14. | — | — | **High** |
| A20 | Persona Pattern (user persona) | ABSENT | — | Not N/A — "the reader of this output is a machinist" vs "a hobbyist with a 3D printer" changes DFM tolerances materially, and `eval/verifiers/dfm.py` already has the knobs. | Low value relative to A3. | Low |
| B | Appendix B: GUI to Real-World Interaction | PARTIAL | `io/surfaces/ui/` (approval, previews, events); `io/render.py`, `io/surfaces/canonical_views.py`, `io/surfaces/render.py` (the agent *sees* its geometry); `domain/vision/` (14 modules) | We render for the *judge*, not for the *agent*: no rendered view is fed back into the planning loop. `agents/generation/stepwise_visual_feedback.py` and `caption_feedback.py` exist and are unwired. | Feed the isometric render back into the retry prompt for vision-capable models. `vlm_judge.py:148` `_views_to_data_uris` already does the encoding. | Medium |
| B1 | Vibe Coding | N/A | — | **Justified:** this is a description of how humans use LLMs to write software, not a pattern an agent implements. Nothing in HarnessCAD could conform or fail to conform to it. | — | — |
| C | Appendix C: Agentic Frameworks (LangChain/LangGraph/CrewAI/ADK/AutoGen/...) | N/A | — | **Justified:** this is a comparative survey of third-party frameworks, not a pattern. HarnessCAD deliberately depends on none of them (stdlib-only core). The *patterns* those frameworks embody are audited individually above (rows 1, 2, 3, 7, 8). | — | — |
| D | Appendix D: Building an Agent with AgentSpace | N/A | — | **Justified:** a product tutorial for a Google Cloud console. There is no transferable pattern; the underlying capabilities (no-code agent config, enterprise knowledge graph) are audited as rows 7 and 14a. | — | — |
| E | Appendix E: AI Agents on the CLI (+ Terminal-Bench) | PRESENT | `core/cli.py:621` — 20+ subcommands (`build`, `apply`, `export`, `selftest`, `pressure`, `bench`, `capabilities`, ...); `eval/selftest/registry.py:1` | We are a CLI agent. No Terminal-Bench-style standardized harness for *our* CLI, but `eval/pressure/` is a stricter equivalent. | — | Low |
| F | Appendix F: Under the Hood — Agents' Reasoning Engines | N/A (source unavailable) | — | **Justified by absence of source:** listed in the TOC at line 59 but **the text is not present in the extracted file** (jump from Appendix E at line 15089 to Appendix G at 15266). No content to audit against. Flagged rather than silently skipped. | Re-extract the PDF if this appendix matters. | Low |
| G | Appendix G: Coding Agents (human-led orchestration, context staging, specialist agents) | PARTIAL | `agents/context/staging.py` (the Context Staging Area, verbatim); `agents/agents/roles.py` (specialist personas: Designer/Modeler/Verifier/DFMCritic/RedTeam/Reviewer — a superset of the book's Scaffolder/TestEngineer/Documenter/Optimizer/ProcessAgent) | The book's Process Agent does **critique THEN reflection on its own critique** (line 15599-15613) — a two-stage critic that "dismisses pedantic or low-impact suggestions". **We have no such stage, and its absence is the direct cause of the pressure-test loss.** Our fleet dumps every finding on the model unfiltered. | `eval/verifiers/soundness.py:497` `model_facing` is the first half of this. The second half — a reflection pass that *ranks and drops* low-impact findings — is row 20. | **High** |
| Concl. | Conclusion: composition of patterns | PARTIAL | `core/harness.py:91` composes session + planner + context + loop-detector + executor + tracer + verifiers + contract | The composition is real, but four of those seven collaborators default to `None` (`harness.py:102-113`) and `pipeline.build` — the only entry point users call — supplies none of them. The composed agent exists in the type signature, not in the product. | Make `pipeline.build` construct the full harness. | **High** |
| Gloss. | Glossary: Critique Model | PARTIAL | `eval/verifiers/vlm_judge.py:190` (orphan); the verifier fleet | — | See row 19d. | — |

**Verdict counts (74 rows): PRESENT 26, PARTIAL 27, ABSENT 15, N/A 6.**

---

## 1. The eval indictment

The book warned us. Not obliquely — in three separate places, in the voice of a caveat, about the
exact failure that cost us the experiment.

### 1.1 It told us a self-judging loop cannot see itself going wrong

**Chapter 11, "Caveats and Considerations" (line 7311-7317):**

> *"An LLM may not fully grasp the intended meaning of a goal and might incorrectly assess its
> performance as successful. Even if the goal is well understood, the model may hallucinate. **When
> the same LLM is responsible for both writing the code and judging its quality, it may have a
> harder time discovering it is going in the wrong direction.**"*

We read this as a warning about *LLM* judges and concluded it did not apply: our judge is a rule,
and rules do not hallucinate. That inference is false and it is the whole bug. A rule that compares
a hole's diameter to a plate's *thickness* — orthogonal dimensions — hallucinates as confidently as
any model, and unlike a model it never expresses doubt. The book's warning is about **an unaudited
evaluator inside a correction loop**. The evaluator's implementation technology is irrelevant. We
exempted ourselves from a warning on a technicality and it fired anyway:
`assets/pressure/report.md:83-88` — *"It fired 40 times and caused every regression."*

### 1.2 It told us the critic must reflect before it speaks

**Appendix G, "The Process Agent: The Code Supervisor" (line 15599-15613):**

> *"**Critique:** The agent performs an initial pass, identifying potential bugs, style violations,
> and logical flaws, much like a static analysis tool. **Reflection:** The agent then analyzes its
> own critique. It synthesizes the findings, prioritizes the most critical issues, **dismisses
> pedantic or low-impact suggestions**, and provides a high-level, actionable summary..."*

The book's critic has **two** stages. Ours has one. `eval/verifiers/registry.py` `run_all` collects
every diagnostic the fleet produces and `core/loop.py:143` appends all of them to the result, from
which `agents/agent/runner.py` hands them to the model. There was no synthesis stage, no
prioritization, and — fatally — no dismissal. A "static analysis tool" (the book's own analogy) is
tolerable precisely *because* a reflection pass stands between it and the person acting on it. We
removed the reflection pass and wired the linter directly to the executor's hands.

### 1.3 It told us the whole system, not the parts, must be evaluated

**Chapter 19 (line 12082-12085):**

> *"Evaluating agents' trajectories is essential, as traditional software tests are insufficient.
> Standard code yields predictable pass/fail results, whereas agents operate probabilistically,
> necessitating qualitative assessment of both the final output **and the agent's trajectory — the
> sequence of steps taken to reach a solution**."*

And (line 12104-12108) the book enumerates the trajectory metrics by name: *"exact match... in-order
match... any-order match... **precision** (measuring the relevance of predicted actions), **recall**
(measuring how many essential actions are captured)."*

**The words precision and recall are in the book, in the evaluation chapter, applied to agent
behaviour.** We implemented recall for 23 verifiers — every one has a unit test proving it fires on
bad input (`tests/eval/verifiers/`) — and we implemented precision for none. Precision is *named in
the source text* as a required metric and we shipped without it.

### 1.4 It told us false feedback poisons a capable follower

**"A Thought Leader's Perspective," Marco Argenti, CIO Goldman Sachs (line 288-291):**

> *"Messy systems plus agents are a recipe for disaster. **An AI trained on 'garbage' data doesn't
> just produce garbage-out; it produces plausible, confident garbage that can poison an entire
> process.**"*

Substitute "fed garbage diagnostics" for "trained on garbage data" and this is the pressure report's
finding verbatim, written a year earlier. Our own report reaches the same sentence independently
(`report.md:53-56`): *"The value of a typed diagnostic is bounded above by its truth, and the
tighter a model's instruction-following, the tighter that bound binds."* The book got there first.

### 1.5 Does the book discuss what we asked?

* **Critic/verifier precision vs recall** — **Yes.** Line 12104-12108, in the trajectory-evaluation
  section, by name. We violated it.
* **False-positive feedback** — **Yes, indirectly but unmistakably.** Line 288-291 (poisoning),
  line 7311-7317 (a judge that is wrong about success), line 15599-15613 (the critic must dismiss
  low-impact findings before they reach the executor). We violated all three.
* **Judge calibration** — **Yes.** Chapter 19's LLM-as-a-Judge section builds a rubric with explicit
  1-5 anchors per criterion (line 11843-11905) and demands a rationale, precisely so the judge's
  output is auditable. Our fleet emits a code and an imperative with no confidence, no rationale and
  no anchor. We *have* `eval/bench/judges/judge_calibration.py:6` `calibrate_threshold` and
  `judge_human_agreement.py` and **we pointed neither at our own verifiers.**
* **LLM-as-judge failure modes** — **Yes.** Line 12069-12074: *"Intermediate steps may be
  overlooked. Limited by LLM capabilities."* We correctly quarantined the LLM judge as advisory
  (`eval/verifiers/vlm_judge.py:7`) and then *failed to apply the same suspicion to the rule judges*,
  which is the inverse error and the more expensive one.
* **Ablations** — **Yes.** Chapter 19's A/B testing (line 11671) and Chapter 17's MASS
  influence-weighted topology search (line 10370-10383), which is ablation as a design method. We
  own `governance/research/ablation_matrix.py:6` `compare_ablation` and
  `governance/research/judge_ablation.py` and **ran neither on the fleet.** A per-verifier
  leave-one-out ablation over the 12 pressure briefs would have found the hole rule in one
  afternoon, before the experiment, for free.
* **Regression testing of the evaluator itself** — **Yes.** Line 12303-12308: *"Evaluating
  intelligent agents goes beyond traditional tests to continuously measure their effectiveness...
  and adherence to requirements."* And Chapter 18's fault-tolerance framing: *"you cannot afford to
  trust blindly"* (line 253). The evaluator is a component of the system; it is subject to the same
  evaluation discipline as the system. We treated the verifier fleet as ground truth rather than as
  a component under test, and ground truth is the one thing a verifier is not.

### 1.6 What has since landed, and why it does not yet count

Credit where due, and then the qualification.

* `src/harnesscad/eval/verifiers/soundness.py:91-100` now defines PROVEN / MEASURED / HEURISTIC and
  `MODEL_FACING_TIERS = (PROVEN, MEASURED)`. `:497` `model_facing` filters the retry channel.
  `:525` `observe` rewrites imperatives ("Reduce the radius below 2.5") into evidence ("fillet r=8
  on a part whose smallest extent is 5"). This is the correct fix and it is well-reasoned.
* `agents/agent/planner.py:36,98` applies that gate on the only channel that reaches a model.
* `eval/verifiers/precheck.py:38-43` records that the unsound hole rule was **deleted**.
* `eval/selftest/fleet_audit.py:178` `VerifierScore` finally computes per-verifier precision,
  recall, F1, and lists the false positives by part name. `:309` `audit` runs it.

**And the number is still -8.3, because the experiment cannot see any of it.**
`eval/pressure/prompts.py:188` `format_typed` — the harness arm's feedback formatter — filters on
`is_actionable(d)` and **nothing else**. It does not import `soundness`. It does not call
`model_facing`. `eval/pressure/loops.py` builds its own loop and does not use `Planner`. The fix
lives in the production planner; the measurement lives in a loop that bypasses it. Until
`eval/pressure/prompts.py` routes through `soundness.model_facing`, the harness has repaired the
bug and cannot prove it, which is the same epistemic position it was in before the repair — and
this is precisely the discipline the report itself demanded when it refused to fix the hole rule
before reporting. The fix is now in. **Rerun the experiment.**

---

## 2. Top 10 gaps, ranked by (harm to correctness) / (effort to fix)

1. **The pressure experiment bypasses the soundness gate.** `eval/pressure/prompts.py:188`
   `format_typed` does not call `eval/verifiers/soundness.py:497` `model_facing`. The one measurement
   of the harness's central claim is measuring the *pre-fix* harness. **Fix:** two imports and one
   list comprehension in `prompts.py`, then rerun `harnesscad pressure` with the published command.
   **Effort: 30 minutes. Harm of not doing it: the product's headline number is a lie in both
   directions and we cannot tell which.**

2. **Zero few-shot exemplars in the system prompt.** `agents/agent/system_prompt.py:81` ships a
   pure zero-shot prompt for a strict-format structured-output task. `agents/rag/exemplar_select.py`
   and `agents/context/exemplar_prompt.py` are written, tested and unwired. The book (line 13424)
   names this the technique for exactly our failure mode. Four of six pressure models were at or
   below 33% solve rate, and the small ones failed at *format*. **Fix: one call in
   `Planner.build_messages`. Effort: 1 hour.**

3. **The fleet audit is never run in CI and its output is not in `assets/`.**
   `eval/selftest/fleet_audit.py:309` computes precision per verifier and
   `eval/selftest/registry.py` exposes it as `harnesscad selftest --fleet`. There is no committed
   report, no test asserting a precision floor, and `--strict` is off. A verifier fleet whose
   precision is not gated in CI will regress to exactly where it was. **Fix: run it, commit
   `assets/fleet/report.md`, add a CI gate at `precision >= 0.95` for any HEURISTIC rule that is
   allowed model-facing. Effort: 2 hours.**

4. **Best-of-N is orphaned.** `eval/reliability/strategies/best_of_n.py` — the book calls this the
   Scaling Inference Law (line 10479: a smaller model with more thinking budget beats a bigger one
   without) and our own domain makes it nearly free, because the verifier is simultaneously reward,
   eval and selector. We spent the inference budget on typed feedback, which lost by 8.3, instead of
   on samples, which we never tried. **Fix: add a `--best-of N` arm to `eval/pressure/loops.py`.
   Effort: half a day. This is the most likely path to a positive headline.**

5. **`Contract` is optional and `pipeline.build` never builds one.** `core/contract.py:104` is the
   book's "Contractor" pillar 1 (Ch.19, line 12180) — a machine-verifiable acceptance spec — and
   `core/pipeline.py:126` `build`, the entry point every user calls, passes no contract, so the
   agent has no goal it can check. Every brief is graded against nothing. **Fix: emit a Contract
   from the brief via `contract_from_brief_schema()` (already written, `contract.py:378`) before
   planning. Effort: 1 day.**

6. **Briefs carry no expected bbox.** `assets/pressure/report.md:92`: *"the brief carries
   `bbox=None`, so the corpus is blind to it too. Fleet and corpus share the blind spot."* Every
   `shell_box_3mm` solve in the published run is a part with wrong outside dimensions and the grader
   passed it. A corpus that cannot detect the bug the verifier misses is not an independent check.
   **Fix: add expected bbox/volume/genus to all 28 briefs in `eval/pressure/briefs.py`. Effort: 1
   day. This is a correctness hole in the *grader*, which is worse than one in the fleet.**

7. **Memory and RAG are wired to nothing.** `agents/memory/store.py:114` `MemoryStore`,
   `agents/memory/skills.py:80` `SkillLibrary`, `agents/rag/retriever.py:54` `HybridRetriever` — all
   built, all tested, none consulted by `agents/agent/planner.py:90-107`, which composes its prompt
   from brief + state + diagnostics and nothing else. The agent cannot learn from a run because it
   never reads what the last run wrote. Ch.4 (line 2472): *"Without memory, each reflection is a
   self-contained event."* **Fix: two calls in `build_messages`. Effort: 1 day including eval.**

8. **No prioritization of model-facing diagnostics.** When 9 findings come back, all 9 go to the
   model. `agents/agents/roles.py:72` `prioritize` exists and is not called from the loop. The book
   (Ch.20, line 12363) and Appendix G (line 15605, "dismisses pedantic or low-impact suggestions")
   both require ranking-and-dropping. An unranked pile of instructions is how a capable model gets
   pulled in nine directions. **Fix: rank by (soundness, severity, op-index) and feed the top 3.
   Effort: half a day.**

9. **Reflexion is orphaned.** `eval/reliability/strategies/reflexion.py:139` `ReflexionLoop`
   implements READ-ACT-REFLECT-WRITE against `MemoryStore` — the book's Ch.4 Reflection pattern with
   Ch.8 memory, exactly as the book says to combine them (line 2472-2480). It is not reachable from
   `AgentHarness`. We have the pattern the book calls essential for quality and we do not run it.
   **Fix: escalation tier after 2 failed iterations. Effort: 1 day.**

10. **The verifier fleet runs serially and so does the 6-engine oracle.** No concurrency exists
    anywhere in `src/harnesscad` (grep for `ThreadPoolExecutor|asyncio|concurrent.futures`: zero
    hits). Ch.3's named use case 5 *is* concurrent validation. `eval/verifiers/registry.py`
    `run_all` iterates 23 pure-reader verifiers in a `for` loop; `eval/selftest/differential.py:310`
    boots six geometry kernels one after another. This costs no correctness, only wall clock — which
    is why it is last — but it is the cheapest 5x in the repo and it is what makes the fleet cheap
    enough to *always* run at `verify_level="full"`. **Fix: `ThreadPoolExecutor` around two loops.
    Effort: 2 hours.**

---

## 3. What we do better than the book

Honestly, and with the caveat attached to each.

**3.1 We have a differential oracle across six independent engines. The book has no equivalent and
does not imagine one.** `eval/selftest/differential.py:236` `compare` runs one op stream on stub,
F-rep, CadQuery, FreeCAD, OpenSCAD and Blender, clusters the observations, and reports disagreement.
Where they disagree, at least one is wrong — *and you did not need to know the right answer to find
that out.* The book's Chapter 19 evaluation table (line 12058-12081) lists three methods — Human
(unscalable), LLM-as-a-Judge (limited by the LLM), Automated Metrics (may not capture full
capability) — and every one of them is either subjective or partial. A six-way differential is
objective, complete, needs no gold, no label and no model. It is a strictly stronger instrument than
anything in the book's evaluation chapter. *Caveat: six engines that share a bug agree, and
`properties.py` and `golden.py` exist precisely because a differential cannot catch a consensus
error. We built the complement. Credit is earned only because we built both.*

**3.2 Our output gate makes a soundness promise the book never asks for.** `io/gate.py:1-11`:
*"Every artifact leaving the harness is either (a) verified valid, or (b) refused with a reason.
There is no third outcome. Silence is not success."* The book's output guardrail (Ch.18) is a
content filter looking for toxicity. Ours is a measured geometric proof — manifoldness, watertight,
signed volume, self-intersection — enforced at the one door every artifact exits through. The book
is protecting a user from a bad *sentence*; we are protecting them from a bad *part*, and we do it
with arithmetic rather than a second model's opinion. *Caveat: soundness, not completeness. The gate
refuses more than it must, and we have never measured how often — which is the same precision blind
spot as the fleet, one layer down. It has simply not cost us anything yet because a refusal is
visible and a false diagnostic is not.*

**3.3 Our structured output cannot drift from the tool set.** `core/grammar.py:169` derives the JSON
Schema and `:255` the GBNF grammar *from the op registry itself* (`cisp.ops._REGISTRY`), and
`agents/agent/system_prompt.py:40` `op_vocabulary` derives the prompt's tool documentation from the
same source. The book's Appendix A treats prompt, schema and tool list as three hand-maintained
artifacts that a developer keeps in sync. Ours are three projections of one source of truth and a
drift is a compile error. *Caveat: this is good engineering, not a novel pattern, and it is table
stakes in any typed system.*

**3.4 Determinism and replay are absolute.** `core/state/opdag.py:54` is content-hashed and
append-only; `core/loop.py:72` `_make_run_id` and `core/harness.py:126` derive run IDs from content,
not from a clock; `core/trace.py:1-15` refuses to depend on wall time. `core/observe.py:737`
`Replayer` reconstructs any run from its event stream. The book's Chapter 19 wants observability;
we have *reproducibility*, which is a strictly stronger property and the precondition for any
credible experiment. This is why the pressure test could be run at all, and it is the reason its
negative result is trustworthy.

**3.5 We published a negative result about our own product and refused to fix the bug first.**
`assets/pressure/report.md:136-144`: *"The broken hole rule was NOT fixed before reporting, even
though fixing it flips the headline from -8.3 to +3.7. It is a bug in the product under test, and
repairing the thing under test to improve its score is the definition of a rigged result."* There is
no equivalent of this anywhere in the book — every measurement in it is a demonstration of a pattern
working. This is the single most valuable artifact in the repository and it is the reason this audit
has anything true to say.

**Where we are behind, plainly: everything above is about *verification*, and verification is not the
agent.** The agent itself — zero-shot, no memory, no retrieval, no reasoning step, no reflection, no
parallelism, no few-shot examples, one model, one prompt — is the least sophisticated component in
the system and the book would recognise almost nothing in it. We have built a world-class *oracle*
and hung a 2023 agent off the front of it. That is the coherent summary of this audit: **the harness
is SOTA-2026 at knowing whether a part is right, and pre-SOTA at making one.**

---

## 4. Incoherences

**4.1 Two retry loops, two different feedback policies.** `agents/agent/runner.py:27` `run` and
`core/harness.py:132` `AgentHarness.run` are both "the correction loop", and
`eval/pressure/loops.py` is a *third*. The first two funnel through `Planner.plan_parsed`, which
applies the soundness gate (`planner.py:98`). The third builds its own prompt via
`eval/pressure/prompts.py:215` `FEEDBACK` and applies no gate at all. The same pattern is therefore
implemented three times with three different levels of correctness, and the one used to *measure* the
product is the least correct. **This is not a stylistic duplication; it is why the headline number
is stale.**

**4.2 The verifier fleet has two orthogonal tier systems and only one is enforced.**
`eval/verifiers/registry.py:63-76` defines COST tiers (`core`/`lint`/`physics`/`domain`) and
`eval/verifiers/soundness.py:91-95` defines TRUST tiers (`proven`/`measured`/`heuristic`). The
registry stamps every diagnostic with its trust tier (`registry.py:951` `_soundness.stamp`) and the
loop then... surfaces all of them equally (`core/loop.py:143`). The trust tier is honoured *only* by
`planner.py:98`, i.e. only if the caller happens to route through the Planner. Any consumer reading
`ApplyOpsResult.diagnostics` directly — which is what `eval/pressure/loops.py` does, and what any
integrator would do — gets the unfiltered pile. **The gate is a property of one call site, not of the
type.** Put it on `Diagnostic` and make the unfiltered list the one you have to ask for.

**4.3 `registry.orphans()` reports 323 orphans (26% of 1,219 modules) and over-reports them.**
`src/harnesscad/registry.py:529` computes reachability from the *static* import graph. But
`eval/verifiers/registry.py:858` discovers verifiers *dynamically* through the capability index, so
`eval/verifiers/precheck.py` shows as an orphan while being the module that caused every regression
in the pressure test. Two different notions of "wired" coexist and neither is authoritative:
static-import-reachable, and dynamically-discovered-at-runtime. A module can be live and look dead,
or look alive (has an importer) and never execute. **Fix: have `registry.orphans()` take the
dynamic-discovery registries as additional roots, and add a third category — `discovered` — so the
orphan list means something.** Right now the single best tool for finding the dead 26% cannot tell
you which of the 26% is actually dead.

**4.4 `AgentHarness` is a composition that is never composed.** `core/harness.py:102-113` takes
`context`, `loop_detector`, `executor`, `tracer`, `verifiers` and `contract` — six collaborators,
all defaulting to `None`, each with a graceful-degradation path documented at `:93-100`. And
`core/pipeline.py:126` `build`, the only entry point in the CLI (`core/cli.py:639` `p_build`),
constructs **none of them**. Every degradation path is the one being taken. The harness's own
docstring describes a system that does not ship.

**4.5 Reflection is implemented three times at three levels of fidelity.**
`eval/reliability/strategies/reflexion.py:139` (full READ-ACT-REFLECT-WRITE with memory),
`agents/agents/roles.py:187-230` (Verifier + DFMCritic + RedTeam personas), and
`agents/agent/runner.py:27` (feed the diagnostics back and hope). The first two are orphans. The
third is what ships and it is the one the book explicitly warns is the weak form: *"a single agent
can perform self-reflection, [but] using two specialized agents... often yields more robust and
unbiased results"* (line 2429-2431).

**4.6 `agents/registry.py`, `agents/exploration/registry.py`, `agents/generation/registry.py`,
`core/registry.py`, `eval/bench/registry.py`, `eval/quality/registry.py`, `eval/selftest/registry.py`,
`eval/verifiers/registry.py`, `governance/registry.py`, `io/formats/registry.py`,
`io/surfaces/registry.py` and the top-level `registry.py` are twelve different things called
"registry"** — some are CLI subcommand dispatchers, some are capability indexes, some are plugin
discoverers. `eval/verifiers/registry.py` is a runtime dispatcher *and* a fleet definition *and* a
soundness stamper. The word carries no information. This is a naming incoherence, not a correctness
one, but it makes the orphan analysis (4.3) genuinely hard to reason about.

**4.7 `core/cisp/explicit_context.py` is a 12-line, unformatted, semicolon-packed module in a
codebase where every other file carries a 15-line design docstring.** It implements handle
generations and stale-handle rejection — a real safety property — and it looks like it was pasted in
and forgotten. Either it is load-bearing (in which case it deserves the same rigour as
`opdag.py`) or it is not (in which case it is one of the 323).
