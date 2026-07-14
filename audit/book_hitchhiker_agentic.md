# HarnessCAD vs. *The Hitchhiker's Guide to Agentic AI* — architecture audit (H3, H4, H5)

Scope: chapters 15–29 (H3: agentic intro / RAG / memory / **harness**; H4: design patterns /
environments / MCP / skills / A2A / multi-agent; H5: frameworks / agentic UI / quiz / quick
reference / conclusion). Read in full from
`resources/extracted_text/chunks/{H3,H4,H5}*.txt`. H3 chapter 18 (the harness chapter) was
read twice.

System under audit: `C:/Users/adria/Downloads/redNoise-main/harnesscad`, commit state of
2026-07-14. Read-only pass over `src/` and `tests/`; nothing outside `audit/` was written.

Everything below is evidenced with `file:line`. Where the verdict is PARTIAL the specific
missing half is named. Where it is ABSENT it is absent, not "planned".

---

## 0. Section count

| Chunk | Chapters | Sections counted |
|---|---|---:|
| H3 | 15 (Introduction), 16 (RAG), 17 (Memory), 18 (**Harness**) | **141** |
| H4 | 19 (Design Patterns), 20 (Environments), 21 (MCP), 22 (Skills), 23 (A2A), 24 (Multi-Agent) | **199** |
| H5 | 25 (Frameworks), 26 (Agentic UI), 27 (Quiz), 28 (Quick Reference), 29 (Conclusion) | **128** |
| | | **468** |

Verdict tally across all 468 rows:

| Verdict | Count | Share |
|---|---:|---:|
| PRESENT | 121 | 25.9% |
| PARTIAL | 133 | 28.4% |
| ABSENT | 92 | 19.7% |
| N/A (justified) | 122 | 26.1% |

The N/A block is dominated by chapter 27 (30 quiz sections), chapter 28 (11 formula-reference
sections) and chapter 29 (3 conclusion sections) — 44 rows that are assessment/reference matter
with no implementable surface — plus the RAG/memory training-theory sections (RL for memory,
joint retriever-generator training, MARL formalism) that belong to the sibling agent's
eval/RL brief, and the UI-framework rows (Vercel/Chainlit/Gradio/Streamlit) for a product with
no browser UI.

---

## 1. Coverage matrix

### H3 — Chapter 15: Introduction to Agentic AI

| # | Section | Verdict | Where in HarnessCAD | Gap | Fix | Pri |
|---|---|---|---|---|---|---|
| 1 | Ch.15 The agentic stack (persistence / grounding / action / coordination / safety; layered architecture) | PARTIAL | `core/harness.py:91`, `core/loop.py:45` | Action + safety layers are real; **persistence** (memory) and **coordination** are built but unwired into the harness | Wire `agents/memory` into `AgentHarness` (see gap #2) | P0 |

### H3 — Chapter 16: Retrieval-Augmented Generation

| # | Section | Verdict | Where | Gap | Fix | Pri |
|---|---|---|---|---|---|---|
| 2 | 16.1 Motivation and Problem Statement | PRESENT | `agents/rag/__init__.py:1`, `agents/rag/retriever.py:1` | — | — | — |
| 3 | 16.1.1 Parametric vs. Non-Parametric Knowledge | PRESENT | `agents/rag/api_knowledge.py:1`, `agents/agent/tool_knowledge.py:1` | — | — | — |
| 4 | 16.1.2 RAG vs. Fine-Tuning vs. Long Context | PARTIAL | `agents/rag/index.py`, `data/dataengine/selftrain/` | The trade-off is never decided anywhere; both paths exist, neither is chosen for the harness | State a policy in `docs/` (RAG for CAD-API knowledge, FT for op-format compliance) | P3 |
| 5 | 16.2 Core RAG Architecture | PRESENT | `agents/rag/index.py:1`, `agents/rag/chunk.py:1`, `agents/rag/retriever.py:1` | — | — | — |
| 6 | 16.2.1 Full Pipeline Diagram | PARTIAL | `agents/rag/__init__.py:1` | Documented in prose, no offline/online split | — | P3 |
| 7 | 16.2.2 Indexing Pipeline | PRESENT | `agents/rag/index.py:1`, `agents/rag/chunk.py:1` | — | — | — |
| 8 | 16.2.3 Retrieval | PRESENT | `agents/rag/retriever.py:1` | — | — | — |
| 9 | 16.2.4 Generation | ABSENT | — | No RAG-into-prompt path: `Planner.build_messages` (`agents/agent/planner.py:84`) injects **no** retrieved context | Add a `retriever` seam to `Planner` | P1 |
| 10 | 16.3 Retrieval Methods | PARTIAL | `agents/rag/retriever.py`, `agents/rag/sphere_knn.py:1` | Token-overlap + kNN only | — | P3 |
| 11 | 16.3.1 Sparse (BM25/TF-IDF) | PARTIAL | `agents/memory/store.py:51` (`TokenOverlapSimilarity`: Jaccard+difflib) | Not BM25; no IDF, no length normalisation | Swap in BM25 behind the existing `Similarity` protocol (`store.py:40`) | P2 |
| 12 | 16.3.2 Dense (DPR) | ABSENT | `agents/memory/store.py:40` declares the `Similarity` seam and says "a real embedder is the future upgrade" | No embedder anywhere in the product | Implement one `Similarity`; the call sites are already abstract | P2 |
| 13 | 16.3.3 Hybrid / RRF | ABSENT | — | No fusion of two rankers | RRF is 10 lines; the book gives it (H3:908) | P3 |
| 14 | 16.3.4 SPLADE / SPLADEv2 | N/A | — | Learned sparse retrieval over a 30k vocab is unjustified for a corpus of CAD API docs + a few thousand episodes | — | — |
| 15 | 16.3.5 ColBERT late interaction | N/A | — | Same: corpus scale does not warrant it | — | — |
| 16 | 16.3.6 Retrieval Method Comparison | ABSENT | — | No benchmark of retrievers | — | P3 |
| 17 | 16.4 Chunking Strategies | PRESENT | `agents/rag/chunk.py:1` | — | — | — |
| 18 | 16.4.1 Fixed-size with overlap | PRESENT | `agents/rag/chunk.py` | — | — | — |
| 19 | 16.4.2 Semantic chunking | ABSENT | — | Requires an embedder (see #12) | — | P3 |
| 20 | 16.4.3 Document-structure-aware chunking | PARTIAL | `agents/rag/api_knowledge.py:1` | Code/API chunking exists; no Markdown/HTML splitters | — | P3 |
| 21 | 16.4.4 Parent-child chunking | ABSENT | — | — | — | P3 |
| 22 | 16.4.5 Chunk-size guidelines | ABSENT | — | No documented sizing policy | — | P3 |
| 23 | 16.5 Advanced RAG Patterns | PARTIAL | `agents/rag/rerank.py:1`, `agents/rag/multimodal_fusion.py:1` | Rerank + multimodal fusion only | — | P3 |
| 24 | 16.5.1 Query Transformation (HyDE / step-back / multi-query) | ABSENT | — | — | — | P3 |
| 25 | 16.5.2 Re-Ranking | PRESENT | `agents/rag/rerank.py:1` (uses `memory/error_notebook.py` as a signal, `rerank.py:27`) | — | — | — |
| 26 | 16.5.3 Contextual Compression | PARTIAL | `agents/context/manager.py:361` (`_evict_middle`) | History eviction, not LLM extraction of relevant spans | — | P3 |
| 27 | 16.5.4 Self-RAG | ABSENT | — | — | — | P3 |
| 28 | 16.5.5 CRAG (corrective) | PARTIAL | `agents/rag/rerank.py` | No retrieval evaluator / web fallback | — | P3 |
| 29 | 16.5.6 Adaptive RAG (complexity routing) | PARTIAL | `core/routing.py:1` routes **models** by task class, not retrieval strategy | The classifier exists; it just doesn't gate retrieval | Reuse `routing.py`'s classifier to gate retrieval | P3 |
| 30 | 16.5.7 Graph RAG | ABSENT | — | — | — | P3 |
| 31 | 16.5.8 RAG-Fusion | ABSENT | — | — | — | P3 |
| 32 | 16.6 REFRAG (efficient decoding) | N/A | — | Inference-server concern; HarnessCAD does not host the model (`agents/llm/litellm_backend.py:1`) | — | — |
| 33 | 16.7 Agentic RAG | ABSENT | — | The agent never decides to retrieve; retrieval is not in its action space | Expose `retrieve` as a CISP-adjacent tool (`io/surfaces/mcp/tools.py:1` already has the non-op tool slot) | P2 |
| 34 | 16.7.1 Limits of static RAG | N/A | — | Motivational | — | — |
| 35 | 16.7.2 Agentic RAG architecture | ABSENT | — | — | — | P2 |
| 36 | 16.7.3 Multi-source routing | PARTIAL | `core/routing.py:1` | Model routing, not knowledge-source routing | — | P3 |
| 37 | 16.7.4 Full agentic RAG implementation | ABSENT | — | — | — | P3 |
| 38 | 16.7.5 Tool-augmented RAG | PARTIAL | `io/surfaces/mcp/tools.py:684` (`resources()`) exposes model state as MCP resources | Resources are *state*, not *knowledge*; no doc corpus is exposed | — | P2 |
| 39 | 16.7.6 Search-R1 (RL-trained retrieval) | N/A | — | Sibling agent's brief (RL) | — | — |
| 40 | 16.8 Evaluation | ABSENT | — | No retrieval eval at all | — | P2 |
| 41 | 16.8.1 Retrieval metrics (Recall@K/MRR/NDCG) | ABSENT | — | `eval/bench` (200 modules) has no retrieval benchmark | Add Recall@k over the exemplar corpus | P2 |
| 42 | 16.8.2 Generation metrics (faithfulness) | PARTIAL | `eval/verifiers/vlm_judge.py:1`, `eval/quality/` | Geometry-grounded, not retrieval-grounded | — | P3 |
| 43 | 16.8.3 RAGAs framework | N/A | — | Third-party dependency; product is stdlib-only by policy (`registry.py:31`) | — | — |
| 44 | 16.8.4 Common failure modes (context poisoning, lost-in-middle) | **PRESENT (ahead)** | `eval/verifiers/soundness.py:1-63` | HarnessCAD *measured* context poisoning (`assets/pressure/report.md:28`) and built a policy against it | — | — |
| 45 | 16.9 Production considerations | PARTIAL | `agents/rag/index.py` | In-memory only | — | P3 |
| 46 | 16.9.1 Embedding model selection | ABSENT | — | See #12 | — | P2 |
| 47 | 16.9.2 Vector database comparison | ABSENT | — | No vector store | — | P3 |
| 48 | 16.9.3 Latency optimization | ABSENT | — | — | — | P3 |
| 49 | 16.9.4 Incremental indexing & versioning | ABSENT | — | — | — | P3 |
| 50 | 16.10 RAG + fine-tuning synergy | N/A | — | Sibling agent (training) | — | — |
| 51 | 16.10.1 When to combine | N/A | — | Sibling agent | — | — |
| 52 | 16.10.2 RAFT | N/A | — | Sibling agent | — | — |
| 53 | 16.10.3 Joint retriever-generator training | N/A | — | Sibling agent | — | — |
| 54 | 16.11 Comprehensive RAG comparison | N/A | — | Reference table | — | — |

### H3 — Chapter 17: Agentic Memory Systems

**This is the chapter the harness fails hardest, and not for the reason the brief assumed.**
A four-type memory store *exists and is tested* (`agents/memory/store.py:114`). It is simply
not connected to the loop: `grep` for `MemoryStore` under `core/`, `agents/agent/` and
`eval/pressure/` returns **nothing**. The agent that builds parts has no memory; the modules
that use `MemoryStore` are `domain/library/catalog.py:21`, `data/datagen/generators.py:70`,
`agents/rag/rerank.py:27` and `eval/reliability/fallback.py:34` — a parts catalogue, a dataset
generator, a reranker and a fallback picker. None of them is the agent.

| # | Section | Verdict | Where | Gap | Fix | Pri |
|---|---|---|---|---|---|---|
| 55 | 17.1 Why agents need memory | PARTIAL | `agents/memory/store.py:1-23` (states the case correctly) | Stated, not honoured: the harness re-solves every brief cold | — | P0 |
| 56 | 17.2 Taxonomy of memory types | **PRESENT** | `agents/memory/store.py:114-120` — `working` / `episodic` / `semantic` / `procedural`, exactly the book's four | — | — | — |
| 57 | 17.2.1 Working memory | PRESENT | `core/state/opdag.py:1` (authoritative), `agents/memory/store.py:117` (scratchpad) | — | — | — |
| 58 | 17.2.2 Episodic memory | PRESENT | `agents/memory/store.py:81` (`Episode`: brief → ops → outcome → digest), `:141` `recall_episodic` | Never written to by the harness | Call `add_episodic` on every `HarnessRun` | P0 |
| 59 | 17.2.3 Semantic memory | PRESENT | `agents/memory/store.py:159` | Written only by `strategies/reflexion.py` | — | P1 |
| 60 | 17.2.4 Procedural memory (skills) | **PRESENT (ahead)** | `agents/memory/skills.py:1` — Voyager-style, **execution-verified** `SkillLibrary`, monotonic (`add_verified` admits a skill only if its expanded ops verify) | Not consulted by `Planner` | Retrieve top-k skills into the prompt | P0 |
| 61 | 17.3 Memory architectures | PARTIAL | `agents/memory/` | Flat store only | — | P2 |
| 62 | 17.3.1 RAG-based memory | PARTIAL | `agents/memory/store.py:141` uses `Similarity`, not a vector index | — | — | P2 |
| 63 | 17.3.2 Summarization-based memory | PARTIAL | `agents/context/manager.py:445` (`feature_tree_summary`) | Summarises *state*, not *history* | — | P2 |
| 64 | 17.3.3 Graph-based memory | ABSENT | — | (`core/state/opdag.py` is a DAG of ops, not a knowledge graph) | — | P3 |
| 65 | 17.3.4 Key-value memory networks | N/A | — | Differentiable memory; training-side | — | — |
| 66 | 17.3.5 MemGPT / virtual context tiers | PARTIAL | `agents/context/progressive_tiers.py:1` | Tiers exist for *context assembly*, not for hot/warm/cold memory paging | Unify `progressive_tiers` with `memory/store` | P2 |
| 67 | 17.4 Memory operations | PARTIAL | `agents/memory/store.py:123-170` | write/read present; update/reflect absent from the store | — | P1 |
| 68 | 17.4.1 Write (importance, contradiction detection) | PARTIAL | `agents/memory/store.py:133` | `add_episodic` is unconditional — no importance threshold, no surprise, no contradiction check | Add `importance` + dedup at write | P1 |
| 69 | 17.4.2 Read/retrieve (temporal decay, recency) | **INCOHERENT** | `agents/memory/decay.py:1` implements Ebbinghaus reinforced decay **but `MemoryStore.recall_episodic` (`store.py:141`) never calls it** | Two half-systems | Attach `decay` to `recall_episodic` scoring | P1 |
| 70 | 17.4.3 Update (conflict resolution, consolidation) | ABSENT | — | No consolidation, no LRU, no eviction in `MemoryStore` | — | P2 |
| 71 | 17.4.4 Reflect (meta-cognition) | PRESENT | `eval/reliability/strategies/reflexion.py:1` — full read-act-reflect-write over `MemoryStore` | Lives in `eval/`, not in the harness; `AgentHarness` cannot reach it | Move/expose reflexion as a harness policy | P0 |
| 72 | 17.5 Memory for multi-turn conversations | PARTIAL | `agents/agent/edit_session.py:1` | Multi-turn editing exists; no cross-session continuity | — | P2 |
| 73 | 17.5.1 User modeling / preference tracking | ABSENT | — | No user model (a CAD user's unit/standard/material preferences are exactly what `semantic` memory is for) | Persist a user profile in `semantic` | P2 |
| 74 | 17.5.2 Session continuity | ABSENT | — | `MemoryStore.save/load` exists (`store.py:182`) and nothing calls it | Load at session start, save at end | P1 |
| 75 | 17.5.3 Personalization | ABSENT | — | — | — | P3 |
| 76 | 17.6 Memory for multi-agent systems | PARTIAL | `agents/agents/blackboard.py:1` | Blackboard exists; no product path uses it | — | P3 |
| 77 | 17.6.1 Shared memory pools | PARTIAL | `agents/agents/blackboard.py` | — | — | P3 |
| 78 | 17.6.2 Blackboard architecture | PRESENT | `agents/agents/blackboard.py:1` (thread-safe, confidence-weighted conflict resolution) | Unwired | — | P3 |
| 79 | 17.6.3 Consensus / conflict in shared knowledge | PARTIAL | `agents/agents/blackboard.py` (confidence wins), `agents/exploration/variant_consensus.py:1` | — | — | P3 |
| 80 | 17.7 Training memory with RL | N/A | — | Sibling agent (RL) | — | — |
| 81 | 17.7.1 Reward signals for memory ops | N/A | — | Sibling agent | — | — |
| 82 | 17.7.2 Learning what to remember | N/A | — | Sibling agent | — | — |
| 83 | 17.7.3 Memory-augmented policy optimization | N/A | — | Sibling agent | — | — |
| 84 | 17.8 Comparison of memory approaches | N/A | — | Reference table | — | — |
| 85 | 17.9 Evaluating memory systems | ABSENT | — | Zero memory metrics anywhere in `eval/` | — | P2 |
| 86 | 17.9.1 Evaluation dimensions (extraction, multi-session, temporal, updates, abstention) | ABSENT | — | — | — | P2 |
| 87 | 17.9.2 Benchmarks (LongMemEval, LOCOMO) | N/A | — | Conversational benchmarks; a CAD equivalent would have to be built | Build "does episodic recall raise solve rate on a repeat brief?" | P1 |
| 88 | 17.9.3 Metrics (recall, precision, staleness, write selectivity) | ABSENT | — | — | — | P2 |
| 89 | 17.10 Implementation patterns | PARTIAL | `agents/memory/store.py` | — | — | P2 |
| 90 | 17.10.1 Vector store memory with embeddings | ABSENT | — | Similarity is lexical (`store.py:51`) | — | P2 |
| 91 | 17.10.2 Hierarchical memory manager (hot/warm/cold) | ABSENT | — | — | — | P2 |
| 92 | 17.10.3 Memory-augmented agent loop (read-act-reflect-write) | **ABSENT in the harness** | `core/harness.py:132` `run()` — no memory read, no memory write | The single most consequential gap in this audit | See gap #2 | **P0** |
| 93 | 17.11 Recent advances | PARTIAL | — | — | — | P3 |
| 94 | 17.11.1 CoALA | PARTIAL | `agents/memory/store.py:1` mirrors the taxonomy; the *decision cycle* half is missing | — | P1 |
| 95 | 17.11.2 Mem0 (automatic extraction) | ABSENT | — | Nothing extracts facts from a run | — | P2 |
| 96 | 17.11.3 Sleep-time compute | ABSENT | — | `agents/exploration/` is search, not offline consolidation | Consolidate episodes into skills between runs — a natural fit for `SkillLibrary` | P2 |
| 97 | 17.11.4 A-MEM (Zettelkasten) | ABSENT | — | — | — | P3 |
| 98 | 17.12 Summary | N/A | — | — | — | — |

### H3 — Chapter 18: Agent Harness (the centrepiece — graded in §2)

| # | Section | Verdict | Where | Gap | Fix | Pri |
|---|---|---|---|---|---|---|
| 99 | 18.1 What is an agent harness (reasoning/execution/memory/communication/observability separation) | PARTIAL | `core/harness.py:91` | 4 of 5 concerns present; **memory** absent from the harness's collaborator set (`harness.py:102-123`) | Add a `memory=` collaborator | P0 |
| 100 | 18.2 Context window management | PARTIAL | `agents/context/manager.py:168` | `ContextManager` is a real budget manager, but `AgentHarness` only calls `preflight()` and **logs** the report (`harness.py:284-309`); it never calls `assemble()` and never sheds load | Make the harness assemble through the manager | P1 |
| 101 | 18.2.1 Context budget problem (C ≥ S+M+T+H+R) | PRESENT | `agents/context/manager.py:5`, `:88-96` — the book's exact five-way partition | — | — | — |
| 102 | 18.2.2 Context allocation strategies | PRESENT | `agents/context/manager.py:79` (`BudgetReport`), `:229` (`preflight`) | — | — | — |
| 103 | 18.2.3 Context compression | PARTIAL | `agents/context/manager.py:361` (`_evict_middle`) | Eviction, not summarization; no summarizer model | Add a summarizing compressor | P2 |
| 104 | 18.2.4 Sliding window / pinned head | PRESENT | `agents/context/manager.py:29-30` (pinned system + first user message, stable prefix for prompt caching) | — | — | — |
| 105 | 18.2.5 Recursive context decomposition (RLM) | ABSENT | — | — | Not needed at current context sizes | P3 |
| 106 | 18.2.6 Token counting & budget monitoring | **PARTIAL — real risk** | `agents/context/manager.py:60` `HeuristicCounter` | The book (H3:3931) is explicit: use the model's exact tokenizer; a heuristic is off by 20–40% on code/JSON — and CISP ops are JSON | Inject a real tokenizer via the existing `TokenCounter` protocol (`manager.py:47`) | P1 |
| 107 | 18.3 Prompt architecture | PARTIAL | `agents/agent/system_prompt.py:81` | Prompt is code, but not versioned and not registry-backed | — | P2 |
| 108 | 18.3.1 System prompt design (persona/capabilities/constraints/output format) | **PRESENT** | `agents/agent/system_prompt.py:19` (ROLE), `:40` (vocabulary, *generated from the op registry so it cannot drift*), `:48` (RULES), `:70` (OUTPUT_CONTRACT) — all four of the book's sections | — | — | — |
| 109 | 18.3.2 Dynamic prompt assembly (blocks, prompt registry, semver) | PARTIAL | `agents/agent/planner.py:84` builds by string concat | No prompt registry, no versioning | — | P2 |
| 110 | 18.3.3 Few-shot management | **INCOHERENT** | `agents/context/exemplar_prompt.py:1` and `agents/rag/exemplar_select.py:1` exist; `Planner.build_messages` uses **neither** | Exemplar selection is built and unreachable from the agent | Wire exemplar selection into `Planner` | P1 |
| 111 | 18.3.4 Tool descriptions (5 components; when-NOT-to-use) | **INCOHERENT** | `io/surfaces/mcp/tools.py:1-12` builds the book's exact 5-component descriptions per op — **but the `Planner` exposes one tool, `emit_ops`, with a one-line description** (`agents/agent/planner.py:41-55`) | The good tool descriptions are shipped to *external* MCP clients and withheld from our own model | Have `Planner` use `ToolCatalog.to_mcp()` | **P0** |
| 112 | 18.4 Tool integration and execution | PARTIAL | `io/surfaces/mcp/tools.py`, `eval/reliability/executor.py:1` | Executor exists but `AgentHarness._dispatch` (`harness.py:323`) defaults to `session.apply_ops` — i.e. **no gate, no approval, no retry** on the default path | Default the harness to the `ToolExecutor` | P0 |
| 113 | 18.4.1 Tool definition schemas | PRESENT | `io/surfaces/mcp/tools.py:181` (`to_mcp`), typed params from op dataclasses | — | — | — |
| 114 | 18.4.2 Tool selection & routing (retrieval-augmented; large tool libraries) | PARTIAL | `agents/agent/tool_knowledge.py:1` | No retrieval over tools; the whole op vocabulary is always in the prompt (`system_prompt.py:40`) — fine at ~12 ops, not at 100 | — | P3 |
| 115 | 18.4.3 Tool output processing (parse, truncate, normalise errors, retry) | PARTIAL | `eval/reliability/executor.py:74` (clipping + note) | Not on the harness's default dispatch path | — | P1 |
| 116 | 18.4.4 Sandboxing and safety | PARTIAL | `governance/security/tool_gate.py:1` (allowlist + trust tier + injection patterns), `eval/reliability/guardrails.py:1` (pre-apply hard gate), `io/backends/external.py:1` (subprocess) | No container/seccomp isolation for OpenSCAD/FreeCAD/Blender subprocesses; the book (H4:521) calls defence-in-depth non-optional | Containerise the external backends | P1 |
| 117 | 18.4.5 MCP | **PRESENT** | `io/surfaces/mcp/server.py:1`, protocol `2025-11-25` + 2 back-compat revisions (`server.py:52`), stdio transport (`mcp/stdio.py`), JSON-RPC (`mcp/jsonrpc.py`) | See §4 incoherence #6 (the soundness tier is stripped on the wire) | — | P1 |
| 118 | 18.5 Orchestration patterns | PARTIAL | `core/harness.py:132` | ReAct only | — | P2 |
| 119 | 18.5.1 ReAct loop | **PRESENT** | `core/harness.py:148-241`: plan → loop-detect → dispatch → verify → contract → repair, `max_iterations` guard at `:112` | — | — | — |
| 120 | 18.5.2 Plan-and-execute | PARTIAL | `agents/agent/plan_envelope.py:1`, `agents/generation/design_plan.py:1` | Typed plan envelope exists; the harness does not run a plan-then-execute mode | — | P2 |
| 121 | 18.5.3 Multi-agent orchestration (supervisor / P2P / hierarchical / swarm) | PARTIAL | `agents/agents/supervisor.py:1`, `overseer.py:1`, `roles.py:1`, `vmodel_workflow.py:1` | All four exist as libraries; **no product path constructs them**. See §5 for the argument that this is correct | — | P3 |
| 122 | 18.5.4 Human-in-the-loop (approval gates, escalation, async approval) | **PARTIAL — the pieces are all there and the harness ignores them** | `io/surfaces/ui/approval.py:37` (three tiers), `:120` (`tier_for(op)`), `:176` (`DryRunPreview`), `:313` (`approval_required` UIEvent); `eval/reliability/executor.py:12` (approval gate); `agents/agent/edit_session.py:1` (propose/approve/apply) | `AgentHarness` has **no approval seam**: `_dispatch` (`harness.py:323`) goes straight to the session unless an executor is injected, and nothing in the product injects one except the ACP surface (`io/surfaces/acp/agent.py:202`) | See §6 | **P0** |
| 123 | 18.5.5 Workflow graphs / state machines | ABSENT | — | The loop is a Python `for` (`harness.py:148`); no explicit state machine, no conditional edges, no resumable graph state | Acceptable for a linear ReAct loop; revisit if plan-and-execute lands | P3 |
| 124 | 18.6 State management | PRESENT | `core/state/opdag.py:1`, `core/state/feature_tree.py:1`, `core/state/constraint_model.py:1` | — | — | — |
| 125 | 18.6.1 Conversation state | PARTIAL | `agents/agent/planner.py:84` rebuilds the message list from (brief, state, diagnostics) each turn | Deliberate and defensible (no history accretion → no lost-in-the-middle), but it means the model cannot see *what it already tried* | Feed the last N op-signatures as observations | P2 |
| 126 | 18.6.2 Task state (progress, checkpoints, rollback) | **PRESENT (ahead)** | `core/loop.py:103` (append), `:126` (`_rollback_last`), `:134` (auto checkpoint per accepted op), `:180` (`rollback(label)`), `core/state/opdag.py` | Transactional per-op rollback is stronger than the book's coarse checkpointing | — | — |
| 127 | 18.6.3 Agent state (plan, pending actions, beliefs) | PARTIAL | `agents/agent/plan_envelope.py:1`, `core/harness.py:74` (`trajectory`) | No belief store | — | P3 |
| 128 | 18.6.4 Persistent state (cross-session) | **ABSENT** | `agents/memory/store.py:182` `save()` / `:196` `load()` exist and **nothing calls them** | Every run starts cold | See gap #2 | **P0** |
| 129 | 18.7 Error handling and recovery | PARTIAL | `core/loop.py:84`, `eval/reliability/` | — | — | P1 |
| 130 | 18.7.1 Retry strategies (backoff, fallback models, graceful degradation) | PARTIAL | `core/routing.py:1` implements a sequential fallback chain + cost tally — **and the harness takes a single `LLM`** (`agents/agent/planner.py:77`) | A `RoutingLLM` satisfies the `LLM` protocol, so this is a one-line wiring change that nobody made | Inject `routing.RoutingLLM` in `core/pipeline.py:28` | P1 |
| 131 | 18.7.2 Loop detection | **PRESENT** | `eval/reliability/loopdetect.py:1` (sliding-window hash of `(op_tag, sorted-args)`), consumed at `core/harness.py:311` *pre-dispatch* so an oscillation never mutates state | — | — | — |
| 132 | 18.7.3 Graceful failure (partial results, reason, resume) | PARTIAL | `core/harness.py:243` (`stop_reason`), `io/gate.py:1` (refuse-with-reason) | No user-facing "here is what I got and why I stopped" rendering; no resume | — | P2 |
| 133 | 18.7.4 Observability (traces/logs/metrics) | **PRESENT (ahead)** | `core/trace.py:38` (`EVENT_KINDS`), `:99` (`JsonlTracer`); `core/observe.py:163` (`SpanCollector`), `:417` (`Metrics`), `:538` (`FailureTaxonomy`), `:737` (`Replayer`), `:804` (`replay`), `:834` (`report`) — plus Wilson intervals at `:279` | The book asks for traces/logs/metrics + replay. HarnessCAD has all three **and** deterministic replay from the event log | — | — |
| 134 | 18.8 Scaling and production concerns | ABSENT | — | Single-process, synchronous, no queue | — | P3 |
| 135 | 18.8.1 Latency optimization (parallel tool calls, streaming, prompt caching, speculative) | PARTIAL | `agents/context/manager.py:29` (stable prefix for prompt caching), `agents/llm/base.py` (`stream`) | Ops are applied strictly serially (`core/loop.py:89`) — correct, because ops are order-dependent; but the *planner* call is also serial with no streaming consumption | — | P3 |
| 136 | 18.8.2 Cost management (token budgets, model routing, caching) | PARTIAL | `core/routing.py:1` (classify-then-route + cost tally), `eval/pressure/cache.py:1` | Unwired to the harness (see #130) | | P1 |
| 137 | 18.8.3 Rate limiting and queuing | ABSENT | — | — | — | P3 |
| 138 | 18.8.4 Evaluation in production (A/B, canary, shadow, LLM-judge) | PARTIAL | `eval/pressure/` (offline A/B of two loops), `eval/verifiers/vlm_judge.py:1`, `governance/research/model_promotion.py:1` | No online A/B, canary or shadow; but the offline arm-vs-arm design (`eval/pressure/loops.py:1-36`) is a textbook A/B | — | P3 |
| 139 | 18.9 Framework comparison | N/A | — | The book's own decision rule ("build custom when the harness is a core differentiator", H3:4649) selects *custom*, which is what this is | — | — |
| 140 | 18.10 Implementation: production agent harness | PRESENT | `core/harness.py:1-413` | Missing vs. the book's reference implementation: approval callback, output truncation, per-tool retry/backoff (all exist elsewhere — see #112, #115, #122) | — | P0 |
| 141 | 18.11 Summary (context is finite; prompts are code; tools are actuators; orchestration is not one-size; state first-class; errors inevitable; observability not optional) | PARTIAL | — | 5 of 7 honoured. "Prompts are code" — yes but unversioned. "Tools are the agent's actuators" — the agent is given one blunt actuator (`emit_ops`) while the good ones sit in the MCP catalogue | — | P0 |

### H4 — Chapter 19: Agent Design Patterns

| # | Section | Verdict | Where | Gap | Fix | Pri |
|---|---|---|---|---|---|---|
| 142 | 19.1 Workflow patterns | PARTIAL | `core/pipeline.py:1` | Pipeline is a thin chain | — | P3 |
| 143 | 19.1.1 Prompt chaining | PRESENT | `core/pipeline.py:1` (brief → planner → session → STEP), `agents/generation/dual_loop.py:1` | — | — | — |
| 144 | 19.1.2 Routing | PRESENT | `core/routing.py:1` (classify-then-route with cost tally) | Unwired to the harness | See #130 | P1 |
| 145 | 19.1.3 Parallelization (sectioning / voting) | PRESENT | `eval/reliability/strategies/best_of_n.py:1`, `agents/exploration/variant_consensus.py:1`, `agents/exploration/tournament.py:1`, `elo.py:1` | Not reachable from `AgentHarness` | — | P2 |
| 146 | 19.1.4 Orchestrator-workers | PARTIAL | `agents/agents/supervisor.py:1` | Unwired | — | P3 |
| 147 | 19.1.5 Evaluator-optimizer | **PRESENT — this is what the harness *is*** | `core/harness.py:148` (generate → verify → feed diagnostics → regenerate), `eval/reliability/repair_loop.py:1` | The book's warning applies exactly: an evaluator-optimizer is only as good as its evaluator, which is the `-8.3` result | See §2 grade | P0 |
| 148 | 19.2 Autonomous agent patterns | PRESENT | `core/harness.py:91` | — | — | — |
| 149 | 19.2.1 ReAct | PRESENT | `core/harness.py:148` | — | — | — |
| 150 | 19.2.2 Planning agents (plan-then-execute / adaptive / continuous / hierarchical) | PARTIAL | `agents/agent/plan_envelope.py:1`, `agents/generation/design_plan.py:1` | The harness re-plans every iteration = "continuous" (the book's most expensive mode, H4:127) and never says so | Consider adaptive replanning (only on failure) to cut model calls | P2 |
| 151 | 19.2.3 Reflection / self-critique (Reflexion) | PRESENT | `eval/reliability/strategies/reflexion.py:1` | Not a harness policy | See gap #2 | P0 |
| 152 | 19.2.4 Tool-use patterns (single/multi/sequential/nested/fallback) | PARTIAL | `agents/agent/tool_trajectory.py:1`, `eval/reliability/fallback.py:1` | The harness's model emits **one** tool call carrying the whole op array; there is no per-op tool loop | Legitimate design choice (ops are a transaction) — but it means the model gets no per-op observation | P1 |
| 153 | 19.3 Design principles (keep it simple; transparency; **provide good tools**; plan for failure; structured outputs; test with diverse inputs) | PARTIAL | `agents/llm/structured.py:1` (structured outputs), `core/trace.py` (transparency), `eval/pressure/briefs.py` (diverse inputs) | "Provide good tools" is violated: see #111 | — | P0 |
| 154 | 19.4 Pattern selection guide | N/A | — | Decision table | — | — |

### H4 — Chapter 20: Agentic Environments and Benchmarks

The brief asks whether the book's environment model supports a seventh, GUI-driving backend.
**It does — but `GeometryBackend` is not the book's environment model.** See §5.

| # | Section | Verdict | Where | Gap | Fix | Pri |
|---|---|---|---|---|---|---|
| 155 | 20.1 Why agents need environments | PRESENT | `io/surfaces/mcp/gym.py:1` | — | — | — |
| 156 | 20.2 Environment design principles (4 axes) | PARTIAL | `io/surfaces/mcp/gym.py:1` | Obs/action/reward defined; episode structure implicit | — | P2 |
| 157 | 20.2.1 Observation space design (text / JSON / multimodal / **hybrid**) | **PRESENT (ahead)** | `io/surfaces/mcp/gym.py:10-16`: JSON geometry summary **+ per-view render availability, with image bytes deliberately kept out of the obs** — the book's hybrid pattern, with the context-blowup failure already engineered around | — | — | — |
| 158 | 20.2.2 Action space design | PRESENT | `io/surfaces/mcp/gym.py:8` (action = CISP ops), `core/cisp/ops.py:1` | — | — | — |
| 159 | 20.2.3 Reward signal design (aligned / learnable / tamper-proof) | PRESENT | `io/surfaces/mcp/tools.py:78` (`reward_from_apply` — execution-based, the book's "tamper-proof" class) | Reward is as sound as the verifier fleet — the `-8.3` result *is* a reward-validity result | Precision harness (gap #4) | P0 |
| 160 | 20.2.4 Episode structure | PARTIAL | `core/harness.py:112` (`max_iterations`) | Early termination present; no budget-based open-ended mode | — | P3 |
| 161 | 20.2.5 Difficulty curriculum / adaptive environments | PRESENT | `agents/generation/difficulty_tiers.py:1`, `agents/generation/training_schedule.py:1` | Not used by the harness (dataset-side) | — | P3 |
| 162 | 20.3 Types of environments | PARTIAL | `io/backends/` (6 kernels) | — | — | — |
| 163 | 20.3.1 Code execution sandboxes | PARTIAL | `io/backends/openscad.py:1`, `freecad_driver.py:1`, `blender.py:1`, `external.py:1` | Subprocesses, not containers (see #116) | — | P1 |
| 164 | 20.3.2 Web environments | N/A | — | Not a web agent | — | — |
| 165 | 20.3.3 Computer-use environments (screenshot + a11y tree, pyautogui-style actions) | **ABSENT — and this is the planned 7th backend** | — | The book's model **is** the right one; `GeometryBackend` (`io/backends/base.py:41`) is not, because it demands `state_digest()` (content hash — a GUI has no content hash), a synchronous non-mutating `apply()` on reject (a GUI mutates before you know), and returns no perception | See §5: add an `Environment` tier above `GeometryBackend`, reuse `mcp/gym.py:1` as the seam | **P0** |
| 166 | 20.3.4 Software engineering environments | N/A | — | Different domain | — | — |
| 167 | 20.3.5 Scientific research environments | N/A | — | Different domain | — | — |
| 168 | 20.3.6 Game and simulation environments | N/A | — | Different domain | — | — |
| 169 | 20.3.7 Multi-agent environments | ABSENT | — | — | — | P3 |
| 170 | 20.4 OpenEnv standardized interfaces | **PRESENT** | `io/surfaces/mcp/gym.py:1` — `reset()/step()/state()/render()/close()`, exactly OpenEnv's shape | Not Docker-served; not registered | Wrap `gym.py` in the OpenEnv `HTTPEnvServer` shape | P2 |
| 171 | 20.4.1 Standardized agent–environment interface | PRESENT | `io/surfaces/mcp/gym.py:1` | Typed actions/observations per-env: yes (`CodeAction`-analogue = `Op`) | — | — |
| 172 | 20.4.2 Environment registries and discovery | PARTIAL | `io/backends/__init__.py`, `io/adapters/registry.py:1`, `registry.py:1` | Local registries only; no HF Spaces / Docker discovery | — | P3 |
| 173 | 20.4.3 Compositional environments (MCP-wrapped tools as env) | PRESENT | `io/surfaces/mcp/gym.py` + `io/surfaces/mcp/tools.py` share one catalogue | — | — | — |
| 174 | 20.4.4 Environment versioning and reproducibility (semver, image pinning, seeded determinism, leaderboard snapshots) | **PRESENT (ahead)** | Determinism is structural, not seeded: `core/loop.py:72` (`_make_run_id` — no wall clock), `core/harness.py:126` (run_id from the brief), `registry.py:19` (byte-identical index), `io/backends/base.py:1` ("replaying the same ops must yield the same digest") | No Docker pinning of external kernels | Pin the OpenSCAD/FreeCAD versions | P2 |
| 175 | 20.5 Building custom environments | PRESENT | `io/surfaces/mcp/gym.py` | — | — | — |
| 176 | 20.5.1 Gymnasium-style API | PRESENT | `io/surfaces/mcp/gym.py:1` | — | — | — |
| 177 | 20.5.2 Reward function engineering (execution-based / LLM-judge / rubric) | PRESENT | `io/surfaces/mcp/tools.py:78`, `eval/verifiers/vlm_judge.py:1`, `eval/quality/` | — | — | — |
| 178 | 20.5.3 State management and checkpointing | PRESENT | `core/state/opdag.py:1`, `core/loop.py:177` | — | — | — |
| 179 | 20.5.4 Parallelization for data collection | PARTIAL | `eval/pressure/runner.py:1` (concurrent processes), `data/dataengine/` | No vectorised envs | — | P3 |
| 180 | 20.6 Environment–agent interface patterns | PARTIAL | `io/surfaces/mcp/gym.py` (structured JSON) | Streaming interaction absent | — | P3 |
| 181 | 20.7 Evaluation harness design | PRESENT | `eval/pressure/runner.py:1`, `eval/bench/` (200 modules), `eval/selftest/` | — | — | — |
| 182 | 20.7.1 Deterministic vs. stochastic environments | PRESENT | `eval/pressure/loops.py:1-36` (temperature 0.0, fixed seed, deterministic backend) | — | — | — |
| 183 | 20.7.2 Held-out test environments | PARTIAL | `eval/pressure/briefs.py:1` | 12 of 28 briefs used; no train/test split declared | Declare a held-out brief set | P2 |
| 184 | 20.7.3 Cross-environment generalization | PARTIAL | `io/backends/` (6 backends behind one protocol makes this *possible*) | No cross-backend transfer measurement | Run the pressure suite across `frep` **and** `cadquery` | P1 |
| 185 | 20.7.4 Human baseline collection | ABSENT | — | No human solve-rate baseline on the briefs; without it "33% vs 25%" has no ceiling | Collect a human baseline on the 12 briefs | P2 |
| 186 | 20.8 Code example: minimal custom env | PRESENT | `io/surfaces/mcp/gym.py:1` (explicitly the book's `FileEditEnv` with "pytest passes" swapped for "verifier passes" — `gym.py:5-6`) | — | — | — |
| 187 | 20.9 Comparison of major environments | N/A | — | Reference table | — | — |
| 188 | 20.10 Summary | N/A | — | — | — | — |

### H4 — Chapter 21: Model Context Protocol

| # | Section | Verdict | Where | Gap | Fix | Pri |
|---|---|---|---|---|---|---|
| 189 | 21.1 The tool-integration problem (N×M → N+M) | PRESENT | `io/surfaces/mcp/__init__.py`, `io/surfaces/mcp/server.py:1` | — | — | — |
| 190 | 21.2 Architecture overview | PRESENT | `io/surfaces/mcp/server.py:1` | Server only; HarnessCAD is not an MCP **client** | See #197 | P1 |
| 191 | 21.2.1 Three-role model (host/client/server) | PARTIAL | `io/surfaces/mcp/server.py:1` (server), `io/surfaces/acp/agent.py:190` (notes "the stub agent runs no MCP") | We are a server, never a host or client | — | P1 |
| 192 | 21.2.2 Transport layers | PARTIAL | `io/surfaces/mcp/stdio.py:1` (stdio) | No Streamable HTTP | Add HTTP transport | P2 |
| 193 | 21.2.3 Protocol lifecycle (initialize / negotiate / operate / shutdown) | PRESENT | `io/surfaces/mcp/server.py:75` (version negotiation), `:134` (`handle`) | — | — | — |
| 194 | 21.2.4 Stateful sessions vs. stateless requests | PRESENT | `io/surfaces/mcp/server.py:1` (wraps a live `HarnessSession`) | — | — | — |
| 195 | 21.2.5 Full architecture diagram | N/A | — | Diagram | — | — |
| 196 | 21.3 Core primitives | PRESENT | `io/surfaces/mcp/tools.py:680` (`to_mcp`), `:684` (`resources`), `:719` (`prompts`) | Sampling absent (see #200) | — | P3 |
| 197 | 21.3.1 Tools | PRESENT | `io/surfaces/mcp/tools.py:1` — CISP ops + `measure`/`query`/`verify`/`export`/`reset`/`render` | — | — | — |
| 198 | 21.3.2 Resources | PRESENT | `io/surfaces/mcp/tools.py:684` (model-state observations, `cad://model/tree`) | No subscriptions | — | P3 |
| 199 | 21.3.3 Prompts | PRESENT | `io/surfaces/mcp/tools.py:719` (op templates) | — | — | — |
| 200 | 21.3.4 Sampling (server → client inference) | ABSENT | — | — | Low value here | P3 |
| 201 | 21.4 Protocol specification (JSON-RPC 2.0) | PRESENT | `io/surfaces/mcp/jsonrpc.py:1` | — | — | — |
| 202 | 21.4.1 JSON-RPC message format | PRESENT | `io/surfaces/mcp/jsonrpc.py` | — | — | — |
| 203 | 21.4.2 Capability negotiation handshake | PRESENT | `io/surfaces/mcp/server.py:53,75` | — | — | — |
| 204 | 21.4.3 Error handling (standard codes) | PRESENT | `io/surfaces/mcp/jsonrpc.py` (`INVALID_PARAMS`, `METHOD_NOT_FOUND`, `RESOURCE_NOT_FOUND`, `INTERNAL_ERROR`) | — | — | — |
| 205 | 21.4.4 Progress reporting | ABSENT | — | Long ops (a 30 s FreeCAD regen) report nothing | Add `notifications/progress` | P2 |
| 206 | 21.5 Tool definition and discovery | PRESENT | `io/surfaces/mcp/tools.py:181` | — | — | — |
| 207 | 21.5.1 Tool schema format | PRESENT | `io/surfaces/mcp/tools.py:181` | — | — | — |
| 208 | 21.5.2 Dynamic tool registration | ABSENT | — | Catalogue is static from `cisp.ops._REGISTRY` | Fine — the op set *should* be static | P3 |
| 209 | 21.5.3 Tool annotations (readOnly/destructive/idempotent/openWorld) | **PRESENT** | `io/surfaces/mcp/annotations.py:39`, and they are **load-bearing**: `io/surfaces/ui/approval.py:107` (`tier_from_annotations`) turns them into approval tiers | — | — | — |
| 210 | 21.6 Security model | PARTIAL | `governance/security/tool_gate.py:1`, `governance/security/policy.py:1` | — | — | P1 |
| 211 | 21.6.1 Trust hierarchy (host > client > server > external) | PARTIAL | `governance/security/tool_gate.py:16` (`TrustTier`) | Not enforced at the MCP server boundary | — | P2 |
| 212 | 21.6.2 User consent | PARTIAL | `io/surfaces/ui/approval.py:1` | The MCP server does not require consent for destructive tools — it has the annotations and does not gate on them (`io/surfaces/mcp/server.py:134`) | Gate `tools/call` on `ApprovalTier.REQUIRE` | **P0** |
| 213 | 21.6.3 Input validation and sanitization | PRESENT | `io/surfaces/mcp/tools.py:1` (typed errors: `ToolValidationError`), `core/cisp/ops.py` (`parse_op`), `eval/reliability/guardrails.py:1` | — | — | — |
| 214 | 21.6.4 Credential management | N/A | — | The server holds no third-party credentials | — | — |
| 215 | 21.6.5 Sandboxing strategies | PARTIAL | `io/backends/external.py:1` | See #116 | — | P1 |
| 216 | 21.7 Implementation patterns | PRESENT | `io/surfaces/mcp/server.py`, `stdio.py` | — | — | — |
| 217 | 21.7.1 Building an MCP server | PRESENT | `io/surfaces/mcp/server.py:1` (hand-rolled, no SDK — consistent with the stdlib-only policy) | — | — | — |
| 218 | 21.7.2 Building an MCP client | ABSENT | — | HarnessCAD cannot consume other MCP servers (e.g. a real FreeCAD MCP bridge — the code even knows one exists, `io/formats/freecad_document.py:3`) | Add a client so external CAD MCP servers become backends | P1 |
| 219 | 21.7.3 Multiple simultaneous servers | ABSENT | — | Follows from #218 | — | P2 |
| 220 | 21.7.4 Error recovery and reconnection | ABSENT | — | Follows from #218 | — | P2 |
| 221 | 21.8 The MCP ecosystem | N/A | — | Survey | — | — |
| 222 | 21.8.1 Popular MCP servers | N/A | — | Survey | — | — |
| 223 | 21.8.2 MCP in production applications | N/A | — | Survey | — | — |
| 224 | 21.8.3 Server registries and discovery | ABSENT | — | HarnessCAD's MCP server is not published to any registry | Publish it — it is genuinely good | P2 |
| 225 | 21.9 MCP vs. alternatives | N/A | — | Decision table; MCP was correctly chosen | — | — |
| 226 | 21.9.1 When to use MCP vs. custom | N/A | — | Both are done (MCP server + native `Planner` path) | — | — |
| 227 | 21.9.2 Migration paths | N/A | — | — | — | — |
| 228 | 21.10 MCP for agent training | PARTIAL | `io/surfaces/mcp/gym.py:1` | — | — | P2 |
| 229 | 21.10.1 MCP servers as RL environment interfaces | **PRESENT (ahead)** | `io/surfaces/mcp/gym.py:1` — the book poses this as an open question ("Could MCP serve as the gymnasium of tool-using LLM training?", H4:2797); HarnessCAD already built it | — | — | — |
| 230 | 21.10.2 Standardized action spaces via MCP | PRESENT | `io/surfaces/mcp/tools.py` from `cisp.ops._REGISTRY` | — | — | — |
| 231 | 21.10.3 Recording tool-use trajectories for SFT | PRESENT | `agents/agent/tool_trajectory.py:1`, `core/harness.py:74` (`trajectory`), `core/trace.py:99` (JSONL) | Never converted to SFT format | Add a `trajectory → messages` exporter | P2 |
| 232 | 21.11 Summary | N/A | — | — | — | — |

### H4 — Chapter 22: Agent Skills

| # | Section | Verdict | Where | Gap | Fix | Pri |
|---|---|---|---|---|---|---|
| 233 | 22.1 What is a skill (prompt + tools + knowledge + workflow + guardrails) | PARTIAL | `agents/memory/skills.py:1` — a skill is an op-template only (params → `list[Op]`) | Narrower than the book's skill, but *verifiable*, which the book's is not | — | P3 |
| 234 | 22.2 Skill architecture patterns | PARTIAL | `agents/memory/skills.py` | — | — | P2 |
| 235 | 22.2.1 Static skill loading | PRESENT | `agents/memory/skills.py` (`build_default_library`), `domain/library/catalog.py:21` | — | — | — |
| 236 | 22.2.2 Dynamic skill discovery (router) | PRESENT | `agents/memory/skills.py` (`find`, text-similarity routing) | Never invoked by the agent | See gap #2 | **P0** |
| 237 | 22.2.3 Hierarchical skill composition | PARTIAL | `agents/memory/skills.py` (`register` for building blocks) | No dependency DAG | — | P3 |
| 238 | 22.3 Case study: Anthropic's agent design | PARTIAL | — | — | — | — |
| 239 | 22.3.1 Core principles (start simple; workflows vs. agents; augmented LLM) | PRESENT | `core/harness.py:1-26` (composes, reimplements nothing) | — | — | — |
| 240 | 22.3.2 Building-block patterns | PARTIAL | See #143–#147 | — | — | P3 |
| 241 | 22.3.3 The Augmented LLM (model + retrieval + tools + memory) | **ABSENT** | `agents/agent/planner.py:77` — the planner takes a bare `LLM`, no retrieval, no memory, one tool | This is the book's *atomic unit* and HarnessCAD does not have it | Gap #2 | **P0** |
| 242 | 22.3.4 Practical implications ("simple loops with good tools") | PARTIAL | `core/harness.py` | The loop is simple; the tools are not good (see #111) | — | P0 |
| 243 | 22.4 Skill lifecycle (discover/select/activate/execute/deactivate/learn) | PARTIAL | `agents/memory/skills.py` (`add_verified` = the "learn" step, Voyager-style) | Discover/select/activate never happen at run time | Gap #2 | P0 |
| 244 | 22.5 Skill registries and marketplaces | PARTIAL | `domain/library/catalog.py:1`, `registry.py:1` | Local only | — | P3 |
| 245 | 22.6 Skills vs. fine-tuning | N/A | — | Sibling agent (training) | — | — |

### H4 — Chapter 23: Agent-to-Agent Communication

| # | Section | Verdict | Where | Gap | Fix | Pri |
|---|---|---|---|---|---|---|
| 246 | 23.1 Why agents must communicate | PARTIAL | `agents/a2a/__init__.py:1` | See §5: the argument for a single loop | — | — |
| 247 | 23.2 The Google A2A protocol | PRESENT | `agents/a2a/messages.py:1`, `agents/a2a/task.py:1`, `io/surfaces/a2a_server/app.py:1` | — | — | — |
| 248 | 23.2.1 Design philosophy (opaque execution, enterprise, modality-agnostic, async-first) | PARTIAL | `io/surfaces/a2a_server/card.py:45` | — | — | P3 |
| 249 | 23.2.2 Agent Cards | PRESENT | `io/surfaces/a2a_server/card.py:45-49` (skills, capabilities, streaming) | — | — | — |
| 250 | 23.2.3 Task lifecycle (submitted/working/input-required/completed/failed/rejected/canceled) | PRESENT | `agents/a2a/task.py:1` | — | — | — |
| 251 | 23.2.4 Streaming via SSE | PARTIAL | `io/surfaces/a2a_server/card.py:48` (`streaming: True`), `wire.py` | Declared; verify the transport actually streams | — | P2 |
| 252 | 23.2.5 Push notifications | ABSENT | `io/surfaces/a2a_server/card.py:48` (`pushNotifications: False`) | Honestly declared as absent | — | P3 |
| 253 | 23.2.6 Message format (TextPart/FilePart/DataPart) | PRESENT | `agents/a2a/messages.py:1` | — | — | — |
| 254 | 23.2.7 Authentication and authorization | ABSENT | `io/surfaces/a2a_server/app.py` | No auth on the A2A server; the book calls this non-negotiable (H4:4899) | Add bearer-token auth before any deployment | P1 |
| 255 | 23.3 Communication patterns | PARTIAL | `agents/a2a/` | — | — | P3 |
| 256 | 23.3.1 Request-response | PRESENT | `io/surfaces/a2a_server/handler.py:51` | — | — | — |
| 257 | 23.3.2 Streaming | PARTIAL | `io/surfaces/a2a_server/wire.py` | — | — | P3 |
| 258 | 23.3.3 Multi-turn interaction (input-required) | PARTIAL | `agents/a2a/task.py`, `agents/agent/edit_session.py:1` | The A2A `input-required` state is not used to ask a human to disambiguate a brief — which is exactly what CAD needs | See §6 | P1 |
| 259 | 23.3.4 Broadcast | ABSENT | — | — | — | P3 |
| 260 | 23.3.5 Pub-sub | ABSENT | — | — | — | P3 |
| 261 | 23.3.6 Negotiation | ABSENT | — | — | — | P3 |
| 262 | 23.3.7 Auction-based allocation | ABSENT | — | — | — | P3 |
| 263 | 23.4 Agent discovery and routing | PARTIAL | `io/surfaces/a2a_server/card.py` (we publish a card) | We never consume one | — | P3 |
| 264 | 23.4.1 Agent registries | ABSENT | — | — | — | P3 |
| 265 | 23.4.2 Capability-based routing | PARTIAL | `agents/agents/roles.py:1` | — | — | P3 |
| 266 | 23.4.3 Load balancing | ABSENT | — | — | — | P3 |
| 267 | 23.4.4 Version management | PARTIAL | `io/surfaces/mcp/server.py:53` (MCP versions negotiated); A2A card has `version` | — | — | P3 |
| 268 | 23.5 Message formats and schemas | PRESENT | `agents/a2a/messages.py:1` | — | — | — |
| 269 | 23.5.1 Structured vs. unstructured | PRESENT | `agents/a2a/messages.py` | — | — | — |
| 270 | 23.5.2 Multi-modal messages | PARTIAL | `agents/agent/attachments.py:1`, `agents/rag/multimodal_fusion.py:1` | — | — | P3 |
| 271 | 23.5.3 Context passing (minimal / summarized / private) | PARTIAL | `agents/context/manager.py:263` (`assemble`) | No privacy scoping | — | P3 |
| 272 | 23.5.4 Conversation threading / correlation IDs | **PRESENT** | `core/trace.py:38` + `core/harness.py:126` (`run_id` derived from the brief — deterministic correlation, better than a UUID for replay) | — | — | — |
| 273 | 23.6 Coordination protocols | PARTIAL | `agents/agents/` | — | — | P3 |
| 274 | 23.6.1 Contract Net Protocol | ABSENT | — | — | — | P3 |
| 275 | 23.6.2 Blackboard systems | PRESENT | `agents/agents/blackboard.py:1` | Unwired | — | P3 |
| 276 | 23.6.3 Consensus protocols | PARTIAL | `agents/exploration/variant_consensus.py:1`, `agents/exploration/elo.py:1` | — | — | P3 |
| 277 | 23.6.4 Leader election | N/A | — | Single-process | — | — |
| 278 | 23.7 A2A vs. MCP (complementary) | PRESENT | `agents/a2a/__init__.py:5` ("MCP is used for [tools]") — the distinction is understood and both surfaces exist | — | — | — |
| 279 | 23.7.1 When to use which | PRESENT | `agents/agent/tool_schema.py:17` explicitly separates the two catalogues | — | — | — |
| 280 | 23.7.2 Combined architecture | PARTIAL | `io/surfaces/a2a_server/__main__.py:50` (A2A server mints an `AgentHarness`) — but that harness has no MCP client | — | — | P2 |
| 281 | 23.8 Security and trust in multi-agent systems | ABSENT | — | See #254 | — | P1 |
| 282 | 23.8.1 Agent identity verification | ABSENT | — | — | — | P1 |
| 283 | 23.8.2 Message integrity / encryption | ABSENT | — | — | — | P2 |
| 284 | 23.8.3 Authorization scopes | PARTIAL | `governance/security/tool_gate.py:16` (`TrustTier`) | Not applied to A2A | — | P1 |
| 285 | 23.8.4 Audit trails and accountability | PRESENT | `core/trace.py:99`, `governance/audit/closure.py:1`, `core/observe.py:834` | — | — | — |
| 286 | 23.9 Implementation example | PRESENT | `io/surfaces/a2a_server/` | — | — | — |
| 287 | 23.10 Summary | N/A | — | — | — | — |

### H4 — Chapter 24: Multi-Agent Systems

Verdict for the whole chapter: **the libraries exist, nothing constructs them, and that is
the right call** (argued in §5). Rows are graded against the book, not against what I think we
should build.

| # | Section | Verdict | Where | Gap | Fix | Pri |
|---|---|---|---|---|---|---|
| 288 | 24.1 Why multiple agents | N/A | — | Argued against in §5 for this domain | — | — |
| 289 | 24.2 Multi-agent architectures | PARTIAL | `agents/agents/` | Libraries only | — | P3 |
| 290 | 24.2.1 Centralized (supervisor) | PRESENT | `agents/agents/supervisor.py:1` | Unwired | — | P3 |
| 291 | 24.2.2 Decentralized (P2P) | ABSENT | — | — | — | P3 |
| 292 | 24.2.3 Hierarchical | PARTIAL | `agents/agents/vmodel_workflow.py:1`, `vmodel_roles.py:1` | — | — | P3 |
| 293 | 24.2.4 Swarm (handoffs) | ABSENT | — | — | — | P3 |
| 294 | 24.3 Coordination mechanisms | PARTIAL | `agents/agents/blackboard.py` | — | — | P3 |
| 295 | 24.3.1 Shared state (blackboard) | PRESENT | `agents/agents/blackboard.py:1` | — | — | P3 |
| 296 | 24.3.2 Message passing | PRESENT | `agents/agents/message_tree.py:1` | — | — | P3 |
| 297 | 24.3.3 Planning and decomposition (task DAG) | PARTIAL | `agents/exploration/decomp_state.py:1`, `decomp_reward.py:1`, `agents/agent/plan_envelope.py:1` | — | — | P3 |
| 298 | 24.3.4 Voting and consensus | PRESENT | `agents/exploration/variant_consensus.py:1`, `tournament.py:1` | — | — | — |
| 299 | 24.3.5 Market-based coordination | ABSENT | — | — | — | P3 |
| 300 | 24.3.6 Stigmergy | PARTIAL | `core/state/opdag.py` (the shared op DAG *is* the environment agents would modify) | — | — | P3 |
| 301 | 24.4 Communication protocols | PARTIAL | `agents/a2a/messages.py` | — | — | P3 |
| 302 | 24.4.1 Structured message formats | PRESENT | `agents/a2a/messages.py:1` | — | — | — |
| 303 | 24.4.2 Performative types (FIPA-ACL) | ABSENT | — | — | — | P3 |
| 304 | 24.4.3 Context sharing strategies | PARTIAL | `agents/context/manager.py` | — | — | P3 |
| 305 | 24.5 Role design and specialization | PRESENT | `agents/agents/roles.py:1`, `vmodel_roles.py:1` | — | — | — |
| 306 | 24.5.1 Defining agent roles | PRESENT | `agents/agents/roles.py:1` | — | — | — |
| 307 | 24.5.2 Capability- vs. role-based assignment | PARTIAL | `agents/agents/roles.py` | — | — | P3 |
| 308 | 24.5.3 Dynamic role reassignment | ABSENT | — | — | — | P3 |
| 309 | 24.5.4 Persona design for diversity | ABSENT | — | — | — | P3 |
| 310 | 24.6 Multi-agent patterns for LLMs | PARTIAL | — | — | — | P3 |
| 311 | 24.6.1 Debate | ABSENT | — | — | — | P3 |
| 312 | 24.6.2 Reflection | PRESENT | `eval/reliability/strategies/reflexion.py:1` | Single-agent reflection, which is the right shape here | — | — |
| 313 | 24.6.3 Division of labor | ABSENT | — | — | — | P3 |
| 314 | 24.6.4 Pipeline | PRESENT | `core/pipeline.py:1` | — | — | — |
| 315 | 24.6.5 Ensemble (best-of-N) | PRESENT | `eval/reliability/strategies/best_of_n.py:1`, `mcts.py:1` | Not reachable from `AgentHarness` | Expose as a harness policy | P2 |
| 316 | 24.6.6 Teacher-student | N/A | — | Distillation; sibling agent | — | — |
| 317 | 24.6.7 Red team | **ABSENT — and this is the one multi-agent pattern the `-8.3` result argues FOR** | `agents/agents/overseer.py:1` is the nearest thing | Nothing adversarially audits the verifier fleet. The fleet's four bugs (`assets/pressure/report.md:81-108`) were found by a human | Build a red-team verifier auditor (gap #4) | **P0** |
| 318 | 24.7 Training multi-agent systems with RL | N/A | — | Sibling agent (RL) | — | — |
| 319 | 24.7.1 Markov game formulation | N/A | — | Sibling agent | — | — |
| 320 | 24.7.2 Independent learning | N/A | — | Sibling agent | — | — |
| 321 | 24.7.3 CTDE | N/A | — | Sibling agent | — | — |
| 322 | 24.7.4 Communication learning | N/A | — | Sibling agent | — | — |
| 323 | 24.7.5 Emergent communication | N/A | — | Sibling agent | — | — |
| 324 | 24.7.6 Self-play | N/A | — | Sibling agent | — | — |
| 325 | 24.7.7 Population-based training | PARTIAL | `agents/exploration/evolution.py:1`, `evolution_strategy.py:1`, `elo.py:1` | Prompt/variant evolution, not agent-population training | — | P3 |
| 326 | 24.7.8 Social welfare / Nash equilibrium | N/A | — | Sibling agent | — | — |
| 327 | 24.8 Challenges and solutions | PARTIAL | — | — | — | P3 |
| 328 | 24.8.1 Coordination overhead | N/A | — | No multi-agent runtime | — | — |
| 329 | 24.8.2 Redundancy vs. efficiency | N/A | — | — | — | — |
| 330 | 24.8.3 Attribution / credit assignment | PARTIAL | `agents/exploration/decomp_reward.py:1`, `agents/agent/tool_reward.py:1` | — | — | P3 |
| 331 | 24.8.4 Scalability | N/A | — | — | — | — |
| 332 | 24.8.5 Emergent behavior and safety (**amplification**) | **PRESENT (ahead)** | `eval/verifiers/soundness.py:1-21` — HarnessCAD independently discovered and *measured* the book's "amplification" risk in a single-agent setting: a wrong diagnostic is amplified by a strong model | — | — | — |
| 333 | 24.8.6 Evaluation (multi-level metrics) | PARTIAL | `core/observe.py:417` (`Metrics`) | Single-agent metrics only | — | P3 |
| 334 | 24.9 Real-world applications | N/A | — | Examples | — | — |
| 335 | 24.9.1 Software development team | N/A | — | Example | — | — |
| 336 | 24.9.2 Research team | N/A | — | Example | — | — |
| 337 | 24.9.3 Customer service | N/A | — | Example | — | — |
| 338 | 24.9.4 Creative team | N/A | — | Example | — | — |
| 339 | 24.10 Architecture comparison | N/A | — | Reference table | — | — |
| 340 | 24.11 Summary ("start simple; evolve only when necessary") | **PRESENT** | `core/harness.py:1-26` | HarnessCAD followed this rule and should keep following it | — | — |

### H5 — Chapter 25: Agent Development Frameworks

| # | Section | Verdict | Where | Gap | Fix | Pri |
|---|---|---|---|---|---|---|
| 341 | 25.1 The engineering gap | PARTIAL | — | Reliability/observability/testability strong; deployment/iteration absent | — | P2 |
| 342 | 25.2 The agent development lifecycle | PARTIAL | — | — | — | P3 |
| 343 | 25.2.1 Phase 1: Design (capability matrix, tool selection, constraint spec) | PRESENT | `registry.py:1` (capability index), `core/contract.py:1` (machine-verifiable acceptance contract) | — | — | — |
| 344 | 25.2.2 Phase 2: Implementation | PRESENT | `core/harness.py`, `agents/agent/planner.py` | — | — | — |
| 345 | 25.2.3 Phase 3: Testing | PRESENT | `tests/` mirrors `src/`; `eval/selftest/` | — | — | — |
| 346 | 25.2.4 Phase 4: Deployment | ABSENT | — | No container, no service, no queue | — | P3 |
| 347 | 25.2.5 Phase 5: Iteration (failure logging + categorization + prompt updates + A/B) | PARTIAL | `core/observe.py:538` (`FailureTaxonomy`), `eval/pressure/` (offline A/B) | No production loop | — | P2 |
| 348 | 25.3 Major frameworks | N/A | — | Custom harness chosen (per the book's own rule) | — | — |
| 349 | 25.3.1 LangGraph | N/A | — | Not used; stdlib-only policy | — | — |
| 350 | 25.3.2 AutoGen | N/A | — | Not used | — | — |
| 351 | 25.3.3 CrewAI | N/A | — | Not used | — | — |
| 352 | 25.3.4 OpenAI Assistants / Agents SDK | N/A | — | Not used | — | — |
| 353 | 25.3.5 DSPy | **N/A — but worth reconsidering** | `agents/exploration/prompt_evolution.py:1` reimplements prompt optimisation | If a metric + 50 examples exist (they do: `eval/pressure/briefs.py`), DSPy's case is strong | Out of scope for stdlib-only, but note it | P3 |
| 354 | 25.3.6 Semantic Kernel | N/A | — | Not used | — | — |
| 355 | 25.4 Open-source agent tooling | N/A | — | Survey | — | — |
| 356 | 25.4.1 Modular agent architectures | PRESENT | `core/harness.py:102-123` (every collaborator optional and injectable) | — | — | — |
| 357 | 25.4.2 Key open-source building blocks | N/A | — | Survey | — | — |
| 358 | 25.4.3 Interoperability standards (MCP / A2A / OpenAPI) | PRESENT | `io/surfaces/mcp/`, `io/surfaces/a2a_server/`, `io/surfaces/acp/` — three protocol surfaces | No OpenAPI-to-tool layer | — | P3 |
| 359 | 25.5 Agent testing and evaluation | PRESENT | `tests/`, `eval/` | — | — | — |
| 360 | 25.5.1 Unit testing individual tools | PRESENT | `tests/` (mirrors every package) | — | — | — |
| 361 | 25.5.2 Integration testing full agent loops | PRESENT | `tests/core/test_harness.py:1` | — | — | — |
| 362 | 25.5.3 Regression testing with golden trajectories | PARTIAL | `core/observe.py:737` (`Replayer`), `core/harness.py:74` (`trajectory`), deterministic `run_id` | Replay exists; **no golden-trajectory corpus is asserted against in CI** | Freeze N golden runs; assert op-sequence + digest | P1 |
| 363 | 25.5.4 Behavioral testing (refusals, max tool calls, allowed domains) | PARTIAL | `io/gate.py:1` (refuse-with-reason), `core/harness.py:112` (max iterations) | No adversarial-brief suite | — | P2 |
| 364 | 25.5.5 Cost and latency testing | ABSENT | — | `core/routing.py` tallies cost; nothing asserts a bound | Add cost/latency bounds to the bench | P2 |
| 365 | 25.6 Observability and debugging | **PRESENT (ahead)** | `core/trace.py`, `core/observe.py` | — | — | — |
| 366 | 25.6.1 Tracing agent execution | PRESENT | `core/trace.py:38,99`; `core/harness.py:45` (`HARNESS_EVENT_KINDS`) | Not OpenTelemetry-shaped | Emit OTel spans (`observe.py:108` already has a `Span`) | P3 |
| 367 | 25.6.2 Failure categorization | **PRESENT (ahead)** | `core/observe.py:538` (`FailureTaxonomy`), `eval/reliability/infeasibility_taxonomy.py:1`, `agents/generation/feedback_taxonomy.py:1` | Three taxonomies (see §4 incoherence #9) | Unify | P2 |
| 368 | 25.6.3 Replay and debugging workflows | **PRESENT (ahead)** | `core/observe.py:693` (`RunReplay`), `:737` (`Replayer`), `:804` (`replay`), `:809` (`render_trajectory`) | — | — | — |
| 369 | 25.7 Production deployment patterns | ABSENT | — | — | — | P3 |
| 370 | 25.7.1 Async agent execution | ABSENT | — | Whole harness is synchronous (`core/harness.py:132`) | — | P3 |
| 371 | 25.7.2 Multi-tenant isolation | ABSENT | — | — | — | P3 |
| 372 | 25.7.3 Cost optimization (model routing, caching) | PARTIAL | `core/routing.py:1`, `eval/pressure/cache.py:1` | Unwired | See #130 | P1 |
| 373 | 25.7.4 Auto-scaling | N/A | — | Not a service | — | — |
| 374 | 25.8 Framework comparison | N/A | — | Reference table | — | — |
| 375 | 25.9 Complete implementation example | PRESENT | `core/pipeline.py:1`, `core/cli.py:1` (27 subcommands) | — | — | — |
| 376 | 25.10 Summary | N/A | — | — | — | — |

### H5 — Chapter 26: Agentic UI Frameworks

| # | Section | Verdict | Where | Gap | Fix | Pri |
|---|---|---|---|---|---|---|
| 377 | 26.1 Beyond the chat box | PARTIAL | `io/surfaces/ui/events.py:1`, `io/surfaces/acp/agent.py:1` | An event protocol exists; no renderer | — | P2 |
| 378 | 26.2 UI paradigms | PARTIAL | `io/surfaces/` | — | — | P3 |
| 379 | 26.2.1 Chat-based interfaces | PARTIAL | `io/surfaces/acp/agent.py:1` (Agent Client Protocol — chat host integration) | — | — | P3 |
| 380 | 26.2.2 Canvas / artifact-based | PARTIAL | `io/surfaces/render.py:1`, `io/surfaces/canonical_views.py:1`, `io/surfaces/edit_views.py:1`, `io/surfaces/id_overlay.py:1` | Artifacts are rendered; no live canvas | — | P3 |
| 381 | 26.2.3 Workflow visualization | PARTIAL | `io/surfaces/graphview.py:1`, `core/observe.py:809` (`render_trajectory`) | — | — | P3 |
| 382 | 26.2.4 Dashboard / monitoring | PARTIAL | `core/observe.py:834` (`report`), `eval/gallery/`, `eval/showcase/` | Offline reports, not live | — | P3 |
| 383 | 26.2.5 Collaborative interfaces | ABSENT | — | — | — | P3 |
| 384 | 26.2.6 Autonomous with checkpoints | **PRESENT** | `io/surfaces/ui/approval.py:37` + `agents/agent/edit_session.py:1` | Not wired to `AgentHarness` | See #122 | P0 |
| 385 | 26.3 Key UI components | PARTIAL | `io/surfaces/ui/events.py:1` | — | — | P2 |
| 386 | 26.3.1 Thought / reasoning display | ABSENT | — | The planner emits ops only, no reasoning trace (`system_prompt.py:24` forbids prose) | Deliberate; but it means there is nothing to show a human and nothing to audit | Allow a `rationale` field on the op envelope | P2 |
| 387 | 26.3.2 Tool-use visualization | PARTIAL | `io/surfaces/ui/events.py:1`, `core/trace.py` | Events exist; no view | — | P3 |
| 388 | 26.3.3 Progress indicators | PARTIAL | `io/surfaces/ui/events.py` | — | — | P3 |
| 389 | 26.3.4 Approval gates (action summary, risk indicator, approve/reject/modify) | **PRESENT (ahead)** | `io/surfaces/ui/approval.py:176` (`DryRunPreview` — predicts the change *without touching the kernel*), `:313` (`approval_required` with risk indicator), `:339` (batch collapse to fight alert fatigue — the book's exact concern, H5:2812) | Unwired to the harness | **P0** | |
| 390 | 26.3.5 Context display | PARTIAL | `agents/context/manager.py:79` (`BudgetReport` is renderable) | — | — | P3 |
| 391 | 26.3.6 Error and recovery UI | PARTIAL | `io/gate.py:1` (refusal carries a reason), `eval/reliability/repair_loop.py:1` | — | — | P2 |
| 392 | 26.3.7 Confidence indicators | **PRESENT (ahead)** | `io/surfaces/confidence.py:1`; and `eval/verifiers/soundness.py:91-97` is a *calibrated* confidence system for diagnostics (proven / measured / heuristic) | — | — | — |
| 393 | 26.4 Frameworks and libraries | N/A | — | No web UI; stdlib-only | — | — |
| 394 | 26.4.1 Vercel AI SDK | N/A | — | — | — | — |
| 395 | 26.4.2 Chainlit | N/A | — | — | — | — |
| 396 | 26.4.3 Gradio | N/A | — | — | — | — |
| 397 | 26.4.4 Streamlit | N/A | — | — | — | — |
| 398 | 26.4.5 OpenAI Assistants Playground | N/A | — | — | — | — |
| 399 | 26.4.6 LangGraph Studio | N/A | — | — | — | — |
| 400 | 26.4.7 Framework comparison | N/A | — | — | — | — |
| 401 | 26.5 Generative UI | ABSENT | — | — | — | P3 |
| 402 | 26.5.1 React Server Components | N/A | — | — | — | — |
| 403 | 26.5.2 Adaptive interfaces by content type | PARTIAL | `io/surfaces/adaptive_ux.py:1` | — | — | P3 |
| 404 | 26.6 Streaming and real-time patterns | PARTIAL | `agents/llm/base.py` (`stream`), `io/surfaces/acp/agent.py:11` (streams UIEvents/trace) | — | — | P3 |
| 405 | 26.6.1 Token streaming | PARTIAL | `agents/llm/base.py` | Not consumed by `Planner` (`planner.py:131` calls `complete`) | — | P3 |
| 406 | 26.6.2 Tool-call streaming | ABSENT | — | — | — | P3 |
| 407 | 26.6.3 Multi-agent streaming | N/A | — | Single agent | — | — |
| 408 | 26.6.4 Optimistic UI updates | N/A | — | — | — | — |
| 409 | 26.6.5 Backpressure handling | N/A | — | — | — | — |
| 410 | 26.7 Human-in-the-loop UI design | PARTIAL | `io/surfaces/ui/approval.py:1` | See §6 | — | P0 |
| 411 | 26.7.1 When to interrupt (reversibility / scope / confidence / cost / novelty) | PARTIAL | `io/surfaces/ui/approval.py:120` (`tier_for`) keys on **reversibility** only | Confidence and cost are not interruption criteria — yet `soundness` gives us a confidence signal and `routing` gives us a cost signal | Escalate on low confidence and on cost, per H3:4372 | P1 |
| 412 | 26.7.2 Tiered approval workflows | **PRESENT (ahead)** | `io/surfaces/ui/approval.py:37` — the book's exact 3 tiers, plus the alert-fatigue batch collapse at `:339` | Unwired | P0 | |
| 413 | 26.7.3 Feedback mechanisms | PARTIAL | `agents/agent/host_feedback.py:1`, `agents/generation/caption_feedback.py:1` | No thumbs/preference capture that reaches memory | Route human corrections into `MemoryStore.procedural` | P1 |
| 414 | 26.7.4 Teaching the agent through UI interaction | ABSENT | — | This is the highest-value HITL pattern for CAD and it does not exist: a human's correction is never generalised | Correction → skill (`SkillLibrary.add_verified`) | P1 |
| 415 | 26.8 Accessibility and trust | PARTIAL | — | — | — | P2 |
| 416 | 26.8.1 Explaining agent decisions | PARTIAL | `core/observe.py:809`, `eval/verifiers/soundness.py:147-378` (every verifier carries a written `reason`) | The *agent's* decisions are not explained; the *verifier's* are | — | P2 |
| 417 | 26.8.2 Showing confidence levels | PRESENT | `io/surfaces/confidence.py:1`, `eval/verifiers/soundness.py` | — | — | — |
| 418 | 26.8.3 Undo and rollback | **PRESENT (ahead)** | `core/loop.py:180` (`rollback(label)`), `:126` (automatic per-op rollback on ERROR), `core/state/opdag.py` | Undo is *structural*, not a UI afterthought | — | — |
| 419 | 26.8.4 Audit trails in the UI | PRESENT | `core/trace.py:99` (JSONL), `core/harness.py:74` (JSON-serialisable trajectory), `governance/audit/closure.py:1` | — | — | — |
| 420 | 26.8.5 Managing user expectations | PARTIAL | `io/gate.py:1` ("verified, or refused with a reason — there is no third outcome") | — | — | P3 |
| 421 | 26.9 Implementation example: full-stack agentic UI | ABSENT | — | No UI ships | — | P3 |
| 422 | 26.9.1 Backend with streaming + approval gates | PARTIAL | `io/surfaces/acp/agent.py:1`, `io/surfaces/server.py:1` | — | — | P2 |
| 423 | 26.9.2 Frontend | ABSENT | — | — | — | P3 |
| 424 | 26.10 Summary | N/A | — | — | — | — |

### H5 — Chapters 27, 28, 29 (assessment and reference)

These 44 sections are quiz questions (27.1–27.30), formula/API reference tables (28.1–28.11)
and a closing essay (29.1–29.3). They contain no prescription that a system can implement or
fail to implement; they restate material already graded above. **All 44 are N/A on that
ground**, with two exceptions marked below where the reference section restates a *design*
constraint that HarnessCAD can be measured against.

| # | Section | Verdict | Where | Gap | Fix | Pri |
|---|---|---|---|---|---|---|
| 425 | 27.1 Foundations Questions | N/A | — | Assessment matter | — | — |
| 426 | 27.2 Core Algorithm Questions | N/A | — | Assessment matter (and sibling agent's RL brief) | — | — |
| 427 | 27.3 System Design Questions | N/A | — | RLHF-cluster design; sibling agent | — | — |
| 428 | 27.4 Practical and Debugging Questions | N/A | — | Assessment matter | — | — |
| 429 | 27.5 GRPO Variants and Advanced RL | N/A | — | Sibling agent | — | — |
| 430 | 27.6 (RL) | N/A | — | Sibling agent | — | — |
| 431 | 27.7 (RL) | N/A | — | Sibling agent | — | — |
| 432 | 27.8 (RL) | N/A | — | Sibling agent | — | — |
| 433 | 27.9 Reward Model and SFT Questions | N/A | — | Sibling agent | — | — |
| 434 | 27.10 System Architecture Extension Questions | N/A | — | Sibling agent | — | — |
| 435 | 27.11 Transformer Architecture Questions | N/A | — | Model internals | — | — |
| 436 | 27.12 Flash Attention Questions | N/A | — | Model internals | — | — |
| 437 | 27.13 LoRA and PEFT Questions | N/A | — | Sibling agent | — | — |
| 438 | 27.14 Model Compression Questions | N/A | — | Model internals | — | — |
| 439 | 27.15 Mixture of Experts Questions | N/A | — | Model internals | — | — |
| 440 | 27.16 Diversity in Training Questions | N/A | — | Sibling agent | — | — |
| 441 | 27.17 Speculative Decoding Questions | N/A | — | Inference serving | — | — |
| 442 | 27.18 Agentic RL Questions | N/A | — | Sibling agent | — | — |
| 443 | 27.19 Listwise Rewards / Advanced RM | N/A | — | Sibling agent | — | — |
| 444 | 27.20 RL for Large Reasoning Models | N/A | — | Sibling agent | — | — |
| 445 | 27.21 LLM Evaluation Questions | N/A | — | Sibling agent | — | — |
| 446 | 27.22 Agentic Memory Questions | N/A | — | Restates ch.17 (graded at #55–#98) | — | — |
| 447 | 27.23 Agent Orchestration Questions | N/A | — | Restates ch.18 (graded at #99–#141) | — | — |
| 448 | 27.24 MCP Protocol Questions | N/A | — | Restates ch.21 | — | — |
| 449 | 27.25 Agent Communication (A2A) Questions | N/A | — | Restates ch.23 | — | — |
| 450 | 27.26 Multi-Agent Systems Questions | N/A | — | Restates ch.24 | — | — |
| 451 | 27.27 Agent Development Framework Questions | N/A | — | Restates ch.25 | — | — |
| 452 | 27.28 Agentic Environments Questions | N/A | — | Restates ch.20 | — | — |
| 453 | 27.29 Agentic UI Framework Questions | N/A | — | Restates ch.26 | — | — |
| 454 | 27.30 RAG and Agentic RAG Questions | N/A | — | Restates ch.16 | — | — |
| 455 | 28.1 Core RL & Alignment Equations | N/A | — | Formula reference | — | — |
| 456 | 28.2 Transformer & Architecture Formulas | N/A | — | Formula reference | — | — |
| 457 | 28.3 Decoding Methods | N/A | — | Reference | — | — |
| 458 | 28.4 Systems & Parallelism | N/A | — | Reference | — | — |
| 459 | 28.5 GPU Hardware Specs | N/A | — | Reference | — | — |
| 460 | 28.6 Hyperparameter Ranges | N/A | — | Reference (sibling agent) | — | — |
| 461 | 28.7 TRL API Quick Reference | N/A | — | Third-party API | — | — |
| 462 | 28.8 RAG Pipeline Formulas | N/A | — | Reference | — | — |
| 463 | 28.9 Agentic Design Patterns | N/A | — | Reference table (restates ch.19) | — | — |
| 464 | 28.10 Agent Communication Protocols | N/A | — | Reference table (restates ch.21/23) | — | — |
| 465 | 28.11 Context Window Budget (C ≥ S+M+T+H+R) | **PARTIAL** | `agents/context/manager.py:5,88-96` implements the identity exactly — but M (memory) is structurally always 0 because nothing writes to it | The budget has a slot for memory and the product has no memory in the loop | Gap #2 | **P0** |
| 466 | 29.1 Summary | N/A | — | Closing essay | — | — |
| 467 | 29.2 The Road Ahead: Open Challenges | N/A | — | Closing essay | — | — |
| 468 | 29.3 Further Reading | N/A | — | Bibliography | — | — |

---

## 2. The harness-chapter grade (H3, chapter 18)

Chapter 18 has 43 sections. HarnessCAD scores:

| | PRESENT | PARTIAL | ABSENT | N/A |
|---|---:|---:|---:|---:|
| Ch.18 rows (#99–#141) | 13 | 19 | 8 | 3 |

Scoring PRESENT = 1, PARTIAL = 0.5, ABSENT = 0, over the 40 non-N/A rows:
**(13 + 9.5) / 40 = 56%.**

**Grade: C+ (56/100). A well-built spine with three of the book's five harness concerns
missing from the object that is actually called `AgentHarness`.**

Section by section, the honest reading:

**Where we meet or beat the chapter**

* **18.6.2 Task state.** `core/loop.py:84-149` is the best code in the repository. Block-and-
  correct (a rejected op never mutates), transactional per-op verify with automatic rollback
  (`:126`), a checkpoint per accepted op (`:134`), and a content digest that makes replay an
  invariant rather than an aspiration (`io/backends/base.py:1`). The book asks for
  checkpoint/rollback; this is finer-grained and stronger.
* **18.7.2 Loop detection.** `eval/reliability/loopdetect.py:1` implements exactly the book's
  sliding-window action hash (H3:4484), and — better than the book's reference implementation —
  `core/harness.py:311` runs it **pre-dispatch**, so a detected oscillation never re-mutates state.
* **18.7.4 Observability.** `core/trace.py` + `core/observe.py` (858 lines) give traces, metrics,
  a failure taxonomy, Wilson confidence intervals, and full trajectory replay. The book's
  "Debugging Gap" sidebar (H3:4510) asks for replay tooling as an aspiration; we have it.
* **18.3.1 System prompt.** All four of the book's sections, with the op vocabulary *generated
  from the op registry* (`agents/agent/system_prompt.py:40`) so the prompt cannot drift from the
  code. That is better practice than the book describes.
* **18.2.1 Context budget.** The five-way partition C ≥ S+M+T+H+R is implemented literally
  (`agents/context/manager.py:5,88`).

**Where we are behind**

* **18.1 / 18.6.4 — memory is not a harness concern here.** The book's definition of a harness
  (H3:3698) names five responsibilities: reasoning, execution, **memory**, communication,
  observability. `AgentHarness.__init__` (`core/harness.py:102-123`) takes `session, planner,
  context, loop_detector, executor, tracer, verifiers`. There is no memory. There is no
  persistence. Every run starts from zero, and a `MemoryStore` with four memory types, a
  Voyager `SkillLibrary`, an `ErrorNotebook` and an Ebbinghaus decay model all sit unused
  three directories away.
* **18.3.4 — the tools are bad, and we know how to make them good.** The book is unusually
  emphatic here ("Tool descriptions are critical", H3:4074; "the LLM selects tools based almost
  entirely on the name and description", H4:1990). `io/surfaces/mcp/tools.py:1-12` builds
  five-component descriptions with when-to-use, when-NOT-to-use, side effects and output specs —
  and hands them to *external* MCP clients. Our own model gets a single tool called `emit_ops`
  described as "Emit the CISP op sequence that builds the requested design"
  (`agents/agent/planner.py:44`). We built the good tools and did not give them to ourselves.
* **18.5.4 — HITL is built and bypassed.** Three approval tiers, a risk indicator, a dry-run
  preview, batch collapse for alert fatigue (`io/surfaces/ui/approval.py`), a gated ToolExecutor
  (`eval/reliability/executor.py:12`) — and `AgentHarness._dispatch` (`core/harness.py:337`)
  falls through to `session.apply_ops` when no executor is injected, which is the default
  everywhere except the ACP surface.
* **18.2.6 — the token counter is a heuristic.** `agents/context/manager.py:60`. The book warns
  that rules of thumb are 20–40% wrong on code and JSON (H3:3931). CISP ops *are* JSON. The
  `TokenCounter` protocol (`manager.py:47`) exists precisely to fix this and nothing implements it.
* **18.7.1 / 18.8.2 — retry, fallback and cost routing exist as an unused module.**
  `core/routing.py` is a drop-in `LLM` with a fallback chain and a cost tally. `core/pipeline.py:28`
  constructs a bare `LiteLLMClient` instead.
* **18.8.x — production concerns are absent** (no async, no queue, no rate limiting). Fine for
  now; stated so it is not mistaken for done.

**The load-bearing observation about the chapter**

Chapter 18 tells you to build the harness as an *operating system* for the model: it schedules,
it remembers, it gates, it observes. HarnessCAD built a superb **kernel driver** (loop.py, the
op DAG, the digest, the verifiers) and a thin **scheduler** (harness.py), and then built the
memory, the gates, the tool descriptions and the routing as *separate products* that the
scheduler never loads. The gap between HarnessCAD's parts and HarnessCAD's harness is larger
than the gap between HarnessCAD's harness and the book.

---

## 3. Top 10 gaps, ranked by harm / effort

| # | Gap | Harm | Effort | Modules |
|---|---|---|---|---|
| **1** | **The pressure experiment does not use the soundness gate, and still issues orders.** `eval/pressure/prompts.py:189` (`format_typed`) renders every diagnostic the fleet emits — no `model_facing()` call anywhere in `eval/pressure/` — and closes with **"Fix exactly these problems"** (`:208`), the exact imperative that `soundness.observe()` (`eval/verifiers/soundness.py:525`) exists to eliminate. The fix that was written in response to the −8.3 result is **not on the path that measured it**. The harness's central claim is therefore still refuted, and the refutation is still un-rerun. | Maximal: the product's headline result is stale and the fix is unmeasured | Hours: import `model_facing` + `observe` into `pressure/prompts.py`, add a third arm (`harness-gated`), re-run the 6×12×2×3 grid (~9 min) | `eval/pressure/prompts.py`, `eval/pressure/loops.py`, `eval/verifiers/soundness.py` |
| **2** | **The harness has no memory.** No episodic recall of a similar past brief, no skill retrieval, no error notebook, no persistence — despite `MemoryStore` (4 types, `agents/memory/store.py:114`), `SkillLibrary` (execution-verified, `skills.py:1`), `ErrorNotebook` (`error_notebook.py:131`), `decay.py` and a working `reflexion.py` all existing and being tested. A CAD agent that cannot remember that it already built this bracket is the book's chapter-17 failure mode #2 verbatim. | Very high: this is the largest available win, and in a domain where briefs repeat it is worth more than any verifier | Days: add `memory=` to `AgentHarness.__init__`; recall top-k episodes + skills in `_preflight`; `add_episodic` + `add_verified` on converge; `save/load` per session | `core/harness.py`, `agents/memory/*`, `agents/agent/planner.py`, `eval/reliability/strategies/reflexion.py` |
| **3** | **Nothing measures verifier precision.** `soundness.py` *declares* tiers from reasoning; no test asks "does this verifier stay silent on the known-good corpus?". The report itself names this as the root cause ("Twenty-three verifiers were written, each with a test asking *does it FIRE on bad input?*. Not one asked *does it stay SILENT on good input?*", `soundness.py:16-21`) — and no harness was built to answer it. A tier table is an assertion until a false-positive rate is measured. | Very high: the tiers are currently *claims*, and the whole feedback policy rests on them | Days: run the full fleet over the known-good corpus (`eval/bench/`, `assets/`), report per-verifier FPR, fail CI on a PROVEN/MEASURED verifier that fires | `eval/verifiers/soundness.py`, `eval/verifiers/registry.py`, `eval/bench/` |
| **4** | **The model is given one blunt tool while the good ones are exported.** `agents/agent/planner.py:41` (`emit_ops`, one line of description) vs. `io/surfaces/mcp/tools.py:1` (5-component descriptions, typed params, annotations, output specs). Book: tool descriptions shift selection accuracy 10–20% (H3:4080). | High | Hours: make `Planner` build its tool list from `ToolCatalog.to_mcp()` | `agents/agent/planner.py`, `io/surfaces/mcp/tools.py` |
| **5** | **No human is in the loop that builds parts.** Approval tiers, risk indicators, dry-run previews and an approval-gated executor all exist; `AgentHarness` defaults to none of them (`core/harness.py:337`). The MCP server likewise executes destructive tools without consent despite carrying the annotations that classify them (`io/surfaces/mcp/server.py:134`, `io/surfaces/ui/approval.py:107`). | High (and a hard blocker for any deployment) | Hours–days: default `AgentHarness.executor` to `reliability.executor.ToolExecutor`; gate MCP `tools/call` on `ApprovalTier.REQUIRE` | `core/harness.py`, `eval/reliability/executor.py`, `io/surfaces/ui/approval.py`, `io/surfaces/mcp/server.py` |
| **6** | **Four loops implement the same pattern.** `core/harness.AgentHarness` (full ReAct), `agents/agent/runner.run` (minimal), `eval/pressure/loops.run_brief` (experiment), `eval/reliability/repair_loop` + `strategies/*` (best-of-N / MCTS / reflexion). `core/pipeline.build` — the product entry point — uses the **minimal** one (`core/pipeline.py:28`), which has no loop detection, no context pre-flight, no contract and no trajectory. | High: the shipping path is the weakest loop, and improvements to `AgentHarness` do not reach it | Days: make `AgentHarness` the only loop; express the others as policies (`strategy=`, `feedback=`) | `core/pipeline.py`, `core/harness.py`, `agents/agent/runner.py`, `eval/pressure/loops.py`, `eval/reliability/*` |
| **7** | **The soundness tier is stripped on the wire.** `Diagnostic.soundness` is deliberately excluded from `to_dict()` (`eval/verifiers/verify.py:39-49`) to keep the pressure wire format byte-identical. Every consumer that crosses a JSON boundary — the MCP server, the A2A server, the JSONL tracer, `eval/pressure` — therefore loses the tier and silently falls back to the code index (`soundness.py:471`), which does not know about per-verifier `by_code` promotions. A remote MCP client cannot tell a theorem from a guess. | High and latent | Hours: add `soundness` to `to_dict()`; keep the pressure comparison by pinning the old serializer in the experiment, not in the type | `eval/verifiers/verify.py`, `io/surfaces/mcp/server.py`, `io/surfaces/a2a_server/wire.py` |
| **8** | **Token counting is heuristic; the harness never sheds load.** `agents/context/manager.py:60` + `core/harness.py:284-309` (pre-flight is *logged*, never acted on). Silent provider truncation is the book's "Silent Truncation Trap" (H3:3758). | Medium-high | Hours: implement `TokenCounter` with the real tokenizer; make `_preflight` call `assemble()` and shed | `agents/context/manager.py`, `core/harness.py` |
| **9** | **No fallback model, no cost ceiling, no retry/backoff in the harness.** `core/routing.py` (498 lines: classify-then-route, sequential fallback, cost tally) satisfies the `LLM` protocol and is not used by anything that ships. | Medium | Hours: `pipeline.build(llm=RoutingLLM(...))` | `core/routing.py`, `core/pipeline.py` |
| **10** | **323 orphan modules (26.5% of 1,219).** `registry.orphans()` — `eval/bench` 112, `data/dataengine` 66, `eval/quality` 34, `data/datagen` 30, `domain/numeric` 14, `eval/reliability` 10, `eval/verifiers` 10. Ten orphaned *verifiers* is the alarming number: a verifier nothing imports is a rule nobody audits, and unaudited rules are what cost 8 briefs. | Medium (but the verifier subset is High) | Days: triage; either import into the fleet registry with a declared tier or delete | `registry.py:529`, `eval/verifiers/registry.py`, `eval/bench/` |

---

## 4. Incoherences

Hunted specifically, per the instruction "make sure all sections are fully up to date and
logical and consistent". Each is two parts of the system implementing the same book pattern in
different, mutually-unaware ways, or a pattern that is half-wired.

1. **The feedback gate is applied in one of the four loops.**
   `agents/agent/planner.py:98` calls `model_facing(diagnostics, self.feedback_tiers)` — the gate.
   `eval/pressure/prompts.py:189` does not, and it is the loop that produced the −8.3.
   `io/surfaces/a2a_server/__main__.py:56` constructs `AgentHarness(session, _PlatePlanner())`
   — a planner that is not `agent.planner.Planner`, so **the A2A surface feeds ungated
   heuristics to whatever model backs it**. `core/harness.py:241` filters by *severity*, not by
   *soundness*, and relies on the planner to gate downstream; any planner that is not
   `agent.planner.Planner` therefore bypasses the policy. The gate is a property of one class,
   not of the architecture. It should be enforced at the harness boundary.

2. **Four loops (§3 gap #6).** `core/harness.py:91`, `agents/agent/runner.py:31`,
   `eval/pressure/loops.py:80`, `eval/reliability/repair_loop.py`. Two system prompts:
   `agents/agent/system_prompt.py:81` (registry-generated, cannot drift) and
   `eval/pressure/prompts.py:88` (a hand-written constant that *can* drift from the op set).
   Two feedback formatters: `planner._format_diagnostics` (`planner.py:135`) and
   `pressure/prompts.format_typed` (`:189`). They do not agree, and the one that disagrees is
   the one that generated the published number.

3. **Two tool catalogues.** `agents/agent/tool_schema.py:1` (TOOLCAD-style FreeCAD tool
   signatures + `InterfaceResult`) and `io/surfaces/mcp/tools.py:1` (CISP-op MCP catalogue).
   The docstring at `tool_schema.py:17` acknowledges the split ("deliberately DISTINCT from
   `surfaces.mcp.tools.ToolCatalog`") and does not resolve it. Two reward functions follow:
   `agents/agent/tool_reward.py:1` (TOOLCAD trajectory reward) and
   `io/surfaces/mcp/tools.py:78` (`reward_from_apply`). Nothing reconciles them.

4. **Five gate-like layers, no single policy object.**
   `eval/reliability/guardrails.py:1` (pre-apply hard gate) →
   `core/loop.py:90` (backend block-and-correct) →
   `core/loop.py:110` (transactional verify + rollback) →
   `io/surfaces/ui/approval.py:37` (three approval tiers) →
   `governance/security/tool_gate.py:1` (prompt/tool trust boundary) →
   `io/gate.py:1` (output gate: "verified or refused, no third outcome").
   Each is individually good. There is no place that states the order, and the default
   `AgentHarness` path traverses only two of them (loop.py's pair).

5. **Two memory-decay systems, unconnected.** `agents/memory/store.py:141` (`recall_episodic`)
   scores by lexical similarity with no recency, importance or decay term.
   `agents/memory/decay.py:1` implements the full Ebbinghaus reinforced-decay curve
   (`R = exp(-Δt/(S·τ))`, `S += 1` on recall) and is imported by exactly one module,
   `agents/generation/registry.py:958`. The memory store does not know its own decay model exists.

6. **`Diagnostic.soundness` does not survive serialization.** `eval/verifiers/verify.py:41`
   declares the field; `:43-49` `to_dict()` omits it, with a comment stating the omission is to
   keep the pressure wire format stable. The result: the tier is in-process only. `soundness.tier_of`
   (`:462`) falls back to `_CODE_INDEX`, which cannot represent a per-verifier `by_code`
   promotion for a code it does not know — so a `preflight-THICKNESS_TOO_LARGE` (PROVEN) that
   crosses a JSON boundary is still recoverable (it is in `_CODE_INDEX`), but a future promoted
   code will not be. The experiment's wire format is holding the type system hostage.

7. **`eval/pressure/prompts.py:208` still gives orders.** `soundness.observe()`
   (`soundness.py:525`) was written specifically to stop this: "observation first, evidence
   attached, imperative last and labelled". `system_prompt.py:64-68` (rule 7) tells the model
   that diagnostics *are* observations with suggestions. `eval/verifiers/registry.py:349`
   phrases kernel-preflight findings through `observe()`. And the experiment's formatter still
   emits "Fix exactly these problems". Three of the four writing surfaces were updated.

8. **Reflection exists twice, in neither place the agent runs.**
   `eval/reliability/strategies/reflexion.py:1` implements read-act-reflect-write over
   `MemoryStore`. `core/harness.py:132` implements act-verify-repair with no reflect and no
   memory. Neither can call the other. The blueprint section they both cite is the same one
   (sec.8).

9. **Three failure taxonomies.** `core/observe.py:538` (`FailureTaxonomy`),
   `eval/reliability/infeasibility_taxonomy.py:1`, `agents/generation/feedback_taxonomy.py:1`.
   Plus `eval/reliability/code_error.py` and `compiler_diagnostics.py`. No mapping between them.

10. **The A2A/MCP/ACP surfaces build harnesses the product does not.**
    `io/surfaces/acp/agent.py:202` constructs an `AgentHarness` with a ToolExecutor-gated
    approval path — the *only* place in the repository that wires HITL into the loop. The CLI
    (`core/cli.py`) and `core/pipeline.py` do not. The best-configured harness in the codebase
    is reachable only through an editor integration.

11. **Ten orphaned verifiers.** `registry.orphans()` reports 10 modules under `eval/verifiers`
    that no other indexed module imports. Given that the fleet auto-discovers
    (`eval/verifiers/registry.py`), some of these may be discovered-but-not-imported, which is
    fine; but `soundness.soundness_of` (`:436`) raises on an undeclared verifier, and the
    quarantine path (`UNDECLARED`, `:448`) silently demotes it to HEURISTIC at runtime. A
    verifier can therefore exist, run, fire, and never reach the model, with no failing test
    unless `tests/eval/verifiers/test_soundness.py` enumerates it. Verify that it does.

---

## 5. Environment / multi-agent: the two architecture questions

### The seventh backend (computer use) does **not** fit `GeometryBackend`

`io/backends/base.py:41` requires:

* `apply(op) -> ApplyResult` that **does not mutate on rejection** ("On invalid references,
  return ok=False WITHOUT mutating");
* `state_digest() -> str`, a **content hash** stable across identical replays
  (`base.py:1`: "The digest is load-bearing");
* `query(q) -> dict`, a synchronous structured read;
* `regenerate()`, a synchronous rebuild.

A GUI-driving backend violates all four. You cannot know whether a click was invalid until
after it has mutated the document. There is no content hash of a running SolidWorks session —
only a screenshot and an accessibility tree. `query("summary")` becomes a perception problem,
not a read. And every operation is asynchronous and slow.

**The book already has the right model and HarnessCAD already has the seam.** H4 §20.2.1
prescribes the *hybrid observation* (screenshot + accessibility tree) and §20.4.1 prescribes
`reset() / step(action) / state() / close()` with per-environment typed actions and
observations. `io/surfaces/mcp/gym.py:1` **is that interface**, already built over the session,
already returning a hybrid observation (JSON summary + per-view render availability, with image
bytes deliberately excluded to protect the context window, `gym.py:10-16`).

Recommendation:

1. Promote `gym.py`'s `Environment` shape to a first-class protocol in `io/` — `reset/step/
   observe/close`, typed per-environment `Action`/`Observation`.
2. Make `GeometryBackend` **one implementation** of it (the transactional, digestible,
   deterministic one), and declare its extra guarantees as *capabilities*
   (`supports_digest`, `supports_rollback`, `supports_block_and_correct`).
3. The GUI backend implements `Environment` and declares none of those capabilities. The harness
   then knows, structurally, that it must not promise deterministic replay or transactional
   rollback for that backend — instead of discovering it at runtime.
4. Verification for the GUI backend runs on the *exported* artifact through `io/gate.py:1`,
   which is already the "verified or refused" door.

Without step 2, adding the seventh backend either breaks the digest invariant that the whole
replay/trace/pressure machinery rests on, or forces a fake digest, which is worse.

### Multi-agent: **no. A single loop is correct here, and the evidence says so.**

Arguing it rather than checking a box:

* **The measured failure was precision, not capacity.** The harness lost 8.3 points because a
  rule was wrong (`assets/pressure/report.md:83`), not because one agent could not hold enough
  in its head. Multi-agent buys specialisation and parallelism (H4:4937). Neither addresses a
  false diagnostic. Worse, H4 §24.8.5 names **amplification** as a multi-agent risk — errors in
  one agent amplified downstream — which is precisely the mechanism that already hurt us with
  *one* agent. Adding agents multiplies the surfaces from which a wrong instruction can enter.
* **CAD has one authoritative state.** The op DAG (`core/state/opdag.py`) is the single shared
  artifact. The book's coordination mechanisms (blackboard, contract net, consensus) exist to
  reconcile *distributed* partial state. There is nothing to reconcile: two agents editing one
  op DAG is a merge conflict, not a collaboration.
* **The book's own rule.** H4 §19.3 ("Keep it simple"), §24.11 ("Start simple: begin with a
  centralized supervisor pattern, measure its limitations, and evolve"). We have not yet
  exhausted the single loop — it has no memory, no good tools and no human. Those are the
  cheaper wins.
* **Where multi-agent *is* justified, and we should do it:**
  1. **The Red Team pattern (H4 §24.6.7).** An adversarial agent whose job is to find verifiers
     that fire on correct geometry. The four fleet bugs in `report.md:81-108` were found by a
     human doing exactly this by hand. This is the one multi-agent pattern the −8.3 result
     positively argues for, and it runs **offline**, so it cannot poison the build loop.
     `agents/agents/overseer.py:1` is the nearest existing thing.
  2. **Best-of-N / ensemble (H4 §24.6.5)** — already built (`strategies/best_of_n.py`,
     `mcts.py`). This is *parallelisation*, not multi-agent; it needs no A2A, no supervisor and
     no blackboard. Expose it as a harness policy.

  Everything else in `agents/agents/` (supervisor, blackboard, roles, vmodel workflow, message
  tree) is speculative capacity. Keep it; do not wire it; do not pretend it is architecture.

### HITL: where the human belongs, and where they are

Per H3 §18.5.4 and H5 §26.7, escalate on **irreversibility**, **low confidence**, and **cost**.
HarnessCAD escalates on none of these in the default loop, and has the machinery for all three.

| Where a human belongs | Status |
|---|---|
| **Before the build**: confirm an ambiguous brief (which of these is the datum? mm or inch?). A2A already has the `input-required` task state (`agents/a2a/task.py:1`) for exactly this. | **ABSENT.** The brief goes straight to the planner (`core/pipeline.py`). |
| **Mid-loop, on low confidence**: `soundness.human_facing()` (`eval/verifiers/soundness.py:509`) already computes the exact set of diagnostics that were *withheld from the model because they might be wrong*. That set is precisely what a human should be shown. | **HALF-WIRED.** The function exists; no surface renders it. This is a two-hour job and it is the single cleanest HITL win available. |
| **Before an irreversible action** (export, delete, overwrite): Tier-3 approval with risk indicator + dry-run preview. | **BUILT AND BYPASSED** (`io/surfaces/ui/approval.py:37`; `core/harness.py:337` skips it; `io/surfaces/mcp/server.py:134` skips it). |
| **After a correction**: generalise the human's fix into a skill (H5 §26.7.4). `SkillLibrary.add_verified` is the right sink. | **ABSENT.** Human corrections are discarded. |

---

## 6. Where HarnessCAD is ahead of the book

Honest, not flattering. Seven items; three of them are genuine contributions the book does not
contain at all.

1. **Soundness tiering of the feedback channel (`eval/verifiers/soundness.py`).** The book
   assumes tools and diagnostics are ground truth. H4 §19.3 says "Provide good tools… clear
   error messages"; H3 §18.4.3 says "normalise errors so the model can reason about them". Not
   one of the 468 sections considers that **a verifier can be wrong, and that a capable model
   will execute a false instruction precisely**. HarnessCAD measured that effect
   (`assets/pressure/report.md:36-56`), named the mechanism ("the value of a typed diagnostic is
   bounded above by its truth, and the tighter a model's instruction-following, the tighter that
   bound binds"), and built a precision policy against it: three declared tiers, a hard error on
   an undeclared verifier (`soundness.py:436`), quarantine-by-default at runtime (`:448`), and a
   model-facing channel narrowed to the diagnostics that cannot lie (`:497`). **This is publishable
   and it is not in the book.**

2. **Feedback phrased as evidence, not orders (`soundness.observe`, `:525`).** The book's
   entire discussion of phrasing is about *humans* (H5 §26.8: explain decisions, show
   confidence). HarnessCAD makes the same distinction on the *model-facing* channel: observation
   first, evidence attached, imperative last and explicitly labelled `SUGGESTION (advisory, not
   a requirement)`. The system prompt's rule 7 (`agents/agent/system_prompt.py:64-68`) teaches
   the model to treat diagnostics as observations and to "not discard correct geometry to satisfy
   a suggestion" — which is a direct structural defence against the regression mode that lost the
   experiment. The book has nothing on this.

3. **Determinism as a structural property, not a seed.** The book asks for "seed-based
   determinism" (H4 §20.4.4). HarnessCAD has no wall clock anywhere on the critical path:
   `run_id` is derived from the brief (`core/harness.py:126`), the session's run id from the op
   batch + a sequence counter (`core/loop.py:72`), the capability index is byte-identical across
   builds (`registry.py:19`), and the backend contract *requires* that replaying the same ops
   yields the same digest (`io/backends/base.py:1`). Replay is an invariant, not a best effort.

4. **The output gate (`io/gate.py:1`).** "Every artifact leaving the harness is either verified
   valid, or refused with a reason. There is no third outcome. Silence is not success." The
   book's guardrails chapter never states a completeness/soundness contract this sharply, and
   the module documents the exact bug that motivated it (the F-rep two-sided shell that dilated
   parts and passed every check).

5. **MCP-as-Gym, already built.** H4 §21.10 poses "Could MCP serve as the gymnasium of
   tool-using LLM training?" as an **open question** with four sub-questions (reward
   specification, episode management, observation schemas, benchmark suites).
   `io/surfaces/mcp/gym.py:1` answers all four over one shared tool catalogue, with an
   execution-based reward (`tools.py:78`), a hybrid observation that deliberately keeps image
   bytes out of the context window, and an explicit no-ground-truth-leak guarantee (`gym.py:17`)
   — the book's own §20.2.1 "observation leakage" failure mode, pre-empted.

6. **A static capability registry with an orphan detector (`registry.py`).** AST-based, no
   imports executed, deterministic, tag-derived, with `orphans()` (`:529`) as a first-class
   architectural-decay metric. The book has no equivalent concept. (It also happens to be how
   this audit found the 323 orphans.)

7. **A pre-registered adversarial experiment against the product's own central claim, published
   as a loss.** `assets/pressure/report.md` states the losing result, refuses to fix the bug that
   caused it before reporting ("repairing the thing under test to improve its score is the
   definition of a rigged result", `:139`), and notes that a second bug *contaminated the
   experiment in the harness's favour* and was also left in ("The −8.3 headline is, if anything,
   generous to the harness", `:108`). H5 §25.5 and H4 §20.7 describe evaluation harnesses; no
   section describes this. It is the strongest evidence in the repository that the engineering
   culture is sound, and it is worth more than any of the code.

---

## Appendix: what I verified and what I did not

* Read in full: H3 (5,443 lines, chapter 18 twice), H4 (6,325), H5 (9,037).
* Read in full or in substantial part: `registry.py`, `_capability_index.json` (1,219 modules),
  `core/harness.py`, `core/loop.py`, `core/pipeline.py`, `core/cisp/{protocol,ops,annotations,
  explicit_context}.py`, `agents/agent/{planner,runner,system_prompt}.py`,
  `agents/memory/store.py`, `eval/verifiers/soundness.py`, `assets/pressure/report.md`,
  `eval/pressure/{loops,prompts}.py`, `io/backends/base.py`.
* Read the docstring + interface of: `agents/{context/manager,memory/{skills,decay,error_notebook},
  rag/*,a2a/*,agents/*,llm/*}`, `io/surfaces/{mcp/*,ui/approval,acp/*,a2a_server/*}`,
  `eval/reliability/{executor,guardrails,loopdetect,strategies/*}`, `io/gate.py`,
  `governance/security/*`, `core/{trace,observe,routing,contract}.py`.
* Ran `registry.orphans()` and `registry.stats()`.
* **Did not** run the test suite, the pressure experiment, or any backend. All verdicts are
  static-read verdicts; where I claim "X does not call Y" it is on the strength of a
  repository-wide `grep`, which is stated inline.
* **Did not** grade the eval/RL question (sibling agent's brief): chapters 16.10, 17.7, 24.7 and
  most of 27 are marked N/A on that ground.
</content>
</invoke>
