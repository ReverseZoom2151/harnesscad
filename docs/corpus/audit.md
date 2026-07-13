# Independent `resources/cad` audit

Status: complete independent pass

## Atomic closure result

The narrative audit is backed by the machine-readable
[`audit/cad_idea_register.json`](audit/cad_idea_register.json) and validator in
`audit/closure.py`.

Current validated result:

- 63/63 physical corpus files covered;
- 67 atomic ideas with source locators;
- 51 implemented ideas with existing code and test evidence;
- 7 external-system ideas with explicit dependency rationale;
- 7 research-heavy ideas with explicit data/compute rationale;
- 2 rejected non-engineering proposals with rationale;
- zero partial, open or undecomposed ideas;
- every major repository layer reverse-mapped to source ideas;
- closure validator: passing.

Closure is defined operationally, not metaphysically: every idea found by the
sequential, adversarial-search, variant-diff and visual passes has a validated
disposition, and no partial item retains an unclassified feasible slice. Adding
a corpus file, deleting evidence, introducing an open disposition or breaking a
reverse-map reference makes validation fail.

This document records an independent source-by-source audit of the 63 files in
`resources/cad`. It does not treat a prior conversation summary as evidence.

## Method

1. The six PDFs are audited through their full extracted-text twins. Page images
   are inspected separately so diagrams and UI elements are not lost.
2. Extracted chunks and summaries are checked as derivative material rather than
   counted as independent sources.
3. Every primary and standalone skill document is reviewed; differences are
   checked for net-new concepts.
4. Candidate ideas are recorded with a source locator and classified as:
   `implemented`, `partial`, `net-new`, `deferred`, or `duplicate`.
5. Implementation claims require a concrete module or test path.

## Canonical inventory

| Type | Count | Canonical treatment |
|---|---:|---|
| PDF | 6 | Full extracted-text twin plus visual figure pass |
| TXT | 9 | Six primary extracts plus three derivative Part-1 chunks |
| PNG | 10 | Direct visual inspection |
| Markdown | 38 | 17 primary skills, 17 standalone variants, four derivative summaries |
| **Total** | **63** | |

## Primary source pairs

| PDF | Full-text representation |
|---|---|
| `pdf/AI CAD Research Paper Part 1 (FINAL).pdf` | `extracted/AI CAD Research Paper Part 1 _FINAL_.txt` |
| `pdf/AI CAD Research Paper Part 2 (FINAL).pdf` | `extracted/AI CAD Research Paper Part 2 _FINAL_.txt` |
| `pdf/Gaudi MVP_ 3-Month Execution Plan.pdf` | `extracted/Gaudi MVP_ 3-Month Execution Plan.txt` |
| `pdf/Kinth Snap.pdf` | `extracted/Kinth Snap.txt` |
| `pdf/Roadmap for Building the Gaudi MVP Architectural Design Foundation Model.pdf` | `extracted/Roadmap for Building the Gaudi MVP Architectural Design Foundation Model.txt` |
| `pdf/Scale AI for Spatial Intelligence (FINAL).pdf` | `extracted/Scale AI for Spatial Intelligence _FINAL_.txt` |

The six pairs were independently checked by sampling the first, middle and last
PDF pages. Every sampled long-word anchor was present in its stated text twin
(1,051/1,051 anchors total), so the text files are valid full-document reading
representations rather than unrelated summaries.

## Derivative-file map

- `extracted/chunks/part1_c1.txt`, `part1_c2.txt`, and `part1_c3.txt` partition
  the Part-1 primary extract. Concatenating them reproduces all 65,790 lines
  exactly.
- `summaries/part1_q1_lines_1_16500.md` through
  `part1_q4_lines_49501_65790.md` summarize the four consecutive quarters of
  the same Part-1 extract.
- `docs/skills/standalone/*.md` are audited as variants of the corresponding
  `docs/skills/*.md` documents; differing bytes alone are not accepted as proof
  of differing ideas. The largest variants are materially different:
  standalone `14` has 53.2% line similarity and standalone `15` has 35.8%.

## Findings

### AI CAD Research Paper Part 1

The four quarter summaries were checked against their corresponding ranges in
the 65,789-line primary extract. The following are distinct build principles,
not repeated product-pitch language:

| Source locator | Idea | Classification | Evidence in repository |
|---|---|---|---|
| lines 18457–19528 | DXF verification with rules, anomaly detection, spatial indexing, plugins and reports | partial | Generic rules/anomaly/reporting exist; no DXF parser or spatial index |
| lines 22412–23226 | Intent/feature graph, token diffs and a visual Git history | partial | `quality/featuregraph.py`, `quality/diff.py`, `state/opdag.py`; no graph/timeline visualization |
| lines 27859–28556 | Diffusion-aware CAD generation and construction-sequence models | deferred | Runtime harness exists; no trained generative model |
| lines 28557–29244 | Confidence overlays and interactive constraint debugging | net-new | Diagnostics exist, but no confidence-overlay/debug UI |
| lines 42340–43460 | Simulation playground with environment actors and intent extraction | deferred | Kinematics and analytic simulation exist; no interactive simulator |
| lines 45760–47230 | Prompt, canvas, feedback, memory, export and construction panels | partial | Backend event/UI contracts exist; no shipped canvas |
| lines 58700–59180 | Learned next-action prediction and geometry-augmented retrieval | partial | MCTS accepts candidate expansions and RAG exists; no learned next-op service |
| lines 59180–62555 | Intent nodes with causal, spatial and functional edges | partial | Feature graph exists but does not model the full typed relation vocabulary |
| lines 63400–65000 | Graph → 2D → 3D debugging pipeline and similarity-stable CAD embeddings | partial | Sketch/feature program exists; no learned stable embedding or interactive 2D debugger |
| lines 65000–65789 | Ingest, normalize, annotate, QC, human review and consumption APIs | implemented | `ingest/`, `dataengine/`, `surfaces/mcp/` |
| lines ~63658 and 65000+ | Capture real modeling sessions/video as aligned intent→operation training data | net-new | Trajectory logging exists, but no video/session capture aligner |

Repeated strategic material—Revit positioning, construction-market pitches,
Ukraine/off-planet plans and duplicate grant drafts—does not add a distinct CAD
mechanism.

### AI CAD Research Paper Part 2

| Source locator | Idea | Classification | Evidence in repository |
|---|---|---|---|
| lines 441–545 | Translator/ingest pairing, graph correspondence, tag normalization, semantic integrity and revision feedback | partial | Import/decompile/fidelity exist; no general cross-format correspondence map |
| lines 2843–2851 | Native CAD/BIM adapters with context extraction and optional writeback | net-new | Generic backend/MCP seams exist; no native tool connector or writeback transaction |
| lines 3222–3255 | Secure upload, IP obfuscation, supported-modality policy, active learning and retrieval | partial | Active learning/retrieval exist; secure upload and IP obfuscation do not |
| lines 3499–3726 | Ontology, simulation feedback, parametric↔mesh bridge, synthetic data, private ingestion and QA | partial | Most data pipeline pieces exist; parametric↔mesh bridge and private enclave do not |
| lines 10568–10571 | Editability, assembly, B-rep and execution metrics | implemented | `bench/metrics.py`, verifier suite |
| lines 10800–11440 | Program/geometry/assembly heads with solver recycling | deferred | Execution/recycling harness exists; neural heads require model training |
| lines 11458–11482 | Clean rooms, federated training, model-in-a-box and retrieval-only sharing | net-new | No privacy-preserving deployment/training layer |
| lines 11599–11698 | Cross-source reconciliation among model, drawings and point clouds with cited quantity takeoff | partial | Reference/fidelity checks exist; multi-source reconciliation is absent |
| lines 15192 and 15571–15576 | Next-operation and graph-rewrite models | partial | No learned next-operation model |
| lines 15655–15657 | Time-to-feasibility, performance delta, clash/code rates and round-trip fidelity | implemented/partial | Most metrics exist; time-to-feasibility is not a first-class benchmark |
| lines 16278–17504 | Editable T-spline generation with topology, continuity and intent heads | deferred | Valuable freeform-CAD research direction; requires a representation, dataset and trained model |
| lines 18043–18055 | Geometry autocomplete and a simulation/QA panel | partial | Checks exist; interactive autocomplete/panel does not |
| lines 18440–18618 | Design lints, deterministic replay and cached asynchronous simulation jobs | partial | Replay/precheck exist; simulation job queue/cache is absent |
| lines 19245+ | Agent success, constraint satisfaction, human corrections, replay rate and p50/p95 latency | partial | Observability covers latency/cost and success aggregates; explicit percentile/report bundle is incomplete |

### Gaudi, Kinth and Scale-AI sources

| Source | Locator | Idea | Classification |
|---|---|---|---|
| Gaudi execution plan | lines 124–138 | staged intent decomposition → operation selection → executable script | implemented |
| Gaudi execution plan | lines 274–293 | multi-turn refinement, human-readable execution summary and graph visualization | partial |
| Gaudi execution plan | lines 314–374 | prompt-injection safety, graceful parsing recovery, latency/cost monitoring and task chunking | partial |
| Kinth | lines 10–18 | sketch/screenshot conditioning, layer/surface edits and concurrent design/simulation/check agents | partial |
| Kinth | lines 2002–2173 | requirements interview, performance generation and reusable parametric model cards | implemented |
| Kinth | lines 3457–3485 | simulation→design iteration, topology optimization and imported-part preference learning | partial |
| Scale AI | lines 336–369 | keyboard/command-first CAD interface | net-new product surface |
| Scale AI | lines 643–711 | decompose expert annotation into scalable non-expert subtasks | partial |
| Scale AI | lines 1287–1343 | distribution-aware synthetic data, on-prem privacy and historical-bias audit | partial |
| Scale AI | lines 2426–2463 | lean intent-first agentic CAD editor | partial |

### Skills and research-governance documents

All 17 primary skill documents were reviewed. Their CAD-transferable value is a
research-governance system rather than geometry functionality:

| Documents | Principle | Classification |
|---|---|---|
| `01`, `02` | dual-source literature search, collision classification and six-perspective idea debate | net-new research tooling |
| `03`, `04` | evidence-weighted advance/refine/pivot gates and adversarial result debate | net-new research tooling |
| `05`, `14` | independent reviewer ensembles, claim/evidence checks and reproducibility scoring | net-new research tooling |
| `06`, standalone `06` | source-grounded writing plus assumption tests, effect sizes and confidence intervals | net-new research tooling |
| `08`, `13` | checkpointed experiment stages with gate rollback and repair | partial; runtime checkpointing exists, research-stage orchestration does not |
| `11` | AlphaCAD-specific research workflow mapping | process reference, not runtime feature |
| `15` | citation/innovation graph for finding sparse research regions | net-new research tooling |

The 17 standalone variants were diff-reviewed. They are not exact conceptual
duplicates: standalone `06`, `13`, `14`, and `15` add concrete statistical
reporting, manual rollback, reviewer-calibration and innovation-graph workflows.

### Figure audit

| Figures | Visual-only finding | Classification |
|---|---|---|
| `gaudi_01.png`–`gaudi_09.png` | Mostly page-rendered prose; `gaudi_03` visually specifies the three-stage intent→component→script pipeline, while `gaudi_06` specifies summary, graph view and platform routing | implemented/partial |
| `kinth_10.png` | GUI exposes sketch, constraints, extrude, parts, measure, simulation, capture, BOM, animation and drawings; surrounding text emphasizes editability, revision control and downstream manufacturability | implemented/partial |

## Independent conclusion

The corpus was **not exhausted** by the previous implementation wave. The
highest-value uncovered or only-partial items are:

1. native CAD adapters with safe writeback;
2. privacy-preserving enterprise ingestion/deployment;
3. learned next-operation/geometry autocomplete;
4. interactive graph/history/constraint-debug visualization;
5. aligned modeling-session or video-to-operation data capture;
6. cross-format/cross-source correspondence and reconciliation;
7. asynchronous cached simulation execution;
8. a keyboard-first interactive CAD surface;
9. a separate research-governance pipeline with evidence gates and reviewer
   ensembles;
10. long-horizon T-spline/freeform parametric generation research.

These findings must be prioritized before claiming the directory is exhausted.

## Prioritized net-new backlog

### P0 — close core harness gaps

| Item | Acceptance boundary | Source |
|---|---|---|
| Native adapter/writeback protocol | Transactional read/apply/verify/rollback contract for at least one external CAD host, with capability discovery and idempotency tests | Part 2 lines 2843–2851; Gaudi pages 4–7 |
| Enterprise privacy boundary | File-policy validation, sensitive metadata redaction, provenance/audit events and an on-prem execution mode; no claim of federated learning until implemented | Part 2 lines 3222–3255, 11458–11482 |
| Next-operation service | Public API that ranks valid next CISP operations from current op-DAG/diagnostics, with top-k benchmark metrics; deterministic heuristic baseline before any learned model | Part 1 lines 58700–59180; Part 2 lines 15192, 15571 |
| Simulation job orchestration | Async job record, content-addressed cache, timeout/cancel/retry and solver provenance around the existing `FEASolver` seam | Part 2 lines 18440–18618 |
| Cross-source reconciliation | Compare imported solid, drawing annotations and reference metadata through persistent correspondence IDs and emit a discrepancy report | Part 2 lines 441–545, 11599–11698 |

### P1 — product and interaction layer

| Item | Acceptance boundary | Source |
|---|---|---|
| Graph/history/debug visualization | Serialize op-DAG, feature graph, branch/merge and diagnostics into a stable visual model consumed by the SSE UI contract | Part 1 lines 22412–23226, 28557–29244; Gaudi page 6 |
| Multi-turn edit session | Conversation state references the current design, produces a semantic diff and applies only an approved patch | Gaudi lines 274–293 and pages 6–7 |
| Prompt/tool security | Detect prompt-injection attempts, enforce tool allowlists and trust-boundary labels, and record blocked attempts | Gaudi page 7 |
| Keyboard-first command surface | Command grammar, discoverable mode/state, undo and accessibility tests over CISP | Scale AI lines 336–369 |
| Sketch/screenshot conditioning seam | Typed image/sketch attachment with provenance and a model-provider interface; no fake vision implementation | Kinth lines 10–18 |

### P2 — data and evaluation

| Item | Acceptance boundary | Source |
|---|---|---|
| Modeling-session capture | Align timestamped UI/video events to accepted/rejected CISP operations and export training records with consent metadata | Part 1 line ~63658 |
| Data bias audit | Extend distribution audit with source, geography, process and geometry-family coverage plus imbalance warnings | Scale AI lines 1287–1343 |
| Time-to-feasibility metrics | Iterations, wall time and solver calls until first valid result, with p50/p95 reporting | Part 2 lines 15655, 19245+ |
| Revision quantity/cost delta | BOM, mass, cost and embodied-carbon deltas between op-DAG revisions | Part 2 lines 11599–11638 |

### Separate research-governance track

- Literature/novelty collision graph.
- Six-perspective proposal and result debate.
- Evidence-consistency, statistical-reporting and reproducibility gates.
- Independent reviewer ensemble with calibrated scoring.
- Checkpointed research stages with explicit rollback.

These belong in research tooling or a plugin, not in the geometry runtime.

### Long-horizon or externally blocked

- Editable T-spline/freeform generation and continuity-aware topology models.
- Trained program, B-rep diffusion and assembly heads.
- Video/scan/point-cloud reconstruction models.
- Production CalculiX/Elmer execution and meshing.
- Full CAM/toolpath generation.
- Federated training or secure clean-room infrastructure.

Interfaces may be added in advance, but these items must not be reported as
implemented without the external systems, datasets and validation they require.

## Feasible-gap implementation status

The immediately usable subset was implemented after this audit:

| Audit item | Implementation |
|---|---|
| Adapter/writeback contract | `adapters/` transactional protocol and deterministic in-memory host; proprietary host connectors remain external |
| Local enterprise boundary | `security/policy.py` file policy, root confinement, redaction, hashes and audit provenance |
| Next-operation baseline | `quality/nextop.py` deterministic ranked suggestions and ranking metrics |
| Simulation orchestration | `quality/simjobs.py` state machine, cache, retry/timeout/cancel and solver provenance |
| Cross-source reconciliation | `ingest/reconcile.py` persistent correspondences and discrepancy reports |
| Graph/history/debug model | `surfaces/graphview.py` deterministic JSON/SVG |
| Multi-turn semantic editing | `agent/edit_session.py` preview, approval, stale-base protection and atomic rollback |
| Prompt/tool security | `security/tool_gate.py` injection indicators, allowlist, trust tiers and audit decisions |
| Keyboard command surface | `surfaces/commands.py` mode-aware, shell-free typed intents |
| Attachment seam | `agent/attachments.py` provenance, MIME/hash/root validation and provider-neutral encoder protocol |
| Modeling-session capture | `dataengine/session_capture.py` consented event/operation alignment and redaction hook |
| Bias/coverage audit | `dataengine/bias_audit.py` provenance-dimension imbalance reporting |
| Time to feasibility | `bench/feasibility.py` first-valid attempts/solver calls/elapsed and p50/p95 |
| Revision deltas | `quality/revision_delta.py` mass/cost/carbon/energy/BOM changes |
| Research governance | `research/governance.py` evidence consistency, reproducibility gates, reviewer ensemble and rollback |

The long-horizon/external list above remains deliberately unimplemented.

## Completion statement

The independent pass covered:

- all six PDFs through verified full-text representations;
- all nine text files, with the three chunks proven exact derivatives;
- all 38 Markdown documents, including semantic diffs of every standalone
  variant;
- all ten PNGs through direct visual inspection;
- repository cross-reference searches for every extracted idea cluster.

The result is not “all ideas were already implemented.” It is a complete audit
showing implemented, partial, net-new and deferred work.
