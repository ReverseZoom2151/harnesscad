# Independent `resources/cad` audit

Status: in progress

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

## Derivative-file map

- `extracted/chunks/part1_c1.txt`, `part1_c2.txt`, and `part1_c3.txt` partition
  the Part-1 primary extract.
- `summaries/part1_q1_lines_1_16500.md` through
  `part1_q4_lines_49501_65790.md` summarize the four consecutive quarters of
  the same Part-1 extract.
- `docs/skills/standalone/*.md` are audited as variants of the corresponding
  `docs/skills/*.md` documents; differing bytes alone are not accepted as proof
  of differing ideas.

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
