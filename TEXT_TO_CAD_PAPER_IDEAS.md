# Text-to-CAD paper idea ledger

This ledger tracks the 186 papers under
`resources/Text-to-CAD + Spatial Intelligence/extracted-md` in manifest order.
Each paper is read individually and cross-referenced against the current
HarnessCAD implementation.

Status: 5 / 186 papers reviewed.

Classifications:

- **implemented** — the repository already contains the operative capability;
- **partial** — a related capability exists, but a distinct paper contribution
  remains;
- **net-new** — deterministic/testable work can be built now;
- **research-heavy** — requires substantial model training, datasets or compute;
- **external** — requires a third-party host, sensor, service or solver.

## Batch 1 — papers 1–5

### 1. 3D-GPT: Procedural 3D Modeling with Large Language Models

Source: `3D-GPT Procedural 3D Modeling with Large Language Models.md`

Core mechanism:

- Treat procedural generation as function-subset selection followed by parameter
  inference, rather than asking an LLM to invent geometry directly.
- Split responsibility among task dispatch, conceptualization and modeling.
- Describe each tool with purpose/parameter documentation, readable code,
  information required to infer parameters and a worked invocation example.
- Preserve scene context across subsequent editing instructions.

| Build idea | Status | Repository comparison |
|---|---|---|
| Structured tool knowledge card containing documentation, required information and examples | **net-new** | MCP tool descriptions exist, but do not model required contextual facts and worked examples as a reusable card |
| Pre-plan tool-subset retrieval/dispatch | **partial** | `quality/nextop.py` ranks state-valid operations; it does not retrieve a minimal task-specific tool set from knowledge cards |
| Per-tool conceptualization that enriches missing parameter context | **partial** | `spec/interview.py` asks general missing questions, but enrichment is not conditioned on selected tool requirements |
| Context-preserving sequential edits | **implemented** | `agent/edit_session.py` |
| Agent-role ablation reporting | **partial** | research governance and metrics exist; no role-specific ablation helper |
| Advanced curve/material/shading and multimodal modeling | **research-heavy** | requires richer procedural tools and multimodal models |

### 2. A Geometric Foundation Model for Crystalline Material Discovery

Source: `A Geometric Foundation Model for Crystalline Material Discovery.md`

This is not a CAD paper, but it contributes transferable geometric-learning
principles:

- encode known symmetry/invariance directly instead of expecting augmentation
  alone to teach it;
- pretrain by matching original and perturbed geometric views;
- evaluate one representation across multiple downstream tasks.

| Build idea | Status | Repository comparison |
|---|---|---|
| Declare transformation invariants/equivariants as representation metadata and test them | **net-new** | geometry checks exist, but representation contracts do not state expected invariance |
| Generate paired perturbation views for consistency testing | **partial** | `datagen/augment.py` augments samples but does not emit invariant-consistency cases |
| Multi-task pretrained geometric foundation model | **research-heavy** | requires large geometric datasets and training |
| Flow-matching/self-supervised coordinate pretraining | **research-heavy** | model-training contribution |

### 3. A Set-based Approach for Feature Extraction of 3D CAD Models

Source: `A Set-based Approach for Feature Extraction of 3D CAD Models.md`

Core mechanism:

- Represent B-rep topology as a two-level attributed adjacency graph combining
  vertex-edge and edge-face relationships.
- Attach convex/concave/transitory attributes to faces, edges and vertices.
- Extract a *set* of plausible feature subgraphs to preserve uncertainty and
  overlapping interpretations instead of forcing one feature label.
- Keep feature extraction separate from downstream feature recognition.

| Build idea | Status | Repository comparison |
|---|---|---|
| Two-level attributed B-rep adjacency graph (TAAG) | **net-new** | `quality/featuregraph.py` reflects operation history, not native face-edge-vertex topology |
| Convexity attributes for faces/edges/vertices | **net-new** | anomaly/DFM modules do not expose this topology annotation |
| Set-valued, overlapping feature hypotheses with provenance | **net-new** | current feature narration assumes recognized operation-derived features |
| Separate extraction candidates from semantic recognition | **net-new** | no reverse-engineering candidate/recognizer boundary |
| General-surface support beyond planes and ruled surfaces | **research-heavy** | paper itself leaves this for future work |

### 4. A Solver-Aided Hierarchical Language for LLM-Driven CAD Design

Source: `A Solver-Aided Hierarchical Language for LLM-Driven CAD Design.md`

Core mechanism:

- A hierarchical DSL lets the LLM reason with semantic parts and local frames
  while a recursive geometric solver handles precision.
- Local constraints are solved inside substructures before global composition,
  improving edit locality and tractability.
- Non-smooth bounding-box constraints are solved by pruning inactive expression
  branches, then validating against the original expression and iterating when
  invalid.
- Evaluate hierarchy through local editability, not only final visual match.

| Build idea | Status | Repository comparison |
|---|---|---|
| Hierarchical semantic part scopes with local coordinate frames | **net-new** | op-DAG and skeleton exist but lack nested solver scopes |
| Recursive local-to-global constraint solving | **net-new** | `ConstraintGraph` is flat |
| Explicit local-editability metric | **net-new** | editability metrics do not measure unaffected sibling/subtree stability |
| Branch-pruned non-smooth constraint solving with original-expression revalidation | **net-new** | constraint relaxation exists but not this solver strategy |
| Solver-aided hierarchical DSL generation | **partial** | CISP is typed and solver-backed but not hierarchical |

### 5. A2Z-10M+: A-to-Z BRep Annotations

Source:
`A2Z-10M+ Geometric Deep Learning with A-to-Z BRep Annotations for AI-Assisted CAD Modeling and Reverse Engineering.md`

Core mechanism:

- Label scan/sketch samples with B-rep face, co-edge, boundary and junction IDs.
- Use proximity-aware, multi-threshold assignment and local frames rather than a
  single nearest-neighbor rule.
- Simulate staged sensor artifacts and multiple sketch skill levels.
- Generate multi-view technical captions/tags with filtering constraints and
  evaluate annotation quality using human/model consensus.
- Build a hierarchical tag ontology and heterogeneous model/tag/category graph
  for retrieval, dataset balancing and frequent design-motif mining.

| Build idea | Status | Repository comparison |
|---|---|---|
| Persistent scan/sketch↔B-rep entity annotation schema | **net-new** | reconciliation has correspondence IDs but no face/co-edge/junction annotation vocabulary |
| Proximity-aware multi-threshold assignment with local geometric frames | **net-new** | no scan-to-B-rep label assignment |
| Staged sensor-artifact augmentation profiles | **net-new** | generic augmentation lacks calibrated capture stages |
| Multi-level sketch skill/style augmentation | **net-new** | no sketch-style hierarchy |
| Multi-view caption/tag job with confidence/filtering constraints | **partial** | description and VLM checks exist separately |
| Hierarchical semantic tag ontology and heterogeneous retrieval graph | **net-new** | intent graph and RAG exist, but not model/tag/category ontology mining |
| Human/model annotation-consensus scorecards | **partial** | `dataengine/consensus.py` exists without annotation-type scorecards |
| Boundary/junction foundation-model training | **research-heavy** | requires the A2Z-scale dataset and GPUs |

## Batch-1 candidate backlog

Highest-value deterministic candidates for a later implementation wave:

1. tool knowledge cards + minimal tool-set dispatch;
2. geometric invariance contracts and perturbation consistency tests;
3. native B-rep TAAG plus set-valued feature hypotheses;
4. hierarchical constraint scopes and local-editability metrics;
5. B-rep entity annotation/correspondence schema;
6. staged scan/sketch augmentation profiles;
7. hierarchical CAD tag ontology and retrieval graph.

No implementation claim is made yet; this phase is paper mining.
