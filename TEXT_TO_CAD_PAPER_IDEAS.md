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
| Structured tool knowledge card containing documentation, required information and examples | **implemented** | `agent/tool_knowledge.py` |
| Pre-plan tool-subset retrieval/dispatch | **implemented** | deterministic minimal retrieval in `agent/tool_knowledge.py` |
| Per-tool conceptualization that enriches missing parameter context | **implemented** | context requirements and missing-context questions in `agent/tool_knowledge.py` |
| Context-preserving sequential edits | **implemented** | `agent/edit_session.py` |
| Agent-role ablation reporting | **implemented** | `research/role_ablation.py` |
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
| Declare transformation invariants/equivariants as representation metadata and test them | **implemented** | `quality/invariance.py` |
| Generate paired perturbation views for consistency testing | **implemented** | perturbation cases and consistency reports in `quality/invariance.py` |
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
| Two-level attributed B-rep adjacency graph (TAAG) | **implemented** | `quality/taag.py` |
| Convexity attributes for faces/edges/vertices | **implemented** | attributed topology records in `quality/taag.py` |
| Set-valued, overlapping feature hypotheses with provenance | **implemented** | `HypothesisSet` in `quality/taag.py` |
| Separate extraction candidates from semantic recognition | **implemented** | extractor/recognizer boundary in `quality/taag.py` |
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
| Hierarchical semantic part scopes with local coordinate frames | **implemented** | `state/constraint_hierarchy.py` |
| Recursive local-to-global constraint solving | **implemented** | child-before-parent solver in `state/constraint_hierarchy.py` |
| Explicit local-editability metric | **implemented** | subtree stability metric in `state/constraint_hierarchy.py` |
| Branch-pruned non-smooth constraint solving with original-expression revalidation | **implemented** | pruned solve and original constraint validation in `state/constraint_hierarchy.py` |
| Solver-aided hierarchical DSL generation | **implemented** | typed hierarchical scope representation in `state/constraint_hierarchy.py` |

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
| Persistent scan/sketch↔B-rep entity annotation schema | **implemented** | `ingest/brep_annotations.py` |
| Proximity-aware multi-threshold assignment with local geometric frames | **implemented** | assignment diagnostics in `ingest/brep_annotations.py` |
| Staged sensor-artifact augmentation profiles | **implemented** | `datagen/capture_augment.py` |
| Multi-level sketch skill/style augmentation | **implemented** | separate geometry-skill and render-style transforms in `datagen/capture_augment.py` |
| Multi-view caption/tag job with confidence/filtering constraints | **implemented** | `dataengine/annotation_scorecard.py` |
| Hierarchical semantic tag ontology and heterogeneous retrieval graph | **implemented** | `quality/tag_ontology.py` |
| Human/model annotation-consensus scorecards | **implemented** | type-specific quality policies in `dataengine/annotation_scorecard.py` |
| Boundary/junction foundation-model training | **research-heavy** | requires the A2Z-scale dataset and GPUs |

## Batch-1 implementation result

All deterministic, locally testable ideas extracted from papers 1–5 are now
implemented. The remaining ideas are explicitly classified as research-heavy:
large-model training, A2Z-scale dataset construction, and advanced multimodal
geometry generation. Those require external data, models, and compute and were
not represented by unusable stubs.
