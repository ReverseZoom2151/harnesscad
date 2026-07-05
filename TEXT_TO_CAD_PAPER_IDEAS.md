# Text-to-CAD paper idea ledger

This ledger tracks the 186 papers under
`resources/Text-to-CAD + Spatial Intelligence/extracted-md` in manifest order.
Each paper is read individually and cross-referenced against the current
HarnessCAD implementation.

Status: 10 / 186 papers reviewed.

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

## Batch 2 — papers 6–10

### 6. Advancements in Computer-Aided Design Automation using Large-Scale Procedural Content Generation from the Video Game Industry

Source:
`Advancements in Computer-Aided Design Automation using Large-Scale Procedural Content Generation from the Video Game Industry.md`

Core mechanism:

- Transfer seeded procedural-content generation into engineering CAD while
  preserving repeatability and real-world constraints.
- Compose reusable assets with terrain/surface mapping, obstacle-aware
  placement and procedural connection routing.
- Run bounded trials derived from a master seed, retain the best result and its
  replay seed, and expose configuration controls to the user.
- Use procedural rules to expand a solution space, while acknowledging that
  manufacturing, tolerance and structural requirements must gate novelty.

| Build idea | Status | Repository comparison |
|---|---|---|
| Seeded, replayable procedural generation with provenance | **implemented** | `datagen/generators.py`, `datagen/pipeline.py` and `exploration/tournament.py` |
| Bounded multi-start search that retains the winning child seed and timeout reason | **implemented** | replayable trial records in `exploration/procedural.py` |
| Constraint-aware modular placement with adjacency, clustering and obstacle rules | **implemented** | placement-rule validation in `exploration/procedural.py` |
| Procedural-technique applicability registry with precision, repeatability and compute-cost tradeoffs | **implemented** | engineering-requirement selector in `exploration/procedural.py` |
| Solution-space coverage and diversity metrics for procedural generators | **implemented** | declared-dimension coverage and diversity in `exploration/procedural.py` |
| Unconstrained game-style terrain/content generation | **research-heavy** | not central to mechanical text-to-CAD and would not improve verified part generation directly |

### 7. Aligning Constraint Generation with Design Intent in Parametric CAD

Source: `Aligning Constraint Generation with Design Intent in Parametric CAD.md`

Core mechanism:

- Treat design intent as expected behavior under parameter edits, not merely a
  visually matching initial sketch.
- Score generated constraints with a solver using five outcomes:
  fully-constrained, under-constrained, over-constrained, unsolvable and stable.
- Measure stability by comparing geometry before and after solving or parameter
  edits at configurable spatial sensitivity.
- Prevent reward hacking by penalizing excessive constraint count and excessive
  use of dimensions instead of semantic geometric constraints.
- Evaluate candidate generation with pass-at-K, not only single-sample accuracy.

| Build idea | Status | Repository comparison |
|---|---|---|
| Unified five-condition sketch design-alignment scorecard | **implemented** | `quality/design_alignment.py` |
| Parameter-perturbation stability test with configurable spatial bins/tolerance | **implemented** | injected perturb-and-solve stability evaluation in `quality/design_alignment.py` |
| Constraint-economy score and dimension-to-geometric-constraint ratio | **implemented** | economy and reward-hacking diagnostics in `quality/design_alignment.py` |
| Constraint-by-constraint solver blame and drop trace | **implemented** | incremental prefix and leave-one-out traces in `quality/design_alignment.py` |
| Solver-verified pass-at-K metric for generated constraint candidates | **implemented** | unbiased solver-verified pass-at-K in `quality/design_alignment.py` |
| Solver-normalized sketch dataset preparation | **partial** | generated samples are solver-filtered; imported sketch corpora are not normalized into solved geometry before use |
| DPO/GRPO/RLOO post-training of a constraint model | **research-heavy** | requires a trained constraint policy, millions of sketches and substantial compute |

### 8. Alignist — CAD-Informed Orientation Distribution Estimation by Fusing Shape and Correspondences

Source:
`Alignist - CAD-Informed Orientation Distribution Estimation by Fusing Shape and Correspondences.md`

Transferable mechanism:

- Represent uncertain orientation as a multi-modal distribution rather than a
  single pose, especially for symmetric or partially occluded parts.
- Fuse independent shape and correspondence evidence as a product of experts.
- Precompute CAD-derived orientation priors and focus subsequent sampling near
  plausible modes while retaining ambiguity.
- Encode rotations through transformed reference-cube corners to avoid
  element-wise rotation-matrix encoding collisions.

| Build idea | Status | Repository comparison |
|---|---|---|
| Symmetry-aware, multi-modal orientation hypothesis distribution | **implemented** | `ingest/orientation.py` |
| Injectable product-of-experts fusion for shape/correspondence orientation scores | **implemented** | named expert fusion in `ingest/orientation.py` |
| Coarse-to-fine mode-focused orientation sampler with deterministic replay | **implemented** | seeded sampler and replay provenance in `ingest/orientation.py` |
| Reference-cube rotation encoding and angular-distance utilities | **implemented** | quaternion and reference-cube utilities in `ingest/orientation.py` |
| Confidence/entropy diagnostics that preserve pose ambiguity for downstream assembly | **implemented** | normalized posterior diagnostics in `ingest/orientation.py` |
| Neural SDF/SurfEmb training and image-conditioned pose inference | **research-heavy** | requires pretrained vision/geometry models and pose datasets |

### 9. Applications of Artificial Intelligence in Computer-Aided Design

Source: `Applications of Artificial Intelligence in Computer-Aided Design.md`

This short survey proposes broad capabilities rather than a new technical
method: automated modeling, material/layout recommendations, simulation,
multidisciplinary optimization, cloud collaboration, privacy and explainable
decisions.

| Build idea | Status | Repository comparison |
|---|---|---|
| Automated modeling, assembly, drawings and simulation checks | **implemented** | CISP/backends, assembly verifier, `quality/drawing.py`, and simulation verifiers |
| Material, cost, energy and sustainability optimization | **implemented** | estimate, fitness, Pareto and embodied-carbon layers |
| Real-time collaboration and cloud execution | **external** | local event/A2A contracts exist; a hosted collaborative product requires deployment infrastructure |
| Sensitive-design privacy and metadata redaction | **implemented** | `security/policy.py` and session-capture redaction |
| Explainable, auditable AI decisions | **implemented** | verifier evidence, provenance reports, traces and tool knowledge cards |
| Learned cross-industry recommendation and performance-prediction models | **research-heavy** | requires proprietary historical datasets and trained predictors |

### 10. Artificial Intelligence-Based Design of Assemblies in the FreeCAD Software

Source: `Artificial Intelligence-Based Design of Assemblies in the FreeCAD Software.md`

Core mechanism and empirical findings:

- Structure assembly prompts around dimensions, layout/constraints, element
  count, element geometry and function.
- Treat text correction, direct code editing and manual CAD editing as distinct
  intervention modes; repeated prompting becomes inefficient for detailed
  features and should trigger a handoff.
- Generate validated families of simple standard parts across parameter ranges
  and persist them as reusable libraries.
- Validate dimensions, placement, detailed features and function separately;
  plausible gross shape is not production readiness.

| Build idea | Status | Repository comparison |
|---|---|---|
| Assembly requirement completeness profile for D/L/N/G/F prompt fields | **implemented** | `quality/assembly_readiness.py` |
| Correction-attempt ledger classifying prompt, code and direct-CAD interventions | **implemented** | intervention records in `quality/assembly_readiness.py` |
| Cost/attempt-based handoff policy from prompting to code or manual CAD editing | **implemented** | diminishing-return handoff policy in `quality/assembly_readiness.py` |
| Parameter-sweep standard-part family generator with naming, validation and manifest | **implemented** | `library/family.py` |
| Separate gross-shape, dimension, placement, detailed-feature and function readiness scorecard | **implemented** | production-readiness gate in `quality/assembly_readiness.py` |
| Compare outputs from multiple generators before intervention | **implemented** | best-of-N, tournament ranking and plural verification |
| Direct FreeCAD Python-console execution | **external** | requires a FreeCAD installation/host adapter; the backend seam can accommodate it |

## Batch-2 implementation result

All deterministic, locally testable candidates from papers 6–10 are now
implemented. Remaining partial items require corpus-wide dataset conversion;
remaining external or research-heavy items require hosted infrastructure,
FreeCAD, proprietary datasets, trained neural models, or substantial compute.
