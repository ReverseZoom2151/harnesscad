# Text-to-CAD paper idea ledger

This ledger tracks the 186 papers under
`resources/Text-to-CAD + Spatial Intelligence/extracted-md` in manifest order.
Each paper is read individually and cross-referenced against the current
HarnessCAD implementation.

Status: 35 / 186 papers reviewed.

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

## Batch 3 — papers 11–15

### 11. Atlas3D — Physically Constrained Self-Supporting Text-to-3D

Source:
`Atlas3D - Physically Constrained Self-Supporting Text-to-3D for Simulation and Fabrication.md`

Core mechanism:

- Add physical standability as a refinement objective instead of optimizing
  visual similarity alone.
- Measure rigid-body stability under gravity/contact, sample tilt directions,
  regularize adjacent normals and flatten the contact base.
- Schedule expensive physics checks intermittently during coarse-to-fine
  refinement and validate in independent simulators and physical trials.

| Build idea | Status | Repository comparison |
|---|---|---|
| COM projection, support polygon, signed stability margin and tilt robustness | **implemented** | `verifiers/standability.py` |
| Adjacent-normal and bottom-surface regularity metrics | **implemented** | `quality/mesh_stability.py` |
| Coarse/refine cadence policy for expensive physics checks | **implemented** | `quality/physics_schedule.py` |
| Stress/deflection/buckling checks | **implemented** | existing analytic simulation verifier |
| Differentiable Warp simulation, SDS/DMTet refinement and learned visual guidance | **research-heavy** | requires neural generators, GPU simulation and differentiable rendering |
| IPC cross-simulation and physical 3D-print validation | **external** | requires external simulators, fabrication and laboratory testing |

### 12. Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models

Source:
`Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models.md`

Core mechanism:

- Generate parameter and appearance descriptions through separate annotation
  channels, reconcile them, and route conflicts to manual review.
- Reverse-generate a command sequence from each description and accept it only
  when ordered LCS recovery reaches 0.9; reflect and retry at most twice.
- Preserve command/type/argument confidence and address command-distribution
  imbalance rather than exposing only one sequence-level score.

| Build idea | Status | Repository comparison |
|---|---|---|
| Ordered reverse-description LCS gate with bounded reflection | **implemented** | `dataengine/reverse_description.py` |
| Image/point-cloud description reconciliation and review routing | **implemented** | `dataengine/annotation_reconcile.py` |
| Per-command, type and argument confidence with selective correction context | **implemented** | `quality/sequence_confidence.py` |
| Command frequency, weighting and rare-operation coverage audit | **implemented** | `dataengine/command_balance.py` |
| Typed executable CAD command grammar and geometric constraints | **implemented** | CISP, grammar and verifier layers |
| Dual DeBERTa encoders, dynamic routing and BiLSTM/transformer decoder | **research-heavy** | requires model training and the annotated corpus |
| Hosted VLM/PointLLM annotation and manual review | **external** | adapters can supply candidates; services and annotators are external |

### 13. Automatic 3D CAD Models Reconstruction from 2D Orthographic Drawings

Source: `Automatic 3D CAD Models Reconstruction from 2D Orthographic Drawings.md`

Core mechanism:

- Parse normalized front/bottom/left SVG engineering views containing visible
  and hidden lines, arcs and circles.
- Match view-projection patterns to reconstruct a 3D wireframe.
- Find coplanar graph cycles, cluster nested outer/inner loops, stitch candidate
  faces, and require every manifold edge to be incident to exactly two faces.
- Evaluate reconstruction with coordinate-tolerant edge and topological face
  precision/recall/F1.

| Build idea | Status | Repository comparison |
|---|---|---|
| Orthographic input contract and safe SVG engineering-entity parser | **implemented** | `reconstruction/` input and parser stages |
| 2D edge normalization, sampling, collinear merge and deduplication | **implemented** | `reconstruction/` normalization stage |
| Table-driven multi-view projection-pattern matcher | **implemented** | `reconstruction/` matching stage |
| Wireframe graph and deterministic coplanar cycle discovery | **implemented** | `reconstruction/` wireframe/loop stages |
| Nested planar loop clustering and manifold incidence gate | **implemented** | `reconstruction/` clustering/validation stages |
| Edge/face reconstruction metrics and stage report | **implemented** | `reconstruction/` metrics and orchestrator |
| Kernel trim/sew for every curved analytic surface | **external** | exposed as an injected stitch adapter; full operation requires OCCT |
| B-spline, partial-view, section-view and raster drawing recovery | **research-heavy** | outside the paper’s own supported assumptions and requires additional inference |

### 14. B-repLer — Language-guided Editing of CAD Models

Source: `B-repLer - Language-guided Editing of CAD Models.md`

Core mechanism:

- Edit construction-history-free B-reps from high-level language.
- Generate verified face-delete/inverse-add pairs and annotate before/after
  changes bidirectionally at multiple semantic detail levels.
- Localize affected faces in selected views, generate multiple edit candidates,
  reject invalid geometry, and preserve unaffected topology and design
  relations.

| Build idea | Status | Repository comparison |
|---|---|---|
| History-free B-rep edit provider, stable face sequence and ranked K candidates | **implemented** | `editing/brep.py`, `ingest/brep_sequence.py` |
| Verified reversible face delete/add data synthesis | **implemented** | `datagen/brep_edit_pairs.py` |
| Bidirectional multilevel edit annotations with leakage checks | **implemented** | `dataengine/brep_edit_annotations.py` |
| Edit-visible view/context selection and projected bounding boxes | **implemented** | `surfaces/edit_views.py` |
| Validity@K, symmetric Chamfer, unchanged topology and relation preservation | **implemented** | `bench/edit_metrics.py` |
| Edit-complexity bins and deterministic balanced sampling | **implemented** | `dataengine/edit_complexity.py` |
| HoLa-BRep latent encoder, DINO/ROIAlign fusion and flow decoder | **research-heavy** | requires the 240K dataset, learned weights and GPU training |
| Fusion 360 face-deletion host | **external** | injected kernel protocol supports it; Fusion is not locally available |

### 15. BlenderLLM — Training Large Language Models for Computer-Aided Design with Self-improvement

Source:
`BlenderLLM - Training Large Language Models for Computer-Aided Design with Self-improvement.md`

Core mechanism:

- Balance generated instructions across 16 object classes, eight styles and
  five length bins; deduplicate and measure unit count, parameter density and
  voxel occupancy entropy.
- Use a cost-aware coarse/fine validation cascade.
- Iterate generate-filter-train-evaluate while retaining the best checkpoint
  and stopping before validation degradation.
- Evaluate per-sample binary criteria by attribute, spatial and instruction
  dimensions, routing each criterion to image or script evidence.
- Use two independent annotators, third-party adjudication, deterministic QC
  sampling and Cohen’s kappa.

| Build idea | Status | Repository comparison |
|---|---|---|
| Balanced instruction taxonomy, seeded quotas and similarity deduplication | **implemented** | `datagen/instruction_taxonomy.py` |
| Unit count, parameter density and occupancy entropy | **implemented** | `datagen/complexity.py` |
| Coarse-to-fine filter with short-circuit and cost accounting | **implemented** | `dataengine/cascade_filter.py` |
| Best-checkpoint self-improvement round controller | **implemented** | `dataengine/self_improvement.py` |
| Typed hybrid benchmark criteria, routing and aggregation | **implemented** | `bench/criteria.py` |
| Source-aware synthetic/wild split leakage and quota audit | **implemented** | `bench/splits.py` |
| Two-reviewer adjudication, deterministic QC and Cohen’s kappa | **implemented** | `dataengine/annotation_workflow.py`, `research/agreement.py` |
| Existing multi-view rendering, CADBench execution and verification | **implemented** | render, vision and bench layers |
| Qwen full-parameter SFT, learned cascade filter and iterative retraining | **research-heavy** | controller is implemented; actual learning requires GPUs and datasets |
| Blender/bpy, GPT-4o judging and forum collection | **external** | require external applications, services and network data |

## Batch-3 implementation result

All deterministic and locally testable ideas from papers 11–15 are implemented.
External kernel/application seams remain explicit, and model-training or
physical-validation claims are not simulated with unusable placeholders.

## Batch 4 — papers 16–20

### 16. BRep Boundary and Junction Detection for CAD Reverse Engineering

Source: `BRep Boundary and Junction Detection for CAD Reverse Engineering.md`

Core mechanism: classify noisy scan points as face, boundary and junction
evidence; suppress spatial duplicates; enforce chain-complex consistency; audit
severe label imbalance and error cascades.

| Build idea | Status | Repository comparison |
|---|---|---|
| Confidence NMS and boundary-first junction eligibility | **implemented** | `reconstruction/point_labels.py` |
| Pointwise scan↔B-rep chain labels and integrity checks | **implemented** | `ingest/scan_brep_labels.py` |
| Density/class prevalence, coverage and weighting audit | **implemented** | `quality/scan_label_audit.py` |
| Wireframe reconstruction and boundary/face metrics | **implemented** | existing `reconstruction/` pipeline |
| DGCNN boundary/junction inference and focal-loss training | **research-heavy** | requires scan datasets, learned models and GPUs |
| ABC/CC3D data and physical scanners | **external** | dataset and capture dependencies |

### 17. BrepGen — A B-rep Generative Diffusion Model with Structured Latent Geometry

Source: `BrepGen - A B-rep Generative Diffusion Model with Structured Latent Geometry.md`

Core mechanism: encode face→edge→vertex sampled geometry with duplicated mating
entities, recover associations through geometric clustering, align and stitch
decoded geometry, and evaluate validity, novelty, coverage and distribution.

| Build idea | Status | Repository comparison |
|---|---|---|
| Structured sampled B-rep tree, duplication/padding and decode validation | **implemented** | `reconstruction/structured_brep.py` |
| Bbox+sample duplicate clustering and mate recovery | **implemented** | `reconstruction/brep_merge.py` |
| Vertex averaging, edge orientation/alignment and consistency seam | **implemented** | `reconstruction/geometry_stitch.py` |
| Valid/unique/novel, COV/MMD and voxel-JSD metrics | **implemented** | `bench/generative_brep_metrics.py` |
| Missing-face, self-intersection, inconsistency and overmerge taxonomy | **implemented** | `reconstruction/failure_audit.py` |
| VAE/DDPM/Transformer B-rep generation | **research-heavy** | requires large geometry corpora and GPU training |
| OCCT B-spline fitting/sewing | **external** | retained as an injected kernel seam |

### 18. Bringing Attention to CAD — Boundary Representation Learning via Transformer

Source: `Bringing Attention to CAD - Boundary Representation Learning via Transformer.md`

Core mechanism: preserve the directed shell→face→loop→coedge→edge→vertex
hierarchy, continuously tokenize Bezier geometry, aggregate local topology
before global attention, and test masking robustness and per-face segmentation.

| Build idea | Status | Repository comparison |
|---|---|---|
| Directed B-rep hierarchy and referential/manifold validation | **implemented** | `ingest/brep_hierarchy.py` |
| Canonical directed cyclic-loop tokens | **implemented** | `ingest/brep_tokens.py` |
| Deterministic hierarchical topology descriptors | **implemented** | `quality/brep_descriptors.py` |
| Morton patch ordering and tokenization fidelity/overflow gates | **implemented** | `ingest/spatial_order.py`, `ingest/tokenization_audit.py` |
| Honest Bezier evaluation and trim/extractor contracts | **implemented** | `ingest/bezier_contracts.py` |
| Seeded B-rep masking robustness and face segmentation metrics | **implemented** | `bench/brep_robustness.py`, `bench/segmentation_metrics.py` |
| Complexity-stratified leakage-safe splits | **implemented** | `bench/brep_splits.py` |
| Learned BRT encoder/classifier/segmenter | **research-heavy** | requires labeled continuous-geometry data and GPU training |

### 19. CAD — Memory Efficient Convolutional Adapter

Source: `CAD - Memory Efficient Convolutional Adapter.md`

Scope note: this is an acronym collision; CAD means convolutional adapter, not
computer-aided design. Only generally useful resource/safety principles were
transferred.

| Build idea | Status | Repository comparison |
|---|---|---|
| Measure actual peak memory, latency and failures rather than parameter count | **implemented** | `research/resource_profile.py` |
| Quality/memory/latency Pareto comparison | **implemented** | `bench/resource_tradeoff.py` |
| Content-addressed frozen embedding cache | **implemented** | `vision/embedding_cache.py` |
| Finite, shape-compatible bounded residual guard | **implemented** | `vision/residual_guard.py` |
| Evidence/resource-aware model promotion gate | **implemented** | `research/model_promotion.py` |
| SAM convolutional-adapter reproduction | **research-heavy** | unrelated to core CAD and requires PyTorch/CUDA/vision data |

### 20. CAD 100K — A Comprehensive Multi-Task Dataset for Car Related Visual Anomaly Detection

Source:
`CAD 100K - A Comprehensive Multi-Task Dataset for Car Related Visual Anomaly Detection.md`

Core transferable mechanism: link hierarchical class, box and mask annotations;
enforce cross-task consistency; create group-safe real/synthetic/open-set
splits; audit long tails; gate privacy and human QC; compare single-task and
multi-task behavior.

| Build idea | Status | Repository comparison |
|---|---|---|
| Generic hierarchy/task-linked anomaly asset schema | **implemented** | `dataengine/anomaly_schema.py` |
| Cross-task box/mask/class integrity | **implemented** | `dataengine/cross_task_consistency.py` |
| Classification, detection and segmentation metrics with slicing | **implemented** | `bench/vision_metrics.py` |
| Group-safe real/synthetic/normal/few-shot/open-set splits | **implemented** | `bench/anomaly_splits.py` |
| Cross-tab rarity and source-ratio audit | **implemented** | `dataengine/anomaly_distribution.py` |
| Deterministic paired anomaly compositor seam and provenance | **implemented** | `datagen/anomaly_pairs.py` |
| Visual QC, privacy release gate and task suitability routing | **implemented** | `dataengine/visual_qc.py`, `security/image_privacy.py`, `dataengine/task_suitability.py` |
| Single/multi-task interaction and negative-transfer reporting | **implemented** | `bench/task_interaction.py` |
| Learned visual detectors/segmenters and diffusion defect synthesis | **research-heavy** | require image corpora, trained models and GPUs |
| Automotive capture, labels and privacy detectors | **external** | domain profile and collection infrastructure |

## Batch-4 implementation result

All deterministic and locally testable ideas from papers 16–20 are implemented.
Paper 19 is retained as an audited acronym false positive with only defensible
cross-domain infrastructure transferred.

## Batch 5 — papers 21–25

### 21. CAD Shape Grammar — Procedural Generation for Massive CAD Model

Source: `CAD Shape Grammar- Procedural Generation for Massive CAD Model.md`

Core mechanism: compact recursive parametric productions, inherited transform
and material state, lazy visible-branch expansion, template instancing and
compression for scenes containing millions of repeated CAD objects.

| Build idea | Status | Repository comparison |
|---|---|---|
| Seeded parameterized productions, inheritance and bounded derivation | **implemented** | `procedural/shape_grammar.py` |
| Lazy branch expansion, culling and instance batching | **implemented** | `procedural/lazy_scene.py` |
| Linear/radial/grid/pipe pattern transforms | **implemented** | `procedural/cad_patterns.py` |
| Grammar compression, reuse and rule-coverage metrics | **implemented** | `quality/grammar_compression.py` |
| GPU-resident generation/rendering and plant-scale datasets | **external** | requires specialized renderer, data and benchmark hardware |

### 22. CAD(Block) — Photorealistic 3D Generation via Adversarial Distillation

Source: `CAD(Block) - Photorealistic 3D Generation via Adversarial Distillation.md`

Core transferable mechanism: prune weak camera evidence, preserve multiview
geometry through an appearance upscaling stage, cache immutable priors, and
evaluate view quality, diversity and angular consistency separately.

| Build idea | Status | Repository comparison |
|---|---|---|
| Geometry/semantic camera pruning with coverage fallback | **implemented** | `quality/camera_pruning.py` |
| Pairwise multiview consistency and outlier localization | **implemented** | `quality/multiview_consistency.py` |
| Raw/upscaled appearance-vs-geometry quality gate | **implemented** | `quality/render_stages.py` |
| Content-addressed prompt/camera/refiner prior cache | **implemented** | `dataengine/prior_cache.py` |
| Render-distribution quality/diversity/coverage summary | **implemented** | `bench/render_distribution.py` |
| StyleGAN/triplane/GAN/diffusion distillation and neural rendering | **research-heavy** | requires models, GPUs and multi-day training |

### 23. CAD-Assistant — Tool-Augmented VLLMs as Generic CAD Task Solvers

Source: `CAD-Assistant - Tool-Augmented VLLMs as Generic CAD Task Solvers.md`

Core mechanism: iteratively plan, execute tools and replan from digest-bound
visual/structured observations; retrieve only relevant tool documentation;
ground sketch JSON in primitive-ID overlays; measure solver impact, tool use,
QA evidence and task cost.

| Build idea | Status | Repository comparison |
|---|---|---|
| Verifier-gated termination and atomic CAD observations | **implemented** | `agent/termination.py`, `agent/cad_observation.py` |
| Tool retrieval recall/budget and trajectory validity metrics | **implemented** | `bench/tool_retrieval.py`, `bench/tool_trajectory.py` |
| Over-parameterized sketch schema and primitive-ID overlay | **implemented** | `quality/sketch_serialization.py`, `surfaces/id_overlay.py` |
| Constraint movement/DOF impact and analytic cross-sections | **implemented** | `quality/constraint_impact.py`, `ingest/cross_section.py` |
| Grounded QA, sketch PF1/CF1 and provider-cost reports | **implemented** | `bench/cad_qa.py`, `bench/sketch_metrics.py`, `bench/agent_cost.py` |
| Learned sketch/scan recognizers and FreeCAD host | **external** | require model weights and installed FreeCAD |
| Arbitrary planner-authored Python | **rejected** | typed CISP/MCP execution is safer and more auditable |

### 24. CAD-Coder — An Open-Source Vision-Language Model for Computer-Aided Design Code Generation

Source:
`CAD-Coder - An Open-Source Vision-Language Model for Computer-Aided Design Code Generation.md`

Core transferable mechanism: deterministic editable code emission, immutable
image/code manifests, explicit complexity/overflow handling, static execution
safety, inertia-normalized geometry alignment, conditioning robustness and
capability-retention evaluation.

| Build idea | Status | Repository comparison |
|---|---|---|
| Restricted typed-op→CadQuery code emitter and AST normalization | **implemented** | `datagen/cadquery_codegen.py`, `quality/cad_code_normalize.py` |
| Image/code and generation provenance manifests | **implemented** | `datagen/image_code_manifest.py`, `dataengine/generation_manifest.py` |
| Code complexity balance and no-silent-truncation contract | **implemented** | `dataengine/code_complexity.py`, `llm/generation_contract.py` |
| Static code safety/VSR and injected solid-IoU alignment | **implemented** | `bench/code_execution.py`, `bench/solid_iou.py` |
| Image-conditioning and capability-retention benchmarks | **implemented** | `bench/image_conditioning.py`, `bench/capability_retention.py` |
| LLaVA/CAD-Coder fine-tuning and real-photo corpus | **research-heavy** | requires datasets, weights, H100-class compute and physical capture |

### 25. CAD-Coder — Text-Guided CAD Files Code Generation

Source: `CAD-Coder - Text-Guided CAD Files Code Generation.md`

Core mechanism: generate editable DXF through standardized source programs,
legal family-specific parameter sweeps and intent comments; evaluate safe AST
structure, parameters, drafting annotations, unbiased pass-at-K, geometry and
cross-platform evidence.

| Build idea | Status | Repository comparison |
|---|---|---|
| Neutral DXF entities/layers/units/annotation contracts | **implemented** | `formats/dxf_contract.py` |
| Typed dimension/tolerance/chamfer/roughness annotations | **implemented** | `cisp/annotations.py` |
| Seeded legal parent-script family generation and replay | **implemented** | `datagen/script_family.py` |
| Intent-comment ambiguity lint and inheritance | **implemented** | `datagen/code_comments.py` |
| Safe AST function/parameter/annotation metrics and unbiased pass-at-K | **implemented** | `bench/code_metrics.py`, `bench/code_passk.py` |
| Geometry distance and evidence-only cross-platform matrix | **implemented** | `bench/geometry_distance.py`, `bench/cross_platform.py` |
| Prompt/script/artifact lineage and paired ablations | **implemented** | `dataengine/cfsc_record.py`, `research/ablation_matrix.py` |
| ezdxf and commercial-platform compatibility | **external** | adapters and licensed hosts are required for real claims |
| LoRA/RL training and the 29K artifact corpus | **research-heavy** | requires model training, curated data and GPU compute |

## Batch-5 implementation result

All deterministic and locally testable findings from papers 21–25 are
implemented. Unsafe arbitrary code execution is explicitly rejected, and
external/model-dependent claims remain behind typed seams.

## Batch 6 — papers 26–30

### 26. CAD-Coder — Text-to-CAD Generation with Chain-of-Thought and Geometric Reward

| Build idea | Status | Repository comparison |
|---|---|---|
| Five-stage CAD plan and strict reasoning/code envelope | **implemented** | `agent/cad_plan.py` |
| Execution-first piecewise geometry/format reward | **implemented** | `quality/cad_reward.py` |
| Best-candidate geometry triplets, quality tiers and lineage | **implemented** | `datagen/geometry_triplets.py` |
| Versioned normalization, squared Chamfer and invalidity protocol | **implemented** | `bench/cad_geometry_protocol.py` |
| Thin/interior/multiresolution sampling guard | **implemented** | `quality/sampling_guard.py` |
| CoT/code/geometry/review provenance and leakage | **implemented** | `dataengine/cot_records.py` |
| Qwen SFT/GRPO and DeepSeek annotation | **research-heavy** | requires datasets and A800-class training |

### 27. CAD-Editor — Locate-then-Infill Framework

| Build idea | Status | Repository comparison |
|---|---|---|
| Canonical LCS locate masks, infill and immutable-context guard | **implemented** | `editing/locate_infill.py` |
| Base/reverse/cross-variant triplet synthesis | **implemented** | `datagen/edit_triplets.py` |
| Staged visual/sequence/difference edit captions and filters | **implemented** | `dataengine/edit_caption.py`, `dataengine/edit_filters.py` |
| Candidate/render/verifier/human-selection records | **implemented** | `dataengine/selective_edits.py` |
| Directional/JSD/VR/CD edit metrics and lineage-safe splits | **implemented** | `bench/edit_alignment.py`, `bench/edit_splits.py` |
| Append-only iterative edit/rollback provenance | **implemented** | `editing/iterative_session.py` |
| Locator/infiller LoRA, CLIP and crowd evaluation | **research-heavy** | requires models, datasets, GPUs and human raters |

### 28. CAD-Editor — Text-Based CAD Editing Through Synthetic Data

| Build idea | Status | Repository comparison |
|---|---|---|
| Visible/executable 1–3 edit triplet contract | **implemented** | `dataengine/edit_triplets.py` |
| Directional image/text edit alignment | **implemented** | `quality/directional_edit_alignment.py` |
| Intended-vs-actual edit locality and collateral diagnostics | **implemented** | `quality/edit_locality.py` |
| Monotonic iterative edit policy with rollback/oscillation stop | **implemented** | `agent/iterative_edit_policy.py` |
| LoRA, MLLM captions, CLIP and human preference collection | **external/research-heavy** | deterministic orchestration exists; learned services do not |

### 29. CAD-GPT — Spatial Reasoning-Enhanced Multimodal CAD Sequences

| Build idea | Status | Repository comparison |
|---|---|---|
| Reversible global-frame/local-sketch spatial tokens | **implemented** | `ingest/sketch_frame_tokens.py` |
| Separate command/scalar/origin/orientation/local-coordinate accuracy | **implemented** | `quality/spatial_sequence_accuracy.py` |
| Frame, extrusion-normal and cumulative-drift coherence | **implemented** | `quality/frame_coherence.py` |
| Deterministic spatial challenge fixtures/report | **implemented** | `bench/spatial_challenge_set.py` |
| LLaVA encoders and learned spatial embeddings | **research-heavy** | requires multimodal data and GPU training |

### 30. CAD-Judge — Efficient Morphological Grading and Verification

| Build idea | Status | Repository comparison |
|---|---|---|
| Cached stage-aware compiler/morphology judge and honest verification levels | **implemented** | `bench/compiler_judge.py` |
| Area-weighted deterministic mesh sampling | **implemented** | `geometry/mesh_sampling.py` |
| Binary preference records/sampling and KTO utility rows | **implemented** | `dataengine/binary_preferences.py`, `binary_sampling.py`, `kto.py` |
| Threshold calibration and compiler diagnostics | **implemented** | `bench/judge_calibration.py`, `reliability/compiler_diagnostics.py` |
| Review plateau, command F1 and failure-aware morphology reports | **implemented** | `bench/review_iterations.py`, `command_metrics.py`, `morphology_report.py` |
| Reward-hacking, efficiency and controlled judge ablations | **implemented** | `bench/reward_hacking.py`, `judge_efficiency.py`, `research/judge_ablation.py` |
| KTO/LoRA model training | **research-heavy** | data rows/math exist; optimization requires models and GPUs |

## Batch-6 implementation result

All deterministic and locally testable findings from papers 26–30 are
implemented. Compiler success, morphology and requirements evidence remain
separate so validity cannot be misreported as semantic correctness.

## Batch 7 — papers 31–35

### 31. CAD-Llama — Parametric 3D Model Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| Lossless Loop/Component structural abstraction and repeated macros | **implemented** | `quality/spcc_structure.py` |
| Evidence-backed five-level CAD complexity | **implemented** | `quality/cad_complexity.py` |
| Hierarchical component/global annotation coverage | **implemented** | `dataengine/hierarchical_cad_annotation.py` |
| Cross-domain structural vocabulary/complexity shift | **implemented** | `bench/cad_domain_shift.py` |
| LLaMA pretraining, LoRA and GPT/CLIP annotation | **research-heavy** | requires models, corpora and GPUs |

### 32. CAD-LLM — Large Language Model for CAD Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| Prefix-ratio completion curves and AUC | **implemented** | `bench/prefix_completion.py` |
| Entity accuracy, strict sketch accuracy and macro/micro CAD F1 | **implemented** | `bench/sketch_sequence_metrics.py` |
| GPT/PEFT training on SketchGraphs | **research-heavy** | requires weights, dataset and training |

### 33. CAD-MLLM — Unifying Multimodality-Conditioned CAD Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| Aligned command/text/view/point-normal records | **implemented** | `dataengine/omnicad_record.py` |
| Split-safe command prefixes and topology-gated modifier ablation | **implemented** | `datagen/command_prefixes.py`, `modifier_ablation.py` |
| Reproducible multiview/point capture and modality curriculum | **implemented** | `datagen/multimodal_capture.py`, `dataengine/modality_schedule.py` |
| SegE/DangEL/SIR/FluxEE mesh metrics | **implemented** | `bench/mesh_topology.py` |
| Modality robustness, complementarity, splits and fusion policy | **implemented** | `bench/modality_robustness.py`, `omnicad_splits.py`, `quality/modality_fusion.py` |
| DINO/Michelangelo/Vicuna multimodal training | **research-heavy** | requires the 453K corpus and H800-class compute |

### 34. CAD-Prompted Generative Models — A Pathway to Feasible and Novel Engineering Designs

| Build idea | Status | Repository comparison |
|---|---|---|
| Verified-feasible CAD render retrieval | **implemented** | `rag/cad_render_retrieval.py` |
| Provider-neutral image-weight/seed sweep | **implemented** | `exploration/image_prompt_sweep.py` |
| Blinded rating QC, Spearman, Mann–Whitney and Pareto analysis | **implemented** | `bench/feasibility_novelty.py` |
| Per-setting prompt similarity and off-diagonal handling | **implemented** | `bench/prompt_similarity.py` |
| Evidence-calibrated design-stage policy without extrapolation | **implemented** | `quality/design_stage_policy.py` |
| Prompt/render/output/rater lineage and perceived-vs-actual claims | **implemented** | `dataengine/cad_prompt_record.py`, `bench/perceived_actual_gap.py` |
| T2I models, BIKED data and human study | **external/research-heavy** | requires services, licensed data and participants |

### 35. CAD-Prompted Generative Models — A Pathway To Feasible And Novel Engineering

**Duplicate alias.** From `## Page 1` through EOF this paper is byte-identical
to paper 34 (SHA-256
`27c8e91207290d7d87f363c4b9a3cc30653500e3e88b4a729b7f180c14ac0d4d`).
Only extraction metadata and the title differ. It contributes no independent
mechanism or evidence and is intentionally implemented once.

## Batch-7 implementation result

All deterministic findings from papers 31–35 are implemented. Exact duplicate
paper 35 is recorded as an alias rather than double-counted.
