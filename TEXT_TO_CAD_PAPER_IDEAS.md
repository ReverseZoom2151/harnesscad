# Text-to-CAD paper idea ledger

This ledger tracks the 186 papers under
`resources/Text-to-CAD + Spatial Intelligence/extracted-md` in manifest order.
Each paper is read individually and cross-referenced against the current
HarnessCAD implementation.

Status: 45 / 186 papers reviewed.

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

## Batch 8 — papers 36–40

### 36. CAD-Prompted SAM3 — Geometry-Conditioned Instance Segmentation

| Build idea | Status | Repository comparison |
|---|---|---|
| Canonical 12-view geometry prompt and foreground-point bundles | **implemented** | `surfaces/canonical_views.py`, `vision/geometry_prompt.py`, `mask_sampling.py` |
| Seeded domain-randomization manifests and shortcut audit | **implemented** | `datagen/domain_randomization.py` |
| One-to-many matching, mask NMS, PQ and instance F1 | **implemented** | `vision/instance_matching.py`, `bench/instance_segmentation.py` |
| Prompt-conditioned split/leakage and appearance invariance | **implemented** | `bench/geometry_prompted_segmentation.py`, `appearance_invariance.py` |
| SAM3 fusion and Isaac/Blender synthetic training | **research-heavy/external** | requires models, simulators, assets and GPUs |

### 37. CAD-Recode — Reverse Engineering CAD Code from Point Clouds

| Build idea | Status | Repository comparison |
|---|---|---|
| Canonical point-cloud normalization/order and Fourier features | **implemented** | `ingest/point_cloud.py`, `fourier_features.py` |
| Validity-first pointcloud candidate selection and K scaling | **implemented** | `reconstruction/pointcloud_candidates.py`, `bench/candidate_scaling.py` |
| Sketch-boolean recipe and verified reverse-engineering records | **implemented** | `datagen/sketch_boolean.py`, `reverse_engineering.py` |
| Code modularity and morphology-gated high-level abstraction | **implemented** | `dataengine/code_modularity.py`, `quality/cad_abstraction.py` |
| Quantization/expressivity risk and point-cloud robustness/budget | **implemented** | `quality/quantization_risk.py`, `reconstruction/expressivity.py`, `bench/pointcloud_robustness.py`, `point_budget.py` |
| Safe semantic parameter exposure | **implemented** | `quality/parameter_exposure.py` |
| Qwen point projector, 1M corpus and real scans | **research-heavy/external** | requires models, data, kernels and H100 training |

### 38. CAD-Tokenizer — Modality-Specific Tokenization

| Build idea | Status | Repository comparison |
|---|---|---|
| Incremental decoder legality FSA | **implemented** | `grammar_fsa.py` |
| Exact-coverage primitive semantic pooling | **implemented** | `quality/primitive_pooling.py` |
| Reconstruction/compression/invalidity Pareto frontier | **implemented** | `bench/tokenizer_frontier.py` |
| Tokenizer/backbone nested exposure split audit | **implemented** | `bench/tokenizer_split_audit.py` |
| VQ-VAE/codebook and LLaMA training | **research-heavy** | requires datasets, learned tokenizer and GPUs |

### 39. CAD-VAE — Correlation-Aware Latents for Fair Disentanglement

**Unrelated acronym collision.** CAD denotes a correlation-aware
disentanglement VAE for fairness-focused image datasets, not computer-aided
design. It contributes no geometry, sketch, B-rep or manufacturing mechanism;
its learned architecture is intentionally excluded.

### 40. CAD-VLM — Language and Vision for Parametric CAD Sketches

| Build idea | Status | Repository comparison |
|---|---|---|
| Versioned 1m/[1,64] primitive and constraint codec | **implemented** | `ingest/cadvlm_codec.py` |
| Full/partial sequence-render records and paired prefix generation | **implemented** | `dataengine/sketch_modal_record.py`, `datagen/paired_sketch_prefix.py` |
| Exact/tolerance entity, sketch, CAD-F1 and sliced metrics | **implemented** | `bench/cadvlm_metrics.py` |
| Crossmodal sketch consistency and precise/handdrawn/noisy conditions | **implemented** | `quality/sketch_crossmodal.py`, `datagen/sketch_image_conditions.py` |
| Constraint-label stability, modality ablations and 13-kind ontology | **implemented** | `quality/constraint_label_stability.py`, `bench/task_modality_ablation.py`, `dataengine/sketch_constraint_ontology.py` |
| ViT-MAE/CodeT5 contrastive multimodal training | **research-heavy** | requires SketchGraphs and week-scale A100 training |

## Batch-8 implementation result

All deterministic findings from papers 36–40 are implemented. Paper 39 is
recorded as an unrelated acronym collision.

## Batch 9 — papers 41–45

### 41. CADDesigner — Conceptual Design Based on a General-Purpose Agent

| Build idea | Status | Repository comparison |
|---|---|---|
| Explicit typed operation bindings, snapshots and stale-handle checks | **implemented** | `cisp/explicit_context.py` |
| Structured parser/runtime/kernel/type error envelopes | **implemented** | `reliability/code_error.py` |
| CAD API knowledge/example coherence and safe chunking | **implemented** | `rag/cad_api_knowledge.py` |
| Correction convergence/recovery/oscillation metrics | **implemented** | `bench/correction_trajectory.py` |
| Hosted LLM/VLM and shell CAD execution | **external** | typed agent seams exist; services/hosts are not bundled |

### 42. CADDreamer — CAD Object Generation from Single-view Images

| Build idea | Status | Repository comparison |
|---|---|---|
| Analytic primitive relation inference/projection | **implemented** | `reconstruction/primitive_relations.py` |
| Adjacency-restricted pair/triple primitive intersections | **implemented** | `reconstruction/primitive_intersections.py` |
| Iterative primitive seam projection with regression rollback | **implemented** | `reconstruction/primitive_stitch.py` |
| View observability/thin-feature audit and view recommendations | **implemented** | `quality/view_coverage.py` |
| Primitive relation, normal and hanging-face metrics | **implemented** | `bench/primitive_reconstruction_metrics.py` |
| Cross-view diffusion, NeuS and segmentation models | **research-heavy** | require image corpora, neural models and GPUs |

### 43. CADEvolve — Creating Realistic CAD via Program Evolution

| Build idea | Status | Repository comparison |
|---|---|---|
| Immutable evolutionary generator lineage and saturation budgets | **implemented** | `datagen/evolution.py` |
| Quality-diversity parameter archive and admission gates | **implemented** | `datagen/parameter_qd.py` |
| Trace slicing and exact 24 cube-rotation rewrites | **implemented** | `datagen/trace_slice.py`, `cube_rotations.py` |
| Template-collapse and operator-distribution audits | **implemented** | `dataengine/template_collapse.py`, `bench/operator_profile.py` |
| Evolution dynamics and ordered admission validation | **implemented** | `bench/evolution_dynamics.py`, `datagen/evolution_validation.py` |
| LLM/VLM proposal, CMA-ES and training corpora | **research-heavy/external** | deterministic orchestration exists; learned proposal systems do not |

### 44. CADgpt — NLP for Enhanced CAD Workflows

| Build idea | Status | Repository comparison |
|---|---|---|
| Host-neutral intent/assumption/clarification envelope | **implemented** | `agent/intent_resolution.py` |
| Safe Rhino/Grasshopper capability DTO/protocol | **implemented** | `adapters/rhino_contract.py` |
| Preview-confirm-execute-refine host lineage | **implemented** | `agent/host_feedback.py` |
| Executable NL-CAD scenario casebook | **implemented** | `bench/nl_cad_casebook.py` |
| Rhino/Grasshopper plugins and hosted GPT | **external** | require proprietary host installation and services |

### 45. CADKnitter — Compositional CAD Generation from Text and Geometry Guidance

| Build idea | Status | Repository comparison |
|---|---|---|
| Directed assembly-pair records and shared normalization | **implemented** | `dataengine/assembly_pair_record.py`, `ingest/assembly_normalization.py` |
| Bidirectional sampled contact evidence and correspondence | **implemented** | `ingest/contact_faces.py`, `quality/contact_correspondence.py` |
| Scheduled contact objectives and guided candidate search | **implemented** | `quality/contact_objective.py`, `exploration/guided_contact_search.py` |
| CD/IV/PR/VR metrics, interaction policy and heatmaps | **implemented** | `bench/compositional_metrics.py`, `quality/assembly_interaction.py`, `bench/contact_heatmap.py` |
| Dataset filters and directional caption workflow | **implemented** | `dataengine/knitcad_filters.py`, `assembly_caption_workflow.py` |
| B-rep diffusion, FGW/ULA guidance and Fusion datasets | **research-heavy/external** | require learned models, optimal-transport tooling and licensed data |

## Batch-9 implementation result

All deterministic and locally testable findings from papers 41–45 are
implemented.

### 46. CADmium — Fine-Tuning Code Language Models for Text-Driven Sequential CAD Design

| Build idea | Status | Repository comparison |
|---|---|---|
| Sphericity discrepancy, discrete mean-curvature difference, exact Euler-characteristic match, mesh-derived watertightness gating, mesh area/volume primitives | **implemented** | `bench/cadmium_mesh_metrics.py` |
| Corpus annotation statistics (conciseness band, unique-word ratio, Heaps vocabulary growth, decimal-precision distribution, head-to-head corpus comparison) | **implemented** | `dataengine/cadmium_annotation_stats.py` |
| GPT-4.1 multimodal annotation, Qwen2.5-Coder LoRA SFT, LLM-as-a-judge, Onshape-FeatureScript JSON normalization | **research-heavy/external** | learned models / proprietary tooling |

### 47. CADMorph — Geometry-Driven Parametric CAD Editing via a Plan-Generate-Verify Loop

| Build idea | Status | Repository comparison |
|---|---|---|
| Voxelised truncated-SDF grid with analytic Boolean algebra + dissimilarity proxies (tSDF distance, voxel IoU, occupancy Hamming) | **implemented** | `geometry/cadmorph_tsdf.py` |
| Relative-contribution planning with top-K masking (leave-one-out proxy for cross-attention read-out) | **implemented** | `editing/cadmorph_plan.py` |
| Distance-to-target verification with a cross-iteration priority queue + structure-preservation objective | **implemented** | `editing/cadmorph_verify.py` |
| Plan-generate-verify orchestrator (queue-seeded, deterministic) | **implemented** | `editing/cadmorph_loop.py` |
| P2S latent-diffusion model, MPP LLM infiller, cross-attention contribution map | **research-heavy/external** | learned models; injected as callables |

### 48. CADParser — A Learning Approach of Sequence Modeling for B-Rep CAD

| Build idea | Status | Repository comparison |
|---|---|---|
| Fixed-width command/token schema (13-symbol vocab, 19-slot param vector, DeepCAD quantization + 257-dim one-hot, NC=32 packing) | **implemented** | `reconstruction/cadparser_schema.py` |
| Sequence-validity grammar/FSA with decode-time legal-continuation masking | **implemented** | `reconstruction/cadparser_grammar.py` |
| B-rep -> coedge graph (faces/edges/coedges, typed adjacency, geometry node features) | **implemented** | `reconstruction/cadparser_brep_graph.py` |
| Dataset statistics + back-to-front truncation augmentation | **implemented** | `reconstruction/cadparser_sequence_stats.py` |
| Learned graph-encoder/Transformer-decoder parser; 40k SolidWorks dataset | **research-heavy/external** | model training / proprietary data |

### 49. CADReasoner — Iterative Program Editing for CAD Reverse Engineering

| Build idea | Status | Repository comparison |
|---|---|---|
| Directional geometry-discrepancy field (nearest-surface offsets + farthest-point selection; t=1 null encoding) | **implemented** | `editing/cadreasoner_discrepancy.py` |
| Closed-loop render-compare-refine reverse-engineering harness (best-so-far by residual; selection-vs-reporting split) | **implemented** | `editing/cadreasoner_edit_loop.py` |
| Geometry-guided beam over edit iterations with render-budget bounds | **implemented** | `editing/cadreasoner_beam.py` |
| Occlusion-based scan-simulation defect pipeline (spherical depth buffer, seeded noise, hole punching) | **implemented** | `datagen/cadreasoner_scansim.py` |
| Qwen2-VL editor + SFT curriculum, multi-view RGB overlay backbone, Poisson reconstruction | **research-heavy/external** | learned VLM / external mesh kernel |

### 50. CADReview — Automatically Reviewing CAD Programs with Error Detection and Correction

| Build idea | Status | Repository comparison |
|---|---|---|
| Eight-scenario CAD-program error taxonomy | **implemented** | `cadreview_taxonomy.py` |
| Brace-aware OpenSCAD block segmenter | **implemented** | `cadreview_blocks.py` |
| Reference-grounded error detector (type + offending block id) | **implemented** | `cadreview_detect.py` |
| Seeded error injector (one error per sample, 8 types) | **implemented** | `cadreview_errorgen.py` |
| Automated corrector with re-detect round-trip guarantee | **implemented** | `cadreview_correct.py` |
| Structured review report + V_d diagnostic reward + Acc scorer | **implemented** | `cadreview_review.py` |
| 8-bit spatial quantization + SGO numeric-token reweighting | **implemented** | `cadreview_quantize.py` |
| Learned feedback generator / code editor (GCR, SGO), RL/DPO with V_v/V_p rewards, multiview rendering | **research-heavy/external** | learned MLLMs / rendering pipeline |

## Batch-10 implementation result

All deterministic and locally testable findings from papers 46-50 are
implemented. Suite: 1791 tests, all passing.

### 51. cadrille — Multi-modal CAD Reconstruction with Reinforcement Learning

| Build idea | Status | Repository comparison |
|---|---|---|
| Verifiable reward shaping (IoU + invalidity) + hard-example mining | **implemented** | `dataengine/cadrille_reward.py` |
| Dr.CPPO (std-free Dr.GRPO advantages + top-\|A\| CPPO selection + clipped PPO surrogate) | **implemented** | `dataengine/cadrille_drcppo.py` |
| DPO preference-pair construction from K samples | **implemented** | `dataengine/cadrille_preference_pairs.py` |
| Point-cloud adapter (unit-cube + furthest-point sampling) | **implemented** | `reconstruction/cadrille_pointcloud_adapter.py` |
| 2x2 multi-view image grid adapter | **implemented** | `vision/cadrille_multiview_grid.py` |
| Reconstruction eval (median Chamfer, IoU%, invalidity ratio) | **implemented** | `bench/cadrille_metrics.py` |
| Orientation-invariant discrete-ICP over 24 axis rotations | **implemented** | `bench/cadrille_orientation_align.py` |
| Learned Qwen2-VL policy + SFT/RL training + LLM caption pipeline | **research-heavy/external** | learned VLM / GPU training |

### 52. CADSmith — Multi-Agent CAD Generation with Programmatic Geometric Validation

| Build idea | Status | Repository comparison |
|---|---|---|
| Dual nested correction loops (execution-error inner + geometric-refinement outer, agents injected) | **implemented** | `cadsmith_dual_loop.py` |
| Structured design-plan schema + JSON handoff + convention checker | **implemented** | `cadsmith_design_plan.py` |
| Error-solution pattern KB over CadQuery/OCCT failure modes | **implemented** | `cadsmith_error_patterns.py` |
| Kernel-metrics record + hard validity gate + plan-discrepancy feedback | **implemented** | `cadsmith_kernel_metrics.py` |
| Three-view render camera spec | **implemented** | `cadsmith_three_view.py` |
| Absolute-mm metrics (Kabsch + ICP, F1@1mm, voxel IoU, Chamfer) | **implemented** | `cadsmith_abs_metrics.py` |
| Judge escalation / anti-oscillation policy | **implemented** | `cadsmith_escalation.py` |
| T1/T2/T3 benchmark difficulty tiers | **implemented** | `cadsmith_tiers.py` |
| VLM-as-Judge (Claude Opus); RAG-over-API-docs generation | **research-heavy/external** | learned VLM / LLM |

### 53. CADTalk — An Algorithm and Benchmark for Semantic Commenting of CAD Programs

| Build idea | Status | Repository comparison |
|---|---|---|
| Hierarchical commentable-block parser (nested OpenSCAD tree, irreducible marking, ancestor collection) | **implemented** | `cadtalk_parser.py` |
| Benchmark metrics (block accuracy + semantic IoU with synonym normalization) | **implemented** | `cadtalk_metrics.py` |
| Multi-view part-label voting with progressive confidence thresholds | **implemented** | `cadtalk_voting.py` |
| Machine-made labelled-primitive program synthesis (round-trips through the parser) | **implemented** | `cadtalk_primitive_program.py` |
| Point-cloud label transfer (max-vote / multi-label / IoU) | **implemented** | `cadtalk_label_transfer.py` |
| ControlNet image translation; Grounding-DINO + SAM; ChatGPT commenting | **research-heavy/external** | learned foundation models |

### 54. CadVLM — Bridging Language and Vision in the Generation of Parametric CAD Sketches

| Build idea | Status | Repository comparison |
|---|---|---|
| Sketch-to-pixel rasteriser (Bresenham lines, midpoint circles, circumcircle arcs) | **implemented** | `vision/cadvlm_sketch_raster.py` |
| ViT-MAE patch-masking pipeline (patchify, 75% masking, image-decoding MSE) | **implemented** | `vision/cadvlm_patch_mask.py` |
| <ENTITY>/<TOKEN> entity-level sequence layout with reversible parse | **implemented** | `ingest/cadvlm_entity_sequence.py` |
| Whole-sketch primitive/constraint validity checker | **implemented** | `ingest/cadvlm_sketch_validity.py` |
| Sketch codec, constraint ontology, Entity/Sketch-Accuracy/CAD-F1, prefix pairs, crossmodal, ablation | **already in repo (prior near-duplicate paper)** | `ingest/cadvlm_codec.py`, `bench/cadvlm_metrics.py`, etc. |
| CadVLM two-stream ViT-MAE + CodeT5+ encoder-decoder training | **research-heavy/external** | learned models / GPU |

### 55. CAM — CAD Point Cloud Part Segmentation via Few-Shot Learning

| Build idea | Status | Repository comparison |
|---|---|---|
| Deterministic geometric point features (local-PCA linearity/planarity/scattering/curvature + max-pooled edge feature) | **implemented** | `reconstruction/fewshot_partseg_features.py` |
| Multi-prototype nearest-prototype segmenter (FPS anchors + bucketed prototypes) | **implemented** | `reconstruction/fewshot_partseg_prototypes.py` |
| C-way K-shot episode construction with background-aware remapping | **implemented** | `reconstruction/fewshot_partseg_episodes.py` |
| Part-seg IoU/mIoU/instance-mIoU metrics | **implemented** | `reconstruction/fewshot_partseg_metrics.py` |
| Transductive label-propagation head (Gaussian kNN affinity, iterative + closed-form) | **implemented** | `reconstruction/fewshot_partseg_labelprop.py` |
| DGCNN backbone training, T-Net, center-loss training strategy | **research-heavy/external** | neural training objectives |

## Batch-11 implementation result

All deterministic and locally testable findings from papers 51-55 are
implemented. Suite: 2106 tests, all passing.

### 56. ChatCAD+ — Towards a Universal and Reliable Interactive CAD using LLMs

| Build idea | Status | Repository comparison |
|---|---|---|
| prob2text graded numeric-score verbaliser (calibrated language bands) | **implemented** | `chatcadplus_prob2text.py` |
| Spherical-projection KD-tree for exact O(log n) cosine top-k | **implemented** | `chatcadplus_sphere_retrieval.py` |
| Hierarchical in-context retrieval, DFS knowledge traversal, domain-argmax routing, medical report NLG | **out-of-scope / research-heavy** | medical computer-aided diagnosis; LLM/CLIP-driven |

### 57. Clarify Before You Draw — Proactive Agents for Robust Text-to-CAD Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| Geometric ambiguity taxonomy + rule classifier + under-specification scorer | **implemented** | `clarify_ambiguity.py` |
| Two-round proactive-clarification MDP state machine | **implemented** | `clarify_dialogue.py` |
| Deterministic ambiguity-synthesis perturbation generator + curation rules | **implemented** | `clarify_perturb.py` |
| Efficiency-F1 + Resolution clarifier metrics | **implemented** | `clarify_metrics.py` |
| Data-leakage auditor | **implemented** | `clarify_leakage.py` |
| Scaling-operation failure detector/rewriter | **implemented** | `clarify_scaling.py` |
| ProCAD SFT, VLM description + LLM-judge + user simulator | **research-heavy/external** | learned models |

### 58. CME-CAD — Heterogeneous Collaborative Multi-Expert Reinforcement Learning for CAD Code Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| Gated multi-objective reward with dedicated work-plane term (IoU + origin/axis deviation, multiplicative gating) | **implemented** | `dataengine/cmecad_reward.py` |
| Expert-internal group-relative advantage with non-negative truncation | **implemented** | `dataengine/cmecad_advantage.py` |
| Multi-expert collaborative learning (best/worst expert, directional-KL credit, routing) | **implemented** | `dataengine/cmecad_collab.py` |
| Hard-negative sample buffering (partitioned rotating split + probabilistic admission) | **implemented** | `dataengine/cmecad_hardneg_buffer.py` |
| MEFT multi-expert CoT training, CADExpert dataset construction | **research-heavy/external** | learned VLM experts / annotation |

### 59. CMT — A Cascade MAR with Topology Predictor for Multimodal Conditional CAD Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| Continuous B-rep tokenization (surface/edge tokens, ascending order, uniform quantization) | **implemented** | `reconstruction/cmt_tokenization.py` |
| Cascade edge-then-surface stage schema + MAR cosine masked-reveal schedule | **implemented** | `reconstruction/cmt_cascade_schedule.py` |
| Geometry-inferred topology predictor (edge-surface incidence, tau=0.5) | **implemented** | `reconstruction/cmt_topology_predictor.py` |
| Topological-validity checker (unbounded regions, non-manifold edges, degenerate edges) | **implemented** | `reconstruction/cmt_topology_validity.py` |
| mmABC curation (quantized-hash dedup + union-find multi-body decomposition) | **implemented** | `reconstruction/cmt_mmabc_dedup.py` |
| Neural MAR sampler + VAEs + multimodal condition encoder + learned cross-attention head | **research-heavy/external** | trained models |

### 60. Comparing Fabrication Workflows in CAD to Support Design Reasoning

| Build idea | Status | Repository comparison |
|---|---|---|
| Fabrication-workflow taxonomy + machine registry + material-stock presets + workflow selector | **implemented** | `fabworkflow_taxonomy.py` |
| Per-workflow feasibility dispatch (machine-fit, FDM time, stock snap, wire-form, mold draft) | **implemented** | `fabworkflow_feasibility.py` |
| Comparison table + intent-based workflow ranker + reflection checklist + exploration trace | **implemented** | `fabworkflow_compare.py` |
| STL voxel/Poisson recovery; React/Three.js UI + N=12 user study | **research-heavy/external / out-of-scope** | mesh kernels / HCI study |

## Batch-12 implementation result

All deterministic and in-scope findings from papers 56-60 are implemented
(ChatCAD+ is medical CAD -- only two domain-agnostic primitives kept;
Fabrication Workflows is an HCI study -- only its workflow artifacts
kept). Suite: 2391 tests, all passing.

### 61. Consistent Flow Distillation for Text-to-3D Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| Equal-area sphere<->square parametrization for uniform viewpoint sampling | **implemented** | `cfd_sphere_square_map.py` |
| Variance-preserving integral-noise transport + closed-form OU noise schedule | **implemented** | `cfd_integral_noise.py` |
| Scaled gradient-variance consistency metric (Adam EMA moments) | **implemented** | `cfd_gradient_variance.py` |
| Clean-flow ODE (Euler) + EDM 2nd-order Heun sampler | **implemented** | `cfd_clean_flow_ode.py` |
| Score-distillation into NeRF/mesh; multi-view rasterized noise; 3D-FID/CLIP eval | **research-heavy/external** | learned diffusion + differentiable renderer |

### 62. Context-Aware Mapping of 2D Drawing Annotations to 3D CAD Features

| Build idea | Status | Repository comparison |
|---|---|---|
| 2D drawing-annotation schema + OCR-callout parser (diameters/threads/radii/counts/GD&T/tolerances) | **implemented** | `annomap_parser.py` |
| Annotation<->feature correspondence scoring (type gate + dimensional agreement + heuristics + greedy assignment) | **implemented** | `annomap_scoring.py` |
| GD&T feature-control-frame representation + ASME/ISO validity checker | **implemented** | `annomap_gdt.py` |
| Manufacturing-spec builder with provenance + precision/recall/F1 link evaluation | **implemented** | `annomap_spec.py` |
| VLM semantic enrichment + constrained-LLM escalation | **research-heavy/external** | learned VLM/LLM |

### 63. ContrastCAD — Contrastive Learning-Based Representation Learning for Computer-Aided Design Models

| Build idea | Status | Repository comparison |
|---|---|---|
| RRE (Random Replace and Extrude) contrastive augmentation | **implemented** | `datagen/contrastcad_rre.py` |
| Shape-preserving permutation augmentations | **implemented** | `datagen/contrastcad_permute.py` |
| Contrastive maths (cosine sim, SimCSE dropout views, NT-Xent/InfoNCE) | **implemented** | `bench/contrastcad_contrastive.py` |
| Latent representation-quality metrics (ED, silhouette, SSE, K-means) | **implemented** | `bench/contrastcad_latent_metrics.py` |
| Position-aligned tolerant reconstruction accuracy | **implemented** | `bench/contrastcad_recon_accuracy.py` |
| Learned Transformer autoencoder + latent-GAN generation | **research-heavy/external** | trained models |

### 64. CraftsMan — High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner

| Build idea | Status | Repository comparison |
|---|---|---|
| Relative Laplacian smoothing (resists thin-feature collapse) + umbrella Laplacian/Taubin/displacement operators | **implemented** | `geometry/craftsman_relative_laplacian.py` |
| Native-3D latent-set diffusion + learned normal-based geometry refiner + MV conditioning | **research-heavy/external** | trained diffusion/ControlNet |

### 65. CReFT-CAD — Boosting Orthographic Projection Reasoning for CAD via Reinforcement Fine-Tuning

| Build idea | Status | Repository comparison |
|---|---|---|
| Forward orthographic projection of a box solid to front/top/side silhouettes | **implemented** | `creft_projection.py` |
| Third-angle inter-view consistency + view-matching + intra-view validity + paired-dimension evaluator | **implemented** | `creft_view_consistency.py` |
| Curriculum reward functions (dichotomous / set-based / difficulty-aware) + attribute difficulty classification | **implemented** | `dataengine/creft_rewards.py` |
| TriView2CAD ortho-reasoning scorer + composite-parameter formula evaluator | **implemented** | `bench/creft_ortho_reasoning.py` |
| Curriculum data-engine (seeded negative sampling + CoT step builder) | **implemented** | `dataengine/creft_data_engine.py` |
| GRPO/SFT fine-tuning of the ViT+Qwen VLM + TriView2CAD raster dataset | **research-heavy/external** | trained VLM / GPU |

## Batch-13 implementation result

All deterministic and in-scope findings from papers 61-65 are implemented
(text-to-3D and mesh-generation papers are mostly learned -- only their
deterministic numeric/geometry primitives kept). Suite: 2680 tests, all
passing.

### 66. DAVINCI — A Single-Stage Architecture for Constrained CAD Sketch Inference

| Build idea | Status | Repository comparison |
|---|---|---|
| 8-token primitive parametrization codec (type + params + construction flag) | **implemented** | `ingest/davinci_primitive_tokens.py` |
| Valid-subreference combination set + constraint-graph consistency filter | **implemented** | `ingest/davinci_subreference_validity.py` |
| Constraint-preserving transformations (permutation + index remapping + proof) | **implemented** | `ingest/davinci_cpt.py` |
| Set-based eval suite (Hungarian matcher, token accuracy, primitive/constraint F1, Chamfer) | **implemented** | `bench/davinci_inference_metrics.py` |
| Learned transformer + FreeCAD-solver CPT generation | **research-heavy/external** | trained model / proprietary solver |

### 67. DeepCAD — A Deep Generative Network for Computer-Aided Design Models

| Build idea | Status | Repository comparison |
|---|---|---|
| Exact DeepCAD command spec (6 types, 16-dim param vector, 256-level quant, invertible vector conversion) | **implemented** | `reconstruction/deepcad_command_spec.py` |
| Sketch-plane orientation + extrusion 3D decode ((theta,phi,gamma) ZYZ, Euler inverse, local<->world) | **implemented** | `reconstruction/deepcad_sketch_plane.py` |
| Loop/profile assembly (SOL-split loops, implicit chaining, canonical CCW ordering) | **implemented** | `reconstruction/deepcad_profile_assembly.py` |
| Command/param accuracy, Chamfer, COV/MMD/JSD, invalid-ratio, F1, quantization, RRE augmentation | **already in repo** | `bench/contrastcad_recon_accuracy.py`, `bench/generative_brep_metrics.py`, etc. |
| Transformer autoencoder + latent-GAN + PointNet++ | **research-heavy/external** | trained models |

### 68. Design-Specification Tiling for ICL-based CAD Code Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| Multi-granular n-gram component tiling + weighted tiling ratio | **implemented** | `context/spectiling_components.py` |
| Greedy submodular exemplar selection ((1-1/e) guarantee) | **implemented** | `rag/spectiling_greedy.py` |
| Spec decomposition into dependency-ordered tiles + coverage metric | **implemented** | `spec/spectiling_decompose.py`, `spec/spectiling_coverage.py` |
| Per-tile ICL prompt assembly | **implemented** | `context/spectiling_prompt.py` |
| Tile-composition/merge of generated fragments | **implemented** | `generation/spectiling_compose.py` |
| Complexity-based corpus stratification | **implemented** | `dataengine/spectiling_complexity.py` |
| LLM CAD-code inference + learned-embedding baselines | **research-heavy/external** | frozen LLM / neural embeddings |

### 69. Diffusion-CAD — Controllable Diffusion Model for Generating CAD Models

| Build idea | Status | Repository comparison |
|---|---|---|
| Sqrt DDPM schedule + forward diffusion + CFG conditional seeding + 256-level quantization | **implemented** | `numeric/diffusioncad_sqrt_schedule.py` |
| Table-II structure-constraint equations as integer-coordinate projection/repair | **implemented** | `geometry/diffusioncad_structure_constraints.py` |
| Unique/novel/sequence-validity generation metrics | **implemented** | `bench/diffusioncad_generation_metrics.py` |
| BERT denoiser + GPT-2 classifier guidance + point-cloud metrics + training | **research-heavy/external** | trained models |

### 70. Don't Mesh with Me — Generating CSG Instead of Meshes by Fine-Tuning a Code-Generation LLM

| Build idea | Status | Repository comparison |
|---|---|---|
| Surface-based CSG (half-spaces + cylinders, cells, union) + membership/occupancy/IoU/validity/overlap | **implemented** | `geometry/dontmesh_halfspace_csg.py` |
| Cell adjacency graph + plausible connected build-ordering enumeration | **implemented** | `reconstruction/dontmesh_cell_graph.py` |
| OpenMC-style script (de)serializer + input-output splitting + augmentation + plausibility metrics | **implemented** | `programs/dontmesh_csg_script.py` |
| DeepSeek-Coder fine-tuning + GPT-4o annotation + GEOUNED decomposition | **research-heavy/external** | trained LLM / proprietary tooling |

## Batch-14 implementation result

All deterministic and in-scope findings from papers 66-70 are implemented
(DeepCAD built modestly since earlier batches covered its metrics; the
diffusion/generation papers kept only their deterministic scaffolding).
Suite: 2929 tests, all passing. All modules placed in packages, not root.

### 71. Draw It Like Euclid — Teaching Transformer Models to Generate CAD Profiles Using Ruler and Compass Construction Steps

| Build idea | Status | Repository comparison |
|---|---|---|
| Ruler-and-compass construction engine (entities, primitives, line/circle intersections, ~15 atomic construction steps) | **implemented** | `geometry/euclid_construction.py` |
| Construction-step DSL + quantization + tokenizer/detokenizer | **implemented** | `geometry/euclid_dsl.py` |
| Construction-sequence -> CAD profile replay compiler | **implemented** | `geometry/euclid_compiler.py` |
| Constructibility/validity checker + profile-validity + construction-accuracy metrics | **implemented** | `geometry/euclid_validity.py` |
| Learned autoregressive transformer + RL fine-tuning + OCCT/ABC dataset extraction | **research-heavy/external** | trained model / kernel / dataset |

### 72. DreamCAD — Scaling Multi-modal CAD Generation using Differentiable Parametric Surfaces

| Build idea | Status | Repository comparison |
|---|---|---|
| Rational Bezier curve/surface evaluation (Bernstein + derivatives, de Casteljau, weighted control grids, normals) | **implemented** | `geometry/dreamcad_rational_bezier.py` |
| Differentiable tessellation (uv sampling -> quads -> triangles, multi-patch welding, C0 continuity) | **implemented** | `geometry/dreamcad_tessellation.py` |
| Analytic CAD primitives (plane/cylinder/cone/sphere/torus) | **implemented** | `geometry/dreamcad_primitives.py` |
| Surface point-sampling + Chamfer/Hausdorff/consistency metrics | **implemented** | `geometry/dreamcad_metrics.py` |
| Multi-modal condition-encoding schema | **implemented** | `reconstruction/dreamcad_condition_schema.py` |
| Learned VAE/SLAT + flow-matching generation + DINOv2/PointNet++ encoders | **research-heavy/external** | trained models |

### 73. E3D-Bench — A Benchmark for End-to-End 3D Geometric Foundation Models

| Build idea | Status | Repository comparison |
|---|---|---|
| Umeyama Sim(3)/SE(3) alignment + stdlib linear-algebra core (Jacobi eigensolver, 3x3 SVD) | **implemented** | `geometry/e3dbench_umeyama.py` |
| Depth metrics (AbsRel, delta-threshold inlier ratios, median scaling) | **implemented** | `bench/e3dbench_depth_metrics.py` |
| Camera-pose metrics (geodesic rotation error, ATE, RPE-trans/rot) | **implemented** | `bench/e3dbench_pose_metrics.py` |
| Point-map metrics (accuracy, completeness, Chamfer-L1, threshold F-score, normal consistency) | **implemented** | `bench/e3dbench_pointmap_metrics.py` |
| Per-scene-normalized cross-scene leaderboard harness | **implemented** | `bench/e3dbench_harness.py` |
| The 16 GFM models + latency/GPU benchmarking | **out-of-scope / external** | trained foundation models / hardware |

### 74. EnzymeCAGE — A Geometric Foundation Model for Enzyme Retrieval with Evolutionary Insights

| Build idea | Status | Repository comparison |
|---|---|---|
| Ranked-retrieval quality metrics (DCG/NDCG, MRR, success-rate@k, enrichment factor) | **implemented** | `bench/ranked_retrieval_metrics.py` |
| Enzyme/pocket GNN encoders, reaction fingerprints, biochemistry data pipeline | **out-of-scope** | molecular biology; no CAD transfer |

### 75. Error Notebook-Guided, Training-Free Part Retrieval in 3D CAD Assemblies via Vision-Language Models

| Build idea | Status | Repository comparison |
|---|---|---|
| Error-notebook corrective memory (entry schema + append-only store + leak-safe recall + few-shot rendering + JSON persistence) | **implemented** | `memory/errornotebook_store.py` |
| Grammar-constraint verifier (strict/relaxed final-answer extraction, corrected-trajectory assembly) | **implemented** | `reliability/errornotebook_gc.py` |
| Training-free re-ranking policy consulting the notebook | **implemented** | `rag/partretr_rerank.py` |
| Part-retrieval eval harness (exact-set accuracy, recall/precision/F1, recall@k, MRR, difficulty buckets) | **implemented** | `bench/partretr_eval.py` |
| VLM part-description/retrieval/correction inference + Fusion 360 dataset | **research-heavy/external** | learned VLM / proprietary data |

## Batch-15 implementation result

All deterministic and in-scope findings from papers 71-75 are implemented
(EnzymeCAGE is biochemistry -- only its ranked-retrieval metrics kept;
the text-to-3D/mesh/foundation-model papers kept only their deterministic
geometry/metric primitives). Suite: 3124 tests, all passing.

### 76. Evaluating Deep Clustering Algorithms on Non-Categorical 3D CAD Models

| Build idea | Status | Repository comparison |
|---|---|---|
| Partition-comparison metrics (NMI, Rand/Adjusted Rand, clustering accuracy via Hungarian, purity) | **implemented** | `bench/deepclustering_partition_metrics.py` |
| Internal validity indices (Davies-Bouldin, Calinski-Harabasz, Dunn) | **implemented** | `bench/deepclustering_internal_indices.py` |
| Pairwise-edge similarity protocol (edge accuracy, balanced accuracy) | **implemented** | `bench/deepclustering_edge_protocol.py` |
| Ensemble evaluation + ranking consistency (Kendall tau) | **implemented** | `bench/deepclustering_ensemble.py` |
| Classic clustering algorithms (k-means++, agglomerative, spectral-lite) | **implemented** | `bench/deepclustering_algorithms.py` |
| Oversegmentation init/annotation protocol | **implemented** | `bench/deepclustering_init_protocol.py` |
| CAD-model distance protocol (Chamfer + voxel-Jaccard, thin-object rule) | **implemented** | `reconstruction/cadcluster_model_distances.py` |
| Learned deep-clustering encoders + human annotation | **research-heavy/external** | trained models / data labor |

### 77. EvoCAD — Evolutionary CAD Code Generation with Vision Language Models

| Build idea | Status | Repository comparison |
|---|---|---|
| Generational evolutionary loop over CAD programs (rank->probability selection, elitism) | **implemented** | `exploration/evocad_evolution.py` |
| Deterministic CAD-program crossover/mutation operators | **implemented** | `exploration/evocad_variation.py` |
| Euler-characteristic topology metrics (T_err/T_corr/genus, dataset aggregation) | **implemented** | `bench/evocad_topology_metrics.py` |
| VLM description + reasoning-model ranking | **research-heavy/external** | learned VLM |

### 78. Exploring the Usability of AI-Generated 3D Models in CAD Workflows and the Metaverse

| Build idea | Status | Repository comparison |
|---|---|---|
| Loop-size CV variability bands + quad-topology fraction + VR/AR polygon budgets + mesh-defect readiness + combined usability verdict | **implemented** | `quality/ai_model_usability_standard.py` |
| UV-layout / animation-topology criteria (expert judgment); NeRF/DreamFusion generation | **out-of-scope / research-heavy** | qualitative / trained models |

### 79. Facilitating the Parametric Definition of Geometric Properties in Programming-Based CAD

| Build idea | Status | Repository comparison |
|---|---|---|
| Linear-form algebra + arithmetic-expression parser/AST + affine reducer | **implemented** | `programs/paramgeom_linform.py` |
| C1..C5 expression classifier + cross-tabulation | **implemented** | `programs/paramgeom_classify.py` |
| Parametric handle grids for primitives + handle-role classifier | **implemented** | `programs/paramgeom_handles.py` |
| Position/delta-vector features via CSG-tree walking | **implemented** | `programs/paramgeom_position.py` |
| Interactive UI + user study | **research-heavy/external** | UI / human subjects |

### 80. Fine-Tuning 3D Foundation Models for Geometric Object Retrieval

| Build idea | Status | Repository comparison |
|---|---|---|
| Geometric-object retrieval eval protocol (NN accuracy/F1, NDCG@N, per-category/macro/micro mAP) | **implemented** | `bench/geomretr_eval.py` |
| Rotation/translation-invariant descriptors (Osada D2, spherical-harmonic-lite, PCA bounding-volume) | **implemented** | `reconstruction/geomretr_descriptors.py` |
| Embedding post-processing (L2 norm, PCA whitening, query expansion) | **implemented** | `bench/geomretr_embedding.py` |
| VICReg + multi-modal contrastive losses | **implemented** | `bench/geomretr_losses.py` |
| VICReg positive-pair augmentation pipeline | **implemented** | `reconstruction/geomretr_augment.py` |
| ULIP-2 foundation-model encoder + fine-tuning | **research-heavy/external** | trained 3D encoder |

## Batch-16 implementation result

All deterministic and in-scope findings from papers 76-80 are implemented.
Also this session: four agent-protocol integrations (MCP server, A2A
server + a2a/ spec conformance, Zed ACP agent). All modules in packages,
not root.

### 81. FlatCAD — Fast Curvature Regularization of Neural SDFs for CAD Models

| Build idea | Status | Repository comparison |
|---|---|---|
| Weingarten shape operator + closed-form implicit-surface Gaussian/mean/principal curvature (validated vs analytic SDFs) | **implemented** | `geometry/flatcad_weingarten.py` |
| Off-diagonal-Weingarten curvature-gap term + ODW L1/L2 loss + closed-form expectations + curvature-regime classification | **implemented** | `geometry/flatcad_weingarten.py` |
| Finite-difference SDF gradient/Hessian + FlatCAD symmetric mixed stencil (O(h^2)) | **implemented** | `numeric/flatcad_sdf_derivatives.py` |
| SIREN network + training pipeline; autodiff HVP route | **research-heavy/external** | learned neural SDF |

### 82. FlexCAD — Unified and Versatile Controllable CAD Generation with Fine-Tuned Large Language Models

| Build idea | Status | Repository comparison |
|---|---|---|
| Hierarchy-aware CAD<->structured-text serializer + round-trip parser + field masker/infill | **implemented** | `reconstruction/flexcad_text.py` |
| 7-level masking scheme (typed curve/loop/face/sketch/extrusion masks) + uniform per-epoch sampling | **implemented** | `dataengine/flexcad_masking.py` |
| Masked-infill training-pair constructor + unconditional template | **implemented** | `dataengine/flexcad_infill_pairs.py` |
| Prediction-validity + controllability metrics | **implemented** | `bench/flexcad_controllability.py` |
| Llama-3 LoRA fine-tuning; COV/MMD/JSD (learned-embedding); Realism study | **research-heavy/external** | learned models / crowd |

### 83. FlexCAD (near-duplicate extraction of paper 82)

| Build idea | Status | Repository comparison |
|---|---|---|
| Serialization / masking / infill / controllability metrics | **covered by paper 82** | `flexcad_*` modules |
| Circle-representation variants (center-radius / diameter / four-points) with round-trip codecs; PV-constrained sampling-config selector + Pareto frontier | **implemented** | `generation/flexcad2_appendix.py` |

### 84. From 2D CAD Drawings to 3D Parametric Models — A Vision-Language Approach

| Build idea | Status | Repository comparison |
|---|---|---|
| Text-based parametric shape-program IR (Python/YAML codecs) + first-octant pose normalization | **implemented** | `reconstruction/cad2program_shape_program.py` |
| 3D box IoU + Hungarian primitive matching + reconstruction/retrieval/param-estimation metrics | **implemented** | `reconstruction/cad2program_metrics.py` |
| Command-template codec + position/size/rotation quantization + quantization-error metric | **implemented** | `reconstruction/cad2program_command_template.py` |
| Orthographic-view-to-3D lifting (constructs prismatic solids) + 2D view parser | **implemented** | `drawings/cad2program_view_lifting.py` |
| Three-view fixed-canvas layout | **implemented** | `drawings/cad2program_canvas_layout.py` |
| ViT/InternVL vision-language model + CLIP model-id token | **research-heavy/external** | trained VLM/CLIP |

### 85. From Concept to Manufacturing — Evaluating Vision-Language Models for Engineering Design

| Build idea | Status | Repository comparison |
|---|---|---|
| Four-area task taxonomy + cross-model scorecard | **implemented** | `bench/engdesign_taxonomy.py` |
| Design-similarity triplet metrics (self-consistency, transitive-violation) | **implemented** | `bench/engdesign_similarity_triplets.py` |
| Multiple-choice description-matching scorer | **implemented** | `bench/engdesign_description_match.py` |
| CAD-generation rubric (description/dimension/feature/non-improvement) | **implemented** | `bench/engdesign_cad_rubric.py` |
| Topology-optimization metrics (volume-fraction / floating-material error) | **implemented** | `bench/engdesign_topopt_metrics.py` |
| DFM scorers (manufacturability + machining-feature recognition) | **implemented** | `bench/engdesign_dfm_scoring.py` |
| Defect-inspection confusion metrics | **implemented** | `bench/engdesign_defect_confusion.py` |
| Textbook + spatial QA scorers | **implemented** | `bench/engdesign_qa_scoring.py` |
| GPT-4V/LLaVA inference; GNMDS embedding; CAT human ratings | **research-heavy/external** | VLMs / human eval |

## Batch-17 implementation result

All deterministic and in-scope findings from papers 81-85 are implemented
(paper 83 is a near-duplicate of 82 -- only its appendix-only circle-repr
and PV-sampling ideas were new). Per the no-README-during-campaign policy,
the suite count is tracked in audit/text_to_cad_progress.json.

### 86. From Idea to CAD — A Language Model-Driven Multi-Agent System for Collaborative Design

| Build idea | Status | Repository comparison |
|---|---|---|
| Shared design-state blackboard + design-feedback composition rule | **implemented** | `agents/idea2cad_blackboard.py` |
| V-model role set + handoff DAG (forward + feedback back-edges) + ast static-check gate | **implemented** | `agents/idea2cad_roles.py` |
| Artifact parsers (SUMMARY addendum, seven-view enum, bounded QA feedback, ambiguity detector) | **implemented** | `agents/idea2cad_artifacts.py` |
| Four nested empty-feedback loops (validation/verification/design/codegen) | **implemented** | `agents/idea2cad_workflow.py` |
| VLM codegen + doc scraping + rendering | **research-heavy/external** | learned VLM / web / kernel |

### 87. From Intent to Execution — Multimodal Chain-of-Thought Reinforcement Learning for Precise CAD Code Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| Gated CAD-RL total reward (exec gate x geometric/eval combo, misalignment deductions) | **implemented** | `dataengine/intent2exec_reward.py` |
| Trust Region Stretch (asymmetric relaxed-bound PPO surrogate) | **implemented** | `dataengine/intent2exec_trs.py` |
| Precision token loss (up-weight numeric/geometry tokens) | **implemented** | `dataengine/intent2exec_precision_token_loss.py` |
| Overlong filtering (exclude truncated sequences from RL loss) | **implemented** | `dataengine/intent2exec_overlong_filter.py` |
| Two-part <Think>/code CoT trace schema + format checker | **implemented** | `dataengine/intent2exec_cot_schema.py` |
| VLM cold-start SFT + RL training; ExeCAD dataset | **research-heavy/external** | trained VLM / proprietary data |

### 88. Future Prospects of Computer-Aided Design (CAD)

| Build idea | Status | Repository comparison |
|---|---|---|
| AI-in-CAD vision, MBR desiderata, VR/AR/MR narrative, 3D-printing survey | **out-of-scope** | survey/vision essay -- no deterministic buildable artifact (nothing built) |

### 89. GaussianCAD — Robust Self-Supervised CAD Reconstruction from Three Orthographic Views Using 3D Gaussian Splatting

| Build idea | Status | Repository comparison |
|---|---|---|
| Single 3D Gaussian forward math (covariance from scale+quaternion, projection/splatting, density, footprint) | **implemented** | `geometry/gaussiancad_splatting.py` |
| ZYX-Euler camera + pinhole intrinsics/extrinsics + three-orthographic-view pose localization + Blender<->COLMAP | **implemented** | `geometry/gaussiancad_camera.py` |
| Visual-hull space-carving Gaussian init from silhouette masks | **implemented** | `reconstruction/gaussiancad_visual_hull.py` |
| Exact-Hungarian Earth Mover's Distance metric | **implemented** | `reconstruction/gaussiancad_emd.py` |
| Sketch image-processing pipeline; 3DGS optimization + diffusion | **research-heavy/external** | image ops / learned optimization |

### 90. GenCAD — Image-Conditioned CAD Generation with Transformer-Based Contrastive Representation and Diffusion Priors

| Build idea | Status | Repository comparison |
|---|---|---|
| Gaussian FID latent-alignment metric (stdlib Jacobi matrix-sqrt) | **implemented** | `bench/gencad_fid.py` |
| Batched R_B image-to-CAD retrieval-accuracy protocol | **implemented** | `bench/gencad_retrieval.py` |
| Command spec / recon accuracy / COV-MMD-JSD / InfoNCE / cosine retrieval / DDPM schedule | **already in repo** | earlier-batch modules |
| Learned transformer/CNN/diffusion encoders + training | **research-heavy/external** | trained models |

## Batch-18 implementation result

All deterministic and in-scope findings from papers 86-90 are implemented
(paper 88 is a survey/vision essay -- correctly no buildable content;
GenCAD built modestly since most machinery already existed). Per the
no-README-during-campaign policy, the suite count is tracked in
audit/text_to_cad_progress.json.

### 91. GenCAD-3D — CAD Program Generation using Multimodal Latent Space Alignment and Synthetic Dataset Balancing

| Build idea | Status | Repository comparison |
|---|---|---|
| SynthBal complexity balancing + noise/replace-sketch augmentation + reduction-balancing + imbalance metrics | **implemented** | `datagen/gencad3d_synthbal.py` |
| Sequence-length normalized metric + relative-improvement conventions | **implemented** | `bench/gencad3d_seqlen_norm.py` |
| Deterministic linear cross-modal latent-alignment surrogate + alignment-quality diagnostics | **implemented** | `bench/gencad3d_latent_alignment.py` |
| Learned encoders + latent diffusion training | **research-heavy/external** | trained models |

### 92. GenCAD-Self-Repairing — Feasibility Enhancement for 3D CAD Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| Infeasibility taxonomy over generated command sequences (pre-kernel) | **implemented** | `reliability/gencadrepair_taxonomy.py` |
| Sequence-level self-repair procedure (idempotent, guaranteed feasible) | **implemented** | `reliability/gencadrepair_sequence.py` |
| Feasibility-rate + repair-success metrics | **implemented** | `reliability/gencadrepair_metrics.py` |
| Detect->repair->re-check loop driver with stall guard | **implemented** | `reliability/gencadrepair_loop.py` |
| Learned latent classifier/regressor + guided diffusion | **research-heavy/external** | trained nets |

### 93. Generating CAD Code with Vision-Language Models for 3D Designs

| Build idea | Status | Repository comparison |
|---|---|---|
| CADPrompt geometric eval protocol (normalization, point-cloud dist, Hausdorff, IoGT, compile penalty) | **implemented** | `bench/vlmcadcode_metrics.py` |
| Geometric-solver feedback baseline (13 categories, paired diff -> NL) | **implemented** | `bench/vlmcadcode_geomsolver.py` |
| CADPrompt data stratification | **implemented** | `bench/vlmcadcode_stratify.py` |
| CADCodeVerify refinement loop | **implemented** | `generation/vlmcadcode_verify_loop.py` |
| Feedback/error-type taxonomy classifiers | **implemented** | `generation/vlmcadcode_feedback_taxonomy.py` |
| VLM question/answer/rewrite + ICP/render | **research-heavy/external** | trained VLMs / Open3D |

### 94. Generating Physically Stable and Buildable Brick Structures from Text

| Build idea | Status | Repository comparison |
|---|---|---|
| Brick representation + library + text format + voxel-overlap collision | **implemented** | `geometry/brick_structure.py` |
| Stud-into-tube connectivity graph (components, grounded) | **implemented** | `geometry/brick_connectivity.py` |
| Physical stability analysis (force/torque equilibrium, score) via two-phase simplex LP | **implemented** | `verifiers/brick_stability.py` |
| Buildability / assembly-order + per-intermediate stability | **implemented** | `verifiers/brick_buildability.py` |
| Stability-aware gate + physics-aware rollback | **implemented** | `verifiers/brick_validity.py` |
| Learned brick LLM + training + mesh-to-brick + robotic assembly | **research-heavy/external** | trained models / hardware |

### 95. Generating Physically Stable and Buildable LEGO Designs from Text

| Build idea | Status | Repository comparison |
|---|---|---|
| Standard 8-brick LEGO library + LEGOGPT text codec + format FSA | **implemented** | `fabrication/lego_brick_library.py` |
| Split-and-remerge legolization (voxel -> brick layout) | **implemented** | `fabrication/lego_legolization.py` |
| Uniform brick-color assignment (nearest LEGO-palette snap) | **implemented** | `fabrication/lego_coloring.py` |
| Shared brick mechanics (collision/connectivity/stability/rollback) | **covered by paper 94** | `brick_*` modules |
| LLaMA fine-tuning + Gurobi + texturing | **research-heavy/external** | trained models / solver |

## Batch-19 implementation result

All deterministic and in-scope findings from papers 91-95 are implemented
(95 LEGOGPT reuses paper 94's brick mechanics -- only its LEGO-specific
library/legolization/coloring were new). Recovered from a mid-batch
session-limit interruption by removing untested partials and re-running all
5 papers fresh. Per the no-README-during-campaign policy, the suite count
is tracked in audit/text_to_cad_progress.json.

### 96. Generative AI and CAD Automation for Diverse and Novel Mechanical Component Designs Under Data Constraints

| Build idea | Status | Repository comparison |
|---|---|---|
| Wheel-rim ISO spec-code parser + derived geometry | **implemented** | `spec/datacon_rim_spec.py` |
| 2D-spoke geometric-feasibility validation (shoelace/centroid/rotational-symmetry) | **implemented** | `verifiers/datacon_rim_validation.py` |
| Diversity/novelty/coverage metrics over component feature vectors | **implemented** | `bench/datacon_diversity.py` |
| Few-shot contour augmentation enforcing rotational symmetry | **implemented** | `datagen/datacon_spoke_augment.py` |
| Scale-invariant low-data dedup + farthest-point curation | **implemented** | `dataengine/datacon_lowdata_dedup.py` |
| Latin-hypercube design-space sampler | **implemented** | `exploration/datacon_designspace_sampler.py` |
| Diffusion/LoRA generator + image processing | **research-heavy/external** | trained models |

### 97. Generative AI for CAD Automation - Leveraging Large Language Models for 3D Modelling

| Build idea | Status | Repository comparison |
|---|---|---|
| Error-driven prompt-evolution loop (stateless rebuild + accumulating constraints) | **implemented** | `generation/llm3dmodel_prompt_evolution.py` |
| Complexity-scaling taxonomy scorer | **implemented** | `bench/llm3dmodel_complexity_scale.py` |
| FreeCAD execution-error taxonomy | **implemented** | `bench/llm3dmodel_freecad_errors.py` |
| Convergence-outcome run-metrics protocol | **implemented** | `bench/llm3dmodel_run_metrics.py` |
| LLM script generation + FreeCAD host | **research-heavy/external** | learned LLM / host |

### 98. Generative AI meets 3D — A Survey on Text-to-3D in AIGC Era

| Build idea | Status | Repository comparison |
|---|---|---|
| Method-taxonomy tables, 3D-representation comparison, cited NeRF/DDPM/SDS equations, future-agenda prose | **out-of-scope** | pure literature survey -- no self-contained algorithm (nothing built) |

### 99. GeoCAD - Local Geometry-Controllable CAD Generation with Large Language Models

| Build idea | Status | Repository comparison |
|---|---|---|
| Closed-form shape captioning (triangle/quad/arc taxonomy + key dims) | **implemented** | `geometry/geocad_vertex_caption.py` |
| Simple/complex part routing from side types | **implemented** | `reconstruction/geocad_part_classifier.py` |
| Closed + non-self-intersecting local-loop validity | **implemented** | `geometry/geocad_local_validity.py` |
| Caption-invariant geometric augmentation | **implemented** | `dataengine/geocad_augment.py` |
| Ver-score text-to-CAD consistency metric | **implemented** | `reconstruction/geocad_verscore.py` |
| Geometry-constrained local-edit prompts (over FlexCAD masking) | **implemented** | `dataengine/geocad_prompt.py` |
| Symmetry-axis reflection editing | **implemented** | `geometry/geocad_sketch_symmetry.py` |
| VLLM captioner/score + LoRA training | **research-heavy/external** | learned VLLM |

### 100. GeoFusion-CAD - Structure-Aware Diffusion with Geometric State Space for Parametric 3D Design

| Build idea | Status | Repository comparison |
|---|---|---|
| Nested sketch-extrusion tree + reversible end-token serialization + 8-bit quantization | **implemented** | `reconstruction/geofusion_hierarchy.py` |
| Geometric state-space selective-scan recurrence + curvature/PE/conv/FiLM fusion | **implemented** | `numeric/geofusion_state_space.py` |
| Structure-consistency metrics (closure validity, structure-F1) | **implemented** | `bench/geofusion_structure_consistency.py` |
| Learned G-Mamba denoiser + diffusion training | **research-heavy/external** | trained SSM/diffusion |

## Batch-20 implementation result

All deterministic and in-scope findings from papers 96-100 are implemented
(98 is a text-to-3D survey -- correctly no buildable content). This closes
the first 100 papers of the 186-paper corpus. Per the no-README-during-
campaign policy, the suite count is tracked in audit/text_to_cad_progress.json.

### 101. Geometric Deep Learning for Computer-Aided Design - A Survey

| Build idea | Status | Repository comparison |
|---|---|---|
| CAD terminology/format glossary, GDL method taxonomy, cited B-rep GNN encoders (UV-Net, Hierarchical CADNet, SB-GCN), generative pipelines | **out-of-scope / research-heavy** | pure literature survey -- cited external neural methods, no self-contained algorithm (nothing built) |

### 102. Geometry of Spatial World Models

| Build idea | Status | Repository comparison |
|---|---|---|
| Probe LLM residual activations for a spatial "world model"; fit geometric structure to activations; toy spatial-laws | **out-of-scope / research-heavy** | LLM-interpretability research proposal, no formalized geometry; SE(3)/frame algebra already in geometry/ (nothing built) |

### 103. GIFT - Bootstrapping Image-to-CAD Program Synthesis via Geometric Feedback

| Build idea | Status | Repository comparison |
|---|---|---|
| Geometric-agreement signal (render + IoU bands -> corrective categories) | **implemented** | `dataengine/gift_geometric_feedback.py` |
| Soft-Rejection-Sampling + Failure-Driven-Augmentation dataset builders | **implemented** | `dataengine/gift_geometric_feedback.py` |
| CDF-based empirical threshold selection | **implemented** | `dataengine/gift_threshold_selection.py` |
| Bootstrapping self-training loop + amortization gap (pass@k - pass@1) + inverse-temp schedule | **implemented** | `dataengine/gift_bootstrap_loop.py` |
| Learned image-to-CAD synthesizer + VLM rendering | **research-heavy/external** | trained VLM |

### 104. GraphBrep - Learning B-Rep in Graph Structure for Efficient CAD Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| Surface-surface weighted adjacency graph (shared-edge counts) + matrix post-processing + edge-list recovery + validity | **implemented** | `reconstruction/graphbrep_surface_graph.py` |
| Permutation-invariant canonicalization (WL refinement, canonical labelling, isomorphism) | **implemented** | `reconstruction/graphbrep_canonical.py` |
| Efficiency/compactness metric (sequence vs graph length, attention-cost reduction) | **implemented** | `reconstruction/graphbrep_efficiency.py` |
| Learned graph-diffusion denoiser + VAEs | **research-heavy/external** | trained models |

### 105. Hierarchical Neural Coding for Controllable CAD Model Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| Solid-Profile-Loop tree with bounding-box abstraction + 6-bit one-hot token encoding | **implemented** | `reconstruction/hnc_spl_tree.py` |
| Nearest-codebook VQ code assignment (3 codebooks) + utilization/perplexity/compression metrics | **implemented** | `reconstruction/hnc_code_assignment.py` |
| Code-tree serialization + three-level control masking | **implemented** | `generation/hnc_code_control.py` |
| Controllability/consistency + edit-locality + diversity metrics | **implemented** | `bench/hnc_code_consistency.py` |
| VQ-VAE + cascaded auto-regressive transformers | **research-heavy/external** | trained models |

## Batch-21 implementation result

All deterministic and in-scope findings from papers 101-105 are implemented
(101 GDL survey + 102 spatial-world-models interpretability proposal are
correctly no-build). Per the no-README-during-campaign policy, the suite
count is tracked in audit/text_to_cad_progress.json.

### 106. HistCAD - Geometrically Constrained Parametric History-based CAD Dataset

| Build idea | Status | Repository comparison |
|---|---|---|
| Constraint-aware sequence schema (plane + Line/Circle/Arc + 10 constraints + rotated extrusion + booleans) + primitive dedup | **implemented** | `reconstruction/histcad_sequence.py` |
| Ten-type constraint model (per-primitive DOF, net-DOF status, conflict/redundancy) | **implemented** | `state/histcad_constraint_model.py` |
| Loop reconstruction + hierarchical outer/hole + 2D OBB + replay-validity | **implemented** | `reconstruction/histcad_replay.py` |
| Inter-part spatial relations (3D OBB + SAT contact + directional labels) | **implemented** | `dataengine/histcad_spatial_relations.py` |
| History-quality metrics | **implemented** | `bench/histcad_history_quality.py` |
| Parametric-edit-consistency (residuals + propagation) | **implemented** | `verifiers/histcad_edit_consistency.py` |
| LLM annotation + full proprietary dataset | **research-heavy/external** | learned model / licensed data |

### 107. How Can Large Language Models Help Humans in Design and Manufacturing

| Build idea | Status | Repository comparison |
|---|---|---|
| Global-coordinate sketch-and-extrude DSL interpreter | **implemented** | `geometry/llmdesign_sketch_extrude_dsl.py` |
| Tri-state box-contact auditor | **implemented** | `geometry/llmdesign_box_contact.py` |
| Analytic mass properties + assembly stability CoM | **implemented** | `quality/llmdesign_primitive_massprops.py` |
| Flat-pack panel decomposition + laser-bed fit/split | **implemented** | `fabrication/llmdesign_flatpack_panels.py` |
| First-order performance formulas (chair/cabinet/quadcopter) | **implemented** | `quality/llmdesign_first_order_performance.py` |
| Constrained design-space enumeration | **implemented** | `exploration/llmdesign_constrained_designspace.py` |
| Process selection / DfM advisor / inverse-design / FEA / CAM | **research-heavy/external** | LLM knowledge / non-stdlib libs |

### 108. Image2CADSeq - Computer-Aided Design Sequence and Knowledge Inference from Product Images

| Build idea | Status | Repository comparison |
|---|---|---|
| Sim-Gallery DSL + feature-matrix (start-point elision + arc-centre reconstruction) | **implemented** | `reconstruction/img2cadseq_gallery_dsl.py` |
| Multi-level H1/H2/H3 evaluation framework (paper's exact equations) | **implemented** | `bench/img2cadseq_eval.py` |
| Knowledge-inference schema (template shapes + parametric DesignRules) | **implemented** | `dataengine/img2cadseq_knowledge.py` |
| TEVAE/ResNet encoder + Fusion 360 rendering | **research-heavy/external** | trained models / host |

### 109. Img2CAD - Conditioned 3D CAD Model Generation from Single Image with Structured Visual Geometry

| Build idea | Status | Repository comparison |
|---|---|---|
| Structured-visual-geometry wireframe schema (junctions/segments, validity, normalization) | **implemented** | `reconstruction/img2cadsvg_representation.py` |
| HAT closed-form 4D attraction field (invertible encode/decode + dense field) | **implemented** | `geometry/img2cadsvg_hat_field.py` |
| Line-endpoint binding + JD-LOI alignment | **implemented** | `reconstruction/img2cadsvg_binding.py`, `img2cadsvg_loi_align.py` |
| Canny edge-extraction pipeline (Gaussian/Sobel/NMS/hysteresis) | **implemented** | `vision/img2cadsvg_edge_extract.py` |
| ANOVA F-test multi-view consistency | **implemented** | `bench/img2cadsvg_multiview_consistency.py` |
| Learned ViT/HAT regressor/decoder + datasets | **research-heavy/external** | trained models |

### 110. Img2CAD - Reverse Engineering 3D CAD Models from Images through VLM-Assisted Conditional Factorization

| Build idea | Status | Repository comparison |
|---|---|---|
| Two-stage conditional factorization (discrete structure / continuous attributes + lossless assembly) | **implemented** | `reconstruction/img2cadrev_factorization.py` |
| Sketch-extrude schema with join/cut as distinct command types + CCW validity | **implemented** | `reconstruction/img2cadrev_schema.py` |
| Sheaf-inspired shared attribute space (aggregate + shared-mean predictor) | **implemented** | `reconstruction/img2cadrev_shared_attributes.py` |
| Reconstruction metrics (Chamfer, symmetry-Chamfer, #SCC, round-trip fidelity) | **implemented** | `bench/img2cadrev_metrics.py` |
| VLM + TrAssembler/GMFlow | **research-heavy/external** | trained VLM/flow |

## Batch-22 implementation result

All deterministic and in-scope findings from papers 106-110 are implemented
(the two distinct Img2CAD papers kept separate via img2cadsvg_/img2cadrev_
prefixes). Per the no-README-during-campaign policy, the suite count is
tracked in audit/text_to_cad_progress.json.

### 111. Integrating Computer Vision and CAD for Precise Dimension Extraction and 3D Solid Model Regeneration for Enhanced Quality Assurance

| Build idea | Status | Repository comparison |
|---|---|---|
| Pixel-to-metric calibration + metric measurement from contours | **implemented** | `vision/cvcad_pixel_calibration.py` |
| Dimension-line/extension-line detection from segments | **implemented** | `drawings/dimext_dimension_lines.py` |
| 3D solid regeneration by extrusion (mesh + mass properties) | **implemented** | `reconstruction/cvcad_solid_regeneration.py` |
| Measured-vs-nominal tolerance QA + MAE/RMSE/MAPE accuracy | **implemented** | `verifiers/cvcad_qa_comparison.py` |
| Learned detector / OCR / robotics / CATScript | **research-heavy/external** | learned/hardware |

### 112. Integrating Deep Learning into CAD, CAE System - Generative Design and Evaluation of 3D Conceptual Wheel

| Build idea | Status | Repository comparison |
|---|---|---|
| Analytic modal relation + inversion + rigid-body-mode count + stiffness-floor screening | **implemented** | `verifiers/dlwheel_modal.py` |
| Surrogate label prep + min-max scaling + RMSE/MAPE + ensemble | **implemented** | `quality/dlwheel_surrogate_eval.py` |
| First-derivative + Sobel edge extraction + edge->coordinate | **implemented** | `geometry/dlwheel_edge.py` |
| Spoke point processing (NN ordering, grouping, reduction, scaling) | **implemented** | `geometry/dlwheel_spoke_points.py` |
| Normal-distribution Latin-hypercube DoE (Acklam probit) | **implemented** | `exploration/dlwheel_lhs.py` |
| Pixelwise L1 dedup + stiffness ranking + diversity score | **implemented** | `quality/dlwheel_design_rank.py` |
| Seven-stage CAD/CAE integration workflow schema | **implemented** | `quality/dlwheel_workflow.py` |
| CNN/surrogate/generative-topology training + real FEA | **research-heavy/external** | trained models / FEA |

### 113. Intelligent CAD 2.0

| Build idea | Status | Repository comparison |
|---|---|---|
| Intensional-vs-extensional ICAD role; 7-module ICAD 2.0 framework; design-phase taxonomy; five research challenges | **out-of-scope / research-heavy** | vision/position essay -- aspirational framework, no specified algorithm (nothing built) |

### 114. Interactive Procedural Computer-Aided Design

| Build idea | Status | Repository comparison |
|---|---|---|
| Procedural serpentine MEMS springs (boustrophedon + mirror + braces) | **implemented** | `geometry/proccad_serpentine.py` |
| Symmetry/repetition operators + parameter reduction | **implemented** | `procedural/proccad_symmetry.py` |
| Incremental dirty-cone re-evaluation of a procedural DAG | **implemented** | `procedural/proccad_incremental.py` |
| Mark-intangible constraint freezing + preservation/projection | **implemented** | `procedural/proccad_constraint_freeze.py` |
| Beauty functionals (arc-length / bending / minimum-variation) | **implemented** | `geometry/proccad_beauty_functionals.py` |
| Key-decision design templates (dependent-parameter propagation) | **implemented** | `procedural/proccad_key_params.py` |
| Narrow-focus finite-difference greedy refinement + multi-start | **implemented** | `exploration/proccad_greedy_refine.py` |
| GA / surface-evolver / UI rendering | **research-heavy/external** | interactive UI / learned |

### 115. Introducing Bidirectional Programming in Constructive Solid Geometry-Based CAD

| Build idea | Status | Repository comparison |
|---|---|---|
| Hierarchical CSG program AST (transforms/booleans/repeat, path identity, serialization) | **implemented** | `programs/bidircsg_ast.py` |
| Forward evaluator get(program)->traced geometry tree + Affine | **implemented** | `programs/bidircsg_forward.py` |
| Reverse/forward navigation + ghost operands + consistency | **implemented** | `editing/bidircsg_navigation.py` |
| Backward put (reuse-or-insert, world->local frame) + GetPut/PutGet lens laws | **implemented** | `editing/bidircsg_backward.py` |
| Variable-combination inference + gizmo UI | **research-heavy/external** | heuristics / UI host |

## Batch-23 implementation result

All deterministic and in-scope findings from papers 111-115 are implemented
(113 Intelligent CAD 2.0 is a vision/position paper -- correctly no-build).
Recovered from a mid-batch session-limit interruption: partials for 112/114/
115 removed and those papers re-run fresh. Per the no-README-during-campaign
policy, the suite count is tracked in audit/text_to_cad_progress.json.

### 116. Joint Neural SDF Reconstruction and Semantic Segmentation for CAD Models

| Build idea | Status | Repository comparison |
|---|---|---|
| kNN label-smoothness consistency + SDF-band surface consistency | **implemented** | `reconstruction/jointsdf_consistency.py` |
| NN label transfer + accuracy + palette-invariant matching | **implemented** | `reconstruction/jointsdf_label_transfer.py` |
| Majority face-labelling + connected-component part segmentation | **implemented** | `geometry/jointsdf_mesh_segments.py` |
| Boundary F-score with graph-hop tolerance | **implemented** | `bench/jointsdf_boundary_fscore.py` |
| Recon<->seg correlation + part-count agreement + joint score | **implemented** | `bench/jointsdf_joint_metrics.py` |
| SIREN/PartField training | **research-heavy/external** | trained nets |

### 117. Large Language and Text-to-3D Models for Engineering Design Optimisation

| Build idea | Status | Repository comparison |
|---|---|---|
| Aerodynamic drag proxy (frontal area + least-squares surrogate) | **implemented** | `verifiers/llmdesopt_drag_proxy.py` |
| Self-adaptive (mu,lambda) evolution strategy | **implemented** | `exploration/llmdesopt_es_optimizer.py` |
| Wu-Palmer taxonomy similarity | **implemented** | `exploration/llmdesopt_wup_similarity.py` |
| Reversible prompt design-variable encodings (bag-of-words + tokenisation) | **implemented** | `exploration/llmdesopt_prompt_encoding.py` |
| Optimisation convergence/diversity metrics | **implemented** | `bench/llmdesopt_convergence.py` |
| Shap-E/Point-E + GPT-4 BPE + OpenFOAM CFD | **research-heavy/external** | learned models / CFD |

### 118. Large Language Models for Computer-Aided Design - A Survey

| Build idea | Status | Repository comparison |
|---|---|---|
| Six-area LLM-in-CAD taxonomy; LLM/dataset/industry tables | **out-of-scope** | pure literature survey -- no self-contained algorithm (nothing built) |

### 119. Learning From Design Procedure To Generate CAD Programs for Data Augmentation

| Build idea | Status | Repository comparison |
|---|---|---|
| B-Spline reference-surface program generation (four families -> CadQuery scripts) | **implemented** | `datagen/designproc_reference_surface.py` |
| Design-procedure step-grammar + prompt template + ablation modes | **implemented** | `datagen/designproc_procedure.py` |
| Program-generation-by-procedure (grammar/template expansion) | **implemented** | `datagen/designproc_program_synthesis.py` |
| B-Spline-ratio validity/diversity/augmentation metrics | **implemented** | `datagen/designproc_bspline_metrics.py` |
| LLM prompting + OCC watertight check | **research-heavy/external** | learned LLM / kernel |

### 120. Lessons on Datasets and Paradigms in Machine Learning for Symbolic Computation - A Case Study on CAD

| Build idea | Status | Repository comparison |
|---|---|---|
| Symmetry-group augmentation + class balancing (orbit/greedy relabel) | **implemented** | `datagen/symcad_symmetry_balance.py` |
| Choice-heuristic evaluation metrics (time-markup, timeout-penalised) + rank-select | **implemented** | `bench/symcad_choice_metrics.py` |
| Outcome-vector leakage dedup | **implemented** | `dataengine/symcad_outcome_dedup.py` |
| Leakage-safe grouped k-fold CV + augment-within-fold | **implemented** | `bench/symcad_grouped_folds.py` |
| Variable-ordering heuristics + trained classifiers (CAD = cylindrical algebraic decomposition) | **research-heavy/external** | symbolic-computation specific / trained models |

## Batch-24 implementation result

All deterministic and in-scope findings from papers 116-120 are implemented
(118 LLMs-for-CAD survey is correctly no-build; 120's "CAD" is cylindrical
algebraic decomposition -- only its transferable domain-agnostic ML dataset
methodology was built). Per the no-README-during-campaign policy, the suite
count is tracked in audit/text_to_cad_progress.json.

### 121. Leveraging Vision-Language Models for Manufacturing Feature Recognition in CAD Designs

| Build idea | Status | Repository comparison |
|---|---|---|
| Hierarchical 5-process manufacturing-feature taxonomy + alias normalization + attribute schema | **implemented** | `fabrication/mfgfeat_taxonomy.py` |
| Four count-sensitive AFR metrics (name accuracy, quantity accuracy, hallucination rate, MAE) | **implemented** | `bench/mfgfeat_afr_metrics.py` |
| Rule-based machining-feature detector (hole subtypes/pocket/slot/step/chamfer/fillet/boss) | **implemented** | `reconstruction/mfgfeat_rule_detector.py` |
| Dimensional-attribute extraction + subtype classification | **implemented** | `fabrication/mfgfeat_attributes.py` |
| VLMs + prompt experiments + CAD2Image rendering | **research-heavy/external** | trained VLMs |

### 122. Leveraging Vision-Language Models for Manufacturing Feature (duplicate of 121)

| Build idea | Status | Repository comparison |
|---|---|---|
| Taxonomy / detector / eval-metrics / attribute-extraction | **covered by paper 121** | `mfgfeat_*` modules |
| Easy/medium/hard difficulty stratification + visually-confusable feature-pair swap diagnostic | **implemented** | `fabrication/mfgfeat2_difficulty.py` |

### 123. LION - Latent Point Diffusion Models for 3D Shape Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| 1-NNA two-sample generative metric + voxel-occupancy JSD | **implemented** | `bench/lion_one_nna.py` |
| Dataset-level global [-1,1] normalization + per-shape variant | **implemented** | `geometry/lion_global_normalize.py` |
| Deterministic DDIM sampler + diffuse-denoise | **implemented** | `numeric/lion_ddim_sampler.py` |
| Chamfer/EMD/FPS/COV-MMD | **already in repo** | earlier modules |
| Learned latent-point VAE + DDMs + SAP + CLIP | **research-heavy/external** | trained models |

### 124. LLaMA-Mesh - Unifying 3D Mesh Generation with Language Models

| Build idea | Status | Repository comparison |
|---|---|---|
| Mesh-as-text OBJ tokenization (quantize/dequantize vertices, serialize/parse round-trip, canonical z-y-x ordering, compression metric) | **implemented** | `formats/llamamesh_tokenization.py` |
| SFT + LLaMA fine-tuning + Objaverse curation | **research-heavy/external** | trained LLM / data |

### 125. Locally Attentional SDF Diffusion for Controllable 3D Shape Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| Surface-occupancy shell + coarse-from-fine 8-subvoxel pooling | **implemented** | `geometry/lasdiff_surface_occupancy.py` |
| Two-stage sparse-voxel subdivision bridge | **implemented** | `geometry/lasdiff_sparse_subdivision.py` |
| View-aware local-attention mask geometry (patch grid + pinhole projection + local-neighbourhood mask) | **implemented** | `geometry/lasdiff_local_attention_mask.py` |
| ViT patch-grid region partition + two-sketch stitch | **implemented** | `geometry/lasdiff_patch_stitch.py` |
| Sketch-CD + 1-NNA + gap-to-50% metrics | **implemented** | `bench/lasdiff_sketch_metrics.py` |
| Learned diffusion U-Net + ViT encoders | **research-heavy/external** | trained models |

## Batch-25 implementation result

All deterministic and in-scope findings from papers 121-125 are implemented
(122 is a materially identical duplicate of 121 -- only its two extra
diagnostics were new). Per the no-README-during-campaign policy, the suite
count is tracked in audit/text_to_cad_progress.json.

### 126. Magic3DSketch - Create Colorful 3D Models From Sketch-Based 3D Modeling Guided by Text and Language-Image Pre-Training

| Build idea | Status | Repository comparison |
|---|---|---|
| Soft silhouette-IoU loss + multi-scale mIoU | **implemented** | `bench/magic3d_silhouette_iou.py` |
| Voxel-IoU + viewpoint MAE/MSE metrics | **implemented** | `bench/magic3d_voxel_metrics.py` |
| Sphere-template deform + flatten loss | **implemented** | `geometry/magic3d_template_deform.py` |
| Barycentric mesh colorization + text-word palette | **implemented** | `geometry/magic3d_mesh_colorize.py` |
| CLIP discriminator + differentiable rendering | **research-heavy/external** | trained CLIP / renderer |

### 127. Make-A-Shape - A Ten-Million-scale 3D Shape Model

| Build idea | Status | Repository comparison |
|---|---|---|
| Separable 3D discrete wavelet transform (Haar + Le Gall 5/3, exact round-trip, multi-level tree) | **implemented** | `numeric/makeashape_wavelet_transform.py` |
| Subband coefficient filtering + adaptive coordinate sets + diffusible packing | **implemented** | `numeric/makeashape_wavelet_tree.py` |
| Wavelet-compression fidelity metrics | **implemented** | `numeric/makeashape_compression_metric.py` |
| Learned U-ViT diffusion + condition encoders | **research-heavy/external** | trained model |

### 128. Mamba-CAD - State Space Model For 3D Computer-Aided Design Generative Modeling

| Build idea | Status | Repository comparison |
|---|---|---|
| ZOH discretization of a continuous SSM (produces geofusion's discrete kernels) | **implemented** | `numeric/mambacad_zoh_discretization.py` |
| Bidirectional/multi-directional scan ordering (reuses selective_scan) | **implemented** | `numeric/mambacad_bidirectional_scan.py` |
| Long-sequence length statistics | **implemented** | `bench/mambacad_length_metrics.py` |
| CAD-rep / Ac-Ap / selective scan | **already in repo** | deepcad / contrastcad / geofusion |
| Learned Mamba model | **research-heavy/external** | trained model |

### 129. MamTiff-CAD - Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence

| Build idea | Status | Repository comparison |
|---|---|---|
| Multi-scale/pyramid sequence encoding (Gaussian + Laplacian pyramid) | **implemented** | `numeric/mamtiff_pyramid.py` |
| Cross-scale adaptive fusion + window-mask attention + scaled PE | **implemented** | `numeric/mamtiff_fusion.py` |
| Complex-parametric-sequence complexity measure + ABC-256 filter | **implemented** | `numeric/mamtiff_complexity.py` |
| Learned Mamba+ encoder + MST-D diffusion | **research-heavy/external** | trained model |

### 130. MeshDiffusion - Score-Based Generative 3D Mesh Modeling

| Build idea | Status | Repository comparison |
|---|---|---|
| Uniform tetrahedral grid (Kuhn/Freudenthal, positive-volume re-orientation) | **implemented** | `geometry/meshdiff_tet_grid.py` |
| Marching tetrahedra (16-case DMTet table, validated vs analytic SDFs) | **implemented** | `geometry/meshdiff_marching_tets.py` |
| DMTet deformable-tet encoding (SDF + deformation, sign-normalization) | **implemented** | `geometry/meshdiff_dmtet.py` |
| Marching-tets edge-crossing noise-sensitivity metric | **implemented** | `geometry/meshdiff_edge_sensitivity.py` |
| Learned score network + diffusion training | **research-heavy/external** | trained model |

## Batch-26 implementation result

All deterministic and in-scope findings from papers 126-130 are implemented.
Recovered from a mid-batch WEEKLY-limit interruption (harder than the earlier
session limits): all partials removed and the papers re-run fresh; 128's
first retry returned empty (0 tool uses) and was re-run again. Notable new
capabilities: a 3D wavelet transform (Make-A-Shape) and marching tetrahedra
(MeshDiffusion), neither of which existed. Per the no-README-during-campaign
policy, the suite count is tracked in audit/text_to_cad_progress.json.

### 131. Meshtron - High-Fidelity, Artist-Like 3D Mesh Generation At Scale

| Build idea | Status | Repository comparison |
|---|---|---|
| Meshtron y-z-x vertex ordering convention (distinct from LLaMA-Mesh z-y-x) | **implemented** | `formats/meshtron_ordering.py` |
| Hourglass coarse-to-fine token layout + sliding-window inference | **implemented** | `formats/meshtron_windowing.py` |
| Decoded-stream order-enforcement/validity checker | **implemented** | `formats/meshtron_order_enforcement.py` |
| Artist-like mesh-quality metrics (aspect/valence/face-area) | **implemented** | `bench/meshtron_mesh_quality.py` |
| Learned 1.1B Hourglass Transformer + Perceiver encoder | **research-heavy/external** | trained model |

### 132. mrCAD - Multimodal Refinement of Computer-Aided Designs

| Build idea | Status | Repository comparison |
|---|---|---|
| 2D CAD-state grammar + typed edit-operation vocabulary + multimodal message/instruction parser | **implemented** | `editing/mrcad_schema.py` |
| Refinement transition function + round/rollout state machine | **implemented** | `editing/mrcad_refinement.py` |
| Refinement metrics (vector chamfer, Proportional-Improvement, edit accuracy, convergence) | **implemented** | `bench/mrcad_metrics.py` |
| Learned designer/maker VLM + crowd pipeline | **research-heavy/external** | trained VLM |

### 133. MUSE - Benchmarking Manufacturable, Functional, and Assemblable Text-to-CAD Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| Manufacturability scorer (table-grounded feasibility) | **implemented** | `bench/muse_manufacturability.py` |
| Functionality scorer (parameter space + support-polygon stability) | **implemented** | `bench/muse_functionality.py` |
| Assemblability scorer (assembly-graph isomorphism + Table-7 joints) | **implemented** | `bench/muse_assemblability.py` |
| Three-stage funnel scorecard | **implemented** | `bench/muse_scorecard.py` |
| Rubric VLM judge + generator | **research-heavy/external** | trained VLM |

### 134. Neural Surrogate-Driven Modelling, Optimisation, and Generation of Engineering Designs - A Concise Review

| Build idea | Status | Repository comparison |
|---|---|---|
| Representation taxonomy, surrogate-modeling narrative, LHS/DoE + surrogate-assisted optimization | **out-of-scope** | pure literature review -- methodologies cited not specified, and already in the repo (nothing built) |

### 135. NeurCADRecon - Neural Representation for Reconstructing CAD Surfaces by Enforcing Zero Gaussian Curvature

| Build idea | Status | Repository comparison |
|---|---|---|
| Zero-Gaussian-curvature developability energy + double-trough quartic + annealing/projection | **implemented** | `geometry/neurcad_developability.py` |
| Developable detector/classifier (shape-operator rank, tip detection) | **implemented** | `geometry/neurcad_developable_detect.py` |
| Developability + Gauss-Bonnet metrics | **implemented** | `bench/neurcad_metrics.py` |
| Reuses FlatCAD Goldman curvature | **already in repo** | `geometry/flatcad_weingarten.py` |
| Neural SDF training | **research-heavy/external** | trained model |

## Batch-27 implementation result

All deterministic and in-scope findings from papers 131-135 are implemented
(134 Neural-Surrogate review is correctly no-build). Recovered from two
back-to-back limit interruptions (weekly then session) on this batch:
partials removed and papers re-run fresh. Notable: NeurCADRecon reuses paper
81's FlatCAD curvature. Per the no-README-during-campaign policy, the suite
count is tracked in audit/text_to_cad_progress.json.

### 136. NURBGen - High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling

| Build idea | Status | Repository comparison |
|---|---|---|
| Cox-de Boor B-spline basis + knot vectors + basis derivatives | **implemented** | `numeric/nurbs_basis.py` |
| Rational NURBS curve evaluation + tangent + tessellation | **implemented** | `geometry/nurbgen_curve.py` |
| Tensor-product NURBS surface + normal + mesh tessellation | **implemented** | `geometry/nurbgen_surface.py` |
| Boehm knot insertion/refinement + Bezier decomposition | **implemented** | `geometry/nurbgen_knot_insertion.py` |
| Hybrid analytic-primitive fidelity gate (Chamfer fallback) | **implemented** | `geometry/nurbgen_hybrid_primitives.py` |
| LLM text->JSON generation | **research-heavy/external** | trained LLM |

### 137. OctFusion - Octree-based Diffusion Models for 3D Shape Generation

| Build idea | Status | Repository comparison |
|---|---|---|
| Volumetric region octree (subdivision, Morton order, neighbor queries, octree<->voxel) | **implemented** | `geometry/octfusion_octree.py` |
| Split-signal encoding of octree structure + round-trip | **implemented** | `geometry/octfusion_split_signal.py` |
| Multi-level Partition-of-Unity blending | **implemented** | `geometry/octfusion_mpu.py` |
| Learned octree VAE + diffusion U-Net | **research-heavy/external** | trained model |

### 138. OpenECAD - An Efficient Visual Language Model for Editable 3D-CAD Design

| Build idea | Status | Repository comparison |
|---|---|---|
| Editable named-variable CAD-script DSL (emitter + ast parser + round-trip) | **implemented** | `programs/openecad_script.py` |
| Sketch/loop validity + loop grouping | **implemented** | `programs/openecad_validity.py` |
| Editability operations (rename/reparametrize/re-emit) | **implemented** | `programs/openecad_edit.py` |
| Dependency-based reference-plane finding | **implemented** | `reconstruction/openecad_refplane.py` |
| Generation scoring metric | **implemented** | `bench/openecad_score.py` |
| VLM + LoRA training | **research-heavy/external** | trained VLM |

### 139. OSCAR - Open-Set CAD Retrieval from a Language Prompt and a Single Image

| Build idea | Status | Repository comparison |
|---|---|---|
| Two-stage multimodal text+image late-fusion retrieval | **implemented** | `rag/oscar_multimodal_fusion.py` |
| Open-set recognition metrics (AUROC, open-set F-measure, rejection, novelty) | **implemented** | `bench/oscar_openset_metrics.py` |
| MI3DOR benchmark criteria (First/Second Tier, ANMRR) | **implemented** | `bench/oscar_mi3dor_metrics.py` |
| Learned CLIP/DINOv2/LLaVA encoders + pose pipelines | **research-heavy/external** | trained models |

### 140. Parametric + Direct CAD integration

| Build idea | Status | Repository comparison |
|---|---|---|
| Two-paradigm data models + edit classification by paradigm/layer | **implemented** | `editing/paramdirect_model.py` |
| Pseudo-Feature integration (anchor-invalidation detection) | **implemented** | `editing/paramdirect_pseudofeature.py` |
| Synchronous Technology partial conversion | **implemented** | `editing/paramdirect_synctech.py` |
| Operation Translating (push-pull -> parameter candidates) | **implemented** | `editing/paramdirect_translate.py` |
| Three-layer consistency reconciliation + bidirectional propagation | **implemented** | `editing/paramdirect_consistency.py` |
| Efficient 3D constraint solving | **out-of-scope** | open problem / needs GCS solver |

## Batch-28 implementation result

All deterministic and in-scope findings from papers 136-140 are implemented.
Notable new capabilities: full NURBS evaluation machinery (Cox-de Boor basis,
rational curve/surface, knot insertion -- NURBGen) and an octree structure
(OctFusion), neither of which existed. Per the no-README-during-campaign
policy, the suite count is tracked in audit/text_to_cad_progress.json.

### 141. Parametric Primitive Analysis of CAD Sketches with Vision Transformer

| Build idea | Status | Repository comparison |
|---|---|---|
| Typed sketch-primitive representation (line/circle/arc/point + flag + 7-slot rows) | **implemented** | `reconstruction/ppa_primitive.py` |
| Coordinate normalization + 6-bit quantization | **implemented** | `reconstruction/ppa_quantization.py` |
| Least-squares primitive fitting (TLS line, Kasa circle, 3-point arc) | **implemented** | `geometry/ppa_primitive_fit.py` |
| Hungarian-matched evaluation protocol (type/flag/param accuracy + Chamfer) | **implemented** | `bench/ppa_primitive_eval.py` |
| ViT image-patch tokenization | **implemented** | `vision/ppa_patch_tokenizer.py` |
| ViT/DETR networks + pointer module | **research-heavy/external** | trained models |

### 142. PHT-CAD - Efficient CAD Parametric Primitive Analysis with Progressive Hierarchical Tuning

| Build idea | Status | Repository comparison |
|---|---|---|
| Efficient Hybrid Parametrization + efficiency metric | **implemented** | `reconstruction/pht_ehp.py` |
| Three-stage progressive-hierarchical-tuning curriculum + coarse-to-fine refinement | **implemented** | `reconstruction/pht_progressive_tuning.py` |
| Parametric-MSE + combined CE/P-MSE objective | **implemented** | `reconstruction/pht_pmse_loss.py` |
| Dimension-Accuracy metric | **implemented** | `reconstruction/pht_dimension_accuracy.py` |
| ViT+Qwen VLM + ParaCAD dataset | **research-heavy/external** | trained model |

### 143. PICASSO - A Feed-Forward Framework for Parametric Inference of CAD Sketches via Rendering Self-Supervision

| Build idea | Status | Repository comparison |
|---|---|---|
| Deterministic sketch-primitive rasterizer (line/arc/circle SDF) | **implemented** | `drawings/picasso_rasterizer.py` |
| Multiscale rendering-consistency loss + image pyramids | **implemented** | `drawings/picasso_render_loss.py` |
| Label-free render-compare self-supervision + gradient-free refinement | **implemented** | `drawings/picasso_self_supervision.py` |
| Image-based eval (ImgMSE, foreground Chamfer, IoU) | **implemented** | `drawings/picasso_metrics.py` |
| Learned SPN/SRN nets + Hungarian matching | **research-heavy/external** | trained models |

### 144. PLLM - Pseudo-Labeling Large Language Models for CAD Program Synthesis

| Build idea | Status | Repository comparison |
|---|---|---|
| Best-of-k pseudo-label selection (Chamfer winner + validity gate) | **implemented** | `dataengine/pllm_pseudo_label_selection.py` |
| Per-candidate confidence/quality score | **implemented** | `dataengine/pllm_confidence_score.py` |
| Self-training accumulator + drift detection + label-efficiency | **implemented** | `dataengine/pllm_selftrain_accumulator.py` |
| Program-level structural augmentation | **implemented** | `datagen/pllm_program_augment.py` |
| LoRA + encoder + executor | **research-heavy/external** | trained LLM |

### 145. Pointer-CAD - Unifying B-Rep and Command Sequences via Pointer-based Edges & Faces Selection

| Build idea | Status | Repository comparison |
|---|---|---|
| Pointer token vocabulary + non-overlapping ID scheme | **implemented** | `reconstruction/pointercad_tokens.py` |
| Stable B-rep entity addressing (well-defined pointers) | **implemented** | `reconstruction/pointercad_indexing.py` |
| Pointer-based commands + dangling-pointer detection | **implemented** | `reconstruction/pointercad_pointer.py` |
| Pointer-based sketch-plane construction | **implemented** | `reconstruction/pointercad_sketchplane.py` |
| Unified B-rep<->command linkage + replay | **implemented** | `reconstruction/pointercad_linkage.py` |
| Pointer-accuracy + invalidity metrics | **implemented** | `reconstruction/pointercad_metrics.py` |
| Learned GNN encoder + Qwen backbone | **research-heavy/external** | trained model |

## Batch-29 implementation result

All deterministic and in-scope findings from papers 141-145 are implemented.
Papers 141/142/143 share a CAD-sketch parametric-primitive theme; 141 built
the foundation (ppa_) and 142/143 built their distinct method contributions
(progressive tuning; rendering self-supervision) without duplication. Per
the no-README-during-campaign policy, the suite count is tracked in
audit/text_to_cad_progress.json.

### 146. PS-CAD - Local Geometry Guidance via Prompting and Selection for CAD Reconstruction

| Build idea | Status | Repository comparison |
|---|---|---|
| Local-geometry-difference detector (bidirectional NN residual regions + radius-graph clustering) | **implemented** | `reconstruction/pscad_residual_regions.py` |
| Planar-prompt representation (RANSAC plane + inlier prompts + hull boundary) | **implemented** | `geometry/pscad_planar_prompt.py` |
| Multi-strategy candidate selection (bbox-IoU fitness) | **implemented** | `reconstruction/pscad_candidate_selection.py` |
| Reconstruction metric suite (CD/HD/ECD/NC/IR) | **implemented** | `bench/pscad_reconstruction_metrics.py` |
| Learned Point-MAE encoder + f-dec selection net | **research-heavy/external** | trained models |

### 147. Query2CAD - Generating CAD Models Using Natural Language Queries

| Build idea | Status | Repository comparison |
|---|---|---|
| FreeCAD Part-macro representation (primitives + booleans + serialization) | **implemented** | `generation/query2cad_macro.py` |
| VQAScore stopping criterion (continuous threshold gate) | **implemented** | `bench/query2cad_vqascore.py` |
| Success/difficulty benchmark metrics + failure taxonomy | **implemented** | `bench/query2cad_metrics.py` |
| Caption-vs-query feedback + human override | **implemented** | `generation/query2cad_feedback.py` |
| LLM + BLIP2 + VQA models | **research-heavy/external** | trained models |

### 148. QueryCAD - Grounded Question Answering for CAD Models

| Build idea | Status | Repository comparison |
|---|---|---|
| Typed CAD-QA query schema (count/measure/existence/position/comparison) | **implemented** | `bench/querycad_query_schema.py` |
| Segmentation-to-answer grounding (mfgfeat taxonomy + coverage/view filters) | **implemented** | `rag/querycad_segmentation_grounding.py` |
| Grounded answer engine (traceable part-id evidence) | **implemented** | `reconstruction/querycad_answer_engine.py` |
| QA evaluation (numeric tolerance + partial credit + error taxonomy) | **implemented** | `bench/querycad_eval.py` |
| SegCAD GroundingDINO/SAM + LLM | **research-heavy/external** | trained models |

### 149. RAG-6DPose - Retrieval-Augmented 6D Pose Estimation via Leveraging CAD as Knowledge Base

| Build idea | Status | Repository comparison |
|---|---|---|
| 6D pose error metrics (ADD/ADD-S, geodesic rotation, translation, 5cm-5deg) | **implemented** | `bench/rag6d_pose_metrics.py` |
| PnP-RANSAC robust pose selection (reuses Umeyama) | **implemented** | `geometry/rag6d_ransac_pose.py` |
| CAD-knowledge-base retrieval + pose-hypothesis ranking | **implemented** | `rag/rag6d_cad_retrieval.py` |
| Learned DINOv2/ReSPC feature matching | **research-heavy/external** | trained model |

### 150. ReCAD - Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models

| Build idea | Status | Repository comparison |
|---|---|---|
| Unified reward (min of IoU + thresholded semantic sim + format reward) | **implemented** | `dataengine/recad_reward.py` |
| Hard-question identification + objective routing | **implemented** | `dataengine/recad_hard_question.py` |
| Hierarchical Primitive Learning curriculum | **implemented** | `dataengine/recad_hpl_curriculum.py` |
| Inertia scale normalization | **already in repo** | `bench/solid_iou.py` |
| VLM SFT/RL training + GRPO gradient | **research-heavy/external** | trained model |

## Batch-30 implementation result

All deterministic and in-scope findings from papers 146-150 are implemented.
Papers 149 and 150 each returned empty (0 tool uses) on first launch -- flaky
starts, not a limit; both re-ran cleanly. Per the no-README-during-campaign
policy, the suite count is tracked in audit/text_to_cad_progress.json.
