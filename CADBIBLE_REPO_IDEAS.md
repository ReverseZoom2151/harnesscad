# CADBible Repo Mining — Build Ideas Log

Deterministic, locally-buildable ideas mined from each repo in
`resources/cadbible/` and implemented as tested stdlib-only modules.
Repos that are paper reference-impls already covered by the 186-paper campaign
(see TEXT_TO_CAD_PAPER_IDEAS.md) yield only genuinely-new implementation-level
ideas. Learned/proprietary/kernel-dependent work is logged research-heavy/external.


### 1. AlphaCAD-main (BrickGPT demo app)

| Build idea | Status | Repository comparison |
|---|---|---|
| Model-free parametric voxel-object generators (table/chair/tower/... + categories) | **implemented** | `procedural/alphacad_brick_templates.py` |
| Fast geometry-only structural heuristics (floating/overhang/mass/entropy/symmetry) | **implemented** | `quality/alphacad_structure_scoring.py` |
| Cross-variant consensus + per-voxel ensemble confidence | **implemented** | `exploration/alphacad_variant_consensus.py` |
| Voxel-scene composition + clamped dimensional mutation | **implemented** | `procedural/alphacad_voxel_compose.py` |
| Text-intent -> brick-category router + prompt linter | **implemented** | `spec/alphacad_intent_categories.py` |
| BrickGPT core (brick/stability/buildability) | **already in repo** | papers 94/95 brick_*/lego_* |
| LLM gen + CLIP + Three.js UI | **research-heavy/external** | trained models / UI |

### 2. AutoCAD-main (manufino Python/COM wrapper)

| Build idea | Status | Repository comparison |
|---|---|---|
| Dimension measurement + placement geometry (generation) | **implemented** | `drawings/autocad_dimension_geometry.py` |
| Linetype dash-pattern application | **implemented** | `drawings/autocad_linetype_dash.py` |
| Bbox align + distribute | **implemented** | `editing/autocad_layout_ops.py` |
| Array/repetition patterns (linear/fit/grid/polar) | **implemented** | `procedural/autocad_array.py` |
| AutoCAD Color Index (ACI) table + nearest matcher | **implemented** | `standards/autocad_aci_color.py` |
| COM entity CRUD / layers / blocks / file IO | **out-of-scope** | proprietary COM host |

### 3. BlenderLLM-main

| Build idea | Status | Repository comparison |
|---|---|---|
| OBJ parse + size-invariant 8-corner camera rig | **implemented** | `geometry/blenderllm_camera_rig.py` |
| bpy-script static analyzer (syntax gate + AST op-sequence) | **implemented** | `programs/blenderllm_bpy_script.py` |
| CADBench task-complexity metrics (unit count/param density/entropy) | **implemented** | `bench/blenderllm_complexity.py` |
| CADBench criteria scoring | **already in repo** | `bench/criteria.py` |
| LLM inference + Blender render | **research-heavy/external** | trained model / runtime |

### 4. CAD-Coder-main (fine-tuned VLM)

| Build idea | Status | Repository comparison |
|---|---|---|
| Correspondence-free inertia/principal-axis shape alignment (SolidAlign) | **implemented** | `geometry/cadcoder_solidalign.py` |
| Inertia-normalized symmetry-enumerated IoU / mesh sampling / eigensolver / CadQuery DSL | **already in repo** | solid_iou / mesh_sampling / e3dbench_umeyama / t2cq_* |
| Fine-tuned VLM + LLaVA training | **research-heavy/external** | trained model |

### 5. CAD-Editor-main (Locate-then-Infill, ICML 2025)

| Build idea | Status | Repository comparison |
|---|---|---|
| Fine-grained partial-token locate mask (component-level) | **implemented** | `editing/cadeditor_partial_mask.py` |
| Edit-type classification (add/delete/modify) + prefix-bucketed pairing | **implemented** | `dataengine/cadeditor_edit_typing.py` |
| Whole-token LCS masking / edit-pair filtering / SkexGen serialization | **already in repo** | locate_infill / edit_filters / paper campaign |
| LLM finetune + CLIP + OCC render | **research-heavy/external** | trained models |

## Batch-1 implementation result

Mined repos 1-5. Strong dedup: AlphaCAD (papers 94/95), CAD-Coder (VLM),
CAD-Editor (ICML paper) all partly covered -- extracted only genuinely-new
implementation-level pieces. AutoCAD (COM wrapper) yielded the most net-new
(drafting algorithms/dash/ACI). Per the no-README-during-campaign policy the
suite count is tracked in audit/cadbible_progress.json.

### 6. CAD-GPT-main

| Build idea | Status | Repository comparison |
|---|---|---|
| Involute spur-gear geometry (radii, involute curve, rack cutter) | **implemented** | `geometry/cadgpt_involute_gear.py` |
| Gear-pair meshing + assembly placement (ratio, centre distance, mesh phase, twist) | **implemented** | `geometry/cadgpt_gear_train.py` |
| Standard ISO-54 gear-module series snapping | **implemented** | `geometry/cadgpt_module_series.py` |
| GPT agent + OpenSCAD render | **research-heavy/external** | trained model / host |

### 7. CAD-MCP-main

| Build idea | Status | Repository comparison |
|---|---|---|
| 2D drawing-command geometry (rectangle/ellipse/arc/polyline + lineweight snap) | **implemented** | `drawings/cadmcp_drawing_commands.py` |
| Coordinate/parameter regex extractor + command classification | **implemented** | `programs/cadmcp_command_parser.py` |
| MCP plumbing / COM driver / ACI / NL semantics | **already in repo / out-of-scope** | surfaces/mcp / autocad_aci_color / nlcad |

### 8. CAD2Program-gh-pages

| Build idea | Status | Repository comparison |
|---|---|---|
| Shape-program representation / pose normalization / metrics / view lifting | **already in repo (paper 84)** | cad2program_* modules |
| ViT/InternVL VLM + website | **out-of-scope** | trained model / static site (nothing built) |

### 9. CADAM-master

| Build idea | Status | Repository comparison |
|---|---|---|
| NACA 4-digit airfoil coordinate generator | **implemented** | `geometry/cadam_naca_airfoil.py` |
| OpenSCAD Customizer parameter parser | **implemented** | `programs/cadam_scad_customizer.py` |
| Cycle-safe conversation/branch message tree | **implemented** | `agents/cadam_message_tree.py` |
| Gear math | **already in repo** | cadgpt_involute_gear |
| React/Three.js UI + WASM engine | **research-heavy/external** | UI / kernel |

### 10. CADCLAW-main

| Build idea | Status | Repository comparison |
|---|---|---|
| Tolerance-stack analyzer (worst-case/RSS/Monte-Carlo + Cpk) | **implemented** | `verifiers/cadclaw_tolerance_stack.py` |
| Static-frame beam screening (section props, torsion, deflection, motor torque) | **implemented** | `quality/cadclaw_beam_screening.py` |
| Clearance-shift suggester (interference -> fix vector) | **implemented** | `verifiers/cadclaw_clearance_shift.py` |
| Exploded-view geometry (radial burst + removal order) | **implemented** | `geometry/cadclaw_explode.py` |
| Claim/honesty text linter | **implemented** | `quality/cadclaw_claim_audit.py` |
| Interference boolean / STEP / FEA solver | **research-heavy/external** | kernel / FEA |

## Batch-2 implementation result

Mined repos 6-10. CAD-GPT surfaced involute-gear geometry (the harness only
had a toothless blank); CADAM added the first aerodynamic geometry (NACA
airfoil) + OpenSCAD-customizer parser; CADCLAW added tolerance stacking, beam
screening, and exploded views. CAD2Program is a static site (paper 84 already
covers it -> no build). Per the no-README policy the suite count is tracked in
audit/cadbible_progress.json.

### 11. CADTestBench-main (paper-169 reference impl)

| Build idea | Status | Repository comparison |
|---|---|---|
| String-based CADTEST execution harness (run test as code string, AST model recovery, check() preamble, replay script) | **implemented** | `bench/cadtb_exec.py` |
| CADTEST predicate / suite / metrics / mutation analysis | **already in repo (paper 169)** | bench/cadtests_* |

### 12. CADTransformer-main (CVPR 2022 panel-symbol spotting)

| Build idea | Status | Repository comparison |
|---|---|---|
| Vectorized floorplan primitive graph (endpoint-KNN adjacency, 6-D node feature, perimeter formulas) | **implemented** | `drawings/cadtransformer_primitive_graph.py` |
| Length-weighted primitive-instance panoptic metrics (mask-free SQ/RQ/PQ + F1) | **implemented** | `bench/cadtransformer_panoptic.py` |
| Instance centroid-offset targets + vote-to-centroid clustering | **implemented** | `reconstruction/cadtransformer_instance_offsets.py` |
| Transformer/HRNet backbones | **research-heavy/external** | trained models |

### 13. CQ-editor-master (CadQuery GUI, ~95% Qt/OCC)

| Build idea | Status | Repository comparison |
|---|---|---|
| Qt-free code-edit text ops (comment toggle, indent, EOL, gutter) | **implemented** | `editing/cqeditor_code_edit.py` |
| show_object result model (binding-name inference, seeded rand_color) | **implemented** | `programs/cqeditor_show_object.py` |
| Module-registry snapshot/restore sandbox | **implemented** | `programs/cqeditor_module_sandbox.py` |
| Qt/OCC viewer / debugger / inspector UI | **out-of-scope** | GUI |

### 14. CQAsk-main (CadQuery LLM assistant) -- partial

| Build idea | Status | Repository comparison |
|---|---|---|
| CadQuery API reference/retrieval index | **implemented** | `generation/cqask_api_reference.py` |
| Code-gen scaffold/sanitizer | **pending** | agent hit session limit mid-write; follow-up |
| LLM | **research-heavy/external** | trained model |

### 15. CascadeStudio-master (OCC.js/WASM web CAD) -- partial

| Build idea | Status | Repository comparison |
|---|---|---|
| CAD entity-selection heuristic (kernel-free edge selection) | **implemented** | `geometry/cascade_entity_selector.py` |
| Sketch-path sampler | **pending** | agent hit session limit mid-write; follow-up |
| OCC.js kernel / Three.js/Monaco UI | **out-of-scope** | kernel / UI |

## Batch-3 implementation result

Mined repos 11-15. CADTestBench (paper 169) + CADTransformer (CVPR) yielded
implementation-level pieces; CQ-editor gave Qt-free text/module helpers. A
session-limit interruption left CQAsk and CascadeStudio at module 1 each
(module 2 pending as a follow-up); untested partials were removed. Per the
no-README policy the suite count is tracked in audit/cadbible_progress.json.

### 16. CodeToCAD-develop

| Build idea | Status | Repository comparison |
|---|---|---|
| Unit-aware length/angle expression evaluator (parser + dimensional type-checking) | **implemented** | `numeric/codetocad_length_expression.py` |
| Cardinal-direction landmark system (compositional anchors, bbox, nearest) | **implemented** | `geometry/codetocad_cardinal_landmark.py` |
| Relative/proportional dimension resolver | **implemented** | `geometry/codetocad_axis_expression.py` |
| General 4x4 transform stack (upstream is NotImplementedError stubs) | **implemented** | `geometry/codetocad_transform_stack.py` |
| Code-CAD operation schema + program type-checker | **implemented** | `programs/codetocad_operation_schema.py` |
| Blender/build123d backends | **research-heavy/external** | runtime |

### 17. ComplexGen-main (SIGGRAPH 2022)

| Build idea | Status | Repository comparison |
|---|---|---|
| 3D surface primitive fitting (plane/sphere/cylinder/cone LSQ + quadric distances) | **implemented** | `geometry/complexgen_surface_fit.py` |
| B-rep chain complex (EV/FE incidence + d1*d2=0 law + watertightness) | **implemented** | `reconstruction/complexgen_chain_complex.py` |
| Probabilistic-complex NMS + constraint repair | **implemented** | `reconstruction/complexgen_complex_nms.py` |
| .complex file format IO | **implemented** | `reconstruction/complexgen_complex_io.py` |
| Complex-structure + topology F1 metrics | **implemented** | `bench/complexgen_complex_metrics.py` |
| Transformer + Gurobi ILP | **research-heavy/external** | trained model / solver |

### 18. DeepCAD-master (paper reference impl)

| Build idea | Status | Repository comparison |
|---|---|---|
| Exact numericalize/denumericalize affine families + NORM_FACTOR | **implemented** | `reconstruction/deepcad2_numericalize.py` |
| Arc macro decode (centre/radius recovery, bulge-aware bbox) | **implemented** | `reconstruction/deepcad2_arc_macro.py` |
| 17-column h5 vector layout with the RELEASED command index order | **implemented** | `reconstruction/deepcad2_vector_layout.py` |
| Official mask-gated ACC_cmd/ACC_param | **implemented** | `bench/deepcad2_ae_accuracy.py` |
| Command spec / sketch plane / profile assembly / Chamfer | **already in repo** | deepcad_* modules |

### 19. GenCAD-main (papers 90/91 reference impl)

| Build idea | Status | Repository comparison |
|---|---|---|
| Arc vector<->geometry closed form + bulge-aware bbox + sampling | **implemented** | `geometry/gencad2_arc_vector.py` |
| Exact loop canonicalization (orientation repair + CCW enforcement) | **implemented** | `reconstruction/gencad2_loop_reorder.py` |
| Exact quantization affine constants | **implemented** | `reconstruction/gencad2_sketch_quantize.py` |
| CMD_ARGS_MASK loss masking + CCIP cross-modal InfoNCE | **implemented** | `bench/gencad2_loss_masks.py` |
| FID / retrieval / latent alignment / seq-len norm / synth-bal | **already in repo** | papers 90/91 |

### 20. Graph-CAD-main (ICLR 2026)

| Build idea | Status | Repository comparison |
|---|---|---|
| FORMAT-v4 decomposition-graph parser + validator + build-wave scheduler | **implemented** | `reconstruction/graphcad_knowledge_graph.py` |
| Align/offset/polar placement solver | **implemented** | `geometry/graphcad_align.py` |
| Grid/polar pattern expansion | **implemented** | `geometry/graphcad_pattern.py` |
| Orientation/rotation directives -> matrices | **implemented** | `geometry/graphcad_orientation.py` |
| Graph -> typed CAD action-plan compiler + plan validator | **implemented** | `reconstruction/graphcad_action_plan.py` |
| CADBench corpus scoring protocol | **implemented** | `bench/graphcad_cadbench_report.py` |
| LLM stages + Blender + VLM judge | **research-heavy/external** | trained models |

## Batch-4 implementation result

Mined repos 16-20 -- the richest batch yet. ComplexGen filled a major gap: 3D
surface primitive fitting (plane/sphere/cylinder/cone), which the harness
lacked entirely (only 2D line/circle/arc). CodeToCAD gave units/landmarks/
transforms/operation-schema. DeepCAD + GenCAD (already-covered papers) still
yielded implementation-level gaps from their REFERENCE CODE -- notably DeepCAD's
released command index order differs from the paper order our spec uses, a
silent permutation for real .h5 data. Per the no-README policy the suite count
is tracked in audit/cadbible_progress.json.

### 21. JoinABLe-main (CVPR 2022, Autodesk)

| Build idea | Status | Repository comparison |
|---|---|---|
| Joint-axis derivation from B-rep entity parameters + colinearity test | **implemented** | `geometry/joinable_joint_axis.py` |
| Joint transform algebra (axis alignment, rotation param, Householder flip) | **implemented** | `geometry/joinable_joint_transform.py` |
| Joint-type motion model (pose params, DOF projection, motion sampling) | **implemented** | `geometry/joinable_joint_motion.py` |
| Joint-prediction metrics (hit@top-k over entity pairs, precision curve, MRR, axis error) | **implemented** | `bench/joinable_joint_metrics.py` |
| B-rep joint entity features (type/area/convexity/dihedral + common-scale norm) | **implemented** | `reconstruction/joinable_entity_features.py` |
| Trained GNN + mesh search | **research-heavy/external** | trained model |

### 22. OCP-master (pybind11 OCCT bindings)

| Build idea | Status | Repository comparison |
|---|---|---|
| C++ header declaration parser | **implemented** | `formats/ocp_cpp_header_parser.py` |
| OCCT API catalog (class/method inventory, arity check, near-miss suggestions) | **implemented** | `backends/ocp_occt_api_catalog.py` |
| Generated bindings / CMake / pywrap / LIEF symbol dump | **out-of-scope** | build tooling |

### 23. OpenCAD-main (modular open CAD system)

| Build idea | Status | Repository comparison |
|---|---|---|
| Face identity across rebuilds (TOPOLOGICAL NAMING: fingerprint + provenance) | **implemented** | `geometry/opencad_face_fingerprint.py` |
| Numeric constraint diagnostics (Jacobian, rank, DOF, Gauss-Newton) | **implemented** | `numeric/opencad_constraint_jacobian.py` |
| Feature-tree rebuild engine (stale propagation, suppression, cascade guard) | **implemented** | `state/opencad_feature_rebuild.py` |
| Parametric expression evaluator + parameter table | **implemented** | `numeric/opencad_param_expression.py` |
| Synthetic analytic topology with semantic tags | **implemented** | `geometry/opencad_synthetic_topology.py` |
| Shape-level kernel preflight + error taxonomy | **implemented** | `verifiers/opencad_kernel_preflight.py` |
| Selector engine / op-log replay / typed registry | **already in repo** | cascade_entity_selector / opdag / cisp |
| OCCT backend + LLM agent + React viewport | **research-heavy/external** | kernel / model / UI |

### 24. ScadLM-main

| Build idea | Status | Repository comparison |
|---|---|---|
| OpenSCAD lexer + parser + typed AST + unparser | **implemented** | `programs/scadlm_ast.py` |
| Deterministic SCAD -> CSG evaluator (kernel-free geometry oracle) | **implemented** | `geometry/scadlm_csg_eval.py` |
| Static OpenSCAD validity gate (binary-free) | **implemented** | `programs/scadlm_check.py` |
| Binary-free SCAD geometry metrics (voxel IoU + failure reasons) | **implemented** | `bench/scadlm_geometry_match.py` |
| LLM + openscad binary + vision judge | **research-heavy/external** | model / binary |

### 25. Sketch2CAD-master (SIGGRAPH Asia 2020)

| Build idea | Status | Repository comparison |
|---|---|---|
| 17-channel training-block codec (stdlib reimpl of a C++ TF op) | **implemented** | `drawings/s2cadsig_block_codec.py` |
| Sketch-op vocabulary + branch router | **implemented** | `reconstruction/s2cadsig_op_router.py` |
| Map -> CAD-parameter decoding (heat-map -> plane, camera-ray stroke lifting) | **implemented** | `reconstruction/s2cadsig_param_decode.py` |
| Sketching-in-context incremental state machine | **implemented** | `editing/s2cadsig_session.py` |
| Per-op evaluation metrics | **implemented** | `reconstruction/s2cadsig_metrics.py` |
| Trained CNN | **research-heavy/external** | trained model |

## Batch-5 implementation result

Mined repos 21-25 (~620 new tests). Highlights: OpenCAD gave the TOPOLOGICAL
NAMING problem (face identity across rebuilds) + the numeric half of
constraint solving; ScadLM yielded a full OpenSCAD parser + kernel-free CSG
evaluator the source repo itself lacks (it shells to the openscad binary);
JoinABLe filled the joint axis/transform/motion gap (harness had only DOF
counts); OCP -- a pure binding repo -- still gave a C++ header parser + OCCT
API catalog that guards kernel-call hallucination. Per the no-README policy
the suite count is tracked in audit/cadbible_progress.json.

### 26. SketchConcept-main (NeurIPS 2022)

| Build idea | Status | Repository comparison |
|---|---|---|
| Concept template (parameterised sub-sketch, slots, interface, canonical signature) | **implemented** | `reconstruction/sketchconcept_template.py` |
| Concept library (dedup, hierarchical flattening, usage) | **implemented** | `library/sketchconcept_library.py` |
| Sketch -> concept decomposition + losslessness certificate | **implemented** | `reconstruction/sketchconcept_decompose.py` |
| DETERMINISTIC concept induction (MDL compression-gain mining) | **implemented** | `library/sketchconcept_induction.py` |
| Library compactness/coverage/compression metrics | **implemented** | `bench/sketchconcept_metrics.py` |
| Trained transformer | **research-heavy/external** | GPU model |

### 27. SketchGraphs-master (paper 165 reference impl)

| Build idea | Status | Repository comparison |
|---|---|---|
| Type-pair DOF table + autoconstrain validity mask (CORRECTNESS FIX: nominal scalar was wrong) | **implemented** | `reconstruction/sgraphs2_dof_mask.py` |
| Onshape FeatureScript entity JSON schema + parser | **implemented** | `formats/sgraphs2_onshape_json.py` |
| Entity geometry evaluation + polyline sampling (wrap-aware arc midpoint) | **implemented** | `drawings/sgraphs2_entity_render.py` |
| Autoconstrain precision/recall metrics | **implemented** | `bench/sgraphs2_autoconstrain_metrics.py` |
| Flat offset-indexed corpus container (pickle-free) | **implemented** | `formats/sgraphs2_flat_array.py` |
| Taxonomy / graph / sequence | **already in repo (paper 165)** | sketchgraphs_* |

### 28. SkexGen-main (ICML 2022)

| Build idea | Status | Repository comparison |
|---|---|---|
| SkexGen token format (6-bit truncating quantization -- differs from BOTH DeepCAD and GenCAD) | **implemented** | `reconstruction/skexgen_token_format.py` |
| 19-token extrude block (rotation MATRIX not Euler) | **implemented** | `reconstruction/skexgen_extrude_tokens.py` |
| CADparser decode + validity oracle | **implemented** | `reconstruction/skexgen_sequence_decode.py` |
| Canonical ordering by bbox + connectivity | **implemented** | `reconstruction/skexgen_canonical_order.py` |
| Branch-wise dataset dedup by token hash | **implemented** | `bench/skexgen_dedup_hash.py` |
| Disentangled code layout (topology/geometry/extrude books) | **implemented** | `reconstruction/skexgen_code_layout.py` |
| COV/MMD/JSD | **already in repo** | generative_brep_metrics |

### 29. SolidPython-master

| Build idea | Status | Repository comparison |
|---|---|---|
| Python object model + SCAD emitter (CLOSES THE OPENSCAD LOOP with scadlm parse/eval) | **implemented** | `programs/solidpy_scad_emit.py` |
| Sweep/loft a 2D section along a 3D path | **implemented** | `geometry/solidpy_extrude_along_path.py` |
| 2D polygon offset + tangent-arc filleting | **implemented** | `geometry/solidpy_path_offset.py` |
| Interpolating Catmull-Rom splines (all prior splines were approximating) | **implemented** | `geometry/solidpy_catmull_rom.py` |
| Helical screw threads | **implemented** | `geometry/solidpy_screw_thread.py` |
| Bbox algebra + planar body splitting with dowel pins | **implemented** | `geometry/solidpy_bounding_box.py` |
| Bill of materials from the object tree | **implemented** | `programs/solidpy_bom.py` |
| py_scadparser (PLY) | **already in repo** | scadlm_ast (dependency-free) |

### 30. SymPoint-main (ECCV 2024)

| Build idea | Status | Repository comparison |
|---|---|---|
| Primitive-as-point 4-anchor representation | **implemented** | `drawings/sympoint_primitive_points.py` |
| 6-D point feature + normalization | **implemented** | `drawings/sympoint_point_features.py` |
| Score-gated GT-driven panoptic eval + thing/stuff split | **implemented** | `bench/sympoint_panoptic_thing_stuff.py` |
| Point-wise semantic eval with ignore label + fwIoU | **implemented** | `bench/sympoint_pointwise_eval.py` |
| Instance-queue CutMix augmentation | **implemented** | `drawings/sympoint_instance_cutmix.py` |
| Winner-takes-all query decoding | **implemented** | `drawings/sympoint_query_grouping.py` |
| Endpoint-KNN graph / FPS | **already in repo** | cadtransformer / cadrille |

## Batch-6 implementation result

Mined repos 26-30 (~660 new tests -- the biggest batch). SolidPython CLOSED THE
OPENSCAD LOOP (emit -> parse -> evaluate offline) and added sweep/loft, 2D
offset+fillet, interpolating splines, and screw threads -- all previously
absent. SketchGraphs' reference impl exposed a CORRECTNESS FIX: constraint DOF
depends on the entity type-pair, not a nominal scalar. SkexGen's quantization
differs from both DeepCAD and GenCAD (three incompatible schemes now modeled).
SketchConcept yielded a deterministic concept-induction miner the source repo
lacks (it learns end-to-end). Per the no-README policy the suite count is
tracked in audit/cadbible_progress.json.
