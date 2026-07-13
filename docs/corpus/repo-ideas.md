# CADBible Repo Mining: Build Ideas Log

Deterministic, locally-buildable ideas mined from each repo in
`resources/cadbible/` and implemented as tested stdlib-only modules.
Repos that are paper reference-impls already covered by the 186-paper campaign
(see docs/corpus/paper-ideas.md) yield only genuinely-new implementation-level
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
CAD-Editor (ICML paper) all partly covered; extracted only genuinely-new
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

### 14. CQAsk-main (CadQuery LLM assistant): partial

| Build idea | Status | Repository comparison |
|---|---|---|
| CadQuery API reference/retrieval index | **implemented** | `generation/cqask_api_reference.py` |
| Code-gen scaffold/sanitizer | **pending** | agent hit session limit mid-write; follow-up |
| LLM | **research-heavy/external** | trained model |

### 15. CascadeStudio-master (OCC.js/WASM web CAD): partial

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

Mined repos 16-20, the richest batch yet. ComplexGen filled a major gap: 3D
surface primitive fitting (plane/sphere/cylinder/cone), which the harness
lacked entirely (only 2D line/circle/arc). CodeToCAD gave units/landmarks/
transforms/operation-schema. DeepCAD + GenCAD (already-covered papers) still
yielded implementation-level gaps from their REFERENCE CODE: DeepCAD's
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
counts); OCP (a pure binding repo) still gave a C++ header parser + OCCT
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
| SkexGen token format (6-bit truncating quantization; differs from BOTH DeepCAD and GenCAD) | **implemented** | `reconstruction/skexgen_token_format.py` |
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

Mined repos 26-30 (~660 new tests, the biggest batch). SolidPython CLOSED THE
OPENSCAD LOOP (emit -> parse -> evaluate offline) and added sweep/loft, 2D
offset+fillet, interpolating splines, and screw threads, all previously
absent. SketchGraphs' reference impl exposed a CORRECTNESS FIX: constraint DOF
depends on the entity type-pair, not a nominal scalar. SkexGen's quantization
differs from both DeepCAD and GenCAD (three incompatible schemes now modeled).
SketchConcept yielded a deterministic concept-induction miner the source repo
lacks (it learns end-to-end). Per the no-README policy the suite count is
tracked in audit/cadbible_progress.json.

## Batch 7 (repos 31-35)

### 31. Text-to-CAD-dean

A 137-line Streamlit app (prompt -> GPT-4 -> OpenSCAD -> STL -> GLB) with no
geometry code of its own. Its value is the pipeline hops it *delegates*, three
of which the harness had no stdlib coverage for.

| Build idea | Status | Repository comparison |
|---|---|---|
| Stdlib STL codec: binary+ASCII parse/write, format detection, normal recompute, volume/area/bounds | implemented `formats/t2cdean_stl_codec.py` | repo calls `trimesh.load()`; harness had only OCCT-backed STL import (`ingest/import_brep.py`) |
| Binary glTF (.glb) writer: GLB 2.0 chunk container, 4-byte alignment, vertex welding, area-weighted normals, POSITION min/max, `parse_glb` inverse, base64 data-URI | implemented `formats/t2cdean_glb_writer.py` | repo delegates to trimesh; harness had ZERO glTF/GLB code |
| OpenSCAD CLI export planner: format/extension table, 2D-vs-3D check, `-D` literal quoting, content-addressed artefact naming, cache key, stderr classifier catching exit-0 empty geometry | implemented `fabrication/t2cdean_openscad_export.py` | repo's `convert_openscad_code_to_stl` uses `uuid4` (cache-miss bug) + `check=True` (misses empty-geometry failure); harness had SCAD *language* tooling but no CLI/export layer |
| LLM-reply -> compilable SCAD: fenced-block extraction (tolerating truncation), language-tag ranking, prose stripping, strict refusal guard | implemented `programs/t2cdean_scad_extract.py` | repo relies on prompt wording alone; harness had only a private `_strip_code_fences` |
| Streamlit UI, PyVista render, model-viewer embed, OpenAI client | out-of-scope/external | UI + trained model |

### 32. Text-to-CadQuery-main

Reference impl of paper 171 (already covered by `programs/t2cq_*`). Yielded
three implementation-level pieces the paper review missed.

| Build idea | Status | Repository comparison |
|---|---|---|
| Raw-generation -> runnable script cleaning (EOS truncation, `### Response:` de-prefix, fence strip, export canonicalisation) | implemented `programs/t2cq2_output_cleaning.py` | `inference/step2_clean_run_CadQuery`; nothing in the harness did this |
| Exact eval protocol: centroid + max-bbox-extent normalisation, CD = mean(d^2)+mean(d^2) x1000, F1 @ 0.02, volumetric IoU @ pitch 0.02, judge Match-Yes gate, mean+median | implemented `bench/t2cq2_eval_protocol.py` | **differs from** `bench/cad_geometry_protocol.py`, which uses unit-sphere normalisation and has no F1/IoU/judge gate; different numbers on the same meshes, so both now coexist |
| Dataset layer: JSONL input/output schema, Instruction/Response template, 2-attempt execution-feedback retry (last-5-stderr-lines), DeepCAD UID bucketing | implemented `dataengine/t2cq2_dataset.py` | `data_annotation/gemini_pipeline.py`; no prior T2CQ record schema |
| DeepCAD-JSON -> CadQuery translation; CadQuery API subset; Invalid-Rate | already in repo (paper 171) | `reconstruction/t2cq_translate.py`, `programs/t2cq_ast.py`, `programs/t2cq_validity.py` |
| Finetuning six LLMs, Gemini annotator/judge, Blender render | research-heavy/external | trained model / GPU / renderer |

### 33. Text2CAD-main

Reference impl of papers 172/173 (already covered by `dataengine/text2cad_*`,
`reconstruction/text2cad2_*`, `bench/text2cad_sequence_f1.py`). Four new pieces.

| Build idea | Status | Repository comparison |
|---|---|---|
| Exact `CADSequence.to_vec`/`from_vec` codec: chained closed loops (line = 1 token, arc = 2, circle = centre+pt1), 11-token extrusion block, START wrapper, pad to 272, `flag_vec`/`index_vec` | implemented `reconstruction/t2c3_cad_vec_codec.py` | `reconstruction/text2cad2_sequence_tokens.py` has the ids but serialises every curve point (2N tokens); the real codec is chained (N tokens) |
| Exact eval protocol (`generate_report`): geometric bbox-L2 loop matching, curve re-matching, **Null 4th class**, per-primitive P/R/F1 from a 4x4 confusion matrix, macro/micro, extrusion count F1 + L1 parameter report rescaled by 1/0.75 | implemented `bench/t2c3_eval_protocol.py` | `bench/text2cad_sequence_f1.py` matches loops by primitive-multiset cost, has no Null class and no extrusion parameter report |
| Minimal-JSON schema (LLM-facing doc: part_N/face_N/loop_N, Euler angles in degrees, negative-zero-normalised 4-decimal rounding) + inverse parser | implemented `dataengine/t2c3_minimal_json_schema.py` | `dataengine/text2cad_minimal_metadata.py` models the transformation, never the document schema/round-trip |
| Training token accuracy (`AccuracyCalculator`: min-length truncation, target>6 mask, per-slot abs(pred-gt) < 3) | implemented `bench/t2c3_token_accuracy.py` | `bench/deepcad2_ae_accuracy.py` scores 17-column rows via `CMD_ARGS_MASK`; different stream, different masking rule |
| L0-L3 prompt taxonomy, prompt generator, minimal metadata, token id table, Invalidity Ratio, 8-bit quantisation | already in repo (papers 172/173) | Text2CAD reuses DeepCAD quantisation maps verbatim (unlike SkexGen/GenCAD) |
| Trained Transformer, BERT encoder, OCC BRep/STEP/STL export, Chamfer | research-heavy/external | GPU / OCC |

**Finding:** the Text2CAD sequence layout differs materially from the paper
Table 3: loops are closed and chained, so curve token cost is N, not 2N, and a
circle is identified structurally by being the only curve in its loop.

### 34. UV-Net-main

Reference impl of the UV-Net paper, but its core representation was genuinely
absent: the harness could describe a B-rep topology but had no way to attach
*sampled geometry* to it.

| Build idea | Status | Repository comparison |
|---|---|---|
| Face UV-grid: per-face regular sampling of the parametric domain -> 7-channel (x,y,z,nx,ny,nz,mask) tensor over plane/cylinder/cone/sphere/torus/NURBS | implemented `geometry/uvnet_uv_grid.py` | `process/solid_to_graph.py` calls occwl `uvgrid`; rebuilt closed-form and OCC-free, NURBS delegated to `geometry/nurbgen_surface.py` |
| Trimming mask / OCC TopAbs_State in the parameter plane: even-odd point-in-loop with hole support -> IN/OUT/ON | implemented `geometry/uvnet_uv_grid.py` | mirrors `visibility_status`; no OCC |
| Bridge from fitted primitives to sampleable surfaces (`surface_from_fit`) | implemented `geometry/uvnet_uv_grid.py` | not in the repo; composes `geometry/complexgen_surface_fit.py` (fit -> sample -> refit round-trips in tests) |
| Edge U-grid: per-edge sampling -> 6-channel (x,y,z,tx,ty,tz) for line/circle/ellipse/polyline/NURBS, degenerate-edge filter, reversed-coedge grid, arc-length + tangent-turning | implemented `geometry/uvnet_u_grid.py` | occwl `ugrid` + the cone-apex has_curve filter |
| Face-adjacency graph with UV-grid node features and U-grid edge features, seam self-loops, bidirectionalisation | implemented `reconstruction/uvnet_face_adjacency.py` | occwl `face_adjacency` + DGL; rebuilt without OCC/DGL. Existing `reconstruction/cadparser_brep_graph.py` uses categorical features, not sampled geometry |
| Mask-aware normalisation: bbox over only in-face grid points, one isotropic solid-level 2/max(diag) transform across faces *and* edges, invertible | implemented `geometry/uvnet_normalize.py` | `datasets/util.py` normalises per-grid; solid-level keeps the graph geometrically consistent |
| Deterministic axis-aligned quarter-turn frames: the 12 (axis, k*90deg) rotations enumerated instead of drawn at random | implemented `geometry/uvnet_normalize.py` | replaces `get_random_rotation` (scipy + random.choice) with a reproducible enumeration; integer-entry orthonormal matrices |
| CNN/GNN encoders, DGL I/O, STEP loading, SolidLetters font pipeline | research-heavy/external | torch / dgl / occwl / OCC |

### 35. WhatsInAName-main

Not topological naming (despite the name): the code+data release for *What's In
A Name? Evaluating Assembly-Part Semantic Knowledge in Language Models through
User-Provided Names in CAD Files* (JCISE 2023). Zero overlap with
`geometry/opencad_face_fingerprint.py`, which solves persistent *entity*
identity; this is the *natural-language semantics* of user-authored names.

| Build idea | Status | Repository comparison |
|---|---|---|
| Default-CAD-name detector + name normaliser (camelCase/snake/instance-suffix tokenizer, corpus dedup, 5 stratification flags) | implemented `library/wian_name_normalizer.py` | repo ships only the *filtered result*; the filter itself is absent. Rebuilt as a default-name grammar over Onshape/SolidWorks/Fusion/FreeCAD stems |
| Two-Parts benchmark generator + metrics (token-disjoint positives, global co-occurrence table, non-co-occurring resampled negatives, class balancing; accuracy, threshold sweep, tie-aware ROC-AUC) | implemented `bench/wian_partname_pairs.py` | reimplements `generate_pairs.py` minus numpy/nltk/pandas; sklearn/torch metrics rewritten in stdlib |
| Corpus templating + Missing-Part / Document-Name task builders + pure-Python stratified 3-way split (largest-remainder per stratum) + acc@k/MRR | implemented `bench/wian_partname_tasks.py` | replaces `generate_corpus.py` + `create_train_val_test_split.py` (sklearn StratifiedShuffleSplit) deterministically |
| Training-free PPMI/SPPMI part-name vector space (token co-occurrence across assembly parts, mean-pooled embeddings, cosine pair scorer, candidate ranker, TF-IDF control) | implemented `library/wian_partname_ppmi.py` | **genuinely new**: every encoder in the paper (BOW, FastText, TechNet, DistilBERT) needs training or downloaded weights. This is the closed-form count-based baseline the paper lacks, and it makes all three tasks runnable locally with no model |
| DistilBERT MLM finetuning, SetTransformer, FastText/word2vec, TechNet embeddings | research-heavy/external | trained models / GPU / downloaded weights |

## Batch-7 implementation result

Five repos, 416 new tests. Two reference-impl repos (32, 33) correctly yielded
only implementation-level gaps, but those gaps included two protocol
*contradictions* worth keeping: the Text-to-CadQuery eval normalisation differs
from the existing `bench/cad_geometry_protocol.py`, and the real Text2CAD
sequence layout is chained (N tokens/loop), not the 2N its own paper implies.
UV-Net contributed the largest genuinely-new capability (sampled-geometry B-rep
graphs); Text-to-CAD-dean, despite being a hobby app, closed three format gaps
(stdlib STL, GLB, OpenSCAD CLI export). Per the no-README policy the suite count
is tracked in audit/cadbible_progress.json.

## Batch 8 (repos 36-40)

### 36. angelcad-master

AngelCAD (C++/AngelScript CSG language). Its real contribution is that the CSG
language is *statically typed*: `solid` and `shape2d` are distinct types, so the
2D/3D mixing bug that OpenSCAD (and every LLM writing OpenSCAD) commits silently
is a compile error there.

| Build idea | Status | Repository comparison |
|---|---|---|
| Typed CSG AST + 2D/3D dimension checker (typed opAdd/opSub/opAnd/opMul, extrudes the only 2D->3D ops, projection2d the only 3D->2D one) with structured diagnostics (dim-mismatch, arity, param missing/type/range, index-range, face-degenerate, unknown-op) | implemented `programs/angelcad_typed_csg.py` | from `as_csg.cpp` register_types + shape/solid/shape2d headers. Genuinely different from the harness's untyped OpenSCAD front end (`programs/scadlm_ast.py`, `geometry/scadlm_csg_eval.py`), which cannot catch `circle(3) + cube(2)` |
| `tmatrix` value type: composable 4x4 transforms, translate/rotate/scale/mirror, `hmatrix` frame orthonormalisation, xdir/ydir/zdir/origin, exclusive deg/rad angles | implemented `programs/angelcad_typed_csg.py` | complements `geometry/codetocad_transform_stack.py` (different convention; here the matrix is a first-class DSL node) |
| XCSG XML CSG-tree interchange (read+write), transform composition collapsed at serialisation, byte-stable output, exact `dumps(loads(x)) == x` | implemented `formats/angelcad_xcsg_xml.py` | harness had mesh formats (STL/GLB) and OpenSCAD text but no typed CSG *tree* format |
| AMF (ISO/ASTM 52915) codec: unit-bearing indexed XML mesh, multi-object/multi-volume lumps, metadata, plain-XML and reproducible-ZIP flavours, lump fusion, per-lump vertex compaction | implemented `formats/angelcad_amf_codec.py` | new format (harness had STL + GLB only) |
| Polyhedron construction/validation + mass properties: Newell normals, `verify` (index range, degenerate/repeated/zero-area/non-planar faces, boundary edge, non-manifold edge, inconsistent orientation, unused vertex, inward winding), signed volume, area, centroid, inertia tensor (tet decomposition + parallel axis), flip/orient | implemented `geometry/angelcad_polyhedron.py` | nothing in the harness checked manifoldness or orientation, or computed inertia |
| Kernel-free bounding-box propagation over the CSG tree (per-operator rules: union/hull enclose, difference bounded by first operand, intersection = overlap, minkowski = box sum, extrudes lift 2D->3D, transform re-encloses 8 corners) + `fits_within` build-volume check + `is_provably_empty` | implemented `geometry/angelcad_csg_boundingbox.py` | distinct from `geometry/solidpy_bounding_box.py` (point-list boxes + print-bed splitting, no per-operator CSG rules). Gives a free pre-kernel gate |
| OpenSCAD `.csg` emission, csgfix/dxfread/polyfix, AngelView GUI, OCCT boolean engine | already in repo / external | `programs/solidpy_scad_emit.py`, `formats/dxf_contract.py`; kernel/UI/fonts |

### 37. arcs-master

`arcs` (Michael-F-Bryan): a 2D CAD framework in Rust on an ECS architecture. The
ECS/rendering layer is app plumbing; the geometry core filled five real gaps.

| Build idea | Status | Repository comparison |
|---|---|---|
| Ramer-Douglas-Peucker polyline decimation (cross-product perpendicular distance, degenerate-base fallback, first-max split for determinism, endpoint preservation) | implemented `geometry/arcs_polyline_simplify.py` | harness only *mentioned* Douglas-Peucker in a `vision/rastercad_vectorize.py` docstring; no implementation existed |
| Closest-point queries with multiplicity (ONE/MANY/INFINITE): clamped segment projection, arc radial projection, exact endpoint-tie detection, polyline variant | implemented `geometry/arcs_closest_point.py` | harness had only `geometry/joinable_joint_axis.closest_point_on_line` (unclamped, 3D infinite line, single result). Also supplies a branch-cut-safe `contains_angle` (the Rust version compares raw angles and breaks across +/-pi) |
| Sagitta/chord-tolerance arc tessellation: `theta = 2*acos(1 - tol/R)`, `N = ceil(sweep/theta)`, inverse `chord_error(N)`, degenerate guards | implemented `geometry/arcs_chord_tolerance.py` | harness had only fixed-count sampling (`geometry/gencad2_arc_vector.sample_arc_points(n=32)`); no tolerance-driven segment count |
| Viewport model: drawing-space <-> canvas-space affine (y-flip, pixels-per-unit, centred window), invert/compose, pixel-vs-drawing `Dimension`, zoom/pan, visible bounds, zoom-to-fit | implemented `drawings/arcs_viewport_transform.py` | no world<->screen viewport existed (`vision/cvcad_pixel_calibration` is a camera scale estimator; `drawings/cad2program_canvas_layout` lays out sheets) |
| AABB quadtree spatial index: insert/modify/remove, query_point(p, radius), query_region, straddling items held at the parent, world auto-grow | implemented `geometry/arcs_quadtree_space.py` | harness had no spatial index, only `ingest/spatial_order.morton2` (z-order key, no queries) |
| Arc-from-three-points / circumcentre / orientation predicate; exact arc bbox; affine transforms | already in repo | `bench/mrcad_metrics`, `geometry/euclid_validity.orient`, `geometry/gencad2_arc_vector.arc_bbox` |
| ECS storage, flagged-storage change events, piet canvas rendering | out-of-scope | specs/piet plumbing |

**Upstream bug not inherited:** `BoundingBox::intersects_with` in the Rust source
is an acknowledged stub (`// FIXME`) that delegates to `fully_contains`, so
overlapping-but-not-contained boxes are silently missed. The port implements the
real separating-axis test and pins it with a regression test.

### 38. cadgenbench-main

A CAD-generation benchmark. Seven modules, each built because its metric
*definition differs* from an existing harness metric rather than duplicating it.

| Build idea | Status | Repository comparison |
|---|---|---|
| Mesh-derived solid Betti numbers (union-find components + even-odd ray-cast parity + `b1 = b0 + b2 - chi/2`) and topology score `exp(-alpha*abs(log((c+1)/(g+1))))`, alpha=2, aggregated as a **product** | implemented `bench/cgb_mesh_betti.py` | `bench/topodiff_topology_consistency.py` computes Betti from *voxels* and scores *exact-match indicators*; `bench/evocad_topology_metrics.py` only compares Euler chi. **Differs**: graded log-ratio (2 vs 4 holes -> 0.36, not 0.60) and product aggregation, so one wrong invariant collapses the axis |
| Interface match: keep-out/keep-in sub-volume contract, bounded pose search (deterministic Halton grid, saturation early-exit), IoU pass/fail ramp (>=0.95 -> 1.0, <=0.80 -> 0.0, linear), group = **min** of features, sample = **mean** over groups | implemented `bench/cgb_interface_match.py` | nothing modelled keep-in/keep-out regions or min-then-mean aggregation; `bench/solid_iou.py` is a raw best-IoU over axis alignments |
| CAD Score composition: validity as a **hard gate**, 0.4/0.4/0.2 generation vs 0.6/0.3/0.1 editing weights, missing axes **renormalize** the weights, editing shape renormalized against the no-op baseline `max(0,(s-b)/(1-b))` | implemented `bench/cgb_cad_score.py` | `bench/criteria.py`, `muse_scorecard.py`, `t2cadbench_scorecard.py` do weighted aggregation but none gates on validity, renormalizes over absent axes, or anchors an edit metric to the no-op |
| Run aggregation + leaderboard: valid/invalid/**missing** trichotomy, zeros included in the aggregate (the anti-gaming rule), validity_rate, open per-task-type buckets, deterministic ranking | implemented `bench/cgb_run_summary.py` | `bench/runner.py`, `graphcad_cadbench_report.py` aggregate but none models "missing candidate" as distinct from "invalid" nor forces non-scorable samples to zero |
| Validity gate + advisory tier: BREP/watertight/manifold gate; advisory-only flags (min face area, face aspect, BREP tolerance) that flag fragile geometry but **never** move the score | implemented `verifiers/cgb_validity_gate.py` | `verifiers/geometry.py:BRepValidityCheck` reports validity as a diagnostic (no gate, no advisory tier) |
| Submission contract: fixed candidate names, empty file != candidate, folders with no candidate preserved as `missing`, meta.json required keys, `agree_to_publish` defaults false and blocks acceptance | implemented `dataengine/cgb_submission_package.py` | no submission/packaging contract existed |
| Canonical-pose contract: bbox centre at origin, extents ordered Lx>=Ly>=Lz, reference face on z=-Lz/2, canonicalization via a **proper** rotation (det +1, never a mirror), symmetry-ambiguity flag | implemented `quality/cgb_canonical_pose.py` | `bench/cadrille_orientation_align.py` brute-forces 24 rotations against a *target* by Chamfer; no target-free canonical frame or ambiguity flag |
| Open3D ICP alignment, manifold3d IoU, renders, LLM baseline agent | research-heavy/external | kernel / ICP / renderer / LLM. The deterministic scaffolding around them is built above |

### 39. cadhub-main

CadHub (RedwoodJS code-CAD sharing platform). The web app is out of scope; the
mineable core is its multi-language code-CAD abstraction layer.

| Build idea | Status | Repository comparison |
|---|---|---|
| Code-CAD language capability matrix / adapter registry (extension, entry file, execution model, artifact kinds, mesh export, parameter flavour, diagnostic dialect; queries + gap explanation) | implemented `adapters/cadhub_language_registry.py` | CadHub encodes this implicitly across `cadPackages/index.ts` + 4 controllers + 4 Dockerfiles; harness had `adapters/base.py` (live-CAD-app protocol) but no per-language code-CAD matrix |
| Multi-language structured diagnostic parser (OpenSCAD / CadQuery traceback / JSCAD JS stack / curv) -> `Diagnostic(severity, file, line, column, message)` + sandbox-path scrubbing + caret source annotation | implemented `programs/cadhub_diagnostics.py` | CadHub only regex-strips the tmp path; the message stays opaque. Harness had OpenSCAD-only *line-bucket* classification (`fabrication/t2cdean_openscad_export.classify_result`); no line/column extraction, no other dialects |
| Unified cross-language customizer parameter schema (OpenSCAD manifest / JSCAD getParameterDefinitions / CadQuery --getparams -> one neutral model) + value validation (clamp/truncate/enum-reset with issue reports) + params.json writer | implemented `programs/cadhub_param_schema.py` | harness had `programs/cadam_scad_customizer.py` (OpenSCAD *source* annotations only); no manifest ingestion, no neutral model, no value validation |
| Render-request canonicalisation + cache key: camera quantisation to 1 dp so orbit jitter hits the same cache entry, DPR viewport sizing, per-language settings whitelist, sha256 key | implemented `backends/cadhub_render_request.py` | harness had only a CLI-export plan key (source+format+defines), no preview/camera/viewport request model |
| Single-blob artifact+metadata envelope codec (gzipped, sentinel-delimited) with lenient decode | implemented `backends/cadhub_concat_payload.py` | ports `runScad.ts`. **Two hardenings**: split on the LAST sentinel (binary STL can contain it by chance) and `mtime=0` gzip so blobs are content-addressable |
| Docker/lambda runners, Prisma schema, React IDE, auth | out-of-scope | web-app/deployment surface |

### 40. cadquery-contrib-master

A corpus of 21 real community CadQuery programs. Valuable not as a library but as
*evidence of how the API is actually used*.

| Build idea | Status | Repository comparison |
|---|---|---|
| Deterministic CadQuery API-usage profiler (ast-based: method counts, positional-arity histograms, kwargs, chain shapes, selector literals) + diff against a declared method table -> unknown-method and arity-violation reports | implemented `programs/cqcontrib_api_profile.py` | harness had no API-surface profiler; `programs/t2cq_ast.py` only hard-codes `CHAIN_METHODS` |
| CadQuery string-selector DSL: tokenizer + recursive-descent parser + evaluator over face/edge sets (`>Z`, `<Y`, `>Z[1]`, `\|Z`, `#Z`, `+Z`, `%CIRCLE`, not/and/or/exc, parens) | implemented `geometry/cqcontrib_selector_dsl.py` | selectors appear 100+ times across the examples; no selector parser existed anywhere in the harness |
| Hole feature geometry: hole/cboreHole/cskHole axial profiles, countersink-depth closed form, exact solid-of-revolution volumes, wall break-through predicate | implemented `geometry/cqcontrib_hole_features.py` | `library/parts.py` mentions a bearing counterbore pocket as a recipe; no general cbore/csk profile geometry existed |
| Parametric enclosure/lid recipe as pure geometry: fillet-ordering rule (larger radius first), validity predicates, inner shell dims, screw-post centres, lid split z + lip footprint, exact rounded-box volumes | implemented `geometry/cqcontrib_enclosure.py` | distils Parametric_Enclosure.py + Remote_Enclosure.py; no enclosure/lid derivation module existed |
| Loft/sweep between profiles; 2D offset+fillet; polar/rect arrays | already in repo | `geometry/solidpy_extrude_along_path.py`, `geometry/solidpy_path_offset.py`, `procedural/autocad_array.py` |
| Assembly constraint solving, DXF export, makeRuledSurface, parametricCurve threads | external | needs the real CadQuery/OCCT runtime |

**Correctness finding against `programs/t2cq_ast.py`** (profiling all 21 programs,
112 distinct methods):
- **Arity bug, fixed:** `moveTo` was declared `(2, 2)`, but CadQuery's signature is
  `moveTo(x=0, y=0)` and the corpus calls it with one argument, so `validate()`
  was rejecting valid programs. Widened to `(0, 2)` with regression tests.
- **Unknown methods** (corpus counts): `val`(31), `constrain`(23), `add`(20),
  `transformed`(14), `parametricCurve`(9), `tag`(9), `rarray`(8), `end`(8),
  `section`(6), `eachpoint`(5), `shell`(5), `toPending`(5), `vals`(5),
  `cboreHole`(4), `polarArray`(4), `split`(4), `makeRuledSurface`(4),
  `located`(4), `makePlane`(4), `exportDXF`(5). `shell`, `loft`, `cboreHole`,
  `cskHole`, `polarArray`, `rarray`, `split` are core fluent modelling methods the
  harness AST does not know. NOT silently widened: `t2cq_ast` is scoped to the
  *paper's* CadQuery subset by design, and expanding its notion of "valid" would
  change what the paper-comparison numbers mean. Left as an open scope decision.

## Batch-8 implementation result

Five repos, 515 new tests. The batch's best idea is angelcad's **typed CSG**: a
dimension checker that catches `circle(3) + cube(2)` with zero geometry, which,
combined with kernel-free bbox propagation, gives a cheap pre-kernel verifier
tier (provably-empty intersections, does-not-fit-the-printer) the harness lacked.
Three repos that looked out-of-scope (a Rust ECS app, a Redwood web app, a folder
of examples) each hid a real gap: no spatial index / no RDP / no viewport; no
multi-dialect diagnostics; no selector DSL and no API-usage evidence.

Separately, a latent suite bug surfaced during this batch's recount: seven
early-campaign test files were written as bare pytest functions, which
`python -m unittest` never collects, so 26 assertions had never executed. All 26
pass on first real execution (no module bug was hiding); 93 tests now run where 0
did, and `tests/test_suite_collectable.py` fails loudly if a non-collectable test
file is added again. Suite: 12,782 tests, zero failures.

## Batch 9 (repos 41-45)

This batch was interrupted once by a session limit and resumed after reset; the
partial-then-clean recovery is documented in the batch-8 close note. Two of the
corpus's heaviest repos land here (CadQuery itself and Curv).

### 41. cadquery-master

The CadQuery library. Geometry delegates to OCCT (external); the pure-Python
selector/plane/state/assembly/export layers were mined. Profiling the real
selector grammar against the harness's own DSL surfaced seven divergences.

| Build idea | Status | Repository comparison |
|---|---|---|
| Programmatic Selector object algebra (composable `&`/`+`/`-` operators; NearestToPoint, BoxSelector with XOR containment + bbox mode, Direction/Parallel/Perpendicular, TypeSelector, Nth-with-tolerance-clustering: Radius/Length/Area/CenterNth, DirectionMinMax, DirectionNth) | implemented `geometry/cq_selector_algebra.py` | harness had `geometry/cqcontrib_selector_dsl.py` (string parser only); the object algebra it lacked, with identity-based order-preserving set combiners |
| Grammar-faithful selector-string compiler targeting the object algebra | implemented `geometry/cq_selector_grammar.py` | fixes 7 documented divergences of `cqcontrib_selector_dsl` from the real grammar (see finding) |
| CadQuery `Plane` named-preset frame algebra (12 presets, toWorld/toLocalCoords, rotated via Rodrigues, setOrigin2d, tolerant equality) | implemented `geometry/cq_plane_frame.py` | `geometry/codetocad_transform_stack.py` is a generic 4x4 lib; new is the CAD-specific named-plane preset table + world/local convention |
| Workplane pending-model state-transition validator (edges->wires->solids; catches extrude-with-no-wire, unfused-edges, close-empty-path, loft<2, boolean-no-base, cross-plane combine) + ast chain extractor | implemented `programs/cq_workplane_state.py` | `programs/t2cq_ast.py` does arity/varref only; new is the semantic pending-wire/edge state layer, a validation advance beyond arity |
| Assembly 6-DOF constraint algebra + Grubler mobility well-posedness (unary/binary kinds, DOF-removed table, under/well/over classification, redundant-constraint + no-anchor detection) | implemented `numeric/cq_assembly_dof.py` | `reconstruction/sgraphs2_dof_mask.py` / `opencad_constraint_jacobian` are 2D sketch DOF; new is 3D rigid-body assembly DOF |
| Orthographic-projection SVG exporter (camera-basis projection, M/L path emission, exact getSVG bbox-fit unitScale/translate, visible/hidden styling) | implemented `formats/cq_svg_projector.py` | harness had AMF/DXF/STL/GLB but no SVG exporter; the OCCT HLR pass is external, the projection+fit+emission algebra is reproduced |
| OCCT booleans/fillets/lofts, scipy constraint numerics | external | kernel / solver |

**Correctness finding (7 divergences of `cqcontrib_selector_dsl` from CadQuery's
real `_makeGrammar`), encoded in `GRAMMAR_FINDINGS`:**
1. **`not` precedence inverted (real bug):** CadQuery lists `not` last in its
   `infix_notation`, making it the *loosest* operator, so `not >X and #XY` means
   `not(>X and #XY)`. The DSL parses `not` at tightest binding, giving
   `(not >X) and #XY`. The result sets differ. The new compiler reproduces the
   correct loosest-`not` precedence.
2. Center-Nth `>>`/`<<` missing (the DSL tokenizes only single `>`/`<`).
3. Named views `front/back/left/right/top/bottom` missing.
4. Bare direction (`X`, `(1,0,0)`) rejected though valid upstream.
5. Compound axes `XY`/`XZ`/`YZ` rejected.
6. `except` spelling unsupported (CadQuery accepts both `exc` and `except`).
7. Phantom binary `+`/`-`/`*` advertised in the DSL docstring but absent from the
   real grammar and unimplemented in the DSL itself.

### 42. cadquery-plugins-main

Community plugins. Two modules committed in the first (interrupted) pass
(teardrop profile, heat-set bore schedule); two more after resume.

| Build idea | Status | Repository comparison |
|---|---|---|
| Overhang-safe teardrop hole profile (self-supporting, avoids >45deg unsupported arcs) | implemented `geometry/cqplug_teardrop_profile.py` | printability-aware variant of a plain hole; no equivalent existed |
| Heat-set insert bore schedule keyed to screw designation | implemented `standards/cqplug_heatsert_schedule.py` | `library/parts.py` had only a bearing-pocket recipe |
| Volumetric region selectors (infinite/finite cylinder, hollow cylinder, sphere, hollow sphere): keep shapes whose centre lies inside a solid via axis-projection `(h, rho)` test | implemented `geometry/cqplug_region_selectors.py` | `cqcontrib_selector_dsl` / `cascade_entity_selector` do direction/size intent; neither has a point-in-solid test. This is the `more_selectors` addition; re-mined correctly after the prior partial failed 5 tests |
| Bevel-gear cone geometry + spherical involute + meshing-pair pitch cones (delta_b, face/root cones, delta_p1 from shaft angle, R from module) | implemented `geometry/cqplug_bevel_gear.py` | `geometry/cadgpt_involute_gear.py` is planar spur only; `cadgpt_gear_train.bevel_scale` is a linear-extrude taper approximation. The plugin's planar Gear class duplicates cadgpt_involute_gear -> skipped |
| localselectors (named-view remap, centre-nth) | skipped (low-value) | thin remap over the existing DSL; centre-nth duplicates DirMinMax index grouping |
| apply_to_each_face, fragment, freecad_import, cq_cache | external | OCCT-runtime iteration/boolean/import or a file cache |

### 43. comet-main

CoMeT (Cognitive Memory Tree): a lossless structured-memory substrate for LLM
agents, deployed behind an autonomous-CAD stack, in-domain for the harness's
own memory layer. Two modules committed pre-interruption, three after resume.

| Build idea | Status | Repository comparison |
|---|---|---|
| Dedup/merge/tag-normalise/prune consolidation sweep | implemented `dataengine/comet_memory_consolidation.py` | new memory maintenance pass |
| Dual-channel WHAT/WHEN fusion + graph 2-hop expansion | implemented `library/comet_dual_channel_fusion.py` | new retrieval fusion |
| Token-budgeted progressive-tier reading (summary->detail->raw, pinned-first admission, risk-escalated deepening) | implemented `context/comet_progressive_tiers.py` | `context/manager.py` has count-based token budgeting but no per-node tier selection. Re-mined with tests after the prior no-test partial |
| MemoryBank reinforced-decay salience (retention R = exp(-dt/(S*tau)), recall reinforcement, curve inversion, threshold sweep), clockless (day numbers as inputs) | implemented `memory/comet_reinforced_decay.py` | CoMeT's MemoryNode doc-strings state the formula but ship no arithmetic ("folded in by the dream pass"); `memory/store.py` has no forgetting curve |
| Terse meta-tag rendering: one-tag-per-axis `(O: A: F: I:)` + consecutive same-origin bundling | implemented `context/comet_meta_tag_render.py` | pure-function extraction of orchestrator render helpers |
| SLM sensor, QueryAnalyzer, compacter LLM, LanceDB index, MCP server | external | trained model / GPU / vector kernel |

### 44. curated-code-cad-main

A curated awesome-list of code-CAD systems + a birdhouse reference-part set.
Documentation, not code: correctly yields a knowledge base, not algorithms.

| Build idea | Status | Repository comparison |
|---|---|---|
| Catalogue of code-CAD systems (name/language/paradigm/kernel/formats/online-editor/maturity/niche) + representation taxonomy + B-rep recommendation + birdhouse set; queries; unstated attributes left UNKNOWN not fabricated | implemented `adapters/ccc_codecad_ecosystem.py` | complements `adapters/cadhub_language_registry.py` (which executes one language: extension/entry/diagnostics); this catalogues/selects a system |
| Deterministic backend selection (explainable rubric) + harness coverage/gap report over the catalogue | implemented `adapters/ccc_backend_selector.py` | selection over the catalogue; no equivalent |
| System-to-system interoperability matrix: format-handoff edges + list-stated transpile/embed bridges, fidelity-ranked interchange format, BFS handoff paths, reachability closure, interop-hub ranking, dense matrix | implemented `adapters/ccc_interop_matrix.py` | turns prose the list only states in English ("FreeCAD best at interoperability", transpilers emit .scad, AngelCAD runs OpenSCAD) into a queryable graph; reuses the ecosystem data, no catalogue duplication |
| Birdhouse per-tool reference sources | not buildable | source code only; captured as a data list |

### 45. curv-master

Curv (Doug Moen): a functional language representing shapes as signed distance
fields. The harness had SDF *consumers* (marching tets, TSDF, FD gradients) but
no way to *author* an SDF, this batch's biggest single capability.

| Build idea | Status | Repository comparison |
|---|---|---|
| Exact SDF primitives (sphere, box exact+mitred, rounded box, cylinder, cone, capped cone, capsule, torus, ellipsoid bound, plane; 2D circle/rect/regular-polygon/half-plane + extrude/revolve lifts) | implemented `geometry/curv_sdf_primitives.py` | no SDF primitive algebra existed (only FD derivatives / TSDF / marching-tets). Formulas from `lib/curv/std.curv` |
| SDF combinators + smooth blends (hard min/max/diff/complement N-ary; polynomial/exponential/power smooth-min; smooth union/intersection/difference; chamfer) | implemented `geometry/curv_sdf_combinators.py` | no combinators/smooth-min existed; tests verify smin->min as k->0 and smooth_union <= hard union |
| Distance-field & domain transforms (offset, shell, round, morph; translate/rotate/scale with `*s` compensation/stretch/mirror; infinite & finite repetition with cell clamp) | implemented `geometry/curv_sdf_transforms.py` | no offset/shell/domain-repetition; scale compensation preserves the Eikonal property |
| TPMS (gyroid, Schwarz-P, Schwarz-D, Neovius), raw implicit + Lipschitz-normalised | implemented `geometry/curv_sdf_tpms.py` | no gyroid/TPMS anywhere. **Correctness finding:** Curv's docs recommend dividing the gyroid field by 4/3 to make it 1-Lipschitz, but the true sup\|grad F\| is sqrt(3) ~ 1.732 (and 7 for Neovius); 4/3 does not bound the gradient, so sphere-tracing can overstep. The module divides by the measured bound and documents it |
| Sphere-tracing / raymarching (Lipschitz-safe stepping, hit/miss) + central-difference normals | implemented `numeric/curv_sphere_trace.py` | no sphere tracer; Hart-1996 stepping with Curv `lipschitz k` compensation |
| C++/GLSL GPU compiler, OpenGL viewer, GLSL codegen | external | non-transferable host/GPU tooling |
| FD gradient/Hessian | already in repo | `numeric/flatcad_sdf_derivatives.py` (composed, not rebuilt) |

## Batch-9 implementation result

Five repos, ~488 new tests. This batch turned the harness from an SDF *consumer*
into an SDF *author* (Curv's full primitive/combinator/transform/TPMS/tracer
algebra), added the programmatic CadQuery selector algebra and a semantic
Workplane state validator (beyond the prior arity-only checks), a 3D assembly DOF
analyser, an SVG exporter, and folded CoMeT's memory-substrate ideas into the
harness's own memory/context layer. Two more reference-material corrections
surfaced: seven grammar divergences in the harness's own selector DSL (including
an inverted-`not`-precedence bug), and Curv's under-stated gyroid Lipschitz bound.
The batch was interrupted mid-flight by a session limit and cleanly resumed;
per the no-README policy the suite count is tracked in
audit/cadbible_progress.json.

## Batch 10 (repos 46-50)

Two FreeCAD integrations, both halves of the Gaudi product, and a covered-paper
reference impl. The frontend correctly yielded nothing; the rest paid out.

### 46. freecad-ai-master

An AI-assistant workbench for FreeCAD (LLM agent loop + Qt panels + MCP glue,
all external). The deterministic core is the ~53-operation tool catalogue, the
FreeCAD expression language, and a relative-edit mini-language.

| Build idea | Status | Repository comparison |
|---|---|---|
| FreeCAD workbench operation catalogue: 53 operations (16 PartDesign, 5 Part, 3 Assembly, 3 Spreadsheet, 2 Sketcher, 1 Draft, + inspection/document/composite), each with workbench + category + typed param schema; `check_call` validates op name / required params / enum domains with difflib near-miss repair; JSON-schema emit | implemented `adapters/fcai_freecad_tool_catalog.py` | distinct from `backends/ocp_occt_api_catalog.py` (OCCT kernel C++ classes) and `generation/query2cad_macro.py` (Part primitives+booleans only); no existing module covers the PartDesign parametric feature-tree taxonomy |
| FreeCAD parametric expression engine: recursive-descent parser + evaluator for dotted property refs (`Variables.height`, `Box.Placement.Base.x`), `[index]` subscripts, arithmetic (right-assoc power), unit literals (mm/cm/m/in/ft/deg/rad -> base units), functions; reports the dependency reference set for recompute ordering; no `eval` | implemented `programs/fcai_expression_engine.py` | `programs/scadlm_ast.py` and `t2cq_ast.py` model OpenSCAD/CadQuery; neither covers FreeCAD's spreadsheet/property expression grammar. Genuinely new |
| Relative property-edit resolver: `modify_property`'s mini-language (`+10%`, `-20%`, `*1.5`, `+5`, absolute) as a pure function returning structured Resolution (kind/previous/resolved/delta) with optional clamp | implemented `library/fcai_relative_value.py` | new small deterministic utility |
| Tool registry mechanics, OpenAI/Anthropic/MCP schema emitters | already in repo | `agent/`, `surfaces/mcp/`, `agent/toolcad_tool_schema.py`; only the FreeCAD-specific content was extracted |
| LLM agent loop (20 providers), Qt GUI, live FreeCAD calls | external | trained model / UI / host |

### 47. freecad_mcp-main

A thin MCP server (~350 LOC) exposing FreeCAD via two `exec` escape-hatch tools.
Generic MCP plumbing the harness already owns; one FreeCAD-specific artifact.

| Build idea | Status | Repository comparison |
|---|---|---|
| FreeCAD document-object-model wire codec: DocumentContext/FreeCADObject/Placement/Rotation/ShapeInfo/DocumentInfo/ViewState dataclasses + encode/decode reproducing the exact `get_document_context()` JSON (TypeId, Name-vs-Label, ViewObject.Visibility, Placement as position + axis-angle, Coin3D camera quaternion), round-trip stable; `parse_type_id`; `validate_context` | implemented `formats/fcmcp_document_model.py` | new. `surfaces/mcp/` encodes CISP ops + a generic `cad://model/tree`, not a FreeCAD object tree; `ocp_occt_api_catalog` catalogues API symbols, not a document instance |
| Placement axis-angle maths (axis_angle<->quaternion in FreeCAD/Coin3D (x,y,z,w) order) | implemented `formats/fcmcp_document_model.py` | new; FreeCAD stores rotations as axis+angle, cameras as quaternions |
| Live FreeCAD socket RPC, FastMCP transport, the two exec tools | external / covered | harness has a full MCP server; arbitrary exec is not a deterministic op catalogue |

### 48. gaudi-backend-main

A text-to-*architecture* backend: a building is an ordered stack of extruded 2D
plates (vertex / parametric / mixed profiles). Web/LLM/Blender layers external.

| Build idea | Status | Repository comparison |
|---|---|---|
| Plate-stack building DSL schema + one-pass validator (per-category required/optional keys, thickness>0, formula x&y, range/steps, rotation, position; building-wide name uniqueness; `normalize_plate` fills defaults) | implemented `spec/gaudi_plate_spec.py` | new. Gaudi feeds raw dicts straight into bpy with no validation; harness has sketch-extrude/command IRs but no stacked-extruded-profile architectural plate IR |
| Safe parametric-curve profile sampler without `eval`: ast-whitelist evaluator with free var `t`, math funcs, constants; half-open `sample_curve`; profile hygiene (signed area, ensure_ccw, dedupe, is_degenerate) | implemented `geometry/gaudi_parametric_profile.py` | **replaces the upstream unsafe `eval`**. `numeric/opencad_param_expression.py` is a named-parameter-table evaluator (no free variable, no curve sampling) |
| Deterministic building assembly: per-category outline resolution (incl. closed Catmull-Rom for `mixed`), AABB xy-centering, rotate_z, z auto-stacking by cumulative thickness, Building/PlacedPlate rings | implemented `generation/gaudi_building_assembly.py` | new orchestration. `geometry/solidpy_extrude_along_path.py` covers generic prism meshing, so no extruder was rebuilt, only the plate-stack placement logic |
| Generic polygon->prism extrusion; LLM prompt/repair loops; Flask/JWT/Docker/bpy | already in repo / external | `geometry/solidpy_extrude_along_path.py`, `generation/*` + `reliability/*repair*`; web/render external |

### 49. gaudi-frontend-main

| Build idea | Status | Repository comparison |
|---|---|---|
| (none) | out-of-scope, nothing built | A Create-React-App chat UI (React 18 + three.js GLB viewer + GLSL blob) for the Gaudi product; all CAD logic is server-side behind `API_BASE_URL`. No client-side CAD schema, constraint system, deterministic geometry algorithm, or DSL; the only math is stock three.js camera/OrbitControls/Box3 auto-fit already covered by `drawings/arcs_viewport_transform.py`. Building anything here would be gold-plating |

### 50. hnc-cad-main

Reference impl of HNC-CAD (ICML 2023, paper 105). Its data pipeline IS SkexGen's,
so canonical ordering / S-P-L tree / 6-bit quant / VQ assignment were already
mined. Two implementation-level findings remained.

| Build idea | Status | Repository comparison |
|---|---|---|
| 25-frame discrete extrude-orientation codebook: clip the plane's three axis vectors to {-1,0,1}, concatenate to length-9, require exact match against 25 canonical patterns -> categorical index in [0,25); doubles as an in-distribution validator | implemented `reconstruction/hnc_rotation_codebook.py` | **representation difference:** the DeepCAD family (`deepcad_sketch_plane`, `deepcad_command_spec`) stores continuous ZYZ angles; SkexGen stores 9 independently-rounded matrix components. HNC collapses orientation to one of 25 categories. The 25 frames are NOT the 24 proper rotations and are not orthonormal (frame 0's x-axis is `(-1,-1,0)`); an empirical clipped set |
| Flat CAD command/param codec: 6-int vocab with end-tokens promoted to command types (SKETCH_END/FACE_END/LOOP_END/LINE/ARC/CIRCLE), implicit-endpoint start-only curves (arc = start+mid, no end; circle = 4 cardinal points), 8-slot params, 11-slot extrude packing, two normalizations (loop-level half-diagonal vs sequence half-extent) | implemented `reconstruction/hnc_cad_vec_codec.py` | distinct vocab/width/normalisation vs `deepcad_command_spec` (16-slot, SOL/EOS pair) and `hnc_spl_tree` (hierarchical). **Code/comment mismatch found:** the `quantize` docstring says `n_bits**2 - 1` but the code uses `2**n_bits - 1`; reproduced the code (authoritative) |
| S-P-L tree, canonical ordering, 6-bit quant, sha256 dedup, VQ nearest-code | already in repo | `reconstruction/hnc_spl_tree.py`, `skexgen_canonical_order.py`, `hnc_code_assignment.py`, `deepcad2_arc_macro.py` |
| VQ-VAE + cascaded autoregressive transformers | research-heavy/external | trained models |

## Batch-10 implementation result

Five repos, ~193 new tests (46: 79, 47: 19, 48: 72, 49: 0, 50: 22). Two FreeCAD
integrations added a FreeCAD-specific layer the harness lacked: a PartDesign
feature-operation catalogue, an eval-free parametric expression engine (dotted
refs, unit literals, dependency tracking), and a document-object wire codec.
Gaudi contributed a text-to-architecture plate-stack DSL and replaced
an upstream unsafe `eval` in its parametric-curve sampler with an AST-whitelisted
one. The frontend was correctly out-of-scope. hnc-cad produced the campaign's
fifth reference-representation finding (a 25-frame orientation codebook unlike the
whole DeepCAD family) plus another code/comment mismatch resolved in favour of the
code. Two gaudi-backend test files were placed in package-local tests/ dirs by the
agent and relocated to the top-level tests/ so the canonical runner collects them.
Per the no-README policy the suite count is tracked in audit/cadbible_progress.json.

## Batch 11 (repos 51-55)

Two geometry kernels (libfive f-rep, manifold mesh-boolean) plus two covered
benchmarks and the OpenCASCADE mirror. The strongest batch for foundational
geometry: the harness gained interval arithmetic, exact predicates, autodiff,
dual contouring, a half-edge mesh, a 3D BVH, and numerical quadrature.

### 51. libfive-master

Matt Keeter's f-rep kernel. Complements Curv: Curv gave fixed Python SDF
functions; libfive gives the reified opcode GRAPH underneath, plus the three ways
to evaluate it (point, interval, dual).

| Build idea | Status | Repository comparison |
|---|---|---|
| f-rep opcode/Tree IR: hash-consed DAG with CSE, constant folding, commutative canonicalisation, evaluator, infix + S-expr printers, small shape stdlib | implemented `geometry/libfive_frep_ir.py` | ports libfive `tree/` + `opcode.cpp`. Distinct from `curv_sdf_*` (fixed callables); this is a reified, introspectable, optimisable graph |
| Interval arithmetic + interval evaluator over the IR for octree pruning (EMPTY/FILLED/AMBIGUOUS) | implemented `numeric/libfive_interval.py` | ports `eval/interval.hpp`; div-straddling-zero conservatism, sqrt/log NaN domains, trig extrema enclosure. **No interval arithmetic existed in the harness**, the basis of efficient implicit meshing |
| Forward-mode automatic differentiation (dual numbers) over the IR for exact gradients/normals | implemented `numeric/libfive_forward_ad.py` | exact vs `numeric/flatcad_sdf_derivatives.py` (finite differences); composes with dual-contouring Hermite data |
| Dual contouring (quadtree) with interval pruning, Hermite data, QEF sharp-feature vertex placement + N-D QEF solver (Jacobi eigendecomposition + rank-aware truncated pseudoinverse, mass-point bias) | implemented `geometry/libfive_dual_contour.py` | sharper-feature alternative to `geometry/meshdiff_marching_tets.py`; QEF/dual-contouring absent from the harness. Tests verify a square keeps its corners and a circle reconstructs as a clean loop |
| C++/OpenGL/Guile-Scheme runtime, GPU | external | reimplemented in pure stdlib |

### 52. manifold-master

Emmett Lalish's guaranteed-manifold mesh-boolean kernel. The harness had SDF
booleans but no mesh-boolean substrate; this fills it (short of the full
retriangulating boolean, which needs the GPU runtime).

| Build idea | Status | Repository comparison |
|---|---|---|
| Adaptive exact-sign predicates orient2d/orient3d/incircle/insphere (float fast path + exact Fraction fallback) | implemented `numeric/manifold_predicates.py` | Shewchuk-style; harness had only inexact 2D orient in `geometry/euclid_validity.py`. Foundational for robust geometry |
| Half-edge triangle mesh + manifoldness invariants (Manifold's 3t+i indexing, next/prev, pair involution, face/vertex circulation, is_manifold/is_2manifold, Euler/genus, boundary loops) | implemented `geometry/manifold_halfedge.py` | `geometry/angelcad_polyhedron.py` had only an undirected edge-use table on polygon faces; no half-edge, no circulation |
| 3D BVH broad-phase (Morton SpreadBits3 + median split + stack-DFS overlap query, self_collisions, query_point) | implemented `geometry/manifold_bvh.py` | harness had only 2D quadtree, fixed-grid octree, 2D morton2; no 3D BVH over boxes. Verified vs brute force |
| Predicate-driven triangle-triangle intersection (Moller coplanar-robust test, intersection segment, segment-plane / segment-segment-2D) | implemented `geometry/manifold_tritri.py` | harness had SDF booleans only, no tri-tri |
| Winding-number point-in-mesh inside test (Van Oosterom-Strackee solid angle, generalised winding) + Kahan mass properties (signed volume, surface area) | implemented `geometry/manifold_winding.py` | degeneracy-robust vs `bench/cgb_mesh_betti.py` ray-cast parity |
| Full retriangulating boolean, CUDA/thrust runtime | external / too large | the exact-predicate substrate is built; the full boolean assembly is out of stdlib scope |

### 53. mrCAD-main

Reference impl of *Multimodal Refinement of CADs* (2025). The refinement schema,
transition function, and metrics were already built from the paper; three
deterministic-geometry gaps remained.

| Build idea | Status | Repository comparison |
|---|---|---|
| Exact curve geometric-relation suite (parallel, perpendicular, overlap-gated parallel_distance, meeting_ends, concentric) + analytic point-to-curve distance (segment / circle / arc-sector via bearing-ordering) keyed to `editing.mrcad_schema.Curve` | implemented `geometry/mrcad2_curve_relations.py` | mrCAD attaches these as methods on Line/Arc/Circle in `design.py`; the harness `Curve` (kind+points) carried none. `geometry/arcs_closest_point.py` uses a different (centre/radius/sweep) parametrization |
| The paper's exact `Design.design_distance` (point-to-CURVE, mean-normalised to [0,1], empty->1.0) | implemented `bench/mrcad2_design_distance.py` | **diverges** from `bench/mrcad_metrics.chamfer_asymmetric`: point-to-curve (analytic) vs point-to-sampled-point; mean-normalised [0,1] vs summed; empty->1.0 vs 0.0. The exact form is always <= the sampled form (tested) |
| Degenerate-curve resolution for edits (line-collapse drop, arc->circle when endpoints coincide, arc drop when adjacent points coincide, circle drop) | implemented `editing/mrcad2_edit_degeneracy.py` | reimplements `editing_actions.py::MovePoint` canonicalization, which the harness `Curve.replace_point`/`apply_action` omit |
| Curve/Design schema, edit vocabulary, transition, rollout, chamfer/PI metrics | already in repo | `editing/mrcad_schema.py`, `editing/mrcad_refinement.py`, `bench/mrcad_metrics.py` |
| VLM eval, DeepSpeed training, cv2 rendering | external | trained model / GPU |

### 54. muse-main

MUSE text-to-CAD benchmark (three-stage funnel: exec -> geometry -> VLM judge).
The funnel + three pillars were already built; four implementation-level pieces
remained, and the source exposed a scoring-semantics issue in the harness.

| Build idea | Status | Repository comparison |
|---|---|---|
| Deterministic deduction-rule rubric engine (start 1.0, subtract per triggered rule_code, clamp, weighted + category aggregation) + plan component-count parser + weight dedup/normalisation | implemented `bench/muse2_rubric_deductions.py` | from `src/judge_system/rubric.py`; no deduction_ratio/rule_code logic in the harness |
| 4-view engineering-drawing SVG metrics (path/text counts, view-label detection, component estimate via single-linkage grouping of overlapping bboxes, mm dimension parse) | implemented `drawings/muse2_svg_view_metrics.py` | from `src/judge_system/svg_metrics.py`; no estimated_component/view_label in the harness drawings/ |
| Geometry-issue -> Stage-2 flag classifier with tri-state no_error semantics, combined-watertight definition, error counts | implemented `bench/muse2_geometry_issue_flags.py` | from `src/judge_system/geometry_metrics.py`; isolates OCCT-payload logic without the kernel |
| Judge-vs-human agreement protocol (Item/Cell/System levels, Pearson/Spearman/Kendall-tau-b, signed bias, seeded bootstrap 95% CI) | implemented `bench/muse2_judge_agreement.py` | harness had only a spearman and ranking-only kendall_tau; no Pearson, tau-b, bias, bootstrap, or multi-level judge protocol |
| Three-stage funnel, pillar averaging, assembly-graph isomorphism, joint/DoF Table 7 | already in repo | `bench/muse_scorecard.py`, `muse_functionality/manufacturability/assemblability.py` |
| Interpenetration volume-ratio, OCCT validator, VLM judge | external / already in repo | OCCT/LLM-in-loop |

**Finding (open, not actioned):** `bench/muse_scorecard.py` ANDs `watertight`,
`manifold`, `self_intersection_free`, `overlap_free` as four independent checks.
In the real MUSE repo, `watertight` is defined to already require no non-manifold
edges, so it implies `manifold`; feeding both independently double-counts
manifoldness. Whether this is a bug depends on how the caller derives the input
flags, so it changes what the benchmark score MEANS. `muse2_geometry_issue_flags`
provides `to_funnel_geometry()` + a `watertight_strict` flag so callers can feed
the funnel consistently, rather than silently editing the scorecard. Left as a
user scope decision, like the t2cq_ast subset question.

### 55. oce-oce-patches

Despite the name, a full 36k-file mirror of the OpenCASCADE (OCCT) C++ kernel,
not a patch set. Almost entirely external and already covered (NURBS, STEP, OCCT
API catalogue). One genuine, self-contained gap.

| Build idea | Status | Repository comparison |
|---|---|---|
| Gauss-Legendre quadrature nodes/weights + 1D/2D definite-integral integrators (Newton-on-Legendre generator reproducing OCCT's `math.cxx` table to 14 dp, `gauss_points_max`) | implemented `numeric/oce_gauss_legendre.py` | OCCT hard-codes the abscissae/weights for N=1..61; **the harness had zero numerical quadrature** (all 71 "gauss" hits were Gaussian noise/diffusion) |
| B-rep/NURBS kernel, STEP I/O, OCCT API, Bernstein/Bezier | already in repo | `numeric/nurbs_basis.py`, `geometry/nurbgen_*`, `formats/stepllm_parser.py`, `backends/ocp_occt_api_catalog.py`, `geometry/dreamcad_rational_bezier.py` |
| "Patches"/known-issues catalogue | N/A | repo is a full OCCT source mirror, not a patch set |
| Rest of OCCT (BOP, meshing, viewer, solvers) | external | non-stdlib-portable C++ kernel |

## Batch-11 implementation result

Five repos, ~211 new tests. The batch's centre of gravity is foundational
geometry the harness had been missing entirely: interval arithmetic and
forward-mode autodiff over a reified f-rep graph, dual contouring with QEF,
adaptive exact-sign 3D predicates, a half-edge mesh with circulation, a 3D BVH,
triangle-triangle intersection, a winding-number inside test, and Gauss-Legendre
quadrature. Together libfive + manifold give the harness both an implicit
(f-rep) and an explicit (mesh-boolean) robust-geometry substrate for the first
time. The two covered benchmarks (mrCAD, muse) yielded only implementation-level
gaps, including a sixth metric divergence (mrCAD point-to-curve vs point-to-point
design distance) and a scoring-semantics finding in the harness's own MUSE
scorecard (watertight/manifold double-count), left as a user decision. Per the
no-README policy the suite count is tracked in audit/cadbible_progress.json.

## Batch 12 (repos 56-60)

An OCCT binding, two covered concepts, and the STEP/OpenSCAD tooling. The
headline is ruststep's EXPRESS schema-language parser: the harness could read
STEP data but not the schema language that defines it.

### 56. pythonocc-core-master

SWIG bindings to the compiled OCCT kernel. Overwhelmingly external; one genuine
kernel-independent algorithm in the pure-Python Extend/ layer.

| Build idea | Status | Repository comparison |
|---|---|---|
| Orientation-aware sub-shape dedup + `MapShapesAndAncestors` inverse-ancestor map, kernel-free (Shape as `(TShape, orientation)` identity; 24 oriented cube edges -> 12 unique; "which faces bound this edge?") | implemented `geometry/pyocc_topology_explorer.py` | lifted from `OCC/Extend/TopologyUtils.py::TopologyExplorer`. Distinct from `reconstruction/cadparser_brep_graph.py` (coedge half-edge graph), `geometry/manifold_halfedge.py` (triangle-mesh half-edge), `geometry/opencad_synthetic_topology.py`; none model the TopAbs containment hierarchy or ancestor-map inversion |
| ShapeFactory/DataExchange/LayerManager wrappers, is_* predicates, SWIG bindings | external / already in repo | thin kernel calls; API surface covered by `backends/ocp_occt_api_catalog.py` |

### 57. querycad-main

Reference impl of QueryCAD (ICRA 2025, paper 148). QA schema, grounding, and eval
already built from the paper; two deterministic B-rep routines remained.

| Build idea | Status | Repository comparison |
|---|---|---|
| Edge-convexity classification (convex/concave/smooth via sign of `dot(cross(n_a,n_b), tangent)`) + Attributed Adjacency Graph (faces as nodes, convexity-labelled arcs, filtered-neighbour queries, convexity histogram) + dihedral angle | implemented `geometry/querycad_edge_convexity.py` | from HierarchicalCADNet `edge_dihedral`. New: harness had `mfgfeat_rule_detector` that *consumes* a per-face concave/convex flag but nothing that *computes* edge convexity from geometry |
| Face-adjacency segmentation: connected-component prune within a feature whitelist (DFS), partition a tagged face-set into part instances, one-ring/N-ring dilation, boundary-face isolation | implemented `geometry/querycad_face_adjacency.py` | from `CADFaceUtils.prune_non_adj_faces`. New: `cascade_entity_selector`/`cq_selector_algebra` select by geometric predicate; none model topological connectivity |
| Typed QA schema, segmentation grounding, grounded answer engine, QA eval | already in repo | `bench/querycad_query_schema.py`, `rag/querycad_segmentation_grounding.py`, `reconstruction/querycad_answer_engine.py`, `bench/querycad_eval.py` |
| Image ray-casting grounding, GNN feature segmentation, GroundedSAM | external | OCCT kernel + trained GNN |

### 58. ruststep-master

ruststep: ISO 10303 in Rust + its espr EXPRESS compiler. The harness had a Part 21
*data* parser but nothing for the EXPRESS schema *language*. Five new modules.

| Build idea | Status | Repository comparison |
|---|---|---|
| EXPRESS (ISO 10303-11) schema-language parser + model: tokenizer (`--` and `(* *)` comments), SCHEMA/ENTITY/TYPE, shared + OPTIONAL attributes, SUBTYPE OF, ABSTRACT, SUPERTYPE OF (ONEOF/ANDOR/AND), DERIVE/INVERSE/UNIQUE/WHERE, ENUMERATION/SELECT, aggregates LIST/SET/ARRAY/BAG with bounds, redeclared attributes | implemented `spec/express_schema_parser.py` | reimplements espr's parser/ + ast/ (nom -> recursive descent). **Not in the harness**; `formats/stepllm_schema.py` is a fixed 24-entity dict. **Parses 662/664 real ISO `.exp` schemas** incl. AP214 (890 entities) |
| Inheritance graph + attribute flattening: transitive supertype/subtype edges, roots/leaves, cycle detection, diamond-safe flattened attribute list (inherited before local), nested SELECT expansion | implemented `spec/express_inheritance.py` | reimplements espr's ir/ legalize stage. Not in the harness |
| Part 21 string X-encoding codec (decode/encode `\X\HH`, `\X2\..\X0\`, `\X4\..\X0\`, `\S\c`, `\\`) | implemented `formats/step_p21_xstring.py` | **fills a gap in `stepllm_parser`**; it returns X-directives verbatim (`\X2\00E9\X0\` instead of `cafe`), with no way to recover the Unicode. Round-trips as literal text so not data-loss, but a semantic gap |
| Structured HEADER model (typed FILE_DESCRIPTION/FILE_NAME/FILE_SCHEMA accessors + schema_name) | implemented `formats/step_header_model.py` | `stepllm_parser` keeps the header as an unstructured `Typed` list |
| Schema-to-Part21 validator: each DATA instance checked against a parsed EXPRESS schema (entity declared, inheritance-aware arity via flattened attrs, structural type-kind match, `$` vs OPTIONAL, complex-instance parts) | implemented `spec/express_p21_validator.py` | only possible now both parsers exist. `stepllm_graph.validate` checks arity against an entity's OWN attributes only, wrong for any subtype, since real records list inherited attributes first |
| Rust struct codegen | external | mechanism; the schema model it consumes was ported |

**Findings vs `formats/stepllm_parser.py`** (all gaps, not overstated as bugs):
X-encoded strings returned verbatim (fixed by the codec); header unstructured
(fixed by the header model); validation ignores inheritance (fixed by the
validator, which flattens the supertype chain); and a minor ed.3 `DATA(...)`
parametrized-section limitation, noted not fixed (out of scope).

### 59. scad-clj-master

A Clojure -> OpenSCAD generator. Its data-first CSG form and the OpenSCAD facet
formula were both absent from the harness.

| Build idea | Status | Repository comparison |
|---|---|---|
| Data-first keyword-tagged S-expression CSG IR + `write_scad` emitter (nested tuples, radian-rotation-to-degrees on emit, `$fn/$fa/$fs` dynamic-binding resolution via context managers, include/use/import/call/define_module, excise, postwalk transform) | implemented `programs/scadclj_data_ir.py` | `programs/solidpy_scad_emit.py` is a mutable object tree with no ambient special-variable resolution and passes rotation angles through untouched; `libfive_frep_ir` is an SDF opcode DAG. Data-first inert-tuple CSG + radian-rotate + dynamic `$fn` is distinct. Output round-trips through `scadlm_ast.parse` + `scadlm_check` |
| OpenSCAD facet-count resolution `get_fragments_from_r` (`$fn/$fa/$fs` -> fragment count: `ceil(max(min(360/fa, 2*pi*r/fs), 5))`) + exact CCW circle polygon, sphere rings, chord-error inverse | implemented `geometry/scadclj_facets.py` | **absent from the harness** (grepped `get_fragments`/`fragments_from`/`fn_fa_fs`: no hits). `scadlm_check` lists `$fn/$fa/$fs` as valid params but never resolves them. This governs how every curved OpenSCAD primitive tessellates |
| Solid line/lines capsule geometry: direction-to-rotation math (`acos(dz/len)`, axis `[-dy,dx,0]`) carrying +Z onto a segment, degenerate cases handled | implemented `geometry/scadclj_line.py` | no existing helper builds a solid strut between two 3D points. Fixes the source's bug (end caps left at origin, not the true endpoints) |
| Clojure/JVM runtime, core.matrix, glyph-outline text | external | needs font data |

### 60. scad-hs-master

A Haskell typed OpenSCAD EDSL. Overlaps heavily with existing typed-CSG/emitter/
gear modules, as anticipated; two Haskell-distinct algebraic pieces remained.

| Build idea | Status | Repository comparison |
|---|---|---|
| SetLike normalising CSG combinator laws (associative union/intersection flattening + difference subtrahend-absorption: `a-b-c-d` -> `difference(a, union(b,c,d))`) | implemented `geometry/scadhs_csg_algebra.py` | genuinely new. angelcad/solidpy/scadlm treat union/intersection/difference as opaque n-ary nodes; none apply the algebraic smart-constructor laws. Ported from `Class.hs` SetLike instance |
| Content-addressed module extraction / CSE (`smodule`/`#`/`##` with `children()` placeholder + `mdl_N` memo naming) + an added `auto_modularize` CSE pass detecting repeated subtrees | implemented `programs/scadhs_module_cse.py` | genuinely new: nothing in the harness hoists shared subtrees into named OpenSCAD modules. Faithful port of scad-hs's memo table |
| Typed 2D/3D phantom dimension; SCAD emission/modifiers; involute/planetary gears | already in repo | `programs/angelcad_typed_csg.py`, `programs/solidpy_scad_emit.py`, `geometry/cadgpt_involute_gear.py` + `cadgpt_gear_train.py` + `cqplug_bevel_gear.py` |

## Batch-12 implementation result

Five repos, ~231 new tests. ruststep contributed the campaign's most complete
STEP work: an EXPRESS schema-language parser validated against 662/664 real ISO
schemas, an inheritance flattener, an X-string codec, a structured header, and a
schema-driven validator, plus three characterised gaps in the harness's
existing Part 21 parser. scad-clj filled a real OpenSCAD gap (the facet-count
formula that governs all curved-primitive tessellation). pythonocc and scad-hs
were correctly disciplined (one/two genuine deltas from otherwise-covered repos),
and querycad yielded its two paper-omitted B-rep routines. Per the no-README
policy the suite count is tracked in audit/cadbible_progress.json.

## Batch 13 (repos 61-65)

Two SDF libraries (both cleared the high dedup bar the Curv/libfive stack set), a
web CAD app, a CADQuery RL reward system, and a UI frontend. Three repos whose
names misled about their contents.

### 61. sdf-csg-master

A TypeScript SDF/CSG library (IQ primitives, isosurface mesher). Primitives,
combinators, transforms, TPMS, and sphere tracing were all already covered; the
genuine gaps were mesh extractors and six missing primitives.

| Build idea | Status | Repository comparison |
|---|---|---|
| Canonical Marching Cubes (Lorensen-Cline 256-case edge + triangle tables, welded vertices, SDF-convention winding) | implemented `geometry/sdfcsg_marching_cubes.py` | **absent from the harness**; it had marching *tets* and dual contouring but not the standard MC. **Bug found+fixed:** Bourke's tables assume "inside = above isolevel" but SDF uses "inside = negative", so normals pointed inward until winding was reversed; a 256-case table-consistency test confirms correct transcription |
| Naive Surface Nets isosurface extractor (one vertex per sign-changing cell = averaged edge crossings) + tiled grid sampling + ASCII STL | implemented `geometry/sdfcsg_surface_nets.py` | the repo's actual mesher; distinct from QEF dual contouring (2D-only) and marching tets. **Bug found+fixed:** watertightness required correct per-axis quad winding |
| Six IQ primitives absent from curv: box_frame, capped_torus, link, hexagonal_prism, triangular_prism, solid_angle | implemented `geometry/sdfcsg_primitives.py` | extend `geometry/curv_sdf_primitives.py` |
| User-data attribute interpolation across the surface (inverse-distance blend) | implemented `geometry/sdfcsg_surface_nets.py` | the repo's getUserData mechanism; no harness equivalent |
| Primitives, CSG, smooth-min, transforms, sphere trace | already in repo | `curv_sdf_*`, `curv_sphere_trace` |

### 62. sdfx-master

deadsy/sdfx, a Go SDF CAD library. Distinctive from pure-SDF-art libraries in its
manufacturing 2D-CAD layer: exact concave-polygon distance and fastener geometry.

| Build idea | Status | Repository comparison |
|---|---|---|
| Exact arbitrary-polygon 2D SDF (winding-number sign + min edge distance) + area/centroid/point-in-polygon | implemented `geometry/sdfx_polygon_sdf.py` | `curv_sdf_primitives.regular_polygon` is a mitred *regular* n-gon field only; `manifold_winding` is a 3D solid-angle mesh test. No exact concave arbitrary-polygon 2D SDF existed |
| Fluent 2D sketch builder: relative/polar vertices, corner fillet (smooth), chamfer, segment->arc replacement, regular n-gon ring | implemented `geometry/sdfx_polygon_builder.py` | no polygon *geometry* builder in the harness; curv/libfive expose only fields |
| Standard thread cross-section profiles: ISO/UTS 60 deg V (7/8 H ext, 1/4 H int truncation, rounded root/crest), 29 deg Acme, ANSI 45/7 buttress (period-wrap-continuous) | implemented `geometry/sdfx_thread_profile.py` | `geometry/solidpy_screw_thread.py` is a helical mesh sweep with a generic tooth section; no standard ISO/Acme/buttress profile math |
| Fastener standard dimension database: ISO metric coarse/fine (M1-M64), UNC/UNF, NPT (tapered), normalised to mm, + hex head radius/height | implemented `standards/sdfx_thread_database.py` | `standards/cqplug_heatsert_schedule.py` is heat-set inserts only; no thread/fastener table existed |
| Cam profiles as exact SDFs (flat-flank + three-arc) with design-parameter solvers | implemented `geometry/sdfx_cam_profile.py` | no cam primitive in curv/libfive/sdf-csg |
| Archimedean spiral exact 2D SDF (polar inversion + whole-turn shifting, thickness offset) | implemented `geometry/sdfx_spiral_sdf.py` | no spiral primitive anywhere in the SDF stack |
| Go runtime, OpenGL, STL/DXF/3MF I/O, 3D helical Screw3D sweep, quadtree render accel | external / already in repo | `geometry/solidpy_screw_thread.py`; STL present |

### 63. solidtype-main

SolidType: a TypeScript/web parametric CAD app on OpenCascade.js. Its topological
naming and constraint-graph DOF were already covered; one genuine delta.

| Build idea | Status | Repository comparison |
|---|---|---|
| Fixed-point integer geometry substrate with shared-grid vertex welding: mm<->nanometre quantisation, exact integer vector ops, division-free exact segment/line/plane intersection with compute-once-snap-once grid snapping, VertexRegistry interning coincident points to one canonical id | implemented `geometry/solidtype_integer_geometry.py` | complementary to `numeric/manifold_predicates.py` (exact-sign predicates *classify* a determinant; this *quantises inputs* so near-coincident vertices weld, eliminating the epsilon non-manifold problem). New |
| Topological naming (fingerprint + evolution + split/merge/delete resolve) | already in repo | `geometry/opencad_face_fingerprint.py` |
| Constraint-graph connected-component DOF (under/over/fully-constrained) | already in repo | the existing constraints module (union-find rank DOF) |
| 2D curve intersection (tolerance-based) | already in repo | existing geometry |

### 64. spatialhero-main

Despite the name, not spatial reasoning: a multi-modal reward/verification system
for RL-training LLMs to write CadQuery. Three deterministic reward modules.

| Build idea | Status | Repository comparison |
|---|---|---|
| Rule-based physical-plausibility gate + hard-constraint checker (fill-ratio/solidity, extreme aspect ratio, SA-to-volume, magnitude bounds -> issues/warnings; constraint-dict -> per-key pass/fail) | implemented `verifiers/spatialhero_plausibility.py` | `quality/anomaly.py` turns the same bbox ratios into a *feature vector* for statistical outlier scoring; `verifiers/cgb_validity_gate.py` gates triangle face aspect. Neither is a fixed-threshold acceptance gate, and fill-ratio/solidity was absent |
| Bounding-box dimensional-accuracy metric (measure w/d/h/volume; bounded relative-error accuracy = max(0, 1-relerr), within-tolerance, average) | implemented `bench/spatialhero_dim_accuracy.py` | `reconstruction/pht_dimension_accuracy.py` matches dimension *annotations* with hard type/value/element; this scores realised solid *extents* continuously. `quality/cad_reward.py` is chamfer-based |
| Gated weighted multi-component composite reward (validated weight-map summing to 1.0; hard gate keys collapse reward to 0) | implemented `quality/spatialhero_composite_reward.py` | `quality/cad_reward.py` is a fixed 2-term reward; this is a generic N-named-component aggregator with configurable hard gates |
| AST static code-safety analysis; VLM eval, CadQuery executor, PPO trainer | already in repo / external | `programs/`, `quality/cad_code_normalize.py`; trained model / kernel |

### 65. structural-frontend-main

| Build idea | Status | Repository comparison |
|---|---|---|
| (none) | out-of-scope, nothing built | The frontend twin of the Gaudi product (same gaudi_logo.png, same Railway backend): a Create-React-App SPA: react-router auth-gated screens, a three.js `@react-three/fiber` BuildingViewer loading server-generated STL/GLB, a chat Assistant that POSTs prompts and polls. A grep of src/ for beam/truss/moment-of-inertia/section-modulus/centroid/deflection/buckling/euler/yield/shear/bending/load-combination/young's-modulus/I-beam/steel-section returned zero hits. All structural analysis is server-side; nothing deterministic is transferable |

## Batch-13 implementation result

Five repos, ~150 new tests. The two SDF libraries validated the dedup discipline:
against the now-deep Curv/libfive SDF stack, sdf-csg still yielded the two
standard mesh extractors the harness lacked (marching cubes, surface nets) plus
six primitives, and sdfx yielded the entire manufacturing 2D-CAD layer (exact
concave-polygon distance, ISO/Acme/buttress thread profiles, a fastener database,
cam and spiral profiles) that pure-SDF-art libraries omit. solidtype added
integer-geometry vertex welding (complementary to the manifold predicates).
spatialhero (a third name-misleads-content repo) gave a plausibility gate and
reward aggregator. structural-frontend was correctly a second UI no-build. Two
marching-cubes/surface-nets winding bugs were found and fixed during testing. Per
the no-README policy the suite count is tracked in audit/cadbible_progress.json.

## Batch 14 (repos 66-70): FINAL BATCH

The campaign's last five. Two repo-identity assumptions in the launch prompts
turned out wrong, and in both cases the agent verified by grep and mined what was
actually there rather than forcing the hypothesis.

### 66. text-to-cad-blender-addon-main

A Blender addon driving Zoo/KittyCAD's text-to-CAD API. The bpy/UI/live-API
layers are external; two deterministic pieces were transferable.

| Build idea | Status | Repository comparison |
|---|---|---|
| Base64 data-URI codec (encode/decode, MIME handling) | implemented `formats/t2cblender_base64data.py` | no data-URI codec existed |
| Zoo/KittyCAD text-to-CAD async API schema (job submission, status polling, result envelope) as a deterministic data model | implemented `adapters/t2cblender_zoo_api_schema.py` | new; the harness had no text-to-CAD job/result schema |
| bpy operators/panels, live API calls, LLM | external | Blender host / network / trained model |

### 67. text-to-cad-main

**Not** Zoo's CLI (the launch prompt's assumption was wrong; the agent grepped:
no `kcl`/`kittycad` anywhere). It is `earthtojake/text-to-cad` "CAD Skills": an
agent-skills library for CAD, robotics, and hardware handoff. It opened a domain
the harness had nothing in: robot description.

| Build idea | Status | Repository comparison |
|---|---|---|
| URDF forward kinematics: 4x4 rigid transforms, rpy/axis-angle, joint clamping, mimic-chain resolution with cycle guard, DFS world-transform solve, frame-relative queries | implemented `geometry/t2cmain_urdf_kinematics.py` | **the harness had no robot kinematics of any kind**; `numeric/cq_assembly_dof.py` counts DOF, it does not pose |
| Strict URDF XML parser: joint-type whitelist, mandatory `<limit>`, radian->degree conversion, single-rooted-tree/cycle/multi-parent/forest rejection, primitive + material validation | implemented `spec/t2cmain_urdf_parser.py` | new (`spec/express_schema_parser.py` parses EXPRESS, unrelated) |
| SRDF semantics with URDF cross-validation: chain joint-path walk, group closure over chains+subgroups (cycle-safe), end-effector adjacency/disjointness rules, group-state limit checks, disabled-collision dedup + reason classification, missing-adjacent-disable detector | implemented `spec/t2cmain_srdf_semantics.py` | entirely new capability |
| CAD-reference token grammar `#o1.2.f3,f4`: occurrence-tree positional selectors with left-to-right occurrence inheritance, canonicalisation, safe STEP-path normalisation | implemented `programs/t2cmain_cad_ref_selectors.py` | categorically different from `geometry/cq_selector_grammar.py` / `cqcontrib_selector_dsl.py` / `cascade_entity_selector.py`, which are *predicate* selectors (">Z", tags); this is an *index into an assembly occurrence tree* |
| Exploded-view layout solver: occurrence-prefix grouping, coplanar layer merge with model-scaled tolerance, non-intersecting stacking gaps, golden-angle spiral fallback for centroid-degenerate radial groups, grounded base, eased progress | implemented `geometry/t2cmain_exploded_view.py` | new; no assembly-presentation solver existed |
| Zoo API schema / base64 codec | already mined (repo 66), and absent from this repo anyway | `adapters/t2cblender_zoo_api_schema.py`, `formats/t2cblender_base64data.py` |
| KCL lexer/parser/AST | not present in this repo | nothing to mine |
| STEP/STL/3MF/GLB export, three.js viewer, implicit GLSL SDF, MoveIt2 server, slicers, printer control | external / already in repo | `formats/` covers the formats; `curv_sdf_*`/`sdfcsg_*` cover SDF |

**Note on method:** three test rounds failed, and each time the agent checked the
source before changing anything, finding the *source's* rule was right and its
own assumption wrong (mimic-cycle fallback resolves to the re-entered joint's own
default; the SRDF end-effector adjacency rule; the common-occurrence-prefix
grouping window). It corrected the tests to reality, not the modules to the tests.

### 68. text-to-cad-ui-main

| Build idea | Status | Repository comparison |
|---|---|---|
| (none) | out-of-scope, nothing built | Zoo's SvelteKit reference frontend for the Text-to-CAD API. Svelte components, a Threlte/three.js ModelViewer (only "math" is stock bounding-box camera-fit), `@kittycad/lib` API client, and trivial base64/time/kebab helpers. Specifically checked for a client-side KCL parser (which would have been genuinely new): KCL appears only as a download-format string and an API request flag; the language is generated server-side. Third correctly-out-of-scope UI repo |

### 69. vitruvion-main

Vitruvion (ICLR 2022), a generative model of parametric CAD sketches. The paper's
concepts were covered indirectly by the SketchGraphs modules, but **no Vitruvion
implementation-level numerics existed**. Six modules, and the campaign's
sharpest quantiser finding.

| Build idea | Status | Repository comparison |
|---|---|---|
| Exact analytic sketch bbox (quadrant-crossing arc rule) + centre/long-axis-1 rescale to `[-0.5,0.5]` + start/mid/end arc parameterisation with circumcentre re-fit | implemented `geometry/vitruvion_sketch_norm.py` | `drawings/sgraphs2_entity_render.bounding_box` is *polyline-sampled* and always under-reports an arc's extremum; `normalize_scene` maps to `[0,1]^2`. Vitruvion needs an *exact* bbox and a `[-0.5,0.5]` domain or the quantiser bins differ |
| Primitive token codec: 3 parallel streams (val/coord/pos), 7 control tokens, per-slot coord ids (Arc 2-7, Circle 8-10, Line 11-14, Point 15-16), isConstruction flags, gather pointer offsets | implemented `reconstruction/vitruvion_primitive_tokens.py` | no 3-stream sketch tokenizer existed; `deepcad2_*`/`gencad2_*` are single-stream command tokenizers |
| Constraint hypergraph as a *pointer* token stream (16-token vocab, ref token = `16 + gather_idx`, arity limited to 2 by the coord vocab, external constraints dropped, refs sorted not argument-ordered) | implemented `reconstruction/vitruvion_constraint_tokens.py` | `reconstruction/sketchgraphs_graph.py` models the hypergraph as a data structure; nothing encoded it as pointers into a primitive stream |
| Seeded truncated-normal primitive noise (parameter-space, arc rejection loop with halving scale) | implemented `datagen/vitruvion_primitive_noise.py` | no perturbation module existed |
| Hand-drawn stroke noise: Matern-GP displacement along arclength (stdlib Cholesky), radial for arcs, 359-degree circle gap | implemented `drawings/vitruvion_hand_drawn_noise.py` | no Matern/Cholesky/GP code existed; `datagen/sketch_image_conditions.py` only *names* a `simulate_hand` callable; this supplies it |
| Dataset curation: 6..16-entity filter + degeneracy rejection, exact length-bucketed token dedup, remainder-balanced shard ranges | implemented `dataengine/vitruvion_sequence_filter.py` | `bench/skexgen_dedup_hash.py` is a *hash* dedup (lossy); Vitruvion's is exact. No shard-range util existed |
| Autoregressive transformers, MobileNet image encoder, samplers | external | trained model / GPU |

**Quantiser finding (the campaign's 8th representation discrepancy).** Vitruvion:
```
bin   = int((v + 0.5) * n)      # TRUNCATION (floor), clamp n -> n-1
value = (bin + 0.5) / n - 0.5   # BIN CENTRE, not the level
```
domain `[-0.5, 0.5]`, default n = 64 (6-bit). Compare:
- DeepCAD (`reconstruction/deepcad2_numericalize.py`) and GenCAD: **256 levels over
  `[-1,1]`, round-half-even, dequantise at the level**.
- SkexGen: 6-bit **truncating**, but also dequantises at the level.

Vitruvion is the only one with **floor-quantise + bin-centre dequantise**, so its
round-trip error is unbiased and bounded by `1/(2n)`, whereas floor/floor
(SkexGen) biases every coordinate *downward* by half a bin. Mixing Vitruvion bins
with a DeepCAD-style dequantiser shifts every primitive by half a bin of the
sketch's long axis: a silent systematic offset.

Two further reference quirks reproduced and documented in-module:
- `truncnorm.rvs(a=-max_diff, b=max_diff, scale=std)`: scipy's bounds are in
  **standard-deviation units**, so the paper's `std = max_diff = 0.15` actually
  bounds the jitter at `0.0225`, 6.6x tighter than the literal reading. Exposed as
  `bounds_in_std_units`.
- `RenderNoise._get_ranges` updates the extent with `+/-radius` **about the origin**
  and the centre separately (not `centre +/- radius`). Cosmetic (it only feeds the
  GP smoothness scale); `bbox_extent=True` opts into the corrected version.
- The constraint coord vocab has only 2 reference slots, so a hyperedge of arity
  >= 3 would silently desync the three streams; this port raises instead.

### 70. CodeToCAD (in `zip/`)

The folder named `zip` was **not** a junk archive; it contains CodeToCAD, a
provider-agnostic CAD scripting API (Blender/FreeCAD/OnShape backends). Two
genuine kinematics deltas; landmarks and the interface taxonomy were already
covered.

| Build idea | Status | Repository comparison |
|---|---|---|
| Per-axis 6-DOF joint limit box: each DOF free / locked / `[min,max]`; clamp a proposed pose into the box (angles wrapped to the nearest branch first); classify joint type from the limit pattern; intersect limit boxes; rigid/revolute/prismatic/cylindrical/planar/ball constructors | implemented `geometry/codetocad2_joint_limit_box.py` | new: `geometry/joinable_joint_motion.py` only *zeroes* disallowed DOF; no ranges, so a revolute joint had no stops; `numeric/cq_assembly_dof.py` only *counts* DOF |
| Gear-ratio rotational driver coupling `driven = -ratio * driver` + train propagation (mesh sign external/internal/shaft, effective ratio, idler sign flip, compound shaft stages, speeds, ideal torques, cycle + multi-driver rejection) | implemented `geometry/codetocad2_gear_coupling.py` | new: `geometry/cadgpt_gear_train.py` does ratio + spatial *placement*, never propagates rotation |
| Landmark system (named bbox anchors, cardinal presets, offsets, nearest-entity search) | already in repo | `geometry/codetocad_cardinal_landmark.py` |
| Provider-agnostic interface taxonomy | already in repo | `adapters/fcai_freecad_tool_catalog.py`, `cadhub_language_registry.py`, `ccc_codecad_ecosystem.py` |
| Length/angle unit expressions; transform stack | already in repo | `numeric/codetocad_length_expression.py`, `geometry/codetocad_transform_stack.py` |
| Blender/FreeCAD/OnShape live backends | external | proprietary hosts |

## Batch-14 implementation result

Five repos, ~332 new tests. The final batch opened a domain the harness had never
touched (robot description: URDF kinematics, URDF/SRDF parsing and cross-
validation), closed the last kinematics gaps (joint limit boxes, gear rotational
propagation), added the Vitruvion sketch stack, and produced the campaign's 8th
and sharpest representation finding: Vitruvion is the only sketch quantiser in
the corpus using floor-quantise + bin-centre dequantise, making it the only one
whose round-trip error is unbiased. Two of the five launch prompts carried a wrong
repo-identity assumption; in both cases the agent verified by grep, said so, and
mined the repo that actually existed. Three UI repos across the campaign were
correctly no-builds.

## CAMPAIGN COMPLETE: 70/70 repos mined.
