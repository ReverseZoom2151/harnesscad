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
