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
