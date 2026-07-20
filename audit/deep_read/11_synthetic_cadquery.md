## synthetic_cadquery-main (double-nested `synthetic_cadquery-main/synthetic_cadquery-main/`; 2225 files, 529 py, 145 json; read ~15 files fully + 6 data probes)

### LICENSE + verdict
**No licence.** Verified: no `LICENSE`/`COPYING`/`licence` file at any depth (searched depth <=3 and full-tree); `README.md` has an Overview/usage section only, no licence line; there is **no `setup.py`/`pyproject.toml`/`setup.cfg`** at all (not a packaged project — just scripts). A grep for "license/copyright/MIT/Apache/BSD/GPL/SPDX" across all `.md/.txt/.py/.toml/.cfg` returns a single incidental word-match inside `Processors/SyntaxValidProcessor.py`, not a licence grant.
**Verdict: MANIFEST-ONLY.** Record paths + SHA, resolve from `resources/` at runtime, vendor nothing. Facts (schema field names, the status-code taxonomy, the two-stage prompt design) remain recordable with citation.
`cad_schema.json` SHA-256 `5ddd15f31df806eea0f30e43a1cd69ceb987b2e99bd79951f42d35d0cef33a4c`.

### WHAT IT IS
A **synthetic-data-generation (SDG) pipeline** that fabricates image->CadQuery training pairs, not a library. Flow (README + `run_structured_pipeline_with_metrics.py`):
1. **VLM** (Qwen2.5-VL-72B) captions 1000 ABC-dataset object renders -> either free-text descriptions (`runVLM.py` -> `object_descriptions.json`) or a **structured JSON per `cad_schema.json`** (`runVLM_structured.py` -> `parse_success` flag).
2. **LLM** (Qwen3-Coder-30B) turns the description/JSON into CadQuery source (`runLLM.py` / `runLLM_from_structured.py`), few-shot-primed by `docs_examples/` and validated inline.
3. `misc_scripts/generate_py_files.py` writes the `valid_syntax==1` entries to `.py` files; `generate_images.py` renders them (`xvfb-run`) for the next SDG round.
Everything is vLLM/CUDA batch inference + model-generated artefacts. `data/sdg_abc_1k_images/`, `generated_code/` (472 py), `generated_code_images/`, `out/` are all pipeline OUTPUT.

### READ (fully)
- `README.md`, `cad_schema.json` (the DSL), `run_structured_pipeline_with_metrics.py`
- `inference_pipelines/runVLM_structured.py`, `runLLM_from_structured.py`
- `inference_pipelines/Processors/SyntaxValidProcessor.py` (the validation core)
- `misc_scripts/generate_py_files.py`
- Data probes: `object_descriptions.json` (1000, keys full_response/valid_syntax/code/image_path), `sdg.json` (1000, +description; **valid_syntax dist: 472x `1`, 528x `-1`**), `docs_examples/examples.json` + `cheatsheet_geometric_commands.json`, one `generated_code/*.py` sample.

### SKIMMED-NOT-READ (honest volume)
Did NOT read: 471 of 472 `generated_code/*.py` (model output), all `generated_code_images/` + `data/sdg_abc_1k_images/` PNGs, `out/**` sample dumps, `DataUtils/Datasets.py`, `runVLM.py`/`runLLM.py` (unstructured twins of the two I read), the four `run_pipeline_*gpu.sh` launchers, 39 of 40 `docs_examples/example*.py`. That is >2200 of 2225 files unread — but they are OUTPUT artefacts, rendered images, or near-duplicates of files I read.

### FINDINGS (ranked)
1. **The exec-based validator with a 6-code failure taxonomy** — `inference_pipelines/Processors/SyntaxValidProcessor.py`. Runs generated code in a `Process` with a timeout, `cq.exporters.export(result, ...)` to force kernel evaluation, and buckets failures: 1=GT-recon-fail, 2=gen-fail, 3=OCC-fail, 4=timeout, 5=None-solid, 6=mp-error; aggregates VSR (valid-syntax rate) + IoU mean/median (`_OCC_IOU.align_shapes`). **Harness equivalent: PRESENT and stronger** — `src/harnesscad/data/dataengine/reward/executability_reward.py` (CAD-RL: binary exec gate x volumetric-IoU R_geom x external-eval R_eval, with named error categories Reference-Frame-Misalignment / Parametric errors); volumetric IoU in `agents/generation/shape_metrics.py` (voxel_iou + ICP + Chamfer + F1); execution-free static IR in `domain/programs/validate/cadquery_validity.py::invalid_rate`. **Verified** by reading all three. Repo's version uses bare `exec()` and doesn't even wire GT (`ground_truth=""` in `runLLM_from_structured`), so its "IoU" path is dead here. **facts-only** (taxonomy is citable; code is unlicensed).
2. **`cad_schema.json` — an 11-op image->build DSL** (workplane enum + operations: sketch{rect/circle/polygon/polyline}, extrude, cut, hole, fillet, chamfer, rotate, translate, mirror, shell, workplane_on_face). A draft-07 JSON-Schema used only as a *prompt scaffold* (embedded verbatim as `SCHEMA_EXAMPLE` in `runVLM_structured.py`), NOT as a validation contract — nothing in the repo validates VLM output against it (`parse_success` is mere `json.loads` success). **Harness equivalent: PRESENT and broader** — `domain/programs/validate/operation_schema.py` (typed op vocabulary + `validate_program` symbol-table checker), `data/datagen/cadquery_codegen.py` (CISP-dict->CadQuery emitter, same schema->source shape), `data/dataengine/schemas/minimal_json.py`, `domain/reconstruction/sequences/sketch_extrude_schema.py`. This schema is a strict subset. **facts-only**.
3. **`object_descriptions.json` / `sdg.json` — 1000-row image->code corpora** with a `valid_syntax` label. **Not machine-checkable ground truth**: the label is the pipeline's own `exec()` result (self-reported), there is no reference solid, and 528/1000 are `-1` (invalid). Source images are ABC renders. **Harness equivalent: `data/dataengine/annotation/cadquery_dataset.py` + the reward modules** cover verified corpora. **manifest-only** (unlicensed model-generated data; low value — noisy, GT-less).
4. **Two-stage VLM-caption -> LLM-code prompt design** (`runVLM_structured.py` + `runLLM_from_structured.py`): decouple perception (structured JSON) from synthesis (few-shot CadQuery), let the coder LLM "repair" incomplete JSON. **Harness equivalent: no exact `structured_cad`/VLM-JSON->code module** (grep for `structured_cad`/vlm-structured in `src/` = empty), but the *pattern* is subsumed by the CAD-RL VLM reward stack + `domain/programs/extract/cadquery_clean.py` (markdown-fence code extraction, same as repo's `extract_code`). **facts-only** (prompt text is expression; the design idea is citable).

### ALREADY COVERED
- Validity / exec gating / IoU -> `data/dataengine/reward/executability_reward.py`, `agents/generation/shape_metrics.py`, `domain/programs/validate/cadquery_validity.py`.
- Op-DSL + schema->CadQuery emit -> `domain/programs/validate/operation_schema.py`, `data/datagen/cadquery_codegen.py`, `data/dataengine/schemas/minimal_json.py`, `domain/reconstruction/sequences/sketch_extrude_schema.py`.
- Code extraction from LLM markdown -> `domain/programs/extract/cadquery_clean.py`.
- CadQuery corpus/annotation -> `data/dataengine/annotation/cadquery_dataset.py`.

### VERDICT
**already-covered / nothing-here.** No vendorable, licence-clean, harness-absent artefact. Everything of substance (exec-validity + IoU + failure taxonomy, an op-DSL, schema->code emit, LLM code extraction, image->code corpora) exists in `src/harnesscad` in a more rigorous, deterministic, verified-GT form; the repo is an unlicensed, GT-less SDG script pipeline whose schema and status-code taxonomy are recordable as facts-only citations but not worth further mining.
