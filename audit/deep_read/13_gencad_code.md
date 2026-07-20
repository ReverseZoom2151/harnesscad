# Deep read — GenCAD-Code (CADCODER: DeepCAD-derived image→CadQuery corpus)

New repo added today. `resources/cad_repos/GenCAD-Code-main/GenCAD-Code-main/`
(double-nested). Genuine read of all 8 py files, the derived-data jsonl/csv, and
the real-photo test-set spreadsheets. Image/HEIC volume stated honestly. Coverage
claims checked against `registry.index()` (1602 modules) and greps of
`src/harnesscad`.

Bottom line up front: GenCAD-Code is **mostly a re-derivation of DeepCAD** the
harness already mines heavily — the DeepCAD command constants and the
h5-vector→CadQuery normalisation/quantisation are **already covered**
(`deepcad_commands`, `deepcad_quantize` NORM_FACTOR=0.75 verified byte-equal,
`gencad_quantize`, `gencad_canonical_order`, `cadquery_codegen`). The **one
genuinely new asset** is the **`real_photo_test_set`: a held-out image→CAD eval
set of 400 real photographs of 50 physically 3D-printed DeepCAD objects**, with
committed per-photo metadata and a DeepCAD-ID mapping — the harness has **no
real-photo→CAD eval set**. Everything is **manifest-only** (no LICENSE; corpus
explicitly *derived from DeepCAD*). NB: distinct from the harness's existing
`gencad_*` modules, which cite a *different* project (GenCAD, Alam & Ahmed
image→B-rep); this repo is CADCODER's GenCAD-**Code** (image→CadQuery source).

---

## GenCAD-Code (819 files: 8 py, 400 png + 400 HEIC, 3 jsonl, 2 xlsx, 1 csv)

### LICENSE — VERDICT: MANIFEST-ONLY (no license; DERIVED from DeepCAD)
Verified absent: no `LICENSE`/`COPYING`/`NOTICE` at depth ≤4; `README.md` has no
license section (only HF dataset links + a Google-Drive DeepCAD-vector link);
`environment.yml` carries no classifiers. `README.md` line 5 states the corpus is
"**derived from the DeepCAD dataset**" and `scripts/deepcad_constants.py` line 1
says "**Copied over from DeepCAD repo**". The harness treats DeepCAD as MIT ©2022
Rundi Wu, but this **derived** corpus commits **no** license of its own and does
not restate DeepCAD's — so it does **not** inherit cleanly on the record here. →
**manifest-only** for all data (paths+SHA, runtime resolve, vendor nothing).
**Constants/schema field names remain recordable as facts with citation.**

### WHAT IT IS
CADCODER's data-prep repo for two HF datasets: `CADCODER/GenCAD-Code` (163k
image↔CadQuery-code pairs, DeepCAD-derived) and `CADCODER/real_photo_test` (400
photos of 3D-printed DeepCAD test objects). The repo commits the **scripts** that
build them + **small derived artifacts + the real-photo metadata**; the bulk
(.h5 vectors, rendered images, generated .py) is gitignored and fetched from
Drive/HF.

### READ (in full)
- `scripts/h5tocadquery.py` (626 lines) — the DeepCAD h5-vector → CadQuery-source
  emitter (sketch/loop/extrude reconstruction, un-quantisation, 2-sided extrude
  handling, STEP export). Read in full.
- `scripts/deepcad_constants.py` — LINE/ARC/CIRCLE/EOS/SOL/EXTRUDE = 0..5,
  EXTRUDE_OPERATIONS, EXTENT_TYPE, NORM_FACTOR=0.75.
- `prompts.py` — single line: the fixed image→code instruction (below).
- `README.md`, `environment.yml`, `.gitignore`.
- `deepcad_derived/{cadquery_test_data_subset100.jsonl, cadquery_test_tokencount.jsonl,
  cadquery_train_tokencount.jsonl, split.csv}` — schema + counts via pandas/json.
- `real_photo_test_set/{RealPhotoTestSet.xlsx, id_deepcad_pairs.xlsx}` — read all
  columns + head rows via pandas.

### SKIMMED — NOT READ
- `scripts/{geom_utils.py, gencadcode_to_hf.py, upload_realphoto_to_hf.py,
  process_heic.py, generate_graphs.py}` — read purpose/imports; geom_utils holds
  the `CoordSystem`/`get_arc` helpers h5tocadquery imports (DeepCAD-derived math).
- 400 real-photo PNGs + 400 HEICs — counted (`find`), not decoded.
- The two token-count jsonls beyond one record each (147289 train / 7355 test
  rows) — llava-specific token stats, low verifier value.

### FINDINGS (ranked)

**1. `real_photo_test_set/` — held-out REAL-PHOTO → CAD eval set (NEW).**
*What:* 50 DeepCAD test-split models were **physically 3D-printed and
photographed 400 times** under varied conditions. Committed ground truth/metadata:
  - `RealPhotoTestSet.xlsx` (sha256 f674eec9e7923ed9) — 400 rows, cols
    `Object_ID, Object_Color, Orientation, Proximity, Background, Lighting, Notes`
    (8 capture conditions × 50 objects). A ready-made **robustness stratification**
    (color/orientation/zoom/background/lighting) for image→CAD.
  - `id_deepcad_pairs.xlsx` (bf96a3b15bbfe294) — 50 rows `DeepCAD_ID → Object_ID`;
    this is the bridge: the **ground-truth CAD is the DeepCAD model** (already in
    the harness), so each photo has a resolvable GT solid.
  - 400 committed PNGs (+ 400 source HEICs).
*Why (verifier-first):* the task flagged "a real-photo held-out set is a genuine
image→CAD eval set if GT is committed" — here GT **is** committed (via the
DeepCAD-ID map to a corpus the harness holds), and the per-photo condition labels
give a domain-shift/robustness axis the synthetic renders lack.
*Harness equivalent:* **NONE.** `grep -rE 'real.?photo|3d.?print' src/harnesscad/eval`
→ only manufacturability/usability modules (no photo corpus). `eval.bench.data.
image_perturbations` synthesises corruptions; this is *real* photographic shift.
*Disposition:* **manifest-only** (no license, derived) — manifest the two xlsx +
the 400 PNGs by path+SHA; resolve GT via the existing DeepCAD corpus + the ID map.

**2. `deepcad_derived/cadquery_test_data_subset100.jsonl` (8806949518063553) — 100 committed image→CadQuery GT pairs.**
*What:* 100 records `{question_id, image (e.g. "0067/00675497_0.png"), text
(prompt), category, ground_truth (full CadQuery source)}`. A small,
self-describing image→code GT slice.
*Why:* machine-checkable image→CAD-code fixtures (execute GT → solid → compare).
*Harness equivalent:* the **generation path** is covered (`data.datagen.
cadquery_codegen`, `domain.reconstruction.translate.cadquery_translate`); a
*committed image↔code fixture list* is not, but this one is DeepCAD-derived and
its images are gitignored (only paths, not pixels, are committed here).
*Disposition:* **manifest-only** (schema field names recordable as facts).

**3. `deepcad_derived/split.csv` (dbd7f95f9d942863) — DeepCAD train/val/test split.**
*What:* 163,671 rows `deepcad_id → split`; distribution
train 147289 / validation 8204 / test 7355 / none 823.
*Why:* a specific, reproducible partition of the full DeepCAD corpus — useful as a
reference split for leakage-audit parity.
*Harness equivalent:* `eval.bench.data.splits` exists (source-aware synthetic/wild
manifests + leakage audits) but is not this exact GenCAD-Code partition.
*Disposition:* **manifest-only / facts** (record the split mapping's provenance +
counts; it is a derived assignment, not licensed data).

**4. `prompts.py` + the jsonl prompt field — the fixed image→code instruction.**
Single exemplar prompt: *"Generate the CADQuery code needed to create the CAD for
the provided image. Just the code, no other words."* UNVERIFIED reference/exemplar
only — never wire into a live prompt. Facts-only (one string, cite source).

### ALREADY COVERED (name the module)
- **DeepCAD command constants** (LINE=0…EXTRUDE=5, EXTRUDE_OPERATIONS, EXTENT_TYPE):
  `domain.reconstruction.tokens.deepcad_commands` (COMMAND_TYPES = SOL,LINE,ARC,
  CIRCLE,EXT,EOS). `deepcad_constants.py` is explicitly a DeepCAD copy.
- **NORM_FACTOR = 0.75 + 256-level un-quantisation affine**: byte-verified identical
  to `domain.reconstruction.tokens.deepcad_quantize` (NORM_FACTOR=0.75; sketch
  scale `(size/2*NORM_FACTOR-1)/bbox`), also `gencad_quantize`.
- **h5-vector → CadQuery source reconstruction** (workplane/loop/extrude emit,
  2-sided extrude, cut/intersect/union): the algorithmic content of
  `h5tocadquery.py` overlaps `domain.reconstruction.sketch.gencad_canonical_order`
  + `data.datagen.cadquery_codegen` + `domain.reconstruction.translate.cadquery_translate`.
- Note: harness `gencad_*` modules cite **GenCAD (image→B-rep, Alam & Ahmed)** — a
  *different* project from this **GenCAD-Code (CADCODER, image→CadQuery)**; the
  names collide but the reconstruction math is the shared DeepCAD lineage.

### VERDICT
**Largely a DeepCAD re-derivation the harness already covers** — constants,
NORM_FACTOR, and the h5→CadQuery pipeline are duplicates (stated plainly). The
**single new asset is the `real_photo_test_set`** (400 real photos of 50 3D-printed
DeepCAD objects, committed metadata + DeepCAD-ID→GT map): a genuine real-photo→CAD
held-out eval set the harness lacks. Everything is **manifest-only** (no license,
DeepCAD-derived); constants/schema/split counts stay recordable as facts with
citation. Recommend: manifest the real-photo set (xlsx + 400 PNGs by path+SHA,
GT resolved through the existing DeepCAD corpus); vendor nothing else.
