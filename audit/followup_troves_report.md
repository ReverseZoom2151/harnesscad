# Follow-up troves report

Findings from the follow-up mining pass over four resource troves: the oce
reviewer-tag catalog (A), pythonocc/ruststep STEP canaries (B), DeepCAD/SkexGen
quantization ranges (C), and kerf's tess/topo/imports sibling packages (D).

Every license verdict below was reached by locating and reading the source
repo's own LICENSE file, not by assuming from the ecosystem.

---

## A and B -- completed in a prior pass

Both landed before this pass and were not revisited.

* **A -- oce reviewer-tag quirk scan.** Delivered a 30-row OCCT kernel-quirk
  catalog (`occt_quirks_oce.py`) mined from oce reviewer tags (`//:abv`,
  `//:pdn`, `skl`, plus OCC bug ids), disjoint from the existing client-side
  catalog. Commit `c5dc373`.
* **B -- pythonocc / ruststep STEP canaries.** Delivered
  `eval/corpus/fixtures/step_canaries.py` with 8 **manifest-only** entries
  (LGPL / permissive mix: SHA-256 verified, paths into `resources/`, nothing
  vendored). Commit `232cd25`.

---

## C -- DeepCAD / SkexGen quantization ranges

**Delivered:** `src/harnesscad/domain/programs/quantization_ranges.py`
(commit `8d15944`).

### License verdicts

| Repo | License file | Verdict |
|---|---|---|
| DeepCAD (`DeepCAD-master/LICENSE`) | MIT, Copyright (c) 2022 Rundi Wu | Vendoring permitted |
| SkexGen (`SkexGen-main/LICENSE`) | MIT, Copyright (c) 2022 Xiang Xu | Vendoring permitted |

Both are MIT, so vendoring was allowed. The module still **reimplements rather
than copies**: each codec is only a few lines, and the content worth having is
the *behaviour*, which is documented and pinned by the selfcheck. Constants are
carried as cited facts with file-and-symbol attribution.

### Constants located

* **DeepCAD** `cadlib/macro.py`: `ARGS_DIM = 256`, `NORM_FACTOR = 0.75`,
  `PAD_VAL = -1`, the `N_ARGS_*` decomposition (5 sketch + 11 extrude = 16), and
  the `MAX_N_EXT/LOOPS/CURVES/TOTAL_LEN` caps (10 / 6 / 15 / 60). Codec
  behaviour from `cadlib/extrude.py`, `cadlib/curves.py`, `cadlib/sketch.py`.
* **SkexGen** `utils/utils.py`: `SKETCH_R = 1`, `RADIUS_R = 1`,
  `EXTRUDE_R = 1.0`, `SCALE_R = 1.4`, `OFFSET_R = 0.9` (duplicated verbatim in
  `dataset.py`). The shipped bit width is 6 (`--bit 6` throughout the README),
  i.e. 64 tokens -- a quarter of DeepCAD's resolution.

### Key finds

1. **DeepCAD's grid is asymmetric.** Forward is `round((x+1)/2*n)` clipped to
   `[0, n-1]`; inverse divides by `n`, not `n-1`. So `x = -1` round-trips
   exactly (a fencepost) while `x = +1` clips and returns `0.9921875` (a clip).
   Round-trip error is `1/n` in range but `2/n` at the top endpoint.
2. **DeepCAD's extent guard is looser than its codec's range.**
   `Extrude.numericalize` asserts `-2.0 <= extent <= 2.0`, but the affine map
   only covers `[-1, 1]`. An extent of `1.5` passes the assert and is then
   silently clipped -- indistinguishable from `1.0`, with no error raised.
3. **SkexGen truncates, it does not round.** `quantize` clips then calls
   `.astype('int32')`. Because the clip runs first the value is non-negative and
   truncation is a floor, so the error is **one-sided** (never overshoots) and
   the worst case is a **full bin, not half**.
4. **The `geom_utils` variant omits the clip entirely.**
   `geom_utils.quantize_verts` has no clip step, so input outside `[-0.5, 0.5]`
   yields a token id outside `[0, 2**b - 1]` -- an out-of-vocabulary id that an
   embedding lookup will not flag the way a range check would.
5. **Truncation costs `SCALE_R` its top token.** Because SkexGen floors, the top
   token is only reached when `(x-min)*R/(max-min)` lands exactly on `R`. It does
   for the binary-exact ranges, but **not** for `SCALE_R = 1.4`:
   `1.4*63/1.4` evaluates to `62.99999999999999`, so the largest scale quantizes
   to token **62** and token **63 is unreachable for that field**. This was
   caught by the selfcheck and then confirmed against a numpy replay of the
   upstream `quantize` -- it is a property of SkexGen, not of the port. A
   rounding codec would not lose it.

Also recorded: a docstring/code discrepancy present in **both** repos. The
quantizer docstrings say the output range is `[0, n_bits**2 - 1]` while every
code path uses `2**n_bits - 1`. At `bit 6` the two readings differ (35 vs 63);
the code is authoritative.

The module also carries a 16-slot field table naming the codec each DeepCAD
argument slot actually uses -- they are not uniform. Five distinct maps are in
play (signed affine, affine-after-pi, raw sketch-grid round-clip, a radius
variant floored at 1 so a degenerate zero-radius circle is unrepresentable, and
an unsigned half-range map for bbox size), plus three slots that are categorical
indices and never quantized at all.

### Skipped

* Arc-angle **inverse**: DeepCAD emits only the *difference* of two quantized
  arc angles (`Arc.to_vector`), so the repo defines no inverse for that codec.
  None was invented here.
* SkexGen's `add_noise` dither: nondeterministic, and a training-time
  augmentation rather than part of the codec.

---

## D -- kerf sibling packages (tess / topo / imports)

**Delivered:** `src/harnesscad/domain/geometry/volumes/topology_optimize.py`
(commit `2d46a2a`). **One** algorithm ported, not two -- see the honest
assessment below.

### License verdict

kerf root `LICENSE`: **MIT, Copyright (c) 2026 Imran Paruk**. Vendoring
permitted with attribution.

One wrinkle worth recording: kerf also ships a second license file,
`LICENSE-CLOUD`, which is **proprietary, all rights reserved**. It scopes only to
`packages/kerf-cloud/**`, `packages/kerf-billing/**` and `src/cloud/**`. None of
tess/topo/imports fall under it, and the port touches none of those paths -- but
a future mining pass over this repo must check that scope before assuming the
root MIT applies to a given file.

(Layout note: the repo is double-nested at `kerf-main/kerf-main/`, and the three
packages are `packages/kerf-tess/`, `packages/kerf-topo/`, `packages/kerf-imports/`
-- not top-level `tess/`, `topo/`, `imports/`.)

### Ported (1 of at most 2): SIMP topology optimization

Source: `packages/kerf-topo/src/kerf_topo/advanced.py` (1061 lines; ~450 form the
algorithm core). Pure Python -- only `math` and `typing` unconditionally; its two
external imports are guarded with working fallbacks. No numpy, scipy, OCC or
dolfinx on the core path. Deterministic: no RNG, fixed iteration counts,
bisection to a fixed tolerance.

**Why it cleared the bar:** nothing in the harness does structural optimization.
`domain/geometry/mesh/` repairs meshes and `domain/geometry/volumes/` handles
density and occupancy fields, but neither has an FE solve or a compliance
objective. This is a genuine capability gap, not an incremental improvement.

What came across: `Mesh2D`, `ke_q4`, `solve_spd_banded` (banded LDL^T, which is
what makes the FE step O(n*band^2) and a hermetic selfcheck affordable) with the
`solve_dense` partial-pivot fallback, the linear-hat density filter, `_oc_step`,
the symmetry / overhang / draw-direction constraints, `mbb_problem`,
`pareto_sweep`, and the three geometric oracles. kerf's never-raise dict contract
is preserved exactly as the source states it, including its one deliberate
exception: the internal solvers raise `ValueError`, caught at the
`_fe_compliance` boundary to trigger the dense fallback.

The selfcheck proves real properties rather than smoke -- KE symmetry, positive
semi-definiteness and the rigid-body translation null space; banded-vs-dense
agreement and zero residual; the filter as a partition of unity; the MBB
regression (compliance 343.8 -> 149.5 over 14 iterations, volume constraint held,
beating the uniform initial design); each manufacturing constraint shown to
**bind** and each oracle shown **non-vacuous**; and the never-raise contract
across 11 malformed inputs. The selfcheck was mutation-tested: breaking the OC
volume target and breaking KE symmetry each produce the expected failures.

### Deliberately skipped

* **`_mma_step`** -- offered as an MMA alternative to OC, but it is the OC update
  with an extra `be ** 0.5` damping term: no moving asymptotes, no subproblem, no
  dual update. Porting it under the name "MMA" would carry the overclaim across,
  so `optimize` here exposes only `update="oc"`.
* **`lattice_infill`** -- its portable path is a one-line Gibson-Ashby estimate
  (`0.5 * period * relative_density`); the real TPMS wall-thickness mapping lives
  in a sibling kerf package that is not vendored.
* **`kerf-topo/routes.py`** -- the FEniCSx SIMP loop is dolfinx + numpy + MPI;
  the meshing/NURBS/marching-cubes helpers are gmsh/OCC/skimage. Its three
  pure-Python islands (`_heaviside_filter`, `_oc_update`, `_heaviside_projection`)
  are strictly weaker duplicates of what `advanced.py` already does better
  (O(n^2) naive filter vs. the windowed one).

### The second port: assessed and declined

**`kerf-imports/heal.py` was the only other serious candidate, and it is
duplicative.** Its 8-stage pipeline is stitch / sliver-removal / edge-merge /
unify-normals / dedup / self-intersection-detect / non-manifold-detect /
hole-fill. The harness already has
`domain/geometry/mesh/repair_toolkit.py` -- itself ported from kerf's
`geom/mesh_repair.py` -- which delivers `weld_vertices`, `unify_normals`,
`fill_holes`, `remove_degenerate`, `decimate`, `is_closed`, `is_manifold` and
`repair_pipeline`. That is a **strict superset** of `heal.py`'s pipeline, and its
`decimate` is a real Garland-Heckbert QEM with accumulated 4x4 quadrics, whereas
`heal.py` has no decimation at all. Porting `heal.py` would have re-landed
existing capability under a second name.

Two further notes for anyone revisiting this trove:

* `heal.py` has a **contract wart**: `heal()` returns `{"model", "report"}` on
  success with **no `"ok": True` key**, but `{"ok": False, "reason"}` on failure.
  Callers must test `"ok" in result and not result["ok"]`, not
  `result.get("ok")`. Normalize on any future port.
* `heal.py::_detect_self_intersections` is explicitly a cheap proxy (AABB overlap
  + no-shared-vertex, capped at 100 pairs), **not** an exact intersection test,
  and will produce false positives. Do not port it as though it were exact.

Also declined: `kerf-imports/tools/mesh.py::_decimate`, advertised as "quadric
edge collapse" but carrying no quadric matrices and no priority queue -- it
rebuilds the full edge set and rescores every edge against every face each pass.
It is O(n^3)-ish and the name oversells it; the harness's existing QEM is better.

### Nothing to port: kerf-tess

`kerf-tess` (1482 lines) contains **zero geometry algorithms**. It is entirely
plumbing: an `asyncpg` NOTIFY-driven job worker plus a FastAPI route that shells
out to a Node sidecar running `occt-import-js`. Hard deps on asyncpg, fastapi,
pydantic and a Node runtime. Skipped in full.

### Surveyed but not pursued: kerf-imports parsers

`kerf-imports` is ~21k lines of source and is, unusually, **almost entirely
stdlib-only** -- a grep for `numpy|OCC|trimesh|scipy|ezdxf|rhino3dm|FreeCAD`
across the whole tree matches exactly four files (`export_3dm.py`,
`freecad/brep_importer.py`, `plugin.py`, `rhino3dm_route.py`). The hand-rolled
readers are real: JT v8/v10 (1031 lines), Parasolid X_T (972), DXF R12/R2004
writer (914), IBIS (785), QIF 3.0 (696), Allegro, gEDA, Eagle, PADS, and an
AP242 PMI/GD&T reader that works at the STEP text level with no kernel.

These are **parsers, not algorithms** -- porting one yields a file-format reader,
not a computation, and the harness already has `io/formats/step.py` +
`step_units.py` for the format it actually consumes. Their value is entirely
contingent on the harness needing to ingest JT / Parasolid / DXF, which is not a
current requirement. Left for a future pass with a concrete ingestion need to
justify it.
