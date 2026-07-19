# Deep read — Zoo/KCL and the JS / functional-CAD language family

Repo set B. Sources under `resources/cad_repos/` (gitignored, double-nested
`X-main/X-main/`). RapCAD deliberately skipped — handled by another agent.

Genuine read pass, not a grep sweep. File counts are **actual** `find -type f`
counts and disagree with the counts in the prior inventories. Volume is stated
honestly per repo; "nothing here" appears several times and is meant.

---

## modeling-app-main (7,292 files)

### LICENSE

`LICENSE` — **MIT**, "Copyright (c) 2023 The Zoo Authors". Verified.

### WHAT IT IS

Zoo's Design Studio: a Tauri/React front end over a large Rust core
(`rust/kcl-lib`, `rust/kcl-error`) that lexes, parses, type-checks and executes
KCL, driving a remote geometry engine over websocket. Prior audits gave it 29
index mentions and never read it end to end.

### READ

Read fully: `rust/kcl-lib/src/errors.rs` (568 lines), `rust/kcl-error/src/error.rs`
(418), `rust/kcl-error/src/lib.rs` (~200), `rust/kcl-lib/src/unit_conversion.rs`
(245), `rust/kcl-lib/src/execution/cad_op.rs` (118),
`docs/kcl-std/functions/std-solid-fillet.md`, and **all 54
`execution_error.snap` fixtures** (message lines of all 54; three read in full
with their paired `input.kcl` source context).

Read in targeted full sections: `execution/types.rs` (~390 of 2,785 lines — the
`NumericType` lattice and the conversion functions), `std/mod.rs`, `std/assert.rs`,
`std/solver.rs`, `frontend/trim.rs`, `execution/artifact.rs` (the enum),
`tests/cube/ops.snap`.

### SKIMMED-NOT-READ

Header/listing only: ~40 files (directory listings, the docs index, 290
simulation-test directories). **Not covered at all**: the other ~2,400 lines of
`types.rs` (the `RuntimeType` coercion machinery), all 34 `std/*.rs`
implementations, the 190 `docs/kcl-std/functions/*.md` individually,
`artifact.rs`'s 2,922-line body and its graph construction, `simulation_tests.rs`
(6,421 lines), the remaining 2,790 non-error `.snap` files, `sketch_solve.rs`,
`lint/`, `parsing/` beyond the suggestion sites, `mermaid_tests.rs` (1,373 lines).

**Realistically: ~10 files read fully, ~10 read in large targeted sections, ~40
header-read, of 7,292.**

### FINDINGS

**1. The 54 committed known-bad fixtures. Highest value here. NONE-verified.**

`rust/kcl-lib/tests/<name>/execution_error.snap`, each paired with an
`input.kcl`. Per `rust/AGENTS.md` ("Inspect generated outputs and check for
`execution_error.snap` before committing") these are **deliberately committed
known-bad pairs**, not accidents. Format is a miette render: error family line,
`× <type>: <message>`, then the offending source span with line and column.

Grouped by what they encode:

- **Units (3)** — `array_range_mismatch_units` ("Range start and range end have
  incompatible units: mm and cm"); `bad_units_in_annotation` ("Unexpected value
  for length units: `nm`; expected one of `mm`, `cm`, `m`, `in`, `ft`, `yd`" —
  the exact accepted set, note **no µm and no km**);
  `sketch_block_failed_unit_conversion` ("sketch variable initial value must be a
  length coercible to the module length unit number(mm), but found a number (deg)").
- **Solid-consumption / linear ownership (8)** —
  `consumed_solid_{appearance,binary_add,binary_subtract,clone,join_surfaces_consumed_input,join_surfaces_reuse_input,subtract_reuse_target}`
  all emit the templated *"`X` was already consumed by a `<op>` operation. The
  operation result is now in `Y`; use that for subsequent operations."* Plus
  engine-level `subtract_self_multiple_tools`: *"Each UUID in brepIds must refer
  to a valid Solid3D. This often occurs when the referenced body was consumed by
  another operation."* **This is a statically checkable ownership discipline
  with a repair message that names the correct replacement variable.**
- **Type/coercion (7)** — `add_arrays`, `argument_error`, `comparisons_multiple`,
  `panic_repro_cube` ("requires a value with type `TaggedEdge`, but found a
  unique ID (uuid)"), `error_inside_fn_also_has_source_range_of_call_site_recursive`,
  `array_push_item_wrong_type`, `ascription_unknown_type`.
- **Indexing (6)** — `invalid_index_fractional` ("1.2 is not a valid index,
  indices must be whole numbers >= 0"), `invalid_index_negative`,
  `invalid_index_str`, `invalid_member_object`, `array_index_oob`,
  `array_elem_pop_empty_fail`.
- **Geometry/engine (5)** — `execute_engine_error_return` ("Cannot solid extrude
  an open profile. Either close the profile, or use a surface extrude."),
  `error_large_fillet_radius` ("engine: Edge cut failed"),
  `consumed_solid_original_issue` ("The Zoo engine cannot handle this 3D
  intersection yet"), `surface_extrude_edge_merge_error`,
  `tangent_line_line_error` ("tangent() does not support Line/Line. Tangency
  requires at least one circular segment.").
- **Tag/edge referencing (3)** — `fillet_duplicate_tags` ("The same edge ID is
  being referenced multiple times"), `get_common_edge_of_segment_edge_tag`
  ("Tag `line1` refers to a sketch edge, but this operation requires a face
  tag"), `face_api_fillet_chamfer_tags_and_edge_refs` ("You must provide either
  'tags' or 'edges' to fillet edges, not both").
- **Scoping/module (7)** — `import_cycle1`, `import_file_not_exist_error`,
  `import_file_parse_error`, `import_only_at_top_level`,
  `export_var_only_at_top_level`, `var_ref_in_own_def`, `cube_with_error`.
- **Sketch-block scoping (4)** — `sketch_block_tags_do_not_leak_to_parent_from_{extrude,region}`,
  `sketch_block_unexpected_argument`, `sketch_block_modeling_command_is_error`.

**Why we want it.** `eval/bench/imports/zoo_kcl_manifest.py` imported the 100
**known-good** kcl-samples. This is the known-bad half of the same vendor's
corpus, with exact expected error text, and it is unmined. A refuse-honestly
harness is graded on the bad half.

**2. The `KclError` taxonomy and its retry rule. NONE-verified.**

`rust/kcl-error/src/error.rs:23-64` — 17 variants, each with its `#[error(...)]`
display string and a stable `serde(tag="kind", rename_all="snake_case")` wire
name: `Lexical, Syntax, Semantic, ImportCycle, Argument, Type, Io, Unexpected,
ValueAlreadyDefined, UndefinedValue, InvalidExpression, MaxCallStack, Refactor,
Engine, EngineHangup, EngineInternal, Internal`. `error.rs:193-213` gives the
canonical lowercase `error_type()` strings and `get_message()` (`:189`) is
exactly `"{error_type}: {message}"` — **error strings are machine-parseable by
prefix.**

The control policy, `error.rs:9-12, 66-70, 155-166, 371-377`:

```rust
const RETRYABLE_ENGINE_MESSAGE_MARKER_SETS: &[&[&str]] = &[
    &["modeling connection", "interrupted", "please reconnect"],
    &["modeling connection", "heartbeats",  "please reconnect"],
];
```

`is_retryable()` is true **only** for `EngineHangup | EngineInternal`.
`new_engine()` classifies by message: exact `"internal error"` (case-insensitive)
→ `EngineInternal`; all markers in any one set present → `EngineHangup`;
otherwise `Engine`. A ready-made retry-vs-abstain rule.

Severity/suggestion model, `kcl-error/src/lib.rs`:
`CompilationIssue{source_range, message, suggestion, severity, tag}`;
`Severity ∈ {Warning, Error, Fatal}`;
`Tag ∈ {Deprecated, Unnecessary, UnknownNumericUnits, None}`;
`Suggestion{title, insert, source_range}` with `apply()` doing literal
splice-replace — **a machine-applicable autofix format.** The concrete
suggestion table sits at `rust/kcl-lib/src/parsing/parser.rs:958, 968, 1364,
2838` (plus sites at `:847, 2859, 2882, 3081, 4138` and
`execution/kcl_value.rs:608`): "Replace `||` with `|`", "Replace `&&` with `&`",
"Replace `:` with `=`" (Tag::Deprecated), "Remove `=`" (Tag::Unnecessary).

**HOW verified NONE.** `src/harnesscad/io/formats/kcl.py:115` defines its own
unrelated `KclError(Exception)` emitter exception. Registry shows error
taxonomies for FreeCAD (`eval.bench.sequence.error_taxonomy`), CADReview
(`domain.programs.review.taxonomy`) and CADCodeVerify
(`agents.generation.feedback_taxonomy`) — none is KCL's, and none carries a
severity/tag/suggestion triple.

**3. The units-of-measure lattice. NONE-verified.**

`rust/kcl-lib/src/execution/types.rs`. `NumericType` is a four-point lattice:
`Known(UnitType) | Default{len, angle} | Unknown | Any`, where
`UnitType ∈ {Count, Length(UnitLength), Angle(UnitAngle), GenericLength, GenericAngle}`.

- `types.rs:976-995` `from_parsed` — the literal-suffix → type table: bare number
  → `Default{module len, module angle}`; `_` → Count; `Length`/`Angle` →
  Generic; `mm,cm,m,in,ft,yd` → `Known(Length)`; `deg,rad` → `Known(Angle)`.
- `types.rs:683-729` `combine_eq` — **conservative**: converts only when *both*
  sides are `Known`; anything else falls through to `Unknown`, silently.
- `types.rs:738-817` `combine_eq_coerce` — **permissive**: also converts `Known`
  vs `Default`. The doc comment states the policy outright: *"Prefer to use
  `combine_eq` if possible since using that prioritises correctness over
  ergonomics."* A deliberate, documented correctness/ergonomics split — the
  design decision itself is the finding.
- `types.rs:863-902` — `combine_mul` / `combine_div` / `combine_mod` dimensional
  algebra. `combine_div:881`: same type ÷ same type → `Known(Count)`, the correct
  dimensional-analysis rule.
- `types.rs:909-974` `combine_range` — the **only** combinator that errors rather
  than coercing; source of the `array_range_mismatch_units` fixture.
- Warning policy: mixing `Known(Angle)` with `Default` angle emits *"Prefer to
  use explicit units for angles"* under `annotations::WARN_ANGLE_UNITS` — **but
  is suppressed when the value is exactly `0.0`** (`:709, 718, 767, 778`), since
  zero converts safely. Same exemption in `combine_eq_array:831, 841-842`.
- `types.rs:1185-1232` — exact conversion tables, pivot-based (mm for metric,
  inch for imperial): `cm×10, m×1000 → mm`; `ft×12, yd×36 → in`; cross-system
  pivot `mm/25.4 → in`, `in×25.4 → mm`; `deg→rad = v/180·π`, `rad→deg = 180v/π`.
- `rust/kcl-lib/src/unit_conversion.rs:101-245` — the full vocabulary across six
  dimensions: Length(6), Volume(10), Mass(3), Area(8), Density(2), Angle(2).

**HOW verified NONE.** Grep for `NumericType|combine_eq|GenericLength|adjust_length`
over `src/harnesscad/` hits only Rust build binaries.
`domain.numeric.unit_expressions` is a CodeToCAD-style `LengthExp` evaluator with
no unknown/default/generic lattice, no mul/div dimensional algebra, and no
coerce-vs-error policy split. `io.formats.step_units` covers only the STEP 1000×
guard.

**4. Tolerance constants. Partial.**

| Constant | File:line | Value | Note |
|---|---|---|---|
| `DEFAULT_TOLERANCE_MM` | `std/mod.rs:743` | `1e-7` | default `tolerance=` for all modeling commands; corroborated by the fillet doc ("should not be changed from its default value of 10⁻⁷ millimeters") |
| `EQUAL_POINTS_DIST_EPSILON` | `std/mod.rs:748` | `2.3283064365386962890625e-10` | = 2⁻³² exactly. Comment: **"WARNING: This must match the tolerance in engine/cpp/engine/scene/constants.h"** — an explicit cross-boundary contract between the KCL layer and the C++ engine |
| `SOLVER_CONVERGENCE_TOLERANCE` | `std/solver.rs:44` | `1e-8` | sketch constraint solver; wired at `exec_ast.rs:1814` |
| `DEFAULT_TOLERANCE` (assert) | `std/assert.rs:137` | `1e-10` | `isEqualTo`/`isNotEqualTo`, user-overridable |
| `EPSILON_PARALLEL` | `frontend/trim.rs:24` | `1e-10` | |
| `EPSILON_POINT_ON_SEGMENT` | `frontend/trim.rs:25` | `1e-6` | |
| `EPSILON_COINCIDENT_TERMINATION_SNAP` | `frontend/trim.rs:26` | `5e-2` | notably coarse — a *snap radius*, not a compare epsilon |
| `DEFAULT_MIN_SIMILARITY` | `tooling/image_comparison.rs:1` | `0.99` | render-regression gate |
| `CONTROL_POINT_SPLINE_SAMPLES_PER_SPAN` | `std/solver.rs:45` | `24` | tessellation density |

`domain.geometry.parametric.chord_tolerance` and `domain.geometry.topology.sew`
have tolerance machinery, but sourced from the `arcs` Rust core, not these
values. **The 1e-7 mm default modeling tolerance and the 2⁻³² point-equality
epsilon are the two a KCL verifier most needs; both NONE-verified.**

**5. The operation stream and artifact graph. Partial — different schema.**

`rust/kcl-lib/src/execution/cad_op.rs:11-51` — `Operation` is a five-variant
**bracketed stream**, not a flat list:
`StdLibCall | VariableDeclaration | GroupBegin | ModuleInstance | GroupEnd`.
`GroupBegin`/`GroupEnd` nest, so a user-defined function call collapses to one
group. `stdlib_entry_source_range` (`:26-28`) deliberately re-points a node's
path at the *user-facing* call (`hole()`) rather than the internal stdlib call
(`subtract()`) — a **provenance/attribution rule** for replaying an op stream
back to a human. `cad_op.rs:53-117` `op_from_kcl_value` is the value→parameter
projection: geometry values (Plane/Face/Sketch/Solid/Helix/Segment) degrade to
bare `artifact_id` references; `SketchConstraint` and unsolved segments degrade
to `KclNone`.

The 382 `ops.snap` files (e.g. `tests/cube/ops.snap`) are JSON keyed by source
path → array of ops, each carrying `type`, `group.{name, unlabeledArg,
labeledArgs}`, and per-arg `{value:{type,value,ty}, sourceRange}` — where **`ty`
carries the full `NumericType`** (`{"type":"Default","len":"mm","angle":"degrees"}`).
A committed corpus of **(KCL program → typed operation stream) pairs**.

`rust/kcl-lib/src/execution/artifact.rs:558-580` — the `Artifact` enum, 21 node
kinds: `CompositeSolid, Plane, Path, Segment, Solid2d, PrimitiveFace,
PrimitiveEdge, PlaneOfFace, StartSketchOnFace, StartSketchOnPlane, SketchBlock,
SketchBlockConstraint, Sweep, Wall, Cap, SweepEdge, EdgeCut, EdgeCutEdge, Helix,
GdtAnnotation, Pattern`. The B-rep-aware distinctions a naive op-DAG lacks:
`Wall` vs `Cap` (side vs end face of a sweep), `SweepEdge`, and `EdgeCut` vs
`EdgeCutEdge` (the fillet/chamfer operation *and* the edge it produces).

**HOW verified partial.** `core.state.opdag` ("git for CAD"), `core.cisp.ops`,
`core.cisp.op_gate` and `domain.programs.validate.operation_schema` cover the
*concept*. But grep for `SweepEdge|EdgeCut|Solid2d|CompositeSolid|artifact_graph`
over `src/harnesscad/` returns **zero hits** — the 21-node artifact taxonomy, the
Wall/Cap/EdgeCutEdge distinctions, and the bracketed `GroupBegin/GroupEnd` shape
are NONE-verified.

**6. Stdlib semantics — names are covered, contracts are not. NONE-verified.**

`docs/kcl-std/` holds **190 function docs, 32 types, 24 consts, 17 modules**,
generated from source and (per `rust/AGENTS.md`) overwritten by tests, so they
track the implementation. Each carries a typed signature block, an **argument
table with type / description / Required yes-no**, a `### Returns` type, and
runnable examples.

`std-solid-fillet.md` shows what a verifier gets per function:
`radius: number(Length)` (unit-constrained), `tags?: [Edge; 1+]` (an
**array-with-minimum-length type**), the documented `tolerance = 10⁻⁷ mm`
default, an **experimental** marker on `edges`, a **"Deprecated as of KCL 2.0"**
marker on `legacyMethod`, and a `version` enum with per-value semantics (0 =
engine chooses, 1 = original, 2 = rolling-ball). Deprecation and experimental
markers are exactly what a generation-time gate needs in order not to emit them.

`AGENTS.md` documents the invariant that a stdlib function must appear in the
`std_fn` dispatch table (`std/mod.rs`, 174 match arms), have a KCL-side doc
comment with ≥1 example, and that the example be registered in
`example_tests.rs::TEST_NAMES` — so **every documented example is also a passing
committed test.**

`domain/spec/zoo_catalog.py` (251 lines) has `KCL_STD_FUNCTIONS` as bare
snake_case name tuples grouped by module, explicitly sourced from the `std_fn`
dispatch table. It has **no signatures, no argument types, no required/optional
flags, no unit constraints, no defaults, no deprecation or experimental markers,
no return types.** The 190 doc files are exactly that missing layer.

### ALREADY COVERED

`kcl.grammar` → `domain/spec/kcl_productions.py`. kcl-samples (the 100 known-good
programs) → `eval/bench/imports/zoo_kcl_manifest.py`. Gear formulas. Operation
streams *as a concept* → `core.state.opdag`, `core.cisp.ops`. ML feedback
vocabulary → `domain.spec.zoo_ml_feedback`. Async operation status →
`io.adapters.zoo_api.OperationStatus`.

*(Confirming the prior finding independently: the kcl-samples manifest count of
100 is correct. The 101st directory is `screenshots`.)*

### VERDICT

**mine-further, heavily.** This is the richest single repo in either set. Six
substantial findings, and the 54 known-bad fixtures alone justify the pass.
Roughly 90% of the repo remains unread; the unread parts most likely to pay are
`simulation_tests.rs`, the 190 individual stdlib docs, and `artifact.rs`'s
graph-construction body.

---

## kittycad.py-main (669) / .rs-main (106) / .ts-main (456) / .go-main (66) / cli-main (126) / Zoo-main (584)

### LICENSE

All six **MIT**, verified from each `LICENSE`: kittycad.py "Copyright (c) 2021
KittyCAD"; .ts "(c) 2022 KittyCAD"; .rs and .go "(c) 2021 The KittyCAD Authors";
cli-main "(c) 2021 The Zoo Authors"; Zoo-main "Copyright (c) 2026 tryAGI".

**Correction worth recording: `Zoo-main` is not Zoo's repository.** It is
tryAGI's third-party .NET SDK, auto-generated from Zoo's OpenAPI spec. Anything
in prior inventories treating it as a first-party Zoo source is mis-attributed.

### WHAT IT IS

Six client SDKs plus the `zoo` CLI over one OpenAPI spec. The spec itself is
already mined into a `zoo_catalog` module; this pass looked for what the spec
does not carry — error semantics, retry policy, and hand-written operational
knowledge.

### READ

kittycad.py: ~40 files fully, ~8 partial, of 669. kittycad.rs: 3 files read in
large targeted regions (`types.rs` is 20k+ lines; read the error module,
`ErrorCode`, and both org-dataset state machines), of 106. kittycad.ts: 24 fully
+ 3 partial, of 456. kittycad.go: 11 fully + 1 header, of 66. cli-main: 10 fully
+ 6 large regions, of 126. Zoo-main: 4 fully + 1 partial, of 584.

### SKIMMED-NOT-READ

kittycad.py's generated model classes beyond the enums and format types.
kittycad.rs's `types.rs` outside the three targeted regions (the great majority
of 20k+ lines). kittycad.ts's `__tests__/gen/**` (~190 generated files — and see
finding 8 for why they are worthless as fixtures). **Zoo-main: 559 of 568 `src/`
files are `.g.cs` generated output** and were not read; the only hand-written C#
is `Tests.cs` (17 lines) and `Generate.cs` (18), neither of which asserts
anything.

### FINDINGS

**1. `ErrorCode` — the API error taxonomy, with retry semantics in the doc
comments. NONE-verified.**

`kittycad.py-main/kittycad/models/error_code.py:4-52`; identical in Rust at
`kittycad.rs-main/kittycad/src/types.rs:8887-8933`. Eleven variants, and the
docstrings *are* the ladder:

| Code | Line | Documented action |
|---|---|---|
| `internal_engine` | :9 | "Graphics engine failed… **consider retrying**" |
| `internal_api` | :13 | "API failed… **consider retrying**" |
| `bad_request` | :17 | "geometrically or graphically impossible. **Don't retry** — read the error message and change your request" |
| `auth_token_missing` / `auth_token_invalid` | :21, :25 | expired / malformed token |
| `invalid_json` / `invalid_bson` | :29, :33 | malformed client payload |
| `wrong_protocol`, `message_type_not_accepted`, `message_type_not_accepted_for_web_r_t_c` | :37-49 | websocket protocol violations |
| `connection_problem` | :41 | transport |

**Why we want it.** A vendor-authored repair-vs-retry-vs-abstain table.
`bad_request` is a genuine refusal predicate: the request is *inherently
impossible*, so a repair loop must mutate the prompt, not resubmit.

**HOW verified NONE.** `eval.verifiers.kernel_preflight` defines its own
`ErrorCode` (`ZERO_VOLUME`, `NON_MANIFOLD`, `BBOX_NO_OVERLAP`, …) — a
geometric-kernel taxonomy, unrelated. `eval/reliability/error_contract.py:19` has
a generic `recoverability_for_error` with no Zoo mapping.

**2. `error_geometry_mismatch` — Zoo ships the harness's own thesis.
NONE-verified. Ranked highest of set B for strategic value.**

`kittycad.rs-main/kittycad/src/types.rs:18007-18042`,
`OrgDatasetFileConversionStatus`:

| Status | Meaning / action |
|---|---|
| `queued` / `in_progress` | in-flight — and *"if `started_at` passes a certain threshold, we assume it got dropped and **will retry**"* (a server-side dropped-job watchdog) |
| `canceled` | will not be converted |
| `success` | result at `output_path` |
| `error_user` | *"user providing a broken file, such as it being empty"* — **do not retry** |
| **`error_geometry_mismatch`** | *"raw KCL result whose geometry **diverged from the source model beyond the accepted threshold**"* |
| `error_unsupported` | *"didn't know how to handle the file. Should be retried **with a new converter version**"* |
| `error_internal` | *"other unrecoverable error. Should be retried **with a new converter version**"* |

**Why we want it.** Zoo runs a geometric-equivalence verifier on its own
CAD→KCL conversion and **fails the job when divergence exceeds a threshold**.
That is precisely this project's thesis, shipped in a vendor's production state
machine — and the three-way retry split (user's fault / retry as-is / retry only
after the *tool* changes) is a sharper policy than a boolean `retryable`.

Adjacent: `types.rs:17938-17989` `OrgDatasetFileConversionPhase` is an explicit
12-phase pipeline, indexed 0-11: `queued → zoo_generated_original_metadata →
snapshot_original → user_provided_metadata → manual_kcl_override →
convert_raw_kcl → zoo_generated_raw_kcl_metadata → snapshot_raw_kcl → salon →
zoo_generated_salon_kcl_metadata → snapshot_salon_kcl → completed`. Phase 3
discovers sidecar metadata (`.json/.yaml/.yml/.toml/.txt`) beside the source CAD
file; phase 4 honours a persisted **manual KCL override**; phase 8 "salon" is a
refactor pass producing *polished* KCL from raw KCL. **Zoo separates "correct
KCL" from "idiomatic KCL" and snapshots both** — a useful decomposition for a
generation pipeline that wants to grade those two properties separately.

**3. The unit enums — 14 of them, exact members. NONE-verified.**

`kittycad.py-main/kittycad/models/unit_*.py`:

- `UnitLength`: `cm, ft, in, m, mm, yd` — **no µm, no km**
- `UnitAngle`: `degrees, radians`
- `UnitArea`: `cm2, dm2, ft2, in2, km2, m2, mm2, yd2`
- `UnitVolume`: `mm3, cm3, ft3, in3, m3, yd3, usfloz, usgal, l, ml`
- `UnitMass`: `g, kg, lb` — only three
- `UnitDensity`: `lb:ft3`, `kg:m3` — **note the colon**, which must survive URL
  encoding; the Go SDK un-escapes `%253A` specifically for this
  (`kittycad.go-main/utils.go:63`, pinned by `utils_test.go:14-63`)
- plus `UnitForce, UnitPressure, UnitTorque, UnitEnergy, UnitPower,
  UnitTemperature, UnitCurrent, UnitFrequency`

**No conversion factors anywhere** — `unit_length_conversion.py` is a
*server-side async job record* (`input`, `input_unit`, `output`, `output_unit`,
`status`), not a table. Conversion is an API round-trip, not local arithmetic.
Worth recording as a negative: do not go looking for Zoo conversion factors.

**HOW verified NONE.** Zero hits for `UnitLength` / `usgal` / `kg:m3` across
`src/harnesscad/`. `domain.numeric.unit_expressions` is a different vocabulary.

**4. The format *option* matrix. Partial — names covered, options not.**

`kittycad.py-main/kittycad/models/output_format3d.py:19-106` and
`input_format3d.py:11-176`. Per format, which options are required vs optional:

| Export | coords | units | storage | presentation |
|---|---|---|---|---|
| `fbx` | — | — | `ascii\|binary` | — |
| `gltf` | — | — | `binary\|standard\|embedded` (embedded default) | `compact\|pretty` |
| `obj` | **required** | **required** | — | — |
| `ply` | **required** | **required** | `ascii\|binary_little_endian\|binary_big_endian` | — |
| `step` | optional (default fwd `-Y` / up `+Z`) | optional, **default `m`** | — | optional, default `pretty` |
| `stl` | **required** | **required** | `ascii\|binary` (binary default) | — |

Three things a verifier wants:

- **`OptionStep.units` defaults to `"m"`** (`output_format3d.py:77`) while
  KCL/modeling defaults to mm. That is the **silent 1000× trap**, and this is its
  *cause*. `io.formats.step_units` detects the symptom; it does not know Zoo's
  default produces it.
- **Import-side coordinate defaults are not uniform.** ACIS/CATIA/Inventor/NX/
  Parasolid/STEP default to forward `-Y` / up `+Z`, but **Creo and SLDPRT default
  to forward `+Z` / up `+Y`** (`input_format3d.py:40-43, :124-127`). Import a
  SolidWorks part assuming the common default and it lands rotated 90°.
- **`Axis` admits only `y` and `z` — there is no `x`** (`models/axis.py`).
  Forward and up cannot be X-aligned. A one-line validity predicate.

Also: CAD-native imports carry `split_closed_faces: bool = False`; mesh imports
(obj/ply/stl) do not. And a documented per-format limitation in
`file_import_format.py` / `file_export_format.py`: OBJ "may or may not have an
attached material (mtl // mtllib)… **we interact with it as if it does not**" —
Zoo drops materials.

`io.formats.registry` and `domain.spec.zoo_catalog` have the format *names* and
the conversion matrix; neither has options, defaults, or coord conventions.
`io/formats/usd.py` has a single `up_axis`, never a forward+up pair.

**5. Retry and poll ladders — four of them, and they disagree. NONE-verified.**

The only real backoff in an official SDK is Rust,
`kittycad.rs-main/kittycad/src/lib.rs:211-232` (wasm twin at `:269-290`):

```rust
// Retry up to 3 times with increasing intervals between attempts.
let retry_policy = reqwest_retry::policies::ExponentialBackoff::builder()
    .build_with_max_retries(3);
```

wrapped in `ConditionalMiddleware` with the predicate
**`|req| req.try_clone().is_some()`** — *only retry requests whose body can be
replayed*, so streaming and multipart uploads are silently non-retryable. Behind
an opt-in `retry` cargo feature (`lib.rs:171-186`); backoff base and max are
`reqwest_retry` defaults, not pinned here.

Everything else has no backoff at all:

| SDK | Timeout | Poll | Retry |
|---|---|---|---|
| Python (`client.py:49-50, 108-134`) | 120 s request, 60 s ws-recv | 60 s deadline / 2 s interval | none |
| CLI (`context.rs:188-199, 226-227`) | 600 s request, 60 s connect, `WS_RESPONSE_TIMEOUT_SECS = 600` | **5 min / 5 s** | **`RetryConfig::no_retries()` under `#[cfg(not(test))]`** — retries exist only in tests |
| Go (`lib.go:51-55, 300-328`) | 600 s, *"since some of the files might take a bit"* | 3 min / 2 s (test only) | none; status code **never** consulted for retryability |
| TS (`src/client.ts`, `errors.ts:19-33`) | none | none | none; throws `ApiError` on any non-`res.ok` without status-class distinction |

**There is no shared policy to inherit — these must be chosen, not merged.** The
CLI's predicate is the strictest: `Self::Other(_) => false` (`context.rs:111-119`)
makes any non-KCL error categorically non-retryable.

Python's status classification (`response_helpers.py:19-112`): 4xx →
`KittyCADClientError`, 5xx → `KittyCADServerError`, else `KittyCADAPIError`; the
named table at `:30-43` includes `429 Too Many Requests` and `402 Payment
Required`; `request_id` from the body falling back to the `x-request-id` header
(`:67-68`). It classifies but never retries.

Async status vocabulary is `queued, uploaded, in_progress, completed, failed`
(`models/api_call_status.py`), terminal set exactly `{completed, failed}`,
lowercased via `_normalize_operation_status` (`client.py:12-21`) — which matters,
because the Go fixtures emit `"Uploaded"` capitalised.

**Already covered:** `io.adapters.zoo_api.OperationStatus` and
`domain.spec.zoo_ml_feedback.ApiCallStatus` both model the status set correctly.

**6. The CLI's refusal predicates and limits — the most concrete control
policies in the family. NONE-verified.**

All hand-written, all in `cli-main`:

- `COPILOT_PROJECT_ENTRY_LIMIT = 25` (`cmd_ml/cmd_kcl.rs:133`) — counts **files
  *and* directories**, iterative DFS, bails on overflow with `"Copilot needs a
  smaller project"`. A hard context-budget gate.
- `join_secure` (`ml/copilot/util.rs:17-60`) — a three-layer path-escape check:
  reject absolute, lexical `..` containment, then canonicalize the nearest
  *existing* ancestor to catch symlink escapes. Distinct message per layer.
  `SCAN_MAX_DEPTH = 256`; skips `.git`/`target`/`node_modules`/dotdirs.
- `FILES_TO_SEND_TO_ENGINE` (`build_kcl_project.rs:11-14`) — the 18-extension
  engine-upload allowlist: `kcl, fbx, glb, gltf, obj, ply, step, stl, sat, sab,
  model, catpart, ipt, prt, xpr, x_t, x_t, sldprt`. **`x_t` appears twice and
  `x_b` is absent** — almost certainly a real bug: Parasolid *binary* cannot be
  uploaded.
- `get_input_format` (`cmd_file.rs:747-772`) — the CLI's actual import matrix is
  narrower than the enum: only Step/Stl/Obj/Gltf/Ply/Fbx/Sldprt, everything else
  → `"Zoo CLI cannot yet handle filetype {other}"`. Units hardcoded to
  `Millimeters` with a `// TODO` at `:227`.
- **The `--allow-errors` gate** (`kcl_error_fmt.rs:31-89`):
  `KclIssueCheck ∈ {DenyErrors, AllowErrors, Ignore}`, default Deny; all issues
  printed, then fail. The help text warns *"Some errors are fatal and are not
  affected by this option. Which errors are considered fatal may change without
  notice."* — an explicit, and explicitly unstable, two-tier severity model.
- **`--deterministic`** (`cmd_kcl.rs:1847-1866`): pins export `created` to epoch
  for **FBX and STEP only** (the only formats carrying the field), then
  regex-scrubs timestamps from UTF-8 output. Directly relevant to byte-stable
  fixture generation.
- Export defaults (`cmd_kcl.rs:804-881`): forward `-Y` / up `+Z` right-handed;
  **STL and PLY export ASCII**, FBX binary; `glb` = Binary+Compact vs `gltf` =
  Embedded+Pretty; units come from the KCL program's
  `meta_settings.default_length_units`, not a flag.
- Error surfacing (`main.rs:306-347`) — the only status→user-action mapping in
  the family: 401 → *"Try authenticating with: `zoo auth login`"*; 403 → not
  authorized. **Exit codes are binary, 0 or 1** — no distinct code for auth vs
  validation vs timeout.

The CLI poll loop (`context.rs:905-981`, twin at `:1001-1082`) has three distinct
failure exits, and the middle one is worth mirroring: `Failed` with an error →
*"Your prompt returned an error: ```…```"*; **`Failed` with no error message →
*"Your prompt returned an error, but no error message. :("*** — an observed real
case; deadline exceeded → *"Your prompt timed out"*. A shape mismatch in the
response variant is fatal (`bail!("Unexpected response type")`), not retried.

**7. Committed fixtures. NONE-verified.**

- `cli-main/src/tests.rs` — a **physics golden corpus** against `tests/gear.kcl`:
  volume `0.05360` (:655), mass `1164.67` (:675), density `0.00085` (:695),
  surface area `1.088` (:711), COM `x: -0.0133` (:727). Plus a clean
  known-good/known-bad pair: `tests/non_fatal_error.kcl` succeeds under
  `--allow-errors` (:1076) and fails without (:1049). `cmd_file.rs:794-895` has
  exact-string negative cases for bad extension and missing file.
- `kittycad.ts-main/__tests__/main.test.ts:14-31` — `example.obj` (37 KB,
  committed) with a golden mass-properties value: `material_density: 0.007 kg:m3`,
  `output_unit: 'g'` → `mass === 1.0375403388552853e-7`. Caveat, stated honestly:
  the assertion sits inside a `try/catch` that passes on any `ApiError`, so it is
  **not enforced in their CI** — but the number is real and reusable.
- `kittycad.go-main/paths_test.go:8-15` — the one known-bad payload: an in-flight
  conversion record with **empty strings where timestamps belong**
  (`"completed_at":"", "started_at":"", "status":"Uploaded"`) that must unmarshal
  without error. The `json_*_test.go` set (5 files, 14 tests) pins a uniform
  tri-state contract for every scalar wrapper: valid / `""` / `null`-and-nil all
  parse.
- `kittycad.py-main/kittycad/tests/test_request_serialization.py:36-63` — a
  known-good `ConversionParams` STL→OBJ payload with explicit
  `System(forward=Y/NEGATIVE, up=Z/POSITIVE)` and `UnitLength.MM`.
  `test_org_dataset_models.py:38-60` — a **deliberately known-bad** payload: a
  response carrying `future_api_field` must raise `ValidationError` under strict
  validation and pass only with `extra="ignore"`. A forward-compat contract.
  `test_ml_copilot_ws_regression.py:38-49` — a bounded-stream policy: iterate **at
  most 200 messages**, fail if no `EndOfStream`, treat a server `Error` frame as
  fatal. Terminal vocabulary `ConversationId | Delta | EndOfStream | Error`.

`kittycad.go-main/lib_test.go:165-208` is the best single artifact in the Go SDK:
a complete hand-written async state machine with the part worth copying —
**`default:` → `t.Fatalf("unexpected async operation status")`**. An unknown
status is fatal, not "keep waiting." The CLI does the opposite (loops to
deadline). The stricter choice is the right one for a verifier.

**8. `gen/expectedToFail.ts` — a maintained map of which Zoo endpoints are
unreliable. NONE-verified.**

`kittycad.ts-main/gen/expectedToFail.ts` (251 lines) is not boilerplate, it is an
ops document:

- `:27-33` — *"All of these **randomly timeout. Unacceptable nondeterminism.**"*
  naming six endpoints.
- `:75-81` `expectedToTimeout` — endpoints **expected to blow the 60 s test
  timeout**, including `ai.create_image_to_3d` and `ai.create_text_to_3d`. The
  generative endpoints time out by design.
- `:103-250` `testsExpectedToThrow` (~150 entries) includes all three
  `ml.create_text_to_cad*` ops and every `file.create_file_*`
  mass/volume/density/COM op.
- `:85-88` — **`testsExpectedToSucceed` is an empty array**, and
  `gen/apiGen.ts:769-772` inverts it so every generated test asserts "truthy
  **or** ApiError". **Their generated suite proves nothing about success.** Treat
  `__tests__/gen/**` (~190 files) as a compile check only — worth knowing before
  anyone mines it as a fixture source.

**9. text-to-cad operational facts. Partly covered.**

From `kittycad.rs-main/kittycad/src/ml.rs:25-40`:

- *"Because our **source of truth for the resulting model is a STEP file**, you
  will always have STEP file contents when you list your generated parts."* → a
  verifier should grade the STEP, not the requested mesh format.
- *"if you **hit the cache**, this endpoint will return right away."* → a poll
  optimisation, and a determinism caveat for benchmarking.
- All three REST text-to-cad ops are **marked deprecated** in favour of
  `/ws/ml/copilot` (`:25`, `:499`, `:530`).
- Iteration semantics (`:499`): *"Even if you give specific ranges to edit, the
  model might change more than just those… You always get the whole code back."*
- Multi-file (`:530`): only changed `.kcl` files are returned, non-KCL imports
  never; and *"Input filepaths will be normalized and re-canonicalized to be
  under the current working directory — care must be taken when handling user
  provided paths"* (a path-traversal caution, and the reason `join_secure` exists).

Schemas: `models/text_to_cad_create_body.py` — `prompt: str` required;
`kcl_version`, `model_version`, `project_name` optional. Response
`models/text_to_cad.py:13-48`: `code`, `conversation_id`, `error`,
`feedback: MlFeedback`, `model: TextToCadModel ∈ {cad, kcl, kcl_iteration}`,
`outputs: Dict[str, Base64Data]`, `status`.

**Prompt constraints: none exist.** `prompt` is a bare `{type: "string"}` in the
spec — no `maxLength`, no `minLength`, no pattern — and no SDK does client-side
validation. The only `maxLength` near ML is `1..80` on a feedback text field
(`kittycad.ts models.ts:10162`). **If the harness wants a prompt-length refusal
predicate it must invent one; Zoo does not supply it.** A useful negative.

**Already covered:** the feedback vocabulary `thumbs_up|thumbs_down|accepted|rejected`
→ `domain.spec.zoo_ml_feedback`, already reframed as an acceptance metric.

**10. TS websocket protocol control. NONE-verified.**

Hand-written, `kittycad.ts-main/src/worker-zookeeper.ts`: ping every **4000 ms**
gated on `readyState === OPEN` (`:219-224`); auth is an **in-band first frame**
`{type:'headers', headers:{Authorization:…}}`, not an HTTP header
(`ml_copilot_ws.ts:51-69`). Two hard-won filters:

- `:30-31, :182-184` — the error *"Please send `{ headers: { Authorization: ... } }`
  over this websocket."* is a **known backend false positive / race** and is
  silently dropped, never surfaced.
- `:24-25, :188-201` — errors containing `conversation not found` /
  `Invalid conversation_id` are re-emitted as `invalid_conversation` so the
  client drops the id and reconnects.

Also: copilot uses **MsgPack**, modeling uses **BSON** — two wire codecs in one
SDK. `models.ts:5814-5815` exposes `enable_dry_run` / `disable_dry_run` as
modeling commands, directly useful for validating ops without committing them.
Batch semantics: *"If any request fails, following requests will not be tried"*,
and `responses` defaults `false`.

### Zoo-main (tryAGI .NET) — near-empty, with two survivors

Judged as instructed: 559 of 568 `src/` files are `.g.cs` generated bindings and
are **not findings**.

1. **`src/libs/Zoo/generate.sh` is a genuine hand-written policy file.** `:9-13`
   — spec-fetch retry with hard numbers: `--retry 5 --retry-delay 10
   --retry-all-errors --connect-timeout 30 --max-time 300`. `:26-39` — renames
   `components.schemas.System` → `CoordinateSystem` and rewrites all `$ref`s,
   because `System` is a reserved C# namespace. Relevant if the harness ever
   codegens Zoo's spec.
2. **A verifiable structural fact: this SDK cannot generate CAD.**
   `generate.sh:41-48` selects six operations including all three text-to-cad
   POSTs, but `:114` passes `--exclude-deprecated-operations`, and all three are
   `deprecated: true` in Zoo's live spec — so they were silently dropped. The
   only generated operation files are `GetTextToCadPartForUser`,
   `ListTextToCadPartsForUser`, `GetAsyncOperation`. **This independently
   corroborates the deprecation signal** from the Rust and TS SDKs: Zoo has moved
   text-to-CAD generation to the copilot websocket, and a third party
   regenerating the spec today gets a read-only client.

Its generated AutoSDK runtime contains the family's only *complete* backoff
policy (`Zoo.OptionsSupport.g.cs:831-843`): `InitialDelay 1s`, `MaxDelay 30s`,
`BackoffMultiplier 2.0`, `JitterRatio 0.2`, `UseRetryAfterHeader = true`, retried
codes **408/429/500/502/503/504**. But **`MaxAttempts = 1` (`:357`) — retries are
off by default**, and `Timeout` defaults to `null`, so an out-of-the-box client
hangs indefinitely on a stalled connection. Generated boilerplate; cited only
because the status-code set is a useful reference. Nothing else: no Polly, no
handlers, no fixtures, no websocket support.

### VERDICT

kittycad.py — **mine-further** (unit enums, format option matrix, ErrorCode).
kittycad.rs — **mine-further** (the geometry-mismatch state machine and the only
real retry ladder). cli-main — **mine-further** (the refusal predicates, limits,
and the physics golden corpus). kittycad.ts — **mine-further, narrowly**
(`expectedToFail.ts` and the websocket filters; its generated test suite is
worthless as fixtures and that is itself worth recording). kittycad.go —
**mine-further, narrowly** (one excellent state-machine test, one known-bad
payload). Zoo-main — **nothing-here**, with two cited exceptions; it is a
generated third-party binding and is mis-attributed in prior inventories.

---

## OpenJSCAD.org-master (2,012 files)

### LICENSE

`LICENSE` — **MIT**, "Copyright (c) 2017-2024 JSCAD Organization". Verified.

### WHAT IT IS

A JS monorepo; `packages/modeling` is a pure-JavaScript BSP-tree CSG kernel with
no native dependency. That makes its tolerance choices unusually legible — every
epsilon is a literal in readable source rather than buried in a C++ build.

### READ

**Read 24 files fully, header/section-read 9 more, of 2,012.** Vendored code
confirmed and excluded: `packages/modeling/src/operations/extrusions/earcut/`
(a Mapbox earcut port).

### SKIMMED-NOT-READ

`packages/web`, `packages/desktop`, `packages/cli` beyond entry points — app
glue. The ~1,900 remaining files, chiefly the per-function test files and the
372 colocated `.d.ts` declarations (enumerated, not read individually).

### FINDINGS

**1. The tolerance ladder — three constants, and two coplanarity predicates that
disagree. NONE-verified.**

`packages/modeling/src/maths/constants.js:7-25`:

```
spatialResolution = 1e5    // "resolution of space, currently one hundred nanometers"
EPS  = 1e-5                // near-zero distances, = 1/spatialResolution
NEPS = 1e-13               // "derived from a series of tests ... optimal precision for
                           //  comparing coplanar polygons, as provided by the sphere
                           //  primitive at high segmentation. NEPS is for 64 bit Number"
```

Two more magic numbers exist **outside** `constants.js`:

- `operations/modifiers/retessellate.js:47` —
  `tolerance = component === 3 ? 0.000000015 : NEPS`. The plane's
  **distance-from-origin component gets 1.5e-8 while the three normal components
  get 1e-13.** A deliberate asymmetry: normals are unit-length so they need a
  tighter *absolute* epsilon than a distance that scales with the model.
- `operations/modifiers/mergePolygons.js:180-185` — a *different* coplanarity
  test: `Math.abs(plane1[3] - plane2[3]) < 0.00000015` (1.5e-7, **ten times
  looser** than retessellate's) then `aboutEqualNormals`
  (`maths/utils/aboutEqualNormals.js:10`, NEPS per component).

**Two coplanarity predicates with different distance tolerances coexist in the
same package.** Why we want it: a worked example of a scale-split tolerance
policy — angular and positional epsilons must be separate scalars — plus a live
instance of the bug class where the "same" predicate drifts between call sites,
which is exactly what a verifier should assert against.

**HOW verified NONE.** `domain.geometry.topology.sew` and
`domain.geometry.parametric.chord_tolerance` exist but neither carries a
normal-vs-distance split coplanarity constant pair. `registry.index()` returns
zero hits for `epsilon`.

**2. Scale-relative epsilon derived from the bounding box. NONE-verified.**

`packages/modeling/src/measurements/calculateEpsilonFromBounds.js:3-9`:

```js
for (i < dimensions) total += bounds[1][i] - bounds[0][i]
return EPS * total / dimensions      // mean extent × 1e-5
```

Consumed by `measureEpsilon.js:11-23` (2D for path2/geom2, 3D for geom3) and used
as *the* epsilon for `snap`, `snapPolygons`, `mergePolygons` and
`triangulatePolygons` (`operations/modifiers/generalize.js:32`).

**Why we want it.** A verifier that hardcodes an absolute epsilon passes a 1 mm
part and fails a 1 m part. This is the cheapest correct fix: one function, mean-extent
scaled. Independently corroborated by OpenCADStudio's `CURVE_REL_TOL` (below) —
two repos converging on "tolerance must scale with the feature."

**HOW verified NONE.** Grep over `src/harnesscad/` for "epsilon.*bounds" /
"relative epsilon" returns nothing; `exploded_view.py` uses `bounds_radius(...)`
only to clamp *against* a fixed `EPSILON`, which is the inverse pattern.

**3. Vertex welding is snap-to-grid, then area-filter — with a non-obvious
length→area derivation. NONE-verified.**

`maths/vec3/snap.js:10-14` — `Math.round(v/eps)*eps + 0` (the `+ 0` normalizes
`-0`). `operations/modifiers/snapPolygons.js:70-86` — snap all vertices, drop
consecutive duplicates via **exact** `vec3.equals` (which is `===`,
`maths/vec3/equals.js:26`) *because snapping made them bit-identical*, then filter
degenerate polygons:

```js
const epsilonArea = (epsilon * epsilon * Math.sqrt(3) / 4)   // area of an equilateral
                                                             // triangle with side = epsilon
newpolygons.filter((p) => Number.isFinite(area) && area > epsilonArea)
```

The `sqrt(3)/4` is the part worth having: a length tolerance must be squared
**and shape-corrected** before it is a valid area tolerance. Verifiers routinely
get this wrong by using `eps*eps`. NONE-verified.

**4. Tree-CSG failure taxonomy — the coplanar fast path is exact float equality.
NONE-verified.**

`packages/modeling/src/operations/booleans/trees/splitPolygonByPlane.js:49`:

```js
if (plane.equals(pplane, splane)) { result.type = 0 }   // coplanar-front
```

`maths/plane/index.js:27` maps `plane.equals` to `vec4/equals.js:37`, four `===`
comparisons. **So the boolean's coplanar fast path only fires when two planes are
bit-identical.** Non-identical-but-coplanar faces fall through to the per-vertex
`±EPS` classification at `:56-62` — which is precisely why `retessellate` (with
its own, looser ladder) must run afterwards. `retessellate.js:7-8` states it:
"After boolean operations all coplanar polygon fragments are joined by a
retesselating operation."

Adjacent scar tissue in the same subsystem:

- `unionGeom3Sub.js:20-25` — a **commented-out line with an ERROR annotation in
  shipped code**:
  ```js
  a.clipTo(b, false)
  // b.clipTo(a, true); // ERROR: doesn't work
  b.clipTo(a); b.invert(); b.clipTo(a); b.invert()
  ```
  The "remove coplanar front" optimisation is known-broken for union, and the
  four-call dance is the workaround.
- `splitPolygonByPlane.js:10, 21` — `EPS_SQUARED = EPS*EPS` with
  `vec3.squaredDistance` drops duplicate vertices produced by the split; then
  `>= 3` vertex guards at `:106-116` **silently drop the front or back fragment**
  if the split degenerated. Silent polygon loss, no error.
- `Node.js:107-113` — splitting-plane choice is `Math.floor(len/2)`, with random
  and index-0 alternatives commented out. `:125, :133` — `stopCondition` when a
  node cannot split its own polygon set: the guard against infinite BSP recursion
  on degenerate input.
- `mayOverlap.js:26-31` — bbox rejection uses `> EPS` per axis; if false,
  `unionForNonIntersecting` **concatenates polygon lists with no boolean at all**
  (`unionGeom3Sub.js:34-38`), commented "Do not use if you are not completely sure
  that the solids do not intersect!" — a short-circuit that yields a non-manifold
  union of merely-touching solids.

**HOW verified partial→NONE.** `domain.geometry.sdf.csg_algebra` / `csg_eval` are
SDF-based, not BSP; `io.backends.manifold` delegates to Manifold. **No BSP /
tree-CSG failure-mode catalogue exists** — `registry.index()` returns one hit for
"bsp" and it is `data.datagen.bspline_metrics` (unrelated). `occt_quirks.py`'s
eight quirks are all OCCT/CadQuery-sourced; none covers tree-CSG.

**5. Committed validators — a ready-made ordered assertion ladder. Partial.**

`packages/modeling/src/geometries/poly3/validate.js:19-61` throws, **in order**,
with distinct messages:

1. `invalid poly3 structure`
2. `poly3 not enough vertices N` (< 3)
3. `poly3 area must be greater than zero`
4. `poly3 duplicate vertex V` (exact `vec3.equals`)
5. `poly3 must be convex`
6. `poly3 invalid vertex V` (non-finite)
7. `poly3 must be coplanar: vertex V distance D` — **only checked when
   `vertices.length > 3`**, threshold `NEPS` (1e-13)

`geometries/geom3/validate.js:34-59` — `validateManifold`: count directed edges
via string keys `"v1/v2"`, then require `count(edge) === count(reversed edge)`;
the error enumerates every offending edge. `:28` carries the honest gap:
`// TODO: check for self-intersecting`. `geometries/geom2/validate.js:83` —
closedness is checked *indirectly*, by calling `toOutlines()` and letting it throw.

`eval.bench.geometry.mesh_topology` covers the manifold-edge count. **The ordered
poly3 predicate ladder with per-failure messages, and the `>3 vertices`
coplanarity carve-out, are NONE-verified.**

**6. Two silent-wrong-answer traps in the measurement API. Partial.**

`measurements/measureVolume.js:39` sums `poly3.measureSignedVolume` over all
polygons with **no manifold precondition** and a `WeakMap` cache (`:8, :41`).
Signed volume on a non-manifold mesh returns a plausible float, not an error.
`measureBoundingSphere.js:125` documents itself as the **"(approximate)"**
bounding sphere: centroid-of-all-vertices plus max radius (`:99-115`), **not** a
minimal enclosing sphere — so it must never be used as a containment proof.
`measureVolume` returns 0 for path2 and geom2 by definition (`:17, :26`).

`eval.hardcorpus.occt` is described as "exact measurements on the solid the model
actually built"; the **vertex-centroid-not-minimal bounding sphere caveat is
NONE-verified.**

**7. Export hygiene: T-junction insertion is mandatory. NONE-verified.**

`packages/io/stl-serializer/index.js:56-65`:

```js
if (objects3d.length === 0) throw new Error('only 3D geometries can be serialized to STL')
objects3d = toArray(modifiers.generalize({ snap: true, triangulate: true }, objects3d))
```

`generalize` (`operations/modifiers/generalize.js:36-50`) runs, in this fixed
order: `snapPolygons(epsilon)` → (`simplify` off) → **`insertTjunctions`** →
`triangulatePolygons(epsilon)`. So **T-junction insertion runs before
triangulation on every STL export** — snapping alone leaves T-junctions that
become visible cracks after fan triangulation.

`packages/io/json-serializer/index.js:31-47` — the JSON round-trip's corner case
is typed arrays: `transforms`/`plane` → `Array.from`, `points`/`vertices` →
per-element `Array.from`, `sides` → nested. Anything not in that switch survives
as-is, so **a typed array under an unlisted key serializes as
`{"0":…,"1":…}`** — a silent format corruption.

**8. The `.jscad` script API surface — already covered.**

There *is* a machine-readable catalogue: **372 `.d.ts` files** colocated under
`packages/modeling/src` (`src/index.d.ts` is only 18 lines — a namespace
re-export; the real surface is per-function). Parameter constraints are a
documented schema: `jsdoc/tutorials/03_usingParameters.md` enumerates every
`getParameterDefinitions()` type with its constraint keys — `int`/`number`/`slider`
take `min`/`max`/`step`; `text`/`url` take `size`/`maxLength`/`placeholder`;
`date` takes `min`/`max` as ISO strings; `choice`/`radio` take parallel
`values`+`captions`; `group` takes `initial: 'closed'`. Parsing lives in
`packages/core/src/parameters/getParameterDefinitionsFromSource.js` (+ its test).

**ALREADY COVERED** — `domain.programs.params.param_schema` explicitly ingests
"JSCAD `getParameterDefinitions()` output" (module docstring line 6, and the
`# JSCAD getParameterDefinitions()` section at line 175).

### EXPLICIT NEGATIVES

- **No known-bad fixture corpus.** `packages/modeling/test/` and
  `packages/io/test/` contain **only `helpers/`**. Boolean test data is inline in
  colocated tests (`unionGeom3.test.js` 10.7 KB, `subtractGeom3.test.js` 8.0 KB,
  `intersectGeom3.test.js` 8.1 KB, `unionGeom2.test.js` 7.5 KB, plus
  `trees/splitPolygonByPlane.test.js` and `mayOverlap.test.js` 3.6 KB). These are
  expected-vertex-array assertions, not known-bad crash inputs.
- **No transferable serialization format.** The sole repo-wide mention is a
  comment at `packages/core/src/code-evaluation/rebuildGeometry.js:27` explaining
  why they *didn't* use transferables. Recorded so it is not chased.

### VERDICT

**mine-further.** The tolerance findings (1, 2, 3) and the tree-CSG failure
taxonomy (4) are the payload. The parameter schema is already covered.

---

## replicad-main (393 files)

### LICENSE

`LICENSE` — **MIT**, "Copyright 2023 QuaroTech Sàrl". Verified (full MIT grant
text, no attribution clause beyond the standard notice).

### WHAT IT IS

A TypeScript API over OCCT compiled to WebAssembly, plus its own 2D blueprint
layer. The interesting part is not the API — it is the wasm boundary, where OCCT
exceptions arrive as bare integers.

### READ

**Read 8 files fully, section-read 9 more, of 393.** Skipped
`packages/replicad-opencascadejs/src/*.d.ts` (9,000-line generated Emscripten
bindings) except targeted lookups; skipped `studio/` and
`replicad-app-example/` as app glue.

### FINDINGS

**1. The wasm exception-recovery recipe — the standout finding here.
NONE-verified.**

replicad ships **two OCCT wasm builds from one binding list**:

- `packages/replicad-opencascadejs/build-source/custom_build_single.yml:3` —
  `buildFlags.extend(["-sDISABLE_EXCEPTION_CATCHING=1"])`. Fast and small; an
  OCCT throw becomes an opaque trap.
- `packages/replicad-opencascadejs/build-source/custom_build_with_exceptions.yml:3-19`
  — adds the `Standard_Failure` symbol plus injected C++:
  ```cpp
  class OCJS {
    static Standard_Failure* getStandard_FailureData(intptr_t exceptionPtr) {
      return reinterpret_cast<Standard_Failure*>(exceptionPtr);
    }
  };
  ```

Consumed at `packages/replicad-evaluator/src/builder.ts:22-44`:

```ts
if (typeof error === "number") {            // a raw wasm exception pointer
  if (oc?.OCJS) message = oc.OCJS.getStandard_FailureData(error).GetMessageString();
  else          message = `Kernel error ${error}`;
}
```

**Under wasm, an OCCT failure surfaces to JS as a bare integer.** Without the
reinterpret-cast shim you get `Kernel error 5243184`; with it you get OCCT's own
`GetMessageString()`.

Shared build flags (`build-source/defaults.yml:295-301`): `-flto -fexceptions
-sEXPORT_ES6=1 -sUSE_ES6_IMPORT_META=0 -sALLOW_MEMORY_GROWTH=1
-sEXPORTED_RUNTIME_METHODS=["FS"] -O3`. Note `-fexceptions` is in the *shared*
defaults and the "single" build adds `-sDISABLE_EXCEPTION_CATCHING=1` on top. The
`EXPORTED_RUNTIME_METHODS=["FS"]` is load-bearing: STEP/STL export writes to
Emscripten's in-memory FS then reads it back (`shapes.ts:142-153, 174-186`).

**Why we want it.** A harness that runs any wasm CAD kernel and reports "kernel
error 5243184" instead of the kernel's own diagnostic is blind. This is the exact
two-build strategy (fast production / diagnostic) plus the ~15-line shim that
makes error text recoverable — and it is the same defect shape as OCCT findings
1-3 in report 01: a structured diagnosis discarded at the boundary.

**HOW verified NONE.** No hits for `Standard_Failure` or wasm exception-pointer
recovery anywhere in `src/harnesscad/`.

**2. Four in-source "OCCT misbehaves on X" notes. NONE-verified.**

- `packages/replicad/src/lib2d/intersections.ts:12-17`:
  ```ts
  try {
    // There seem to be a bug in occt where it returns segments but fails to fetch them.
    intersector.Segment(i, h1, h2);
  } catch (e) { continue; }
  ```
  `NbSegments()` over-reports; `Segment(i,…)` throws for some `i`. Workaround is
  per-index try/continue.
- `packages/replicad/src/lib2d/offset.ts:93-101` — `Geom2d_OffsetCurve` is
  **never returned directly**: "While returning the offset curve itself would be
  the more correct thing to do, opencascade does some weird stuff with it (for
  instance after mirroring it)". It is approximated as a continuous B-spline.
  Then `:98-107`: if the offset self-intersects, the curve is **collapsed to its
  endpoints** (`{collapsed: true, firstPoint, lastPoint}`) — "We need a better
  way to handle curves that self intersect, for now we replace them with a line."
- `packages/replicad/src/shapeHelpers.ts:103` — building a helix:
  `// We do not GC this surface (or it can break for some reason)`.
  `Geom_CylindricalSurface_1` is deliberately excluded from the `GCWithScope`
  register while every neighbouring object is registered. **A use-after-free the
  author worked around by leaking.**
- `packages/replicad/src/Sketcher.ts:280-283` — counter-clockwise elliptical arc:
  `arc.wrapped.Reverse()` with `// This does not work, we may need to hack a bit
  more within makeEllipseArc`. **Shipped known-broken.**

Directly appendable to `occt_quirks.py`, whose eight entries
(`boolean-fuzzy-value`, `revolve-zero-degrees`, `infinite-face-center-sentinel`,
`saddle-boolean-adjacent-holes`, `revolve-touching-axis`, `loft-invalid-solid`,
`no-occt-booleans-for-metrics`, `bbox-near-tangent-refusal`) cover none of them.

**3. The mesh parameter chain — with a 1000× trap. NONE-verified.**

`packages/replicad/src/shapes.ts:423, 439, 476, 636` — **`tolerance = 1e-3,
angularTolerance = 0.1`** at all four call sites (`_mesh`, `mesh`, `meshEdges`,
`blobSTL`). `angularTolerance` is OCCT's `theAngDeflection` in **radians**
(`BRepMesh_IncrementalMesh_2(shape, tol, /*isRelative*/ false, angTol,
/*isParallel*/ false)`, `:424-431`) — 0.1 rad ≈ 5.7°.

Edge sampling uses a different hardcoded recipe (`shapes.ts:566-570`):

```ts
new GCPnts_TangentialDeflection_2(adaptorCurve, tolerance, angularTolerance,
                                  /*MinimumOfPoints*/ 2, /*UTol*/ 1e-9, /*MinimumLength*/ 1e-7)
```

And in `meshShape` (`shapes.ts:1173`), the mesh→Manifold handoff uses
`tolerance ?? 1e-6` — **a thousand times tighter than the meshing tolerance** —
as the *vertex-merge grid* (`scale = 1/tol`, integer-rounded key at `:1174-1180`),
producing Manifold's `mergeFromVert`/`mergeToVert` arrays. `tol === 0` is
special-cased to exact string keys.

The trap: **the merge tolerance must be far tighter than the mesh tolerance or
the mesh collapses.** `registry.index()` returns **zero hits for "deflection"**;
`domain.geometry.parametric.chord_tolerance` covers sagitta/chord for arcs (from
`arcs`), not the `BRepMesh_IncrementalMesh` argument recipe, the angular-deflection
default, or the merge-grid rule. NONE-verified.

**4. Error taxonomy — 60+ throw sites with real structure. Partial.**

Distinguishable classes, all under `packages/replicad/src/`:

- **"Bug in the …" self-accusations** — the algorithm knows it reached an
  impossible state: `blueprints/boolean2D.ts:42, 71, 143, 381`;
  `blueprints/booleanOperations.ts:148`; `blueprints/customCorners.ts:41, 46, 54`;
  `blueprints/offset.ts:255, 307`. **Ten sites.** A verifier can treat these as
  "kernel-harness bug", categorically different from user error.
- **Cast failures** — `Could not {fillet,chamfer,shell,offset,revolve,sweep,loft}
  as/to a 3d shape` (`shapes.ts:1379, 1451, 1462`; `addThickness.ts:57, 133, 320`;
  `shapeHelpers.ts:542`). The op *succeeded* but returned a lower-dimensional
  shape — a distinct failure mode from "it threw".
- **Selector failures** — `Could not fillet, no edge was selected` /
  `Could not chamfer, no edge was selected` (`shapes.ts:1377, 1447`),
  `Finder has not found a unique solution` (`finders/definitions.ts:117`),
  `Could not find face for chamfer` (`shapes.ts:1425`). **This is the dominant
  text-to-CAD failure mode and it is a distinct class from geometric failure.**
- **Precondition failures** — `The minor radius must be smaller than the major
  one` (`shapeHelpers.ts:63, 151`), `You need at least 3 points to make a polygon`
  (`:641`), `Failed to build the face. Your wire might be non planar.` (`:315`).
- **Loop guard** — `Infinite loop detected` (`lib2d/stitching.ts:33`). A committed
  anti-hang guard.
- **I/O** — `WRITE STEP FILE FAILED.` / `WRITE STL FILE FAILED.`
  (`shapes.ts:155, 185`; `export/assemblyExporter.ts:149`),
  `Failed to load STEP file` / `Failed to load STL file` /
  `STL file contains no triangles` (`importers.ts:35, 79, 181`).
- **Lifetime** — `This object has been deleted` (`register.ts:40`).

`eval.verifiers.edge_fillet` and `domain.geometry.features.fillet_feasibility`
exist. The **selector-failure-as-its-own-class** distinction and the "Bug in the
…" self-accusation category are NONE-verified.

**5. Fillet/chamfer failure handling — the *absence* is the finding.**

`packages/replicad/src/shapes.ts:1347-1381` (fillet) and `1400-1453` (chamfer):
the only pre-kernel guard is `if (!edgesFound) throw` — an edge **count**.
`BRepFilletAPI_MakeFillet` is constructed with
`ChFi3d_FilletShape.ChFi3d_Rational` (`:1354`) and `filletBuilder.Shape()` is
called **without checking `IsDone()`**. No radius-vs-local-geometry feasibility
check, no try/catch, no fallback to a smaller radius. A stray `console.log(e)`
sits in the two-radius branch at `:1372`.

Chamfer's asymmetric forms require a face:
`r.selectedFace(finder).find(this, {unique: true})`, throwing
`Could not find face for chamfer` (`:1425`); `Add_3(d0, d1, edge, face)` for
`distances`, `AddDA(distance, angle*DEG2RAD, edge, face)` for `distance`+`angle`
(`:1428-1444`).

Adjacent: `shell()` (`:1220-1284`) defaults `tolerance = 1e-3` and passes
`-thickness` (negated) with `BRepOffset_Skin` + `GeomAbs_Arc` join type.
`intersect()` (`:1153-1166`) calls **`intersector.SimplifyResult(true, true,
1e-3)`** after `Build()` — an unconditional post-boolean simplification with its
own tolerance.

**This validates the harness's own design**: `domain.geometry.features.fillet_feasibility`
("deterministic, stdlib-only") is the preflight replicad lacks. **NONE-verified**:
the `SimplifyResult` post-boolean policy and the `ChFi3d_Rational` vs
`ChFi3d_Polynomial` flag choice.

**6. FinalizationRegistry-based GC — informative, not a gap.**

`packages/replicad/src/register.ts` — three disciplines: `WrappingObj`
(per-object, unregister-on-reassign, `:44-52`), `GCWithScope()` (registers
against the *closure itself*, so everything dies when the function's frame is
collected, `:61-68`), and `localGC()` (explicit `Set` + manual flush, `:79-100`).
`:8-15`: if `FinalizationRegistry` is absent it stubs to no-ops and logs
"Garbage collection will not work" — **silent unbounded wasm heap growth on old
runtimes.** N/A to the harness (Python, subprocess-driver), recorded as context.

### EXPLICIT NEGATIVE

`packages/replicad/__tests__/` holds `revolve-angle.test.ts`, `drawing/`, and an
**SVG-snapshot differ** (`diffSVGToSnapshot.ts`, `toMatchSVGSnapshot.ts`).
Golden-image regression for 2D drawings is a *technique* worth noting, but
**there is no known-bad geometry corpus.**

### VERDICT

**mine-further.** The wasm exception recipe and the four OCCT notes are the
payload; the mesh parameter chain is a close third.

---

## CascadeStudio-master (93 files)

### LICENSE

`LICENSE` — **MIT**, "Copyright (c) 2020 Johnathon Selstad". Verified.

### WHAT IT IS — and a provenance caveat that must be stated

Browser-based parametric CAD: JavaScript (or OpenSCAD) in a Monaco editor,
evaluated in a Web Worker against OCCT compiled to wasm.

**Provenance caveat.** This checkout is **not upstream CascadeStudio.**
`package.json` reads `"name": "cascadestudio-monorepo", "version": "2.0.0"` with a
`packages/` split, Playwright config, `vercel.json`, and — notably — a
**`CLAUDE.md`** at the root. Upstream CascadeStudio is a single-package repo at
0.x with no such file. This is a substantially rewritten fork.

That matters for how the findings below are weighted, and the two categories are
kept separate:

- **Code-level claims are verifiable and were verified**, by reading
  `packages/cascade-core/src/worker/StandardLibrary.js` directly (finding 2 below
  quotes the source).
- **`CLAUDE.md`'s 13-item pitfall table is a fork author's agent-guide prose**,
  not upstream documentation and not an OCCT-documented behaviour. Several items
  are cross-checkable against the code and several are not. It is recorded below
  as a **lead**, not as established fact.

### READ

**Read 1 file fully (`CLAUDE.md`, 341 lines), section-read 3
(`StandardLibrary.js`, `ShapeToMesh.js`, `package.json`), of 93.**
`packages/cascade-studio/lib/` is vendored (dockview, golden-layout,
openscad-parser) and was skipped. I additionally re-read
`StandardLibrary.js:470-505` and the three `fuzzValue` sites myself to verify
finding 2 before recording it.

### FINDINGS

**1. A post-boolean volume-ratio sanity check — verified in source, and the
cheapest verifier in either report. NONE-verified.**

`packages/cascade-core/src/worker/StandardLibrary.js:470-476` defines
`_quickVolume(shape)` — `BRepGProp.VolumeProperties_1` → `Math.abs(props.Mass())`,
`catch → 0`. Every boolean then compares input to output:

```js
// Union       :501  if (totalInput > 1 && resultVol < totalInput * 0.01)
// Difference  :539  if (mainVol    > 1 && resultVol < mainVol    * 0.01)
// Intersection:581  if (minInput   > 1 && resultVol < minInput   * 0.01)
```

with, for union, the message *"Union produced near-zero volume (…). Do the shapes
overlap? Non-touching shapes cannot be Unioned — keep them as separate scene
objects instead."*

**A 1%-of-input volume floor, with the reference quantity chosen per operation**
— sum of inputs for union, the *main* shape for difference, the *smaller* input
for intersection — and a `> 1` guard so tiny parts do not false-positive. The
per-op reference choice is the non-obvious part.

**Why we want it.** It turns OCCT's silent empty boolean into a diagnosable
message for about five lines of code, and it is a *measurement*, which is the
harness's stated bar. `eval.hardcorpus.occt` measures the built solid but no
per-boolean input-vs-output volume-ratio assertion with per-op reference
selection exists. NONE-verified.

**2. Tolerance defaults and flag recipes — verified in source. Partly new.**

- **Boolean fuzz value `1e-7`**, defaulted identically in all three booleans:
  `StandardLibrary.js:480` (Union), `:513` (Difference), `:560` (Intersection),
  passed to `oc.OCJS.BooleanFuse/BooleanCut/BooleanCommon(a, b, fuzzValue)`.
  *(The harness's existing `boolean-fuzzy-value` quirk in `occt_quirks.py` covers
  the practice; this pins a concrete default value to it.)*
- **Post-union face unification**:
  `new ShapeUpgrade_UnifySameDomain_2(combined, true, true, false); fusor.Build()`
  unless `keepEdges` (`:494-497`). The three booleans are unifyEdges / unifyFaces
  / concatBSplines.
- **Meshing**: `packages/cascade-core/src/worker/ShapeToMesh.js:73` —
  `new oc.BRepMesh_IncrementalMesh_2(shape, maxDeviation, false, maxDeviation * 5, false)`.
  **Angular deflection is derived as `5 × linear deviation`**, unlike replicad's
  independent `0.1`. A one-knob tessellation API — worth contrasting with
  replicad's two-knob one.
- Other defaults: `Offset` tolerance `0.1` (`:620`); several ops at `1e-4`
  (`:1175, 1189, 1373, 1385`) and `1e-3` (`:1232, 1456`); zero-vector guards at
  `1e-10` (`:1138, 1152, 1539`) throwing
  `Cannot normalize a zero-length vector; check your axis parameter`.
- **OCCT build provenance** (`CLAUDE.md:303-305`): "Custom fork of **OCCT 8.0.0
  RC4** compiled with **emsdk 4.0.23**" — which, if accurate, matches the OCCT
  8.0.0 sources in `oce-oce-patches` (report 01) and makes this the only
  wasm-side companion to that tree.
  `oc.OCJS.HashCode(shape, 100000000)` is the topology-identity primitive
  (`ShapeToMesh.js:59, 91`); `BRep_Tool.Triangulation(face, loc, 0 /* Poly_MeshPurpose_NONE */)`
  with a null-check that logs `"Encountered Null Face!"` and **clears the entire
  arg cache** (`:83`).

**3. The 13-item pitfall table — recorded as LEADS, provenance-flagged.**

`CLAUDE.md:90-224`, each with a GOOD/BAD code pair. Fork-authored prose; the
items marked *(code-verifiable)* can be checked against `StandardLibrary.js`,
the rest are the author's claims:

1. `Loft()` prefers `TopoDS_Wire` — after `Translate`/`Rotate` a wire downcasts to
   generic `TopoDS_Shape`; must re-extract with `GetWire()` (`:92-102`).
2. **Fillet before hollowing** — "After Difference/Union, the edge topology
   changes and the selector may not find the edges you expect" (`:104-121`).
   **The most important item on the list: booleans invalidate edge selectors.**
   It generalises past this repo to any selector-based CAD DSL, and it is the
   same failure replicad surfaces as `Could not fillet, no edge was selected`.
3. `Offset(face, d)` returns a wire/face, not a solid — must offset then extrude
   separately (`:123-133`).
4. `Volume()` can be negative on inverted face normals — cosmetic, use
   `Math.abs`. *(code-verifiable: `_quickVolume` does exactly this at `:474`.)*
5. **Sketch `.Fillet()` must follow `.LineTo()`** — fillets the most recent
   vertex; calling first **fails silently** (`:140-150`).
6. Transforms return new shapes; the original is consumed unless
   `keepOriginal: true` (`:152-156`).
7. `Circle(r, true)` → wire; `Circle(r, false)` → face (`:158-163`).
8. `Scale()` takes a **scalar only**; non-uniform scaling unsupported (`:165-170`).
9. `BSpline(points, closed)` — `false` for Pipe rails, `true` for rings.
10. **`Union()` on non-overlapping shapes** produces unexpected results
    (`:178-190`). *(code-verifiable: this is precisely what the `:501` volume
    check detects and what its message says.)*
11. **`Extrude()` consumes its input face** (`keepFace=false` default) — reusing
    it for `Offset()` fails (`:192-206`). *(Same shape as Zoo's solid-consumption
    ownership rule above — two independent systems, one discipline.)*
12. **Sketch plane for revolve profiles** — `new Sketch([x,y],'XZ')` maps
    `[x,y]→[X,0,Z]`; default XY + revolve-around-Z "produces a flat disk!"
    (`:208-222`).
13. **Null-shape cascading** — one null shape poisons every downstream op;
    Extrude/FilletEdges/ChamferEdges/Offset/Pipe/Difference null-check with a
    descriptive early return "instead of cascading cryptic failures" (`:224-230`).

Items 2, 5, 10, 11 and 12 are **silent-wrong-output or silent-no-op** modes — the
program runs clean and the geometry is wrong, which is exactly what a verifier
must catch. `agents.generation.quirk_preflight` ("warn before OCCT bites") is the
natural consumer and has no selector-invalidation or consumed-input rules.
NONE-verified — but any of these adopted from prose alone should be reproduced
against a live kernel first, given the provenance caveat.

**4. Build-step-indexed rendering — a verifier affordance worth stealing.**

`window.CascadeAPI` is deliberately four methods — `getQuickStart()`,
`runCode(code) → {success, errors, logs, historySteps}`, `saveScreenshot(filename)`,
`setCameraAngle(azimuth, elevation)` (`CLAUDE.md:61-70`). `getHistorySteps()`
returns `[{fnName, lineNumber, shapeCount}, …]` and `screenshotHistoryStep(i)`
renders any intermediate build state (`:270-278`) — **per-operation visual
bisection of a CAD program.** Not a kernel fact; recorded because
build-step-indexed rendering is a diagnostic the harness's op-DAG could support
and currently does not surface.

### VERDICT

**mine-further, with the provenance caveat attached.** Finding 1 is the single
cheapest high-value item in report 02. Finding 3 is a rich lead list that should
be reproduced before being recorded as fact.

---

## OpenCADStudio-main (804 files)

### LICENSE

`LICENSE` — **GPL-3.0**. Behavioural facts only, cited by path and line.
**Nothing vendored, no source text reproduced.**

### WHAT IT IS

A Rust / `iced` / `wgpu` DWG editor. **It has no kernel of its own** — 3D work is
delegated to `truck-modeling` 0.6 / `truck-meshalgo` 0.4 / `truck-shapeops`.

### READ

**Read 2 files fully (`docs/tessellation.md`, `docs/native-vs-web.md`),
section-read 5 (`Cargo.toml`, `src/scene/model/solid_model.rs`,
`src/scene/convert/truck_tess.rs`, `src/scene/convert/acis_to_truck.rs`, the docs
listing), of 804.**

### SKIMMED-NOT-READ

`src/ui/`, `src/app/`, `src/plugin/`, `crates/dwg-thumbnailer*`, `web/`,
`packaging/`, `Trunk.toml`, `docs/pane-grid-migration.md`, `docs/PR-plugin-host.md`,
`docs/plugin-architecture.md` (409 lines, an out-of-process plugin ABI over
interprocess/libloading/memmap2/rkyv), `STATUSBAR.md`, `COMMANDS.md` — **all app
glue, UI layout, plugin ABI and packaging. No kernel facts.** That is the great
majority of the repo.

### VERDICT FIRST

**Not "nothing here" — but close.** Three things survive the filter. The prior
audit's instinct was directionally right; it was wrong only in degree.

### FINDINGS

**1. A per-entity-type tessellation-path table — a genuine domain table.
Partial.**

`docs/tessellation.md:48-93` — 43 DWG/DXF entity types × {output: mesh / wire /
both / none} × {path: truck B-rep mesh / truck curve topology / direct}, with
per-row notes. `:5-19` draws a distinction the doc insists must not be conflated:
**truck B-rep meshing** (`MeshableShape::triangulation()`, the only path
producing kernel-derived filled triangles) vs **truck curve topology**
(`Edge`/`Wire` sampled by `ParameterDivision1D` — touches topology but never
triangulates a surface; output is a polyline).

Only `Solid3D`, `Region`, `Body` and `Surface` take the B-rep mesh path, and all
four are **truck-with-direct-fallback**: `acis_to_truck::tessellate_sat_truck`
first, falling back to a bespoke per-surface LOD sampler (`tessellate_sat_lods`)
"when truck cannot rebuild a face" (`:38-42`). Rows carrying real behaviour:
`Hatch` — "boundary outline **not** emitted to the wire set (**#131 OOM**)";
`Ray` — `[base, base + dir×1e6]`; `LwPolyline` — path depends on the `plinegen`
flag (truck `Contour` with bulge arcs vs direct segments).

`domain.reconstruction.ortho.edges` / `ortho.topology` handle 2D reconstruction;
no per-entity DWG tessellation-path table exists. Given the `rust-integrations`
work already covers truck, this is complementary — and the
truck-fails-→-fallback-sampler pattern is the honest architecture for an
unreliable B-rep rebuild.

**2. Three tolerance constants with their stated reasoning. Partial.**

- `src/scene/model/solid_model.rs:143` — **`BOOL_TOL: f64 = 0.05`**, passed to
  `truck_shapeops::or/and` (`:150-155`). Subtract is `b.not()` then `and`.
  `boolean()` returns `Option<Solid>` and the doc comment states it returns
  `None` "when the operation fails (e.g. the solids don't actually overlap)"
  (`:145-146`) — **the same non-overlap failure CascadeStudio detects by volume
  ratio, here surfaced as `None` rather than a bad solid.** Two independent
  designs for the same trap.
- `src/scene/convert/acis_to_truck.rs:41-46` — `MESH_TOL = 0.01` (planar faces,
  "the surface itself adds no curvature") and **`CURVE_REL_TOL = 0.1`,
  radius-*relative***, commented: "as a fraction of the surface radius, so the
  facet count is radius-independent instead of exploding on large radii. `0.1` =
  10% chord error (~7 facets per full circle)". Applied at `:211` as
  `(radius.abs() * CURVE_REL_TOL).max(1e-6)` — **relative with an absolute
  floor.** Also `FULL = TAU + 0.2` (`:36-37`), "slightly over 2π so revolution
  builders close the loop" — a degenerate-seam workaround.
- `src/scene/convert/truck_tess.rs:20-23` — `CURVE_TOL = 0.005`, `MESH_TOL = 0.01`,
  plus a **zoom-adaptive per-frame override** stored in an `AtomicU64` of f64
  bits (`:25-35`), targeting ~0.5 px chord height, deliberately global "so callers
  don't need to thread a tolerance parameter through every entity-converter
  signature."

**Why we want it.** The radius-relative-with-absolute-floor chord tolerance is
the same shape as OpenJSCAD's bounds-relative epsilon, **arrived at
independently** — two repos converging on "tolerance must scale with the
feature." The pixel-targeted variant is a distinct third policy: view-dependent
rather than model-dependent.

`domain.geometry.parametric.chord_tolerance` **HAS** sagitta/chord tessellation
(from `arcs`). The **relative-with-absolute-floor form** and the **view-dependent
variant** are NONE-verified.

**3. A capability-degradation matrix — borderline, included because it is a
*kernel availability* fact. NONE-verified.**

`docs/native-vs-web.md:10-32` + `Cargo.toml:26-30`: the `solid3d` feature gates
`truck-meshalgo` + `truck-shapeops`, which pull `vtkio → xz2 → lzma-sys`, **a C
library that cannot cross-compile to wasm32**. The web build therefore runs
`--no-default-features`, and **"the Model tab's solid primitives and boolean
operations are no-ops"** and "solid tessellation and ACIS (SAT) import produce no
geometry" — **silently, not as errors.** Related: `lzma-sys` is forced to
`features = ["static"]` because dynamic linking baked a Homebrew
`liblzma.5.dylib` path into the macOS binary, crashing on other machines (issue
#56, `Cargo.toml:41-46`).

**Why we want it.** "The boolean kernel is compiled out and every boolean
silently returns nothing" is a verifier's nightmare configuration, and a strong
argument for a **startup capability probe** rather than trusting a backend's
presence. Reported as a behavioural fact per the GPL constraint.
`io.backends.*` is the natural home; NONE-verified as a named pattern.

### EXPLICIT "NOTHING HERE"

`docs/plugin-architecture.md`, `docs/pane-grid-migration.md`, `STATUSBAR.md`,
`COMMANDS.md`, `crates/dwg-thumbnailer-win` (a Windows COM shell handler),
`build.rs`, `Trunk.toml`, `packaging/`, and the whole of `src/ui/` and `src/app/`
— **no kernel scar tissue.** The prior audit's dismissal was very nearly right.

### VERDICT

**mine-further, narrowly** — three findings, all small, one (the capability
probe) more architectural than geometric. Everything else is app glue.

---

## curv-master (1,553 files)

### LICENSE

`LICENSE` — **Apache-2.0**. Verified. `extern/` is vendored third-party and was
excluded throughout.

### WHAT IT IS

Doug Moore's functional f-rep / signed-distance-field CAD language. It has no
B-rep backend, which is why a prior audit skipped it. That verdict is
**overturned below, with evidence.**

### READ

Read fully: all of `docs/shapes/` (`Shapes.rst`, `Boolean.rst`,
`Transformations.rst`, `Distance_Field_Operations.rst`, `Debug.rst`,
`Future_Work.rst`), `docs/Theory.rst`, `docs/Mesh_Export.rst`, `docs/Viewer.rst`,
`docs/lib/Blend.rst`, `docs/language/Grammar.rst`, `lib/curv/lib/blend.curv`,
`ideas/Error_Messages`. **16 files read fully.**

Targeted full sections: ~450 of 1,744 lines of `lib/curv/std.curv`; 140 of 499
lines of `libcurv/frag.cc`. Header/index-read: ~30 further files
(`libcurv/render.{h,cc}`, `libcurv/io/`, the `ideas/` tree, directory listings).

**Read 16 files fully, 2 in large targeted sections, ~30 header-read, of 1,553.**

Harness side verified by full read of `combinators.py`, `field_transforms.py`,
`primitives.py`, `sphere_tracing.py`, `csg_bounds.py`, `rounded_csg.py`, plus
repo-wide greps.

### THE HEADLINE

The harness already contains Curv-derived ports — `sdf.combinators`,
`sdf.primitives`, `sdf.field_transforms`, `sdf.tpms` and `numeric.sphere_tracing`
all say "from Curv" in their own summaries. **What was ported is the arithmetic.
What was not ported is the correctness metadata — which is the only part a
verifier actually needs.**

### FINDINGS

**1. The SDF correctness taxonomy — the central finding. NONE-verified.**

`docs/shapes/Shapes.rst:52-70` — a five-level total order:

| class | definition (verbatim from Shapes.rst) |
|---|---|
| `exact` | exact Euclidean distance to nearest boundary; `offset` gives a rounded offset |
| `mitred` | vertex/edge information preserved in all isosurfaces; `offset` gives a mitred offset |
| `approximate` | implementation-dependent, **may change between releases** |
| `bad` | Lipschitz continuous with **constant > 1**. "Sphere tracing won't work unless you correct the SDF using the `lipschitz` operator. The correction factor needs to be determined experimentally." |
| `discontinuous` | not Lipschitz continuous at all; **`lipschitz` cannot help you** |

`docs/shapes/Distance_Field_Operations.rst:1-15` states the refusal predicate
directly: *"if you want predictable and repeatable behaviour, you should restrict
distance field arguments to shape expressions that are documented to produce
either an exact or a mitred distance field."*

**Why we want it.** This is a four-way decision procedure, not advice.
`discontinuous` → refuse to render at all. `bad` → refuse until a measured `k` is
supplied. `approximate` → refuse `offset`/`shell`/`perimeter_extrude`, because the
results are unspecified *and version-unstable*. `exact` / `mitred` → admit, but
with **different documented semantics for the same call**.

**HOW verified NONE.** The five words appear in `sdf/primitives.py:6-18` and
`sdf/combinators.py:21-33` as English docstring prose. There is no enum, no
dataclass, no `Literal` type; **`bad` and `discontinuous` appear nowhere at all.**
Every combinator takes and returns a bare `float`.

**2. Per-operation class propagation — including an inside/outside asymmetry.
NONE-verified.**

`docs/shapes/Boolean.rst:58-82` — the finding is that **the class of the interior
differs from the class of the exterior**, so field class is a *pair*, not a scalar:

- `complement s` → out(result) = in(s); in(result) = out(s) — a swap
- `union` of exacts → **outside exact, inside approximate**
- `intersection` of exacts → **outside mitred, inside exact**
- `difference[s1,s2]` → outside mitred if out(s1) and in(s2) exact; inside exact
  if in(s1) and out(s2) exact

Confirmed against implementation: `std.curv:791` comments `_union2` as "produces
a mitred SDF inside"; `std.curv:783-789` `complement` negates and sets bbox to
infinite.

`docs/shapes/Transformations.rst:5, 21-23` — similarity transformations
(translate / rotate / reflect / **isotropic** scale) are the **only**
class-preserving ops: "If the input has an exact or mitred distance field, the
output is also exact or mitred."

Class-destroying operations, with the source line stating the damage:

| op | resulting class | source |
|---|---|---|
| `stretch` (anisotropic scale) | approximate — `dist * min(s)` | `std.curv:947-959`; `Theory.rst:506-511` |
| `shear_x` | **bad** — `fix_distance = d; // TODO` (no correction at all) | `std.curv:1131-1136`; `Transformations.rst:113` |
| `twist` | **bad**, and *unboundedly* so | `std.curv:1518-1539`; `Transformations.rst:147-151` |
| `bend` | **bad** | `std.curv:1541-1567`; `Transformations.rst:180` |
| `local_taper_x/xy` | corrected by `* min[k...]` | `std.curv:1139-1193` |
| `gyroid` | **bad, Lipschitz constant exactly 4/3** | `std.curv:1511` |
| `loft` | "TODO: bad distance field" | `Distance_Field_Operations.rst:82` |
| `columns` blend | "the distance field is bad" | `lib/Blend.rst:79` |
| `pipe` blend | "WARNING: bad distance field" | `lib/Blend.rst:89` |

**The `twist` rule is the sharpest refusal predicate in the repo**
(`Transformations.rst:149-151`): *"As the twist rate increases, the distance field
becomes more distorted… The distortion also increases without bounds with
increasing distance from the Z axis, so you can only twist shapes with a finite
diameter."* That is a checkable precondition — finite radial extent — with a
Lipschitz bound scaling as `twist_rate × radius`.

**HOW verified NONE.** No propagation exists (`combinators.py:23-26` states the
union rule as a comment only). `bend`, `shear`, `repeat_xy`, `repeat_radial` are
absent entirely; `twist` and `taper` exist only as extrusion operators
(`sweep.py:90, 104`), not as domain warps of an arbitrary field.

**3. Lipschitz correction and — crucially — its measurement protocol.
NONE-verified.**

`std.curv:1621-1632`, the corrector (trivially simple, and absent from the
harness as a composable operator):

```
lipschitz k shape = make_shape { dist p : shape.dist p / k, ... }
```

`docs/shapes/Debug.rst:39-42` + `std.curv:1634-1652` — the measurement protocol,
which is the real finding:

> "To find the Lipschitz constant of a shape with a bad distance field, start
> with `[j,k]=[1,2]`, then use binary search to find the smallest value of `k`
> that doesn't produce white."

`show_gradient` samples four axis-neighbours at `eps = 0.01` and takes
`g = max(|d−up|,|d−down|,|d−left|,|d−right|) / eps` (`std.curv:1643-1649`) — a
finite-difference sup-norm gradient estimate. Wrapped in a bisection over a
sampled domain it becomes an **automated Lipschitz-constant estimator**: exactly
the missing primitive that turns "field is bad" from an unusable verdict into a
repairable one.

Also `Debug.rst:21-30`, `show_dist` — the **NaN/inf failure taxonomy**: NaN →
white ("something that can only happen on the GPU"), `+inf` → vivid cyan, `−inf`
→ dark cyan, gradient > 1 ramping red to full at gradient ≥ 2. A ready-made
field-sanity checklist.

**HOW verified NONE.** No composable `lipschitz(k, field)`; only a hardcoded
per-primitive bool in `tpms.py:33, 50` and a step divisor in
`sphere_tracing.py:87`. Nothing anywhere samples a domain and takes a max of
`|∇f|` — `numeric/finite_differences.py:41` and `sphere_tracing.py:93` are
pointwise only.

**4. Sphere-tracing parameters and failure modes. Partial.**

`libcurv/render.h:31, 33` — shipped defaults: `ray_max_iter_ = 200`,
`ray_max_depth_ = 2000.0`.

`libcurv/frag.cc:194-204` — the loop, and one detail the harness gets differently:

```c
for (int i=0; i<ray_max_iter; i++) {
    float precis = 0.0005*t;          // epsilon is RELATIVE to marched distance
    float d = dist(p);
    if (abs(d) < abs(precis)) { c = colour(p); break; }
    t += d;
    if (t > tmax) break;
}
```

`precis = 0.0005*t` is a **distance-proportional epsilon** — it widens with depth
to match the shrinking pixel footprint. The harness uses a fixed `epsilon=1e-6`
(`sphere_tracing.py:58-66`), which over-iterates near and under-resolves far.
`frag.cc:212` also gives the normal-estimation constant: a tetrahedral 4-tap at
`0.5773*0.0005` (vs the harness's 6-tap central difference).

Documented failure modes: `Mesh_Export.rst:30-32` — mesh export exists precisely
as the escape hatch for *"models that are not compatible with the viewer (because
their distance function is too slow or not Lipschitz-continuous)"*.
`Shapes.rst:66` — sphere tracing simply "won't work" on `bad` fields.
`frag.cc:182-184` carries an unresolved TODO that the hard-coded `tmax` breaks
`tetrahedron`.

`numeric.sphere_tracing` has `max_steps` / `epsilon` / `lipschitz`. It **lacks**
the relative epsilon and — more importantly — **cannot distinguish its two
failure returns**: both max-distance escape (`:89`) and step exhaustion (`:90`)
return a bare `None`. For a verifier those are opposite diagnoses (a genuine miss
vs an under-converged or overstepped ray). No sign-flip or tunnelling detection
exists despite being documented at `:22-24`.

**5. The shape protocol and the bbox conservativeness contract. Partial.**

`std.curv:125-138` — `make_shape`, the enforced invariant set:

```
assert (shape.is_2d || shape.is_3d);   // at least one dimensionality
assert (is_bbox3(shape.bbox));         // always 3D-shaped, even for 2D shapes
assert (is_func(shape.colour));
assert (defined(shape.dist) && is_func(shape.dist));
```

Defaults: `bbox = [[-inf,-inf,-inf],[inf,inf,inf]]` (**conservative by default**),
`colour = [.8,.8,.5]^2.2`. `dist` and `colour` are `(x,y,z,t) → …`: **time is a
coordinate, not a global** (`Theory.rst:561-591`), so animation composes without
special-casing.

Bbox contract, `Shapes.rst:72-84` + `Distance_Field_Operations.rst:12-23`:

- A bbox is `exact` or `approximate`; approximate means **larger** than
  necessary. "Bad" means **too small to contain the shape** — a soundness
  violation.
- All shape *constructors* produce exact bboxes; many *combinators* do not.
- The refusal rule: *"For all of the distance field operations, we only guarantee
  to compute a 'good' bounding box estimate if the distance field arguments are
  exact. Otherwise, the bounding box may be 'bad'."* Reason given (`:17-21`):
  computing a good bbox needs a **lower bound on the ratio by which the field
  underestimates true distance**, and *"for mitred distance fields in general,
  there is no lower bound."*
- Escape hatch: `set_bbox bbox shape` (`Debug.rst:18-19`), plus `show_bbox` which
  renders in-box geometry green and **out-of-box geometry red** (`Debug.rst:6-16`)
  — a visual soundness oracle.

Per-op bbox rules worth lifting: `offset d` is good iff (field exact) **or**
(`d <= 0`) (`Distance_Field_Operations.rst:49-52`); and
`smooth r .union[s,s] ≡ offset (r/4) s` is *"the worst case for bounding box
inflation, so we can use this to compute bounding boxes"* (`:209-211`), matching
`std.curv:848-849` which inflates by `k/4`.

**Why we want it.** Bbox correctness is *entangled with field class*. A harness
that computes bounds from a CSG tree while ignoring field exactness can silently
emit unsound (too-small) boxes — and a too-small bbox is a wrong containment
proof, not a loose one.

`sdf.csg_bounds` documents conservativeness in prose (`csg_bounds.py:27-28`,
`csg_eval.py:21`) but `BBox3` carries only `_lo`/`_hi` (`:62`) with no
exact/conservative/bad tag, and there is no `set_bbox` override anywhere in
`domain/geometry`. **The `offset`-sign rule and the `k/4` smooth-union inflation
rule are NONE-verified.**

**6. The blend kernel catalogue with exact formulas. Partial.**

`lib/curv/lib/blend.curv` ports MERCURY `hg_sdf`. Every kernel is a record of
`{union, intersection, difference}` derived from one `fmin`, with
`bmax[a,b] = -fmin[-a,-b]` (`blend.curv:5-9`) — a De Morgan generator worth
having as an abstraction in itself. Blended-union bbox rule at `:21-23`: inflate
by `bulge = -bmin[0,0]`.

| kernel | formula | source |
|---|---|---|
| `smooth r` | `h=clamp[.5+.5*(b-a)/r,0,1]; lerp[b,a,h] - r*h*(1-h)` | `blend.curv:47-54`, `std.curv:869-877` |
| `chamfer r` | `e=max[r-abs(a-b),0]; min[a,b] - e*.5` | `blend.curv:56-63` |
| `stairs [r,n]` | `s=r/(n+1); u=b-r; min[min[a,b], .5*(u+a+abs(mod[u-a+s,2s]-s))]` | `blend.curv:65-73` |
| `columns [r,n]` | full scalloped kernel, separate `fmax`/`diff` (not derivable by negation) | `blend.curv:75-107` |
| `pipe d` | `mag[s1.dist p, s2.dist p] - d/2` | `blend.curv:109-119` |
| `engrave r` | `max[d1, (d1 + r - abs(d2))/sqrt 2]` | `blend.curv:121-129` |
| `groove [ra,rb]` | `max[d1, min[d1+ra, rb-abs(d2)]]` | `blend.curv:131-139` |
| `tongue [ra,rb]` | `min[d1, max[d1-ra, abs(d2)-rb]]` | `blend.curv:141-150` |

Stated caveats, all refusal-grade: `smooth`/`chamfer` — distance field
approximate, bbox approximate (`Distance_Field_Operations.rst:213-215, 223-225`);
both **binary only, non-associative, do not generalise to N shapes** (`:165-166`)
— a real trap for LLM-generated N-ary blend chains. `stairs` — *"WARNING:
experimental, the API may change… What is the relationship of the `r` parameter
to the size of the blending band?"* (`lib/Blend.rst:66-69`). `columns` — *"the
parameters don't make sense and the distance field is bad"* (`:78-79`). `pipe` —
*"bad distance field"* (`:89`).

Plus the geometric caveat at `Distance_Field_Operations.rst:191-207`: the
elliptic blend gives a true quarter-circle fillet **only at 90°**, deforming to
an ellipse otherwise — *"This might be bad for engineering, if you need a
constant radius fillet"* — and it produces a documented "bulge". **That one
matters commercially: "add a 3 mm fillet" is a canonical text-to-CAD request
that this operator silently fails to honour off-90°.**

`sdf.combinators` has `smooth_min_poly` (the same IQ formula), `chamfer_min`,
plus exp/power variants, and `rounded_csg` has arc fillets. **`stairs`, `columns`,
`pipe`, `engrave`, `groove`, `tongue` are NONE-verified — 6 of 8 absent.** The
non-associativity warning and the 90°-only constant-radius caveat are also
NONE-verified.

**7. Formal grammar and precedence. NONE-verified.**

`docs/language/Grammar.rst` — a complete, genuinely formal artefact:

- Lexical: `:22-31` identifiers including a 15-word reserved list; `:47` symbols;
  `:52` characters; `:59-64` numerals; `:70-81` strings with `$`-interpolation
  productions.
- **`:96-159` — an explicit 11-level precedence chain**, lowest to highest:
  `listing → item → ritem → pipeline → disjunction → conjunction → relation →
  sum → product → power → postfix → primary`.
- `:161-207` — a second, deeper **phrase-type system** with six primitive types
  (definition, pattern, expression, locative, generator, statement), and the
  honest note at `:92-93` that *"Not all program texts that have a parse tree are
  syntactically correct"* — i.e. **parse ≠ well-formed, a two-stage validation
  contract**, which is exactly the shape a generation harness needs.

### EXPLICIT NEGATIVE

**No error catalogue exists.** `ideas/Error_Messages` (82 lines, read fully) is an
aspirational design note citing Clang and Elm, not a catalogue. Diagnostic
strings are scattered inline across `libcurv/*.cc`. The prior audit was right on
this specific point.

Also: Curv names no specific published algorithm for its `#smooth`/`#sharp` mesh
generators in the docs, and `libcurv/io/` contains no `export_mesh.cc` in this
snapshot — so the algorithm *identities* are unestablished. Only the selection
rules and artifacts (below) transfer.

**8. Meshing: algorithm-selection rules keyed on field class. Partial.**

`docs/Mesh_Export.rst:34-72` — two generators with an explicit, **field-class-keyed**
selection rule:

- `#smooth` (default): output *"guaranteed to be topologically correct
  (watertight, manifold, no self intersections)"*, mostly quads, but **rounds off
  all sharp features**. The fallback when `#sharp` fails.
- `#sharp`: preserves sharp features, built-in simplifier, fewer faces —
  *"recommended for CAD-like models with exact or mitred distance fields…
  combined using boolean operations… transformed with similarity
  transformations."* **That precondition list is exactly the class-preserving set
  from finding 2.**
- `#sharp` failure modes (`:58-66`): `bend`/`twist` produce warped fields that
  *"cause the `#sharp` algorithm to misbehave and create spiky artifacts"*; it
  *"requires surface areas to have smoothly varying normals, so it can't be used
  with pure fractal shapes"*; and it is *"not guaranteed to create a
  topologically correct mesh."*

Parameter contract (`:89-115`): `vsize` — *"should be half the size of the
smallest detail you want to capture, and half the thickness of the thinnest wall.
If `vsize` is too large, small details will disappear and holes will appear in
thin walls."* **That is a quantitative pre-flight check**: given a target
minimum wall thickness, `vsize` is determined, and a violation *predicts holes in
the output*. `vcount` defaults to 100,000; 1-2M triangles is the practical print
ceiling (`:79-81`); Shapeways' documented hard limit is 2M (`:166`).

`geometry/volumes/marching_cubes` and `surface_nets` exist as isosurface
extractors. The **algorithm-selection rule keyed on field class**, the
**`vsize ≤ min_wall/2` predicate**, and the **spiky-artifact-from-warped-field
failure mode** are NONE-verified.

### VERDICT — the prior "skip, no backend" call was WRONG

Three independent lines of evidence:

1. **It was already contradicted when it was made.** Five harness modules carry
   "from Curv" / "ported from Curv" in their own summaries (`sdf.combinators`,
   `sdf.primitives`, `sdf.field_transforms`, `sdf.tpms`, `numeric.sphere_tracing`).
   The repo had already proven useful before being marked useless — the audit
   judged it on backend criteria and missed that it had been mined as a
   *specification*.
2. **The stated criterion was the wrong one.** "No backend" is true and
   irrelevant. Curv's transferable asset is a **five-level correctness lattice
   with per-operation propagation rules and a documented repair operator** —
   which is the shape of a refusal predicate, and which **no B-rep-backed repo in
   the corpus can supply**, because the concept does not exist outside f-rep.
3. **The gap is real, verified and load-bearing.** The harness ported Curv's
   arithmetic and dropped its metadata. Confirmed absent: the field-class type
   (no enum anywhere; `bad`/`discontinuous` appear zero times), all propagation
   rules, any Lipschitz measurement, any composable `lipschitz` corrector, any
   exactness-keyed guard on `offset`/`shell` (`field_transforms.py:52, 61, 72, 85`
   are unguarded one-liners), the bbox good/bad tag and `set_bbox`, six of eight
   blend kernels, and four transforms.

The consequence is concrete and currently live: **the harness will happily
evaluate `twist(rate, shape) >> offset(2.0)` and return a confident number.**
Curv documents that this composition is meaningless — `twist` yields a `bad`
field with unbounded distortion, and `offset` on a non-exact field has no defined
semantics. The harness cannot presently express that refusal, and Curv is where
the vocabulary to express it comes from.

**mine-further, at high priority.** Extraction order: (1) reify the five-class
lattice and its propagation as a type carried alongside every SDF node; (2) the
`show_gradient` bisection as an automated Lipschitz estimator; (3) exactness-keyed
guards on `offset`/`shell`/`perimeter_extrude`; (4) bbox conservativeness tagging
plus the `offset`-sign and `k/4` rules; (5) the six missing blend kernels; (6) the
`vsize ≤ min_wall/2` mesh pre-flight predicate.

---

## Cross-repo notes for set B

**The same defect, five times.** Report 01's recurring finding — a structured
diagnosis discarded at a boundary — repeats here: replicad's wasm exception
pointer (`Kernel error 5243184` instead of `GetMessageString()`), OpenJSCAD's
`measureVolume` returning a plausible float on non-manifold input, the harness's
`sphere_tracing` collapsing two opposite failures into one `None`, Zoo's TS
generated suite asserting "success **or** error", and OpenCADStudio's wasm build
silently no-op-ing every boolean. In every case the machinery to distinguish
exists and the last inch throws it away.

**Two independent inventions of "tolerance must scale with the feature."**
OpenJSCAD's `EPS × mean-extent / dimensions`
(`calculateEpsilonFromBounds.js:8`) and OpenCADStudio's
`(radius.abs() * CURVE_REL_TOL).max(1e-6)` (`acis_to_truck.rs:211`) arrive at the
same policy from different directions, and the second adds the absolute floor the
first lacks. Curv's `precis = 0.0005*t` is a third instance in the ray-marching
domain. The harness's epsilons are absolute throughout.

**Two independent inventions of linear ownership.** Zoo's solid-consumption
fixtures (*"`X` was already consumed by a `<op>` operation"*, 8 committed
known-bad cases) and CascadeStudio's `Extrude()`-consumes-its-face rule are the
same discipline in two unrelated systems. That convergence is itself evidence
it is a real property of CAD op-streams worth checking statically, not a quirk
of either implementation.

**Two independent designs for the non-overlapping-boolean trap.**
CascadeStudio detects it after the fact by volume ratio
(`StandardLibrary.js:501`); OpenCADStudio's truck wrapper surfaces it as
`Option::None` up front (`solid_model.rs:145-146`); OpenJSCAD *creates* it by
short-circuiting to list concatenation (`mayOverlap.js:26-31`). Three points on
the same problem.

**Negative result worth recording once.** **No known-bad geometry fixture corpus
exists in OpenJSCAD, replicad, CascadeStudio or OpenCADStudio.** OpenJSCAD's
`test/` holds only `helpers/`; replicad's `__tests__/` is an SVG golden-image
differ. The only committed adversarial corpora found anywhere in set B are Zoo's
54 `execution_error.snap` pairs and the CLI's physics golden values. The harness's
`eval.corpus.fixtures.manifold_meshes` remains its own best adversarial source.
