# Deep inventory: missed engineering domain tables and schemas/DSLs

Read-only sweep (2026-07-16) of resources/cad_repos (~120 repos) + OpenCAD-main.
Harness baseline read first: thread_database (ISO+UNC+NPT), heatsert_bores (M3-M6),
cad_defaults (M3/M4/M5 clearance), contract.py (12-material density-only),
embodied_carbon (small uncited dict), printability_verdict (min_wall/min_feature/45deg
only), library/parts.py (5 ModelCards), kcl_grammar.py (lexer mirror only).

## Top-10 highest-value finds (ranked)

1. **anvilate ISO 286 fits** (`tolerance/data/iso286_{grades,deviations}.yaml` +
   `tolerance/iso286.py`) -- full IT5-IT16 grade widths + fundamental deviations
   (d..u with hole delta rule), `fit("H7/g6") -> clearance range + kind`. The harness
   has ZERO fit/tolerance capability. Complete, cited, CC0-flagged. Build:
   `domain/standards/iso286_fits.py` + contract fit predicates.
2. **anvilate standards catalog set** (11 provenance-tagged YAMLs: metric_thread,
   metric_clearance M2-M30 x3 fits, cap_screws, hex_bolts, hex_nuts, washers,
   dowel_pins, bearings, T-slot extrusions 20/30/40/45, NEMA 17/23/34 frames +
   resolver.py). Build: `domain/standards/part_catalog.py` keyed by designation;
   NEMA + T-slot directly bind "motor mount bracket" hole patterns.
3. **cad-cae-copilot 46 JSON Schemas** (`aieng/src/aieng/schemas/`), especially
   `allowed_operations_catalog.schema.json` (per-feature op permissions:
   allowed/forbidden/conditional + preconditions + blocked_by_constraints),
   parameter_edit, patch_proposal, protected_regions. The missing OP-GATING contract
   layer over core/cisp/ops.py. Build: `core/cisp/op_gate.py` validating an op stream
   against an allowed-operations catalog before execution.
4. **anvilate materials.yaml (390 ln, citation-tagged) + cad-cae materials.py**
   (Al 6061/7075/2024/5052/5083/6082, steels 1045/A36/4140/4340, H13...) --
   E/nu/rho/yield/ultimate per material ("retrieval not recall"). Upgrades
   contract.py from density-only; unlocks mass->stress predicates. Build:
   `domain/standards/materials_db.py` (anvilate citations + copilot breadth).
5. **kerf `kerf-lca/data/ice_v3.json`** -- ICE v3.0 embodied-carbon DB (kg CO2e/kg,
   aliases, recycled content, source URLs). Drop-in cited upgrade of
   embodied_carbon.py's ad-hoc dict.
6. **modeling-app `codemirror-lang-kcl/src/kcl.grammar`** -- complete Lezer
   PRODUCTION-RULE grammar for KCL (precedence, statements, expressions).
   kcl_grammar.py mirrors the lexer only. Build: production-rule table /
   recursive-descent checker.
7. **AgentSCAD `printable_rules.md`** -- feature-typed FDM minima (wall 1.2/2.0,
   through-hole 2.0/3.0, blind 3.0/4.0, boss 3.0/5.0, text 0.5, gap 0.2/0.4,
   bridge 20mm, merge tol 0.2). Extends printability_verdict with FEATURE_MINIMA.
8. **sdfx `obj/servo.go` + `gridfinity.go`** -- named servo dimension DB (Hitec
   HS-40/55/85BB...), Gridfinity envelope constants, geneva/keyway/standoff
   generators. Only sdf/screw.go was ported. Servo mounts are a classic prompt.
9. **IntentForge rule-pack DSL** (`knowledge/packs/data/*.yaml` +
   capability/evidence schemas + review_policies + golden_cases) -- versioned
   declarative rules (condition expressions, required_metrics, severity, confidence,
   tradeoffs, depends_on). Build: `domain/fabrication/rule_packs.py` evaluating
   condition expressions against measured metrics.
10. **OpenCAD `caid-design-artifact/patch-v1` schemas** -- versioned strict
    artifact/patch interchange. NOTE: already built this wave as
    `domain/spec/design_patch.py` by the OpenCAD build agent -- mark satisfied.

Also notable: anvilate ISO 2768 general tolerances + process-capability table;
Forma-OSS electronics component catalog (ESP32/Arduino/DHT22 pin schemas -- if
enclosure flow in scope); text-to-cad printer wrapper-profile schema + G-code bounds
validation (build printer_profiles.py); RapCAD openscad.bnf (OpenSCAD grammar FSA for
validating generated .scad); anvilate dimension-checked Quantity pattern (optional
small type for contract predicates).

## Skips (verified)

SKIP-exists: modeling-app gear formulas (gear_train/bevel_gear ported), BikeBench
validation (bikebench_metrics mined), kittycad spec.json (zoo_catalog mined),
BrickGPT brick_library (ported), anvilate evidence/citation idea (mined into
evidence_bundle/accounting). SKIP-irrelevant: Roshera prompt vibes-numbers (uncited,
contrary to citation discipline), CAD-MCP config, curv grammar (no backend),
bench prompt assets (bench-mined separately).

## Honest coverage note

~25 of ~120 repos inspected in depth (targeted greps: thread/DIN/ISO286/density/
gear/grammar/schema + every data/, standards/, knowledge/, schemas/, grammar dir).
NOT inspected file-by-file: big kernels (pythonocc, OCP, oce, manifold, libfive,
solvespace, ruststep -- no standards tables surfaced), ML-dataset repos (schemas
appear mined per csg_vocabulary/autobrep_serialize/cadmium_sequence), ~15 UI repos.
Possible residual: DeepCAD/SkexGen quantization parameter-range constants not
verified against reconstruction modules. Full-tree grep timed out on .history/venv
noise (AlphaCAD, Text-to-CAD-dean) -- switched to rg with exclusions; matches only
inside vendored/noise dirs deliberately skipped. zip/CodeToCAD-develop duplicate not
separately swept.
