# `resources/cad` coverage ledger

This ledger makes the research pass auditable. It separates source coverage from
implementation status; “covered” means the source was inspected for build ideas,
not that every cited research result was reimplemented.

> This high-level ledger has been superseded for idea-level completeness by the
> independent [CAD_CORPUS_AUDIT.md](CAD_CORPUS_AUDIT.md). The independent pass
> found additional partial and net-new ideas; therefore the corpus must not be
> described as exhausted.

## Inventory

| Source group | Files | Coverage | Evidence / disposition |
|---|---:|---|---|
| `pdf/*.pdf` | 6 | Covered through extracted twins | Each PDF has a corresponding full-text file under `extracted/`; figures were checked separately. |
| `extracted/*.txt` | 6 | Mined | Primary text used for the CAD, data-engine, Gaudi/BIM-transfer and competitor passes. |
| `extracted/chunks/*.txt` | 3 | Duplicate-checked | Partitioned copies of AI CAD Part 1; no independent source content. |
| `_fig/*.png` | 10 | Visually inspected | Nine Gaudi page renders duplicate the prose; `kinth_10.png` exposed the 2D engineering-drawing capability. |
| `docs/skills/*.md` | 17 | Mined | Research-agent workflow, review, experiment, orchestration and AlphaCAD patterns. |
| `docs/skills/standalone/*.md` | 17 | Variant-checked | Standalone rewrites were compared with the main skill set; no net-new workflow concepts were found. |
| `summaries/*.md` | 4 | Duplicate-checked | Derived summaries of AI CAD Part 1; primary full text takes precedence. |

Total inventory: 63 files (6 PDF, 9 TXT, 10 PNG and 38 Markdown).

## Primary sources

| File | Extracted twin | Main build ideas / transferable principles | Status |
|---|---|---|---|
| `pdf/AI CAD Research Paper Part 1 (FINAL).pdf` | `extracted/AI CAD Research Paper Part 1 _FINAL_.txt` | Expanded feature ops, repair, import/reference matching, standards, measurement/BOM, semantic diff, anomaly detection, physics, assembly and evaluation | Implemented across `cisp/`, `backends/`, `verifiers/`, `quality/`, `ingest/`, `bench/` |
| `pdf/AI CAD Research Paper Part 2 (FINAL).pdf` | `extracted/AI CAD Research Paper Part 2 _FINAL_.txt` | CISP editability, `SetParam`, assembly instances/mates, third verifier family, op-DAG bisect, metrics/export | Implemented |
| `pdf/Scale AI for Spatial Intelligence (FINAL).pdf` | `extracted/Scale AI for Spatial Intelligence _FINAL_.txt` | Trajectory logging, audit, active learning, consensus, intent capture, preference data | Implemented in `dataengine/` and `datagen/` |
| `pdf/Kinth Snap.pdf` | `extracted/Kinth Snap.txt` | Skeleton/master-sketch workflow, engineering sizing, parts, measure, simulation, BOM, animation and drawings | Implemented; drawing capability recovered from `_fig/kinth_10.png` |
| `pdf/Gaudi MVP_ 3-Month Execution Plan.pdf` | `extracted/Gaudi MVP_ 3-Month Execution Plan.txt` | Requirements graph, modular tools, clash/rules, traceability, conformance, fallback and QA | Mechanical-transfer implementations completed |
| `pdf/Roadmap for Building the Gaudi MVP Architectural Design Foundation Model.pdf` | `extracted/Roadmap for Building the Gaudi MVP Architectural Design Foundation Model.txt` | BIM-to-mechanical transfers: sequencing, access/serviceability, versioned standards, completeness, carbon, branching/merge and certification | Implemented in the recovered wave |

## Implemented idea map

| Idea cluster | Implementation |
|---|---|
| Mechanical operation vocabulary and editability | `cisp/ops.py`, `backends/`, `state/opdag.py` |
| Assembly, interference and kinematics | `verifiers/assembly.py`, `verifiers/interference.py`, `quality/kinematics.py`, `quality/assemblyseq.py` |
| Geometry validity, healing and fault localisation | `verifiers/geometry.py`, `reliability/repair.py`, `state/opdag.py` |
| Requirements, standards and compliance | `spec/`, `verifiers/requirements.py`, `verifiers/standards.py`, `verifiers/compliance.py`, `standards/` |
| Physics, tool access and plan feasibility | `verifiers/simulation.py`, `verifiers/access.py`, `verifiers/precheck.py` |
| Completeness, functional acceptance and certification | `verifiers/completeness.py`, `verifiers/functional.py`, `verifiers/report.py` |
| Quantitative design and sustainability | `quality/estimate.py`, `quality/fitness.py`, `quality/pareto.py` |
| Drawings, narration, QA and traceability | `quality/drawing.py`, `quality/describe.py`, `quality/ask.py`, `quality/traceability.py` |
| Import, decompile, reference and round-trip fidelity | `ingest/`, `verifiers/reference.py` |
| Recovery and human feedback | `reliability/fallback.py`, `dataengine/edit_pairs.py` |
| Search, exploration and evaluation | `reliability/strategies/`, `exploration/`, `bench/` |

## Explicit deferrals

These ideas require external systems or a materially larger product surface and
are not represented as completed features:

- Production FEA meshing/solving with CalculiX or Elmer. The analytic verifier
  and solver protocol exist; no solver result is fabricated.
- Scan/point-cloud-to-parametric-feature reconstruction.
- Full CAM/toolpath generation and machine simulation.
- A shipped interactive CAD canvas.
- GPU model training runs; only trace/preference exporters are implemented.
- Rust-native geometry-kernel replacement.

## Verification

The recovered corpus-derived wave is covered by focused tests and the integrated
suite. The current baseline is 1,178 passing tests with 48 optional-dependency
skips. Context-dependent checks remain opt-in; the default verifier set stays
limited to universally valid sketch, solid-presence and B-rep checks.
