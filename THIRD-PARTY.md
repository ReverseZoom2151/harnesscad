# Third-party material redistributed by HarnessCAD

This file records third-party work that HarnessCAD **redistributes** -- material
that is present in this repository as content, not merely as an influence on how
something was written. Each entry names the upstream project, its licence, what
was taken, and what was changed.

Borrowing an idea, an algorithm or a taxonomy does not create an entry here;
shipping someone else's expression or data does.

---

## IntentForge -- fabrication rule packs (Apache-2.0)

**Upstream:** IntentForge (`intentforge`), version 0.10.2
**Licence:** Apache License, Version 2.0 -- full text in
[`THIRD-PARTY-LICENSES/Apache-2.0.txt`](THIRD-PARTY-LICENSES/Apache-2.0.txt),
also at <https://www.apache.org/licenses/LICENSE-2.0>
**Local copy of the upstream tree:**
`resources/cad_repos/IntentForge-main/IntentForge-main`

**What is redistributed:** the four bracket rule packs -- `assembly.yaml`,
`manufacturing.yaml`, `mechanical.yaml`, `structural.yaml` from
`src/intentforge/knowledge/packs/data/` -- comprising ten fabrication rules with
their conditions, severities, confidences, recommendations, source references
and reasoning metadata. They live in
`src/harnesscad/domain/fabrication/rule_packs.py` as the `VENDORED_PACKS` dict.

**Changes made (Apache-2.0 s4(b)):** the rule data was transcribed from YAML
into a Python dict literal so it can be loaded without a YAML dependency. The
rule content itself is unmodified: every id, expression, threshold, severity,
confidence, `created_by`, `last_updated`, `source_reference` and
`metadata.migrated_from` field is carried through as upstream wrote it, and each
pack's `source` field records the YAML path it came from. The surrounding
evaluator, dataclasses and expression interpreter in that module are HarnessCAD's
own code and are not covered by this entry.

**Attribution notices retained (Apache-2.0 s4(c)):** upstream's LICENSE carries
the unfilled `Copyright [yyyy] [name of copyright owner]` placeholder from the
Apache appendix, so there is no named copyright holder to reproduce. The
authorship marker upstream does carry -- `created_by: "intentforge-team"` on
every rule -- is preserved verbatim in the vendored data.

**NOTICE file (Apache-2.0 s4(d)):** the upstream distribution ships no `NOTICE`
file, so there are no NOTICE contents to reproduce. If IntentForge adds one, its
contents must be added here.

---

## cadgenbench -- manifold3d metric-boolean conversion and policy (Apache-2.0)

**Upstream:** cadgenbench (`cadgenbench`)
**Licence:** Apache License, Version 2.0 -- full text in
[`THIRD-PARTY-LICENSES/Apache-2.0.txt`](THIRD-PARTY-LICENSES/Apache-2.0.txt),
also at <https://www.apache.org/licenses/LICENSE-2.0>
**Local copy of the upstream tree:**
`resources/cad_repos/cadgenbench-main/cadgenbench-main`

**What is redistributed:** the mesh<->manifold conversion logic of
`src/cadgenbench/eval/booleans.py` -- building a `manifold3d.Mesh` from a
vertex array plus a `uint32` triangle-index array, constructing a
`manifold3d.Manifold` from it, and rejecting the result by comparing the
manifold's status enum *by name* against `"NoError"` (a forward-compatibility
detail upstream arrived at, not an obvious one) and by `is_empty()`. It lives in
`src/harnesscad/eval/verifiers/metric_booleans.py` as `mesh_to_manifold` /
`manifold_to_mesh_arrays`, and again as `_ingest` in
`src/harnesscad/eval/verifiers/_metric_boolean_worker.py`.

Taken with it, and credited here rather than treated as independent work: the
**policy** that no metric may be computed with an OCCT boolean and that every
metric-side boolean routes through `manifold3d` instead; the marker set that
defines what "an OCCT boolean can reach this module" means (`BRepAlgoAPI`,
`BOPAlgo`, and the `build123d` operator route), from upstream's
`tests/eval/test_interface_viz_no_occt.py`; and the **1.0 mm^3 sub-epsilon
noise rule** of `eval/interface_match_viz.py` -- the specific threshold below
which an overlap volume is tessellation residue rather than a clash, carried
over as `OVERLAP_NOISE_EPSILON`.

**Changes made (Apache-2.0 s4(b)):** the conversion was retyped onto
HarnessCAD's own `MeshData` (plain Python vertex/triangle lists, so the module
has no module-scope numpy import) and made total -- it returns `None` instead of
raising on any failure, because every caller degrades to a bounding-box
approximation rather than propagating. The policy test was reimplemented as a
*token-level* source scan (`occt_boolean_offenders`) that fails closed on an
unreadable or untokenisable module, so prose discussing the banned APIs is not
an offence. Everything around the conversion is HarnessCAD's own code and is not
covered by this entry: the OCCT tessellation and grid-quantised vertex weld
(`shape_to_mesh`) that feeds it -- our inputs are B-rep shapes where
cadgenbench's were already meshes -- the native manifold3d swept-cylinder
construction, and the subprocess watchdog (`intersection_volume_isolated` and
the `_metric_boolean_worker` module) that time-bounds one boolean in a killable
child process.

**Attribution notices retained (Apache-2.0 s4(c)):** upstream's `LICENSE` names
a copyright holder on its first line -- `Copyright 2026 Hugging Face` -- which is
reproduced in this entry and in the `ATTRIBUTION` section of both HarnessCAD
modules named above.

**NOTICE file (Apache-2.0 s4(d)):** the upstream distribution ships no `NOTICE`
file (its root holds `LICENSE`, `README.md`, `pyproject.toml`, `src/`, `tests/`,
`docs/` and dotfiles only), so there are no NOTICE contents to reproduce. If
cadgenbench adds one, its contents must be added here.
