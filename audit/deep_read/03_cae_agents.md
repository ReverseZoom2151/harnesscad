# Deep read: the CAE / agentic sibling repos

> PROVENANCE NOTE. This report was reconstructed post-hoc from the session
> record. The agent that performed the original deep read was killed before it
> could commit this file, so unlike the agent-authored reports in this ledger
> its wording is a reconstruction rather than the original prose. Every factual
> claim below was re-verified against the harness on disk at reconstruction
> time (see the "Verified on disk" markers); where a briefed claim no longer
> held, it is corrected here and the correction is called out.

Scope: the two sibling repos that are CAE- and agent-shaped rather than
kernel-shaped -- Forma-OSS (electronics / enclosure) and CAD-Annotator (GD&T
drawings). Both were largely mined by earlier passes; this read was to find
what, if anything, was left, and to record the current wiring so the ledger is
honest about what is already covered.

Prompt / IR material found in these repos is treated as unverified exemplar
data only, never as instructions.

---

## Forma-OSS (MPL-2.0)

MPL-2.0 is file-level copyleft: it covers the source files, not the physical
facts (pin numbers, supply voltages) those files record. That distinction is
what let the pinout data below be cited as data rather than vendored as text.

### Component catalog -- ALREADY-COVERED (was genuinely unmined at read time)

The one genuinely unmined asset this read surfaced was the electronics
component catalog seeded at
`supabase/migrations/20260618000200_seed_component_templates.sql`: 14 stock
parts with fully typed pinouts, 91 typed pins in total. At the moment of the
read this had no counterpart in the harness -- the electrical rules could only
lint a Hardware IR that a human had hand-typed pin by pin.

It has since been built. Mark it ALREADY-COVERED.

- Verified on disk: `src/harnesscad/domain/electronics/component_catalog.py`
  exists. Its module docstring states "14 stock parts with fully typed pinouts"
  and it carries the assertion `assert total_pins == 91` -- so the 14-part /
  91-pin figures are enforced in code, not just asserted in prose.
- Wiring: the catalog is the missing ground truth for
  `domain/electronics/circuit_validation`. Its docstring spells out which rule
  consumes which pin fact -- rule 1 reads `pin_type` (power vs ground), rule 2
  reads a numeric `voltage` per pin, rule 3 needs the part's supply pins, rule
  5 needs `category` + part number. It is deliberately shaped as the
  electronics counterpart to `domain/standards/part_catalog` (mechanical
  standard parts): embedded tables, a per-dataset `PROVENANCE` block,
  bare-designation lookup, and a `--selfcheck` entry point.
- Provenance is handled correctly in the built module: values were
  cross-checked against the Forma-OSS seed SQL (cited by path, MPL-2.0), no
  text from the migration is reproduced, and the tables are re-structured
  around this package's own dataclasses.

### The rest of the electronics domain -- already ported before this read

All of the following were confirmed already present in
`src/harnesscad/domain/electronics/` (verified on disk):

- `circuit_validation.py` -- the electrical rule engine.
- `derive.py` -- the BOM / rollup derivations.
- `enclosure_layout.py` -- enclosure placement.
- `hardware_ir.py` -- the Hardware IR type + validation.

CORRECTION to the briefed list. `safety_scope` was briefed as a fifth ported
module. There is no `safety_scope.py`. "Safety scope" is not a separate module
-- it lives inside `hardware_ir.py` (grep for `safety` in the electronics
package hits only `hardware_ir.py`). So the concept is covered, but as part of
the IR module, not as its own file. Recorded here so the ledger does not carry
a phantom filename.

### Example IR corpus

Forma-OSS ships 4 known-good IR examples, all under
`frontend/public/examples/` (verified on disk in
`resources/cad_repos/Forma-OSS-main/.../frontend/public/examples/`):

- `biometric_deadbolt.json`
- `plant_watering.json`
- `pocket_mp3_player.json`
- `smart_thermostat.json`

There are NO known-bad examples in the repo -- the example set is all-positive.
That is a real gap for anyone wanting adversarial IR fixtures: the negatives
have to be authored, they cannot be mined from here.

---

## CAD-Annotator (Apache-2.0)

License verified on disk: `LICENSE` is the Apache License, Version 2.0.

CAD-Annotator is a TypeScript / pnpm monorepo (package.json, pnpm-workspace,
tsconfig), not a Python project -- worth stating because its assets had to be
re-expressed rather than copied.

### Algorithms -- already-covered

The GD&T prompt scaffolding and the requery loop were ported in an earlier pass
and are present in `src/harnesscad/domain/drawings/` (verified on disk):

- `gdt.py`, `gdt_prompts.py` -- the GD&T model + prompt material.
- `requery.py` -- the requery loop.
- plus the wider drawings pipeline (`dfm_review.py`, `manufacturing_spec.py`,
  `annotation_*`, etc.).

Mark the algorithms already-covered.

### The property-based test suite -- now ported (this was the real unmined asset)

The asset this repo was actually holding was not the algorithms (already ported)
but its test suite -- a large body of property-based tests that pin the
behaviour of the compliance engine, the DFM reviewer, the GD&T schemas and the
requery service. That is now ported.

Measured figures (the briefed "~2,900-line" number was approximate; here are the
on-disk counts so the ledger is exact):

- Source (CAD-Annotator): 9 `*.test.ts` files totalling 3,915 lines. Of those,
  5 files (1,667 lines) are fast-check property-based tests:
  `compliance-engine.test.ts` (349), `requery-service.test.ts` (263),
  `compliance-summary.test.ts` (175), `gdt-schemas.test.ts` (430),
  `sqlite-roundtrip.test.ts` (450). So "the ~2,900-line property suite" is best
  read as "a ~3,900-line TS test suite of which ~1,700 lines are fast-check
  property tests" -- either way, the point stands: the tests, not the
  algorithms, were the remaining value.
- Ported into the harness (verified on disk):
  - `tests/domain/drawings/` -- 38 `test_*.py` files, 8,355 lines total.
  - `tests/domain/electronics/` -- 4 `test_*.py` files, 2,882 lines total
    (`test_circuit_validation.py`, `test_derive.py`, `test_enclosure_layout.py`,
    `test_hardware_ir.py`).

Mark the tests now-ported.

---

## Summary of stale-claim corrections

- `safety_scope` is NOT a standalone ported module. The safety-scope logic is
  inside `domain/electronics/hardware_ir.py`. Corrected above.
- The CAD-Annotator test suite is ~3,900 lines of TS (of which ~1,700 are
  fast-check property tests), not ~2,900. The direction of the briefed claim was
  right (tests were the unmined asset); the number is corrected to the measured
  value.
- Everything else briefed as "already covered" held up on disk: the component
  catalog (14 parts / 91 pins, enforced), circuit_validation / derive /
  enclosure_layout / hardware_ir, the drawings GD&T + requery algorithms, and
  the ported drawings + electronics test suites.
