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
