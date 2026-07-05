# Text-to-CAD Paper Implementation Campaign — Loop Protocol

**Goal:** exhaustively scan every paper in the 186-paper corpus
`resources/Text-to-CAD + Spatial Intelligence/extracted-md/`, extract every
DETERMINISTIC, LOCALLY-BUILDABLE idea not already in the codebase, and implement
each as a tested module — continuously, in batches of 5, until all 186 are done.

## State (single source of truth)

`audit/text_to_cad_progress.json`
- `total_papers` (186), `reviewed_count`, `next_manifest_index`, `reviewed[]`.
- Paper order = row order in
  `resources/Text-to-CAD + Spatial Intelligence/extracted-md/_manifest.md`
  (row N of the `| [paper](...) |` table = paper index N).

Idea log: `TEXT_TO_CAD_PAPER_IDEAS.md` — one `### N. <Paper>` section per paper
with a `| Build idea | Status | Repository comparison |` table. Status is
`implemented` (with `path/module.py`), or `research-heavy/external` /
`external` for learned-model/training/proprietary/licensed ideas.

## Per-batch loop (5 papers)

1. Read `next_manifest_index` from the progress JSON. The batch is papers
   `[N .. N+4]` in manifest order.
2. For each paper, launch one implementation agent (fork/general-purpose) that:
   - reads the paper markdown fully;
   - checks the existing package inventory to avoid duplication;
   - implements every deterministic buildable idea as NEW files (distinctive
     names prefixed by the paper concept; do NOT edit existing files, any
     `__init__.py`, or `pyproject.toml`);
   - adds `tests/test_<name>.py` (unittest, stdlib, deterministic) and iterates
     until green;
   - returns the `### N.` idea-table rows.
3. Integrate: verify per-module tests pass, register new top-level packages in
   `pyproject.toml`, reconcile any registry-derived consumers (mcp/grammar
   fallbacks already auto-cover new ops).
4. Commit granularly, no `Co-Authored-By` trailer:
   - `feat: <idea>` per module group,
   - `build: register <packages>` if new packages,
   - `docs: close papers N through N+4`,
   - `docs: update suite count after <k>th paper batch` (bump README badge to
     the real per-module test total; a monolithic `unittest discover` segfaults
     at OCCT teardown — count per module).
   Push after each batch.
5. Update `audit/text_to_cad_progress.json` (`reviewed_count += 5`,
   `next_manifest_index += 5`, append `reviewed[]` entries) and append the
   idea-tables to `TEXT_TO_CAD_PAPER_IDEAS.md` with a
   `## Batch-<k> implementation result` note.
6. Schedule the next batch (continuous loop) until `next_manifest_index > 186`.

## Conventions

Python stdlib-only, absolute imports, deterministic (no wall clock; seed any
randomness). No emojis. Tests in `tests/` MUST be `unittest.TestCase` classes
(NOT bare pytest `def test_` functions) so they are collected by the canonical
`python -m unittest` runner — verify each paper's tests print `OK` (not
`NO TESTS RAN`) before committing. Skip — do not fake — anything that
needs a trained model, GPU, optimal-transport tooling, proprietary host
(Rhino/Revit/Fusion), or licensed data; log it as `research-heavy/external`.

## Health check

`for t in $(ls tests/test_*.py | sed 's#tests/##;s#\.py##'); do python -m unittest tests.$t; done`
(run per module; sum `Ran N`). As of batch 9: 1575 tests, all passing.
