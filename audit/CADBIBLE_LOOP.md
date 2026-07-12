# CADBible Repo Mining Campaign — Loop Protocol

**Goal:** read every repo in `resources/cadbible/` one by one, extract every
DETERMINISTIC, LOCALLY-BUILDABLE idea NOT already in the harness, and implement
each as a tested stdlib-only module — continuously, in batches of ~5 repos,
until all repos are done.

## State (single source of truth)

`audit/cadbible_progress.json`
- `total_repos`, `reviewed_count`, `next_repo_index`, `reviewed[]`.
- Repo order = alphabetical (see the `reviewed[]` list / `/tmp/cadbible_repos.txt`).

Idea log: `CADBIBLE_REPO_IDEAS.md` — one `### N. <repo>` section per repo with a
`| Build idea | Status | Repository comparison |` table.

## Critical context: the harness is already large

The 186-paper text-to-CAD campaign already produced ~700 tested modules
(`reviewed_count=186`, `campaign_complete=true` in
`audit/text_to_cad_progress.json`; the full inventory is in
`TEXT_TO_CAD_PAPER_IDEAS.md`). MANY cadbible repos are the reference
implementations of those papers (DeepCAD, GenCAD, hnc-cad, SketchGraphs,
Text2CAD, mrCAD, muse, querycad, CADTestBench, UV-Net, vitruvion, SkexGen,
Sketch2CAD, CAD2Program, Text-to-CadQuery, ...). For those, the paper-level
ideas are ALREADY built — so extract only genuinely-new IMPLEMENTATION-level
ideas the paper review missed (a concrete algorithm, data structure, numeric
routine, parser, utility) and mark the rest `already in repo (paper N)`.

The richest NEW source is the pure-tool repos that are NOT papers: geometry
kernels (manifold, libfive, sdf-csg, sdfx, curv), CadQuery / SolidPython /
OpenSCAD ecosystems, STEP tooling (ruststep, pythonocc-core, OCP), etc. These
contain transferable deterministic geometry algorithms (mesh boolean, SDF ops,
marching cubes variants, tessellation, constraint solving, transforms) worth
reimplementing in stdlib Python.

## Per-batch loop (~5 repos)

1. Read `next_repo_index`. The batch is repos `[N .. N+4]` in manifest order.
2. For each repo, launch one agent that:
   - reads the repo's key source (README + core modules) — NOT every file, but
     enough to find the deterministic algorithmic content;
   - checks the existing harness inventory to AVOID DUPLICATION (grep the
     package tree; consult TEXT_TO_CAD_PAPER_IDEAS.md for paper coverage);
   - implements every genuinely-new deterministic idea as NEW files
     (distinctive names prefixed by the repo concept; do NOT edit existing
     files, any `__init__.py`, or `pyproject.toml`). PLACE in the most-fitting
     topical package, NOT the repo root;
   - writes `tests/test_<name>.py` (unittest, stdlib, deterministic) IMMEDIATELY
     after each module and verifies OK before the next;
   - returns the `### N.` idea-table rows.
3. Integrate: verify per-module tests pass; commit granularly, no
   `Co-Authored-By` trailer; `feat: <idea> (<repo>)` per module group.
4. Update `audit/cadbible_progress.json` (`reviewed_count`, `next_repo_index`,
   append `reviewed[]`) + append idea-tables to `CADBIBLE_REPO_IDEAS.md`.
   Record the authoritative per-module suite total in the progress json
   (`suite_tests`) — a monolithic `unittest discover` segfaults at OCCT
   teardown, so count per module. Do NOT edit README during the campaign.
5. Schedule the next batch (continuous loop) until `next_repo_index > total`.

## Conventions

Python stdlib-only, absolute imports, deterministic (no wall clock; seed any
randomness). No emojis. Tests MUST be `unittest.TestCase` classes verified with
`python -m unittest`. Skip — do not fake — anything needing a trained model,
GPU, OCCT/OpenCascade kernel, proprietary host, or licensed data; log it as
`research-heavy/external`. A repo that is a pure paper re-impl already covered,
or is out-of-scope (pure UI/frontend/build-tooling with no transferable
deterministic algorithm), correctly yields little or nothing — that is fine.
Source language is irrelevant (Rust/Haskell/Clojure/C++/JS repos still yield
transferable deterministic algorithms to reimplement in stdlib Python).
