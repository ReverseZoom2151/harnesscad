# Independent `resources/cad` audit

Status: in progress

This document records an independent source-by-source audit of the 63 files in
`resources/cad`. It does not treat a prior conversation summary as evidence.

## Method

1. The six PDFs are audited through their full extracted-text twins. Page images
   are inspected separately so diagrams and UI elements are not lost.
2. Extracted chunks and summaries are checked as derivative material rather than
   counted as independent sources.
3. Every primary and standalone skill document is reviewed; differences are
   checked for net-new concepts.
4. Candidate ideas are recorded with a source locator and classified as:
   `implemented`, `partial`, `net-new`, `deferred`, or `duplicate`.
5. Implementation claims require a concrete module or test path.

## Canonical inventory

| Type | Count | Canonical treatment |
|---|---:|---|
| PDF | 6 | Full extracted-text twin plus visual figure pass |
| TXT | 9 | Six primary extracts plus three derivative Part-1 chunks |
| PNG | 10 | Direct visual inspection |
| Markdown | 38 | 17 primary skills, 17 standalone variants, four derivative summaries |
| **Total** | **63** | |

## Primary source pairs

| PDF | Full-text representation |
|---|---|
| `pdf/AI CAD Research Paper Part 1 (FINAL).pdf` | `extracted/AI CAD Research Paper Part 1 _FINAL_.txt` |
| `pdf/AI CAD Research Paper Part 2 (FINAL).pdf` | `extracted/AI CAD Research Paper Part 2 _FINAL_.txt` |
| `pdf/Gaudi MVP_ 3-Month Execution Plan.pdf` | `extracted/Gaudi MVP_ 3-Month Execution Plan.txt` |
| `pdf/Kinth Snap.pdf` | `extracted/Kinth Snap.txt` |
| `pdf/Roadmap for Building the Gaudi MVP Architectural Design Foundation Model.pdf` | `extracted/Roadmap for Building the Gaudi MVP Architectural Design Foundation Model.txt` |
| `pdf/Scale AI for Spatial Intelligence (FINAL).pdf` | `extracted/Scale AI for Spatial Intelligence _FINAL_.txt` |

## Derivative-file map

- `extracted/chunks/part1_c1.txt`, `part1_c2.txt`, and `part1_c3.txt` partition
  the Part-1 primary extract.
- `summaries/part1_q1_lines_1_16500.md` through
  `part1_q4_lines_49501_65790.md` summarize the four consecutive quarters of
  the same Part-1 extract.
- `docs/skills/standalone/*.md` are audited as variants of the corresponding
  `docs/skills/*.md` documents; differing bytes alone are not accepted as proof
  of differing ideas.

## Findings

Findings are appended in source order as the audit proceeds.
