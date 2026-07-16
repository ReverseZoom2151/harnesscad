---
name: judge
description: deterministic CAD graders: cad-score, betti, best-of-n, compiler-review
---

# harnesscad judge

deterministic CAD graders: cad-score, betti, best-of-n, compiler-review

## Usage

```bash
harnesscad judge [--list] [--review <review>] [--betti <betti>] [--n-voids <n_voids>] [--unadapted] [--json]
```

## Arguments

- `--list`: list every judge route
- `--review`: a sketch-extrude op sequence (JSON or @file) to review
- `--betti`: a triangle-face list (JSON or @file) -> Betti triple
- `--n-voids`: void-shell count for --betti (default 0) (default: 0)
- `--unadapted`: list judge modules with no route
- `--json`: emit JSON instead of text

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
