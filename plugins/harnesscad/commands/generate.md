---
name: generate
description: generation strategies driving a session (deterministic stub planner)
---

# harnesscad generate

generation strategies driving a session (deterministic stub planner)

## Usage

```bash
harnesscad generate [brief] [--list] [--rivals] [--unadapted] [--strategy <strategy>] [--planner <stub|llm>] [--backend <stub|cadquery>] [--retrieve <retrieve>] [--retrieval-backend <hybrid|sphere_knn>] [--context] [--json]
```

## Arguments

- `brief`: the natural-language design brief
- `--list`: list the generation strategies
- `--rivals`: list the rival families (never blended)
- `--unadapted`: list generation modules with no call site yet
- `--strategy`: the generation strategy to run (default: direct) (default: direct)
- `--planner`: stub = deterministic, no model (default) (choices: stub, llm; default: stub)
- `--backend`:  (choices: stub, cadquery; default: stub)
- `--retrieve`: retrieve capability modules for a query
- `--retrieval-backend`:  (choices: hybrid, sphere_knn; default: hybrid)
- `--context`: print the assembled, token-budgeted context
- `--json`:

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
