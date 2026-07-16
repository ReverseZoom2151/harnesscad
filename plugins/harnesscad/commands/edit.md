---
name: edit
description: edit surface: apply a parametric edit, diff it, or run an edit loop
---

# harnesscad edit

edit surface: apply a parametric edit, diff it, or run an edit loop

## Usage

```bash
harnesscad edit [--list] [--rivals] [--unadapted] [--ops <ops>] [--params] [--set <set>] [--strategy <strategy>] [--target <target>] [--seed <seed>] [--backend <stub|cadquery>] [--json]
```

## Arguments

- `--list`: list the edit kinds and the edit strategies
- `--rivals`: list the rival strategy families (never blended)
- `--unadapted`: list editing modules with no call site yet
- `--ops`: the op stream to edit (default: the built-in demo)
- `--params`: list the editable parameters of the op stream
- `--set`: apply one parametric edit and print the diff
- `--strategy`: run a named edit strategy toward --target
- `--target`: the target op stream the strategy edits toward
- `--seed`: strategy seed (default: 0)
- `--backend`:  (choices: stub, cadquery; default: stub)
- `--json`: emit the result as JSON

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
