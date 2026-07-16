---
name: export
description: run ops (or the demo) and write the model to <out>
---

# harnesscad export

run ops (or the demo) and write the model to <out>

## Usage

```bash
harnesscad export <out> [--ops <ops>] [--backend <stub|cadquery|build123d|frep|blender|openscad|freecad|manifold|rhino3dm|microcad|truck>] [--force]
```

## Arguments

- `out`: output path; the extension picks the format (required)
- `--ops`: path to a JSON array of ops (default: the built-in demo)
- `--backend`:  (choices: stub, cadquery, build123d, frep, blender, openscad, freecad, manifold, rhino3dm, microcad, truck; default: stub)
- `--force`: write the artifact even when the output gate refuses it. The file is written AND a <name>.INVALID.json sidecar naming every failed measurement is written beside it. For debugging only.

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
