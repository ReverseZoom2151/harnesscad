---
name: ingest
description: decode an existing model (CAD tokens or a mesh) into editable CISP ops
---

# harnesscad ingest

decode an existing model (CAD tokens or a mesh) into editable CISP ops

## Usage

```bash
harnesscad ingest <source> --family <deepcad|skexgen|hnc|vitruvion|mesh> [--backend <stub|cadquery|build123d|frep|blender|openscad|freecad|manifold|rhino3dm|microcad|truck>] [--arc-policy <chord|reject>]
```

## Arguments

- `source`: path to a tokens JSON document, or a .obj/.xyz mesh (required)
- `--family`: the token family to decode with -- MANDATORY and never guessed: the quantisers are mutually incompatible ('mesh' takes the point-cloud path) (choices: deepcad, skexgen, hnc, vitruvion, mesh; required)
- `--backend`:  (choices: stub, cadquery, build123d, frep, blender, openscad, freecad, manifold, rhino3dm, microcad, truck; default: stub)
- `--arc-policy`: CISP has no arc op: approximate arcs by their chord (default) or refuse them (choices: chord, reject; default: chord)

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
