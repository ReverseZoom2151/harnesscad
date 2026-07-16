---
name: gallery
description: rendered parts gallery: 16 parts, each exercising a different capability (--list / --build [--out DIR] [--only NAME])
---

# harnesscad gallery

rendered parts gallery: 16 parts, each exercising a different capability (--list / --build [--out DIR] [--only NAME])

## Usage

```bash
harnesscad gallery [--list] [--build] [--out <out>] [--only <only>] [--no-compare] [--json]
```

## Arguments

- `--list`: print the catalogue (name, capability, backend)
- `--build`: build + render + QC every catalogued part
- `--out`: output directory (default: assets\gallery) (default: assets\gallery)
- `--only`: render just this part
- `--no-compare`: skip the cross-backend comparison strip
- `--json`: emit the catalogue / manifest as JSON

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
