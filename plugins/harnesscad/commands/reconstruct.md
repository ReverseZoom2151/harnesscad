---
name: reconstruct
description: reconstruction route registry (input kind -> output kind)
---

# harnesscad reconstruct

reconstruction route registry (input kind -> output kind)

## Usage

```bash
harnesscad reconstruct [--list] [--from <from_kind>] [--to <to_kind>] [--kinds] [--rivals] [--unadapted] [--show <show>] [--json]
```

## Arguments

- `--list`: list every discovered reconstruction route
- `--from`: filter by input kind (e.g. point_cloud)
- `--to`: filter by output kind (e.g. primitives)
- `--kinds`: list the input/output kinds routes are keyed by
- `--rivals`: list the rival families that are selected, never blended
- `--unadapted`: list reconstruction modules with no route yet
- `--show`: show one route by name
- `--json`: emit JSON

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
