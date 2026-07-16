---
name: fabricate
description: manufacturing surface: workflows, feasibility, readiness, flat-pack, bricks, export planning
---

# harnesscad fabricate

manufacturing surface: workflows, feasibility, readiness, flat-pack, bricks, export planning

## Usage

```bash
harnesscad fabricate [--list] [--workflows] [--machines] [--analyze <analyze>] [--bbox <bbox>] [--volume <volume>] [--machine <machine>] [--material <material>] [--readiness <readiness>] [--unadapted] [--json]
```

## Arguments

- `--list`: list every manufacturing route
- `--workflows`: list the manufacturing workflows
- `--machines`: list the machines
- `--analyze`: feasibility-check --bbox against this workflow
- `--bbox`: the part envelope in mm
- `--volume`: the part volume in mm^3 (default: 0.0)
- `--machine`: the machine id
- `--material`: the material id
- `--readiness`: a descriptor (JSON object or @file) for the readiness gate
- `--unadapted`: list fabrication modules with no route
- `--json`: emit JSON instead of text

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
