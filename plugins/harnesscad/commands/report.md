---
name: report
description: quality analysis surface (--list/--rivals/--unadapted, or analyse a model)
---

# harnesscad report

quality analysis surface (--list/--rivals/--unadapted, or analyse a model)

## Usage

```bash
harnesscad report [--ops <ops>] [--backend <stub|cadquery|frep>] [--brief <brief>] [--extras <extras>] [--kind <geometry|sequence|physics|graph|assembly|edit|sketch|perception|reward|report>] [--only <only>] [--list] [--rivals] [--unadapted] [--json]
```

## Arguments

- `--ops`: path to a JSON array of ops (default: the built-in demo)
- `--backend`:  (choices: stub, cadquery, frep; default: stub)
- `--brief`: the design brief, so the traceability matrix can be built
- `--extras`: path to a JSON object of extra state projections (anomaly reference corpus, reward components, mesh, ...)
- `--kind`: restrict the report to one analyser kind (choices: geometry, sequence, physics, graph, assembly, edit, sketch, perception, reward, report)
- `--only`: comma-separated analyser names to run
- `--list`: list every analyser
- `--rivals`: list the rival analyser families (never averaged)
- `--unadapted`: list quality modules with no analyser yet
- `--json`: print the report as JSON

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
