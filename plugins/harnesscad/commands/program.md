---
name: program
description: code-CAD program surface: parse/validate/emit/review, by --lang
---

# harnesscad program

code-CAD program surface: parse/validate/emit/review, by --lang

## Usage

```bash
harnesscad program [--list] [--lang <cadquery|openscad|openecad|typed_csg|openmc_csg|freecad_expr|bpy>] [--parse <parse>] [--validate <validate>] [--review <review>] [--reference <reference>] [--emit <emit>] [--operations] [--unadapted] [--json]
```

## Arguments

- `--list`: list the languages and what each can do
- `--lang`: the language -- MANDATORY for any real work, never guessed (choices: cadquery, openscad, openecad, typed_csg, openmc_csg, freecad_expr, bpy)
- `--parse`: parse FILE as --lang and print the round-tripped source
- `--validate`: validate FILE as --lang
- `--review`: statically review FILE as --lang
- `--reference`: the reference program to review against (openscad)
- `--emit`: emit a neutral op program (JSON array) as --lang source
- `--operations`: list the neutral operation vocabulary emit() accepts
- `--unadapted`: list program modules with no capability binding yet
- `--json`: emit findings as JSON

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
