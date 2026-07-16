---
name: procedural
description: named procedural generators that emit CISP ops (--list/--rivals/--gen)
---

# harnesscad procedural

named procedural generators that emit CISP ops (--list/--rivals/--gen)

## Usage

```bash
harnesscad procedural [--list] [--rivals] [--unadapted] [--gen <gen>] [--param <param>] [--apply] [--json]
```

## Arguments

- `--list`: list every registered procedural route
- `--rivals`: list the rival families (selected by name, never blended)
- `--unadapted`: list procedural modules with no route
- `--gen`: the generator to run (see --list)
- `--param`: a generator parameter (repeatable); V is parsed as JSON (default: [])
- `--apply`: apply the emitted ops to a stub-backed HarnessSession
- `--json`: emit JSON instead of text

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
