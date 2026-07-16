---
name: search
description: design-space search over a session (--list/--rivals/--strategy)
---

# harnesscad search

design-space search over a session (--list/--rivals/--strategy)

## Usage

```bash
harnesscad search [--list] [--rivals] [--unadapted] [--strategy <strategy>] [--ops <ops>] [--target <target>] [--seed <seed>] [--space] [--json]
```

## Arguments

- `--list`: list the search strategies
- `--rivals`: list the rival families (selected by name, never blended)
- `--unadapted`: list exploration modules with no call site yet
- `--strategy`: the strategy to run
- `--ops`: the starting design (default: the built-in demo)
- `--target`: the target model the objective minimises toward
- `--seed`:  (default: 0)
- `--space`: print the searchable design space and exit
- `--json`:

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
