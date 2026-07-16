---
name: agent
description: agent surface: envelopes, gates, approval-gated edits, tool metrics
---

# harnesscad agent

agent surface: envelopes, gates, approval-gated edits, tool metrics

## Usage

```bash
harnesscad agent [--list] [--intent <intent>] [--envelope <envelope>] [--tools <tools>] [--unadapted] [--json]
```

## Arguments

- `--list`: list every agent route
- `--intent`: resolve one natural-language CAD instruction
- `--envelope`: parse a model reply (a file) into a typed plan envelope
- `--tools`: which tools this task needs, and what context is missing
- `--unadapted`: list agent modules with no route, and why
- `--json`: emit JSON instead of text

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
