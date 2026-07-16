---
name: ui
description: interaction surface: command grammar, prediction, overlays, views
---

# harnesscad ui

interaction surface: command grammar, prediction, overlays, views

## Usage

```bash
harnesscad ui [--list] [--parse <parse>] [--commands] [--views <views>] [--unadapted] [--json]
```

## Arguments

- `--list`: list every interaction route
- `--parse`: parse one command line into a typed intent
- `--commands`: list the commands available in the initial state
- `--views`: print N canonical prompt cameras
- `--unadapted`: list surface modules with no route, and why
- `--json`: emit JSON instead of text

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
