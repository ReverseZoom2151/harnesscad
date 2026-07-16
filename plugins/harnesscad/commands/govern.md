---
name: govern
description: governance surface: security gates, research evidence, audit closure
---

# harnesscad govern

governance surface: security gates, research evidence, audit closure

## Usage

```bash
harnesscad govern [--list] [--prompt <prompt>] [--trust <trust>] [--tool <tool>] [--ingest <ingest>] [--effect <effect>] [--unadapted] [--json]
```

## Arguments

- `--list`: list every governance route
- `--prompt`: run a prompt through the trust gate
- `--trust`: the trust tier of --prompt / --tool (default: untrusted)
- `--tool`: authorise a tool call at --trust
- `--ingest`: run a file path through the ingest policy gate
- `--effect`: two samples: {"a": [...], "b": [...]}
- `--unadapted`: list governance modules with no route
- `--json`: emit JSON instead of text

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
