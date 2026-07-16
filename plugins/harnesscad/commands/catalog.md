---
name: catalog
description: parts catalogue + standards knowledge base (--parts/--find/--part/--thread/--heatsert/--aci)
---

# harnesscad catalog

parts catalogue + standards knowledge base (--parts/--find/--part/--thread/--heatsert/--aci)

## Usage

```bash
harnesscad catalog [--list] [--parts] [--find <find>] [--part <part>] [--param <param>] [--apply] [--thread <thread>] [--heatsert <heatsert>] [--wall <wall>] [--aci <aci>] [--unadapted] [--json]
```

## Arguments

- `--list`: list every catalogue route
- `--parts`: list the execution-verified parts
- `--find`: retrieve parts by function tag or free query
- `--part`: instantiate this part (with --param K=V) and print its ops
- `--param`: a part parameter (repeatable); V is parsed as JSON (default: [])
- `--apply`: apply the instantiated part's ops to a stub session
- `--thread`: look up a standard thread (e.g. M6)
- `--heatsert`: look up a heat-set insert bore schedule (e.g. M4)
- `--wall`: wall thickness for the --heatsert fit check
- `--aci`: an ACI colour name or index
- `--unadapted`: list library/standards modules with no route
- `--json`: emit JSON instead of text

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
