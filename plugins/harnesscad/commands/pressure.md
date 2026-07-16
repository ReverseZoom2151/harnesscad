---
name: pressure
description: pressure test: does a typed diagnostic beat a blind retry? (runs local ollama models through both loops and scores the geometry)
---

# harnesscad pressure

pressure test: does a typed diagnostic beat a blind retry? (runs local ollama models through both loops and scores the geometry)

## Usage

```bash
harnesscad pressure [--model <model>] [--loop <blind|harness|oracle_bon|self_consistency|both|all>] [--briefs <briefs>] [--seed <seed>] [--temperature <temperature>] [--max-attempts <max_attempts>] [--out <out>] [--cache <cache>] [--no-resume] [--report <report>] [--list-briefs] [--json]
```

## Arguments

- `--model`: ollama model to test; repeat to test several (default: the four frontier tags -- qwen3.6:27b, qwen3.6:35b, ornith:9b, ornith:35b)
- `--loop`: 'blind' = raw kernel errors, 'harness' = SOUND typed diagnostics, 'oracle_bon' = Best-of-N selected by the differential oracle + the output gate, 'self_consistency' = majority vote over the SAME N draws (free), 'both' = blind+harness (the v1 A/B), 'all' = every arm. Repeatable. Default: both (choices: blind, harness, oracle_bon, self_consistency, both, all)
- `--briefs`: 'all', 'traps', 'notraps', a category (plate/hole/bracket/flange/shell/fillet/trap_shell/trap_fillet/trap_hole), or a CSV of brief ids (default: all)
- `--seed`: model seed (recorded in the results file) (default: 20260713)
- `--temperature`:  (default: 0.0)
- `--max-attempts`: attempt budget per brief per arm (default: 4)
- `--out`: write/append the results JSON here (resumable) (default: pressure_results.json)
- `--cache`: directory for the model-output cache (default: .pressure_cache)
- `--no-resume`: ignore any existing cells in --out and re-run them
- `--report`: print the report for an existing results file and exit
- `--list-briefs`: print the brief corpus and exit
- `--json`: with --report, dump the aggregate as JSON

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
