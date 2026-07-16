---
name: dataset
description: data-engine pipeline (--list/--presets/--rivals, or run a named pipeline)
---

# harnesscad dataset

data-engine pipeline (--list/--presets/--rivals, or run a named pipeline)

## Usage

```bash
harnesscad dataset [--list] [--presets] [--rivals] [--unadapted] [--kind <generate|annotate|curate|augment|reward|preference|selftrain|audit|emit>] [--pipeline <pipeline>] [--input <input>] [--out <out>] [--seed <seed>] [--target-size <target_size>] [--generate <generate>] [--json]
```

## Arguments

- `--list`: list every discovered stage
- `--presets`: list the named pipelines and the stages each selects
- `--rivals`: list the rival stage families that must never be blended
- `--unadapted`: list data modules with no stage yet
- `--kind`: filter --list by stage kind (choices: generate, annotate, curate, augment, reward, preference, selftrain, audit, emit)
- `--pipeline`: run this named pipeline
- `--input`: path to a JSON array of records for --pipeline
- `--out`: write the emitted records as JSONL
- `--seed`: the run seed (default 0) (default: 0)
- `--target-size`: down-selection budget for curate.subset (0 = keep all) (default: 0)
- `--generate`: generate N verified samples first (bootstrap pipeline) (default: 0)
- `--json`: print the dataset as JSON

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
