---
name: numeric
description: numeric building blocks: diffusion, flow/ODE, noise schedules, multiscale, distillation, state-space
---

# harnesscad numeric

numeric building blocks: diffusion, flow/ODE, noise schedules, multiscale, distillation, state-space

## Usage

```bash
harnesscad numeric [--list] [--groups] [--schedule <schedule>] [--transition <transition>] [--unadapted] [--json]
```

## Arguments

- `--list`: list every numeric route (default)
- `--groups`: list the numeric theme groups
- `--schedule`: sample the sqrt noise schedule at step T of STEPS
- `--transition`: a uniform categorical transition matrix
- `--unadapted`: list numeric modules with no route
- `--json`: emit JSON instead of text

This file is generated from the live CLI parser by
`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.
