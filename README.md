# harnesscad

Native agentic harness for engineering/mechanical text-to-CAD.

## Phase-0 spine

- **CISP typed ops** — a small, typed constraint/intent op vocabulary.
- **Event-sourced ops-DAG** — deterministic replay with checkpoint/rollback.
- **`GeometryBackend` protocol + stub backend** — pluggable geometry engine.
- **Plural verifier** — many independent checks over the produced state.
- **The loop** — `applyOps → regen → verify → checkpoint` with block-and-correct.

## Layout

```
harnesscad/
├── cisp/               # CISP typed ops
├── state/              # event-sourced ops-DAG, replay, checkpoint/rollback
├── backends/           # GeometryBackend protocol + stub backend
├── verify.py           # plural verifier
├── loop.py             # applyOps/regen/verify/checkpoint loop
├── tests/              # unittest suite
├── HARNESS_BLUEPRINT.md
├── pyproject.toml
└── README.md
```

## Run the tests

```
python -m unittest discover -s tests -t . -v
```

## Design doc

See [HARNESS_BLUEPRINT.md](HARNESS_BLUEPRINT.md) — the founding design doc.

## Note

Research/reference material (PDFs and research folders) lives under the
`resources/` directory, which is gitignored and not part of the product.
