"""datagen — the synthetic-data bootstrap generator (docs/blueprint.md sec.21).

Data is the #1 named risk (there is no "GitHub for CAD"). This package is the
cold-start answer: **synthetic parametric generation + solver-in-the-loop for ground
truth**. Generators (datagen.generators) cheaply emit (NL brief -> CISP ops) pairs;
the pipeline (datagen.pipeline) runs each through a HarnessSession and keeps only the
parts that verifiably build, tagging each with its digest + summary. It complements the
bench/ eval harness (which scores an agent) and the memory/skills library (which stores
verified skills) by manufacturing the training/eval examples both consume.
"""

from __future__ import annotations

from harnesscad.data.datagen.generators import (
    ParametricSampler,
    Generator,
    DEFAULT_GENERATORS,
    gen_plate,
    gen_bracket,
    gen_plate_with_holes,
)
from harnesscad.data.datagen.pipeline import (
    Sample,
    DatasetReport,
    generate_dataset,
    generate_dataset_report,
    to_jsonl,
    read_jsonl,
    verifiers_as_labor,
)

__all__ = [
    "ParametricSampler",
    "Generator",
    "DEFAULT_GENERATORS",
    "gen_plate",
    "gen_bracket",
    "gen_plate_with_holes",
    "Sample",
    "DatasetReport",
    "generate_dataset",
    "generate_dataset_report",
    "to_jsonl",
    "read_jsonl",
    "verifiers_as_labor",
]
