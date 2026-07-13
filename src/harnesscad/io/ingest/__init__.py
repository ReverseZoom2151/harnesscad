"""ingest — STEP/B-rep import, feature-decompile and metadata/BOM extraction.

The reference-ingestion layer for HarnessCAD: it reads existing solids back into
the harness (the inverse of the CadQuery backend's export path) so they can be
used as *reference solids* to score generation against, *decompiled* into a
best-effort CISP feature tree, or *ingested* into the RAG store as retrievable
precedents.

Three entry points, all OCCT-guarded so the whole package imports and runs with
no geometry kernel installed:

    from harnesscad.io.ingest import import_solid, decompile, extract_metadata
    part = import_solid("bracket.step")     # measurable reference solid
    tree = decompile(part)                  # best-effort CISP op list
    meta = extract_metadata("bracket.step") # BOM / PMI / assembly tree (XCAF)

The reference-match verifier that scores a generated model against an imported
reference lives in the top-level ``checks_reference`` module.
"""

from __future__ import annotations

from harnesscad.io.ingest.import_brep import (
    ImportedPart, import_solid, detect_format,
    precedent_text, index_precedent,
)
from harnesscad.io.ingest.decompile import DecompileResult, decompile
from harnesscad.io.ingest.metadata import PartMetadata, extract_metadata
from harnesscad.io.ingest.fidelity import FidelityReport, import_fidelity, roundtrip_fidelity

__all__ = [
    "ImportedPart",
    "import_solid",
    "detect_format",
    "precedent_text",
    "index_precedent",
    "DecompileResult",
    "decompile",
    "PartMetadata",
    "extract_metadata",
    "FidelityReport",
    "import_fidelity",
    "roundtrip_fidelity",
]
