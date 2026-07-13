"""Tests for the ingest layer — import / decompile / metadata.

Every path here PASSES without cadquery/OCCT installed: the modules import
cleanly and degrade to clear 'unavailable' / 'metrics-only' results rather than
crashing. Real-kernel assertions are guarded behind cadquery availability.
"""

import os
import tempfile
import unittest

from harnesscad.io.ingest import (
    ImportedPart, import_solid, detect_format,
    DecompileResult, decompile,
    PartMetadata, extract_metadata,
    precedent_text, index_precedent,
)
from harnesscad.io.ingest.decompile import _decompile_from_metrics
from harnesscad.core.cisp.ops import NewSketch, AddRectangle, Extrude
from harnesscad.io.backends.stub import StubBackend


try:  # optional real-kernel gate
    import cadquery as _cq  # noqa: F401
    _HAVE_CQ = True
except Exception:  # noqa: BLE001
    _HAVE_CQ = False


class _MeasuringBackend:
    """A fake backend that answers 'measure' like a real kernel would."""

    def __init__(self, bbox=(20.0, 10.0, 5.0), volume=1000.0):
        self._bbox = bbox
        self._volume = volume

    def query(self, q):
        if q == "measure":
            return {"bbox": list(self._bbox), "volume": self._volume}
        return {}


def _build_plate(backend):
    for op in (NewSketch(plane="XY"),
               AddRectangle(sketch="sk1", w=20.0, h=10.0),
               Extrude(sketch="sk1", distance=5.0)):
        res = backend.apply(op)
        assert res.ok, res.diagnostics
    return backend


# --------------------------------------------------------------------------- #
# import_solid
# --------------------------------------------------------------------------- #
class TestImportSolid(unittest.TestCase):
    def test_detect_format(self):
        self.assertEqual(detect_format("a.step"), "step")
        self.assertEqual(detect_format("a.STP"), "step")
        self.assertEqual(detect_format("a.iges"), "iges")
        self.assertEqual(detect_format("a.igs"), "iges")
        self.assertEqual(detect_format("a.stl"), "stl")
        self.assertEqual(detect_format("a.obj"), "unknown")

    def test_missing_path_is_clean_unavailable(self):
        part = import_solid("does/not/exist.step")
        self.assertIsInstance(part, ImportedPart)
        self.assertFalse(part.available)
        self.assertFalse(part.ok)
        self.assertIn("not found", part.note)
        self.assertEqual(part.bbox, [0.0, 0.0, 0.0])
        self.assertEqual(part.volume, 0.0)

    def test_unsupported_format_is_clean_unavailable(self):
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as fh:
            fh.write(b"not a solid")
            path = fh.name
        try:
            part = import_solid(path)
            self.assertFalse(part.available)
            self.assertIn("unsupported format", part.note)
        finally:
            os.remove(path)

    def test_empty_string_path(self):
        part = import_solid("")
        self.assertFalse(part.available)
        self.assertTrue(part.note)

    def test_to_dict_is_structured(self):
        d = import_solid("nope.step").to_dict()
        for key in ("path", "fmt", "available", "metrics", "bbox", "note"):
            self.assertIn(key, d)

    @unittest.skipUnless(_HAVE_CQ, "cadquery/OCCT not installed")
    def test_roundtrip_real_step(self):
        # Export a plate from the real backend, then import it back and measure.
        from harnesscad.io.backends.cadquery_backend import CadQueryBackend
        be = _build_plate(CadQueryBackend())
        step_text = be.export("step")
        with tempfile.NamedTemporaryFile(
                suffix=".step", delete=False, mode="w") as fh:
            fh.write(step_text)
            path = fh.name
        try:
            part = import_solid(path)
            self.assertTrue(part.available, part.note)
            self.assertGreater(part.volume, 0.0)
            self.assertEqual(len(part.bbox), 3)
        finally:
            os.remove(path)


# --------------------------------------------------------------------------- #
# decompile
# --------------------------------------------------------------------------- #
class TestDecompile(unittest.TestCase):
    def test_stub_part_metrics_only_note(self):
        # StubBackend cannot measure -> clear metrics-only note, no crash.
        backend = _build_plate(StubBackend())
        result = decompile(backend)
        self.assertIsInstance(result, DecompileResult)
        self.assertEqual(result.method, "none")
        self.assertFalse(result.ops)
        self.assertIn("metrics-only", result.note)
        self.assertEqual(result.confidence, 0.0)

    def test_measurable_backend_yields_prismatic_ops(self):
        # A backend that answers 'measure' -> a plausible prismatic op list.
        result = decompile(_MeasuringBackend(bbox=(20.0, 10.0, 5.0)))
        self.assertTrue(result.ops)
        self.assertEqual(result.method, "metrics-bbox")
        kinds = [type(op).__name__ for op in result.ops]
        self.assertEqual(kinds, ["NewSketch", "AddRectangle", "Extrude"])
        rect = result.ops[1]
        self.assertAlmostEqual(rect.w, 20.0)
        self.assertAlmostEqual(rect.h, 10.0)
        self.assertAlmostEqual(result.ops[2].distance, 5.0)
        self.assertGreater(result.confidence, 0.0)

    def test_imported_part_metrics_path(self):
        part = ImportedPart(path="x.step", fmt="step",
                            metrics={"bbox": [8.0, 8.0, 3.0], "volume": 192.0},
                            bbox=[8.0, 8.0, 3.0], available=True)
        result = decompile(part)
        self.assertTrue(result.ops)
        self.assertEqual(result.method, "metrics-bbox")

    def test_no_metrics_returns_empty_note(self):
        result = _decompile_from_metrics({})
        self.assertFalse(result.ops)
        self.assertIn("no measurable", result.note)

    def test_result_to_dict(self):
        d = decompile(_MeasuringBackend()).to_dict()
        for key in ("ops", "confidence", "note", "method", "face_summary"):
            self.assertIn(key, d)
        self.assertTrue(all("op" in o for o in d["ops"]))

    def test_deterministic(self):
        a = decompile(_MeasuringBackend(bbox=(20.0, 10.0, 5.0)))
        b = decompile(_MeasuringBackend(bbox=(20.0, 10.0, 5.0)))
        self.assertEqual(a.to_dict(), b.to_dict())

    @unittest.skipUnless(_HAVE_CQ, "cadquery/OCCT not installed")
    def test_real_box_recovers_prismatic(self):
        from harnesscad.io.backends.cadquery_backend import CadQueryBackend
        be = _build_plate(CadQueryBackend())
        result = decompile(be)
        self.assertTrue(result.ops)
        self.assertEqual(result.method, "brep-faces")
        self.assertGreaterEqual(result.confidence, 0.3)
        self.assertEqual(result.face_summary.get("planar"), 6)


# --------------------------------------------------------------------------- #
# extract_metadata
# --------------------------------------------------------------------------- #
class TestMetadata(unittest.TestCase):
    def test_missing_path_is_empty_with_note(self):
        meta = extract_metadata("nope.step")
        self.assertIsInstance(meta, PartMetadata)
        self.assertFalse(meta.available)
        self.assertIn("not found", meta.note)
        self.assertEqual(meta.bom_lines, [])

    def test_non_step_is_empty_with_note(self):
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as fh:
            path = fh.name
        try:
            meta = extract_metadata(path)
            self.assertFalse(meta.available)
            self.assertIn("STEP", meta.note)
        finally:
            os.remove(path)

    def test_to_dict_structured(self):
        d = extract_metadata("nope.step").to_dict()
        for key in ("path", "name", "material", "bom_lines", "pmi",
                    "assembly_tree", "available", "note"):
            self.assertIn(key, d)

    @unittest.skipUnless(
        _HAVE_CQ and os.environ.get("HARNESSCAD_XCAF_TESTS"),
        "XCAF real-STEP test is opt-in (set HARNESSCAD_XCAF_TESTS=1); the "
        "cadquery-ocp XCAF application singleton segfaults the interpreter at "
        "teardown on some builds, which would crash the whole test process")
    def test_real_step_metadata(self):
        from harnesscad.io.backends.cadquery_backend import CadQueryBackend
        be = _build_plate(CadQueryBackend())
        step_text = be.export("step")
        with tempfile.NamedTemporaryFile(
                suffix=".step", delete=False, mode="w") as fh:
            fh.write(step_text)
            path = fh.name
        try:
            meta = extract_metadata(path)
            # A single-part STEP may or may not carry a product name; the record
            # must at least be structurally valid and non-crashing.
            self.assertIsInstance(meta, PartMetadata)
            self.assertIsInstance(meta.bom_lines, list)
        finally:
            os.remove(path)


# --------------------------------------------------------------------------- #
# RAG precedent ingestion
# --------------------------------------------------------------------------- #
class TestPrecedentIngestion(unittest.TestCase):
    def test_precedent_text_without_kernel(self):
        part = import_solid("nope.step")
        text = precedent_text(part)
        self.assertIn("Imported part", text)
        self.assertIn("format: step", text)

    def test_precedent_text_with_metrics(self):
        part = ImportedPart(path="p.step", fmt="step",
                            metrics={"volume": 100.0, "faces": 6},
                            bbox=[10.0, 5.0, 2.0], available=True)
        text = precedent_text(part)
        self.assertIn("volume: 100", text)
        self.assertIn("bbox:", text)

    def test_index_precedent_into_retriever(self):
        from harnesscad.agents.rag import HybridRetriever
        r = HybridRetriever()
        part = index_precedent(r, "bracket.step")
        self.assertIsInstance(part, ImportedPart)
        hits = r.retrieve("imported bracket step", k=3)
        self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main()
