"""Tests for quality.drawing — 2D engineering-drawing sheet generation.

The module must produce a usable dimensioned drawing on the dependency-free
StubBackend (schematic path: no OCCT views, dimensions derived from the sketch
profiles + extrude depth), emit a self-contained SVG carrying the title-block
fields and the bounding-dimension text, save to a file, and be deterministic
under an injected date. When CadQuery/OCCT is installed a real solid embeds
multiple projected view groups whose dimensions reflect the model bbox.
"""

import os
import tempfile
import unittest

from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.cisp.ops import NewSketch, AddRectangle, Extrude

try:
    from harnesscad.core.cisp.ops import Hole
    HAVE_HOLE = True
except Exception:  # noqa: BLE001
    Hole = None
    HAVE_HOLE = False

from harnesscad.eval.quality.report.drawing import Drawing, make_drawing


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


HAVE_CQ = _cadquery_available()


def _stub_plate(w=60.0, h=40.0, thick=8.0, n_holes=0):
    b = StubBackend()
    assert b.apply(NewSketch(plane="XY")).ok
    assert b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=w, h=h)).ok
    assert b.apply(Extrude(sketch="sk1", distance=thick)).ok
    if HAVE_HOLE:
        for i in range(n_holes):
            assert b.apply(Hole(face_or_sketch="", x=5.0 + i, y=5.0,
                                diameter=5.0, through=True)).ok
    return b


class TestSchematicDrawing(unittest.TestCase):
    def test_returns_drawing_with_nonempty_svg(self):
        d = make_drawing(_stub_plate())
        self.assertIsInstance(d, Drawing)
        self.assertTrue(d.svg)
        self.assertIn("<svg", d.svg)
        self.assertTrue(d.svg.strip().endswith("</svg>"))

    def test_svg_contains_title_block_fields(self):
        d = make_drawing(_stub_plate(),
                         title_block={"part": "BRACKET-7", "material": "AL 6061"})
        for token in ("TITLE", "MATERIAL", "SCALE", "UNITS", "DATE",
                      "DRAWING NO.", "BRACKET-7", "AL 6061", "mm"):
            self.assertIn(token, d.svg, token)
        self.assertEqual(d.title_block["part"], "BRACKET-7")

    def test_svg_contains_bbox_dimension_text(self):
        d = make_drawing(_stub_plate(w=60.0, h=40.0, thick=8.0))
        self.assertEqual(d.dimensions["bbox"], [60.0, 40.0, 8.0])
        self.assertEqual(d.dimensions["source"], "derived")
        for token in ("60", "40", "8"):
            self.assertIn(token, d.svg, token)

    def test_schematic_has_note(self):
        d = make_drawing(_stub_plate())
        # Stub has no OCCT views -> schematic fallback carries a note.
        self.assertIsInstance(d.note, str)
        self.assertTrue(d.note)

    def test_save_writes_file(self):
        d = make_drawing(_stub_plate())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sheet.svg")
            d.save(path)
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            self.assertEqual(content, d.svg)

    def test_deterministic_with_injected_date(self):
        a = make_drawing(_stub_plate(), date="2026-07-05")
        b = make_drawing(_stub_plate(), date="2026-07-05")
        self.assertEqual(a.svg, b.svg)
        self.assertIn("2026-07-05", a.svg)

    def test_date_defaults_to_placeholder(self):
        d = make_drawing(_stub_plate())
        self.assertEqual(d.title_block["date"], "YYYY-MM-DD")

    def test_to_dict_roundtrip_keys(self):
        d = make_drawing(_stub_plate())
        data = d.to_dict()
        for key in ("svg", "views", "dimensions", "title_block", "note"):
            self.assertIn(key, data)
        self.assertEqual(data["views"], ["front", "top", "right", "iso"])

    @unittest.skipUnless(HAVE_HOLE, "Hole op not available")
    def test_hole_callouts_present(self):
        d = make_drawing(_stub_plate(n_holes=3))
        holes = d.dimensions["holes"]
        self.assertEqual(sum(h["count"] for h in holes), 3)
        self.assertIn("Ø5", d.svg)
        self.assertIn("3×", d.svg)

    def test_first_angle_projection_label(self):
        d = make_drawing(_stub_plate(), angle="first")
        self.assertIn("FIRST ANGLE", d.svg)
        self.assertEqual(d.title_block["projection"], "FIRST ANGLE")

    def test_drawing_number_is_deterministic(self):
        a = make_drawing(_stub_plate())
        b = make_drawing(_stub_plate())
        self.assertEqual(a.title_block["drawing_number"],
                         b.title_block["drawing_number"])
        self.assertTrue(a.title_block["drawing_number"].startswith("HC-"))

    def test_custom_views_subset(self):
        d = make_drawing(_stub_plate(), views=("front", "top"))
        self.assertEqual(d.views, ["front", "top"])
        self.assertIn("FRONT", d.svg)
        self.assertIn("TOP", d.svg)


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestRealDrawing(unittest.TestCase):
    def _cq_plate(self, w=60.0, h=40.0, thick=8.0):
        from harnesscad.io.backends.cadquery_backend import CadQueryBackend
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=w, h=h))
        b.apply(Extrude(sketch="sk1", distance=thick))
        return b

    def test_real_solid_embeds_multiple_view_groups(self):
        d = make_drawing(self._cq_plate())
        # nested <svg> per embedded view + <g> view groups
        self.assertGreater(d.svg.count("<g>"), 1)
        self.assertGreater(d.svg.count("<svg"), 2)
        self.assertIsNone(d.note)

    def test_dimensions_reflect_model_bbox(self):
        d = make_drawing(self._cq_plate(w=60.0, h=40.0, thick=8.0))
        self.assertEqual(d.dimensions["source"], "metrics")
        bbox = d.dimensions["bbox"]
        self.assertAlmostEqual(bbox[0], 60.0, delta=0.5)
        self.assertAlmostEqual(bbox[1], 40.0, delta=0.5)
        self.assertAlmostEqual(bbox[2], 8.0, delta=0.5)

    def test_deterministic_with_injected_date(self):
        a = make_drawing(self._cq_plate(), date="2026-07-05")
        b = make_drawing(self._cq_plate(), date="2026-07-05")
        self.assertEqual(a.svg, b.svg)


if __name__ == "__main__":
    unittest.main()
