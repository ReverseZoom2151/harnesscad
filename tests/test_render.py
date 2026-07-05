"""Tests for render.py — multi-view rendering with a graceful headless skip.

The module must import and behave (returning None per view + a note) even when
cadquery/OCP is absent or the backend has no solid. When cadquery IS installed a
real CadQuery solid renders to non-empty bytes.
"""

import os
import tempfile
import unittest

from render import (
    STANDARD_VIEWS, DEFAULT_VIEWS, ViewSpec, RenderResult,
    render, render_views, save_views, resolve_views,
)
from backends.stub import StubBackend
from cisp.ops import NewSketch, AddRectangle, Extrude


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


HAVE_CQ = _cadquery_available()


def _stub_with_solid() -> StubBackend:
    b = StubBackend()
    b.apply(NewSketch(plane="XY"))
    b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=20.0, h=10.0))
    b.apply(Extrude(sketch="sk1", distance=5.0))
    return b


class TestViewSpecs(unittest.TestCase):
    def test_default_views_are_known(self):
        for name in DEFAULT_VIEWS:
            self.assertIn(name, STANDARD_VIEWS)

    def test_iso_is_isometric_projection(self):
        self.assertEqual(STANDARD_VIEWS["iso"].projection, "iso")
        self.assertEqual(STANDARD_VIEWS["front"].projection, "ortho")

    def test_resolve_mixes_names_and_specs(self):
        custom = ViewSpec("custom", (1.0, 0.0, 0.0))
        specs = resolve_views(["iso", custom])
        self.assertEqual([s.name for s in specs], ["iso", "custom"])

    def test_unknown_view_raises(self):
        with self.assertRaises(KeyError):
            resolve_views(["nope"])


class TestHeadlessSkip(unittest.TestCase):
    """Stub backend has no _combined() OCCT shape -> graceful None + note."""

    def test_render_views_returns_none_per_view(self):
        images = render_views(_stub_with_solid())
        self.assertEqual(set(images), set(DEFAULT_VIEWS))
        self.assertTrue(all(v is None for v in images.values()))

    def test_render_result_carries_note_and_never_raises(self):
        result = render(_stub_with_solid())
        self.assertIsInstance(result, RenderResult)
        self.assertFalse(result.any_rendered)
        self.assertIsInstance(result.note, str)
        self.assertTrue(result.note)

    def test_save_views_writes_no_files_on_skip(self):
        with tempfile.TemporaryDirectory() as d:
            paths = save_views(_stub_with_solid(), d)
            self.assertTrue(all(p is None for p in paths.values()))
            self.assertEqual(os.listdir(d), [])


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestRealRender(unittest.TestCase):
    def _cq_plate(self):
        from backends.cadquery_backend import CadQueryBackend
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=20.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        return b

    def test_render_returns_bytes_per_view(self):
        result = render(self._cq_plate())
        self.assertTrue(result.any_rendered)
        self.assertIsNone(result.note)
        for name, data in result.images.items():
            self.assertIsInstance(data, (bytes, bytearray), name)
            self.assertGreater(len(data), 0, name)

    def test_deterministic_bytes(self):
        a = render_views(self._cq_plate(), views=("iso",))
        b = render_views(self._cq_plate(), views=("iso",))
        self.assertEqual(a["iso"], b["iso"])

    def test_save_views_writes_files(self):
        with tempfile.TemporaryDirectory() as d:
            paths = save_views(self._cq_plate(), d, views=("front", "top"))
            for name in ("front", "top"):
                self.assertTrue(os.path.exists(paths[name]))


if __name__ == "__main__":
    unittest.main()
