"""The orthographic engineering-drawing export route."""

from __future__ import annotations

import os
import tempfile
import unittest

from harnesscad.core.cisp.ops import parse_op
from harnesscad.core.loop import HarnessSession
from harnesscad.io import drawing
from harnesscad.io.backends.frep import FRepBackend
from harnesscad.io.formats import registry as fmt

# A plate with a through hole: it has hidden edges in the top view.
OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0},
    {"op": "extrude", "sketch": "sk1", "distance": 6.0},
    {"op": "hole", "face_or_sketch": "sk1", "x": 10.0, "y": 5.0,
     "diameter": 5.0, "through": True},
]

# A unit cube, given directly as a mesh: its geometry is known exactly.
CUBE_V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
          (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]
CUBE_F = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
          (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
          (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]


def _model(resolution=16):
    backend = FRepBackend(resolution=resolution)
    session = HarnessSession(backend)
    result = session.apply_ops([parse_op(o) for o in OPS])
    assert result.ok, result.diagnostics
    return backend


class FeatureEdgeTest(unittest.TestCase):
    def test_a_cube_has_exactly_its_twelve_feature_edges(self):
        edges = drawing.feature_edges((CUBE_V, CUBE_F), angle=25.0)
        # 12 creases (90 degrees) -- the 6 in-face diagonals are coplanar and drop
        self.assertEqual(len(edges), 12)

    def test_a_flat_tessellation_has_no_creases(self):
        verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        faces = [(0, 1, 2), (0, 2, 3)]
        # every edge here is a BOUNDARY edge (one adjacent face) except the shared
        # diagonal, which is coplanar
        edges = drawing.feature_edges((verts, faces))
        self.assertEqual(len(edges), 4)


class VisibilityTest(unittest.TestCase):
    def test_the_far_face_of_a_cube_is_hidden_and_the_near_face_is_not(self):
        oracle = drawing._Visibility((CUBE_V, CUBE_F))
        toward_viewer = (0.0, -1.0, 0.0)              # the front view
        near = (0.5, 0.0, 0.5)                        # y = 0 face: visible
        far = (0.5, 1.0, 0.5)                         # y = 1 face: behind the solid
        self.assertFalse(oracle.hidden(near, toward_viewer))
        self.assertTrue(oracle.hidden(far, toward_viewer))

    def test_the_bvh_prunes_the_visibility_query(self):
        backend = _model()
        oracle = drawing._Visibility(backend.mesh())
        oracle.hidden((10.0, 5.0, 3.0), (0.0, -1.0, 0.0))
        self.assertGreater(oracle.brute, 0)
        self.assertLess(oracle.tested, oracle.brute,
                        "the BVH must hand back fewer triangles than brute force")


class OrthographicDrawingTest(unittest.TestCase):
    def test_the_sheet_carries_three_views_hidden_lines_and_dimensions(self):
        svg = drawing.orthographic_drawing(_model().mesh())
        for name in ("FRONT", "TOP", "SIDE"):
            self.assertIn(name, svg)
        self.assertIn("#888", svg)          # the dashed hidden-line layer
        self.assertIn("#2255aa", svg)       # the dimension layer
        self.assertIn("views sufficient: true", svg)
        metrics = drawing.drawing_metrics(svg)
        self.assertGreater(metrics["total_path_count"], 0)
        self.assertGreater(metrics["text_count"], 0)

    def test_hidden_lines_can_be_switched_off(self):
        mesh = _model().mesh()
        with_hidden = drawing.orthographic_drawing(mesh)
        without = drawing.orthographic_drawing(mesh, show_hidden=False)
        self.assertIn("#888", with_hidden)
        self.assertNotIn("#888", without)

    def test_first_and_third_angle_place_the_views_differently(self):
        mesh = _model().mesh()
        third = drawing.orthographic_drawing(mesh, angle_convention="third_angle")
        first = drawing.orthographic_drawing(mesh, angle_convention="first_angle")
        self.assertNotEqual(third, first)

    def test_the_drawing_is_deterministic(self):
        mesh = _model().mesh()
        self.assertEqual(drawing.orthographic_drawing(mesh),
                         drawing.orthographic_drawing(mesh))

    def test_an_insufficient_view_set_is_reported_not_hidden(self):
        svg = drawing.orthographic_drawing(_model().mesh(), views=("front",))
        self.assertIn("views sufficient: false", svg)

    def test_an_unknown_view_is_refused(self):
        with self.assertRaises(drawing.DrawingError):
            drawing.orthographic_drawing(_model().mesh(), views=("isometric",))

    def test_an_empty_mesh_is_refused(self):
        with self.assertRaises(drawing.DrawingError):
            drawing.orthographic_drawing(([], []))


class SvgExportRouteTest(unittest.TestCase):
    def test_the_default_svg_route_is_unchanged(self):
        backend = _model()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "wire.svg")
            fmt.write(backend, path)
            text = open(path, encoding="utf-8").read()
        self.assertIn("<svg", text)
        self.assertNotIn("FRONT", text)     # the wireframe route has no views

    def test_views_selects_the_orthographic_drawing_route(self):
        backend = _model()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ortho.svg")
            fmt.write(backend, path, views=True)
            text = open(path, encoding="utf-8").read()
        self.assertIn("FRONT", text)
        self.assertIn("TOP", text)
        self.assertIn("SIDE", text)

    def test_a_view_subset_can_be_named(self):
        backend = _model()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "two.svg")
            fmt.write(backend, path, views=["front", "top"])
            text = open(path, encoding="utf-8").read()
        self.assertIn("FRONT", text)
        self.assertNotIn("SIDE", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
