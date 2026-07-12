import math
import unittest

from formats.cq_svg_projector import (
    bounding_box_2d,
    camera_basis,
    fit_transform,
    get_svg,
    path_data,
    project_point,
    project_polyline,
)


def unit_cube_edges(size=1.0):
    s = size
    v = [
        (0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0),
        (0, 0, s), (s, 0, s), (s, s, s), (0, s, s),
    ]
    idx = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),   # top
        (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
    ]
    return [[v[a], v[b]] for a, b in idx]


class TestProjection(unittest.TestCase):
    def test_camera_basis_orthonormal(self):
        right, up, out = camera_basis((-1.75, 1.1, 5))
        for a in (right, up, out):
            self.assertAlmostEqual(sum(c * c for c in a), 1.0)
        # mutually perpendicular
        self.assertAlmostEqual(sum(r * u for r, u in zip(right, up)), 0.0)
        self.assertAlmostEqual(sum(r * o for r, o in zip(right, out)), 0.0)

    def test_project_along_z_is_xy(self):
        # viewing along +Z, u,v == x,y
        self.assertEqual(project_point((3, 4, 9), (0, 0, 1)), (3.0, 4.0))

    def test_project_polyline(self):
        pl = project_polyline([(0, 0, 5), (2, 3, 5)], (0, 0, 1))
        self.assertEqual(pl, [(0.0, 0.0), (2.0, 3.0)])

    def test_depth_is_dropped(self):
        # points differing only in view depth collapse
        a = project_point((1, 2, 0), (0, 0, 1))
        b = project_point((1, 2, 100), (0, 0, 1))
        self.assertEqual(a, b)


class TestPathData(unittest.TestCase):
    def test_path_string(self):
        d = path_data([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
        self.assertTrue(d.startswith("M0.0,0.0 "))
        self.assertIn("L1.0,0.0 ", d)
        self.assertIn("L1.0,1.0 ", d)

    def test_empty(self):
        self.assertEqual(path_data([]), "")


class TestBBoxAndFit(unittest.TestCase):
    def test_bbox(self):
        edges = [project_polyline(e, (0, 0, 1)) for e in unit_cube_edges(2.0)]
        self.assertEqual(bounding_box_2d(edges), (0.0, 0.0, 2.0, 2.0))

    def test_fit_fixed_canvas(self):
        # 2x2 bbox into 800x240, bb_scale 0.75
        w, h, scale, tx, ty = fit_transform((0, 0, 2, 2), 800, 240)
        self.assertEqual((w, h), (800.0, 240.0))
        # unitScale = min(800/2*0.75, 240/2*0.75) = min(300, 90) = 90
        self.assertAlmostEqual(scale, 90.0)

    def test_fit_derived_height(self):
        # height=None -> derived from aspect
        w, h, scale, tx, ty = fit_transform((0, 0, 4, 2), 800, None)
        self.assertEqual(w, 800.0)
        # unitScale = (800 - 400) / 4 = 100
        self.assertAlmostEqual(scale, 100.0)

    def test_degenerate_raises(self):
        with self.assertRaises(ValueError):
            fit_transform((0, 0, 0, 5), 100, 100)


class TestGetSVG(unittest.TestCase):
    def test_document_structure(self):
        svg = get_svg(unit_cube_edges(2.0), opts={"projectionDir": (0, 0, 1)})
        self.assertIn("<svg", svg)
        self.assertIn("</svg>", svg)
        self.assertIn("<path", svg)
        # 12 visible edges -> 12 paths
        self.assertEqual(svg.count("<path"), 12)

    def test_hidden_styling_and_toggle(self):
        vis = unit_cube_edges(1.0)
        hid = [[(0, 0, 0), (0, 0, 1)]]
        svg = get_svg(vis, hid, opts={"projectionDir": (0, 1, 0)})
        self.assertIn("stroke-dasharray", svg)
        # showHidden False drops the hidden path
        svg2 = get_svg(vis, hid, opts={"projectionDir": (0, 1, 0), "showHidden": False})
        self.assertEqual(svg2.count("<path"), len(vis))

    def test_auto_stroke_width(self):
        svg = get_svg(unit_cube_edges(2.0), opts={"projectionDir": (0, 0, 1)})
        # stroke-width should be present and finite (1/unitScale)
        self.assertIn("stroke-width=", svg)

    def test_deterministic(self):
        a = get_svg(unit_cube_edges(2.0), opts={"projectionDir": (0, 0, 1)})
        b = get_svg(unit_cube_edges(2.0), opts={"projectionDir": (0, 0, 1)})
        self.assertEqual(a, b)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            get_svg([])


if __name__ == "__main__":
    unittest.main()
