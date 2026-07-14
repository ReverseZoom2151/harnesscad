"""The projection maths, tested WITHOUT an application.

The whole point of an analytic projection is that it is checkable on paper. The
camera below is a verbatim capture from a live FreeCAD 1.1.1 isometric view, so
these tests pin the exact conventions that a GUI cannot be trusted to tell you
twice: axis-angle vs quaternion, ADJUST_CAMERA's aspect rule, and y-up vs y-down.
Every one of those has a wrong answer that still produces a plausible-looking
corpus.

The tests that need a live FreeCAD SKIP when it is absent. They never hang and
they never fail on absence.
"""

from __future__ import annotations

import math
import unittest

from harnesscad.io.cua import viewport as vp

#: Verbatim from ``View3DInventorPy.getCamera()`` on FreeCAD 1.1.1, isometric,
#: zoom-fitted on a 60x40x20 box with a 12mm bore, in a 1534x770 viewport.
CAMERA_TEXT = """#Inventor V2.1 ascii


OrthographicCamera {
  viewportMapping ADJUST_CAMERA
  position 174.08299 -124.09424 154.52979
  orientation 0.74290609 0.30772209 0.59447283  1.2171158
  nearDistance 214.96733
  farDistance 284.74902
  aspectRatio 1
  focalDistance 250
  height 74.833145

}
"""
VIEWPORT = (1534, 770)


class TestQuaternion(unittest.TestCase):
    def test_axis_angle_is_not_a_quaternion(self):
        """The .iv text is AXIS+ANGLE; pivy hands back a QUATERNION.

        The two look alike -- four floats, all order-1 -- and reading one as the
        other rotates the camera by a plausible amount rather than an obviously
        broken one. The quaternion pivy reported for exactly this camera was
        (0.42470816, 0.17592005, 0.33985111, 0.82047331).
        """
        got = vp.axis_angle_to_quat((0.74290609, 0.30772209, 0.59447283), 1.2171158)
        for a, b in zip(got, (0.42470816, 0.17592005, 0.33985111, 0.82047331)):
            self.assertAlmostEqual(a, b, places=6)

    def test_unit_norm(self):
        q = vp.axis_angle_to_quat((1.0, 2.0, 3.0), 0.7)
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in q)), 1.0, places=12)

    def test_degenerate_axis_is_identity(self):
        self.assertEqual(vp.axis_angle_to_quat((0.0, 0.0, 0.0), 1.0),
                         (0.0, 0.0, 0.0, 1.0))

    def test_inverse_round_trips(self):
        q = vp.axis_angle_to_quat((0.3, -0.5, 0.81), 2.1)
        v = (3.0, -7.0, 11.0)
        back = vp.quat_rotate(vp.quat_inverse(q), vp.quat_rotate(q, v))
        for a, b in zip(back, v):
            self.assertAlmostEqual(a, b, places=9)


class TestOrthoCamera(unittest.TestCase):
    def setUp(self):
        self.cam = vp.parse_camera(CAMERA_TEXT, *VIEWPORT)

    def test_parsed(self):
        self.assertAlmostEqual(self.cam.height, 74.833145, places=5)
        self.assertAlmostEqual(self.cam.position[0], 174.08299, places=4)
        self.assertEqual((self.cam.width_px, self.cam.height_px), VIEWPORT)

    def test_aspect_comes_from_the_VIEWPORT_not_the_camera_field(self):
        """Coin's ADJUST_CAMERA ignores ``aspectRatio``. The file says 1. It is 2.

        A projection that trusted the camera's own field would be wrong by ~2x in
        x -- and would still land inside the part, which is why this is a test and
        not a comment.
        """
        hw, hh = self.cam.half_extents()
        self.assertAlmostEqual(hh, 74.833145 / 2.0, places=5)
        self.assertAlmostEqual(hw / hh, 1534.0 / 770.0, places=9)
        self.assertNotAlmostEqual(hw / hh, 1.0, places=2)

    def test_portrait_viewport_is_rescaled(self):
        tall = vp.OrthoCamera(position=self.cam.position,
                              orientation=self.cam.orientation,
                              height=self.cam.height, width_px=400, height_px=800)
        hw, hh = tall.half_extents()
        vpa = 0.5
        self.assertAlmostEqual(hh, (self.cam.height * 0.5) / vpa, places=6)
        self.assertAlmostEqual(hw / hh, vpa, places=9)

    def test_the_camera_position_projects_to_the_centre(self):
        """The point the camera looks at lands in the middle of the viewport."""
        # A point straight ahead of the camera: position + forward*d. Coin's
        # camera looks down its own -Z.
        forward = vp.quat_rotate(self.cam.orientation, (0.0, 0.0, -1.0))
        ahead = tuple(self.cam.position[i] + forward[i] * 250.0 for i in range(3))
        px, py, depth = self.cam.project(ahead)
        self.assertAlmostEqual(px, VIEWPORT[0] / 2.0, places=6)
        self.assertAlmostEqual(py, VIEWPORT[1] / 2.0, places=6)
        self.assertLess(depth, 0.0)          # in front of the camera

    def test_depth_orders_front_to_back(self):
        forward = vp.quat_rotate(self.cam.orientation, (0.0, 0.0, -1.0))
        near = tuple(self.cam.position[i] + forward[i] * 220.0 for i in range(3))
        far = tuple(self.cam.position[i] + forward[i] * 280.0 for i in range(3))
        self.assertGreater(self.cam.project(near)[2], self.cam.project(far)[2])

    def test_image_and_picker_y_are_flipped(self):
        """y-up is the picker's frame; y-down is the screenshot's. Both, explicitly."""
        _ix, iy = self.cam.to_image_xy(100.0, 0.0)
        self.assertAlmostEqual(iy, VIEWPORT[1] - 1)
        _ix, iy = self.cam.to_image_xy(100.0, float(VIEWPORT[1] - 1))
        self.assertAlmostEqual(iy, 0.0)

    def test_screen_xy_offsets_by_the_viewport_rect(self):
        rect = (646, 365, 1534, 770)
        sx, sy = self.cam.to_screen_xy(10.0, float(VIEWPORT[1] - 1), rect)
        self.assertEqual((sx, sy), (656, 365))

    def test_in_view(self):
        self.assertTrue(self.cam.in_view(700.0, 400.0))
        self.assertFalse(self.cam.in_view(-1.0, 400.0))
        self.assertFalse(self.cam.in_view(700.0, 900.0))

    def test_round_trip_dict(self):
        again = vp.OrthoCamera.from_dict(self.cam.to_dict())
        self.assertEqual(again, self.cam)

    def test_perspective_is_refused_not_approximated(self):
        with self.assertRaises(vp.ViewportError):
            vp.parse_camera("PerspectiveCamera { heightAngle 0.78 }", 100, 100)

    def test_garbage_is_refused(self):
        with self.assertRaises(vp.ViewportError):
            vp.parse_camera("not a camera", 100, 100)


class TestNamedViews(unittest.TestCase):
    def test_seven_named_views_and_no_orbit(self):
        self.assertEqual(len(vp.NAMED_VIEWS), 7)
        self.assertEqual(set(vp.VIEW_ORDER), set(vp.NAMED_VIEWS))
        # There is deliberately no orbit anywhere on the controller: an orbit
        # destroys the coordinate frame, and a projection you cannot write down
        # is a label you cannot trust.
        self.assertFalse(any("orbit" in name.lower()
                             for name in dir(vp.ViewportController)))

    def test_every_view_has_a_method_and_a_key(self):
        for name, (method, key) in vp.NAMED_VIEWS.items():
            self.assertTrue(method.startswith("view"), name)
            self.assertTrue(key.isdigit(), name)


class TestDescriptions(unittest.TestCase):
    TOPO = {"bbox": [0.0, 0.0, 0.0, 60.0, 40.0, 20.0]}

    def test_axis_faces_get_human_names(self):
        rec = {"kind": "face", "surface": "Plane", "normal": [0, 0, 1],
               "centroid": [30.0, 20.0, 20.0]}
        self.assertEqual(vp.describe_entity(rec, self.TOPO),
                         "the top face of the part")

    def test_cylindrical_face_is_located(self):
        rec = {"kind": "face", "surface": "Cylinder", "normal": None,
               "centroid": [15.0, 20.0, 10.0]}
        got = vp.describe_entity(rec, self.TOPO)
        self.assertIn("cylindrical face", got)
        self.assertIn("left", got)

    def test_edge_and_vertex(self):
        edge = {"kind": "edge", "surface": "Line", "centroid": [0.0, 0.0, 10.0]}
        self.assertIn("straight edge", vp.describe_entity(edge, self.TOPO))
        vert = {"kind": "vertex", "surface": "", "centroid": [60.0, 40.0, 20.0]}
        self.assertIn("vertex", vp.describe_entity(vert, self.TOPO))

    def test_descriptions_are_deterministic(self):
        rec = {"kind": "face", "surface": "Plane", "normal": [0, 0, -1],
               "centroid": [30.0, 20.0, 0.0]}
        self.assertEqual(vp.describe_entity(rec, self.TOPO),
                         vp.describe_entity(rec, self.TOPO))


@unittest.skipUnless(vp.gui_available(), "the FreeCAD GUI is not installed")
class TestLiveViewport(unittest.TestCase):
    """The claim that matters: a computed pixel selects the entity we meant.

    Skipped, never failed, when FreeCAD is absent.
    """

    BUILD = """
import Part
doc = App.newDocument("t")
o = doc.addObject("Part::Feature", "Model")
o.Shape = Part.makeBox(60, 40, 20).cut(
    Part.makeCylinder(6, 40, App.Vector(30, 20, -10)))
doc.recompute()
Gui.activeDocument().activeView().setCameraType("Orthographic")
Gui.SendMsgToActiveView("ViewFit")
Gui.updateGui()
RESULT = len(o.Shape.Faces)
"""

    @classmethod
    def setUpClass(cls):
        cls.bridge = vp.FreeCADGuiBridge(timeout=180.0).start()
        cls.ctl = vp.ViewportController(cls.bridge, "Model")
        cls.bridge.call(cls.BUILD, timeout=120)

    @classmethod
    def tearDownClass(cls):
        cls.bridge.close()          # never leave a GUI running

    def test_projection_agrees_with_the_apps_own_world_to_screen(self):
        """Our maths vs FreeCAD's ``getPointOnScreen``. Sub-pixel, or it is wrong."""
        self.ctl.set_named_view("isometric")
        cam = self.ctl.camera()
        points = [p for e in self.ctl.entities(("face",)) for p in e.candidates]
        theirs = self.bridge.call(
            "import json\n"
            "v = Gui.activeDocument().activeView()\n"
            "out = []\n"
            "for p in json.loads(%r):\n"
            "    s = v.getPointOnScreen(App.Vector(p[0], p[1], p[2]))\n"
            "    out.append([float(s[0]), float(s[1])])\n"
            "RESULT = out" % __import__("json").dumps([list(p) for p in points]))
        worst = 0.0
        for point, (tx, ty) in zip(points, theirs):
            px, py, _d = cam.project(point)
            worst = max(worst, math.hypot(px - tx, py - ty))
        # A pixel-centre convention differs by half a pixel; anything beyond ~2 px
        # is a real disagreement about the projection.
        self.assertLess(worst, 2.0, "projection differs from the app by %.3f px" % worst)

    def test_a_computed_pixel_selects_the_intended_face(self):
        self.ctl.set_named_view("isometric")
        cam = self.ctl.camera()
        picks = self.ctl.adjudicate(self.ctl.entities(("face",)), cam)
        verified = [p for p in picks if p.verified]
        self.assertTrue(verified, "not one face was verifiable from an isometric view")
        for pick in verified:
            self.assertEqual(pick.selected, pick.entity)
            self.assertTrue(0 <= pick.x < cam.width_px)
            self.assertTrue(0 <= pick.y < cam.height_px)

    def test_discards_carry_their_reason(self):
        """A discard is evidence, not a hole. It says WHY, and the why is the finding."""
        self.ctl.set_named_view("top")
        picks = self.ctl.adjudicate(self.ctl.entities(("face",)))
        discarded = [p for p in picks if not p.verified]
        self.assertTrue(discarded, "a top view must occlude the bottom face")
        for pick in discarded:
            self.assertTrue(pick.reason)
            self.assertNotEqual(pick.selected, pick.entity)

    def test_named_views_are_orthographic_and_repeatable(self):
        """Leave a named view and come back: the same camera, to well under a pixel.

        Not bit-identical, and it is worth being precise about why rather than
        loosening the assertion until it passes. ``ViewFit`` recentres by a float
        computation whose last couple of digits move: the measured drift is ~3e-4
        mm on a ~75 mm view volume, i.e. ~4e-6 of the viewport, or ~0.006 px. The
        ORIENTATION and the view HEIGHT -- the two things the projection's shape
        depends on -- are exactly equal. So the pixel a label lands on is
        unchanged, which is the claim that actually matters, and it is the claim
        the corpus independently confirms: two full sweeps produced 938 verified
        pairs each, with identical pixels.
        """
        self.ctl.set_named_view("front")
        first = self.ctl.camera()
        self.ctl.set_named_view("top")
        self.ctl.set_named_view("front")
        second = self.ctl.camera()

        self.assertEqual(first.orientation, second.orientation)
        self.assertEqual(first.height, second.height)
        self.assertEqual((first.width_px, first.height_px),
                         (second.width_px, second.height_px))
        # And the drift is sub-pixel where it counts: on the projected point.
        probe = (30.0, 20.0, 10.0)
        ax, ay, _ = first.project(probe)
        bx, by, _ = second.project(probe)
        self.assertLess(math.hypot(ax - bx, ay - by), 0.5)

    def test_unknown_view_is_refused(self):
        with self.assertRaises(vp.ViewportError):
            self.ctl.set_named_view("dramatic-angle")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
