"""The op -> control table is DATA, and it is honest about what it cannot bind."""

import unittest

from harnesscad.core.cisp.ops import (
    AddRectangle, Extrude, Fillet, NewSketch, _REGISTRY,
)
from harnesscad.io.cua import bindings_freecad as B


class TestIsomorphism(unittest.TestCase):
    def test_op_to_button_is_a_static_dict_not_a_vision_problem(self):
        """Every button name below was read off the live FreeCAD UIA tree."""
        self.assertEqual(B.OP_TO_BUTTON["extrude"], "Pad")
        self.assertEqual(B.OP_TO_BUTTON["shell"], "Thickness")
        self.assertEqual(B.OP_TO_BUTTON["circular_pattern"], "Polar Pattern")
        self.assertEqual(B.OP_TO_BUTTON["boolean"], "Boolean Operation")

    def test_every_op_is_either_bound_or_explicitly_refused(self):
        # Asserted as a SET, not per-op subTests: a subTest failure is reported
        # one at a time by some runners, which makes a 12-op hole look like a
        # 1-op hole. The whole gap is named at once or not at all.
        declared = set(B.OP_TO_BUTTON) | set(B.REQUIRES_VIEWPORT)
        self.assertEqual(
            sorted(set(_REGISTRY) - declared), [],
            "ops neither bound nor refused: implement-or-refuse means a CISP op "
            "the GUI cannot drive must say so in REQUIRES_VIEWPORT")

    def test_the_viewport_is_selected_by_id_path_never_by_classname(self):
        """There is a DECOY 100x30 QOpenGLWidget outside the window. A pipeline that
        cropped to it would silently feed the model a sliver of desktop."""
        self.assertEqual(B.VIEWPORT_AID_CONTAINS, "View3DInventorViewer")


class TestRecipes(unittest.TestCase):
    def test_a_rectangle_padded_at_the_origin_IS_the_box_primitive(self):
        ops = [NewSketch(plane="XY"),
               AddRectangle(sketch="sk1", x=0, y=0, w=30, h=20),
               Extrude(sketch="sk1", distance=10)]
        matches, reasons = B.match_recipes(ops)
        self.assertEqual(reasons, [])
        self.assertEqual(len(matches), 1)
        recipe, matched = matches[0]
        self.assertEqual(recipe.id, "box")
        self.assertEqual(B.check_guards(recipe, matched), [])
        self.assertEqual(B.bind_values(recipe, matched),
                         {"boxLength": 30.0, "boxWidth": 20.0, "boxHeight": 10.0})

    def test_two_boxes_segment_into_two_recipes(self):
        ops = [NewSketch(plane="XY"),
               AddRectangle(sketch="sk1", x=0, y=0, w=30, h=20),
               Extrude(sketch="sk1", distance=10),
               NewSketch(plane="XY"),
               AddRectangle(sketch="sk2", x=0, y=0, w=10, h=10),
               Extrude(sketch="sk2", distance=40)]
        matches, reasons = B.match_recipes(ops)
        self.assertEqual(reasons, [])
        self.assertEqual([r.id for r, _ in matches], ["box", "box"])

    def test_an_off_origin_rectangle_is_REFUSED_not_built_in_the_wrong_place(self):
        """The Box primitive's attachment-offset fields are disabled until a
        reference is PICKED IN THE VIEWPORT. So an off-origin box cannot be built
        coordinate-free, and the guard says so rather than quietly building it at
        the origin -- which would be a silently wrong part."""
        ops = [NewSketch(plane="XY"),
               AddRectangle(sketch="sk1", x=5, y=0, w=30, h=20),
               Extrude(sketch="sk1", distance=10)]
        (recipe, matched), = B.match_recipes(ops)[0]
        bad = B.check_guards(recipe, matched)
        self.assertTrue(bad)
        self.assertIn("PICKED IN THE VIEWPORT", bad[0])

    def test_an_op_needing_a_pick_is_reported_with_its_reason(self):
        matches, reasons = B.match_recipes([Fillet(edges=("|Z",), radius=2.0)])
        self.assertEqual(matches, [])
        self.assertEqual(len(reasons), 1)
        self.assertIn("EDGE selection in the viewport", reasons[0])


if __name__ == "__main__":
    unittest.main()
