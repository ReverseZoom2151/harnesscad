"""The instrument, verified against closed forms it cannot fake.

If ``section_at`` did not return the true second moment of area, every structural
constraint would be meaningless; if ``classify`` were not exact point membership,
every probe would be. So both are pinned against arithmetic: a rectangular bar's
I = b*h^3/12, and a hole's exact bore radius.
"""

from __future__ import annotations

import math
import unittest

from harnesscad.core.cisp.ops import (AddCircle, AddRectangle, Extrude, Hole,
                                      NewSketch)
from harnesscad.eval.hardcorpus import occt


def _plate(a, b, c, hole=None):
    ops = [NewSketch("XY"), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c)]
    if hole is not None:
        x, y, d = hole
        ops.append(Hole("sk1", x, y, d, None, True, "simple"))
    return occt.build(ops)


class TestOcct(unittest.TestCase):

    def test_volume_is_exact(self):
        b = _plate(60, 40, 12)
        self.assertAlmostEqual(occt.volume_of(b.shape), 60 * 40 * 12, places=3)

    def test_point_membership_is_exact(self):
        b = _plate(60, 40, 12, hole=(20, 20, 10))
        self.assertEqual(occt.classify(b.shape, (5, 5, 6)), "in")     # solid corner
        self.assertEqual(occt.classify(b.shape, (20, 20, 6)), "out")  # hole axis
        self.assertEqual(occt.classify(b.shape, (30, 20, 30)), "out")  # above part

    def test_bore_radius_recovers_the_hole(self):
        b = _plate(60, 40, 12, hole=(20, 20, 12))
        r = occt.bore_radius_at(b.shape, (20, 20), 6)
        self.assertAlmostEqual(r, 6.0, places=1)      # 12 mm dia -> r = 6

    def test_section_second_moment_matches_bh3_over_12(self):
        # a 20 (Y) x 6 (Z) bar: I about the Y-centroidal axis = b*h^3/12 with
        # b along Y (20) and h along Z (6) -> 20*6^3/12 = 360.
        b = _plate(50, 20, 6)
        sec = occt.section_at(b.shape, (10, 0, 0), (1, 0, 0), (0, 1, 0))
        self.assertTrue(sec.ok)
        self.assertAlmostEqual(sec.area, 20 * 6, places=3)
        self.assertAlmostEqual(sec.inertia, 20 * 6 ** 3 / 12.0, places=2)
        self.assertAlmostEqual(sec.c, 3.0, places=3)
        self.assertAlmostEqual(sec.section_modulus, 360.0 / 3.0, places=2)

    def test_boolean_iou_of_a_part_with_itself_is_one(self):
        b = _plate(60, 40, 12)
        self.assertAlmostEqual(occt.boolean_iou(b.shape, b.shape), 1.0, places=3)

    def test_draft_does_not_build(self):
        # Recorded reason the op is dropped from the corpus.
        from harnesscad.core.cisp.ops import Draft
        built = occt.build([NewSketch("XY"), AddRectangle("sk1", 0, 0, 40, 40),
                            Extrude("sk1", 20), Draft(("|Z",), 5.0, "<Z")])
        self.assertFalse(built, "draft now builds -- add a draft factory and drop "
                                "the DROPPED_OPS note")


if __name__ == "__main__":
    unittest.main()
