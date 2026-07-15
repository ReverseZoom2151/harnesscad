"""The field's oracle, faithfully weak: it must PASS the parts the field passes.

The whole package rests on the claim that Text2CAD-Bench's IoU + Chamfer and MUSE's
geometric check score certain WRONG parts as correct. If our implementation of those
metrics were secretly stricter than the field's, the claim would be rigged. So these
tests pin that the weak grader PASSES the 8-vs-12 mm hole and the displaced hole --
the two failures the field is on record as not catching.
"""

from __future__ import annotations

import unittest

from harnesscad.core.cisp.ops import AddRectangle, Extrude, Hole, NewSketch
from harnesscad.eval.hardcorpus import weak


def _plate_hole(x, y, d):
    return [NewSketch("XY"), AddRectangle("sk1", 0, 0, 60, 40), Extrude("sk1", 12),
            Hole("sk1", x, y, d, None, True, "simple")]


class TestWeak(unittest.TestCase):

    def test_the_weak_grader_passes_a_correct_answer(self):
        ops = _plate_hole(20, 20, 12)
        s = weak.score_weak(ops, ops)
        self.assertTrue(s.passes)
        self.assertAlmostEqual(s.iou, 1.0, places=2)

    def test_the_weak_grader_passes_the_8_vs_12mm_hole(self):
        # The measured blind spot: an 8 mm hole where 12 mm was asked scores high
        # IoU and passes. If this ever fails, our IoU has been made stricter than
        # the field's and the discriminative claim is no longer honest.
        near = _plate_hole(20, 20, 8)
        correct = _plate_hole(20, 20, 12)
        s = weak.score_weak(near, correct)
        self.assertTrue(s.valid, "MUSE's geometric check must pass this")
        self.assertGreaterEqual(s.iou, weak.IOU_MATCH,
                                "IoU %s dropped below the pre-registered bar" % s.iou)
        self.assertTrue(s.passes, "the field's grader must be fooled here")

    def test_the_weak_grader_passes_the_displaced_hole(self):
        near = _plate_hole(40, 20, 10)
        correct = _plate_hole(20, 20, 10)
        s = weak.score_weak(near, correct)
        self.assertTrue(s.passes)

    def test_invalidity_is_constant_across_correct_and_wrong(self):
        # The pressure_correlation finding, in miniature: valid does not vary
        # between a right and a wrong (but well-formed) part.
        good = weak.score_weak(_plate_hole(20, 20, 12), _plate_hole(20, 20, 12))
        bad = weak.score_weak(_plate_hole(20, 20, 8), _plate_hole(20, 20, 12))
        self.assertEqual(good.valid, bad.valid)

    def test_thresholds_are_the_preregistered_ones(self):
        # Imported from eval/corpus/shape, never redefined here.
        from harnesscad.eval.corpus.shape import IOU_MATCH
        self.assertEqual(weak.IOU_MATCH, IOU_MATCH)


if __name__ == "__main__":
    unittest.main()
