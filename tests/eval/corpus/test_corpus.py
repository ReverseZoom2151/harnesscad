"""The corpus's own invariants -- the ones the pressure corpus violated.

FAST BY DESIGN. Building a solid on the F-rep sampler marches a grid and costs
seconds; a suite that built all 32 briefs would take minutes and nobody would run
it. So the structural invariants (every brief has a bbox, every brief cites an
independent source, the splits do not overlap) are pure and cost nothing, and only
a small, named subset is actually BUILT. The full sweep is
``harnesscad.eval.corpus.run`` and it is a report a human asks for.
"""

from __future__ import annotations

import os
import unittest

from harnesscad.eval.corpus import analytic, dev, grade, score, shape, spec
from harnesscad.eval.corpus.spec import Source, Split

FULL = os.environ.get("HARNESSCAD_CORPUS_FULL") == "1"

#: Two briefs, one arithmetic and one from a standard, cheap enough to build in a
#: unit test. Named, not sampled: a randomly-chosen subset makes a flaky suite.
_FAST = ("dev_plate_60x40x10", "dev_washer_iso7089_m8")


class TestBriefInvariants(unittest.TestCase):
    """Structural. No geometry engine is touched, so these are instant."""

    def all_briefs(self):
        # The held-out split is reached ONLY through the scorer. Its SIZE is not a
        # leak; its contents are, so this test never asks for them.
        return list(dev.BRIEFS)

    def test_every_brief_carries_an_expected_bbox(self):
        """report.md:92. The pressure briefs carried ``bbox=None``, so a shell that
        dilated a 60x40x20 box into 63x43x23 scored a pass. A brief that does not
        state the envelope cannot catch an envelope bug."""
        for b in self.all_briefs():
            self.assertIsNotNone(b.bbox, b.id)
            self.assertEqual(len(b.bbox), 3, b.id)
            for v in b.bbox:
                self.assertGreater(v, 0.0, b.id)

    def test_a_brief_without_a_bbox_cannot_be_constructed(self):
        """The invariant is enforced by the type, not by this test's vigilance."""
        with self.assertRaises((ValueError, TypeError)):
            spec.Brief(id="x", split=Split.DEV, source=Source.ANALYTIC,
                       citation="arithmetic", text="t",
                       reference=(), volume=1.0, bbox=None)   # type: ignore[arg-type]

    def test_every_brief_derives_its_truth_from_outside_this_repo(self):
        for b in self.all_briefs():
            self.assertIn(b.source, Source.ALL, b.id)
            self.assertTrue(b.citation.strip(), b.id)

    def test_there_is_no_source_meaning_the_harness_said_so(self):
        """The enum has no member for a number read off one of our own runs, and
        that absence is load-bearing: a recorded fixture defends whatever bug was
        live the day it was recorded."""
        for s in Source.ALL:
            self.assertNotIn(s, ("measured", "recorded", "fixture", "golden_run"))

    def test_the_splits_do_not_overlap(self):
        n_dev = len(dev.BRIEFS)
        n_held = score.size()          # a COUNT is not a leak
        self.assertGreater(n_dev, 0)
        self.assertGreater(n_held, 0)
        dev_ids = set(dev.ids())
        self.assertEqual(len(dev_ids), n_dev, "duplicate ids in the dev split")

    def test_analytic_factories_refuse_a_degenerate_part(self):
        """A brief factory that will build an impossible part will eventually build
        one, and then the corpus is scoring the engine's despair."""
        with self.assertRaises(ValueError):        # 2t >= min extent: no cavity
            analytic.hollow_box("x", Split.DEV, 20.0, 20.0, 10.0, 5.0)
        with self.assertRaises(ValueError):        # 2r >= min extent: degenerate
            analytic.filleted_plate("x", Split.DEV, 50.0, 30.0, 6.0, 3.0)
        with self.assertRaises(ValueError):        # the hole does not fit
            analytic.plate_with_holes("x", Split.DEV, 20.0, 20.0, 5.0, 30.0,
                                      ((10.0, 10.0),))

    def test_the_fillet_ceiling_is_the_real_one_not_the_harness_rule(self):
        """r = 2.99 on a 6 mm plate is a REAL PART (2r < 6). The harness's own
        RADIUS_TOO_LARGE rule used to fire at r = 3.1 and stay silent at r = 3.0;
        the pressure briefs were written to agree with it. This corpus is written
        to agree with geometry."""
        b = analytic.filleted_plate("x", Split.DEV, 50.0, 30.0, 6.0, 2.99)
        self.assertGreater(b.volume, 0.0)


class TestGraderIsIndependentOfTheFleet(unittest.TestCase):

    def test_the_grader_never_runs_the_fleet(self):
        """A grader that consulted the verifiers would be scoring a model on its
        ability to please the thing under test -- and when a rule is wrong, on its
        ability to please a BUG. Checked against the parsed CODE, not the text:
        these modules discuss the fleet at length in their docstrings, and a check
        that tripped on prose would be switched off within the week."""
        import ast
        import inspect

        for module in (grade, shape):
            tree = ast.parse(inspect.getsource(module))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        self.assertNotIn("verifiers", a.name, module.__name__)
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn("verifiers", node.module or "",
                                     module.__name__)
                elif isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if kw.arg != "verify_level":
                            continue
                        self.assertNotEqual(
                            getattr(kw.value, "value", None), "full",
                            "%s builds at verify_level='full', which runs the "
                            "verifier fleet. The fleet is the system under test."
                            % module.__name__)


class TestReferenceSolutionsPassTheirOwnGrader(unittest.TestCase):
    """A corpus whose reference solution fails its own grader is measuring the
    engine's bugs and billing them to the model. The pressure corpus failed exactly
    this on two shell briefs and shipped, because nobody ever ran it."""

    def test_fast_subset(self):
        for bid in _FAST:
            b = dev.by_id(bid)
            s = grade.grade(b, list(b.reference))
            self.assertTrue(s.solved,
                            "%s: its own reference solution fails its own grader:\n"
                            "  %s" % (bid, "\n  ".join(s.reasons)))
            self.assertIsNotNone(s.iou)
            self.assertGreaterEqual(s.iou, 0.99)

    @unittest.skipUnless(FULL, "the full corpus sweep builds ~30 solids on a "
                               "grid-marching engine and takes minutes; set "
                               "HARNESSCAD_CORPUS_FULL=1 to run it")
    def test_every_dev_brief(self):
        for b in dev.BRIEFS:
            s = grade.grade(b, list(b.reference))
            if s.unmeasurable:
                continue          # a GOOD part this engine cannot resolve
            self.assertTrue(s.solved, "%s: %s" % (b.id, "; ".join(s.reasons)))

    @unittest.skipUnless(FULL, "the held-out sweep builds ~15 solids; set "
                               "HARNESSCAD_CORPUS_FULL=1 to run it")
    def test_every_heldout_brief(self):
        r = score.reference_score()
        self.assertEqual(r.failed, {}, "held-out briefs whose own reference "
                                       "solution fails: %s" % score.failures(r))


class TestShapeMetric(unittest.TestCase):
    """What the shape metric does, AND what it provably cannot do."""

    def test_iou_catches_a_gross_shape_error_the_envelope_cannot(self):
        """A boolean that cut the wrong corner: identical volume, identical bbox,
        and a completely different part. Every envelope family passes it."""
        from harnesscad.core.cisp.ops import (AddRectangle, Boolean, Extrude,
                                              NewSketch)
        good = (NewSketch("XY"), AddRectangle("sk1", 0, 0, 40, 40),
                Extrude("sk1", 20.0),
                NewSketch("XY"), AddRectangle("sk2", 0, 0, 20, 20),
                Extrude("sk2", 20.0), Boolean("cut", "f1", "f2"))
        wrong_corner = (NewSketch("XY"), AddRectangle("sk1", 0, 0, 40, 40),
                        Extrude("sk1", 20.0),
                        NewSketch("XY"), AddRectangle("sk2", 20, 20, 20, 20),
                        Extrude("sk2", 20.0), Boolean("cut", "f1", "f2"))
        s = shape.iou_of_ops(wrong_corner, good)
        self.assertTrue(s.ok, s.reason)
        self.assertLess(s.iou, shape.IOU_MATCH)
        self.assertFalse(s.matched)

    def test_iou_is_BLIND_to_a_small_misplaced_hole_and_we_say_so(self):
        """THE LIMITATION, PINNED BY A TEST SO IT CANNOT BE FORGOTTEN.

        An 8 mm hole moved 20 mm across a 60x40x12 plate is a WRONG PART, and its
        IoU is ~0.957 -- it passes the threshold. The symmetric difference is two
        holes' worth of material against a union of 29,400 mm3, and that is
        arithmetic, not a defect in the implementation.

        This test asserts the blindness ON PURPOSE. Raising IOU_MATCH above 0.957
        to make this one wrong part fail would fail CORRECT parts, because a
        correct rebuild on a sampled engine already differs from itself by a
        surface band. The probe points are what catch this (see the next test),
        which is exactly why both families are reported and neither is sufficient.
        """
        from harnesscad.core.cisp.ops import (AddRectangle, Extrude, Hole,
                                              NewSketch)
        good = (NewSketch("XY"), AddRectangle("sk1", 0, 0, 60, 40),
                Extrude("sk1", 12.0),
                Hole("sk1", 30.0, 20.0, 8.0, None, True, "simple"))
        moved = (NewSketch("XY"), AddRectangle("sk1", 0, 0, 60, 40),
                 Extrude("sk1", 12.0),
                 Hole("sk1", 12.0, 8.0, 8.0, None, True, "simple"))
        s = shape.iou_of_ops(moved, good)
        self.assertTrue(s.ok, s.reason)
        # It SEES the difference...
        self.assertLess(s.iou, 0.99)
        # ...and it does not CATCH it. Stated, not hidden.
        self.assertGreater(
            s.iou, shape.IOU_MATCH,
            "the blindness this test documents has changed (IoU %.3f). If the "
            "metric now catches a small misplaced hole, good -- but check that "
            "IOU_MATCH was not simply raised to a value that also fails correct "
            "parts, which is how a benchmark gets tuned into agreeing with you."
            % s.iou)

    def test_the_probes_catch_exactly_what_iou_misses(self):
        """The complement. A probe on the hole's AXIS is a point assertion, and a
        point assertion is what an integral over the whole part cannot make."""
        from harnesscad.core.cisp.ops import (AddRectangle, Extrude, Hole,
                                              NewSketch)
        b = analytic.plate_with_holes("t", Split.DEV, 60.0, 40.0, 12.0, 8.0,
                                      ((30.0, 20.0),))
        moved = (NewSketch("XY"), AddRectangle("sk1", 0, 0, 60, 40),
                 Extrude("sk1", 12.0),
                 Hole("sk1", 12.0, 8.0, 8.0, None, True, "simple"))
        s = grade.grade(b, moved)
        self.assertFalse(s.solved,
                         "a hole 20 mm from where the brief put it was scored as "
                         "SOLVED; the probes are not doing their job")
        self.assertFalse(s.probes_ok)
        self.assertTrue(s.volume_ok, "the volume is unchanged -- that is the point")
        self.assertTrue(s.bbox_ok, "the bbox is unchanged -- that is the point")

    def test_the_same_stream_scores_one(self):
        from harnesscad.core.cisp.ops import AddRectangle, Extrude, NewSketch
        ops = (NewSketch("XY"), AddRectangle("sk1", 0, 0, 60, 40),
               Extrude("sk1", 10.0))
        s = shape.iou_of_ops(ops, ops)
        self.assertTrue(s.ok, s.reason)
        self.assertGreaterEqual(s.iou, 0.999)

    def test_the_metric_is_deterministic(self):
        from harnesscad.core.cisp.ops import AddRectangle, Extrude, NewSketch
        a = (NewSketch("XY"), AddRectangle("sk1", 0, 0, 50, 30), Extrude("sk1", 8.0))
        b = (NewSketch("XY"), AddRectangle("sk1", 0, 0, 50, 30), Extrude("sk1", 9.0))
        first = shape.iou_of_ops(b, a).iou
        second = shape.iou_of_ops(b, a).iou
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
