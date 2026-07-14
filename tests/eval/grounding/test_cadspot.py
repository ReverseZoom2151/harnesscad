"""CADSpot: the metric, the regions, and the harness check that caught itself."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from harnesscad.eval.grounding import cadspot, corpus
from harnesscad.io.cua import viewport as vp


def _target(region="viewport", instruction="the top face", bbox=(90, 90, 110, 110),
            entity="Face3", image="a.png"):
    return cadspot.Target(region=region, instruction=instruction, image=image,
                          width=200, height=200, bbox=bbox, entity=entity,
                          sample="s0000", view="isometric")


class TestTarget(unittest.TestCase):
    def test_point_in_bbox_is_the_whole_screenspot_metric(self):
        t = _target()
        self.assertTrue(t.contains(100, 100))
        self.assertTrue(t.contains(90, 110))      # inclusive on the border
        self.assertFalse(t.contains(89, 100))
        self.assertFalse(t.contains(100, 111))

    def test_centre(self):
        self.assertEqual(_target().center, (100, 100))

    def test_round_trip(self):
        t = _target()
        self.assertEqual(cadspot.Target.from_dict(t.to_dict()), t)


class TestRegions(unittest.TestCase):
    def test_four_regions_and_the_fourth_is_the_point(self):
        self.assertEqual(cadspot.REGIONS,
                         ("toolbar", "dialog", "tree", "viewport"))

    def test_classification_prefers_the_qt_object_path(self):
        class E:
            def __init__(self, aid, ctype):
                self.automation_id, self.control_type = aid, ctype

        self.assertEqual(cadspot._classify(
            E("...Gui__Dialog__Placement.GroupBox5.xPos", "SpinnerControl")), "dialog")
        self.assertEqual(cadspot._classify(
            E("...MainWindow.Part Design.QToolButton", "ButtonControl")), "toolbar")
        self.assertEqual(cadspot._classify(
            E("...Gui::TreeWidget", "TreeItemControl")), "tree")
        # A button with no useful id still lands somewhere sensible.
        self.assertEqual(cadspot._classify(E("", "ButtonControl")), "toolbar")
        self.assertEqual(cadspot._classify(E("", "TextControl")), "")

    def test_static_labels_and_containers_are_not_targets(self):
        """A label is the caption OF a target, not a target."""
        self.assertNotIn("TextControl", cadspot._ACTIONABLE)
        self.assertNotIn("GroupControl", cadspot._ACTIONABLE)
        self.assertIn("SpinnerControl", cadspot._ACTIONABLE)
        self.assertIn("ButtonControl", cadspot._ACTIONABLE)


class TestViewportTargets(unittest.TestCase):
    def test_only_verified_pairs_become_targets(self):
        pairs = [
            corpus.GroundingPair(sample="s", view="top", screenshot="a.png",
                                 entity="Face1", kind="face", description="the top face",
                                 x=100, y=50, point=(0, 0, 0), verified=True,
                                 width=800, height=600),
            corpus.GroundingPair(sample="s", view="top", screenshot="a.png",
                                 entity="Face2", kind="face", description="the bottom face",
                                 x=-1, y=-1, point=(0, 0, 0), verified=False,
                                 reason="occluded by Face1"),
        ]
        targets = cadspot.viewport_targets(pairs, radius=10)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].entity, "Face1")
        self.assertEqual(targets[0].bbox, (90, 40, 110, 60))
        self.assertEqual((targets[0].width, targets[0].height), (800, 600))


class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.targets = [_target(), _target(entity="Face4", bbox=(10, 10, 30, 30))]

    def test_the_oracle_is_a_harness_check_and_must_be_perfect(self):
        """If the oracle is not 1.0, the plumbing is broken -- not the projection.

        This test exists because the first version of the oracle scored 0.0: it
        keyed on ``(image, instruction)``, and a part has several entities with the
        same description. It looked exactly like a catastrophic projection bug.
        """
        report = cadspot.evaluate(self.targets, cadspot.oracle_predictor(),
                                  name="oracle")
        self.assertEqual(report.regions["viewport"].accuracy, 1.0)
        self.assertEqual(report.overall.error_rate, 0.0)

    def test_the_oracle_survives_duplicate_instructions(self):
        dupes = [_target(entity="Edge1", bbox=(0, 0, 10, 10)),
                 _target(entity="Edge2", bbox=(180, 180, 200, 200))]
        report = cadspot.evaluate(dupes, cadspot.oracle_predictor(), name="oracle")
        self.assertEqual(report.regions["viewport"].hits, 2)

    def test_centre_and_random_are_floors(self):
        centre = cadspot.evaluate(self.targets, cadspot.center_predictor,
                                  name="center")
        self.assertLessEqual(centre.overall.accuracy, 0.5)
        rand = cadspot.evaluate(self.targets, cadspot.random_predictor(0),
                                name="random")
        self.assertLessEqual(rand.overall.accuracy, 1.0)

    def test_random_is_seeded_and_therefore_comparable(self):
        a = cadspot.evaluate(self.targets, cadspot.random_predictor(7)).to_dict()
        b = cadspot.evaluate(self.targets, cadspot.random_predictor(7)).to_dict()
        self.assertEqual(a["overall"]["hits"], b["overall"]["hits"])

    def test_report_shape(self):
        report = cadspot.evaluate(self.targets, cadspot.center_predictor, name="c")
        payload = report.to_dict()
        self.assertIn("point_in_bbox", payload["overall"])
        self.assertIn("error_rate", payload["overall"])
        self.assertIn("latency_ms", payload["overall"])


class TestPersistence(unittest.TestCase):
    def test_round_trip(self):
        d = tempfile.mkdtemp(prefix="hc_cadspot_")
        try:
            path = os.path.join(d, "b.jsonl")
            cadspot.save(path, [_target(), _target(region="toolbar",
                                                   instruction="Pad", entity="")])
            got = cadspot.load(path)
            self.assertEqual(len(got), 2)
            self.assertEqual({t.region for t in got}, {"viewport", "toolbar"})
        finally:
            shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(vp.gui_available(), "the FreeCAD GUI is not installed")
class TestLiveBenchmark(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="hc_cadspot_live_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_build_populates_the_viewport_region(self):
        targets, meta = cadspot.build(self.dir, count=1, seed=0,
                                      views=("isometric",))
        viewport = [t for t in targets if t.region == "viewport"]
        self.assertTrue(viewport, "the viewport region must not be empty")
        for t in viewport:
            self.assertTrue(t.entity)
            self.assertTrue(t.instruction)
        # The oracle must still be perfect on a real, freshly-built benchmark.
        report = cadspot.evaluate(targets, cadspot.oracle_predictor(),
                                  name="oracle", root=self.dir)
        self.assertEqual(report.regions["viewport"].accuracy, 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
