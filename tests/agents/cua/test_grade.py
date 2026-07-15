"""The geometric grader: the scripted differential, and (live) the full loop.

The scripted half runs wherever freecadcmd is installed and needs no GUI and no
model: it proves the differential and the target oracle on ground truth. The live
half (HARNESSCAD_CUA_LIVE=1) drives a REAL model in a REAL FreeCAD GUI and grades
the built part — the actual "agent in the environment" end to end.
"""

import os
import unittest

from harnesscad.agents.cua.briefs import Target, by_id
from harnesscad.agents.cua.grade import (
    differential, grade_ops, scripted_measure,
)
from harnesscad.core.cisp.ops import AddRectangle, Extrude, NewSketch

LIVE = os.environ.get("HARNESSCAD_CUA_LIVE") == "1"

BOX = [NewSketch(plane="XY"), AddRectangle(sketch="sk1", x=0, y=0, w=30, h=20),
       Extrude(sketch="sk1", distance=10)]


def _skip_if_no_freecadcmd():
    try:
        scripted_measure(BOX)
    except Exception as exc:  # noqa: BLE001
        if "Unavailable" in type(exc).__name__ or "freecad" in str(exc).lower():
            raise unittest.SkipTest("scripted FreeCAD backend not available: %s" % exc)
        raise


class TestScriptedGroundTruth(unittest.TestCase):
    def setUp(self):
        _skip_if_no_freecadcmd()

    def test_scripted_box_is_analytic(self):
        metrics, gate_ok, failures = scripted_measure(BOX)
        self.assertAlmostEqual(metrics["volume"], 6000.0, places=6)
        self.assertTrue(gate_ok, failures)

    def test_target_oracle_accepts_the_right_box(self):
        metrics, _ok, _f = scripted_measure(BOX)
        ok, misses = by_id("block_30x20x10").target.satisfied(metrics)
        self.assertTrue(ok, misses)

    def test_target_oracle_rejects_the_wrong_box(self):
        metrics, _ok, _f = scripted_measure(BOX)
        wrong = Target(volume=1.0, bbox=(1, 1, 1))
        ok, misses = wrong.satisfied(metrics)
        self.assertFalse(ok)
        self.assertTrue(misses)


class TestDifferential(unittest.TestCase):
    def test_identical_measurements_agree(self):
        m = {"volume": 6000.0, "surface_area": 2200.0, "bbox": [30, 20, 10],
             "center_of_mass": [15, 10, 5], "faces": 6, "edges": 12, "solids": 1}
        d = differential(m, dict(m))
        self.assertTrue(d.agree)
        self.assertEqual(d.max_delta, 0.0)

    def test_ten_x_error_is_caught(self):
        # THE locale bug: 37.5 -> 375 makes the volume 10x. The differential must
        # not shrug at that.
        good = {"volume": 2929.6875, "bbox": [37.5, 12.5, 6.25], "faces": 6,
                "edges": 12, "solids": 1}
        bad = {"volume": 29296.875, "bbox": [375, 12.5, 6.25], "faces": 6,
               "edges": 12, "solids": 1}
        d = differential(bad, good)
        self.assertFalse(d.agree)
        self.assertTrue(d.mismatches)

    def test_axis_permutation_is_not_a_mismatch(self):
        a = {"volume": 6000.0, "bbox": [30, 20, 10], "faces": 6, "edges": 12, "solids": 1}
        b = {"volume": 6000.0, "bbox": [20, 10, 30], "faces": 6, "edges": 12, "solids": 1}
        self.assertTrue(differential(a, b).agree)


@unittest.skipUnless(LIVE, "live GUI + model (set HARNESSCAD_CUA_LIVE=1)")
class TestLiveAgentInTheEnvironment(unittest.TestCase):
    """A real model, a real GUI, graded on geometry. The whole thing."""

    @classmethod
    def setUpClass(cls):
        from harnesscad.io.cua import environment_freecad as E

        ok, why = E.available()
        if not ok:
            raise unittest.SkipTest(why)
        from harnesscad.agents.cua.models import largest_available
        cls.model = largest_available()
        if cls.model is None:
            raise unittest.SkipTest("no Ollama model installed")
        cls.E = E

    def test_a_model_builds_the_block_and_it_grades_out(self):
        from harnesscad.agents.cua.loop import solve
        from harnesscad.agents.cua.models import make_llm

        brief = by_id("block_30x20x10")
        env = self.E.FreeCADGuiEnvironment()
        try:
            env.reset()
            result = solve(env, make_llm(self.model), brief, max_iterations=3)
        finally:
            env.close()
            env.scratch.cleanup()
        # Whatever the model did, the grade must be honest: if it solved it, the
        # differential against the scripted kernel must agree.
        if result.solved:
            self.assertIsNotNone(result.grade)
            self.assertTrue(result.grade.diff.agree, result.grade.reason)
            self.assertTrue(result.grade.gui_valid)


if __name__ == "__main__":
    unittest.main()
