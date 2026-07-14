"""THE DIFFERENTIAL ORACLE ACROSS THE GUI BOUNDARY.

The scripted ``io/backends/freecad.py`` matches ANALYTIC on all 20 CISP ops to
4.5e-16, so it is ground truth. The GUI environment drives the *same op stream*
through the running application's toolbar, dialogs and spinboxes -- and the two
must produce THE SAME PART.

That is a test nobody else in the computer-use field can write, because nobody
else knows what is supposed to be on the screen. A CUA whose output is checked by
a model's opinion of a screenshot cannot tell 37.5 mm from 375 mm. This one is
checked by the volume of the solid it built.

SKIPs cleanly when FreeCAD / uiautomation / Windows are absent. The live GUI half
additionally requires ``HARNESSCAD_CUA_LIVE=1`` so a routine test run never
hijacks the user's desktop. No user file is ever opened; FreeCAD is killed on
teardown; every export goes to a scratch directory the harness owns.
"""

import os
import unittest

from harnesscad.core.cisp.ops import AddRectangle, Extrude, NewSketch
from harnesscad.core.environment import BackendEnvironment, CapabilityError, require
from harnesscad.io.backends.base import BackendUnavailable

LIVE = os.environ.get("HARNESSCAD_CUA_LIVE") == "1"

#: The op streams driven through BOTH sides. Every one is a sketch-rectangle-pad,
#: which is exactly FreeCAD's additive Box primitive.
CASES = {
    # name: (ops, why this case)
    "box_30x20x10": [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=30.0, h=20.0),
        Extrude(sketch="sk1", distance=10.0),
    ],
    # THE LOCALE TRAP: 37.5 typed naively into this machine's comma-locale FreeCAD
    # reads back as 375,00 mm -- a silent 10x error. If the GUI side ships that,
    # the volume is 10x out and this test fails loudly.
    "box_fractional_37.5": [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=37.5, h=12.5),
        Extrude(sketch="sk1", distance=6.25),
    ],
    # Two additive primitives FUSE into the active body -- the same union the op
    # stream's successive extrudes produce.
    "two_boxes_fused": [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=30.0, h=20.0),
        Extrude(sketch="sk1", distance=10.0),
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk2", x=0.0, y=0.0, w=10.0, h=10.0),
        Extrude(sketch="sk2", distance=40.0),
    ],
}


def scripted_reference(ops):
    """The scripted FreeCAD backend's exact B-rep measurement of the op stream."""
    from harnesscad.io.backends.freecad import FreeCADBackend

    backend = FreeCADBackend()
    env = BackendEnvironment(backend)
    require(env, "content_digest", "nonmutating_reject", "synchronous_read")
    env.reset()
    for op in ops:
        result = env.step(op)
        if not result.ok:
            raise AssertionError("the scripted reference rejected %r: %s"
                                 % (op, [d.message for d in result.diagnostics]))
    return backend.query("metrics"), env.state_digest()


class TestScriptedReferenceIsGroundTruth(unittest.TestCase):
    """The reference side, on its own. Runs wherever freecadcmd is installed."""

    def setUp(self):
        try:
            self.metrics, self.digest = scripted_reference(CASES["box_30x20x10"])
        except BackendUnavailable as exc:
            raise unittest.SkipTest(str(exc))

    def test_the_reference_is_analytic(self):
        self.assertAlmostEqual(self.metrics["volume"], 30 * 20 * 10, places=9)
        self.assertEqual([round(v, 9) for v in self.metrics["bbox"]],
                         [30.0, 20.0, 10.0])

    def test_the_reference_has_a_content_digest(self):
        self.assertTrue(self.digest)


@unittest.skipUnless(LIVE, "live GUI differential (set HARNESSCAD_CUA_LIVE=1)")
class TestGuiMatchesScripted(unittest.TestCase):
    """THE MONEY TEST. Same op stream, two sides of the GUI boundary, one part."""

    @classmethod
    def setUpClass(cls):
        from harnesscad.io.cua import environment_freecad as E

        ok, why = E.available()
        if not ok:
            raise unittest.SkipTest(why)
        cls.E = E

    def _differential(self, name):
        try:
            reference, _digest = scripted_reference(CASES[name])
        except BackendUnavailable as exc:
            raise unittest.SkipTest(str(exc))
        env = self.E.FreeCADGuiEnvironment()
        try:
            env.reset()
            result = env.step(CASES[name])
            self.assertTrue(result.ok, [d.message for d in result.diagnostics])
            # Every GUI action was PROVEN by a read-back, not by a return code.
            self.assertTrue(result.verified)
            gui = env.measure("metrics")
            step = env.export("step")
        finally:
            env.close()
            env.scratch.cleanup()
        return reference, gui, step

    def _assert_same_part(self, reference, gui):
        self.assertAlmostEqual(gui["volume"], reference["volume"], places=6)
        self.assertAlmostEqual(gui["surface_area"], reference["surface_area"],
                               places=6)
        for got, want in zip(gui["bbox"], reference["bbox"]):
            self.assertAlmostEqual(got, want, places=9)
        for got, want in zip(gui["center_of_mass"], reference["center_of_mass"]):
            self.assertAlmostEqual(got, want, places=9)
        self.assertEqual(gui["faces"], reference["faces"])
        self.assertEqual(gui["edges"], reference["edges"])
        self.assertEqual(gui["solids"], reference["solids"])

    def test_box(self):
        reference, gui, _ = self._differential("box_30x20x10")
        self.assertAlmostEqual(gui["volume"], 6000.0, places=9)   # EXACTLY 6000
        self._assert_same_part(reference, gui)

    def test_fractional_dimensions_survive_the_comma_locale(self):
        """37.5 x 12.5 x 6.25 = 2929.6875 mm3. Typed naively into this machine's
        comma-locale FreeCAD, 37.5 becomes 375 and the volume is 10x out. The
        read-back assert is what stands between us and shipping that."""
        reference, gui, _ = self._differential("box_fractional_37.5")
        self.assertAlmostEqual(gui["volume"], 37.5 * 12.5 * 6.25, places=6)
        self._assert_same_part(reference, gui)

    def test_two_additive_primitives_fuse_like_two_extrudes(self):
        reference, gui, _ = self._differential("two_boxes_fused")
        self._assert_same_part(reference, gui)

    def test_the_export_came_through_the_harness_channel_not_the_apps_save(self):
        _reference, _gui, step = self._differential("box_30x20x10")
        self.assertIn("ISO-10303-21", step)
        self.assertIn("AUTOMOTIVE_DESIGN", step)   # AP214, the kernel's own writer


@unittest.skipUnless(LIVE, "live GUI honesty checks (set HARNESSCAD_CUA_LIVE=1)")
class TestTheGuiIsHonestAboutWhatItCannotDo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from harnesscad.io.cua import environment_freecad as E

        ok, why = E.available()
        if not ok:
            raise unittest.SkipTest(why)
        cls.env = E.FreeCADGuiEnvironment()
        cls.env.reset()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        cls.env.scratch.cleanup()

    def test_state_digest_REFUSES_rather_than_faking_one(self):
        """The book audit's whole point: a GUI is not a GeometryBackend, because it
        cannot produce a content digest. The correct behaviour is to say so."""
        caps = self.env.capabilities()
        self.assertFalse(caps.content_digest)
        self.assertFalse(caps.nonmutating_reject)
        self.assertFalse(caps.synchronous_read)
        with self.assertRaises(CapabilityError) as ctx:
            self.env.state_digest()
        self.assertIn("has no content hash", str(ctx.exception))
        # And the observation reports None -- it never substitutes something else.
        self.assertIsNone(self.env.observe().digest)

    def test_an_op_needing_a_viewport_pick_is_refused_not_faked(self):
        from harnesscad.core.cisp.ops import Fillet

        result = self.env.step(Fillet(edges=("|Z",), radius=2.0))
        self.assertFalse(result.ok)
        self.assertFalse(result.verified)
        self.assertIn("EDGE selection in the viewport",
                      result.diagnostics[0].message)

    def test_it_declares_resolve_before_act_which_a_kernel_cannot(self):
        self.assertTrue(self.env.capabilities().resolve_before_act)


if __name__ == "__main__":
    unittest.main()
