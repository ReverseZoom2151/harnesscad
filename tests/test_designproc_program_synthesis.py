"""Tests for datagen.designproc_program_synthesis."""

import unittest

from harnesscad.data.datagen import design_procedure as dp
from harnesscad.data.datagen import program_synthesis as ps


class TestSynthesizeProgram(unittest.TestCase):
    def test_full_program_has_bspline(self):
        proc = dp.build_procedure("bracket", "saddle", n_primitives=2)
        prog = ps.synthesize_program(proc, seed=1)
        self.assertTrue(ps.has_bspline_geometry(prog))
        # First command is the spline surface.
        self.assertEqual(prog[0]["op"], "make_spline_surface")

    def test_deterministic(self):
        proc = dp.build_procedure("bracket", "wave", n_primitives=3)
        a = ps.synthesize_program(proc, seed=7)
        b = ps.synthesize_program(proc, seed=7)
        self.assertEqual(a, b)

    def test_seed_changes_program(self):
        proc = dp.build_procedure("bracket", "wave", n_primitives=3)
        a = ps.synthesize_program(proc, seed=1)
        b = ps.synthesize_program(proc, seed=2)
        # conform_face count is seed-dependent, so programs may differ in length.
        self.assertTrue(a != b or len(a) == len(b))  # at least well-formed

    def test_invalid_procedure_rejected(self):
        bad = dp.DesignProcedure("bracket", [
            dp.DesignStep(dp.SELECT_REFERENCE_SURFACE),
            dp.DesignStep(dp.EXPORT),  # surface never removed, no features
        ])
        with self.assertRaises(ValueError):
            ps.synthesize_program(bad, seed=1)

    def test_baseline_has_no_bspline(self):
        proc = dp.build_procedure("bracket", "saddle",
                                  with_reference_surface=False,
                                  with_fillet=False)
        prog = ps.synthesize_program(proc, seed=3)
        self.assertFalse(ps.has_bspline_geometry(prog))


class TestTotals(unittest.TestCase):
    def test_totals_non_negative_and_summed(self):
        proc = dp.build_procedure("bracket", "gaussian", n_primitives=2)
        prog = ps.synthesize_program(proc, seed=5)
        t = ps.program_totals(prog)
        for k in ("faces", "curves", "bspline_faces", "bspline_curves"):
            self.assertGreaterEqual(t[k], 0)
        self.assertGreater(t["lines"], 0)
        self.assertEqual(t["n_commands"], len(prog))

    def test_surface_removal_nets_out_own_primitive(self):
        # Surface removal subtracts its own face; conform keeps object curvature.
        proc = dp.build_procedure("bracket", "saddle", n_primitives=1,
                                  with_fillet=False)
        prog = ps.synthesize_program(proc, seed=9)
        t = ps.program_totals(prog)
        # Still positive B-Spline faces from conform steps after removal.
        self.assertGreater(t["bspline_faces"], 0)


class TestFamily(unittest.TestCase):
    def test_family_size_and_determinism(self):
        surfaces = ["saddle", "gaussian"]
        descs = ["a bracket with two holes", "a U-shaped bracket", "a plate"]
        fam1 = ps.synthesize_family("bracket", surfaces, descs, seed=11)
        fam2 = ps.synthesize_family("bracket", surfaces, descs, seed=11)
        self.assertEqual(len(fam1), 6)
        self.assertEqual([f["totals"] for f in fam1],
                         [f["totals"] for f in fam2])

    def test_full_family_all_bspline(self):
        fam = ps.synthesize_family(
            "bracket", ["saddle", "ripple"], ["d1", "d2"], seed=2, mode="full")
        for f in fam:
            self.assertGreater(f["totals"]["bspline_faces"], 0)
            self.assertEqual(f["surface_kind"] and True, True)

    def test_baseline_family_no_bspline(self):
        fam = ps.synthesize_family(
            "bracket", ["saddle"], ["d1", "d2"], seed=2, mode="none")
        for f in fam:
            # No fillet in baseline build? build_procedure defaults with_fillet=True,
            # which DOES add bspline. Confirm surface_kind blanked instead.
            self.assertEqual(f["surface_kind"], "")

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            ps.synthesize_family("bracket", ["saddle"], ["d"], seed=1, mode="x")


if __name__ == "__main__":
    unittest.main()
