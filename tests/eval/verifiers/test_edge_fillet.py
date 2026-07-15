"""Tests for the SOUND per-edge fillet ceiling (verifiers.edge_fillet).

The rule must (a) catch the degenerate all-edges fillet, (b) stay SILENT on the
same body filleted below the ceiling, (c) NOT fire on the exact cases the
whole-body rule falsely condemned (a large fillet on edges away from the thin
dimension), and (d) ABSTAIN outside the box scope where the proof does not hold.
"""

import unittest

from harnesscad.core.cisp.ops import (AddCircle, AddRectangle, Chamfer, Extrude,
                                      Fillet, Hole, NewSketch, Revolve, Shell)
from harnesscad.eval.verifiers.edge_fillet import (box_extents,
                                                   degenerate_fillets)


def _plate(w, h, d, *feature):
    return [NewSketch("XY"), AddRectangle("sk1", 0, 0, w, h),
            Extrude("sk1", d), *feature]


class TestTheorem(unittest.TestCase):
    def test_all_edges_over_ceiling_fires(self):
        # 50x30x6 plate, all edges R5. On the 6 mm-thick side faces both parallel
        # edges are rounded: 2*5 = 10 >= 6. Degenerate.
        self.assertTrue(degenerate_fillets(_plate(50, 30, 6, Fillet((), 5.0))))

    def test_all_edges_under_ceiling_silent(self):
        # Same plate, R2: 2*2 = 4 < 6. Valid, and the rule must be silent.
        self.assertEqual(degenerate_fillets(_plate(50, 30, 6, Fillet((), 2.0))), [])

    def test_exact_bullnose_is_silent(self):
        # 2r == extent: the arcs are tangent at the mid-plane -- a valid BULLNOSE
        # (a fully rounded edge), NOT a degeneracy. r=3 on a 6 mm plate is exactly
        # test_soundness's known-good filleted_thin_plate, which builds. A PROVEN
        # rule must not condemn it.
        self.assertEqual(degenerate_fillets(_plate(50, 30, 6, Fillet((), 3.0))), [])

    def test_just_past_the_bullnose_fires(self):
        self.assertTrue(degenerate_fillets(_plate(50, 30, 6, Fillet((), 3.001))))

    def test_just_below_half_extent_silent(self):
        self.assertEqual(degenerate_fillets(_plate(50, 30, 6, Fillet((), 2.999))), [])

    def test_vertical_edges_large_radius_silent(self):
        # THE false positive the whole-body rule caused: R5 on the four vertical
        # edges of a 6 mm plate is valid -- those edges are adjacent to the 50 and
        # 30 mm faces, not the 6 mm thickness. min(50,30)=30; 2*5=10 < 30.
        self.assertEqual(
            degenerate_fillets(_plate(50, 30, 6, Fillet(("|Z",), 5.0))), [])

    def test_vertical_edges_true_degenerate_fires(self):
        # ... but R16 on the same vertical edges IS degenerate: 2*16=32 >= 30.
        self.assertTrue(
            degenerate_fillets(_plate(50, 30, 6, Fillet(("|Z",), 16.0))))

    def test_single_face_large_radius_silent(self):
        # Only the top face's edges rounded (opposite edge NOT filleted): a single
        # strip of width r fits while r < the perpendicular extent. R5 < 6: valid.
        self.assertEqual(
            degenerate_fillets(_plate(50, 30, 6, Fillet((">Z",), 5.0))), [])

    def test_single_face_at_full_extent_is_silent(self):
        # Single strip, r == extent: the tangent point lands exactly on the far
        # edge -- a valid bullnose of the whole face, not a degeneracy.
        self.assertEqual(
            degenerate_fillets(_plate(50, 30, 6, Fillet((">Z",), 6.0))), [])

    def test_single_face_over_full_extent_fires(self):
        # The single-strip ceiling is the FULL extent, not half: R6.1 > 6.
        self.assertTrue(
            degenerate_fillets(_plate(50, 30, 6, Fillet((">Z",), 6.1))))

    def test_chamfer_treated_like_fillet(self):
        self.assertTrue(degenerate_fillets(_plate(50, 30, 6, Chamfer((), 5.0))))

    def test_negative_radius_abstains(self):
        # A non-positive radius is a typo handled by INVALID_INPUT elsewhere; the
        # ceiling rule has no opinion on it.
        self.assertEqual(degenerate_fillets(_plate(40, 40, 10, Fillet((), -3.0))), [])

    def test_deterministic(self):
        a = degenerate_fillets(_plate(50, 30, 6, Fillet((), 5.0)))
        b = degenerate_fillets(_plate(50, 30, 6, Fillet((), 5.0)))
        self.assertEqual([f.edge for f in a], [f.edge for f in b])


class TestScope(unittest.TestCase):
    def test_shell_present_abstains(self):
        self.assertIsNone(box_extents(_plate(60, 40, 20, Shell((), 3.0))))

    def test_hole_present_abstains(self):
        self.assertIsNone(
            box_extents(_plate(20, 20, 10, Hole("sk1", 10, 10, 30, None, True, "simple"))))

    def test_circle_sketch_abstains(self):
        ops = [NewSketch("XY"), AddCircle("sk1", 0, 0, 40.0), Extrude("sk1", 8.0),
               Fillet((), 5.0)]
        self.assertIsNone(box_extents(ops))

    def test_revolve_abstains(self):
        ops = [NewSketch("XY"), AddRectangle("sk1", 10, 0, 5, 20),
               Revolve("sk1", (0, 0, 0, 0, 1, 0), 360.0)]
        self.assertIsNone(box_extents(ops))

    def test_non_xy_plane_abstains(self):
        ops = [NewSketch("XZ"), AddRectangle("sk1", 0, 0, 50, 30),
               Extrude("sk1", 6.0), Fillet((), 5.0)]
        self.assertIsNone(box_extents(ops))
        self.assertEqual(degenerate_fillets(ops), [])

    def test_zero_extrude_abstains(self):
        self.assertIsNone(box_extents(_plate(50, 30, 0.0, Fillet((), 5.0))))

    def test_clean_box_in_scope(self):
        self.assertIsNotNone(box_extents(_plate(50, 30, 6, Fillet((), 2.0))))


class TestFleetIntegration(unittest.TestCase):
    """The rule, as the fleet actually runs it: an ERROR diagnostic, PROVEN tier."""

    def test_wired_into_fleet_and_declared(self):
        from harnesscad.eval.verifiers import registry as reg
        from harnesscad.eval.verifiers import soundness as snd
        names = {getattr(v, "name", "") for v in reg.discover(refresh=True)}
        self.assertIn("edge-fillet", names)
        self.assertEqual(snd.soundness_of("edge-fillet").default, snd.PROVEN)

    def test_emits_error_on_degenerate_and_is_model_facing(self):
        from harnesscad.eval.verifiers import registry as reg
        from harnesscad.eval.verifiers import soundness as snd
        from harnesscad.eval.verifiers.verify import Severity

        class _StubDag:
            def __init__(self, ops):
                self._ops = ops

            def ops(self):
                return list(self._ops)

        class _StubBackend:
            def query(self, what):
                return {}

        ops = _plate(50, 30, 6, Fillet((), 5.0))
        state = reg.model_state(_StubBackend(), _StubDag(ops))
        diags = reg.run_all(state, tiers=reg.TIERS, only=["edge-fillet"])
        errors = [d for d in diags if d.severity is Severity.ERROR]
        self.assertTrue(errors)
        self.assertEqual(errors[0].code, "edge-fillet-degenerate")
        # A PROVEN ERROR is allowed to instruct the model.
        self.assertTrue(snd.model_facing(errors))

    def test_silent_on_valid_plate(self):
        from harnesscad.eval.verifiers import registry as reg

        class _StubDag:
            def __init__(self, ops):
                self._ops = ops

            def ops(self):
                return list(self._ops)

        class _StubBackend:
            def query(self, what):
                return {}

        ops = _plate(50, 30, 6, Fillet(("|Z",), 5.0))
        state = reg.model_state(_StubBackend(), _StubDag(ops))
        diags = reg.run_all(state, tiers=reg.TIERS, only=["edge-fillet"])
        self.assertEqual(diags, [])


if __name__ == "__main__":
    unittest.main()
