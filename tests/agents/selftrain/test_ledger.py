"""The capability ledger must stay honest, and a test is the only thing that
makes a docstring load-bearing."""

from __future__ import annotations

import unittest

from harnesscad.agents.selftrain import ledger
from harnesscad.eval.pressure import briefs as briefs_mod


class TestInstruments(unittest.TestCase):

    def test_every_instrument_declares_a_blind_spot(self):
        # An instrument that claims to see everything is an instrument nobody has
        # audited. There are four and each one names what it cannot do.
        self.assertEqual(len(ledger.INSTRUMENTS), 4)
        for inst in ledger.INSTRUMENTS:
            self.assertTrue(inst.can_see, inst.name)
            self.assertTrue(inst.cannot_see,
                            "%s declares no blind spot" % inst.name)

    def test_reference_free_split(self):
        # Only the reference-free instruments can ever grade a brief a user typed.
        self.assertEqual(set(ledger.REFERENCE_FREE), {"differential", "gate"})
        self.assertEqual(set(ledger.REFERENCE_BOUND), {"envelope", "shape"})

    def test_shell_brief_blind_spot_is_named(self):
        # shell_box_3mm carries bbox=None. That is the hole the shell bug shipped
        # through: the fleet did not fire AND the corpus could not have known.
        brief = briefs_mod.brief_by_id("shell_box_3mm")
        holes = ledger.blind_spots_of_brief(brief)
        self.assertTrue(any("expect.bbox is None" in h for h in holes), holes)


class TestCertify(unittest.TestCase):

    def test_reference_stream_is_certified(self):
        brief = briefs_mod.brief_by_id("trap_hole_oversize")
        cert = ledger.certify(brief, list(brief.reference))
        self.assertTrue(cert.accepted)
        self.assertTrue(cert.gate_ok)
        self.assertTrue(cert.envelope_ok)
        self.assertTrue(cert.shape_ok)

    def test_the_14b_regression_is_refused(self):
        # The op stream the 14b produced after the fleet's false diagnostic.
        brief = briefs_mod.brief_by_id("trap_hole_oversize")
        ops = [
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0,
             "w": 40, "h": 40},
            {"op": "extrude", "sketch": "sk1", "distance": 10},
            {"op": "hole", "face_or_sketch": "solid", "x": 20, "y": 20,
             "diameter": 8, "through": True, "kind": "simple", "depth": None},
        ]
        cert = ledger.certify(brief, ops)
        self.assertFalse(cert.accepted)
        self.assertFalse(cert.envelope_ok)

    def test_the_shape_metric_cannot_see_the_14b_regression(self):
        # THE HONEST WARNING, ASSERTED. The 8mm-for-12mm hole scores IoU 0.963 and
        # the SHAPE channel calls it a match. If this test ever starts failing
        # because the metric got sharper, DELETE IT and celebrate -- but until
        # then, nobody gets to claim the shape metric closes the many-to-one hole.
        brief = briefs_mod.brief_by_id("trap_hole_oversize")
        ops = [
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0,
             "w": 40, "h": 40},
            {"op": "extrude", "sketch": "sk1", "distance": 10},
            {"op": "hole", "face_or_sketch": "solid", "x": 20, "y": 20,
             "diameter": 8, "through": True, "kind": "simple", "depth": None},
        ]
        cert = ledger.certify(brief, ops)
        self.assertIsNotNone(cert.shape_iou)
        self.assertGreater(cert.shape_iou, 0.90)
        self.assertTrue(cert.shape_ok)
        # It is the ENVELOPE's op-level assertion that catches it, not the shape.
        self.assertFalse(cert.envelope_ok)

    def test_unparseable_gets_no_certificate(self):
        brief = briefs_mod.brief_by_id("l_bracket")
        cert = ledger.certify(brief, [])
        self.assertFalse(cert.accepted)

    def test_accepted_certificates_still_carry_blind_spots(self):
        brief = briefs_mod.brief_by_id("trap_hole_oversize")
        cert = ledger.certify(brief, list(brief.reference))
        self.assertTrue(cert.accepted)
        self.assertTrue(cert.blind_spots,
                        "an accepted candidate must still say what nobody looked at")


if __name__ == "__main__":                                # pragma: no cover
    unittest.main()
