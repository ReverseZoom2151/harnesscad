"""The dead-field ratchet: an op field the kernel ignores may not be introduced."""

from __future__ import annotations

import unittest

from harnesscad.eval.gates import liveness_floor as lf


class _Cell:
    def __init__(self, backend, op, field_):
        self._d = {"backend": backend, "op": op, "field": field_}

    def to_dict(self):
        return dict(self._d)


class _Report:
    def __init__(self, dead, backends=("frep",), unmapped=()):
        self.dead = list(dead)
        self.backends = list(backends)
        self.unmapped = list(unmapped)


CENSUS = {"dead": ["frep:hole.kind", "frep:fillet.edges"]}


class TestRatchet(unittest.TestCase):
    def test_the_committed_census_is_real(self):
        """Every census entry is a well-formed backend:op.field.

        This used to also assert the census was NON-EMPTY -- a guard against a
        census stubbed out to nothing, written when frep had 15 dead fields and an
        empty list could only mean somebody had switched the gate off. It is now
        empty because all 15 were FIXED, so that assertion had inverted into a test
        that fails on success. What is worth pinning is the shape of an entry (the
        gate keys off it) and the debt's DIRECTION, below.
        """
        base = lf.baseline()
        self.assertIsInstance(base["dead"], list)
        for entry in base["dead"]:
            self.assertRegex(entry, r"^[a-z]+:[a-z_]+\.[a-z_0-9]+$")

    def test_the_census_never_grows(self):
        """The ratchet's one rule, pinned against the committed file.

        The census may only ever shrink. It is empty today; if a future diff adds
        an entry, that is a new dead field being written down instead of fixed, and
        this test is where that argument has to be had.
        """
        self.assertEqual(lf.baseline()["dead"], [],
                         "the dead-field census grew. A dead field is a bug to "
                         "fix, not a debt to record -- see the module docstring")

    def test_the_known_debt_passes(self):
        rep = _Report([_Cell("frep", "hole", "kind"),
                       _Cell("frep", "fillet", "edges")])
        gate = lf.check(rep, census=CENSUS)
        self.assertTrue(gate.ok)
        self.assertEqual(len(gate.known_dead), 2)

    def test_a_NEW_dead_field_fails_the_build(self):
        rep = _Report([_Cell("frep", "hole", "kind"),
                       _Cell("frep", "fillet", "edges"),
                       _Cell("frep", "extrude", "distance")])
        gate = lf.check(rep, census=CENSUS)
        self.assertFalse(gate.ok)
        self.assertEqual(gate.new_dead, ["frep:extrude.distance"])

    def test_a_revived_field_fails_until_the_census_is_tightened(self):
        rep = _Report([_Cell("frep", "hole", "kind")])
        gate = lf.check(rep, census=CENSUS)
        self.assertFalse(gate.ok)
        self.assertEqual(gate.revived, ["frep:fillet.edges"])

    def test_an_unmeasured_backend_is_not_claimed_as_revived(self):
        census = {"dead": ["cadquery:hole.kind"]}
        gate = lf.check(_Report([], backends=("frep",)), census=census)
        self.assertEqual(gate.revived, [])
        self.assertTrue(gate.ok)

    def test_a_schema_field_the_oracle_does_not_know_fails(self):
        gate = lf.check(_Report([], unmapped=[("extrude", "taper")]), census={"dead": []})
        self.assertFalse(gate.ok)


if __name__ == "__main__":
    unittest.main()
