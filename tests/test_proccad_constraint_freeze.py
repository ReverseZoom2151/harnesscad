"""Tests for procedural.proccad_constraint_freeze."""

import unittest

from procedural.proccad_constraint_freeze import (
    build_constraints,
    check_preserved,
    freeze_region,
    project_onto_constraints,
)


SOL = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}


class FreezeTest(unittest.TestCase):
    def test_freeze_region_snapshot(self):
        frozen = freeze_region(SOL, ["a", "c"])
        self.assertEqual(frozen, {"a": 1.0, "c": 3.0})

    def test_freeze_missing_variable(self):
        with self.assertRaises(KeyError):
            freeze_region(SOL, ["z"])

    def test_bad_kind(self):
        with self.assertRaises(ValueError):
            freeze_region(SOL, ["a"], kind="bogus")

    def test_build_constraints_splits(self):
        cs = build_constraints(SOL, intangible_regions=[["a", "b"]], desirable_features=[["c"]])
        self.assertEqual(cs.intangible, {"a": 1.0, "b": 2.0})
        self.assertEqual(cs.desirable, {"c": 3.0})

    def test_frozen_and_free(self):
        cs = build_constraints(SOL, intangible_regions=[["a"]], desirable_features=[["b"]])
        self.assertEqual(cs.frozen_variables(), {"a", "b"})
        self.assertEqual(cs.free_variables(SOL.keys()), {"c", "d"})


class PreservationTest(unittest.TestCase):
    def test_preserved_ok(self):
        cs = build_constraints(SOL, intangible_regions=[["a", "b"]])
        new = {"a": 1.0, "b": 2.0, "c": 99.0, "d": 99.0}  # only free vars changed
        ok, viols = check_preserved(new, cs)
        self.assertTrue(ok)
        self.assertEqual(viols, [])

    def test_intangible_violation_fails(self):
        cs = build_constraints(SOL, intangible_regions=[["a"]])
        ok, viols = check_preserved({"a": 5.0}, cs)
        self.assertFalse(ok)
        self.assertEqual(len(viols), 1)
        self.assertEqual(viols[0].variable, "a")
        self.assertEqual(viols[0].kind, "intangible")

    def test_desirable_violation_reported_but_ok(self):
        cs = build_constraints(SOL, desirable_features=[["c"]])
        ok, viols = check_preserved({"c": 7.0}, cs)
        self.assertTrue(ok)  # desirable does not fail the hard check
        self.assertEqual(len(viols), 1)
        self.assertEqual(viols[0].kind, "desirable")

    def test_missing_variable_is_violation(self):
        cs = build_constraints(SOL, intangible_regions=[["a"]])
        ok, viols = check_preserved({}, cs)
        self.assertFalse(ok)
        self.assertEqual(len(viols), 1)

    def test_project_restores_intangible(self):
        cs = build_constraints(SOL, intangible_regions=[["a", "b"]], desirable_features=[["c"]])
        candidate = {"a": 5.0, "b": 6.0, "c": 7.0, "d": 8.0}
        repaired = project_onto_constraints(candidate, cs)
        self.assertEqual(repaired["a"], 1.0)
        self.assertEqual(repaired["b"], 2.0)
        self.assertEqual(repaired["c"], 7.0)  # desirable not forced
        self.assertEqual(repaired["d"], 8.0)
        ok, _ = check_preserved(repaired, cs)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
