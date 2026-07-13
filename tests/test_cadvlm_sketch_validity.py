import unittest

from harnesscad.io.ingest.cadvlm_sketch_validity import (
    ValidityReport, check_sketch, constraint_issues, constraint_name,
    entity_issues,
)


class CadVLMSketchValidityTests(unittest.TestCase):
    def test_valid_entities(self):
        self.assertEqual(entity_issues(("line", 1, 32, 64, 32)), ())
        self.assertEqual(entity_issues(("arc", 16, 32, 32, 48, 48, 32)), ())
        self.assertEqual(
            entity_issues(("circle", 48, 32, 32, 48, 16, 32, 32, 16)), ())

    def test_entity_bad_count_and_range(self):
        self.assertTrue(any("bad-token-count" in m
                            for m in entity_issues(("line", 1, 2, 3))))
        self.assertTrue(any("coord-out-of-range" in m
                            for m in entity_issues(("line", 0, 2, 65, 2))))
        self.assertEqual(entity_issues(("spline", 1, 2)), ("unknown-entity:spline",))
        self.assertEqual(entity_issues(()), ("empty-entity",))

    def test_constraint_references(self):
        # parallel (token 73) needs >=2 refs, indices must exist.
        self.assertEqual(constraint_issues((73, 0, 1), n_entities=2), ())
        self.assertTrue(any("insufficient-references" in m
                            for m in constraint_issues((73, 0), 2)))
        self.assertTrue(any("reference-out-of-range" in m
                            for m in constraint_issues((73, 0, 5), 2)))
        self.assertEqual(constraint_issues((999, 0, 1), 2),
                         ("unknown-constraint-token:999",))

    def test_check_sketch_collects_indexed_issues(self):
        report = check_sketch(
            entities=(("line", 1, 32, 64, 32), ("line", 1, 2, 3)),
            constraints=((73, 0, 1), (73, 0, 9)))
        self.assertIsInstance(report, ValidityReport)
        self.assertFalse(report.valid)
        self.assertTrue(any(i == 1 for i, _ in report.entity_issues))
        self.assertTrue(any(j == 1 for j, _ in report.constraint_issues))
        self.assertEqual(report.all_issues,
                         report.entity_issues + report.constraint_issues)

    def test_valid_whole_sketch(self):
        report = check_sketch(
            entities=(("line", 1, 32, 64, 32),
                      ("circle", 48, 32, 32, 48, 16, 32, 32, 16)),
            constraints=((69, 0),))               # horizontal on entity 0
        self.assertTrue(report.valid)
        self.assertEqual(report.all_issues, ())

    def test_constraint_name(self):
        self.assertEqual(constraint_name(73), "parallel")
        self.assertEqual(constraint_name(65), "coincident")


if __name__ == "__main__":
    unittest.main()
