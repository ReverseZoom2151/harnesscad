"""Tests for domain.spec.nurbs_json_validator."""

import unittest

from harnesscad.domain.spec.nurbs_json_validator import (
    is_valid_face,
    validate_face,
    validate_model,
)


def _valid_face():
    # 2x2 poles, bilinear (degree 1 in both directions), clamped.
    # non-periodic: sum(mults) == n_poles + degree + 1 = 2 + 1 + 1 = 4 -> mults [2,2]
    return {
        "poles": [[[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [1, 1, 1]]],
        "u_knots": [0.0, 1.0],
        "v_knots": [0.0, 1.0],
        "u_mults": [2, 2],
        "v_mults": [2, 2],
        "u_degree": 1,
        "v_degree": 1,
        "u_periodic": 0,
        "v_periodic": 0,
        "weights": [[1.0, 1.0], [1.0, 1.0]],
    }


class ValidFaceTest(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(validate_face(_valid_face()), [])
        self.assertTrue(is_valid_face(_valid_face()))


class InvalidFaceTest(unittest.TestCase):
    def test_missing_field(self):
        f = _valid_face()
        del f["u_knots"]
        self.assertIn("missing required field: u_knots", validate_face(f))

    def test_non_rectangular_poles(self):
        f = _valid_face()
        f["poles"] = [[[0, 0, 0], [1, 0, 0]], [[0, 1, 0]]]
        self.assertTrue(validate_face(f))

    def test_knot_mult_relationship(self):
        f = _valid_face()
        f["u_mults"] = [2, 3]  # sum 5 != 4
        errs = validate_face(f)
        self.assertTrue(any("non-periodic" in e for e in errs))

    def test_non_increasing_knots(self):
        f = _valid_face()
        f["u_knots"] = [1.0, 1.0]
        self.assertTrue(any("strictly increasing" in e for e in validate_face(f)))

    def test_bad_weights_shape(self):
        f = _valid_face()
        f["weights"] = [[1.0, 1.0]]
        self.assertTrue(any("weights grid" in e for e in validate_face(f)))

    def test_negative_weight(self):
        f = _valid_face()
        f["weights"] = [[1.0, -1.0], [1.0, 1.0]]
        self.assertTrue(any("strictly positive" in e for e in validate_face(f)))

    def test_degree_zero(self):
        f = _valid_face()
        f["u_degree"] = 0
        self.assertTrue(any("degree must be" in e for e in validate_face(f)))

    def test_periodic_rule(self):
        # periodic: sum(mults) == n_poles
        f = _valid_face()
        f["u_periodic"] = 1
        f["u_mults"] = [1, 1]  # sum 2 == n_poles 2 -> ok in u
        errs = validate_face(f)
        self.assertFalse(any(e.startswith("u (periodic)") for e in errs))


class ValidateModelTest(unittest.TestCase):
    def test_all_valid(self):
        self.assertEqual(validate_model({"face_0": _valid_face()}), {})

    def test_reports_bad_face(self):
        bad = _valid_face()
        del bad["poles"]
        out = validate_model({"face_0": _valid_face(), "face_1": bad})
        self.assertIn("face_1", out)
        self.assertNotIn("face_0", out)

    def test_empty_model(self):
        self.assertIn("<model>", validate_model({}))


if __name__ == "__main__":
    unittest.main()
