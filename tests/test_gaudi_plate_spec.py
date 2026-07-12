import unittest

from spec.gaudi_plate_spec import (
    CATEGORIES,
    HANDLE_TYPES,
    PlateSpecError,
    is_valid_building,
    is_valid_plate,
    normalize_plate,
    validate_building,
    validate_plate,
)


def _vertex_plate():
    return {
        "name": "base",
        "category": "vertex",
        "thickness": 1.5,
        "vertices": [(0, 0), (2, 0), (2, 2), (0, 2)],
    }


def _parametric_plate():
    return {
        "name": "wave",
        "category": "parametric",
        "thickness": 1.0,
        "formula": {"x": "4 * math.cos(t)", "y": "4 * math.sin(t)"},
        "range": (0, 6.283185),
        "steps": 100,
    }


def _mixed_plate():
    return {
        "name": "curve",
        "category": "mixed",
        "thickness": 0.5,
        "vertices": [(0, 0), (1, 0), (1, 1)],
        "handle_types": [("AUTO", "AUTO"), ("VECTOR", "FREE"), ("AUTO", "ALIGNED")],
    }


class ValidPlatesTests(unittest.TestCase):
    def test_vertex_valid(self):
        self.assertEqual(validate_plate(_vertex_plate()), [])
        self.assertTrue(is_valid_plate(_vertex_plate()))

    def test_parametric_valid(self):
        self.assertEqual(validate_plate(_parametric_plate()), [])

    def test_mixed_valid(self):
        self.assertEqual(validate_plate(_mixed_plate()), [])

    def test_categories_constant(self):
        self.assertEqual(CATEGORIES, ("vertex", "parametric", "mixed"))
        self.assertIn("AUTO", HANDLE_TYPES)


class InvalidPlateTests(unittest.TestCase):
    def test_not_a_dict(self):
        self.assertEqual(validate_plate([1, 2]), ["plate: must be a dict"])

    def test_missing_name(self):
        p = _vertex_plate()
        del p["name"]
        self.assertIn("name: must be a non-empty string", validate_plate(p))

    def test_bad_thickness(self):
        p = _vertex_plate()
        p["thickness"] = -1
        self.assertIn("thickness: must be positive", validate_plate(p))
        p["thickness"] = "x"
        self.assertIn("thickness: must be a number", validate_plate(p))

    def test_bool_is_not_number(self):
        p = _vertex_plate()
        p["thickness"] = True
        self.assertIn("thickness: must be a number", validate_plate(p))

    def test_unknown_category(self):
        p = _vertex_plate()
        p["category"] = "blob"
        issues = validate_plate(p)
        self.assertTrue(any("category:" in i for i in issues))

    def test_too_few_vertices(self):
        p = _vertex_plate()
        p["vertices"] = [(0, 0), (1, 1)]
        self.assertTrue(any("at least 3 points" in i for i in validate_plate(p)))

    def test_bad_vertex_point(self):
        p = _vertex_plate()
        p["vertices"] = [(0, 0), (1, 0), (1, "y")]
        self.assertTrue(any("vertices[2]" in i for i in validate_plate(p)))

    def test_parametric_missing_formula_axis(self):
        p = _parametric_plate()
        p["formula"] = {"x": "t"}
        self.assertTrue(any("formula.y" in i for i in validate_plate(p)))

    def test_parametric_bad_range(self):
        p = _parametric_plate()
        p["range"] = (1, 1)
        self.assertTrue(any("range: start and end must differ" in i for i in validate_plate(p)))

    def test_parametric_bad_steps(self):
        p = _parametric_plate()
        p["steps"] = 2
        self.assertTrue(any("steps: need at least 3" in i for i in validate_plate(p)))
        p["steps"] = 5.0
        self.assertTrue(any("steps: must be an integer" in i for i in validate_plate(p)))

    def test_mixed_handle_length_mismatch(self):
        p = _mixed_plate()
        p["handle_types"] = [("AUTO", "AUTO")]
        self.assertTrue(any("length" in i for i in validate_plate(p)))

    def test_mixed_bad_handle_value(self):
        p = _mixed_plate()
        p["handle_types"] = [("AUTO", "AUTO"), ("NOPE", "FREE"), ("AUTO", "AUTO")]
        self.assertTrue(any("not in" in i for i in validate_plate(p)))

    def test_bad_rotation(self):
        p = _vertex_plate()
        p["rotation"] = "left"
        self.assertIn("rotation: must be a number (degrees)", validate_plate(p))

    def test_bad_position(self):
        p = _vertex_plate()
        p["position"] = {"x": "far"}
        self.assertTrue(any("position.x" in i for i in validate_plate(p)))


class BuildingTests(unittest.TestCase):
    def test_valid_building(self):
        plates = [_vertex_plate(), _parametric_plate(), _mixed_plate()]
        self.assertEqual(validate_building(plates), [])
        self.assertTrue(is_valid_building(plates))

    def test_empty_building(self):
        self.assertTrue(any("at least one" in i for i in validate_building([])))

    def test_not_a_list(self):
        self.assertEqual(validate_building({}), ["building: must be a list of plate dicts"])

    def test_duplicate_names(self):
        a = _vertex_plate()
        b = _parametric_plate()
        b["name"] = "base"
        issues = validate_building([a, b])
        self.assertTrue(any("duplicate name 'base'" in i for i in issues))

    def test_issue_paths_prefixed(self):
        bad = {"name": "x", "category": "vertex", "thickness": 1, "vertices": [(0, 0)]}
        issues = validate_building([_vertex_plate(), bad])
        self.assertTrue(any(i.startswith("plate[1]") for i in issues))


class NormalizeTests(unittest.TestCase):
    def test_fills_defaults(self):
        out = normalize_plate(_vertex_plate())
        self.assertEqual(out["rotation"], 0.0)
        self.assertEqual(out["position"], {"x": 0.0, "y": 0.0, "z": 0.0})

    def test_preserves_given_position(self):
        p = _vertex_plate()
        p["position"] = {"x": 5.0}
        out = normalize_plate(p)
        self.assertEqual(out["position"]["x"], 5.0)
        self.assertEqual(out["position"]["z"], 0.0)

    def test_does_not_mutate_input(self):
        p = _vertex_plate()
        normalize_plate(p)
        self.assertNotIn("rotation", p)

    def test_raises_on_invalid(self):
        with self.assertRaises(PlateSpecError):
            normalize_plate({"name": "", "category": "vertex", "thickness": 1})


if __name__ == "__main__":
    unittest.main()
