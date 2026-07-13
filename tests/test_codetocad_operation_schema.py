import unittest

from harnesscad.domain.programs.validate.codetocad_operation_schema import (
    CATEGORIES,
    OPERATIONS,
    Call,
    describe_operation,
    entity_kinds,
    operation_names,
    validate_call,
    validate_program,
)


def valid_program():
    return [
        Call("rectangle", {"center": (0, 0, 0), "width": "40mm", "height": "20mm"}, "base"),
        Call("extrude", {"profile": "base", "height": "5mm"}, "plate"),
        Call("circle", {"center": (0, 0, 0), "radius": "3mm"}, "pin_profile"),
        Call("extrude", {"profile": "pin_profile", "height": "10mm"}, "pin"),
        Call("union", {"this": "plate", "that": "pin"}, "part"),
        Call("fillet", {"solid": "part", "radius": "1mm", "at": "top_front_left"}, "rounded"),
        Call("landmark", {"of": "rounded", "at": "top_center", "offset_z": "2mm"}, "mount"),
    ]


class TestVocabulary(unittest.TestCase):
    def test_categories(self):
        for category in ("draw", "solid", "boolean", "feature", "transform", "joint"):
            self.assertIn(category, CATEGORIES)

    def test_operation_names(self):
        names = operation_names()
        for expected in ("extrude", "revolve", "loft", "sweep", "union", "subtract",
                         "fillet", "chamfer", "mirror", "revolute", "prismatic"):
            self.assertIn(expected, names)
        self.assertEqual(names, sorted(names))

    def test_names_by_category(self):
        self.assertEqual(
            operation_names("boolean"), ["concat", "intersection", "subtract", "union"]
        )

    def test_describe(self):
        text = describe_operation("extrude")
        self.assertIn("extrude(", text)
        self.assertIn("profile: entity", text)
        self.assertIn("[draft_angle: angle]", text)
        self.assertTrue(text.endswith("-> solid"))

    def test_describe_unknown(self):
        with self.assertRaises(KeyError):
            describe_operation("teleport")

    def test_specs_are_self_consistent(self):
        for name, spec in OPERATIONS.items():
            self.assertEqual(name, spec.name)
            self.assertTrue(spec.params)
            seen = set()
            for param in spec.params:
                self.assertNotIn(param.name, seen)
                seen.add(param.name)
                if param.kind in ("entity", "entity_list"):
                    self.assertTrue(param.accepts)


class TestValidateCall(unittest.TestCase):
    def test_valid(self):
        call = Call("extrude", {"profile": "sketch1", "height": "5mm"}, "solid1")
        self.assertEqual(validate_call(call), [])

    def test_unknown_operation(self):
        errors = validate_call(Call("bevelify", {}, "x"))
        self.assertEqual(len(errors), 1)
        self.assertIn("unknown operation", errors[0])

    def test_missing_required(self):
        errors = validate_call(Call("extrude", {"profile": "s"}, "r"))
        self.assertTrue(any("missing required argument 'height'" in e for e in errors))

    def test_unknown_argument(self):
        errors = validate_call(Call("extrude", {"profile": "s", "height": "5mm", "colour": "red"}, "r"))
        self.assertTrue(any("unknown argument 'colour'" in e for e in errors))

    def test_bad_length(self):
        errors = validate_call(Call("extrude", {"profile": "s", "height": "5 bananas"}, "r"))
        self.assertTrue(any("not a valid length" in e for e in errors))

    def test_angle_in_length_slot(self):
        errors = validate_call(Call("extrude", {"profile": "s", "height": "90deg"}, "r"))
        self.assertTrue(any("not a valid length" in e for e in errors))

    def test_bad_angle(self):
        errors = validate_call(
            Call("revolve", {"profile": "s", "axis": "a", "angle": "5mm"}, "r")
        )
        self.assertTrue(any("not a valid angle" in e for e in errors))

    def test_bad_plane(self):
        errors = validate_call(Call("mirror", {"solid": "s", "plane": "zz"}, "r"))
        self.assertTrue(any("must be one of" in e for e in errors))

    def test_bad_cardinal(self):
        errors = validate_call(
            Call("fillet", {"solid": "s", "radius": "1mm", "at": "northwest"}, "r")
        )
        self.assertTrue(any("not a cardinal direction" in e for e in errors))

    def test_bad_point(self):
        errors = validate_call(Call("circle", {"center": (0, 0), "radius": "1mm"}, "r"))
        self.assertTrue(any("3-element point" in e for e in errors))

    def test_bad_bool_and_scalar(self):
        errors = validate_call(
            Call("loft", {"profile": "a", "to": "b", "merge": "yes"}, "r")
        )
        self.assertTrue(any("must be a bool" in e for e in errors))
        errors = validate_call(
            Call("linear_pattern", {"solid": "s", "count": "three", "spacing": "5mm"}, "r")
        )
        self.assertTrue(any("must be a number" in e for e in errors))

    def test_missing_result_name(self):
        errors = validate_call(Call("extrude", {"profile": "s", "height": "5mm"}))
        self.assertTrue(any("no result name" in e for e in errors))

    def test_result_on_void_operation(self):
        errors = validate_call(Call("translate", {"target": "s", "x": "5mm"}, "r"))
        self.assertTrue(any("produces nothing" in e for e in errors))

    def test_void_operation_is_valid(self):
        self.assertEqual(validate_call(Call("translate", {"target": "s", "x": "5mm"})), [])


class TestValidateProgram(unittest.TestCase):
    def test_valid_program(self):
        self.assertEqual(validate_program(valid_program()), [])

    def test_symbol_table(self):
        table = entity_kinds(valid_program())
        self.assertEqual(table["base"], "sketch")
        self.assertEqual(table["plate"], "solid")
        self.assertEqual(table["mount"], "landmark")

    def test_undefined_reference(self):
        errors = validate_program([Call("extrude", {"profile": "ghost", "height": "5mm"}, "s")])
        self.assertTrue(any("undefined entity 'ghost'" in e for e in errors))

    def test_forward_reference(self):
        program = [
            Call("extrude", {"profile": "base", "height": "5mm"}, "plate"),
            Call("rectangle", {"center": (0, 0, 0), "width": "1mm", "height": "1mm"}, "base"),
        ]
        errors = validate_program(program)
        self.assertTrue(any("forward reference" in e for e in errors))

    def test_kind_mismatch_extruding_a_solid(self):
        program = [
            Call("rectangle", {"center": (0, 0, 0), "width": "1mm", "height": "1mm"}, "base"),
            Call("extrude", {"profile": "base", "height": "5mm"}, "plate"),
            Call("extrude", {"profile": "plate", "height": "5mm"}, "bad"),
        ]
        errors = validate_program(program)
        self.assertTrue(
            any("expects sketch but 'plate' is a solid" in e for e in errors)
        )

    def test_kind_mismatch_filleting_a_sketch(self):
        program = [
            Call("circle", {"center": (0, 0, 0), "radius": "1mm"}, "c"),
            Call("fillet", {"solid": "c", "radius": "1mm"}, "f"),
        ]
        errors = validate_program(program)
        self.assertTrue(any("expects solid but 'c' is a sketch" in e for e in errors))

    def test_duplicate_result(self):
        program = [
            Call("circle", {"center": (0, 0, 0), "radius": "1mm"}, "c"),
            Call("circle", {"center": (0, 0, 0), "radius": "2mm"}, "c"),
        ]
        errors = validate_program(program)
        self.assertTrue(any("duplicate result name 'c'" in e for e in errors))

    def test_entity_list_argument(self):
        program = [
            Call("rectangle", {"center": (0, 0, 0), "width": "1mm", "height": "1mm"}, "outer"),
            Call("circle", {"center": (0, 0, 0), "radius": "1mm"}, "hole"),
            Call("extrude", {"profile": "outer", "height": "2mm", "subtract": ["hole"]}, "part"),
        ]
        self.assertEqual(validate_program(program), [])

    def test_entity_list_undefined_member(self):
        program = [
            Call("rectangle", {"center": (0, 0, 0), "width": "1mm", "height": "1mm"}, "outer"),
            Call("extrude", {"profile": "outer", "height": "2mm", "subtract": ["ghost"]}, "part"),
        ]
        errors = validate_program(program)
        self.assertTrue(any("undefined entity 'ghost'" in e for e in errors))

    def test_joint_requires_landmarks(self):
        program = [
            Call("rectangle", {"center": (0, 0, 0), "width": "1mm", "height": "1mm"}, "s"),
            Call("extrude", {"profile": "s", "height": "1mm"}, "a"),
            Call("rigid", {"this": "a", "at": "a"}),
        ]
        errors = validate_program(program)
        self.assertTrue(any("expects landmark but 'a' is a solid" in e for e in errors))

    def test_errors_are_deterministic(self):
        program = [Call("extrude", {"profile": "ghost"}, "s")]
        self.assertEqual(validate_program(program), validate_program(program))

    def test_empty_program(self):
        self.assertEqual(validate_program([]), [])


if __name__ == "__main__":
    unittest.main()
