"""Tests for agent.toolcad_tool_schema."""

import unittest

from harnesscad.agents.agent.toolcad_tool_schema import (
    ArgSpec,
    InterfaceResult,
    ToolCall,
    ToolExecutionState,
    ToolLibrary,
    ToolSignature,
    default_toolcad_library,
)


class ArgSpecTest(unittest.TestCase):
    def test_type_checks(self):
        self.assertIsNone(ArgSpec("d", "number").check(3))
        self.assertIsNone(ArgSpec("d", "number").check(2.5))
        self.assertIsNotNone(ArgSpec("d", "number").check("x"))

    def test_bool_rejected_for_number(self):
        self.assertIsNotNone(ArgSpec("d", "number").check(True))

    def test_literal(self):
        spec = ArgSpec("op", "literal", choices=("cut", "fuse"))
        self.assertIsNone(spec.check("cut"))
        self.assertIsNotNone(spec.check("common"))

    def test_literal_requires_choices(self):
        with self.assertRaises(ValueError):
            ArgSpec("op", "literal")

    def test_list_accepts_tuple(self):
        self.assertIsNone(ArgSpec("e", "list").check((1, 2)))


class ToolSignatureTest(unittest.TestCase):
    def setUp(self):
        self.sig = ToolSignature(
            "extrude_face",
            "extrude",
            (ArgSpec("sketch_name", "str"), ArgSpec("distance", "number"),
             ArgSpec("name", "str", required=False)),
        )

    def test_required_args(self):
        self.assertEqual(self.sig.required_args, ("sketch_name", "distance"))

    def test_valid(self):
        self.assertEqual(self.sig.validate({"sketch_name": "s", "distance": 5}), ())

    def test_missing_required(self):
        errs = self.sig.validate({"sketch_name": "s"})
        self.assertTrue(any("distance" in e for e in errs))

    def test_unknown_arg(self):
        errs = self.sig.validate({"sketch_name": "s", "distance": 5, "foo": 1})
        self.assertTrue(any("foo" in e for e in errs))

    def test_wrong_type(self):
        errs = self.sig.validate({"sketch_name": "s", "distance": "big"})
        self.assertTrue(any("distance" in e for e in errs))

    def test_duplicate_arg_rejected(self):
        with self.assertRaises(ValueError):
            ToolSignature("t", "s", (ArgSpec("a", "str"), ArgSpec("a", "int")))


class InterfaceResultTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(InterfaceResult(True, "ok").label, "success")
        self.assertEqual(InterfaceResult(False, "no").label, "fail")


class ToolLibraryTest(unittest.TestCase):
    def setUp(self):
        self.lib = default_toolcad_library()

    def test_names(self):
        self.assertIn("boolean_operation", self.lib.names())
        self.assertIn("set_coord_system", self.lib.names())
        self.assertEqual(len(self.lib.names()), 6)

    def test_unknown_tool(self):
        errs = self.lib.validate_call(ToolCall("nope", {}))
        self.assertTrue(any("unknown tool" in e for e in errs))

    def test_duplicate_registration(self):
        with self.assertRaises(ValueError):
            ToolLibrary((
                ToolSignature("t", "s"),
                ToolSignature("t", "s"),
            ))

    def test_validate_boolean_literal(self):
        call = ToolCall("boolean_operation", {
            "base_object_name": "a", "tool_object_name": "b", "operation": "bad"})
        errs = self.lib.validate_call(call)
        self.assertTrue(any("operation" in e for e in errs))


class ToolExecutionStateTest(unittest.TestCase):
    def setUp(self):
        self.lib = default_toolcad_library()
        self.state = ToolExecutionState(self.lib)

    def test_create_and_extrude(self):
        r1 = self.state.execute(ToolCall("create_simple_sketch", {
            "profile": "rect", "sketch_name": "s1"}))
        self.assertTrue(r1.success)
        self.assertIn("s1", self.state.objects)
        r2 = self.state.execute(ToolCall("extrude_face", {
            "sketch_name": "s1", "distance": 10, "name": "p1"}))
        self.assertTrue(r2.success)
        self.assertEqual(r2.produced_object, "p1")

    def test_boolean_requires_existing_operands(self):
        r = self.state.execute(ToolCall("boolean_operation", {
            "base_object_name": "missing", "tool_object_name": "also_missing",
            "operation": "cut"}))
        self.assertFalse(r.success)
        self.assertIn("does not exist", r.description)

    def test_boolean_success_after_parts_exist(self):
        self.state.execute(ToolCall("extrude_face", {
            "sketch_name": "s1", "distance": 1, "name": "a"}))
        self.state.execute(ToolCall("extrude_face", {
            "sketch_name": "s2", "distance": 1, "name": "b"}))
        r = self.state.execute(ToolCall("boolean_operation", {
            "base_object_name": "a", "tool_object_name": "b",
            "operation": "fuse", "name": "c"}))
        self.assertTrue(r.success)
        self.assertIn("c", self.state.objects)

    def test_invalid_args_produce_fail(self):
        r = self.state.execute(ToolCall("extrude_face", {"sketch_name": "s1"}))
        self.assertFalse(r.success)

    def test_duplicate_name_fails(self):
        self.state.execute(ToolCall("extrude_face", {
            "sketch_name": "s1", "distance": 1, "name": "dup"}))
        r = self.state.execute(ToolCall("extrude_face", {
            "sketch_name": "s2", "distance": 1, "name": "dup"}))
        self.assertFalse(r.success)
        self.assertIn("already exists", r.description)

    def test_auto_naming_is_deterministic(self):
        r1 = self.state.execute(ToolCall("create_simple_sketch", {"profile": "a"}))
        r2 = self.state.execute(ToolCall("create_simple_sketch", {"profile": "b"}))
        self.assertEqual(r1.produced_object, "create_simple_sketch_1")
        self.assertEqual(r2.produced_object, "create_simple_sketch_2")


if __name__ == "__main__":
    unittest.main()
