"""Tests for the FreeCAD workbench operation catalogue."""

import unittest

from adapters.fcai_freecad_tool_catalog import (
    CallCheck,
    FreeCadToolCatalog,
    Operation,
    ParamSpec,
    WORKBENCHES,
    default_catalog,
)


class CatalogShapeTest(unittest.TestCase):
    def setUp(self):
        self.cat = default_catalog()

    def test_operation_count(self):
        self.assertEqual(len(self.cat), 53)

    def test_default_is_shared(self):
        self.assertIs(default_catalog(), default_catalog())

    def test_every_operation_well_formed(self):
        for name in self.cat.names():
            op = self.cat.get(name)
            self.assertIsInstance(op, Operation)
            self.assertIn(op.workbench, WORKBENCHES)
            self.assertIn(op.category,
                          {"modeling", "query", "file", "general",
                           "interactive", "view"})
            for p in op.params:
                self.assertIsInstance(p, ParamSpec)
                self.assertIn(p.type,
                              {"string", "number", "integer", "boolean",
                               "array", "object"})

    def test_names_sorted(self):
        names = self.cat.names()
        self.assertEqual(names, sorted(names))

    def test_known_operations_present(self):
        for name in ("create_body", "pad_sketch", "pocket_sketch",
                     "revolve_sketch", "boolean_operation", "set_expression",
                     "fillet_edges", "linear_pattern", "create_sketch"):
            self.assertIn(name, self.cat)


class WorkbenchGroupingTest(unittest.TestCase):
    def setUp(self):
        self.cat = default_catalog()

    def test_histogram_sums_to_total(self):
        hist = self.cat.workbench_histogram()
        self.assertEqual(sum(hist.values()), len(self.cat))

    def test_partdesign_is_largest(self):
        hist = self.cat.workbench_histogram()
        self.assertEqual(max(hist, key=hist.get), "PartDesign")
        self.assertEqual(hist["PartDesign"], 16)

    def test_by_workbench(self):
        part = self.cat.by_workbench("Part")
        self.assertTrue(all(o.workbench == "Part" for o in part))
        self.assertIn("boolean_operation", [o.name for o in part])

    def test_by_category(self):
        queries = self.cat.by_category("query")
        self.assertTrue(all(o.category == "query" for o in queries))
        self.assertIn("describe_model", [o.name for o in queries])


class RequiredParamTest(unittest.TestCase):
    def setUp(self):
        self.cat = default_catalog()

    def test_required_params_of_boolean(self):
        op = self.cat.get("boolean_operation")
        self.assertEqual(set(op.required_params()),
                         {"operation", "object1", "object2"})

    def test_optional_params_excluded(self):
        op = self.cat.get("boolean_operation")
        self.assertNotIn("label", op.required_params())

    def test_param_lookup(self):
        op = self.cat.get("create_primitive")
        spec = op.param("shape_type")
        self.assertTrue(spec.required)
        self.assertEqual(spec.enum,
                         ("box", "cylinder", "sphere", "cone", "torus"))
        self.assertIsNone(op.param("nonexistent"))


class CheckCallTest(unittest.TestCase):
    def setUp(self):
        self.cat = default_catalog()

    def test_valid_call(self):
        res = self.cat.check_call("pad_sketch", {"sketch_name": "Sketch"})
        self.assertTrue(res)
        self.assertIsInstance(res, CallCheck)
        self.assertEqual(res.errors, [])

    def test_unknown_operation(self):
        res = self.cat.check_call("extrude", {})
        self.assertFalse(res)
        self.assertTrue(res.errors)

    def test_unknown_operation_suggestion(self):
        # "create_boddy" -> "create_body"
        res = self.cat.check_call("create_boddy", {})
        self.assertFalse(res)
        self.assertEqual(res.suggestion, "create_body")

    def test_missing_required(self):
        res = self.cat.check_call("boolean_operation",
                                  {"operation": "fuse"})
        self.assertFalse(res)
        self.assertTrue(any("object1" in e for e in res.errors))
        self.assertTrue(any("object2" in e for e in res.errors))

    def test_enum_domain_violation(self):
        res = self.cat.check_call("boolean_operation",
                                  {"operation": "subtract",
                                   "object1": "A", "object2": "B"})
        self.assertFalse(res)
        self.assertTrue(any("not in" in e for e in res.errors))

    def test_valid_enum(self):
        res = self.cat.check_call("boolean_operation",
                                  {"operation": "cut",
                                   "object1": "A", "object2": "B"})
        self.assertTrue(res)

    def test_unknown_param_is_warning_not_error(self):
        res = self.cat.check_call(
            "pad_sketch", {"sketch_name": "S", "lenght": 10})
        self.assertTrue(res)  # still ok - warning only
        self.assertTrue(res.warnings)
        self.assertTrue(any("length" in w for w in res.warnings))


class JsonSchemaTest(unittest.TestCase):
    def setUp(self):
        self.cat = default_catalog()

    def test_schema_structure(self):
        schema = self.cat.to_json_schema("boolean_operation")
        self.assertEqual(schema["type"], "object")
        self.assertIn("operation", schema["properties"])
        self.assertIn("enum", schema["properties"]["operation"])
        self.assertEqual(set(schema["required"]),
                         {"operation", "object1", "object2"})

    def test_schema_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.cat.to_json_schema("nope")

    def test_custom_operations(self):
        op = Operation("x", "Part", "modeling",
                       (ParamSpec("a", "string", True),))
        cat = FreeCadToolCatalog({"x": op})
        self.assertEqual(len(cat), 1)
        self.assertTrue(cat.check_call("x", {"a": "v"}))


if __name__ == "__main__":
    unittest.main()
