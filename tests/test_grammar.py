"""Tests for the grammar-constrained op-decoding layer (grammar.py).

Everything is derived from ``cisp.ops._REGISTRY`` so these tests avoid hardcoding
the op list: they iterate the live registry and assert coverage, and one test
registers a brand-new op tag to prove the schema/grammar auto-cover it.
"""

import json
import unittest
from dataclasses import dataclass
from typing import ClassVar, Optional

import grammar
from cisp.ops import Op, _REGISTRY, CONSTRAINT_DOF

REGISTRY_TAGS = set(_REGISTRY)


class TestJsonSchema(unittest.TestCase):
    def setUp(self):
        self.schema = grammar.op_json_schema()
        self.branches = {b["properties"]["op"]["const"]: b
                         for b in self.schema["oneOf"]}

    def test_one_branch_per_registered_op(self):
        # Derived from the registry, not a hardcoded list.
        self.assertEqual(set(self.branches), REGISTRY_TAGS)
        self.assertEqual(len(self.schema["oneOf"]), len(_REGISTRY))

    def test_discriminated_union_shape(self):
        self.assertEqual(self.schema["discriminator"]["propertyName"], "op")
        for tag, branch in self.branches.items():
            self.assertEqual(branch["type"], "object")
            self.assertFalse(branch["additionalProperties"])
            self.assertIn("op", branch["required"])
            self.assertEqual(branch["properties"]["op"]["const"], tag)

    def test_field_types(self):
        # extrude.distance is a number; new_sketch.plane is a string.
        ex = self.branches["extrude"]["properties"]
        self.assertEqual(ex["distance"]["type"], "number")
        self.assertEqual(ex["sketch"]["type"], "string")
        # fillet.edges is a string array.
        fe = self.branches["fillet"]["properties"]["edges"]
        self.assertEqual(fe["type"], "array")
        self.assertEqual(fe["items"]["type"], "string")

    def test_optional_fields_are_nullable_and_not_required(self):
        con = self.branches["constrain"]
        props = con["properties"]
        # Constrain.b and .value are Optional -> nullable, not required.
        self.assertEqual(props["b"]["type"], ["string", "null"])
        self.assertEqual(props["value"]["type"], ["number", "null"])
        self.assertNotIn("b", con["required"])
        self.assertNotIn("value", con["required"])
        # kind and a are required.
        self.assertIn("kind", con["required"])
        self.assertIn("a", con["required"])

    def test_enums_present(self):
        plane = self.branches["new_sketch"]["properties"]["plane"]
        self.assertEqual(set(plane["enum"]), {"XY", "YZ", "XZ"})
        bkind = self.branches["boolean"]["properties"]["kind"]
        self.assertEqual(set(bkind["enum"]), {"union", "cut", "intersect"})
        ckind = self.branches["constrain"]["properties"]["kind"]
        self.assertEqual(set(ckind["enum"]), set(CONSTRAINT_DOF))


class TestGbnfGrammar(unittest.TestCase):
    def test_non_empty(self):
        g = grammar.op_grammar()
        self.assertTrue(g.strip())
        self.assertIn("root ::=", g)

    def test_mentions_every_op_tag(self):
        g = grammar.op_grammar()
        for tag in _REGISTRY:
            self.assertIn('"\\"%s\\""' % tag, g,
                          f"grammar does not fix op tag {tag!r}")

    def test_mentions_enum_literals(self):
        g = grammar.op_grammar()
        for lit in ("XY", "YZ", "XZ", "union", "cut", "intersect"):
            self.assertIn(lit, g)
        for lit in CONSTRAINT_DOF:
            self.assertIn(lit, g)

    def test_has_shared_json_rules(self):
        g = grammar.op_grammar()
        for rule in ("number", "string", "ws", "strarray"):
            self.assertIn("%s " % rule, g)


class TestGrammarConstraintAccepts(unittest.TestCase):
    def setUp(self):
        self.gc = grammar.GrammarConstraint()

    def test_accepts_valid_ops(self):
        valid = [
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_point", "sketch": "sk1", "x": 1.0, "y": 2.0},
            {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 20, "h": 10},
            {"op": "constrain", "kind": "distance", "a": "e1", "value": 20},
            {"op": "constrain", "kind": "coincident", "a": "e1"},  # optionals omitted
            {"op": "extrude", "sketch": "sk1", "distance": 5},
            {"op": "fillet", "edges": ["e1", "e2"], "radius": 0.5},
            {"op": "boolean", "kind": "cut", "target": "f1", "tool": "f2"},
        ]
        for op in valid:
            with self.subTest(op=op["op"]):
                errs = self.gc.validate(json.dumps(op))
                self.assertEqual(errs, [], f"{op} -> {[repr(e) for e in errs]}")
                self.assertTrue(self.gc.accepts(op))

    def test_check_returns_parsed_op(self):
        op = self.gc.check({"op": "extrude", "sketch": "sk1", "distance": 5})
        self.assertIsInstance(op, Op)
        self.assertEqual(op.OP, "extrude")

    def test_accepts_dict_and_string_equally(self):
        d = {"op": "new_sketch", "plane": "YZ"}
        self.assertTrue(self.gc.accepts(d))
        self.assertTrue(self.gc.accepts(json.dumps(d)))


class TestGrammarConstraintRejects(unittest.TestCase):
    def setUp(self):
        self.gc = grammar.GrammarConstraint()

    def _kinds(self, candidate):
        return {e.kind for e in self.gc.validate(candidate)}

    def test_rejects_malformed_json(self):
        errs = self.gc.validate("{not json")
        self.assertTrue(errs)
        self.assertEqual(errs[0].kind, "json")

    def test_rejects_empty(self):
        self.assertEqual(self.gc.validate("")[0].kind, "json")

    def test_rejects_non_object(self):
        self.assertEqual(self.gc.validate("[1, 2, 3]")[0].kind, "structure")

    def test_rejects_missing_op_tag(self):
        self.assertEqual(self._kinds({"plane": "XY"}), {"required"})

    def test_rejects_unknown_tag(self):
        errs = self.gc.validate({"op": "teleport", "x": 1})
        self.assertEqual(errs[0].kind, "unknown_tag")

    def test_rejects_enum_violation(self):
        self.assertIn("enum", self._kinds({"op": "new_sketch", "plane": "diagonal"}))
        self.assertIn("enum", self._kinds(
            {"op": "boolean", "kind": "merge", "target": "f1", "tool": "f2"}))
        self.assertIn("enum", self._kinds(
            {"op": "constrain", "kind": "tangentish", "a": "e1"}))

    def test_rejects_wrong_type(self):
        self.assertIn("type", self._kinds(
            {"op": "extrude", "sketch": "sk1", "distance": "far"}))
        # bool is not a number.
        self.assertIn("type", self._kinds(
            {"op": "extrude", "sketch": "sk1", "distance": True}))
        # edges must be a string array.
        self.assertIn("type", self._kinds(
            {"op": "fillet", "edges": [1, 2], "radius": 0.5}))

    def test_rejects_missing_required_field(self):
        self.assertIn("required", self._kinds({"op": "extrude", "sketch": "sk1"}))

    def test_rejects_unexpected_field(self):
        self.assertIn("additional", self._kinds(
            {"op": "new_sketch", "plane": "XY", "bogus": 1}))

    def test_errors_are_typed_grammar_errors(self):
        for e in self.gc.validate({"op": "extrude", "sketch": "sk1", "distance": "x"}):
            self.assertIsInstance(e, grammar.GrammarError)
            self.assertTrue(e.kind)


class TestStateHook(unittest.TestCase):
    def test_no_extrude_before_sketch(self):
        allowed = grammar.allowed_ops_for_state(has_sketch=False)
        self.assertIn("new_sketch", allowed)
        self.assertNotIn("extrude", allowed)
        self.assertNotIn("add_line", allowed)

    def test_sketch_unlocks_geometry_and_features(self):
        allowed = grammar.allowed_ops_for_state(has_sketch=True)
        for tag in ("add_line", "add_circle", "constrain", "extrude", "fillet"):
            self.assertIn(tag, allowed)

    def test_boolean_needs_solid(self):
        self.assertNotIn("boolean", grammar.allowed_ops_for_state(has_sketch=True))
        self.assertIn("boolean",
                      grammar.allowed_ops_for_state(has_sketch=True, has_solid=True))

    def test_constraint_for_state_rejects_disallowed_op(self):
        gc = grammar.constraint_for_state(has_sketch=False)
        errs = gc.validate({"op": "extrude", "sketch": "sk1", "distance": 5})
        self.assertEqual(errs[0].kind, "not_allowed")

    def test_grammar_for_state_shrinks_root(self):
        g = grammar.grammar_for_state(has_sketch=False)
        self.assertIn("new_sketch", g)
        self.assertNotIn(grammar._gbnf_rule_name("extrude") + " ::=", g)
        sch = grammar.grammar_for_state(has_sketch=False, as_schema=True)
        tags = {b["properties"]["op"]["const"] for b in sch["oneOf"]}
        self.assertNotIn("extrude", tags)


class TestAutoCoversNewOp(unittest.TestCase):
    """The schema/grammar must auto-cover a newly registered op (no hardcoding)."""

    def setUp(self):
        # A hypothetical op tag NOT in the real registry, to prove the
        # schema/grammar auto-cover a newly registered op (chamfer is now a
        # real op, so this fixture uses an unregistered fictional tag).
        @dataclass(frozen=True)
        class Groove(Op):
            OP: ClassVar[str] = "groove"
            edges: tuple = ()
            distance: float = 1.0
            label: Optional[str] = None

        self.tag = "groove"
        self.cls = Groove
        self._added = self.tag not in _REGISTRY
        if self._added:
            _REGISTRY[self.tag] = Groove

    def tearDown(self):
        if self._added:
            _REGISTRY.pop(self.tag, None)

    def test_schema_covers_new_op(self):
        schema = grammar.op_json_schema()
        tags = {b["properties"]["op"]["const"] for b in schema["oneOf"]}
        self.assertIn(self.tag, tags)
        branch = next(b for b in schema["oneOf"]
                      if b["properties"]["op"]["const"] == self.tag)
        self.assertEqual(branch["properties"]["distance"]["type"], "number")
        self.assertEqual(branch["properties"]["edges"]["type"], "array")
        # Optional field nullable + not required.
        self.assertEqual(branch["properties"]["label"]["type"], ["string", "null"])
        self.assertNotIn("label", branch["required"])

    def test_grammar_covers_new_op(self):
        g = grammar.op_grammar()
        self.assertIn('"\\"%s\\""' % self.tag, g)

    def test_constraint_accepts_new_op(self):
        gc = grammar.GrammarConstraint()
        self.assertTrue(gc.accepts(
            {"op": self.tag, "edges": ["e1"], "distance": 2.0}))


if __name__ == "__main__":
    unittest.main()
