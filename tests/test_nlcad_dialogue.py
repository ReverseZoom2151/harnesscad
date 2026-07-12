import unittest

from spec.nlcad_dialogue import (
    EntityRegistry, resolve_reference, extract_fragment, DialogueState,
)


class TestReferenceResolution(unittest.TestCase):
    def setUp(self):
        self.reg = EntityRegistry()
        self.c1 = self.reg.add("circle", ("big",))
        self.r1 = self.reg.add("rectangle", ())
        self.c2 = self.reg.add("circle", ("small",))

    def test_anaphora_it(self):
        r = resolve_reference("it", self.reg)
        self.assertTrue(r.resolved)
        self.assertEqual(r.entity, self.c2)

    def test_the_last_one(self):
        r = resolve_reference("the last one", self.reg)
        self.assertEqual(r.entity, self.c2)

    def test_the_first_one(self):
        r = resolve_reference("the first one", self.reg)
        self.assertEqual(r.entity, self.c1)

    def test_big_circle_unique(self):
        r = resolve_reference("the big circle", self.reg)
        self.assertTrue(r.resolved)
        self.assertEqual(r.entity, self.c1)

    def test_the_rectangle_unique_by_type(self):
        r = resolve_reference("the rectangle", self.reg)
        self.assertEqual(r.entity, self.r1)

    def test_ambiguous_circle(self):
        r = resolve_reference("the circle", self.reg)
        self.assertFalse(r.resolved)
        self.assertTrue(r.ambiguous)
        self.assertEqual(set(r.candidates), {self.c1, self.c2})

    def test_last_circle_resolves_ambiguity(self):
        r = resolve_reference("the last circle", self.reg)
        self.assertEqual(r.entity, self.c2)

    def test_no_entities(self):
        r = resolve_reference("it", EntityRegistry())
        self.assertFalse(r.resolved)
        self.assertEqual(r.reason, "no-entities")

    def test_no_match(self):
        r = resolve_reference("the sphere", self.reg)
        self.assertFalse(r.resolved)
        self.assertEqual(r.reason, "no-match")


class TestExtractFragment(unittest.TestCase):
    def test_location_fragment(self):
        f = extract_fragment("at (20, 0)")
        self.assertEqual(f.location, (20.0, 0.0))

    def test_dimension_fragment(self):
        f = extract_fragment("radius 8")
        self.assertEqual(f.dimensions, {"radius": 8.0})

    def test_target_fragment(self):
        f = extract_fragment("to (5, 5)")
        self.assertEqual(f.target, (5.0, 5.0))

    def test_empty(self):
        self.assertTrue(extract_fragment("the and of").empty)


class TestEllipsis(unittest.TestCase):
    def test_full_then_location_ellipsis(self):
        d = DialogueState()
        first = d.interpret("draw a circle of radius 5 at (0, 0)")
        self.assertEqual(first.location, (0.0, 0.0))
        # elliptical follow-up: only the location changes
        second = d.interpret("at (20, 0)")
        self.assertEqual(second.action, "create")
        self.assertEqual(second.obj, "circle")
        self.assertEqual(second.dimensions, {"radius": 5.0})   # inherited
        self.assertEqual(second.location, (20.0, 0.0))         # overridden

    def test_dimension_ellipsis_overrides(self):
        d = DialogueState()
        d.interpret("draw a circle of radius 5 at (0, 0)")
        third = d.interpret("radius 9")
        self.assertEqual(third.dimensions, {"radius": 9.0})
        self.assertEqual(third.location, (0.0, 0.0))           # inherited

    def test_ellipsis_without_prior_returns_none(self):
        d = DialogueState()
        self.assertIsNone(d.interpret("at (1, 1)"))

    def test_created_entity_registered(self):
        d = DialogueState()
        d.interpret("draw a big circle of radius 5")
        self.assertEqual(len(d.registry), 1)
        ref = resolve_reference("the big circle", d.registry)
        self.assertTrue(ref.resolved)

    def test_history_grows(self):
        d = DialogueState()
        d.interpret("draw a circle of radius 5 at (0,0)")
        d.interpret("at (1, 1)")
        self.assertEqual(len(d.history), 2)

    def test_deterministic(self):
        d1, d2 = DialogueState(), DialogueState()
        d1.interpret("draw a square of side 4 at (0,0)")
        d2.interpret("draw a square of side 4 at (0,0)")
        self.assertEqual(d1.interpret("at (2,2)").to_dict(),
                         d2.interpret("at (2,2)").to_dict())


if __name__ == "__main__":
    unittest.main()
