import unittest

from harnesscad.agents.generation.query2cad_feedback import (
    resolve_caption, build_difference_feedback, CorrectiveFeedback,
    assemble_feedback, SOURCE_CAPTION, SOURCE_HUMAN,
)


class TestResolveCaption(unittest.TestCase):
    def test_auto_when_no_human(self):
        r = resolve_caption("a pentagon plate")
        self.assertEqual(r["source"], SOURCE_CAPTION)
        self.assertFalse(r["human_intervened"])
        self.assertEqual(r["caption"], "a pentagon plate")

    def test_human_override(self):
        r = resolve_caption("a pentagon plate", "a five-pointed star plate")
        self.assertEqual(r["source"], SOURCE_HUMAN)
        self.assertTrue(r["human_intervened"])
        self.assertEqual(r["caption"], "a five-pointed star plate")

    def test_blank_human_falls_back_to_auto(self):
        r = resolve_caption("a torus", "   ")
        self.assertEqual(r["source"], SOURCE_CAPTION)

    def test_human_rescues_empty_auto(self):
        r = resolve_caption("", "a bookshelf")
        self.assertEqual(r["source"], SOURCE_HUMAN)
        self.assertEqual(r["caption"], "a bookshelf")

    def test_no_caption_at_all(self):
        with self.assertRaises(ValueError):
            resolve_caption("   ", None)


class TestDifferenceFeedback(unittest.TestCase):
    def test_contains_both(self):
        fb = build_difference_feedback("a star plate", "a pentagon plate")
        self.assertIn("a star plate", fb)
        self.assertIn("a pentagon plate", fb)
        self.assertIn("difference", fb.lower())

    def test_empty_query(self):
        with self.assertRaises(ValueError):
            build_difference_feedback("  ", "a shape")

    def test_empty_caption(self):
        with self.assertRaises(ValueError):
            build_difference_feedback("a shape", "  ")


class TestCorrectiveFeedback(unittest.TestCase):
    def test_render_with_steps(self):
        cf = CorrectiveFeedback("an open pentagon outline",
                                ["close the pentagon", "extrude to add thickness"])
        out = cf.render()
        self.assertIn("looks like", out)
        self.assertIn("1. close the pentagon", out)
        self.assertIn("2. extrude to add thickness", out)
        self.assertTrue(cf.is_actionable)

    def test_render_without_steps(self):
        cf = CorrectiveFeedback("a torus")
        self.assertNotIn("Steps", cf.render())
        self.assertFalse(cf.is_actionable)

    def test_empty_looks_like(self):
        with self.assertRaises(ValueError):
            CorrectiveFeedback("  ")

    def test_empty_step_rejected(self):
        cf = CorrectiveFeedback("a shape", ["  "])
        with self.assertRaises(ValueError):
            cf.render()


class TestAssemble(unittest.TestCase):
    def test_full_pipeline_auto(self):
        r = assemble_feedback("a star plate", "a pentagon plate")
        self.assertEqual(r["source"], SOURCE_CAPTION)
        self.assertFalse(r["human_intervened"])
        self.assertFalse(r["actionable"])
        self.assertIn("a pentagon plate", r["difference_feedback"])

    def test_full_pipeline_human_and_steps(self):
        r = assemble_feedback(
            "a bookshelf", "a flat panel",
            human_caption="a single shelf plank",
            correction_steps=["add vertical sides", "add a second shelf"])
        self.assertTrue(r["human_intervened"])
        self.assertEqual(r["source"], SOURCE_HUMAN)
        self.assertTrue(r["actionable"])
        self.assertIn("a single shelf plank", r["difference_feedback"])
        self.assertIn("add vertical sides", r["corrective_feedback"])

    def test_deterministic(self):
        a = assemble_feedback("q", "c", correction_steps=["s1"])
        b = assemble_feedback("q", "c", correction_steps=["s1"])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
