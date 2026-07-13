"""Tests for programs.t2cdean_scad_extract."""

import unittest

from harnesscad.domain.programs.extract.openscad_extract import (
    CodeBlock,
    ScadExtractionError,
    extract_code_blocks,
    extract_scad,
    looks_like_scad,
    normalise_scad,
    scad_score,
    strip_prose_lines,
)

FENCED = """Sure! Here's the OpenSCAD code for a 10mm cube:

```scad
cube([10, 10, 10]);
```

This creates a cube with 10mm sides. Let me know if you want it hollow!
"""


class TestExtractCodeBlocks(unittest.TestCase):
    def test_single_tagged_block(self):
        blocks = extract_code_blocks(FENCED)
        self.assertEqual(blocks, [CodeBlock("scad", "cube([10, 10, 10]);")])

    def test_untagged_block(self):
        blocks = extract_code_blocks("```\nsphere(5);\n```")
        self.assertEqual(blocks[0].language, "")
        self.assertEqual(blocks[0].code, "sphere(5);")

    def test_multiple_blocks_in_order(self):
        text = "```python\nprint(1)\n```\ntext\n```openscad\ncube(1);\n```"
        blocks = extract_code_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].language, "python")
        self.assertEqual(blocks[1].language, "openscad")

    def test_unterminated_fence_is_recovered(self):
        text = "Here you go:\n```scad\ncube(10);\ntranslate([1,0,0])"
        blocks = extract_code_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("cube(10);", blocks[0].code)

    def test_tilde_fences(self):
        blocks = extract_code_blocks("~~~scad\ncube(2);\n~~~")
        self.assertEqual(blocks[0].code, "cube(2);")

    def test_no_fences(self):
        self.assertEqual(extract_code_blocks("just prose"), [])

    def test_empty_text(self):
        self.assertEqual(extract_code_blocks(""), [])


class TestLooksLikeScad(unittest.TestCase):
    def test_true_for_primitives(self):
        self.assertTrue(looks_like_scad("cube([1,2,3]);"))
        self.assertTrue(looks_like_scad("difference() { sphere(5); cube(2); }"))

    def test_true_for_user_module_call(self):
        self.assertTrue(looks_like_scad("my_bracket(width=10);"))

    def test_false_for_prose(self):
        self.assertFalse(looks_like_scad("This creates a cube with 10mm sides."))

    def test_false_for_python(self):
        self.assertFalse(looks_like_scad("import cadquery as cq\ndef make():\n    pass"))

    def test_false_for_empty(self):
        self.assertFalse(looks_like_scad(""))
        self.assertFalse(looks_like_scad("   \n "))


class TestScore(unittest.TestCase):
    def test_tagged_scad_outranks_untagged(self):
        tagged = CodeBlock("scad", "cube(1);")
        untagged = CodeBlock("", "cube(1);")
        self.assertGreater(scad_score(tagged), scad_score(untagged))

    def test_python_block_is_penalised(self):
        self.assertLess(scad_score(CodeBlock("python", "print(1)")), 0)


class TestStripProse(unittest.TestCase):
    def test_drops_english_keeps_code(self):
        text = "Here is the code:\ncube([2,2,2]);\nThis makes a small cube.\n"
        self.assertEqual(strip_prose_lines(text), "cube([2,2,2]);")

    def test_keeps_comments_and_braces(self):
        text = "Explanation follows.\n// a plate\ndifference() {\n  cube(10);\n}\nEnjoy!"
        out = strip_prose_lines(text)
        self.assertIn("// a plate", out)
        self.assertIn("difference() {", out)
        self.assertIn("}", out)
        self.assertNotIn("Enjoy", out)

    def test_keeps_assignments_and_special_vars(self):
        out = strip_prose_lines("Sure.\n$fn = 64;\nwidth = 10;\ncylinder(h=5, r=2);")
        self.assertEqual(out, "$fn = 64;\nwidth = 10;\ncylinder(h=5, r=2);")

    def test_all_prose_yields_empty(self):
        self.assertEqual(strip_prose_lines("I cannot help with that request."), "")


class TestExtractScad(unittest.TestCase):
    def test_fenced_reply(self):
        self.assertEqual(extract_scad(FENCED), "cube([10, 10, 10]);")

    def test_prefers_scad_block_over_python_block(self):
        text = "```python\nprint('hi')\n```\n```scad\nsphere(4);\n```"
        self.assertEqual(extract_scad(text), "sphere(4);")

    def test_unfenced_reply_falls_back_to_prose_stripping(self):
        text = "Certainly! Here it is:\ncylinder(h=10, r=3);\nHope this helps."
        self.assertEqual(extract_scad(text), "cylinder(h=10, r=3);")

    def test_bare_code_passes_through(self):
        self.assertEqual(extract_scad("cube(5);"), "cube(5);")

    def test_refusal_raises(self):
        with self.assertRaises(ScadExtractionError):
            extract_scad("I'm sorry, I can't do that.")

    def test_strict_rejects_non_scad_block(self):
        text = "```\nLorem ipsum dolor sit amet\n```"
        with self.assertRaises(ScadExtractionError):
            extract_scad(text, strict=True)

    def test_strict_accepts_real_scad(self):
        self.assertEqual(extract_scad(FENCED, strict=True), "cube([10, 10, 10]);")

    def test_empty_reply_raises(self):
        with self.assertRaises(ScadExtractionError):
            extract_scad("")

    def test_deterministic(self):
        self.assertEqual(extract_scad(FENCED), extract_scad(FENCED))


class TestNormalise(unittest.TestCase):
    def test_crlf_and_trailing_space(self):
        self.assertEqual(normalise_scad("cube(1);  \r\nsphere(2);\r\n"), "cube(1);\nsphere(2);\n")

    def test_blank_edges_trimmed_and_final_newline(self):
        self.assertEqual(normalise_scad("\n\ncube(1);\n\n\n"), "cube(1);\n")

    def test_empty_stays_empty(self):
        self.assertEqual(normalise_scad("   \n\n"), "")

    def test_idempotent(self):
        once = normalise_scad("cube(1);\r\n")
        self.assertEqual(normalise_scad(once), once)


if __name__ == "__main__":
    unittest.main()
