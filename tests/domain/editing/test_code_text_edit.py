import unittest

from harnesscad.domain.editing.code_text_edit import (
    detect_eol,
    indent_lines,
    unindent_lines,
    toggle_comment_block,
    line_number_gutter_digits,
)


class TestDetectEol(unittest.TestCase):
    def test_crlf_wins(self):
        self.assertEqual(detect_eol(b"a\r\nb\nc"), "\r\n")

    def test_lone_cr(self):
        self.assertEqual(detect_eol(b"a\rb\rc"), "\r")

    def test_default_lf(self):
        self.assertEqual(detect_eol(b"a\nb\nc"), "\n")

    def test_no_newline_defaults_lf(self):
        self.assertEqual(detect_eol(b"abc"), "\n")

    def test_type_error(self):
        with self.assertRaises(TypeError):
            detect_eol("not bytes")


class TestIndent(unittest.TestCase):
    def test_indent_adds_four_spaces(self):
        self.assertEqual(indent_lines(["x = 1", "y = 2"]), ["    x = 1", "    y = 2"])

    def test_indent_skips_blank(self):
        self.assertEqual(indent_lines(["a", "", "b"]), ["    a", "", "    b"])

    def test_unindent_removes_one_level(self):
        self.assertEqual(unindent_lines(["        a", "    b"]), ["    a", "b"])

    def test_unindent_leaves_partial(self):
        self.assertEqual(unindent_lines(["  a", "b"]), ["  a", "b"])

    def test_indent_roundtrip(self):
        src = ["def f():", "    return 1"]
        self.assertEqual(unindent_lines(indent_lines(src)), src)

    def test_does_not_mutate_input(self):
        src = ["a"]
        indent_lines(src)
        self.assertEqual(src, ["a"])


class TestToggleComment(unittest.TestCase):
    def test_comment_single_line(self):
        self.assertEqual(toggle_comment_block(["x = 1"]), ["# x = 1"])

    def test_uncomment_single_line(self):
        self.assertEqual(toggle_comment_block(["# x = 1"]), ["x = 1"])

    def test_comment_at_indent_level(self):
        self.assertEqual(toggle_comment_block(["    x = 1"]), ["    # x = 1"])

    def test_uncomment_at_indent_level(self):
        self.assertEqual(toggle_comment_block(["    # x = 1"]), ["    x = 1"])

    def test_uniform_block_toggles_off(self):
        self.assertEqual(
            toggle_comment_block(["# a", "# b"]), ["a", "b"]
        )

    def test_uniform_block_toggles_on_shared_column(self):
        # left-most column is 0 (line "a"), so comment inserted at col 0 for both
        self.assertEqual(
            toggle_comment_block(["a", "  b"]), ["# a", "#   b"]
        )

    def test_mixed_block_comments_all_at_col0(self):
        self.assertEqual(
            toggle_comment_block(["# a", "b"]), ["# # a", "# b"]
        )

    def test_blank_lines_ignored(self):
        self.assertEqual(
            toggle_comment_block(["a", "", "b"]), ["# a", "", "# b"]
        )

    def test_all_blank_unchanged(self):
        self.assertEqual(toggle_comment_block(["", "  "]), ["", "  "])

    def test_uncomment_without_trailing_space(self):
        self.assertEqual(toggle_comment_block(["#x"]), ["x"])

    def test_roundtrip_uniform(self):
        src = ["    x = 1", "    y = 2"]
        once = toggle_comment_block(src)
        self.assertEqual(toggle_comment_block(once), src)

    def test_does_not_mutate_input(self):
        src = ["x"]
        toggle_comment_block(src)
        self.assertEqual(src, ["x"])


class TestGutterDigits(unittest.TestCase):
    def test_small(self):
        self.assertEqual(line_number_gutter_digits(1), 1)
        self.assertEqual(line_number_gutter_digits(9), 1)

    def test_boundaries(self):
        self.assertEqual(line_number_gutter_digits(10), 2)
        self.assertEqual(line_number_gutter_digits(99), 2)
        self.assertEqual(line_number_gutter_digits(100), 3)

    def test_zero_and_negative_reserve_one(self):
        self.assertEqual(line_number_gutter_digits(0), 1)
        self.assertEqual(line_number_gutter_digits(-5), 1)


if __name__ == "__main__":
    unittest.main()
