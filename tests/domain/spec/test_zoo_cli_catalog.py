"""Tests for domain.spec.zoo_cli_catalog (Zoo CLI command surface)."""

import unittest

from harnesscad.domain.spec.zoo_cli_catalog import (
    COMMANDS,
    GEOMETRY_QUERIES,
    ML_SUBCOMMANDS,
    command_path_exists,
    geometry_query_commands,
    is_geometry_query,
    subcommands,
    top_level_commands,
)


class TestTree(unittest.TestCase):
    def test_top_level_has_core_verbs(self):
        tl = top_level_commands()
        for verb in ("kcl", "file", "ml", "auth", "config"):
            self.assertIn(verb, tl)

    def test_kcl_subcommands(self):
        subs = subcommands("kcl")
        self.assertIn("export", subs)
        self.assertIn("format", subs)
        self.assertIn("bounding-box", subs)

    def test_file_subcommands(self):
        subs = subcommands("file")
        self.assertIn("convert", subs)
        self.assertNotIn("bounding-box", subs)  # file has no bounding-box

    def test_leaf_verb_empty(self):
        self.assertEqual(subcommands("version"), ())

    def test_unknown_command(self):
        self.assertEqual(subcommands("nope"), ())

    def test_ml_nested(self):
        self.assertIn("text-to-cad", COMMANDS["ml"])
        self.assertEqual(ML_SUBCOMMANDS["text-to-cad"], ("export", "snapshot", "view"))
        self.assertEqual(ML_SUBCOMMANDS["kcl"], ("edit", "copilot"))


class TestGeometryQueries(unittest.TestCase):
    def test_is_geometry_query(self):
        self.assertTrue(is_geometry_query("volume"))
        self.assertTrue(is_geometry_query("center-of-mass"))
        self.assertFalse(is_geometry_query("export"))

    def test_geometry_query_commands(self):
        pairs = geometry_query_commands()
        self.assertIn(("kcl", "volume"), pairs)
        self.assertIn(("file", "mass"), pairs)
        # bounding-box only exists under kcl
        self.assertIn(("kcl", "bounding-box"), pairs)
        self.assertNotIn(("file", "bounding-box"), pairs)

    def test_all_queries_present_under_kcl(self):
        kcl_subs = subcommands("kcl")
        for q in GEOMETRY_QUERIES:
            self.assertIn(q, kcl_subs)


class TestPathValidation(unittest.TestCase):
    def test_valid_paths(self):
        self.assertTrue(command_path_exists("kcl"))
        self.assertTrue(command_path_exists("kcl", "volume"))
        self.assertTrue(command_path_exists("file", "convert"))

    def test_invalid_paths(self):
        self.assertFalse(command_path_exists("kcl", "nope"))
        self.assertFalse(command_path_exists("nope"))
        self.assertFalse(command_path_exists("nope", "volume"))


if __name__ == "__main__":
    unittest.main()
