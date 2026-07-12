"""Tests for fabrication.t2cdean_openscad_export."""

import unittest

from fabrication.t2cdean_openscad_export import (
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_WRONG_DIMENSION,
    ExportPlan,
    OpenScadExportError,
    artifact_name,
    check_output_extension,
    classify_result,
    define_args,
    extension_for,
    format_dimension,
    is_success,
    plan_cache_key,
    plan_export,
    scad_literal,
    sorted_formats,
    source_digest,
    summarize,
    warnings_only,
)

CUBE = "cube([10,10,10]);"


class TestFormats(unittest.TestCase):
    def test_extension_lookup(self):
        self.assertEqual(extension_for("stl"), ".stl")
        self.assertEqual(extension_for("binstl"), ".stl")
        self.assertEqual(extension_for("SVG"), ".svg")

    def test_unknown_format(self):
        with self.assertRaises(OpenScadExportError):
            extension_for("step")

    def test_dimension_classification(self):
        self.assertEqual(format_dimension("stl"), "3d")
        self.assertEqual(format_dimension("dxf"), "2d")
        self.assertEqual(format_dimension("png"), "other")

    def test_sorted_formats(self):
        self.assertEqual(sorted_formats("2d"), ["dxf", "pdf", "svg"])
        with self.assertRaises(OpenScadExportError):
            sorted_formats("4d")


class TestContentAddressing(unittest.TestCase):
    def test_digest_is_stable(self):
        self.assertEqual(source_digest(CUBE), source_digest(CUBE))

    def test_line_endings_normalised(self):
        self.assertEqual(source_digest("a();\nb();"), source_digest("a();\r\nb();"))

    def test_artifact_name_is_a_pure_function_of_source(self):
        self.assertEqual(artifact_name(CUBE), artifact_name(CUBE))
        self.assertNotEqual(artifact_name(CUBE), artifact_name("sphere(5);"))

    def test_cache_key_depends_on_defines_and_format(self):
        base = plan_cache_key(CUBE, "stl")
        self.assertEqual(base, plan_cache_key(CUBE, "stl"))
        self.assertNotEqual(base, plan_cache_key(CUBE, "off"))
        self.assertNotEqual(base, plan_cache_key(CUBE, "stl", {"size": 2}))

    def test_cache_key_ignores_define_ordering(self):
        a = plan_cache_key(CUBE, "stl", {"a": 1, "b": 2})
        b = plan_cache_key(CUBE, "stl", {"b": 2, "a": 1})
        self.assertEqual(a, b)


class TestScadLiteral(unittest.TestCase):
    def test_bool_is_lowercase(self):
        self.assertEqual(scad_literal(True), "true")
        self.assertEqual(scad_literal(False), "false")

    def test_numbers(self):
        self.assertEqual(scad_literal(3), "3")
        self.assertEqual(scad_literal(2.5), "2.5")

    def test_string_quoted_and_escaped(self):
        self.assertEqual(scad_literal('a"b'), '"a\\"b"')
        self.assertEqual(scad_literal("c:\\x"), '"c:\\\\x"')

    def test_vector(self):
        self.assertEqual(scad_literal([1, 2, 3]), "[1,2,3]")
        self.assertEqual(scad_literal((True, "x")), '[true,"x"]')

    def test_unsupported_type(self):
        with self.assertRaises(OpenScadExportError):
            scad_literal({"a": 1})

    def test_define_args_sorted(self):
        self.assertEqual(
            define_args({"b": 2, "a": 1}), ["-D", "a=1", "-D", "b=2"]
        )

    def test_define_args_reject_bad_identifier(self):
        with self.assertRaises(OpenScadExportError):
            define_args({"not a name": 1})


class TestPlanExport(unittest.TestCase):
    def test_argv_shape(self):
        plan = plan_export(CUBE, "stl", out_dir="build")
        stem = artifact_name(CUBE)
        self.assertEqual(
            plan.argv,
            [
                "openscad",
                "-o",
                "build/%s.stl" % stem,
                "--export-format",
                "stl",
                "build/%s.scad" % stem,
            ],
        )
        self.assertEqual(plan.output_path, "build/%s.stl" % stem)
        self.assertEqual(plan.scad_path, "build/%s.scad" % stem)

    def test_plan_is_idempotent(self):
        self.assertEqual(plan_export(CUBE), plan_export(CUBE))
        self.assertIsInstance(plan_export(CUBE), ExportPlan)

    def test_defines_appear_before_input_file(self):
        plan = plan_export(CUBE, "stl", defines={"size": 10})
        self.assertEqual(plan.argv[-3:-1], ["-D", "size=10"])
        self.assertTrue(plan.argv[-1].endswith(".scad"))

    def test_custom_executable(self):
        plan = plan_export(CUBE, executable="/usr/bin/openscad")
        self.assertEqual(plan.argv[0], "/usr/bin/openscad")

    def test_empty_source_rejected(self):
        with self.assertRaises(OpenScadExportError):
            plan_export("   \n")

    def test_unknown_format_rejected(self):
        with self.assertRaises(OpenScadExportError):
            plan_export(CUBE, "step")

    def test_output_extension_guard(self):
        check_output_extension("out.stl", "binstl")
        with self.assertRaises(OpenScadExportError):
            check_output_extension("out.stl", "svg")


class TestClassifyResult(unittest.TestCase):
    def test_clean_run_is_ok(self):
        self.assertEqual(classify_result(0, ""), (STATUS_OK, []))
        self.assertTrue(is_success(0, ""))

    def test_empty_geometry_despite_zero_exit(self):
        stderr = "Current top level object is empty.\n"
        status, _ = classify_result(0, stderr)
        self.assertEqual(status, STATUS_EMPTY)
        self.assertFalse(is_success(0, stderr))

    def test_wrong_dimension(self):
        stderr = "ERROR: Current top level object is not a 2D object.\n"
        status, messages = classify_result(0, stderr)
        self.assertEqual(status, STATUS_WRONG_DIMENSION)
        self.assertEqual(len(messages), 1)

    def test_nonzero_exit_is_error(self):
        status, _ = classify_result(1, "")
        self.assertEqual(status, STATUS_ERROR)

    def test_error_line_with_zero_exit_is_error(self):
        status, messages = classify_result(0, "ERROR: Parser error in line 3\n")
        self.assertEqual(status, STATUS_ERROR)
        self.assertEqual(messages, ["ERROR: Parser error in line 3"])

    def test_warnings_do_not_fail_the_run(self):
        stderr = "WARNING: $fn is too small\n"
        self.assertEqual(classify_result(0, stderr)[0], STATUS_OK)
        self.assertTrue(is_success(0, stderr))
        self.assertEqual(warnings_only(stderr), ["WARNING: $fn is too small"])


class TestSummarize(unittest.TestCase):
    def test_summary_record(self):
        plan = plan_export(CUBE, "stl", out_dir="out")
        rec = summarize(plan, 0, "WARNING: slow\n")
        self.assertEqual(rec["status"], STATUS_OK)
        self.assertEqual(rec["format"], "stl")
        self.assertEqual(rec["returncode"], 0)
        self.assertEqual(rec["digest"], source_digest(CUBE))
        self.assertEqual(rec["warnings"], ["WARNING: slow"])
        self.assertTrue(rec["output"].startswith("out/"))

    def test_summary_is_deterministic(self):
        plan = plan_export(CUBE)
        self.assertEqual(summarize(plan, 0, ""), summarize(plan, 0, ""))


if __name__ == "__main__":
    unittest.main()
