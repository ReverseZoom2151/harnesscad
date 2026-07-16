"""Tests for fabrication/printer_profiles.py."""

from __future__ import annotations

import copy
import unittest

from harnesscad.domain.fabrication import printer_profiles as pp
from harnesscad.domain.fabrication.printability_verdict import PrinterProfile, check_fit

BASE_PROFILE = {
    "backend": "orcaslicer",
    "native_config": "/profiles/machine.json",
    "machine": {"name": "Example 256", "bed_size_mm": [256, 256], "z_height_mm": 256},
    "filament": {"type": "PLA", "nozzle_temp_c": 210, "bed_temp_c": 60},
}

GOOD_GCODE = "\n".join(
    [
        "; header",
        "G21",
        "G90",
        "M104 S210",
        "M140 S60",
        "G28",
        "G1 X10 Y10 Z0.2 E1",
        "G1 X200 Y200 Z10 E5",
    ]
)


def profile(**overrides):
    data = copy.deepcopy(BASE_PROFILE)
    data.update(overrides)
    return data


class TestLoadProfile(unittest.TestCase):
    def test_loads_valid_profile(self):
        spec = pp.load_profile(BASE_PROFILE)
        self.assertEqual(spec.backend, "orcaslicer")
        self.assertEqual(spec.machine_name, "Example 256")
        self.assertEqual(spec.bed_size_mm, (256.0, 256.0))
        self.assertEqual(spec.z_height_mm, 256.0)
        self.assertEqual(spec.filament_type, "PLA")
        self.assertEqual(spec.nozzle_temp_c, 210.0)
        self.assertEqual(spec.bed_temp_c, 60.0)

    def test_motion_bounds_default_to_envelope(self):
        spec = pp.load_profile(BASE_PROFILE)
        self.assertEqual(
            spec.motion_bounds_mm,
            {"x": (0.0, 256.0), "y": (0.0, 256.0), "z": (0.0, 256.0)},
        )

    def test_native_settings_defaults_to_native_config(self):
        spec = pp.load_profile(BASE_PROFILE)
        self.assertEqual(spec.native_settings, ("/profiles/machine.json",))
        self.assertEqual(spec.native_filaments, ())

    def test_explicit_native_settings_and_filaments(self):
        spec = pp.load_profile(
            profile(native_settings=["/a.ini", "/b.ini"], native_filaments=["/pla.ini"])
        )
        self.assertEqual(spec.native_settings, ("/a.ini", "/b.ini"))
        self.assertEqual(spec.native_filaments, ("/pla.ini",))

    def test_present_but_empty_list_rejected(self):
        with self.assertRaises(pp.ProfileError):
            pp.load_profile(profile(native_settings=[]))

    def test_motion_bounds_override_is_per_axis(self):
        data = profile()
        data["machine"]["motion_bounds_mm"] = {"x": [-5, 260]}
        spec = pp.load_profile(data)
        self.assertEqual(spec.motion_bounds_mm["x"], (-5.0, 260.0))
        # Unnamed axes keep their envelope defaults.
        self.assertEqual(spec.motion_bounds_mm["y"], (0.0, 256.0))
        self.assertEqual(spec.motion_bounds_mm["z"], (0.0, 256.0))

    def test_spec_is_frozen(self):
        spec = pp.load_profile(BASE_PROFILE)
        with self.assertRaises(Exception):
            spec.backend = "curaengine"  # type: ignore[misc]

    def test_str_is_informative(self):
        self.assertIn("Example 256", str(pp.load_profile(BASE_PROFILE)))

    def test_all_preferred_backends_accepted(self):
        for backend in pp.PREFERRED_BACKEND_ORDER:
            self.assertEqual(pp.load_profile(profile(backend=backend)).backend, backend)

    def test_backend_is_case_insensitive(self):
        self.assertEqual(pp.load_profile(profile(backend="OrcaSlicer")).backend, "orcaslicer")


class TestLoadProfileRejections(unittest.TestCase):
    def assert_rejects(self, field, **overrides):
        with self.assertRaises(pp.ProfileError) as ctx:
            pp.load_profile(profile(**overrides))
        self.assertIn(field, str(ctx.exception))

    def test_unknown_backend(self):
        self.assert_rejects("backend", backend="slic3r")

    def test_missing_native_config(self):
        self.assert_rejects("native_config", native_config="")

    def test_machine_must_be_object(self):
        self.assert_rejects("machine", machine=None)

    def test_filament_must_be_object(self):
        self.assert_rejects("filament", filament=[])

    def test_missing_machine_name(self):
        self.assert_rejects(
            "machine.name", machine={"bed_size_mm": [10, 10], "z_height_mm": 10}
        )

    def test_bed_size_must_be_pair(self):
        self.assert_rejects(
            "bed_size_mm", machine={"name": "m", "bed_size_mm": [1, 2, 3], "z_height_mm": 10}
        )

    def test_bed_size_rejects_non_numbers(self):
        self.assert_rejects(
            "bed_size_mm", machine={"name": "m", "bed_size_mm": ["a", 2], "z_height_mm": 10}
        )

    def test_bool_is_not_a_number(self):
        # bool subclasses int; the loader must still reject it.
        self.assert_rejects(
            "z_height_mm", machine={"name": "m", "bed_size_mm": [10, 10], "z_height_mm": True}
        )

    def test_nonpositive_envelope_rejected(self):
        self.assert_rejects(
            "z_height_mm", machine={"name": "m", "bed_size_mm": [10, 10], "z_height_mm": 0}
        )
        self.assert_rejects(
            "bed_size_mm[0]", machine={"name": "m", "bed_size_mm": [-1, 10], "z_height_mm": 10}
        )

    def test_missing_filament_type(self):
        self.assert_rejects(
            "filament.type", filament={"nozzle_temp_c": 210, "bed_temp_c": 60}
        )

    def test_missing_nozzle_temp(self):
        self.assert_rejects("filament.nozzle_temp_c", filament={"type": "PLA", "bed_temp_c": 60})

    def test_motion_bounds_must_be_object(self):
        data = profile()
        data["machine"]["motion_bounds_mm"] = [0, 10]
        with self.assertRaises(pp.ProfileError):
            pp.load_profile(data)

    def test_profile_must_be_mapping(self):
        with self.assertRaises(pp.ProfileError):
            pp.load_profile([])  # type: ignore[arg-type]


class TestParseAxisBounds(unittest.TestCase):
    def test_valid_pair(self):
        self.assertEqual(pp.parse_axis_bounds([-5, 260], "x"), (-5.0, 260.0))

    def test_inverted_range_rejected(self):
        with self.assertRaises(pp.ProfileError):
            pp.parse_axis_bounds([10, 5], "x")

    def test_empty_range_rejected(self):
        with self.assertRaises(pp.ProfileError) as ctx:
            pp.parse_axis_bounds([10, 10], "x")
        self.assertIn("min less than max", str(ctx.exception))

    def test_wrong_shape_rejected(self):
        for bad in ([1], [1, 2, 3], "10", None, [True, 5]):
            with self.assertRaises(pp.ProfileError):
                pp.parse_axis_bounds(bad, "x")


class TestLexing(unittest.TestCase):
    def test_strip_comment(self):
        self.assertEqual(pp.strip_gcode_comment("G1 X10 ; move"), "G1 X10")
        self.assertEqual(pp.strip_gcode_comment("  ; only a comment"), "")

    def test_parse_command(self):
        self.assertEqual(pp.parse_command("G1 X10"), "G1")
        self.assertEqual(pp.parse_command("  g28  "), "G28")
        self.assertEqual(pp.parse_command("; comment"), "")
        self.assertEqual(pp.parse_command(""), "")

    def test_subnumbered_command_reduces(self):
        self.assertEqual(pp.parse_command("G1.1 X5"), "G1")

    def test_parse_numeric_tokens(self):
        self.assertEqual(
            pp.parse_numeric_tokens("G1 X10.5 Y-2 E.3"),
            {"G": 1.0, "X": 10.5, "Y": -2.0, "E": 0.3},
        )

    def test_tokens_ignore_comments(self):
        self.assertNotIn("Z", pp.parse_numeric_tokens("G1 X1 ; Z99"))


class TestValidateGcode(unittest.TestCase):
    def setUp(self):
        self.spec = pp.load_profile(BASE_PROFILE)

    def test_good_body_passes(self):
        report = pp.validate_gcode(GOOD_GCODE, self.spec)
        self.assertTrue(report.ok)
        self.assertEqual(report.errors, [])

    def test_stats_counted(self):
        report = pp.validate_gcode(GOOD_GCODE, self.spec)
        self.assertEqual(report.stats["movement_commands"], 2)
        self.assertEqual(report.stats["extrusion_moves"], 2)
        self.assertEqual(report.stats["temperature_commands"], 2)
        self.assertEqual(report.stats["lines"], 8)

    def test_empty_body_fails(self):
        report = pp.validate_gcode("", self.spec)
        self.assertFalse(report.ok)
        self.assertIn("G-code file is empty.", report.errors)

    def test_no_movement_fails(self):
        report = pp.validate_gcode("M104 S210\nM140 S60", self.spec)
        self.assertFalse(report.ok)
        self.assertIn("No G0/G1/G2/G3 movement commands found.", report.errors)

    def test_no_extrusion_fails(self):
        report = pp.validate_gcode("G90\nM104 S210\nG0 X10 Y10", self.spec)
        self.assertIn("No extrusion moves found.", report.errors)

    def test_no_temperature_fails(self):
        report = pp.validate_gcode("G90\nG1 X10 Y10 E1", self.spec)
        self.assertIn("No nozzle or bed temperature commands found.", report.errors)

    def test_out_of_bounds_x_fails(self):
        report = pp.validate_gcode(GOOD_GCODE + "\nG1 X300 Y10 Z1 E6", self.spec)
        self.assertFalse(report.ok)
        self.assertEqual(len(report.errors), 1)
        self.assertIn("X=300", report.errors[0])
        self.assertIn("0..256", report.errors[0])

    def test_out_of_bounds_each_axis(self):
        for axis in ("X", "Y", "Z"):
            body = GOOD_GCODE + "\nG1 %s300 E6" % axis
            report = pp.validate_gcode(body, self.spec)
            self.assertFalse(report.ok)
            self.assertIn("%s=300" % axis, report.errors[0])

    def test_negative_move_out_of_default_bounds(self):
        self.assertFalse(pp.validate_gcode(GOOD_GCODE + "\nG1 X-1 Y10 E6", self.spec).ok)

    def test_override_bounds_admit_off_bed_move(self):
        data = profile()
        data["machine"]["motion_bounds_mm"] = {"x": [-5, 260]}
        bounded = pp.load_profile(data)
        self.assertTrue(pp.validate_gcode(GOOD_GCODE + "\nG1 X-1 Y10 E6", bounded).ok)
        # Still bounded: -6 is below the override's floor.
        self.assertFalse(pp.validate_gcode(GOOD_GCODE + "\nG1 X-6 Y10 E6", bounded).ok)

    def test_boundary_values_are_inclusive(self):
        report = pp.validate_gcode(GOOD_GCODE + "\nG1 X0 Y256 E6", self.spec)
        self.assertTrue(report.ok, report.errors)

    def test_relative_positioning_warns_and_skips_bounds(self):
        body = "\n".join(["G90", "M104 S210", "G1 X10 Y10 E1", "G91", "G1 X9999 Y9999 E2"])
        report = pp.validate_gcode(body, self.spec)
        self.assertTrue(report.ok, report.errors)
        self.assertTrue(any("Relative positioning" in w for w in report.warnings))

    def test_g90_resumes_bounds_checking(self):
        body = "\n".join(
            ["G90", "M104 S210", "G1 X10 Y10 E1", "G91", "G1 X9999 E2", "G90", "G1 X9999 E3"]
        )
        report = pp.validate_gcode(body, self.spec)
        self.assertFalse(report.ok)
        self.assertEqual(len(report.errors), 1)

    def test_relative_warning_emitted_once(self):
        body = "\n".join(["G90", "M104 S210", "G1 X1 Y1 E1", "G91", "G1 X1 E2", "G91", "G1 X1 E3"])
        report = pp.validate_gcode(body, self.spec)
        relative = [w for w in report.warnings if "Relative positioning" in w]
        self.assertEqual(len(relative), 1)

    def test_unknown_command_warns_but_passes(self):
        report = pp.validate_gcode(GOOD_GCODE + "\nM1234 S1", self.spec)
        self.assertTrue(report.ok)
        self.assertTrue(any("M1234 line" in w for w in report.warnings))

    def test_tool_change_is_recognised(self):
        report = pp.validate_gcode(GOOD_GCODE + "\nT0", self.spec)
        self.assertTrue(report.ok)
        self.assertFalse(any("T0" in w for w in report.warnings))

    def test_supported_commands_do_not_warn(self):
        body = GOOD_GCODE + "\n" + "\n".join(sorted(pp.SUPPORTED_GCODE_COMMANDS))
        report = pp.validate_gcode(body, self.spec)
        self.assertFalse(any("Unknown" in w for w in report.warnings))

    def test_comments_are_not_counted(self):
        report = pp.validate_gcode("; just a comment\n; another", self.spec)
        self.assertEqual(report.stats["non_comment_lines"], 0)

    def test_report_as_dict_is_plain_data(self):
        payload = pp.validate_gcode(GOOD_GCODE, self.spec).as_dict()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["errors"], [])
        self.assertIsInstance(payload["stats"], dict)

    def test_deterministic(self):
        first = pp.validate_gcode(GOOD_GCODE + "\nG1 X300 E6", self.spec).as_dict()
        second = pp.validate_gcode(GOOD_GCODE + "\nG1 X300 E6", self.spec).as_dict()
        self.assertEqual(first, second)


class TestPrintabilityBridge(unittest.TestCase):
    def setUp(self):
        self.spec = pp.load_profile(BASE_PROFILE)

    def test_bridges_to_printer_profile(self):
        built = pp.to_printability_profile(self.spec)
        self.assertIsInstance(built, PrinterProfile)
        self.assertEqual(built.build_volume_mm, (256.0, 256.0, 256.0))

    def test_defaults_preserved(self):
        built = pp.to_printability_profile(self.spec)
        self.assertEqual(built.margin_mm, 2.0)
        self.assertEqual(built.min_wall_mm, 0.8)
        self.assertEqual(built.support_free_angle_deg, 45.0)

    def test_thresholds_overridable(self):
        built = pp.to_printability_profile(self.spec, margin_mm=5.0, min_wall_mm=1.2)
        self.assertEqual(built.margin_mm, 5.0)
        self.assertEqual(built.min_wall_mm, 1.2)

    def test_motion_bounds_extent_drives_build_volume(self):
        data = profile()
        data["machine"]["motion_bounds_mm"] = {"x": [-5, 260]}
        built = pp.to_printability_profile(pp.load_profile(data))
        self.assertEqual(built.build_volume_mm, (265.0, 256.0, 256.0))

    def test_bridged_envelope_drives_fit_check(self):
        built = pp.to_printability_profile(self.spec)
        self.assertTrue(check_fit((100.0, 100.0, 100.0), built).fits)
        self.assertFalse(check_fit((300.0, 10.0, 10.0), built).fits)
        # 254 exceeds the usable 256 - 2*margin = 252 mm.
        self.assertFalse(check_fit((254.0, 10.0, 10.0), built).fits)


class TestSelfcheck(unittest.TestCase):
    def test_selfcheck_passes(self):
        self.assertEqual(pp.main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
