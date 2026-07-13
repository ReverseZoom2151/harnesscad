"""Tests for bench.muse_manufacturability."""

import unittest

from harnesscad.eval.bench.protocols.muse_manufacturability import (
    MATERIALS,
    PROCESSES,
    material_process_compatible,
    muse_manufacturability,
    process_tolerance,
    score_manufacturable,
    score_well_toleranced,
)


class TableTests(unittest.TestCase):
    def test_material_process_compatibility(self):
        self.assertTrue(material_process_compatible("Timber", "CNC Milling"))
        self.assertTrue(material_process_compatible("PLA", "3D Printing"))
        self.assertFalse(material_process_compatible("PLA", "CNC Milling"))
        self.assertFalse(material_process_compatible("Steel", "Laser Cutting"))

    def test_process_tolerance(self):
        self.assertEqual(process_tolerance("CNC Milling"), (0.05, 0.10))
        self.assertEqual(process_tolerance("3D Printing"), (0.10, 0.50))

    def test_unknown_names_raise(self):
        with self.assertRaises(ValueError):
            material_process_compatible("Unobtanium", "CNC Milling")
        with self.assertRaises(ValueError):
            process_tolerance("Warp Drive")

    def test_tables_present(self):
        self.assertIn("Aluminum", MATERIALS)
        self.assertIn("Injection Molding", PROCESSES)


class ManufacturableTests(unittest.TestCase):
    def test_clean_timber_chair_passes(self):
        design = {
            "material": "Timber", "process": "CNC Milling",
            "components": [
                {"name": "seat", "wall_thickness": 18.0, "bbox": (400, 400, 20)},
                {"name": "leg", "wall_thickness": 30.0, "bbox": (30, 30, 420)},
            ],
        }
        r = score_manufacturable(design)
        self.assertEqual(r["manufacturable"], 1)
        self.assertEqual(r["violations"], ())

    def test_incompatible_material_process(self):
        r = score_manufacturable({"material": "PLA", "process": "CNC Milling",
                                  "components": []})
        self.assertEqual(r["manufacturable"], 0)
        self.assertIn("incompatible_material_process", r["violations"])

    def test_zero_and_thin_walls(self):
        r = score_manufacturable({
            "material": "PLA", "process": "3D Printing",
            "components": [
                {"name": "shell", "wall_thickness": 0.0},
                {"name": "rib", "wall_thickness": 0.3},
            ]})
        self.assertEqual(r["manufacturable"], 0)
        self.assertIn("zero_thickness:shell", r["violations"])
        self.assertIn("thin_wall:rib", r["violations"])

    def test_laser_sheet_thickness_limit(self):
        r = score_manufacturable({
            "material": "Acrylic", "process": "Laser Cutting",
            "components": [{"name": "panel", "wall_thickness": 6.0,
                            "thickness": 6.0}]})
        self.assertIn("exceeds_sheet_thickness:panel", r["violations"])

    def test_build_volume_and_max_edge(self):
        big_print = score_manufacturable({
            "material": "PLA", "process": "3D Printing",
            "components": [{"name": "body", "wall_thickness": 2.0,
                            "bbox": (350, 100, 100)}]})
        self.assertIn("exceeds_build_volume:body", big_print["violations"])
        big_cnc = score_manufacturable({
            "material": "Aluminum", "process": "CNC Milling",
            "components": [{"name": "beam", "wall_thickness": 5.0,
                            "bbox": (2500, 50, 50)}]})
        self.assertIn("exceeds_max_edge:beam", big_cnc["violations"])

    def test_inaccessible_cavity_only_subtractive(self):
        cnc = score_manufacturable({
            "material": "Aluminum", "process": "CNC Milling",
            "components": [{"name": "block", "wall_thickness": 5.0,
                            "internal_dead_cavity": True}]})
        self.assertIn("inaccessible_cavity:block", cnc["violations"])
        # 3D printing can produce internal cavities -> no violation.
        printed = score_manufacturable({
            "material": "PLA", "process": "3D Printing",
            "components": [{"name": "block", "wall_thickness": 2.0,
                            "internal_dead_cavity": True}]})
        self.assertEqual(printed["manufacturable"], 1)

    def test_brittle_cantilever(self):
        r = score_manufacturable({
            "material": "Acrylic", "process": "CNC Milling",
            "components": [{"name": "arm", "wall_thickness": 3.0,
                            "cantilever": True, "load_bearing": True}]})
        self.assertIn("fragile_brittle_cantilever:arm", r["violations"])


class WellTolerancedTests(unittest.TestCase):
    def test_good_clearances_pass(self):
        r = score_well_toleranced({
            "material": "Timber", "process": "CNC Milling",
            "clearances": [("leg-socket", 0.08), ("backrest-slot", 0.1)],
            "components": [{"name": "seat", "wall_thickness": 18.0}]})
        self.assertEqual(r["well_toleranced"], 1)

    def test_illegal_fusion(self):
        r = score_well_toleranced({
            "material": "Timber", "process": "CNC Milling",
            "clearances": [("seam", 0.0)]})
        self.assertEqual(r["well_toleranced"], 0)
        self.assertIn("illegal_fusion:seam", r["violations"])

    def test_exaggerated_gap(self):
        r = score_well_toleranced({
            "material": "Timber", "process": "CNC Milling",
            "clearances": [5.0]})
        self.assertIn("exaggerated_gap:?", r["violations"])

    def test_bare_number_clearances(self):
        r = score_well_toleranced({
            "material": "PLA", "process": "3D Printing",
            "clearances": [0.2, 0.3]})
        self.assertEqual(r["well_toleranced"], 1)

    def test_bad_factor(self):
        with self.assertRaises(ValueError):
            score_well_toleranced({"material": "PLA", "process": "3D Printing"},
                                  max_gap_factor=0)


class PillarTests(unittest.TestCase):
    def test_full_pass(self):
        design = {
            "material": "Timber", "process": "CNC Milling",
            "components": [{"name": "seat", "wall_thickness": 18.0,
                            "bbox": (400, 400, 20)}],
            "clearances": [0.08],
        }
        r = muse_manufacturability(design)
        self.assertEqual(r["average"], 1.0)

    def test_half_pass(self):
        # manufacturable but badly toleranced.
        design = {
            "material": "PLA", "process": "3D Printing",
            "components": [{"name": "body", "wall_thickness": 2.0,
                            "bbox": (100, 100, 100)}],
            "clearances": [0.0],
        }
        r = muse_manufacturability(design)
        self.assertEqual(r["manufacturable"], 1)
        self.assertEqual(r["well_toleranced"], 0)
        self.assertEqual(r["average"], 0.5)


if __name__ == "__main__":
    unittest.main()
