"""Tests for datagen.designproc_procedure."""

import unittest

from harnesscad.data.datagen import design_procedure as dp


class TestDesignStep(unittest.TestCase):
    def test_valid_kind(self):
        s = dp.DesignStep(dp.ADD_PRIMITIVE, "hole")
        self.assertEqual(s.kind, dp.ADD_PRIMITIVE)
        self.assertEqual(s.to_dict(), {"kind": dp.ADD_PRIMITIVE, "detail": "hole"})

    def test_invalid_kind(self):
        with self.assertRaises(ValueError):
            dp.DesignStep("sculpt")


class TestBuildProcedure(unittest.TestCase):
    def test_full_procedure_order(self):
        proc = dp.build_procedure("bracket", "saddle", n_primitives=2)
        kinds = proc.kinds()
        self.assertEqual(kinds[0], dp.SELECT_REFERENCE_SURFACE)
        self.assertEqual(kinds[1], dp.CONFORM_TO_SURFACE)
        self.assertEqual(kinds[-1], dp.EXPORT)
        self.assertEqual(kinds[-2], dp.REMOVE_REFERENCE_SURFACE)
        self.assertTrue(proc.uses_reference_surface())
        self.assertEqual(proc.surface_kind, "saddle")

    def test_full_procedure_is_valid(self):
        proc = dp.build_procedure("bracket", "gaussian")
        self.assertTrue(dp.is_valid_procedure(proc))

    def test_baseline_no_surface_is_valid(self):
        proc = dp.build_procedure("bracket", "wave",
                                  with_reference_surface=False)
        self.assertFalse(proc.uses_reference_surface())
        self.assertNotIn(dp.REMOVE_REFERENCE_SURFACE, proc.kinds())
        self.assertTrue(dp.is_valid_procedure(proc))

    def test_organic_step_count(self):
        proc = dp.build_procedure("bracket", "ripple", with_fillet=True)
        # conform + fillet = 2 organic steps
        self.assertEqual(proc.organic_step_count(), 2)
        base = dp.build_procedure("bracket", "ripple",
                                  with_reference_surface=False, with_fillet=False)
        self.assertEqual(base.organic_step_count(), 0)

    def test_negative_primitives(self):
        with self.assertRaises(ValueError):
            dp.build_procedure("bracket", "saddle", n_primitives=-1)


class TestValidateProcedure(unittest.TestCase):
    def test_empty(self):
        ok, errs = dp.validate_procedure(dp.DesignProcedure("bracket"))
        self.assertFalse(ok)
        self.assertIn("empty procedure", errs)

    def test_surface_not_removed(self):
        proc = dp.DesignProcedure("bracket", [
            dp.DesignStep(dp.SELECT_REFERENCE_SURFACE),
            dp.DesignStep(dp.CONFORM_TO_SURFACE),
            dp.DesignStep(dp.EXPORT),
        ])
        ok, errs = dp.validate_procedure(proc)
        self.assertFalse(ok)
        self.assertTrue(any("never removed" in e for e in errs))

    def test_surface_not_first(self):
        proc = dp.DesignProcedure("bracket", [
            dp.DesignStep(dp.ADD_PRIMITIVE),
            dp.DesignStep(dp.SELECT_REFERENCE_SURFACE),
            dp.DesignStep(dp.REMOVE_REFERENCE_SURFACE),
            dp.DesignStep(dp.EXPORT),
        ])
        ok, errs = dp.validate_procedure(proc)
        self.assertFalse(ok)
        self.assertTrue(any("selected first" in e for e in errs))

    def test_remove_without_select(self):
        proc = dp.DesignProcedure("bracket", [
            dp.DesignStep(dp.ADD_PRIMITIVE),
            dp.DesignStep(dp.REMOVE_REFERENCE_SURFACE),
        ])
        ok, errs = dp.validate_procedure(proc)
        self.assertFalse(ok)
        self.assertTrue(any("without a select" in e for e in errs))

    def test_export_not_last(self):
        proc = dp.DesignProcedure("bracket", [
            dp.DesignStep(dp.ADD_PRIMITIVE),
            dp.DesignStep(dp.EXPORT),
            dp.DesignStep(dp.FILLET),
        ])
        ok, errs = dp.validate_procedure(proc)
        self.assertFalse(ok)
        self.assertTrue(any("final step" in e for e in errs))

    def test_no_features(self):
        proc = dp.DesignProcedure("bracket", [
            dp.DesignStep(dp.SELECT_REFERENCE_SURFACE),
            dp.DesignStep(dp.REMOVE_REFERENCE_SURFACE),
            dp.DesignStep(dp.EXPORT),
        ])
        ok, errs = dp.validate_procedure(proc)
        self.assertFalse(ok)
        self.assertTrue(any("no object features" in e for e in errs))

    def test_conform_out_of_place(self):
        # conform appears after remove -> invalid.
        proc = dp.DesignProcedure("bracket", [
            dp.DesignStep(dp.SELECT_REFERENCE_SURFACE),
            dp.DesignStep(dp.ADD_PRIMITIVE),
            dp.DesignStep(dp.REMOVE_REFERENCE_SURFACE),
            dp.DesignStep(dp.CONFORM_TO_SURFACE),
            dp.DesignStep(dp.EXPORT),
        ])
        ok, errs = dp.validate_procedure(proc)
        self.assertFalse(ok)
        self.assertTrue(any("between select and remove" in e for e in errs))


class TestPrompt(unittest.TestCase):
    def test_full_prompt_has_all_slots(self):
        script = "# saddle.py\nimport cadquery as cq"
        slots = dp.build_prompt("bracket", "A rectangular bracket.",
                                script, mode="full")
        self.assertIn("CadQuery", slots["prefix"])
        self.assertIn("reference surface", slots["design_context"])
        self.assertIn("watertight", slots["postfix"])
        self.assertEqual(slots["reference_surface_program"], script)
        self.assertIn(script, slots["text"])
        self.assertIn("A rectangular bracket.", slots["text"])

    def test_text_mode_drops_surface(self):
        slots = dp.build_prompt("bracket", "desc", "SURFACE", mode="text")
        self.assertEqual(slots["reference_surface_program"], "")
        self.assertIn("organic", slots["design_context"])
        self.assertNotIn("SURFACE", slots["text"])

    def test_none_mode_drops_context(self):
        slots = dp.build_prompt("bracket", "desc", "SURFACE", mode="none")
        self.assertEqual(slots["design_context"], "")
        self.assertNotIn("SURFACE", slots["text"])

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            dp.build_prompt("bracket", "desc", mode="weird")

    def test_procedure_from_prompt_mode(self):
        full = dp.procedure_from_prompt_mode("bracket", "saddle", "full")
        self.assertTrue(full.uses_reference_surface())
        rt = dp.procedure_from_prompt_mode("bracket", "saddle", "none")
        self.assertFalse(rt.uses_reference_surface())
        self.assertTrue(dp.is_valid_procedure(full))
        self.assertTrue(dp.is_valid_procedure(rt))


if __name__ == "__main__":
    unittest.main()
