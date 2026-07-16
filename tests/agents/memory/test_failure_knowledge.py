import unittest

from harnesscad.agents.memory.failure_knowledge import (
    FAILURE_PACK_NAME,
    build_failure_pack,
    main,
    register_failure_knowledge,
)
from harnesscad.agents.memory.skillpack import (
    SkillPack,
    unverified_names,
    verified_prompt_lines,
)
from harnesscad.agents.memory.skills import SkillLibrary


class PackTests(unittest.TestCase):
    def setUp(self):
        self.pack = build_failure_pack()

    def test_three_failure_diagnoses(self):
        self.assertEqual(self.pack.name, FAILURE_PACK_NAME)
        self.assertEqual(self.pack.names(), [
            "repair-floating-parts",
            "repair-missing-holes",
            "repair-non-manifold-boolean",
        ])

    def test_each_skill_is_a_full_diagnosis(self):
        for ps in self.pack.skills:
            self.assertTrue(ps.triggers, ps.name)      # symptom
            self.assertTrue(ps.workflow, ps.name)      # repair strategy
            self.assertTrue(ps.safety_rules, ps.name)  # prevention
            self.assertTrue(ps.verification, ps.name)
            self.assertEqual(set(ps.sections), {
                "Symptom", "Common Causes", "Repair Strategy", "Prevention"})

    def test_provenance_names_the_source_file_and_license(self):
        for ps in self.pack.skills:
            self.assertEqual(ps.provenance["repo"], "AgentSCAD")
            self.assertEqual(ps.provenance["license"], "MIT")
            self.assertTrue(ps.provenance["file"].endswith(".md"))
            self.assertEqual(ps.provenance["status"], "unverified-reference")

    def test_round_trip_lossless(self):
        self.assertEqual(SkillPack.from_dict(self.pack.to_dict()).to_dict(),
                         self.pack.to_dict())

    def test_deterministic(self):
        self.assertEqual(build_failure_pack().to_dict(), self.pack.to_dict())


class VerificationFirstTests(unittest.TestCase):
    def setUp(self):
        self.library = SkillLibrary()
        self.added = register_failure_knowledge(self.library)

    def test_registers_every_skill(self):
        self.assertEqual(self.added, build_failure_pack().names())

    def test_all_entries_land_unverified(self):
        self.assertEqual(sorted(unverified_names(self.library)),
                         sorted(self.added))
        for name in self.added:
            self.assertFalse(self.library.get(name).verified, name)

    def test_unverified_recipe_refuses_to_expand(self):
        for name in self.added:
            with self.assertRaises(RuntimeError):
                self.library.get(name).template()

    def test_never_surfaced_to_a_model_prompt(self):
        self.assertEqual(verified_prompt_lines(self.library, "non-manifold"), [])
        self.assertEqual(verified_prompt_lines(self.library), [])

    def test_reimport_does_not_displace(self):
        self.assertEqual(register_failure_knowledge(self.library), [])

    def test_registration_never_flips_verified(self):
        register_failure_knowledge(self.library, overwrite=True)
        for name in self.added:
            self.assertFalse(self.library.get(name).verified, name)


class SelfcheckTests(unittest.TestCase):
    def test_selfcheck_exits_zero(self):
        self.assertEqual(main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
