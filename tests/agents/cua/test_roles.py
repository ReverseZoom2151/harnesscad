"""Grounding is a specialist: it must be its own model, and tier-0 declares none."""

import unittest

from harnesscad.agents.cua.roles import (
    CAD_TIER0_SPLIT, REFERENCE_SPLIT, Role, RoleAssignment, RoleError, ROLE_IO,
    validate,
)


class TestRoleIO(unittest.TestCase):
    def test_action_model_needs_no_image(self):
        self.assertFalse(ROLE_IO[Role.ACTION]["needs_image"])

    def test_vision_and_grounding_need_the_image(self):
        self.assertTrue(ROLE_IO[Role.VISION]["needs_image"])
        self.assertTrue(ROLE_IO[Role.GROUNDING]["needs_image"])


class TestValidate(unittest.TestCase):
    def test_reference_split_is_sound(self):
        validate(REFERENCE_SPLIT, require_grounding=True, require_vision=True)

    def test_no_action_model_refused(self):
        with self.assertRaises(RoleError):
            validate(RoleAssignment(vision="v", action=None, grounding="g"))

    def test_grounding_collapsed_into_action_refused(self):
        with self.assertRaises(RoleError):
            validate(RoleAssignment(action="m", grounding="m"))

    def test_grounding_collapsed_into_vision_refused(self):
        with self.assertRaises(RoleError):
            validate(RoleAssignment(vision="m", action="a", grounding="m"))

    def test_tier0_has_no_specialists(self):
        validate(CAD_TIER0_SPLIT)  # sound: action only
        self.assertEqual(CAD_TIER0_SPLIT.present_roles(), [Role.ACTION])

    def test_tier0_refused_when_grounding_required(self):
        with self.assertRaises(RoleError):
            validate(CAD_TIER0_SPLIT, require_grounding=True)


class TestAssignment(unittest.TestCase):
    def test_model_for_and_present_roles(self):
        a = RoleAssignment(vision="v", action="a", grounding="g")
        self.assertEqual(a.model_for(Role.GROUNDING), "g")
        self.assertEqual(set(a.present_roles()), set(Role))

    def test_to_dict(self):
        self.assertEqual(REFERENCE_SPLIT.to_dict(),
                         {"vision": "qwen-2.5-vl", "action": "llama-3.3",
                          "grounding": "os-atlas"})


if __name__ == "__main__":
    unittest.main()
