import unittest

from harnesscad.domain.editing.paramdirect_model import (
    FeatureTree, ParametricFeature, Face, DirectBRep, ParameterEdit, PushPullEdit,
)
from harnesscad.domain.editing.paramdirect_pseudofeature import (
    append_pseudo_feature, regenerate, transform_to_feature_redefinition,
    RegenResult, PSEUDO_PARAM,
)


def _setup():
    tree = FeatureTree([
        ParametricFeature("slot", "sketch_extrude", {"P10": 33.0}),
    ])
    brep = DirectBRep()
    brep.add_face(Face("slot_face", 0, 0, 1, 33.0, origin="slot"))
    return tree, brep


class TestAppend(unittest.TestCase):
    def test_append_leaves_history_unchanged(self):
        tree, brep = _setup()
        new = append_pseudo_feature(tree, brep, PushPullEdit("slot_face", 5.0))
        # original history untouched
        self.assertEqual(len(tree.features), 1)
        # pseudo-feature appended at the end referencing the anchor
        self.assertEqual(len(new.features), 2)
        pf = new.features[-1]
        self.assertTrue(pf.direct_edit)
        self.assertEqual(pf.refs, ("slot",))
        self.assertEqual(pf.params[PSEUDO_PARAM], 5.0)

    def test_append_requires_anchor(self):
        tree, brep = _setup()
        brep.add_face(Face("floating", 1, 0, 0, 2.0))  # no origin
        with self.assertRaises(ValueError):
            append_pseudo_feature(tree, brep, PushPullEdit("floating", 1.0))


class TestRegenerate(unittest.TestCase):
    def test_p10_change_fails_regeneration(self):
        tree, brep = _setup()
        new = append_pseudo_feature(tree, brep, PushPullEdit("slot_face", 5.0))
        # P10 33 -> 68 invalidates the pseudo-feature's anchor (Fig. 3)
        res = regenerate(new, ParameterEdit("slot", "P10", 68.0))
        self.assertIsInstance(res, RegenResult)
        self.assertFalse(res.ok)
        self.assertEqual(res.broken, ["pseudo1"])
        self.assertIn("invalidating", res.reason)

    def test_change_without_pseudo_ok(self):
        tree, _ = _setup()
        res = regenerate(tree, ParameterEdit("slot", "P10", 68.0))
        self.assertTrue(res.ok)
        self.assertEqual(res.tree.parameter("slot", "P10"), 68.0)

    def test_unrelated_feature_edit_ok(self):
        tree, brep = _setup()
        tree.features.append(ParametricFeature("other", "hole", {"d": 2.0}))
        new = append_pseudo_feature(tree, brep, PushPullEdit("slot_face", 5.0))
        res = regenerate(new, ParameterEdit("other", "d", 9.0))
        self.assertTrue(res.ok)

    def test_regenerate_type_error(self):
        tree, _ = _setup()
        with self.assertRaises(TypeError):
            regenerate(tree, PushPullEdit("slot_face", 1.0))


class TestTransform(unittest.TestCase):
    def test_perfect_solution_regenerates(self):
        tree, brep = _setup()
        new = append_pseudo_feature(tree, brep, PushPullEdit("slot_face", 5.0))
        fixed = transform_to_feature_redefinition(new, "pseudo1")
        # pseudo-feature removed, anchor redefined (33 + 5)
        self.assertEqual(len(fixed.features), 1)
        self.assertEqual(fixed.parameter("slot", "P10"), 38.0)
        # now the P10 edit regenerates cleanly (no fragile anchor)
        res = regenerate(fixed, ParameterEdit("slot", "P10", 68.0))
        self.assertTrue(res.ok)

    def test_transform_rejects_non_pseudo(self):
        tree, _ = _setup()
        with self.assertRaises(ValueError):
            transform_to_feature_redefinition(tree, "slot")


if __name__ == "__main__":
    unittest.main()
