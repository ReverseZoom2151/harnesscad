import unittest

from editing.paramdirect_model import FeatureTree, ParametricFeature, PushPullEdit
from editing.paramdirect_translate import (
    FaceParamLink, Translation, translate_push_pull, is_achievable, is_unique,
)


def _tree():
    return FeatureTree([
        ParametricFeature("box", "sketch_extrude", {"height": 10.0, "half": 4.0}),
    ])


class TestFaceParamLink(unittest.TestCase):
    def test_zero_gain_rejected(self):
        with self.assertRaises(ValueError):
            FaceParamLink("top", "box", "height", 0.0)


class TestTranslate(unittest.TestCase):
    def test_achievable_unique(self):
        t = _tree()
        links = [FaceParamLink("top", "box", "height", 1.0)]
        cands = translate_push_pull(t, PushPullEdit("top", 5.0), links)
        self.assertEqual(len(cands), 1)
        pe = cands[0].param_edits[0]
        self.assertEqual((pe.target_fid, pe.param, pe.new_value),
                         ("box", "height", 15.0))
        self.assertTrue(is_achievable(t, PushPullEdit("top", 5.0), links))
        self.assertTrue(is_unique(t, PushPullEdit("top", 5.0), links))

    def test_gain_scales_delta(self):
        t = _tree()
        # a face governed with gain 2 -> parameter moves half the push distance
        links = [FaceParamLink("top", "box", "height", 2.0)]
        cands = translate_push_pull(t, PushPullEdit("top", 6.0), links)
        self.assertEqual(cands[0].param_edits[0].new_value, 13.0)  # 10 + 6/2

    def test_non_achievable_empty(self):
        t = _tree()
        links = [FaceParamLink("top", "box", "height", 1.0)]
        # a push-pull on a face with no governing parameter -> not achievable
        cands = translate_push_pull(t, PushPullEdit("mystery", 5.0), links)
        self.assertEqual(cands, [])
        self.assertFalse(is_achievable(t, PushPullEdit("mystery", 5.0), links))

    def test_non_unique_with_symmetric(self):
        t = _tree()
        links = [FaceParamLink("top", "box", "height", 1.0)]
        cands = translate_push_pull(
            t, PushPullEdit("top", 5.0), links,
            symmetric_params=[("box", "half")])
        self.assertEqual(len(cands), 2)  # not unique
        self.assertFalse(is_unique(t, PushPullEdit("top", 5.0), links,
                                   symmetric_params=[("box", "half")]))

    def test_multiple_links_same_face_non_unique(self):
        t = FeatureTree([
            ParametricFeature("box", "sketch_extrude", {"height": 10.0}),
            ParametricFeature("pad", "extrude", {"len": 2.0}),
        ])
        links = [FaceParamLink("top", "box", "height", 1.0),
                 FaceParamLink("top", "pad", "len", 1.0)]
        cands = translate_push_pull(t, PushPullEdit("top", 3.0), links)
        self.assertEqual(len(cands), 2)


class TestApply(unittest.TestCase):
    def test_apply_param_edit(self):
        t = _tree()
        links = [FaceParamLink("top", "box", "height", 1.0)]
        cand = translate_push_pull(t, PushPullEdit("top", 5.0), links)[0]
        out = cand.apply(t)
        self.assertEqual(out.parameter("box", "height"), 15.0)
        self.assertEqual(t.parameter("box", "height"), 10.0)  # source untouched

    def test_apply_reorder(self):
        t = FeatureTree([
            ParametricFeature("a", "x", {}),
            ParametricFeature("b", "y", {}),
            ParametricFeature("c", "z", {}),
        ])
        tr = Translation(reorder=("c", 0))
        out = tr.apply(t)
        self.assertEqual([f.fid for f in out.features], ["c", "a", "b"])


if __name__ == "__main__":
    unittest.main()
