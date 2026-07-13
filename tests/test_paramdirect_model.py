import unittest

from harnesscad.domain.editing.paramdirect_model import (
    Paradigm, InfoLayer, ParametricFeature, FeatureTree, Face, DirectBRep,
    ParameterEdit, PushPullEdit, edit_from_dict, classify_edit, edit_layer,
)


def _tree():
    return FeatureTree([
        ParametricFeature("f0", "sketch_extrude", {"height": 10.0, "width": 4.0}),
        ParametricFeature("f1", "hole", {"depth": 3.0}, refs=("f0",)),
        ParametricFeature("f2", "fillet", {"radius": 1.0}, refs=("f0", "f1"),
                          direct_edit=True),
    ])


class TestFeatureTree(unittest.TestCase):
    def test_lookup_and_set(self):
        t = _tree()
        self.assertEqual(t.parameter("f0", "height"), 10.0)
        t.set_parameter("f0", "height", 68.0)
        self.assertEqual(t.parameter("f0", "height"), 68.0)

    def test_set_unknown_param(self):
        with self.assertRaises(KeyError):
            _tree().set_parameter("f0", "nope", 1.0)

    def test_unknown_feature(self):
        with self.assertRaises(KeyError):
            _tree().index_of("zz")

    def test_variation_space_sorted(self):
        vs = _tree().variation_space()
        self.assertEqual(vs, (("f0", "height"), ("f0", "width"),
                              ("f1", "depth"), ("f2", "radius")))

    def test_dependents(self):
        t = _tree()
        self.assertEqual(t.dependents("f0"), ("f1", "f2"))
        self.assertEqual(t.dependents("f1"), ("f2",))
        self.assertEqual(t.dependents("f2"), ())

    def test_roundtrip_and_copy(self):
        t = _tree()
        self.assertEqual(FeatureTree.from_dict(t.to_dict()).to_dict(), t.to_dict())
        c = t.copy()
        c.set_parameter("f0", "height", 1.0)
        self.assertEqual(t.parameter("f0", "height"), 10.0)  # copy is deep


class TestDirectBRep(unittest.TestCase):
    def _brep(self):
        b = DirectBRep()
        b.add_face(Face("top", 0, 0, 1, 10.0, origin="f0"))
        b.add_face(Face("bottom", 0, 0, -1, 0.0, origin="f0"))
        b.add_face(Face("side", 1, 0, 0, 4.0))
        b.connect("top", "side")
        b.connect("bottom", "side")
        b.connect("top", "side")  # dedup
        return b

    def test_push_pull_moves_offset(self):
        b = self._brep()
        b.push_pull("top", 5.0)
        self.assertEqual(b.faces["top"].offset, 15.0)

    def test_adjacency_dedup_and_neighbours(self):
        b = self._brep()
        self.assertEqual(len(b.adjacency), 2)
        self.assertEqual(b.neighbours("side"), ("bottom", "top"))

    def test_geometry_key_changes(self):
        b = self._brep()
        k0 = b.faces["top"].geometry_key()
        b.push_pull("top", 1.0)
        self.assertNotEqual(k0, b.faces["top"].geometry_key())

    def test_roundtrip(self):
        b = self._brep()
        self.assertEqual(DirectBRep.from_dict(b.to_dict()).to_dict(), b.to_dict())


class TestEditClassification(unittest.TestCase):
    def test_classify(self):
        self.assertIs(classify_edit(ParameterEdit("f0", "h", 1.0)),
                      Paradigm.PARAMETRIC)
        self.assertIs(classify_edit(PushPullEdit("top", 2.0)), Paradigm.DIRECT)

    def test_layer(self):
        self.assertIs(edit_layer(ParameterEdit("f0", "h", 1.0)),
                      InfoLayer.CONSTRAINT)
        self.assertIs(edit_layer(PushPullEdit("top", 2.0)), InfoLayer.GEOMETRY)

    def test_classify_bad(self):
        with self.assertRaises(TypeError):
            classify_edit(object())

    def test_edit_roundtrip(self):
        for e in (ParameterEdit("f0", "h", 1.5), PushPullEdit("top", -2.0)):
            self.assertEqual(edit_from_dict(e.to_dict()).to_dict(), e.to_dict())

    def test_edit_from_dict_bad(self):
        with self.assertRaises(ValueError):
            edit_from_dict({"kind": "xyz"})


if __name__ == "__main__":
    unittest.main()
