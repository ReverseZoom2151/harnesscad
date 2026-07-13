import unittest

from harnesscad.domain.editing.paramdirect_model import (
    DirectBRep, Face, FeatureTree, ParametricFeature, InfoLayer,
    ParameterEdit, PushPullEdit,
)
from harnesscad.domain.editing.paramdirect_consistency import (
    DistanceConstraint, ParallelConstraint, Inconsistency, HybridModel,
    check_consistency, is_consistent, propagate_parametric_to_geometry,
    propagate_direct_to_constraint, recognize_constraints, design_intent_drift,
)


def _brep():
    b = DirectBRep()
    b.add_face(Face("a", 0, 0, 1, 0.0))
    b.add_face(Face("b", 0, 0, 1, 10.0))
    b.connect("a", "b")
    return b


class TestCheckConsistency(unittest.TestCase):
    def test_consistent(self):
        m = HybridModel(_brep(), [DistanceConstraint("a", "b", 10.0)])
        self.assertTrue(is_consistent(m))
        self.assertEqual(check_consistency(m), [])

    def test_geometry_violates_distance(self):
        b = _brep()
        b.push_pull("b", 5.0)  # gap now 15, constraint wants 10
        m = HybridModel(b, [DistanceConstraint("a", "b", 10.0)])
        incs = check_consistency(m)
        self.assertEqual(len(incs), 1)
        self.assertIs(incs[0].layer, InfoLayer.GEOMETRY)
        self.assertEqual(incs[0].entities, ("a", "b"))

    def test_missing_face(self):
        m = HybridModel(_brep(), [DistanceConstraint("a", "zz", 10.0)])
        incs = check_consistency(m)
        self.assertIs(incs[0].layer, InfoLayer.CONSTRAINT)

    def test_parallel_violation(self):
        b = DirectBRep()
        b.add_face(Face("a", 0, 0, 1, 0.0))
        b.add_face(Face("b", 1, 0, 0, 0.0))
        m = HybridModel(b, [ParallelConstraint("a", "b")])
        incs = check_consistency(m)
        self.assertEqual(len(incs), 1)
        self.assertIs(incs[0].layer, InfoLayer.GEOMETRY)


class TestParametricToGeometry(unittest.TestCase):
    def test_propagation_keeps_consistent(self):
        b = DirectBRep()
        b.add_face(Face("box:len:a", 0, 0, 1, 0.0))
        b.add_face(Face("box:len:b", 0, 0, 1, 10.0))
        tree = FeatureTree([ParametricFeature("box", "extrude", {"len": 10.0})])
        m = HybridModel(b, [DistanceConstraint("box:len:a", "box:len:b", 10.0)],
                        tree)
        self.assertTrue(is_consistent(m))
        out = propagate_parametric_to_geometry(m, ParameterEdit("box", "len", 25.0))
        # constraint updated and geometry re-solved -> still consistent
        self.assertTrue(is_consistent(out))
        self.assertEqual(out.brep.faces["box:len:b"].offset, 25.0)
        self.assertEqual(out.tree.parameter("box", "len"), 25.0)


class TestDirectToConstraint(unittest.TestCase):
    def test_pushpull_updates_constraint(self):
        m = HybridModel(_brep(), [DistanceConstraint("a", "b", 10.0)])
        out = propagate_direct_to_constraint(m, PushPullEdit("b", 5.0))
        # geometry moved AND constraint value reconciled to the new gap
        self.assertEqual(out.brep.faces["b"].offset, 15.0)
        self.assertTrue(is_consistent(out))
        self.assertEqual(out.constraints[0].value, 15.0)

    def test_original_model_untouched(self):
        m = HybridModel(_brep(), [DistanceConstraint("a", "b", 10.0)])
        propagate_direct_to_constraint(m, PushPullEdit("b", 5.0))
        self.assertEqual(m.brep.faces["b"].offset, 10.0)


class TestRecognition(unittest.TestCase):
    def test_recognize_parallel_and_distance(self):
        rec = recognize_constraints(_brep())
        kinds = sorted(c.ctype for c in rec)
        self.assertEqual(kinds, ["distance", "parallel"])

    def test_design_intent_drift(self):
        # original design intent had a distance of 10 between a and c;
        # recognition from geometry only sees a<->b, so a<->c drifts (is lost)
        original = [DistanceConstraint("a", "c", 10.0),
                    DistanceConstraint("a", "b", 10.0)]
        recognized = recognize_constraints(_brep())  # only a<->b
        drift = design_intent_drift(original, recognized)
        self.assertEqual(len(drift), 1)
        self.assertEqual((drift[0].face_a, drift[0].face_b), ("a", "c"))

    def test_no_drift_when_matching(self):
        recognized = recognize_constraints(_brep())
        self.assertEqual(design_intent_drift(recognized, recognized), [])


if __name__ == "__main__":
    unittest.main()
