import unittest

from harnesscad.domain.geometry.topology.selector_algebra import (
    AndSelector,
    AreaNthSelector,
    BoxSelector,
    CenterNthSelector,
    DirectionMinMaxSelector,
    DirectionNthSelector,
    DirectionSelector,
    InverseSelector,
    LengthNthSelector,
    NearestToPointSelector,
    ParallelDirSelector,
    PerpendicularDirSelector,
    RadiusNthSelector,
    Selector,
    SelectorError,
    Shape,
    SubtractSelector,
    SumSelector,
    TypeSelector,
)


def cube_faces(size=2.0):
    h = size / 2.0
    return [
        Shape("Face", (0, 0, h), (0, 0, 1), "PLANE", area=size * size, name="top"),
        Shape("Face", (0, 0, -h), (0, 0, -1), "PLANE", area=size * size, name="bot"),
        Shape("Face", (h, 0, 0), (1, 0, 0), "PLANE", area=size * size, name="right"),
        Shape("Face", (-h, 0, 0), (-1, 0, 0), "PLANE", area=size * size, name="left"),
        Shape("Face", (0, h, 0), (0, 1, 0), "PLANE", area=size * size, name="front"),
        Shape("Face", (0, -h, 0), (0, -1, 0), "PLANE", area=size * size, name="back"),
    ]


class TestDirectional(unittest.TestCase):
    def test_max_z(self):
        sel = DirectionMinMaxSelector((0, 0, 1), directionMax=True)
        out = sel.filter(cube_faces())
        self.assertEqual([s.name for s in out], ["top"])

    def test_min_z(self):
        sel = DirectionMinMaxSelector((0, 0, 1), directionMax=False)
        out = sel.filter(cube_faces())
        self.assertEqual([s.name for s in out], ["bot"])

    def test_direction_selector_same_sense(self):
        sel = DirectionSelector((0, 0, 1))
        out = sel.filter(cube_faces())
        self.assertEqual([s.name for s in out], ["top"])

    def test_parallel_both_senses(self):
        sel = ParallelDirSelector((0, 0, 1))
        out = {s.name for s in sel.filter(cube_faces())}
        self.assertEqual(out, {"top", "bot"})

    def test_perpendicular(self):
        sel = PerpendicularDirSelector((0, 0, 1))
        out = {s.name for s in sel.filter(cube_faces())}
        self.assertEqual(out, {"right", "left", "front", "back"})


class TestType(unittest.TestCase):
    def test_type(self):
        faces = cube_faces() + [Shape("Face", (0, 0, 5), (0, 0, 1), "CYLINDER")]
        out = TypeSelector("plane").filter(faces)
        self.assertEqual(len(out), 6)


class TestNth(unittest.TestCase):
    def test_center_nth_clustering(self):
        # three z levels: two faces share z=0 (one cluster)
        shapes = [
            Shape("Face", (0, 0, 0.0), name="a"),
            Shape("Face", (1, 0, 0.0), name="b"),
            Shape("Face", (0, 0, 2.0), name="c"),
            Shape("Face", (0, 0, 4.0), name="d"),
        ]
        # ascending order (directionMax=True): cluster 0 is the min z level
        out = CenterNthSelector((0, 0, 1), 0, directionMax=True).filter(shapes)
        self.assertEqual({s.name for s in out}, {"a", "b"})
        # last cluster ascending is the top
        out = CenterNthSelector((0, 0, 1), -1, directionMax=True).filter(shapes)
        self.assertEqual([s.name for s in out], ["d"])
        # reversed (directionMax=False): index 0 is now the top
        out = CenterNthSelector((0, 0, 1), 0, directionMax=False).filter(shapes)
        self.assertEqual([s.name for s in out], ["d"])

    def test_radius_nth_drops_non_circular(self):
        edges = [
            Shape("Edge", geom_type="LINE"),  # no radius -> dropped
            Shape("Edge", geom_type="CIRCLE", radius=1.0, name="r1"),
            Shape("Edge", geom_type="CIRCLE", radius=3.0, name="r3"),
        ]
        out = RadiusNthSelector(0).filter(edges)
        self.assertEqual([s.name for s in out], ["r1"])
        out = RadiusNthSelector(-1).filter(edges)
        self.assertEqual([s.name for s in out], ["r3"])

    def test_length_nth(self):
        edges = [
            Shape("Edge", length=5.0, name="l5"),
            Shape("Edge", length=1.0, name="l1"),
        ]
        out = LengthNthSelector(0).filter(edges)
        self.assertEqual([s.name for s in out], ["l1"])

    def test_area_nth(self):
        faces = [
            Shape("Face", area=9.0, name="big"),
            Shape("Face", area=1.0, name="small"),
        ]
        out = AreaNthSelector(0).filter(faces)
        self.assertEqual([s.name for s in out], ["small"])

    def test_empty_raises(self):
        with self.assertRaises(SelectorError):
            RadiusNthSelector(0).filter([])

    def test_direction_nth_parallel_then_center(self):
        shapes = [
            Shape("Face", (0, 0, 0), (0, 0, 1), "PLANE", name="z0"),
            Shape("Face", (0, 0, 3), (0, 0, 1), "PLANE", name="z3"),
            Shape("Face", (5, 0, 0), (1, 0, 0), "PLANE", name="x"),  # not parallel
        ]
        out = DirectionNthSelector((0, 0, 1), 0, directionMax=True).filter(shapes)
        self.assertEqual([s.name for s in out], ["z0"])


class TestBox(unittest.TestCase):
    def test_center_mode(self):
        shapes = [
            Shape("Vertex", (0.5, 0.5, 0.5), name="in"),
            Shape("Vertex", (5, 5, 5), name="out"),
        ]
        out = BoxSelector((0, 0, 0), (1, 1, 1)).filter(shapes)
        self.assertEqual([s.name for s in out], ["in"])

    def test_reversed_corners(self):
        shapes = [Shape("Vertex", (0.5, 0.5, 0.5), name="in")]
        out = BoxSelector((1, 1, 1), (0, 0, 0)).filter(shapes)
        self.assertEqual([s.name for s in out], ["in"])

    def test_boundingbox_mode(self):
        shapes = [
            Shape("Solid", bbox=((0.1, 0.1, 0.1), (0.9, 0.9, 0.9)), name="inside"),
            Shape("Solid", bbox=((0.1, 0.1, 0.1), (2.0, 0.9, 0.9)), name="poking"),
        ]
        out = BoxSelector((0, 0, 0), (1, 1, 1), boundingbox=True).filter(shapes)
        self.assertEqual([s.name for s in out], ["inside"])


class TestNearest(unittest.TestCase):
    def test_nearest(self):
        shapes = [
            Shape("Vertex", (0, 0, 0), name="a"),
            Shape("Vertex", (10, 0, 0), name="b"),
        ]
        out = NearestToPointSelector((9, 0, 0)).filter(shapes)
        self.assertEqual([s.name for s in out], ["b"])


class TestAlgebra(unittest.TestCase):
    def test_and(self):
        faces = cube_faces()
        # parallel Z AND max Z -> just top
        sel = ParallelDirSelector((0, 0, 1)) & DirectionMinMaxSelector((0, 0, 1))
        out = sel.filter(faces)
        self.assertEqual([s.name for s in out], ["top"])
        self.assertIsInstance(sel, AndSelector)

    def test_sum_union_order(self):
        faces = cube_faces()
        sel = DirectionMinMaxSelector((0, 0, 1)) + DirectionMinMaxSelector((1, 0, 0))
        out = {s.name for s in sel.filter(faces)}
        self.assertEqual(out, {"top", "right"})
        self.assertIsInstance(sel, SumSelector)

    def test_subtract(self):
        faces = cube_faces()
        sel = ParallelDirSelector((0, 0, 1)) - DirectionMinMaxSelector((0, 0, 1))
        out = [s.name for s in sel.filter(faces)]
        self.assertEqual(out, ["bot"])
        self.assertIsInstance(sel, SubtractSelector)

    def test_inverse(self):
        faces = cube_faces()
        sel = -DirectionMinMaxSelector((0, 0, 1))
        out = {s.name for s in sel.filter(faces)}
        self.assertEqual(out, {"bot", "right", "left", "front", "back"})
        self.assertIsInstance(sel, InverseSelector)

    def test_union_no_duplicates(self):
        faces = cube_faces()
        sel = DirectionMinMaxSelector((0, 0, 1)) + DirectionMinMaxSelector((0, 0, 1))
        out = [s.name for s in sel.filter(faces)]
        self.assertEqual(out, ["top"])

    def test_base_selector_identity(self):
        faces = cube_faces()
        self.assertEqual(len(Selector().filter(faces)), 6)


if __name__ == "__main__":
    unittest.main()
