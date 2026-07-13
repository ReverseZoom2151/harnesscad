import unittest

from harnesscad.domain.reconstruction.cmt_topology_validity import (
    check_adjacency, check_edge_geometry, is_valid,
    quantized_is_valid, valid_ratio,
)


def cube_adjacency():
    # 12 edges, 6 surfaces, each edge shared by exactly 2 faces,
    # each face bounded by 4 edges -> valid manifold.
    faces = [
        (0, 1, 2, 3),      # bottom
        (4, 5, 6, 7),      # top
        (0, 8, 4, 9),      # side
        (1, 9, 5, 10),
        (2, 10, 6, 11),
        (3, 11, 7, 8),
    ]
    adjacency = []
    for e in range(12):
        adjacency.append(tuple(e in face for face in faces))
    return tuple(adjacency)


class TestCheckAdjacency(unittest.TestCase):
    def test_valid_cube(self):
        ok, diags = check_adjacency(cube_adjacency())
        self.assertTrue(ok)
        self.assertEqual(diags, ())

    def test_unbounded_open_region(self):
        # a surface (column) with no incident edge
        adj = ((True, False), (True, False))
        ok, diags = check_adjacency(adj)
        self.assertFalse(ok)
        self.assertTrue(any(d.code == "unbounded-open-region" for d in diags))

    def test_dangling_edge(self):
        # edge bounds only 1 surface
        adj = ((True, True), (True, False))
        ok, diags = check_adjacency(adj)
        self.assertFalse(ok)
        self.assertTrue(any(d.code == "dangling-edge" for d in diags))

    def test_over_shared_edge(self):
        adj = ((True, True, True), (True, True, False))  # row0 sums to 3
        ok, diags = check_adjacency(adj)
        self.assertFalse(ok)
        self.assertTrue(any(d.code == "over-shared-edge" for d in diags))

    def test_empty_ok(self):
        ok, diags = check_adjacency(())
        self.assertTrue(ok)

    def test_ragged_raises(self):
        with self.assertRaises(ValueError):
            check_adjacency(((True, False), (True,)))


class TestEdgeGeometry(unittest.TestCase):
    def test_zero_length(self):
        ok, diags = check_edge_geometry((((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),))
        self.assertFalse(ok)
        self.assertTrue(any(d.code == "degenerate-edge" for d in diags))

    def test_duplicate_edges(self):
        e = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        rev = ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        ok, diags = check_edge_geometry((e, rev))
        self.assertFalse(ok)
        self.assertTrue(any(d.code == "self-intersecting-edge" for d in diags))

    def test_clean(self):
        ok, _ = check_edge_geometry((
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ))
        self.assertTrue(ok)


class TestIsValid(unittest.TestCase):
    def test_valid_model(self):
        self.assertTrue(is_valid(cube_adjacency()))

    def test_invalid_with_geometry(self):
        adj = cube_adjacency()
        bad_edges = tuple(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) for _ in range(12))
        self.assertFalse(is_valid(adj, bad_edges))


class TestQuantizedValidity(unittest.TestCase):
    def test_collapse_under_quantization(self):
        # two endpoints closer than one 4-bit level collapse to zero length
        adj = ((True, True),)
        edges = (((0.0, 0.0, 0.0), (0.01, 0.0, 0.0)),)
        self.assertFalse(quantized_is_valid(adj, edges, bits=4))

    def test_survives_quantization(self):
        adj = ((True, True),)
        edges = (((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),)
        self.assertTrue(quantized_is_valid(adj, edges, bits=4))


class TestValidRatio(unittest.TestCase):
    def test_ratio(self):
        good = (cube_adjacency(), None)
        bad = (((True, False), (True, False)), None)
        self.assertAlmostEqual(valid_ratio((good, bad, good)), 2 / 3)

    def test_empty(self):
        self.assertEqual(valid_ratio(()), 0.0)


if __name__ == "__main__":
    unittest.main()
