import unittest

from harnesscad.domain.reconstruction.edges import normalize_edges, projection_feature
from harnesscad.domain.reconstruction.metrics import edge_prf, face_prf
from harnesscad.domain.reconstruction.model import Edge2D, Edge3D, FaceCluster, FaceLoop, OrthographicInput, View2D
from harnesscad.domain.reconstruction.patterns import PATTERNS, match_patterns
from harnesscad.domain.reconstruction.pipeline import reconstruct
from harnesscad.domain.reconstruction.svg import parse_svg, validate_input
from harnesscad.domain.reconstruction.topology import (
    cluster_planar_loops, find_face_loops, manifold_gate, wireframe_graph,
)


SVG = """<svg xmlns="http://www.w3.org/2000/svg">
<g id="front"><path id="f" d="M 0 0 H 10 V 5 H 0 Z"/></g>
<g data-view="bottom"><line id="b" x1="0" y1="0" x2="10" y2="0"/></g>
<g id="left"><circle id="l" cx="0" cy="0" r="2"/></g>
</svg>"""


class InputAndSvgTests(unittest.TestCase):
    def test_safe_svg_parser_and_basic_paths(self):
        drawing, diagnostics = parse_svg(SVG)
        self.assertIsNotNone(drawing)
        self.assertFalse([d for d in diagnostics if d.severity == "error"])
        self.assertEqual(len(drawing.views["front"].edges), 4)
        self.assertEqual(drawing.views["left"].edges[0].kind, "circle")
        self.assertEqual(len(drawing.views["left"].edges[0].points), 3)

    def test_rejects_active_or_external_svg(self):
        for body in ("<script/>", "<use href='https://example.test/a'/>",
                     "<g transform='scale(2)'/>"):
            drawing, diagnostics = parse_svg(f"<svg>{body}</svg>")
            self.assertIsNone(drawing)
            self.assertEqual(diagnostics[0].code, "unsafe-svg")

    def test_contract_reports_scale_tolerance_and_views(self):
        drawing = OrthographicInput(
            {"front": View2D("front", ())}, scale=0, tolerance=-1)
        codes = {item.code for item in validate_input(drawing)}
        self.assertEqual(codes, {"invalid-scale", "invalid-tolerance",
                                 "empty-view", "missing-view"})


class NormalizationTests(unittest.TestCase):
    def test_feature_classification(self):
        self.assertEqual(projection_feature(
            Edge2D("front", "line", ((0, 0), (2, 0))), .01), "H")
        self.assertEqual(projection_feature(
            Edge2D("front", "line", ((0, 0), (0, 2))), .01), "V")
        self.assertEqual(projection_feature(
            Edge2D("front", "line", ((0, 0), (2, 2))), .01), "I")
        self.assertEqual(projection_feature(
            Edge2D("front", "circle", ((1, 0), (0, 1), (-1, 0))), .01), "A")

    def test_deduplicates_and_merges_three_segments(self):
        edges = [
            Edge2D("front", "line", ((0, 0), (1, 0)), source_id="a"),
            Edge2D("front", "line", ((1, 0), (2, 0)), source_id="b"),
            Edge2D("front", "line", ((2, 0), (3, 0)), source_id="c"),
            Edge2D("front", "line", ((1, 0), (0, 0)), source_id="duplicate"),
        ]
        result = normalize_edges(edges, 1e-6)
        self.assertEqual(len(result), 1)
        self.assertEqual(set((result[0].start, result[0].end)), {(0, 0), (3, 0)})


class PatternTests(unittest.TestCase):
    def test_all_fourteen_named_patterns_are_explicit(self):
        self.assertEqual(set(PATTERNS), {
            "L1", "L2", "L3", "L4", "L5", "L6", "L7",
            "C1", "C2", "C3", "C4", "C5", "C6", "C7",
        })

    def test_matches_l7_and_reports_duplicate_ambiguity(self):
        front = [
            Edge2D("front", "line", ((0, 0), (2, 2)), source_id="f1"),
            Edge2D("front", "line", ((0, 0), (2, 2)), source_id="f2"),
        ]
        bottom = [Edge2D("bottom", "line", ((0, 0), (2, 2)), source_id="b")]
        left = [Edge2D("left", "line", ((0, 0), (2, 2)), source_id="l")]
        result, diagnostics = match_patterns(
            {"front": front, "bottom": bottom, "left": left}, 1e-6)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].pattern, "L7")
        self.assertEqual(result[0].start, (0, 0, 0))
        self.assertEqual(result[0].end, (2, 2, 2))
        self.assertEqual(diagnostics[0].code, "ambiguous-edge-match")


def cube_edges():
    p = [(x, y, z) for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)]
    return tuple(Edge3D(a, b) for i, a in enumerate(p)
                 for b in p[i + 1:] if sum(a[j] != b[j] for j in range(3)) == 1)


class TopologyTests(unittest.TestCase):
    def test_wireframe_graph_and_cube_face_loops_are_deterministic(self):
        edges = cube_edges()
        _, graph = wireframe_graph(edges, 1e-6)
        self.assertEqual(len(graph), 8)
        self.assertTrue(all(len(neighbors) == 3 for neighbors in graph.values()))
        a = find_face_loops(edges, 1e-6)
        b = find_face_loops(tuple(reversed(edges)), 1e-6)
        self.assertEqual(len(a), 6)
        self.assertEqual([loop.vertices for loop in a],
                         [loop.vertices for loop in b])
        faces = cluster_planar_loops(a, 1e-6)
        ok, diagnostics = manifold_gate(faces, len(edges))
        self.assertTrue(ok, diagnostics)

    def test_clusters_nested_planar_loop_as_inner_trim(self):
        def loop(points, indices):
            return FaceLoop(tuple(points), tuple(indices), (0, 0, 1, 0))
        outer = loop(((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)), (0, 1, 2, 3))
        inner = loop(((1, 1, 0), (2, 1, 0), (2, 2, 0), (1, 2, 0)), (4, 5, 6, 7))
        faces = cluster_planar_loops((inner, outer), 1e-6)
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].inner, (inner,))

    def test_manifold_gate_reports_incidence(self):
        loop = FaceLoop(((0, 0, 0), (1, 0, 0), (0, 1, 0)), (0, 1, 2),
                        (0, 0, 1, 0))
        ok, diagnostics = manifold_gate((FaceCluster(loop),), 3)
        self.assertFalse(ok)
        self.assertEqual(diagnostics[0].code, "non-manifold-edge-incidence")


class MetricAndPipelineTests(unittest.TestCase):
    def test_coordinate_tolerant_edge_and_topology_face_prf(self):
        expected = [Edge3D((0, 0, 0), (1, 0, 0))]
        actual = [Edge3D((1.0000001, 0, 0), (0, 0, 0))]
        self.assertEqual(edge_prf(actual, expected, 1e-5)["f1"], 1)
        loop = FaceLoop(((0, 0, 0),) * 3, (2, 1, 0), (0, 0, 1, 0))
        self.assertEqual(face_prf([FaceCluster(loop)], [FaceCluster(loop)])["f1"], 1)

    def test_pipeline_has_all_reports_and_structured_stitch_status(self):
        result = reconstruct(SVG)
        self.assertEqual([stage.name for stage in result.reports], [
            "parse", "normalize", "match_edges", "detect_loops",
            "cluster_faces", "manifold_gate", "stitch",
        ])
        self.assertEqual(result.stitch.status, "unavailable")

    def test_stitch_adapter_failure_is_structured(self):
        def fail(*_):
            raise RuntimeError("kernel offline")
        result = reconstruct(SVG, stitcher=fail)
        self.assertEqual(result.stitch.status, "failed")
        self.assertIn("kernel offline", result.stitch.message)


if __name__ == "__main__":
    unittest.main()
