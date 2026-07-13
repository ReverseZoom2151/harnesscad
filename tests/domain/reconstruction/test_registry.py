"""The reconstruction route registry: discovery, real routes, and rival safety."""

import unittest

from harnesscad.domain.reconstruction import registry as R


# Three orthographic views of a 10 x 5 plate (the pipeline's own fixture shape).
SVG = """<svg xmlns="http://www.w3.org/2000/svg">
<g id="front"><path id="f" d="M 0 0 H 10 V 5 H 0 Z"/></g>
<g data-view="bottom"><line id="b" x1="0" y1="0" x2="10" y2="0"/></g>
<g id="left"><circle id="l" cx="0" cy="0" r="2"/></g>
</svg>"""


def box_cloud(lo=(-1.0, -2.0, 0.0), hi=(3.0, 2.0, 5.0), n=3):
    """A deterministic point cloud sampled on the surface of an axis-aligned box."""
    pts = []
    for i in range(n + 1):
        for j in range(n + 1):
            u = i / n
            v = j / n
            x = lo[0] + u * (hi[0] - lo[0])
            y = lo[1] + v * (hi[1] - lo[1])
            z = lo[2] + u * (hi[2] - lo[2])
            pts.append((x, y, lo[2]))
            pts.append((x, y, hi[2]))
            pts.append((x, lo[1], z))
            pts.append((x, hi[1], z))
            pts.append((lo[0], y, z))
            pts.append((hi[0], y, z))
    return pts


class DiscoveryTests(unittest.TestCase):
    def test_registry_discovers_many_real_modules(self):
        routes = R.routes()
        modules = {d for r in routes for d in r.dotted}
        self.assertGreater(len(routes), 10)
        self.assertGreater(len(modules), 10)
        # Every module a route claims is a real, indexed module.
        from harnesscad import registry as capability_registry

        indexed = {e.dotted for e in capability_registry.index()}
        for dotted in modules:
            self.assertIn(dotted, indexed)

    def test_routes_are_keyed_by_input_and_output_kind(self):
        # "what can turn a point cloud into CAD primitives?" -- a real answer.
        answers = R.routes_for("point_cloud", "primitives")
        self.assertTrue(answers)
        self.assertIn("fit.bbox_primitive", [r.name for r in answers])
        for r in answers:
            self.assertIn("point_cloud", r.inputs)
            self.assertEqual(r.output, "primitives")
        self.assertIn("point_cloud", R.inputs())
        self.assertIn("primitives", R.outputs())

    def test_discovery_is_deterministic_and_sorted(self):
        names = [r.name for r in R.routes()]
        self.assertEqual(names, sorted(names))
        self.assertEqual(names, [r.name for r in R.routes()])

    def test_unknown_route_and_unknown_kind_raise(self):
        with self.assertRaises(R.UnknownRoute):
            R.route("no.such.route")
        with self.assertRaises(R.RouteError):
            R.routes_for("banana")

    def test_unadapted_is_reported_not_hidden(self):
        # The registry is honest about what it has not routed.
        self.assertIsInstance(R.unadapted(), tuple)


class PointCloudToPrimitiveTests(unittest.TestCase):
    """A real reconstruction route, end to end, with a checked answer."""

    def test_point_cloud_fits_a_primitive_with_the_right_size(self):
        lo, hi = (-1.0, -2.0, 0.0), (3.0, 2.0, 5.0)
        result = R.run("fit.bbox_primitive", points=box_cloud(lo, hi))

        prim = result["primitives"][0]
        self.assertEqual(prim["shape"], "cube")
        # The recovered size is the box's true extent...
        self.assertEqual(prim["size"], (4.0, 4.0, 5.0))
        # ...centred on the footprint, sitting on the cloud's floor.
        self.assertEqual(prim["position"], (1.0, 0.0, 0.0))
        # ...and the reconstructed mesh reproduces the cloud's bounding box.
        fit_lo, fit_hi = result["mesh"].bounding_box()
        for i in range(3):
            self.assertAlmostEqual(fit_lo[i], lo[i])
            self.assertAlmostEqual(fit_hi[i], hi[i])
        self.assertAlmostEqual(result["residual"], 0.0)
        self.assertEqual(result["method"], "axis-aligned bounding-box fit")

    def test_fit_is_deterministic(self):
        a = R.run("fit.bbox_primitive", points=box_cloud())
        b = R.run("fit.bbox_primitive", points=box_cloud())
        self.assertEqual(a["primitives"], b["primitives"])

    def test_empty_cloud_is_refused_not_guessed(self):
        with self.assertRaises(R.RouteError):
            R.run("fit.bbox_primitive", points=[])

    def test_sampling_uses_the_cadrille_point_adapter(self):
        result = R.run("fit.bbox_primitive", points=box_cloud(), sample=8)
        self.assertEqual(result["n_points"], 8)


class DrawingToTopologyTests(unittest.TestCase):
    """The SVG / orthographic route runs end to end and reports every stage."""

    def test_svg_reconstructs_to_edges_and_stages(self):
        result = R.run("ortho.reconstruct", svg=SVG)
        stages = [name for name, _n in result["reports"]]
        self.assertEqual(stages, ["parse", "normalize", "match_edges",
                                  "detect_loops", "cluster_faces",
                                  "manifold_gate", "stitch"])
        self.assertEqual(result["stitch"], "unavailable")
        self.assertFalse([d for d in result["diagnostics"] if d.severity == "error"])

    def test_a_mesh_cross_section_produces_a_contour_that_extrudes_to_a_solid(self):
        # mesh -> contour2d -> solid: two routes composed, both real.
        triangles = [
            ((0.0, 0.0, -1.0), (4.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
            ((4.0, 0.0, -1.0), (4.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        ]
        section = R.run("ingest.cross_section", triangles=triangles,
                        origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        self.assertTrue(section["contours"])

        solid = R.run("fit.extrude_contour",
                      contour=[(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)],
                      depth=3.0)
        self.assertAlmostEqual(solid["volume"], 24.0)


class RivalTests(unittest.TestCase):
    """Rivals are exposed by name, selected explicitly, and never merged."""

    def test_every_rival_family_is_named_and_documented(self):
        families = R.rivals()
        for expected in ("token_family", "brep_graph_encoding",
                         "canonical_sketch_ordering"):
            self.assertIn(expected, families)
            self.assertGreaterEqual(len(families[expected]), 2)
            self.assertTrue(R.rival_doc(expected))

    def test_selecting_a_rival_from_the_wrong_family_raises(self):
        self.assertEqual(R.select("brep_graph_encoding", "brep.graph.uvnet"),
                         "brep.graph.uvnet")
        with self.assertRaises(R.RivalMismatch) as ctx:
            R.select("brep_graph_encoding", "sketch.order.gencad")
        self.assertIn("canonical_sketch_ordering", str(ctx.exception))
        with self.assertRaises(R.RivalMismatch):
            R.select("brep_graph_encoding", "not.a.member")
        with self.assertRaises(R.RivalMismatch):
            R.select("no_such_family", "brep.graph.uvnet")

    def test_the_three_brep_graph_encodings_stay_separate_routes(self):
        members = R.rivals()["brep_graph_encoding"]
        routes = {name: R.route(name) for name in members}
        # Each rival is its own route, over its own module, taking its own input.
        self.assertEqual(len(routes), 3)
        self.assertEqual(len({r.dotted for r in routes.values()}), 3)
        self.assertEqual(len({r.inputs for r in routes.values()}), 3)
        for r in routes.values():
            self.assertEqual(r.family, "brep_graph_encoding")
        # There is no merged/blended encoding on offer.
        graph_routes = [r.name for r in R.routes(family="brep_graph_encoding")]
        self.assertEqual(sorted(graph_routes), sorted(members))

    def test_the_cadparser_encoding_runs_and_is_categorical(self):
        brep = {
            "faces": [
                {"id": "f1", "surface_type": "plane", "area": 4.0,
                 "loops": [[("e1", True), ("e2", True)]]},
                {"id": "f2", "surface_type": "cylinder", "area": 6.0,
                 "loops": [[("e1", False), ("e2", False)]]},
            ],
            "edges": [{"id": "e1", "curve_type": "line", "length": 2.0},
                      {"id": "e2", "curve_type": "circle", "length": 3.0}],
        }
        result = R.run("brep.graph.cadparser", **brep)
        self.assertEqual(result["encoding"], "cadparser")
        # faces + edges + coedges are all nodes -- the CADParser choice.
        self.assertEqual(result["n_nodes"], 2 + 2 + 4)
        self.assertTrue(result["node_features"])

    def test_the_graphbrep_encoding_runs_and_is_permutation_canonical(self):
        a = ((0, 1, 0), (1, 0, 1), (0, 1, 0))
        # The same graph under a node relabelling.
        b = ((0, 0, 1), (0, 0, 1), (1, 1, 0))
        ka = R.run("brep.graph.graphbrep", matrix=a)["key"]
        kb = R.run("brep.graph.graphbrep", matrix=b)["key"]
        self.assertEqual(ka, kb)      # canonical: relabelling does not matter

    def test_the_uvnet_encoding_runs_on_sampled_geometry(self):
        from harnesscad.domain.geometry.parametric.curve_grid import Line
        from harnesscad.domain.geometry.parametric.surface_grid import Plane

        faces = [{"surface": Plane(origin=(0.0, 0.0, 0.0)), "name": "a"},
                 {"surface": Plane(origin=(0.0, 0.0, 1.0)), "name": "b"}]
        edges = [{"curve": Line(origin=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0)),
                  "faces": (0, 1), "name": "e"}]
        result = R.run("brep.graph.uvnet", faces=faces, edges=edges)
        self.assertEqual(result["encoding"], "uvnet")
        self.assertEqual(result["n_nodes"], 2)
        self.assertTrue(result["connected"])

    def test_rival_encodings_take_incompatible_inputs(self):
        # Handing the cadparser payload to the graphbrep route is refused, not
        # coerced: they are different encodings of different things.
        with self.assertRaises(TypeError):
            R.run("brep.graph.graphbrep", faces=[], edges=[])

    def test_the_token_families_are_still_refused_across_decoders(self):
        from harnesscad.domain.reconstruction import ingest_pipeline as ip

        tokens = ip.TokenSequence("vitruvion", {"val": [0], "num_bins": 64})
        with self.assertRaises(ip.FamilyMismatch):
            R.run("tokens.to_cisp", tokens=tokens, family="deepcad")

    def test_the_canonical_orderings_are_three_separate_routes(self):
        members = R.rivals()["canonical_sketch_ordering"]
        for name in members:
            self.assertEqual(R.route(name).family, "canonical_sketch_ordering")
        self.assertEqual(len({R.route(n).dotted for n in members}), len(members))


if __name__ == "__main__":
    unittest.main()
