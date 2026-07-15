"""picks — the selector->concrete-entity resolution, pure (no GUI, no model).

A synthetic box topology (the exact shape viewport.ViewportController.topology
emits) is enough to prove that a CISP selector picks out the right edges and
faces. The live half — the computed click adjudicated by the app — is exercised
in test_viewport.py's corpus and behind HARNESSCAD_CUA_LIVE=1.
"""

import math
import unittest

from harnesscad.io.cua import picks
from harnesscad.io.cua import viewport as vp


def _sample(a, b, n=5):
    return [[a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
             a[2] + (b[2] - a[2]) * t] for t in (0.5, 0.35, 0.65, 0.2, 0.8)]


def box_topology(lx=30.0, ly=20.0, lz=10.0):
    """A corner-at-origin box as viewport topology: 6 faces, 12 edges, 8 verts."""
    corners = {
        (0, 0, 0): (0, 0, 0), (1, 0, 0): (lx, 0, 0), (1, 1, 0): (lx, ly, 0),
        (0, 1, 0): (0, ly, 0), (0, 0, 1): (0, 0, lz), (1, 0, 1): (lx, 0, lz),
        (1, 1, 1): (lx, ly, lz), (0, 1, 1): (0, ly, lz),
    }
    edge_pairs = [
        # 4 vertical (parallel Z)
        ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)),
        ((1, 1, 0), (1, 1, 1)), ((0, 1, 0), (0, 1, 1)),
        # 4 along X
        ((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)),
        ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
        # 4 along Y
        ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)),
        ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
    ]
    edges = []
    for i, (a, b) in enumerate(edge_pairs):
        pa, pb = corners[a], corners[b]
        c = [(pa[k] + pb[k]) / 2 for k in range(3)]
        edges.append({"name": "Edge%d" % (i + 1), "kind": "edge", "surface": "Line",
                      "length": 1.0, "centroid": c, "points": _sample(pa, pb)})
    faces = [
        {"name": "Face1", "surface": "Plane", "normal": [0, 0, -1],
         "centroid": [lx / 2, ly / 2, 0], "area": lx * ly, "points": [[lx / 2, ly / 2, 0]]},
        {"name": "Face2", "surface": "Plane", "normal": [0, 0, 1],
         "centroid": [lx / 2, ly / 2, lz], "area": lx * ly, "points": [[lx / 2, ly / 2, lz]]},
        {"name": "Face3", "surface": "Plane", "normal": [-1, 0, 0],
         "centroid": [0, ly / 2, lz / 2], "area": ly * lz, "points": [[0, ly / 2, lz / 2]]},
        {"name": "Face4", "surface": "Plane", "normal": [1, 0, 0],
         "centroid": [lx, ly / 2, lz / 2], "area": ly * lz, "points": [[lx, ly / 2, lz / 2]]},
        {"name": "Face5", "surface": "Plane", "normal": [0, -1, 0],
         "centroid": [lx / 2, 0, lz / 2], "area": lx * lz, "points": [[lx / 2, 0, lz / 2]]},
        {"name": "Face6", "surface": "Plane", "normal": [0, 1, 0],
         "centroid": [lx / 2, ly, lz / 2], "area": lx * lz, "points": [[lx / 2, ly, lz / 2]]},
    ]
    return {"faces": faces, "edges": edges, "vertices": [],
            "bbox": [0, 0, 0, lx, ly, lz]}


class TestEdgeTangent(unittest.TestCase):
    def test_vertical_edge_tangent_is_z(self):
        t = picks._edge_tangent(_sample((0, 0, 0), (0, 0, 10)))
        self.assertAlmostEqual(abs(t[2]), 1.0, places=6)
        self.assertAlmostEqual(t[0], 0.0, places=6)
        self.assertAlmostEqual(t[1], 0.0, places=6)


class TestResolve(unittest.TestCase):
    def setUp(self):
        self.topo = box_topology()

    def test_parallel_z_selects_the_four_vertical_edges(self):
        names = picks.resolve("|Z", self.topo, ("edge",))
        self.assertEqual(set(names), {"Edge1", "Edge2", "Edge3", "Edge4"})

    def test_empty_selector_is_every_edge(self):
        names = picks.resolve("", self.topo, ("edge",))
        self.assertEqual(len(names), 12)

    def test_top_face_selector(self):
        names = picks.resolve(">Z", self.topo, ("face",))
        self.assertEqual(names, ["Face2"])  # normal +Z, highest in Z

    def test_parallel_x_selects_the_four_x_edges(self):
        names = picks.resolve("|X", self.topo, ("edge",))
        self.assertEqual(set(names), {"Edge5", "Edge6", "Edge7", "Edge8"})


class TestResolveOpEdges(unittest.TestCase):
    def test_fillet_vertical_edges(self):
        from harnesscad.core.cisp.ops import Fillet
        names = picks.resolve_op_edges(Fillet(edges=("|Z",), radius=3.0), box_topology())
        self.assertEqual(set(names), {"Edge1", "Edge2", "Edge3", "Edge4"})

    def test_fillet_empty_is_all_edges(self):
        from harnesscad.core.cisp.ops import Fillet
        names = picks.resolve_op_edges(Fillet(edges=(), radius=1.0), box_topology())
        self.assertEqual(len(names), 12)


# --------------------------------------------------------------------------
# The computed pick, exercised offline with the REAL projection.
#
# There is no GUI here. The controller below subclasses the real
# ViewportController and stubs ONLY the methods that would touch the app's Python
# bridge: the camera, the topology, the named-view set, and the picker. Every
# other line -- crucially ViewportController.adjudicate and OrthoCamera.project --
# is the production code, so this proves the pick PROJECTION maths and the
# adjudication/read-back logic together, without a live picker.
#
# The stub picker models FreeCAD's SoRayPickAction as: the entity whose projected
# candidate is nearest the queried pixel AND frontmost (max camera-space depth)
# wins, i.e. depth-ordered occlusion. That is exactly the property the real
# adjudicator relies on, so a pick the app would refuse (an occluded face) is
# refused here too.
# --------------------------------------------------------------------------
def _top_view_camera():
    """A verbatim top-down orthographic camera over the 30x20x10 box.

    Identity orientation: camera axes == world axes, so it looks down world -Z
    with +Y up. Square 400x400 viewport, view height 44 mm -> half-extents 22 mm,
    which comfortably contains the 30x20 footprint centred at (15, 10)."""
    return vp.OrthoCamera(position=(15.0, 10.0, 100.0),
                          orientation=(0.0, 0.0, 0.0, 1.0),
                          height=44.0, width_px=400, height_px=400)


class OfflineViewportController(vp.ViewportController):
    """Real projection + adjudication; a simulated, depth-ordered picker."""

    PICK_RADIUS = 3.0

    def __init__(self, topo, camera):
        super().__init__(bridge=None, obj_name="Model")
        self._topology = topo
        self._camera = camera
        self.view = None
        self._sel = set()
        self.misclick = set()          # entities a real click "steals" to nothing
        # Precompute every candidate's projected pixel + depth, once.
        self._catalog = []             # (name, px, py, depth)
        for ent in self.entities(("edge", "face", "vertex")):
            for point in ent.candidates:
                px, py, depth = camera.project(point)
                self._catalog.append((ent.name, px, py, depth))

    # -- stubbed bridge-touching surface -----------------------------------
    def topology(self, refresh=False):
        return self._topology

    def camera(self):
        return self._camera

    def set_named_view(self, view, tries=40, poll=0.05):
        self.view = view             # no orbit, no bridge; just record it

    def _under(self, px, py):
        """The frontmost entity within the pick radius of a viewport-local pixel."""
        best_name, best_depth = None, -1e18
        for name, cx, cy, depth in self._catalog:
            if math.hypot(cx - px, cy - py) <= self.PICK_RADIUS and depth > best_depth:
                best_name, best_depth = name, depth
        return best_name

    def pick(self, points):
        return [self._under(px, py) for px, py in points]

    # -- the real-mouse read-back, simulated -------------------------------
    def clear_selection(self):
        self._sel = set()

    def focus_window(self, settle=0.5):
        pass

    def viewport_rect(self):
        return (0, 0, 400, 400)

    def selection(self):
        return ["Model.%s" % n for n in sorted(self._sel)]

    def mouse_click(self, pick, camera, rect=None, settle=0.35):
        # A real plain click clears then selects what is under the pixel (matching
        # viewport.mouse_click's clear-then-click). A "misclick" entity is stolen
        # by the pick radius and selects nothing -- the honest failure mode.
        self.clear_selection()
        if pick.entity not in self.misclick:
            self._sel = {pick.entity}
        return self.selection()


class TestComputedPickOffline(unittest.TestCase):
    def setUp(self):
        self.topo = box_topology()
        self.ctl = OfflineViewportController(self.topo, _top_view_camera())

    def test_projection_maps_a_vertical_edge_to_its_corner_pixel(self):
        # Edge1 is the vertical edge at the (0,0) corner; from straight above every
        # sample projects to the SAME pixel (only z changes, and z is depth).
        cam = self.ctl.camera()
        ent = {e.name: e for e in self.ctl.entities(("edge",))}["Edge1"]
        pxs = {tuple(round(v, 4) for v in cam.project(p)[:2]) for p in ent.candidates}
        self.assertEqual(len(pxs), 1, "a top-view vertical edge is one pixel")

    def test_four_vertical_edges_verify_without_a_real_mouse(self):
        names = picks.resolve("|Z", self.topo, ("edge",))
        out = picks.pick_entities(self.ctl, names, view="top", use_real_mouse=False)
        self.assertTrue(out.verified, out.reason)
        self.assertEqual(set(out.selected), {"Edge1", "Edge2", "Edge3", "Edge4"})
        self.assertEqual(self.ctl.view, "top")     # named view, never an orbit

    def test_top_face_is_verifiable_from_above(self):
        names = picks.resolve(">Z", self.topo, ("face",))
        out = picks.pick_entities(self.ctl, names, view="top", use_real_mouse=False)
        self.assertTrue(out.verified, out.reason)
        self.assertEqual(out.selected, ["Face2"])

    def test_the_occluded_bottom_face_is_discarded_with_a_reason(self):
        # From directly above, the bottom face sits behind the top face at the same
        # pixel: the app's picker returns the top face, so the bottom is not a pick.
        out = picks.pick_entities(self.ctl, ["Face1"], view="top",
                                  use_real_mouse=False)
        self.assertFalse(out.verified)
        self.assertIn("Face1", out.reason)
        self.assertEqual(out.selected, [])
        occluded = [r for r in out.per_entity if not r["computed"]]
        self.assertTrue(occluded and "occluded" in occluded[0]["reason"])

    def test_real_mouse_read_back_confirms_the_selection(self):
        names = picks.resolve("|Z", self.topo, ("edge",))
        out = picks.pick_entities(self.ctl, names, view="top", use_real_mouse=True)
        self.assertTrue(out.verified, out.reason)
        self.assertEqual(set(out.selected), {"Edge1", "Edge2", "Edge3", "Edge4"})
        self.assertTrue(all(r.get("selected_now") for r in out.per_entity))

    def test_a_click_the_app_does_not_confirm_is_not_a_selection(self):
        # The pixel is computed and the picker agrees, but the synthesised click is
        # stolen by the pick radius: read-back shows nothing, so it does NOT verify.
        self.ctl.misclick = {"Edge3"}
        names = picks.resolve("|Z", self.topo, ("edge",))
        out = picks.pick_entities(self.ctl, names, view="top", use_real_mouse=True)
        self.assertFalse(out.verified)
        self.assertIn("Edge3", out.reason)
        self.assertNotIn("Edge3", out.selected)


class TestPickOpEdgesUnlock(unittest.TestCase):
    """The whole unlock in one call: a refused Fillet op -> verified edges."""

    def setUp(self):
        self.ctl = OfflineViewportController(box_topology(), _top_view_camera())

    def test_fillet_selectors_become_a_verified_edge_selection(self):
        from harnesscad.core.cisp.ops import Fillet
        op = Fillet(edges=("|Z",), radius=3.0)
        out = picks.pick_op_edges(self.ctl, op, view="top", use_real_mouse=False)
        self.assertTrue(out.verified, out.reason)
        self.assertEqual(set(out.selected), {"Edge1", "Edge2", "Edge3", "Edge4"})

    def test_a_selector_that_denotes_no_edge_is_reported_honestly(self):
        from harnesscad.core.cisp.ops import Fillet
        # A cube-only selector against the box denotes nothing here.
        op = Fillet(edges=(">Z and |Z",), radius=1.0)
        out = picks.pick_op_edges(self.ctl, op, view="top", use_real_mouse=False)
        self.assertEqual(out.selected, [])
        self.assertFalse(out.verified)


if __name__ == "__main__":
    unittest.main()
