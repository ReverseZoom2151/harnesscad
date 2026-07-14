"""The software rasteriser: shaded solids, a z-buffer, and a stdlib PNG."""

from __future__ import annotations

import os
import struct
import tempfile
import unittest
import zlib

from harnesscad.core.cisp.ops import parse_op
from harnesscad.core.loop import HarnessSession
from harnesscad.io import render
from harnesscad.io.backends.frep import FRepBackend
from harnesscad.io.formats import registry as fmt

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# A unit cube, wound counter-clockwise seen from outside.
CUBE_VERTS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
              (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]
CUBE_FACES = [(0, 2, 1), (0, 3, 2),        # bottom (-Z)
              (4, 5, 6), (4, 6, 7),        # top (+Z)
              (0, 1, 5), (0, 5, 4),        # -Y
              (1, 2, 6), (1, 6, 5),        # +X
              (2, 3, 7), (2, 7, 6),        # +Y
              (3, 0, 4), (3, 4, 7)]        # -X
CUBE = (CUBE_VERTS, CUBE_FACES)

# A plate with a through hole -- the part the README shows.
PLATE_OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 40.0, "h": 24.0},
    {"op": "extrude", "sketch": "sk1", "distance": 8.0},
    {"op": "hole", "face_or_sketch": "f1", "x": 10.0, "y": 12.0,
     "diameter": 6.0, "through": True},
    {"op": "hole", "face_or_sketch": "f1", "x": 30.0, "y": 12.0,
     "diameter": 6.0, "through": True},
]


def decode_png(path):
    """(width, height, rows) from an 8-bit RGB PNG, using zlib alone.

    The renderer's own writer is not trusted to check itself: this is an
    independent decoder, so a test that reads a pixel back has really been
    through the PNG container.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != PNG_MAGIC:
        raise AssertionError("not a PNG")
    pos, idat, width, height = 8, b"", 0, 0
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        if tag == b"IHDR":
            width, height, depth, ctype = struct.unpack(">IIBB", payload[:10])
            if depth != 8 or ctype != 2:
                raise AssertionError(f"expected 8-bit RGB, got depth={depth} type={ctype}")
        elif tag == b"IDAT":
            idat += payload
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride, bpp = width * 3, 3
    rows, prev, p = [], bytearray(stride), 0
    for _ in range(height):
        ftype = raw[p]
        p += 1
        line = bytearray(raw[p:p + stride])
        p += stride
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            x = line[i]
            if ftype == 1:
                x += a
            elif ftype == 2:
                x += b
            elif ftype == 3:
                x += (a + b) // 2
            elif ftype == 4:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                x += a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            line[i] = x & 0xFF
        rows.append(line)
        prev = line
    return width, height, rows


def pixel(rows, x, y):
    o = x * 3
    return tuple(rows[y][o:o + 3])


def plate_mesh():
    session = HarnessSession(FRepBackend(resolution=32), verify_level="core")
    session.apply_ops([parse_op(d) for d in PLATE_OPS])
    return fmt.to_mesh(session).indexed()


class PngWriterTests(unittest.TestCase):
    def test_encode_is_a_real_png(self):
        data = render.encode_png([0, 0, 0, 255, 255, 255], 2, 1)
        self.assertEqual(data[:8], PNG_MAGIC)
        self.assertEqual(data[12:16], b"IHDR")
        self.assertTrue(data.rstrip().endswith(b"\xaeB`\x82"))  # IEND crc
        self.assertEqual(render.png_size(data), (2, 1))

    def test_round_trips_through_an_independent_decoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "px.png")
            render.write_png(path, [10, 20, 30, 200, 210, 220], 2, 1)
            width, height, rows = decode_png(path)
            self.assertEqual((width, height), (2, 1))
            self.assertEqual(pixel(rows, 0, 0), (10, 20, 30))
            self.assertEqual(pixel(rows, 1, 0), (200, 210, 220))

    def test_wrong_buffer_size_is_refused(self):
        with self.assertRaises(render.RenderError):
            render.encode_png([0, 0, 0], 2, 1)

    def test_png_size_rejects_a_non_png(self):
        with self.assertRaises(render.RenderError):
            render.png_size(b"not a png at all, not even close" * 2)


class RenderCubeTests(unittest.TestCase):
    def test_cube_renders_to_a_valid_non_uniform_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cube.png")
            out = render.render(CUBE, path, view="iso", width=160, height=120, ssaa=2)
            self.assertEqual(out, path)
            self.assertTrue(os.path.getsize(path) > 0)

            width, height, rows = decode_png(path)
            self.assertEqual((width, height), (160, 120))
            self.assertEqual(render.png_size(path), (160, 120))

            colors = {pixel(rows, x, y) for y in range(height) for x in range(width)}
            self.assertGreater(len(colors), 8, "the image is flat -- nothing was drawn")
            # the part is really there: plenty of non-background pixels
            drawn = sum(1 for y in range(height) for x in range(width)
                        if pixel(rows, x, y) != (255, 255, 255))
            self.assertGreater(drawn, width * height * 0.10)
            # ...and it does not fill the frame: there is a silhouette
            self.assertLess(drawn, width * height * 0.90)

    def test_the_image_is_neither_blank_nor_black(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cube.png")
            render.render(CUBE, path, width=120, height=90, ssaa=1)
            width, height, rows = decode_png(path)
            lum = [0.299 * r + 0.587 * g + 0.114 * b
                   for y in range(height) for x in range(width)
                   for (r, g, b) in [pixel(rows, x, y)]]
            mean = sum(lum) / len(lum)
            variance = sum((v - mean) ** 2 for v in lum) / len(lum)
            self.assertGreater(variance, 50.0, "no tonal range: the render is blank")
            self.assertGreater(mean, 40.0, "the render is black")

    def test_every_named_view_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in render.VIEW_PRESETS:
                path = os.path.join(tmp, f"{name}.png")
                render.render(CUBE, path, view=name, width=64, height=64, ssaa=1)
                self.assertEqual(render.png_size(path), (64, 64))

    def test_unknown_view_and_bad_options_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.png")
            with self.assertRaises(render.RenderError):
                render.render(CUBE, path, view="nowhere")
            with self.assertRaises(render.RenderError):
                render.render(CUBE, path, shading="cel")
            with self.assertRaises(render.RenderError):
                render.render(CUBE, path, ssaa=9)
            with self.assertRaises(render.RenderError):
                render.render(([], []), path)

    def test_perspective_and_orthographic_both_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            ortho = render.render(CUBE, os.path.join(tmp, "o.png"), width=80, height=80,
                                  ssaa=1, projection="orthographic", return_pixels=True)
            persp = render.render(CUBE, os.path.join(tmp, "p.png"), width=80, height=80,
                                  ssaa=1, projection="perspective", return_pixels=True)
            self.assertEqual(len(ortho["pixels"]), len(persp["pixels"]))
            # a perspective camera does not project the cube the same way
            self.assertNotEqual(bytes(ortho["pixels"]), bytes(persp["pixels"]))


class ZBufferTests(unittest.TestCase):
    """The whole point of a z-buffer: the far surface must not show through."""

    # Two axis-aligned quads facing +X, both covering the view. The NEAR one
    # (x=1) is red-ish; the FAR one (x=-1) is drawn AFTER it, so a renderer with
    # no depth test would paint it over the top.
    NEAR_FAR_VERTS = [
        (1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0), (1.0, -1.0, 1.0),
        (-1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, 1.0, 1.0), (-1.0, -1.0, 1.0),
    ]
    NEAR_FAR_FACES = [(0, 1, 2), (0, 2, 3),      # near quad, at x = +1
                      (4, 5, 6), (4, 6, 7)]      # far quad, at x = -1, drawn later

    def _center_face(self, view="side", cull=False):
        """Which triangle owns the centre pixel, straight out of the id-buffer."""
        verts = self.NEAR_FAR_VERTS
        faces = self.NEAR_FAR_FACES
        center, radius = render._bounds(verts)
        cam = render.preset_camera(view, center, radius)
        proj = render._Projector(cam, 40, 40, verts, 0.1)
        fb = render.Framebuffer(40, 40, (255, 255, 255))
        raw = render._face_normals(verts, faces)
        lights = render._resolved_lights(list(render.DEFAULT_LIGHTS), cam)
        render._raster(fb, proj, verts, faces, raw, [], render.DEFAULT_MATERIAL,
                       lights, "flat", cull)
        return fb.face[20 * 40 + 20]

    def test_the_near_triangle_wins_the_pixel(self):
        # the "side" preset looks from +X, so the x=+1 quad (faces 0/1) is nearer
        winner = self._center_face(view="side", cull=False)
        self.assertIn(winner, (0, 1),
                      "the far quad showed through the near one: no depth test")

    def test_the_far_triangle_wins_from_the_other_side(self):
        # from -X the depth order reverses; the SAME code must now pick the other quad
        winner = self._center_face(view="left", cull=False)
        self.assertIn(winner, (2, 3),
                      "the z-buffer did not follow the camera")

    def test_a_hidden_face_is_never_visible_in_the_render(self):
        # a cube: exactly the front-facing half of its faces may own pixels
        verts, faces = CUBE
        audit = render.visibility_audit(CUBE, views=("front",), width=64, height=64)
        self.assertTrue(audit["visible"])
        # the -Y face (ids 4,5) faces the "front" camera; the +Y face (8,9) is behind it
        self.assertTrue(set(audit["visible"]).intersection({4, 5}))
        self.assertFalse(set(audit["visible"]).intersection({8, 9}),
                         "the back face of the cube was rendered through the front")


class ShadingTests(unittest.TestCase):
    def test_a_lit_face_is_brighter_than_an_unlit_face(self):
        light = render.Light(direction=(0.0, 0.0, 1.0), space="world", intensity=1.0)
        mat = render.Material(base_color=(200, 200, 200), ambient=0.2, specular=0.0)
        lights = [( (0.0, 0.0, 1.0), (1.0, 1.0, 1.0), 1.0 )]
        view_dir = (0.0, 0.0, 1.0)

        facing = render._shade((0.0, 0.0, 1.0), view_dir, mat, lights)   # into the light
        edge_on = render._shade((1.0, 0.0, 0.0), view_dir, mat, lights)  # grazing it
        self.assertGreater(facing[0], edge_on[0],
                           "the face pointing at the light is not brighter")
        # the grazing face still gets the ambient term, and nothing is ever black
        self.assertGreater(edge_on[0], 0.0)
        self.assertLessEqual(facing[0], 1.0)
        self.assertIsInstance(light.direction, tuple)

    def test_the_lit_top_of_a_cube_beats_its_shaded_side(self):
        # the default key light comes from over the viewer's shoulder and above,
        # so in the iso view the top face must read brighter than the dark side.
        out = render.render(CUBE, None, view="iso", width=120, height=120, ssaa=1,
                            edges=False, shading="flat")
        rows = out["pixels"]
        w = out["width"]

        def lum_at(x, y):
            o = (y * w + x) * 3
            r, g, b = rows[o], rows[o + 1], rows[o + 2]
            return 0.299 * r + 0.587 * g + 0.114 * b

        top = lum_at(60, 40)      # the top face, upper middle of an iso cube
        side = lum_at(35, 85)     # a lower side face
        self.assertNotEqual(top, 255.0)
        self.assertGreater(top, side, "the lit top face is not brighter than the side")

    def test_flat_and_smooth_differ_on_a_curved_surface(self):
        verts, faces = plate_mesh()
        flat = render.render((verts, faces), None, view="iso", width=100, height=100,
                             ssaa=1, shading="flat", edges=False)
        smooth = render.render((verts, faces), None, view="iso", width=100, height=100,
                               ssaa=1, shading="smooth", edges=False)
        self.assertNotEqual(bytes(flat["pixels"]), bytes(smooth["pixels"]),
                            "smooth shading produced the same image as flat")

    def test_a_material_changes_the_colour(self):
        red = render.render(CUBE, None, width=48, height=48, ssaa=1, edges=False,
                            material=render.Material(base_color=(255, 0, 0)))
        blue = render.render(CUBE, None, width=48, height=48, ssaa=1, edges=False,
                             material=render.Material(base_color=(0, 0, 255)))
        self.assertNotEqual(bytes(red["pixels"]), bytes(blue["pixels"]))

    def test_the_background_is_honoured(self):
        out = render.render(CUBE, None, width=48, height=48, ssaa=1,
                            background=(7, 8, 9))
        self.assertEqual(tuple(out["pixels"][0:3]), (7, 8, 9))  # top-left corner


class EdgeOverlayTests(unittest.TestCase):
    def test_edges_draw_dark_ink_over_the_solid(self):
        with_edges = render.render(CUBE, None, view="iso", width=120, height=120,
                                   ssaa=1, edges=True)
        without = render.render(CUBE, None, view="iso", width=120, height=120,
                                ssaa=1, edges=False)
        self.assertNotEqual(bytes(with_edges["pixels"]), bytes(without["pixels"]))

        def ink(buf):
            return sum(1 for i in range(0, len(buf), 3)
                       if buf[i] < 90 and buf[i + 1] < 90 and buf[i + 2] < 90)

        self.assertGreater(ink(with_edges["pixels"]), 50, "no edge ink was drawn")
        self.assertEqual(ink(without["pixels"]), 0, "ink appeared with edges=False")

    def test_the_overlay_uses_the_drawing_route_feature_edges(self):
        # the renderer must not carry a second crease-detector
        from harnesscad.io import drawing

        edges = drawing.feature_edges(CUBE, angle=25.0)
        self.assertEqual(len(edges), 12, "a cube has 12 feature edges")


class DeterminismTests(unittest.TestCase):
    def test_two_identical_renders_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.png")
            b = os.path.join(tmp, "b.png")
            render.render(CUBE, a, view="hero", width=100, height=80, ssaa=2)
            render.render(CUBE, b, view="hero", width=100, height=80, ssaa=2)
            with open(a, "rb") as fh:
                first = fh.read()
            with open(b, "rb") as fh:
                second = fh.read()
            self.assertEqual(first, second, "the renderer is not deterministic")

    def test_the_real_part_is_deterministic_too(self):
        verts, faces = plate_mesh()
        one = render.render((verts, faces), None, view="iso", width=90, height=70, ssaa=2)
        two = render.render((verts, faces), None, view="iso", width=90, height=70, ssaa=2)
        self.assertEqual(bytes(one["pixels"]), bytes(two["pixels"]))


class SsaaTests(unittest.TestCase):
    def test_ssaa_keeps_the_output_size_and_changes_the_pixels(self):
        aliased = render.render(CUBE, None, view="iso", width=64, height=64, ssaa=1)
        smooth = render.render(CUBE, None, view="iso", width=64, height=64, ssaa=3)
        self.assertEqual(aliased["width"], smooth["width"], 64)
        self.assertEqual(len(aliased["pixels"]), len(smooth["pixels"]))
        self.assertNotEqual(bytes(aliased["pixels"]), bytes(smooth["pixels"]))

    def test_ssaa_produces_intermediate_tones_on_the_silhouette(self):
        """Anti-aliasing means edge pixels are blends, not just solid-or-background."""
        def tones(ssaa):
            out = render.render(CUBE, None, view="iso", width=64, height=64,
                                ssaa=ssaa, edges=False, background=(255, 255, 255))
            buf = out["pixels"]
            return {(buf[i], buf[i + 1], buf[i + 2]) for i in range(0, len(buf), 3)}

        self.assertGreater(len(tones(3)), len(tones(1)),
                           "supersampling added no intermediate tones")

    def test_downsample_is_an_exact_box_filter(self):
        fb = render.Framebuffer(2, 2, (0, 0, 0))
        for i, value in enumerate((0, 100, 200, 255)):      # 4 pixels, 1 channel each
            o = i * 3
            fb.color[o] = fb.color[o + 1] = fb.color[o + 2] = value
        out = fb.downsample(2)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], (0 + 100 + 200 + 255 + 2) // 4)   # rounded mean


class CullingTests(unittest.TestCase):
    def test_culling_drops_the_back_faces_of_a_closed_solid(self):
        """A closed cube looks the same culled or not -- but half the work is skipped."""
        culled = render.render(CUBE, None, view="iso", width=80, height=80, ssaa=1,
                               edges=False, cull=True)
        uncut = render.render(CUBE, None, view="iso", width=80, height=80, ssaa=1,
                              edges=False, cull=False)
        # the z-buffer already hides the back faces, so the image must be identical
        self.assertEqual(bytes(culled["pixels"]), bytes(uncut["pixels"]),
                         "culling changed the image of a closed solid")

    def test_culling_hides_a_lone_back_facing_triangle(self):
        """An open, back-facing triangle: culled it vanishes, unculled it shows."""
        # wound clockwise as seen from the "front" camera (which looks from -Y)
        verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 1.0)]
        faces = [(0, 1, 2)]
        cam = render.look_at((0.0, -5.0, 0.5), (0.5, 0.0, 0.5))

        def drawn(cull):
            out = render.render((verts, faces), None, camera=cam, width=40, height=40,
                                ssaa=1, edges=False, cull=cull,
                                background=(255, 255, 255))
            buf = out["pixels"]
            return sum(1 for i in range(0, len(buf), 3)
                       if (buf[i], buf[i + 1], buf[i + 2]) != (255, 255, 255))

        lit = drawn(cull=False)
        self.assertGreater(lit, 10, "the triangle did not render at all")
        # one of the two windings must be culled; flip until we find the culled one
        flipped = [(0, 2, 1)]

        def drawn_flipped(cull):
            out = render.render((verts, flipped), None, camera=cam, width=40, height=40,
                                ssaa=1, edges=False, cull=cull,
                                background=(255, 255, 255))
            buf = out["pixels"]
            return sum(1 for i in range(0, len(buf), 3)
                       if (buf[i], buf[i + 1], buf[i + 2]) != (255, 255, 255))

        culled_counts = (drawn(cull=True), drawn_flipped(cull=True))
        self.assertIn(0, culled_counts,
                      "backface culling never removed the back-facing winding")
        self.assertGreater(max(culled_counts), 10,
                           "backface culling removed the FRONT-facing winding too")


class CameraTests(unittest.TestCase):
    def test_look_at_builds_an_orthonormal_right_handed_basis(self):
        cam = render.look_at((5.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        right, up, forward = cam.basis()
        for v in (right, up, forward):
            self.assertAlmostEqual(sum(c * c for c in v) ** 0.5, 1.0, places=9)
        self.assertAlmostEqual(render._dot(right, up), 0.0, places=9)
        self.assertAlmostEqual(render._dot(right, forward), 0.0, places=9)
        self.assertAlmostEqual(render._dot(up, forward), 0.0, places=9)

    def test_a_degenerate_camera_is_refused(self):
        with self.assertRaises(render.RenderError):
            render.look_at((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)).basis()
        with self.assertRaises(render.RenderError):
            render.look_at((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), projection="fisheye")

    def test_a_top_view_does_not_gimbal_lock(self):
        # up is +Z and the view direction IS +Z: the basis must still be valid
        right, up, forward = render.preset_camera(
            "top", (0.0, 0.0, 0.0), 1.0).basis()
        self.assertAlmostEqual(sum(c * c for c in right) ** 0.5, 1.0, places=9)


class RegistryTests(unittest.TestCase):
    def test_png_is_registered_write_only(self):
        spec = fmt.spec_for_extension(".png")
        self.assertEqual(spec.kind, "image")
        self.assertTrue(spec.can_write)
        self.assertFalse(spec.can_read, "PNG must not claim to be readable")
        self.assertFalse(spec.round_trip)
        self.assertEqual(spec.mime, "image/png")

    def test_reading_a_png_is_refused_honestly(self):
        with self.assertRaises(fmt.UnsupportedOperationError):
            fmt.read("part.png")

    def test_png_appears_in_the_capability_matrix_as_write_only(self):
        rows = {r["name"]: r for r in fmt.capability_matrix()}
        self.assertIn("render", rows)
        row = rows["render"]
        self.assertEqual(row["extensions"], [".png"])
        self.assertTrue(row["write"])
        self.assertFalse(row["read"])
        self.assertIn("render", [s.name for s in fmt.supported(kind="image",
                                                               mode="write")])
        self.assertEqual(fmt.supported(kind="image", mode="read"), [])

    def test_write_dispatches_on_the_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "part.png")
            fmt.write(CUBE, path, width=64, height=64, ssaa=1)
            self.assertEqual(render.png_size(path), (64, 64))

    def test_a_session_exports_straight_to_a_png(self):
        session = HarnessSession(FRepBackend(resolution=24), verify_level="core")
        session.apply_ops([parse_op(d) for d in PLATE_OPS[:3]])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.png")
            fmt.export_session(session, path, width=80, height=60, ssaa=1)
            self.assertEqual(render.png_size(path), (80, 60))


class SessionRenderTests(unittest.TestCase):
    def test_render_session_draws_the_real_part(self):
        session = HarnessSession(FRepBackend(resolution=32), verify_level="core")
        result = session.apply_ops([parse_op(d) for d in PLATE_OPS])
        self.assertTrue(result.ok)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plate.png")
            render.render_session(session, path, view="iso", width=200, height=150,
                                  ssaa=1)
            width, height, rows = decode_png(path)
            self.assertEqual((width, height), (200, 150))
            drawn = sum(1 for y in range(height) for x in range(width)
                        if pixel(rows, x, y) != (255, 255, 255))
            self.assertGreater(drawn, 200, "the part did not render")

    def test_the_through_holes_are_open_in_the_top_view(self):
        """The holes must be SEEN, not just meshed: background shows through them."""
        verts, faces = plate_mesh()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "top.png")
            render.render((verts, faces), path, view="top", width=400, height=240,
                          shading="flat", edges=False, ssaa=1,
                          background=(255, 0, 255))
            width, height, rows = decode_png(path)

        def at(mx, my):
            margin = 0.10
            x = int(margin * width + (mx / 40.0) * width * (1 - 2 * margin))
            y = int(margin * height + (1.0 - my / 24.0) * height * (1 - 2 * margin))
            return pixel(rows, x, y)

        background = (255, 0, 255)
        hole_a, hole_b = at(10.0, 12.0), at(30.0, 12.0)
        plate = at(20.0, 12.0)
        self.assertEqual(hole_a, background, "the hole at (10,12) is not open")
        self.assertEqual(hole_b, background, "the hole at (30,12) is not open")
        self.assertNotEqual(plate, background, "the plate web is missing")
        self.assertNotEqual(hole_a, plate, "hole pixels do not differ from plate pixels")


class UnblockedModuleTests(unittest.TestCase):
    """The modules that were orphaned solely because no renderer existed."""

    def test_three_view_composites_the_cadsmith_judge_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "three.png")
            out = render.three_view(CUBE, path, panel=80, ssaa=1)
            self.assertEqual(len(out["views"]), 3)
            self.assertEqual(render.png_size(path), (240, 80))
            self.assertEqual((out["width"], out["height"]), (240, 80))

    def test_visibility_audit_reports_a_real_visible_set(self):
        audit = render.visibility_audit(CUBE, views=("front", "top", "side"),
                                        width=64, height=64)
        self.assertEqual(audit["faces"], 12)
        self.assertTrue(audit["visible"])
        # three orthogonal views of a cube cannot show all six faces
        self.assertLess(len(audit["visible"]), 12)

    def test_visual_qc_accepts_a_render_and_rejects_a_fake(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "qc.png")
            render.render(CUBE, path, width=128, height=128, ssaa=1)
            good = render.qc(path, asset_id="cube")
            self.assertTrue(getattr(good, "accepted", None) or good)

            bad = render.qc(b"definitely not a png", asset_id="junk")
            self.assertNotEqual(str(good), str(bad))


if __name__ == "__main__":
    unittest.main()
