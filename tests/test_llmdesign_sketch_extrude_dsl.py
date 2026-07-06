"""Tests for the global-coordinate sketch-and-extrude CAD DSL interpreter."""

import unittest

from geometry.llmdesign_sketch_extrude_dsl import (
    CapPlane,
    Circle,
    Interpreter,
    Rectangle,
)


class TestDefaultPlaneExtrusion(unittest.TestCase):
    def test_circle_on_xy_plane(self):
        interp = Interpreter()
        s = interp.create_sketch(Circle(2.0, 3.0, 5.0, 4.0), "XY_PLANE")
        sid = interp.extrude(s, 10.0)
        solid = interp.solid(sid)
        self.assertEqual(solid.kind, "cylinder")
        self.assertEqual(solid.normal, (0.0, 0.0, 1.0))
        # spans [cx-r,cx+r] x [cy-r,cy+r] x [cz, cz+L]
        self.assertEqual(solid.aabb, (-2.0, -1.0, 5.0, 6.0, 7.0, 15.0))

    def test_circle_on_xz_plane(self):
        interp = Interpreter()
        s = interp.create_sketch(Circle(0.0, 0.0, 0.0, 3.0), "XZ_PLANE")
        sid = interp.extrude(s, 6.0)
        solid = interp.solid(sid)
        # normal +Y; in-plane axes (X,Z); extrude along Y
        self.assertEqual(solid.aabb, (-3.0, 0.0, -3.0, 3.0, 6.0, 3.0))

    def test_circle_on_zy_plane(self):
        interp = Interpreter()
        s = interp.create_sketch(Circle(0.0, 0.0, 0.0, 2.0), "ZY_PLANE")
        sid = interp.extrude(s, 5.0)
        solid = interp.solid(sid)
        # normal +X; in-plane axes (Z,Y); extrude along X
        self.assertEqual(solid.aabb, (0.0, -2.0, -2.0, 5.0, 2.0, 2.0))

    def test_rectangle_on_each_plane(self):
        # XY: length along X, width along Y
        interp = Interpreter()
        s = interp.create_sketch(Rectangle(0.0, 0.0, 0.0, 4.0, 2.0), "XY_PLANE")
        solid = interp.solid(interp.extrude(s, 3.0))
        self.assertEqual(solid.kind, "box")
        self.assertEqual(solid.aabb, (-2.0, -1.0, 0.0, 2.0, 1.0, 3.0))

        # XZ: length along X, width along Z
        interp = Interpreter()
        s = interp.create_sketch(Rectangle(0.0, 0.0, 0.0, 4.0, 2.0), "XZ_PLANE")
        solid = interp.solid(interp.extrude(s, 3.0))
        self.assertEqual(solid.aabb, (-2.0, 0.0, -1.0, 2.0, 3.0, 1.0))

        # ZY: length along Z, width along Y
        interp = Interpreter()
        s = interp.create_sketch(Rectangle(0.0, 0.0, 0.0, 4.0, 2.0), "ZY_PLANE")
        solid = interp.solid(interp.extrude(s, 3.0))
        self.assertEqual(solid.aabb, (0.0, -1.0, -2.0, 3.0, 1.0, 2.0))


class TestExtrudeDirection(unittest.TestCase):
    def test_extrude_along_positive_normal(self):
        interp = Interpreter()
        s = interp.create_sketch(Circle(0.0, 0.0, 4.0, 1.0), "XY_PLANE")
        solid = interp.solid(interp.extrude(s, 7.0))
        # from z=4 upward to z=11
        self.assertEqual(solid.zmin, 4.0)
        self.assertEqual(solid.zmax, 11.0)


class TestCapFaceSelection(unittest.TestCase):
    def _base(self):
        interp = Interpreter()
        # box centred at origin spanning [-1,1]^3
        s = interp.create_sketch(Rectangle(0.0, 0.0, -1.0, 2.0, 2.0), "XY_PLANE")
        base = interp.extrude(s, 2.0)  # z in [-1,1], x in [-1,1], y in [-1,1]
        return interp, base

    def test_cap_max_z_places_at_top(self):
        interp, base = self._base()
        s = interp.create_sketch(Circle(0.0, 0.0, 999.0, 1.0), interp.cap(base, "max_z"))
        solid = interp.solid(interp.extrude(s, 3.0))
        # sits on top face z=1, normal +Z, extrudes to z=4
        self.assertEqual(solid.zmin, 1.0)
        self.assertEqual(solid.zmax, 4.0)

    def test_cap_min_z_places_at_bottom(self):
        interp, base = self._base()
        s = interp.create_sketch(Circle(0.0, 0.0, 50.0, 1.0), interp.cap(base, "min_z"))
        solid = interp.solid(interp.extrude(s, 3.0))
        # sits on bottom face z=-1, normal -Z, extrudes downward to z=-4
        self.assertEqual(solid.zmax, -1.0)
        self.assertEqual(solid.zmin, -4.0)

    def test_cap_max_x_and_min_x(self):
        interp, base = self._base()
        s = interp.create_sketch(Circle(0.0, 0.0, 0.0, 1.0), interp.cap(base, "max_x"))
        solid = interp.solid(interp.extrude(s, 2.0))
        self.assertEqual(solid.xmin, 1.0)
        self.assertEqual(solid.xmax, 3.0)

        interp2, base2 = self._base()
        s2 = interp2.create_sketch(Circle(0.0, 0.0, 0.0, 1.0), interp2.cap(base2, "min_x"))
        solid2 = interp2.solid(interp2.extrude(s2, 2.0))
        self.assertEqual(solid2.xmax, -1.0)
        self.assertEqual(solid2.xmin, -3.0)

    def test_cap_max_y_and_min_y(self):
        interp, base = self._base()
        s = interp.create_sketch(Circle(0.0, 0.0, 0.0, 1.0), interp.cap(base, "max_y"))
        solid = interp.solid(interp.extrude(s, 2.0))
        self.assertEqual(solid.ymin, 1.0)
        self.assertEqual(solid.ymax, 3.0)

        interp2, base2 = self._base()
        s2 = interp2.create_sketch(Circle(0.0, 0.0, 0.0, 1.0), interp2.cap(base2, "min_y"))
        solid2 = interp2.solid(interp2.extrude(s2, 2.0))
        self.assertEqual(solid2.ymax, -1.0)
        self.assertEqual(solid2.ymin, -3.0)

    def test_cap_normal_axis_override(self):
        # The given center's normal-axis component is ignored; in-plane coords
        # are preserved.
        interp, base = self._base()
        s = interp.create_sketch(
            Circle(0.5, -0.5, 12345.0, 0.25), interp.cap(base, "max_z")
        )
        solid = interp.solid(interp.extrude(s, 1.0))
        # in-plane center (0.5,-0.5) preserved; z overridden to top face 1.0
        self.assertEqual(solid.aabb, (0.25, -0.75, 1.0, 0.75, -0.25, 2.0))


class TestRoundTableFigure84(unittest.TestCase):
    def _build(self):
        interp = Interpreter()
        leg_base = interp.extrude(
            interp.create_sketch(Circle(0.0, 0.0, 0.0, 3.0), "XY_PLANE"), 1.0
        )
        leg = interp.extrude(
            interp.create_sketch(
                Circle(0.0, 0.0, 1.0, 1.0), interp.cap(leg_base, "max_z")
            ),
            10.0,
        )
        top = interp.extrude(
            interp.create_sketch(
                Circle(0.0, 0.0, 11.0, 8.0), interp.cap(leg, "max_z")
            ),
            1.0,
        )
        return interp, leg_base, leg, top

    def test_leg_base_aabb(self):
        interp, leg_base, _, _ = self._build()
        s = interp.solid(leg_base)
        self.assertEqual(s.aabb, (-3.0, -3.0, 0.0, 3.0, 3.0, 1.0))

    def test_leg_sits_on_base(self):
        interp, _, leg, _ = self._build()
        s = interp.solid(leg)
        # leg on top face of legBase (z=1), radius 1, extruded 10 -> z 1..11
        self.assertEqual(s.aabb, (-1.0, -1.0, 1.0, 1.0, 1.0, 11.0))

    def test_top_sits_on_leg(self):
        interp, _, _, top = self._build()
        s = interp.solid(top)
        # top on leg's top face z=11, radius 8, extruded 1 -> z 11..12
        self.assertEqual(s.aabb, (-8.0, -8.0, 11.0, 8.0, 8.0, 12.0))

    def test_assembly_aabb(self):
        interp, _, _, _ = self._build()
        self.assertEqual(interp.assembly_aabb(), (-8.0, -8.0, 0.0, 8.0, 8.0, 12.0))
        self.assertEqual(len(interp.solids), 3)


class TestErrorCases(unittest.TestCase):
    def test_unknown_plane(self):
        interp = Interpreter()
        with self.assertRaises(ValueError):
            interp.create_sketch(Circle(0, 0, 0, 1), "AB_PLANE")

    def test_unknown_cap_side(self):
        interp = Interpreter()
        s = interp.create_sketch(Circle(0, 0, 0, 1), "XY_PLANE")
        base = interp.extrude(s, 1.0)
        with self.assertRaises(ValueError):
            interp.cap(base, "top")

    def test_cap_unknown_solid(self):
        interp = Interpreter()
        with self.assertRaises(ValueError):
            interp.cap(999, "max_z")

    def test_extrude_unknown_sketch(self):
        interp = Interpreter()
        with self.assertRaises(ValueError):
            interp.extrude(42, 1.0)

    def test_nonpositive_extrude_length(self):
        interp = Interpreter()
        s = interp.create_sketch(Circle(0, 0, 0, 1), "XY_PLANE")
        with self.assertRaises(ValueError):
            interp.extrude(s, 0.0)
        with self.assertRaises(ValueError):
            interp.extrude(s, -2.0)

    def test_negative_radius(self):
        with self.assertRaises(ValueError):
            Circle(0, 0, 0, -1.0)

    def test_negative_rectangle_dimensions(self):
        with self.assertRaises(ValueError):
            Rectangle(0, 0, 0, -1.0, 2.0)
        with self.assertRaises(ValueError):
            Rectangle(0, 0, 0, 1.0, -2.0)

    def test_solid_unknown_id(self):
        interp = Interpreter()
        with self.assertRaises(ValueError):
            interp.solid(0)

    def test_assembly_aabb_empty(self):
        interp = Interpreter()
        with self.assertRaises(ValueError):
            interp.assembly_aabb()


if __name__ == "__main__":
    unittest.main()
