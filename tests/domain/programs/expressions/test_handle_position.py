"""Tests for programs.paramgeom_position (Position + Delta-vector CSG derivation)."""

import unittest
from fractions import Fraction

from harnesscad.domain.programs.expressions.handles import cube_handles, cylinder_handles, sphere_handles
from harnesscad.domain.programs.expressions.linear_form import LinearForm
from harnesscad.domain.programs.expressions.handle_position import (
    DerivationError,
    PrimitiveNode,
    delta_vector,
    derive_position,
    opaque_transform,
    translate,
    translate_statement,
    vec_to_code,
)


def _vec(*forms):
    return tuple(forms)


class PositionTest(unittest.TestCase):
    def test_bottom_face_of_translated_cube(self):
        # translate([tx,ty,tz]) cube([size_x,size_y,size_z]); handle = bottom face
        cube = PrimitiveNode("cube1", handles=cube_handles("size_x", "size_y", "size_z"))
        root = translate(
            "t1",
            _vec(LinearForm.var("tx"), LinearForm.var("ty"), LinearForm.var("tz")),
            cube,
        )
        pos = derive_position(root, "cube1", "xmid_ymid_zmin")
        # x = tx, y = ty, z = tz - size_z/2  (paper Figure 2 derivation)
        self.assertEqual(pos[0], LinearForm.var("tx"))
        self.assertEqual(pos[1], LinearForm.var("ty"))
        self.assertEqual(
            pos[2], LinearForm.var("tz") + LinearForm.var("size_z", Fraction(-1, 2))
        )

    def test_nested_translations_accumulate(self):
        cube = PrimitiveNode("c", handles=cube_handles(2, 2, 2))
        inner = translate(
            "t2", _vec(LinearForm.const(0), LinearForm.const(0), LinearForm.var("b")), cube
        )
        root = translate(
            "t1", _vec(LinearForm.var("a"), LinearForm.const(0), LinearForm.const(0)), inner
        )
        pos = derive_position(root, "c", "center")
        self.assertEqual(pos[0], LinearForm.var("a"))
        self.assertEqual(pos[2], LinearForm.var("b"))

    def test_trivial_zero_translate_vanishes(self):
        # a translate([0,0,0]) contributes nothing (paper's simplification concern)
        cube = PrimitiveNode("c", handles=cube_handles("s", "s", "s"))
        root = translate("t0", _vec(*(LinearForm.const(0),) * 3), cube)
        pos = derive_position(root, "c", "center")
        self.assertTrue(all(p.is_zero for p in pos))

    def test_missing_node_raises(self):
        cube = PrimitiveNode("c", handles=cube_handles(1, 1, 1))
        with self.assertRaises(DerivationError):
            derive_position(cube, "nope", "center")

    def test_missing_handle_raises(self):
        cube = PrimitiveNode("c", handles=cube_handles(1, 1, 1))
        with self.assertRaises(DerivationError):
            derive_position(cube, "c", "no_such_handle")

    def test_rotate_on_path_blocks(self):
        cube = PrimitiveNode("c", handles=cube_handles(1, 1, 1))
        root = opaque_transform("rot", cube)
        with self.assertRaises(DerivationError):
            derive_position(root, "c", "center")

    def test_non_primitive_node_center(self):
        cube = PrimitiveNode("c", handles=cube_handles(1, 1, 1))
        root = translate("t1", _vec(LinearForm.var("a"), LinearForm.const(0), LinearForm.const(0)), cube)
        pos = derive_position(root, "t1", "center")
        self.assertEqual(pos[0], LinearForm.var("a"))


class DeltaVectorTest(unittest.TestCase):
    def test_cup_sphere_alignment(self):
        # Paper worked example (Section 4.2): aligning a sphere to the cylinder
        # top yields [r_top - r_sphere, 0, thickness + h_stem + h_top].
        stem = PrimitiveNode("stem", handles=cylinder_handles("r_stem1", "r_top", "h_stem"))
        base_translate = translate(
            "base_t",
            _vec(LinearForm.const(0), LinearForm.const(0), LinearForm.var("thickness")),
            stem,
        )
        sphere = PrimitiveNode("sphere", handles=sphere_handles("d_sphere"))
        sphere_translate = translate(
            "sph_t",
            _vec(
                LinearForm.const(0),
                LinearForm.const(0),
                LinearForm.var("thickness")
                + LinearForm.var("h_stem")
                + LinearForm.var("h_top"),
            ),
            sphere,
        )
        root = translate("root", _vec(*(LinearForm.const(0),) * 3), base_translate, sphere_translate)

        # origin: sphere x-extreme; destination: cylinder-top x-extreme.
        delta = delta_vector(
            root,
            origin=("sphere", "xmax_ymid_zmid"),
            destination=("stem", "xmax_ymid_zmax"),
        )
        # x: (r_top) - (d_sphere/2) ; here d_sphere is the sphere diameter.
        self.assertEqual(delta[0].coefficient("r_top"), Fraction(1))
        self.assertEqual(delta[0].coefficient("d_sphere"), Fraction(-1, 2))

    def test_simple_delta(self):
        a = PrimitiveNode("a", handles=cube_handles(2, 2, 2))
        b = PrimitiveNode("b", handles=cube_handles(2, 2, 2))
        root = translate(
            "root",
            _vec(*(LinearForm.const(0),) * 3),
            translate("ta", _vec(LinearForm.const(0), LinearForm.const(0), LinearForm.const(0)), a),
            translate("tb", _vec(LinearForm.var("d"), LinearForm.const(0), LinearForm.const(0)), b),
        )
        delta = delta_vector(root, origin=("a", "center"), destination=("b", "center"))
        self.assertEqual(delta[0], LinearForm.var("d"))
        self.assertTrue(delta[1].is_zero and delta[2].is_zero)


class RenderTest(unittest.TestCase):
    def test_vec_to_code(self):
        vec = _vec(LinearForm.var("x"), LinearForm.const(0), LinearForm.const(3))
        self.assertEqual(vec_to_code(vec), "[x, 0, 3]")

    def test_translate_statement(self):
        vec = _vec(LinearForm.const(0), LinearForm.const(0), LinearForm.var("h"))
        self.assertEqual(translate_statement(vec), "translate([0, 0, h])")


if __name__ == "__main__":
    unittest.main()
