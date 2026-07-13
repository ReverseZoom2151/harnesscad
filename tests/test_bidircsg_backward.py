"""Tests for editing.bidircsg_backward (the backward put + lens laws)."""

import unittest

from harnesscad.domain.programs.bidircsg_ast import (
    Difference,
    Primitive,
    Repeat,
    Rotate,
    Scale,
    Translate,
    Union,
    node_at,
)
from harnesscad.domain.programs.bidircsg_forward import find_instance, get, leaves
from harnesscad.domain.editing.bidircsg_backward import (
    get_put_holds,
    put_get_probe,
    put_get_translate_holds,
    put_rotate,
    put_scale,
    put_scale_primitive,
    put_translate,
    roundtrip_anchor_neutral,
    world_point,
)


def approx(a, b, tol=1e-6):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


class PutTranslateInsertTest(unittest.TestCase):
    def test_insert_translate_on_bare_primitive(self):
        prog = Primitive("sphere", (2.0,))
        res = put_translate(prog, (), (3.0, 0.0, 0.0))
        self.assertFalse(res.reused)
        self.assertIsInstance(res.program, Translate)
        self.assertTrue(approx(res.program.offset, (3.0, 0.0, 0.0)))
        # PutGet: the element moved by exactly the delta
        self.assertTrue(put_get_translate_holds(prog, (), (3.0, 0.0, 0.0)))

    def test_insert_converts_world_to_local_frame(self):
        # element sits inside a 90deg-about-z rotation; a world +x drag must
        # become a local +... offset that yields world +x.
        prog = Rotate((0, 0, 90), Primitive("sphere", (1.0,)))
        # selected = the primitive at path (0,)
        self.assertTrue(put_get_translate_holds(prog, (0,), (1.0, 0.0, 0.0)))


class PutTranslateReuseTest(unittest.TestCase):
    def test_reuse_existing_translate(self):
        prog = Translate((1.0, 0.0, 0.0), Primitive("cube", (1, 1, 1)))
        # select the primitive; its parent is a Translate -> reuse
        res = put_translate(prog, (0,), (2.0, 0.0, 0.0))
        self.assertTrue(res.reused)
        # offset updated, no new translate node inserted
        self.assertIsInstance(res.program, Translate)
        self.assertIsInstance(res.program.child, Primitive)
        self.assertTrue(approx(res.program.offset, (3.0, 0.0, 0.0)))

    def test_reuse_put_get(self):
        prog = Translate((1.0, 0.0, 0.0), Primitive("cube", (1, 1, 1)))
        self.assertTrue(put_get_translate_holds(prog, (0,), (2.0, 5.0, -1.0)))


class PutRotateTest(unittest.TestCase):
    def test_insert_rotate(self):
        prog = Primitive("cube", (2, 2, 2))
        res = put_rotate(prog, (), (0.0, 0.0, 90.0))
        self.assertFalse(res.reused)
        self.assertIsInstance(res.program, Rotate)
        # probe a local +x point: after 90deg about z it must land near +y
        self.assertTrue(
            put_get_probe(prog, (), res, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        )

    def test_reuse_rotate(self):
        prog = Rotate((0.0, 0.0, 45.0), Primitive("cube", (1, 1, 1)))
        res = put_rotate(prog, (0,), (0.0, 0.0, 45.0))
        self.assertTrue(res.reused)
        self.assertTrue(approx(res.program.angles, (0.0, 0.0, 90.0)))


class PutScaleTest(unittest.TestCase):
    def test_insert_scale(self):
        prog = Primitive("sphere", (1.0,))
        res = put_scale(prog, (), (2.0, 2.0, 2.0))
        self.assertFalse(res.reused)
        self.assertIsInstance(res.program, Scale)
        # probe (1,0,0) -> (2,0,0)
        self.assertTrue(
            put_get_probe(prog, (), res, (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        )

    def test_reuse_scale_multiplies(self):
        prog = Scale((2.0, 2.0, 2.0), Primitive("sphere", (1.0,)))
        res = put_scale(prog, (0,), (3.0, 1.0, 1.0))
        self.assertTrue(res.reused)
        self.assertTrue(approx(res.program.factors, (6.0, 2.0, 2.0)))

    def test_scale_primitive_updates_params(self):
        prog = Primitive("cube", (1.0, 2.0, 3.0))
        res = put_scale_primitive(prog, (), (2.0, 2.0, 2.0))
        self.assertTrue(res.reused)
        self.assertEqual(res.program.params, (2.0, 4.0, 6.0))

    def test_scale_primitive_rejects_non_primitive(self):
        prog = Union((Primitive("sphere", (1.0,)),))
        with self.assertRaises(ValueError):
            put_scale_primitive(prog, (), (2.0, 2.0, 2.0))


class LensLawTest(unittest.TestCase):
    def _prog(self):
        return Difference((
            Translate((2, 0, 0), Rotate((0, 0, 30), Primitive("cube", (1, 1, 1)))),
            Primitive("sphere", (0.5,)),
        ))

    def test_get_put_identity(self):
        self.assertTrue(get_put_holds(self._prog()))

    def test_get_put_zero_returns_same_object(self):
        prog = self._prog()
        self.assertIs(put_translate(prog, (0,), (0.0, 0.0, 0.0)).program, prog)

    def test_put_get_translate(self):
        prog = self._prog()
        # translate the cube (path (0,0,0)) by a world delta
        self.assertTrue(
            put_get_translate_holds(prog, (0, 0, 0), (1.0, -2.0, 0.5))
        )

    def test_roundtrip_anchor_neutral(self):
        prog = self._prog()
        self.assertTrue(
            roundtrip_anchor_neutral(prog, (0, 0, 0), (3.0, 1.0, -1.0))
        )


class LoopEditTest(unittest.TestCase):
    def test_edit_propagates_to_all_instances(self):
        prog = Repeat(3, (2, 0, 0), Primitive("cube", (1, 1, 1)))
        # select instance 1; parent (path (0,)->primitive) has AST parent = Repeat,
        # so a new translate is inserted, affecting ALL instances (impacted).
        before = [l.anchor[1] for l in leaves(get(prog))]
        res = put_translate(prog, (0,), (0.0, 5.0, 0.0), call_stack=(1,))
        after = [l.anchor[1] for l in leaves(get(res.program))]
        # every instance shifted by +5 in y
        for b, a in zip(sorted(before), sorted(after)):
            self.assertAlmostEqual(a - b, 5.0, places=6)


if __name__ == "__main__":
    unittest.main()
