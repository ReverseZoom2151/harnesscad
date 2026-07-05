"""Tests for the ruler-and-compass construction engine, DSL, compiler and
validity metrics (paper 71, "Draw It Like Euclid")."""

import math
import unittest

from geometry import euclid_construction as ec
from geometry import euclid_dsl as dsl
from geometry.euclid_compiler import (
    Profile, assemble_profile, compile_profile, replay, ReplayError,
)
from geometry import euclid_validity as ev


P = ec.Point


class PrimitiveTests(unittest.TestCase):
    def test_line_through_points_direction_and_membership(self):
        line = ec.line_through_points(P(0.0, 0.0), P(1.0, 0.0))
        self.assertAlmostEqual(line.phi, 0.0)
        self.assertTrue(line.contains(P(0.5, 0.0)))
        self.assertFalse(line.contains(P(0.5, 0.2)))
        # left normal of +x line points +y; a point above has positive dist.
        self.assertGreater(line.signed_distance(P(0.0, 0.3)), 0.0)

    def test_perpendicular_and_bisector(self):
        base = ec.line_through_points(P(-1.0, 0.0), P(1.0, 0.0))
        perp = ec.perpendicular_line(base, P(0.0, 0.0))
        self.assertAlmostEqual(_ang(perp.phi), math.pi / 2.0, places=9)
        pb = ec.perpendicular_bisector(P(-0.4, 0.0), P(0.4, 0.0))
        self.assertTrue(pb.contains(P(0.0, 0.9)))
        self.assertTrue(pb.contains(P(0.0, -0.9)))

    def test_parallel_line(self):
        base = ec.line_through_points(P(0.0, 0.0), P(1.0, 0.0))
        par = ec.parallel_line(base, P(0.3, 0.2))
        self.assertTrue(base.is_parallel(par))
        self.assertTrue(par.contains(P(-0.5, 0.2)))

    def test_angle_bisector(self):
        b = ec.angle_bisector(P(0.0, 0.0), P(1.0, 0.0), P(0.0, 1.0))
        self.assertAlmostEqual(_ang(b.phi), math.pi / 4.0, places=9)


class IntersectionTests(unittest.TestCase):
    def test_line_line(self):
        l1 = ec.line_through_points(P(0.0, 0.0), P(1.0, 0.0))
        l2 = ec.line_through_points(P(0.3, -1.0), P(0.3, 1.0))
        p = ec.line_x_line(l1, l2)
        self.assertTrue(p.almost_equals(P(0.3, 0.0), tol=1e-9))

    def test_parallel_lines_raise(self):
        l1 = ec.line_through_points(P(0.0, 0.0), P(1.0, 0.0))
        l2 = ec.line_through_points(P(0.0, 0.2), P(1.0, 0.2))
        self.assertIsNone(ec.line_line_intersection(l1, l2))
        with self.assertRaises(ValueError):
            ec.line_x_line(l1, l2)

    def test_line_circle_two_points(self):
        c = ec.Circle(P(0.0, 0.0), 0.5)
        line = ec.line_through_points(P(-1.0, 0.0), P(1.0, 0.0))
        pts = ec.line_x_circle(line, c)
        self.assertEqual(len(pts), 2)
        xs = sorted(p.x for p in pts)
        self.assertAlmostEqual(xs[0], -0.5, places=9)
        self.assertAlmostEqual(xs[1], 0.5, places=9)

    def test_line_circle_tangent_and_miss(self):
        c = ec.Circle(P(0.0, 0.0), 0.5)
        tangent = ec.line_through_points(P(-1.0, 0.5), P(1.0, 0.5))
        self.assertEqual(len(ec.line_x_circle(tangent, c)), 1)
        miss = ec.line_through_points(P(-1.0, 0.9), P(1.0, 0.9))
        self.assertEqual(len(ec.line_x_circle(miss, c)), 0)

    def test_circle_circle(self):
        c1 = ec.Circle(P(0.0, 0.0), 0.5)
        c2 = ec.Circle(P(0.5, 0.0), 0.5)
        pts = ec.circle_circle_intersection(c1, c2)
        self.assertEqual(len(pts), 2)
        for p in pts:
            self.assertAlmostEqual(p.x, 0.25, places=9)


class StepTests(unittest.TestCase):
    def test_circle_offset(self):
        c = ec.Circle(P(0.1, 0.1), 0.2, ccw=True)
        off = ec.circle_offset_circle(c, 0.1)
        self.assertAlmostEqual(off.radius, 0.3)
        self.assertTrue(off.center.almost_equals(c.center, 1e-12))

    def test_line_offset_left(self):
        line = ec.line_through_points(P(0.0, 0.0), P(1.0, 0.0))
        off = ec.line_offset_line(line, 0.2)
        # +x line offset left moves it to +y by 0.2
        self.assertTrue(off.contains(P(0.0, 0.2)))

    def test_line_reverse(self):
        line = ec.line_through_points(P(0.0, 0.0), P(1.0, 0.0))
        rev = ec.line_reverse_line(line)
        self.assertAlmostEqual(rev.phi, math.pi, places=9)
        # same set of points
        self.assertTrue(rev.contains(P(0.5, 0.0)))

    def test_circle_reverse_flips_flag_only(self):
        c = ec.Circle(P(0.1, 0.0), 0.3, ccw=True)
        r = ec.circle_reverse_circle(c)
        self.assertFalse(r.ccw)
        self.assertAlmostEqual(r.radius, c.radius)
        self.assertTrue(r.center.almost_equals(c.center, 1e-12))

    def test_point_line_sym(self):
        sym = ec.line_through_points(P(0.0, -1.0), P(0.0, 1.0))  # y-axis
        img = ec.point_line_sym_point(P(0.3, 0.2), sym)
        self.assertTrue(img.almost_equals(P(-0.3, 0.2), tol=1e-9))

    def test_line_sym_line(self):
        sym = ec.line_through_points(P(0.0, -1.0), P(0.0, 1.0))
        line = ec.line_through_points(P(0.2, 0.0), P(0.4, 0.3))
        img = ec.line_sym_line_line(line, sym)
        # reflected line passes through reflected points
        self.assertTrue(img.contains(P(-0.2, 0.0)))
        self.assertTrue(img.contains(P(-0.4, 0.3)))

    def test_line_axis_rotated(self):
        line = ec.line_through_points(P(0.0, 0.0), P(1.0, 0.0))
        rot = ec.line_axis_rotated_line(line, P(0.0, 0.0), math.pi / 2.0, ccw=True)
        self.assertAlmostEqual(_ang(rot.phi), math.pi / 2.0, places=9)

    def test_point_radius_circle(self):
        c = ec.point_radius_circle(P(0.1, 0.1), 0.25)
        self.assertAlmostEqual(c.radius, 0.25)

    def test_line_datum_parallel(self):
        line = ec.line_through_points(P(0.0, 0.0), P(1.0, 0.0))
        par = ec.line_datum_parallel_line(line, P(0.0, 0.3))
        self.assertTrue(line.is_parallel(par))
        self.assertTrue(par.contains(P(0.5, 0.3)))

    def test_line_circle_parallel_line_is_tangent(self):
        line = ec.line_through_points(P(-1.0, 0.0), P(1.0, 0.0))
        circle = ec.Circle(P(0.0, 0.0), 0.3)
        tan = ec.line_circle_parallel_line(line, circle)
        self.assertTrue(line.is_parallel(tan))
        self.assertAlmostEqual(abs(tan.signed_distance(circle.center)),
                               circle.radius, places=9)

    def test_circle_point_point_arc_midpoint_on_circle(self):
        c = ec.Circle(P(0.0, 0.0), 0.5, ccw=True)
        start = P(0.5, 0.0)
        end = P(0.0, 0.5)
        arc = ec.circle_point_point_arc(c, start, end)
        # ccw quarter arc: mid at 45 degrees
        self.assertTrue(arc.mid.almost_equals(
            P(0.5 * math.cos(math.pi / 4), 0.5 * math.sin(math.pi / 4)), tol=1e-9))
        self.assertAlmostEqual(c.center.dist(arc.mid), 0.5, places=9)

    def test_line_line_fillet_tangency(self):
        l1 = ec.line_through_points(P(-1.0, 0.0), P(1.0, 0.0))   # +x axis
        l2 = ec.line_through_points(P(0.0, -1.0), P(0.0, 1.0))   # +y axis
        r = 0.2
        arc = ec.line_line_fillet(l1, l2, r)
        center = ev._arc_center(arc)
        self.assertIsNotNone(center)
        # tangent points lie on the respective lines at distance r from centre
        self.assertAlmostEqual(center.dist(arc.start), r, places=9)
        self.assertAlmostEqual(center.dist(arc.end), r, places=9)
        self.assertAlmostEqual(center.dist(arc.mid), r, places=9)
        self.assertTrue(l1.contains(arc.start, tol=1e-9))
        self.assertTrue(l2.contains(arc.end, tol=1e-9))

    def test_symline_offset_pair_is_mirror(self):
        sym = ec.line_through_points(P(0.0, -1.0), P(0.0, 1.0))
        a, b = ec.symline_offset_line_line(sym, 0.3)
        # a and b are mirror images across the y-axis at x = -/+ 0.3
        # (sign follows the symmetry line's left normal).
        self.assertTrue(a.contains(P(-0.3, 0.0)))
        self.assertTrue(b.contains(P(0.3, 0.0)))
        self.assertTrue(sym.is_parallel(a))


class QuantizationTests(unittest.TestCase):
    def test_length_exact_endpoints(self):
        for v in (-1.0, 0.0, 1.0):
            self.assertAlmostEqual(
                dsl.dequantize_length(dsl.quantize_length(v)), v, places=12)

    def test_length_roundtrip_monotone(self):
        idx = dsl.quantize_length(0.5)
        self.assertGreater(dsl.dequantize_length(idx), 0.4)
        self.assertLess(dsl.dequantize_length(idx), 0.6)

    def test_point_center_exact(self):
        ix, iy = dsl.quantize_point(0.0, 0.0)
        x, y = dsl.dequantize_point(ix, iy)
        self.assertAlmostEqual(x, 0.0, places=12)
        self.assertAlmostEqual(y, 0.0, places=12)

    def test_angle_common_values_exact(self):
        for a in (0.0, math.pi, math.pi / 2, math.pi / 3):
            self.assertAlmostEqual(
                dsl.dequantize_angle(dsl.quantize_angle(a)), a, places=12)

    def test_angle_wraps(self):
        self.assertEqual(dsl.quantize_angle(2 * math.pi), 0)

    def test_infline_roundtrip(self):
        iphi, irho = dsl.quantize_infline(math.pi / 2, 0.0)
        phi, rho = dsl.dequantize_infline(iphi, irho)
        self.assertAlmostEqual(phi, math.pi / 2, places=12)
        self.assertAlmostEqual(rho, 0.0, places=12)


class DslTokenTests(unittest.TestCase):
    def _sample_sequence(self):
        seq = dsl.ConstructionSequence()
        seq.parameters[0] = 0.2
        seq.steps.append(dsl.Step(
            "LineOffsetLine", ("srcline",), ("l1",), (0,), False))
        seq.steps.append(dsl.Step(
            "LineXLine", ("l1", "srcline2"), ("v0",), (), False))
        seq.steps.append(dsl.Step(
            "PointRadiusCircle", ("v0",), ("c0",), (1,), True))
        seq.parameters[1] = 0.15
        return seq

    def test_tokenize_detokenize_roundtrip(self):
        seq = self._sample_sequence()
        toks = dsl.tokenize(seq)
        self.assertEqual(toks[0][0], dsl.START_OF_CONSTRUCTION)
        self.assertEqual(toks[-1][0], dsl.END_OF_CONSTRUCTION)
        back = dsl.detokenize(toks)
        self.assertEqual(len(back.steps), 3)
        self.assertEqual(back.steps[0].op, "LineOffsetLine")
        self.assertEqual(back.steps[0].param_indices, (0,))
        self.assertTrue(back.steps[2].creates_curve)
        self.assertAlmostEqual(back.parameters[0], 0.2, delta=0.02)

    def test_detokenize_rejects_bad_stream(self):
        with self.assertRaises(ValueError):
            dsl.detokenize([("Op", "LineXLine", ("a",), ("b",))])

    def test_vocabulary_is_injective(self):
        vocab = dsl.build_vocabulary()
        self.assertEqual(len(vocab), len(set(vocab.values())))
        self.assertIn("Op:LineLineFillet", vocab)
        self.assertIn("UseParameter31", vocab)


class CompilerTests(unittest.TestCase):
    def test_replay_undefined_input_raises(self):
        seq = dsl.ConstructionSequence()
        seq.steps.append(dsl.Step("LineReverseLine", ("nope",), ("r",)))
        with self.assertRaises(ReplayError):
            replay(seq, {})

    def test_replay_builds_square_profile(self):
        # Build a unit square [-0.25,0.25]^2 from four axis lines via LineXLine.
        prompt = {
            "top": ec.line_through_points(P(-1.0, 0.25), P(1.0, 0.25)),
            "right": ec.line_through_points(P(0.25, 1.0), P(0.25, -1.0)),
            "bottom": ec.line_through_points(P(1.0, -0.25), P(-1.0, -0.25)),
            "left": ec.line_through_points(P(-0.25, -1.0), P(-0.25, 1.0)),
        }
        seq = dsl.ConstructionSequence()
        seq.steps += [
            dsl.Step("LineXLine", ("left", "top"), ("tl",)),
            dsl.Step("LineXLine", ("top", "right"), ("tr",)),
            dsl.Step("LineXLine", ("right", "bottom"), ("br",)),
            dsl.Step("LineXLine", ("bottom", "left"), ("bl",)),
        ]
        env, created = replay(seq, prompt)
        self.assertTrue(env["tl"].almost_equals(P(-0.25, 0.25), tol=1e-9))
        self.assertTrue(env["tr"].almost_equals(P(0.25, 0.25), tol=1e-9))
        self.assertTrue(env["br"].almost_equals(P(0.25, -0.25), tol=1e-9))
        self.assertTrue(env["bl"].almost_equals(P(-0.25, -0.25), tol=1e-9))

    def test_assemble_square_into_closed_loop(self):
        corners = [P(-0.25, 0.25), P(0.25, 0.25), P(0.25, -0.25), P(-0.25, -0.25)]
        curves = [ec.Segment(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        prof = assemble_profile(curves)
        self.assertEqual(len(prof.loops), 1)
        self.assertEqual(len(prof.loops[0]), 4)

    def test_circle_isolated_into_own_loop(self):
        curves = [
            ec.Segment(P(-0.25, 0.25), P(0.25, 0.25)),
            ec.Segment(P(0.25, 0.25), P(-0.25, 0.25)),
            ec.Circle(P(0.0, 0.0), 0.1),
        ]
        prof = assemble_profile(curves)
        circle_loops = [l for l in prof.loops if isinstance(l, ec.Circle)]
        self.assertEqual(len(circle_loops), 1)

    def test_parametric_edit_changes_geometry(self):
        prompt = {"src": ec.line_through_points(P(0.0, 0.0), P(1.0, 0.0)),
                  "cut": ec.line_through_points(P(0.0, -1.0), P(0.0, 1.0))}
        seq = dsl.ConstructionSequence()
        seq.parameters[0] = 0.2
        seq.steps += [
            dsl.Step("LineOffsetLine", ("src",), ("o",), (0,)),
            dsl.Step("LineXLine", ("o", "cut"), ("v",)),
        ]
        env1, _ = replay(seq, prompt)
        seq.parameters[0] = 0.4
        env2, _ = replay(seq, prompt)
        self.assertAlmostEqual(env1["v"].y, 0.2, places=9)
        self.assertAlmostEqual(env2["v"].y, 0.4, places=9)


class ValidityTests(unittest.TestCase):
    def _square_prompt_seq(self):
        prompt = {
            "top": ec.line_through_points(P(-1.0, 0.25), P(1.0, 0.25)),
            "right": ec.line_through_points(P(0.25, 1.0), P(0.25, -1.0)),
            "bottom": ec.line_through_points(P(1.0, -0.25), P(-1.0, -0.25)),
            "left": ec.line_through_points(P(-0.25, -1.0), P(-0.25, 1.0)),
        }
        seq = dsl.ConstructionSequence()
        seq.steps += [
            dsl.Step("LineXLine", ("left", "top"), ("tl",)),
            dsl.Step("LineXLine", ("top", "right"), ("tr",)),
            dsl.Step("LineXLine", ("right", "bottom"), ("br",)),
            dsl.Step("LineXLine", ("bottom", "left"), ("bl",)),
        ]
        return prompt, seq

    def test_check_passes_for_valid_sequence(self):
        prompt, seq = self._square_prompt_seq()
        res = ev.check_sequence(seq, prompt)
        self.assertTrue(res.ok, res.errors)
        self.assertTrue(ev.syntactic_validity(seq, prompt))

    def test_check_flags_use_before_definition(self):
        prompt, _ = self._square_prompt_seq()
        seq = dsl.ConstructionSequence()
        seq.steps.append(dsl.Step("LineXLine", ("ghost", "top"), ("p",)))
        res = ev.check_sequence(seq, prompt)
        self.assertFalse(res.ok)
        self.assertTrue(any("before definition" in e for e in res.errors))

    def test_check_flags_wrong_type(self):
        prompt = {"pt": P(0.0, 0.0), "line": ec.line_through_points(P(0, 0), P(1, 0))}
        seq = dsl.ConstructionSequence()
        # LineXLine wants two lines; give it a point in slot 0.
        seq.steps.append(dsl.Step("LineXLine", ("pt", "line"), ("q",)))
        res = ev.check_sequence(seq, prompt)
        self.assertFalse(res.ok)
        self.assertTrue(any("expected line" in e for e in res.errors))

    def test_check_flags_missing_param_value(self):
        prompt = {"src": ec.line_through_points(P(0, 0), P(1, 0))}
        seq = dsl.ConstructionSequence()
        seq.steps.append(dsl.Step("LineOffsetLine", ("src",), ("o",), (0,)))
        # note: parameter 0 has no value set
        res = ev.check_sequence(seq, prompt)
        self.assertFalse(res.ok)
        self.assertTrue(any("no value" in e for e in res.errors))

    def test_syntactic_validity_false_on_parallel_intersection(self):
        prompt = {
            "a": ec.line_through_points(P(0, 0), P(1, 0)),
            "b": ec.line_through_points(P(0, 0.2), P(1, 0.2)),
        }
        seq = dsl.ConstructionSequence()
        seq.steps.append(dsl.Step("LineXLine", ("a", "b"), ("p",)))
        self.assertTrue(ev.check_sequence(seq, prompt).ok)  # syntactically fine
        self.assertFalse(ev.syntactic_validity(seq, prompt))  # replay fails

    def test_no_self_intersection_square_vs_bowtie(self):
        square = Profile(loops=[[
            ec.Segment(P(-0.25, 0.25), P(0.25, 0.25)),
            ec.Segment(P(0.25, 0.25), P(0.25, -0.25)),
            ec.Segment(P(0.25, -0.25), P(-0.25, -0.25)),
            ec.Segment(P(-0.25, -0.25), P(-0.25, 0.25)),
        ]])
        self.assertTrue(ev.no_self_intersection(square))
        bowtie = Profile(loops=[[
            ec.Segment(P(-0.25, 0.25), P(0.25, -0.25)),
            ec.Segment(P(0.25, -0.25), P(0.25, 0.25)),
            ec.Segment(P(0.25, 0.25), P(-0.25, -0.25)),
            ec.Segment(P(-0.25, -0.25), P(-0.25, 0.25)),
        ]])
        self.assertFalse(ev.no_self_intersection(bowtie))

    def test_no_short_edges(self):
        good = Profile(loops=[[
            ec.Segment(P(-0.25, 0.0), P(0.25, 0.0)),
            ec.Segment(P(0.25, 0.0), P(-0.25, 0.0)),
        ]])
        self.assertTrue(ev.no_short_edges(good))
        bad = Profile(loops=[[ec.Segment(P(0.0, 0.0), P(0.001, 0.0))]])
        self.assertFalse(ev.no_short_edges(bad))

    def test_profile_area_square(self):
        square = Profile(loops=[[
            ec.Segment(P(-0.25, 0.25), P(0.25, 0.25)),
            ec.Segment(P(0.25, 0.25), P(0.25, -0.25)),
            ec.Segment(P(0.25, -0.25), P(-0.25, -0.25)),
            ec.Segment(P(-0.25, -0.25), P(-0.25, 0.25)),
        ]])
        self.assertAlmostEqual(ev.profile_area(square), 0.25, places=6)

    def test_end_to_end_compile_square_profile_valid(self):
        prompt = {
            "top": ec.line_through_points(P(-1.0, 0.25), P(1.0, 0.25)),
            "right": ec.line_through_points(P(0.25, 1.0), P(0.25, -1.0)),
            "bottom": ec.line_through_points(P(1.0, -0.25), P(-1.0, -0.25)),
            "left": ec.line_through_points(P(-0.25, -1.0), P(-0.25, 1.0)),
        }
        seq = dsl.ConstructionSequence()
        seq.steps += [
            dsl.Step("LineXLine", ("left", "top"), ("tl",)),
            dsl.Step("LineXLine", ("top", "right"), ("tr",)),
            dsl.Step("LineXLine", ("right", "bottom"), ("br",)),
            dsl.Step("LineXLine", ("bottom", "left"), ("bl",)),
        ]
        # Emit the four edges as created curves via segments in a follow-up.
        env, _ = replay(seq, prompt)
        square = Profile(loops=[[
            ec.Segment(env["tl"], env["tr"]),
            ec.Segment(env["tr"], env["br"]),
            ec.Segment(env["br"], env["bl"]),
            ec.Segment(env["bl"], env["tl"]),
        ]])
        self.assertTrue(ev.no_self_intersection(square))
        self.assertTrue(ev.no_short_edges(square))
        self.assertAlmostEqual(ev.profile_area(square), 0.25, places=6)


class AccuracyMetricTests(unittest.TestCase):
    def test_perfect_match(self):
        pred = [P(0.1, 0.2), ec.Circle(P(0, 0), 0.3, ccw=True)]
        ref = [P(0.1, 0.2), ec.Circle(P(0, 0), 0.3, ccw=True)]
        m = ev.construction_accuracy(pred, ref)
        self.assertAlmostEqual(m["mean_dist"], 0.0, places=12)
        self.assertAlmostEqual(m["mean_flag_agree"], 1.0)
        self.assertAlmostEqual(m["solution_exists"], 1.0)

    def test_flag_disagreement(self):
        pred = [ec.Circle(P(0, 0), 0.3, ccw=True)]
        ref = [ec.Circle(P(0, 0), 0.3, ccw=False)]
        m = ev.construction_accuracy(pred, ref)
        self.assertAlmostEqual(m["mean_flag_agree"], 0.0)

    def test_distance_error(self):
        pred = [P(0.1, 0.0)]
        ref = [P(0.0, 0.0)]
        m = ev.construction_accuracy(pred, ref)
        self.assertAlmostEqual(m["mean_dist"], 0.1, places=12)

    def test_type_mismatch_marks_unsolved(self):
        pred = [P(0.0, 0.0)]
        ref = [ec.Circle(P(0, 0), 0.1)]
        m = ev.construction_accuracy(pred, ref)
        self.assertAlmostEqual(m["solution_exists"], 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ev.construction_accuracy([P(0, 0)], [])


def _ang(a):
    """Normalise an angle onto [0, pi) for direction comparison."""
    a = a % math.pi
    return a


if __name__ == "__main__":
    unittest.main()
