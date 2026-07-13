"""Tests for DeepCAD loop / profile assembly."""

import unittest

from harnesscad.domain.reconstruction.tokens.deepcad_command_spec import Command, command, SOL, LINE, CIRCLE, EXT, EOS
from harnesscad.domain.reconstruction.sketch import deepcad_profile_assembly as pa


def _square():
    # Unit square loop: four line curves whose absolute endpoints are the corners.
    return [
        Command(SOL),
        command(LINE, x=1.0, y=0.0),
        command(LINE, x=1.0, y=1.0),
        command(LINE, x=0.0, y=1.0),
        command(LINE, x=0.0, y=0.0),
    ]


def _ext():
    return command(EXT, theta=0, phi=0, gamma=0, px=0, py=0, pz=0,
                   s=1, e1=0.5, e2=0, b=0, u=0)


class TestSplit(unittest.TestCase):
    def test_split_loops_two(self):
        cmds = _square() + [Command(SOL), command(CIRCLE, x=0.5, y=0.5, r=0.2)]
        loops = pa.split_loops(cmds)
        self.assertEqual(len(loops), 2)
        self.assertEqual(loops[0][0].type, SOL)
        self.assertEqual(loops[1][1].type, CIRCLE)

    def test_split_stops_at_ext(self):
        cmds = _square() + [_ext(), Command(SOL), command(LINE, x=1, y=1)]
        loops = pa.split_loops(cmds)
        self.assertEqual(len(loops), 1)

    def test_split_profiles(self):
        cmds = _square() + [_ext()] + [Command(SOL), command(CIRCLE, x=0, y=0, r=0.3), _ext(), Command(EOS)]
        profiles = pa.split_profiles(cmds)
        self.assertEqual(len(profiles), 2)
        loops0, ext0 = profiles[0]
        self.assertEqual(len(loops0), 1)
        self.assertEqual(ext0.type, EXT)

    def test_incomplete_profile_ignored(self):
        # Sketch with no closing Ext -> no profile emitted.
        self.assertEqual(pa.split_profiles(_square()), [])


class TestReconstruct(unittest.TestCase):
    def test_closure_chaining(self):
        segs = pa.reconstruct_segments(_square())
        self.assertEqual(len(segs), 4)
        # First curve starts where the last curve ends (closure).
        self.assertEqual(segs[0].start, (0.0, 0.0))
        self.assertEqual(segs[0].end, (1.0, 0.0))
        self.assertEqual(segs[-1].end, segs[0].start)
        # Each start equals the predecessor's end.
        for prev, cur in zip(segs, segs[1:]):
            self.assertEqual(cur.start, prev.end)

    def test_circle_is_standalone(self):
        loop = [Command(SOL), command(CIRCLE, x=0.5, y=0.5, r=0.25)]
        segs = pa.reconstruct_segments(loop)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].type, CIRCLE)
        self.assertEqual(segs[0].start, (0.5, 0.5))
        self.assertEqual(segs[0].end, (0.5, 0.5))


class TestBBox(unittest.TestCase):
    def test_square_bbox(self):
        self.assertEqual(pa.loop_bbox(_square()), (0.0, 0.0, 1.0, 1.0))

    def test_circle_bbox_uses_radius(self):
        loop = [Command(SOL), command(CIRCLE, x=1.0, y=1.0, r=0.5)]
        self.assertEqual(pa.loop_bbox(loop), (0.5, 0.5, 1.5, 1.5))


class TestCanonical(unittest.TestCase):
    def test_canonical_loop_starts_bottom_left(self):
        # Rotate the square so it doesn't begin at (0,0); canonicalisation should
        # restore a first curve whose start vertex is the bottom-left (0,0).
        rotated = [
            Command(SOL),
            command(LINE, x=0.0, y=1.0),
            command(LINE, x=0.0, y=0.0),
            command(LINE, x=1.0, y=0.0),
            command(LINE, x=1.0, y=1.0),
        ]
        canon = pa.canonical_loop(rotated)
        segs = pa.reconstruct_segments(canon)
        self.assertEqual(segs[0].start, (0.0, 0.0))

    def test_canonical_loop_preserves_count(self):
        canon = pa.canonical_loop(_square())
        self.assertEqual(canon[0].type, SOL)
        self.assertEqual(len(pa._curves(canon)), 4)

    def test_sort_loops_by_bbox_corner(self):
        far = [Command(SOL), command(CIRCLE, x=5.0, y=5.0, r=0.1)]
        near = _square()
        ordered = pa.sort_loops([far, near])
        self.assertEqual(pa.loop_bbox(ordered[0])[:2], (0.0, 0.0))

    def test_canonical_profile(self):
        far = [Command(SOL), command(CIRCLE, x=5.0, y=5.0, r=0.1)]
        prof = pa.canonical_profile([far, _square()])
        self.assertEqual(pa.loop_bbox(prof[0])[:2], (0.0, 0.0))
        self.assertEqual(pa.reconstruct_segments(prof[1])[0].start, (5.0, 5.0))


if __name__ == "__main__":
    unittest.main()
