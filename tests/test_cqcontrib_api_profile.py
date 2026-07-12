import unittest

from programs.cqcontrib_api_profile import (
    ApiProfile,
    arity_violations,
    format_profile,
    profile_source,
    profile_sources,
    selector_strings,
    unknown_methods,
)

SRC = """
import cadquery as cq
result = (cq.Workplane("XY").box(2, 2, 2)
          .faces(">Z").shell(-0.2)
          .faces(">Z").edges("not(<X or >X)")
          .chamfer(0.125))
other = cq.Workplane("XY").rect(1, 2).extrude(3, False)
"""


class TestProfileSource(unittest.TestCase):
    def test_method_counts(self):
        p = profile_source(SRC)
        self.assertEqual(p.sources, 1)
        self.assertEqual(p.methods["faces"].count, 2)
        self.assertEqual(p.methods["box"].count, 1)
        self.assertIn("extrude", p.methods)

    def test_arity_recorded(self):
        p = profile_source(SRC)
        self.assertEqual(p.methods["box"].arity_counts, {3: 1})
        self.assertEqual(p.methods["extrude"].arity_counts, {2: 1})
        self.assertEqual(p.methods["faces"].min_args, 1)

    def test_selectors_collected(self):
        p = profile_source(SRC)
        self.assertEqual(p.selectors[">Z"], 2)
        self.assertEqual(p.selectors["not(<X or >X)"], 1)
        self.assertEqual(selector_strings(p)[0], (">Z", 2))

    def test_kwargs(self):
        p = profile_source("a = w.cskHole(diameter=1, cskDiameter=2, cskAngle=90)")
        self.assertEqual(p.methods["cskHole"].kwargs["cskAngle"], 1)
        self.assertEqual(p.methods["cskHole"].arity_counts, {0: 1})

    def test_chain_shape(self):
        p = profile_source("r = w.faces('>Z').workplane().hole(2)")
        chains = p.top_chains(5)
        self.assertEqual(chains[0][0], ("faces", "workplane", "hole"))
        self.assertEqual(chains[0][1], 1)

    def test_determinism(self):
        self.assertEqual(profile_source(SRC).as_dict(),
                         profile_source(SRC).as_dict())


class TestCorpusDiff(unittest.TestCase):
    def setUp(self):
        self.p = profile_sources([SRC, "x = w.cboreHole(2, 4, 1).loft()"])

    def test_unknown_methods(self):
        known = {"box": (3, 4), "faces": (0, 1), "extrude": (1, 2)}
        unknown = dict(unknown_methods(self.p, known))
        self.assertIn("shell", unknown)
        self.assertIn("cboreHole", unknown)
        self.assertNotIn("box", unknown)

    def test_arity_violation_detected(self):
        known = {"extrude": (1, 1)}  # declared max 1, corpus uses 2
        viol = arity_violations(self.p, known)
        self.assertEqual(viol, [("extrude", 2, 1, 1)])

    def test_no_violation_when_range_ok(self):
        self.assertEqual(arity_violations(self.p, {"extrude": (1, 2)}), [])

    def test_report_is_text(self):
        text = format_profile(self.p, top=3)
        self.assertIn("distinct methods", text)
        self.assertIn("top chains", text)

    def test_empty_profile(self):
        p = ApiProfile()
        self.assertEqual(p.top_methods(), [])
        self.assertEqual(unknown_methods(p, {}), [])


class TestHarnessGap(unittest.TestCase):
    def test_gap_against_t2cq_ast(self):
        from programs.t2cq_ast import CHAIN_METHODS
        p = profile_source(SRC)
        gaps = dict(unknown_methods(p, CHAIN_METHODS))
        # 'shell' is genuinely absent from the harness CHAIN_METHODS table.
        self.assertIn("shell", gaps)
        self.assertNotIn("box", gaps)


if __name__ == "__main__":
    unittest.main()
