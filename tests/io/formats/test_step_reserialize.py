import unittest

from harnesscad.io.formats.step import parse, entity_refs, Real
from harnesscad.io.formats import step_reserialize as sr


SAMPLE = """
ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#10 = CLOSED_SHELL('shell', (#20, #30));
#20 = ADVANCED_FACE('f1', (#40), #50, .T.);
#30 = ADVANCED_FACE('f2', (#40), #50, .F.);
#40 = CARTESIAN_POINT('p', (1.2500000, 2.0, 3.333333333));
#50 = PLANE('pl', #40);
ENDSEC;
END-ISO-10303-21;
"""


class TestDfsOrder(unittest.TestCase):
    def setUp(self):
        self.step = parse(SAMPLE)

    def test_root_first(self):
        order = sr.dfs_order(self.step)
        self.assertEqual(order[0], 10)  # CLOSED_SHELL is the only root

    def test_all_ids_once(self):
        order = sr.dfs_order(self.step)
        self.assertEqual(sorted(order), [10, 20, 30, 40, 50])
        self.assertEqual(len(order), len(set(order)))

    def test_depth_first_locality(self):
        # After the shell, face #20 then its children before sibling #30.
        order = sr.dfs_order(self.step)
        self.assertLess(order.index(20), order.index(30))
        self.assertLess(order.index(40), order.index(30))


class TestReserialize(unittest.TestCase):
    def setUp(self):
        self.step = parse(SAMPLE)
        self.reser = sr.reserialize(self.step, digits=3)

    def test_sequential_ids(self):
        self.assertEqual(self.reser.order, [1, 2, 3, 4, 5])

    def test_topology_preserved(self):
        # The renumbered shell must still reference two faces.
        shell = self.reser.entities[1]
        self.assertEqual(shell.keyword, "CLOSED_SHELL")
        self.assertEqual(len(entity_refs(shell)), 2)

    def test_real_normalization(self):
        # 3.333333333 rounded to 3 decimals -> 3.333
        pt = next(e for e in self.reser.entities.values()
                  if e.keyword == "CARTESIAN_POINT")
        reals = [p for grp in pt.params if isinstance(grp, list) for p in grp
                 if isinstance(p, Real)]
        texts = {r.text for r in reals}
        self.assertIn("3.333", texts)
        self.assertIn("2.", texts)  # integer-valued real keeps a point

    def test_deterministic(self):
        a = sr.annotated_text(self.step, digits=3)
        b = sr.annotated_text(self.step, digits=3)
        self.assertEqual(a, b)


class TestBranchAnnotations(unittest.TestCase):
    def setUp(self):
        self.step = parse(SAMPLE)

    def test_subtree_and_depth(self):
        stats = sr.branch_annotations(self.step)
        shell = stats[10]
        # reachable: 10,20,30,40,50 = 5
        self.assertEqual(shell.subtree_size, 5)
        self.assertEqual(shell.children, 2)
        self.assertGreaterEqual(shell.depth, 2)

    def test_leaf(self):
        stats = sr.branch_annotations(self.step)
        pt = stats[40]  # CARTESIAN_POINT references nothing
        self.assertEqual(pt.children, 0)
        self.assertEqual(pt.subtree_size, 1)
        self.assertEqual(pt.depth, 0)

    def test_annotated_text_has_comments(self):
        txt = sr.annotated_text(self.step, digits=3)
        self.assertIn("/* c=", txt)
        self.assertIn("CLOSED_SHELL", txt)


if __name__ == "__main__":
    unittest.main()
