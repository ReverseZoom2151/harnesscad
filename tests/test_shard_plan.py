"""The CI shard plan must be a PARTITION, or the suite silently shrinks.

``shard_plan`` decides which test units each CI shard runs. A packing bug here
does not announce itself: a unit dropped from every shard simply never runs, and
the matrix goes green because nothing failed. That is the same silent-green
failure mode as a pytest-style file the runner never collects (see
:mod:`tests.test_suite_collectable`), so it gets the same kind of guard.

The invariants that matter:

* every module is covered, at whatever granularity it was split to;
* no unit is run twice (wasted runner minutes, and a flaky test gets two rolls);
* the plan is deterministic, since two shards computing different plans from the
  same tree would drop units without any single shard noticing.
"""

from __future__ import annotations

import unittest

from tests import shard_plan


class ShardPlanIsAPartition(unittest.TestCase):
    NSHARD = 8

    @classmethod
    def setUpClass(cls):
        # Computed once: units() walks the whole tests tree and parses the
        # split candidates, which is not something to repeat per test method.
        cls.all_units = shard_plan.units()
        cls.bins = shard_plan.pack(cls.all_units, cls.NSHARD)
        cls.modules = set(shard_plan.test_modules())

    def test_every_unit_lands_in_exactly_one_shard(self):
        assigned = [u for b in self.bins for u in b]
        self.assertEqual(
            sorted(assigned), sorted(u for u, _ in self.all_units),
            "the shards must partition the unit list -- a unit in no shard never "
            "runs, and a unit in two shards burns a runner twice")
        self.assertEqual(len(assigned), len(set(assigned)), "a unit is duplicated")

    def test_every_test_module_is_still_covered(self):
        # A split module contributes `module.Class[.method]` units rather than
        # the bare module, so compare on the module prefix.
        modules = self.modules
        covered = set()
        for unit, _ in self.all_units:
            mod = unit
            while mod and mod not in modules:
                mod = mod.rpartition(".")[0]
            if mod:
                covered.add(mod)
        self.assertEqual(covered, modules,
                         "a test module vanished from the plan entirely")

    def test_the_plan_is_deterministic(self):
        again = shard_plan.pack(shard_plan.units(), self.NSHARD)
        self.assertEqual(self.bins, again,
                         "two shards computing different plans would drop units")

    def test_packing_beats_round_robin_on_the_measured_data(self):
        # The point of the exercise: the old `index % nshard` rule left the
        # matrix waiting on one shard. Packing must actually even the load out,
        # or the extra machinery is not paying for itself.
        cost = dict(self.all_units)
        loads = [sum(cost[u] for u in b) for b in self.bins]
        by_name = [[] for _ in range(self.NSHARD)]
        for i, (unit, _) in enumerate(sorted(self.all_units)):
            by_name[i % self.NSHARD].append(unit)
        rr_loads = [sum(cost[u] for u in b) for b in by_name]
        self.assertLess(max(loads), max(rr_loads),
                        "packing did not reduce the slowest shard")

    def test_a_module_absent_from_the_runtime_table_still_runs(self):
        # The cost table is a hint, never a correctness input: a brand-new test
        # file nobody has measured must still be assigned somewhere.
        runtimes, default = shard_plan.load_runtimes()
        self.assertGreater(default, 0.0, "unmeasured modules need a positive estimate")
        unmeasured = [m for m in shard_plan.test_modules() if m not in runtimes]
        self.assertTrue(unmeasured, "expected most modules to be unmeasured noise")
        assigned = {u for b in self.bins for u in b}
        self.assertIn(unmeasured[0], assigned)


if __name__ == "__main__":
    unittest.main()
