"""Pack the CI test matrix by MEASURED COST rather than by name.

The runner executes every test unit in its own process (a monolithic
``unittest discover`` segfaults at OCCT teardown, so per-process isolation is why
a green tick means anything). What changed here is only WHICH units go to which
shard.

The old rule was ``index % nshard`` over the sorted file list. That is oblivious
to cost, and the cost distribution is brutally skewed: of 1165 modules, 52 carry
96% of the wall time and two of them run ~850s each. Round-robin dealt those two
into different shards and left others nearly empty -- a measured
``tests(7)=892s`` against ``tests(2)=205s``. The matrix finishes when its SLOWEST
shard finishes, so more than half the fleet's time was spent idle.

Two changes fix it:

* **Longest-processing-time bin packing.** Sort the units by measured cost and
  drop each into whichever shard is currently lightest. LPT is the classic greedy
  makespan heuristic and is provably within 4/3 of optimal, which is far inside
  the noise of a CI runner.

* **Split the giants by TestCase class.** A module is atomic to ``unittest``, so
  the slowest single module is a FLOOR no packing can get under -- 874s against a
  perfect-balance target of 458s. Modules above :data:`SPLIT_THRESHOLD_S` are
  therefore expanded into their ``TestCase`` classes, which the runner can invoke
  individually (``python -m unittest tests.x.test_y.TestZ``). None of the heavy
  modules use ``setUpClass``/``setUpModule``, so splitting re-pays no shared
  fixture; it costs one extra interpreter start per class.

The cost table is :data:`RUNTIMES_PATH`, scraped from the ``MODTIME`` lines the
runner prints. It is a hint, never a correctness input: a module missing from it
(a new test, a renamed one) is estimated at the recorded median and still runs.
The UNION of shards is always every unit, which :mod:`tests.test_shard_plan`
asserts.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import sys
from typing import Dict, List, Tuple

TESTS_DIR = pathlib.Path(__file__).resolve().parent
RUNTIMES_PATH = TESTS_DIR / "module_runtimes.json"

#: A module costing more than this is expanded into its TestCase classes. Set
#: just under the perfect-balance target (total/nshard) so that only genuine
#: floor-setters are split and the process count stays close to the module count.
SPLIT_THRESHOLD_S = 60.0


def load_runtimes() -> Tuple[Dict[str, float], float]:
    """The measured cost table and the fallback for anything absent from it."""
    try:
        with open(RUNTIMES_PATH, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return {}, 0.1
    return dict(doc.get("modules", {})), float(doc.get("_default_seconds", 0.1))


def test_modules() -> List[str]:
    """Every test module, as a dotted name, in a stable order."""
    out = []
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        rel = path.relative_to(TESTS_DIR.parent)
        out.append(str(rel.with_suffix("")).replace(os.sep, ".").replace("/", "."))
    return out


def testcase_classes(module: str) -> List[str]:
    """The TestCase class names in ``module``, read statically.

    ``ast``, not ``import``: importing a test module here would run its
    module-level code in the planner, which is both slow and a side effect. A
    class is taken to be a test class when any base's name ends in ``TestCase``
    or starts with ``Test`` -- the same shape the suite's own collectability
    guard relies on.
    """
    path = TESTS_DIR.parent / (module.replace(".", os.sep) + ".py")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    names = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
            if name.endswith("TestCase") or name.startswith("Test"):
                names.append(node.name)
                break
    return names


def testcase_methods(module: str) -> List[str]:
    """``Class.method`` for every test method in ``module``, read statically."""
    path = TESTS_DIR.parent / (module.replace(".", os.sep) + ".py")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    wanted = set(testcase_classes(module))
    out = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in wanted:
            for sub in node.body:
                if (isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and sub.name.startswith("test")):
                    out.append("%s.%s" % (node.name, sub.name))
    return out


def split_of(module: str) -> List[str]:
    """The finest useful sub-units of a module, or [] if it must stay atomic.

    Classes first, because a class is the coarsest split that removes the floor
    and costs the fewest extra interpreter starts. A module that is one big class
    (``test_reference_loop`` is seven tests in one) would still be a floor, so it
    falls through to its individual methods.
    """
    classes = testcase_classes(module)
    if len(classes) >= 2:
        return classes
    methods = testcase_methods(module)
    return methods if len(methods) >= 2 else []


def units() -> List[Tuple[str, float]]:
    """(unit, estimated seconds) for the whole suite, heaviest first.

    A unit is a module, except for modules over :data:`SPLIT_THRESHOLD_S`, which
    are split by :func:`split_of`.

    A split part is costed from its OWN measurement when the table has one, and
    only falls back to an even share of the module otherwise. The even share is a
    poor estimate and was measurably so: splitting the giants evenly predicted
    455s per shard and delivered 577s, because a module's cost is rarely spread
    evenly across its classes -- one class usually dominates and lands in a shard
    that then finishes late. The runner records per-unit times precisely so this
    guess can be replaced by a measurement on the next regeneration.
    """
    runtimes, default = load_runtimes()
    out: List[Tuple[str, float]] = []
    for module in test_modules():
        cost = runtimes.get(module, default)
        parts = split_of(module) if cost > SPLIT_THRESHOLD_S else []
        if not parts:
            out.append((module, cost))
            continue
        names = ["%s.%s" % (module, part) for part in parts]
        measured = {n: runtimes[n] for n in names if n in runtimes}
        # Whatever is unmeasured shares out the module time the measured parts
        # do not already account for, so the parts still sum to the module.
        rest = max(cost - sum(measured.values()), 0.0)
        spare = len(names) - len(measured)
        each = (rest / spare) if spare else 0.0
        for name in names:
            out.append((name, measured.get(name, each)))
    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out


def pack(all_units: List[Tuple[str, float]], nshard: int) -> List[List[str]]:
    """Longest-processing-time bin packing: heaviest unit to the lightest shard."""
    loads = [0.0] * nshard
    bins: List[List[str]] = [[] for _ in range(nshard)]
    for unit, cost in all_units:
        i = loads.index(min(loads))
        bins[i].append(unit)
        loads[i] += cost
    return bins


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="shard_plan",
        description="Print the test units assigned to one CI shard.")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--explain", action="store_true",
                        help="report the predicted load of every shard instead "
                             "of printing one shard's units")
    args = parser.parse_args(argv)

    nshard = max(1, args.nshard)
    all_units = units()
    bins = pack(all_units, nshard)

    if args.explain:
        cost = dict(all_units)
        total = sum(cost.values())
        heaviest = all_units[0] if all_units else ("-", 0.0)
        print("units       : %d (from %d modules)" % (len(all_units), len(test_modules())))
        print("total       : %.0fs serial" % total)
        print("perfect     : %.0fs per shard at nshard=%d" % (total / nshard, nshard))
        print("floor       : %.0fs (%s) -- no shard can beat its slowest unit"
              % (heaviest[1], heaviest[0]))
        for i, b in enumerate(bins):
            print("  shard %-2d  %7.0fs  %4d units" % (i, sum(cost[u] for u in b), len(b)))
        return 0

    if not 0 <= args.shard < nshard:
        raise SystemExit("--shard must be in [0, %d)" % nshard)
    for unit in sorted(bins[args.shard]):
        print(unit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
