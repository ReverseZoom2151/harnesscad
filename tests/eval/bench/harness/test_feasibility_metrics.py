import unittest

from harnesscad.eval.bench.harness.feasibility import (
    FeasibilityResult,
    FeasibilityTracker,
    Percentiles,
    aggregate_feasibility,
)


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


class FeasibilityTrackerTests(unittest.TestCase):
    def test_tracks_first_valid_attempt(self):
        clock = FakeClock(10)
        tracker = FeasibilityTracker(clock)
        clock.value = 11
        self.assertIsNone(tracker.record_attempt(valid=False, solver_calls=2))
        clock.value = 14.5
        result = tracker.record_attempt(valid=True, solver_calls=1)
        self.assertEqual(
            FeasibilityResult(True, 2, 3, 4.5),
            result,
        )

    def test_snapshot_before_success_is_right_censored(self):
        tracker = FeasibilityTracker(FakeClock())
        tracker.record_attempt(valid=False, solver_calls=3)
        self.assertEqual(
            FeasibilityResult(False, 1, 3, None),
            tracker.snapshot(),
        )

    def test_first_valid_is_frozen(self):
        clock = FakeClock()
        tracker = FeasibilityTracker(clock)
        clock.value = 2
        first = tracker.record_attempt(valid=True, solver_calls=1)
        clock.value = 99
        second = tracker.record_attempt(valid=False, solver_calls=100)
        self.assertIs(first, second)
        self.assertEqual(FeasibilityResult(True, 1, 1, 2), tracker.snapshot())

    def test_zero_elapsed_is_valid(self):
        tracker = FeasibilityTracker(FakeClock(5))
        self.assertEqual(0, tracker.record_attempt(valid=True).elapsed_seconds)

    def test_rejects_bad_solver_counts(self):
        tracker = FeasibilityTracker(FakeClock())
        for value, error in ((-1, ValueError), (1.5, TypeError), (True, TypeError)):
            with self.subTest(value=value), self.assertRaises(error):
                tracker.record_attempt(valid=False, solver_calls=value)

    def test_rejects_backwards_clock_at_success(self):
        clock = FakeClock(10)
        tracker = FeasibilityTracker(clock)
        clock.value = 9
        with self.assertRaisesRegex(ValueError, "backwards"):
            tracker.record_attempt(valid=True)

    def test_requires_injected_clock(self):
        with self.assertRaises(TypeError):
            FeasibilityTracker(None)


class AggregateFeasibilityTests(unittest.TestCase):
    def test_empty_aggregate(self):
        aggregate = aggregate_feasibility([])
        self.assertEqual(0, aggregate.runs)
        self.assertEqual(0.0, aggregate.success_rate)
        self.assertEqual(Percentiles(None, None), aggregate.elapsed_seconds)

    def test_failures_affect_rate_not_percentiles(self):
        aggregate = aggregate_feasibility([
            FeasibilityResult(False, 10, 20, None),
            FeasibilityResult(True, 2, 4, 1.5),
        ])
        self.assertEqual(2, aggregate.runs)
        self.assertEqual(1, aggregate.successes)
        self.assertEqual(0.5, aggregate.success_rate)
        self.assertEqual(Percentiles(2.0, 2.0), aggregate.attempts)
        self.assertEqual(Percentiles(1.5, 1.5), aggregate.elapsed_seconds)

    def test_nearest_rank_p50_p95_are_deterministic(self):
        results = [
            FeasibilityResult(True, n, n * 2, float(n) / 10)
            for n in range(1, 21)
        ]
        aggregate = aggregate_feasibility(reversed(results))
        self.assertEqual(Percentiles(10.0, 19.0), aggregate.attempts)
        self.assertEqual(Percentiles(20.0, 38.0), aggregate.solver_calls)
        self.assertEqual(Percentiles(1.0, 1.9), aggregate.elapsed_seconds)

    def test_input_is_not_mutated(self):
        results = [
            FeasibilityResult(True, 2, 3, 1.0),
            FeasibilityResult(True, 1, 2, 0.5),
        ]
        before = list(results)
        aggregate_feasibility(results)
        self.assertEqual(before, results)


if __name__ == "__main__":
    unittest.main()
