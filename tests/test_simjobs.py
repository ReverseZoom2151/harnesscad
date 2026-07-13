import unittest

from harnesscad.eval.quality.simjobs import (
    FEASolverAdapter,
    JobState,
    SimulationJobs,
    SolverProvenance,
    content_key,
)


class Clock:
    def __init__(self):
        self.value = 10.0

    def __call__(self):
        self.value += 1.0
        return self.value


class Solver:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def solve(self, payload):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return {"input": payload, "answer": outcome}


class TestSimulationJobs(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.prov = SolverProvenance(
            "calculix", "2.22", executable_digest="sha256:abc",
            configuration={"threads": 1},
        )

    def test_content_key_is_order_independent_and_provenance_sensitive(self):
        self.assertEqual(
            content_key({"b": 2, "a": [1]}, self.prov),
            content_key({"a": [1], "b": 2}, self.prov),
        )
        other = SolverProvenance("calculix", "2.23")
        self.assertNotEqual(content_key({"a": 1}, self.prov), content_key({"a": 1}, other))

    def test_success_records_provenance_and_caches(self):
        jobs = SimulationJobs(clock=self.clock)
        solver = Solver([42])
        first = jobs.run_now({"mesh": "m1"}, self.prov, solver)
        self.assertEqual(first.state, JobState.SUCCEEDED)
        self.assertEqual(first.attempt, 1)
        self.assertFalse(first.cache_hit)
        self.assertEqual(first.provenance.version, "2.22")

        second = jobs.run_now({"mesh": "m1"}, self.prov, solver)
        self.assertEqual(second.state, JobState.SUCCEEDED)
        self.assertTrue(second.cache_hit)
        self.assertEqual(second.result, first.result)
        self.assertEqual(solver.calls, 1)

    def test_failure_retries_to_success(self):
        jobs = SimulationJobs(clock=self.clock)
        job = jobs.run_now(
            {"case": 1}, self.prov, Solver([RuntimeError("transient"), 7]),
            max_retries=1,
        )
        self.assertEqual(job.state, JobState.SUCCEEDED)
        self.assertEqual(job.attempt, 2)
        self.assertIsNone(job.error)

    def test_failed_after_retry_budget(self):
        jobs = SimulationJobs(clock=self.clock)
        job = jobs.run_now(
            {}, self.prov, Solver([ValueError("bad"), ValueError("still bad")]),
            max_retries=1,
        )
        self.assertEqual(job.state, JobState.FAILED)
        self.assertEqual(job.attempt, 2)
        self.assertIn("still bad", job.error)
        self.assertIsNotNone(job.finished_at)

    def test_timeout_is_typed_and_retriable(self):
        timeouts = []

        def executor(call, timeout):
            timeouts.append(timeout)
            return call()

        jobs = SimulationJobs(clock=self.clock, executor=executor)
        job = jobs.run_now(
            {}, self.prov, Solver([TimeoutError("deadline"), 9]),
            timeout_s=2.5, max_retries=1,
        )
        self.assertEqual(job.state, JobState.SUCCEEDED)
        self.assertEqual(timeouts, [2.5, 2.5])

    def test_pending_job_can_be_cancelled_idempotently(self):
        jobs = SimulationJobs(clock=self.clock)
        job = jobs.submit({}, self.prov)
        jobs.cancel(job.id)
        jobs.cancel(job.id)
        jobs.run(job.id, Solver([1]))
        self.assertEqual(job.state, JobState.CANCELLED)
        self.assertEqual(job.attempt, 0)

    def test_executor_can_request_cancellation_while_running(self):
        holder = {}

        def executor(call, timeout):
            del timeout
            holder["jobs"].cancel(holder["job"].id)
            return call()

        jobs = SimulationJobs(clock=self.clock, executor=executor)
        job = jobs.submit({}, self.prov)
        holder.update(jobs=jobs, job=job)
        jobs.run(job.id, Solver([1]))
        self.assertEqual(job.state, JobState.CANCELLED)
        self.assertNotIn(job.cache_key, jobs.cache)

    def test_cached_none_is_a_valid_hit(self):
        key = content_key({}, self.prov)
        jobs = SimulationJobs(clock=self.clock, cache={key: None})
        job = jobs.submit({}, self.prov)
        self.assertTrue(job.cache_hit)
        self.assertEqual(job.state, JobState.SUCCEEDED)

    def test_fea_solver_two_argument_protocol_adapts_without_import(self):
        class FEA:
            def solve(self, mesh, load_case):
                return (mesh, load_case)

        jobs = SimulationJobs(clock=self.clock)
        job = jobs.run_now(
            {"mesh": "mesh-1", "load_case": {"force": 10}},
            self.prov,
            FEASolverAdapter(FEA()),
        )
        self.assertEqual(job.result, ("mesh-1", {"force": 10}))

    def test_invalid_options_and_payload_rejected(self):
        jobs = SimulationJobs(clock=self.clock)
        with self.assertRaises(ValueError):
            jobs.submit({}, self.prov, timeout_s=0)
        with self.assertRaises(ValueError):
            jobs.submit({}, self.prov, max_retries=-1)
        with self.assertRaises(TypeError):
            jobs.submit({"bad": object()}, self.prov)


if __name__ == "__main__":
    unittest.main()
