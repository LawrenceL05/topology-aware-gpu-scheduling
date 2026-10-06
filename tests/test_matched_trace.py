import json
import unittest
from unittest.mock import Mock, patch

from topology_scheduler import Node, PolicyName, TraceJob, Workload, run_matched_trace


class MatchedTraceTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [Node("a", "A", 2, 80)]
        self.workload = Workload(2, 40, {"A": 1})
        self.worker = lambda rank: rank
        self.jobs = [TraceJob(name, self.workload, self.worker) for name in ("first", "second")]

    def run_trace(self, jobs=None, **options):
        return run_matched_trace(
            self.nodes, self.jobs if jobs is None else jobs, {},
            accelerator_type="A", **options,
        )

    def test_every_policy_replays_same_order_workers_and_backend_options(self):
        seen = []

        def backend(plan, worker, **options):
            seen.append((plan, worker, options))
            return [worker(rank) for rank in range(len(plan.workers))]

        # A generator must be materialized once, not exhausted after the baseline.
        records = self.run_trace(iter(self.jobs), backend=backend, execution_timeout=7)
        self.assertEqual([(r.planning.policy, r.job_id) for r in records],
                         [(p.value, j.job_id) for p in PolicyName for j in self.jobs])
        self.assertEqual(len(seen), 10)
        for plan, worker, options in seen:
            self.assertIs(worker, self.worker)
            self.assertEqual(options, {"execution_timeout": 7})
            self.assertEqual([node.name for node in plan.workers], ["a", "a"])
        for record in records:
            value = record.as_dict()
            self.assertEqual(value["status"], "succeeded")
            self.assertEqual(value["backend"], "backend")
            self.assertIsNone(value["failure_stage"])
            self.assertEqual(value["inputs"]["workload"]["workers"], 2)
            self.assertEqual(value["inputs"]["accelerator_type"],
                             "A" if value["policy"] == "accelerator_type" else None)
        json.dumps([record.as_dict() for record in records], allow_nan=False)

    def test_default_and_named_ray_use_shared_adapter_for_all_policies(self):
        for backend in (None, "ray"):
            with self.subTest(backend=backend), patch(
                "topology_scheduler.ray_backend.run", return_value=[0, 1]
            ) as run:
                records = self.run_trace(backend=backend, reservation_timeout=3)
                self.assertEqual(run.call_count, 10)
                self.assertTrue(all(r.execution.backend == "ray" for r in records))
                for call in run.call_args_list:
                    self.assertIs(call.args[1], self.worker)
                    self.assertEqual(call.kwargs, {"reservation_timeout": 3})

    def test_boundaries_include_planning_reservation_execution_and_cleanup(self):
        ticks = iter(range(0, 10000, 10))
        with patch("topology_scheduler.comparison.perf_counter_ns", side_effect=lambda: next(ticks)):
            records = self.run_trace(backend=lambda *args: [])
        for record in records:
            timing = record.as_dict()["timing"]
            self.assertEqual([
                timing[name] - record.submitted_ns for name in (
                    "submitted_ns", "planning_started_ns", "planning_finished_ns",
                    "execution_started_ns", "execution_finished_ns", "terminal_ns",
                )
            ], [0, 10, 20, 30, 40, 50])
            self.assertEqual(record.jct_ns, 50)
            self.assertEqual(record.baseline_jct_ns, 50)
            self.assertEqual(record.normalized_jct, 1.0)

    def test_ratios_use_matched_job_observed_duration_not_estimates(self):
        ticks = []
        durations = {}
        start = 0
        for index, policy in enumerate(PolicyName):
            for job_index, job in enumerate(self.jobs):
                duration = 100 * (index + 1) * (job_index + 1)
                ticks.extend([start, start + 1, start + 2, start + 3, start + 4, start + duration])
                durations[(policy.value, job.job_id)] = duration
                start += duration + 10
        with patch("topology_scheduler.comparison.perf_counter_ns", side_effect=ticks):
            records = self.run_trace(backend=lambda *args: [])
        for record in records:
            baseline = durations[("gpu_count", record.job_id)]
            self.assertEqual(record.baseline_jct_ns, baseline)
            self.assertEqual(record.normalized_jct,
                             durations[(record.planning.policy, record.job_id)] / baseline)

    def test_execution_failure_retained_and_later_jobs_still_run(self):
        def fail(rank):
            raise TimeoutError("worker deadline")

        def backend(plan, worker):
            return [worker(rank) for rank in range(len(plan.workers))]

        records = self.run_trace(
            [TraceJob("fails", self.workload, fail), self.jobs[0]], backend=backend,
        )
        self.assertEqual(len(records), 10)
        for failed, success in zip(records[::2], records[1::2]):
            value = failed.as_dict()
            self.assertEqual(value["failure_stage"], "execution")
            self.assertIn("TimeoutError: worker deadline", value["error"])
            self.assertIsNone(value["jct_ns"])
            self.assertIsNone(failed.normalized_jct)
            self.assertEqual(failed.normalization_reason, "job_failed")
            self.assertEqual(success.status, "succeeded")

    def test_planning_failure_retains_inputs_and_does_not_call_backend(self):
        impossible = Workload(3, 40, {"A": 1})
        backend = Mock(return_value=[])
        records = self.run_trace(
            [TraceJob("too-large", impossible, self.worker), self.jobs[0]], backend=backend,
        )
        self.assertEqual(backend.call_count, 5)
        for failed, success in zip(records[::2], records[1::2]):
            value = failed.as_dict()
            self.assertEqual(value["failure_stage"], "planning")
            self.assertEqual(value["inputs"]["workload"]["workers"], 3)
            self.assertEqual(value["chosen_placement"], [])
            self.assertIsNone(value["estimated_seconds"])
            self.assertNotIn("execution_started_ns", value["timing"])
            self.assertIn("capacity", value["error"])
            self.assertEqual(success.status, "succeeded")

    def test_failed_baseline_does_not_normalize_successful_other_policies(self):
        def backend(plan, worker):
            if plan.policy_name == "gpu_count":
                raise RuntimeError("reservation failed")
            return []

        records = self.run_trace(backend=backend)
        for record in records:
            self.assertIsNone(record.normalized_jct)
            self.assertIsNone(record.baseline_jct_ns)
            self.assertEqual(record.normalization_reason,
                             "job_failed" if record.planning.policy == "gpu_count" else "baseline_failed")

    def test_zero_duration_baseline_does_not_divide_by_zero(self):
        with patch("topology_scheduler.comparison.perf_counter_ns", return_value=1):
            records = self.run_trace(backend=lambda *args: [])
        for record in records:
            self.assertIsNone(record.normalized_jct)
            self.assertEqual(record.normalization_reason, "nonpositive_baseline_jct")

    def test_invalid_trace_rejected_before_execution(self):
        for jobs in ([], [self.jobs[0]] * 2,
                     [TraceJob("", self.workload, self.worker)],
                     [TraceJob("bad-worker", self.workload, None)]):
            with self.subTest(jobs=jobs):
                backend = Mock()
                with self.assertRaises(ValueError):
                    self.run_trace(jobs, backend=backend)
                backend.assert_not_called()
        for options in ({"accelerator_type": ""}, {"accelerator_type": "A", "backend": "kai"}):
            with self.assertRaises(ValueError):
                run_matched_trace(self.nodes, self.jobs, {}, **options)

    def test_interrupts_propagate(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_trace(backend=Mock(side_effect=KeyboardInterrupt))


if __name__ == "__main__":
    unittest.main()
