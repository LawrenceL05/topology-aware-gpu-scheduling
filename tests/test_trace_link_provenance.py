"""The measured-cost API and matched trace runner must preserve one snapshot."""

import json
import unittest
from unittest.mock import Mock, patch

from topology_scheduler import (
    LinkCost, LinkCostSource, Node, PolicyName, TraceJob, Workload,
    run_matched_trace,
)


class TraceLinkProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [Node(name, "X", 1, 80) for name in ("a", "b", "c")]
        self.workload = Workload(2, 1, {"X": 1}, 1)
        self.costs = {
            ("a", "b"): LinkCost(2, LinkCostSource.MEASURED, 1000),
            ("a", "c"): LinkCost(3, LinkCostSource.ADVERTISED),
            ("b", "c"): LinkCost(1, LinkCostSource.FALLBACK),
        }
        self.bandwidth = {pair: cost.gb_per_second for pair, cost in self.costs.items()}
        self.sources = {"|".join(pair): cost.provenance() for pair, cost in self.costs.items()}
        self.jobs = [TraceJob("job", self.workload, lambda rank: rank)]

    def run_trace(self, **options):
        return run_matched_trace(self.nodes, self.jobs, self.bandwidth,
                                 accelerator_type="X", **options)

    def assert_sources(self, records):
        values = json.loads(json.dumps([r.as_dict() for r in records], allow_nan=False))
        self.assertEqual(len(values), len(self.jobs) * len(PolicyName))
        for value in values:
            self.assertEqual(value["inputs"]["bandwidth_sources"], self.sources)
            self.assertEqual(value["inputs"]["bandwidth_gbps"],
                             {"a|b": 2, "a|c": 3, "b|c": 1})
        return values

    def test_mixed_sources_reach_every_policy_without_leaking_to_backend(self):
        # A strict signature catches accidental forwarding via **backend_options.
        def execute(plan, worker, *, execution_timeout):
            self.assertEqual(execution_timeout, 7)
            return [worker(rank) for rank in range(len(plan.workers))]

        backend = Mock(side_effect=execute)
        records = self.run_trace(link_costs=self.costs, backend=backend, execution_timeout=7)
        self.assertTrue(all(r.status == "succeeded" for r in records))
        self.assertEqual(backend.call_count, 5)
        self.assert_sources(records)

    def test_planning_and_execution_failures_keep_sources_and_continue(self):
        self.jobs = [TraceJob("capacity", Workload(4, 1, {"X": 1}, 1), lambda rank: rank),
                     TraceJob("execution", self.workload, lambda rank: rank),
                     TraceJob("success", self.workload, lambda rank: rank)]
        backend = Mock(side_effect=[TimeoutError("deadline"), []] * len(PolicyName))
        values = self.assert_sources(self.run_trace(link_costs=self.costs, backend=backend))
        self.assertEqual(backend.call_count, 10)
        self.assertEqual([v["failure_stage"] for v in values],
                         ["planning", "execution", None] * len(PolicyName))
        self.assertEqual([v["status"] for v in values],
                         ["failed", "failed", "succeeded"] * len(PolicyName))

    def test_default_and_named_ray_do_not_receive_link_costs(self):
        for backend in (None, "ray"):
            with self.subTest(backend=backend), patch(
                "topology_scheduler.ray_backend.run", return_value=[]
            ) as run:
                records = self.run_trace(link_costs=self.costs, backend=backend,
                                         reservation_timeout=3)
                self.assert_sources(records)
                self.assertEqual(run.call_count, 5)
                self.assertTrue(all(r.status == "succeeded" for r in records))
                for call in run.call_args_list:
                    self.assertEqual(call.kwargs, {"reservation_timeout": 3})

    def test_missing_extra_and_different_costs_rejected_before_any_planning(self):
        invalid = ({}, {("a", "b"): self.costs[("a", "b")]},
                   {**self.costs, ("c", "d"): LinkCost(1, LinkCostSource.FALLBACK)},
                   {**self.costs, ("a", "b"): LinkCost(9, LinkCostSource.MEASURED, 1000)})
        for costs in invalid:
            with self.subTest(costs=costs), patch(
                "topology_scheduler.comparison.choose_placement"
            ) as choose:
                backend = Mock()
                with self.assertRaisesRegex(ValueError, "exactly the supplied"):
                    self.run_trace(link_costs=costs, backend=backend)
                choose.assert_not_called()
                backend.assert_not_called()

    def test_supplied_values_keep_their_label_on_success_and_planning_failure(self):
        self.jobs.append(TraceJob("capacity", Workload(4, 1, {"X": 1}, 1), lambda rank: rank))
        self.sources = {pair: {"source": "supplied", "measured_at": None}
                        for pair in self.sources}
        values = self.assert_sources(self.run_trace(backend=lambda *args: []))
        self.assertEqual([v["failure_stage"] for v in values],
                         [None, "planning"] * len(PolicyName))

    def test_caller_mutation_cannot_change_later_policy_inputs_or_old_records(self):
        def backend(plan, worker):
            self.bandwidth.clear()
            self.costs.clear()
            return []

        records = self.run_trace(link_costs=self.costs, backend=backend)
        self.assertTrue(all(r.status == "succeeded" for r in records))
        self.assert_sources(records)
        records[0].planning.inputs["bandwidth_sources"]["a|b"]["source"] = "changed"
        self.assertEqual(records[1].planning.inputs["bandwidth_sources"], self.sources)


if __name__ == "__main__":
    unittest.main()
