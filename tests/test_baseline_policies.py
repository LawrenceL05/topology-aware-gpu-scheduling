import unittest
from unittest.mock import patch

from topology_scheduler import (
    Node, PolicyName, RecordedExecutionError, Workload, choose_placement,
    plan_with_record, run_with_record,
)


class BaselinePolicyTests(unittest.TestCase):
    def setUp(self):
        specs = [
            ("a", "A"), ("b", "A"),
            ("c", "X"), ("d", "X"),
            ("e", "FAST"), ("f", "FAST"),
            ("g", "TOPO"), ("h", "TOPO"),
            ("i", "MID"), ("j", "MID"),
        ]
        self.nodes = [Node(name, model, 1, 80) for name, model in specs]
        self.workload = Workload(
            2, 40, {"A": 10, "X": 8, "FAST": 1, "TOPO": 20, "MID": 2}, 10
        )
        self.bandwidth = {
            tuple(sorted((left.name, right.name))): 1
            for index, left in enumerate(self.nodes)
            for right in self.nodes[index + 1:]
        }
        self.bandwidth[("e", "f")] = 0.1
        self.bandwidth[("g", "h")] = 100
        self.bandwidth[("i", "j")] = 50

    def test_policies_intentionally_choose_different_placements(self):
        cases = [
            (PolicyName.GPU_COUNT, {}, ("a", "b")),
            (PolicyName.ACCELERATOR_TYPE, {"accelerator_type": "X"}, ("c", "d")),
            (PolicyName.WORKLOAD_COMPUTE, {}, ("e", "f")),
            (PolicyName.TOPOLOGY_ONLY, {}, ("g", "h")),
            (PolicyName.COMBINED, {}, ("i", "j")),
        ]
        for policy, options, expected in cases:
            with self.subTest(policy=policy):
                plan = choose_placement(
                    self.nodes, self.workload, self.bandwidth,
                    policy=policy, **options,
                )
                self.assertEqual(tuple(node.name for node in plan.workers), expected)
                self.assertEqual(plan.policy_name, policy.value)

    def test_unscored_baselines_report_no_estimate(self):
        for policy, options in [
            (PolicyName.GPU_COUNT, {}),
            (PolicyName.ACCELERATOR_TYPE, {"accelerator_type": "X"}),
        ]:
            plan = choose_placement(
                self.nodes, self.workload, self.bandwidth,
                policy=policy, **options,
            )
            self.assertIsNone(plan.estimated_seconds)

    def test_all_policies_share_memory_and_capacity_checks(self):
        nodes = [Node("small", "A", 2, 20), Node("one", "A", 1, 80)]
        workload = Workload(2, 40, {"A": 1})
        for policy in PolicyName:
            options = {"accelerator_type": "A"} if policy is PolicyName.ACCELERATOR_TYPE else {}
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "capacity"):
                choose_placement(nodes, workload, {}, policy=policy, **options)

    def test_lexical_tie_break_is_independent_of_input_order(self):
        workload = Workload(2, 1, {"A": 1})
        nodes = [Node("b", "A", 1, 80), Node("a", "A", 1, 80)]
        for policy in PolicyName:
            options = {"accelerator_type": "A"} if policy is PolicyName.ACCELERATOR_TYPE else {}
            plan = choose_placement(nodes, workload, {}, policy=policy, **options)
            self.assertEqual(tuple(node.name for node in plan.workers), ("a", "b"))

    def test_planning_record_is_machine_readable(self):
        plan, record = plan_with_record(
            self.nodes, self.workload, self.bandwidth,
            policy=PolicyName.ACCELERATOR_TYPE, accelerator_type="X",
        )
        value = record.as_dict()
        self.assertEqual(value["policy"], "accelerator_type")
        self.assertEqual(value["chosen_placement"], ["c", "d"])
        self.assertIsNone(value["estimated_seconds"])
        self.assertEqual(value["inputs"]["accelerator_type"], "X")
        self.assertLessEqual(
            value["timing"]["planning_started_ns"],
            value["timing"]["planning_finished_ns"],
        )
        self.assertEqual(plan.as_dict()["policy"], "accelerator_type")

    def test_accelerator_policy_requires_constraint(self):
        with self.assertRaisesRegex(ValueError, "requires accelerator_type"):
            choose_placement(
                self.nodes, self.workload, self.bandwidth,
                policy=PolicyName.ACCELERATOR_TYPE,
            )

    def test_every_policy_uses_same_backend_contract(self):
        seen = []

        def backend(plan, worker, **options):
            seen.append((plan.policy_name, options))
            return [worker(rank) for rank in range(len(plan.workers))]

        for policy in PolicyName:
            options = {"accelerator_type": "X"} if policy is PolicyName.ACCELERATOR_TYPE else {}
            plan, planning = plan_with_record(
                self.nodes, self.workload, self.bandwidth,
                policy=policy, **options,
            )
            results, execution = run_with_record(
                plan, planning, lambda rank: rank,
                backend=backend, execution_timeout=12,
            )
            self.assertEqual(results, [0, 1])
            self.assertEqual(execution.status, "succeeded")
            self.assertEqual(execution.backend, "backend")
            self.assertIn("execution_started_ns", execution.as_dict()["timing"])
        self.assertEqual([name for name, _ in seen], [item.value for item in PolicyName])

    def test_backend_failure_is_retained_in_machine_readable_record(self):
        plan, planning = plan_with_record(
            self.nodes, self.workload, self.bandwidth,
            policy=PolicyName.GPU_COUNT,
        )

        def fail(*args, **kwargs):
            raise TimeoutError("reservation timed out")

        with self.assertRaises(RecordedExecutionError) as caught:
            run_with_record(plan, planning, lambda rank: rank, backend=fail)
        value = caught.exception.record.as_dict()
        self.assertEqual(value["status"], "failed")
        self.assertEqual(value["error"], "TimeoutError: reservation timed out")

    def test_named_backend_validation_is_actionable(self):
        plan, planning = plan_with_record(
            self.nodes, self.workload, self.bandwidth,
            policy=PolicyName.GPU_COUNT,
        )
        with self.assertRaisesRegex(ValueError, "'ray', 'kai'"):
            run_with_record(plan, planning, None, backend="unknown")

    def test_every_policy_can_use_named_kai_backend(self):
        kai_workload = object()
        with patch(
            "topology_scheduler.kai_backend.run", return_value="complete"
        ) as selected:
            for policy in PolicyName:
                options = (
                    {"accelerator_type": "X"}
                    if policy is PolicyName.ACCELERATOR_TYPE else {}
                )
                plan, planning = plan_with_record(
                    self.nodes, self.workload, self.bandwidth,
                    policy=policy, **options,
                )
                result, execution = run_with_record(
                    plan, planning, kai_workload,
                    backend="kai", client="client",
                )
                self.assertEqual(result, "complete")
                self.assertEqual(execution.backend, "kai")
        self.assertEqual(selected.call_count, len(PolicyName))


if __name__ == "__main__":
    unittest.main()
