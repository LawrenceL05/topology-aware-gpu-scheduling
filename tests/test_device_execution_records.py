"""Identity evidence must survive the common execution and trace record path."""
import json
import unittest
from unittest.mock import Mock, patch

from topology_scheduler import (
    DeviceAssignment, DeviceBindingError, DevicePlacement, LinkCost,
    LinkCostSource, Node, PolicyName, RecordedExecutionError, TraceJob, Workload,
    plan_with_record, run_matched_trace, run_with_record,
)


class DeviceExecutionRecordTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [Node("a", "X", 1, 80), Node("b", "Y", 1, 80)]
        self.workload = Workload(1, 1, {"X": 4, "Y": 1})
        self.plan, self.planning = plan_with_record(
            self.nodes, self.workload, {}, policy=PolicyName.GPU_COUNT)
        self.placement = DevicePlacement(("GPU-a",))
        self.assignment = DeviceAssignment(0, "a", "GPU-a", "GPU-a",
                                           identity_source="injected")

    def execute(self, **options):
        return run_with_record(self.plan, self.planning, lambda rank: rank,
                               device_placement=self.placement, **options)

    def test_success_keeps_results_and_serializes_identity(self):
        with patch("topology_scheduler.device_binding.run_with_devices",
                   return_value=([42], (self.assignment,))) as run:
            result, record = self.execute(execution_timeout=7)
        self.assertEqual(result, [42])
        value = json.loads(json.dumps(record.as_dict()))["device_verification"]
        self.assertEqual(value, {"mode": "verify", "requested_uuids": ["GPU-a"],
                                 "assignments": [self.assignment.as_dict()]})
        self.assertEqual(run.call_args.args[:2], (self.plan, self.placement))
        self.assertEqual(run.call_args.kwargs, {"mode": "verify", "execution_timeout": 7})

    def test_refusal_and_workload_failure_retain_available_assignments(self):
        for message in ("device mismatch", "worker failed"):
            error = DeviceBindingError(message, [self.assignment])
            with self.subTest(message=message), patch(
                "topology_scheduler.device_binding.run_with_devices", side_effect=error
            ), self.assertRaises(RecordedExecutionError) as caught:
                self.execute()
            record = caught.exception.record
            self.assertEqual(record.status, "failed")
            self.assertEqual(record.as_dict()["device_verification"]["assignments"],
                             [self.assignment.as_dict()])
            self.assertIs(caught.exception.__cause__, error)

    def test_preflight_failure_keeps_request_without_inventing_observations(self):
        with patch("topology_scheduler.device_binding.run_with_devices",
                   side_effect=ValueError("missing UUID resource")), self.assertRaises(
                       RecordedExecutionError) as caught:
            self.execute()
        value = caught.exception.record.as_dict()["device_verification"]
        self.assertEqual(value["requested_uuids"], ["GPU-a"])
        self.assertEqual(value["assignments"], [])

    def test_observe_mode_is_explicit_and_default_execution_remains_unchanged(self):
        with patch("topology_scheduler.device_binding.run_with_devices",
                   return_value=([42], (self.assignment,))) as run:
            _, record = self.execute(device_mode="observe")
        self.assertEqual(record.as_dict()["device_verification"]["mode"], "observe")
        self.assertEqual(run.call_args.kwargs["mode"], "observe")
        with patch("topology_scheduler.ray_backend.run", return_value=[7]) as run:
            results, record = run_with_record(self.plan, self.planning, lambda rank: rank)
        self.assertEqual(results, [7])
        self.assertNotIn("device_verification", record.as_dict())
        self.assertEqual(run.call_args.kwargs, {})

    def test_incompatible_backend_or_invalid_mode_cannot_run_work(self):
        custom = Mock()
        for options in ({"backend": "kai"}, {"backend": custom}, {"device_mode": "pin"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.execute(**options)
        custom.assert_not_called()

    def test_trace_resolves_each_plan_and_retains_link_and_device_evidence(self):
        def select(plan):
            return DevicePlacement(tuple("GPU-" + node.name for node in plan.workers))

        def execute(plan, placement, worker, **options):
            self.assertEqual(options, {"mode": "verify"})
            self.assertEqual(placement, select(plan))
            assignments = tuple(DeviceAssignment(rank, node.name, uuid, uuid)
                                for rank, (node, uuid) in enumerate(zip(plan.workers, placement.uuids)))
            return [worker(rank) for rank in range(len(plan.workers))], assignments

        with patch("topology_scheduler.device_binding.run_with_devices", side_effect=execute):
            records = run_matched_trace(
                self.nodes, [TraceJob("job", self.workload, lambda rank: rank)],
                {("a", "b"): 2}, accelerator_type="X", device_placement=select,
                link_costs={("a", "b"): LinkCost(2, LinkCostSource.MEASURED, 1000)})
        self.assertEqual(len(records), len(PolicyName))
        self.assertGreater(len({r.planning.chosen_placement for r in records}), 1)
        for record in records:
            value = json.loads(json.dumps(record.as_dict()))
            self.assertEqual(value["status"], "succeeded")
            self.assertEqual(value["inputs"]["bandwidth_sources"]["a|b"]["source"], "measured")
            self.assertEqual(value["device_verification"]["requested_uuids"],
                             ["GPU-" + node for node in record.planning.chosen_placement])

    def test_trace_keeps_refusals_and_continues_with_failed_baseline(self):
        error = DeviceBindingError("refused", [self.assignment])
        with patch("topology_scheduler.device_binding.run_with_devices",
                   side_effect=[error] + [([0], (self.assignment,))] * 4):
            records = run_matched_trace(
                self.nodes, [TraceJob("job", self.workload, lambda rank: rank)], {},
                accelerator_type="X", device_placement=self.placement)
        self.assertEqual(records[0].as_dict()["device_verification"]["assignments"],
                         [self.assignment.as_dict()])
        self.assertEqual([r.status for r in records], ["failed"] + ["succeeded"] * 4)
        self.assertTrue(all(r.normalized_jct is None for r in records))

    def test_selector_failure_becomes_terminal_record_without_execution(self):
        for selector in (lambda plan: None, Mock(side_effect=ValueError("no mapped UUID"))):
            with self.subTest(selector=selector), patch(
                "topology_scheduler.device_binding.run_with_devices"
            ) as run, self.assertRaises(RecordedExecutionError) as caught:
                run_with_record(self.plan, self.planning, lambda rank: rank,
                                device_placement=selector)
            run.assert_not_called()
            value = caught.exception.record.as_dict()["device_verification"]
            self.assertIsNone(value["requested_uuids"])
            self.assertEqual(value["assignments"], [])
