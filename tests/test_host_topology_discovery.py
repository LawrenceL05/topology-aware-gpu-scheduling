"""Ray orchestration tests; NVML and physical GPU execution are not exercised."""
import importlib.util
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from topology_scheduler.host_topology import HostTopology, discover_host_topology


@unittest.skipUnless(importlib.util.find_spec("ray"), "Install .[ray] for discovery tests")
class HostDiscoveryTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.initialized = stack.enter_context(patch("ray.is_initialized", return_value=True))
        self.nodes = stack.enter_context(patch("ray.nodes", return_value=[
            {"Alive": True, "NodeID": "b" * 56, "Resources": {"GPU": 1, "topology_node:b": 1}},
            {"Alive": True, "NodeID": "a" * 56, "Resources": {"GPU": 1, "topology_node:a": 1}},
            {"Alive": False, "NodeID": "dead", "Resources": {"GPU": 1}},
            {"Alive": True, "NodeID": "cpu", "Resources": {"CPU": 1}},
        ]))
        self.remote = stack.enter_context(patch("ray.remote"))
        self.probe = self.remote.return_value.return_value
        self.probe.options.return_value.remote.side_effect = ["ref-a", "ref-b"]
        self.get = stack.enter_context(patch("ray.get", return_value=[
            {"node_id": name * 56, "topology": HostTopology("local", name * 56, (), (), ("unknown PCI",))}
            for name in ("a", "b")
        ]))
        self.cancel = stack.enter_context(patch("ray.cancel"))

    def test_one_pinned_probe_per_live_gpu_node_with_stable_identity(self):
        result = discover_host_topology(timeout=7)
        self.assertEqual([(r.node_id, r.node_name) for r in result],
                         [("a" * 56, "a"), ("b" * 56, "b")])
        self.assertEqual(result[0].diagnostics, ("unknown PCI",))
        self.remote.assert_called_once_with(num_cpus=0, max_retries=0)
        self.assertEqual(self.probe.options.call_count, 2)
        for call, node_id in zip(self.probe.options.call_args_list, ("a", "b")):
            strategy = call.kwargs["scheduling_strategy"]
            self.assertEqual(strategy.node_id, node_id * 56)
            self.assertFalse(strategy.soft)
        self.get.assert_called_once_with(["ref-a", "ref-b"], timeout=7)
        self.cancel.assert_not_called()

    def test_invalid_node_markers_fail_before_any_submission(self):
        for resources in ({"GPU": 1}, {"GPU": 1, "topology_node:b": 1},
                          {"GPU": 1, "topology_node:x": 1, "topology_node:y": 1}):
            with self.subTest(resources=resources):
                self.nodes.return_value[1]["Resources"] = resources
                with self.assertRaises(ValueError):
                    discover_host_topology()
                self.remote.assert_not_called()

    def test_timeout_and_probe_failure_cancel_submitted_tasks(self):
        for error in (TimeoutError("deadline"), RuntimeError("probe failed")):
            with self.subTest(error=error):
                self.probe.options.return_value.remote.side_effect = ["ref-a", "ref-b"]
                self.get.side_effect = error
                self.cancel.reset_mock()
                with self.assertRaises(type(error)):
                    discover_host_topology()
                self.assertEqual(self.cancel.call_count, 2)
                self.cancel.assert_any_call("ref-a", force=True)
                self.cancel.assert_any_call("ref-b", force=True)

    def test_partial_submission_failure_cancels_previous_probe(self):
        self.probe.options.return_value.remote.side_effect = ["ref-a", RuntimeError("submit")]
        with self.assertRaisesRegex(RuntimeError, "submit"):
            discover_host_topology()
        self.cancel.assert_called_once_with("ref-a", force=True)
        self.get.assert_not_called()

    def test_cleanup_failure_preserves_original_error(self):
        self.get.side_effect = TimeoutError("original deadline")
        self.cancel.side_effect = RuntimeError("disconnected")
        with self.assertRaisesRegex(TimeoutError, "original deadline"):
            discover_host_topology()
        self.assertEqual(self.cancel.call_count, 2)

    def test_wrong_node_identity_is_rejected(self):
        self.get.return_value[0]["node_id"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "wrong Ray node"):
            discover_host_topology()

    def test_invalid_timeout_fails_before_submission(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                discover_host_topology(timeout=timeout)
        self.remote.assert_not_called()

    def test_no_gpu_nodes_or_uninitialized_ray(self):
        self.nodes.return_value = []
        with self.assertRaisesRegex(ValueError, "No live Ray"):
            discover_host_topology()
        self.initialized.return_value = False
        with self.assertRaisesRegex(RuntimeError, "ray.init"):
            discover_host_topology()
        self.remote.assert_not_called()
