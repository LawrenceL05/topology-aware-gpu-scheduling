import importlib.util
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from topology_scheduler.nic_inventory import NodeNICInventory, discover_nic_inventory


@unittest.skipUnless(importlib.util.find_spec("ray"), "Install .[ray] for discovery tests")
class NICDiscoveryTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.initialized = stack.enter_context(patch("ray.is_initialized", return_value=True))
        self.nodes = stack.enter_context(patch("ray.nodes", return_value=[
            {"Alive": True, "NodeID": "b" * 56, "Resources": {"topology_node:b": 1}},
            {"Alive": True, "NodeID": "a" * 56, "Resources": {"topology_node:a": 1}},
            {"Alive": False, "NodeID": "dead", "Resources": {"topology_node:c": 1}},
            {"Alive": True, "NodeID": "head", "Resources": {"CPU": 1}},
        ]))
        self.remote = stack.enter_context(patch("ray.remote"))
        self.probe = self.remote.return_value.return_value
        self.probe.options.return_value.remote.side_effect = ["ref-a", "ref-b"]
        self.get = stack.enter_context(patch("ray.get", return_value=[
            {"node_id": name * 56, "inventory": NodeNICInventory(
                name * 56, "local", (), ("sysfs unavailable",))}
            for name in ("a", "b")
        ]))
        self.cancel = stack.enter_context(patch("ray.cancel"))

    def test_each_live_marked_node_probed_once_without_gpu_requirement(self):
        inventories = discover_nic_inventory(timeout=7)
        self.assertEqual([(r.node_name, r.node_id) for r in inventories],
                         [("a", "a" * 56), ("b", "b" * 56)])
        self.assertEqual(inventories[0].problems, ("sysfs unavailable",))
        self.remote.assert_called_once_with(num_cpus=0, max_retries=0)
        self.assertEqual(self.probe.options.call_count, 2)
        for call, node_id in zip(self.probe.options.call_args_list, ("a" * 56, "b" * 56)):
            strategy = call.kwargs["scheduling_strategy"]
            self.assertEqual(strategy.node_id, node_id)
            self.assertFalse(strategy.soft)
        self.get.assert_called_once_with(["ref-a", "ref-b"], timeout=7)
        self.cancel.assert_not_called()

    def test_invalid_markers_rejected_before_any_probe_is_submitted(self):
        cases = [
            [self.nodes.return_value[0], {"Alive": True, "NodeID": "bad", "Resources": {
                "topology_node:x": 1, "topology_node:y": 1}}],
            [self.nodes.return_value[0], {"Alive": True, "NodeID": "other", "Resources": {
                "topology_node:b": 1}}],
            [],
        ]
        for nodes in cases:
            with self.subTest(nodes=nodes):
                self.nodes.return_value = nodes
                with self.assertRaises(ValueError):
                    discover_nic_inventory()
                self.remote.assert_not_called()

    def test_wrong_node_identity_is_rejected(self):
        self.get.return_value[0]["node_id"] = "unexpected"
        with self.assertRaisesRegex(RuntimeError, "wrong Ray node"):
            discover_nic_inventory()

    def test_timeout_or_probe_failure_cancels_submitted_tasks(self):
        for error in (TimeoutError("deadline"), RuntimeError("node disappeared")):
            with self.subTest(error=error):
                self.probe.options.return_value.remote.side_effect = ["ref-a", "ref-b"]
                self.cancel.reset_mock()
                self.get.side_effect = error
                with self.assertRaises(type(error)):
                    discover_nic_inventory()
                self.assertEqual(self.cancel.call_count, 2)
                self.cancel.assert_any_call("ref-a", force=True)
                self.cancel.assert_any_call("ref-b", force=True)

    def test_partial_submission_failure_cancels_earlier_task(self):
        self.probe.options.return_value.remote.side_effect = ["ref-a", RuntimeError("submit failed")]
        with self.assertRaisesRegex(RuntimeError, "submit failed"):
            discover_nic_inventory()
        self.cancel.assert_called_once_with("ref-a", force=True)
        self.get.assert_not_called()

    def test_cleanup_error_does_not_hide_original_failure(self):
        self.get.side_effect = TimeoutError("original deadline")
        self.cancel.side_effect = RuntimeError("Ray disconnected")
        with self.assertRaisesRegex(TimeoutError, "original deadline"):
            discover_nic_inventory()
        self.assertEqual(self.cancel.call_count, 2)

    def test_invalid_timeout_fails_before_submission(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                discover_nic_inventory(timeout=timeout)
        self.remote.assert_not_called()

    def test_ray_must_be_initialized(self):
        self.initialized.return_value = False
        with self.assertRaisesRegex(RuntimeError, "ray.init"):
            discover_nic_inventory()
        self.remote.assert_not_called()


if __name__ == "__main__":
    unittest.main()
