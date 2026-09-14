import importlib.util
import unittest
from unittest.mock import MagicMock, patch
from topology_scheduler import Node, Plan
from topology_scheduler.ray_backend import run


@unittest.skipUnless(importlib.util.find_spec("ray"), "Install .[ray] for backend tests")
class BackendTests(unittest.TestCase):
    def setUp(self):
        self.plan = Plan((Node("a", "A", 1, 80),), 1)
        self.live = [{"Alive": True, "Resources": {"CPU": 1, "GPU": 1, "topology_node:a": 1}}]

    def test_timeout_removes_reservation(self):
        with patch("ray.is_initialized", return_value=True), \
             patch("ray.nodes", return_value=self.live), \
             patch("ray.util.placement_group.placement_group") as create, \
             patch("ray.util.placement_group.remove_placement_group") as remove, \
             patch("ray.get", side_effect=TimeoutError("busy")):
            with self.assertRaises(TimeoutError):
                run(self.plan, lambda rank: rank)
            remove.assert_called_once_with(create.return_value)

    def test_duplicate_node_marker_rejected_before_reservation(self):
        with patch("ray.is_initialized", return_value=True), \
             patch("ray.nodes", return_value=self.live * 2), \
             patch("ray.util.placement_group.placement_group") as create:
            with self.assertRaisesRegex(ValueError, "exactly one"):
                run(self.plan, lambda rank: rank)
            create.assert_not_called()

    def test_worker_failure_cancels_tasks_and_removes_group(self):
        ref = MagicMock()
        with patch("ray.is_initialized", return_value=True), \
             patch("ray.nodes", return_value=self.live), \
             patch("ray.util.placement_group.placement_group") as create, \
             patch("ray.util.placement_group.remove_placement_group") as remove, \
             patch("ray.remote") as remote, \
             patch("ray.cancel") as cancel, \
             patch("ray.get", side_effect=[None, RuntimeError("worker failed")]):
            remote.return_value.options.return_value.remote.return_value = ref
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                run(self.plan, lambda rank: rank)
            cancel.assert_called_once_with(ref, force=True)
            remove.assert_called_once_with(create.return_value)
