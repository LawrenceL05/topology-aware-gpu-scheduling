import importlib.util
import unittest
from unittest.mock import MagicMock, patch
from topology_scheduler import Node, Plan, PolicyName, Workload, choose_placement
from topology_scheduler.ray_backend import run


@unittest.skipUnless(importlib.util.find_spec("ray"), "Install .[ray] for backend tests")
class BackendTests(unittest.TestCase):
    def setUp(self):
        self.plan = Plan((Node("a", "A", 1, 80),), 1)
        self.live = [{"Alive": True, "Resources": {"CPU": 1, "GPU": 1, "topology_node:a": 1}}]

    def test_all_policies_share_reservation_execution_and_cleanup(self):
        for policy in PolicyName:
            plan = choose_placement(
                [Node("a", "A", 1, 80)], Workload(1, 40, {"A": 1}), {},
                policy=policy, accelerator_type="A",
            )
            for outcome in ([0], TimeoutError("deadline")):
                with self.subTest(policy=policy, outcome=outcome), \
                     patch("ray.is_initialized", return_value=True), \
                     patch("ray.nodes", return_value=self.live), \
                     patch("ray.util.placement_group.placement_group") as create, \
                     patch("ray.util.placement_group.remove_placement_group") as remove, \
                     patch("ray.remote") as remote, \
                     patch("ray.cancel") as cancel, \
                     patch("ray.get", side_effect=[None, outcome]) as get:
                    worker = lambda rank: rank
                    if isinstance(outcome, Exception):
                        with self.assertRaises(TimeoutError):
                            run(plan, worker, reservation_timeout=7, execution_timeout=9)
                    else:
                        self.assertEqual(run(plan, worker, reservation_timeout=7,
                                             execution_timeout=9), [0])
                    create.assert_called_once_with(
                        [{"CPU": 1, "GPU": 1, "topology_node:a": 1}], strategy="PACK",
                    )
                    remote.assert_called_once_with(worker)
                    options = remote.return_value.options.call_args.kwargs
                    self.assertEqual(options["num_cpus"], 1)
                    self.assertEqual(options["num_gpus"], 1)
                    self.assertEqual(options["resources"], {"topology_node:a": 1})
                    self.assertEqual(options["max_retries"], 0)
                    strategy = options["scheduling_strategy"]
                    self.assertIs(strategy.placement_group, create.return_value)
                    self.assertEqual(strategy.placement_group_bundle_index, 0)
                    self.assertFalse(strategy.placement_group_capture_child_tasks)
                    self.assertEqual([c.kwargs["timeout"] for c in get.call_args_list], [7, 9])
                    cancel.assert_called_once_with(
                        remote.return_value.options.return_value.remote.return_value, force=True,
                    )
                    remove.assert_called_once_with(create.return_value)

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
