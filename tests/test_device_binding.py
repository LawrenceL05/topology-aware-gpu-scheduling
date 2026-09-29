import json
import unittest
from unittest.mock import patch

from topology_scheduler import GPUDevice, Node, Plan
from topology_scheduler.device_binding import (
    DEVICE_ORDER_ENV_VAR, OBSERVE, PCI_BUS_ID, VERIFY, DeviceAssignment,
    DeviceBindingError, DevicePlacement, assignment_for, bind_worker,
    bundle_resources, check_assignments, device_resource_key, device_resources,
    preflight_devices, resolve_device as resolve_observed_device,
)

PLAN = Plan((Node("a", "H100", 2, 80), Node("a", "H100", 2, 80)), 1.0)
PLACEMENT = DevicePlacement(("GPU-aaa", "GPU-bbb"))
PCI_ORDER = {DEVICE_ORDER_ENV_VAR: PCI_BUS_ID}


def device(index, uuid):
    return GPUDevice(index, uuid, "NVIDIA H100 80GB HBM3", "H100", 80,
                     f"0000:{index:02x}:00.0")


DEVICES = (device(0, "GPU-aaa"), device(1, "GPU-bbb"))


def resolve_device(ids, devices, *, device_order):
    """Existing cases simulate a CUDA observation agreeing with the fixture."""
    cuda = {"available": True, "device_count": 1, "uuid": "GPU-aaa"}
    if len(ids) == 1 and str(ids[0]).isdecimal() and int(ids[0]) < len(devices):
        cuda["uuid"] = devices[int(ids[0])].uuid
    return resolve_observed_device(ids, devices, device_order=device_order, cuda_device=cuda)


def node(name, resources):
    return {"Alive": True, "NodeID": f"id-{name}", "Resources": resources}


class ResourceTests(unittest.TestCase):
    def test_resource_key_is_derived_from_the_uuid(self):
        self.assertEqual(device_resource_key("GPU-aaa"), "topology_gpu:GPU-aaa")
        self.assertEqual(device_resource_key("  GPU-aaa  "), "topology_gpu:GPU-aaa")
        for bad in ("", "   "):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                device_resource_key(bad)

    def test_node_resources_give_one_unit_per_device(self):
        self.assertEqual(device_resources(DEVICES),
                         {"topology_gpu:GPU-aaa": 1, "topology_gpu:GPU-bbb": 1})

    def test_duplicate_devices_on_one_node_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate device"):
            device_resources((device(0, "GPU-aaa"), device(1, "GPU-aaa")))

    def test_placement_rejects_empty_and_shared_devices(self):
        with self.assertRaisesRegex(ValueError, "at least one device"):
            DevicePlacement(())
        with self.assertRaisesRegex(ValueError, "cannot serve two ranks"):
            DevicePlacement(("GPU-aaa", "GPU-aaa"))
        with self.assertRaisesRegex(ValueError, "nonempty string"):
            DevicePlacement(("GPU-aaa", ""))

    def test_bundle_resources_match_ranks_one_to_one(self):
        self.assertEqual(bundle_resources(PLAN, PLACEMENT),
                         [{"topology_gpu:GPU-aaa": 1}, {"topology_gpu:GPU-bbb": 1}])
        with self.assertRaisesRegex(ValueError, "names 1 devices for 2"):
            bundle_resources(PLAN, DevicePlacement(("GPU-aaa",)))


class ResolutionTests(unittest.TestCase):
    def test_an_index_resolves_to_a_uuid_when_orders_agree(self):
        resolved = resolve_device([1], DEVICES, device_order=PCI_BUS_ID)
        self.assertEqual(resolved["assigned_uuid"], "GPU-bbb")
        self.assertEqual(resolved["assigned_index"], 1)
        self.assertEqual(resolved["pci_bus_id"], "0000:01:00.0")
        self.assertIsNone(resolved.get("problem"))

    def test_the_default_cuda_ordering_is_treated_as_a_problem(self):
        for order in (None, "FASTEST_FIRST"):
            with self.subTest(order=order):
                resolved = resolve_device([0], DEVICES, device_order=order)
                # The UUID is still reported, so the record stays useful.
                self.assertEqual(resolved["assigned_uuid"], "GPU-aaa")
                self.assertIn(PCI_BUS_ID, resolved["problem"])
                self.assertIn(DEVICE_ORDER_ENV_VAR, resolved["problem"])

    def test_more_or_fewer_than_one_device_is_refused(self):
        for ids in ([], [0, 1]):
            with self.subTest(ids=ids):
                self.assertIn("exactly one GPU",
                              resolve_device(ids, DEVICES,
                                             device_order=PCI_BUS_ID)["problem"])

    def test_an_index_outside_the_node_inventory_is_refused(self):
        resolved = resolve_device([7], DEVICES, device_order=PCI_BUS_ID)
        self.assertIn("raylet and NVML disagree", resolved["problem"])

    def test_a_non_index_accelerator_id_is_refused(self):
        self.assertIn("is not an index",
                      resolve_device(["cuda:0"], DEVICES,
                                     device_order=PCI_BUS_ID)["problem"])

    def test_string_indices_from_ray_are_accepted(self):
        self.assertEqual(resolve_device(["1"], DEVICES,
                                        device_order=PCI_BUS_ID)["assigned_uuid"],
                         "GPU-bbb")


class AssignmentTests(unittest.TestCase):
    def resolved(self, index=0, order=PCI_BUS_ID):
        return resolve_device([index], DEVICES, device_order=order)

    def test_a_matching_device_is_recorded_as_matched(self):
        item = assignment_for(0, "a", "GPU-aaa", self.resolved())
        self.assertTrue(item.matched)
        self.assertEqual(item.as_dict()["matched"], True)
        json.dumps(item.as_dict())

    def test_a_different_device_never_counts_as_matched(self):
        item = assignment_for(0, "a", "GPU-bbb", self.resolved(index=0))
        self.assertFalse(item.matched)
        self.assertEqual((item.requested_uuid, item.assigned_uuid),
                         ("GPU-bbb", "GPU-aaa"))

    def test_an_unsafe_ordering_never_counts_as_matched(self):
        item = assignment_for(0, "a", "GPU-aaa", self.resolved(order=None))
        self.assertEqual(item.assigned_uuid, "GPU-aaa")
        self.assertFalse(item.matched)

    def test_records_round_trip_through_a_dictionary(self):
        item = assignment_for(1, "a", "GPU-bbb", self.resolved(index=1))
        self.assertEqual(DeviceAssignment.from_dict(item.as_dict()), item)

    def test_observe_mode_reports_while_verify_mode_refuses(self):
        wrong = assignment_for(0, "a", "GPU-bbb", self.resolved(index=0))
        check_assignments([wrong], OBSERVE)
        with self.assertRaises(DeviceBindingError) as caught:
            check_assignments([wrong], VERIFY)
        message = str(caught.exception)
        self.assertIn("rank 0 on a", message)
        self.assertIn("asked for GPU-bbb", message)
        self.assertIn("received GPU-aaa", message)
        self.assertEqual(len(caught.exception.assignments), 1)
        json.dumps(caught.exception.as_dict())

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            check_assignments([], "pin")


class PreflightTests(unittest.TestCase):
    def test_accepts_a_node_advertising_both_marker_and_devices(self):
        preflight_devices(PLAN, PLACEMENT, [node("a", {
            "topology_node:a": 2, "topology_gpu:GPU-aaa": 1,
            "topology_gpu:GPU-bbb": 1})])

    def test_rejects_a_device_no_live_node_advertises(self):
        with self.assertRaisesRegex(ValueError, "topology_gpu:GPU-bbb"):
            preflight_devices(PLAN, PLACEMENT, [node("a", {
                "topology_node:a": 2, "topology_gpu:GPU-aaa": 1})])

    def test_rejects_a_device_advertised_by_the_wrong_node(self):
        with self.assertRaisesRegex(ValueError, "rank 1"):
            preflight_devices(PLAN, PLACEMENT, [
                node("a", {"topology_node:a": 2, "topology_gpu:GPU-aaa": 1}),
                node("b", {"topology_node:b": 1, "topology_gpu:GPU-bbb": 1})])

    def test_ignores_dead_nodes(self):
        dead = node("a", {"topology_node:a": 2, "topology_gpu:GPU-aaa": 1,
                          "topology_gpu:GPU-bbb": 1})
        dead["Alive"] = False
        with self.assertRaisesRegex(ValueError, "exactly one live node"):
            preflight_devices(PLAN, PLACEMENT, [dead])


class BoundWorkerTests(unittest.TestCase):
    def bound(self, *, mode=VERIFY, ids=(0,), order=PCI_BUS_ID, devices=DEVICES):
        environ = {DEVICE_ORDER_ENV_VAR: order} if order else {}
        worker = bind_worker(lambda rank: f"work-{rank}", PLACEMENT, ("a", "a"),
                             mode=mode, devices_provider=lambda: devices)
        with patch("topology_scheduler.device_binding.observe_worker_device",
                   side_effect=lambda **_: resolve_device(
                       ids, devices, device_order=environ.get(DEVICE_ORDER_ENV_VAR))):
            return worker(0)

    def test_a_matching_device_runs_the_workload(self):
        payload = self.bound()
        self.assertEqual(payload["result"], "work-0")
        self.assertTrue(payload["device"]["matched"])

    def test_a_wrong_device_fails_before_the_workload_runs(self):
        calls = []

        def worker(rank):
            calls.append(rank)
            return rank

        bound = bind_worker(worker, PLACEMENT, ("a", "a"), mode=VERIFY,
                            devices_provider=lambda: DEVICES)
        with patch("topology_scheduler.device_binding.observe_worker_device",
                   return_value=resolve_device([1], DEVICES, device_order=PCI_BUS_ID)):
            with self.assertRaises(DeviceBindingError):
                bound(0)
        self.assertEqual(calls, [])

    def test_observe_mode_records_the_mismatch_and_still_runs(self):
        payload = self.bound(mode=OBSERVE, ids=(1,))
        self.assertEqual(payload["result"], "work-0")
        self.assertFalse(payload["device"]["matched"])
        self.assertEqual(payload["device"]["assigned_uuid"], "GPU-bbb")

    def test_an_unsafe_ordering_stops_a_verified_run(self):
        with self.assertRaises(DeviceBindingError):
            self.bound(order=None)

    def test_binding_rejects_an_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            bind_worker(lambda rank: rank, PLACEMENT, ("a", "a"), mode="pin")


class SingleDeviceNodeTests(unittest.TestCase):
    """The degenerate case the issue calls out: one GPU per node."""

    def test_one_device_per_node_resolves_without_ambiguity(self):
        plan = Plan((Node("a", "H100", 1, 80), Node("b", "H100", 1, 80)), 1.0)
        placement = DevicePlacement(("GPU-aaa", "GPU-zzz"))
        preflight_devices(plan, placement, [
            node("a", {"topology_node:a": 1, "topology_gpu:GPU-aaa": 1}),
            node("b", {"topology_node:b": 1, "topology_gpu:GPU-zzz": 1})])
        resolved = resolve_device([0], (device(0, "GPU-zzz"),),
                                  device_order=PCI_BUS_ID)
        self.assertTrue(assignment_for(1, "b", "GPU-zzz", resolved).matched)


if __name__ == "__main__":
    unittest.main()
