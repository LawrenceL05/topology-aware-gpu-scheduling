import ctypes
from dataclasses import replace
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from topology_scheduler import GPUDevice, Node, Plan
from topology_scheduler.cuda_identity import read_cuda_identity
from topology_scheduler.device_binding import (
    DeviceAssignment, DeviceBindingError, DevicePlacement, assignment_for,
    bind_worker, observe_worker_device, preflight_devices, resolve_device,
)

DEVICES = (
    GPUDevice(0, "GPU-a", "A", "A", 80, "0000:81:00.0"),
    GPUDevice(1, "GPU-b", "A", "A", 80, "0000:01:00.0"),
)
ORDER = "PCI_BUS_ID"


def cuda(uuid="GPU-a", count=1):
    return {"available": True, "device_count": count, "uuid": uuid}


class DriverQueryTests(unittest.TestCase):
    def driver(self, count=1, fail=None, version=2):
        def write_count(pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = count
            return 0

        def write_uuid(pointer, device):
            ctypes.memmove(pointer, bytes(range(16)), 16)
            return 0

        functions = {
            "cuInit": Mock(return_value=0),
            "cuDeviceGetCount": Mock(side_effect=write_count),
            "cuDeviceGet": Mock(return_value=0),
            "cuDeviceGetUuid_v2" if version == 2 else "cuDeviceGetUuid": Mock(side_effect=write_uuid),
        }
        if fail:
            functions[fail] = Mock(return_value=5)
        return SimpleNamespace(**functions)

    def read(self, driver):
        with patch("topology_scheduler.cuda_identity.sys.platform", "linux"), \
             patch("topology_scheduler.cuda_identity.ctypes.CDLL", return_value=driver):
            return read_cuda_identity()

    def test_driver_uuid_is_read_from_the_only_visible_device(self):
        for version in (1, 2):
            with self.subTest(version=version):
                driver = self.driver(version=version)
                self.assertEqual(self.read(driver), {"available": True, "device_count": 1,
                    "uuid": "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"})
                self.assertEqual(driver.cuDeviceGet.call_args.args[1], 0)
                self.assertEqual(driver.cuInit.argtypes, [ctypes.c_uint])
                self.assertEqual(driver.cuDeviceGet.restype, ctypes.c_int)

    def test_zero_or_multiple_visible_devices_are_refused(self):
        for count in (0, 2):
            with self.subTest(count=count):
                driver = self.driver(count=count)
                result = self.read(driver)
                self.assertFalse(result["available"])
                self.assertEqual(result["device_count"], count)
                driver.cuDeviceGet.assert_not_called()

    def test_driver_failures_and_missing_entry_points_are_reported(self):
        for name in ("cuInit", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid_v2"):
            with self.subTest(name=name):
                result = self.read(self.driver(fail=name))
                self.assertFalse(result["available"])
                self.assertIn(name, result["reason"])
        self.assertFalse(self.read(SimpleNamespace())["available"])

    def test_missing_driver_is_not_a_verified_assignment(self):
        with patch("topology_scheduler.cuda_identity.sys.platform", "linux"), \
             patch("topology_scheduler.cuda_identity.ctypes.CDLL", side_effect=OSError("absent")):
            self.assertFalse(read_cuda_identity()["available"])


class IdentityVerificationTests(unittest.TestCase):
    def resolve(self, ids=(0,), devices=DEVICES, observation=None):
        return resolve_device(ids, devices, device_order=ORDER, cuda_device=observation)

    def test_nvml_candidate_alone_never_proves_actual_assignment(self):
        resolved = self.resolve()
        self.assertIsNone(resolved["assigned_uuid"])
        self.assertEqual(resolved["ray_assigned_uuid"], "GPU-a")
        self.assertFalse(assignment_for(0, "a", "GPU-a", resolved).matched)

    def test_cuda_nvml_order_disagreement_is_refused_even_with_pci_bus_id(self):
        resolved = self.resolve(observation=cuda("GPU-b"))
        self.assertEqual(resolved["assigned_uuid"], "GPU-b")
        self.assertEqual(resolved["pci_bus_id"], "0000:01:00.0")
        self.assertEqual(resolved["ray_assigned_uuid"], "GPU-a")
        self.assertIn("numbering or visibility differs", resolved["problem"])
        self.assertFalse(assignment_for(0, "a", "GPU-a", resolved).matched)

    def test_explicit_nvml_index_is_used_instead_of_sequence_position(self):
        resolved = self.resolve(devices=DEVICES[::-1], observation=cuda())
        self.assertEqual(resolved["ray_assigned_uuid"], "GPU-a")
        self.assertTrue(assignment_for(0, "a", "GPU-a", resolved).matched)

    def test_full_uuid_visibility_tokens_are_supported_without_index_guessing(self):
        resolved = self.resolve(ids=("GPU-b",), observation=cuda("GPU-b"))
        self.assertEqual(resolved["assigned_index"], 1)
        self.assertTrue(assignment_for(0, "a", "GPU-b", resolved).matched)

    def test_fractional_and_boolean_ids_are_not_coerced_to_indices(self):
        for ids in ((0.5,), (True,), ("0.5",), (-1,)):
            with self.subTest(ids=ids):
                self.assertIn("problem", self.resolve(ids=ids, observation=cuda()))

    def test_duplicate_nvml_indices_or_uuids_cannot_be_verified(self):
        for devices in ((DEVICES[0], replace(DEVICES[1], index=0)),
                        (DEVICES[0], replace(DEVICES[1], uuid="GPU-a"))):
            with self.subTest(devices=devices):
                self.assertIn("problem", self.resolve(devices=devices, observation=cuda()))

    def test_driver_query_failure_is_retained_as_a_refusal_record(self):
        resolved = observe_worker_device(accelerator_ids=[0], devices_provider=lambda: DEVICES,
            environ={"CUDA_DEVICE_ORDER": ORDER}, cuda_provider=Mock(side_effect=OSError("driver lost")))
        record = assignment_for(3, "node-a", "GPU-a", resolved)
        self.assertFalse(record.matched)
        self.assertIn("driver lost", record.problem)
        self.assertEqual(record.identity_source, "injected")
        self.assertEqual(DeviceAssignment.from_dict(record.as_dict()), record)

    def test_production_path_uses_driver_query_and_blocks_work_on_mismatch(self):
        for observation in (cuda("GPU-b"), {"available": False, "reason": "driver missing"}, cuda(count=2)):
            with self.subTest(observation=observation), \
                 patch("ray.get_gpu_ids", return_value=[0]), \
                 patch("topology_scheduler.device_binding._node_devices", return_value=DEVICES), \
                 patch("topology_scheduler.cuda_identity.read_cuda_identity", return_value=observation):
                worker = Mock()
                bound = bind_worker(worker, DevicePlacement(("GPU-a",)), ("node-a",),
                                    environ={"CUDA_DEVICE_ORDER": ORDER})
                with self.assertRaises(DeviceBindingError) as caught:
                    bound(0)
                worker.assert_not_called()
                self.assertEqual(caught.exception.assignments[0].identity_source, "cuda_driver")
                self.assertIn("rank 0 on node-a", str(caught.exception))

    def test_matching_production_driver_observation_allows_work(self):
        with patch("ray.get_gpu_ids", return_value=[0]), \
             patch("topology_scheduler.device_binding._node_devices", return_value=DEVICES), \
             patch("topology_scheduler.cuda_identity.read_cuda_identity", return_value=cuda()):
            worker = Mock(return_value="result")
            payload = bind_worker(worker, DevicePlacement(("GPU-a",)), ("node-a",),
                                  environ={"CUDA_DEVICE_ORDER": ORDER})(0)
        worker.assert_called_once_with(0)
        self.assertTrue(payload["device"]["matched"])
        self.assertEqual(payload["device"]["identity_source"], "cuda_driver")

    def test_nvml_failure_keeps_an_already_observed_cuda_uuid(self):
        resolved = observe_worker_device(accelerator_ids=[0],
            devices_provider=Mock(side_effect=OSError("NVML unavailable")),
            cuda_provider=lambda: cuda("GPU-b"), environ={"CUDA_DEVICE_ORDER": ORDER})
        self.assertEqual(resolved["assigned_uuid"], "GPU-b")
        self.assertIn("NVML unavailable", resolved["problem"])
        self.assertFalse(assignment_for(0, "a", "GPU-b", resolved).matched)

    def test_worker_exception_retains_its_device_record_and_cause(self):
        failure = RuntimeError("workload failed")
        with patch("ray.get_gpu_ids", return_value=[0]):
            bound = bind_worker(Mock(side_effect=failure), DevicePlacement(("GPU-a",)), ("a",),
                devices_provider=lambda: DEVICES, cuda_provider=cuda,
                environ={"CUDA_DEVICE_ORDER": ORDER})
            with self.assertRaises(DeviceBindingError) as caught:
                bound(0)
        self.assertIs(caught.exception.__cause__, failure)
        self.assertEqual(caught.exception.assignments[0].assigned_uuid, "GPU-a")
        self.assertTrue(caught.exception.assignments[0].matched)


class ReservationValidationTests(unittest.TestCase):
    def test_whitespace_does_not_hide_duplicate_planned_devices(self):
        with self.assertRaisesRegex(ValueError, "cannot serve two ranks"):
            DevicePlacement(("GPU-a", " GPU-a "))
        self.assertEqual(DevicePlacement([" GPU-a "]).uuids, ("GPU-a",))

    def test_device_resource_must_be_one_unit_on_one_live_node(self):
        plan = Plan((Node("a", "A", 1, 80),), None)
        placement = DevicePlacement(("GPU-a",))
        for amount in (0.5, 2):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                preflight_devices(plan, placement, [{"Alive": True,
                    "Resources": {"topology_node:a": 1, "topology_gpu:GPU-a": amount}}])
        with self.assertRaises(ValueError):
            preflight_devices(plan, placement, [
                {"Alive": True, "Resources": {f"topology_node:{name}": 1, "topology_gpu:GPU-a": 1}}
                for name in ("a", "b")])

    def test_extra_resources_cannot_override_reservations(self):
        from topology_scheduler.ray_backend import run

        plan = Plan((Node("a", "A", 1, 80),), None)
        with patch("ray.is_initialized", return_value=True), \
             patch("ray.nodes", return_value=[{"Alive": True, "Resources": {
                 "CPU": 1, "GPU": 1, "topology_node:a": 1}}]), \
             patch("ray.util.placement_group.placement_group") as create:
            for extra in ([], [{"GPU": 0}], [{"CPU": 2}], [{"topology_node:a": 0}],
                          [{"topology_node:b": 1}], [{"topology_gpu:GPU-a": float("nan")}],
                          [{"topology_gpu:GPU-a": True}]):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    run(plan, lambda rank: rank, extra_resources=extra)
            create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
