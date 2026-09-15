import unittest
from types import SimpleNamespace
from unittest.mock import patch

from topology_scheduler.inventory import (
    GPUDevice,
    RayNodeInventory,
    _node_marker,
    _read_nvml,
)


class FakeNvml:
    def __init__(self):
        self.initialized = False

    def nvmlInit(self):
        self.initialized = True

    def nvmlShutdown(self):
        self.initialized = False

    def nvmlDeviceGetCount(self):
        return 2

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetName(self, handle):
        return b"NVIDIA H100 80GB HBM3"

    def nvmlDeviceGetUUID(self, handle):
        return f"GPU-{handle}".encode()

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(total=80_000_000_000)

    def nvmlDeviceGetPciInfo(self, handle):
        return SimpleNamespace(busId=f"0000:{handle:02x}:00.0".encode())


class InventoryTests(unittest.TestCase):
    def test_reads_nvml_properties_via_ray_accelerator_parser(self):
        nvml = FakeNvml()
        devices = _read_nvml(nvml)
        self.assertFalse(nvml.initialized)
        self.assertEqual([d.accelerator_type for d in devices], ["H100", "H100"])
        self.assertEqual([d.memory_gb for d in devices], [80, 80])
        self.assertEqual([d.uuid for d in devices], ["GPU-0", "GPU-1"])

    def test_converts_detected_inventory_to_planner_node(self):
        device = GPUDevice(0, "GPU-0", "NVIDIA H100", "H100", 80, "0000:00:00.0")
        node = RayNodeInventory("id", "a", "topology_node:a", 1, (device,)).as_planner_node()
        self.assertEqual((node.name, node.gpu_model, node.available_gpus, node.memory_gb_per_gpu),
                         ("a", "H100", 1, 80))

    def test_rejects_mixed_gpu_models(self):
        a = GPUDevice(0, "0", "A", "A", 80, "0")
        b = GPUDevice(1, "1", "B", "B", 80, "1")
        with self.assertRaisesRegex(ValueError, "Mixed GPU models"):
            RayNodeInventory("id", "n", "topology_node:n", 2, (a, b)).as_planner_node()

    def test_requires_one_node_marker(self):
        self.assertEqual(_node_marker({"topology_node:a": 2}, "id"),
                         ("topology_node:a", "a"))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _node_marker({}, "id")


if __name__ == "__main__":
    unittest.main()
