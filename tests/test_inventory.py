import unittest
from types import SimpleNamespace
from unittest.mock import patch

from topology_scheduler.inventory import (
    GPUDevice,
    RayNodeInventory,
    _node_marker,
    _read_nvml,
    _read_nvml_snapshot,
)


class FakeNvml:
    NVML_NVLINK_MAX_LINKS = 4
    NVML_TOPOLOGY_INTERNAL = 0
    NVML_TOPOLOGY_SINGLE = 10
    NVML_TOPOLOGY_MULTIPLE = 20
    NVML_TOPOLOGY_HOSTBRIDGE = 30
    NVML_TOPOLOGY_NODE = 40
    NVML_TOPOLOGY_SYSTEM = 50

    class NVMLError(Exception):
        pass

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

    def nvmlDeviceGetTopologyCommonAncestor(self, left, right):
        return self.NVML_TOPOLOGY_INTERNAL

    def nvmlDeviceGetNvLinkState(self, handle, link):
        return link in (0, 1)

    def nvmlDeviceGetNvLinkRemotePciInfo(self, handle, link):
        peer = 1 - handle
        return SimpleNamespace(busId=f"0000:{peer:02x}:00.0".encode())


class UnsupportedTopologyNvml(FakeNvml):
    def nvmlDeviceGetTopologyCommonAncestor(self, left, right):
        raise self.NVMLError()

    def nvmlDeviceGetNvLinkState(self, handle, link):
        raise self.NVMLError()


class InventoryTests(unittest.TestCase):
    def test_reads_nvml_properties_via_ray_accelerator_parser(self):
        nvml = FakeNvml()
        devices = _read_nvml(nvml)
        self.assertFalse(nvml.initialized)
        self.assertEqual([d.accelerator_type for d in devices], ["H100", "H100"])
        self.assertEqual([d.memory_gb for d in devices], [80, 80])
        self.assertEqual([d.uuid for d in devices], ["GPU-0", "GPU-1"])

    def test_reads_pair_topology_and_direct_nvlinks(self):
        devices, connections = _read_nvml_snapshot(FakeNvml())
        self.assertEqual(len(devices), 2)
        self.assertEqual(len(connections), 1)
        self.assertEqual(connections[0].source_uuid, "GPU-0")
        self.assertEqual(connections[0].target_uuid, "GPU-1")
        self.assertEqual(connections[0].common_ancestor, "internal")
        self.assertEqual(connections[0].direct_nvlink_count, 2)

    def test_preserves_devices_when_topology_queries_are_unsupported(self):
        devices, connections = _read_nvml_snapshot(UnsupportedTopologyNvml())
        self.assertEqual(len(devices), 2)
        self.assertEqual(len(connections), 1)
        self.assertIsNone(connections[0].common_ancestor)
        self.assertIsNone(connections[0].direct_nvlink_count)

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
