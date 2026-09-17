import json
import sys
import tempfile
import unittest
from pathlib import Path

from topology_scheduler import GPUDevice
from topology_scheduler.host_topology import (
    GPULocality, HostTopology, Proximity, SysfsReader, collect_host_topology,
    normalize_pci_address,
)

ROOT_COMPLEX = "pci0000:00"
FAR_COMPLEX = "pci0000:80"
GPU_ADDRESS = "0000:17:00.0"
ETH = {"speed": 100000, "operstate": "up"}


class FakeSysfs(SysfsReader):
    """An in-memory sysfs tree, as ``{"sys/devices/x": {"numa_node": "0"}}``.

    Real directories cannot be used here: PCI addresses contain colons, which
    Windows rejects in a path. A file value of ``None`` reads like a file the
    process is not allowed to open.
    """

    def __init__(self, tree):
        self.files, self.dirs = {}, set()
        for path, files in tree.items():
            parts = tuple(path.split("/"))
            for index in range(1, len(parts) + 1):
                self.dirs.add(parts[:index])
            for name, value in files.items():
                self.files[parts + (name,)] = value

    def directories(self, *parts):
        return tuple(sorted({entry[len(parts)] for entry in self.dirs
                             if len(entry) > len(parts) and entry[:len(parts)] == parts}))

    def read(self, *parts):
        if parts not in self.files:
            return None, "missing"
        value = self.files[parts]
        if value is None:
            return None, "unreadable (PermissionError)"
        return str(value).strip(), None


def device(gpu_address=GPU_ADDRESS, uuid="GPU-0"):
    return GPUDevice(0, uuid, "NVIDIA H100 80GB HBM3", "H100", 80, gpu_address)


def sysfs(layout, virtual=("lo",)):
    """Build a fixture from (PCI path, numa_node, interfaces) entries.

    ``numa_node`` is an integer, ``-1`` when the kernel does not know, ``None``
    to leave the file out, or ``"unreadable"`` for a permission-limited host.
    """
    tree = {}
    for path, numa, interfaces in layout:
        directory = "/".join(("sys", "devices") + tuple(path))
        tree[directory] = (
            {} if numa is None else {"numa_node": None if numa == "unreadable" else numa})
        for name, fields in interfaces.items():
            tree[f"{directory}/net/{name}"] = dict(fields)
    for name in virtual:
        tree[f"sys/devices/virtual/net/{name}"] = {}
    return FakeSysfs(tree)


class TopologyFixture(unittest.TestCase):
    def collect(self, layout, *, devices=None, virtual=("lo",)):
        return collect_host_topology(
            devices if devices is not None else [device()],
            sysfs=sysfs(layout, virtual))

    def proximity(self, topology, nic_name):
        (gpu,) = topology.gpus
        return next(item for item in gpu.nics if item.nic_name == nic_name)


class ClassificationTests(TopologyFixture):
    def test_shared_switch_beats_shared_root_complex(self):
        topology = self.collect([
            ((ROOT_COMPLEX, "0000:00:01.0", GPU_ADDRESS), 0, {}),
            ((ROOT_COMPLEX, "0000:00:01.0", "0000:18:00.0"), 0, {"eth0": ETH}),
            ((ROOT_COMPLEX, "0000:00:02.0", "0000:19:00.0"), 0, {"eth1": ETH}),
        ])
        close, far = self.proximity(topology, "eth0"), self.proximity(topology, "eth1")
        self.assertEqual(close.proximity, Proximity.SAME_SWITCH)
        self.assertEqual(close.shared_pci_ancestor, "0000:00:01.0")
        self.assertEqual(far.proximity, Proximity.SAME_ROOT_COMPLEX)
        self.assertEqual(far.shared_pci_ancestor, ROOT_COMPLEX)
        self.assertEqual(topology.gpus[0].nearest_nic.nic_name, "eth0")

    def test_multi_socket_separates_same_numa_from_cross_numa(self):
        topology = self.collect([
            ((ROOT_COMPLEX, "0000:00:01.0", GPU_ADDRESS), 1, {}),
            ((FAR_COMPLEX, "0000:80:01.0", "0000:81:00.0"), 1, {"eth0": ETH}),
            ((FAR_COMPLEX, "0000:80:02.0", "0000:82:00.0"), 0, {"eth1": ETH}),
        ])
        self.assertEqual(self.proximity(topology, "eth0").proximity, Proximity.SAME_NUMA)
        cross = self.proximity(topology, "eth1")
        self.assertEqual(cross.proximity, Proximity.CROSS_NUMA)
        self.assertEqual((cross.gpu_numa_node, cross.nic_numa_node), (1, 0))
        self.assertIsNone(cross.shared_pci_ancestor)

    def test_same_multifunction_device_is_closest(self):
        topology = self.collect([
            ((ROOT_COMPLEX, "0000:00:01.0", GPU_ADDRESS), 0, {}),
            ((ROOT_COMPLEX, "0000:00:01.0", "0000:17:00.1"), 0, {"eth0": ETH}),
        ])
        self.assertEqual(self.proximity(topology, "eth0").proximity,
                         Proximity.SAME_DEVICE)

    def test_virtual_interface_stays_present_as_unknown(self):
        topology = self.collect([((ROOT_COMPLEX, GPU_ADDRESS), 0, {})])
        loopback = self.proximity(topology, "lo")
        self.assertEqual(loopback.proximity, Proximity.UNKNOWN)
        self.assertIn("no PCI device", loopback.reason)
        self.assertTrue(next(nic for nic in topology.nics if nic.name == "lo").virtual)
        self.assertIsNone(topology.gpus[0].nearest_nic)


class MissingEvidenceTests(TopologyFixture):
    def test_kernel_without_numa_node_is_unknown_not_guessed(self):
        topology = self.collect([
            ((ROOT_COMPLEX, GPU_ADDRESS), -1, {}),
            ((FAR_COMPLEX, "0000:81:00.0"), 0, {"eth0": ETH}),
        ])
        far = self.proximity(topology, "eth0")
        self.assertEqual(far.proximity, Proximity.UNKNOWN)
        self.assertIn("GPU NUMA node is unknown", far.reason)
        self.assertIsNone(topology.gpus[0].numa_node)
        self.assertTrue(any("no NUMA node (-1)" in line for line in topology.diagnostics))

    def test_absent_numa_file_is_reported(self):
        topology = self.collect([
            ((ROOT_COMPLEX, GPU_ADDRESS), None, {}),
            ((FAR_COMPLEX, "0000:81:00.0"), 0, {"eth0": ETH}),
        ])
        self.assertTrue(any("numa_node missing" in line for line in topology.diagnostics))

    def test_unreadable_numa_file_is_reported_for_gpu_and_interface(self):
        topology = self.collect([
            ((ROOT_COMPLEX, GPU_ADDRESS), "unreadable", {}),
            ((FAR_COMPLEX, "0000:81:00.0"), "unreadable", {"eth0": ETH}),
        ])
        self.assertTrue(any("numa_node unreadable" in line and "GPU" in line
                            for line in topology.diagnostics))
        self.assertTrue(any("interface eth0" in line and "unreadable" in line
                            for line in topology.diagnostics))
        self.assertEqual(self.proximity(topology, "eth0").proximity, Proximity.UNKNOWN)

    def test_gpu_missing_from_pci_tree_is_kept_with_a_reason(self):
        topology = self.collect(
            [((ROOT_COMPLEX, "0000:81:00.0"), 0, {"eth0": ETH})],
            devices=[device("0000:99:00.0")])
        (gpu,) = topology.gpus
        self.assertEqual(gpu.pci_path, ())
        self.assertIsNone(gpu.numa_node)
        self.assertEqual(gpu.nics[0].proximity, Proximity.UNKNOWN)
        self.assertTrue(any("0000:99:00.0 is not present" in line
                            for line in topology.diagnostics))

    def test_virtualized_host_without_pci_or_interfaces(self):
        topology = self.collect([], devices=[device()], virtual=())
        self.assertEqual(topology.nics, ())
        self.assertEqual(topology.gpus[0].nics, ())
        self.assertIsNone(topology.gpus[0].nearest_nic)
        self.assertTrue(any("no network interfaces" in line
                            for line in topology.diagnostics))

    def test_interface_speed_of_minus_one_is_unknown(self):
        topology = self.collect([
            ((ROOT_COMPLEX, GPU_ADDRESS), 0, {}),
            ((ROOT_COMPLEX, "0000:81:00.0"), 0,
             {"eth0": {"speed": -1, "operstate": "down"}}),
        ])
        (nic,) = [item for item in topology.nics if item.name == "eth0"]
        self.assertIsNone(nic.speed_mbps)
        self.assertEqual(nic.operstate, "down")


class StabilityTests(TopologyFixture):
    LAYOUT = [
        ((ROOT_COMPLEX, "0000:00:01.0", GPU_ADDRESS), 0, {}),
        ((ROOT_COMPLEX, "0000:00:01.0", "0000:18:00.0"), 0, {"eth1": ETH}),
        ((ROOT_COMPLEX, "0000:00:02.0", "0000:19:00.0"), 0, {"eth0": ETH}),
        ((FAR_COMPLEX, "0000:80:01.0", "0000:81:00.0"), 1, {"eth2": ETH}),
    ]

    def test_repeated_discovery_serializes_identically(self):
        reader = sysfs(self.LAYOUT)
        devices = [device("00000000:17:00.0"), device("0000:19:00.0", "GPU-1")]
        first = collect_host_topology(devices, sysfs=reader)
        second = collect_host_topology(list(reversed(devices)), sysfs=reader)
        self.assertEqual(json.dumps(first.as_dict(), sort_keys=True),
                         json.dumps(second.as_dict(), sort_keys=True))

    def test_ordering_is_by_address_name_and_proximity(self):
        topology = self.collect(self.LAYOUT)
        self.assertEqual([nic.name for nic in topology.nics], ["eth0", "eth1", "eth2", "lo"])
        (gpu,) = topology.gpus
        self.assertEqual([item.nic_name for item in gpu.nics],
                         ["eth1", "eth0", "eth2", "lo"])
        # eth2 sits on the other socket, so it ranks after the local interfaces.
        self.assertEqual([item.proximity.value for item in gpu.nics],
                         ["same-switch", "same-root-complex", "cross-numa", "unknown"])

    def test_serialization_keeps_evidence_and_sources(self):
        value = self.collect(self.LAYOUT).as_dict()
        nearest = value["gpus"][0]["nics"][0]
        self.assertEqual(value["gpus"][0]["nearest_nic"], "eth1")
        self.assertEqual(nearest["proximity"], "same-switch")
        self.assertEqual(nearest["shared_pci_ancestor"], "0000:00:01.0")
        self.assertEqual(value["gpus"][0]["pci_path"],
                         [ROOT_COMPLEX, "0000:00:01.0", GPU_ADDRESS])
        self.assertIn("numa_node", value["sources"])
        json.dumps(value)


class RealSysfsTests(unittest.TestCase):
    """Exercise the reader that talks to a real filesystem."""

    def test_missing_root_reads_as_empty_rather_than_raising(self):
        reader = SysfsReader(Path(tempfile.gettempdir()) / "topology-scheduler-absent")
        self.assertEqual(reader.directories("sys", "devices"), ())
        self.assertEqual(reader.read("sys", "devices", "numa_node"), (None, "missing"))

    @unittest.skipUnless(sys.platform.startswith("linux"), "sysfs is Linux-only")
    def test_this_linux_host_reports_interfaces(self):
        topology = collect_host_topology([])
        names = [nic.name for nic in topology.nics]
        self.assertIn("lo", names)
        self.assertEqual(names, sorted(names))
        self.assertTrue(all(nic.virtual for nic in topology.nics if nic.name == "lo"))
        for nic in topology.nics:
            with self.subTest(nic=nic.name):
                if nic.virtual:
                    self.assertEqual((nic.pci_address, nic.pci_path), (None, ()))
                else:
                    # A PCI-attached interface was found by walking the real tree.
                    self.assertIsNotNone(nic.pci_address)
                    self.assertEqual(nic.pci_path[-1], nic.pci_address)
        json.dumps(topology.as_dict())


class AddressTests(unittest.TestCase):
    def test_nvml_and_sysfs_forms_normalize_together(self):
        for value in ("00000000:17:00.0", "0000:17:00.0", "  0000:17:00.0  ",
                      "00000000:17:00.0".upper()):
            self.assertEqual(normalize_pci_address(value), "0000:17:00.0")

    def test_rejects_unrecognized_addresses(self):
        for value in ("0000:17:00", "not-an-address", "zzzz:17:00.0", ""):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "Unrecognized PCI address"):
                normalize_pci_address(value)


class ModelTests(unittest.TestCase):
    def test_proximity_ranks_closest_first_and_unknown_last(self):
        ranks = [item.rank for item in Proximity]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(Proximity.SAME_DEVICE.rank, 0)
        self.assertEqual(Proximity.UNKNOWN.rank, len(list(Proximity)) - 1)

    def test_empty_topology_serializes(self):
        value = HostTopology("a", "id", (), ()).as_dict()
        self.assertEqual((value["gpus"], value["nics"], value["diagnostics"]),
                         ([], [], []))

    def test_locality_without_interfaces_has_no_nearest(self):
        self.assertIsNone(GPULocality("GPU-0", "0000:17:00.0", (), 0, ()).nearest_nic)


if __name__ == "__main__":
    unittest.main()
