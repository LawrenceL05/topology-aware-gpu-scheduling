import json
import sys
import tempfile
import unittest
from pathlib import Path

from topology_scheduler import GPUDevice
from topology_scheduler.host_topology import (
    GPULocality, HostTopology, Proximity, collect_host_topology,
    normalize_pci_address,
)
from topology_scheduler.nic_inventory import (
    LOOPBACK, PHYSICAL, REPORTED, UNAVAILABLE, UNREADABLE, SysfsReader,
    collect_nic_inventory,
)

ROOT_COMPLEX = "pci0000:00"
FAR_COMPLEX = "pci0000:80"
GPU_ADDRESS = "0000:17:00.0"
# A file value of None reads like a file the process may not open.
DENIED = None
ETH = {"address": "ac:1f:6b:00:00:01", "operstate": "up", "mtu": "9000",
       "speed": "100000", "type": "1"}


class FakeSysfs(SysfsReader):
    """An in-memory sysfs tree, as ``{"sys/devices/x": {"numa_node": "0"}}``.

    Real directories cannot be used here: PCI addresses contain colons, which
    Windows rejects in a path.
    """

    def __init__(self, tree):
        self.files, self.dirs = {}, set()
        for path, files in tree.items():
            parts = tuple(path.split("/"))
            self.dirs.update(parts[:index] for index in range(1, len(parts) + 1))
            self.files.update((parts + (name,), value) for name, value in files.items())

    def directories(self, *parts):
        return tuple(sorted({entry[len(parts)] for entry in self.dirs
                             if len(entry) > len(parts) and entry[:len(parts)] == parts}))

    def read(self, *parts):
        if parts not in self.files:
            return None, UNAVAILABLE
        value = self.files[parts]
        if value is DENIED:
            return None, UNREADABLE
        return str(value).strip(), REPORTED


def device(gpu_address=GPU_ADDRESS, uuid="GPU-0"):
    return GPUDevice(0, uuid, "NVIDIA H100 80GB HBM3", "H100", 80, gpu_address)


def sysfs(layout, virtual=("lo",)):
    """Build a fixture from (PCI path, numa_node, interfaces) entries.

    Each entry places one PCI function under ``/sys/devices`` and, for every
    interface on it, the ``/sys/class/net`` entry the NIC inventory reads,
    including the ``device/uevent`` that names its PCI slot. ``numa_node`` is
    an integer, ``-1`` when the kernel does not know, ``None`` to leave the
    file out, or ``"unreadable"`` for a permission-limited host.
    """
    tree = {}
    for path, numa, interfaces in layout:
        numa_file = ({} if numa is None
                     else {"numa_node": DENIED if numa == "unreadable" else numa})
        tree["/".join(("sys", "devices") + tuple(path))] = dict(numa_file)
        for name, fields in interfaces.items():
            tree[f"sys/class/net/{name}"] = dict(ETH, **fields)
            tree[f"sys/class/net/{name}/device"] = dict(
                numa_file, uevent=f"PCI_SLOT_NAME={path[-1]}\nDRIVER=test\n")
    for name in virtual:
        tree[f"sys/class/net/{name}"] = {"type": "772" if name == "lo" else "1"}
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

    def nic(self, topology, name):
        return next(item for item in topology.nics if item.name == name)


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

    def test_loopback_stays_present_as_unknown(self):
        topology = self.collect([((ROOT_COMPLEX, GPU_ADDRESS), 0, {})])
        loopback = self.proximity(topology, "lo")
        self.assertEqual(loopback.proximity, Proximity.UNKNOWN)
        self.assertIn("no PCI device", loopback.reason)
        self.assertEqual(self.nic(topology, "lo").kind, LOOPBACK)
        self.assertIsNone(self.nic(topology, "lo").pci_address)
        self.assertIsNone(topology.gpus[0].nearest_nic)


class InventoryReuseTests(TopologyFixture):
    """The interfaces come from the NIC inventory, not a second sysfs walk."""

    LAYOUT = [((ROOT_COMPLEX, "0000:00:01.0", GPU_ADDRESS), 0, {}),
              ((ROOT_COMPLEX, "0000:00:01.0", "0000:18:00.0"), 0, {"eth0": ETH})]

    def test_each_interface_keeps_its_inventory_provenance(self):
        nic = self.nic(self.collect(self.LAYOUT), "eth0")
        self.assertEqual(nic.kind, PHYSICAL)
        self.assertEqual(nic.interface.driver.value, "test")
        self.assertEqual(nic.interface.mac.value, "ac:1f:6b:00:00:01")
        self.assertEqual(nic.interface.mtu.value, 9000)
        self.assertEqual(nic.speed_mbps, 100000)
        self.assertEqual(nic.operstate, "up")
        # Provenance survives: every field still names the file it came from.
        self.assertEqual(nic.interface.speed_mbps.confidence, REPORTED)
        self.assertIn("/sys/class/net/eth0/speed", nic.interface.speed_mbps.source)

    def test_a_supplied_inventory_is_used_instead_of_re_reading(self):
        reader = sysfs(self.LAYOUT)
        collected = collect_host_topology([device()], sysfs=reader)
        reused = collect_host_topology(
            [device()], sysfs=reader,
            inventory=collect_nic_inventory(sysfs=reader))
        self.assertEqual(json.dumps(collected.as_dict(), sort_keys=True),
                         json.dumps(reused.as_dict(), sort_keys=True))

    def test_an_interface_outside_the_pci_tree_is_kept_with_a_reason(self):
        # The interface names a PCI slot that /sys/devices does not contain.
        reader = sysfs([((ROOT_COMPLEX, GPU_ADDRESS), 0, {})])
        reader.files[("sys", "class", "net", "eth9", "type")] = "1"
        reader.files[("sys", "class", "net", "eth9", "device", "uevent")] = (
            "PCI_SLOT_NAME=0000:aa:00.0\n")
        reader.dirs.update({("sys", "class", "net", "eth9"),
                            ("sys", "class", "net", "eth9", "device")})
        topology = collect_host_topology([device()], sysfs=reader)
        nic = self.nic(topology, "eth9")
        self.assertEqual((nic.pci_address, nic.pci_path), ("0000:aa:00.0", ()))
        self.assertEqual(self.proximity(topology, "eth9").proximity, Proximity.UNKNOWN)
        self.assertTrue(any("0000:aa:00.0 is not present" in line
                            for line in topology.diagnostics))

    def test_node_level_inventory_problems_reach_the_diagnostics(self):
        topology = self.collect([], devices=[device()], virtual=())
        self.assertTrue(any("no interfaces were found" in line
                            for line in topology.diagnostics))


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
        self.assertTrue(any(f"numa_node is {UNAVAILABLE}" in line
                            for line in topology.diagnostics))

    def test_unreadable_numa_file_is_reported_for_gpu_and_interface(self):
        topology = self.collect([
            ((ROOT_COMPLEX, GPU_ADDRESS), "unreadable", {}),
            ((FAR_COMPLEX, "0000:81:00.0"), "unreadable", {"eth0": ETH}),
        ])
        self.assertTrue(any(f"numa_node is {UNREADABLE}" in line and "GPU" in line
                            for line in topology.diagnostics))
        self.assertTrue(any("interface eth0" in line and UNREADABLE in line
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

    def test_interface_speed_of_minus_one_is_unknown(self):
        topology = self.collect([
            ((ROOT_COMPLEX, GPU_ADDRESS), 0, {}),
            ((ROOT_COMPLEX, "0000:81:00.0"), 0,
             {"eth0": dict(ETH, speed="-1", operstate="down")}),
        ])
        nic = self.nic(topology, "eth0")
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
        interface = next(item for item in value["nics"] if item["name"] == "eth1")
        self.assertEqual(interface["normalized_pci_address"], "0000:18:00.0")
        self.assertEqual(interface["pci_path"][-1], "0000:18:00.0")
        # The inventory's per-field provenance is part of the record.
        self.assertEqual(interface["speed_mbps"]["confidence"], REPORTED)
        self.assertIn("gpu_numa_node", value["sources"])
        self.assertIn("nic_numa_node", value["sources"])
        json.dumps(value)


class RealSysfsTests(unittest.TestCase):
    """Exercise the reader that talks to a real filesystem."""

    def test_missing_root_reads_as_empty_rather_than_raising(self):
        reader = SysfsReader(Path(tempfile.gettempdir()) / "topology-scheduler-absent")
        self.assertEqual(reader.directories("sys", "devices"), ())
        self.assertEqual(reader.read("sys", "devices", "numa_node"),
                         (None, UNAVAILABLE))

    @unittest.skipUnless(sys.platform.startswith("linux"), "sysfs is Linux-only")
    def test_this_linux_host_reports_interfaces(self):
        topology = collect_host_topology([])
        names = [nic.name for nic in topology.nics]
        self.assertIn("lo", names)
        self.assertEqual(names, sorted(names))
        self.assertEqual(self.nic_kind(topology, "lo"), LOOPBACK)
        for nic in topology.nics:
            with self.subTest(nic=nic.name):
                if nic.pci_address is None:
                    self.assertEqual(nic.pci_path, ())
                elif nic.pci_path:
                    # The address was found by walking the real device tree.
                    self.assertEqual(nic.pci_path[-1], nic.pci_address)
        json.dumps(topology.as_dict())

    def nic_kind(self, topology, name):
        return next(nic.kind for nic in topology.nics if nic.name == name)


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
