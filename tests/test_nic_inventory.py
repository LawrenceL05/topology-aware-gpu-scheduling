import errno
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from topology_scheduler.nic_inventory import (
    LOOPBACK, PHYSICAL, REPORTED, UNAVAILABLE, UNREADABLE, UNSUPPORTED, VIRTUAL,
    NodeNICInventory, SysfsReader, collect_nic_inventory,
)

# A file value of None reads like a permission error, and PERMISSION/EINVAL
# mark the two ways sysfs refuses to answer.
DENIED, INVALID = None, "einval"
ETHERNET = {
    "address": "ac:1f:6b:00:00:01", "operstate": "up", "mtu": "9000",
    "speed": "100000", "type": "1",
}
MELLANOX_UEVENT = "DRIVER=mlx5_core\nPCI_SLOT_NAME=0000:3b:00.0\nINTERFACE=eth0\n"


class FakeSysfs(SysfsReader):
    """An in-memory sysfs: {"sys/class/net/eth0": {"mtu": "9000"}}."""

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
        if value == INVALID:
            return None, UNSUPPORTED
        return str(value).strip(), REPORTED


def net(name, files, *, uevent=None):
    tree = {f"sys/class/net/{name}": dict(files)}
    tree[f"sys/class/net/{name}/device"] = {"uevent": uevent} if uevent else {}
    return tree


def collect(*trees, **options):
    merged = {}
    for tree in trees:
        merged.update(tree)
    return collect_nic_inventory(sysfs=FakeSysfs(merged), **options)


class InterfaceKindTests(unittest.TestCase):
    def test_ethernet_with_a_pci_function_is_physical(self):
        inventory = collect(net("eth0", dict(ETHERNET, **{"device/numa_node": "0"}),
                                uevent=MELLANOX_UEVENT))
        (interface,) = inventory.interfaces
        self.assertEqual(interface.kind, PHYSICAL)
        self.assertEqual(interface.pci_address.value, "0000:3b:00.0")
        self.assertEqual(interface.driver.value, "mlx5_core")
        self.assertEqual(interface.speed_mbps.value, 100000)
        self.assertEqual(interface.mtu.value, 9000)
        self.assertEqual(interface.mac.value, "ac:1f:6b:00:00:01")
        self.assertTrue(interface.up)

    def test_numa_node_is_read_from_the_pci_function(self):
        tree = net("eth0", ETHERNET, uevent=MELLANOX_UEVENT)
        tree["sys/class/net/eth0/device"]["numa_node"] = "1"
        (interface,) = collect(tree).interfaces
        self.assertEqual(interface.numa_node.value, 1)
        self.assertEqual(interface.numa_node.confidence, REPORTED)

    def test_loopback_and_bridges_are_not_physical(self):
        inventory = collect(
            net("lo", {"address": "00:00:00:00:00:00", "operstate": "unknown",
                       "mtu": "65536", "type": "772", "speed": INVALID}),
            net("br0", {"address": "02:42:00:00:00:01", "operstate": "down",
                        "mtu": "1500", "type": "1", "speed": INVALID}))
        kinds = {item.name: item.kind for item in inventory.interfaces}
        self.assertEqual(kinds, {"lo": LOOPBACK, "br0": VIRTUAL})
        self.assertEqual(inventory.physical, ())
        bridge = next(item for item in inventory.interfaces if item.name == "br0")
        self.assertFalse(bridge.up)
        self.assertEqual(bridge.operstate.value, "down")

    def test_an_interface_the_kernel_barely_describes_is_unknown(self):
        (interface,) = collect(net("weird", {})).interfaces
        self.assertEqual(interface.kind, "unknown")
        self.assertEqual(interface.mac.confidence, UNAVAILABLE)


class ConfidenceTests(unittest.TestCase):
    def test_unsupported_speed_is_not_a_measured_zero(self):
        (interface,) = collect(net("br0", dict(ETHERNET, speed=INVALID))).interfaces
        self.assertIsNone(interface.speed_mbps.value)
        self.assertEqual(interface.speed_mbps.confidence, UNSUPPORTED)
        self.assertFalse(interface.speed_mbps.known)

    def test_driver_reporting_minus_one_is_unsupported(self):
        (interface,) = collect(net("eth0", dict(ETHERNET, speed="-1"))).interfaces
        self.assertIsNone(interface.speed_mbps.value)
        self.assertEqual(interface.speed_mbps.confidence, UNSUPPORTED)

    def test_absent_file_is_unavailable_and_keeps_its_source(self):
        files = {key: value for key, value in ETHERNET.items() if key != "speed"}
        (interface,) = collect(net("eth0", files)).interfaces
        self.assertEqual(interface.speed_mbps.confidence, UNAVAILABLE)
        self.assertEqual(interface.speed_mbps.source, "/sys/class/net/eth0/speed")

    def test_permission_errors_are_reported_per_field_not_dropped(self):
        (interface,) = collect(
            net("eth0", dict(ETHERNET, speed=DENIED, address=DENIED))).interfaces
        self.assertEqual(interface.speed_mbps.confidence, UNREADABLE)
        self.assertEqual(interface.name, "eth0")
        self.assertEqual(len(interface.problems), 2)
        self.assertTrue(any("/sys/class/net/eth0/speed" in problem
                            for problem in interface.problems))

    def test_a_node_with_no_interfaces_still_returns(self):
        inventory = collect({})
        self.assertEqual(inventory.interfaces, ())
        self.assertIn("no interfaces were found", inventory.problems[0])


class RDMATests(unittest.TestCase):
    def fabric(self, link_layer="InfiniBand"):
        return {
            "sys/class/infiniband/mlx5_0": {"node_type": "1: CA"},
            "sys/class/infiniband/mlx5_0/device": {"uevent": MELLANOX_UEVENT},
            "sys/class/infiniband/mlx5_0/ports/1": {
                "state": "4: ACTIVE", "link_layer": link_layer},
        }

    def test_rdma_device_is_attached_to_the_interface_sharing_its_pci_function(self):
        inventory = collect(net("eth0", ETHERNET, uevent=MELLANOX_UEVENT), self.fabric())
        (interface,) = inventory.interfaces
        (device,) = interface.rdma
        self.assertEqual(device.name, "mlx5_0")
        self.assertEqual(device.pci_address, "0000:3b:00.0")
        self.assertEqual(device.link_layer, "InfiniBand")
        self.assertEqual(device.port_states, ("1:4: ACTIVE",))

    def test_rdma_is_not_attached_to_an_unrelated_interface(self):
        other = "DRIVER=igb\nPCI_SLOT_NAME=0000:04:00.0\n"
        inventory = collect(net("eth1", ETHERNET, uevent=other), self.fabric())
        (interface,) = inventory.interfaces
        self.assertEqual(interface.rdma, ())

    def test_interface_without_pci_never_inherits_an_rdma_device(self):
        inventory = collect(net("br0", ETHERNET), self.fabric())
        self.assertEqual(inventory.interfaces[0].rdma, ())


class SerializationTests(unittest.TestCase):
    def test_records_are_stable_ordered_and_json_safe(self):
        trees = (net("eth1", ETHERNET, uevent=MELLANOX_UEVENT),
                 net("eth0", ETHERNET, uevent=MELLANOX_UEVENT),
                 net("lo", {"type": "772"}))
        first = collect(*trees, node_id="ray-1", node_name="a")
        second = collect(*reversed(trees), node_id="ray-1", node_name="a")
        self.assertEqual([item.name for item in first.interfaces],
                         ["eth0", "eth1", "lo"])
        self.assertEqual(json.dumps(first.as_dict(), sort_keys=True),
                         json.dumps(second.as_dict(), sort_keys=True))

    def test_serialization_carries_provenance_identity_and_units(self):
        value = collect(net("eth0", ETHERNET, uevent=MELLANOX_UEVENT),
                        node_id="ray-1", node_name="a").as_dict()
        self.assertEqual((value["node_id"], value["node_name"]), ("ray-1", "a"))
        speed = value["interfaces"][0]["speed_mbps"]
        self.assertEqual(speed["value"], 100000)
        self.assertEqual(speed["source"], "/sys/class/net/eth0/speed")
        self.assertEqual(speed["confidence"], REPORTED)
        self.assertIn("not measured throughput", value["units"]["speed_mbps"])

    def test_empty_inventory_serializes(self):
        self.assertEqual(NodeNICInventory("id", "a", ()).as_dict()["interfaces"], [])


class RealSysfsTests(unittest.TestCase):
    def test_missing_root_reads_as_empty_rather_than_raising(self):
        reader = SysfsReader("/nonexistent-topology-scheduler-root")
        self.assertEqual(reader.directories("sys", "class", "net"), ())
        self.assertEqual(reader.read("sys", "class", "net", "eth0", "mtu"),
                         (None, UNAVAILABLE))

    @unittest.skipUnless(sys.platform.startswith("linux"), "sysfs is Linux-only")
    def test_this_linux_host_reports_its_own_interfaces(self):
        inventory = collect_nic_inventory()
        names = [item.name for item in inventory.interfaces]
        self.assertIn("lo", names)
        self.assertEqual(names, sorted(names))
        loopback = next(item for item in inventory.interfaces if item.name == "lo")
        self.assertEqual(loopback.kind, LOOPBACK)
        # Loopback has no driver-reported speed; it must not read as zero.
        self.assertIsNone(loopback.speed_mbps.value)
        for interface in inventory.physical:
            with self.subTest(interface=interface.name):
                self.assertTrue(interface.pci_address.known)
        json.dumps(inventory.as_dict())


class ReaderTests(unittest.TestCase):
    """Exercise the real reader, since fixtures bypass its error handling."""

    def read_raising(self, error):
        with patch.object(Path, "read_text", side_effect=error):
            return SysfsReader("/").read("sys", "class", "net", "eth0", "speed")

    def test_sysfs_refusals_are_told_apart(self):
        for error, expected in (
            (OSError(errno.EINVAL, "Invalid argument"), UNSUPPORTED),
            (PermissionError(errno.EACCES, "Permission denied"), UNREADABLE),
            (OSError(errno.EIO, "I/O error"), UNREADABLE),
            (FileNotFoundError(), UNAVAILABLE),
        ):
            with self.subTest(error=error):
                self.assertEqual(self.read_raising(error), (None, expected))

    def test_values_are_stripped(self):
        with patch.object(Path, "read_text", return_value=" 9000 \n"):
            self.assertEqual(SysfsReader("/").read("a"), ("9000", REPORTED))


if __name__ == "__main__":
    unittest.main()
