"""Print the discovered network interface records.

Without arguments this reads a synthetic host, so the output is deterministic
and needs no NIC hardware. ``--host`` reads this machine's sysfs, and ``--live``
collects from every live Ray node beside the GPU inventory.
"""
import argparse
import json

from topology_scheduler import SysfsReader, collect_nic_inventory

# A synthetic host: a Mellanox port with RDMA, an onboard NIC whose driver
# reports no speed, a bridge, and loopback.
DEMO_TREE = {
    "sys/class/net/eth0": {"address": "ac:1f:6b:00:00:01", "operstate": "up",
                           "mtu": "9000", "speed": "100000", "type": "1"},
    "sys/class/net/eth0/device": {"uevent": "DRIVER=mlx5_core\n"
                                            "PCI_SLOT_NAME=0000:3b:00.0\n",
                                  "numa_node": "0"},
    "sys/class/net/eth1": {"address": "ac:1f:6b:00:00:02", "operstate": "down",
                           "mtu": "1500", "speed": "-1", "type": "1"},
    "sys/class/net/eth1/device": {"uevent": "DRIVER=igb\n"
                                            "PCI_SLOT_NAME=0000:04:00.0\n",
                                  "numa_node": "1"},
    "sys/class/net/br0": {"address": "02:42:00:00:00:01", "operstate": "up",
                          "mtu": "1500", "type": "1"},
    "sys/class/net/lo": {"address": "00:00:00:00:00:00", "operstate": "unknown",
                         "mtu": "65536", "type": "772"},
    "sys/class/infiniband/mlx5_0": {"node_type": "1: CA"},
    "sys/class/infiniband/mlx5_0/device": {"uevent": "PCI_SLOT_NAME=0000:3b:00.0\n"},
    "sys/class/infiniband/mlx5_0/ports/1": {"state": "4: ACTIVE",
                                            "link_layer": "InfiniBand"},
}


class DemoSysfs(SysfsReader):
    """Serve DEMO_TREE from memory, so the example needs no NIC."""

    def __init__(self, tree=DEMO_TREE):
        self.files, self.dirs = {}, set()
        for path, files in tree.items():
            parts = tuple(path.split("/"))
            self.dirs.update(parts[:index] for index in range(1, len(parts) + 1))
            self.files.update((parts + (name,), value) for name, value in files.items())

    def directories(self, *parts):
        return tuple(sorted({entry[len(parts)] for entry in self.dirs
                             if len(entry) > len(parts) and entry[:len(parts)] == parts}))

    def read(self, *parts):
        from topology_scheduler.nic_inventory import REPORTED, UNAVAILABLE

        if parts not in self.files:
            return None, UNAVAILABLE
        return str(self.files[parts]).strip(), REPORTED


def summarize(inventory):
    return [
        {
            "name": interface.name, "kind": interface.kind, "up": interface.up,
            "pci_address": interface.pci_address.value,
            "numa_node": interface.numa_node.value,
            "driver": interface.driver.value,
            "speed_mbps": interface.speed_mbps.value,
            "speed_confidence": interface.speed_mbps.confidence,
            "rdma": [device.name for device in interface.rdma],
        }
        for interface in inventory.interfaces
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--host", action="store_true",
                        help="read this machine's sysfs instead of the synthetic host")
    source.add_argument("--live", action="store_true",
                        help="collect from every live Ray node")
    arguments = parser.parse_args()

    if arguments.live:
        import ray

        from topology_scheduler import discover_nic_inventory

        ray.init(address="auto")
        try:
            inventories = discover_nic_inventory()
        finally:
            ray.shutdown()
        print(json.dumps({"synthetic_inputs": False,
                          "nodes": [item.as_dict() for item in inventories]}, indent=2))
        return

    if arguments.host:
        inventory = collect_nic_inventory()
        print(json.dumps({"synthetic_inputs": False, "summary": summarize(inventory),
                          "inventory": inventory.as_dict()}, indent=2))
        return

    inventory = collect_nic_inventory(sysfs=DemoSysfs(), node_name="demo")
    print(json.dumps({"synthetic_inputs": True, "summary": summarize(inventory),
                      "inventory": inventory.as_dict()}, indent=2))


if __name__ == "__main__":
    main()
