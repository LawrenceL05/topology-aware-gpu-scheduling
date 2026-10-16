"""Print GPU to NUMA to nearest-NIC mappings.

Without arguments this uses a synthetic two-socket host so the output is
deterministic and needs no GPU. ``--host`` reads this machine's sysfs, and
``--live`` maps every GPU node of a running Ray cluster.
"""
import argparse
import json

from topology_scheduler import GPUDevice, SysfsReader, collect_host_topology
from topology_scheduler.nic_inventory import REPORTED, UNAVAILABLE

ETHERNET = {"address": "ac:1f:6b:00:00:01", "operstate": "up", "mtu": "9000",
            "speed": "200000", "type": "1"}


def interface(name, pci_address, numa_node):
    """The `/sys/class/net` entry the NIC inventory reads for one interface."""
    return {
        f"sys/class/net/{name}": dict(ETHERNET),
        f"sys/class/net/{name}/device": {
            "numa_node": numa_node,
            "uevent": f"PCI_SLOT_NAME={pci_address}\nDRIVER=demo\n"},
    }


# A synthetic two-socket host: one GPU per socket, a NIC beside the first
# behind the same switch, and the second socket's NIC one switch away.
DEMO_TREE = {
    "sys/devices/pci0000:00/0000:00:01.0/0000:17:00.0": {"numa_node": "0"},
    "sys/devices/pci0000:00/0000:00:01.0/0000:18:00.0": {"numa_node": "0"},
    "sys/devices/pci0000:80/0000:80:01.0/0000:99:00.0": {"numa_node": "1"},
    "sys/devices/pci0000:80/0000:80:02.0/0000:9a:00.0": {"numa_node": "1"},
    "sys/class/net/lo": {"type": "772"},
    "sys/devices/virtual/net/lo": {},
    **interface("eth0", "0000:18:00.0", "0"),
    **interface("eth1", "0000:9a:00.0", "1"),
}
DEMO_GPUS = (
    GPUDevice(0, "GPU-aaa", "NVIDIA H100 80GB HBM3", "H100", 80, "00000000:17:00.0"),
    GPUDevice(1, "GPU-bbb", "NVIDIA H100 80GB HBM3", "H100", 80, "00000000:99:00.0"),
)


class DemoSysfs(SysfsReader):
    """Serve DEMO_TREE in memory; PCI addresses cannot be real directories."""

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
        if parts not in self.files:
            return None, UNAVAILABLE
        return self.files[parts], REPORTED


def summarize(topology):
    return {
        "gpus": [
            {
                "gpu": gpu.uuid,
                "pci_address": gpu.pci_address,
                "numa_node": gpu.numa_node,
                "nearest_nic": gpu.nearest_nic.nic_name if gpu.nearest_nic else None,
                "proximity": gpu.nics[0].proximity.value if gpu.nics else None,
                "shared_pci_ancestor":
                    gpu.nics[0].shared_pci_ancestor if gpu.nics else None,
            }
            for gpu in topology.gpus
        ],
        # The interfaces come from the NIC inventory; advertised, not measured.
        "interfaces": [
            {"name": nic.name, "kind": nic.kind, "numa_node": nic.numa_node,
             "advertised_mbps": nic.speed_mbps}
            for nic in topology.nics
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--host", action="store_true",
                        help="read this machine's sysfs instead of the synthetic host")
    source.add_argument("--live", action="store_true",
                        help="map every GPU node of a running Ray cluster")
    arguments = parser.parse_args()

    if arguments.live:
        import ray

        from topology_scheduler import discover_host_topology

        ray.init(address="auto")
        try:
            topologies = discover_host_topology()
        finally:
            ray.shutdown()
        print(json.dumps({"synthetic_inputs": False,
                          "nodes": [item.as_dict() for item in topologies]}, indent=2))
        return

    if arguments.host:
        topology = collect_host_topology([])
        print(json.dumps({"synthetic_inputs": False, "gpus_supplied": False,
                          "topology": topology.as_dict()}, indent=2))
        return

    topology = collect_host_topology(DEMO_GPUS, sysfs=DemoSysfs(), node_name="demo")
    print(json.dumps({"synthetic_inputs": True, "summary": summarize(topology),
                      "topology": topology.as_dict()}, indent=2))


if __name__ == "__main__":
    main()
