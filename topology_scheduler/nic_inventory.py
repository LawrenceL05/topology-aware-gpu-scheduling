"""Discover the network interfaces of every live Ray node from host sysfs.

This is an inventory, not a measurement. Advertised link speed is what the
driver reports the link negotiated; it is never application throughput, and a
value the kernel does not provide is recorded as unknown rather than as zero.

Every value carries where it came from and how much it is worth, so a partial
answer is still usable: a node with an unreadable file keeps its other
interfaces instead of disappearing.

Reads go through ``SysfsReader``, so the collector runs against an in-memory
fixture as easily as against a host.
"""

import errno
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from .inventory import _node_marker
from .policy import positive

# How much one reading is worth.
REPORTED = "reported"          # The kernel gave a usable value.
UNAVAILABLE = "unavailable"    # The file does not exist on this interface.
UNREADABLE = "unreadable"      # The file exists but could not be read.
UNSUPPORTED = "unsupported"    # The driver declined to answer, such as speed -1.

PHYSICAL, VIRTUAL, LOOPBACK, UNKNOWN = "physical", "virtual", "loopback", "unknown"
# Linux ARPHRD values found in /sys/class/net/<name>/type.
ARPHRD_ETHER, ARPHRD_INFINIBAND, ARPHRD_LOOPBACK = 1, 32, 772
UNITS = {"speed_mbps": "Mbit/s advertised by the driver, not measured throughput",
         "mtu": "bytes", "numa_node": "kernel NUMA node id"}


@dataclass(frozen=True)
class Reading:
    """One discovered value, its sysfs source, and its confidence."""

    value: object | None
    source: str
    confidence: str

    @property
    def known(self) -> bool:
        return self.confidence == REPORTED

    def as_dict(self) -> dict:
        return {"value": self.value, "source": self.source,
                "confidence": self.confidence}


@dataclass(frozen=True)
class RDMADevice:
    """An RDMA device and the PCI function it shares with a network interface."""

    name: str
    pci_address: Reading
    node_type: Reading
    link_layer: Reading
    port_states: tuple[Reading, ...] = ()
    problems: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "pci_address": self.pci_address.as_dict(),
            "node_type": self.node_type.as_dict(),
            "link_layer": self.link_layer.as_dict(),
            "port_states": [state.as_dict() for state in self.port_states],
            "problems": list(self.problems),
        }


@dataclass(frozen=True)
class NetworkInterface:
    """One interface as the host describes it, with per-field provenance."""

    name: str
    kind: str
    mac: Reading
    pci_address: Reading
    numa_node: Reading
    driver: Reading
    operstate: Reading
    mtu: Reading
    speed_mbps: Reading
    rdma: tuple[RDMADevice, ...] = ()
    problems: tuple[str, ...] = ()

    @property
    def up(self) -> bool:
        return self.operstate.value == "up"

    def as_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "up": self.up,
            "mac": self.mac.as_dict(), "pci_address": self.pci_address.as_dict(),
            "numa_node": self.numa_node.as_dict(), "driver": self.driver.as_dict(),
            "operstate": self.operstate.as_dict(), "mtu": self.mtu.as_dict(),
            "speed_mbps": self.speed_mbps.as_dict(),
            "rdma": [device.as_dict() for device in self.rdma],
            "problems": list(self.problems),
        }


@dataclass(frozen=True)
class NodeNICInventory:
    """Every interface one node reported, beside its Ray identity."""

    node_id: str
    node_name: str
    interfaces: tuple[NetworkInterface, ...]
    problems: tuple[str, ...] = ()
    unattached_rdma: tuple[RDMADevice, ...] = ()

    @property
    def physical(self) -> tuple[NetworkInterface, ...]:
        return tuple(item for item in self.interfaces if item.kind == PHYSICAL)

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id, "node_name": self.node_name,
            "interfaces": [item.as_dict() for item in self.interfaces],
            "unattached_rdma": [device.as_dict() for device in self.unattached_rdma],
            "problems": list(self.problems), "units": dict(UNITS),
        }


class SysfsReader:
    """Read-only access to one sysfs tree.

    Tests and examples subclass this instead of building directories, so the
    collector can be exercised without NIC hardware or root.
    """

    def __init__(self, root: Path | str = "/"):
        self.root = Path(root)

    def directories(self, *parts: str) -> tuple[str, ...]:
        """The names of a path's subdirectories, sorted, or empty if unreadable."""
        try:
            return tuple(sorted(
                entry.name for entry in self.root.joinpath(*parts).iterdir()
                if entry.is_dir()))
        except OSError:
            return ()

    def read(self, *parts: str) -> tuple[str | None, str]:
        """A file's contents and the confidence that the answer deserves."""
        try:
            text = self.root.joinpath(*parts).read_text(
                encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return None, UNAVAILABLE
        except OSError as error:
            # sysfs answers EINVAL for a value the driver cannot supply, such
            # as the speed of a virtual interface, and EACCES when the reader
            # lacks permission. Neither is a measured zero, and they are not
            # the same problem.
            if error.errno == errno.EINVAL:
                return None, UNSUPPORTED
            return None, UNREADABLE
        return text.strip(), REPORTED


def _path(*parts: str) -> str:
    return "/" + "/".join(parts)


def _reading(sysfs: SysfsReader, parts: Sequence[str], *, convert=None) -> Reading:
    """Read one sysfs file into a value that remembers where it came from."""
    text, confidence = sysfs.read(*parts)
    source = _path(*parts)
    if text is None or text == "":
        return Reading(None, source, UNAVAILABLE if text == "" else confidence)
    if convert is None:
        return Reading(text, source, REPORTED)
    try:
        value = convert(text)
    except (TypeError, ValueError):
        return Reading(None, source, UNSUPPORTED)
    return Reading(value, source, REPORTED if value is not None else UNSUPPORTED)


def _positive_int(text: str) -> int | None:
    """Return a nonnegative integer, or None for the kernel's unknown values."""
    value = int(text)
    return value if value >= 0 else None


def _uevent(sysfs: SysfsReader, parts: Sequence[str]) -> dict[str, Reading]:
    """Parse a sysfs uevent file, which names the PCI slot and the driver.

    Reading ``device/uevent`` avoids following the ``device`` symlink, so the
    same code works against a fixture that cannot contain symlinks.
    """
    text, confidence = sysfs.read(*parts, "device", "uevent")
    source = _path(*parts, "device", "uevent")
    values = {}
    for line in (text or "").splitlines():
        key, separator, value = line.partition("=")
        if separator and value:
            values[key.strip()] = value.strip()
    return {
        key: Reading(values.get(key), source,
                     REPORTED if key in values else
                     confidence if confidence != REPORTED else UNAVAILABLE)
        for key in ("PCI_SLOT_NAME", "DRIVER")
    }


def _classify(kind_type: Reading, pci: Reading, name: str,
              virtual_names: Sequence[str]) -> str:
    """Sort an interface into physical, virtual, loopback, or unknown."""
    if kind_type.value == ARPHRD_LOOPBACK or name == "lo":
        return LOOPBACK
    if pci.known:
        return PHYSICAL
    if name in virtual_names:
        return VIRTUAL
    return UNKNOWN


def _rdma_devices(sysfs: SysfsReader) -> tuple[RDMADevice, ...]:
    """Read every RDMA device, including devices that cannot be PCI-matched."""
    base = ("sys", "class", "infiniband")
    devices = []
    for name in sysfs.directories(*base):
        parts = base + (name,)
        pci = _uevent(sysfs, parts)["PCI_SLOT_NAME"]
        node_type = _reading(sysfs, parts + ("node_type",))
        states, link_layer = [], None
        for port in sysfs.directories(*parts, "ports"):
            state = _reading(sysfs, parts + ("ports", port, "state"))
            states.append(Reading(
                f"{port}:{state.value}" if state.value is not None else None,
                state.source, state.confidence))
            if link_layer is None:
                link_layer = _reading(
                    sysfs, parts + ("ports", port, "link_layer"))
        if link_layer is None:
            link_layer = Reading(None, _path(*parts, "ports"), UNAVAILABLE)
        readings = (pci, node_type, link_layer, *states)
        unreadable = tuple(
            f"{reading.source} could not be read"
            for reading in readings if reading.confidence == UNREADABLE)
        devices.append(RDMADevice(
            name=name, pci_address=pci, node_type=node_type,
            link_layer=link_layer, port_states=tuple(states),
            problems=unreadable))
    return tuple(devices)


def collect_nic_inventory(
    *, sysfs: SysfsReader | None = None, node_id: str = "local",
    node_name: str = "local",
) -> NodeNICInventory:
    """Read every interface this host exposes under ``/sys/class/net``.

    Nothing is entered by hand and nothing is inferred: each field records the
    file it came from and whether the kernel actually answered. An interface
    whose files cannot be read is kept with its problems listed.
    """
    sysfs = sysfs or SysfsReader()
    base = ("sys", "class", "net")
    rdma_devices = _rdma_devices(sysfs)
    rdma_by_pci: dict[str, list[RDMADevice]] = {}
    for device in rdma_devices:
        if device.pci_address.known:
            rdma_by_pci.setdefault(str(device.pci_address.value), []).append(device)
    interfaces, problems = [], []
    attached_rdma = set()
    names = sysfs.directories(*base)
    virtual_names = sysfs.directories("sys", "devices", "virtual", "net")
    if not names:
        problems.append(f"no interfaces were found under {_path(*base)}")
    for name in names:
        parts = base + (name,)
        uevent = _uevent(sysfs, parts)
        pci = uevent["PCI_SLOT_NAME"]
        driver = uevent["DRIVER"]
        kind_type = _reading(sysfs, parts + ("type",), convert=int)
        matching_rdma = tuple(rdma_by_pci.get(str(pci.value), ())) if pci.known else ()
        attached_rdma.update(matching_rdma)
        interface = NetworkInterface(
            name=name,
            kind=_classify(kind_type, pci, name, virtual_names),
            mac=_reading(sysfs, parts + ("address",)),
            pci_address=pci,
            numa_node=_reading(sysfs, parts + ("device", "numa_node"),
                               convert=_positive_int),
            driver=driver,
            operstate=_reading(sysfs, parts + ("operstate",)),
            mtu=_reading(sysfs, parts + ("mtu",), convert=_positive_int),
            speed_mbps=_reading(sysfs, parts + ("speed",), convert=_positive_int),
            rdma=matching_rdma,
        )
        unreadable = tuple(
            f"{field}: {getattr(interface, field).source} could not be read"
            for field in ("mac", "pci_address", "driver", "numa_node",
                          "operstate", "mtu", "speed_mbps")
            if getattr(interface, field).confidence == UNREADABLE
        )
        interfaces.append(replace(interface, problems=unreadable))
    unattached_rdma = tuple(
        device for device in rdma_devices if device not in attached_rdma)
    for device in rdma_devices:
        problems.extend(
            f"RDMA device {device.name}: {problem}"
            for problem in device.problems)
    for device in unattached_rdma:
        if device.pci_address.known:
            problems.append(
                f"RDMA device {device.name}: no network interface shares PCI "
                f"address {device.pci_address.value}")
        else:
            problems.append(
                f"RDMA device {device.name}: {device.pci_address.source} is "
                f"{device.pci_address.confidence}; cannot associate it with a "
                "network interface")
    return NodeNICInventory(
        node_id=node_id, node_name=node_name,
        interfaces=tuple(interfaces), problems=tuple(problems),
        unattached_rdma=unattached_rdma)


def _probe_node_nics() -> dict:
    """Run inside a Ray worker pinned to the node being inventoried."""
    import ray

    node_id = ray.get_runtime_context().get_node_id()
    return {"node_id": node_id,
            "inventory": collect_nic_inventory(node_id=node_id)}


def discover_nic_inventory(*, timeout: float = 30) -> tuple[NodeNICInventory, ...]:
    """Collect the NIC inventory of every live Ray node that has a marker.

    This runs beside ``discover_ray_gpu_inventory()`` and uses the same node
    identity, so GPU and NIC records describe the same machine. Unlike GPU
    discovery it does not require the node to have a GPU.
    """
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    positive(timeout, "timeout")
    if not ray.is_initialized():
        raise RuntimeError("Call ray.init() before NIC discovery")

    metadata = []
    for node in ray.nodes():
        resources = node["Resources"]
        if not node["Alive"] or not any(
                key.startswith("topology_node:") and quantity > 0
                for key, quantity in resources.items()):
            continue
        node_id = node["NodeID"]
        _, name = _node_marker(resources, node_id)
        metadata.append((node_id, name))
    if not metadata:
        raise ValueError("No live Ray node advertises a topology_node:<name> resource")
    if len({name for _, name in metadata}) != len(metadata):
        raise ValueError("Live Ray nodes must have unique topology_node names")

    metadata.sort(key=lambda item: item[1])
    probe = ray.remote(num_cpus=0, max_retries=0)(_probe_node_nics)
    pending = []
    try:
        for node_id, name in metadata:
            pending.append(probe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False)).remote())
        results = ray.get(pending, timeout=timeout)
    except BaseException:
        for ref in pending:
            try:
                ray.cancel(ref, force=True)
            except Exception:
                pass  # Preserve the original failure if Ray itself is unavailable.
        raise
    inventories = []
    for (expected_id, name), result in zip(metadata, results):
        if result["node_id"] != expected_id:
            raise RuntimeError(f"NIC probe for {name} ran on the wrong Ray node")
        inventory = result["inventory"]
        inventories.append(NodeNICInventory(
            node_id=expected_id, node_name=name,
            interfaces=inventory.interfaces, problems=inventory.problems,
            unattached_rdma=inventory.unattached_rdma))
    return tuple(sorted(inventories, key=lambda item: item.node_name))
