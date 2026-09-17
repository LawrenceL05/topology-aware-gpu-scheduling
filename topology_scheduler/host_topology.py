"""Map each GPU to its NUMA node and nearest NIC by reading host sysfs.

These are observations. Proximity is derived from the PCI hierarchy and the
kernel's NUMA files, never entered by a user. Being close to a NIC does not
promise measured bandwidth, and nothing here binds a GPU or a NIC to a
workload; Ray, KAI, and the driver still choose devices.

Every read goes through ``SysfsReader``, so the collector can be exercised
against an in-memory fixture instead of a live host.
"""

import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from .inventory import GPUDevice, _node_marker

# sysfs locations this collector reads, for the record it returns.
SOURCES = (
    ("pci_hierarchy", "/sys/devices/pci*/**"),
    ("interfaces", "/sys/devices/**/net/*"),
    ("virtual_interfaces", "/sys/devices/virtual/net/*"),
    ("numa_node", "<pci device>/numa_node"),
    ("speed", "<interface>/speed"),
    ("operstate", "<interface>/operstate"),
)
_PCI_ADDRESS = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$")


class Proximity(str, Enum):
    """How close a GPU and a NIC are on the host, closest first.

    ``SAME_SWITCH`` means the two functions share a PCI bridge below the host
    bridge. ``SAME_ROOT_COMPLEX`` means they share only the host bridge. The
    NUMA values are used when the PCI hierarchy does not relate them.
    """

    SAME_DEVICE = "same-device"
    SAME_SWITCH = "same-switch"
    SAME_ROOT_COMPLEX = "same-root-complex"
    SAME_NUMA = "same-numa"
    CROSS_NUMA = "cross-numa"
    UNKNOWN = "unknown"

    @property
    def rank(self) -> int:
        """Sort key: 0 is closest, and ``UNKNOWN`` always sorts last."""
        return list(Proximity).index(self)


@dataclass(frozen=True)
class HostNIC:
    """A network interface as the host describes it.

    ``pci_address`` and ``numa_node`` are ``None`` for virtual interfaces such
    as loopback and bridges, and whenever the kernel does not report them.
    """

    name: str
    pci_address: str | None
    pci_path: tuple[str, ...]
    numa_node: int | None
    speed_mbps: float | None
    operstate: str | None
    virtual: bool


@dataclass(frozen=True)
class NICProximity:
    """Why one GPU and one NIC were classified as close or far.

    The evidence is kept so that a classification can be checked without
    rerunning discovery: the deepest shared PCI ancestor, both NUMA nodes, and
    the reason whenever the answer is ``UNKNOWN``.
    """

    nic_name: str
    proximity: Proximity
    shared_pci_ancestor: str | None = None
    gpu_numa_node: int | None = None
    nic_numa_node: int | None = None
    reason: str | None = None

    def as_dict(self) -> dict:
        return dict(asdict(self), proximity=self.proximity.value)


@dataclass(frozen=True)
class GPULocality:
    """One GPU's host placement and its distance to every interface."""

    uuid: str
    pci_address: str
    pci_path: tuple[str, ...]
    numa_node: int | None
    nics: tuple[NICProximity, ...]

    @property
    def nearest_nic(self) -> NICProximity | None:
        """The closest interface, or ``None`` when none could be related."""
        if self.nics and self.nics[0].proximity is not Proximity.UNKNOWN:
            return self.nics[0]
        return None

    def as_dict(self) -> dict:
        return {
            "uuid": self.uuid,
            "pci_address": self.pci_address,
            "pci_path": list(self.pci_path),
            "numa_node": self.numa_node,
            "nearest_nic": self.nearest_nic.nic_name if self.nearest_nic else None,
            "nics": [item.as_dict() for item in self.nics],
        }


@dataclass(frozen=True)
class HostTopology:
    """GPU, NUMA, and NIC placement observed on one host."""

    node_name: str
    node_id: str
    gpus: tuple[GPULocality, ...]
    nics: tuple[HostNIC, ...]
    diagnostics: tuple[str, ...] = ()
    sources: tuple[tuple[str, str], ...] = SOURCES

    def as_dict(self) -> dict:
        return {
            "node_name": self.node_name,
            "node_id": self.node_id,
            "gpus": [gpu.as_dict() for gpu in self.gpus],
            "nics": [asdict(nic) | {"pci_path": list(nic.pci_path)}
                     for nic in self.nics],
            "diagnostics": list(self.diagnostics),
            "sources": dict(self.sources),
        }


def normalize_pci_address(value: str) -> str:
    """Return a sysfs-style ``dddd:bb:dd.f`` address.

    NVML reports an eight-digit domain, such as ``00000000:17:00.0``, while
    sysfs uses four. Both forms, in either case, normalize to the same string.
    """
    parts = value.strip().lower().split(":")
    if len(parts) != 3:
        raise ValueError(f"Unrecognized PCI address {value!r}")
    try:
        address = f"{int(parts[0], 16):04x}:{parts[1]}:{parts[2]}"
    except ValueError as error:
        raise ValueError(f"Unrecognized PCI address {value!r}") from error
    if not _PCI_ADDRESS.match(address):
        raise ValueError(f"Unrecognized PCI address {value!r}")
    return address


class SysfsReader:
    """Read-only access to one sysfs tree.

    Tests and examples subclass this rather than building real directories,
    because PCI addresses contain colons, which some filesystems reject.
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

    def read(self, *parts: str) -> tuple[str | None, str | None]:
        """A file's stripped contents, or why it could not be read."""
        try:
            text = self.root.joinpath(*parts).read_text(
                encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return None, "missing"
        except OSError as error:
            return None, f"unreadable ({type(error).__name__})"
        return text.strip(), None


def _numa_node(sysfs: SysfsReader, parts: tuple[str, ...]) -> tuple[int | None, str | None]:
    """Read a device's NUMA node, distinguishing absent from unreadable."""
    text, problem = sysfs.read(*parts, "numa_node")
    if text is None:
        return None, f"numa_node {problem}"
    try:
        value = int(text)
    except ValueError:
        return None, f"numa_node is unparsable ({text!r})"
    if value < 0:
        return None, "kernel reports no NUMA node (-1)"
    return value, None


def _number(text: str | None) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _pci_devices(sysfs: SysfsReader) -> dict[str, tuple[str, ...]]:
    """Map every PCI address under ``/sys/devices`` to its ancestry.

    Each value starts at the host bridge and ends with the device itself. Only
    directories whose names are PCI addresses are followed, so unrelated sysfs
    entries cannot lead the walk astray.
    """
    base = ("sys", "devices")
    stack = [base + (name,) for name in sysfs.directories(*base)
             if name.startswith("pci")]
    found: dict[str, tuple[str, ...]] = {}
    while stack:
        parts = stack.pop()
        for name in sysfs.directories(*parts):
            if not _PCI_ADDRESS.match(name):
                continue
            found[name] = parts[len(base):] + (name,)
            stack.append(parts + (name,))
    return found


def _interfaces(sysfs: SysfsReader, devices: dict[str, tuple[str, ...]]
                ) -> tuple[list[HostNIC], list[str]]:
    """Read every interface, PCI-attached or virtual, in a stable order."""
    base = ("sys", "devices")
    diagnostics: list[str] = []
    nics: list[HostNIC] = []

    def read_interface(parts, name, address, path, numa) -> HostNIC:
        speed, _ = sysfs.read(*parts, "speed")
        operstate, _ = sysfs.read(*parts, "operstate")
        return HostNIC(
            name=name, pci_address=address, pci_path=path, numa_node=numa,
            speed_mbps=_number(speed), operstate=operstate,
            virtual=address is None,
        )

    for address, path in sorted(devices.items()):
        device = base + path
        names = sysfs.directories(*device, "net")
        if not names:
            continue
        numa, problem = _numa_node(sysfs, device)
        for name in names:
            if problem:
                diagnostics.append(f"interface {name} on {address}: {problem}")
            nics.append(read_interface(device + ("net", name), name, address, path, numa))

    virtual = base + ("virtual", "net")
    for name in sysfs.directories(*virtual):
        nics.append(read_interface(virtual + (name,), name, None, (), None))

    if not nics:
        diagnostics.append("no network interfaces were found under /sys/devices")
    return sorted(nics, key=lambda nic: nic.name), diagnostics


def _classify(gpu_address: str, gpu_path: tuple[str, ...], gpu_numa: int | None,
              nic: HostNIC) -> NICProximity:
    """Relate one GPU and one NIC, keeping the evidence for the answer."""
    common = ()
    for left, right in zip(gpu_path[:-1], nic.pci_path[:-1]):
        if left != right:
            break
        common += (left,)
    shared = common[-1] if common else None
    partial = dict(nic_name=nic.name, shared_pci_ancestor=shared,
                   gpu_numa_node=gpu_numa, nic_numa_node=nic.numa_node)

    if nic.pci_address is None:
        return NICProximity(proximity=Proximity.UNKNOWN, reason=(
            "interface has no PCI device, so it is virtual or not enumerated"
        ), **partial)
    if gpu_address.rsplit(".", 1)[0] == nic.pci_address.rsplit(".", 1)[0]:
        return NICProximity(proximity=Proximity.SAME_DEVICE, **partial)
    if len(common) >= 2:
        return NICProximity(proximity=Proximity.SAME_SWITCH, **partial)
    if len(common) == 1:
        return NICProximity(proximity=Proximity.SAME_ROOT_COMPLEX, **partial)
    if gpu_numa is None or nic.numa_node is None:
        missing = "GPU" if gpu_numa is None else "interface"
        return NICProximity(proximity=Proximity.UNKNOWN, reason=(
            f"no shared PCI ancestor and the {missing} NUMA node is unknown"
        ), **partial)
    if gpu_numa == nic.numa_node:
        return NICProximity(proximity=Proximity.SAME_NUMA, **partial)
    return NICProximity(proximity=Proximity.CROSS_NUMA, **partial)


def collect_host_topology(
    devices: Iterable[GPUDevice], *,
    sysfs: SysfsReader | None = None,
    node_name: str = "local",
    node_id: str = "local",
) -> HostTopology:
    """Map each GPU to its NUMA node and its distance to every interface.

    ``devices`` are the GPUs NVML reported; their UUIDs and PCI bus IDs stay
    the identifiers throughout. Everything else is read through ``sysfs``,
    which defaults to this host. A GPU or interface the kernel does not
    describe is kept with an ``UNKNOWN`` classification and a reason rather
    than dropped. Ordering is stable: GPUs by PCI address, interfaces by name,
    and each GPU's interfaces by proximity and then name.
    """
    sysfs = sysfs or SysfsReader()
    pci = _pci_devices(sysfs)
    nics, diagnostics = _interfaces(sysfs, pci)

    localities = []
    for device in devices:
        address = normalize_pci_address(device.pci_bus_id)
        path = pci.get(address, ())
        if not path:
            diagnostics.append(
                f"GPU {device.uuid}: PCI device {address} is not present under "
                "/sys/devices, so its NUMA node and NIC distances are unknown")
            numa = None
        else:
            numa, problem = _numa_node(sysfs, ("sys", "devices") + path)
            if problem:
                diagnostics.append(f"GPU {device.uuid} at {address}: {problem}")
        localities.append(GPULocality(
            uuid=device.uuid, pci_address=address, pci_path=path, numa_node=numa,
            nics=tuple(sorted(
                (_classify(address, path, numa, nic) for nic in nics),
                key=lambda item: (item.proximity.rank, item.nic_name),
            )),
        ))
    return HostTopology(
        node_name=node_name, node_id=node_id,
        gpus=tuple(sorted(localities, key=lambda gpu: gpu.pci_address)),
        nics=tuple(nics), diagnostics=tuple(diagnostics),
    )


def _probe_host_topology() -> dict:
    """Run inside a Ray worker pinned to the node being mapped."""
    import ray
    import ray._private.thirdparty.pynvml as pynvml

    from .inventory import _read_nvml_snapshot

    devices, _ = _read_nvml_snapshot(pynvml)
    node_id = ray.get_runtime_context().get_node_id()
    return {"node_id": node_id,
            "topology": collect_host_topology(devices, node_id=node_id)}


def discover_host_topology(*, timeout: float = 30) -> tuple[HostTopology, ...]:
    """Map GPUs to NUMA nodes and NICs on every live Ray GPU node.

    Each node is probed by its own pinned task, so nothing is entered by hand.
    Results are ordered by topology node name.
    """
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if not ray.is_initialized():
        raise RuntimeError("Call ray.init() before host topology discovery")

    nodes = [node for node in ray.nodes()
             if node["Alive"] and node["Resources"].get("GPU", 0) > 0]
    if not nodes:
        raise ValueError("No live Ray nodes advertise GPU resources")
    probe = ray.remote(num_cpus=0)(_probe_host_topology)
    pending, metadata = [], []
    for node in nodes:
        node_id = node["NodeID"]
        _, name = _node_marker(node["Resources"], node_id)
        pending.append(probe.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False)).remote())
        metadata.append((node_id, name))

    results = ray.get(pending, timeout=timeout)
    topologies = []
    for (expected_id, name), result in zip(metadata, results):
        if result["node_id"] != expected_id:
            raise RuntimeError(f"Host topology probe for {name} ran on the wrong Ray node")
        topology = result["topology"]
        topologies.append(HostTopology(
            node_name=name, node_id=expected_id, gpus=topology.gpus,
            nics=topology.nics, diagnostics=topology.diagnostics,
            sources=topology.sources,
        ))
    return tuple(sorted(topologies, key=lambda item: item.node_name))
