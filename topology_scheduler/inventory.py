"""Discover planner GPU inventory by probing every live Ray node."""

from dataclasses import dataclass
from itertools import combinations
from math import isclose

from .policy import Node


@dataclass(frozen=True)
class GPUDevice:
    """Static NVIDIA GPU properties reported by NVML."""

    index: int
    uuid: str
    name: str
    accelerator_type: str
    memory_gb: float
    pci_bus_id: str


@dataclass(frozen=True)
class GPUConnection:
    """Undirected NVML relationship between two GPUs on the same node."""

    source_uuid: str
    target_uuid: str
    common_ancestor: str | None
    direct_nvlink_count: int | None


@dataclass(frozen=True)
class RayNodeInventory:
    """The detected GPUs and Ray scheduling identity of one node."""

    node_id: str
    node_name: str
    resource_key: str
    configured_gpus: int
    devices: tuple[GPUDevice, ...]
    connections: tuple[GPUConnection, ...] = ()

    def as_planner_node(self) -> Node:
        """Convert a homogeneous Ray node into the V1 planner input."""
        if not self.devices:
            raise ValueError(f"No NVIDIA GPUs detected on {self.node_name}")
        models = {device.accelerator_type for device in self.devices}
        if len(models) != 1:
            raise ValueError(
                f"Mixed GPU models on {self.node_name} are not supported by V1.1: "
                f"{sorted(models)}"
            )
        memories = [device.memory_gb for device in self.devices]
        if not all(isclose(memories[0], value, rel_tol=0.001) for value in memories[1:]):
            raise ValueError(
                f"Non-uniform GPU memory on {self.node_name} is not supported by V1.1"
            )
        if self.configured_gpus > len(self.devices):
            raise ValueError(
                f"Ray advertises {self.configured_gpus} GPUs on {self.node_name}, "
                f"but NVML detected {len(self.devices)}"
            )
        return Node(
            self.node_name,
            next(iter(models)),
            self.configured_gpus,
            memories[0],
        )


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _topology_name(nvml, value: int) -> str:
    names = {
        nvml.NVML_TOPOLOGY_INTERNAL: "internal",
        nvml.NVML_TOPOLOGY_SINGLE: "single-pci-bridge",
        nvml.NVML_TOPOLOGY_MULTIPLE: "multiple-pci-bridges",
        nvml.NVML_TOPOLOGY_HOSTBRIDGE: "host-bridge",
        nvml.NVML_TOPOLOGY_NODE: "numa-node",
        nvml.NVML_TOPOLOGY_SYSTEM: "system",
    }
    return names.get(value, f"unknown-{value}")


def _direct_nvlink_count(nvml, handle, target_pci_bus_id: str) -> int | None:
    """Count active links to a GPU, or return ``None`` when NVML cannot tell."""
    count = 0
    state_queries = 0
    unresolved_active_link = False
    for link in range(nvml.NVML_NVLINK_MAX_LINKS):
        try:
            if not nvml.nvmlDeviceGetNvLinkState(handle, link):
                state_queries += 1
                continue
            state_queries += 1
        except nvml.NVMLError:
            # NVLink inspection is unavailable on some drivers and devices.
            continue
        try:
            remote = nvml.nvmlDeviceGetNvLinkRemotePciInfo(handle, link)
            if _text(remote.busId).lower() == target_pci_bus_id.lower():
                count += 1
        except nvml.NVMLError:
            # An active NVSwitch link may not expose a remote GPU PCI identity.
            unresolved_active_link = True
    if state_queries == 0 or unresolved_active_link:
        return None
    return count


def _common_ancestor(nvml, left_handle, right_handle) -> str | None:
    """Return the normalized PCI/NUMA relationship when NVML supports it."""
    try:
        value = nvml.nvmlDeviceGetTopologyCommonAncestor(left_handle, right_handle)
    except nvml.NVMLError:
        return None
    return _topology_name(nvml, value)


def _undirected_nvlink_count(
        nvml, left_handle, right_handle, left_pci: str, right_pci: str
) -> int | None:
    """Reconcile the NVLink count reported by both endpoints of a GPU pair."""
    forward = _direct_nvlink_count(nvml, left_handle, right_pci)
    reverse = _direct_nvlink_count(nvml, right_handle, left_pci)
    known = [count for count in (forward, reverse) if count is not None]
    if not known:
        return None
    if len(known) == 2 and forward != reverse:
        return None
    return known[0]


def _read_nvml_snapshot(
        nvml) -> tuple[tuple[GPUDevice, ...], tuple[GPUConnection, ...]]:
    """Read static GPU properties and pair topology in one NVML session."""
    from ray._private.accelerators.nvidia_gpu import NvidiaGPUAcceleratorManager

    nvml.nvmlInit()
    try:
        devices = []
        handles = []
        for index in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            handles.append(handle)
            name = _text(nvml.nvmlDeviceGetName(handle))
            accelerator_type = NvidiaGPUAcceleratorManager._gpu_name_to_accelerator_type(name)
            if not accelerator_type:
                raise RuntimeError(f"Ray could not derive an accelerator type from {name!r}")
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            pci = nvml.nvmlDeviceGetPciInfo(handle)
            devices.append(GPUDevice(
                index=index,
                uuid=_text(nvml.nvmlDeviceGetUUID(handle)),
                name=name,
                accelerator_type=accelerator_type,
                memory_gb=memory.total / 1_000_000_000,
                pci_bus_id=_text(pci.busId),
            ))
        connections = []
        for left, right in combinations(range(len(devices)), 2):
            connections.append(GPUConnection(
                source_uuid=devices[left].uuid,
                target_uuid=devices[right].uuid,
                common_ancestor=_common_ancestor(
                    nvml, handles[left], handles[right]
                ),
                direct_nvlink_count=_undirected_nvlink_count(
                    nvml,
                    handles[left],
                    handles[right],
                    devices[left].pci_bus_id,
                    devices[right].pci_bus_id,
                ),
            ))
        return tuple(devices), tuple(connections)
    finally:
        nvml.nvmlShutdown()


def _read_nvml(nvml) -> tuple[GPUDevice, ...]:
    """Backward-compatible device-only view of the NVML snapshot."""
    return _read_nvml_snapshot(nvml)[0]


def _probe_nvidia_node() -> dict:
    """Run inside a Ray worker pinned to the node being inventoried."""
    import ray
    import ray._private.thirdparty.pynvml as pynvml

    devices, connections = _read_nvml_snapshot(pynvml)
    return {
        "node_id": ray.get_runtime_context().get_node_id(),
        "devices": devices,
        "connections": connections,
    }


def _node_marker(resources: dict, node_id: str) -> tuple[str, str]:
    markers = sorted(
        key for key, quantity in resources.items()
        if key.startswith("topology_node:") and quantity > 0
    )
    if len(markers) != 1:
        raise ValueError(
            f"Ray node {node_id} must advertise exactly one topology_node:<name> resource"
        )
    marker = markers[0]
    return marker, marker.removeprefix("topology_node:")


def discover_ray_gpu_inventory(*, timeout: float = 30) -> tuple[RayNodeInventory, ...]:
    """Probe NVIDIA hardware on each live Ray node and return stable inventory.

    Ray's configured ``GPU`` value is used as schedulable capacity. NVML supplies
    the model, UUID, total memory, PCI identity, and pairwise intra-node
    topology. Free VRAM is intentionally not used as capacity because observing
    it does not reserve it. A ``None`` connection field means that NVML could not
    report that property; zero NVLinks means it successfully found no direct
    links between the pair.
    """
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if not ray.is_initialized():
        raise RuntimeError("Call ray.init() before GPU discovery")

    nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) > 0]
    pending = []
    metadata = []
    probe = ray.remote(num_cpus=0)(_probe_nvidia_node)
    for node in nodes:
        node_id = node["NodeID"]
        marker, name = _node_marker(node["Resources"], node_id)
        configured = node["Resources"]["GPU"]
        if configured != int(configured) or configured < 1:
            raise ValueError(f"Ray node {name} must advertise a positive integer GPU capacity")
        pending.append(probe.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        ).remote())
        metadata.append((node_id, name, marker, int(configured)))

    if not pending:
        raise ValueError("No live Ray nodes advertise GPU resources")
    results = ray.get(pending, timeout=timeout)
    inventory = []
    for (expected_id, name, marker, configured), result in zip(metadata, results):
        if result["node_id"] != expected_id:
            raise RuntimeError(f"GPU probe for {name} ran on the wrong Ray node")
        inventory.append(RayNodeInventory(
            node_id=expected_id,
            node_name=name,
            resource_key=marker,
            configured_gpus=configured,
            devices=tuple(result["devices"]),
            connections=tuple(result["connections"]),
        ))
    return tuple(sorted(inventory, key=lambda item: item.node_name))


def discover_planner_nodes(*, timeout: float = 30) -> list[Node]:
    """Return automatically discovered inputs for ``choose_placement``."""
    return [item.as_planner_node() for item in discover_ray_gpu_inventory(timeout=timeout)]
