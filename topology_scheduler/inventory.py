"""Discover planner GPU inventory by probing every live Ray node."""

from dataclasses import dataclass
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
class RayNodeInventory:
    """The detected GPUs and Ray scheduling identity of one node."""

    node_id: str
    node_name: str
    resource_key: str
    configured_gpus: int
    devices: tuple[GPUDevice, ...]

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


def _read_nvml(nvml) -> tuple[GPUDevice, ...]:
    """Read static properties for every NVIDIA device visible to NVML."""
    from ray._private.accelerators.nvidia_gpu import NvidiaGPUAcceleratorManager

    nvml.nvmlInit()
    try:
        devices = []
        for index in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
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
        return tuple(devices)
    finally:
        nvml.nvmlShutdown()


def _probe_nvidia_node() -> dict:
    """Run inside a Ray worker pinned to the node being inventoried."""
    import ray
    import ray._private.thirdparty.pynvml as pynvml

    return {
        "node_id": ray.get_runtime_context().get_node_id(),
        "devices": _read_nvml(pynvml),
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
    the model, UUID, total memory and PCI identity. Free VRAM is intentionally not
    used as capacity because observing it does not reserve it.
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
        ))
    return tuple(sorted(inventory, key=lambda item: item.node_name))


def discover_planner_nodes(*, timeout: float = 30) -> list[Node]:
    """Return automatically discovered inputs for ``choose_placement``."""
    return [item.as_planner_node() for item in discover_ray_gpu_inventory(timeout=timeout)]
